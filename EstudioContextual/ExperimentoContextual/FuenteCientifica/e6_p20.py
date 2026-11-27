"""P20: controlled 3 x 3 A/P/O contextual-representation experiment.

Frozen K80 S task, canonical occurrence positions, multiplicity-exact operators,
TRAIN-A-only common calibration, word-level classifier training and inference.
No historical predictions are imported. All sources are opened read-only.
"""
from __future__ import annotations
import contextlib, gc, hashlib, io, json, os, platform, shutil, sys, threading, time, warnings, zipfile
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn, pyarrow, pyarrow.parquet as pq, pyarrow.compute as pc
import psutil, joblib
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold
import e6_p18_core as core
import e6_p18_static as static
import e6_p18_compute as compute
import e6_p18_numeric as numeric

PACKAGE=Path(__file__).resolve().parent
VERSION='P20_CONTROLLED_APO_V1'
BETAS=compute.CAL_BETAS.copy()
NETS=tuple(compute.NETS)
ARMS=tuple(m+q for m in 'APO' for q in 'APO')
MODELS=(dict(name='PRIMARY',family='S_TENSOR_SHARED',alpha=1e-4),
        dict(name='SENSITIVITY',family='H_BETA_LINEAR_Z',alpha=.1))
OCC_SHA='ea073b657e9a30008e9a6764871d0bae920c392ce3dba840721e3977457e373f'
DATASET_SHA='d19b40b890dad8850d2407f45bf5269f6aef494b0df6e08e3d2e9cbafceb0b33'
req,sha,digest,readj,writej,frame,savez,cached,seal=(compute.req,compute.sha,compute.digest,compute.readj,
    compute.writej,compute.frame,compute.savez,compute.cached,compute.seal)

@dataclass
class Config:
    # Code lives beside P19; short output paths avoid Windows path-length problems.
    output_root: Path=field(default_factory=lambda:Path.home()/'E6R')
    project_root: Path=field(default_factory=lambda:Path.home()/'Workspace/Tesis/Exp2def')
    dataset_dir: Path=field(default_factory=lambda:Path.home()/'Workspace/Tesis/Exp2/Results/E6_DATASET_PSEUDOSENTIDOS_WORDNET/run_4b7a935bdfb47252c452/exports/dataset_20261022T211853_eeab30')
    occurrence_dir: Path=field(default_factory=lambda:Path.home()/'Workspace/Tesis/Exp2/Results/contextual_dataset')
    audit_root: Path|None=None
    embedding_path: Path|None=None
    max_workers: int=8
    reserve_gib: float=3.
    reserve_disk_gib: float=8.
    beta_chunk: int=7
    query_block: int=32
    show_figures: bool=True
    figure_dpi: int=150
    force_rehash_sources: bool=False
    def __post_init__(self):
        for k in ('output_root','project_root','dataset_dir','occurrence_dir','audit_root','embedding_path'):
            v=getattr(self,k)
            if v is not None:setattr(self,k,Path(v).expanduser().resolve())
        req(1<=self.max_workers<=32 and self.beta_chunk in (1,2,7,14),'Invalid scheduling settings.')
        req(1<=self.query_block<=128 and self.reserve_gib>=1 and self.reserve_disk_gib>=2,'Invalid resource reserve.')

def environment():
    return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,pyarrow=pyarrow.__version__,joblib=joblib.__version__)

def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def readcsv(path):return pd.read_csv(path,keep_default_na=False)

def put_npy(path,arr):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.npy')
    with tmp.open('wb') as f:np.save(f,arr,allow_pickle=False)
    os.replace(tmp,path)

def frozen(path,data):
    if path.exists():req(readj(path)==core.safe85(data),'Frozen identity mismatch: '+str(path))
    else:writej(path,data)

def verify_package():
    for name,h in readj(PACKAGE/'PACKAGE_SHA256.json').items():
        p=(PACKAGE/name).resolve()
        req(p.is_relative_to(PACKAGE) and p.is_file() and sha(p)==h,'Package altered: '+name)

def source_context(cfg):
    override={k:v for k,v in dict(audit_root=cfg.audit_root,embedding_path=cfg.embedding_path).items() if v is not None}
    sc=static.StaticConfig(project_root=cfg.project_root,output_root=cfg.output_root,
        datasets=('prototypes',),mode='reuse_only',overrides={'prototypes':override})
    return static._configuration(sc,'prototypes')

def preflight(cfg=None):
    cfg=cfg or Config();verify_package();s=source_context(cfg)
    paths={'S audit':s['paths']['audit_root'],'Frozen pseudo-sense export':cfg.dataset_dir,
        'Canonical occurrences':cfg.occurrence_dir/'contextual_occurrences_definitive.parquet',
        'Occurrence row-group index':cfg.occurrence_dir/'word_row_group_index.csv'}
    rows=[dict(role=k,path=str(p),exists=p.exists()) for k,p in paths.items()]
    print('P20: 9 arms; K80; 640 words; 8 classes; 100 canonical positions/word.')
    print('449,820 new ensembles; each evaluates A, P and O queries. No P19 prediction reuse.')
    print('Output parent:',cfg.output_root)
    print('Environment:',environment())
    print('Available RAM (GiB):',round(psutil.virtual_memory().available/2**30,2))
    return pd.DataFrame(rows)

def guard(c,need=0.):
    root=Path(c.get('main_root',c['root']));cfg=c['cfg']
    if (root/'STOP').exists():raise compute.TechnicalPause('STOP present. Checkpoints retained. Remove STOP to resume.')
    free=psutil.virtual_memory().available/2**30
    if free<cfg.reserve_gib+need:raise compute.TechnicalPause(f'RAM pause: {free:.2f} GiB available; need {cfg.reserve_gib+need:.2f}.')
    if shutil.disk_usage(root).free/2**30<cfg.reserve_disk_gib:raise compute.TechnicalPause('Disk reserve reached. Checkpoints retained.')
    for p,st in c.get('source_stats',{}).items():
        z=Path(p).stat();req((z.st_size,z.st_mtime_ns)==tuple(st),'Source changed: '+str(p))

@contextlib.contextmanager
def stage(c,name):
    root=Path(c.get('main_root',c['root']))
    with core.run_lock(root):
        try:
            guard(c);yield;guard(c)
            writej(root/'execution_status.json',dict(status='STAGE_COMPLETE',stage=name,resumable=True))
        except BaseException as e:
            writej(root/'execution_status.json',dict(status='TECHNICAL_PAUSE',stage=name,error=repr(e),resumable=True))
            raise

def worker_context(c):
    return dict(root=c.get('main_root',c['root']),cache=c['cache'],reserve_gib=c['cfg'].reserve_gib,
                kernel_id=c['kernel_id'],split_files=c['split_files'])

def checked_source(c,path,want=None):
    path=Path(path).resolve();req(path.is_file(),'Required local source missing: '+str(path))
    # This ledger is operational; a known expected hash is still mandatory for frozen sources.
    ledger_path=c['root']/'source_hash_cache.json'
    ledger=readj(ledger_path) if ledger_path.exists() else {}
    st=path.stat();entry=ledger.get(str(path),{})
    same=entry.get('size')==st.st_size and entry.get('mtime_ns')==st.st_mtime_ns
    if not same or c['cfg'].force_rehash_sources:
        print('Authenticating source:',path.name,flush=True);h=hashlib.sha256();last=time.monotonic();seen=0
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):
                h.update(block);seen+=len(block)
                if time.monotonic()-last>20:
                    print(f'  {100*seen/max(st.st_size,1):.1f}% verified',flush=True);guard(c);last=time.monotonic()
        entry=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=h.hexdigest());ledger[str(path)]=entry
        writej(ledger_path,ledger)
    req(want is None or entry['sha256']==want,'Frozen source hash mismatch: '+str(path))
    c.setdefault('source_stats',{})[str(path)]=(st.st_size,st.st_mtime_ns)
    return entry['sha256']

def prepare(cfg=None):
    cfg=cfg or Config();verify_package();kernel=readj(PACKAGE/'evidence/p18_kernel.json')
    req(environment()==kernel['environment'],'Use the same Python environment as P19. Expected '+str(kernel['environment'])+'; actual '+str(environment()))
    s=source_context(cfg);static._read_audit_contract(s);g=static._load_geometry(s);cat=static._catalog_embeddings(s)
    words=sorted(static.build_word_class_map(g['neighbors']));E=cat.loc[words].copy();del cat;gc.collect()
    req(E.shape==(640,768) and len(g['full'])==414 and len(g['common'])==376,'Frozen S K80 inventory mismatch.')
    req(s['audit_id']==kernel['source_audit_id'] and g['signature']==kernel['geometry_identity'],'Wrong S task.')
    req(hashlib.sha256(E.to_numpy(dtype='<f8').tobytes()).hexdigest()==kernel['augmented_array_sha'],'S means changed.')
    for old,want in readj(PACKAGE/'evidence/p18_source_ledger.json').items():
        suffix=old.replace('\\','/').split('/audit_3ee527e0fa4f7cfa423f/')[-1]
        req(sha(s['paths']['audit_root']/suffix)==want,'Frozen S artifact changed: '+suffix)
    contract=dict(version=VERSION,arms=ARMS,networks=NETS,betas=BETAS,splits=list(range(51)),seeds=list(range(10)),
        K=80,n_classes=8,n_words=640,positions_per_word=100,iterations=1,models=MODELS,
        reference_S=digest(kernel),dataset_sha=DATASET_SHA,occurrences_sha=OCC_SHA,
        code=readj(PACKAGE/'PACKAGE_SHA256.json'),environment=environment(),
        calibration='TRAIN A, 100 positions per word, class-balanced effective fraction; shared across all arms',
        rho='median TRAIN A norm, one entry per word',keys='rho * RAW / norm(RAW)',values='RAW',queries='RAW',
        classifier_training='one recovered A query per TRAIN word; shared across inference A/P/O',
        geometry='S FULL/COMMON/competitors frozen; no outcome selection',CV='5x5 word-grouped shared folds',
        multiplicity='A=100; P=frozen support; O=1; exact group-mass operator',
        inference='100 canonical positions per word; equal word/class-bin weights; not independent replicates')
    rid=digest(contract);root=cfg.output_root/('p20_'+rid[:12]);root.mkdir(parents=True,exist_ok=True)
    for old in (s['paths']['audit_root'],cfg.dataset_dir,cfg.occurrence_dir,PACKAGE):
        req(not root.is_relative_to(old) and not old.is_relative_to(root),'Output must be separate from source folders.')
    c=dict(cfg=cfg,root=root,cache=root,run_id=rid,kernel_id=rid,contract=contract,split_files={},source_stats={},geometries={'S':g})
    with stage(c,'PREPARE_S'),threadpool_limits(limits=1):
        frozen(root/'protocol.json',contract)
        for p,h in s['protected'].items():checked_source(c,p,h)
        classmap=static.build_word_class_map(g['neighbors']);replay=[]
        for seed in range(51):
            guard(c)
            sp=static.prepare_classification_split(E,g['neighbors'],frequency_threshold=2,validation_split=.33,
                random_seed_split=seed,pairwise_different_class=True)
            old=next(a for a in g['split_audit'] if a['split_seed']==seed)
            X=sp.train_embeddings.to_numpy(dtype='<f8');Q=sp.validation_embeddings.to_numpy(dtype='<f8')
            req(static.digest85(sp.train_words)==old['train_words_sha'],'TRAIN word order mismatch.')
            req(hashlib.sha256(X.tobytes()).hexdigest()==old['train_array_sha'],'TRAIN vector order mismatch.')
            got=pd.DataFrame(dict(word=sp.validation_words,true_class=sp.true_classes,validation_similarity=sp.validation_similarities)).sort_values('word').reset_index(drop=True)
            want=g['expected'].loc[g['expected'].split_seed.eq(seed),got.columns].sort_values('word').reset_index(drop=True)
            pd.testing.assert_frame_equal(got,want,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
            req(not sp.drop_info['dropped_validation_words'],'Unexpected query exclusions.')
            req(set(sp.train_words).isdisjoint(sp.validation_words) and set(sp.train_words)|set(sp.validation_words)==set(words),'Word-role leakage or missing words.')
            z=dict(X=X,Q=Q,y=np.array([classmap[w] for w in sp.train_words],np.int32),truth=np.asarray(sp.true_classes,np.int32),
                train_words=np.asarray(sp.train_words,str),query_words=np.asarray(sp.validation_words,str))
            path=root/'inputs'/f's{seed:02d}.npz'
            if path.exists():req(all(np.array_equal(arrays(path)[k],v) for k,v in z.items()),'Existing split differs.')
            else:savez(path,**z)
            c['split_files'][str(seed)]=sha(path)
            replay.append(dict(split=seed,train_words=len(X),inference_words=len(Q),classes=8,word_overlap=0,verified=True))
        for name in ('full','common','expected'):
            g[name]=g[name].copy();g[name]['source_word']=g[name].word
            frame(root/'inputs'/(name+'.parquet'),g[name])
        writej(root/'inputs/geometry.json',dict(edges=g['edges'],neighbors=g['neighbors'],split_audit=g['split_audit']))
        units=g['full'][['word','true_class']].sort_values('word').reset_index(drop=True);folds=[]
        for rep in range(5):
            for f,(_,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=rep).split(units,units.true_class)):
                folds.extend(dict(word=w,repeat=rep,fold=f) for w in units.iloc[b].word)
        frame(root/'inputs/cv_folds.parquet',pd.DataFrame(folds));frame(root/'split_replay.csv',pd.DataFrame(replay))
        prepare_occurrences(c,E)
        writej(root/'source_stats.json',c['source_stats']);writej(root/'split_files.json',c['split_files'])
        frame(root/'betas.csv',pd.DataFrame(dict(beta_index=np.arange(98),beta=BETAS)))
    print('P20 ready:',root,flush=True)
    return c

def prepare_occurrences(c,E):
    cfg=c['cfg'];d=cfg.dataset_dir;folder=c['root']/'inputs'/'apo';iid=digest(dict(run=c['run_id'],stage='CANONICAL_APO'))
    checked_source(c,d/'dataset_manifest.json',DATASET_SHA);manifest=readj(d/'dataset_manifest.json')
    req(manifest.get('status')=='COMPLETE','Dataset export is incomplete.')
    for n in ('units.parquet','membership.parquet','prototypes_RAW_float64.npy'):
        checked_source(c,d/n,manifest['files'][n]['sha256'])
    source=cfg.occurrence_dir/'contextual_occurrences_definitive.parquet';checked_source(c,source,OCC_SHA)
    indexpath=cfg.occurrence_dir/'word_row_group_index.csv';checked_source(c,indexpath)
    if cached(folder,iid) is not None:
        print('Canonical A/P/O inputs verified and reused.',flush=True);return
    guard(c,2.);words=E.index.tolist();units=pd.read_parquet(d/'units.parquet')
    req(np.array_equal(units.unit_index,np.arange(len(units))) and not units.unit_id.duplicated().any(),'Noncanonical unit inventory.')
    u=units.loc[units.word.isin(words)].sort_values(['word','sense_id']).reset_index(drop=True)
    req(set(u.word)==set(words) and u.groupby('word').support.sum().eq(100).all(),'Expected all 100 positions for every S word.')
    raw=np.load(d/'prototypes_RAW_float64.npy',mmap_mode='r',allow_pickle=False)
    req(raw.dtype==np.float64 and raw.shape==(len(units),768),'Frozen prototype schema mismatch.')
    P=np.asarray(raw[u.unit_index],dtype=np.float64).copy();members=[];selected=set(u.unit_index)
    for batch in pq.ParquetFile(d/'membership.parquet').iter_batches(batch_size=65536):
        guard(c);q=batch.to_pandas();q=q.loc[q.unit_index.isin(selected)]
        if len(q):members.append(q)
    members=pd.concat(members,ignore_index=True).sort_values('occ_id').reset_index(drop=True)
    byidx=u.set_index('unit_index')
    req(len(members)==64000 and not members.occ_id.duplicated().any(),'Canonical positions missing or duplicated.')
    req(members.word.eq(members.unit_index.map(byidx.word)).all() and members.sense_id.eq(members.unit_index.map(byidx.sense_id)).all(),'Membership identity mismatch.')
    req(np.array_equal(members.groupby('unit_index').size().reindex(u.unit_index),u.support),'Prototype multiplicities differ.')
    ix=pd.read_csv(indexpath,keep_default_na=False).set_index('word',verify_integrity=True).loc[words]
    req(ix.n_occurrences.eq(100).all() and (ix.row_end_exclusive-ix.row_start).eq(100).all(),'Noncanonical occurrence index.')
    parts=[];vectors=[];cols=['word','occ_id','source_doc_key','word_pos_original','source_name','source_split','source_row_idx','embedding']
    with pq.ParquetFile(source) as pf:
        starts=np.r_[0,np.cumsum([pf.metadata.row_group(j).num_rows for j in range(pf.num_row_groups)])]
        groups=list(ix.groupby('row_group_id'))
        for num,(rg,rows) in enumerate(groups):
            guard(c);rg=int(rg);req(0<=rg<pf.num_row_groups,'Invalid row group.')
            tab=pf.read_row_group(rg,columns=cols,use_threads=False)
            req(np.array_equal(tab['occ_id'].to_numpy(),np.arange(starts[rg],starts[rg+1])),'Occurrence IDs not canonical.')
            tab=tab.filter(pc.is_in(tab['word'],value_set=pyarrow.array(rows.index.tolist())))
            a=tab['embedding'].combine_chunks()
            req(pyarrow.types.is_fixed_size_list(a.type) and a.type.list_size==768 and pyarrow.types.is_float32(a.type.value_type),'RAW occurrence schema differs.')
            req(a.null_count==0 and a.values.null_count==0,'Null embedding.')
            v=a.values.to_numpy(zero_copy_only=False).reshape(len(a),768).astype(np.float64)
            meta=tab.drop(['embedding']).to_pandas()
            req(not meta.isna().any().any() and np.isfinite(v).all() and (np.linalg.norm(v,axis=1)>0).all(),'Invalid RAW input.')
            for word,g in meta.groupby('word'):
                r=ix.loc[word];req(np.array_equal(np.sort(g.occ_id),np.arange(r.row_start,r.row_end_exclusive)),'Occurrence range mismatch: '+word)
            parts.append(meta);vectors.append(v)
            if num%20==0:print(f'Canonical row groups: {num+1}/{len(groups)}',flush=True)
    meta=pd.concat(parts,ignore_index=True);O=np.concatenate(vectors);del vectors
    order=np.argsort(meta.occ_id.to_numpy(),kind='stable');meta=meta.iloc[order].reset_index(drop=True);O=np.ascontiguousarray(O[order])
    req(np.array_equal(meta.occ_id,members.occ_id) and np.array_equal(meta.word,members.word),'Membership/source alignment mismatch.')
    req(not meta.duplicated(['word','source_doc_key','word_pos_original']).any(),'Duplicate lexical occurrence identity.')
    keys=meta.source_name.astype(str)+':'+meta.source_split.astype(str)+':'+meta.source_row_idx.astype(str)
    req(keys.eq(meta.source_doc_key.astype(str)).all() and meta.word_pos_original.ge(0).all(),'Invalid source provenance.')
    proto_map={int(v):i for i,v in enumerate(u.unit_index)};word_map={w:i for i,w in enumerate(words)}
    meta['p_index']=members.unit_index.map(proto_map).to_numpy();meta['a_index']=meta.word.map(word_map)
    meta['sense_id']=members.sense_id.to_numpy();meta['unit_id']=members.unit_index.map(byidx.unit_id)
    A=E.to_numpy(dtype=np.float64);checks=[];diagnostics=[]
    for p,ids in meta.groupby('p_index').indices.items():
        avg=O[ids].mean(0);delta=float(abs(avg-P[p]).max())
        req(np.allclose(avg,P[p],rtol=1e-12,atol=1e-12),'O mean does not reproduce frozen P.')
        checks.append(dict(kind='prototype',name=u.iloc[p].unit_id,n=len(ids),max_abs_delta=delta))
    for ai,w in enumerate(words):
        ids=np.flatnonzero(meta.a_index.to_numpy()==ai);sel=u.word.eq(w).to_numpy()
        avg=O[ids].mean(0);weighted=np.average(P[sel],axis=0,weights=u.loc[sel,'support'])
        req(len(ids)==100 and np.allclose(A[ai],avg,rtol=1e-12,atol=1e-12),'S word mean differs from canonical occurrences: '+w)
        req(np.allclose(A[ai],weighted,rtol=1e-12,atol=1e-12),'Weighted pseudo-sense mean differs from A: '+w)
        checks.append(dict(kind='word',name=w,n=100,max_abs_delta=float(max(abs(avg-A[ai]).max(),abs(weighted-A[ai]).max()))))
        diagnostics.append(dict(word=w,prototypes=int(sel.sum()),occurrences=100,norm_A=float(np.linalg.norm(A[ai])),
            mean_norm_P=float(np.average(np.linalg.norm(P[sel],axis=1),weights=u.loc[sel,'support'])),
            mean_norm_O=float(np.linalg.norm(O[ids],axis=1).mean()),
            variance_P=float(np.average(np.square(P[sel]-A[ai]).sum(1),weights=u.loc[sel,'support'])),
            variance_O=float(np.square(O[ids]-A[ai]).sum(1).mean())))
    put_npy(folder/'O.npy',O);put_npy(folder/'P.npy',P);put_npy(folder/'A.npy',A)
    frame(folder/'occurrences.parquet',meta);frame(folder/'units.parquet',u)
    writej(folder/'words.json',words);frame(folder/'means_verified.csv',pd.DataFrame(checks))
    frame(folder/'representation_diagnostics.csv',pd.DataFrame(diagnostics))
    overlap=[]
    for seed in range(51):
        z=split_arrays(c,seed);L=meta.loc[meta.word.isin(z['train_words'])];I=meta.loc[meta.word.isin(z['query_words'])]
        overlap.append(dict(split=seed,word_overlap=len(set(L.word)&set(I.word)),occurrence_overlap=len(set(L.occ_id)&set(I.occ_id)),
            source_record_overlap=len(set(L.source_doc_key)&set(I.source_doc_key)),document_disjointness_claim=False))
    frame(folder/'role_overlap.csv',pd.DataFrame(overlap))
    seal(folder,iid,['O.npy','P.npy','A.npy','occurrences.parquet','units.parquet','words.json','means_verified.csv','role_overlap.csv','representation_diagnostics.csv'],
        dict(words=640,occurrences=64000,prototypes=len(P),classes=8,max_abs_mean_delta=max(r['max_abs_delta'] for r in checks)))
    print('All canonical positions and weighted mean identities verified.',flush=True)

def split_arrays(c,seed):
    p=c['root']/'inputs'/f's{seed:02d}.npz'
    req(sha(p)==c['split_files'][str(seed)],'Split input changed.')
    return arrays(p)

_DATA={}
def clear_data():
    _DATA.clear();gc.collect()

def representations(c,seed,memory):
    key=(str(c['root']),seed,memory)
    if key in _DATA:return _DATA[key]
    clear_data();z=split_arrays(c,seed);folder=c['root']/'inputs/apo'
    words=readj(folder/'words.json');pos={w:i for i,w in enumerate(words)}
    A=np.load(folder/'A.npy',mmap_mode='r');P=np.load(folder/'P.npy',mmap_mode='r');O=np.load(folder/'O.npy',mmap_mode='r')
    u=pd.read_parquet(folder/'units.parquet');meta=pd.read_parquet(folder/'occurrences.parquet')
    L=z['train_words'];I=z['query_words'];li=np.array([pos[w] for w in L]);qi=np.array([pos[w] for w in I])
    pl=np.flatnonzero(u.word.isin(L));pi=np.flatnonzero(u.word.isin(I))
    ol=np.flatnonzero(meta.word.isin(L));oi=np.flatnonzero(meta.word.isin(I))
    req(len(ol)==100*len(L) and len(oi)==100*len(I) and not set(ol)&set(oi),'Occurrence-role mismatch.')
    if memory=='A':X=np.array(A[li]);multiplicity=np.full(len(li),100,np.int64)
    elif memory=='P':X=np.array(P[pl]);multiplicity=u.iloc[pl].support.to_numpy(np.int64)
    elif memory=='O':X=np.array(O[ol]);multiplicity=np.ones(len(ol),np.int64)
    else:raise ValueError(memory)
    req(multiplicity.sum()==100*len(L),'Memory positions differ across arms.')
    rho=float(np.median(np.linalg.norm(A[li],axis=1)));norm=np.linalg.norm(X,axis=1)
    req((norm>0).all() and rho>0,'Zero norm in memory.')
    # Unique inference representations; maps recover each canonical occurrence prediction.
    Q=np.concatenate([A[li],A[qi],P[pi],O[oi]])
    sl=dict(A=(len(L),len(L)+len(I)),P=(len(L)+len(I),len(L)+len(I)+len(pi)),
            O=(len(L)+len(I)+len(pi),len(Q)))
    qword={w:i for i,w in enumerate(I)};ppos={int(p):i for i,p in enumerate(pi)}
    om=meta.iloc[oi].copy().reset_index(drop=True)
    maps=dict(A=om.word.map(qword).to_numpy(np.int32),P=om.p_index.map(ppos).to_numpy(np.int32),O=np.arange(len(oi),dtype=np.int32))
    req(all((v>=0).all() for v in maps.values()),'Incomplete canonical query map.')
    a=dict(X=X,keys=X*(rho/norm)[:,None],multiplicity=multiplicity,Q=Q,L=L,I=I,y=z['y'],truth=z['truth'],
        rho=rho,slices=sl,maps=maps,query_word_index=maps['A'],occ_ids=om.occ_id.to_numpy(np.int64),n_train=len(L),
        n_query=len(Q),dimension=Q.shape[1],query_words=I)
    _DATA[key]=a;return a

def calibration_location(c,seed):
    return c['root']/'cal'/f's{seed:02d}',digest(dict(run=c['run_id'],stage='SHARED_A_CALIBRATION',split=seed))

def calibrate(c,seed):
    folder,iid=calibration_location(c,seed);old=cached(folder,iid)
    if old is not None:return np.array(old['scales'],float)
    guard(c,.5);z=split_arrays(c,seed);X=z['X'];y=z['y'];rho=float(np.median(np.linalg.norm(X,axis=1)))
    raw=(X*(rho/np.linalg.norm(X,axis=1))[:,None])@X.T
    raw=raw.T;m0=np.full(len(X),100,np.int64);s,o,m=numeric.prepare_scores(raw,m0);pref=numeric.prefixes(s,m)
    target=compute.curve((s,o,m,pref),y,0,1.);rows=[];traces=[];scales=[];curves=[]
    with threadpool_limits(limits=1):
        for ni,n in enumerate(NETS):
            seen={}
            def objective(loglam):
                guard(c);lam=float(np.exp(loglam));key=lam.hex()
                if key not in seen:
                    q=compute.curve((s,o,m,pref),y,ni,lam)
                    seen[key]=(lam,float(np.square(np.log(q)-np.log(target)).mean()),q)
                return seen[key][1]
            before=objective(0.)
            if ni:
                objective(np.log(.001));opt=minimize_scalar(objective,bounds=(np.log(.001),0.),method='bounded',options=dict(xatol=.015,maxiter=32))
                req(opt.success,'HFY calibration did not converge; no automatic parameter changes.')
            lam,loss,q=min(seen.values(),key=lambda v:(v[1],abs(np.log(v[0]))))
            req(loss<=before+1e-12,'Calibration worsened the lambda=1 reference.')
            anchors=[]
            for bi in (0,18,97):
                w=numeric.reference_weights(raw[:2],m0,BETAS[bi]*lam,ni)
                f=1/(m0.sum()*(w*w/m0).sum(1));f1=numeric.fraction(s[:2],m[:2],[p[:2] for p in pref],BETAS[bi]*lam,ni)
                anchors.append(float(abs(f-f1).max()))
            req(max(anchors)<1e-8,'Multiplicity-aware calibration anchor failed.')
            scales.append(lam);curves.append(q)
            rows.append(dict(network=n,split=seed,lambda_value=lam,rho=rho,TRAIN_words=len(X),memory_positions=int(m0.sum()),
                loss_before=before,loss_after=loss,max_anchor_delta=max(anchors),uses_validation=False,uses_errors=False))
            traces.extend(dict(network=n,lambda_value=v[0],objective=v[1]) for v in seen.values())
    frame(folder/'calibration.csv',pd.DataFrame(rows));frame(folder/'trace.csv',pd.DataFrame(traces))
    savez(folder/'curves.npz',target=target,curves=np.array(curves),betas=BETAS)
    seal(folder,iid,['calibration.csv','trace.csv','curves.npz'],dict(scales=scales,rho=rho,shared_by_all_arms=True))
    return np.array(scales)

def trial_location(c,memory,seed,ni,bi,r):
    folder=c['root']/'trials'/memory/f's{seed:02d}'/f'n{ni}'/f'b{bi:02d}'
    return folder/f'r{r:02d}.npz',digest(dict(run=c['run_id'],memory=memory,split=seed,network=ni,beta_index=bi,seed=r))

def load_trial(path,iid):
    if not path.exists():return None
    z=arrays(path);req(str(z.pop('identity'))==iid,'Trial identity mismatch: '+str(path))
    want=str(z.pop('payload_sha'))
    got=hashlib.sha256(b''.join(np.ascontiguousarray(z[k]).tobytes() for k in sorted(z))).hexdigest()
    req(want==got,'Trial payload altered: '+str(path))
    for q in 'APO':req(z[q].dtype==np.int8 and ((z[q]>=0)&(z[q]<8)).all(),'Invalid prediction labels.')
    return z

def save_trial(path,iid,payload):
    h=hashlib.sha256(b''.join(np.ascontiguousarray(payload[k]).tobytes() for k in sorted(payload))).hexdigest()
    savez(path,identity=np.array(iid),payload_sha=np.array(h),**payload)

def scratch(c,memory,seed):return c['root']/'tmp'/memory/f's{seed:02d}'

def score_location(c,memory,seed,left,right):
    return scratch(c,memory,seed)/'scores'/f'{left:06d}_{right:06d}',digest(dict(run=c['run_id'],memory=memory,split=seed,stage='SCORES',left=left,right=right))

def score_task(c,memory,seed,left,right):
    folder,iid=score_location(c,memory,seed,left,right)
    old=cached(folder,iid)
    if old is not None:return old
    guard(c);a=representations(c,seed,memory)
    raw=a['Q'][left:right]@a['keys'].T
    s,o,m=numeric.prepare_scores(raw,a['multiplicity'])
    put_npy(folder/'s.npy',s);put_npy(folder/'order.npy',o.astype(np.int32))
    row=dict(left=left,right=right,memory=memory,split=seed,n_positions=int(a['multiplicity'].sum()))
    seal(folder,iid,['s.npy','order.npy'],row);return row

def recall_location(c,memory,seed,indices,left,right):
    group='b'+digest(list(indices))[:10]
    return scratch(c,memory,seed)/group/f'{left:06d}_{right:06d}',digest(dict(run=c['run_id'],memory=memory,split=seed,
        stage='RECALL',indices=indices,left=left,right=right))

def recall_task(c,memory,seed,indices,scales,left,right):
    folder,iid=recall_location(c,memory,seed,indices,left,right)
    old=cached(folder,iid)
    if old is not None:return old
    guard(c);a=representations(c,seed,memory);sf,si=score_location(c,memory,seed,left,right)
    req(cached(sf,si) is not None,'Scores incomplete.')
    s=np.load(sf/'s.npy');order=np.load(sf/'order.npy');m=a['multiplicity'][order].astype(float);pref=numeric.prefixes(s,m)
    R=np.empty((3,len(indices),right-left,a['dimension']),np.float64);delta=0.
    for ni in range(3):
        for j,bi in enumerate(indices):
            guard(c);gain=float(BETAS[bi]*scales[ni]);v=numeric.recall_sorted(s,order,m,pref,gain,ni,a['X'])
            req(np.isfinite(v).all(),'Nonfinite recovered vector.');R[ni,j]=v
            if left==0 and j in (0,len(indices)-1):
                raw=a['Q'][left:left+2]@a['keys'].T
                ref=numeric.reference_weights(raw,a['multiplicity'],gain,ni)@a['X']
                err=float(abs(ref-v[:2]).max());delta=max(delta,err)
                req(np.allclose(ref,v[:2],rtol=1e-8,atol=1e-9),'Independent recall anchor failed.')
    put_npy(folder/'R.npy',R)
    row=dict(memory=memory,split=seed,left=left,right=right,indices=list(indices),anchor_delta=delta)
    seal(folder,iid,['R.npy'],row);return row

def merge_recalls(c,memory,seed,indices,blocks):
    group=scratch(c,memory,seed)/('b'+digest(list(indices))[:10]);dest=group/'merged'
    iid=digest(dict(run=c['run_id'],memory=memory,seed=seed,indices=indices,stage='MERGED_RECALL'))
    if cached(dest,iid) is not None:return dest
    a=representations(c,seed,memory);dest.mkdir(parents=True,exist_ok=True);names=[]
    for ni in range(3):
        for j,bi in enumerate(indices):
            guard(c);name=f'n{ni}_b{bi:02d}.npy';tmp=dest/(name+'.partial.npy')
            out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float64,shape=(a['n_query'],a['dimension']))
            for left,right in blocks:
                folder,ident=recall_location(c,memory,seed,indices,left,right)
                # Receipts were verified immediately before merge by the worker scheduler.
                req((folder/'done.json').is_file(),'Missing recall block.')
                q=np.load(folder/'R.npy',mmap_mode='r');out[left:right]=q[ni,j];del q
            out.flush();del out;os.replace(tmp,dest/name);names.append(name)
    seal(dest,iid,names,dict(indices=list(indices)));return dest

def vote_once(model,T,block=256):
    classes=model.ensemble_classifier.le_.classes_;pred=[];tie=[]
    for left in range(0,len(T),block):
        v=model.individual_votes(T[left:left+block]);counts=np.stack([(v==cl).sum(axis=1) for cl in classes],axis=1)
        mx=counts.max(1);pred.append(classes[counts.argmax(1)]);tie.append((counts==mx[:,None]).sum(1)>1)
    return np.concatenate(pred).astype(np.int8),np.concatenate(tie)

def classification_task(c,memory,seed,indices,ni,bi,r):
    path,iid=trial_location(c,memory,seed,ni,bi,r)
    if load_trial(path,iid) is not None:return dict(new=0)
    guard(c);a=representations(c,seed,memory)
    group=scratch(c,memory,seed)/('b'+digest(list(indices))[:10])/'merged'
    R=np.load(group/f'n{ni}_b{bi:02d}.npy',mmap_mode='r');n=a['n_train'];start=time.perf_counter()
    model=core.MulticlassEnsembleClassifier(pd.DataFrame(R[:n],index=a['L']),c['geometries']['S']['neighbors'],
        random_seed_cls=r,n_jobs=1,voting_n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit_only();pred,ties=vote_once(model,R[n:])
    anchors=np.unique([0,a['slices']['P'][0]-n,a['slices']['O'][0]-n,len(pred)-1])
    req(np.array_equal(pred[anchors],model.classify(R[n:][anchors])),'Hard-vote disagrees with sklearn.')
    payload={}
    for q,(lo,hi) in a['slices'].items():
        payload[q]=pred[lo-n:hi-n];payload[q+'_tie']=ties[lo-n:hi-n]
    payload['warnings']=np.array(json.dumps([dict(category=w.category.__name__,message=str(w.message)) for w in caught],ensure_ascii=False))
    payload['seconds']=np.array(time.perf_counter()-start)
    save_trial(path,iid,payload)
    return dict(new=1,memory=memory,split=seed,network=ni,beta_index=bi,seed=r,seconds=float(payload['seconds']))

def light(c):
    return {k:c[k] for k in ('cfg','root','cache','run_id','kernel_id','split_files','source_stats','geometries')}

def run(c,pilot=False):
    """Pilot: split 0, beta 0, seed 0, all 3 memories/networks; useful official trials."""
    with stage(c,'PILOT' if pilot else 'CONTROLLED_TRIALS'):
        # Deterministic schedule; each completed ensemble is atomic and reusable.
        for seed in ([0] if pilot else range(51)):
            scales=calibrate(c,seed)
            for memory in 'APO':
                guard(c);clear_data();a=representations(c,seed,memory)
                seeds=[0] if pilot else list(range(10));bis=[0] if pilot else list(range(98))
                pending=[bi for bi in bis if any(load_trial(*trial_location(c,memory,seed,ni,bi,r)) is None for ni in range(3) for r in seeds)]
                if not pending:
                    print(f'{memory} split {seed}: requested trials complete.',flush=True);continue
                blocks=[(l,min(l+c['cfg'].query_block,a['n_query'])) for l in range(0,a['n_query'],c['cfg'].query_block)]
                N=len(a['X']);D=a['dimension'];nq=a['n_query'];chunk=1 if pilot else c['cfg'].beta_chunk
                need_disk=(nq*N*12+2*3*chunk*nq*D*8)/2**30+c['cfg'].reserve_disk_gib
                req(shutil.disk_usage(c['root']).free/2**30>=need_disk,f'Need approximately {need_disk:.1f} GiB free for this memory/split temporary cache.')
                need=max(1.,(N*D*16+nq*D*8+c['cfg'].query_block*N*8*13)/2**30+.6)
                wc=light(c)
                maps=c['root']/'maps'/f's{seed:02d}.npz'
                mdata=dict(query_words=a['I'],occ_ids=a['occ_ids'],truth=a['truth'],word_index=a['query_word_index'],**a['maps'])
                if maps.exists():req(all(np.array_equal(arrays(maps)[k],v) for k,v in mdata.items()),'Query mapping changed across memories.')
                else:savez(maps,**mdata)
                clear_data()
                run_tasks(c,f'{memory} s{seed:02d} SCORES',score_task,[(wc,memory,seed,l,r) for l,r in blocks],need)
                for off in range(0,len(pending),chunk):
                    indices=tuple(pending[off:off+chunk]);start=time.perf_counter()
                    checks=run_tasks(c,f'{memory} s{seed:02d} RECALL {indices[0]}',recall_task,[(wc,memory,seed,indices,scales,l,r) for l,r in blocks],need)
                    frame(c['root']/'diagnostics'/f'{memory}_s{seed:02d}_{indices[0]:02d}_recall.csv',pd.DataFrame(checks))
                    merge_recalls(c,memory,seed,indices,blocks);clear_data()
                    tasks=[(wc,memory,seed,indices,ni,bi,r) for bi in indices for ni in range(3) for r in seeds
                        if load_trial(*trial_location(c,memory,seed,ni,bi,r)) is None]
                    rows=run_tasks(c,f'{memory} s{seed:02d} CLASSIFY {indices[0]}',classification_task,tasks,max(1.5,need))
                    clear_data();frame(c['root']/'timing'/f'{memory}_s{seed:02d}_b{indices[0]:02d}_{time.time_ns()}.csv',pd.DataFrame(rows))
                    shutil.rmtree(scratch(c,memory,seed)/('b'+digest(list(indices))[:10]))
                    print(f'{memory} split {seed}: completed {len(indices)} beta(s), {time.perf_counter()-start:.1f}s including recall.',flush=True)
                if not pilot:
                    writej(c['root']/'trials'/memory/f's{seed:02d}'/'complete.json',dict(run_id=c['run_id'],ensembles=2940))
                    shutil.rmtree(scratch(c,memory,seed),ignore_errors=True)
        clear_data()
    return status(c)

def status(c):
    rows=[]
    for memory in 'APO':
        paths=list((c['root']/'trials'/memory).glob('s*/n*/b*/r*.npz'))
        rows.append(dict(memory=memory,completed_ensemble_files=len(paths),required_ensembles=149940,
                         percent=100*len(paths)/149940))
    print('Counts below are progress indicators. Aggregation verifies identities and payload hashes.')
    return pd.DataFrame(rows)

DIRECT_CONTRASTS=(('PP','AA'),('OO','AA'),('PA','AA'),('OA','AA'),('AP','AA'),('AO','AA'))
CONTRASTS={a+'-'+b:{a:1,b:-1} for a,b in DIRECT_CONTRASTS}
CONTRASTS.update({'P interaction':{'PP':1,'PA':-1,'AP':-1,'AA':1},
                  'O interaction':{'OO':1,'OA':-1,'AO':-1,'AA':1}})

def aggregate(c):
    """Complete balanced panel only; integer trial denominators remain explicit."""
    receipt=c['root']/'counts_complete.json'
    if receipt.exists():
        r=readj(receipt);req(r['run_id']==c['run_id'],'Counts belong to another run.')
        for name,h in r['files'].items():req(sha(c['root']/'counts'/name)==h,'Counts changed.')
        print('Verified complete counts reused.');return pd.DataFrame(r['summary'])
    with stage(c,'AGGREGATE_COMPLETE_PANEL'):
        full=c['geometries']['S']['full'].sort_values('word').reset_index(drop=True)
        words=full.word.tolist();pos={w:i for i,w in enumerate(words)};W=len(words)
        errors=np.zeros((9,3,W,98),np.int64);ties=np.zeros_like(errors);exposure=np.zeros((W,98),np.int64)
        changes=np.zeros((6,3,W,98,3),np.int64);last=time.monotonic();verified=0;ledger=[]
        for seed in range(51):
            guard(c);z=split_arrays(c,seed);maps_path=c['root']/'maps'/f's{seed:02d}.npz'
            req(maps_path.exists(),'Missing canonical query mapping. Run all trials first.')
            m=arrays(maps_path);qw=m['query_words'];wi=m['word_index'];idx=np.array([pos[w] for w in qw])
            req(np.array_equal(qw,z['query_words']) and np.array_equal(m['truth'],z['truth']),'Mapping identity mismatch.')
            req(len(m['occ_ids'])==100*len(qw) and len(np.unique(m['occ_ids']))==len(m['occ_ids']),'Invalid canonical positions.')
            req(np.array_equal(np.bincount(wi,minlength=len(qw)),np.full(len(qw),100)),'Word position counts differ.')
            truth=m['truth'][wi]
            for bi in range(98):
                exposure[idx,bi]+=10
                for ni in range(3):
                    for r in range(10):
                        err={}
                        for memory in 'APO':
                            path,iid=trial_location(c,memory,seed,ni,bi,r);p=load_trial(path,iid)
                            req(p is not None,f'Panel incomplete: {memory}, split={seed}, beta={bi}, network={ni}, seed={r}. Resume run(C).')
                            ledger.append(dict(path=path.relative_to(c['root']).as_posix(),sha256=sha(path)))
                            for q in 'APO':
                                arm=memory+q;ai=ARMS.index(arm);mp=m[q]
                                req(mp.max()<len(p[q]) and len(p[q])==len(p[q+'_tie']),'Prediction/map shape mismatch.')
                                e=p[q][mp]!=truth;err[arm]=e
                                errors[ai,ni,idx,bi]+=np.bincount(wi,weights=e,minlength=len(qw)).astype(np.int64)
                                ties[ai,ni,idx,bi]+=np.bincount(wi,weights=p[q+'_tie'][mp],minlength=len(qw)).astype(np.int64)
                            verified+=1
                        for ci,(a,b) in enumerate(DIRECT_CONTRASTS):
                            for k,v in enumerate((err[a]&~err[b],~err[a]&err[b],err[a]&err[b])):
                                changes[ci,ni,idx,bi,k]+=np.bincount(wi,weights=v,minlength=len(qw)).astype(np.int64)
                    if time.monotonic()-last>20:
                        print(f'Authenticated ensembles: {verified:,}/449,820',flush=True);guard(c);last=time.monotonic()
        req(verified==449820 and (exposure>0).all(),'Incomplete exposure panel.')
        expected=c['geometries']['S']['expected'].groupby('word').size().reindex(words).to_numpy()*10
        req(np.array_equal(exposure,np.repeat(expected[:,None],98,axis=1)),'Exposure mismatch with frozen S.')
        base=pd.DataFrame(dict(word=np.repeat(words,98),source_word=np.repeat(words,98),
            true_class=np.repeat(full.true_class.to_numpy(),98),beta=np.tile(BETAS,W),beta_key=np.tile(np.round(BETAS,12),W),
            validation_similarity=np.repeat(full.validation_similarity.to_numpy(),98),n_items=exposure.ravel(),n_trials=100*exposure.ravel()))
        hashes={};summary=[]
        for ai,arm in enumerate(ARMS):
            for ni,n in enumerate(NETS):
                d=base.copy();d['n_errors']=errors[ai,ni].ravel();d['n_ties']=ties[ai,ni].ravel();d['rate']=d.n_errors/d.n_trials
                name=f'{arm}_n{ni}.parquet';frame(c['root']/'counts'/name,d);hashes[name]=sha(c['root']/'counts'/name)
                summary.append(dict(arm=arm,network=n,words=W,rows=len(d),occurrence_opportunities=int(d.n_trials.sum()),
                    word_exposures=int(d.n_items.sum()),mean_word_error=float(d.groupby('word').rate.mean().mean())))
        for ci,(a,b) in enumerate(DIRECT_CONTRASTS):
            for ni,n in enumerate(NETS):
                d=base.copy();v=changes[ci,ni].reshape(-1,3)
                d['new_errors']=v[:,0];d['fixed_errors']=v[:,1];d['both_errors']=v[:,2]
                d['both_correct']=d.n_trials-v.sum(axis=1);d['paired_delta_rate']=(d.new_errors-d.fixed_errors)/d.n_trials
                req((d.both_correct>=0).all(),'Invalid paired transition counts.')
                req(np.allclose(d.paired_delta_rate,(errors[ARMS.index(a),ni]-errors[ARMS.index(b),ni]).ravel()/d.n_trials,rtol=0,atol=1e-15),'Paired rate identity failed.')
                name=f'paired_{a}_{b}_n{ni}.parquet';frame(c['root']/'counts'/name,d);hashes[name]=sha(c['root']/'counts'/name)
        frame(c['root']/'verified_trial_ledger.parquet',pd.DataFrame(ledger));frame(c['root']/'counts_summary.csv',pd.DataFrame(summary))
        writej(receipt,dict(run_id=c['run_id'],files=hashes,verified_ensembles=verified,summary=summary,
            trial_ledger_sha=sha(c['root']/'verified_trial_ledger.parquet'),positions_are_independent=False))
    return pd.DataFrame(summary)

def empirical(c):
    import e6_p20_stats as st
    req((c['root']/'counts_complete.json').is_file(),'Aggregate the complete trial panel first.')
    rows=[]
    with stage(c,'EMPIRICAL_REPORT'):
        for arm in ARMS:
            a=st.load_analysis(c,arm)
            for ni,n in enumerate(NETS):
                for case in [(),*[(k,) for k in range(8)]]:
                    label='FULL' if not case else f'L{case[0]:02d}';s=st.surface(a,a['Y'][:,ni],case)
                    frame(c['root']/'empirical'/arm/f'n{ni}_{label}.csv',s,index=True)
                    rows.append(dict(arm=arm,network=n,case=label,omitted_label='' if not case else a['labels'][case[0]],
                        **st.trends(s)))
        result=pd.DataFrame(rows);frame(c['root']/'empirical_trends.csv',result)
        contrasts=paired_trends(result,scopes=('',),keys=['network','case','omitted_label'])
        frame(c['root']/'empirical_contrasts.csv',contrasts)
        writej(c['root']/'empirical_complete.json',dict(run_id=c['run_id'],files={n:sha(c['root']/n) for n in ('empirical_trends.csv','empirical_contrasts.csv')}))
    return result

def paired_trends(table,scopes,keys):
    rows=[]
    for kval,d in table.groupby(keys,sort=False,dropna=False):
        if not isinstance(kval,tuple):kval=(kval,)
        req(set(d.arm)==set(ARMS) and not d.arm.duplicated().any(),'Contrasts require all nine arms with identical keys.')
        v=d.set_index('arm')
        for scope in scopes:
            prefix=scope+'_' if scope else ''
            for name,terms in CONTRASTS.items():
                row=dict(zip(keys,kval));row.update(contrast=name,scope=scope or 'empirical')
                for k,sign in [('T1',1),('T2',-1),('T3',-1)]:
                    row[k+'_delta']=float(sum(coef*v.loc[arm,prefix+k] for arm,coef in terms.items()))
                    row[k+'_signed_improvement']=sign*row[k+'_delta']
                    if len(terms)==2:
                        target=next(a for a in terms if terms[a]==1);ref=next(a for a in terms if terms[a]==-1)
                        row[k+'_reference']=float(v.loc[ref,prefix+k]);row[k+'_target']=float(v.loc[target,prefix+k])
                        row[k+'_reference_pass']=bool(sign*v.loc[ref,prefix+k]>0);row[k+'_target_pass']=bool(sign*v.loc[target,prefix+k]>0)
                rows.append(row)
    return pd.DataFrame(rows)

def statistics(c):
    import e6_p20_stats as st
    for arm in ARMS:
        a=st.load_analysis(c,arm);st.run_oof(a);st.run_refits(a)
    with stage(c,'STATISTICAL_CONTRASTS'):
        fits=pd.concat([readcsv(c['root']/'analysis'/arm/'FULL_LOCO.csv') for arm in ARMS],ignore_index=True)
        oof=pd.concat([readcsv(c['root']/'analysis'/arm/'OOF_metrics.csv') for arm in ARMS],ignore_index=True)
        robust=st.summarize_robustness(fits)
        contrasts=paired_trends(fits,('empirical','matched5','grid30'),['model','candidate','network','case','omitted_label'])
        for name,d in [('statistical_FULL_LOCO',fits),('OOF_metrics',oof),('robustness',robust),('statistical_contrasts',contrasts)]:
            frame(c['root']/(name+'.csv'),d)
        writej(c['root']/'statistics_complete.json',dict(run_id=c['run_id'],fixed_models=MODELS,CV_tasks=1350,refits=486,
            files={n+'.csv':sha(c['root']/(n+'.csv')) for n in ('statistical_FULL_LOCO','OOF_metrics','robustness','statistical_contrasts')}))
    return contrasts

def show(c,include_models=True):
    """Always render PNG bytes in notebook output; independent of Matplotlib backend."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from IPython.display import display,Image
    cmap=LinearSegmentedColormap.from_list('thesis',['#6A0DAD','#B755C1','#D890D1','#F2C2A3','#F9E2B7','#FFFF00'])
    out=c['root']/'figures';out.mkdir(exist_ok=True)
    def emit(fig,name):
        fig.savefig(out/(name+'.png'),dpi=c['cfg'].figure_dpi,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        if c['cfg'].show_figures:display(Image(filename=str(out/(name+'.png'))))
        plt.close(fig)
    def read_surface(p):
        s=pd.read_csv(p,index_col=0);s.index=s.index.astype(float);s.columns=s.columns.astype(float);return s
    for ni,n in enumerate(NETS):
        surfaces={arm:read_surface(c['root']/'empirical'/arm/f'n{ni}_FULL.csv') for arm in ARMS}
        for restricted in (True,False):
            mask=BETAS<=.10+1e-12 if restricted else np.ones(98,bool)
            fig,axs=plt.subplots(3,3,figsize=(16,8),sharex=True,sharey=True,layout='constrained')
            for ax,arm in zip(axs.flat,ARMS):
                s=surfaces[arm];im=ax.pcolormesh(BETAS[mask],s.index,s.to_numpy()[:,mask],shading='nearest',cmap=cmap,vmin=0,vmax=1)
                ax.set_title(f'Memory {arm[0]} / Query {arm[1]}');ax.set_xlabel(r'$\beta$')
            for ax in axs[:,0]:ax.set_ylabel('Within-class similarity (z)')
            fig.colorbar(im,ax=axs,label='Error rate');fig.suptitle(n+' — controlled representations')
            emit(fig,f'n{ni}_nine_arms_'+('restricted' if restricted else 'full'))
        fig,axs=plt.subplots(1,2,figsize=(14,3.5),sharey=True,layout='constrained')
        ds=[surfaces[a].to_numpy()-surfaces['AA'].to_numpy() for a in ('PP','OO')];lim=max(max(abs(d).max() for d in ds),1e-9)
        for ax,arm,d in zip(axs,('PP','OO'),ds):
            im=ax.pcolormesh(BETAS,surfaces['AA'].index,d,shading='nearest',cmap='RdBu_r',vmin=-lim,vmax=lim)
            ax.set_title(arm+' − AA');ax.set_xlabel(r'$\beta$')
        axs[0].set_ylabel('Within-class similarity (z)');fig.colorbar(im,ax=axs,label='Change in error rate');fig.suptitle(n)
        emit(fig,f'n{ni}_paired_surface_difference')
        fig,axs=plt.subplots(1,2,figsize=(12,3.5),layout='constrained')
        for arm,color in [('AA','#6A0DAD'),('PP','#B755C1'),('OO','#157E8C')]:
            s=surfaces[arm];x=s.index.to_numpy();slope=(x-x.mean())@s.to_numpy()/np.square(x-x.mean()).sum()
            axs[0].plot(BETAS,s.mean(axis=0),label=arm,color=color);axs[1].plot(BETAS,slope,label=arm,color=color)
        axs[0].set_ylabel('Mean error rate');axs[1].set_ylabel('Similarity slope');axs[1].axhline(0,color='gray',lw=.8)
        for ax in axs:ax.set_xlabel(r'$\beta$');ax.legend()
        fig.suptitle(n);emit(fig,f'n{ni}_error_and_similarity_slope')
    ep=readcsv(c['root']/'empirical_trends.csv');display(ep.loc[ep.case.eq('FULL'),['arm','network','T1','T2','T3','T1T2']])
    display(readcsv(c['root']/'empirical_contrasts.csv').query("case == 'FULL'"))
    if include_models and (c['root']/'statistics_complete.json').exists():
        robust=readcsv(c['root']/'robustness.csv');display(robust)
        for ni,n in enumerate(NETS):
            for mi,model in enumerate(MODELS):
                fig,axs=plt.subplots(3,3,figsize=(16,8),sharex=True,sharey=True,layout='constrained')
                for ri,arm in enumerate(('AA','PP','OO')):
                    folder=c['root']/'analysis'/arm/'fits'/f'm{mi}'/f'n{ni}'/'FULL'
                    for ci,(name,title) in enumerate((('empirical','Empirical'),('matched5','Fitted at observed words'),('grid30','Fitted on common grid'))):
                        s=read_surface(folder/(name+'.csv'));ax=axs[ri,ci]
                        im=ax.pcolormesh(BETAS,s.index,s.to_numpy(),shading='nearest',cmap=cmap,vmin=0,vmax=1)
                        ax.set_title(arm+' — '+title);ax.set_xlabel(r'$\beta$')
                    axs[ri,0].set_ylabel('Within-class similarity (z)')
                fig.colorbar(im,ax=axs,label='Error rate');fig.suptitle(n+' — '+model['name'])
                emit(fig,f'n{ni}_m{mi}_empirical_and_fitted')
            fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for mi,model in enumerate(MODELS):
                d=robust.loc[(robust.candidate==core.candidate_name(model['family'],model['alpha']))&(robust.network==n)&(robust.scope=='grid30')].set_index('arm').loc[list(ARMS)]
                ax=axs[mi];x=np.arange(9)
                ax.bar(x-.17,d.T1_worst_signed_margin,.34,label='T1');ax.bar(x+.17,d.T2_worst_signed_margin,.34,label='T2')
                ax.axhline(0,color='gray',lw=.8);ax.set_xticks(x,ARMS);ax.set_title(model['name']);ax.set_ylabel('Worst signed LOCO margin');ax.legend()
            fig.suptitle(n);emit(fig,f'n{ni}_LOCO_margins')
    print('Figures and tables:',c['root'])
    print('A = word mean; P = frozen pseudo-sense; O = original occurrence. No independence claim for occurrence copies.')
    return out

def export_review(c):
    """Small review ZIP with scientific metadata, counts, contrasts and figures; excludes raw corpus and models."""
    req((c['root']/'counts_complete.json').is_file(),'Full counts are required for a review export.')
    target=c['root']/f'P20_Review_{c["run_id"][:12]}.zip';tmp=target.with_suffix('.partial.zip')
    chosen=[]
    for p in c['root'].rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(c['root']);parts=rel.parts
        if parts[0] in ('trials','tmp','cal','resources','timing'):continue
        if parts[0]=='inputs' and len(parts)>1 and parts[1]=='apo' and p.suffix in ('.npy',):continue
        if parts[0]=='inputs' and p.suffix=='.npz':continue
        if parts[0]=='analysis' and any(x in parts for x in ('CV','fits')):continue
        if p.suffix in ('.zip',) or p.name=='STOP':continue
        chosen.append((p,rel.as_posix()))
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in chosen:z.write(p,name)
        z.writestr('REVIEW_MANIFEST.json',json.dumps(dict(run_id=c['run_id'],statistics_complete=(c['root']/'statistics_complete.json').exists(),
            files={name:sha(p) for p,name in chosen}),indent=2))
    os.replace(tmp,target);print('Review ZIP:',target);return target

def finalize(c):
    """Verify the delivered scientific summaries without requiring every cache to remain."""
    for stage_name in ('counts','empirical','statistics'):
        receipt=c['root']/(stage_name+'_complete.json')
        req(receipt.exists(),'Pending stage: '+stage_name)
        r=readj(receipt);req(r['run_id']==c['run_id'],'Summary identity mismatch.')
        for name,h in r['files'].items():
            path=c['root']/('counts' if stage_name=='counts' else '')/name
            req(sha(path)==h,'Summary artifact altered: '+str(path))
    fits=readcsv(c['root']/'statistical_FULL_LOCO.csv');oof=readcsv(c['root']/'OOF_metrics.csv')
    req(len(fits)==486 and fits.stable.all() and fits.groupby(['arm','network','model']).size().eq(9).all(),'Incomplete FULL/LOCO refits.')
    req(len(oof)==270 and oof.groupby(['arm','network','model']).size().eq(5).all(),'Incomplete OOF summaries.')
    req(set(fits.arm)==set(ARMS) and set(fits.network)==set(NETS),'Missing experimental conditions.')
    report=dict(status='COMPLETE',run_id=c['run_id'],ensembles=449820,arms=9,words=640,classes=8,
        splits=51,classifier_seeds=10,beta_points=98,CV_fits=1350,full_and_LOCO_refits=486,
        empirical_cases=243,occurrence_copies_are_independent=False,
        effects='Conditional on fixed K80 BERT source, task, multiplicities, scales and observer recipe.',
        new_classifier_fit_per_memory=True,source_inputs_read_only=True)
    writej(c['root']/'FINAL_CHECK.json',report);print(json.dumps(report,indent=2));return report


class Monitor:
    @staticmethod
    def rss():
        try:return psutil.Process().memory_info().rss/2**30
        except (psutil.Error,OSError):
            try:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if sys.platform=='darwin' else 2**20)
            except (ImportError,OSError):return .8
    @staticmethod
    def private():
        try:return float(psutil.Process().memory_full_info().uss)/2**30
        except (psutil.Error,OSError,AttributeError):return float('nan')
    def __enter__(self):
        self.stop=threading.Event();self.peak=self.rss();self.peak_private=self.private();self.minimum=psutil.virtual_memory().available/2**30
        def sample():
            while not self.stop.wait(.25):
                self.peak=max(self.peak,self.rss());private=self.private()
                if np.isfinite(private):self.peak_private=max(self.peak_private,private) if np.isfinite(self.peak_private) else private
                self.minimum=min(self.minimum,psutil.virtual_memory().available/2**30)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()

def invoke(fn,args):
    with threadpool_limits(limits=1),Monitor() as m:r=fn(*args)
    return r,dict(peak_RSS_GiB=m.peak,peak_USS_GiB=m.peak_private,min_available_GiB=m.minimum)
def indexed_invoke(index,fn,args):
    result,measurement=invoke(fn,args)
    return index,result,measurement

def close_pool(kill=False):
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True,kill_workers=kill)

def run_tasks(c,label,fn,tasks,need=.6):
    tasks=list(tasks)
    if not tasks:return []
    close_pool();gc.collect();guard(c,need);start=time.perf_counter();last=0.;logs=[];results=[];failed=True
    print(f'{label}: {len(tasks)} tasks; first useful task measures RAM.',flush=True)
    try:
        r,m=invoke(fn,tasks[0]);results.append(r);ordered={0:r}
        observed=m['peak_USS_GiB'] if np.isfinite(m['peak_USS_GiB']) else m['peak_RSS_GiB']
        per=max(need,observed*1.3)
        free=psutil.virtual_memory().available/2**30-c['cfg'].reserve_gib
        jobs=max(1,min(c['cfg'].max_workers,os.cpu_count() or 1,max(len(tasks)-1,1),int(free/per)))
        req(len(tasks)==1 or free>=per,'RAM insuficiente para el siguiente proceso.')
        logs.append(dict(task=1,workers=1,elapsed=time.perf_counter()-start,**m))
        print(f'{label}: {jobs} workers maximum; BLAS=1, {per:.2f} GiB budget per worker.',flush=True)
        if len(tasks)>1:
            with parallel_config(backend='loky',inner_max_num_threads=1):
                stream=Parallel(n_jobs=jobs,return_as='generator_unordered',batch_size=1,pre_dispatch=jobs,max_nbytes='8M',temp_folder=str(c['root']/'tmp'))(delayed(indexed_invoke)(i,fn,args) for i,args in enumerate(tasks[1:],1))
                for i,r,m in stream:
                    ordered[i]=r;results.append(r);elapsed=time.perf_counter()-start;done=len(results)
                    # Parallel throughput excludes the initial serial RAM probe.
                    parallel_elapsed=max(elapsed-logs[0]['elapsed'],1e-6);eta=parallel_elapsed/max(done-1,1)*(len(tasks)-done)
                    logs.append(dict(task=done,workers=jobs,elapsed=elapsed,**m))
                    writej(Path(c.get('main_root',c['root']))/'progress.json',dict(stage=label,done=done,total=len(tasks),workers=jobs,elapsed_seconds=elapsed,eta_seconds_approx=eta,ETA_scope='pending tasks in this stage only'))
                    if elapsed-last>=20 or done==len(tasks):
                        print(f'{label}: {done}/{len(tasks)} | {elapsed/60:.1f} min | stage remaining approximately {eta/60:.1f} min',flush=True);last=elapsed
                    guard(c)
        failed=False
    finally:
        close_pool(kill=failed)
        frame(Path(c.get('main_root',c['root']))/'resources'/f'{digest(label)[:12]}_{time.time_ns()}.csv',pd.DataFrame(logs))
    # Scheduling is unordered; reductions consume results in canonical task order.
    return [ordered[i] for i in range(len(tasks))]

