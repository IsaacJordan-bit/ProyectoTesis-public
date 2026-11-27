"""Deterministic synthetic tests. Never reads or writes historical experiment data."""
from pathlib import Path
import tempfile, json, time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import e6_p20 as p
import e6_p20_stats as st

def test_multiplicity():
    rng=np.random.default_rng(19);max_delta=0.
    for m in [np.ones(7,int),np.full(7,100),np.array([1,3,17,40,2,99,6])]:
        X=rng.normal(size=(7,9));Q=rng.normal(size=(5,9));raw=Q@X.T
        s,o,ms=p.numeric.prepare_scores(raw,m);pref=p.numeric.prefixes(s,ms)
        expanded=np.repeat(raw,m,axis=1);V=np.repeat(X,m,axis=0)
        for ni,n in enumerate(p.NETS):
            for gain in (.00001,.01,.495,10.):
                ref=p.static.attention_weights((expanded*gain).T,n).T
                got=p.numeric.recall_sorted(s,o,ms,pref,gain,ni,X)
                delta=float(abs(got-ref@V).max());max_delta=max(max_delta,delta)
                np.testing.assert_allclose(got,ref@V,rtol=1e-8,atol=1e-9)
                f=p.numeric.fraction(s,ms,pref,gain,ni)
                np.testing.assert_allclose(f,1/(m.sum()*(ref*ref).sum(1)),rtol=1e-7,atol=1e-9)
    return dict(expanded_operator_max_abs_delta=max_delta)

def synthetic_context(root):
    rng=np.random.default_rng(20);words=[f'w{k}_{j}' for k in range(8) for j in range(4)]
    neighbors={f'class{k}':[f'w{k}_{j}' for j in range(4)] for k in range(8)}
    A0=rng.normal(size=(32,8));O=np.repeat(A0,100,axis=0)+rng.normal(scale=.2,size=(3200,8))
    A=O.reshape(32,100,8).mean(1);P=[];meta=[];units=[]
    for ai,w in enumerate(words):
        for si,(lo,hi) in enumerate(((0,30),(30,100))):
            pi=len(P);P.append(O[ai*100+lo:ai*100+hi].mean(0))
            units.append(dict(word=w,sense_id=si+1,unit_index=pi,unit_id=w+f'::sense{si+1:02d}',support=hi-lo))
            for j in range(lo,hi):meta.append(dict(word=w,a_index=ai,p_index=pi,occ_id=ai*100+j))
    cfg=p.Config(output_root=root,max_workers=1,reserve_gib=1,reserve_disk_gib=2,query_block=16,beta_chunk=1,show_figures=False)
    c=dict(root=root,cache=root,cfg=cfg,run_id='SYNTHETIC_TEST_ONLY',kernel_id='SYNTHETIC_TEST_ONLY',source_stats={},split_files={},
           geometries={'S':{'neighbors':neighbors}})
    folder=root/'inputs/apo'
    p.put_npy(folder/'A.npy',A);p.put_npy(folder/'P.npy',np.array(P));p.put_npy(folder/'O.npy',O)
    p.writej(folder/'words.json',words);p.frame(folder/'units.parquet',pd.DataFrame(units));p.frame(folder/'occurrences.parquet',pd.DataFrame(meta))
    L=[i for i in range(32) if i%4!=3];I=[i for i in range(32) if i%4==3]
    path=root/'inputs/s00.npz';p.savez(path,X=A[L],Q=A[I],y=np.array(L)//4,truth=np.array(I)//4,
        train_words=np.array(words)[L],query_words=np.array(words)[I]);c['split_files']['0']=p.sha(path)
    return c

def test_execution(root):
    c=synthetic_context(root);scales=p.calibrate(c,0);deltas=[];fit_seconds=[]
    np.testing.assert_array_equal(scales,p.calibrate(c,0))
    mappings=[];rhos=[]
    for memory in 'APO':
        a=p.representations(c,0,memory);mappings.append(a['maps']);rhos.append(a['rho'])
        assert a['multiplicity'].sum()==2400
        blocks=[(l,min(l+16,a['n_query'])) for l in range(0,a['n_query'],16)];indices=(0,)
        for lo,hi in blocks:
            p.score_task(c,memory,0,lo,hi);deltas.append(p.recall_task(c,memory,0,indices,scales,lo,hi)['anchor_delta'])
        p.merge_recalls(c,memory,0,indices,blocks)
        for ni in range(3):
            r=p.classification_task(c,memory,0,indices,ni,0,0);fit_seconds.append(r['seconds'])
            assert p.classification_task(c,memory,0,indices,ni,0,0)['new']==0
            z=p.load_trial(*p.trial_location(c,memory,0,ni,0,0))
            for q in 'APO':assert len(z[q][a['maps'][q]])==800
            # Query compression is exact: repeating before prediction must be identical.
            R=np.load(p.scratch(c,memory,0)/('b'+p.digest([0])[:10])/'merged'/f'n{ni}_b00.npy')
            mod=p.core.MulticlassEnsembleClassifier(pd.DataFrame(R[:24],index=a['L']),c['geometries']['S']['neighbors'],random_seed_cls=0,n_jobs=1,voting_n_jobs=1)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');mod.fit_only()
            for q in 'APO':
                lo,hi=a['slices'][q]
                np.testing.assert_array_equal(mod.classify(R[lo:hi][a['maps'][q]]),z[q][a['maps'][q]])
        p.clear_data()
    assert len(set(rhos))==1
    for q in 'APO':
        for mp in mappings[1:]:np.testing.assert_array_equal(mappings[0][q],mp[q])
    # Tamper detection is an actual payload checksum check, not only a filename check.
    path,iid=p.trial_location(c,'A',0,0,0,0);original=path.read_bytes();z=p.arrays(path);z['A'][0]=(z['A'][0]+1)%8;p.savez(path,**z)
    try:p.load_trial(path,iid)
    except RuntimeError:pass
    else:raise AssertionError('Tampering was not detected')
    path.write_bytes(original)
    (root/'STOP').write_text('test')
    try:p.guard(c)
    except p.compute.TechnicalPause:pass
    else:raise AssertionError('STOP not honored')
    (root/'STOP').unlink()
    return dict(ensembles=9,independent_replay_ensembles=9,max_recall_delta=max(deltas),scales=scales.tolist(),fit_seconds=sum(fit_seconds),
        resume_verified=True,tamper_detected=True,canonical_prediction_expansion_verified=True)

def statistics_context(root):
    cfg=p.Config(output_root=root,max_workers=1,reserve_gib=1,reserve_disk_gib=2,show_figures=False)
    words=[f'c{k}w{j}' for k in range(8) for j in range(15)];cl=np.repeat(np.arange(8),15)
    z=np.tile(np.linspace(-1.5,1.5,15),8);z=z/z[:15].std();sim=.7+.04*z+.005*cl
    full=pd.DataFrame(dict(word=words,source_word=words,true_class=cl,validation_similarity=sim,similarity_wc_z=z))
    edges=np.linspace(z.min(),z.max(),6);common=full.copy();common['similarity_bin']=np.searchsorted(edges[1:-1],z,side='right')
    g=dict(full=full,common=common,edges=edges,neighbors={f'class{k}':[w for w in words if w.startswith(f'c{k}w')] for k in range(8)})
    c=dict(cfg=cfg,root=root,cache=root,run_id='SYNTHETIC_STATISTICS',kernel_id='test',split_files={},source_stats={},geometries={'S':g})
    base=full.loc[full.index.repeat(98)].reset_index(drop=True);base['beta']=np.tile(p.BETAS,len(full));base['beta_key']=base.beta.round(12);base['n_items']=10;base['n_trials']=1000
    files={}
    for ai,arm in enumerate(p.ARMS):
        for ni in range(3):
            d=base.copy();d['rate']=1/(1+np.exp(-(-2.2+.1*d.similarity_wc_z-3*d.beta+.01*ai+.015*ni*d.similarity_wc_z)))
            d['n_errors']=np.round(d.rate*1000).astype(int);d['rate']=d.n_errors/1000
            name=f'{arm}_n{ni}.parquet';p.frame(root/'counts'/name,d);files[name]=p.sha(root/'counts'/name)
    p.writej(root/'counts_complete.json',dict(run_id=c['run_id'],files=files,summary=[]))
    units=full.sort_values('word').reset_index(drop=True);rows=[]
    for rep in range(5):
        for f,(_,b) in enumerate(p.StratifiedKFold(5,shuffle=True,random_state=rep).split(units,units.true_class)):
            rows.extend(dict(word=w,repeat=rep,fold=f) for w in units.iloc[b].word)
    p.frame(root/'inputs/cv_folds.parquet',pd.DataFrame(rows));return c

def test_statistics(root):
    c=statistics_context(root);a=st.load_analysis(c,'AA');b=st.load_analysis(c,'OO')
    for x,y in zip(a['folds'],b['folds']):
        np.testing.assert_array_equal(x['train'],y['train']);np.testing.assert_array_equal(x['test'],y['test'])
    np.testing.assert_allclose(a['weights'].sum(),1)
    E=st.surface(a,np.ones(len(a['data'])));np.testing.assert_allclose(E,1)
    np.testing.assert_allclose(st.surface(a,np.ones(len(a['data'])),(0,)),1)
    for mi in range(2):
        r=st.cv_task(a,mi,0,0);assert r['stable']
        r=st.refit_task(a,mi,0);assert r['stable']
        r=st.refit_task(a,mi,0,(0,));assert r['stable'] and r['n_classes']==7
    t=p.empirical(c);assert len(t)==9*3*9
    ct=pd.read_csv(root/'empirical_contrasts.csv');assert len(ct)==3*9*8
    p.show(c,include_models=False)
    assert len(list((root/'figures').glob('*.png')))==12
    archive=p.export_review(c);assert archive.is_file()
    return dict(shared_CV_verified=True,fits_and_LOCO_refits=6,empirical_cases=len(t),paired_contrasts=len(ct),PNG_figures=12)

def test_source_extraction(root):
    """Exercise the real 640 x 100 x 768 extraction path against a synthetic corpus."""
    from unittest.mock import patch
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng=np.random.default_rng(21);root.mkdir(parents=True,exist_ok=True)
    dataset=root/'dataset';occ=root/'occ';run=root/'run'
    for d in (dataset,occ,run):d.mkdir()
    words=[f'w{i:03d}' for i in range(640)];raw=rng.standard_normal((64000,768),dtype=np.float32)
    A=np.array([raw[100*i:100*(i+1)].astype(float).mean(0) for i in range(640)])
    P=[];units=[];membership=[]
    for i,w in enumerate(words):
        for si,(lo,hi) in enumerate(((0,30),(30,100))):
            ix=len(P);P.append(raw[100*i+lo:100*i+hi].astype(float).mean(0))
            units.append(dict(unit_index=ix,unit_id=w+f'::sense{si+1:02d}',word=w,sense_id=si+1,support=hi-lo))
            membership.extend(dict(occ_id=100*i+j,word=w,sense_id=si+1,unit_index=ix) for j in range(lo,hi))
    p.frame(dataset/'units.parquet',pd.DataFrame(units));p.frame(dataset/'membership.parquet',pd.DataFrame(membership));p.put_npy(dataset/'prototypes_RAW_float64.npy',np.array(P))
    m=pd.DataFrame(dict(word=np.repeat(words,100),occ_id=np.arange(64000),source_doc_key=[f'test:train:{j}' for j in range(64000)],
        word_pos_original=np.zeros(64000,int),source_name='test',source_split='train',source_row_idx=np.arange(64000)))
    tab=pa.Table.from_pandas(m,preserve_index=False).append_column('embedding',pa.FixedSizeListArray.from_arrays(pa.array(raw.ravel()),768))
    pq.write_table(tab,occ/'contextual_occurrences_definitive.parquet',row_group_size=100,compression='snappy')
    del tab,raw,m
    p.frame(occ/'word_row_group_index.csv',pd.DataFrame(dict(word=words,n_occurrences=100,row_start=np.arange(640)*100,row_end_exclusive=(np.arange(640)+1)*100,row_group_id=np.arange(640))))
    manifest=dict(status='COMPLETE',files={n:dict(sha256=p.sha(dataset/n)) for n in ('units.parquet','membership.parquet','prototypes_RAW_float64.npy')})
    p.writej(dataset/'dataset_manifest.json',manifest)
    cfg=p.Config(output_root=root,dataset_dir=dataset,occurrence_dir=occ,reserve_gib=1,reserve_disk_gib=2,max_workers=1)
    c=dict(cfg=cfg,root=run,cache=run,run_id='SYNTHETIC_SOURCE_ONLY',source_stats={},split_files={})
    for s in range(51):
        f=run/'inputs'/f's{s:02d}.npz';p.savez(f,train_words=np.array(words)[np.arange(640)%4!=3],query_words=np.array(words)[np.arange(640)%4==3]);c['split_files'][str(s)]=p.sha(f)
    with patch.object(p,'OCC_SHA',p.sha(occ/'contextual_occurrences_definitive.parquet')),patch.object(p,'DATASET_SHA',p.sha(dataset/'dataset_manifest.json')):
        p.prepare_occurrences(c,pd.DataFrame(A,index=words))
        p.prepare_occurrences(c,pd.DataFrame(A,index=words))
    r=p.readj(run/'inputs/apo/done.json')['summary'];assert r['words']==640 and r['occurrences']==64000
    stats=p.readcsv(run/'inputs/apo/representation_diagnostics.csv')
    assert (stats.variance_P<=stats.variance_O+1e-12).all()
    return dict(**r,source_extraction_and_resume_verified=True)

def run_tests():
    started=time.perf_counter();report=dict(test_data='Synthetic only; not scientific outcomes',environment=p.environment())
    with tempfile.TemporaryDirectory(prefix='p20_tests_') as tmp,threadpool_limits(limits=1):
        root=Path(tmp);report['multiplicity']=test_multiplicity();print('Expanded operator equivalence: PASS',flush=True)
        report['execution']=test_execution(root/'execution');print('Controlled execution, resume and tamper checks: PASS',flush=True)
        report['statistics']=test_statistics(root/'statistics');print('Shared CV, statistical refits and figure export: PASS',flush=True)
        report['sources']=test_source_extraction(root/'sources');print('Canonical source extraction and mean checks: PASS',flush=True)
    report.update(status='PASS',seconds=time.perf_counter()-started);return report

if __name__=='__main__':
    report=run_tests();print(json.dumps(report,indent=2))
    # Optional explicit destination avoids changing any scientific package file.
    import sys
    if len(sys.argv)>1:p.writej(Path(sys.argv[1]),report)
