"""W2V K40 RAW: selección estadística y LOCO, sin repetir ensayos.

Recetas matemáticas importadas del módulo estático congelado. Los signos son
resultados descriptivos; no seleccionan K, familia, Ridge ni checkpoints.
"""
from __future__ import annotations
import contextlib, gc, hashlib, json, os, platform, shutil, threading, time, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
import e2_k40_model_reference as ref

PACKAGE=Path(__file__).resolve().parent
VERSION='E2_W2V_K40_MODEL_LOCO_V1'
COMPLETION_ID='3539f481e978b9499b795b8aeb1f9757df16788858e0843802f5c07241fe9ae9'
CORE3=ref.CORE3
LABELS=ref.CLASSES85
req=ref.req85
sha=ref.sha85
digest=ref.digest85
readj=ref.read85
writej=ref.write85
readcsv=lambda p:pd.read_csv(p,float_precision='round_trip')

class TechnicalPause(RuntimeError): pass

@dataclass
class Config:
    output_root: Path=Path.home()/'E2R'
    max_workers: int=14
    reserve_gib: float=3.0
    reserve_disk_gib: float=2.0
    show_figures: bool=True
    figure_dpi: int=150
    def __post_init__(self):
        self.output_root=Path(self.output_root).expanduser().resolve()
        req(type(self.max_workers) is int and 1<=self.max_workers<=14,'max_workers debe estar entre 1 y 14.')
        req(self.reserve_gib>=3 and self.reserve_disk_gib>=1,'Conserva al menos 3 GiB RAM y 1 GiB disco.')
        req(self.figure_dpi>=72,'DPI inválido.')

@contextlib.contextmanager
def _lock(root):
    p=Path(root)/'RUNNING.lock';p.parent.mkdir(parents=True,exist_ok=True)
    f=p.open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        f.close();raise TechnicalPause('Otra ejecución utiliza '+str(root))
    try: yield
    finally:
        f.seek(0)
        if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
        f.close()

def _guard(ctx, need=.05):
    cfg=ctx['cfg']
    if (ctx['root']/'STOP').exists():raise TechnicalPause('STOP: pausa reanudable; elimina STOP para continuar.')
    if psutil.virtual_memory().available/2**30 < cfg.reserve_gib+need:
        raise TechnicalPause('RAM insuficiente para el siguiente ajuste; conserva checkpoints.')
    if shutil.disk_usage(ctx['root']).free/2**30 < cfg.reserve_disk_gib:
        raise TechnicalPause('Disco por debajo de la reserva configurada.')

def _verify_package():
    req(Path(ref.__file__).resolve()==PACKAGE/'e2_k40_model_reference.py','Otra referencia está cargada. Reinicia el kernel.')
    manifest=readj(PACKAGE/'PACKAGE_SHA256.json')
    req('e2_k40_model.py' in manifest and 'e2_k40_model_reference.py' in manifest,'Manifest de código incompleto.')
    for name,wanted in manifest.items():
        p=(PACKAGE/name).resolve()
        req(p.is_relative_to(PACKAGE) and p.is_file() and sha(p)==wanted,'Paquete incompleto/modificado: '+name)
    return manifest

def _frozen(path,value):
    value=ref.safe85(value)
    if path.exists():req(readj(path)==value,'Contrato congelado distinto: '+str(path))
    else:writej(path,value)

def prepare(config=None):
    cfg=config or Config();manifest=_verify_package()
    src=PACKAGE/'evidence/completion';sp=readj(PACKAGE/'evidence/source_protocol.json')
    cp=readj(src/'protocol.json');sourceid=digest(sp)
    req(digest(cp)==COMPLETION_ID==readj(src/'report/summary.json')['run_id'],'Identidad de conteos fuente distinta.')
    req(sourceid==cp['source_audit'] and cp['K']==40 and cp['condition']=='RAW','Fuente de otra condición.')
    req(cp['networks']==list(CORE3) and cp['split_seeds']==list(range(51)) and cp['classifier_seeds']==list(range(10)), 'Contrato de redes/semillas distinto.')
    req(np.array_equal(cp['betas'],ref.BETAS),'Se requieren los 98 betas congelados.')
    for name,wanted in readj(src/'EXPORT_SHA256.json').items():
        p=(src/name).resolve();req(p.is_relative_to(src) and p.is_file() and sha(p)==wanted,'Export fuente alterado: '+name)
    geo=PACKAGE/'evidence/geometry';gid=digest(dict(parent=sourceid,K=40,phase='geometry'))
    ref._receipt(geo,gid)
    gate=readj(geo/'gate.json');req(gate['A'],'K40 no tiene admisibilidad estructural.')
    g={k:pd.read_parquet(geo/(k+'.parquet')) for k in ('full','common','expected')}
    g.update(K=40,gate=gate,edges=np.array(gate['edges']),centers=np.array(gate['centers']),
             split_audit=readj(geo/'split_audit.json'),neighbors=readj(geo/'neighbors.json'),signature=gid)
    ref._validate_geometry(g)
    for name in ('full','common','expected'):
        pd.testing.assert_frame_equal(g[name],pd.read_parquet(src/f'report/K40/geometry/{name}.parquet'),check_dtype=False,check_exact=True)
    ctx=dict(g,cfg=cfg,betas=ref.BETAS.copy(),keys=np.round(ref.BETAS,12),labels=LABELS,
             counts={},empirical={},dataset_label='W2V300 RAW',protected={})
    calibration=readcsv(src/'report/calibration.csv');calibration=calibration.loc[calibration.K.eq(40)].copy()
    req(len(calibration)==102 and not calibration.duplicated(['network','split_seed']).any(),'Calibración K40 incompleta.')
    audits={a['split_seed']:a for a in g['split_audit']};checks=[]
    oldfull=readcsv(src/'report/FULL.csv');oldloco=readcsv(src/'report/LOCO.csv')
    empirical_rows=[]
    for n in CORE3:
        folder=src/'report/K40'/n;p=folder/'word_beta.parquet'
        binding=readj(folder/'source_binding.json');receipt=readj(folder/'source_receipt.json')
        req(sha(p)==binding['source_count_sha256']==binding['exported_count_sha256']==receipt['outputs']['word_beta.parquet'],'Bytes de conteos no corresponden al recibo.')
        if n==CORE3[0]:
            ident=digest(dict(parent=sourceid,geometry=gid,network=n,scales=np.ones(51),phase='counts',
                classifier_seeds=list(range(10)),seed_merge_strategy='AUTHENTICATED_DISJOINT_SEEDS_V1'))
        else:
            ident=digest(dict(run=COMPLETION_ID,stage='counts',network=n))
            cal=calibration.loc[calibration.network.eq(n)].sort_values('split_seed')
            req(cal.split_seed.tolist()==list(range(51)) and cal.lambda_value.between(.001,1).all(),'Escalas HFY incompletas.')
            req(cal.calibration_uses_errors.eq(False).all() and cal.calibration_uses_validation.eq(False).all(),'Calibración no TRAIN-only.')
            for row in cal.to_dict('records'):
                a=audits[row['split_seed']]
                req(row['train_words_sha']==a['train_words_sha'] and row['train_array_sha']==a['train_array_sha'],'Calibración de otro TRAIN.')
        req(receipt['identity']==ident,'Recibo de conteos de otra corrida.')
        d=ref.validate_wb85(ctx,pd.read_parquet(p));ctx['counts'][n]=d
        s=ref.surface85(ctx,d);ctx['empirical'][n]=s
        for c in [None,*range(8)]:
            t=ref.trends85(ref.surface85(ctx,d,c))
            old=(oldfull.loc[oldfull.K.eq(40)&oldfull.network.eq(n)] if c is None else
                 oldloco.loc[oldloco.K.eq(40)&oldloco.network.eq(n)&oldloco.omitted_class.eq(c)])
            req(len(old)==1 and all(np.isclose(t[k],old.iloc[0][k],rtol=0,atol=1e-12) for k in ('T1','T2','T3')),'Tendencias fuente no reproducidas.')
            empirical_rows.append(dict(network=n,omitted_class=c,**t))
        checks.append(dict(network=n,K=40,n_words=d.word.nunique(),n_common=len(ctx['common']),
            n_rows=len(d),n_betas=98,n_split_seeds=51,n_classifier_seeds=10,counts_verified=True,
            calibration='Softmax λ=1' if n==CORE3[0] else 'TRAIN-only verificada'))
    ref._prepare_model_inputs(ctx)
    for f in ctx['folds']:
        a=ctx['data'].iloc[f['train']];b=ctx['data'].iloc[f['test']]
        req(not set(a.word)&set(b.word) and set(a.true_class)==set(b.true_class)==set(range(8)),'CV con fuga o clase ausente.')
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  scipy=ref.scipy.__version__,sklearn=ref.sklearn.__version__)
    contract=dict(version=VERSION,K=40,condition='RAW',dataset='W2V300',networks=list(CORE3),
        K_policy='FIXED_K40_STRUCTURAL_ADMISSIBILITY_IN_AUDITED_DOMAIN_GE40',
        signs_are_results_not_admissibility=True,K_search=False,source_completion=COMPLETION_ID,
        source_audit=sourceid,geometry=gid,counts_sha256={n:sha(src/'report/K40'/n/'word_beta.parquet') for n in CORE3},
        code_sha256={k:manifest[k] for k in ('e2_k40_model.py','e2_k40_model_reference.py')},
        betas=ctx['betas'],split_seeds=list(range(51)),classifier_seeds=list(range(10)),
        families=ref.FAMILIES85,alphas=ref.ALPHAS85,
        CV=dict(repeats=5,folds=5,group='word',splitter='StratifiedKFold',seeds=list(range(5)),
                folds_sha256=digest(ctx['fold_ledger'].to_dict('records')),preprocessing_train_only=True),
        selection=dict(rule='two_one_SE_then_complexity_stronger_ridge_id',signs_used=False,LOCO_used=False,
                       shared_family_and_ridge_CORE3=True,force_K50_winner=False),
        fitting='weighted_logistic_cross_entropy_of_error_fractions',weights='equal_class_equal_word_beta_exposure_within_word',
        intercept_penalized=False,empirical_bins=5,model_grid=30,COMMON_fixed=True,
        LOCO='refit_seven_classes_rebuild_preprocessing_no_hyperparameter_retune',
        experimental_class_ablation=False,independent_validation=False,software=versions)
    pid=digest(contract);root=cfg.output_root/('k40_model_'+pid[:12])
    req(not root.is_relative_to(PACKAGE),'La salida debe quedar fuera de la entrega.')
    root.mkdir(parents=True,exist_ok=True)
    ctx.update(root=root,id=pid,id85=pid,out85=root/'MODEL_SEARCH',
        out88=root/'FULL_LOCO',id88=digest(dict(parent=pid,stage='FULL_LOCO')),
        input_audit=pd.DataFrame(checks),calibration=calibration,protocol=contract,
        empirical_rows=pd.DataFrame(empirical_rows),settings=dict(show_figures=cfg.show_figures,figure_dpi=cfg.figure_dpi))
    with _lock(root):
        _guard(ctx);ctx['out85'].mkdir(exist_ok=True);ctx['out88'].mkdir(exist_ok=True)
        _frozen(root/'protocol.json',contract)
        _frozen(ctx['out85']/'model_search_contract_frozen.json',contract)
        _frozen(ctx['out88']/'protocol_frozen.json',dict(parent=pid,stage='FULL_LOCO',retune=False))
        ref.csv85(root/'input_audit.csv',ctx['input_audit']);ref.csv85(root/'empirical_FULL_LOCO.csv',ctx['empirical_rows'])
        ref.csv85(root/'calibration.csv',calibration)
        ref.csv85(ctx['out85']/'candidates_frozen.csv',ctx['registry'])
        ref.csv85(ctx['out85']/'folds_frozen.csv',ctx['fold_ledger'])
        for n in CORE3:ref.frame85(root/'inputs'/n/'word_beta.parquet',ctx['counts'][n])
        for n in ('full','common','expected'):ref.frame85(root/'inputs/geometry'/(n+'.parquet'),ctx[n])
        writej(root/'inputs/geometry/axis.json',dict(edges=ctx['edges'],centers=ctx['centers'],signature=gid))
        writej(root/'source_ledger.json',manifest)
        writej(root/'environment.json',dict(software=versions,source_software=cp['expected_scientific_versions'],
                    max_workers=cfg.max_workers,reserve_gib=cfg.reserve_gib))
    print('K40 autenticado: 226 palabras FULL, 192 COMMON; 98×51×10. Sin nuevos ensayos.',flush=True)
    print('Salida:',root,flush=True)
    return ctx

def _payload(ctx):
    return {k:ctx[k] for k in ('cfg','root','data','outcomes','betas','out85','id85')}

def _alphas(family):
    return [None] if family['kind'] in ('constant','class_constant') else list(reversed(ref.ALPHAS85))

def _cv_group(ctx,fold,family):
    """Una matriz TRAIN/TEST para tres redes; checkpoints separados por ajuste."""
    alphas=_alphas(family);old={}
    for n in CORE3:
        for a in alphas:
            folder,ident=ref.location85(ctx,n,fold,family,a)
            old[n,a]=ref.completed85(folder,ident)
            if old[n,a]:req(old[n,a]['summary']['stable'],'Checkpoint inestable.')
    total=len(old)
    if all(old.values()):return dict(computed=0,reused=total)
    _guard(ctx)
    d=ctx['data'];train=d.iloc[fold['train']];test=d.iloc[fold['test']]
    w=ref.weights85(train);wt=ref.weights85(test);X=Xt=None
    info=dict(declared_parameters=family['p'],active_parameters=family['p'])
    computed=0
    with threadpool_limits(limits=1):
        if alphas!=[None]:X,Xt,w,info=ref.design85(ctx,fold,family)
        for n in CORE3:
            y=ctx['outcomes'][n][fold['train']];yt=ctx['outcomes'][n][fold['test']];theta=None
            for a in alphas:
                folder,ident=ref.location85(ctx,n,fold,family,a)
                if old[n,a]:
                    with np.load(folder/'arrays.npz',allow_pickle=False) as z:theta=z['coef'].copy()
                    continue
                _guard(ctx);started=time.perf_counter()
                if family['kind']=='constant':
                    theta=np.array([float(w@y)]);p=np.full(len(test),theta[0])
                    fit=dict(stable=True,success=True,gradient_inf=0.,n_iter=0,attempts=0)
                elif family['kind']=='class_constant':
                    q=pd.DataFrame(dict(c=train.true_class.to_numpy(),wy=w*y,w=w)).groupby('c').sum()
                    theta=(q.wy/q.w).reindex(range(8)).to_numpy();p=theta[test.true_class.to_numpy(int)]
                    fit=dict(stable=True,success=True,gradient_inf=0.,n_iter=0,attempts=0)
                else:
                    theta,fit=ref.solve90(X,y,w,float(a),theta);p=ref.expit(Xt@theta)
                req(fit['stable'] and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Ajuste CV inválido.')
                pc=np.clip(p,1e-12,1-1e-12)
                row=dict(candidate=ref.candidate85(family['name'],a),family=family['name'],alpha=a,
                    network=n,repeat=fold['repeat'],fold=fold['fold'],**fit,n_parameters=family['p'],
                    active_parameters=info['active_parameters'],n_test_words=len(fold['test_words']),n_test_rows=len(test),
                    fold_logloss=float(wt@(-yt*np.log(pc)-(1-yt)*np.log1p(-pc))),seconds=time.perf_counter()-started)
                folder.mkdir(parents=True,exist_ok=True)
                ref.write_arrays88(folder/'arrays.npz',prediction=p,row_id=fold['test'],coef=theta)
                writej(folder/'summary.json',row);writej(folder/'preprocessing.json',info)
                ref.receipt85(folder,ident,['arrays.npz','summary.json','preprocessing.json'],row)
                computed+=1
    return dict(computed=computed,reused=total-computed)

class _Monitor:
    def __init__(self,ctx,stage,workers):self.ctx=ctx;self.stage=stage;self.workers=workers;self.rows=[]
    def sample(self):
        rss=np.nan
        try:
            me=psutil.Process();rss=me.memory_info().rss/2**30
            for child in me.children(recursive=True):
                try:rss+=child.memory_info().rss/2**30
                except psutil.Error:pass
        except psutil.Error:pass
        self.rows.append(dict(stage=self.stage,workers=self.workers,elapsed_seconds=time.perf_counter()-self.start,
            available_GiB=psutil.virtual_memory().available/2**30,process_tree_RSS_GiB=rss,cpu_percent=psutil.cpu_percent()))
    def __enter__(self):
        self.start=time.perf_counter();self.stop=threading.Event();self.sample()
        def loop():
            while not self.stop.wait(1.):self.sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.sample()
        p=self.ctx['root']/'resources.csv';new=pd.DataFrame(self.rows)
        if p.exists():new=pd.concat([readcsv(p),new],ignore_index=True)
        ref.csv85(p,new)

def _workers(ctx,p,remaining):
    # Diseño bruto/estandarizado, test, productos temporales y proceso científico.
    per=max(.45,len(ctx['data'])*p*8*7/2**30+.25)
    available=psutil.virtual_memory().available/2**30
    jobs=min(ctx['cfg'].max_workers,os.cpu_count() or 1,remaining,int((available-ctx['cfg'].reserve_gib)/per))
    if jobs<1:raise TechnicalPause(f'RAM insuficiente: reserva {ctx["cfg"].reserve_gib:.1f} + tarea {per:.2f} GiB.')
    return jobs,per

def _status(ctx,**kw):writej(ctx['root']/'execution_status.json',dict(run_id=ctx['id'],resumable=True,**kw))

def search(ctx):
    with _lock(ctx['root']),threadpool_limits(limits=1):
        try:
            _guard(ctx);computed=reused=0;total=3825;payload=_payload(ctx)
            for family in ref.FAMILIES85:
                missing=[]
                for fold in ctx['folds']:
                    receipts=[ref.completed85(*ref.location85(ctx,n,fold,family,a)) for n in CORE3 for a in _alphas(family)]
                    if all(receipts):
                        req(all(r['summary']['stable'] for r in receipts),'Ajuste guardado inestable.')
                        reused+=len(receipts)
                    else:missing.append(fold)
                if not missing:print(f'{family["name"]}: reutilizada completa.',flush=True);continue
                jobs,per=_workers(ctx,family['p'],len(missing));start=time.perf_counter()
                print(f'{family["name"]}: {len(missing)} grupos; hasta {jobs} procesos; estimación {per:.2f} GiB/tarea.',flush=True)
                with _Monitor(ctx,family['name'],jobs),parallel_config(backend='loky',inner_max_num_threads=1):
                    results=Parallel(n_jobs=jobs,return_as='generator_unordered',batch_size=1,pre_dispatch=jobs,
                        max_nbytes='1M')(delayed(_cv_group)(payload,f,family) for f in missing)
                    for i,r in enumerate(results,1):
                        computed+=r['computed'];reused+=r['reused']
                        eta=(time.perf_counter()-start)/i*(len(missing)-i)/60
                        _status(ctx,status='CV_RUNNING',family=family['name'],completed_fits=computed+reused,
                                total_fits=total,workers=jobs,ETA_family_minutes=eta)
                        print(f'  grupos {i}/{len(missing)} · ajustes {computed+reused}/{total} · ETA familia {eta:.1f} min',flush=True)
                        _guard(ctx)
                gc.collect()
            req(computed+reused==total,'Inventario CV incompleto.')
            _verify_package()
            selected=ref.rank85(ctx)
            req(selected['n_candidates']==51 and selected['n_fits']==3825,'Selección parcial.')
            ref.receipt85(ctx['out85'],digest(dict(parent=ctx['id'],stage='selection')),
                ['model_search_contract_frozen.json','candidates_frozen.csv','folds_frozen.csv',
                 'model_selection_frozen.json','candidate_ranking.csv','network_comparison.csv',
                 'OOF_metrics_by_network_repeat.csv','all_fit_diagnostics.csv'],selected)
            _status(ctx,status='MODEL_SELECTED',computed_fits=computed,reused_fits=reused,
                    selected=selected['selected'],signs_used_for_selection=False)
            ctx['selection']=selected
            return selected
        except BaseException as exc:
            _status(ctx,status='TECHNICAL_PAUSE',error=repr(exc),scientific_rejection=False)
            raise

def _load_selection(ctx):
    p=ctx['out85']/'model_selection_frozen.json'
    req(p.is_file(),'Ejecuta primero la celda de búsqueda estadística.')
    req(ref.completed85(ctx['out85'],digest(dict(parent=ctx['id'],stage='selection'))) is not None,
        'Falta recibo de búsqueda completa. Reejecuta la celda de búsqueda para autenticar checkpoints y ranking.')
    s=readj(p);req(s['protocol_id']==ctx['id85'] and s['n_fits']==3825 and s['n_candidates']==51,'Selección de otro contrato o incompleta.')
    req(s['selection_uses_T1_T2_T3'] is False and s['selection_uses_LOCO'] is False,'Selección condicionada a signos.')
    ranking=readcsv(ctx['out85']/'candidate_ranking.csv')
    req(len(ranking)==51 and not ranking.candidate.duplicated().any(),'Ranking incompleto.')
    chosen,_,_=ref.choose85(ranking)
    req(chosen['candidate']==s['selected']['candidate'],'Ranking no reproduce la selección.')
    ctx.update(selection=s,selected90=s['selected'],alpha88=s['selected']['alpha'])
    return s

def _fidelity(e,p,network,case,role):
    delta=p.to_numpy()-e.to_numpy();beta=e.columns.to_numpy(float);rows=[]
    for region,mask in [('ALL',np.ones(len(beta),bool)),('BETA_LE_010',beta<=.10),('BETA_GT_010',beta>.10)]:
        q=delta[:,mask]
        rows.append(dict(network=network,case=case,role=role,region=region,RMSE=float(np.sqrt(np.square(q).mean())),
                         MAE=float(abs(q).mean()),bias=float(q.mean()),max_abs_error=float(abs(q).max())))
    return rows

def _oof(ctx):
    selected=ctx['selected90'];fam=next(f for f in ref.FAMILIES85 if f['name']==selected['family'])
    arrays={n:[] for n in CORE3};anchors=[];metrics=[];fidelity=[]
    for repeat in range(5):
        pred={n:np.full(len(ctx['data']),np.nan) for n in CORE3};cover=np.zeros(len(ctx['data']),int)
        for fold in (f for f in ctx['folds'] if f['repeat']==repeat):
            _guard(ctx);a=ctx['data'].iloc[fold['train']];b=ctx['data'].iloc[fold['test']]
            X=Xt=None;w=ref.weights85(a)
            if selected['alpha'] is not None:X,Xt,w,_=ref.design85(ctx,fold,fam)
            for n in CORE3:
                folder,ident=ref.location85(ctx,n,fold,fam,selected['alpha']);m=ref.completed85(folder,ident)
                req(m is not None and m['summary']['stable'],'Falta checkpoint seleccionado.')
                with np.load(folder/'arrays.npz',allow_pickle=False) as z:ids=z['row_id'];p=z['prediction'];coef=z['coef']
                req(np.array_equal(ids,fold['test']),'Índices OOF distintos.')
                y=ctx['outcomes'][n][fold['train']]
                if X is not None:
                    _,grad,eta=ref._objective85R(X,y,w,selected['alpha'],coef)
                    gradmax=float(abs(grad).max());reconstructed=ref.expit(Xt@coef)
                    req(gradmax<=2e-6,'Gradiente OOF no certificado.')
                elif selected['kind']=='constant':
                    gradmax=0.;reconstructed=np.full(len(b),w@y)
                    req(np.allclose(coef,[w@y],rtol=0,atol=1e-12),'Constante global no reproducida.')
                else:
                    q=pd.DataFrame(dict(c=a.true_class.to_numpy(),w=w,wy=w*y)).groupby('c').sum()
                    values=q.wy/q.w;reconstructed=b.true_class.map(values).to_numpy();gradmax=0.
                    req(np.allclose(coef,values.reindex(range(8)),rtol=0,atol=1e-12),'Constantes por clase distintas.')
                delta=float(abs(p-reconstructed).max());req(delta<=1e-10,'Predicciones OOF no reproducidas.')
                pred[n][ids]=p;anchors.append(dict(network=n,repeat=repeat,fold=fold['fold'],gradient_inf=gradmax,prediction_delta=delta))
            cover[fold['test']]+=1
        req((cover==1).all(),'Cobertura OOF distinta de uno por repetición.')
        for n in CORE3:
            p=pred[n];req(np.isfinite(p).all(),'OOF incompleta.');y=ctx['outcomes'][n];w=ctx['weights'];pc=np.clip(p,1e-12,1-1e-12)
            metrics.append(dict(network=n,repeat=repeat,OOF_logloss=float(w@(-y*np.log(pc)-(1-y)*np.log1p(-pc))),
                OOF_surface_RMSE=ref.rmse_surface85(ctx,n,p),OOF_brier_fraction=float(w@np.square(y-p)),
                OOF_brier_bernoulli=float(w@(y*(1-p)**2+(1-y)*p**2))))
            frame=ctx['data'].assign(prediction=p);ref.frame85(ctx['root']/'OOF'/f'{n}_r{repeat}.parquet',frame)
            surface=ref.surface85(ctx,frame,value='prediction');ref.csv85(ctx['root']/'OOF'/f'{n}_r{repeat}_matched5.csv',surface,index=True)
            fidelity.extend(_fidelity(ctx['empirical'][n],surface,n,f'OOF_r{repeat}','OOF_SELECTION_CONDITIONAL'))
            arrays[n].append(p)
    historical=readcsv(ctx['out85']/'OOF_metrics_by_network_repeat.csv')
    for row in metrics:
        old=historical.loc[historical.candidate.eq(selected['candidate'])&historical.network.eq(row['network'])&historical.repeat.eq(row['repeat'])]
        req(len(old)==1 and all(np.isclose(row[k],old.iloc[0][k],rtol=1e-9,atol=1e-12) for k in ('OOF_logloss','OOF_surface_RMSE','OOF_brier_fraction')),'OOF no reproduce ranking.')
    ctx['oof_means']={}
    for n,values in arrays.items():
        frame=ctx['data'].assign(prediction=np.mean(values,axis=0));s=ref.surface85(ctx,frame,value='prediction')
        ctx['oof_means'][n]=s;ref.csv85(ctx['root']/'OOF'/f'{n}_mean_matched5.csv',s,index=True)
        fidelity.extend(_fidelity(ctx['empirical'][n],s,n,'OOF_mean','REPEAT_AVERAGED_OOF'))
    ref.csv85(ctx['root']/'OOF/anchors.csv',pd.DataFrame(anchors))
    ref.csv85(ctx['root']/'OOF/metrics.csv',pd.DataFrame(metrics))
    ref.csv85(ctx['root']/'OOF/fidelity.csv',pd.DataFrame(fidelity))
    return fidelity

def _refit_job(ctx,n,c):
    _guard(ctx)
    with threadpool_limits(limits=1):return ref.one_refit90(ctx,n,c)

def _case_path(ctx,n,c):return ctx['out88']/'fits'/n/('FULL' if c is None else f'LOCO_{c}_{LABELS[c]}')

def _verify_refit(ctx,n,c):
    path=_case_path(ctx,n,c);ident=digest(dict(protocol=ctx['id88'],network=n,omitted=c))
    receipt=ref.completed85(path,ident);req(receipt is not None and receipt['summary']['stable'],'Falta refit estable.')
    ids=np.flatnonzero(np.ones(len(ctx['data']),bool) if c is None else ctx['data'].true_class.ne(c))
    d=ctx['data'].iloc[ids];y=ctx['outcomes'][n][ids]
    fam=next(f for f in ref.FAMILIES85 if f['name']==ctx['selected90']['family'])
    X,w,state=ref.design90(ctx,d,fam)
    with np.load(path/'parameters.npz',allow_pickle=False) as z:theta=z['theta'];classes=z['class_labels'];cube=z['class_grid']
    req(set(classes)==set(range(8))-({c} if c is not None else set()),'LOCO conserva/predice la clase omitida.')
    if X is not None:
        _,grad,_=ref._objective85R(X,y,w,ctx['alpha88'],theta);req(abs(grad).max()<=2e-6,'Refit sin certificado de gradiente.')
    elif fam['kind']=='constant':req(np.allclose(theta,[w@y],rtol=0,atol=1e-12),'Constante FULL/LOCO distinta.')
    else:
        q=pd.DataFrame(dict(c=d.true_class.to_numpy(),wy=w*y,w=w)).groupby('c').sum()
        req(np.allclose(theta,(q.wy/q.w).reindex(classes),rtol=0,atol=1e-12),'Constantes por clase FULL/LOCO distintas.')
    p=ref.predict90(d,state,theta);stored=pd.read_parquet(path/'word_predictions.parquet')
    pd.testing.assert_frame_equal(stored[['word','true_class','beta_key']].reset_index(drop=True),d[['word','true_class','beta_key']].reset_index(drop=True),check_dtype=False)
    req(np.allclose(p,stored.prediction,rtol=0,atol=1e-10),'Predicciones FULL/LOCO distintas.')
    grid,newcube=ref.grid90(ctx,state,theta);req(np.allclose(cube,newcube,rtol=0,atol=1e-10),'GRID30 no reproducida.')
    matched=ref.surface85(ctx,stored,c,value='prediction');emp=ref.surface85(ctx,ctx['counts'][n],c)
    for filename,s in [('empirical_5bins.csv',emp),('fitted_5bins.csv',matched),('fitted_30points.csv',grid)]:
        old=ref.read_surface88(path/filename)
        req(np.allclose(old.index,s.index,rtol=0,atol=1e-12) and np.allclose(old.columns,s.columns,rtol=0,atol=1e-12) and np.allclose(old,s,rtol=0,atol=1e-10),'Superficie guardada no reproducida: '+filename)
    return emp,matched,grid

def finish(ctx):
    with _lock(ctx['root']),threadpool_limits(limits=1):
        try:
            _guard(ctx);_load_selection(ctx)
            print('Verificando 75 checkpoints OOF seleccionados…',flush=True)
            fidelity=_oof(ctx)
            tasks_full=[(n,None) for n in CORE3];tasks_loco=[(n,c) for c in range(8) for n in CORE3]
            selected=ctx['selected90'];p=next(f['p'] for f in ref.FAMILIES85 if f['name']==selected['family'])
            payload={k:ctx[k] for k in ('cfg','root','data','outcomes','betas','keys','full','common','edges','centers','counts','out88','id88','selected90','alpha88')}
            for name,tasks in [('FULL',tasks_full),('LOCO',tasks_loco)]:
                jobs,_=_workers(ctx,p,len(tasks))
                with _Monitor(ctx,name,jobs),parallel_config(backend='loky',inner_max_num_threads=1):
                    results=Parallel(n_jobs=jobs,return_as='generator_unordered',batch_size=1,pre_dispatch=jobs)(delayed(_refit_job)(payload,n,c) for n,c in tasks)
                    for i,r in enumerate(results,1):
                        print(f'{name} {i}/{len(tasks)}: {r["network"]} · {r["fit"]}',flush=True)
                        _status(ctx,status='REFITTING',stage=name,completed=i,total=len(tasks));_guard(ctx)
            details=[];surfaces={}
            for n in CORE3:
                for c in [None,*range(8)]:
                    _guard(ctx);emp,m5,m30=_verify_refit(ctx,n,c);surfaces[n,c]=(emp,m5,m30)
                    tag='FULL' if c is None else 'LOCO_'+LABELS[c]
                    row=dict(network=n,case=tag,omitted_class=c,stable=True)
                    for prefix,s in [('empirical',emp),('matched5',m5),('grid30',m30)]:
                        t=ref.trends85(s)
                        row.update({prefix+'_'+k:t[k] for k in ('T1','T2','T3')})
                        row[prefix+'_T1_expected']=t['T1']>0;row[prefix+'_T2_expected']=t['T2']<0;row[prefix+'_T3_expected']=t['T3']<0
                        row[prefix+'_T12_expected']=t['T1']>0 and t['T2']<0
                        row[prefix+'_ALL3_expected']=t['ALL3_expected']
                    if c is not None:
                        only=ref.read_surface88(_case_path(ctx,n,c)/'aggregation_only_30points.csv')
                        t=ref.trends85(only)
                        for k in ('T1','T2','T3'):row['aggregation_only_'+k]=t[k];row['refit_minus_aggregation_'+k]=row['grid30_'+k]-t[k]
                        row['RMSE_refit_vs_aggregation_only']=float(np.sqrt(np.square(m30.to_numpy()-only.to_numpy()).mean()))
                    details.append(row);fidelity.extend(_fidelity(emp,m5,n,tag,'IN_SAMPLE_MATCHED5'))
            detail=pd.DataFrame(details);decisions=[]
            for n in CORE3:
                alln=detail.loc[detail.network.eq(n)];f=alln.loc[alln.case.eq('FULL')].iloc[0];l=alln.loc[alln.case.ne('FULL')]
                req(len(l)==8,'Falta LOCO.')
                for scope in ('empirical','matched5','grid30'):
                    row=dict(network=n,scope=scope)
                    for key in ('T1','T2','T3','T12','ALL3'):
                        col=scope+'_'+key+'_expected';row[key+'_FULL_expected']=bool(f[col]);row[key+'_LOCO_passed']=int(l[col].sum())
                        row[key+'_robust']=bool(f[col] and l[col].all())
                    decisions.append(row)
            decisions=pd.DataFrame(decisions);ff=pd.DataFrame(fidelity);out=ctx['root']/'report';out.mkdir(exist_ok=True)
            for name,frame in [('FULL_LOCO',detail),('decisions',decisions),('fidelity',ff)]:ref.csv85(out/(name+'.csv'),frame)
            _plots(ctx,surfaces,detail)
            summary=dict(status='COMPLETE',run_id=ctx['id'],K=40,selected=selected,n_CV=3825,n_refits=27,
                signs_used_for_K_or_model_selection=False,all_refits_verified=True,LOCO_is_experimental_ablation=False,
                independent_external_validation=False,decisions=decisions.to_dict('records'))
            writej(out/'summary.json',summary);_html_report(ctx,decisions,detail,ff)
            ref.receipt85(out,digest(dict(parent=ctx['id'],stage='report')),
                [p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p.name!='completed.json'],summary)
            _verify_package();_status(ctx,status='COMPLETE',selected=selected)
            return dict(summary=summary,decisions=decisions,details=detail,fidelity=ff)
        except BaseException as exc:
            _status(ctx,status='TECHNICAL_PAUSE',error=repr(exc),scientific_rejection=False)
            raise

def _curves(s):
    x=s.index.to_numpy(float);x=(x-x.mean())/np.square(x-x.mean()).sum();a=s.to_numpy()
    return x@a,a.mean(axis=0)

def _savefig(ctx,fig,name,show=False):
    p=ctx['root']/'report/figures'/name;p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=ctx['cfg'].figure_dpi,bbox_inches='tight')
    if show and ctx['cfg'].show_figures:plt.show()
    plt.close(fig)

def _plots(ctx,surfaces,detail):
    from matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('official',['#6A0DAD','#B755C1','#D890D1','#F2C2A3','#F9E2B7','#FFFF00'])
    vmax=max(float(s.to_numpy().max()) for triplet in surfaces.values() for s in triplet);vmax=max(vmax,1e-12)
    for n in CORE3:
        emp,m5,m30=surfaces[n,None]
        for c in [None,*range(8)]:
            cases=[('GRID30',surfaces[n,c][2])]
            if c is None:cases.insert(0,('EMPIRICAL',emp))
            for kind,s in cases:
                tag='FULL' if c is None else 'LOCO_'+LABELS[c]
                fig,ax=plt.subplots(figsize=(14,3.2));a=s.sort_index(ascending=False)
                im=ax.imshow(a,cmap=cmap,vmin=0,vmax=vmax,aspect='auto',interpolation='nearest');ax.set_box_aspect(.08)
                ax.set_xticks(np.linspace(0,97,5),[f'{v:.2f}' for v in np.linspace(.01,.495,5)])
                ax.set_yticks(np.linspace(0,len(a)-1,5),[f'{v:.2f}' for v in np.linspace(a.index.max(),a.index.min(),5)])
                ax.set(xlabel='Beta nominal',ylabel='Similitud WITHIN_CLASS_Z',title=f'W2V RAW K40 · {ref.NAMES85[n]} · {tag} · {kind}')
                fig.colorbar(im,ax=ax,label='FAR / probabilidad',fraction=.025,pad=.02)
                fig.tight_layout();_savefig(ctx,fig,f'{n}_{tag}_{kind}.png')
        fig,axes=plt.subplots(1,3,figsize=(17,4.7));curves={}
        for label,s,color in [('Empírico 5 bins',emp,'#6A0DAD'),('Modelo MATCHED5',m5,'#C07A00'),('Modelo GRID30',m30,'#187E87')]:
            slope,mean=_curves(s);curves[label]=(slope,mean)
            axes[0].plot(ctx['betas'],slope,label=label,color=color)
            axes[2].plot(ctx['betas'],mean,label=label,color=color)
        x=curves['Empírico 5 bins'][0]
        for label,color in [('Modelo MATCHED5','#C07A00'),('Modelo GRID30','#187E87')]:
            axes[1].scatter(x,curves[label][0],s=9,label=label,color=color)
        lim=np.r_[x,curves['Modelo MATCHED5'][0],curves['Modelo GRID30'][0]]
        axes[1].plot([lim.min(),lim.max()],[lim.min(),lim.max()],color='.5',ls='--')
        axes[0].axhline(0,color='.5',lw=.8)
        axes[0].set(title='A · Pendiente semántica por beta',xlabel='Beta nominal',ylabel='Pendiente FAR–similitud')
        axes[1].set(title='B · Concordancia de pendientes',xlabel='Pendiente empírica',ylabel='Pendiente modelada')
        axes[2].set(title='C · FAR medio por beta',xlabel='Beta nominal',ylabel='FAR / probabilidad')
        for ax in axes:ax.grid(alpha=.15)
        axes[0].legend(fontsize=8);fig.suptitle(ref.NAMES85[n]+' · FULL · '+ctx['selected90']['candidate'])
        fig.tight_layout();_savefig(ctx,fig,n+'_ABC.png',show=True)
        fig,axes=plt.subplots(1,3,figsize=(17,5));rows=detail.loc[detail.network.eq(n)]
        for ax,key in zip(axes,('T1','T2','T3')):
            for prefix,label,color in [('empirical','Empírico','#6A0DAD'),('matched5','MATCHED5','#C07A00'),('grid30','GRID30','#187E87')]:
                ax.plot(range(9),rows[prefix+'_'+key],marker='o',ms=3,label=label,color=color)
            ax.axhline(0,color='.4',lw=.8);ax.set_xticks(range(9),['FULL',*LABELS],rotation=55,ha='right')
            ax.set_title(key+(' > 0' if key=='T1' else ' < 0'));ax.grid(alpha=.15)
        axes[0].legend(fontsize=9);fig.suptitle(ref.NAMES85[n]+' · omisión de clase / reajuste estadístico')
        fig.tight_layout();_savefig(ctx,fig,n+'_LOCO.png',show=True)
        delta=ctx['oof_means'][n].to_numpy()-emp.to_numpy();fig,ax=plt.subplots(figsize=(9,3.5))
        ax.plot(ctx['betas'],np.sqrt(np.square(delta).mean(axis=0)),color='#6A0DAD');ax.axvline(.1,ls=':',color='.5')
        ax.set(title=ref.NAMES85[n]+' · OOF promediado entre repeticiones',xlabel='Beta nominal',ylabel='RMSE MATCHED5')
        fig.tight_layout();_savefig(ctx,fig,n+'_OOF.png')

def _html_report(ctx,decisions,detail,fidelity):
    selected=ctx['selected90'];out=ctx['root']/'report'
    full=detail.loc[detail.case.eq('FULL'),['network',*[p+'_'+k for p in ('empirical','matched5','grid30') for k in ('T1','T2','T3')]]]
    sections=[]
    for n in CORE3:
        sections.append(f'<h2>{ref.NAMES85[n]}</h2><img src="figures/{n}_ABC.png"><img src="figures/{n}_LOCO.png"><img src="figures/{n}_OOF.png">')
    html='''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>W2V K40 · modelo y LOCO</title>
    <style>body{font:15px/1.6 system-ui;color:#263346;margin:32px auto;max-width:1200px;padding:20px}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}img{max-width:100%}.scroll{overflow:auto}h1{font-size:28px}</style>
    <h1>W2V RAW · K40 · modelo y LOCO estadístico</h1>'''
    html+=f'<p>Modelo seleccionado: <b>{selected["candidate"]}</b>. 51 candidatas, 3.825 ajustes CV y 27 reajustes FULL/LOCO completos y verificados.</p>'
    html+='''<p>K40 se fija por admisibilidad estructural en el dominio auditado desde K40. Los signos se reportan y no seleccionan K ni el modelo. El ganador no se fuerza a coincidir con K50. La CV comparte familia/Ridge entre CORE3 y conserva todas las filas de una palabra en un mismo fold.</p>
    <p>Los conteos y las calibraciones se reutilizaron. La CV es condicional a geometría/calibración/ensayos y participa en la selección; no es validación externa. Los reajustes LOCO mantienen hiperparámetros y reconstruyen el preprocesamiento con siete clases. No constituyen una ablación experimental de las memorias.</p>
    <h2>Tendencias FULL</h2><div class="scroll">'''+full.to_html(index=False,float_format=lambda v:f'{v:.7g}',border=0)+'</div>'
    html+='<h2>Signos y conservación LOCO</h2><p>LOCO_passed cuenta omisiones con el signo esperado (0..8); robust exige FULL y 8/8. No es una probabilidad ni una prueba de significación.</p><div class="scroll">'+decisions.to_html(index=False,border=0)+'</div>'
    html+='<h2>Fidelidad sobre soporte comparable</h2><p>MATCHED5 promedia predicciones en las palabras de cada clase/bin como el empírico. GRID30 es una media uniforme sobre puntos funcionales: no se interpreta su diferencia celda a celda frente a cinco bins como RMSE. La comparación de curvas GRID30 es descriptiva.</p><div class="scroll">'
    subset=fidelity.loc[fidelity.region.eq('ALL')&fidelity.case.isin(['FULL','OOF_mean'])]
    html+=subset.to_html(index=False,border=0,float_format=lambda v:f'{v:.7g}')+'</div>'+''.join(sections)
    html+='<p>Las tablas FULL_LOCO.csv, decisions.csv y fidelity.csv conservan valores y discrepancias. Que un modelo cambie un signo no demuestra una mejora: debe evaluarse junto con su fidelidad OOF y MATCHED5. Los signos son globales y no exigen monotonicidad local.</p></html>'
    (out/'INFORME.html').write_text(html,encoding='utf-8')

def export_results(ctx):
    with _lock(ctx['root']):
        _load_selection(ctx)
        req(readj(ctx['root']/'execution_status.json')['status']=='COMPLETE','Termina primero FULL/LOCO.')
        req(ref.completed85(ctx['root']/'report',digest(dict(parent=ctx['id'],stage='report'))) is not None,'Reporte incompleto.')
        files=[]
        for name in ('protocol.json','source_ledger.json','environment.json','input_audit.csv','empirical_FULL_LOCO.csv','calibration.csv','execution_status.json','resources.csv'):
            p=ctx['root']/name
            if p.is_file():files.append(p)
        for name in ('inputs','OOF','FULL_LOCO','report'):
            files.extend(p for p in (ctx['root']/name).rglob('*') if p.is_file() and p.name!='RUNNING.lock' and '.partial' not in p.name)
        files.extend(p for p in ctx['out85'].iterdir() if p.is_file())
        fam=next(f for f in ref.FAMILIES85 if f['name']==ctx['selected90']['family'])
        for n in CORE3:
            for fold in ctx['folds']:
                folder,ident=ref.location85(ctx,n,fold,fam,ctx['alpha88'])
                req(ref.completed85(folder,ident) is not None,'Falta checkpoint seleccionado al exportar.')
                files.extend(p for p in folder.iterdir() if p.is_file())
        filename=ctx['root']/('E2_W2V_K40_modelo_'+ctx['id'][:12]+'_resultados.zip');temp=filename.with_suffix('.partial')
        hashes={}
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(files)):
                name=p.relative_to(ctx['root']).as_posix();z.write(p,name);hashes[name]=sha(p)
            for name in ('e2_k40_model.py','e2_k40_model_reference.py','PACKAGE_SHA256.json'):
                p=PACKAGE/name;z.write(p,'code/'+name);hashes['code/'+name]=sha(p)
            z.writestr('EXPORT_SHA256.json',json.dumps(hashes,indent=2))
            z.writestr('ALCANCE_EXPORT.txt','Incluye los 75 checkpoints del ganador y diagnósticos de 3.825 ajustes. Los demás checkpoints CV permanecen en la corrida local. No se declara reproducción de ensayos clasificadores.\n')
        with zipfile.ZipFile(temp) as z:req(z.testzip() is None,'ZIP dañado.')
        os.replace(temp,filename);print(filename,flush=True);return filename
