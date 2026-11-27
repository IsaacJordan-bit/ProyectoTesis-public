"""Fixed P17 recipe, 98-point OOF fidelity and genuine statistical FULL/LOCO refits.
No family/ridge selection on the extended outcomes. Fractions retain unit weights.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import e6_p18_core as core
import e6_p18_compute as compute
from e6_p20 import (MODELS, NETS, PACKAGE, req, readj, writej, readcsv, frame, savez,
    digest, sha, cached, seal, stage, worker_context, run_tasks, BETAS)

def load_analysis(main,arm):
    root=main['root']/"analysis"/arm
    g=main['geometries']['S']
    receipt=readj(main['root']/'counts_complete.json')
    req(receipt['run_id']==main['run_id'], 'Conteos de otra ampliación.')
    counts={}
    for ni,n in enumerate(NETS):
        path=main['root']/'counts'/f'{arm}_n{ni}.parquet'
        req(sha(path)==receipt['files'][path.name], 'Conteos alterados: '+str(path))
        counts[n]=pd.read_parquet(path)
    cols=['word','source_word','true_class','beta','beta_key','validation_similarity','n_items']
    data=counts[NETS[0]][cols].copy()
    for n in NETS[1:]:pd.testing.assert_frame_equal(data,counts[n][cols])
    Y=np.column_stack([counts[n]['rate'].to_numpy(float) for n in NETS])
    classes=sorted(data.true_class.unique().tolist());C=len(classes)
    base={k:main[k] for k in ('cfg','cache','kernel_id','split_files','source_stats')}
    base['geometries']={'S':{'neighbors':main['geometries']['S']['neighbors']}}
    c=dict(base,root=root,main_root=main['root'],arm=arm,betas=BETAS.copy(),keys=np.round(BETAS,12),
        counts=counts,data=data,Y=Y,weights=core.weights88(data),classes=classes,labels=list(g['neighbors']),
        full=g['full'],observed_full=g['full'],common=g['common'],edges=np.asarray(g['edges']),geometry=g)
    c['run_id']=digest(dict(parent=main['run_id'],arm=arm,counts=[receipt['files'][f'{arm}_n{i}.parquet'] for i in range(3)]))
    meta=c['common'].set_index('word');bid=data.word.map(meta.similarity_bin);valid=bid.notna().to_numpy()
    cov=c['common'].groupby(['true_class','similarity_bin']).size();bb=bid.fillna(0).to_numpy(int)
    bi=pd.Index(c['keys']).get_indexer(data.beta_key)
    req((bi>=0).all(),'Beta fuera de la grilla.')
    den=np.array([cov.loc[cl,k] if v else 1 for cl,k,v in zip(data.true_class,bb,valid)])
    c.update(surface_codes=bb*len(BETAS)+bi,surface_weights=valid.astype(float)/(C*den),row_classes=data.true_class.to_numpy())
    units=data[['word','source_word','true_class']].drop_duplicates('word').sort_values('word').reset_index(drop=True)
    rows=[]
    for rep in range(5):
        for fold,(a,b) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=rep).split(units,units.true_class)):
            rows.extend(dict(word=w,repeat=rep,fold=fold) for w in units.iloc[b].word)
    ledger=pd.DataFrame(rows)
    common_ledger=main['root']/'inputs'/'cv_folds.parquet'
    pd.testing.assert_frame_equal(ledger,pd.read_parquet(common_ledger),check_dtype=False)
    folds=[]
    for rep in range(5):
        ld=ledger.loc[ledger.repeat.eq(rep)]
        req(len(ld)==len(units) and not ld.word.duplicated().any(),'Cobertura CV duplicada/incompleta.')
        for f in range(5):
            test=units.loc[units.word.isin(ld.loc[ld.fold.eq(f),'word'])];train=units.loc[~units.word.isin(test.word)]
            req(set(train.true_class)==set(classes)==set(test.true_class),'CV sin todas las clases.')
            req(not set(train.source_word)&set(test.source_word),'CV comparte palabra de origen.')
            folds.append(dict(repeat=rep,fold=f,train=np.flatnonzero(data.word.isin(train.word)),test=np.flatnonzero(data.word.isin(test.word))))
    c['folds']=folds;frame(root/'CV/folds.parquet',ledger)
    return c

def surface(c,values,omitted=()):
    values=np.asarray(values,float);req(values.shape==(len(c['data']),),'Forma incorrecta para superficie.')
    w=c['surface_weights'].copy()
    if omitted:w[np.isin(c['row_classes'],omitted)]=0;w*=len(c["classes"])/(len(c["classes"])-len(omitted))
    req(np.isfinite(values[w>0]).all(),'Predicciones no finitas en COMMON.')
    a=np.bincount(c['surface_codes'],weights=w*np.where(w>0,values,0),minlength=5*len(c['betas'])).reshape(5,len(c['betas']))
    return pd.DataFrame(a,index=(c['edges'][:-1]+c['edges'][1:])/2,columns=c['betas'])

def trends(s):
    t=core.trends85(s);t['T1T2']=bool(t['T1']>0 and t['T2']<0)
    return t

def curves(s):
    x=s.index.to_numpy(float);u=(x-x.mean())/np.square(x-x.mean()).sum();a=s.to_numpy(float)
    return u@a,a.mean(axis=0)

def fidelity(c,E,M):
    delta=M.to_numpy()-E.to_numpy();es,em=curves(E);ms,mm=curves(M)
    result=dict(RMSE_matched5=float(np.sqrt(np.square(delta).mean())),MAE_matched5=float(abs(delta).mean()),bias_matched5=float(delta.mean()),max_abs_matched5=float(abs(delta).max()),RMSE_similarity_slope_curve=float(np.sqrt(np.square(ms-es).mean())),RMSE_beta_mean_curve=float(np.sqrt(np.square(mm-em).mean())))
    result['OOF_curve_RMSE']=float(np.sqrt(result['RMSE_beta_mean_curve']**2+np.var(E.index.to_numpy())*result['RMSE_similarity_slope_curve']**2))
    for label,mask in [('beta_le_010',c['betas']<=.10+1e-12),('beta_gt_010',c['betas']>.10+1e-12)]:result['RMSE_'+label]=float(np.sqrt(np.square(delta[:,mask]).mean()))
    return result

def stat_location(c,label,*parts):
    folder=c['root']/label
    for p in parts:folder=folder/str(p)
    return folder,digest(dict(panel=c['run_id'],stage=label,parts=parts))

def cv_task(c,mi,fi,ni):
    folder,identity=stat_location(c,'CV',f'm{mi}',f'f{fi:02d}',f'n{ni}')
    old=cached(folder,identity)
    if old is not None:return old
    compute.guard(worker_context(c));start=time.perf_counter();model=MODELS[mi];f=c['folds'][fi]
    a=c['data'].iloc[f['train']];b=c['data'].iloc[f['test']];family=next(v for v in core.model_families(len(c["classes"])) if v['name']==model['family'])
    with threadpool_limits(limits=1):
        theta,state,info=core.fit_model(c,a,c['Y'][f['train'],ni],family,model['alpha'])
        pred=core.predict90(b,state,theta)
    req(np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all(),'OOF inválida.')
    savez(folder/'fit.npz',theta=theta,prediction=pred,row_ids=f['test'])
    writej(folder/'preprocessing.json',core.state90(state));writej(folder/'diagnostics.json',info)
    row=dict(arm=c['arm'],model=model['name'],candidate=core.candidate_name(model['family'],model['alpha']),network=NETS[ni],repeat=f['repeat'],fold=f['fold'],stable=info['stable'],gradient_inf=info['gradient_inf'],seconds=time.perf_counter()-start)
    seal(folder,identity,['fit.npz','preprocessing.json','diagnostics.json'],row)
    return row

def run_oof(c):
    with stage(c,'STATISTICAL_OOF'):
        tasks=[(c,mi,fi,ni) for mi in range(len(MODELS)) for fi in range(25) for ni in range(3)
            if cached(*stat_location(c,'CV',f'm{mi}',f'f{fi:02d}',f'n{ni}')) is None]
        run_tasks(c,'STATISTICAL_OOF',cv_task,tasks,need=.7+len(c['data'])*(16*len(c['classes']))*8*6/2**30)
        rows=[];anchors=[];curve_rows=[]
        with threadpool_limits(limits=1):
            for mi,model in enumerate(MODELS):
                for ni,n in enumerate(NETS):
                    reps=[];E=surface(c,c['Y'][:,ni]);family=next(f for f in core.model_families(len(c["classes"])) if f['name']==model['family'])
                    for rep in range(5):
                        pred=np.full(len(c['data']),np.nan);coverage=np.zeros(len(pred),int)
                        for fi,f in enumerate(c['folds']):
                            if f['repeat']!=rep:continue
                            folder,identity=stat_location(c,'CV',f'm{mi}',f'f{fi:02d}',f'n{ni}');req(cached(folder,identity) is not None,'Falta CV.')
                            with np.load(folder/'fit.npz',allow_pickle=False) as z:ids=z['row_ids'];theta=z['theta'];p=z['prediction']
                            req(np.array_equal(ids,f['test']),'Índices OOF distintos.')
                            a=c['data'].iloc[f['train']];X,w,state=core.design90(c,a,family)
                            _,grad,_=core._objective85R(X,c['Y'][f['train'],ni],w,model['alpha'],theta)
                            replay=core.predict90(c['data'].iloc[ids],state,theta);delta=float(abs(replay-p).max())
                            req(np.max(abs(grad))<=2e-6 and delta<=1e-10,'Coeficientes OOF no reproducidos.')
                            pred[ids]=p;coverage[ids]+=1
                            anchors.append(dict(model=model['name'],network=n,repeat=rep,fold=f['fold'],gradient_inf=float(abs(grad).max()),prediction_delta=delta))
                        req((coverage==1).all() and np.isfinite(pred).all(),'Cobertura OOF incompleta.')
                        M=surface(c,pred);y=c['Y'][:,ni];pc=np.clip(pred,1e-12,1-1e-12)
                        row=dict(arm=c['arm'],model=model['name'],candidate=core.candidate_name(model['family'],model['alpha']),network=n,repeat=rep,
                            OOF_logloss=float(c['weights']@(-y*np.log(pc)-(1-y)*np.log1p(-pc))),
                            OOF_brier_fraction=float(c['weights']@np.square(y-pred)),**fidelity(c,E,M))
                        for prefix,v in [('empirical',E),('OOF',M)]:
                            row.update({prefix+'_'+k:t for k,t in trends(v).items()})
                        rows.append(row);reps.append(pred)
                    avg=np.mean(reps,axis=0);M=surface(c,avg);frame(c['root']/'OOF'/f'm{mi}_n{ni}.csv',M,index=True)
                    savez(c['root']/'OOF'/f'm{mi}_n{ni}.npz',prediction=avg)
                    es,em=curves(E);ms,mm=curves(M)
                    curve_rows.extend(dict(model=model['name'],network=n,beta=b,empirical_mean=em[j],OOF_mean=mm[j],empirical_slope=es[j],OOF_slope=ms[j]) for j,b in enumerate(c['betas']))
        frame(c['root']/'OOF_metrics.csv',pd.DataFrame(rows));frame(c['root']/'OOF_anchors.csv',pd.DataFrame(anchors));frame(c['root']/'OOF_curves.csv',pd.DataFrame(curve_rows))
    return pd.DataFrame(rows)

def refit_task(c,mi,ni,omitted=()):
    tag='FULL' if not omitted else f'L{omitted[0]:02d}'
    folder,identity=stat_location(c,'fits',f'm{mi}',f'n{ni}',tag);old=cached(folder,identity)
    if old is not None:return old
    compute.guard(worker_context(c));model=MODELS[mi];ids=np.flatnonzero(~np.isin(c['row_classes'],omitted));d=c['data'].iloc[ids]
    family=next(f for f in core.model_families(len(c["classes"])-len(omitted)) if f['name']==model['family'])
    with threadpool_limits(limits=1):
        theta,state,info=core.fit_model(c,d,c['Y'][ids,ni],family,model['alpha'])
        pred=core.predict90(d,state,theta);allpred=np.full(len(c['data']),np.nan);allpred[ids]=pred
        E=surface(c,c['Y'][:,ni],omitted);M=surface(c,allpred,omitted);G,cube=core.grid_predictions(c,state,theta)
    row=dict(model=model['name'],candidate=core.candidate_name(model['family'],model['alpha']),arm=c['arm'],network=NETS[ni],case=tag,
        omitted_label='' if not omitted else c['labels'][omitted[0]],n_classes=len(c["classes"])-len(omitted),n_units=d.word.nunique(),
        family=model['family'],alpha=model['alpha'],n_parameters=family['p'],stable=info['stable'],gradient_inf=info['gradient_inf'],**fidelity(c,E,M))
    for prefix,s in [('empirical',E),('matched5',M),('grid30',G)]:
        t=trends(s)
        for k in ('T1','T2','T3'):row[prefix+'_'+k]=t[k]
        row[prefix+'_T1T2']=t['T1T2'];row[prefix+'_ALL3']=t['ALL3_expected']
    for k in ('T1','T2','T3'):
        row['fit_delta_'+k]=row['matched5_'+k]-row['empirical_'+k]
        row['grid_delta_'+k]=row['grid30_'+k]-row['matched5_'+k]
    names=['fit.npz','preprocessing.json','diagnostics.json','summary.json','empirical.csv','matched5.csv','grid30.csv']
    if omitted:
        full,_=stat_location(c,'fits',f'm{mi}',f'n{ni}','FULL')
        with np.load(full/'fit.npz',allow_pickle=False) as z:
            only=pd.DataFrame(z['class_grid'][~np.isin(z['classes'],omitted)].mean(0),index=G.index,columns=G.columns)
            full_grid=pd.DataFrame(z['class_grid'].mean(0),index=G.index,columns=G.columns)
        row['RMSE_refit_vs_aggregation_only']=float(np.sqrt(np.square(G.to_numpy()-only.to_numpy()).mean()))
        for k,v in trends(only).items():row['aggregation_only_'+k]=v
        for k in ('T1','T2','T3'):row['relative_change_'+k]=(row['grid30_'+k]-trends(full_grid)[k])/max(abs(trends(full_grid)[k]),1e-12)
        frame(folder/'aggregation_only.csv',only,index=True);names.append('aggregation_only.csv')
    savez(folder/'fit.npz',theta=theta,prediction=pred,row_ids=ids,classes=np.asarray(state['classes']),class_grid=cube)
    writej(folder/'preprocessing.json',core.state90(state));writej(folder/'diagnostics.json',info)
    for name,s in [('empirical',E),('matched5',M),('grid30',G)]:frame(folder/(name+'.csv'),s,index=True)
    writej(folder/'summary.json',row);seal(folder,identity,names,row)
    return row

def summarize_robustness(fits):
    rows=[]
    for (candidate,arm,n),d in fits.groupby(['candidate','arm','network'],sort=False):
        f=d.loc[d.case.eq('FULL')].iloc[0];l=d.loc[~d.case.eq('FULL')]
        for scope in ('empirical','matched5','grid30'):
            row=dict(candidate=candidate,arm=arm,network=n,scope=scope,n_LOCO=len(l),n_classes=int(f.n_classes),
                FULL_T1T2=bool(f[scope+'_T1T2']),LOCO_T1T2_passed=int(l[scope+'_T1T2'].sum()),
                R_T1T2=bool(f[scope+'_T1T2'] and l[scope+'_T1T2'].all()))
            for k,sign in [('T1',1),('T2',-1),('T3',-1)]:
                v=sign*l[scope+'_'+k];at=v.idxmin()
                row.update({k+'_FULL':f[scope+'_'+k],k+'_LOCO_passed':int((v>0).sum()),
                    k+'_worst_signed_margin':float(v.min()),k+'_worst_omission':l.loc[at,'omitted_label'],
                    k+'_robust':bool(sign*f[scope+'_'+k]>0 and (v>0).all())})
            rows.append(row)
    return pd.DataFrame(rows)

def run_refits(c):
    with stage(c,'FULL_LOCO'):
        for cases,label in [([()], 'FULL'),([(i,) for i in range(len(c["classes"]))],'LOCO')]:
            tasks=[]
            for mi in range(len(MODELS)):
                for ni in range(3):
                    for omitted in cases:
                        tag='FULL' if not omitted else f'L{omitted[0]:02d}'
                        if cached(*stat_location(c,'fits',f'm{mi}',f'n{ni}',tag)) is None:tasks.append((c,mi,ni,omitted))
            run_tasks(c,'STATISTICAL_'+label,refit_task,tasks,need=.7+len(c['data'])*(16*len(c['classes']))*8*6/2**30)
        rows=[cached(*stat_location(c,'fits',f'm{mi}',f'n{ni}',tag)) for mi in range(len(MODELS)) for ni in range(3) for tag in ['FULL',*[f'L{i:02d}' for i in range(len(c["classes"]))]]]
        req(all(r is not None and r['stable'] for r in rows),'Faltan reajustes estables.')
        fits=pd.DataFrame(rows);frame(c['root']/'FULL_LOCO.csv',fits);frame(c['root']/'robustness.csv',summarize_robustness(fits))
    return fits

