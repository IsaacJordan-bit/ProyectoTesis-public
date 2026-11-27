"""P18 retrieval kernel. Stable identities permit later 33/30/7 -> 98/51/10 reuse.

The driver controls the requested panel. This module never expands that panel.
"""
from __future__ import annotations
import hashlib, json, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
import e6_p18_static as static
import e6_p18_core as core
import e6_p18_numeric as numeric

NETS=tuple(static.CORE3)
CAL_BETAS=.01+.005*np.arange(98)
req=core.req85
sha=core.file_hash
digest=core.digest
readj=core.read_json
writej=core.atomic_json
frame=core.save_frame
savez=core.save_npz

class TechnicalPause(RuntimeError): pass

def guard(c):
    if (Path(c['root'])/'STOP').exists() or (Path(c['cache'])/'STOP').exists():
        raise TechnicalPause('STOP presente. Checkpoints conservados; elimina STOP para reanudar.')
    if psutil.virtual_memory().available/2**30<c['reserve_gib']:
        raise TechnicalPause('RAM disponible por debajo de la reserva. Reanuda tras liberar memoria.')

def seal(folder,identity,names,summary):
    folder=Path(folder)
    writej(folder/'done.json',dict(identity=identity,files={n:sha(folder/n) for n in names},summary=summary))

def cached(folder,identity):
    folder=Path(folder);p=folder/'done.json'
    if not p.is_file():return None
    r=readj(p);req(r['identity']==identity,'Identidad de checkpoint distinta: '+str(folder))
    for name,h in r['files'].items():
        q=(folder/name).resolve()
        req(q.is_relative_to(folder.resolve()) and q.is_file() and sha(q)==h,'Checkpoint alterado: '+str(q))
    return r['summary']

def split_arrays(c,seed):
    p=Path(c['cache'])/'inputs'/f's{seed:02d}.npz'
    req(sha(p)==c['split_files'][str(seed)],'Entrada de split modificada: '+str(p))
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def scores(X,Q):
    norms=np.linalg.norm(X,axis=1)
    req(np.isfinite(X).all() and np.isfinite(Q).all() and (norms>0).all(),'Vectores inválidos.')
    median=float(np.median(norms))
    keys=X*(median/norms)[:,None]
    return (keys@Q.T).T,median

def prepared(raw):
    s,o,m=numeric.prepare_scores(raw,np.ones(raw.shape[1]))
    return s,o,m,numeric.prefixes(s,m)

def curve(prep,classes,ni,scale,betas=CAL_BETAS):
    s,o,m,pref=prep;C=len(np.unique(classes));den=np.bincount(classes,minlength=C)
    req((den>0).all(),'Clase TRAIN ausente en calibración.')
    out=np.empty((C,len(betas)))
    for j,b in enumerate(betas):
        f=numeric.fraction(s,m,pref,float(b)*float(scale),ni)
        out[:,j]=np.bincount(classes,weights=f,minlength=C)/den
    return out

def calibration_location(c,seed):
    return Path(c['cache'])/'cal'/f's{seed:02d}',digest(dict(parent=c['kernel_id'],stage='keynorm_calibration',split=seed))

def calibrate_split(c,seed):
    folder,identity=calibration_location(c,seed);old=cached(folder,identity)
    if old is not None:return dict(reused=True,split=seed,seconds=0.)
    guard(c);start=time.perf_counter();z=split_arrays(c,seed);X=z['X'];y=z['y']
    with threadpool_limits(limits=1):
        raw,median=scores(X,X);prep=prepared(raw);target=curve(prep,y,0,1.)
        rows=[];traces=[];curves=[]
        for ni,n in enumerate(NETS):
            seen={}
            def objective(loglam):
                guard(c);lam=float(np.exp(loglam));key=lam.hex()
                if key not in seen:
                    q=curve(prep,y,ni,lam);seen[key]=(lam,float(np.square(np.log(q)-np.log(target)).mean()),q)
                return seen[key][1]
            before=objective(0.)
            if ni:
                objective(np.log(.001))
                opt=minimize_scalar(objective,bounds=(np.log(.001),0.),method='bounded',options=dict(xatol=.015,maxiter=32))
                req(opt.success,'Calibración sin convergencia; no se cambian límites.')
            lam,loss,q=min(seen.values(),key=lambda x:(x[1],abs(np.log(x[0]))))
            req(loss<=before+1e-12,'Calibración empeoró la referencia lambda=1.')
            # Independent bisection anchors, not the optimized prefix operator.
            anchor=raw[:min(12,len(X))];s,o,m,pref=prepared(anchor);deltas=[]
            for b in (CAL_BETAS[0],CAL_BETAS[18],CAL_BETAS[-1]):
                a=static.attention_weights((anchor*(b*lam)).T,n).T
                f0=1/(len(X)*np.square(a).sum(1))
                f1=numeric.fraction(s,m,pref,b*lam,ni)
                deltas.append(float(abs(f0-f1).max()))
            req(max(deltas)<1e-8,'N_eff optimizado no reproduce bisección.')
            rows.append(dict(network=n,split_seed=seed,lambda_value=lam,median_train_norm=median,
                objective_reference=before,objective_calibrated=loss,anchor_max_delta=max(deltas),
                train_array_sha=hashlib.sha256(X.astype('<f8').tobytes()).hexdigest(),
                train_words_sha=static.digest85(z['train_words'].tolist()),
                calibration_uses_validation=False,calibration_uses_errors=False,calibration_betas=98,
                lambda_at_boundary=bool(ni and min(abs(np.log(lam/.001)),abs(np.log(lam)))<=.015),
                n_memories=len(X),reference='Softmax_keynorm_same_TRAIN_SELF'))
            traces.extend(dict(network=n,lambda_value=a,objective=v) for a,v,_ in seen.values())
            curves.append(q)
    writej(folder/'calibration.json',rows);writej(folder/'trace.json',traces)
    savez(folder/'curves.npz',target=target,curves=np.stack(curves),betas=CAL_BETAS)
    result=dict(split=seed,seconds=time.perf_counter()-start,reused=False)
    seal(folder,identity,['calibration.json','trace.json','curves.npz'],result)
    return result

def unit_location(c,seed,ni,beta_index,classifier_seed):
    # Absolute beta index in the frozen 98-point grid, never its position in a panel.
    folder=Path(c['cache'])/'trials'/f's{seed:02d}'/f'n{ni}'/f'b{beta_index:02d}'
    identity=digest(dict(parent=c['kernel_id'],stage='classifier_seed',split=seed,network=NETS[ni],
        beta_index=beta_index,beta=float(CAL_BETAS[beta_index]),classifier_seed=classifier_seed,
        calibration_identity=calibration_location(c,seed)[1]))
    return folder/f'r{classifier_seed:02d}.npz',identity

def payload_digest(identity,pred,ties,truth,words,warning_text):
    return digest(dict(identity=identity,prediction=np.asarray(pred).tolist(),tie=np.asarray(ties).tolist(),
        truth=np.asarray(truth).tolist(),words=np.asarray(words).tolist(),warnings=warning_text))

def read_unit(path,identity):
    if not Path(path).is_file():return None
    with np.load(path,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    req(str(d['identity'])==identity,'Ensayo de otra configuración: '+str(path))
    req(str(d['payload_sha'])==payload_digest(identity,d['prediction'],d['tie'],d['truth'],d['words'],str(d['warnings'])),
        'Checksum interno de ensayo inválido: '+str(path))
    req(d['prediction'].shape==d['truth'].shape==d['words'].shape==d['tie'].shape and
        np.isin(d['prediction'],range(8)).all() and np.isin(d['truth'],range(8)).all(),'Predicción/etiqueta inválida.')
    return d

def trial_beta(c,seed,ni,beta_index,classifier_seeds):
    start=time.perf_counter();pending=[]
    for r in classifier_seeds:
        if read_unit(*unit_location(c,seed,ni,beta_index,r)) is None:pending.append(r)
    if not pending:return dict(split=seed,network=NETS[ni],beta_index=beta_index,new=0,reused=len(classifier_seeds),seconds=0.)
    guard(c);folder,ident=calibration_location(c,seed)
    req(cached(folder,ident) is not None,'Falta calibración keynorm autenticada.')
    cal=readj(folder/'calibration.json')[ni];lam=cal['lambda_value'];z=split_arrays(c,seed)
    X=z['X'];Q=z['Q'];beta=float(CAL_BETAS[beta_index]);gain=beta*lam
    with threadpool_limits(limits=1):
        query=np.vstack([X,Q]);raw,median=scores(X,query);s,o,m,pref=prepared(raw)
        recalled=numeric.recall_sorted(s,o,m,pref,gain,ni,X)
        anchors=np.unique([0,len(X)-1,len(X),len(query)-1])
        direct=static.attention_weights((raw[anchors]*gain).T,NETS[ni]).T@X
        req(np.allclose(recalled[anchors],direct,rtol=1e-9,atol=1e-11),'Recuperación optimizada no reproduce bisección.')
        R,T=recalled[:len(X)],recalled[len(X):]
        req(abs(median-cal['median_train_norm'])<=1e-12,'Norma de TRAIN diferente.')
        for r in pending:
            guard(c);model=static.MulticlassEnsembleClassifier(pd.DataFrame(R,index=z['train_words']),c['neighbors'],random_seed_cls=int(r),n_jobs=1,voting_n_jobs=1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit_only();pred=model.classify(T)
            diag=model.hard_vote_diagnostics(T,ensemble_predictions=pred)
            req(np.all(diag['tie_winner_is_smallest_class'][diag['tie']]),'Desempate hard-vote distinto.')
            warning_text=json.dumps([dict(category=w.category.__name__,message=str(w.message)) for w in caught],ensure_ascii=False)
            path,identity=unit_location(c,seed,ni,beta_index,r)
            pred=np.asarray(pred,np.int32);truth=np.asarray(z['truth'],np.int32);tie=np.asarray(diag['tie'],bool);words=z['query_words']
            # One atomic file per classifier seed. Interruptions lose at most the active fit.
            savez(path,identity=np.array(identity),prediction=pred,tie=tie,truth=truth,words=words,
                warnings=np.array(warning_text),payload_sha=np.array(payload_digest(identity,pred,tie,truth,words,warning_text)),
                beta=np.array(beta),lambda_value=np.array(lam),classifier_seed=np.array(r))
    return dict(split=seed,network=NETS[ni],beta_index=beta_index,new=len(pending),reused=len(classifier_seeds)-len(pending),seconds=time.perf_counter()-start)
