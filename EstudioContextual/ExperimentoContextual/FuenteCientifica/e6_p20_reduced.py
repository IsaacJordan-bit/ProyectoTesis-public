"""P20 fixed 33 x 51 x 3 panel. Reuses authenticated original P20 trials.

The original module, full-grid calibration and trial identities remain unchanged.
This module owns a separate reduced-panel analysis and completion receipt.
"""
from e6_p20 import *
import e6_p20 as bank_api

VERSION_REDUCED='P20_33x51x3_V1'
BETA_INDICES=tuple(int(i) for i in np.rint(np.linspace(0,97,33)))
BETAS=bank_api.BETAS[np.array(BETA_INDICES)].copy()
SPLITS=tuple(range(51))
CLASSIFIER_SEEDS=(0,1,2)
EXPECTED_ENSEMBLES=33*51*3*3*3

def verify_addon():
    bank_api.verify_package()
    manifest=readj(PACKAGE/'REDUCED_SHA256.json')
    for name,h in manifest.items():
        req(sha(PACKAGE/name)==h,'Reduced-panel package altered: '+name)
    req(len(BETA_INDICES)==len(set(BETA_INDICES))==33 and BETA_INDICES[0]==0 and BETA_INDICES[-1]==97,
        'Invalid fixed beta subset.')
    return manifest

def guard(c,need=0.):
    bank_api.guard(c,need)
    if (Path(c['cache'])/'STOP').exists():
        raise compute.TechnicalPause('STOP present in original trial bank. Checkpoints retained.')

@contextlib.contextmanager
def stage(c,name):
    root=Path(c.get('main_root',c['root']))
    # Share the original bank lock: full and reduced runs must not write concurrently.
    with core.run_lock(c['cache']):
        try:
            guard(c);yield;guard(c)
            writej(root/'execution_status.json',dict(status='STAGE_COMPLETE',stage=name,resumable=True))
        except BaseException as e:
            writej(root/'execution_status.json',dict(status='TECHNICAL_PAUSE',stage=name,error=repr(e),resumable=True))
            raise

def prepare(bank):
    code=verify_addon();contract=bank['contract']
    req(contract['version']==bank_api.VERSION,'Wrong original experiment version.')
    req(list(contract['splits'])==list(SPLITS) and list(contract['seeds'])==list(range(10)),
        'Original trial bank is not the 98 x 51 x 10 protocol.')
    np.testing.assert_array_equal(contract['betas'],bank_api.BETAS)
    protocol=dict(version=VERSION_REDUCED,source_bank_id=bank['run_id'],source_contract_sha=digest(contract),
        original_beta_indices=BETA_INDICES,betas=BETAS,splits=SPLITS,classifier_seeds=CLASSIFIER_SEEDS,
        required_ensembles=EXPECTED_ENSEMBLES,arms=ARMS,networks=NETS,models=MODELS,code=code,
        beta_selection='Outcome-independent: round(linspace(0,97,33)); retain original indices.',
        calibration='Original shared TRAIN-A calibration on all 98 betas, unchanged.',
        geometry='Original S FULL/COMMON, all 51 splits, shared word-level CV folds, unchanged.',
        interpretation='Paired reduced-grid experiment; not a claim of completed 98 x 51 x 10 execution.',
        analysis='Fixed model families and penalties refitted on the selected 33-beta panel.',
        source_environment=contract['environment'])
    rid=digest(protocol);root=bank['cfg'].output_root/('p20r_'+rid[:12]);root.mkdir(parents=True,exist_ok=True)
    c={k:bank[k] for k in ('cfg','cache','kernel_id','split_files','source_stats','geometries')}
    c.update(root=root,run_id=rid,bank=bank,protocol=protocol)
    with stage(c,'PREPARE_REDUCED_PANEL'):
        frozen(root/'protocol.json',protocol)
        writej(root/'source_bank.json',dict(root=str(bank['root']),run_id=bank['run_id']))
        for name in ('cv_folds.parquet','full.parquet','common.parquet','expected.parquet','geometry.json'):
            origin=bank['root']/'inputs'/name;dest=root/'inputs'/name;dest.parent.mkdir(exist_ok=True)
            if dest.exists():req(sha(dest)==sha(origin),'Reduced input metadata changed: '+name)
            else:shutil.copy2(origin,dest)
        frame(root/'betas.csv',pd.DataFrame(dict(panel_index=range(len(BETAS)),original_beta_index=BETA_INDICES,beta=BETAS)))
        code_dir=root/'code';code_dir.mkdir(exist_ok=True)
        for name in (*code,'REDUCED_SHA256.json'):
            shutil.copy2(PACKAGE/name,code_dir/name)
    print('Original checkpoint bank:',bank['root'])
    print('Reduced panel results:',root)
    print('33 betas x 51 splits x 3 classifier seeds x 3 memories x 3 networks = 45,441 ensembles.')
    print('Classifier seeds:',CLASSIFIER_SEEDS,'| Worker limit:',c['cfg'].max_workers)
    print('Calibration remains on the original 98-beta grid.')
    return c

def status(c,authenticate=False):
    bank=c['bank'];rows=[]
    for memory in 'APO':
        completed=0
        for split in SPLITS:
            for ni in range(3):
                for bi in BETA_INDICES:
                    for r in CLASSIFIER_SEEDS:
                        path,iid=trial_location(bank,memory,split,ni,bi,r)
                        completed+=int(load_trial(path,iid) is not None if authenticate else path.is_file())
        required=len(SPLITS)*len(BETAS)*len(CLASSIFIER_SEEDS)*3
        rows.append(dict(memory=memory,selected_complete=completed,selected_pending=required-completed,
            selected_required=required,percent=100*completed/required))
    print('Selected-panel progress only. Each reused trial is authenticated before use.')
    return pd.DataFrame(rows)

def run(c):
    """Resume selected original-index trials; never mark the full-grid bank complete."""
    return _run_selected(c['bank'],c)

def _run_selected(c,panel):
    """Use original scientific workers, calibration and trial identities."""
    with stage(panel,'REDUCED_CONTROLLED_TRIALS'):
        # Deterministic schedule; each completed ensemble is atomic and reusable.
        for seed in SPLITS:
            scales=calibrate(c,seed)
            for memory in 'APO':
                guard(c);clear_data();a=representations(c,seed,memory)
                seeds=list(CLASSIFIER_SEEDS);bis=list(BETA_INDICES)
                pending=[bi for bi in bis if any(load_trial(*trial_location(c,memory,seed,ni,bi,r)) is None for ni in range(3) for r in seeds)]
                if not pending:
                    print(f'{memory} split {seed}: requested trials complete.',flush=True);continue
                blocks=[(l,min(l+c['cfg'].query_block,a['n_query'])) for l in range(0,a['n_query'],c['cfg'].query_block)]
                N=len(a['X']);D=a['dimension'];nq=a['n_query'];chunk=c['cfg'].beta_chunk
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
                writej(panel['root']/'progress'/f'{memory}_s{seed:02d}.json',dict(panel_id=panel['run_id'],original_beta_indices=BETA_INDICES,classifier_seeds=CLASSIFIER_SEEDS,ensembles=len(BETAS)*len(CLASSIFIER_SEEDS)*3))
                shutil.rmtree(scratch(c,memory,seed),ignore_errors=True)
        clear_data()
    return status(panel)

def aggregate(c):
    """Complete balanced panel only; integer trial denominators remain explicit."""
    bank=c['bank'];receipt=c['root']/'counts_complete.json'
    if receipt.exists():
        r=readj(receipt);req(r['run_id']==c['run_id'],'Counts belong to another run.')
        req(r['verified_ensembles']==EXPECTED_ENSEMBLES,'Wrong authenticated ensemble count.')
        req(sha(c['root']/'verified_trial_ledger.parquet')==r['trial_ledger_sha'],'Verified trial ledger changed.')
        for name,h in r['files'].items():req(sha(c['root']/'counts'/name)==h,'Counts changed.')
        print('Verified complete counts reused.');return pd.DataFrame(r['summary'])
    with stage(c,'AGGREGATE_COMPLETE_PANEL'):
        full=c['geometries']['S']['full'].sort_values('word').reset_index(drop=True)
        words=full.word.tolist();pos={w:i for i,w in enumerate(words)};W=len(words)
        errors=np.zeros((9,3,W,len(BETAS)),np.int64);ties=np.zeros_like(errors);exposure=np.zeros((W,len(BETAS)),np.int64)
        changes=np.zeros((6,3,W,len(BETAS),3),np.int64);last=time.monotonic();verified=0;ledger=[]
        for seed in SPLITS:
            guard(c);z=split_arrays(bank,seed);maps_path=bank['root']/'maps'/f's{seed:02d}.npz'
            req(maps_path.exists(),'Missing canonical query mapping. Run all trials first.')
            m=arrays(maps_path);qw=m['query_words'];wi=m['word_index'];idx=np.array([pos[w] for w in qw])
            req(np.array_equal(qw,z['query_words']) and np.array_equal(m['truth'],z['truth']),'Mapping identity mismatch.')
            req(len(m['occ_ids'])==100*len(qw) and len(np.unique(m['occ_ids']))==len(m['occ_ids']),'Invalid canonical positions.')
            req(np.array_equal(np.bincount(wi,minlength=len(qw)),np.full(len(qw),100)),'Word position counts differ.')
            truth=m['truth'][wi]
            for bi,original_bi in enumerate(BETA_INDICES):
                exposure[idx,bi]+=len(CLASSIFIER_SEEDS)
                for ni in range(3):
                    for r in CLASSIFIER_SEEDS:
                        err={}
                        for memory in 'APO':
                            path,iid=trial_location(bank,memory,seed,ni,original_bi,r);p=load_trial(path,iid)
                            req(p is not None,f'Panel incomplete: {memory}, split={seed}, beta_index={original_bi}, network={ni}, seed={r}. Resume run(C).')
                            ledger.append(dict(path=path.relative_to(bank['root']).as_posix(),sha256=sha(path)))
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
                        print(f'Authenticated ensembles: {verified:,}/{EXPECTED_ENSEMBLES:,}',flush=True);guard(c);last=time.monotonic()
        req(verified==EXPECTED_ENSEMBLES and (exposure>0).all(),'Incomplete exposure panel.')
        expected=c['geometries']['S']['expected'].groupby('word').size().reindex(words).to_numpy()*len(CLASSIFIER_SEEDS)
        req(np.array_equal(exposure,np.repeat(expected[:,None],len(BETAS),axis=1)),'Exposure mismatch with frozen S.')
        base=pd.DataFrame(dict(word=np.repeat(words,len(BETAS)),source_word=np.repeat(words,len(BETAS)),
            true_class=np.repeat(full.true_class.to_numpy(),len(BETAS)),beta=np.tile(BETAS,W),beta_key=np.tile(np.round(BETAS,12),W),
            validation_similarity=np.repeat(full.validation_similarity.to_numpy(),len(BETAS)),n_items=exposure.ravel(),n_trials=100*exposure.ravel()))
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
            source_bank_id=bank['run_id'],source_bank_root=str(bank['root']),original_beta_indices=BETA_INDICES,classifier_seeds=CLASSIFIER_SEEDS,trial_ledger_sha=sha(c['root']/'verified_trial_ledger.parquet'),positions_are_independent=False))
    return pd.DataFrame(summary)

def empirical(c):
    import e6_p20_reduced_stats as st
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
    import e6_p20_reduced_stats as st
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
            mask=BETAS<=.10+1e-12 if restricted else np.ones(len(BETAS),bool)
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
    req(readj(c['root']/'counts_complete.json')['verified_ensembles']==EXPECTED_ENSEMBLES,'Wrong authenticated ensemble count.')
    report=dict(status='COMPLETE',run_id=c['run_id'],ensembles=EXPECTED_ENSEMBLES,arms=9,words=640,classes=8,
        splits=len(SPLITS),classifier_seeds=len(CLASSIFIER_SEEDS),beta_points=len(BETAS),CV_fits=1350,full_and_LOCO_refits=486,
        empirical_cases=243,occurrence_copies_are_independent=False,
        effects='Conditional on fixed K80 BERT source, task, multiplicities, scales and observer recipe.',
        new_classifier_fit_per_memory=True,source_inputs_read_only=True)
    writej(c['root']/'FINAL_CHECK.json',report);print(json.dumps(report,indent=2));return report

def plot_saved(root,show_figures=True,dpi=150,include_models=True):
    """Regenerate figures from saved numerical surfaces; no data loading or refitting."""
    root=Path(root).expanduser().resolve()
    protocol=readj(root/'protocol.json')
    req(protocol['version']==VERSION_REDUCED,'Not a reduced-panel result folder.')
    np.testing.assert_allclose(protocol['betas'],BETAS,rtol=0,atol=1e-14)
    cfg=Config(show_figures=show_figures,figure_dpi=dpi)
    return show(dict(root=root,cfg=cfg),include_models=include_models)

def export_review(c):
    """All numerical figure sources and fitted surfaces; no raw corpus or trial bank."""
    root=c['root'];req((root/'counts_complete.json').exists(),'Aggregate the complete selected panel first.')
    target=root/f'P20_33x51x3_Review_{c["run_id"][:12]}.zip';tmp=target.with_suffix('.partial.zip')
    chosen=[]
    for p in root.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(root);parts=rel.parts
        if parts[0] in ('tmp','resources','timing','progress') or p.suffix=='.zip' or p.name in ('STOP','RUN.lock'):continue
        # Include fitted empirical/matched5/grid30 CSVs, parameters and preprocessing.
        # CV checkpoints are unnecessary for changing figure titles or labels.
        if parts[0]=='analysis' and 'CV' in parts:continue
        chosen.append((p,rel.as_posix()))
    bank=c['bank']['root']
    # Preserve small calibration/structural evidence alongside reduced outcomes.
    evidence=[bank/n for n in ('protocol.json','split_replay.csv','split_files.json','source_stats.json',
        'inputs/apo/means_verified.csv','inputs/apo/role_overlap.csv','inputs/apo/representation_diagnostics.csv')]
    evidence.extend(p for p in (bank/'cal').rglob('*') if p.is_file())
    chosen.extend((p,'source_bank/'+p.relative_to(bank).as_posix()) for p in evidence if p.is_file())
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in chosen:z.write(p,name)
        z.writestr('REVIEW_MANIFEST.json',json.dumps(dict(run_id=c['run_id'],
            source_bank_id=c['bank']['run_id'],beta_points=33,classifier_seeds=[0,1,2],splits=51,
            numerical_plot_sources_included=True,files={name:sha(p) for p,name in chosen}),indent=2))
    os.replace(tmp,target);print('Review ZIP:',target);return target
