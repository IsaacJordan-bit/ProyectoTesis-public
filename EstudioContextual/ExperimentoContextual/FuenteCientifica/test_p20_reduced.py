"""Synthetic integration checks for the P20 reduced-panel bridge."""
from pathlib import Path
import json, tempfile, shutil, zipfile, time
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import e6_p20 as original
import e6_p20_reduced as reduced
import e6_p20_reduced_stats as stats
import plot_p20_reduced as plots
import test_p20 as fixtures

def execution_test(root):
    bank=fixtures.synthetic_context(root)
    a=original.representations(bank,0,'A');words=a['I'];truth=a['truth']
    full=pd.DataFrame(dict(word=words,source_word=words,true_class=truth,
        validation_similarity=np.linspace(.5,.9,len(words)),similarity_wc_z=np.zeros(len(words))))
    bank['geometries']['S'].update(full=full,common=full.assign(similarity_bin=0),expected=full.assign(split_seed=0),edges=np.linspace(-2,2,6))
    bank['contract']=dict(version=original.VERSION,splits=list(range(51)),seeds=list(range(10)),
        betas=original.BETAS,environment=original.environment())
    for name in ('full','common','expected'):
        original.frame(root/'inputs'/(name+'.parquet'),bank['geometries']['S'][name])
    original.frame(root/'inputs/cv_folds.parquet',pd.DataFrame(dict(word=[],repeat=[],fold=[])))
    original.writej(root/'inputs/geometry.json',dict(edges=list(np.linspace(-2,2,6))))
    panel=reduced.prepare(bank)
    # One real completed original ensemble is the bridge/reuse reference.
    scales=original.calibrate(bank,0);a=original.representations(bank,0,'A')
    blocks=[(l,min(l+16,a['n_query'])) for l in range(0,a['n_query'],16)]
    for lo,hi in blocks:
        original.score_task(bank,'A',0,lo,hi);original.recall_task(bank,'A',0,(0,),scales,lo,hi)
    original.merge_recalls(bank,'A',0,(0,),blocks)
    original.classification_task(bank,'A',0,(0,),0,0,0)
    path,iid=original.trial_location(bank,'A',0,0,0,0);prior_sha=original.sha(path)
    cal_sha=original.sha(root/'cal/s00/curves.npz')
    # An authenticated off-panel trial must survive and not enter reduced counts.
    off_path,off_id=original.trial_location(bank,'A',0,0,1,9)
    original.save_trial(off_path,off_id,original.load_trial(path,iid));off_sha=original.sha(off_path)
    with patch.multiple(reduced,SPLITS=(0,),BETA_INDICES=(0,97),BETAS=original.BETAS[[0,97]],
                        CLASSIFIER_SEEDS=(0,2),EXPECTED_ENSEMBLES=36):
        progress=reduced.run(panel)
        assert progress.selected_complete.sum()==36 and progress.selected_pending.sum()==0
        assert original.sha(path)==prior_sha and original.sha(off_path)==off_sha
        assert original.sha(root/'cal/s00/curves.npz')==cal_sha
        assert len(original.arrays(root/'cal/s00/curves.npz')['betas'])==98
        assert not list((root/'trials').rglob('complete.json'))
        selected=list((root/'trials').rglob('r*.npz'));hashes={p:original.sha(p) for p in selected}
        with patch.object(reduced,'score_task',side_effect=AssertionError('Completed trials recomputed')):
            reduced.run(panel)
        assert all(original.sha(p)==h for p,h in hashes.items())
        missing,identity=original.trial_location(bank,'O',0,2,97,2)
        data=missing.read_bytes();missing.unlink()
        try:reduced.aggregate(panel)
        except RuntimeError as e:assert 'Panel incomplete' in str(e)
        else:raise AssertionError('Incomplete panel was accepted')
        missing.write_bytes(data)
        table=reduced.aggregate(panel)
        assert len(table)==27
        m=original.arrays(root/'maps/s00.npz')
        for memory in 'APO':
            for q in 'APO':
                for ni in range(3):
                    d=pd.read_parquet(panel['root']/'counts'/f'{memory+q}_n{ni}.parquet')
                    assert set(d.beta)==set(original.BETAS[[0,97]])
                    assert d.n_items.eq(2).all() and d.n_trials.eq(200).all()
                    for bi in (0,97):
                        expected=np.zeros(len(words),dtype=int)
                        for seed in (0,2):
                            z=original.load_trial(*original.trial_location(bank,memory,0,ni,bi,seed))
                            e=z[q][m[q]]!=m['truth'][m['word_index']]
                            expected+=np.bincount(m['word_index'],weights=e,minlength=len(words)).astype(int)
                        got=d.loc[d.beta.eq(original.BETAS[bi])].set_index('word').loc[words]
                        np.testing.assert_array_equal(got.n_errors,expected)
        assert original.readj(panel['root']/'counts_complete.json')['verified_ensembles']==36
        assert len(pd.read_parquet(panel['root']/'verified_trial_ledger.parquet'))==36
        (root/'STOP').write_text('test')
        try:reduced.run(panel)
        except original.compute.TechnicalPause:pass
        else:raise AssertionError('Bank STOP ignored')
        (root/'STOP').unlink()
        reduced.aggregate(panel)
    return dict(actual_synthetic_ensembles=36,prior_checkpoint_reused=True,off_panel_preserved=True,
        original_98_beta_calibration_preserved=True,original_index_97_verified=True,
        independent_integer_count_replay=True,incomplete_panel_rejected=True,STOP_respected=True)

def statistical_test(root):
    c=fixtures.statistics_context(root)
    c['bank']=dict(run_id='SYNTHETIC_BANK',root=root)
    c['protocol']=dict(version=reduced.VERSION_REDUCED,betas=reduced.BETAS)
    original.writej(root/'protocol.json',c['protocol'])
    receipt=original.readj(root/'counts_complete.json')
    for name in receipt['files']:
        path=root/'counts'/name;d=pd.read_parquet(path)
        d=d.loc[d.beta_key.isin(np.round(reduced.BETAS,12))].copy()
        original.frame(path,d);receipt['files'][name]=original.sha(path)
    receipt['verified_ensembles']=reduced.EXPECTED_ENSEMBLES
    original.writej(root/'counts_complete.json',receipt)
    a=stats.load_analysis(c,'AA');b=stats.load_analysis(c,'OO')
    assert a['Y'].shape==(120*33,3)
    for f,g in zip(a['folds'],b['folds']):
        np.testing.assert_array_equal(f['train'],g['train']);np.testing.assert_array_equal(f['test'],g['test'])
    np.testing.assert_allclose(stats.surface(a,np.ones(len(a['data']))),1)
    for mi in range(2):
        assert stats.cv_task(a,mi,0,0)['stable']
        assert stats.refit_task(a,mi,0)['stable']
        assert stats.refit_task(a,mi,0,(0,))['stable']
    # Also exercise actual bounded parallel worker serialization.
    c['cfg'].max_workers=2
    for arm in ('AA','PP','OO'):
        ctx=stats.load_analysis(c,arm)
        jobs=[(ctx,mi,ni,()) for mi in range(2) for ni in range(3)]
        rows=reduced.run_tasks(ctx,'SYNTHETIC_REFIT',stats.refit_task,jobs,.1)
        assert len(rows)==6 and all(r['stable'] for r in rows)
    empirical=reduced.empirical(c);assert len(empirical)==243
    assert len(original.readcsv(root/'empirical_contrasts.csv'))==216
    reduced.show(c,include_models=False)
    assert len(list((root/'figures').glob('*.png')))==12
    # Full model rendering uses saved fitted surfaces, never fitting inside show().
    robust=[]
    for model in reduced.MODELS:
        for network in reduced.NETS:
            for arm in reduced.ARMS:
                robust.append(dict(candidate=original.core.candidate_name(model['family'],model['alpha']),network=network,
                    scope='grid30',arm=arm,T1_worst_signed_margin=.1,T2_worst_signed_margin=.2))
    original.frame(root/'robustness.csv',pd.DataFrame(robust))
    original.writej(root/'statistics_complete.json',dict(test_only=True))
    plots.plot_saved(root,show_figures=False,include_models=True)
    assert len(list((root/'figures').glob('*.png')))==21
    archive=reduced.export_review(c)
    with zipfile.ZipFile(archive) as z:
        assert 'analysis/AA/fits/m0/n0/FULL/grid30.csv' in z.namelist()
        assert 'analysis/AA/fits/m0/n0/FULL/fit.npz' in z.namelist()
        assert 'empirical/AA/n0_FULL.csv' in z.namelist()
    return dict(beta_points=33,shared_25_word_folds=True,models_and_LOCO_executed=True,
        bounded_parallel_refits=True,empirical_cases=243,contrasts=216,figures_PNG_PDF=21,
        numerical_figure_sources_exported=True)

def main():
    started=time.perf_counter()
    report=dict(test_data='SYNTHETIC ONLY; no execution on the user real data.',environment=original.environment())
    reduced.verify_addon()
    with tempfile.TemporaryDirectory(prefix='p20r_test_') as tmp,threadpool_limits(limits=1):
        root=Path(tmp)
        report['execution']=execution_test(root/'bank');print('Execution/reuse/counts: PASS',flush=True)
        report['statistics']=statistical_test(root/'stats');print('Statistics/figures/export: PASS',flush=True)
        shutil.copy2(root/'stats/figures/n0_nine_arms_full.png',Path(__file__).parent/'synthetic_preview.png')
    report.update(status='PASS',seconds=time.perf_counter()-started)
    original.writej(Path(__file__).parent/'VALIDACION_33x51x3.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
