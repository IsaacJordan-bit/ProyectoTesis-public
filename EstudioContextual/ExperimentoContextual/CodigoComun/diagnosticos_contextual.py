"""Primary-model diagnostics, using authenticated P20 saved results only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextual_guardado import ARMS, NETS, TITLES, CASES

SERIES=(('empirical','Empirical','#202020','-'),
        ('matched5','Model · MATCHED5','#9A279B','--'),
        ('grid30','Model · GRID30','#00878B',':'))

def curves(s):
    x=s.index.to_numpy(float);x=x-x.mean();a=s.to_numpy(float)
    return x@a/(x@x),a.mean(axis=0)

def plot_diagnostics(study, config, arms=ARMS, networks=(0,1,2), show=True):
    """Full beta range; crop settings affect heatmaps only. No training or refitting."""
    from IPython.display import display
    out=Path(config.output_root)/'diagnostics'; records=[]
    def save(fig,folder,name):
        try:
            for fmt in config.formats:
                p=folder/(name+'.'+fmt);fig.savefig(p,dpi=config.dpi,bbox_inches='tight')
                records.append(dict(figure=name,path=str(p)))
            if show:display(fig)
        finally:plt.close(fig)
    for arm in arms:
        folder=out/arm;folder.mkdir(parents=True,exist_ok=True)
        for ni in networks:
            title=f'BERT · K = 80 · {TITLES[ni]} · Memory {arm[0]} / Query {arm[1]}'
            E=study.surface(arm,ni,kind='empirical')
            M=study.surface(arm,ni,kind='primary',representation='matched5')
            G=study.surface(arm,ni,kind='primary',representation='grid30')
            pd.testing.assert_index_equal(E.index,M.index)
            beta=E.columns.to_numpy(float); cs=[curves(s) for s in (E,M,G)]
            # Cross-check the diagnostic against the frozen scientific report.
            report=study.tables['statistical_FULL_LOCO']
            d=report[(report.arm==arm)&(report.network==NETS[ni])&(report.model=='PRIMARY')]
            if len(d)!=9 or set(d.case)!=set(CASES):raise ValueError('Incomplete primary FULL/LOCO diagnostics')
            full=d[d.case=='FULL'].iloc[0]
            rmse=np.sqrt(np.mean((M.to_numpy()-E.to_numpy())**2))
            np.testing.assert_allclose(rmse,full.RMSE_matched5,rtol=0,atol=1e-12)
            fig,ax=plt.subplots(1,3,figsize=(14.5,4.8),layout='constrained')
            rows=[]
            for (key,label,color,style),(slope,mean) in zip(SERIES,cs):
                ax[0].plot(beta,slope,label=label,color=color,ls=style)
                ax[2].plot(beta,mean,label=label,color=color,ls=style)
                rows.extend(dict(surface=key,beta=b,similarity_slope=s,mean_far=m) for b,s,m in zip(beta,slope,mean))
            for (_,label,color,_),(slope,_) in zip(SERIES[1:],cs[1:]):
                ax[1].scatter(cs[0][0],slope,s=14,label=label,color=color)
            lo=min(c[0].min() for c in cs);hi=max(c[0].max() for c in cs)
            ax[1].plot([lo,hi],[lo,hi],ls='--',color='.5')
            ax[0].axhline(0,color='.5',lw=.8)
            ax[0].set(title='A. Semantic slope',xlabel='Nominal β',ylabel='Slope of FAR vs. similarity')
            ax[1].set(title='B. Slope agreement',xlabel='Empirical slope',ylabel='Model slope')
            ax[2].set(title='C. Mean FAR',xlabel='Nominal β',ylabel='Mean FAR / probability')
            ax[0].legend(fontsize=8)
            for a in ax:a.grid(alpha=.15)
            fig.suptitle(title+'\nPrimary model · Descriptive fit · All classes · Full β range',fontsize=12)
            pd.DataFrame(rows).to_csv(folder/f'n{ni}_fit_curves.csv',index=False)
            save(fig,folder,f'n{ni}_fit_quality')
            # Exactly the same FULL + 8 omissions shown in the static study.
            d=d.set_index('case').loc[list(CASES)]
            fig,ax=plt.subplots(1,3,figsize=(14.5,5),layout='constrained')
            for a,trend in zip(ax,('T1','T2','T3')):
                for key,label,color,style in SERIES:
                    a.plot(range(9),d[key+'_'+trend],label=label,color=color,ls=style,marker='o',ms=3)
                a.axhline(0,color='.5',lw=.8)
                a.set_xticks(range(9),['All classes',*study.labels],rotation=55,ha='right')
                a.set(title=trend,xlabel='Omitted class (except first point)',ylabel='Global slope')
                a.grid(alpha=.15)
            ax[0].legend(fontsize=8)
            fig.suptitle(title+'\nPrimary model · Leave-one-class-out sensitivity · Full β range',fontsize=12)
            d.to_csv(folder/f'n{ni}_LOCO_trends.csv')
            save(fig,folder,f'n{ni}_LOCO_sensitivity')
            # Saved OOF surface is formed AFTER averaging predictions over repeats.
            O=study.csv(f'analysis/{arm}/OOF/m0_n{ni}.csv',index_col=0)
            O.index=O.index.astype(float);O.columns=O.columns.astype(float)
            np.testing.assert_allclose(O.index,E.index,rtol=0,atol=1e-14)
            np.testing.assert_allclose(O.columns,E.columns,rtol=0,atol=1e-14)
            values=O.to_numpy()
            if not np.isfinite(values).all():raise ValueError('Nonfinite OOF predictions')
            curve=np.sqrt(np.mean((values-E.to_numpy())**2,axis=0))
            # Reconcile OOF means and slopes with the original saved curves.
            old=study.csv(f'analysis/{arm}/OOF_curves.csv')
            old=old[(old.model=='PRIMARY')&(old.network==NETS[ni])].sort_values('beta')
            slope,mean=curves(O)
            np.testing.assert_allclose(mean,old.OOF_mean,rtol=0,atol=1e-12)
            np.testing.assert_allclose(slope,old.OOF_slope,rtol=0,atol=1e-12)
            fig,a=plt.subplots(figsize=(8.5,4),layout='constrained')
            a.plot(beta,curve,color='#6A0DAD',lw=2)
            a.set(title=title+'\nOut-of-fold fidelity · Predictions averaged over repeats',
                  xlabel='Nominal β',ylabel='RMSE (MATCHED5)')
            a.grid(alpha=.15)
            pd.DataFrame(dict(beta=beta,rmse_matched5=curve)).to_csv(folder/f'n{ni}_OOF_curve.csv',index=False)
            save(fig,folder,f'n{ni}_OOF')
    return pd.DataFrame(records)
