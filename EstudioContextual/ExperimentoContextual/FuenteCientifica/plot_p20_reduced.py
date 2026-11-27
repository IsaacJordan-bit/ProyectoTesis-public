"""Editable presentation layer. No training or statistical fitting.
Change titles, axis labels, colors or layout here without altering checkpoints.
"""
from e6_p20_reduced import *

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

def plot_saved(root,show_figures=True,dpi=150,include_models=True):
    """Regenerate figures from saved numerical surfaces; no data loading or refitting."""
    root=Path(root).expanduser().resolve()
    protocol=readj(root/'protocol.json')
    req(protocol['version']==VERSION_REDUCED,'Not a reduced-panel result folder.')
    np.testing.assert_allclose(protocol['betas'],BETAS,rtol=0,atol=1e-14)
    cfg=Config(show_figures=show_figures,figure_dpi=dpi)
    return show(dict(root=root,cfg=cfg),include_models=include_models)

