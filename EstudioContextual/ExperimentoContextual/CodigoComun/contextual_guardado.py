"""Read-only replay of the frozen P20 controlled experiment. No fitting imports."""
from dataclasses import dataclass
from pathlib import Path
import hashlib, io, json, zipfile
import numpy as np
import pandas as pd

ARMS = ('AA','AP','AO','PA','PP','PO','OA','OP','OO')
NETS = ('mhncs_softmax','hfy_entmax_a15','hfy_normmax_g2')
TITLES = ('MHNCS Softmax', 'HFY Entmax (α = 1.5)', 'HFY Normmax (γ = 2)')
KINDS = ('empirical','primary','sensitivity')
CASES = ('FULL',) + tuple(f'L{i:02d}' for i in range(8))
RUN_ID = 'bf02d71452872d063717c4b8770a4708b8f1ca85e6eed7616c9fc7a712f09dd8'

class SavedStudy:
    def __init__(self, archive):
        self.archive = Path(archive)
        with zipfile.ZipFile(self.archive) as z:
            self.data = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
        self.protocol = self.json('protocol.json')
        self.geometry = self.json('inputs/geometry.json')
        self.labels = list(self.geometry['neighbors'])
        self.tables = {n: self.csv(n+'.csv') for n in (
            'counts_summary','empirical_trends','statistical_FULL_LOCO',
            'robustness','OOF_metrics','empirical_contrasts','statistical_contrasts')}

    def json(self, name): return json.loads(self.data[name])
    def csv(self, name, **kw): return pd.read_csv(io.BytesIO(self.data[name]), **kw)
    def parquet(self, name): return pd.read_parquet(io.BytesIO(self.data[name]))

    def surface(self, arm, network, case='FULL', kind='primary', representation='grid30'):
        if arm not in ARMS or network not in range(3) or case not in CASES or kind not in KINDS:
            raise ValueError('Unknown condition, network, omission or surface kind')
        if representation not in ('grid30','matched5'): raise ValueError('Unknown representation')
        path = (f'empirical/{arm}/n{network}_{case}.csv' if kind=='empirical' else
                f'analysis/{arm}/fits/m{0 if kind=="primary" else 1}/n{network}/{case}/{representation}.csv')
        s = self.csv(path, index_col=0)
        s.index = s.index.astype(float); s.columns = s.columns.astype(float)
        return s

    def verify(self):
        manifest = self.json('REVIEW_MANIFEST.json')
        for name, expected in manifest['files'].items():
            if name not in self.data or hashlib.sha256(self.data[name]).hexdigest()!=expected:
                raise ValueError('Missing or altered artifact: '+name)
        if self.protocol['version']!='P20_33x51x3_V1' or manifest['run_id']!=RUN_ID:
            raise ValueError('Unexpected scientific protocol')
        f=self.tables['statistical_FULL_LOCO']
        expected={(a,n,m,c) for a in ARMS for n in NETS for m in ('PRIMARY','SENSITIVITY') for c in CASES}
        actual=set(f[['arm','network','model','case']].itertuples(index=False,name=None))
        if len(f)!=486 or actual!=expected or not f.stable.eq(True).all():
            raise ValueError('Incomplete or unstable FULL/LOCO results')
        checked=0
        for arm in ARMS:
            for ni in range(3):
                for case in CASES:
                    for kind in KINDS:
                        for representation in (('grid30',) if kind=='empirical' else ('grid30','matched5')):
                            s=self.surface(arm,ni,case,kind,representation)
                            y=s.to_numpy(); rows=30 if kind!='empirical' and representation=='grid30' else 5
                            if y.shape!=(rows,33) or not np.isfinite(y).all() or y.min() < -1e-12 or y.max()>1+1e-12:
                                raise ValueError('Invalid saved probability surface')
                            np.testing.assert_allclose(s.columns,self.protocol['betas'],rtol=0,atol=1e-14)
                            checked+=1
        # Independently reproduce empirical FULL and LOCO using fixed COMMON,
        # equal class/bin weights, and saved word error rates (no classifiers).
        common=self.parquet('inputs/common.parquet')[['word','true_class','similarity_bin']]
        max_delta=0.
        for arm in ARMS:
            for ni in range(3):
                counts=self.parquet(f'counts/{arm}_n{ni}.parquet')
                d=counts.merge(common,on=['word','true_class'],validate='many_to_one')
                for case in CASES:
                    subset=d if case=='FULL' else d[d.true_class!=int(case[1:])]
                    reconstructed=(subset.groupby(['true_class','similarity_bin','beta']).rate.mean()
                                   .groupby(['similarity_bin','beta']).mean().unstack('beta'))
                    saved=self.surface(arm,ni,case,'empirical')
                    delta=float(np.max(np.abs(reconstructed.to_numpy()-saved.to_numpy())))
                    max_delta=max(max_delta,delta)
                    np.testing.assert_allclose(reconstructed.to_numpy(),saved.to_numpy(),rtol=0,atol=1e-12)
        return dict(status='VERIFIED',run_id=RUN_ID,authenticated_files=len(manifest['files']),
                    refits=486,numerical_surfaces=checked,empirical_cases_reproduced=243,
                    empirical_max_abs_delta=max_delta,experiment_executed=False,models_refitted=False)

@dataclass
class FigureConfig:
    output_root: Path = Path('SalidasRevision/Figuras')
    beta_range: tuple | None = (0.01, 0.15)
    heatmap_width: float = 6.
    color_limits: tuple | None = (0., 1.)
    dpi: int = 200
    formats: tuple = ('png','pdf')
    display_view: str = 'manuscript'


def draw_heatmap(study, arm, network, case, kind, config, view='manuscript'):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    s=study.surface(arm,network,case,kind); x=s.columns.to_numpy(); y=s.index.to_numpy()
    limits=(float(x.min()),float(x.max()))
    if view=='manuscript' and config.beta_range is not None:
        limits=tuple(map(float,config.beta_range))
        if not x.min()<=limits[0]<limits[1]<=x.max():
            raise ValueError('beta_range must lie within the saved beta domain')
    elif view not in ('manuscript','appendix'): raise ValueError('Unknown view')
    if config.heatmap_width<4: raise ValueError('Use a width of at least 4 inches for readable labels')
    cmap=LinearSegmentedColormap.from_list('thesis',['#6A0DAD','#B755C1','#D890D1','#F2C2A3','#F9E2B7','#FFFF00'])
    # Use the full-domain limits in BOTH views, even with automatic color scaling.
    vmin,vmax=config.color_limits if config.color_limits is not None else (float(s.min().min()),float(s.max().max()))
    fig,ax=plt.subplots(figsize=(config.heatmap_width+1.2,config.heatmap_width+1.4),layout='constrained')
    im=ax.pcolormesh(x,y,s.to_numpy(),shading='nearest',cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True)
    ax.set_xlim(*limits)
    if kind=='empirical': ax.set_ylim(study.geometry['edges'][0],study.geometry['edges'][-1])
    else: ax.set_ylim(y.min(),y.max())
    ax.set_box_aspect(1)  # Physical data panel is square, independently of beta crop.
    ax.set_xlabel('Nominal β',fontsize=11)
    ax.set_ylabel('Within-class semantic\nsimilarity (z)',fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    label={'empirical':'Empirical surface','primary':'Primary fitted surface','sensitivity':'Alternative fitted surface'}[kind]
    case_label='All eight classes' if case=='FULL' else 'Class omitted: '+study.labels[int(case[1:])]
    span=('Zoomed' if view=='manuscript' and config.beta_range else 'Full')+f' β range: {limits[0]:.3f}–{limits[1]:.3f}'
    ax.set_title(f'BERT · K = 80 · {TITLES[network]}\nMemory {arm[0]} / Query {arm[1]} · {label}\n{case_label} · {span}',fontsize=11,pad=12)
    fig.colorbar(im,ax=ax,shrink=.85,pad=.03,label='FAR / error probability')
    return fig,ax


def plot_heatmaps(study, config, kinds=KINDS, cases=CASES, arms=ARMS, networks=(0,1,2), show=False):
    import matplotlib.pyplot as plt
    from IPython.display import display
    config.output_root=Path(config.output_root)
    rows=[]
    for kind in kinds:
        for arm in arms:
            for ni in networks:
                for case in cases:
                    for view in ('manuscript','appendix'):
                        fig,ax=draw_heatmap(study,arm,ni,case,kind,config,view)
                        try:
                            folder=config.output_root/view/kind/arm; folder.mkdir(parents=True,exist_ok=True)
                            for fmt in config.formats:
                                path=folder/f'n{ni}_{case}.{fmt}'
                                fig.savefig(path,dpi=config.dpi,bbox_inches='tight')
                                rows.append(dict(kind=kind,arm=arm,network=NETS[ni],case=case,view=view,path=str(path)))
                            if show and view==config.display_view: display(fig)
                        finally: plt.close(fig)
    result=pd.DataFrame(rows)
    return result
