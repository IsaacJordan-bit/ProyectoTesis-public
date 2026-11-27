"""English figures from verified results. Plotting never fits or trains a model.

Heatmap zoom is an axis limit on the original matrix. The full beta grid is
retained for color limits, curves and T1/T2/T3, including all LOCO results.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, FuncFormatter

import exp2_pipeline_integrated as exp

NETWORK_NAMES = {
    'mhncs_softmax': 'MHNCS Softmax',
    'hfy_entmax_a15': 'HFY Entmax (α = 1.5)',
    'hfy_normmax_g2': 'HFY Normmax (γ = 2)',
}
DATASET_NAMES = {'glove': 'GloVe (50D)', 'w2v': 'Word2Vec (300D)',
                 'prototypes': 'BERT prototypes'}
SERIES = [('empirical', 'Empirical (5 bins)', '#6A0DAD', '-'),
          ('matched5', 'Model (MATCHED5)', '#B27600', '--'),
          ('grid30', 'Model (GRID30)', '#187E87', ':')]


@dataclass
class FigureConfig:
    output_root: Path
    beta_range: tuple[float, float] | None = (0.01, 0.15)
    heatmap_aspect: float = 3.5  # width / height of the data panel
    heatmap_width: float = 10.5  # inches
    color_limits: tuple[float, float] | None = None
    dpi: int = 200
    formats: tuple[str, ...] = ('png', 'pdf')
    display_view: str = 'manuscript'  # manuscript, appendix, both, none
    show_diagnostics: bool = True

    def __post_init__(self):
        self.output_root = Path(self.output_root)
        if self.display_view not in ('manuscript', 'appendix', 'both', 'none'):
            raise ValueError('display_view must be manuscript, appendix, both or none.')
        if not np.isfinite(self.heatmap_aspect) or not 1 <= self.heatmap_aspect <= 15:
            raise ValueError('heatmap_aspect must be between 1 and 15 (width / height).')
        if not np.isfinite(self.heatmap_width) or self.heatmap_width < 6:
            raise ValueError('heatmap_width must be at least 6 inches.')
        if self.dpi < 72 or 'png' not in self.formats or not set(self.formats) <= {'png', 'pdf', 'svg'}:
            raise ValueError('Use dpi >= 72; formats must include png, optionally pdf and svg.')
        if self.color_limits is not None:
            lo, hi = self.color_limits
            if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi <= 1:
                raise ValueError('color_limits must satisfy 0 <= lower < upper <= 1.')


def _surface(surface):
    s = surface.copy(deep=True)
    s.index = s.index.astype(float)
    s.columns = s.columns.astype(float)
    s = s.sort_index().sort_index(axis=1)
    if s.index.has_duplicates or s.columns.has_duplicates or min(s.shape) < 2:
        raise ValueError('A surface must have distinct coordinates and at least two rows and columns.')
    if not np.isfinite(s.index).all() or not np.isfinite(s.columns).all():
        raise ValueError('Non-finite surface coordinates.')
    if np.isinf(s.to_numpy()).any() or not np.isfinite(s.to_numpy()).any():
        raise ValueError('Invalid surface values.')
    return s


def _edges(centers):
    x = np.asarray(centers, float)
    mid = (x[:-1] + x[1:]) / 2
    return np.r_[x[0] - (mid[0] - x[0]), mid, x[-1] + (x[-1] - mid[-1])]


def _limits(betas, requested):
    full = (float(np.min(betas)), float(np.max(betas)))
    if requested is None:
        return full
    try:
        lo, hi = map(float, requested)
    except (TypeError, ValueError) as err:
        raise ValueError('beta_range must be None or (lower, upper).') from err
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError('beta_range must contain two finite, increasing values.')
    if lo < full[0] - 1e-12 or hi > full[1] + 1e-12:
        raise ValueError(f'beta_range must lie inside the observed beta domain {full}.')
    return max(lo, full[0]), min(hi, full[1])


def _range_name(bounds):
    return 'manuscript_beta_' + '_'.join(f'{v:.12g}'.replace('.', 'p') for v in bounds)


def _hash_surface(s):
    h = hashlib.sha256()
    for a in (s.index.to_numpy(float), s.columns.to_numpy(float), s.to_numpy(float)):
        h.update(np.asarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


def _palette():
    cmap = LinearSegmentedColormap.from_list('thesis_purple_yellow',
        ['#6A0DAD', '#B755C1', '#D890D1', '#F2C2A3', '#F9E2B7', '#FFFF00'])
    cmap.set_bad('#DDDDDD')
    return cmap


def _save(fig, folder, stem, config, show=False):
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    try:
        for fmt in config.formats:
            path = folder / f'{stem}.{fmt}'
            fig.savefig(path, dpi=config.dpi, bbox_inches='tight', facecolor='white')
            records[fmt] = str(path)
            records[fmt + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        records['figure_text'] = [x.get_text() for x in fig.findobj(Text) if x.get_text()]
    finally:
        plt.close(fig)
    if show:
        from IPython.display import Image, display
        display(Image(filename=records['png'], width=1050))
    return records


def _draw_heatmap(s, title, subtitle, color_label, color_limits, xlim, config,
                  *, y_limits=None, y_edges=None):
    """No interpolation or smoothing; xlim clips the full, unchanged matrix."""
    width = config.heatmap_width
    fig, ax = plt.subplots(figsize=(width, width/config.heatmap_aspect + 1.7), layout='constrained')
    im = ax.pcolormesh(_edges(s.columns), _edges(s.index) if y_edges is None else y_edges,
        np.ma.masked_invalid(s.to_numpy()), shading='flat', cmap=_palette(),
        vmin=color_limits[0], vmax=color_limits[1], rasterized=True)
    ax.set_xlim(*xlim)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_box_aspect(1/config.heatmap_aspect)
    ax.set_xlabel('Nominal β', fontsize=11)
    ax.set_ylabel('Within-class semantic\nsimilarity (z)', fontsize=11)
    ax.set_title(title + '\n' + subtitle, fontsize=12, pad=12)
    # Endpoints remain explicit, even when the interval is not aligned with the beta grid.
    ax.set_xticks(np.linspace(*xlim, 5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.3f}'.rstrip('0').rstrip('.')))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    # Attach the bar to the data panel, not to the taller layout slot.
    cb = fig.colorbar(im, cax=ax.inset_axes([1.025, 0, .025, 1]))
    cb.set_label(color_label, fontsize=10)
    return fig, ax


def heatmap_views(surface, config, *, folder, stem, title, subtitle,
                  color_limits, color_label='FAR / probability', y_limits=None, y_edges=None):
    """Save full and zoomed versions with identical values and color limits.

    Used by the main study and the optional support-geometry contrast.
    """
    s = _surface(surface)
    full = _limits(s.columns, None)
    zoom = _limits(s.columns, config.beta_range)
    views = [('appendix', full, 'appendix_full')]
    if config.beta_range is not None:
        views.append(('manuscript', zoom, _range_name(zoom)))
    records = []
    for view, bounds, child in views:
        detail = f'{subtitle}\n' + ('Full β range' if view == 'appendix' else 'Zoomed β range')
        detail += f': {bounds[0]:.3f}–{bounds[1]:.3f}'
        fig, ax = _draw_heatmap(s, title, detail, color_label, color_limits, bounds, config,
                                y_limits=y_limits, y_edges=y_edges)
        show = config.display_view in ('both', view) or (
            config.beta_range is None and config.display_view == 'manuscript')
        rec = _save(fig, Path(folder)/child, stem, config, show=show)
        rec.update(kind='heatmap', view=view, beta_min=bounds[0], beta_max=bounds[1],
                   full_beta_min=full[0], full_beta_max=full[1], n_betas=s.shape[1],
                   n_similarity_points=s.shape[0], vmin=color_limits[0], vmax=color_limits[1],
                   surface_sha256=_hash_surface(s), interpolation=False, smoothing=False,
                   analysis_beta_range=list(full))
        records.append(rec)
    return records


def _surfaces(result):
    ctx = result.context
    result_surfaces = {}
    for network in exp.CORE3:
        for omitted in (None, *range(8)):
            tag = 'FULL' if omitted is None else f'LOCO_{omitted}_{ctx["labels"][omitted]}'
            folder = ctx['out88'] / 'fits' / network / tag
            emp = ctx['empirical'][network] if omitted is None else exp.read_surface88(folder/'empirical_5bins.csv')
            matched = exp.read_surface88(folder/'fitted_5bins.csv')
            grid = exp.verified_model_surface89(ctx, network, omitted)
            result_surfaces[network, omitted] = tuple(_surface(s) for s in (emp, matched, grid))
    return result_surfaces


def _case_label(ctx, omitted):
    return 'All eight classes' if omitted is None else 'Leave-one-class-out: ' + ctx['labels'][omitted]


def _title(result, network):
    return f'{DATASET_NAMES.get(result.dataset, result.dataset)} · K = {result.context["K"]} · {NETWORK_NAMES[network]}'


def _curves(s):
    z = s.index.to_numpy(float); z = z - z.mean()
    a = s.to_numpy(float)
    return z @ a / (z @ z), a.mean(axis=0)


def _quality(result, network, surfaces, folder, config):
    curves = [_curves(s) for s in surfaces]
    betas = surfaces[0].columns.to_numpy(float)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), layout='constrained')
    for (_, label, color, style), (slope, mean) in zip(SERIES, curves):
        axes[0].plot(betas, slope, label=label, color=color, ls=style, lw=1.8)
        axes[2].plot(betas, mean, label=label, color=color, ls=style, lw=1.8)
    for (_, label, color, _), (slope, _) in zip(SERIES[1:], curves[1:]):
        axes[1].scatter(curves[0][0], slope, s=14, label=label, color=color, alpha=.8)
    lo = min(v[0].min() for v in curves); hi = max(v[0].max() for v in curves)
    axes[1].plot([lo, hi], [lo, hi], color='.5', ls='--', lw=1)
    axes[0].axhline(0, color='.5', lw=.8)
    axes[0].set(title='A. Semantic slope', xlabel='Nominal β', ylabel='Slope of FAR vs. similarity')
    axes[1].set(title='B. Slope agreement', xlabel='Empirical slope', ylabel='Model slope')
    axes[2].set(title='C. Mean FAR', xlabel='Nominal β', ylabel='Mean FAR / probability')
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle(_title(result, network) + '\nDescriptive fit · All classes · Full β range', fontsize=13)
    rows = []
    for (kind, _, _, _), (slope, mean) in zip(SERIES, curves):
        rows.extend(dict(network=network, surface=kind, beta=b, similarity_slope=s, mean_far=m)
                    for b, s, m in zip(betas, slope, mean))
    pd.DataFrame(rows).to_csv(folder/f'{network}_fit_curves.csv', index=False)
    return dict(kind='fit_quality', network=network,
        **_save(fig, folder, network+'_fit_quality', config, show=config.show_diagnostics))


def _sensitivity(result, network, surfaces, folder, config):
    ctx = result.context; rows = []
    for omitted in (None, *range(8)):
        row = dict(network=network, omitted_class=omitted)
        for (prefix, _, _, _), surface in zip(SERIES, surfaces[network, omitted]):
            row.update({prefix+'_'+k: v for k,v in exp.trends85(surface).items() if k in ('T1','T2','T3')})
        rows.append(row)
    table = pd.DataFrame(rows)
    # Reconcile the plotted values against the verified scientific report.
    original = result.tables.get('sensitivity', result.tables['FULL_LOCO'])
    original = original.loc[original.network.eq(network)].copy()
    for row in rows:
        mask = original.omitted_class.isna() if row['omitted_class'] is None else original.omitted_class.eq(row['omitted_class'])
        old = original.loc[mask]
        if len(old) != 1:
            raise ValueError('Missing or duplicate FULL/LOCO record.')
        for name, value in row.items():
            if name in ('network','omitted_class'): continue
            if not np.isclose(value, old.iloc[0][name], rtol=0, atol=1e-12):
                raise ValueError('A plotted trend differs from the stored scientific result: ' + name)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), layout='constrained')
    labels = ['All classes', *ctx['labels']]
    for ax, trend in zip(axes, ('T1','T2','T3')):
        for prefix, label, color, style in SERIES:
            ax.plot(range(9), table[prefix+'_'+trend], label=label, color=color,
                    ls=style, marker='o', ms=3)
        ax.axhline(0, color='.4', lw=.8)
        ax.set_xticks(range(9), labels, rotation=55, ha='right')
        ax.set_title(trend + (' (expected > 0)' if trend=='T1' else ' (expected < 0)'))
        ax.set_ylabel('Global slope')
        ax.set_xlabel('Omitted class (except first point)')
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle(_title(result, network)+'\nLeave-one-class-out sensitivity · Full β range', fontsize=13)
    table.to_csv(folder/f'{network}_LOCO_trends.csv', index=False)
    return dict(kind='loco_sensitivity', network=network,
        **_save(fig, folder, network+'_LOCO_sensitivity', config, show=config.show_diagnostics))


def _oof(result, network, folder, config):
    ctx = result.context
    if 'oof_predictions90' in ctx:
        surface = exp.surface85(ctx, ctx['data'].assign(prediction=ctx['oof_predictions90'][network]), value='prediction')
    else:
        surface = exp.read_surface88(ctx['root']/'OOF'/f'{network}_mean_matched5.csv')
    s, empirical = _surface(surface), _surface(ctx['empirical'][network])
    pd.testing.assert_index_equal(s.index, empirical.index, check_names=False)
    pd.testing.assert_index_equal(s.columns, empirical.columns, check_names=False)
    curve = np.sqrt(np.square(s.to_numpy()-empirical.to_numpy()).mean(axis=0))
    fig, ax = plt.subplots(figsize=(8.5, 4), layout='constrained')
    ax.plot(s.columns, curve, color='#6A0DAD', lw=2)
    ax.set(title=_title(result, network)+'\nOut-of-fold fidelity · Predictions averaged over repeats',
           xlabel='Nominal β', ylabel='RMSE (MATCHED5)')
    ax.grid(alpha=.15)
    pd.DataFrame(dict(beta=s.columns, rmse_matched5=curve)).to_csv(folder/f'{network}_OOF_curve.csv', index=False)
    return dict(kind='oof', network=network,
        **_save(fig, folder, network+'_OOF', config, show=config.show_diagnostics))


def plot_results(result, config, *, heatmaps=True, fit_quality=True, oof=True, loco=True):
    """Generate English figures for GloVe K40, W2V K40 or the W2V K50 appendix."""
    ctx = result.context
    _limits(ctx['betas'], config.beta_range)  # Fail before writing any output for an invalid range.
    folder = config.output_root / f'{result.dataset}_k{ctx["K"]}'
    folder.mkdir(parents=True, exist_ok=True)
    surfaces = _surfaces(result)
    # Always use the full matrices, including all networks and LOCO cases.
    maximum = max(float(s.to_numpy().max()) for triplet in surfaces.values() for s in triplet)
    color_limits = config.color_limits or (0.0, maximum if maximum > 0 else 1.0)
    records = []
    if heatmaps:
        for network in exp.CORE3:
            for omitted in ((None, *range(8)) if loco else (None,)):
                emp, _, model = surfaces[network, omitted]
                choices = [('empirical', emp), ('model', model)]
                for kind, s in choices:
                    tag = 'full' if omitted is None else 'loco_'+ctx['labels'][omitted]
                    subtitle = ('Empirical surface' if kind=='empirical' else 'Model surface')
                    subtitle += ' · ' + _case_label(ctx, omitted)
                    figures = heatmap_views(s, config, folder=folder, stem=f'{network}_{tag}_{kind}',
                        title=_title(result, network), subtitle=subtitle, color_limits=color_limits,
                        color_label='FAR' if kind=='empirical' else 'Predicted FAR',
                        y_limits=(ctx['edges'][0],ctx['edges'][-1]))
                    for item in figures:item.update(network=network, case=tag, surface=kind)
                    records.extend(figures)
    diagnostic_folder = folder/'diagnostics'; diagnostic_folder.mkdir(exist_ok=True)
    for network in exp.CORE3:
        if fit_quality:
            records.append(_quality(result, network, surfaces[network,None], diagnostic_folder, config))
            if loco:records.append(_sensitivity(result, network, surfaces, diagnostic_folder, config))
        if oof:records.append(_oof(result, network, diagnostic_folder, config))
    name = 'heatmaps' if heatmaps and not fit_quality and not oof else 'diagnostics' if not heatmaps else 'all'
    suffix = _range_name(_limits(ctx['betas'],config.beta_range)) if config.beta_range is not None else 'full_only'
    report = dict(dataset=result.dataset, K=ctx['K'], configuration=asdict(config),
                  scientific_results_modified=False, new_model_fits=0, figures=records)
    path = folder/f'figure_index_{name}_{suffix}.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print(f'{DATASET_NAMES.get(result.dataset,result.dataset)} K{ctx["K"]}: {len(records)} figures saved to {folder}')
    return pd.DataFrame(records)
