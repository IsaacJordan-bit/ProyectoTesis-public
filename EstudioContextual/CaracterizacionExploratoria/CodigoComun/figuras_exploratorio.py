"""Presentation extracted from the frozen P22 module. No scientific calculation.
Edits: optional DPI/formats, panel labels, scoped style, no forced backend.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def require(ok,message):
    if not ok: raise ValueError(message)

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def _make_figures(run, dpi=220, formats=("png", "pdf", "svg")):
    import matplotlib
    import matplotlib.pyplot as plt
    root = Path(run["root"])
    require((root/"SUMMARY_COMPLETE.json").is_file(),"Run summarize() first.")
    g = pd.read_csv(root/"word_geometry.csv")
    summary = pd.read_csv(root/"tables/heldout_summary.csv")
    out = root/"figures";out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"svg.fonttype":"none"})
    def save(fig,name):
        fig.tight_layout()
        for ext in formats:
            fig.savefig(out/(name+"."+ext),dpi=dpi,bbox_inches="tight")
        plt.close(fig)
    categories = sorted(g.category.unique())
    colors = ["#187F91","#AB2DAC","#53677F"]
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].boxplot([g.loc[g.category.eq(c),"directional_variance"].dropna() for c in categories],tick_labels=categories,showfliers=False)
    axs[0].set(ylabel="Mean squared deviation of unit directions",title="(A) Within-word directional variation")
    axs[0].tick_params(axis="x",rotation=45)
    for metric,label,color in [("raw_self_cosine","Same word",colors[0]),("other_word_cosine","Other words",colors[2])]:
        values=np.sort(g[metric].dropna())
        axs[1].plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color)
    axs[1].set(xlabel="Mean cosine similarity",ylabel="Fraction of words",title="(B) Similarity with an anisotropy reference")
    axs[1].legend(frameon=False)
    save(fig,"01_contextual_geometry")
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].scatter(g.n_prototypes,g.frozen_P_retained_fraction,s=12,alpha=.4,color=colors[0])
    ks=np.sort(g.n_prototypes.unique())
    axs[0].plot(ks,(ks-1)/99,"--",color=colors[2],label="Random membership expectation")
    axs[0].set(xlabel="Frozen prototypes per word",ylabel="Between-prototype / total variance",title="(A) Descriptive variance accounting")
    axs[0].legend(frameon=False)
    axs[1].boxplot([g.loc[g.category.eq(c),"top_neighbor_change_rate"] for c in categories],tick_labels=categories,showfliers=False)
    axs[1].set(ylabel="Occurrence vs. mean: top-neighbour change rate",title="(B) Changes in lexical neighbourhood")
    axs[1].tick_params(axis="x",rotation=45)
    save(fig,"02_aggregation_and_neighbours")
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,metric,title in zip(axs,["relative_gain","gain_minus_random","gain_minus_gaussian"],
                             ["(A) Held-out compression gain","(B) Gain beyond random partitions","(C) Gain beyond Gaussian reference"]):
        for scope,color in [("COHORT",colors[0]),("COMMON",colors[1])]:
            sub=summary.loc[summary.scope.eq(scope)&summary.training_fraction.eq(1)&summary.metric.eq(metric)&summary.status.eq("OK")].sort_values("k")
            if len(sub):
                ax.plot(sub.k,sub.estimate,"o-",label=scope,color=color)
                ax.fill_between(sub.k,sub.bootstrap_low,sub.bootstrap_high,color=color,alpha=.15)
        ax.axhline(0,color="grey",lw=.8)
        ax.set(xlabel="Diagnostic codebook size k",ylabel="Relative MSE reduction" if metric=="relative_gain" else "Difference in relative gain",title=title)
        ax.set_xticks([k for k in run["config"]["probe_k"] if metric=="relative_gain" or k>1])
    axs[0].legend(frameon=False)
    save(fig,"03_heldout_controls")
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    sub=summary.loc[summary.scope.eq("COHORT")&summary.metric.eq("relative_gain")&summary.status.eq("OK")]
    for k in sorted(sub.k.unique()):
        q=sub.loc[sub.k.eq(k)].sort_values("training_fraction")
        axs[0].plot(q.training_fraction,q.estimate,"o-",label=f"k={k}")
        axs[0].fill_between(q.training_fraction,q.bootstrap_low,q.bootstrap_high,alpha=.12)
    axs[0].axhline(0,color="grey",lw=.8)
    axs[0].set(xlabel="Fraction of available training groups",ylabel="Held-out relative MSE reduction",title="(A) Sampling sensitivity")
    axs[0].legend(frameon=False)
    q=summary.loc[summary.scope.eq("COHORT")&summary.training_fraction.eq(1)&summary.metric.eq("stability_ARI")&summary.status.eq("OK")].sort_values("k")
    if len(q):
        axs[1].plot(q.k,q.estimate,"o-",color=colors[1]);axs[1].fill_between(q.k,q.bootstrap_low,q.bootstrap_high,color=colors[1],alpha=.15)
    axs[1].axhline(0,color="grey",lw=.8)
    axs[1].set(xlabel="Diagnostic codebook size k",ylabel="Adjusted Rand index on shared held-out rows",title="(B) Stability across disjoint training halves")
    axs[1].set_xticks([k for k in run["config"]["probe_k"] if k>1])
    save(fig,"04_sampling_and_stability")
    occ=pd.read_csv(root/"tables/occurrence_reference_summary.csv")
    if len(occ) and occ.status.eq("OK").any():
        fig,ax=plt.subplots(figsize=(7,4))
        for scope,color in [("COHORT",colors[0]),("COMMON",colors[1])]:
            q=occ.loc[occ.scope.eq(scope)&occ.status.eq("OK")].sort_values("training_fraction")
            if len(q):
                ax.plot(q.training_fraction,q.estimate,"o-",label=scope,color=color)
                ax.fill_between(q.training_fraction,q.bootstrap_low,q.bootstrap_high,color=color,alpha=.15)
        ax.axhline(0,color="grey",lw=.8)
        ax.set(xlabel="Fraction of available training groups",ylabel="Relative held-out MSE reduction vs. TRAIN mean",
               title="TRAIN-occurrence codebook: a larger-capacity reference")
        ax.legend(frameon=False)
        save(fig,"06_occurrence_codebook_reference")
    t=pd.read_csv(root/"text_alignment.csv")
    if "spearman" in t and t.status.eq("OK").any():
        q=t.loc[t.status.eq("OK")]
        fig,ax=plt.subplots(figsize=(7,4))
        ax.boxplot([q.loc[q.category.eq(c),"rho_minus_null"].dropna() for c in categories],tick_labels=categories,showfliers=False)
        ax.axhline(0,color="grey",lw=.8);ax.tick_params(axis="x",rotation=45)
        ax.set(ylabel="Spearman correlation minus permutation mean",title="Context-lexical association (not sense validation)")
        save(fig,"05_text_alignment")
    write_json(root/"FIGURES.json",dict(files=sorted(p.name for p in out.iterdir())))
    return sorted(out.glob("*.png"))

def make_figures(run, dpi=220, formats=("png","pdf","svg")):
    import matplotlib.pyplot as plt
    with plt.rc_context():
        return _make_figures(run,dpi=dpi,formats=formats)
