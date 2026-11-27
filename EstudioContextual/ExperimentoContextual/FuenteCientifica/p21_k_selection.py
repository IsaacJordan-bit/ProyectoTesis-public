"""P21: criterios de soporte/geometría independientes de errores y de T3.

La selección solo abre geometry/* y INPUT_SHA256.json. El diagnóstico opcional
de FAR se ejecuta después, en otra función, y no modifica la selección.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "1.0.0"
CLASSES = ["apple", "bicycle", "bread", "canoe", "chair", "jellyfish",
           "screwdriver", "skunk"]
COLORS = {40: "#9c32aa", 50: "#138592"}
DEFAULT_CONFIG = {
    "K_values": [40, 50], "seed": 21021, "n_resamples": 500,
    "word_fraction": 0.80, "split_panel_sizes": [10, 26, 40],
    "primary_split_panel": 26, "primary_policy": "joint",
    "thresholds": {
        "max_reference_se": 0.10,
        "min_loo_estimable_fraction": 0.95,
        "max_loo_state_switch_q95": 0.10,
        "max_loo_edge_shift_q95_bins": 0.50,
        "min_word_estimable_fraction": 0.80,
        "max_word_state_switch_q95": 0.25,
        "min_split_estimable_fraction": 0.95,
        "max_split_state_switch_q95": 0.10,
    },
    # Sensibilidad un criterio a la vez; no se busca el umbral que favorezca K50.
    "threshold_sensitivity": {
        "max_reference_se": [0.075, 0.10, 0.125],
        "min_loo_estimable_fraction": [0.90, 0.95, 1.0],
        "max_loo_state_switch_q95": [0.05, 0.10, 0.20],
        "max_loo_edge_shift_q95_bins": [0.25, 0.50, 1.0],
        "min_word_estimable_fraction": [0.70, 0.80, 0.90],
        "max_word_state_switch_q95": [0.15, 0.25, 0.35],
        "min_split_estimable_fraction": [0.90, 0.95, 1.0],
        "max_split_state_switch_q95": [0.05, 0.10, 0.20],
    },
}


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                  allow_nan=False), encoding="utf-8")


def resolve_runtime(runtime=None):
    """Prefiere RUNTIME del notebook P21; no busca fuera de la carpeta actual."""
    if runtime is not None:
        root = Path(runtime).resolve()
        if (root / "data/w2v").is_dir() and (root / "INPUT_SHA256.json").is_file():
            return root
        raise FileNotFoundError(f"RUNTIME no contiene los datos verificados de P21: {root}")
    candidates = [Path.cwd(), Path.cwd() / "P21", *Path.cwd().glob("_p21_runtime_*")]
    found = sorted({p.resolve() for p in candidates
                    if (p / "data/w2v").is_dir() and (p / "INPUT_SHA256.json").is_file()})
    if len(found) != 1:
        raise RuntimeError("Ejecuta primero la celda de preparación de P21 que define RUNTIME. "
                           "Si hay varios runtimes, pasa RUNTIME explícitamente.")
    return found[0]


def _verified_path(root, relative, ledger, receipts):
    path = root / relative
    expected = ledger.get(relative)
    if expected is None:
        raise ValueError(f"Archivo sin hash en INPUT_SHA256.json: {relative}")
    actual = _digest(path)
    if actual != expected:
        raise ValueError(f"Hash incorrecto: {relative}. No se mezclan versiones.")
    receipts[relative] = actual
    return path


def _geometry(similarity, classes, selected):
    """Reproduce ddof=0, intersección de rangos de clase y cinco bins uniformes."""
    ids = np.asarray(selected, dtype=int)
    y, x = classes[ids], similarity[ids]
    sizes = np.bincount(y, minlength=8)
    if np.any(sizes < 2):
        return None
    means = np.bincount(y, weights=x, minlength=8) / sizes
    sd = np.sqrt(np.bincount(y, weights=(x - means[y]) ** 2, minlength=8) / sizes)
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        return None
    z = (x - means[y]) / sd[y]
    lower = max(z[y == c].min() for c in range(8))
    upper = min(z[y == c].max() for c in range(8))
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        return None
    edges = np.linspace(lower, upper, 6)
    inside = (z >= lower - 1e-12) & (z <= upper + 1e-12)
    state = np.full(len(ids), -1, dtype=int)
    state[inside] = np.searchsorted(edges[1:-1], z[inside], side="right").clip(0, 4)
    counts = np.bincount(y[inside] * 5 + state[inside], minlength=40).reshape(8, 5)
    return dict(ids=ids, z=z, edges=edges, state=state, counts=counts,
                estimable=bool((counts > 0).all()))


def load_geometries(runtime, K_values):
    """Entrada exclusivamente geométrica. No lee predicciones ni estadísticas T."""
    root = resolve_runtime(runtime)
    ledger = json.loads((root / "INPUT_SHA256.json").read_text(encoding="utf-8"))
    receipts, geometries = {}, {}
    for K in K_values:
        prefix = f"data/w2v/K{K}/geometry/"
        def read(name):
            p = _verified_path(root, prefix + name, ledger, receipts)
            return pd.read_parquet(p) if p.suffix == ".parquet" else json.loads(p.read_text(encoding="utf-8"))
        full = read("full.parquet").sort_values("word").reset_index(drop=True)
        common, expected = read("common.parquet"), read("expected.parquet")
        gate, audits = read("gate.json"), read("split_audit.json")
        if full.word.duplicated().any() or common.word.duplicated().any():
            raise ValueError(f"K{K}: las palabras de FULL/COMMON deben ser distintas.")
        if expected.duplicated(["split_seed", "word"]).any():
            raise ValueError(f"K{K}: observaciones query-split duplicadas.")
        if set(full.true_class) != set(range(8)) or not np.isfinite(full.validation_similarity).all():
            raise ValueError(f"K{K}: clases o similitudes inválidas.")
        if not set(common.word).issubset(set(full.word)) or set(expected.word) != set(full.word):
            raise ValueError(f"K{K}: inventarios incompatibles.")
        byword = full.set_index("word")
        for frame in [common, expected]:
            if not np.array_equal(frame.true_class, frame.word.map(byword.true_class)):
                raise ValueError(f"K{K}: etiquetas no alineadas por identidad.")
            if not np.allclose(frame.validation_similarity, frame.word.map(byword.validation_similarity), rtol=0, atol=1e-12):
                raise ValueError(f"K{K}: similitud variable o mal alineada.")
        x, y = full.validation_similarity.to_numpy(float), full.true_class.to_numpy(int)
        base = _geometry(x, y, np.arange(len(full)))
        if base is None or not base["estimable"]:
            raise ValueError(f"K{K}: geometría de referencia no admisible.")
        if not np.allclose(base["z"], full.similarity_wc_z, rtol=0, atol=1e-12):
            raise ValueError(f"K{K}: no se reproduce z con ddof=0.")
        if not np.allclose(base["edges"], gate["edges"], rtol=0, atol=1e-12):
            raise ValueError(f"K{K}: no se reproducen los límites de COMMON.")
        positions = pd.Series(np.arange(len(full)), index=full.word)
        if set(full.word[base["state"] >= 0]) != set(common.word):
            raise ValueError(f"K{K}: COMMON no se reproduce.")
        if not np.array_equal(base["state"][positions.loc[common.word]], common.similarity_bin):
            raise ValueError(f"K{K}: bins no se reproducen.")
        seeds = np.sort(expected.split_seed.unique())
        if set(seeds) != set(a["split_seed"] for a in audits) or len(audits) != len(seeds):
            raise ValueError(f"K{K}: recibos de splits incompletos.")
        structural = all(bool(gate.get(k)) for k in ["A", "A1", "A2", "A4"])
        presence = np.zeros((len(seeds), len(full)), dtype=bool)
        for i, s in enumerate(seeds):
            presence[i, positions.loc[expected.loc[expected.split_seed.eq(s), "word"]]] = True
        geometries[int(K)] = dict(K=int(K), full=full, common=common,
            x=x, y=y, words=full.word.to_numpy(), base=base, seeds=seeds,
            presence=presence, original_admissible=structural)
    first = next(iter(geometries.values()))["seeds"]
    if any(not np.array_equal(first, g["seeds"]) for g in geometries.values()):
        raise ValueError("Los K deben compartir los identificadores de split para el submuestreo pareado.")
    return root, geometries, receipts


def _precision_rows(g):
    """Referencia bajo independencia de palabras; no es un IC empírico."""
    rows = []
    for b, sizes in enumerate(g["base"]["counts"].T):
        sum_w2 = float(np.sum(1.0 / sizes) / 64.0)
        rows.append(dict(K=g["K"], bin=b,
            center=float(np.mean(g["base"]["edges"][b:b+2])),
            n_words=int(sizes.sum()), min_words_per_class=int(sizes.min()),
            singleton_cells=int((sizes == 1).sum()),
            max_word_weight=float(1.0 / (8 * sizes.min())),
            effective_weight_count=1.0 / sum_w2,
            reference_se=0.5 * np.sqrt(sum_w2)))
    return rows


def _perturb(g, selected):
    base = g["base"]
    ids = np.sort(np.asarray(selected, dtype=int))
    old_state = base["state"][ids]
    old_inside = old_state >= 0
    fixed_counts = np.bincount(g["y"][ids][old_inside] * 5 + old_state[old_inside], minlength=40)
    fixed = dict(fixed_occupied_cells=int((fixed_counts > 0).sum()),
                 fixed_estimable=bool((fixed_counts > 0).all()))
    new = _geometry(g["x"], g["y"], ids)
    if new is None:
        return dict(n_words=len(ids), n_common=0, occupied_cells=0,
            geometry_valid=False, estimable=False, state_switch=1.0,
            common_jaccard=0.0, edge_shift_bins=float("inf"), z_rmse=float("nan"), **fixed)
    new_state = new["state"]
    a, b = old_state >= 0, new_state >= 0
    union = int(np.sum(a | b))
    return dict(n_words=len(ids), n_common=int(b.sum()),
        occupied_cells=int((new["counts"] > 0).sum()), geometry_valid=True,
        estimable=new["estimable"], state_switch=float(np.mean(old_state != new_state)),
        common_jaccard=float(np.sum(a & b) / union) if union else 1.0,
        edge_shift_bins=float(np.max(np.abs(new["edges"] - base["edges"])) / np.diff(base["edges"])[0]),
        z_rmse=float(np.sqrt(np.mean((new["z"] - base["z"][ids]) ** 2))), **fixed)


def _q95(values):
    # nearest evita interpolar infinito con infinito en una geometría fallida.
    return float(np.quantile(np.asarray(values, float), .95, method="nearest"))


def _aggregate(precision, perturbations, geometries, cfg):
    rows = []
    for K, g in geometries.items():
        p, r = precision.query("K == @K"), perturbations.query("K == @K")
        row = dict(K=K, original_admissible=g["original_admissible"],
            full_words=len(g["full"]), common_words=int(g["base"]["counts"].sum()),
            singleton_cells=int((g["base"]["counts"] == 1).sum()),
            max_reference_se=float(p.reference_se.max()),
            min_effective_weight_count=float(p.effective_weight_count.min()),
            max_word_weight=float(p.max_word_weight.max()))
        panels = {"loo": r[r.scheme.eq("delete_one_word")],
                  "word": r[r.scheme.eq("word_subsample")],
                  "split": r[r.scheme.eq(f"splits_{cfg['primary_split_panel']}")]}
        for label, panel in panels.items():
            row[label + "_estimable_fraction"] = float(panel.estimable.mean())
            row[label + "_state_switch_q95"] = _q95(panel.state_switch)
            row[label + "_edge_shift_q95_bins"] = _q95(panel.edge_shift_bins)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_policy(summary, thresholds):
    """Solo métricas de soporte/precisión de diseño/geometría. Sin acceso a T3."""
    rules = [
        ("precision", "max_reference_se", "max_reference_se", "<="),
        ("geometry", "loo_estimable_fraction", "min_loo_estimable_fraction", ">="),
        ("geometry", "loo_state_switch_q95", "max_loo_state_switch_q95", "<="),
        ("geometry", "loo_edge_shift_q95_bins", "max_loo_edge_shift_q95_bins", "<="),
        ("sampling", "word_estimable_fraction", "min_word_estimable_fraction", ">="),
        ("sampling", "word_state_switch_q95", "max_word_state_switch_q95", "<="),
        ("sampling", "split_estimable_fraction", "min_split_estimable_fraction", ">="),
        ("sampling", "split_state_switch_q95", "max_split_state_switch_q95", "<="),
    ]
    checks, decisions = [], []
    for _, row in summary.iterrows():
        K = int(row.K)
        for group, metric, key, op in rules:
            value, limit = float(row[metric]), float(thresholds[key])
            passed = np.isfinite(value) and (value <= limit if op == "<=" else value >= limit)
            checks.append(dict(K=K, group=group, metric=metric, value=value,
                               operator=op, threshold=limit, passed=bool(passed)))
    checks = pd.DataFrame(checks)
    original = summary.set_index("K").original_admissible
    for policy in ["precision", "geometry", "sampling", "joint"]:
        part = checks if policy == "joint" else checks[checks.group.eq(policy)]
        eligible = [int(K) for K, p in part.groupby("K") if p.passed.all() and original.loc[K]]
        decisions.append(dict(policy=policy, eligible_K=eligible,
                              selected_K=min(eligible) if eligible else None))
    return checks, decisions


def _validate_config(config):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    if cfg["primary_policy"] != "joint":
        raise ValueError("La regla principal de esta versión es joint; las otras son diagnósticas.")
    if not (0 < cfg["word_fraction"] < 1) or cfg["n_resamples"] < 100:
        raise ValueError("Usa una fracción en (0,1) y al menos 100 remuestras.")
    if sorted(cfg["K_values"]) != [40, 50] or cfg["primary_split_panel"] not in cfg["split_panel_sizes"]:
        raise ValueError("Esta extensión contrasta K40/K50 y exige el panel principal entre los tamaños declarados.")
    if set(cfg["thresholds"]) != set(DEFAULT_CONFIG["thresholds"]):
        raise ValueError("Deben declararse los ocho umbrales de esta versión.")
    for key, value in cfg["thresholds"].items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Umbral inválido: {key}")
    return cfg


def run_selection(runtime=None, output=None, config=None, show=True):
    cfg = _validate_config(config)
    root, geometries, receipts = load_geometries(runtime, cfg["K_values"])
    seeds = next(iter(geometries.values()))["seeds"]
    if any(m < 2 or m > len(seeds) for m in cfg["split_panel_sizes"]):
        raise ValueError("Tamaño de panel incompatible con los splits disponibles.")
    plan = dict(version=VERSION, config=cfg, input_sha256=receipts,
                scope="W2V K40 versus K50; observed surface geometry only",
                status="retrospective exploratory; not independent confirmation",
                outcome_files_used_for_selection=[],
                software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    if globals().get("_SOURCE_SHA256"):
        plan["module_sha256"] = _SOURCE_SHA256
    elif "__file__" in globals() and Path(__file__).is_file():
        plan["module_sha256"] = _digest(__file__)
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    parent = Path(output) if output is not None else Path.cwd() / "outputs" / "k_selection"
    folder = parent / identity
    folder.mkdir(parents=True, exist_ok=True)
    plan_path = folder / "PLAN.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError("Existe otro plan bajo este identificador; no se sobrescribe.")
    if not plan_path.exists():
        _json(plan_path, plan)  # escrito ANTES de calcular las perturbaciones
    print(f"Plan fijado para esta ejecución: {identity}. Alcance: exploratorio retrospectivo.")
    print("Selección: no se leen errores, predicciones ni T1–T3.")
    precision = pd.DataFrame([row for g in geometries.values() for row in _precision_rows(g)])
    coverage = pd.DataFrame([dict(K=K, true_class=CLASSES[c], bin=b, n_words=int(n))
        for K, g in geometries.items() for c in range(8) for b, n in enumerate(g["base"]["counts"][c])])
    rows, draws = [], []
    for K, g in geometries.items():
        ids = np.arange(len(g["full"]))
        for omitted in ids:
            rows.append(dict(K=K, scheme="delete_one_word", replicate=int(omitted),
                removed_word=str(g["words"][omitted]), removed_class=CLASSES[g["y"][omitted]],
                **_perturb(g, ids[ids != omitted])))
        print(f"K{K}: {len(ids)} omisiones léxicas exactas completadas.")
    union_words = np.array(sorted(set().union(*(set(g["words"]) for g in geometries.values()))))
    positions = {w: i for i, w in enumerate(union_words)}
    word_indices = {K: np.array([positions[w] for w in g["words"]]) for K, g in geometries.items()}
    rng_words = np.random.default_rng(cfg["seed"])
    rng_splits = np.random.default_rng(cfg["seed"] + 1)
    for rep in range(cfg["n_resamples"]):
        priority = rng_words.random(len(union_words))
        for K, g in geometries.items():
            chosen = []
            for c in range(8):
                candidates = np.flatnonzero(g["y"] == c)
                n = max(2, int(np.floor(cfg["word_fraction"] * len(candidates))))
                chosen.extend(candidates[np.argsort(priority[word_indices[K][candidates]], kind="stable")[:n]])
            rows.append(dict(K=K, scheme="word_subsample", replicate=rep, **_perturb(g, chosen)))
        # Un mismo orden aleatorio para ambos K y paneles anidados de 10/26/40.
        ordered = rng_splits.permutation(len(seeds))
        for m in cfg["split_panel_sizes"]:
            selected_seeds = ordered[:m]
            draws.append(dict(replicate=rep, n_splits=m,
                              split_seeds=";".join(map(str, seeds[selected_seeds]))))
            for K, g in geometries.items():
                chosen = np.flatnonzero(g["presence"][selected_seeds].any(axis=0))
                rows.append(dict(K=K, scheme=f"splits_{m}", replicate=rep, **_perturb(g, chosen)))
        if (rep + 1) % 100 == 0:
            print(f"Remuestras geométricas: {rep+1}/{cfg['n_resamples']}.")
    perturbations = pd.DataFrame(rows)
    summary = _aggregate(precision, perturbations, geometries, cfg)
    checks, decisions = apply_policy(summary, cfg["thresholds"])
    sensitivity = []
    for key, values in cfg["threshold_sensitivity"].items():
        for value in values:
            thresholds = dict(cfg["thresholds"], **{key: value})
            _, variants = apply_policy(summary, thresholds)
            joint = next(d for d in variants if d["policy"] == "joint")
            sensitivity.append(dict(changed_threshold=key, value=value, **joint))
    # Dominancia de Pareto: informa compensaciones sin inventar una suma ponderada.
    metrics = summary[["max_reference_se", "loo_state_switch_q95", "loo_edge_shift_q95_bins",
                       "word_state_switch_q95", "split_state_switch_q95"]].copy()
    for key in ["loo", "word", "split"]:
        metrics[key + "_failure_fraction"] = 1 - summary[key + "_estimable_fraction"]
    dominated = []
    for i in range(len(summary)):
        a = metrics.iloc[i].to_numpy()
        dominated.append(any(np.all(metrics.iloc[j].to_numpy() <= a) and
            np.any(metrics.iloc[j].to_numpy() < a) for j in range(len(summary)) if j != i))
    summary["pareto_dominated"] = dominated
    scheme_summary = perturbations.groupby(["K", "scheme"]).agg(
        n_perturbations=("estimable", "size"), estimable_fraction=("estimable", "mean"),
        fixed_geometry_estimable_fraction=("fixed_estimable", "mean"),
        geometry_valid_fraction=("geometry_valid", "mean"),
        mean_words=("n_words", "mean"), mean_common_words=("n_common", "mean"),
        state_switch_q95=("state_switch", _q95), edge_shift_q95_bins=("edge_shift_bins", _q95),
        mean_common_jaccard=("common_jaccard", "mean")).reset_index()
    for name, frame in {"precision_by_bin": precision, "coverage": coverage,
            "perturbations": perturbations, "summary": summary, "criteria": checks,
            "sampling_summary": scheme_summary, "threshold_sensitivity": pd.DataFrame(sensitivity),
            "split_draws": pd.DataFrame(draws)}.items():
        frame.to_csv(folder / (name + ".csv"), index=False)
    perturbations.to_parquet(folder / "perturbations.parquet", index=False)
    _json(folder / "decisions.json", dict(primary_policy="joint", decisions=decisions,
          not_a_confirmatory_selection=True, T3_used=False))
    receipt = dict(input_hashes_verified=True, original_z_common_bins_reproduced=True,
                   distinct_words_used=True, paired_split_draws=True, timestamp_utc=datetime.now(timezone.utc).isoformat())
    _json(folder / "validation.json", receipt)
    result = dict(root=root, output=folder, config=cfg, geometries=geometries,
        precision=precision, coverage=coverage, perturbations=perturbations,
        summary=summary, criteria=checks, sampling_summary=scheme_summary,
        decisions=decisions, sensitivity=pd.DataFrame(sensitivity))
    if show:
        show_selection(result)
    print("Regla principal:", next(d for d in decisions if d["policy"] == "joint"))
    print("Resultados guardados en:", folder.resolve())
    return result


def _save_show(fig, folder, name, show=True):
    fig.savefig(folder / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def show_selection(result):
    from IPython.display import display
    p, r, cfg, folder = result["precision"], result["perturbations"], result["config"], result["output"]
    display(result["summary"].round(4))
    display(result["criteria"].round(4))
    display(pd.DataFrame(result["decisions"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), constrained_layout=True)
    for K, part in p.groupby("K"):
        for ax, metric in zip(axes, ["reference_se", "effective_weight_count", "max_word_weight"]):
            ax.plot(part.bin + 1, part[metric], "o-", color=COLORS[K], label=f"K{K}")
            ax.set_xticks(range(1, 6)); ax.set_xlabel("COMMON bin (within each K)")
    axes[0].axhline(cfg["thresholds"]["max_reference_se"], color="0.4", ls="--", label="Proposed limit")
    axes[0].set_ylabel("Worst-case SE reference"); axes[0].set_title("Independent-word design benchmark")
    axes[1].set_ylabel("1 / sum of squared word weights"); axes[1].set_title("Effective weight count")
    axes[2].set_ylabel("Maximum weight of one word"); axes[2].set_title("Single-word concentration")
    axes[0].legend(fontsize=8)
    _save_show(fig, folder, "01_precision_design")
    schemes = ["delete_one_word", "word_subsample", *[f"splits_{m}" for m in cfg["split_panel_sizes"]]]
    labels = ["Delete one word", f"Keep {cfg['word_fraction']:.0%} per class", *[f"{m} splits" for m in cfg["split_panel_sizes"]]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for i, K in enumerate(cfg["K_values"]):
        parts = [r[r.K.eq(K) & r.scheme.eq(s)] for s in schemes]
        x = np.arange(len(schemes)) + (i - .5) * .34
        axes[0].bar(x, [d.estimable.mean() for d in parts], .32, color=COLORS[K], label=f"K{K}")
        axes[1].plot(np.arange(len(schemes)), [_q95(d.state_switch) for d in parts], "o-", color=COLORS[K], label=f"K{K}")
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Fraction retaining all 40 occupied cells")
    axes[1].set_ylabel("95th percentile of word-state changes"); axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xticks(range(len(schemes)), labels, rotation=20, ha="right"); ax.legend()
    axes[0].set_title("Support sensitivity"); axes[1].set_title("Bin / COMMON membership sensitivity")
    _save_show(fig, folder, "02_sampling_and_geometry")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for ax, K in zip(axes, cfg["K_values"]):
        parts = [r[r.K.eq(K) & r.scheme.eq(s)] for s in schemes]
        x = np.arange(len(schemes))
        ax.bar(x - .17, [d.fixed_estimable.mean() for d in parts], .32,
               color="0.65", label="Keep reference geometry")
        ax.bar(x + .17, [d.estimable.mean() for d in parts], .32,
               color=COLORS[K], label="Recompute geometry")
        ax.set_xticks(x, labels, rotation=20, ha="right"); ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction retaining all 40 occupied cells")
        ax.set_title(f"K{K}: lexical loss versus rebinning")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.30))
    _save_show(fig, folder, "02b_fixed_vs_recomputed_geometry")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharex=True)
    for ax, K in zip(axes, cfg["K_values"]):
        worst = r[r.K.eq(K) & r.scheme.eq("delete_one_word")].nlargest(10, "state_switch").sort_values("state_switch")
        ax.barh(worst.removed_word, worst.state_switch, color=COLORS[K])
        ax.set_xlabel("Fraction of remaining words changing bin / COMMON status")
        ax.set_title(f"K{K}: most influential word omissions")
    _save_show(fig, folder, "03_influential_words")
    print("Los bins tienen límites propios en cada K. Estas figuras comparan calidad del diseño, no FAR en idénticos intervalos semánticos.")


def far_word_sensitivity(result, n_boot=1000, seed=21023, network="mhncs_softmax", show=True):
    """Diagnóstico posterior y condicional. NO participa en apply_policy.

    Remuestrea curvas completas por palabra dentro de cada celda clase-bin.
    No estima variabilidad entre splits ni entre entrenamientos.
    """
    if n_boot < 100:
        raise ValueError("Usa al menos 100 remuestras para el diagnóstico descriptivo.")
    folder = result["output"] / "far_diagnostic"
    folder.mkdir(exist_ok=True)
    root = result["root"]
    ledger = json.loads((root / "INPUT_SHA256.json").read_text(encoding="utf-8"))
    receipts, frames = {}, []
    rng = np.random.default_rng(seed)
    # Se conserva exactamente el archivo de decisión, incluso tras este diagnóstico.
    decision_sha = _digest(result["output"] / "decisions.json")
    for K, g in result["geometries"].items():
        relative = f"data/w2v/K{K}/counts/{network}/word_beta.parquet"
        wb = pd.read_parquet(_verified_path(root, relative, ledger, receipts))
        if wb.duplicated(["word", "beta"]).any() or (wb.n_items <= 0).any() or not wb.n_errors.between(0, wb.n_items).all():
            raise ValueError("Conteos por palabra/beta inválidos o duplicados.")
        cls = g["full"].set_index("word").true_class
        if set(wb.word) != set(cls.index) or not np.array_equal(wb.true_class, wb.word.map(cls)):
            raise ValueError("Conteos sin alineación por palabra/clase.")
        wb = wb.assign(rate=wb.n_errors / wb.n_items)
        rates = wb.pivot(index="word", columns="beta", values="rate").sort_index(axis=1)
        if rates.isna().any().any():
            raise ValueError("Grilla beta incompleta por palabra.")
        beta = rates.columns.to_numpy(float)
        for b in range(5):
            observed = np.zeros(len(beta)); bootstrap = np.zeros((n_boot, len(beta)))
            singleton_classes = []
            for c in range(8):
                words = g["words"][(g["y"] == c) & (g["base"]["state"] == b)]
                values = rates.loc[words].to_numpy(float)
                n = len(words)
                if n == 1:
                    singleton_classes.append(CLASSES[c])
                observed += values.mean(axis=0) / 8
                multiplicity = rng.multinomial(n, np.full(n, 1/n), size=n_boot)
                bootstrap += multiplicity @ values / (8 * n)
            low, high = np.quantile(bootstrap, [.025, .975], axis=0)
            # Guardar remuestras permite cambiar títulos, curvas, percentiles sin recalcular.
            np.savez_compressed(folder / f"K{K}_bin{b}_word_bootstrap.npz",
                                beta=beta, bootstrap=bootstrap, observed=observed)
            frames.append(pd.DataFrame(dict(K=K, bin=b, beta=beta, observed_far=observed,
                q025=low, q975=high, band_width=high-low,
                singleton_classes=";".join(singleton_classes), n_singleton_classes=len(singleton_classes))))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(folder / "conditional_word_bands.csv", index=False)
    summary = frame.groupby(["K", "bin"]).agg(mean_band_width=("band_width", "mean"),
        max_band_width=("band_width", "max"), n_singleton_classes=("n_singleton_classes", "max")).reset_index()
    summary.to_csv(folder / "conditional_word_summary.csv", index=False)
    _json(folder / "DIAGNOSTIC.json", dict(n_boot=n_boot, seed=seed, network=network,
        input_sha256=receipts, conditional_on_existing_word_rates=True,
        confidence_interval_for_generalization=False, selection_uses_this_diagnostic=False,
        decision_sha256=decision_sha))
    if decision_sha != _digest(result["output"] / "decisions.json"):
        raise RuntimeError("El diagnóstico modificó indebidamente la decisión.")
    if show:
        from IPython.display import display
        display(summary.round(4))
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), constrained_layout=True)
        vmax = max(float(frame.band_width.max()), 1e-12)
        for ax, K in zip(axes, result["config"]["K_values"]):
            part = frame[frame.K.eq(K)]
            grid = part.pivot(index="bin", columns="beta", values="band_width")
            im = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                extent=[grid.columns.min(), grid.columns.max(), .5, 5.5])
            ax.set_xlabel(r"$\beta$"); ax.set_ylabel("COMMON bin")
            ax.set_yticks(range(1, 6)); ax.set_title(f"K{K}: conditional word-resampling width")
        fig.colorbar(im, ax=axes, label="97.5th minus 2.5th percentile (FAR)")
        _save_show(fig, folder, "04_conditional_far_width")
    print("Diagnóstico descriptivo, fuera del selector. Una celda de una palabra no aporta variación al bootstrap;")
    print("su banda estrecha no acredita precisión. No es un intervalo de confianza entre splits/entrenamientos.")
    return frame, summary


def export_results(result):
    folder = result["output"].resolve()
    target = folder.parent / ("P21_K_criteria_" + folder.name + ".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                archive.write(p, arcname=str(Path(folder.name) / p.relative_to(folder)))
    print("Paquete de resultados:", target)
    return target
