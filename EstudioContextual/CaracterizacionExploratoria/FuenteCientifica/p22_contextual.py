"""P22: outcome-independent characterization of the frozen P20/P21 A/P/O bank.

All source files are read-only. This module never imports retrieval/classifier
code, changes the historical partition, or uses T1/T2/T3 to select anything.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import socket
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

VERSION = "P22_CONTEXTUAL_PROPERTIES_1.0"
HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference"
_WORKER = {}


class AuditError(RuntimeError):
    pass


class TechnicalPause(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def digest(obj):
    return hashlib.sha256(json.dumps(clean_json(obj), sort_keys=True,
        ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


def environment():
    names = ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow",
             "matplotlib", "psutil", "threadpoolctl", "jinja2"]
    versions = {n: importlib.metadata.version(n) for n in names}
    return dict(python=platform.python_version(), packages=versions,
                platform=platform.platform(), machine=platform.machine())


@dataclass
class Config:
    bank: Path = field(default_factory=lambda: Path.home() / "E6R/p20_68682213c891")
    output_parent: Path = field(default_factory=lambda: Path.home() / "E6R")
    # Optional source texts. None discovers the known source if it still exists.
    source_parquet: Path | None = None
    use_text: bool = True
    split_seeds: tuple = (0, 1, 2, 3, 4)
    probe_k: tuple = (1, 2, 4, 8)
    training_fractions: tuple = (0.25, 0.5, 1.0)
    test_group_fraction: float = 0.20
    kmeans_n_init: int = 5
    kmeans_max_iter: int = 100
    control_repetitions: int = 20
    text_permutations: int = 199
    bootstrap_repetitions: int = 2000
    nearest_neighbors: int = 10
    min_train_occurrences: int = 16
    min_test_occurrences: int = 10
    min_source_groups: int = 10
    # Operational settings do not change the scientific contract.
    max_workers: int = 6
    reserve_gib: float = 3.0
    worker_budget_gib: float = 0.8
    max_hours_per_call: float = 4.0

    def __post_init__(self):
        self.bank = Path(self.bank).expanduser().resolve()
        self.output_parent = Path(self.output_parent).expanduser().resolve()
        if self.source_parquet is not None:
            self.source_parquet = Path(self.source_parquet).expanduser().resolve()
        require(not self.output_parent.is_relative_to(self.bank), "Outputs must be outside the source bank.")
        require(self.probe_k == tuple(sorted(set(self.probe_k))) and self.probe_k[0] == 1,
                "probe_k must be sorted, unique, and include 1.")
        require(len(set(self.split_seeds)) == len(self.split_seeds) and len(self.split_seeds) >= 2,
                "Use at least two distinct resampling seeds.")
        require(1.0 in self.training_fractions and all(0 < f <= 1 for f in self.training_fractions),
                "Training fractions must lie in (0, 1] and include 1.")
        require(0 < self.test_group_fraction < 0.5, "Invalid held-out fraction.")
        require(self.control_repetitions >= 2 and self.text_permutations >= 19,
                "Insufficient control repetitions.")
        require(self.max_workers >= 1 and self.reserve_gib >= 1 and self.max_hours_per_call > 0,
                "Invalid resource settings.")

    def scientific(self):
        out = asdict(self)
        for key in ("output_parent", "max_workers", "reserve_gib", "worker_budget_gib", "max_hours_per_call"):
            out.pop(key)
        return clean_json(out)


def preflight(cfg):
    info = environment()  # Fails early with the actual missing dependency.
    folder = cfg.bank / "inputs/apo"
    required = [cfg.bank / "protocol.json"] + [folder / n for n in
        ("done.json", "A.npy", "P.npy", "O.npy", "words.json", "units.parquet", "occurrences.parquet")]
    missing = [str(p) for p in required if not p.is_file()]
    require(not missing, "Frozen P20 bank not found. Set BANK in the notebook. Missing:\n" + "\n".join(missing))
    require(psutil.virtual_memory().available / 2**30 >= cfg.reserve_gib + 0.5,
            "Insufficient free RAM for the audit. Close other heavy jobs and retry.")
    cfg.output_parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(cfg.output_parent).free / 2**30 >= 1,
            "At least 1 GiB of output disk space is required.")
    return info


def load_bank(bank):
    folder = Path(bank) / "inputs/apo"
    return dict(A=np.load(folder / "A.npy", mmap_mode="r", allow_pickle=False),
                P=np.load(folder / "P.npy", mmap_mode="r", allow_pickle=False),
                O=np.load(folder / "O.npy", mmap_mode="r", allow_pickle=False),
                words=read_json(folder / "words.json"),
                units=pd.read_parquet(folder / "units.parquet"),
                meta=pd.read_parquet(folder / "occurrences.parquet"))


def validate_alignment(d, expected_n=640, expected_dim=768, expected_per_word=100):
    """Identity-first audit; never reorders metadata to conceal an array mismatch."""
    A, P, O, words, u, m = (d[k] for k in ("A", "P", "O", "words", "units", "meta"))
    require(len(words) == expected_n and len(set(words)) == len(words), "Word inventory mismatch.")
    require(A.shape == (expected_n, expected_dim) and O.shape == (expected_n * expected_per_word, expected_dim),
            "A/O array shapes do not match the frozen inventory.")
    require(P.shape == (len(u), expected_dim) and len(m) == len(O), "P/O metadata shape mismatch.")
    require({"word", "support", "unit_id", "unit_index"} <= set(u), "Missing prototype identity fields.")
    require({"word", "occ_id", "source_doc_key", "word_pos_original", "a_index", "p_index"} <= set(m),
            "Missing occurrence identity fields.")
    require(not m[["word", "occ_id", "source_doc_key", "word_pos_original", "a_index", "p_index"]].isna().any().any(),
            "Null occurrence identity.")
    require(not m.occ_id.duplicated().any() and not u.unit_id.duplicated().any() and not u.unit_index.duplicated().any(), "Duplicated occurrence/prototype IDs.")
    require(not m.duplicated(["word", "source_doc_key", "word_pos_original"]).any(), "Duplicated lexical occurrence identity.")
    for col, upper in (("a_index", len(A)), ("p_index", len(P))):
        v = m[col].to_numpy()
        require(np.issubdtype(v.dtype, np.integer) and np.all((v >= 0) & (v < upper)), "Invalid " + col)
    require(np.array_equal(np.asarray(words)[m.a_index], m.word.to_numpy()), "A index/word identity mismatch.")
    require(np.array_equal(u.word.to_numpy()[m.p_index], m.word.to_numpy()), "P index/word identity mismatch.")
    for name in ("unit_id", "sense_id"):
        if name in m and name in u:
            require(np.array_equal(u[name].to_numpy()[m.p_index], m[name].to_numpy()), "Prototype metadata mismatch: "+name)
    require(set(u.word) == set(words), "Prototype word inventory mismatch.")
    require((u.support.to_numpy() > 0).all(), "Every frozen prototype must have positive support.")
    require(np.array_equal(np.bincount(m.p_index, minlength=len(P)), u.support.to_numpy()), "Prototype support mismatch.")
    require(np.array_equal(np.bincount(m.a_index, minlength=len(A)), np.full(len(A), expected_per_word)),
            "Unequal or missing occurrence counts.")
    for name, x in (("A", A), ("P", P), ("O", O)):
        require(x.dtype == np.float64, "Expected canonical float64 array: " + name)
        for left in range(0, len(x), 4096):
            block = x[left:left+4096]
            require(np.isfinite(block).all() and (np.linalg.norm(block, axis=1) > 0).all(), "Invalid vectors: " + name)
    max_a, max_p = 0.0, 0.0
    for pi, ids in m.groupby("p_index", sort=True).indices.items():
        avg = np.asarray(O[ids], dtype=np.float64).mean(0)
        require(np.allclose(avg, P[pi], rtol=1e-12, atol=1e-12), "Occurrence mean does not reproduce P: " + str(pi))
        max_p = max(max_p, float(np.max(abs(avg-P[pi]))))
    for ai, ids in m.groupby("a_index", sort=True).indices.items():
        avg = np.asarray(O[ids], dtype=np.float64).mean(0)
        mask = u.word.eq(words[ai]).to_numpy()
        weighted = np.average(P[mask], axis=0, weights=u.loc[mask, "support"])
        require(np.allclose(avg, A[ai], rtol=1e-12, atol=1e-12), "Occurrence mean does not reproduce A: " + words[ai])
        require(np.allclose(weighted, A[ai], rtol=1e-12, atol=1e-12), "Weighted P mean does not reproduce A: " + words[ai])
        max_a = max(max_a, float(np.max(abs(avg-A[ai]))))
    return dict(words=len(words), occurrences=len(O), prototypes=len(P), dimension=expected_dim,
                occurrences_per_word=expected_per_word, max_abs_A_delta=max_a, max_abs_P_delta=max_p)


def audit_bank(cfg, reference=None):
    reference = REFERENCE if reference is None else Path(reference)
    preflight(cfg)
    expected = read_json(reference / "bank_protocol.json")
    bank_id = read_json(reference / "bank_identity.json")["run_id"]
    require(read_json(cfg.bank / "protocol.json") == expected,
            "The bank protocol differs from the reviewed P20/P21 source. Do not substitute another export.")
    folder = cfg.bank / "inputs/apo"
    receipt = read_json(folder / "done.json")
    require(receipt["identity"] == digest(dict(run=bank_id, stage="CANONICAL_APO")), "Bank/receipt identity mismatch.")
    required = {"A.npy", "P.npy", "O.npy", "words.json", "units.parquet", "occurrences.parquet"}
    require(required <= set(receipt["files"]), "Incomplete canonical input receipt.")
    source_hashes = {}
    for name, expected_hash in receipt["files"].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()) and path.is_file(), "Unsafe or missing receipt path.")
        actual = sha(path)
        require(actual == expected_hash, "Canonical input hash changed: " + name)
        source_hashes[name] = actual
    d = load_bank(cfg.bank)
    detail = validate_alignment(d, expected["n_words"], 768, expected["positions_per_word"])
    geometry = read_json(reference / "geometry.json")
    labels = {w: c for c, ws in geometry["neighbors"].items() for w in ws}
    require(len(labels) == len(d["words"]) and set(labels) == set(d["words"]), "Historical cohort differs from the bank.")
    common = pd.read_parquet(reference / "common.parquet")
    full = pd.read_parquet(reference / "full.parquet")
    detail.update(bank_id=bank_id, source_hashes=source_hashes, label_map=labels,
                  common_words=sorted(set(common.word)), full_words=sorted(set(full.word)),
                  group_note="source_doc_key identifies a source record, not necessarily an independent document")
    return d, detail


def source_text_path(cfg):
    if cfg.source_parquet is not None:
        require(cfg.source_parquet.is_file(), "Explicit source_parquet does not exist.")
        return cfg.source_parquet
    path = Path.home() / "Workspace/Tesis/Exp2/Results/contextual_dataset/contextual_occurrences_definitive.parquet"
    return path if path.is_file() else None


def extract_texts(cfg, meta):
    """Read only selected text/identity columns; never the 40-GB embedding column."""
    if not cfg.use_text:
        return None, dict(status="DISABLED", semantic_validation=False)
    source = source_text_path(cfg)
    if source is None:
        return None, dict(status="NOT_AVAILABLE", reason="Canonical source text parquet was not found.", semantic_validation=False)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(source)
    schema = set(pf.schema_arrow.names)
    candidates = [c for c in ("context_window", "text") if c in schema]
    require(candidates, "No context_window or text column in the canonical source parquet.")
    column = candidates[0]
    cols = ["occ_id", "word", "source_doc_key", "word_pos_original", column]
    require(set(cols) <= schema, "Source text identities are incomplete.")
    ids = np.sort(meta.occ_id.to_numpy(np.int64))
    starts = np.r_[0, np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])]
    # The reviewed source uses occ_id == absolute row index. Verify rather than assume it.
    groups = np.unique(np.searchsorted(starts[1:], ids, side="right"))
    require(groups.max() < pf.num_row_groups, "Source row positions no longer match occurrence IDs.")
    pieces = []
    for count, group in enumerate(groups):
        block = pf.read_row_group(int(group), columns=cols, use_threads=False).to_pandas()
        require(np.array_equal(block.occ_id.to_numpy(), np.arange(starts[group], starts[group+1])),
                "Noncanonical source row/occurrence alignment.")
        pieces.append(block.loc[block.occ_id.isin(ids)])
        if count % 25 == 0:
            print(f"Reading canonical text row groups: {count+1}/{len(groups)}", flush=True)
    text = pd.concat(pieces, ignore_index=True).set_index("occ_id", verify_integrity=True).loc[meta.occ_id].reset_index()
    for name in ("occ_id", "word", "source_doc_key", "word_pos_original"):
        require(np.array_equal(text[name].astype(str).to_numpy(), meta[name].astype(str).to_numpy()),
                "Text identity mismatch: " + name)
    require(not text[column].isna().any(), "Missing canonical context text.")
    text = text.rename(columns={column: "context"})
    # Hash the exact selected material, not a guessed hash of the full source file.
    selected_hash = digest(text.to_dict("records"))
    return text, dict(status="AVAILABLE", path=str(source), column=column,
                     selected_content_sha256=selected_hash, source_size=source.stat().st_size,
                     full_source_rehashed=False, canonical_identities_verified=True,
                     semantic_validation=False)


def prepare(cfg):
    """Authenticate once, freeze the plan, and create a separate content-addressed run."""
    with threadpool_limits(limits=1):
        d, audit = audit_bank(cfg)
        texts, text_status = extract_texts(cfg, d["meta"])
    plan = dict(version=VERSION, scientific_config=cfg.scientific(),
                code_sha256=sha(__file__), environment=environment(),
                source_hashes=audit["source_hashes"], bank_id=audit["bank_id"],
                reference_hashes={p.name: sha(p) for p in sorted(REFERENCE.iterdir()) if p.is_file()},
                text_status=text_status,
                scope="Frozen 640-word controlled-experiment cohort; not the entire 50,455-word export.",
                status="EXPLORATORY_AFTER_PRIOR_RESULTS; no T1/T2/T3 used by this program")
    contract = digest(plan)
    root = cfg.output_parent / ("p22_contextual_" + contract[:12])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "PLAN.json").exists():
        require(read_json(root / "PLAN.json") == clean_json(plan), "Run contract collision.")
    write_json(root / "PLAN.json", plan)
    write_json(root / "BANK_AUDIT.json", audit)
    write_json(root / "TEXT_STATUS.json", text_status)
    if texts is not None:
        text_file = root / "canonical_texts.parquet"
        if not text_file.exists():
            texts.to_parquet(text_file, index=False)
        require(digest(pd.read_parquet(text_file).to_dict("records")) == text_status["selected_content_sha256"],
                "Cached text selection changed.")
    run = dict(root=str(root), bank=str(cfg.bank), contract_id=contract,
               config=clean_json(asdict(cfg)), words=d["words"], classes=audit["label_map"])
    write_json(root / "RUN.json", run)
    print(f"Authenticated {len(d['words'])} words and {len(d['O']):,} occurrences.\nOutput: {root}\nText: {text_status['status']}")
    return run


def unit_rows(x):
    norm = np.linalg.norm(x, axis=1)
    require((norm > 0).all(), "Zero vector in angular diagnostic.")
    return x / norm[:, None]


def connected_groups(meta, X, contexts=None):
    """Union source records, exact vectors, and (when present) identical texts."""
    n = len(X)
    parent = np.arange(n)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def link(values):
        seen = {}
        for i, v in enumerate(values):
            if v in seen:
                parent[root(i)] = root(seen[v])
            else:
                seen[v] = i
    link(meta.source_doc_key.astype(str))
    link(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in X)
    if contexts is not None:
        link(re.sub(r"\s+", " ", str(t)).strip().casefold() for t in contexts)
    return np.array([root(i) for i in range(n)], dtype=int)


def group_split(groups, seed, fraction):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    ntest = max(1, int(math.ceil(len(unique) * fraction)))
    test = np.flatnonzero(np.isin(groups, order[:ntest]))
    train = np.flatnonzero(~np.isin(groups, order[:ntest]))
    return train, test


def geometry_metrics(X, P, local_assignment, support, reference_A, reference_unit_means,
                     same_class_mask, own_word, topk):
    require(1 <= topk < len(reference_A), "Neighbourhood size must be smaller than the reference vocabulary.")
    n = len(X)
    a = X.mean(0)
    residual = X-a
    total = float(np.square(residual).sum()/n)
    within = float(np.square(X-P[local_assignment]).sum()/n)
    between = float(np.sum(support * np.square(P-a).sum(1))/n)
    require(np.isclose(total, within+between, rtol=1e-9, atol=1e-10), "Variance decomposition identity failed.")
    U = unit_rows(X)
    mean_u = U.mean(0)
    self_cosine = float((np.square(U.sum(0)).sum()-n)/(n*(n-1)))
    global_excluded = np.arange(len(reference_A)) != own_word
    other = float(mean_u @ reference_unit_means[global_excluded].mean(0))
    same = same_class_mask & global_excluded
    same_cos = float(mean_u @ reference_unit_means[same].mean(0))
    different_cos = float(mean_u @ reference_unit_means[~same_class_mask].mean(0))
    eig = np.maximum(np.linalg.eigvalsh(residual @ residual.T / n), 0)
    evr = float(eig[-1]/eig.sum()) if eig.sum() > 1e-20 else np.nan
    positive = eig[eig > 1e-20]
    entropy_rank = float(np.exp(-np.sum((positive/positive.sum())*np.log(positive/positive.sum())))) if len(positive) else 0.0
    targets = unit_rows(reference_A)
    scores_a = unit_rows(a[None])[0] @ targets.T
    scores_o = U @ targets.T
    scores_a[own_word] = -np.inf
    scores_o[:, own_word] = -np.inf
    order_a = np.argsort(-scores_a, kind="stable")[:topk]
    order_o = np.argsort(-scores_o, axis=1, kind="stable")[:, :topk]
    overlap = np.isin(order_o, order_a).sum(1)
    jaccard = overlap/(2*topk-overlap)
    unique_first = len(np.unique(order_o[:, 0]))
    # A -> 0 and O -> 1 are accounting endpoints, not predictive results.
    return dict(total_variance=total, within_P_variance=within, between_P_variance=between,
                frozen_P_retained_fraction=between/total if total > 1e-20 else np.nan,
                random_partition_expected_fraction=(len(P)-1)/(n-1),
                raw_self_cosine=self_cosine, other_word_cosine=other,
                adjusted_self_cosine=self_cosine-other,
                same_class_other_word_cosine=same_cos, different_class_cosine=different_cos,
                directional_variance=float(np.square(U-mean_u).sum()/n),
                norm_shrinkage=float(np.linalg.norm(a)/np.linalg.norm(X, axis=1).mean()),
                centered_PC1_fraction=evr, centered_effective_rank=entropy_rank,
                top_neighbor_change_rate=float(np.mean(order_o[:, 0] != order_a[0])),
                topk_jaccard_with_mean=float(jaccard.mean()), distinct_top_neighbors=unique_first,
                n_prototypes=len(P), min_frozen_support=int(support.min()))


def check_resources(run, deadline):
    root = Path(run["root"])
    if (root / "STOP").exists():
        raise TechnicalPause("STOP file detected; completed words remain reusable.")
    if time.time() >= deadline:
        raise TechnicalPause("Time budget reached; rerun to resume.")
    if psutil.virtual_memory().available / 2**30 < run["config"]["reserve_gib"]:
        raise TechnicalPause("RAM reserve reached; close other work and resume.")


def fit_centers(train, k, seed, cfg):
    if k == 1:
        return train.mean(0, keepdims=True), np.zeros(len(train), dtype=int), 0
    model = KMeans(n_clusters=k, random_state=seed, n_init=cfg["kmeans_n_init"],
                   max_iter=cfg["kmeans_max_iter"], algorithm="lloyd", tol=1e-4)
    model.fit(train)
    return model.cluster_centers_, model.labels_, int(model.n_iter_)


def nearest(queries, centers):
    distances = (np.square(queries).sum(1)[:, None] + np.square(centers).sum(1)[None, :]
                 - 2*queries @ centers.T)
    labels = np.argmin(distances, axis=1)
    errors = np.square(queries-centers[labels]).sum(1)
    return labels, errors


def loss(test, centers):
    return float(nearest(test, centers)[1].mean())


def heldout_probes(X, groups, word, cfg, guard=lambda: None):
    """Fresh centroids only. Frozen full-data A/P are never used on held-out rows."""
    rows, null_rows = [], []
    for rs in cfg["split_seeds"]:
        train_ids, test_ids = group_split(groups, seed_for("split", word, rs), cfg["test_group_fraction"])
        train_groups = np.unique(groups[train_ids])
        order = np.random.default_rng(seed_for("nested", word, rs)).permutation(train_groups)
        for fraction in cfg["training_fractions"]:
            selected = order[:max(1, int(math.ceil(len(order)*fraction)))]
            ids = train_ids[np.isin(groups[train_ids], selected)]
            train, test = X[ids], X[test_ids]
            base = dict(seed=int(rs), training_fraction=float(fraction), n_train=len(train), n_test=len(test),
                        n_train_groups=len(selected), n_test_groups=len(np.unique(groups[test_ids])))
            valid = (len(np.unique(groups)) >= cfg["min_source_groups"] and
                     len(train) >= cfg["min_train_occurrences"] and len(test) >= cfg["min_test_occurrences"])
            prepared = None
            if valid:
                guard()
                mean = train.mean(0)
                centered = train-mean
                _, singular, vt = np.linalg.svd(centered, full_matrices=False)
                rank = int(np.sum(singular > max(centered.shape)*np.finfo(float).eps*singular[0])) if singular[0] > 0 else 0
                if rank:
                    basis = vt[:rank]
                    train_z, test_z = centered @ basis.T, (test-mean) @ basis.T
                    outside = np.maximum(np.square(test-mean).sum(1)-np.square(test_z).sum(1), 0)
                    a_error = float(np.square(test-mean).sum(1).mean())
                    prepared = (mean, singular, rank, basis, train_z, test_z, outside, a_error)
            for k in cfg["probe_k"]:
                guard()
                info = dict(base, k=int(k))
                if not valid or len(train) < 2*k or len(np.unique(train, axis=0)) < k:
                    rows.append(dict(info, status="NOT_EVALUABLE", reason="Insufficient grouped train/test support"))
                    continue
                # Exact train-span reduction; no test fit and no variance truncation.
                if prepared is None:
                    rows.append(dict(info, status="NOT_EVALUABLE", reason="Zero training variance"))
                    continue
                mean, singular, rank, basis, train_z, test_z, outside, a_error = prepared
                if a_error <= 1e-20:
                    rows.append(dict(info,status="NOT_EVALUABLE",reason="Zero held-out baseline loss"))
                    continue
                centers, labels, iterations = fit_centers(train_z, k, seed_for("fit", word, rs, fraction, k), cfg)
                assign, squared = nearest(test_z, centers)
                k_error = float((squared+outside).mean())
                restored = mean + centers @ basis
                norms = np.linalg.norm(restored[assign], axis=1)
                angle_error = float(np.mean(1-np.sum(unit_rows(test)*unit_rows(restored[assign]), axis=1))) if (norms > 0).all() else np.nan
                gain = 1-k_error/a_error if a_error > 1e-20 else np.nan
                info.update(status="OK", A_test_mse=a_error, K_test_mse=k_error,
                            relative_gain=gain, angular_test_error=angle_error,
                            occupied_train_clusters=len(np.unique(labels)), min_train_cluster=int(np.bincount(labels, minlength=k).min()),
                            iterations=iterations, reached_iteration_limit=iterations >= cfg["kmeans_max_iter"],
                            train_span_rank=rank, test_outside_train_span_fraction=float(outside.mean()/a_error))
                if k == 1:
                    occurrence_error = loss(test_z, train_z)+float(outside.mean())
                    info.update(O_train_test_mse=occurrence_error, O_train_codebook_size=len(train),
                                O_train_gain=1-occurrence_error/a_error if a_error>1e-20 else np.nan)
                if k > 1 and fraction == 1.0:
                    counts = np.bincount(labels, minlength=k)
                    if (counts == 0).any():
                        info["control_status"] = "NOT_EVALUABLE_EMPTY_CLUSTER"
                    else:
                        random_gains, gaussian_gains = [], []
                        scales = singular[:rank]/np.sqrt(len(train)-1)
                        for repetition in range(cfg["control_repetitions"]):
                            guard()
                            rng = np.random.default_rng(seed_for("controls", word, rs, k, repetition))
                            permuted = rng.permutation(len(train_z))
                            chunks = np.split(permuted, np.cumsum(counts)[:-1])
                            random_centers = np.stack([train_z[g].mean(0) for g in chunks])
                            random_error = loss(test_z, random_centers)+float(outside.mean())
                            random_gain = 1-random_error/a_error
                            # Independent synthetic samples, covariance estimated ONLY from real TRAIN.
                            gaussian_train = rng.standard_normal((len(train), rank))*scales
                            gaussian_test = rng.standard_normal((len(test), rank))*scales
                            gc, _, _ = fit_centers(gaussian_train, k, seed_for("gaussian", word, rs, k, repetition), cfg)
                            ga = loss(gaussian_test, gaussian_train.mean(0, keepdims=True))
                            gaussian_gain = 1-loss(gaussian_test, gc)/ga if ga > 1e-20 else np.nan
                            random_gains.append(random_gain)
                            gaussian_gains.append(gaussian_gain)
                            null_rows.append(dict(seed=int(rs), k=int(k), repetition=repetition,
                                                  random_partition_gain=random_gain, gaussian_gain=gaussian_gain))
                        info.update(control_status="OK", random_gain_mean=float(np.mean(random_gains)),
                                    gaussian_gain_mean=float(np.mean(gaussian_gains)),
                                    gain_minus_random=float(gain-np.mean(random_gains)),
                                    gain_minus_gaussian=float(gain-np.mean(gaussian_gains)),
                                    gaussian_MC_se=float(np.std(gaussian_gains, ddof=1)/np.sqrt(len(gaussian_gains))))
                    # Two disjoint TRAIN group halves, common unseen TEST, fixed k.
                    half = len(order)//2
                    left = train_ids[np.isin(groups[train_ids], order[:half])]
                    right = train_ids[np.isin(groups[train_ids], order[half:])]
                    if min(len(left), len(right)) >= 2*k:
                        c1, _, _ = fit_centers(X[left], k, seed_for("stability1", word, rs, k), cfg)
                        c2, _, _ = fit_centers(X[right], k, seed_for("stability2", word, rs, k), cfg)
                        l1, _ = nearest(test, c1)
                        l2, _ = nearest(test, c2)
                        info.update(stability_occupied_left=len(np.unique(l1)), stability_occupied_right=len(np.unique(l2)))
                        if min(len(np.unique(l1)), len(np.unique(l2))) > 1:
                            info.update(stability_ARI=float(adjusted_rand_score(l1, l2)), stability_status="OK")
                        else:
                            info.update(stability_ARI=None, stability_status="DEGENERATE_ONE_TEST_CLUSTER")
                    else:
                        info["stability_status"] = "NOT_EVALUABLE_INSUFFICIENT_HALF_SUPPORT"
                rows.append(info)
    return rows, null_rows


def correlation(x, y):
    a, b = rankdata(x), rankdata(y)
    a, b = a-a.mean(), b-b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den > 0 else np.nan


def text_alignment(X, meta, contexts, word, groups, repetitions):
    """Context-lexical proxy, not human sense labels or a causal test."""
    if contexts is None:
        return dict(status="NOT_AVAILABLE")
    token = re.compile(r"(?<!\w)"+re.escape(word)+r"(?!\w)", flags=re.I)
    masked = [token.sub(" ", str(t)) for t in contexts]
    ids = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    if len(ids) < 20:
        return dict(status="NOT_EVALUABLE", reason="Fewer than 20 source/duplicate groups")
    # One row per connected group; remove repeated windows within a group.
    documents = [" ".join(dict.fromkeys(masked[j] for j in ix)) for ix in ids]
    vectors = np.stack([X[ix].mean(0) for ix in ids])
    if not (np.linalg.norm(vectors,axis=1) > 0).all():
        return dict(status="NOT_EVALUABLE", reason="Zero group-mean vector in angular text diagnostic")
    try:
        tfidf = TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, max_df=0.95,
                               ngram_range=(1, 1), sublinear_tf=True, max_features=5000).fit_transform(documents)
    except ValueError as e:
        return dict(status="NOT_EVALUABLE", reason=str(e))
    if tfidf.shape[1] < 5 or np.any(np.asarray(tfidf.power(2).sum(1)).ravel() == 0):
        return dict(status="NOT_EVALUABLE", reason="Insufficient nonempty lexical contexts")
    S = (tfidf @ tfidf.T).toarray()
    E = unit_rows(vectors) @ unit_rows(vectors).T
    ix = np.triu_indices(len(ids), 1)
    observed = correlation(E[ix], S[ix])
    if not np.isfinite(observed):
        return dict(status="NOT_EVALUABLE", reason="Constant pairwise similarity")
    rng = np.random.default_rng(seed_for("text", word))
    null = []
    for _ in range(repetitions):
        perm = rng.permutation(len(ids))
        null.append(correlation(E[ix], S[np.ix_(perm, perm)][ix]))
    return dict(status="OK", group_count=len(ids), vocabulary_size=tfidf.shape[1],
                spearman=observed, permutation_mean=float(np.nanmean(null)),
                rho_minus_null=float(observed-np.nanmean(null)),
                null_q025=float(np.nanquantile(null, .025)), null_q975=float(np.nanquantile(null, .975)),
                null_MC_se=float(np.nanstd(null, ddof=1)/np.sqrt(len(null))),
                interpretation="Association with lexical context; no independent sense validation; no pairwise-iid p-value")


def worker_init(run):
    # Applied before BLAS work; threadpoolctl also covers libraries already loaded.
    global _WORKER
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    limiter = threadpool_limits(limits=1)
    d = load_bank(run["bank"])
    d["groups"] = d["meta"].groupby("word", sort=False).indices
    textpath = Path(run["root"]) / "canonical_texts.parquet"
    texts = pd.read_parquet(textpath) if textpath.exists() else None
    if texts is not None:
        require(np.array_equal(texts.occ_id, d["meta"].occ_id), "Worker text/occurrence misalignment.")
    unit_means = np.stack([unit_rows(np.asarray(d["O"][d["groups"][w]])).mean(0) for w in d["words"]])
    _WORKER = dict(run=run, data=d, limiter=limiter, texts=texts, unit_means=unit_means,
                   labels=np.array([run["classes"][w] for w in d["words"]]))


def word_job(payload):
    word, deadline = payload
    run, d = _WORKER["run"], _WORKER["data"]
    start = time.perf_counter()
    guard = lambda: check_resources(run, deadline)
    guard()
    ids = d["groups"][word]
    m = d["meta"].iloc[ids].reset_index(drop=True)
    X = np.asarray(d["O"][ids], dtype=np.float64)
    ai = d["words"].index(word)
    pids = np.flatnonzero(d["units"].word.eq(word))
    pmap = {int(v): i for i, v in enumerate(pids)}
    assignment = np.array([pmap[int(v)] for v in m.p_index])
    P = np.asarray(d["P"][pids])
    support = d["units"].iloc[pids].support.to_numpy()
    contexts = _WORKER["texts"].iloc[ids].context.tolist() if _WORKER["texts"] is not None else None
    groups = connected_groups(m, X, contexts)
    g = geometry_metrics(X, P, assignment, support, np.asarray(d["A"]), _WORKER["unit_means"],
                         _WORKER["labels"] == run["classes"][word], ai, run["config"]["nearest_neighbors"])
    g.update(n_source_records=m.source_doc_key.nunique(), n_connected_groups=len(np.unique(groups)),
             duplicate_vector_rows=len(X)-len(np.unique(X, axis=0)))
    cv, null = heldout_probes(X, groups, word, run["config"], guard)
    guard()
    t = text_alignment(X, m, contexts, word, groups, run["config"]["text_permutations"])
    result = dict(word=word, category=run["classes"][word], contract_id=run["contract_id"],
                  geometry=g, probes=cv, controls=null, text=t, seconds=time.perf_counter()-start)
    path = Path(run["root"]) / "words" / (hashlib.sha256(word.encode()).hexdigest()[:20]+".json")
    write_json(path, result)
    write_json(path.with_suffix(".receipt.json"), dict(word=word, contract_id=run["contract_id"], sha256=sha(path)))
    return dict(word=word, seconds=result["seconds"])


def completed(run):
    result = {}
    for path in (Path(run["root"]) / "words").glob("*.receipt.json"):
        receipt = read_json(path)
        target = path.with_name(path.name.replace(".receipt.json", ".json"))
        require(receipt["contract_id"] == run["contract_id"] and receipt["word"] in run["words"], "Checkpoint contract mismatch.")
        require(target.is_file() and sha(target) == receipt["sha256"], "Incomplete or modified word checkpoint: " + receipt["word"])
        require(receipt["word"] not in result, "Duplicate word checkpoint.")
        result[receipt["word"]] = target
    return result


@contextlib.contextmanager
def run_lock(root):
    path = Path(root) / "RUNNING.lock"
    if path.exists():
        old = read_json(path)
        if old["host"] == socket.gethostname() and not psutil.pid_exists(old["pid"]):
            path.unlink()
        else:
            raise AuditError("Another run may be active. Do not run two notebooks against this output folder.")
    with path.open("x", encoding="utf-8") as f:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), f)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def capacity(cfg):
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    available = psutil.virtual_memory().available/2**30
    ram = int((available-cfg["reserve_gib"]-0.3)/cfg["worker_budget_gib"])
    return max(0, min(cfg["max_workers"], max(1, physical-1), ram))


def pilot_words(run):
    selected = []
    for category in sorted(set(run["classes"].values())):
        pool = [w for w in run["words"] if run["classes"][w] == category]
        selected.extend(sorted(pool, key=lambda w: seed_for("pilot", w))[:2])
    return selected


def run_analysis(run, pilot=False):
    """Pilot tasks are real full-configuration words and are reused in the full run."""
    root = Path(run["root"])
    cfg = run["config"]
    require(read_json(root/"PLAN.json")["code_sha256"] == sha(__file__), "Code changed: run prepare() again.")
    require(environment() == read_json(root/"PLAN.json")["environment"], "Environment changed: run prepare() again.")
    selected = pilot_words(run) if pilot else run["words"]
    ready = completed(run)
    todo = [w for w in selected if w not in ready]
    workers = capacity(cfg)
    require(workers > 0, "RAM reserve leaves no worker capacity; close other jobs and resume.")
    print(f"{'Pilot' if pilot else 'Full analysis'}: {len(todo)} pending words; {workers} processes; one BLAS thread per process.", flush=True)
    start, deadline = time.perf_counter(), time.time()+cfg["max_hours_per_call"]*3600
    records, reason = [], None
    if todo:
        with run_lock(root):
            # Spawn explicitly: same semantics in Windows and Linux; no notebook-local functions.
            import multiprocessing as mp
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=worker_init, initargs=(run,)) as pool:
                iterator = iter(todo)
                futures = {}
                for _ in range(min(workers, len(todo))):
                    w = next(iterator)
                    futures[pool.submit(word_job, (w, deadline))] = w
                while futures:
                    finished, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                    if not finished:
                        print(f"Running: {len(ready)+len(records)}/{len(run['words'])} words saved; "
                              f"available RAM {psutil.virtual_memory().available/2**30:.1f} GiB.", flush=True)
                    for future in finished:
                        w = futures.pop(future)
                        try:
                            records.append(future.result())
                        except TechnicalPause as exc:
                            reason = str(exc)
                        except Exception:
                            write_json(root/"ERROR.json", dict(word=w, traceback=traceback.format_exc()))
                            raise
                        if reason is None:
                            following = next(iterator, None)
                            if following is not None:
                                futures[pool.submit(word_job, (following, deadline))] = following
                    if finished:
                        print(f"Saved {len(ready)+len(records)}/{len(run['words'])} words.", flush=True)
    n_done = len(completed(run))
    elapsed = time.perf_counter()-start
    timing = dict(mode="pilot" if pilot else "full", status="COMPLETE" if n_done == len(run["words"]) else "PARTIAL",
                  completed_words=n_done, planned_words=len(run["words"]), new_words=len(records),
                  wall_seconds=elapsed, workers=workers, pause_reason=reason, word_times=records)
    if records:
        timing["estimated_remaining_hours_at_observed_throughput"] = (len(run["words"])-n_done)*elapsed/len(records)/3600
        timing["estimate_note"] = "Approximation from these words and current load, including startup; not a guarantee."
    write_json(root/("PILOT.json" if pilot else "PROGRESS.json"), timing)
    if n_done == len(run["words"]):
        write_json(root/"COMPUTATION_COMPLETE.json", dict(contract_id=run["contract_id"], words=n_done))
    return timing


def stratified_summary(df, metric, repetitions=2000, salt="summary"):
    """Average repeats within word before calling. Resample words, not pairs/folds."""
    data = df[["word", "category", metric]].dropna()
    require(not data.word.duplicated().any(), "Bootstrap input must have one row per word.")
    categories = sorted(df.category.unique())
    values = [data.loc[data.category.eq(c), metric].to_numpy(float) for c in categories]
    if not values or any(len(v) == 0 for v in values):
        return dict(metric=metric, status="NOT_EVALUABLE_MISSING_CATEGORY", n_words=len(data), n_categories=sum(len(v)>0 for v in values))
    point = float(np.mean([v.mean() for v in values]))
    rng = np.random.default_rng(seed_for(salt, metric))
    draws = np.zeros(repetitions)
    for v in values:
        draws += v[rng.integers(0, len(v), (repetitions, len(v)))].mean(1)/len(values)
    return dict(metric=metric, status="OK", estimate=point,
                bootstrap_low=float(np.quantile(draws, .025)), bootstrap_high=float(np.quantile(draws, .975)),
                n_words=len(data), n_categories=len(values),
                interval="95% within-category word-bootstrap; conditional, not independent-corpus inference")


def table(df, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder/(name+".csv"), index=False)
    df.to_latex(folder/(name+".tex"), index=False, escape=True, float_format=lambda x: f"{x:.5g}")


def summarize(run):
    root = Path(run["root"])
    files = completed(run)
    require(len(files) == len(run["words"]), "Analysis is incomplete. Resume before generating manuscript-ready summaries.")
    audit = read_json(root/"BANK_AUDIT.json")
    geometries, probes, controls, texts = [], [], [], []
    for word in run["words"]:
        r = read_json(files[word])
        key = dict(word=word, category=r["category"], in_common=word in audit["common_words"], in_full=word in audit["full_words"])
        geometries.append(dict(key, **r["geometry"]))
        probes.extend(dict(key, **p) for p in r["probes"])
        controls.extend(dict(key, **p) for p in r["controls"])
        texts.append(dict(key, **r["text"]))
    g, p, c, t = map(pd.DataFrame, (geometries, probes, controls, texts))
    out = root/"tables"
    for name, df in (("word_geometry",g),("heldout_probes",p),("controls",c),("text_alignment",t)):
        df.to_csv(root/(name+".csv"), index=False)
    summary = []
    for scope, mask in (("COHORT", np.ones(len(g), bool)), ("COMMON",g.in_common.to_numpy())):
        for metric in ("total_variance", "frozen_P_retained_fraction", "random_partition_expected_fraction",
                       "raw_self_cosine", "other_word_cosine", "adjusted_self_cosine", "directional_variance",
                       "norm_shrinkage", "centered_PC1_fraction", "centered_effective_rank",
                       "top_neighbor_change_rate", "topk_jaccard_with_mean"):
            summary.append(dict(scope=scope, **stratified_summary(g.loc[mask], metric, run["config"]["bootstrap_repetitions"])))
    table(pd.DataFrame(summary),out,"geometry_summary")
    # Include failed cells in the denominator; finite endpoints have explicit support.
    support = p.groupby(["category","training_fraction","k","status"],dropna=False).size().reset_index(name="word_split_cells")
    table(support,out,"probe_coverage")
    metric_cols = [v for v in ("relative_gain","gain_minus_random","gain_minus_gaussian","stability_ARI",
                              "gaussian_MC_se","angular_test_error","test_outside_train_span_fraction") if v in p]
    pp = p.loc[p.status.eq("OK")].groupby(["word","category","in_common","training_fraction","k"],as_index=False)[metric_cols].mean()
    if len(pp):
        repeat_summary = p.loc[p.status.eq("OK")].groupby(["word","training_fraction","k"],as_index=False).agg(
            n_evaluable_splits=("seed","nunique"), gain_split_sd=("relative_gain","std"),
            gain_split_min=("relative_gain","min"), gain_split_max=("relative_gain","max"))
        pp = pp.merge(repeat_summary,on=["word","training_fraction","k"],validate="one_to_one")
    # Reindex each cell onto all words to retain missing-category detection.
    ps = []
    for (fraction,k), part in pp.groupby(["training_fraction","k"]):
        complete = g[["word","category","in_common"]].merge(part.drop(columns=["category","in_common"]),on="word",how="left")
        for scope, subset in (("COHORT",complete),("COMMON",complete.loc[complete.in_common])):
            for metric in metric_cols:
                ps.append(dict(scope=scope,training_fraction=fraction,k=k,
                               **stratified_summary(subset,metric,run["config"]["bootstrap_repetitions"],f"{scope}:{fraction}:{k}")))
    ps_frame = pd.DataFrame(ps,columns=["scope","training_fraction","k","metric","status","estimate",
         "bootstrap_low","bootstrap_high","n_words","n_categories","interval"])
    table(ps_frame,out,"heldout_summary")
    table(pp,out,"word_averaged_probes")
    occurrence_summary=[]
    if "O_train_gain" in p:
        occurrence=p.loc[p.k.eq(1)&p.status.eq("OK")].groupby(
            ["word","category","in_common","training_fraction"],as_index=False)[["O_train_gain","O_train_codebook_size"]].mean()
        for fraction,part in occurrence.groupby("training_fraction"):
            complete=g[["word","category","in_common"]].merge(part.drop(columns=["category","in_common"]),on="word",how="left")
            for scope,subset in (("COHORT",complete),("COMMON",complete.loc[complete.in_common])):
                occurrence_summary.append(dict(scope=scope,training_fraction=fraction,
                    mean_codebook_size=subset.groupby("category").O_train_codebook_size.mean().mean(),
                    **stratified_summary(subset,"O_train_gain",run["config"]["bootstrap_repetitions"],f"O:{scope}:{fraction}")))
        table(occurrence,out,"word_occurrence_reference")
    table(pd.DataFrame(occurrence_summary,columns=["scope","training_fraction","mean_codebook_size","metric","status",
        "estimate","bootstrap_low","bootstrap_high","n_words","n_categories","interval"]),out,"occurrence_reference_summary")
    for status_column in ("control_status","stability_status"):
        if status_column in p:
            endpoint_coverage=p.loc[p.training_fraction.eq(1)&p.k.gt(1)].groupby(
                ["category","k",status_column],dropna=False).size().reset_index(name="word_split_cells")
            table(endpoint_coverage,out,status_column+"_coverage")
    table(g.groupby("category",as_index=False).agg(n_words=("word","size"),
        mean_retained=("frozen_P_retained_fraction","mean"),mean_directional_variance=("directional_variance","mean"),
        mean_neighbor_change=("top_neighbor_change_rate","mean"),minimum_source_groups=("n_connected_groups","min")),out,"category_geometry")
    loco = []
    for (fraction,k), part in pp.loc[pp.training_fraction.eq(1)].groupby(["training_fraction","k"]):
        for omitted in sorted(g.category.unique()):
            for metric in metric_cols:
                remaining = part.loc[part.category.ne(omitted)]
                means = remaining.groupby("category")[metric].mean()
                value = means.mean() if means.notna().sum() == len(g.category.unique())-1 else np.nan
                loco.append(dict(k=k,omitted_category=omitted,metric=metric,estimate=value,
                                 scope="COHORT",interpretation="Descriptive omission; no refitting or T1/T2/T3"))
    table(pd.DataFrame(loco,columns=["k","omitted_category","metric","estimate","scope","interpretation"]),out,"category_omissions")
    text_summary = []
    if "spearman" in t:
        for metric in ("spearman","rho_minus_null"):
            text_summary.append(stratified_summary(t,metric,run["config"]["bootstrap_repetitions"]))
    table(pd.DataFrame(text_summary,columns=["metric","status","estimate","bootstrap_low","bootstrap_high","n_words","n_categories","interval"]),out,"text_summary")
    m = load_bank(run["bank"])["meta"]
    record_word_count = m.groupby("source_doc_key").word.nunique()
    dependence = dict(source_records=len(record_word_count), shared_by_multiple_words=int((record_word_count>1).sum()),
                      maximum_words_per_source_record=int(record_word_count.max()),
                      full_document_independence="NOT_ESTABLISHED", bootstrap_limitation="Words may share source records; intervals are conditional sensitivity summaries.")
    write_json(root/"DEPENDENCE_AUDIT.json",dependence)
    # All null controls, unsuccessful fits, and individual words stay in the review bundle.
    write_json(root/"SUMMARY_COMPLETE.json",dict(contract_id=run["contract_id"], words=len(g),
               probe_status_counts=p.status.value_counts().to_dict(),text_status_counts=t.status.value_counts().to_dict(),
               inference="Exploratory; no validation of linguistic senses or proof of a retrieval mechanism"))
    write_report(run,g,p,t,dependence)
    return dict(geometry=g,probes=p,controls=c,text=t,geometry_summary=pd.DataFrame(summary),probe_summary=ps_frame)


def write_report(run,g,p,t,dependence):
    root = Path(run["root"])
    text = f"""# P22 exploratory contextual-representation report

Status: computational analysis complete for {len(g)} words. Contract: `{run['contract_id']}`.

This study characterizes the frozen P20/P21 cohort. It does not rerun memory retrieval,
fit the thesis classifier, select K, or use T1, T2, or T3. All comparisons are conditional
on this corpus, representation extraction, and lexical inventory.

## What to inspect

1. `tables/geometry_summary.csv`: contextual dispersion, anisotropy reference,
   prototype variance accounting, and changes in nearest lexical neighbours.
2. `tables/heldout_summary.csv`: every fixed probe-k and training fraction. Read
   relative compression gains jointly with random-partition and Gaussian controls.
3. `tables/probe_coverage.csv`: non-evaluable cells and their denominators.
4. `tables/category_omissions.csv`: category dependence without refitting.
5. `tables/text_summary.csv`: lexical-context association, when source texts exist.
6. `tables/occurrence_reference_summary.csv`: all TRAIN occurrences as a codebook,
   compared with the TRAIN mean; codebook capacities differ and are reported.

## Interpretation limits

The A=0 and O=1 endpoints for retained within-word variance follow by construction.
Frozen P is fitted on the same occurrences; its retained fraction is descriptive.
Held-out probes therefore refit all centroids on TRAIN records only. These diagnostic
K-means centroids are not the historical P induction algorithm or new retrieval arms.
Their test queries retain their full occurrence vectors to choose a nearest centroid;
this is a quantization test, not a deployment test with only a word identifier.
Within-word source-group hold-outs differ from the main word-disjoint retrieval
partitions and do not constitute additional retrieval trials.

A multi-centroid gain alone cannot establish distinct senses. A continuous unimodal
cloud can also benefit from quantization. The Gaussian control uses the full empirical
TRAIN covariance in its train span, with no shrinkage or added residual variance;
its contrast is sensitive to that finite-sample reference and is not a test of a
universal unimodality null. Inspect the test variance outside the train span as well.

Bootstrap intervals resample words within each of the fixed categories after averaging
repeated splits. They do not treat occurrence pairs, null draws, or repeated splits as
independent observations. They are conditional word-resampling summaries, not evidence
of replication in a new corpus. {dependence['shared_by_multiple_words']} source records
are shared by multiple cohort words; document independence is not established.

Text status counts: {json.dumps(t.status.value_counts().to_dict())}.
Text association uses masked-target TF-IDF and record/duplicate groups. It measures
lexical context association, not independent human sense annotation. No automatic
semantic interpretation or favourable conclusion is generated.

## Manuscript placement

Methods: cohort, authenticated identities, descriptive metrics, held-out protocol,
controls, weighting, missing-cell policy, resource/software record.
Results: select the most informative geometry and held-out figures; preserve null
or unfavourable findings and coverage. T1/T2 remain the main thesis outcomes.
Discussion: distinguish contextual variation, compression, linguistic interpretation,
and the separate controlled retrieval findings.
Appendix: all word-level results, grids, controls, omissions, and implementation details.
The approved abstract, Spanish summary, and symbols section need no automatic revision.
"""
    (root/"REPORT.md").write_text(text,encoding="utf-8")


def make_figures(run):
    import matplotlib
    matplotlib.use("Agg")
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
        for ext in ("png","pdf","svg"):
            fig.savefig(out/(name+"."+ext),dpi=220,bbox_inches="tight")
        plt.close(fig)
    categories = sorted(g.category.unique())
    colors = ["#187F91","#AB2DAC","#53677F"]
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].boxplot([g.loc[g.category.eq(c),"directional_variance"].dropna() for c in categories],tick_labels=categories,showfliers=False)
    axs[0].set(ylabel="Mean squared deviation of unit directions",title="A. Within-word directional variation")
    axs[0].tick_params(axis="x",rotation=45)
    for metric,label,color in [("raw_self_cosine","Same word",colors[0]),("other_word_cosine","Other words",colors[2])]:
        values=np.sort(g[metric].dropna())
        axs[1].plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color)
    axs[1].set(xlabel="Mean cosine similarity",ylabel="Fraction of words",title="B. Similarity with an anisotropy reference")
    axs[1].legend(frameon=False)
    save(fig,"01_contextual_geometry")
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].scatter(g.n_prototypes,g.frozen_P_retained_fraction,s=12,alpha=.4,color=colors[0])
    ks=np.sort(g.n_prototypes.unique())
    axs[0].plot(ks,(ks-1)/99,"--",color=colors[2],label="Random membership expectation")
    axs[0].set(xlabel="Frozen prototypes per word",ylabel="Between-prototype / total variance",title="A. Descriptive variance accounting")
    axs[0].legend(frameon=False)
    axs[1].boxplot([g.loc[g.category.eq(c),"top_neighbor_change_rate"] for c in categories],tick_labels=categories,showfliers=False)
    axs[1].set(ylabel="Occurrence vs. mean: top-neighbour change rate",title="B. Changes in lexical neighbourhood")
    axs[1].tick_params(axis="x",rotation=45)
    save(fig,"02_aggregation_and_neighbours")
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,metric,title in zip(axs,["relative_gain","gain_minus_random","gain_minus_gaussian"],
                             ["A. Held-out compression gain","B. Gain beyond random partitions","C. Gain beyond Gaussian reference"]):
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
    axs[0].set(xlabel="Fraction of available training groups",ylabel="Held-out relative MSE reduction",title="A. Sampling sensitivity")
    axs[0].legend(frameon=False)
    q=summary.loc[summary.scope.eq("COHORT")&summary.training_fraction.eq(1)&summary.metric.eq("stability_ARI")&summary.status.eq("OK")].sort_values("k")
    if len(q):
        axs[1].plot(q.k,q.estimate,"o-",color=colors[1]);axs[1].fill_between(q.k,q.bootstrap_low,q.bootstrap_high,color=colors[1],alpha=.15)
    axs[1].axhline(0,color="grey",lw=.8)
    axs[1].set(xlabel="Diagnostic codebook size k",ylabel="Adjusted Rand index on shared held-out rows",title="B. Stability across disjoint training halves")
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


def export_review(run):
    root=Path(run["root"])
    require((root/"SUMMARY_COMPLETE.json").exists(),"Complete and summarize the analysis before exporting.")
    target=root.parent/(root.name+"_Review.zip")
    allowed=[]
    for p in root.rglob("*"):
        if p.is_file() and p.name not in ("canonical_texts.parquet","RUNNING.lock") and p.suffix in (".json",".csv",".tex",".md",".png",".pdf",".svg"):
            allowed.append(p)
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(allowed):z.write(p,p.relative_to(root))
        for p in (HERE/"p22_contextual.py",HERE/"METHODS.md",HERE/"references.bib"):
            z.write(p,"code/"+p.name)
    print(f"Review bundle: {target} ({target.stat().st_size/2**20:.1f} MiB). Raw embeddings and source texts are excluded.")
    return target


def show_status(run):
    count=len(completed(run))
    return dict(completed_words=count,planned_words=len(run["words"]),remaining_words=len(run["words"])-count,
                text_status=read_json(Path(run["root"])/"TEXT_STATUS.json")["status"],output=run["root"])


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,default=Path.home()/"E6R/p20_68682213c891")
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--no-text",action="store_true")
    args=parser.parse_args()
    cfg=Config(bank=args.bank,max_workers=args.workers,use_text=not args.no_text)
    run=prepare(cfg)
    run_analysis(run,pilot=True)
    run_analysis(run)
    if show_status(run)["remaining_words"]==0:
        summarize(run);make_figures(run);export_review(run)
