# Paste in the STATIC notebook after RESULTS["glove"] is available.
# Optional: set GLOVE_FILE to your original GloVe50 .txt/.pkl file.
# Alternatively set GLOVE_VECTORS to an already loaded embedding DataFrame.
from pathlib import Path
import json, hashlib, importlib.util, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, Image


def make_static_pair_figures(result, module=None, embedding_file=None, vectors=None):
    ctx = result.context
    if ctx.get("K") != 40 or len(ctx["labels"]) != 8:
        raise ValueError("This cell is for the audited eight-class GloVe50 K40 result.")
    project = Path(ctx["config"].project_root)
    if module is None or not hasattr(module, "prepare_classification_split"):
        path = project / "exp2_pipeline_integrated.py"
        spec = importlib.util.spec_from_file_location("_static_pair_reader", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    folder = ctx.get("folder")
    geo = Path(folder) if folder is not None else Path(ctx["geometry_root"])/"K40"/"geometry"
    receipt_path = geo / "geometry_identity.json"
    if not receipt_path.exists():
        receipt_path = geo / "completed.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw_neighbors = (geo / "neighbors.json").read_bytes()
    if hashlib.sha256(raw_neighbors).hexdigest() != receipt["outputs"]["neighbors.json"]:
        raise RuntimeError("Catalogue checksum mismatch.")
    neighbors = json.loads(raw_neighbors)
    labels = list(neighbors)
    if labels != list(ctx["labels"]):
        raise ValueError("Catalogue class order differs from the static result.")
    owner = {w: i for i, values in enumerate(neighbors.values()) for w in values}
    words = sorted(owner)
    if len(words) != sum(map(len, neighbors.values())):
        raise ValueError("Overlapping catalogue classes.")

    # Read only the required words from TXT, keeping RAM use small.
    source = "Provided in-memory DataFrame"
    if vectors is None:
        explicit = embedding_file or ctx.get("paths", {}).get("embedding_path")
        candidates = [Path(explicit)] if explicit is not None else []
        if explicit is None:
            for base in dict.fromkeys([project, project.parent, project.parent/"Data"]):
                for pattern in ("glove.6B.50d.txt", "*/glove.6B.50d.txt", "*/*/glove.6B.50d.txt"):
                    candidates.extend(sorted(base.glob(pattern)))
        candidates = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
        if not candidates:
            raise FileNotFoundError(
                "GloVe50 vectors were not found. Set GLOVE_FILE to the original "
                "glove.6B.50d.txt or its embedding DataFrame .pkl, then rerun this cell."
            )
        path = candidates[0]
        source = str(path)
        print("Reading vectors:", path)
        if path.suffix.lower() in (".pkl", ".pickle"):
            vectors = pd.read_pickle(path)
        elif path.suffix.lower() == ".txt":
            wanted, rows = set(words), {}
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    token, sep, values = line.rstrip().partition(" ")
                    if token in wanted:
                        if token in rows:
                            raise ValueError("Duplicate embedding word: " + token)
                        rows[token] = np.fromstring(values, sep=" ", dtype=float)
            vectors = pd.DataFrame.from_dict(rows, orient="index")
        else:
            raise ValueError("Use an embedding DataFrame .pkl or the original GloVe50 .txt.")
    if not isinstance(vectors, pd.DataFrame) or vectors.index.has_duplicates:
        raise ValueError("Expected a uniquely indexed embedding DataFrame.")
    missing = sorted(set(words) - set(vectors.index))
    if missing:
        raise ValueError("Catalogue words missing from the embedding file: " + str(missing[:15]))
    embeddings = vectors.loc[words].astype(float).copy()
    if embeddings.shape[1] != 50 or not np.isfinite(embeddings.values).all():
        raise ValueError("Expected finite, 50-dimensional GloVe embeddings.")

    expected = ctx["expected"].copy()
    if expected.duplicated(["split_seed", "word"]).any() or set(expected.split_seed) != set(range(51)):
        raise ValueError("The saved validation inventory is not the complete 51-split experiment.")
    records = []
    for seed in range(51):
        split = module.prepare_classification_split(
            embeddings, neighbors, metric="cosine", frequency_threshold=2,
            validation_split=0.33, random_seed_split=seed,
            pairwise_different_class=True, debug=False,
        )
        observed = pd.DataFrame({
            "word": split.validation_words, "true_class": split.true_classes,
            "validation_similarity": np.clip(split.validation_similarities, -1, 1),
        }).sort_values("word").reset_index(drop=True)
        saved = expected.loc[expected.split_seed.eq(seed), observed.columns].sort_values("word").reset_index(drop=True)
        pd.testing.assert_frame_equal(observed, saved, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
        mapping = {w: (v, 1-float(distance)) for w, v, distance in split.nearest_neighbor_pairs}
        train = set(split.train_words)
        for word in split.validation_words:
            neighbor, similarity = mapping[word]
            if owner[word] == owner[neighbor] or neighbor not in train:
                raise ValueError("Invalid class restriction or missing TRAIN neighbor.")
            records.append(dict(split_seed=seed, query_word=word,
                query_class=labels[owner[word]], neighbor_word=neighbor,
                neighbor_class=labels[owner[neighbor]], cosine_similarity=similarity))
        if seed == 0 or (seed + 1) % 10 == 0 or seed == 50:
            print(f"Verified split reconstructions: {seed + 1}/51", flush=True)

    pairs = pd.DataFrame(records)
    unique = pairs.drop_duplicates("query_word").drop(columns="split_seed").copy()
    if pairs.groupby("query_word").neighbor_word.nunique().ne(1).any():
        raise ValueError("The selected neighbor varies across splits.")
    unique["split_exposures"] = unique.query_word.map(pairs.groupby("query_word").size())
    counts = pd.crosstab(pairs.query_class, pairs.neighbor_class).reindex(index=labels, columns=labels, fill_value=0)
    percentages = 100 * counts.div(counts.sum(axis=1), axis=0)
    same = pairs.query_class.eq(pairs.neighbor_class)
    table_hash = hashlib.sha256(pairs.to_csv(index=False).encode()).hexdigest()
    out = Path.home()/"E2_Contrasts"/"static_K40_pairs"/table_hash[:12]
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out/"pair_exposures.csv", index=False)
    unique.to_csv(out/"unique_query_pairs.csv", index=False)
    counts.to_csv(out/"pair_class_counts.csv")
    percentages.to_csv(out/"pair_class_percentages.csv")

    cmap = LinearSegmentedColormap.from_list("thesis", ["#6A0DAD", "#B755C1", "#D890D1", "#F2C2A3", "#F9E2B7", "#FFFF00"])
    with plt.rc_context({"font.size": 11, "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(8.4, 7), constrained_layout=True)
        im = ax.imshow(percentages.values, cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks(range(8), labels, rotation=45, ha="right")
        ax.set_yticks(range(8), labels)
        ax.set(title="Proposed static experiment", xlabel="Nearest-neighbor class", ylabel="Query class")
        for i in range(8):
            for j in range(8):
                value = percentages.iloc[i, j]
                if value > 0 or i == j:
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                            color="black" if value > 45 else "white", fontsize=10)
        fig.colorbar(im, ax=ax, label="Percentage within each query class")
        for ext in ("png", "pdf"):
            fig.savefig(out/("static_pair_composition."+ext), dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    display(Image(filename=str(out/"static_pair_composition.png"), width=1000))

    metrics = pd.DataFrame({"Diagnostic": ["Splits", "Query-split observations", "Distinct validation words",
        "Same-class pairs (%)", "Different-class pairs (%)", "Missing TRAIN neighbors", "Verified split reconstructions"],
        "Value": [51, len(pairs), len(unique), round(100*same.mean(), 2), round(100*(~same).mean(), 2), 0, 51]})
    display(metrics.set_index("Diagnostic"))
    metrics.to_csv(out/"pairing_metrics.csv", index=False)

    # Two examples per class: lowest and highest cosine similarity.
    # Selection uses geometry only, never classification errors.
    examples = []
    for label in labels:
        group = unique.loc[unique.query_class.eq(label)].sort_values(["cosine_similarity", "query_word"])
        examples.extend([group.iloc[0].to_dict(), group.iloc[-1].to_dict()])
    examples = pd.DataFrame(examples).drop_duplicates("query_word")
    examples.to_csv(out/"pair_examples.csv", index=False)
    print("Examples: lowest and highest similarity in each query class.")
    display(examples.round({"cosine_similarity": 4}).rename(columns={"query_word":"Query word", "query_class":"Query class",
        "neighbor_word":"Nearest neighbor", "neighbor_class":"Neighbor class", "cosine_similarity":"Cosine similarity",
        "split_exposures":"Split exposures"}).reset_index(drop=True))
    (out/"provenance.json").write_text(json.dumps(dict(
        dataset="GloVe50", K=40, condition="RAW", source_geometry=str(geo), embedding_source=source,
        pairwise_different_class=True, scope="Full evaluated query inventory, before COMMON restriction",
        denominator="Query-split observations; rows normalized independently",
        reconstructed_splits=51, similarity_tolerance=1e-12, classifier_training_performed=False,
        table_sha256=table_hash, module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        embedding_subset_sha256=hashlib.sha256(embeddings.to_numpy(dtype='<f8').tobytes()).hexdigest(),
        embedding_word_order=words, labels=labels), indent=2), encoding="utf-8")
    print("Saved figures and tables:", out)
    return pairs, out


if "RESULTS" not in globals() or "glove" not in RESULTS:
    raise RuntimeError('Load RESULTS["glove"] in the static notebook first.')
_pair_module = globals().get("static_module") or globals().get("exp")
STATIC_PAIR_EXPOSURES, STATIC_PAIR_OUTPUT = make_static_pair_figures(
    RESULTS["glove"], module=_pair_module,
    embedding_file=globals().get("GLOVE_FILE"), vectors=globals().get("GLOVE_VECTORS"),
)
