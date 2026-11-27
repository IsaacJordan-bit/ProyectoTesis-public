"""Recopila artefactos existentes; no genera embeddings ni ejecuta notebooks.

Solo necesita Python 3.11+. PyArrow es opcional para inspeccionar el Parquet
y extraer una muestra de las 24 palabras del experimento.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

METADATA_NAMES = {
    "common_pool_eligible_vocabulary_min100.csv",
    "eligible_vocabulary_min100.csv", "selection_report.json",
    "contextual_dataset_meta.json", "generation_config.json",
    "validation_report.json", "tokenizer_alignment_audit.json",
    "word_row_group_index.csv", "word_row_group_index_report.json",
    "word_counts.csv", "category_words_coverage.csv",
    "exp2_metadata_words_coverage.csv", "sample_contexts_by_category_word.csv",
    "contextual_exp2_vocabulary_wordnet_v1.csv",
    "contextual_exp2_vocabulary_wordnet_v1_audit.csv",
    "contextual_exp2_input_manifest.json",
}
PROTOTYPE_STEM = "contextual_exp2_embeddings_rawmean100_wordnet_v1"
LARGE_NAMES = {
    "contextual_occurrences_definitive.parquet",
    "_contextual_occurrences_definitive.partial.parquet",
    "contextual_occurrence_candidates_sorted.parquet",
    "_contextual_occurrence_candidates_unsorted.parquet",
    "selected_context_hashes_top100.parquet",
}
WORDS = sorted({
    "apple", "olive", "pear", "bicycle", "parachute", "scooter",
    "bread", "cheese", "peanut", "canoe", "bus", "raft",
    "chair", "curtain", "stool", "jellyfish", "ant", "crab",
    "screwdriver", "sling", "wrench", "skunk", "frog", "raccoon",
})
SKIP_DIRS = {".git", ".venv", "venv", ".venvs", "__pycache__",
             "node_modules", ".ipynb_checkpoints", ".cache", "_duckdb_tmp"}


def compatible(path: Path) -> str:
    """Compatibilidad con rutas largas de Windows, también al crear el ZIP."""
    value = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        if value.startswith("\\\\"):
            return "\\\\?\\UNC\\" + value[2:]
        return "\\\\?\\" + value
    return value


def is_reference(name: str) -> bool:
    stem = Path(name).stem
    return name.endswith(".ipynb") and any(
        stem == x or stem.startswith(x + "(")
        for x in ("corpus_audit", "contextual_def_generation")
    )


def discover(root: Path, warnings: list[str]) -> list[Path]:
    found = []
    def walk_error(exc):
        warnings.append(f"No se pudo inspeccionar una carpeta: {exc}")
    for parent, dirs, names in os.walk(compatible(root), onerror=walk_error,
                                       followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and not os.path.islink(os.path.join(parent, d)))
        for name in sorted(names):
            # Los informes con nombres genéricos solo se toman de la carpeta
            # contextual: otras ramas de la tesis también tienen validation_report.
            if name in {"selection_report.json", "validation_report.json",
                        "generation_config.json", "word_counts.csv"}:
                if Path(parent).name != "contextual_dataset":
                    continue
            if (name in METADATA_NAMES or name in LARGE_NAMES
                or name in {PROTOTYPE_STEM + ".pkl", PROTOTYPE_STEM + ".parquet"}
                or is_reference(name)):
                p = Path(parent) / name
                if not p.is_symlink():
                    found.append(p)
    return found


def safe_relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(compatible(path), compatible(root))).as_posix()


def parquet_details(path: Path) -> dict:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(compatible(path))
    try:
        return {"rows": pf.metadata.num_rows,
                "row_groups": pf.metadata.num_row_groups,
                "schema": str(pf.schema_arrow)}
    finally:
        pf.close()


def sample_words(parquet: Path, index: Path, destination: Path) -> dict:
    """Lee únicamente los row groups indicados por el índice existente."""
    import csv
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    groups = set()
    present = set()
    with open(compatible(index), encoding="utf-8-sig", newline="") as src:
        for row in csv.DictReader(src):
            word = row.get("word")
            if word in WORDS:
                if word in present:
                    raise ValueError(f"Palabra duplicada en el índice: {word}")
                present.add(word)
                groups.add(int(row["row_group_id"]))
    if present != set(WORDS):
        raise ValueError(f"Faltan palabras en el índice: {sorted(set(WORDS) - present)}")

    pf = pq.ParquetFile(compatible(parquet))
    pieces = []
    try:
        for group in sorted(groups):
            if group < 0 or group >= pf.metadata.num_row_groups:
                raise ValueError(f"Row group inválido: {group}")
            if pf.metadata.row_group(group).num_rows > 10_000:
                raise ValueError("Row group inesperadamente grande; se omite la muestra.")
            table = pf.read_row_group(group)
            pieces.append(table.filter(pc.is_in(table["word"], value_set=pa.array(WORDS))))
        sample = pa.concat_tables(pieces)
    finally:
        pf.close()
    counts = Counter(sample["word"].to_pylist())
    if counts != Counter({w: 100 for w in WORDS}):
        raise ValueError(f"La muestra no tiene 100 ocurrencias por palabra: {dict(counts)}")
    sample = sample.sort_by([("word", "ascending"), ("occ_id", "ascending")])
    pq.write_table(sample, compatible(destination), compression="zstd")
    return {"n_words": len(WORDS), "n_rows": sample.num_rows,
            "words": WORDS, "kind": "subset_only_not_full_dataset",
            "source_row_groups": sorted(groups)}


def add_file(z: zipfile.ZipFile, source: Path, name: str) -> str:
    """Calcula SHA256 sobre los mismos bytes que escribe en el ZIP."""
    h = hashlib.sha256()
    before = os.stat(compatible(source))
    info = zipfile.ZipInfo(name, date_time=datetime.now().timetuple()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    with open(compatible(source), "rb") as src, z.open(info, "w", force_zip64=True) as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            dst.write(block)
            h.update(block)
    after = os.stat(compatible(source))
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"El archivo cambió durante la copia: {source}")
    return h.hexdigest()


def collect(root: Path, output: Path, include_hashes=False,
            make_samples=True, metadata_only=False) -> dict:
    root = root.expanduser().resolve()
    output = output.expanduser().absolute()
    if not os.path.isdir(compatible(root)):
        raise FileNotFoundError(f"No existe la raíz: {root}")
    if os.path.exists(compatible(output)):
        raise FileExistsError(f"El ZIP ya existe; usa otra ruta --salida: {output}")
    os.makedirs(compatible(output.parent), exist_ok=True)
    warnings = []
    found = discover(root, warnings)
    if not found:
        raise FileNotFoundError("No se encontraron artefactos contextuales. Revisa --raiz.")
    print(f"Encontrados {len(found)} artefactos relevantes.", flush=True)
    entries, selected = [], []
    entry_by_path = {}
    for path in found:
        name = path.name
        relative = safe_relative(path, root)
        entry = {"source_relative": relative,
                 "size_bytes": os.stat(compatible(path)).st_size,
                 "included": False}
        entry_by_path[path] = entry
        entries.append(entry)
        if name in METADATA_NAMES or is_reference(name):
            if entry["size_bytes"] <= 32 * 1024**2:
                selected.append(path)
            else:
                entry["omitted_reason"] = "metadata_exceeds_32_MiB"
                warnings.append(f"Metadato demasiado grande, no incluido: {relative}")
        elif name == PROTOTYPE_STEM + ".pkl":
            if not metadata_only:
                selected.append(path)
            else:
                entry["omitted_reason"] = "metadata_only_requested"
        elif name == PROTOTYPE_STEM + ".parquet":
            if metadata_only:
                entry["omitted_reason"] = "metadata_only_requested"
            elif (path.parent / (PROTOTYPE_STEM + ".pkl")) in found:
                entry["omitted_reason"] = "pickle_preferred_equivalence_not_yet_verified"
            else:
                selected.append(path)
        elif name == "selected_context_hashes_top100.parquet" and include_hashes:
            selected.append(path)
        else:
            entry["omitted_reason"] = "large_file_remains_on_user_machine"
        if name == "contextual_occurrences_definitive.parquet":
            try:
                entry["parquet_footer"] = parquet_details(path)
            except Exception as exc:
                warnings.append(f"No se inspeccionó el footer de {relative}: {exc}")

    total = sum(entry_by_path[p]["size_bytes"] for p in selected)
    print(f"Para comprimir: {len(selected)} archivos, {total / 1024**2:.1f} MiB.", flush=True)
    print("El Parquet completo y los candidatos permanecen en su ubicación.", flush=True)
    if total > 450 * 1024**2:
        print("Aviso: recopilación grande. --solo-metadatos permite una entrega más pequeña.", flush=True)
    environment = {"python": sys.version, "packages": {}}
    for name in ["numpy", "pandas", "pyarrow", "torch", "transformers",
                 "datasets", "nltk", "duckdb", "scikit-learn"]:
        try:
            environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment["packages"][name] = None
    inventory = {
        "status": "collection_only_not_scientific_validation",
        "source_root": str(root), "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment_at_collection_not_historical": environment,
        "files": entries, "samples": [], "warnings": warnings,
        "notebooks_executed": False, "bert_executed": False,
        "full_occurrence_dataset_included": False,
    }
    names = {p.name for p in found}
    inventory["missing_key_artifacts"] = sorted({
        "common_pool_eligible_vocabulary_min100.csv",
        "eligible_vocabulary_min100.csv", "word_row_group_index.csv",
        "selection_report.json", "validation_report.json", "generation_config.json",
        "contextual_dataset_meta.json", "tokenizer_alignment_audit.json",
        "contextual_exp2_vocabulary_wordnet_v1.csv",
        "contextual_exp2_vocabulary_wordnet_v1_audit.csv",
        "contextual_exp2_input_manifest.json",
        "contextual_occurrences_definitive.parquet",
    } - names)
    if not any(n in names for n in [PROTOTYPE_STEM + ".pkl", PROTOTYPE_STEM + ".parquet"]):
        inventory["missing_key_artifacts"].append(PROTOTYPE_STEM + ".pkl OR .parquet")
    # Escritura exclusiva para no reemplazar una recopilación anterior.
    partial = Path(str(output) + ".partial")
    with tempfile.TemporaryDirectory(prefix="ctx_review_") as temp:
        with zipfile.ZipFile(compatible(partial), "x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=3, allowZip64=True) as z:
            for number, path in enumerate(selected, 1):
                entry = entry_by_path[path]
                arc = "archivos/" + entry["source_relative"]
                entry["sha256"] = add_file(z, path, arc)
                entry["archive_path"] = arc
                entry["included"] = True
                print(f"Comprimido {number}/{len(selected)}: {path.name}", flush=True)
            if make_samples:
                for i, path in enumerate(found):
                    if path.name != "contextual_occurrences_definitive.parquet":
                        continue
                    index = path.parent / "word_row_group_index.csv"
                    if index not in found:
                        warnings.append(f"Sin índice junto a {safe_relative(path, root)}; muestra omitida.")
                        continue
                    try:
                        sample_path = Path(temp) / f"muestra_24_palabras_{i:03d}.parquet"
                        details = sample_words(path, index, sample_path)
                        arc = "muestras/" + sample_path.name
                        details.update(source_relative=safe_relative(path, root),
                                       archive_path=arc,
                                       sha256=add_file(z, sample_path, arc))
                        inventory["samples"].append(details)
                        print(f"Muestra incluida: {details['n_rows']} filas, sin ejecutar BERT.", flush=True)
                    except Exception as exc:
                        warnings.append(f"Muestra omitida de {safe_relative(path, root)}: {exc}")
            z.writestr("INVENTARIO.json", json.dumps(inventory, indent=2, ensure_ascii=False))
    os.rename(compatible(partial), compatible(output))
    print(f"\nArchivo para adjuntar: {output}", flush=True)
    print(f"Faltantes: {len(inventory['missing_key_artifacts'])}; advertencias: {len(warnings)}.")
    print("Detalles en INVENTARIO.json. No se ejecutó el experimento.")
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", required=True, type=Path,
                        help="Raíz original Tesis que contiene Exp2/Results.")
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--incluir-hashes", action="store_true",
                        help="Incluye también el manifiesto grande de contextos seleccionados.")
    parser.add_argument("--sin-muestra", action="store_true")
    parser.add_argument("--solo-metadatos", action="store_true",
                        help="No incluye las matrices de prototipos; tendrán que enviarse después.")
    args = parser.parse_args()
    output = args.salida or (Path.cwd() / (
        "PendientesContextual_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".zip"))
    try:
        collect(args.raiz, output, args.incluir_hashes,
                not args.sin_muestra, args.solo_metadatos)
    except Exception as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
