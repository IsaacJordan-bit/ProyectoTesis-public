#!/usr/bin/env python3
"""Localiza resultados existentes y prepara sus rutas; no importa ML ni ajusta modelos."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
EXPECTED = {
    "glove": {
        "model_root": "search_b77d2b37e15428d6bf13",
        "refit_root": "refit_addb067bc5bf230730d5",
    },
    "prototypes": {
        "audit_root": "audit_3ee527e0fa4f7cfa423f",
        "model_root": "search_9720a6f271ceb1f336ae",
        "refit_root": "refit_9639c317f3f49105a694",
        "closure_root": "close_3267a23756b30d371b80",
    },
    "w2v": {  # Sólo sensibilidad K50 del módulo anterior; K40 está incluido en el ZIP.
        "audit_root": "audit_f4766816e364f73e7b47",
        "model_root": "search_8ab640f6ba94317eb9ec",
        "refit_root": "refit_3d35ee4f9e55880d936a",
    },
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def scan(root):
    matches = {key: {role: [] for role in roles} for key, roles in EXPECTED.items()}
    matches["glove"].update(gk83_root=[], geometry_search_root=[])
    by_name = {}
    protocols = []
    warnings = []
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__"}
    for folder, directories, files in os.walk(root, onerror=lambda e: warnings.append(str(e))):
        directories[:] = [n for n in directories if n not in skip]
        p = Path(folder)
        by_name.setdefault(p.name, []).append(p)
        if "protocol_frozen.json" in files:
            try:
                protocols.append((p, read_json(p / "protocol_frozen.json")))
            except (OSError, ValueError) as exc:
                warnings.append(str(exc))
    for key, roles in EXPECTED.items():
        for role, name in roles.items():
            matches[key][role] = by_name.get(name, [])
    for p, contract in protocols:
        if (contract.get("version") == "GK83G_TRAIN_ONLY_NEFF_SCALE_FULL98_V1"
                and contract.get("dataset") == "glove.6B.50d" and contract.get("K") == 40
                and p.name == "scale_" + digest(contract)[:20]):
            matches["glove"]["gk83_root"].append(p)
    if len(matches["glove"]["gk83_root"]) == 1:
        source_id = read_json(matches["glove"]["gk83_root"][0] / "protocol_frozen.json")["source_search_id"]
        matches["glove"]["geometry_search_root"] = [p for p, d in protocols if digest(d) == source_id]
    return matches, warnings


def inventory(matches):
    rows = []
    for dataset, roles in matches.items():
        for role, paths in roles.items():
            row = dict(dataset=dataset, role=role, status="ENCONTRADO" if len(paths) == 1 else
                       "AUSENTE" if not paths else "AMBIGUO", paths=[str(p) for p in paths])
            row["alcance"] = "sensibilidad_K50_opcional" if dataset == "w2v" else "estudio_principal"
            if len(paths) == 1:
                path = paths[0]
                if role == "model_root":
                    row["checkpoints_CV"] = sum(1 for _ in (path / "checkpoints").rglob("completed.json"))
                    row["seleccion_congelada"] = (path / "model_selection_frozen.json").is_file()
                    row["expected_checkpoints_CV"] = 3825
                if role == "refit_root":
                    row["refits_FULL_LOCO"] = sum(1 for _ in (path / "fits").rglob("completed.json"))
                    row["expected_refits_FULL_LOCO"] = 27
                if role == "audit_root":
                    row["K_status"] = (path / "K_status.csv").is_file()
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, required=True, help="Carpeta Exp2def o padre que conserva Results")
    parser.add_argument("--guardar", type=Path, default=REPO / "rutas_resultados.json")
    parser.add_argument("--informe", type=Path, default=REPO / "InventarioResultados.json")
    args = parser.parse_args()
    root = args.raiz.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"La carpeta no existe: {root}")
    matches, warnings = scan(root)
    rows = inventory(matches)
    previous = read_json(args.guardar) if args.guardar.is_file() else {}
    config = dict(previous)
    config.update(project_root=str(root), overrides=dict(previous.get("overrides", {})))
    for dataset, roles in matches.items():
        target = dict(config["overrides"].get(dataset, {}))
        for role, paths in roles.items():
            if len(paths) == 1:
                target[role] = str(paths[0])
        config["overrides"][dataset] = target
    args.guardar.parent.mkdir(parents=True, exist_ok=True)
    args.guardar.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.informe.parent.mkdir(parents=True, exist_ok=True)
    args.informe.write_text(json.dumps(dict(root=str(root), entries=rows, warnings=warnings,
        note="Inventario de presencia; la identidad, los hashes y la integridad completa se verifican al cargar."),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['dataset']:12} {row['role']:22} {row['status']:10} " + " | ".join(row['paths']))
    print("Rutas:", args.guardar)
    print("Informe:", args.informe)
    print("W2V K40: se usa el paquete incluido; no depende de localizar K50.")


if __name__ == "__main__":
    main()
