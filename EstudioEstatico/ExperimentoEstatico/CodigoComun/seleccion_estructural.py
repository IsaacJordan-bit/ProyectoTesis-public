"""Primera K admisible en K20–K100 usando sólo geometría; sin leer tasas de error.

Integra la evidencia de K20/K30 con la auditoría K40–K100. Reconstruye COMMON,
cobertura, particiones y diseños, sin entrenar ni usar T1–T3.
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import reutilizacion as reuse

DOMINIO_K = tuple(range(20, 101, 10))
DATASETS = {"glove": "GloVe", "w2v": "W2V"}


def seleccionar_primera_admisible(table, dataset, dominio=DOMINIO_K):
    """Una geometría ausente pausa la selección; no equivale a un rechazo.

    Los únicos campos leídos son dataset, K y five_bin_geometry_admissible.
    La tabla de entrada debe proceder de auditar_estructura, que comprueba los
    demás requisitos y su integridad. T1/T2/T3, LOCO y errores no participan.
    """
    if dataset not in DATASETS:
        raise ValueError("Esta auditoría contiene sólo GloVe y W2V")
    dominio = tuple(dominio)
    if not dominio or tuple(sorted(set(dominio))) != dominio:
        raise ValueError("El dominio debe estar ordenado y no contener duplicados")
    rows = table.loc[table.dataset.eq(DATASETS[dataset]), ["K", "five_bin_geometry_admissible"]]
    if rows.K.duplicated().any():
        raise reuse.exp.ProtocolError("Geometrías distintas para una misma K: resolver la identidad")
    visited = []
    for k in dominio:
        found = rows.loc[rows.K.eq(k)]
        if len(found) != 1:
            raise reuse.exp.MissingArtifactError(
                f"Falta evidencia estructural de {dataset} K{k}. "
                "No se salta una K sin evidencia ni se supone un rechazo."
            )
        valid = found.iloc[0].five_bin_geometry_admissible
        if pd.isna(valid):
            raise reuse.exp.MissingArtifactError(f"{dataset} K{k} no está disponible; no se puede continuar la selección")
        if not isinstance(valid, (bool, np.bool_)):
            raise reuse.exp.ProtocolError("Admisibilidad debe ser un booleano verificado")
        visited.append(int(k))
        if valid:
            return dict(dataset=dataset, K=int(k), criterio="FIRST_STRUCTURALLY_ADMISSIBLE",
                        grilla_candidata=list(dominio), K_evaluadas_hasta_seleccion=visited,
                        usa_T1_T2_T3=False, usa_LOCO=False)
    raise reuse.exp.ProtocolError("Ninguna K del dominio es estructuralmente admisible")


def _extract_geometry(source, destination):
    source = reuse.ruta_compatible(source)
    destination = reuse.ruta_compatible(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".geometrias_", dir=destination.parent))
    try:
        with zipfile.ZipFile(source) as z:
            for item in z.infolist():
                path = (temp / item.filename).resolve()
                reuse._check(path.is_relative_to(temp.resolve()), "Ruta inválida en geometrías")
                reuse._check((item.external_attr >> 16) & 0o170000 != 0o120000, "Enlace no permitido")
            z.extractall(temp)
        temp.rename(destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def auditar_estructura(repo, *, reconstruir=True):
    """Comprueba la evidencia estructural. Reconstruir nunca ejecuta el experimento.

    reconstruir=False lee las tablas de la auditoría previamente guardada,
    verificando sus hashes; True recalcula los diagnósticos geométricos.
    """
    repo = reuse.ruta_compatible(repo)
    package = reuse._archive(repo, "E2_Auditoria_estructural_K40_K100.zip") / "E2_Auditoria_estructural_K40_K100"
    reuse._verify_manifest(package, "checksums.json")
    inputs = repo / "SalidasRevision/geometrias_estructurales"
    _extract_geometry(package / "geometrias_originales.zip", inputs)
    manifest = reuse.json_read(inputs / "manifest.json")
    for item in manifest["files"]:
        path = (inputs / item["archive_name"]).resolve()
        reuse._check(path.is_relative_to(inputs) and path.is_file(), "Falta entrada geométrica")
        reuse._check(path.stat().st_size == item["bytes"] and reuse.sha256(path) == item["sha256"], "Geometría alterada")
    if reconstruir:
        out = repo / "SalidasRevision/auditoria_estructural"
        subprocess.run([sys.executable, str(package / "audit_geometry.py"), "--inputs", str(inputs),
                        "--source", str(repo / "CodigoComun/exp2_pipeline_integrated.py"),
                        "--out", str(out)], check=True)
    else:
        out = package / "tablas"
    table = pd.read_csv(out / "resumen_geometrias.csv")
    report = reuse.json_read(out / "resultado.json")
    reuse._check(not report["outcomes_read"] and not report["models_fitted"] and not report["classification_rerun"],
                 "La selección debe ser independiente de los resultados científicos")
    from auditoria_k_inferiores import auditar_inferiores
    lower, lower_coverage, lower_report, glove = auditar_inferiores(repo)
    table["estado"] = np.where(table.five_bin_geometry_admissible, "ADMISIBLE", "RECHAZADA")
    table["motivo"] = np.where(table.five_bin_geometry_admissible, "Estructura y cobertura fija5 válidas", "A4: celda categoría×bin vacía")
    table["evidencia"] = "Geometría archivada y auditada"
    table = pd.concat([lower, table], ignore_index=True)
    grid = pd.DataFrame([(d,k) for d in DATASETS.values() for k in DOMINIO_K], columns=["dataset","K"])
    table = grid.merge(table, on=["dataset","K"], how="left", validate="one_to_one")
    table["estado"] = table.estado.fillna("NO_DISPONIBLE")
    table["motivo"] = table.motivo.fillna("No se incluyó geometría compatible; no es un rechazo")
    selections = {key: seleccionar_primera_admisible(table, key) for key in DATASETS}
    coverage = pd.concat([lower_coverage, pd.read_csv(out / "cobertura_clase_bin.csv")], ignore_index=True)
    report = dict(report, grilla=list(DOMINIO_K), auditoria_K20_K30=lower_report,
                  seleccion=selections, lower_K_claim=True,
                  alcance="Primera K admisible en la grilla 20–100; W2V K60–K100 no requeridos tras aceptar K40")
    combined = repo / "SalidasRevision/auditoria_grilla_K20_K100"
    combined.mkdir(parents=True, exist_ok=True)
    table.to_csv(combined / "resumen_geometrias.csv", index=False)
    coverage.to_csv(combined / "cobertura_clase_bin.csv", index=False)
    (combined / "resultado.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Fuente exacta de la geometría GloVe declarada por GK83.
    protocol = reuse.json_read(glove / "protocol_frozen.json")
    reuse._check(glove.name == "search_" + reuse.exp.digest85(protocol)[:20], "Otra geometría GloVe")
    return SimpleNamespace(seleccion=selections, tabla=table,
                           cobertura=coverage,
                           informe=report, output_dir=combined, geometry_search_root=glove,
                           reconstruida=reconstruir)
