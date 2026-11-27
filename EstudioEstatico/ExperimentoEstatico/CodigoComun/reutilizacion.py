"""Lectura y verificación de resultados; este módulo no ofrece entrenamiento.

GloVe/AA y el anexo W2V K50 usan el adaptador original en modo reuse_only. W2V K40 lee la
exportación posterior, con 75 checkpoints del ganador y 27 ajustes FULL/LOCO.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import ntpath
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import exp2_pipeline_integrated as exp


def _windows_extended_path(value):
    """Ruta Win32 extendida; admite unidades locales y recursos UNC."""
    value = os.fspath(value)
    if value.startswith("\\\\?\\"):
        return value
    value = ntpath.abspath(value)
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def ruta_compatible(path):
    """Permite extraer y leer rutas largas en Windows sin cambiar sus nombres."""
    value = os.path.abspath(Path(path).expanduser())
    if os.name == "nt":
        value = _windows_extended_path(value)
    return Path(value).resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check(condition, message):
    if not condition:
        raise exp.ProtocolError(message)


def configurar(repo, datasets):
    """Las rutas relativas del JSON se resuelven respecto de ExperimentoEstatico."""
    repo = ruta_compatible(repo)
    path = repo / "rutas_resultados.json"
    cfg = json_read(path) if path.is_file() else {}

    def resolved(value):
        p = Path(value).expanduser()
        return ruta_compatible(p if p.is_absolute() else repo / p)

    overrides = {
        key: {name: resolved(value) for name, value in values.items() if value}
        for key, values in cfg.get("overrides", {}).items()
    }
    # El estudio principal usa fuentes incluidas y verificadas, independientemente
    # de las rutas antiguas del computador. El JSON sigue disponible para AA.
    if "glove" in datasets and (repo / "ResultadosDisponibles/GLOVE_INCLUIDO.json").is_file():
        overrides.setdefault("glove", {}).update(_fuentes_glove_incluidas(repo))
    for key in datasets:
        overrides.setdefault(key, {}).update(
            hop_module=repo / "CodigoComun/hopfield.py",
            hfy_module=repo / "CodigoComun/hopfield_fenchel_young.py",
        )
    return exp.StaticConfig(
        project_root=resolved(cfg.get("project_root", ".")),
        output_root=repo / "SalidasRevision" / ("AA" if datasets == ("prototypes",) else "estatico"),
        datasets=datasets, mode="reuse_only", overrides=overrides,
        verify_all_cv_receipts=True, verify_oof_predictions=True,
        max_workers=6, reserve_gib=3.0, block_betas=14,
        figure_dpi=180, show_figures=True,
    )


def protocolo_principal(datasets):
    rows = []
    for key in datasets:
        spec = exp.DATASETS[key]
        rows.append(dict(dataset=key, K=40 if key == "w2v" else spec.selected_k,
                         modelo="H_BETA_LINEAR_Z", ridge=1e-3 if key == "w2v" else spec.ridge_alpha,
                         modo="reuse_only", bins=5, puntos_modelo=30,
                         seleccion_K="Admisibilidad estructural; signos como resultados" if key in ("glove", "w2v")
                         else "AA K80 heredado; cierre exploratorio"))
    return pd.DataFrame(rows)


def _archive(repo, name):
    """Extrae a una carpeta nueva y verifica el ZIP contra la entrega congelada."""
    repo = ruta_compatible(repo)
    source = repo / "ResultadosDisponibles" / name
    manifest = json_read(repo / "ResultadosDisponibles/SHA256.json")
    if not source.is_file():
        raise exp.MissingArtifactError(f"Falta el archivo incluido en la entrega: {source}")
    _check(sha256(source) == manifest[name], f"ZIP alterado: {source}")
    target = repo / "SalidasRevision/archivos" / source.stem
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".extraccion_", dir=target.parent))
        try:
            with zipfile.ZipFile(source) as z:
                for member in z.infolist():
                    destination = (temporary / member.filename).resolve()
                    _check(destination.is_relative_to(temporary.resolve()), "Ruta inválida dentro del ZIP")
                    _check((member.external_attr >> 16) & 0o170000 != 0o120000, "Enlace no admitido en el ZIP")
                z.extractall(temporary)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target


def _verify_manifest(root, name):
    root = ruta_compatible(root)
    manifest = json_read(root / name)
    for relative, expected in manifest.items():
        p = (root / relative).resolve()
        _check(p.is_relative_to(root) and p.is_file(), f"Archivo faltante o ruta inválida: {p}")
        _check(sha256(p) == expected, f"Contenido alterado: {p}")
    return len(manifest)


def _fuentes_en_partes(repo, metadata_file, cache_folder):
    """Comprueba y extrae paquetes independientes con un manifiesto común."""
    meta = json_read(repo / "ResultadosDisponibles" / metadata_file)
    hashes = json_read(repo / "ResultadosDisponibles/SHA256.json")
    for name in meta["partes"]:
        _check(Path(name).name == name, "Nombre de ZIP inválido")
        source = repo / "ResultadosDisponibles" / name
        if not source.is_file():
            raise exp.MissingArtifactError(f"Falta una parte incluida: {source}")
        _check(sha256(source) == hashes[name], "ZIP alterado: " + name)
    target = repo / "SalidasRevision" / cache_folder
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".fuentes_", dir=target.parent))
        try:
            seen = set()
            for name in meta["partes"]:
                with zipfile.ZipFile(repo / "ResultadosDisponibles" / name) as z:
                    for item in z.infolist():
                        path = (temp / item.filename).resolve()
                        _check(path.is_relative_to(temp.resolve()), "Ruta inválida en los resultados")
                        _check((item.external_attr >> 16) & 0o170000 != 0o120000, "Enlace no permitido")
                        _check(item.filename not in seen, "Archivo duplicado entre partes")
                        seen.add(item.filename)
                    z.extractall(temp)
            _verify_manifest(temp, "SHA256.json")
            temp.rename(target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    _verify_manifest(target, "SHA256.json")
    result = {}
    for role, relative in meta["paths"].items():
        path = (target / relative).resolve()
        _check(path.is_relative_to(target.resolve()) and path.is_dir(), "Fuente fuera del paquete")
        result[role] = path
    return result


def _fuentes_glove_incluidas(repo):
    meta = json_read(repo / "ResultadosDisponibles/GLOVE_INCLUIDO.json")
    result = _fuentes_en_partes(repo, "GLOVE_INCLUIDO.json", "glove_incluido_v1")
    geometry = _archive(repo, meta["geometry_archive"])
    _verify_manifest(geometry, "SHA256.json")
    result["geometry_search_root"] = geometry / meta["geometry_root"]
    return result


def cargar_w2v_k50(repo, estructura):
    """Anexo K50 desde archivos incluidos; no modifica la selección principal."""
    repo = ruta_compatible(repo)
    _check(estructura.seleccion["w2v"]["K"] == 40,
           "El anexo K50 corresponde al estudio principal W2V K40")
    _check(exp.DATASETS["w2v"].selected_k == 50, "Contrato K50 incompatible")
    config = configurar(repo, ("w2v",))
    config.overrides["w2v"].update(
        _fuentes_en_partes(repo, "W2V_K50_INCLUIDO.json", "w2v_k50_incluido_v1"))
    config.output_root = repo / "SalidasRevision/anexo_w2v_k50"
    result = exp.run_dataset(config, "w2v", plots=False, replay_k=False)
    historical_policy = result.summary["selected_K_policy"]
    decision = dict(dataset="w2v", K=50, rol="ANEXO_SENSIBILIDAD", K_principal=40,
                    criterio="K50_FIJADA_PARA_EL_ANEXO", seleccion_principal_modificada=False,
                    ridge_K40=0.001, ridge_K50=0.01)
    result.tables["K_selection"] = pd.DataFrame([decision])
    result.summary.update(selected_K_policy=decision["criterio"], K_selection=decision,
                          role=decision["rol"], historical_K_policy=historical_policy)
    exp.csv85(result.output_dir / "tables/K_selection.csv", result.tables["K_selection"])
    exp.write85(result.output_dir / "report/summary.json", result.summary)
    exp.write85(result.output_dir / "report/alcance_anexo_K50.json", decision)
    return result


def cargar_glove(config, estructura):
    """Carga GloVe K40 incluido después de seleccionar K por estructura."""
    selection = estructura.seleccion["glove"]
    _check(config.mode == "reuse_only", "Esta entrega sólo reutiliza resultados")
    _check(selection["K"] == exp.DATASETS["glove"].selected_k,
           "La K seleccionada no corresponde a los resultados GloVe incluidos")
    result = exp.run_dataset(config, "glove", plots=False, replay_k=False)
    result.tables["K_selection"] = pd.DataFrame([selection])
    result.summary["selected_K_policy"] = selection["criterio"]
    result.summary["K_selection"] = selection
    # Son informes derivados, nunca modificaciones a los checkpoints fuente.
    exp.csv85(result.output_dir / "tables/K_selection.csv", result.tables["K_selection"])
    exp.write85(result.output_dir / "report/summary.json", result.summary)
    exp.write85(result.output_dir / "report/seleccion_estructural.json", selection)
    return result


def _verify_oof(ctx):
    """Predicción a partir de coeficientes guardados. Nunca llama a fit."""
    family = next(f for f in exp.FAMILIES85 if f["name"] == ctx["selected90"]["family"])
    anchors = []
    for fold in ctx["folds"]:
        X, Xt, w, _ = exp.design85(ctx, fold, family)
        for network in exp.CORE3:
            path, identity = exp.location85(ctx, network, fold, family, ctx["alpha88"])
            exp._receipt(path, identity)
            with np.load(path / "arrays.npz", allow_pickle=False) as z:
                ids, stored, coef = z["row_id"], z["prediction"], z["coef"]
            _check(np.array_equal(ids, fold["test"]), "Índices OOF incompatibles")
            _, gradient, _ = exp._objective85R(X, ctx["outcomes"][network][fold["train"]], w, ctx["alpha88"], coef)
            delta = float(np.max(np.abs(stored - exp.expit(Xt @ coef))))
            gradient_inf = float(np.max(np.abs(gradient)))
            _check(delta <= 1e-10 and gradient_inf <= 2e-6, "Coeficientes OOF no reproducidos")
            anchors.append(dict(network=network, repeat=fold["repeat"], fold=fold["fold"],
                                prediction_delta=delta, gradient_inf=gradient_inf))
    return pd.DataFrame(anchors)


def cargar_w2v_k40(repo, *, verificar_coeficientes=True, estructura=None):
    """Carga conteos, geometría, selección y ajustes existentes de W2V K40.

    La exportación incluye todos los diagnósticos de selección, pero solamente
    los 75 checkpoints del ganador. No se afirma verificar los otros 3.750.
    """
    repo = ruta_compatible(repo)
    if estructura is None:
        from seleccion_estructural import auditar_estructura
        estructura = auditar_estructura(repo, reconstruir=False)
    _check(estructura.seleccion["w2v"]["K"] == 40,
           "La primera K estructural no corresponde a la exportación K40; no se cambia el experimento")
    root = _archive(repo, "E2_W2V_K40_modelo_3ffd8554b799_resultados.zip")
    completed = _archive(repo, "E2_K40_CORE3_resultados_3539f481e978.zip")
    catalog = _archive(repo, "E2_Revision_Catalogo_W2V_K40_K50.zip") / "E2_Revision_Catalogo_W2V_K40_K50"
    checked_files = _verify_manifest(root, "EXPORT_SHA256.json")
    _verify_manifest(completed, "EXPORT_SHA256.json")
    _verify_manifest(catalog, "SHA256.json")
    protocol = json_read(root / "protocol.json")
    pid = exp.digest85(protocol)
    _check(pid == "3ffd8554b799eb5b8ea87abe3d9ec120450d1784b75801298199eb35d2cbdde7", "Corrida K40 distinta")
    _check(json_read(root / "execution_status.json")["status"] == "COMPLETE", "La corrida no está completa")
    _check(protocol["K"] == 40 and protocol["signs_are_results_not_admissibility"], "Política K40 incompatible")
    cp = json_read(completed / "protocol.json")
    _check(exp.digest85(cp) == protocol["source_completion"], "Conteos de otra ejecución")
    source = catalog / "sources"
    _check(exp.digest85(json_read(source / "protocol_frozen.json")) == protocol["source_audit"], "Otro dataset de origen")
    geo = source / "K40/geometry"
    exp._receipt(geo, protocol["geometry"])
    gate = json_read(geo / "gate.json")
    _check(gate["A"], "K40 no supera la validación estructural")
    ctx = {k: pd.read_parquet(geo / (k + ".parquet")) for k in ("full", "common", "expected")}
    ctx.update(K=40, gate=gate, edges=np.asarray(gate["edges"]), centers=np.asarray(gate["centers"]),
               split_audit=json_read(geo / "split_audit.json"), neighbors=json_read(geo / "neighbors.json"),
               signature=protocol["geometry"], labels=exp.CLASSES85, betas=exp.BETAS.copy(),
               keys=np.round(exp.BETAS, 12), counts={}, empirical={}, root=root, id=pid, id85=pid,
               out85=root / "MODEL_SEARCH", out88=root / "FULL_LOCO", dataset_label="W2V300 RAW")
    exp._validate_geometry(ctx)
    for key in ("full", "common", "expected"):
        pd.testing.assert_frame_equal(ctx[key], pd.read_parquet(root / "inputs/geometry" / (key + ".parquet")),
                                      check_dtype=False, check_exact=True)
    calibration = pd.read_csv(root / "calibration.csv", float_precision="round_trip")
    _check(len(calibration) == 102 and not calibration.duplicated(["network", "split_seed"]).any(), "Escalas incompletas")
    for network in exp.CORE3:
        count_path = root / "inputs" / network / "word_beta.parquet"
        source_path = completed / "report/K40" / network
        binding = json_read(source_path / "source_binding.json")
        _check(sha256(count_path) == protocol["counts_sha256"][network] == binding["source_count_sha256"], "Conteos distintos")
        _check(count_path.read_bytes() == (source_path / "word_beta.parquet").read_bytes(), "Conteos y fuente discrepan")
        if network != exp.CORE3[0]:
            cal = calibration.loc[calibration.network.eq(network)].sort_values("split_seed")
            _check(cal.split_seed.tolist() == list(range(51)) and cal.lambda_value.between(.001, 1).all(), "Escalas de otro diseño")
            _check(cal.calibration_uses_errors.eq(False).all() and cal.calibration_uses_validation.eq(False).all(), "Escalas no TRAIN-only")
            audits = {a["split_seed"]: a for a in ctx["split_audit"]}
            for row in cal.to_dict("records"):
                _check(all(row[k] == audits[row["split_seed"]][k] for k in ("train_words_sha", "train_array_sha")), "Otro TRAIN")
        ctx["counts"][network] = exp.validate_wb85(ctx, pd.read_parquet(count_path))
        ctx["empirical"][network] = exp.surface85(ctx, ctx["counts"][network])
    ctx["calibration"] = calibration
    exp._prepare_model_inputs(ctx)
    _check(exp.digest85(ctx["fold_ledger"].to_dict("records")) == protocol["CV"]["folds_sha256"], "CV distinta")
    original = repo / "CodigoComun/w2v_k40_original"
    for name, h in protocol["code_sha256"].items():
        _check(sha256(original / name) == h, "Código K40 alterado: " + name)
    sys.path.insert(0, str(original))
    model = importlib.import_module("e2_k40_model")
    _check(Path(model.__file__).resolve() == original / "e2_k40_model.py", "Reinicia el kernel: otro módulo K40 está cargado")
    exp._receipt(ctx["out85"], exp.digest85(dict(parent=pid, stage="selection")))
    model._load_selection(ctx)
    _check(ctx["selected90"]["candidate"] == "H_BETA_LINEAR_Z__ridge_1e-03", "Otro ganador")
    diagnostics = pd.read_csv(root / "MODEL_SEARCH/all_fit_diagnostics.csv")
    _check(len(diagnostics) == 3825 and not diagnostics.duplicated(["candidate", "network", "repeat", "fold"]).any(), "Diagnósticos incompletos")
    _check(diagnostics.stable.astype(str).str.lower().eq("true").all(), "Diagnósticos inestables")
    refit_protocol = json_read(root / "FULL_LOCO/protocol_frozen.json")
    _check(refit_protocol == dict(parent=pid, stage="FULL_LOCO", retune=False), "Protocolo FULL/LOCO distinto")
    ctx["id88"] = exp.digest85(dict(parent=pid, stage="FULL_LOCO"))
    anchors = pd.DataFrame()
    with threadpool_limits(limits=1):
        if verificar_coeficientes:
            anchors = _verify_oof(ctx)
        for network in exp.CORE3:
            for omitted in [None, *range(8)]:
                tag = "FULL" if omitted is None else f"LOCO_{omitted}_{exp.CLASSES85[omitted]}"
                exp._receipt(ctx["out88"] / "fits" / network / tag,
                             exp.digest85(dict(protocol=ctx["id88"], network=network, omitted=omitted)))
                if verificar_coeficientes:
                    model._verify_refit(ctx, network, omitted)
    tables = {name: pd.read_csv(root / rel, float_precision="round_trip") for name, rel in {
        "input_audit": "input_audit.csv", "calibration": "calibration.csv",
        "empirical_FULL_LOCO": "empirical_FULL_LOCO.csv", "model_ranking": "MODEL_SEARCH/candidate_ranking.csv",
        "OOF": "OOF/metrics.csv", "FULL_LOCO": "report/FULL_LOCO.csv", "decisions": "report/decisions.csv",
        "fidelity": "report/fidelity.csv"}.items()}
    tables["verificacion_OOF"] = anchors
    tables["sensibilidad_K50"] = pd.read_csv(completed / "report/K50_minus_K40.csv", float_precision="round_trip")
    tables["K_selection"] = pd.DataFrame([estructura.seleccion["w2v"]])
    result = SimpleNamespace(dataset="w2v", context=ctx, output_dir=root, tables=tables,
                             summary=json_read(root / "report/summary.json"))
    result.verification = dict(archivos_export_verificados=checked_files, diagnosticos_CV=3825,
                              checkpoints_CV_incluidos=75, checkpoints_CV_no_incluidos=3750,
                              refits_verificados=27 if verificar_coeficientes else 0,
                              coeficientes_verificados=verificar_coeficientes, nuevos_ajustes=0, nuevas_clasificaciones=0)
    print("W2V K40: geometría, conteos y selección verificados; no se han ajustado modelos ni ejecutado clasificadores.")
    return result


def mostrar_w2v(result, names):
    from IPython.display import display
    for name in names:
        print("W2V K40 ·", name)
        display(result.tables[name])


def graficar_w2v(result, *, heatmaps=True, fit_quality=True, oof=True, loco=False):
    """Muestra PNG guardados, previamente autenticados al cargar la exportación."""
    from IPython.display import Image, display
    kinds = (["FULL_EMPIRICAL", "FULL_GRID30"] if heatmaps else [])
    kinds += ["ABC"] if fit_quality else []
    kinds += ["OOF"] if oof else []
    kinds += ["LOCO", *[f"LOCO_{label}_GRID30" for label in exp.CLASSES85]] if loco else []
    for network in exp.CORE3:
        for kind in kinds:
            path = result.output_dir / "report/figures" / f"{network}_{kind}.png"
            display(Image(filename=str(path)))


def resumen_conjunto(results):
    rows = []
    for key, result in results.items():
        if key == "w2v" and result.context["K"] == 40:
            for row in result.tables["decisions"].query("scope == 'grid30'").to_dict("records"):
                rows.append(dict(dataset=key, K=40, network=row["network"],
                                 modelo=result.summary["selected"]["candidate"],
                                 T1_modelo=row["T1_FULL_expected"], T2_modelo=row["T2_FULL_expected"],
                                 T3_modelo=row["T3_FULL_expected"], LOCO_tres_signos=row["ALL3_LOCO_passed"]))
        else:
            for row in result.tables["final_status"].to_dict("records"):
                rows.append(dict(dataset=key, K=result.context["K"], network=row["network"],
                                 modelo=result.summary["model"],
                                 tres_signos_modelo=row["M_primary"], LOCO_tres_signos=row["model_LOCO_passed_30"]))
    return pd.DataFrame(rows)
