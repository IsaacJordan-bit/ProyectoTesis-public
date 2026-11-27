"""Completa exclusivamente HFY/W2V/K40 RAW; auditoría empírica CORE3.

Recetas numéricas importadas, sin modificar, del módulo estático autenticado.
No selecciona K, modelos ni umbrales usando los signos observados.
"""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
import zipfile
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import e2_k40_reference as ref

PACKAGE = Path(__file__).resolve().parent
VERSION = "E2_W2V_K40_HFY_COMPLETION_V1"
AUDIT_ID = "f4766816e364f73e7b47"
CORE3 = ref.CORE3
HFY = CORE3[1:]
BLOCK = 7
req = ref.req85
readj = ref.read85
writej = ref.write85
sha = ref.sha85
digest = ref.digest85  # Serialización histórica, incluidos sus espacios.


class TechnicalPause(RuntimeError):
    pass


@dataclass
class Config:
    project_root: Path = Path.home() / "Workspace/Tesis/Exp2def"
    output_root: Path = Path.home() / "E2R"
    audit_root: Path | None = None
    max_workers: int = 14
    reserve_gib: float = 3.0
    reserve_disk_gib: float = 2.0

    def __post_init__(self):
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.output_root = Path(self.output_root).expanduser().resolve()
        self.audit_root = (Path(self.audit_root).expanduser().resolve() if self.audit_root
            else self.project_root / "Results/W2V/_PROTOCOL_RECTOR_V3" / ("audit_" + AUDIT_ID))
        req(isinstance(self.max_workers, int) and 1 <= self.max_workers <= 14,
            "max_workers debe estar entre 1 y 14.")
        req(self.reserve_gib >= 3 and self.reserve_disk_gib >= 1, "Reservas insuficientes.")


def _versions():
    return dict(numpy=np.__version__, pandas=pd.__version__,
                scipy=ref.scipy.__version__, sklearn=ref.sklearn.__version__)


def _package_check():
    req(Path(ref.__file__).resolve() == PACKAGE / "e2_k40_reference.py",
        "Otro módulo de referencia está cargado. Reinicia el kernel.")
    expected = readj(PACKAGE / "CODE_SHA256.json")
    for name, value in expected.items():
        p = (PACKAGE / name).resolve()
        req(p.is_relative_to(PACKAGE) and p.is_file() and sha(p) == value,
            "Código incompleto o modificado: " + name)
    return expected


def _pin_receipt(ctx, folder, identity):
    m = ref._receipt(folder, identity)
    ref._pin(ctx, folder / "completed.json")
    for name, value in m["outputs"].items():
        ref._pin(ctx, folder / name, value)
    return m


def _cal_valid(g, n, seed, row):
    req(row.get("network") == n and row.get("split_seed") == seed
        and row.get("K", g["K"]) == g["K"], "Escala de otra red/K/partición.")
    req(.001 <= row["lambda_value"] <= 1
        and row.get("calibration_uses_errors") is False
        and row.get("calibration_uses_validation") is False, "Escala no TRAIN-only.")
    a = next(a for a in g["split_audit"] if a["split_seed"] == seed)
    for field in ("train_words_sha", "train_array_sha"):
        req(row[field] == a[field], "Escala de otro TRAIN: " + field)
    return row


def _historical_cal(ctx, g, n, seed):
    f = g["folder"].parent / "calibration" / n / f"split_{seed:02d}"
    if not (f / "completed.json").is_file():
        return None
    ident = digest(dict(parent=ctx["audit_id"], geometry=g["signature"],
                        network=n, split=seed, phase="calibration"))
    _pin_receipt(ctx, f, ident)
    return _cal_valid(g, n, seed, readj(f / "calibration.json"))


def _counts_identity(ctx, g, n, cal):
    scales = np.ones(51) if n == CORE3[0] else np.array([r["lambda_value"] for r in cal])
    return digest(dict(parent=ctx["audit_id"], geometry=g["signature"], network=n,
        scales=scales, phase="counts", classifier_seeds=list(range(10)),
        seed_merge_strategy="AUTHENTICATED_DISJOINT_SEEDS_V1"))


def _historical_counts(ctx, g, n):
    f = g["folder"].parent / "counts" / n
    if not (f / "completed.json").is_file():
        return None
    cal = None if n == CORE3[0] else [_historical_cal(ctx, g, n, s) for s in range(51)]
    if cal is not None and any(r is None for r in cal):
        raise ref.MissingArtifactError("Hay conteos completos, pero falta su calibración autenticada: " + str(f))
    _pin_receipt(ctx, f, _counts_identity(ctx, g, n, cal))
    return ref.validate_wb85(dict(ctx, **g), pd.read_parquet(f / "word_beta.parquet"))


def _own_location(ctx, stage, n, seed=None, left=None, scale=None):
    args = dict(run=ctx["id"], stage=stage, network=n)
    f = ctx["root"] / stage / n
    if seed is not None:
        args["split"] = seed
        f = f / f"s{seed:02d}"
    if left is not None:
        args.update(betas=ref.BETAS[left:left + BLOCK], scale=float(scale), classifier_seeds=list(range(10)))
        f = f / f"b{left:02d}_{min(left + BLOCK, 98):02d}"
    return f, digest(args)


def _own_cal(ctx, n, seed):
    f, identity = _own_location(ctx, "calibration", n, seed)
    if ref.completed85(f, identity) is None:
        return None
    return _cal_valid(ctx["g"], n, seed, readj(f / "calibration.json"))


def _own_counts(ctx, n):
    f, identity = _own_location(ctx, "counts", n)
    if ref.completed85(f, identity) is None:
        return None
    return ref.validate_wb85(dict(ctx, **ctx["g"]), pd.read_parquet(f / "word_beta.parquet"))


def prepare(config=None):
    cfg = config or Config()
    code = _package_check()
    overrides = dict(audit_root=cfg.audit_root, hop_module=PACKAGE / "reference/hopfield.py",
                     hfy_module=PACKAGE / "reference/hopfield_fenchel_young.py")
    sc = ref.StaticConfig(project_root=cfg.project_root, output_root=cfg.output_root,
        datasets=("w2v",), max_workers=cfg.max_workers, reserve_gib=cfg.reserve_gib,
        show_figures=False, overrides={"w2v": overrides})
    ctx = ref._configuration(sc, "w2v")
    protocol = ref._read_audit_contract(ctx)
    req(ctx["audit_id"].startswith(AUDIT_ID), "La corrida fuente no es la W2V auditada.")
    g = ref._load_geometry(ctx, 40)
    req(g["gate"]["A"], "K40 no tiene geometría estructural admisible.")
    _pin_receipt(ctx, g["folder"], g["signature"])
    contract = dict(version=VERSION, source_audit=ctx["audit_id"], geometry=g["signature"],
        code_sha256=code, dataset="W2V300", K=40, condition="RAW", networks=list(CORE3),
        compute_networks=list(HFY), betas=ref.BETAS, split_seeds=list(range(51)),
        classifier_seeds=list(range(10)), empirical_bins=5, axis="WITHIN_CLASS_Z",
        COMMON="frozen_per_K", class_omissions=list(range(8)),
        calibration=ref.CALIBRATION_RECIPE, calibration_train_only=True,
        expected_scientific_versions=protocol["scientific_versions"],
        K_selection=False, fit_statistical_models=False, stop_on_sign_failure=False,
        LOCO="aggregation_of_existing_experiment; not_class_ablation",
        inferential_claim="descriptive_sensitivity_no_p_values_or_confidence_intervals")
    identity = digest(contract)
    root = cfg.output_root / ("k40_hfy_" + identity[:12])
    req(not root.is_relative_to(cfg.audit_root), "La salida no puede estar dentro del archivo histórico.")
    root.mkdir(parents=True, exist_ok=True)
    with ref.lock85(root):
        p = root / "protocol.json"
        if p.exists():
            req(readj(p) == ref.safe85(contract), "Colisión de protocolo de la ampliación.")
        else:
            writej(p, contract)
    ctx.update(K=40, cfg=cfg, root=root, id=identity, protocol=contract, g=g,
               counts40={}, source40={}, calibration={}, preflight=[])
    for n in CORE3:
        d = _historical_counts(ctx, g, n)
        origin = "HISTORICAL_AUTHENTICATED"
        if d is None:
            d = _own_counts(ctx, n)
            origin = "COMPLETION_AUTHENTICATED"
        if d is not None:
            ctx["counts40"][n] = d
            ctx["source40"][n] = origin
        elif n == CORE3[0]:
            raise ref.MissingArtifactError("Falta el conteo Softmax K40 ya calculado: " +
                str(g["folder"].parent / "counts" / n / "word_beta.parquet"))
        ctx["preflight"].append(dict(network=n, K=40,
            status="COMPLETE_REUSE" if d is not None else "COMPLETE_MISSING_WORK",
            n_betas=98, n_splits=51, n_classifier_seeds=10))
    current = _versions()
    ctx["environment"] = pd.DataFrame([dict(package=k, current=current[k], required=v,
        matches=current[k] == v) for k, v in protocol["scientific_versions"].items()])
    print("Salida:", root)
    print("K40 RAW · 98 betas × 51 particiones × 10 semillas. Softmax reutilizado.")
    print("Procesos máximos:", cfg.max_workers, "| reserva RAM:", cfg.reserve_gib, "GiB")
    return ctx


def _validate_block(g, d, seed, grid, scale):
    expected = g["expected"].loc[lambda q: q.split_seed.eq(seed)].set_index("word")
    req(len(d) == len(expected) * len(grid) and not d.duplicated(["word", "beta_key"]).any(),
        "Bloque con filas duplicadas/ausentes.")
    req(set(d.word) == set(expected.index) and set(d.beta_key) == set(np.round(grid, 12))
        and d.groupby("word").size().eq(len(grid)).all(), "Inventario beta/palabra incorrecto.")
    req(d.n_items.eq(10).all() and np.isfinite(d.n_errors).all()
        and d.n_errors.between(0, 10).all() and np.equal(d.n_errors, np.floor(d.n_errors)).all(),
        "Semillas/errores incorrectos.")
    req(d.true_class.eq(d.word.map(expected.true_class)).all()
        and np.allclose(d.validation_similarity, d.word.map(expected.validation_similarity), rtol=0, atol=1e-12),
        "Geometría incorrecta en bloque.")
    req(np.allclose(d.beta_effective, d.beta_key * scale, rtol=1e-10, atol=1e-12)
        and np.allclose(d.lambda_value, scale, rtol=0, atol=1e-14), "Escala distinta.")
    return d


def _historical_block(ctx, n, seed, left, scale, cal):
    g = ctx["g"]
    width = int(ctx["audit_contract"].get("block_betas", 14))
    pieces = []
    grid = ref.BETAS[left:left + BLOCK]
    parent = _counts_identity(ctx, g, n, cal)
    for j, a in enumerate(range(0, 98, width)):
        old_grid = ref.BETAS[a:a + width]
        wanted = np.intersect1d(np.round(grid, 12), np.round(old_grid, 12))
        if not len(wanted):
            continue
        folder = g["folder"].parent / "counts" / n / f"split_{seed:02d}" / f"block_{j:02d}"
        if not (folder / "completed.json").is_file():
            return None
        _pin_receipt(ctx, folder, digest(dict(parent=parent, split=seed, betas=old_grid)))
        q = pd.read_parquet(folder / "counts.parquet")
        pieces.append(q.loc[q.beta_key.isin(wanted)].copy())
    return _validate_block(g, pd.concat(pieces, ignore_index=True), seed, grid, scale)


# Los procesos reciben sólo datos pequeños; nunca módulos dinámicos ni modelos ajustados.
_W = None
_SPLITS = None
_THREAD_LIMIT = None


def _init_worker(payload):
    global _W, _SPLITS, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    _W = payload
    _W["g"]["backend"] = ref._audited_backend(_W["ctx"])
    _SPLITS = OrderedDict()


def _split(seed):
    if seed not in _SPLITS:
        _SPLITS[seed] = ref._prepared(_W["ctx"], _W["g"], seed)
        if len(_SPLITS) > 2:
            _SPLITS.popitem(last=False)
    _SPLITS.move_to_end(seed)
    return _SPLITS[seed]


def _compute_task(task):
    folder = Path(task["folder"])
    old = ref.completed85(folder, task["identity"])
    if old is not None:
        return dict(task=task, reused=True, seconds=0.)
    started = time.perf_counter()
    sp = _split(task["seed"])
    n = task["network"]
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if task["stage"] == "calibration":
            row, curves, trace = ref._calibrate_one(sp, _W["g"]["neighbors"], n, _W["g"]["backend"])
            row.update(network=n, K=40, split_seed=task["seed"],
                train_words_sha=digest(sp.train_words),
                train_array_sha=hashlib.sha256(sp.train_embeddings.to_numpy(dtype="<f8").tobytes()).hexdigest())
            _cal_valid(_W["g"], n, task["seed"], row)
            writej(folder / "calibration.json", row)
            ref.frame85(folder / "curves.parquet", curves)
            writej(folder / "trace.json", trace)
            names = ["calibration.json", "curves.parquet", "trace.json"]
        else:
            grid = ref.BETAS[task["left"]:task["left"] + BLOCK]
            d = ref._classification_block(_W["g"], sp, n, task["seed"], grid, task["scale"])
            _validate_block(_W["g"], d, task["seed"], grid, task["scale"])
            ref.frame85(folder / "counts.parquet", d)
            row = dict(network=n, K=40, split_seed=task["seed"], betas=grid,
                       lambda_value=task["scale"], classifier_seeds=list(range(10)))
            names = ["counts.parquet"]
    warning_counts = {}
    for w in caught:
        key = (w.category.__name__, str(w.message))
        warning_counts[key] = warning_counts.get(key, 0) + 1
    writej(folder / "warnings.json", [dict(category=k[0], message=k[1], count=v)
                                        for k, v in warning_counts.items()])
    writej(folder / "runtime.json", dict(seconds=time.perf_counter() - started,
        scientific_versions=_versions(), pid=os.getpid(), log=log.getvalue()))
    names += ["warnings.json", "runtime.json"]
    ref.receipt85(folder, task["identity"], names, row)
    return dict(task=task, reused=False, seconds=time.perf_counter() - started)


def _runtime_guard(ctx):
    bad = ctx["environment"].loc[lambda x: ~x.matches]
    if len(bad):
        raise TechnicalPause("Usa el kernel del experimento estático para los ensayos nuevos. "
            "Las versiones numéricas deben coincidir con el archivo histórico:\n" + bad.to_string(index=False)
            + "\nNo se modificó ningún entorno. Consulta LEEME.txt si necesitas un entorno separado.")


def _payload(ctx):
    # Verificar datos y los 51 TRAIN antes de iniciar trabajo largo.
    g = dict(ctx["g"])
    catalog = ref._catalog_embeddings(ctx)
    words = list(ref.build_word_class_map(g["neighbors"]))
    req(set(words) <= set(catalog.index), "El catálogo no cubre K40.")
    g["embeddings"] = catalog.loc[words].copy()
    del catalog
    g["backend"] = ref._audited_backend(ctx)
    print("Verificando los 51 TRAIN/VALIDATION y sus hashes…", flush=True)
    with threadpool_limits(limits=1):
        for seed in range(51):
            ref._prepared(ctx, g, seed)
    g.pop("backend")
    worker_ctx = {k: ctx[k] for k in ("config", "paths", "spec", "audit_id")}
    worker_ctx["protected"] = {}
    return dict(ctx=worker_ctx, g=g)


def _resource_snapshot():
    p = psutil.Process()
    rss = p.memory_info().rss
    for child in p.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.Error:
            pass
    return dict(available_GiB=psutil.virtual_memory().available / 2**30,
                process_tree_RSS_GiB=rss / 2**30, cpu_percent=psutil.cpu_percent(None))


def _schedule(ctx, pool, tasks, workers, stage):
    total = len(tasks)
    if not total:
        return
    cursor = done = 0
    pending = {}
    started = last = time.perf_counter()
    paused = False
    history = ctx.setdefault("runtime_log", [])
    print(f"{stage}: {total} tareas nuevas; hasta {workers} procesos.", flush=True)
    try:
        while cursor < total or pending:
            now = time.perf_counter()
            stats = _resource_snapshot()
            if (ctx["root"] / "STOP").exists():
                paused = True
            disk = ref.shutil.disk_usage(ctx["root"]).free / 2**30
            if disk < ctx["cfg"].reserve_disk_gib:
                paused = True
            # Las tareas activas terminan; si no queda margen no se encolan nuevas.
            while not paused and cursor < total and len(pending) < workers and stats["available_GiB"] > ctx["cfg"].reserve_gib + .25:
                task = tasks[cursor]
                pending[pool.submit(_compute_task, task)] = task
                cursor += 1
            if not pending:
                if cursor < total:
                    raise TechnicalPause("Pausa por STOP, RAM o disco. Los bloques completos se conservan.")
                break
            ready, _ = wait(pending, timeout=3, return_when=FIRST_COMPLETED)
            for future in ready:
                future.result()  # Propaga el error y conserva los demás checkpoints.
                del pending[future]
                done += 1
            elapsed = time.perf_counter() - started
            if now - last >= 30 or done == total:
                # Estimación de esta etapa solamente; calibración y clasificación tienen costes distintos.
                eta = (total - done) * elapsed / done / 60 if done >= max(workers, 3) else None
                row = dict(stage=stage, completed=done, total_new=total, active=len(pending),
                           worker_cap=workers, elapsed_minutes=elapsed / 60, ETA_stage_minutes=eta, **stats)
                history.append(row)
                writej(ctx["root"] / "progress.json", row)
                ref.csv85(ctx["root"] / "resources.csv", pd.DataFrame(history))
                print(f"{stage}: {done}/{total} | activos {len(pending)} | RAM libre {stats['available_GiB']:.1f} GiB"
                      + (f" | restante de esta etapa ≈ {eta:.1f} min" if eta is not None else ""), flush=True)
                last = now
        if paused:
            raise TechnicalPause("Pausa solicitada. Retira STOP y repite run para continuar.")
    except BaseException:
        for future in pending:
            future.cancel()
        print("Cerrando tareas activas; sus checkpoints se conservarán. No vuelvas a interrumpir el cierre.", flush=True)
        raise


def _task(ctx, stage, n, seed, left=None, scale=None):
    f, identity = _own_location(ctx, stage, n, seed, left, scale)
    return dict(stage=stage, network=n, seed=seed, left=left, scale=scale,
                folder=str(f), identity=identity)


def _save_reused(ctx, task, filenames, origin, row):
    f = Path(task["folder"])
    writej(f / "origin.json", origin)
    ref.receipt85(f, task["identity"], filenames + ["origin.json"], row)


def run(ctx):
    missing = [n for n in HFY if n not in ctx["counts40"]]
    with ref.lock85(ctx["root"]):
        try:
            if missing:
                if (ctx["root"] / "STOP").exists():
                    raise TechnicalPause("STOP presente. Elimínalo para reanudar.")
                cal_tasks = []
                for n in missing:
                    for seed in range(51):
                        if _own_cal(ctx, n, seed) is not None:
                            continue
                        old = _historical_cal(ctx, ctx["g"], n, seed)
                        task = _task(ctx, "calibration", n, seed)
                        if old is None:
                            cal_tasks.append(task)
                        else:
                            writej(Path(task["folder"]) / "calibration.json", old)
                            _save_reused(ctx, task, ["calibration.json"],
                                dict(kind="historical_calibration", source=str(ctx["g"]["folder"].parent / "calibration" / n / f"split_{seed:02d}")), old)
                # Crear procesos/vectores sólo si efectivamente falta cálculo numérico.
                with contextlib.ExitStack() as stack:
                    pool = None
                    workers = None

                    def schedule_new(tasks, stage):
                        nonlocal pool, workers
                        if not tasks:
                            return
                        if pool is None:
                            _runtime_guard(ctx)
                            payload = _payload(ctx)
                            available = psutil.virtual_memory().available / 2**30
                            workers = min(ctx["cfg"].max_workers,
                                int((available - ctx["cfg"].reserve_gib) / .65))
                            if workers < 1:
                                raise TechnicalPause("Falta RAM para iniciar un proceso conservando la reserva.")
                            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                                os.environ[key] = "1"
                            pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers,
                                mp_context=mp.get_context("spawn"), initializer=_init_worker, initargs=(payload,)))
                        _schedule(ctx, pool, tasks, workers, stage)

                    schedule_new(cal_tasks, "Calibración TRAIN-only")
                    block_tasks = []
                    for n in missing:
                        cal = [_own_cal(ctx, n, s) for s in range(51)]
                        req(all(r is not None for r in cal), "Calibración incompleta.")
                        ctx["calibration"][n] = cal
                        for seed, row in enumerate(cal):
                            scale = row["lambda_value"]
                            for left in range(0, 98, BLOCK):
                                task = _task(ctx, "blocks", n, seed, left, scale)
                                f = Path(task["folder"])
                                if ref.completed85(f, task["identity"]) is not None:
                                    _validate_block(ctx["g"], pd.read_parquet(f / "counts.parquet"), seed,
                                                    ref.BETAS[left:left + BLOCK], scale)
                                    continue
                                old = _historical_block(ctx, n, seed, left, scale, cal)
                                if old is None:
                                    block_tasks.append(task)
                                else:
                                    ref.frame85(f / "counts.parquet", old)
                                    _save_reused(ctx, task, ["counts.parquet"], dict(kind="historical_counts_block",
                                        source=str(ctx["g"]["folder"].parent / "counts" / n), split=seed, left=left),
                                        dict(network=n, split_seed=seed, lambda_value=scale))
                    # Intercalar redes/particiones permite avanzar en ambas, sin veto por signos.
                    block_tasks.sort(key=lambda t: (t["seed"], t["left"], t["network"]))
                    schedule_new(block_tasks, "Clasificación HFY")
                for n in missing:
                    parts = []
                    cal = ctx["calibration"][n]
                    for seed, row in enumerate(cal):
                        for left in range(0, 98, BLOCK):
                            f, ident = _own_location(ctx, "blocks", n, seed, left, row["lambda_value"])
                            req(ref.completed85(f, ident) is not None, "Falta un checkpoint.")
                            d = _validate_block(ctx["g"], pd.read_parquet(f / "counts.parquet"), seed,
                                                ref.BETAS[left:left + BLOCK], row["lambda_value"])
                            parts.append(d.assign(split_seed=seed))
                    allrows = pd.concat(parts, ignore_index=True)
                    req(not allrows.duplicated(["split_seed", "word", "beta_key"]).any(), "Doble exposición.")
                    wb = allrows.groupby(["word", "true_class", "beta_key"], as_index=False).agg(
                        n_items=("n_items", "sum"), n_errors=("n_errors", "sum"),
                        validation_similarity=("validation_similarity", "first"))
                    wb["beta"] = wb.beta_key
                    wb = ref.validate_wb85(dict(ctx, **ctx["g"]), wb)
                    f, identity = _own_location(ctx, "counts", n)
                    ref.frame85(f / "word_beta.parquet", wb)
                    ref.csv85(f / "calibration.csv", pd.DataFrame(cal))
                    ref.receipt85(f, identity, ["word_beta.parquet", "calibration.csv"],
                                  dict(network=n, n_splits=51, n_betas=98, n_classifier_seeds=10))
                    ctx["counts40"][n] = wb
                    ctx["source40"][n] = "COMPLETION_AUTHENTICATED"
            result = report(ctx)
            writej(ctx["root"] / "execution_status.json", dict(status="COMPLETE", run_id=ctx["id"]))
            return result
        except BaseException as exc:
            writej(ctx["root"] / "execution_status.json", dict(status="TECHNICAL_PAUSE", resumable=True,
                error=repr(exc), scientific_failure=False, run_id=ctx["id"]))
            raise


def _direction(row):
    t1, t2, t3 = row["T1"] > 0, row["T2"] < 0, row["T3"] < 0
    return dict(T1_expected=bool(t1), T2_expected=bool(t2), T3_expected=bool(t3),
                T12_expected=bool(t1 and t2), ALL3_expected=bool(t1 and t2 and t3))


def report(ctx):
    out = ctx["root"] / "report"
    out.mkdir(exist_ok=True)
    full, loco, status, contribution, concentration, curves, calibration = [], [], [], [], [], [], []
    cases = {}
    for K in (40, 50):
        if K == 40:
            g = ctx["g"]
        else:
            if not (ctx["paths"]["audit_root"] / "K50/geometry/completed.json").is_file():
                for n in CORE3:
                    status.append(dict(K=50, network=n, status="NOT_AVAILABLE", reason="Falta geometría K50"))
                continue
            g = ref._load_geometry(ctx, 50)
            _pin_receipt(ctx, g["folder"], g["signature"])
        for n in CORE3:
            d = ctx["counts40"].get(n) if K == 40 else _historical_counts(ctx, g, n)
            if d is None:
                status.append(dict(K=K, network=n, status="NOT_AVAILABLE", reason="Sin conteos completos"))
                continue
            a = ref.assess90(ctx, g, n, d, "EMPIRICAL_POSTHOC_K40_COMPLETION")
            scope = out / f"K{K}" / n
            ref.save_assessment90(scope, a)
            # Evidencia compacta suficiente para reconstruir todas las superficies.
            ref.frame85(scope / "word_beta.parquet", d)
            source_folder = (ctx["root"] / "counts" / n
                if K == 40 and ctx["source40"][n] == "COMPLETION_AUTHENTICATED"
                else g["folder"].parent / "counts" / n)
            writej(scope / "source_receipt.json", readj(source_folder / "completed.json"))
            writej(scope / "source_binding.json", dict(source_folder=str(source_folder),
                source_count_sha256=sha(source_folder / "word_beta.parquet"),
                exported_count_sha256=sha(scope / "word_beta.parquet"),
                transformation="validate, canonical beta keys, sort class/word/beta, serialize",
                source_receipt_sha256=sha(source_folder / "completed.json")))
            for name in ("full", "common", "expected"):
                ref.frame85(out / f"K{K}" / "geometry" / (name + ".parquet"), g[name])
            writej(out / f"K{K}" / "geometry/axis.json", dict(edges=g["edges"], centers=g["centers"],
                source_geometry_identity=g["signature"]))
            ll = a["loco"].copy()
            for key in ("T1", "T2", "T3"):
                ll[key + "_expected"] = ll[key].gt(0) if key == "T1" else ll[key].lt(0)
            ll["T12_expected"] = ll.T1_expected & ll.T2_expected
            ll["K"] = K
            loco.append(ll)
            row = dict(a["row"], **_direction(a["row"]))
            for key in ("T1", "T2", "T3", "T12", "ALL3"):
                field = key + "_expected"
                row["LOCO_" + key + "_passed"] = int(ll[field].sum())
                row["robust_" + key] = bool(row[field] and ll[field].all())
            row["n_LOCO"] = 8
            full.append(row)
            for key in ("T1", "T2", "T3"):
                values = ll[key].to_numpy()
                sign = 1 if key == "T1" else -1
                row["worst_LOCO_" + key + "_margin"] = float(np.min(sign * values))
            contribution.append(a["words"].assign(K=K, network=n))
            concentration.append(a["concentration"].assign(K=K, network=n))
            s = a["surface"]
            x = s.index.to_numpy(float); x = (x-x.mean()) / np.square(x-x.mean()).sum()
            curves.append(pd.DataFrame(dict(K=K, network=n, beta=ref.BETAS,
                mean_FAR=s.to_numpy().mean(axis=0), semantic_slope=x @ s.to_numpy())))
            cases[K, n] = s
            status.append(dict(K=K, network=n, status="COMPLETE",
                origin=ctx["source40"][n] if K == 40 else "HISTORICAL_AUTHENTICATED"))
            if n in HFY:
                cal = ([_own_cal(ctx, n, seed) for seed in range(51)]
                    if K == 40 and ctx["source40"][n] == "COMPLETION_AUTHENTICATED"
                    else [_historical_cal(ctx, g, n, seed) for seed in range(51)])
                req(all(v is not None for v in cal), "Falta trazabilidad de escalas.")
                calibration.extend(dict(v, K=K) for v in cal)
    full_df = pd.DataFrame(full)
    loco_df = pd.concat(loco, ignore_index=True)
    req(set(full_df.loc[full_df.K.eq(40), "network"]) == set(CORE3), "K40 CORE3 incompleto.")
    tables = dict(FULL=full_df, LOCO=loco_df, availability=pd.DataFrame(status),
        contributions=pd.concat(contribution, ignore_index=True), concentration=pd.concat(concentration, ignore_index=True),
        curves=pd.concat(curves, ignore_index=True), calibration=pd.DataFrame(calibration))
    tables["class_contributions"] = tables["contributions"].groupby(
        ["K", "network", "true_class", "class_label"], as_index=False)[
        ["T1_contribution", "T2_contribution", "T3_contribution"]].sum()
    warning_rows = []
    for n in HFY:
        for stage in ("calibration", "blocks"):
            for path in (ctx["root"] / stage / n).glob("**/warnings.json"):
                warning_rows.extend(dict(r, network=n, stage=stage) for r in readj(path))
    warnings_df = pd.DataFrame(warning_rows, columns=["network", "stage", "category", "message", "count"])
    tables["warnings_new_computation"] = (warnings_df.groupby(
        ["network", "stage", "category", "message"], as_index=False)["count"].sum() if len(warnings_df) else warnings_df)
    deltas = []
    for n in CORE3:
        p = full_df.loc[full_df.network.eq(n)].set_index("K")
        if {40, 50} <= set(p.index):
            deltas.append(dict(network=n, **{key + "_K50_minus_K40": float(p.loc[50, key] - p.loc[40, key]) for key in ("T1", "T2", "T3")}))
    tables["K50_minus_K40"] = pd.DataFrame(deltas)
    for name, table in tables.items():
        ref.csv85(out / (name + ".csv"), table)
    _plot(ctx, tables)
    comparison_complete = len(full_df) == 6
    summary = dict(run_id=ctx["id"], status="COMPLETE_K40_CORE3", K50_comparison_complete=comparison_complete,
        scope="W2V RAW, empirical 5-bin FULL and eight class omissions",
        K_chosen=False, model_fitted=False, structural_rule_inferred=False,
        T3_failure_networks_K40=full_df.loc[full_df.K.eq(40) & ~full_df.T3_expected, "network"].tolist(),
        LOCO_note="Descriptive class influence; retained counts, fixed axis and bins. Not a rerun without a class.",
        comparisons_note="K40 vs K50 changes memories, query inventory, splits and semantic geometry jointly.")
    writej(out / "summary.json", summary)
    writej(out / "protocol.json", ctx["protocol"])
    writej(out / "source_ledger.json", ctx["protected"])
    (out / "LEEME_RESULTADOS.txt").write_text(
        "FULL.csv: T1/T2/T3 y número de omisiones que conservan cada signo (0..8).\n"
        "LOCO.csv: todos los valores por clase omitida. robust_X exige FULL y 8/8.\n"
        "T3_expected=False significa T3>=0; no significa que T1/T2 fallen.\n"
        "No se ajustó un modelo ni se usó GRID30/MATCHED5: esta auditoría es empírica.\n"
        "K50 sólo se lee; cualquier ausencia aparece como NOT_AVAILABLE, nunca FAIL.\n"
        "Los gráficos muestran tendencias globales, sin imponer monotonía local.\n"
        "warnings_new_computation.csv cubre los cálculos nuevos; no acredita ausencia de advertencias históricas.\n"
        "La comparación es diagnóstica y no demuestra por sí misma una insuficiencia estructural.\n",
        encoding="utf-8")
    # Proteger de cambios concurrentes también las fuentes utilizadas al preparar.
    for p, h in ctx["protected"].items():
        req(Path(p).is_file() and sha(p) == h, "Fuente modificada durante la ampliación: " + p)
    return tables


def _plot(ctx, tables):
    import matplotlib.pyplot as plt
    c, l, f = tables["curves"], tables["LOCO"], tables["FULL"]
    for n in CORE3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for K, color in ((40, "#762a83"), (50, "#087e8b")):
            d = c.loc[c.network.eq(n) & c.K.eq(K)]
            if len(d):
                axes[0].plot(d.beta, d.semantic_slope, color=color, label=f"K{K}")
                axes[1].plot(d.beta, d.mean_FAR, color=color, label=f"K{K}")
        axes[0].axhline(0, color=".6", linewidth=.8)
        axes[0].set(ylabel="Pendiente FAR ~ Z dentro de clase", title="Curva cuya pendiente en beta define T3")
        axes[1].set(ylabel="FAR medio sobre los cinco bins", title="Curva cuya pendiente define T2")
        for ax in axes:
            ax.set_xlabel("Beta nominal"); ax.grid(alpha=.15); ax.legend()
        fig.suptitle("W2V RAW · " + ref.NAMES85[n] + " · empírico FULL")
        fig.tight_layout(); fig.savefig(ctx["root"] / "report" / (n + "_curves.png"), dpi=150); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, key in zip(axes, ("T1", "T2", "T3")):
            for K, color in ((40, "#762a83"), (50, "#087e8b")):
                full = f.loc[f.network.eq(n) & f.K.eq(K)]
                d = l.loc[l.network.eq(n) & l.K.eq(K)].sort_values("omitted_class")
                if len(full) == 1 and len(d) == 8:
                    ax.plot(range(9), [full.iloc[0][key], *d[key]], marker="o", color=color, label=f"K{K}")
            ax.axhline(0, color=".4", linewidth=.8)
            ax.set_title(key + (" > 0" if key == "T1" else " < 0"))
            ax.set_xticks(range(9), ["FULL", *ref.CLASSES85], rotation=55, ha="right")
            ax.grid(alpha=.15)
        axes[0].legend(); fig.suptitle(ref.NAMES85[n] + " · FULL y LOCO empírico (clase omitida)")
        fig.tight_layout(); fig.savefig(ctx["root"] / "report" / (n + "_LOCO.png"), dpi=150); plt.close(fig)


def export_results(ctx):
    root = ctx["root"]
    req((root / "report/summary.json").is_file(), "Ejecuta run antes de exportar resultados finales.")
    files = sorted((root / "report").rglob("*"))
    files = [p for p in files if p.is_file()]
    for name in ("protocol.json", "resources.csv", "execution_status.json"):
        if (root / name).is_file():
            files.append(root / name)
    target = root / ("E2_K40_CORE3_resultados_" + ctx["id"][:12] + ".zip")
    temp = target.with_suffix(".partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        hashes = {}
        for p in files:
            rel = p.relative_to(root).as_posix()
            z.write(p, rel); hashes[rel] = sha(p)
        z.writestr("EXPORT_SHA256.json", json.dumps(hashes, indent=2))
        for p in (PACKAGE / "e2_k40_hfy.py", PACKAGE / "e2_k40_reference.py", PACKAGE / "CODE_SHA256.json"):
            z.write(p, "code/" + p.name)
    temp.replace(target)
    print("Envíame este ZIP:", target)
    return target


def request_stop(ctx):
    (ctx["root"] / "STOP").write_text("Stop after active checkpoints\n", encoding="utf-8")


def clear_stop(ctx):
    (ctx["root"] / "STOP").unlink(missing_ok=True)
