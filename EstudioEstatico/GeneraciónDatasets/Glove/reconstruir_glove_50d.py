"""Reproduce y verifica el dataset GloVe 6B 50D utilizado en la tesis."""
import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ARCHIVO_SHA = "d8f717f8dd4b545cb7f418ef9f3d0c3e6e68a6f48b97d32f8b7aae40cb31f96f"
MATRIZ_SHA = "2ed0d282c625464365043d10405cf31064babdbbe3a6aee1095e2c0589165da1"
NORMAS = (0.04693154286091541, 4.474556674375726, 14.121561584771847)


def verificar_contenido(tabla):
    if tabla.shape != (399997, 50) or tabla.index.name != 0:
        raise ValueError("Forma o nombre del índice distintos de la referencia.")
    if list(tabla.columns) != list(range(1, 51)):
        raise ValueError("Se esperaban las columnas originales de 1 a 50.")
    if tabla.index.hasnans or tabla.index.has_duplicates:
        raise ValueError("El índice contiene valores ausentes o duplicados.")
    if any(tipo != np.dtype("float64") for tipo in tabla.dtypes):
        raise ValueError("Se esperaban valores float64.")
    matriz = tabla.to_numpy(dtype="<f8", copy=False)
    if not np.isfinite(matriz).all():
        raise ValueError("La matriz contiene valores no finitos.")
    huella = hashlib.sha256(
        json.dumps(tabla.index.tolist(), ensure_ascii=False).encode("utf-8")
    )
    for inicio in range(0, len(tabla), 10000):
        huella.update(matriz[inicio:inicio + 10000].tobytes(order="C"))
    if huella.hexdigest() != MATRIZ_SHA:
        raise ValueError("El índice ordenado o los vectores difieren del dataset de referencia.")
    normas = np.linalg.norm(matriz, axis=1)
    observadas = [float(normas.min()), float(normas.mean()), float(normas.max())]
    if not np.allclose(observadas, NORMAS, rtol=0, atol=1e-12):
        raise ValueError("Las normas de los vectores difieren de la referencia.")
    return {"forma": list(tabla.shape), "dtype": "float64",
            "sha256_indice_y_matriz": huella.hexdigest(), "normas_l2": observadas}


def cargar_glove(ruta):
    with ruta.open("rb") as archivo:
        if hashlib.file_digest(archivo, "sha256").hexdigest() != ARCHIVO_SHA:
            raise ValueError("El TXT no coincide con glove.6B.50d.txt de referencia.")
    tabla = pd.read_csv(ruta, sep=" ", quoting=3, header=None,
                        index_col=0, keep_default_na=True)
    if tabla.shape != (400000, 50):
        raise ValueError(f"Dimensiones de entrada inesperadas: {tabla.shape}")
    # Preservar el comportamiento que interpreta nan, null y n/a como índices NA.
    tabla = tabla.loc[tabla.index.notna()].copy()
    tabla.index = tabla.index.astype(str)
    tabla = tabla.loc[~tabla.index.duplicated(keep="first")]
    verificar_contenido(tabla)
    return tabla


def guardar(tabla, ruta, sobrescribir=False):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists() and not sobrescribir:
        raise FileExistsError(f"Ya existe {ruta}. Usa --solo-verificar o --sobrescribir.")
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(dir=ruta.parent, delete=False) as archivo:
            temporal = Path(archivo.name)
        tabla.to_pickle(temporal, protocol=5)
        # Comprobar también lo que realmente se ha serializado antes de publicarlo.
        comprobar = pd.read_pickle(temporal)
        pd.testing.assert_frame_equal(tabla, comprobar, check_exact=True)
        os.replace(temporal, ruta)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt", type=Path, default=BASE / "datos/glove.6B.50d.txt")
    parser.add_argument("--salida", "--guardar-pkl", dest="salida", type=Path,
                        default=BASE / "salida/glove_6B_50D_definitivo.pkl")
    parser.add_argument("--solo-verificar", action="store_true")
    parser.add_argument("--sobrescribir", action="store_true")
    args = parser.parse_args()
    if not args.txt.is_file():
        parser.error(f"Falta {args.txt}. Ejecuta descargar_glove.py o copia allí el TXT.")
    tabla = cargar_glove(args.txt)
    informe = verificar_contenido(tabla)
    informe.update({"sha256_txt": ARCHIVO_SHA, "python": platform.python_version(),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "indice_y_vectores_verificados": True})
    if not args.solo_verificar:
        guardar(tabla, args.salida, args.sobrescribir)
        informe["pkl_verificado_tras_lectura"] = True
        informe["archivo_salida"] = args.salida.name
        informe_path = args.salida.with_suffix(".verificacion.json")
        informe_path.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        print(f"Dataset: {args.salida.resolve()}")
        print(f"Verificación: {informe_path.resolve()}")
    print("Verificado: 399.997 entradas × 50 dimensiones; índice y vectores idénticos.")


if __name__ == "__main__":
    main()
