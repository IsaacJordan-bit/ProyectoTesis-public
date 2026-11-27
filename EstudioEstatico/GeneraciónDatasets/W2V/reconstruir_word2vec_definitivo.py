"""Reproduce el dataset Word2Vec de la tesis mediante el mapeo completo incluido."""
import argparse
import csv
import gzip
import hashlib
import json
import mmap
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATASET = "w2v_google_news_top3000000_relaxed_lowercase_clean"
FORMA = (653346, 300)
ENTRADAS_ORIGINALES = 3000000
SHA_MAPEO = "1f0672692faab63b3d9fc49431a3ced3369a6e4e3d606d9c91e068b9c22d06af"
SHA_INDICE = "28eaaf7828a2dac985708acc71417a41c3a00e83a3b645a4c986b4f3e8b20b19"
SHA_VECTORES = "c03107624c7bdd21453b1dbf01f3cc74b9de5d3b125c56fd0b230dc464781ef1"


def leer_mapeo(ruta):
    palabras, origenes, posiciones = [], [], []
    huella = hashlib.sha256()
    abrir = gzip.open if ruta.suffix == ".gz" else open
    with abrir(ruta, "rt", encoding="utf-8-sig", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            palabra, origen = fila["clean_word"], fila["source_word"]
            posicion = int(fila["source_index"])
            if posicion < 0 or (posiciones and posicion <= posiciones[-1]):
                raise ValueError("Las posiciones del mapeo deben ser crecientes y no negativas.")
            huella.update(f"{posicion}\t{origen}\t{palabra}\n".encode("utf-8"))
            palabras.append(palabra)
            origenes.append(origen.encode("utf-8"))
            posiciones.append(posicion)
    if len(palabras) != FORMA[0] or huella.hexdigest() != SHA_MAPEO:
        raise ValueError("El mapeo de selección difiere de la referencia.")
    return palabras, origenes, posiciones


def extraer_vectores(ruta, origenes, posiciones):
    dimensiones = FORMA[1]
    ancho = dimensiones * np.dtype("<f4").itemsize
    matriz = np.empty((len(posiciones), dimensiones), dtype="<f4")
    with ruta.open("rb") as archivo:
        cabecera = archivo.readline(128).split()
        if cabecera != [str(ENTRADAS_ORIGINALES).encode(), str(dimensiones).encode()]:
            raise ValueError("Se esperaba el binario Word2Vec de 3000000 × 300.")
        cursor = archivo.tell()
        with mmap.mmap(archivo.fileno(), 0, access=mmap.ACCESS_READ) as datos:
            fila = 0
            for posicion in range(posiciones[-1] + 1):
                espacio = datos.find(b" ", cursor)
                fin = espacio + 1 + ancho
                if espacio < 0 or fin > len(datos):
                    raise ValueError(f"Registro incompleto en la posición {posicion}.")
                if posicion == posiciones[fila]:
                    token = datos[cursor:espacio].lstrip(b"\r\n")
                    if token != origenes[fila]:
                        raise ValueError(f"Token de origen distinto en la posición {posicion}.")
                    matriz[fila] = np.frombuffer(datos, dtype="<f4", count=dimensiones,
                                                offset=espacio + 1)
                    fila += 1
                cursor = fin
    return matriz


def verificar_contenido(tabla):
    if tabla.shape != FORMA or tabla.index.name != "word":
        raise ValueError("Dimensiones o nombre del índice distintos de la referencia.")
    if not tabla.columns.equals(pd.RangeIndex(FORMA[1])):
        raise ValueError("Las columnas deben estar numeradas de 0 a 299.")
    if any(tipo != np.dtype("float32") for tipo in tabla.dtypes):
        raise ValueError("Se esperaban vectores float32.")
    indice, vectores = hashlib.sha256(), hashlib.sha256()
    for palabra in tabla.index:
        indice.update((palabra + "\n").encode("utf-8"))
    matriz = tabla.to_numpy(copy=False)
    for inicio in range(0, len(tabla), 4096):
        bloque = np.ascontiguousarray(matriz[inicio:inicio + 4096], dtype="<f4")
        if not np.isfinite(bloque).all():
            raise ValueError("La matriz contiene valores no finitos.")
        vectores.update(bloque)
    if indice.hexdigest() != SHA_INDICE or vectores.hexdigest() != SHA_VECTORES:
        raise ValueError("El índice ordenado o los vectores difieren del dataset de referencia.")
    return {"forma": list(FORMA), "dtype": "float32",
            "sha256_indice": indice.hexdigest(), "sha256_vectores": vectores.hexdigest(),
            "sha256_mapeo_logico": SHA_MAPEO}


def guardar(tabla, ruta, sobrescribir=False):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists() and not sobrescribir:
        raise FileExistsError(f"Ya existe {ruta}. Usa --solo-verificar o --sobrescribir.")
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(dir=ruta.parent, delete=False) as archivo:
            temporal = Path(archivo.name)
        tabla.to_pickle(temporal, protocol=5)
        comprobar = pd.read_pickle(temporal)
        verificar_contenido(comprobar)
        os.replace(temporal, ruta)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path,
                        default=BASE / "datos/GoogleNews-vectors-negative300.bin")
    parser.add_argument("--mapeo", type=Path, default=BASE / "seleccion_vocabulario.csv.gz")
    parser.add_argument("--salida", type=Path, default=BASE / "salida" / f"{DATASET}.pkl")
    parser.add_argument("--solo-verificar", action="store_true")
    parser.add_argument("--sobrescribir", action="store_true")
    args = parser.parse_args()
    if not args.bin.is_file():
        parser.error(f"Falta {args.bin}. Ejecuta descargar_word2vec.py o indica --bin.")
    palabras, origenes, posiciones = leer_mapeo(args.mapeo)
    matriz = extraer_vectores(args.bin, origenes, posiciones)
    tabla = pd.DataFrame(matriz, index=pd.Index(palabras, name="word"), copy=False)
    informe = verificar_contenido(tabla)
    informe.update({"dataset": DATASET, "python": platform.python_version(),
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
    print("Verificado: 653.346 palabras × 300 dimensiones; índice y vectores idénticos.")


if __name__ == "__main__":
    main()
