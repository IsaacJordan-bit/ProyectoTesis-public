"""Descarga la distribución GloVe 6B y conserva únicamente el TXT de 50 dimensiones."""
import argparse
import hashlib
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
SHA_TXT = "d8f717f8dd4b545cb7f418ef9f3d0c3e6e68a6f48b97d32f8b7aae40cb31f96f"
TXT_BYTES = 171350079


def sha256(ruta):
    with ruta.open("rb") as archivo:
        return hashlib.file_digest(archivo, "sha256").hexdigest()


def extraer(zip_path, destino):
    temporal = None
    try:
        with zipfile.ZipFile(zip_path) as archivo:
            entrada = archivo.getinfo("glove.6B.50d.txt")
            if entrada.file_size != TXT_BYTES:
                raise ValueError("Tamaño del TXT distinto de la referencia.")
            with archivo.open(entrada) as fuente, tempfile.NamedTemporaryFile(
                    dir=destino.parent, delete=False) as salida:
                temporal = Path(salida.name)
                while bloque := fuente.read(1024 * 1024):
                    salida.write(bloque)
        if sha256(temporal) != SHA_TXT:
            raise ValueError("La huella del TXT descargado no coincide con la referencia.")
        os.replace(temporal, destino)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo-zip", type=Path, help="Usar una distribución ya descargada.")
    parser.add_argument("--destino", type=Path, default=BASE / "datos/glove.6B.50d.txt")
    args = parser.parse_args()
    args.destino.parent.mkdir(parents=True, exist_ok=True)
    if args.destino.exists():
        if sha256(args.destino) != SHA_TXT:
            parser.error("Ya existe otro TXT en el destino. Usa --destino para elegir otra ruta.")
        print(f"TXT ya disponible y verificado: {args.destino.resolve()}")
        return
    if args.archivo_zip:
        extraer(args.archivo_zip, args.destino)
    else:
        print("Descargando glove.6B.zip (contiene también otras dimensiones)...", flush=True)
        with tempfile.TemporaryDirectory(dir=args.destino.parent) as carpeta:
            zip_path = Path(carpeta) / "glove.6B.zip"
            solicitud = urllib.request.Request(URL, headers={"User-Agent": "ProyectoTesis/1.1"})
            with urllib.request.urlopen(solicitud, timeout=120) as fuente, zip_path.open("wb") as salida:
                acumulado, siguiente = 0, 128 * 1024**2
                while bloque := fuente.read(1024 * 1024):
                    salida.write(bloque)
                    acumulado += len(bloque)
                    if acumulado >= siguiente:
                        print(f"Descargados {acumulado / 1024**2:.0f} MiB", flush=True)
                        siguiente += 128 * 1024**2
            extraer(zip_path, args.destino)
    print(f"TXT verificado: {args.destino.resolve()}")


if __name__ == "__main__":
    main()
