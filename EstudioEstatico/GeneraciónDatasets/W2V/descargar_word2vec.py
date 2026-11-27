"""Descarga y descomprime Google News de Gensim, verificando gzip y binario."""
import argparse
import gzip
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
URL = ("https://github.com/piskvorky/gensim-data/releases/download/"
       "word2vec-google-news-300/word2vec-google-news-300.gz")
MD5_GZ = "a5e5354d40acb95f9ec66d5977d140ef"
SHA_BIN = "6cb30ecf940501bdb0a0b2921562d4008ab3e461c342b44b48f801211f9c0f8e"
GZ_BYTES = 1743563840
BIN_BYTES = 3644258522


class LectorConHuella:
    """Registra todos los bytes comprimidos sin guardar una segunda copia en disco."""
    def __init__(self, archivo):
        self.archivo = archivo
        self.md5 = hashlib.md5(usedforsecurity=False)
        self.nbytes = 0

    def read(self, cantidad=-1):
        bloque = self.archivo.read(cantidad)
        self.md5.update(bloque)
        self.nbytes += len(bloque)
        return bloque


def descomprimir(fuente, destino):
    lector = LectorConHuella(fuente)
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(dir=destino.parent, delete=False) as salida:
            temporal = Path(salida.name)
            huella, acumulado, siguiente = hashlib.sha256(), 0, 512 * 1024**2
            with gzip.GzipFile(fileobj=lector, mode="rb") as comprimido:
                while bloque := comprimido.read(1024 * 1024):
                    acumulado += len(bloque)
                    if acumulado > BIN_BYTES:
                        raise ValueError("El binario excede el tamaño de referencia.")
                    salida.write(bloque)
                    huella.update(bloque)
                    if acumulado >= siguiente:
                        print(f"Descomprimidos {acumulado / 1024**3:.2f} GiB", flush=True)
                        siguiente += 512 * 1024**2
        if acumulado != BIN_BYTES or huella.hexdigest() != SHA_BIN:
            raise ValueError("El binario descargado no coincide con la referencia.")
        if lector.nbytes != GZ_BYTES or lector.md5.hexdigest() != MD5_GZ:
            raise ValueError("La distribución gzip no coincide con la referencia de Gensim.")
        os.replace(temporal, destino)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo-gz", type=Path, help="Usar el gzip ya descargado de Gensim.")
    parser.add_argument("--destino", type=Path,
                        default=BASE / "datos/GoogleNews-vectors-negative300.bin")
    args = parser.parse_args()
    args.destino.parent.mkdir(parents=True, exist_ok=True)
    if args.destino.exists():
        with args.destino.open("rb") as archivo:
            if hashlib.file_digest(archivo, "sha256").hexdigest() != SHA_BIN:
                parser.error("Ya existe otro binario en el destino. Usa --destino para elegir otra ruta.")
        print(f"Binario ya disponible y verificado: {args.destino.resolve()}")
        return
    if args.archivo_gz:
        with args.archivo_gz.open("rb") as fuente:
            descomprimir(fuente, args.destino)
    else:
        print("Descargando Google News; se descomprime durante la descarga...", flush=True)
        solicitud = urllib.request.Request(URL, headers={"User-Agent": "ProyectoTesis/1.1"})
        with urllib.request.urlopen(solicitud, timeout=120) as fuente:
            descomprimir(fuente, args.destino)
    print(f"Binario verificado: {args.destino.resolve()}")


if __name__ == "__main__":
    main()
