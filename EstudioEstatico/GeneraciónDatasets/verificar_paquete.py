"""Comprueba los archivos versionados de esta entrega; no requiere los datasets grandes."""
import hashlib
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    manifiesto = json.loads((base / "MANIFIESTO_ARCHIVOS.json").read_text(encoding="utf-8"))
    errores = []
    for entrada in manifiesto["archivos"]:
        ruta = (base / entrada["ruta"]).resolve()
        if not ruta.is_relative_to(base):
            errores.append(f"Ruta fuera del paquete: {entrada['ruta']}")
            continue
        if not ruta.is_file():
            errores.append(f"Falta: {entrada['ruta']}")
            continue
        with ruta.open("rb") as archivo:
            huella = hashlib.file_digest(archivo, "sha256").hexdigest()
        if huella != entrada["sha256"] or ruta.stat().st_size != entrada["bytes"]:
            errores.append(f"Contenido diferente: {entrada['ruta']}")
    if errores:
        raise SystemExit("\n".join(errores))
    print(f"Integridad correcta: {len(manifiesto['archivos'])} archivos verificados.")
    print("Los datos fuente y PKL se verifican al ejecutar sus reconstructores.")


if __name__ == "__main__":
    main()
