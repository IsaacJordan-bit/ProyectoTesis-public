#!/usr/bin/env python3
"""Comprueba integridad local. --w2v añade verificación de coeficientes, sin ajustes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2v", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "MANIFIESTO_ENTREGA.json").read_text(encoding="utf-8"))
    problems = []
    for name, expected in manifest["sha256"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            problems.append("Falta: " + name)
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            problems.append("Cambió: " + name)
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Integridad correcta: {len(manifest['sha256'])} archivos.")
    if args.w2v:
        sys.path.insert(0, str(root / "CodigoComun"))
        from reutilizacion import cargar_w2v_k40
        result = cargar_w2v_k40(root)
        print(json.dumps(result.verification, ensure_ascii=False, indent=2))
    else:
        print("Para cotejar también coeficientes W2V: python verificar_entrega.py --w2v")


if __name__ == "__main__":
    main()
