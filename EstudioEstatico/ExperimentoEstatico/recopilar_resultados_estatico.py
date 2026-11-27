#!/usr/bin/env python3
"""Comprime resultados existentes de GloVe y geometrías K20–K100; no ejecuta experimentos."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

GLOVE_BASE = ('Results/computador_secundario/'
    'GK60G_SELECTED_K_FULL98_GENERAL_GATE_VALIDATION_PC_PRINCIPAL/'
    '20261008T190301Z_6e346ce9/GK62G_K100_PILOT9/GK67G_K100_FULL98_Z/'
    'GK80G_FULL98_SEARCH')
SCALE = GLOVE_BASE + '/GK83G_HFY_SCALE_MATCHED/scale_d3399476d688141441a5'
MODEL = GLOVE_BASE + '/GK83G_HFY_SCALE_MATCHED/GK85G_MODEL_SEARCH/search_b77d2b37e15428d6bf13'
REFIT = MODEL + '/GK88G_SELECTED_FULL_LOCO/refit_addb067bc5bf230730d5'
W2V = 'Results/W2V/_PROTOCOL_RECTOR_V3/audit_f4766816e364f73e7b47'
GRID = tuple(range(20, 101, 10))


def compatible(value):
    s = os.path.abspath(Path(value).expanduser())
    if os.name == 'nt' and not s.startswith('\\\\?\\'):
        s = '\\\\?\\UNC\\' + s[2:] if s.startswith('\\\\') else '\\\\?\\' + s
    return Path(s).resolve()


def collect(root):
    selected = {}
    notices = []
    roles = []
    geometries = []

    def add(path):
        path = path.resolve()
        if path.is_file() and not path.is_symlink() and path.is_relative_to(root):
            selected[path.relative_to(root).as_posix()] = path

    def tree(path):
        if not path.is_dir(): return
        for folder, dirs, files in os.walk(path, onerror=lambda e: notices.append(str(e))):
            dirs[:] = [d for d in dirs if d not in {'__pycache__','.git'} and not (Path(folder)/d).is_symlink()]
            for name in files: add(Path(folder)/name)

    for role,rel in [('GloVe_GK83',SCALE),('GloVe_GK85',MODEL),('GloVe_GK88',REFIT)]:
        p=root/rel
        roles.append(dict(rol=role,ruta=rel,existe=p.is_dir()))
        tree(p)  # Deduplicado: GK88 está dentro del árbol GK85.
    # Diagnóstico geométrico y contratos, sin conteos del resto de corridas.
    search=root/'Results'
    skipped={'__pycache__','.git','checkpoints','fits','evaluation','figures','MODEL_SEARCH','GK85G_MODEL_SEARCH'}
    for folder, dirs, files in os.walk(search, onerror=lambda e: notices.append(str(e))):
        dirs[:] = [d for d in dirs if d not in skipped and not (Path(folder)/d).is_symlink()]
        p=Path(folder)
        if p.name!='geometry': continue
        match=re.fullmatch(r'K(\d+)',p.parent.name)
        if not match or int(match[1]) not in GRID: continue
        upper=p.as_posix().upper()
        if 'PROTOTIP' in upper: continue
        dataset='W2V' if '/W2V/' in upper else 'GloVe' if ('GLOVE' in upper or re.search(r'/GK\d+G_',upper)) else None
        if dataset is None: continue
        geometries.append(dict(dataset=dataset,K=int(match[1]),ruta=p.relative_to(root).as_posix()))
        tree(p)
        parent=p.parent
        while parent!=root and parent.is_relative_to(root):
            for item in parent.iterdir():
                if item.is_file() and item.suffix.lower() in {'.json','.csv'}: add(item)
            if parent==search: break
            parent=parent.parent
        dirs[:]=[]
    # Incluye estados de rechazo aunque no se haya escrito una geometría completa.
    for rel in (GLOVE_BASE,W2V):
        p=root/rel
        if p.is_dir():
            for item in p.iterdir():
                if item.is_file() and item.suffix.lower() in {'.json','.csv'}: add(item)
    report=dict(raiz_original=str(root),grilla=list(GRID),fuentes=roles,geometrias=geometries,
        geometrias_no_encontradas=[dict(dataset=d,K=k) for d in ('GloVe','W2V') for k in GRID
            if not any(g['dataset']==d and g['K']==k for g in geometries)],
        advertencias=notices,solo_lectura=True,nuevos_ajustes=0,nuevas_clasificaciones=0,
        nota='Localizar archivos no certifica su identidad científica; se verificará al integrarlos.')
    return selected,report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raiz',default=r'C:\Users\jisaa\Workspace\Tesis\Exp2def')
    ap.add_argument('--salida',default='PendientesEstudioEstatico_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip')
    args=ap.parse_args()
    root=compatible(args.raiz)
    if not (root/'Results').is_dir():
        ap.error('No existe Results en la raíz indicada. Usa --raiz con la ubicación del proyecto original.')
    output=compatible(args.salida)
    if output.exists(): ap.error('La salida ya existe; elige otro nombre con --salida.')
    selected,report=collect(root)
    size=sum(p.stat().st_size for p in selected.values())
    print(f'Encontrados {len(selected)} archivos, {size/2**20:.1f} MiB antes de comprimir.',flush=True)
    for row in report['fuentes']: print(row['rol'], 'ENCONTRADO' if row['existe'] else 'AUSENTE',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.recopilacion_',dir=output.parent)
    os.close(fd)
    tmp=Path(tmp)
    manifest=[]
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
            for i,(name,p) in enumerate(sorted(selected.items()),1):
                before=p.stat();h=hashlib.sha256();n=0
                with p.open('rb') as src, z.open('fuentes/'+name,'w',force_zip64=True) as dst:
                    while chunk:=src.read(1024*1024):
                        dst.write(chunk);h.update(chunk);n+=len(chunk)
                after=p.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or n!=before.st_size:
                    raise RuntimeError('Un archivo cambió durante la lectura: '+str(p))
                manifest.append(dict(archivo='fuentes/'+name,bytes=n,sha256=h.hexdigest()))
                if i%250==0: print(f'Comprimidos {i}/{len(selected)}',flush=True)
            z.writestr('INVENTARIO.json',json.dumps(report,ensure_ascii=False,indent=2))
            z.writestr('MANIFIESTO.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        tmp.rename(output)
    finally:
        if tmp.exists(): tmp.unlink()
    print('Archivo para adjuntar:',output,flush=True)
    print('Faltantes y advertencias se detallan en INVENTARIO.json. No se ejecutó el experimento.')


if __name__=='__main__': main()
