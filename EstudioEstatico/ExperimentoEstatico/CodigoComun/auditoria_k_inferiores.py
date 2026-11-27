"""Verifica K20/K30 sin leer errores: rechazos de partición y cobertura COMMON fija5."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import reutilizacion as reuse

LABELS = ['apple','bicycle','bread','canoe','chair','jellyfish','screwdriver','skunk']
GLOVE_ROOT = 'GloVe/search_5e904ca15f3dcc8dd6d7'
W2V_ROOT = 'W2V/audit_f4766816e364f73e7b47'


def _neighbors(n,k,n40):
    reuse._check(list(n)==LABELS,'Catálogo de clases distinto')
    words=[w for c in LABELS for w in n[c]]
    reuse._check(all(len(n[c])==k and n[c]==n40[c][:k] for c in LABELS),'Vecinos no corresponden al prefijo K40')
    reuse._check(len(set(words))==len(words) and not set(words)&set(LABELS),'Vecinos repetidos/centros incluidos')
    return words


def auditar_inferiores(repo):
    root=reuse._archive(repo,'E2_Geometrias_K20_K30.zip')
    nfiles=reuse._verify_manifest(root,'SHA256.json')
    rows=[];details={};cells=[]
    for dataset,rel in [('GloVe',GLOVE_ROOT),('W2V',W2V_ROOT)]:
        source=root/rel;protocol=reuse.json_read(source/'protocol_frozen.json')
        pid=reuse.exp.digest85(protocol)
        reuse._check(source.name.endswith(pid[:20]),'Identidad del protocolo geométrico distinta')
        reuse._check(protocol['dimension']==(50 if dataset=='GloVe' else 300),'Dimensión incompatible')
        n40=reuse.json_read(source/'K40/geometry/neighbors.json')
        for k in (20,30):
            folder=source/f'K{k}/geometry'
            row=dict(dataset=dataset,K=k,n_full=None,n_common=None,min_words_cell=None,empty_cells=None,
                     five_bin_geometry_admissible=False,estado='RECHAZADA',usa_errores=False)
            if dataset=='GloVe' and k==20:
                failure=reuse.json_read(folder/'structural_failure.json')
                words=_neighbors(failure['neighbors'],k,n40)
                reuse._check(failure['A1'] is False and failure['K']==20 and failure['failed_split']==0,'Fallo K20 distinto')
                with np.load(root/'reconstruccion/glove_K20_vectores.npz',allow_pickle=False) as z:
                    names=z['words'].tolist();vectors=z['vectors']
                reuse._check(names==words and vectors.shape==(160,50),'Vectores K20 incompatibles')
                fingerprint=hashlib.sha256(json.dumps(names,ensure_ascii=False).encode()+vectors.astype('<f8').tobytes()).hexdigest()
                reuse._check(fingerprint==failure['embedding_sha256'],'Vectores K20 distintos del registro de rechazo')
                # Sólo la partición geométrica; no se evalúa Hopfield ni un clasificador.
                split=reuse.exp.prepare_classification_split(pd.DataFrame(vectors,index=names),failure['neighbors'],
                    pairwise_different_class=True,frequency_threshold=2,validation_split=.33,random_seed_split=0)
                observed=split.drop_info
                saved=reuse.json_read(root/'reconstruccion/glove_K20_reconstruccion.json')['drop_info']
                for name in ('validation_classes_final','dropped_validation_words','n_validation_after_filter','n_validation_before_filter'):
                    reuse._check(observed[name]==saved[name],'Rechazo GloVe K20 no reproducido: '+name)
                reuse._check(len(observed['validation_classes_final'])<8 and bool(observed['dropped_validation_words']),'K20 no reproduce el rechazo A1')
                row.update(failed_split=0,motivo='A1: split 0 pierde skunk y descarta 2 palabras',evidencia='Partición reproducida con 160 vectores autenticados')
                details['GloVe_K20']=observed
            elif dataset=='W2V':
                gate=reuse.json_read(folder/'gate.json')
                receipt=reuse.exp._receipt(folder,reuse.exp.digest85(dict(parent=pid,K=k,phase='geometry')))
                reuse._check(receipt['summary']==gate,'Gate/recibo W2V incompatibles')
                _neighbors(reuse.json_read(folder/'neighbors.json'),k,n40)
                audits=reuse.json_read(folder/'split_audit.json');seed=gate['failed_split']
                reuse._check(gate['A1'] is False and gate['A'] is False and gate['K']==k,'Rechazo estructural W2V inválido')
                reuse._check([a['split_seed'] for a in audits]==list(range(seed+1)),'Secuencia de particiones incompleta')
                reuse._check(all(a['complete'] for a in audits[:-1]),'Fallo anterior no documentado')
                a=audits[-1];counts={int(c):v for c,v in a['validation_counts_before_filter'].items()}
                unsupported={c for c in range(8) if counts.get(c,0)<3}
                missing=set(range(8))-set(a['validation_classes_final'])
                reuse._check(a['complete'] is False and unsupported and unsupported<=missing and bool(a['dropped_validation_words']),
                             'No hay evidencia del fallo de frecuencia/clases')
                reuse._check(a['effective_min_class_frequency']==3 and a['pairwise_different_class'] is True,'Otro contrato de partición')
                reuse._check(a['n_validation_before_filter']-a['n_validation_after_filter']==len(a['dropped_validation_words']),'Descartes inconsistentes')
                row.update(failed_split=seed,motivo=f'A1: split {seed} pierde categorías y descarta {len(a["dropped_validation_words"])} palabras',
                           evidencia='Auditoría de particiones y recibos autenticados')
                details[f'W2V_K{k}']=a
            else:
                # GloVe K30 histórico llegó a aceptar cuatro bins. Aquí se conservan CINCO.
                receipt=reuse.json_read(folder/'geometry_identity.json')
                for name,h in receipt['outputs'].items():
                    p=(folder/name).resolve()
                    reuse._check(p.is_relative_to(folder.resolve()) and p.is_file() and reuse.sha256(p)==h,'Geometría K30 alterada')
                neighbors=reuse.json_read(folder/'neighbors.json');_neighbors(neighbors,k,n40)
                owners={w:c for c,l in enumerate(LABELS) for w in neighbors[l]}
                full=pd.read_parquet(folder/'full_words.parquet');expected=pd.read_parquet(folder/'expected_splits.parquet')
                reuse._check(not full.word.duplicated().any() and not expected.duplicated(['split_seed','word']).any(),'Palabras/splits duplicados')
                reuse._check(set(expected.split_seed)==set(range(51)) and set(full.word)==set(expected.word),'FULL no coincide con la unión de validaciones')
                reuse._check(expected.true_class.eq(expected.word.map(owners)).all(),'Clases inconsistentes')
                cols=['word','true_class','validation_similarity']
                observed=expected[cols].drop_duplicates().sort_values('word').reset_index(drop=True)
                pd.testing.assert_frame_equal(observed,full[cols].sort_values('word').reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-12)
                splits=pd.read_csv(folder/'split_structure.csv')
                reuse._check(len(splits)==51 and splits.complete.astype(str).str.lower().eq('true').all(),'Particiones K30 incompletas')
                for _,part in expected.groupby('split_seed'):
                    train=set(owners)-set(part.word)
                    tc=pd.Series([owners[w] for w in train]).value_counts().reindex(range(8),fill_value=0)
                    qc=part.true_class.value_counts().reindex(range(8),fill_value=0)
                    reuse._check(tc.ge(3).all() and qc.ge(3).all(),'Frecuencia mínima K30 no satisfecha')
                stats=full.groupby('true_class').validation_similarity.agg(mean='mean',sd=lambda x:x.std(ddof=0))
                reuse._check(set(stats.index)==set(range(8)) and stats.sd.gt(1e-14).all(),'Estandarizador inválido')
                z=(full.validation_similarity-full.true_class.map(stats['mean']))/full.true_class.map(stats.sd)
                reuse._check(np.allclose(z,full.similarity_wc_z,rtol=0,atol=1e-12),'Z K30 diferente')
                limits=full.groupby('true_class').similarity_wc_z.agg(['min','max'])
                lo=float(limits['min'].max());hi=float(limits['max'].min())
                reuse._check(hi>lo,'COMMON degenerado')
                edges=np.linspace(lo,hi,6)
                common=full.loc[full.similarity_wc_z.between(lo-1e-12,hi+1e-12)].copy()
                common['similarity_bin']=np.searchsorted(edges[1:-1],common.similarity_wc_z,side='right').clip(0,4)
                original=pd.read_parquet(folder/'common_Z.parquet')
                reuse._check(set(common.word)==set(original.word),'Inventario COMMON distinto')
                cov=pd.crosstab(common.true_class,common.similarity_bin).reindex(index=range(8),columns=range(5),fill_value=0)
                for c in range(8):
                    for b in range(5):cells.append(dict(dataset=dataset,K=k,true_class=c,label=LABELS[c],bin=b,n_words=int(cov.loc[c,b])))
                valid=bool(cov.ge(1).all().all())
                row.update(n_full=len(full),n_common=len(common),min_words_cell=int(cov.min().min()),empty_cells=int(cov.eq(0).sum().sum()),
                           common_lower=lo,common_upper=hi,five_bin_geometry_admissible=valid,estado='ADMISIBLE' if valid else 'RECHAZADA',
                           motivo='A4: cobertura recalculada con cinco bins fijos',evidencia='51 particiones, Z y COMMON reconstruidos')
                details['GloVe_K30']=dict(native_bins=len(reuse.json_read(folder/'gate.json')['edges'])-1,fixed_bins=5,
                    empty_cells=[dict(class_label=LABELS[c],bin=b) for c in range(8) for b in range(5) if cov.loc[c,b]==0])
            rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(cells),dict(archivos_verificados=nfiles,detalles=details,
        outcomes_read=False,models_fitted=False,classification_rerun=False),root/GLOVE_ROOT
