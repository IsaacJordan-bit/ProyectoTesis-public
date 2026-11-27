"""Carga portable, comprobación y reproducción desde ocurrencias guardadas."""
from pathlib import Path
import hashlib
import json
import os
import zipfile
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
N_WORDS, N_OCCURRENCES, DIM = 103029, 100, 768
N_PROTOTYPES = 50455
SELECTION_SHA = 'e130061a093ff98c3265cda041a3a86e77c6204c3509e0aa0a56405af87af087'
VOCAB_SHA = '95d87d7ce442dd36699ab0ff7bdd8bd12c10e5460f642f6ba4a514369e088edc'
EMBEDDING_SHA = '0590ffe496e1c0da00b399e4773d715f835b17ce2c0b0004ff19fcd3c7269593'
STEM = 'contextual_exp2_embeddings_rawmean100_wordnet_v1'
WORDS = sorted(['apple','olive','pear','bicycle','parachute','scooter','bread','cheese','peanut','canoe','bus','raft','chair','curtain','stool','jellyfish','ant','crab','screwdriver','sling','wrench','skunk','frog','raccoon'])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, na_filter=False)


def fingerprint(words, matrix):
    words = list(words)
    vh = hashlib.sha256(json.dumps(words, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    eh = hashlib.sha256()
    for word in words:
        eh.update(word.encode('utf-8')); eh.update(b'\0')
    array = np.ascontiguousarray(matrix, dtype='<f8')
    eh.update(memoryview(array).cast('B'))
    return vh, eh.hexdigest()


def verificar_prototipos(df):
    require(isinstance(df, pd.DataFrame), 'Los prototipos deben ser un DataFrame.')
    require(df.shape == (N_PROTOTYPES, DIM), 'Dimensiones inesperadas de prototipos.')
    require(df.index.is_unique and df.index.is_monotonic_increasing, 'Vocabulario duplicado o desordenado.')
    require(all(isinstance(w, str) for w in df.index), 'El vocabulario debe contener strings.')
    matrix = df.to_numpy()
    require(matrix.dtype == np.dtype('float64'), 'Se requieren los valores float64 originales.')
    require(np.isfinite(matrix).all(), 'Prototipos no finitos.')
    require((np.linalg.norm(matrix, axis=1) > 0).all(), 'Prototipo de norma cero.')
    vh, eh = fingerprint(df.index, matrix)
    require(vh == VOCAB_SHA, 'El vocabulario no coincide con el dataset definitivo.')
    require(eh == EMBEDDING_SHA, 'Los valores no coinciden con el dataset definitivo.')
    return {'shape': list(df.shape), 'vocabulary_sha256': vh, 'embedding_sha256': eh, 'status': 'VERIFIED'}


def cargar_prototipos(base=BASE):
    path = Path(base) / 'Datos/Prototipos' / (STEM + '.npz')
    with np.load(path, allow_pickle=False) as data:
        df = pd.DataFrame(data['embeddings'], index=pd.Index(data['words'].tolist(), name='word'), columns=data['columns'].tolist())
    verificar_prototipos(df)
    frozen = read_csv(Path(base) / 'Datos/Prototipos' / (STEM.replace('embeddings_rawmean100_', 'vocabulary_') + '.csv'))
    require(df.index.tolist() == frozen.word.tolist(), 'El CSV y el NPZ tienen vocabularios diferentes.')
    return df


def verificar_metadatos(base=BASE):
    base = Path(base)
    directory = base / 'Datos/Ocurrencias'
    vocab = read_csv(directory / 'eligible_vocabulary_min100.csv')
    index = read_csv(directory / 'word_row_group_index.csv')
    counts = read_csv(directory / 'word_counts.csv')
    require(len(vocab) == N_WORDS and vocab.word.is_unique and vocab.word.is_monotonic_increasing, 'Vocabulario maestro inválido.')
    require('nan' not in set(vocab.word) and 'null' in set(vocab.word), 'Se alteró el contrato léxico nan/null.')
    require(index.word.tolist() == vocab.word.tolist(), 'Índice incompatible con el vocabulario.')
    require(index.n_occurrences.eq(100).all(), 'El índice no declara 100 ocurrencias por palabra.')
    for column, expected in [('row_start', np.arange(N_WORDS)*100), ('row_end_exclusive', (np.arange(N_WORDS)+1)*100), ('row_group_id', np.arange(N_WORDS)//10), ('word_index',np.arange(N_WORDS))]:
        require(np.array_equal(index[column].to_numpy(), expected), 'Índice inválido: '+column)
    require(counts.word.tolist() == vocab.word.tolist() and counts.n_occurrences.eq(100).all(), 'Conteos guardados incompatibles.')
    meta = json.loads((directory/'contextual_dataset_meta.json').read_text(encoding='utf-8'))
    report = json.loads((directory/'validation_report.json').read_text(encoding='utf-8'))
    selection = json.loads((directory/'selection_report.json').read_text(encoding='utf-8'))
    require(meta['selection_sha256'] == SELECTION_SHA, 'Selección incorrecta en metadata.')
    require((selection.get('selection_sha256_final') or selection.get('selection_sha256')) == SELECTION_SHA, 'Selección incorrecta en reporte.')
    require(report['status'] == 'VALID' and report['n_rows'] == N_WORDS*N_OCCURRENCES, 'Reporte de validación incompatible.')
    return {'master_words':len(vocab),'expected_rows':N_WORDS*N_OCCURRENCES,'reports_and_index':'VERIFIED', 'occurrence_file_present':(directory/'contextual_occurrences_definitive.parquet').is_file(), 'occurrence_bytes_verified_by_this_function':False}


def _parquet_path(base, parquet):
    path = Path(parquet) if parquet is not None else Path(base)/'Datos/Ocurrencias/contextual_occurrences_definitive.parquet'
    if not path.is_file():
        raise FileNotFoundError('Copia el Parquet de ocurrencias a: '+str(path))
    return path


def _check_footer(pf):
    import pyarrow as pa
    require(pf.metadata.num_rows == N_WORDS*100, 'Número de ocurrencias inesperado.')
    require(pf.metadata.num_row_groups == (N_WORDS+9)//10, 'Número de row groups inesperado.')
    dtype = pf.schema_arrow.field('embedding').type
    require(pa.types.is_fixed_size_list(dtype) and dtype.list_size == DIM and dtype.value_type == pa.float32(), 'Schema de embeddings incompatible.')


def cargar_ocurrencias(words, base=BASE, parquet=None):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    words = sorted(set(words))
    require(bool(words) and all(isinstance(w,str) for w in words), 'Indica palabras como strings.')
    index = read_csv(Path(base)/'Datos/Ocurrencias/word_row_group_index.csv')
    require(set(words) <= set(index.word), 'Hay palabras ausentes del vocabulario.')
    groups = sorted(index.loc[index.word.isin(words),'row_group_id'].astype(int).unique())
    path = _parquet_path(base,parquet)
    pf = pq.ParquetFile(path)
    try:
        _check_footer(pf)
        pieces = []
        for group in groups:
            require(pf.metadata.row_group(int(group)).num_rows <= 1000, 'Row group demasiado grande.')
            table = pf.read_row_group(int(group))
            pieces.append(table.filter(pc.is_in(table['word'], value_set=pa.array(words))))
        table = pa.concat_tables(pieces).sort_by([('word','ascending'),('occ_id','ascending')])
    finally:
        pf.close()
    counts = pd.Series(table['word'].to_pylist()).value_counts()
    require(set(counts.index) == set(words) and counts.eq(100).all(), 'No se recuperaron 100 ocurrencias por palabra.')
    x = table['embedding'].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(-1,DIM)
    require(np.isfinite(x).all() and (np.linalg.norm(x,axis=1)>0).all(), 'Embeddings inválidos en la muestra.')
    return table


def verificar_muestra(base=BASE, parquet=None, output=None):
    import pyarrow.parquet as pq
    table = cargar_ocurrencias(WORDS,base,parquet)
    df = cargar_prototipos(base)
    words = table['word'].to_pylist()
    require(words == [w for w in WORDS for _ in range(100)], 'Orden incorrecto de la muestra.')
    x = table['embedding'].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(WORDS),100,DIM)
    means = x.astype(np.float64).mean(axis=1)
    require(set(WORDS) <= set(df.index), 'Faltan palabras experimentales en los prototipos.')
    require(np.array_equal(means, df.loc[WORDS].to_numpy()), 'Las medias de la muestra no coinciden con los prototipos.')
    report = {'status':'VERIFIED_SAMPLE_ONLY','n_words':len(WORDS),'n_occurrences':table.num_rows,'prototype_means_exact':True,'full_occurrence_dataset_verified':False}
    if output is not None:
        import io
        path = Path(output); path.parent.mkdir(parents=True,exist_ok=True)
        buffer = io.BytesIO(); pq.write_table(table,buffer,compression='zstd')
        # No se sobrescribe una muestra anterior.
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
            z.writestr('muestra_24_palabras.parquet',buffer.getvalue())
            z.writestr('VERIFICACION_MUESTRA.json',json.dumps(report,indent=2))
    return report


def reconstruir_prototipos(base=BASE, parquet=None, output=None):
    """Lee el Parquet guardado; no carga BERT. Guarda solo tras verificar huellas."""
    import pyarrow.parquet as pq
    base = Path(base)
    vocab = read_csv(base/'Datos/Ocurrencias/eligible_vocabulary_min100.csv').word.tolist()
    kept = read_csv(base/'Datos/Prototipos/contextual_exp2_vocabulary_wordnet_v1.csv').word.tolist()
    require(len(vocab) == N_WORDS and len(kept) == N_PROTOTYPES, 'Vocabularios inesperados.')
    positions = {w:i for i,w in enumerate(kept)}
    result = np.empty((len(kept),DIM),dtype=np.float64)
    seen = set(); scanned = 0
    pf = pq.ParquetFile(_parquet_path(base,parquet))
    try:
        _check_footer(pf)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg,columns=['word','embedding'])
            require(table.num_rows % 100 == 0 and table.num_rows <= 1000, 'Row group incompatible.')
            start = scanned//100; n = table.num_rows//100
            words = vocab[start:start+n]
            require(table['word'].to_pylist() == [w for w in words for _ in range(100)], 'Bloques de palabras fuera de orden.')
            x = table['embedding'].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(n,100,DIM)
            require(np.isfinite(x).all() and (np.linalg.norm(x,axis=2)>0).all(), 'Ocurrencias no finitas o nulas.')
            means = x.astype(np.float64).mean(axis=1)
            for word, vector in zip(words,means):
                if word in positions:
                    result[positions[word]] = vector; seen.add(word)
            scanned += table.num_rows
            if rg % 1000 == 0:print(f'Row groups: {rg}/{pf.metadata.num_row_groups}',flush=True)
    finally:
        pf.close()
    require(scanned == N_WORDS*100 and seen == set(kept), 'Reconstrucción incompleta.')
    columns = [f'dim_{i:03d}' for i in range(DIM)]
    df = pd.DataFrame(result,index=pd.Index(kept,name='word'),columns=columns)
    verificar_prototipos(df)
    if output is not None:
        output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
        # No reemplaza las entradas congeladas.
        with output.open('xb') as f:
            np.savez_compressed(f,words=np.asarray(kept,dtype=str),columns=np.asarray(columns,dtype=str),embeddings=result)
    return df
