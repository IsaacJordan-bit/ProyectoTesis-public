# Word2Vec Google News · 300 dimensiones

## Reproducción

Desde esta carpeta, con Python 3.11 o 3.12 de 64 bits:

```bash
python -m pip install -r requirements.txt
python descargar_word2vec.py
python reconstruir_word2vec_definitivo.py
```

Si ya tienes `GoogleNews-vectors-negative300.bin`, cópialo a `datos/` y omite la
descarga, o indica su ubicación:

```bash
python reconstruir_word2vec_definitivo.py --bin "ruta/al/GoogleNews-vectors-negative300.bin"
```

Salida: `salida/w2v_google_news_top3000000_relaxed_lowercase_clean.pkl` y su
archivo de verificación JSON. `--salida` cambia el destino; `--solo-verificar`
evita escribir; `--sobrescribir` autoriza reemplazar una salida anterior.
`--mapeo` acepta un CSV o CSV gzip con la selección completa de referencia.

## Qué se reproduce

`seleccion_vocabulario.csv.gz` conserva íntegramente las 653.346 filas del CSV
de correspondencias del dataset definitivo. No es una muestra. Contiene
`clean_word`, `source_word`, `source_index`, `is_exact_lowercase` y
`selection_reason`; el script lo lee directamente sin descomprimirlo en disco.

El código:

1. Comprueba la huella lógica del mapeo y sus posiciones crecientes.
2. Recorre el binario de Google News de 3.000.000 × 300; para cada posición
   seleccionada exige que coincida el token original.
3. Recupera su vector float32 y aplica la etiqueta final conservada en el CSV.
4. Comprueba todos los nombres, su orden y todos los valores de la matriz.
5. Exporta el PKL y vuelve a leerlo para verificar su contenido.

Resultado: **653.346 × 300**, `float32`, índice `word`, columnas `0` a `299`.
No se entrenan embeddings, no se promedian vectores y no se aplica L2.
El criterio definitivo es la identidad del índice ordenado y los vectores;
una exportación equivalente del binario es válida si supera estas comprobaciones.

## Archivos de referencia

- `referencia_dataset.json`: identidad del dataset y serialización de huellas.
- `metadata_dataset.json`: configuración documentada del dataset de referencia.
- `informe_limpieza.csv`: contadores aportados con el dataset.
- `verificacion_reconstruccion_aportada.json`: registro conservado de la
  reconstrucción completa desde la distribución de Google News.

El mapeo contiene 153.070 formas originalmente minúsculas y 500.276 convertidas.
Los contadores `kept_*` del informe corresponden al proceso de selección y
difieren de esa composición final en los 24.444 reemplazos registrados.

Esta implementación es el código de reproducción de referencia. La selección
se reproduce mediante el mapeo, sin volver a decidir los filtros ni los desempates.
No es necesario recuperar otro limpiador para generar el mismo dataset.

## Uso del resultado

```python
import pandas as pd
w2v_df = pd.read_pickle(
    "salida/w2v_google_news_top3000000_relaxed_lowercase_clean.pkl"
)
```

El dataset completo sirve tanto para K40 como para K50. K controla la selección
posterior de vecinos, no cambia el archivo de embeddings base.
