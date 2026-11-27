# GloVe 6B · 50 dimensiones

## Reproducción

Desde esta carpeta, con Python 3.11 o 3.12 de 64 bits:

```bash
python -m pip install -r requirements.txt
python descargar_glove.py
python reconstruir_glove_50d.py
```

Si ya tienes `glove.6B.50d.txt`, cópialo a `datos/` y omite la descarga, o usa:

```bash
python reconstruir_glove_50d.py --txt "ruta/al/glove.6B.50d.txt"
```

Salida: `salida/glove_6B_50D_definitivo.pkl` y su archivo de verificación JSON.
Los directorios se resuelven respecto de este script. `--salida` cambia el
destino; `--solo-verificar` evita escribir; `--sobrescribir` autoriza reemplazar
una salida previa. `python reconstruir_glove_50d.py --help` muestra las opciones.

## Qué se reproduce

1. Se comprueba el SHA-256 del TXT incluido en la referencia de la tesis.
2. Se lee con `pandas.read_csv(sep=" ", quoting=3, header=None, index_col=0,
   keep_default_na=True)`: 400.000 entradas y 50 componentes.
3. Se eliminan los índices ausentes; los demás se convierten a texto y se
   conserva la primera fila si hay índices repetidos.
4. Se obtienen **399.997 × 50 valores float64**. No se normalizan los vectores.
5. Se verifican el índice ordenado y todos los valores mediante SHA-256,
   además de las normas de referencia. Después de escribir el PKL, se vuelve
   a leer y se comprueba la igualdad exacta del DataFrame.

Las tres filas retiradas corresponden a los tokens literales `nan`, `null` y
`n/a`, interpretados como ausentes por el lector. El dataset conserva palabras
funcionales, signos y entradas numéricas; no recibe los filtros léxicos de W2V.
El índice se llama `0` y las columnas conservan las etiquetas `1` a `50`.

`referencia_dataset.json` contiene dimensiones, huellas y detalles de lectura.
La huella de contenido usa el índice serializado mediante
`json.dumps(lista, ensure_ascii=False)` en UTF-8, seguido por todos los valores
float64 little-endian en orden de filas. No es el hash de los bytes del PKL.

## Uso del resultado

```python
import pandas as pd
glove_df = pd.read_pickle("salida/glove_6B_50D_definitivo.pkl")
```

El TXT de `datos/` también sirve para los cargadores originales de los notebooks,
aplicando después la limpieza estructural descrita arriba. La selección de
vecinos y K se realiza en la etapa experimental, sobre este dataset base.
