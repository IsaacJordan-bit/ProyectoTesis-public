# Generación de los datasets estáticos · etapa 1.1

Esta carpeta conserva las implementaciones de referencia para reproducir los
datasets de GloVe y Word2Vec utilizados en la tesis. El criterio de identidad es
el vocabulario ordenado y los valores de sus vectores. No se requiere que la
serialización PKL tenga los mismos bytes entre versiones de pandas.

| Carpeta | Dataset definitivo | Forma | Valores |
|---|---|---:|---|
| `Glove/` | GloVe 6B 50D | 399.997 × 50 | float64 |
| `W2V/` | Google News, selección relaxed-lowercase | 653.346 × 300 | float32 |

La etapa 1.1 de estos dos datasets queda cerrada. El TXT de GloVe y el binario
de Google News se obtienen por descarga o copia local. El mapeo completo de
W2V está incluido. Los datasets base son independientes de K: usar K40 o K50
selecciona diferentes vecinos del mismo vocabulario completo.

## Ubicación dentro del proyecto

La ruta de esta carpeta es `ProyectoTesis/Estudio estático/GeneraciónDatasets/`.
Sus subcarpetas `Glove/` y `W2V/` son independientes: cada una incluye su código,
dependencias, referencias e instrucciones de reproducción.

## Inicio rápido

Usar Python 3.11 o 3.12 de 64 bits. La entrega se comprueba con Python 3.12.14,
NumPy 2.3.5 y pandas 2.2.3; ambas carpetas fijan las mismas dependencias.
Ejecutar desde esta carpeta:

```bash
python -m pip install -r "Glove/requirements.txt"
python verificar_paquete.py
```

GloVe, descargando el archivo fuente si aún no está disponible:

```bash
python "Glove/descargar_glove.py"
python "Glove/reconstruir_glove_50d.py"
```

Word2Vec:

```bash
python "W2V/descargar_word2vec.py"
python "W2V/reconstruir_word2vec_definitivo.py"
```

Las rutas predeterminadas se resuelven respecto de cada script, no respecto
del directorio de ejecución. Se pueden usar archivos que ya estén en otra
carpeta con `--txt` o `--bin`, sin copiarlos ni descargarlos de nuevo.

## Datos fuente y salidas

| Operación | Archivo relativo a esta carpeta |
|---|---|
| Entrada GloVe | `Glove/datos/glove.6B.50d.txt` |
| Entrada W2V | `W2V/datos/GoogleNews-vectors-negative300.bin` |
| Salida GloVe | `Glove/salida/glove_6B_50D_definitivo.pkl` |
| Salida W2V | `W2V/salida/w2v_google_news_top3000000_relaxed_lowercase_clean.pkl` |

Cada generación comprueba las huellas y vuelve a leer el PKL escrito antes
de darlo por válido. Guarda al lado un archivo `*.verificacion.json` con las
dimensiones, huellas y versiones del entorno. `--solo-verificar` reconstruye
y comprueba sin escribir un PKL. Para reemplazar una salida ya generada,
usar explícitamente `--sobrescribir`.

Los PKL se leen con `pandas.read_pickle`. El cargador de GloVe de los notebooks
puede seguir leyendo el TXT; el PKL es una exportación equivalente. Si los
notebooks usan sus rutas anteriores, hay que apuntarlos a las rutas de esta
carpeta o copiar allí el TXT/PKL correspondiente.

## Preparación para GitHub

Se versionan código, documentación, dependencias, referencias y el mapeo
comprimido completo de W2V. Los archivos fuente grandes y las salidas PKL
quedan excluidos mediante `.gitignore`; cada carpeta de datos y salidas
conserva un README. El mapeo `.csv.gz` sí debe subirse al repositorio.

GitHub bloquea archivos de más de 100 MiB en Git ordinario. Por ello este
paquete no contiene el TXT de 171 MB, el binario de 3,64 GB ni los PKL.
No es necesario configurar Git LFS para esta distribución de código.

Consultar [INSTRUCCIONES_DATOS.md](INSTRUCCIONES_DATOS.md) para la copia manual.
No queda ningún archivo de código pendiente de incorporar.

## Alcance de las verificaciones

`VERIFICACION_ENTREGA.json` registra las comprobaciones realizadas al preparar
este paquete. `MANIFIESTO_ARCHIVOS.json` permite verificar su integridad después
de extraerlo o clonarlo. Los resultados y notebooks de las corridas K40/K50
pertenecen a las etapas experimentales y se organizan fuera de esta carpeta.
Los prototipos BERT tienen su propia generación, separada de GloVe y W2V.

## Fuentes

- [GloVe, Stanford](https://nlp.stanford.edu/projects/glove/).
- [Distribución GloVe 6B](https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip).
- [Distribución Google News mantenida por Gensim](https://github.com/piskvorky/gensim-data).
- [Límites de archivos de GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Las distribuciones externas mantienen sus condiciones de uso; este paquete
no les asigna una licencia nueva.
