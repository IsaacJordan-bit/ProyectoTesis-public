# Archivos grandes que se obtienen aparte

Todos los archivos de código y el mapeo de selección de W2V están incluidos.
Solo se necesitan dos archivos fuente grandes, que se pueden descargar con
los scripts o indicar desde sus ubicaciones actuales.

| Archivo | Tamaño de la distribución verificada | Destino predeterminado |
|---|---:|---|
| `glove.6B.50d.txt` | 171.350.079 bytes | `Glove/datos/glove.6B.50d.txt` |
| `GoogleNews-vectors-negative300.bin` | 3.644.258.522 bytes | `W2V/datos/GoogleNews-vectors-negative300.bin` |

Las rutas anteriores son relativas a `GeneraciónDatasets/`.

## Si ya tienes los archivos

- **GloVe:** en `GloVe_6B_50D_definitivo_y_trazabilidad(1).zip` está en
  `glove.6B/glove.6B.50d.txt`. Copia ese TXT al destino indicado.
- **W2V:** utiliza tu archivo `GoogleNews-vectors-negative300.bin`. El ZIP del
  dataset depurado contiene el PKL y el mapeo, pero no ese binario completo.
  Si no conservas el binario, `descargar_word2vec.py` lo obtiene y descomprime.

Para trabajar directamente con tus rutas actuales, desde `GeneraciónDatasets/`:

```powershell
python "Glove/reconstruir_glove_50d.py" --txt "C:/Datos/glove.6B.50d.txt"
python "W2V/reconstruir_word2vec_definitivo.py" --bin "C:/Datos/GoogleNews-vectors-negative300.bin"
```

Sustituye esas rutas si los archivos están en otro lugar. También puedes usar
`--salida "ruta/dataset.pkl"` para elegir dónde escribir el resultado.

## Si necesitas descargarlos

```bash
python "Glove/descargar_glove.py"
python "W2V/descargar_word2vec.py"
```

GloVe se distribuye en un ZIP con varias dimensiones: el script descarga el
ZIP y extrae únicamente el TXT de 50 dimensiones. Borra el ZIP temporal al
terminar. Si ya descargaste la distribución, acepta `--archivo-zip`.

Word2Vec se descarga como gzip y se descomprime durante la transferencia,
sin almacenar otra copia comprimida. Si ya descargaste ese gzip de Gensim,
acepta `--archivo-gz`. Se comprueban el tamaño, el MD5 de la distribución y
el SHA-256 del binario antes de colocarlo en su destino definitivo.

La descarga requiere conexión a las fuentes externas. Una conexión interrumpida
no deja un archivo definitivo incompleto; se puede repetir el comando.
Reserva espacio para los datos fuente y los PKL; para W2V, al menos unos
5 GB libres y un proceso Python de 64 bits. La generación completa requiere
memoria adicional para el DataFrame y su comprobación tras la serialización.

## Qué subir a GitHub

Sube el contenido de esta entrega respetando `.gitignore`. Mantén localmente
los archivos de `datos/` y `salida/`; no hay que incorporarlos manualmente al
historial de Git. El archivo `W2V/seleccion_vocabulario.csv.gz` es parte
obligatoria del código de reproducción y sí se incluye en el repositorio.
