# Etapa 2.1 — Dataset contextual

**Prototipos integrados y verificados. Pendiente: incorporar localmente el
Parquet maestro y ejecutar su comprobación.** No se ha vuelto a ejecutar BERT.

## Qué abrir

Abre **GeneracionDataset.ipynb** con el entorno de tesis en VS Code.
El notebook carga los prototipos guardados y comprueba las huellas y los
metadatos. La reconstrucción completa desde las ocurrencias queda desactivada
por defecto. La vía rápida no importa torch ni descarga modelos o corpus.

La carpeta puede estar dentro de ProyectoTesis/EstudioContextual. Las rutas del
código son relativas al proyecto. Los JSON originales conservan las rutas
antiguas como procedencia documental; esas rutas no se utilizan para cargar.

## Contenido integrado

| Carpeta o archivo | Función |
|---|---|
| GeneracionDataset.ipynb | Reutilizar, verificar y, opcionalmente, reconstruir prototipos |
| GenerarOcurrencias.ipynb | Generar desde corpus con BERT, desactivado por defecto |
| Codigo/contextual.py | Carga indexada, validaciones y medias RAW mean100 |
| Entradas | Vocabulario auditado y lista fija de 318 stopwords |
| Datos/Ocurrencias | Vocabulario final, índice, conteos, selección e informes originales |
| Datos/Prototipos | Matriz NPZ verificada, vocabulario conceptual y manifiesto original |
| Referencia | Los dos notebooks originales, para consulta |
| PRUEBAS_INTEGRACION.json | Pruebas ejecutadas y límites de verificación |

Dataset maestro: 103.029 palabras, 100 ocurrencias por palabra, 10.302.900
filas, 768 dimensiones float32. Prototipos: 50.455 palabras, 768 dimensiones
float64. Se promedian las 100 ocurrencias RAW después de convertirlas a
float64. No se aplica normalización L2 antes ni después.

El NPZ contiene exactamente los valores del PKL recibido. Se verificaron su
forma y las huellas científicas de vocabulario y matriz. La conversión evita
la dependencia de la serialización de StringDtype entre versiones de pandas;
se carga con allow_pickle=False. No se redujo la precisión numérica.

La auditoría WordNet detallada no venía en la entrega. La reproducción aplica
el vocabulario conceptual congelado y verificado. El archivo
Datos/Prototipos/auditoria_pertenencia_congelada.csv informa qué palabras del
maestro pertenecen a esa selección; **no es una reconstrucción ficticia de la
auditoría de synsets o morphy**. No hace falta repetir WordNet para conservar
exactamente el mismo dataset final.

## Archivo que debes incorporar manualmente

Copia este archivo existente:

    C:\Users\jisaa\Workspace\Tesis\Exp2\Results\contextual_dataset\contextual_occurrences_definitive.parquet

A esta ubicación, dentro de tu proyecto nuevo:

    EstudioContextual\GeneracionDataset\Datos\Ocurrencias\contextual_occurrences_definitive.parquet

Desde la raíz de ProyectoTesis, PowerShell:

    Copy-Item -LiteralPath "C:\Users\jisaa\Workspace\Tesis\Exp2\Results\contextual_dataset\contextual_occurrences_definitive.parquet" -Destination ".\EstudioContextual\GeneracionDataset\Datos\Ocurrencias\contextual_occurrences_definitive.parquet"

El inventario recibido mide 41.281.045.347 bytes (38,446 GiB). Necesitas ese
espacio adicional para copiarlo. No está dentro de este ZIP.

Después, ejecuta GeneracionDataset.ipynb completo con el entorno de tesis.
Por defecto verifica las 24 palabras experimentales (2.400 ocurrencias), exige
que sus medias sean exactamente iguales a los prototipos y crea
SalidasRevision/MuestraContextual_<fecha>.zip. Ese ZIP es pequeño y permite
revisar aquí las ocurrencias reales sin transferir el maestro completo.

Si falta el Parquet, la carga de prototipos sigue funcionando y se informa del
pendiente. Una muestra correcta no constituye validación completa de los
10.302.900 vectores.

## Entorno

Las dos advertencias de la recopilación ocurrieron porque se ejecutó el Python
general sin pyarrow. El inventario tampoco encontró numpy, pandas, torch u
otras dependencias en ese intérprete. Esto no demuestra que falten en el
entorno de tesis: selecciona en VS Code el kernel de ese entorno existente.

Si fuera necesario instalar las dependencias de lectura en ese entorno:

    python -m pip install -r requirements.txt

Para regenerar BERT desde corpus, se necesitan además las dependencias de
requirements-generacion.txt y el modelo/corpus descargables o su caché local.
Las versiones de la ejecución histórica no se conocen: no se inventa un lock
histórico. El entorno de comprobación de esta entrega está registrado en
VERIFICACION_ENTORNO.json.

## Reconstrucción opcional sin BERT

En GeneracionDataset.ipynb cambia RECONSTRUIR_PROTOTIPOS a True. Leerá el
Parquet guardado, comprobará el orden, las 100 filas por palabra, embeddings
finitos y no nulos, y reproducirá los prototipos de las 50.455 palabras.
Solo escribe un NPZ nuevo en SalidasRegeneracion después de comprobar las dos
huellas definitivas. No sobrescribe las entradas congeladas. La lectura de
38,45 GiB puede tardar; no equivale a recalcular embeddings o experimentos.

## Regeneración desde corpus

GenerarOcurrencias.ipynb está ordenado en una secuencia única, sin las celdas
duplicadas de rescate o reinicio. Para activarlo debe cambiarse explícitamente
GENERAR_DESDE_CORPUS a True. Es una operación costosa.

Conserva el algoritmo del notebook recibido: WikiText-103 raw completo y los
primeros 800.000 registros de Wikipedia 20231101.en; selección de los 100
contextos únicos de menor hash BLAKE2b por palabra; ventana ±40 palabras;
BERT base uncased, últimas cuatro capas y promedio de WordPieces; recorte
centrado que conserva el objetivo dentro de los 128 tokens. Se excluye nan
desde el inicio y se conserva null como texto literal.

Usa el vocabulario final incluido. El contrato léxico aplicado al vocabulario
auditado reproduce exactamente las 103.029 palabras. No necesita ejecutar
corpus_audit.ipynb. Las salidas se escriben en SalidasRegeneracion/Ocurrencias.
Se verifica la huella de selección antes de continuar con BERT y, al finalizar,
la identidad de los prototipos reconstruidos.

La revisión de Wikipedia se recuperó de las salidas del notebook original.
No constan las revisiones históricas de WikiText ni BERT; quedan explícitamente
sin fijar. No se promete identidad numérica al cambiar esas fuentes, GPU o
librerías. Si cambia la selección o la matriz, las comprobaciones rechazan el
resultado. La regeneración BERT completa no se ejecutó en esta integración.
Para reutilización exacta, usa los artefactos guardados.

## Qué se verificó y qué falta

- Integridad SHA256 de los 19 archivos incluidos en la recopilación.
- Identidad numérica y de vocabulario del PKL recibido y del NPZ convertido.
- Vocabulario final, índice y conteos de las 103.029 palabras.
- Contrato léxico: 103.033 palabras auditadas → 103.029 palabras definitivas.
- Notebook principal ejecutado con los prototipos reales.
- Generador completo compilado y ejecutado con generación desactivada.
- Carga, muestra, reconstrucción y rechazo de datos alterados probados con
  un Parquet sintético; no se presenta esa prueba como validación del maestro.

Pendiente: copiar el Parquet real y comprobar sus ocurrencias en tu equipo.
No se ha cerrado la verificación integral de la etapa 2.1. Para publicación,
los datos grandes deberán distribuirse como artefactos versionados o mediante
una descarga verificable; no existe todavía una URL pública configurada.
