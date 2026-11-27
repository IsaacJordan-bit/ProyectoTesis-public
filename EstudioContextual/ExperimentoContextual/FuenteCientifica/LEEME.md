# P20 — Contextualización controlada A/P/O

Ejecutar en el **computador principal**, con el mismo entorno Python de P19.
Este paquete implementa la comparación controlada de la sección metodológica
«Comparación controlada del aporte contextual». No continúa el experimento GloVe.

## Ubicación y entrada

Extrae la carpeta completa `E6_P20_Contextualizacion_Controlada` junto a
`E6_P19_Ampliacion_98x51x10`, dentro de `C:\Users\jisaa\Workspace\Tesis\Exp6def`.
Abre `E6_P20_Contextualizacion_Controlada.ipynb` en VS Code, selecciona el entorno
de P19 y utiliza un kernel nuevo. Ejecuta las celdas en orden. Los módulos necesarios
están incluidos: no hay que copiarlos dentro de P19 ni modificar esa carpeta.

La configuración editable está en `p20_config.py`; la notebook la importa.
La primera celda de comprobación muestra las rutas y si existen. Si moviste alguna
fuente, cambia únicamente su ruta en ese archivo y vuelve a importar la configuración.
Los resultados nuevos van a `C:\Users\jisaa\E6R\p20_<identidad>` para mantener rutas
cortas en Windows. La carpeta del código permanece junto a P19.

## Archivos locales requeridos

El ZIP contiene código y evidencia de identidad; **no duplica los grandes datos RAW**
que ya están en el computador principal:

- Auditoría estática S: `Exp2def/Results/PROTOTIPOS/_PROTOCOL_RECTOR_V3/audit_3ee527e0fa4f7cfa423f`.
  Se requieren su protocolo, geometría K80 y catálogo autenticado.
- Exportación congelada de pseudo-sentidos:
  `Exp2/Results/E6_DATASET_PSEUDOSENTIDOS_WORDNET/run_4b7a935bdfb47252c452/exports/dataset_20261022T211853_eeab30`.
  Se leen `dataset_manifest.json`, `units.parquet`, `membership.parquet` y
  `prototypes_RAW_float64.npy`.
- Ocurrencias: `Exp2/Results/contextual_dataset/contextual_occurrences_definitive.parquet`
  e índice `word_row_group_index.csv`.

Todas son entradas de solo lectura. No se extrae BERT otra vez, no se induce un nuevo
clustering y no se descargan datos. El subset P14, con otras palabras/clases, no basta.
Una fuente ausente o con otra identidad detiene la preparación y muestra la causa.

La primera autenticación recorre los archivos grandes para verificar SHA-256 y puede
tardar. Al reanudar se reutiliza el registro de hashes si tamaño y fecha no cambiaron;
`force_rehash_sources=True` fuerza una nueva lectura completa. Los checkpoints A/P/O
seleccionados siempre se verifican mediante su recibo.

## Qué compara

| Memoria / consulta de inferencia | A: promedio por palabra | P: pseudo-sentido | O: ocurrencia |
|---|---|---|---|
| A | AA | AP | AO |
| P | PA | PP | PO |
| O | OA | OP | OO |

- Mismas 640 palabras K80, ocho clases, competidores y 51 particiones de S.
  FULL contiene 414 palabras; COMMON, 376. Se conservan cinco bins y sus coordenadas.
- Cada palabra aporta las mismas 100 posiciones canónicas. A repite el promedio,
  P repite cada prototipo según su soporte y O conserva cada vector original.
  La compresión calcula masas de atención con multiplicidades; no elimina copias
  sin compensación. Se verifica A = media de O = media ponderada de P.
- Claves `rho * RAW / norm(RAW)`, valores y consultas RAW; una iteración.
  `rho` se calcula una vez por split sobre las medias A de TRAIN, una por palabra.
- Calibración HFY solo con A de TRAIN, sus multiplicidades reales y las 98 betas;
  las mismas escalas se usan en las nueve condiciones. No se calibra con errores.
- El clasificador siempre aprende de **una consulta A por palabra TRAIN**, recuperada
  con la memoria correspondiente. Un ensemble por memoria/red/beta/split/semilla
  evalúa las tres consultas A/P/O. La receta y las semillas son comunes; los
  coeficientes se ajustan de nuevo cuando cambia la memoria.
- CORE3: MHNCS Softmax, HFY Entmax 1.5 y HFY Normmax 2. Betas .010–.495, paso .005;
  51 particiones y diez semillas. Son **449.820 ensembles nuevos** y 1.349.460
  evaluaciones de condición. No se importan predicciones históricas P19.

Esto estima el efecto de conservar diferenciación contextual **condicionado al corpus,
encoder BERT, tarea K80 y observador fijados**. Incluye los cambios de norma y dispersión
producidos por promediar. No es una comparación de BERT contra GloVe ni una prueba de
una noción lingüística de sentido aislada de toda geometría representacional.

## Ejecución y pausas

1. Configuración, comprobación de fuentes y prueba numérica rápida.
2. `prepare`: reproduce S y sus 51 particiones; prepara las 64.000 ocurrencias y
   verifica sus identidades, pertenencias, medias, multiplicidades y roles.
3. Piloto útil: split 0, beta .010, semilla 0, tres memorias y tres redes: nueve
   ensembles oficiales. Construye también las cachés de scores de ese split.
   Se reutiliza en la corrida completa; **no hay un resultado smoke que confundir con el definitivo**.
4. Corrida completa. Se procesan bloques de consultas y betas, con BLAS=1 por proceso.
   El número de procesos se limita por RAM medida. La estimación de tiempo mostrada
   corresponde a la etapa activa; el piloto no permite prometer la duración total.
5. Conteos completos, superficies empíricas y contrastes; luego modelos, CV y LOCO.
6. Figuras en el output, comprobación final y ZIP de revisión.

`Ctrl+C`/interrumpir la celda conserva los ensembles ya guardados. Tras reiniciar el
kernel ejecuta configuración y `prepare`, y vuelve a ejecutar la etapa pendiente.
La identidad detecta incompatibilidades; no reinicies borrando resultados.
También puedes crear un archivo `STOP` dentro de la nueva carpeta de resultados;
elimínalo antes de reanudar. La notebook incluye celdas opcionales para ello.

No ejecutes dos escritores sobre la misma corrida a la vez. Las etapas usan un lock.
No es necesario cerrar VS Code. El programa comprueba RAM y espacio; una pausa técnica
no es un rechazo científico. Las cachés temporales de O pueden requerir varias decenas
de GiB; antes de cada bloque se calcula el espacio necesario. Los temporales completados
se eliminan al finalizar la memoria/split; las predicciones se conservan.

Si cambias de versión del paquete o de entorno no se mezclan resultados. El entorno
de referencia está en `requirements_referencia.txt`; no se instala ni actualiza nada
automáticamente. Se requieren además psutil, threadpoolctl, matplotlib, IPython e ipykernel.

## Análisis y salidas

- Errores y oportunidades enteras por palabra. `n_trials = 100 * n_items`;
  `n_items` cuenta exposiciones split×semilla. Las copias no son nuevas unidades
  independientes ni aumentan artificialmente los pesos de los modelos.
- Superficies con igual peso de palabra dentro de clase/bin e igual peso de clase.
  Los modelos se ajustan en FULL y se comparan con la superficie empírica en COMMON.
- Principal: `S_TENSOR_SHARED`, Ridge 1e-4. Sensibilidad: `H_BETA_LINEAR_Z`, Ridge .1.
  Familias y penalizaciones fijas, sin selección por los resultados nuevos.
- CV5×5 agrupada por palabra y compartida entre condiciones; 1.350 ajustes CV.
  MATCHED5 para fidelidad y GRID30 para tendencias del modelo.
- FULL y ocho omisiones LOCO, empíricas y con reajuste estadístico real:
  486 reajustes. LOCO evalúa la influencia de las clases en el análisis; no elimina
  clases de la memoria ni vuelve a entrenar los clasificadores.
- Contrastes PP−AA, OO−AA, PA−AA, OA−AA, AP−AA, AO−AA e interacciones
  PP−PA−AP+AA y OO−OA−AO+AA, en T1/T2/T3 y cada omisión. Se distinguen cambios de
  signo y cambios de magnitud. T1>0/T2<0 son prioritarios; T3<0 se conserva.
- Se conservan también transiciones emparejadas: errores nuevos, corregidos, comunes
  y aciertos comunes, usando los mismos IDs de ocurrencia. No se calculan intervalos
  binomiales ni p-valores que supongan independencia entre copias o particiones.

Las figuras tienen textos en inglés, escala común 0–1 y se muestran como imágenes PNG
en el output de la notebook. Se guardan PNG y PDF. Las diferencias usan escala divergente
centrada en cero; una reducción de error no equivale por sí sola a cumplir T1/T2.

`P20_Review_<identidad>.zip` contiene protocolos, diagnósticos, conteos, contrastes,
tablas y figuras para revisar el manuscrito. No incluye el corpus ni todas las
predicciones; esas permanecen en la carpeta de resultados, que debes conservar.

## Validación de entrega

`test_p20.py` contiene pruebas sintéticas reproducibles del operador comprimido frente
al expandido, calibración, consultas, entrenamientos, reanudación, corrupción, CV,
reajuste LOCO y exportación. `VALIDACION_ENTREGA.json` registra el entorno y las pruebas
realizadas. Estas pruebas **no son resultados del experimento**. La autenticación con
tus fuentes completas y la corrida científica se realizarán en el computador principal.

Alternativa por terminal: `python run_p20.py preflight`, luego `python run_p20.py all`.
La notebook es preferible si quieres ver todas las figuras en su output.
