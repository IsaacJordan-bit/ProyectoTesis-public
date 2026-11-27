# P20: ejecución reducida 33 × 51 × 3

## Dónde y cómo instalar

En el **computador principal**. Detén la celda larga anterior, espera a que termine la interrupción y apaga su kernel.

Copia **todos los archivos de este ZIP** directamente dentro de tu carpeta existente `E6_P20_Contextualizacion_Controlada`, junto a `e6_p20.py` y `p20_config.py`. Este complemento no contiene reemplazos de esos archivos originales ni borra resultados.

Abre `E6_P20_33x51x3.ipynb`, selecciona el mismo entorno Python de P19/P20 y ejecuta sus secciones 1–6 en orden. La sección 2 muestra qué checkpoints seleccionados ya existen; la sección 3 es la ejecución larga que reemplaza a la anterior.

No uses «Ejecutar todo» en el notebook antiguo. Mantén las rutas de datos y de salida de la corrida original. Si existía un archivo `STOP`, elimina únicamente ese marcador tras detener la corrida; conserva el resto de archivos.

## Panel fijado antes de sus resultados

- 33 índices originales: `0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97`.
- Beta original = `0.010 + 0.005 × índice`; rango 0.010–0.495, incluidos extremos y 0.100. No se interpolan resultados de betas omitidos.
- Splits: 0–50, los mismos 51 que antes. Semillas del clasificador: 0, 1, 2.
- Se conservan tres memorias × tres consultas A/P/O y tres redes. Cada ensamble entrenado por memoria/red/split/beta/semilla evalúa las tres consultas.
- Total: 33 × 51 × 3 × 3 × 3 = **45.441 ensambles**, frente a 449.820 originales (89,9 % menos). Esto no garantiza reducir el tiempo total en la misma proporción: calibración, preparación de scores, lectura y análisis estadístico tienen otros costes.
- Máximo solicitado: 14 workers. La RAM disponible y la medida del primer trabajo pueden reducir el paralelismo efectivo.

## Qué permanece y qué cambia

Se conserva la calibración compartida TRAIN-A sobre los **98 betas originales**, rho, las fuentes, las 100 posiciones canónicas por palabra, K80, las ocho clases, los 51 splits, las geometrías FULL/COMMON y los folds CV compartidos. Las predicciones existentes compatibles conservan su identidad original y se verifican antes de reutilizarse.

El análisis se recalcula exclusivamente sobre los 33 betas y las tres semillas seleccionadas. Por tanto, T1/T2/T3 y los modelos ajustados pueden variar respecto a la grilla completa. Las familias y penalizaciones estadísticas están fijas, pero su ajuste y estandarización de beta corresponden al panel reducido. No se eligen semillas ni betas por su comportamiento observado.

Los 51 splits mantienen la cobertura y geometría previamente fijadas; no son un mínimo universal de validez estructural. Las 100 posiciones canónicas no se consideran réplicas independientes.

## Checkpoints y salidas

El banco original `p20_…` conserva las predicciones individuales. El complemento agrega sólo las pendientes seleccionadas. Los resultados de otras semillas/betas permanecen intactos, excluidos del panel reducido. Se pueden regenerar cachés temporales de scores/recall cuando sea necesario; no se prometen reutilizar todos los cálculos temporales interrumpidos.

El análisis reducido va a una carpeta hermana `p20r_…`, con protocolo e identidad propios. El complemento **no marca como completa** la corrida original de 98 × 51 × 10. La finalización del panel reducido exige los 45.441 ensambles, el análisis estadístico y sus comprobaciones.

Las figuras y tablas están en inglés. Los gráficos aparecen en el output y se guardan en PNG/PDF; también se guardan sus matrices y tablas para modificar la presentación posteriormente sin volver a entrenar. Puedes editar títulos, labels, colores y formato en `plot_p20_reduced.py`; este archivo de presentación no está dentro de la identidad científica, por lo que cambiarlo no invalida los checkpoints. El ZIP de revisión incluye las superficies ajustadas, además de las empíricas.

## Validación de entrega

`VALIDACION_33x51x3.json` describe pruebas sintéticas del puente de checkpoints, los índices beta, los conteos, los ajustes estadísticos y las figuras. No contiene resultados científicos de tus datos reales. Los archivos fuente originales se verificaron sin modificaciones.
