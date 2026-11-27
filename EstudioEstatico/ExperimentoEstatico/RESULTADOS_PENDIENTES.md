# Alcance de los resultados incluidos y pendientes

## Estudio principal y anexo K50: completos para reutilización

El ZIP incorpora las geometrías y rechazos para escoger K40 desde la grilla K20–K100,
los conteos CORE3, la selección estadística, los coeficientes y FULL/LOCO de GloVe/W2V K40.
También incluye el modelo W2V K50 del anexo, con las 3.825 unidades CV, OOF del ganador,
27 ajustes FULL/LOCO, calibraciones y conteos. No requiere rutas externas ni nuevos entrenamientos.

K40 sigue siendo la primera K estructuralmente admisible. K50 está fijada para el anexo.
Los modelos archivados usan Ridge=0.001 (K40) y 0.01 (K50); no es un contraste que varíe únicamente K.

## Archivos externos fuera del alcance de esta ejecución

1. **AA K80:** falta `Results/PROTOTIPOS/_PROTOCOL_RECTOR_V3/audit_3ee527e0fa4f7cfa423f/`,
   con GK97P/GK98P. Su notebook permanece separado del estudio estático.
2. **W2V K40, todas las candidatas CV:** se incluyen los diagnósticos de las 3.825 unidades,
   los 75 checkpoints individuales del ganador y los 27 FULL/LOCO. Para auditar individualmente
   cada candidata K40 faltan los otros 3.750 checkpoints de
   `E2R/k40_model_3ffd8554b799/MODEL_SEARCH/checkpoints/`. No son necesarios para ejecutar este notebook.

Las geometrías posteriores W2V K60–K100 del protocolo V3 no están incluidas; no son necesarias
para seleccionar la primera admisible, K40, y no se presentan como fallos estructurales.
