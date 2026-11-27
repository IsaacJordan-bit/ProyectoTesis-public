# Estudio estático: ejecución desde resultados incluidos

Abre `EstudioEstatico.ipynb` y ejecuta sus celdas en orden. La preparación conecta automáticamente las
fuentes incluidas de GloVe. W2V utiliza su exportación K40 incluida. No hace falta copiar `Results` del
computador original ni crear `rutas_resultados.json` para el estudio principal.

## Verificación y salidas

Desde esta carpeta:

```bash
python verificar_entrega.py
```

El notebook realiza la auditoría de K, la verificación de conteos/modelos y la presentación de superficies,
heatmaps, OOF y FULL/LOCO. La configuración es `reuse_only`; no hay cálculos científicos nuevos si falta
un archivo. Las salidas derivadas se guardan en `SalidasRevision` y pueden regenerarse desde los paquetes.
Las rutas largas de Windows se manejan mediante rutas extendidas durante extracción y lectura.

- GloVe: 3.825 checkpoints CV autenticados, 75 del modelo elegido cotejados numéricamente y 27 FULL/LOCO.
- W2V: 3.825 diagnósticos de selección, 75 checkpoints del ganador incluidos y 27 FULL/LOCO. Los otros
  3.750 checkpoints W2V no son necesarios para volver a presentar el ganador; su auditoría individual requeriría esos archivos.
- La selección de K recorre 20,30,…,100, admite K40 y se detiene. El error y la robustez no eligen K.

Los módulos científicos recibidos se conservan sin modificaciones. Los adaptadores conectan las fuentes
incluidas y registran la selección estructural en los informes actuales.

## Archivos

| Ruta | Función |
|---|---|
| `CodigoComun/seleccion_estructural.py` | Primera K estructuralmente admisible en la grilla definida. |
| `CodigoComun/auditoria_k_inferiores.py` | Rechazos K20/K30 y reproducción geométrica GloVe K20. |
| `CodigoComun/reutilizacion.py` | Conexión portable, verificación y lectura de resultados. |
| `ResultadosDisponibles/` | Geometrías, conteos y modelos autenticados. Conservar todos los ZIP. |
| `Referencias/` | Diagnósticos estructurales adicionales opcionales. |
| `VERIFICACION_ETAPA_1_2.json` | Resultado de la ejecución comprobada de este notebook. |

El apéndice K50 está incluido y activado mediante `VERIFICAR_MODELO_K50=True`: verifica 3.825 checkpoints CV,
las predicciones OOF del ganador y 27 ajustes FULL/LOCO desde los archivos guardados. No requiere rutas externas.
K40 sigue siendo la selección principal. El modelo K50 usa Ridge=0.01, frente a 0.001 en K40;
la comparación de modelos no aísla sólo K. AA permanece separado y requiere sus resultados externos.
Consulta `RESULTADOS_PENDIENTES.md` para el alcance de las fuentes adicionales.

## Figuras en inglés e intervalo editable

`figuras_estudio.py` dibuja los resultados verificados de GloVe K40, W2V K40 y W2V K50.
Reemplaza la presentación de PNG históricos en español en este notebook. Los paquetes originales
permanecen intactos. Cada heatmap produce una vista completa y otra del intervalo editable de beta,
con la misma matriz y los mismos límites de color. Los gráficos A/B/C, LOCO y OOF también están en inglés.

Las opciones están en la sección 1.1 del notebook. `FIGURAS_MANUSCRITO.md` explica cómo regenerar
sólo los heatmaps sin volver a cargar los resultados ni ejecutar clasificadores o ajustes.
