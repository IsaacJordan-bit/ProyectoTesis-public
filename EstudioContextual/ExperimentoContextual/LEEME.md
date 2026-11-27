# Gráficos de ajuste contextual

Extrae ESTE ZIP directamente dentro de ProyectoTesis/EstudioContextual/ExperimentoContextual. Contiene CodigoComun/diagnosticos_contextual.py y DiagnosticosContextual.ipynb. No reemplaza tu notebook principal ni su configuración.

Opción recomendada: añade al final de EstudioContextual.ipynb, después de cargar RESULTS y definir FIGURES, la celda:

```python
from diagnosticos_contextual import plot_diagnostics
DIAGNOSTICOS = plot_diagnostics(RESULTS, FIGURES, show=True)
```

Alternativamente, ejecuta DiagnosticosContextual.ipynb desde la misma carpeta. Ambas opciones producen lo mismo; no es necesario ejecutar ambas. Requiere el módulo contextual_guardado.py y ResultadosGuardados/P20_controlado.zip ya entregados.

Genera 81 figuras (9 variantes × 3 redes × 3 diagnósticos), exportadas en PNG y PDF:
- fit_quality: pendiente semántica, concordancia de pendientes y FAR medio; compara empírico, MATCHED5 y GRID30.
- LOCO_sensitivity: T1/T2/T3 en FULL y las ocho omisiones, comparando las mismas tres superficies.
- OOF: RMSE MATCHED5 por beta de las predicciones fuera de muestra, promediadas entre repeticiones ANTES de calcular el RMSE. No es un gráfico del ajuste en entrenamiento.

Se guardan en SalidasRevision/Figuras/diagnostics/<variante>. Se exportan también los CSV de las curvas. Usa show=False para guardar sin mostrar las 81 figuras. DPI y formatos provienen de FIGURES. El recorte de los heatmaps no recorta estos diagnósticos: se evalúa siempre la grilla científica completa.

Solo utiliza el modelo principal. No cambia datos, parámetros, familia, regularización, resultados ni signos. El RMSE del ajuste y las curvas OOF se cotejan con las tablas originales antes de exportar. La comparación MATCHED5 es la de fidelidad directa; GRID30 usa otra malla y se muestra como representación del modelo.

No modifica los heatmaps existentes. Para evitar exportar el modelo alternativo, comenta la llamada FIGURAS_ALTERNATIVAS del notebook principal y llama regenerar_heatmaps con kinds=("empirical", "primary") cuando quieras regenerar solo las figuras elegidas.
