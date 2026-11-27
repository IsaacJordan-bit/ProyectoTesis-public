# Figuras en inglés para el manuscrito y los anexos

## Uso en VS Code

1. Abre `EstudioEstatico.ipynb` y busca **1.1. Figuras para el manuscrito y los anexos**.
2. Edita `beta_range=(0.01, 0.15)` con el intervalo que prefieras.
3. Ejecuta el notebook en orden. GloVe K40, W2V K40 y el anexo W2V K50 usan la misma configuración.

El dominio disponible es 0.010–0.495. El rango debe estar dentro de ese dominio y tener límite
inferior menor que el superior. No se exige que los límites coincidan exactamente con puntos de la grilla.
`beta_range=None` genera sólo las vistas completas.

## Cambiar el intervalo después de cargar los resultados

Ejecuta una celda como ésta:

```python
FIGURES.beta_range = (0.01, 0.12)
regenerar_heatmaps()
```

Sólo se redibujan las superficies: no se ejecutan entrenamientos, búsquedas de K, modelos ni LOCO nuevos.
Para esta operación no debes volver a ejecutar las celdas de carga o validación.

## Opciones de presentación

| Opción | Función |
|---|---|
| `beta_range` | Límites horizontales comunes a todos los heatmaps con beta en el eje x. |
| `heatmap_aspect=3.5` | Ancho/alto del panel de datos; menor valor hace un panel más alto. |
| `heatmap_width=10.5` | Ancho en pulgadas. |
| `color_limits=None` | Escala automática por dataset/K, común a sus tres redes y casos FULL/LOCO, calculada antes de recortar. |
| `color_limits=(0.0, 1.0)` | Alternativa: una escala absoluta común también entre datasets y K. |
| `display_view="manuscript"` | Muestra las vistas ampliadas. También admite `"appendix"`, `"both"`, `"none"`. Ambas versiones se guardan. |
| `dpi=200` | Resolución PNG; puede subirse a 300. |
| `formats=("png", "pdf")` | Exporta ambos formatos. También se admite SVG manteniendo PNG. |
| `show_diagnostics=True` | Muestra las figuras A/B/C, LOCO y OOF. |

## Archivos exportados

En `SalidasRevision/Figuras/` se crean las carpetas `glove_k40`, `w2v_k40` y `w2v_k50`.
Dentro de cada una:

- `manuscript_beta_0p01_0p15/`: vistas del intervalo solicitado (el nombre cambia con el rango).
- `appendix_full/`: vistas completas, independientes del intervalo elegido.
- `diagnostics/`: curvas de ajuste, sensibilidad LOCO y fidelidad OOF sobre todo el dominio.
- `figure_index_*.json`: coordenadas, escala de color, huellas de las matrices y de cada figura.

Hay 30 superficies por configuración: 3 empíricas FULL, 3 modelos FULL y 24 modelos LOCO.
Con un intervalo definido se exportan 60 heatmaps por configuración, más 9 figuras de diagnóstico.
Las imágenes que se muestran en el notebook son sólo una selección de las versiones exportadas.

## Interpretación y conservación de resultados

El zoom cambia exclusivamente los límites del eje horizontal y se identifica en inglés dentro de la figura.
Las 98 columnas de beta y sus valores permanecen en la matriz: no hay interpolación, suavizado,
reestimación ni sustitución de valores. Las dos versiones de cada superficie tienen la misma escala de color.
Los indicadores T1/T2/T3, FULL/LOCO y los diagnósticos se calculan sobre el dominio completo.
Los gráficos de ajuste muestran empírico, MATCHED5 y GRID30 sobre sus coordenadas originales.

La paleta morado–amarillo se conserva y la similitud aumenta hacia arriba. Los títulos se simplificaron;
los ejes, leyendas, barras de color y anotaciones de las figuras del notebook están en inglés.
Las palabras que identifican clases (apple, bicycle, etc.) son las originales.

El contraste opcional de soporte estructural también usa el mismo intervalo para sus heatmaps con beta.
Las matrices de cobertura y de pares entre clases no tienen beta en el eje horizontal y no se recortan.
Los códigos y PNG históricos se conservan para trazabilidad; el notebook usa el nuevo módulo de figuras.

Para la leyenda del manuscrito puede utilizarse, ajustando el intervalo:

> Zoomed view of the nominal beta range 0.01–0.15. The full-range surface is shown in Appendix X.
> Trend and robustness statistics are evaluated over the full protocol domain.
