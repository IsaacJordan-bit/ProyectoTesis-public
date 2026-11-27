# Estudio contextual controlado

Copiar EstudioContextual/ExperimentoContextual dentro de ProyectoTesis. Abrir EstudioContextual.ipynb en VS Code, seleccionar el entorno habitual y ejecutar en orden. Dependencias de lectura y gráficos: requirements.txt.

## Alcance
Solo incluye el experimento controlado P20 de nueve variantes. No modifica GeneracionDataset ni el estudio estático. El notebook preliminar EstudioContextual_AA.ipynb y su módulo antiguo quedan fuera de esta entrega y deben excluirse del repositorio definitivo.

La reproducción de resultados guardados, inspección de modelos y generación de heatmaps es autocontenida. ResultadosGuardados/P20_controlado.zip se lee directamente: no se extraen rutas profundas ni se buscan carpetas del computador original. Las rutas absolutas dentro de los metadatos son procedencia, no dependencias del notebook.

FuenteCientifica conserva el motor numérico y sus evidencias sin cambiar los hashes. No se importa durante esta ejecución. Recalcular desde cero requiere las entradas congeladas externas: auditoría S K80, exportación de pseudosentidos (dataset_manifest.json, units.parquet, membership.parquet, prototypes_RAW_float64.npy), ocurrencias e índices. Esta entrega no acredita que esas entradas estén completas en el proyecto. No ejecutar el notebook original ni p20_config como vía de reproducción rápida.

## Figuras
Inglés; panel de datos cuadrado; recorte beta común editable; dominio completo siempre exportado. Escala idéntica entre las dos vistas de cada superficie. FULL y ocho omisiones para datos empíricos, modelo principal y alternativo. Los heatmaps ajustados usan GRID30; MATCHED5 y diagnósticos OOF siguen disponibles en los resultados. Se generan 729 superficies, 1458 vistas y 2916 archivos PNG/PDF. La exportación puede tardar varios minutos y ocupar varios cientos de MB; no repite el experimento.

## Validación
5039 archivos coinciden con los hashes; 486 ajustes estables completos y 1215 matrices empíricas/MATCHED5/GRID30 verificadas. Las 243 superficies empíricas se reconstruyeron desde los conteos usando COMMON y pesos uniformes por clase/celda; máxima diferencia 1.53e-16. Se verificó la forma cuadrada de 12 paneles y se inspeccionó un ejemplo. No se ejecutó BERT, clasificación ni ajustes estadísticos. No se exportaron las 2916 figuras durante la revisión; se comprobó la exportación con una muestra de casos.
