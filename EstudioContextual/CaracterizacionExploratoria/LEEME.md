# Caracterización exploratoria contextual: P22

## Instalación
Extraer ESTE ZIP dentro de `ProyectoTesis/EstudioContextual`. El ZIP ya contiene la carpeta `CaracterizacionExploratoria`.
Abrir `CaracterizacionExploratoria/CaracterizacionExploratoria.ipynb` y ejecutar en orden en el entorno habitual. No modifica GeneracionDataset ni ExperimentoContextual. Requisitos de lectura y figuras: requirements.txt.

## Qué se integra
Caracterización exploratoria de la cohorte contextual congelada: 640 palabras, 64.000 ocurrencias, 1.286 prototipos y subconjunto COMMON de 376 palabras. Mismo banco P20 usado en las nueve variantes. No es el experimento preliminar AA excluido ni la adaptación del clasificador P21.

Se incluyen todos los registros por palabra, tablas, controles, casos no evaluables, figuras originales e informe. El notebook de reproducción genera seis figuras nuevas en inglés (PNG/PDF/SVG), con indicadores de panel. Se conservan las tablas numéricas para el manuscrito; no se requiere repetir los análisis.

## Ubicación en el manuscrito
Esta caracterización, sus controles y sus gráficos van en anexos, referenciados desde las secciones pertinentes. Su función es examinar variación contextual y pérdida de información por agregación. No demuestra que los grupos sean sentidos lingüísticos verdaderos ni la causa de la diferencia de robustez del experimento principal. No justifica modificar retrospectivamente el K ni elegir un tamaño diagnóstico favorable.

## Reutilización y fuente científica
ResultadosGuardados/P22_resultados.zip conserva íntegro el paquete de revisión original. SHA256.json fija sus bytes para detectar alteraciones. CodigoComun solo lee, verifica y dibuja; no importa el motor de clustering ni abre las rutas absolutas históricas. Las únicas escrituras se hacen en SalidasRevision.

FuenteCientifica contiene el código, metodología, referencias, pruebas y notebook ejecutado que enviaste. Sus README/VALIDATION describen la entrega original anterior a la corrida; no deben confundirse con VERIFICACION.json ni con las evidencias de ejecución del ZIP de resultados. La comparación con el clasificador adaptado que mencionan algunos textos originales no forma parte de esta integración. Para recalcular P22 desde cero sí se necesitan las matrices y textos del banco original; no se necesitan para este notebook.

## Verificación realizada
Se cotejaron la identidad del plan, el código científico y los 640 comprobantes de integridad. Se reconstruyeron las cuatro tablas de detalle (geometría, pruebas, controles y asociación textual) desde los registros por palabra. Se reconciliaron los estimadores puntuales de geometría y reconstrucción con las tablas guardadas. Las referencias FULL, COMMON y geometría coinciden con P20. Se regeneraron las seis figuras en PNG/PDF/SVG y se ejecutaron secuencialmente las celdas del notebook.
No se volvieron a leer los embeddings originales, calcular centroides, permutaciones ni intervalos bootstrap. La comprobación se realiza sobre las evidencias guardadas y no constituye una nueva corrida del experimento. Se conservan 632 celdas de prueba no evaluables y 15 palabras no evaluables para asociación textual.
