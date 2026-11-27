# Revisión de la implementación

Se conserva el contenido de los seis módulos científicos recibidos y las fuentes de W2V K40. Se añaden
adaptadores que conectan resultados incluidos, admiten rutas largas en Windows y registran la selección
estructural previa a la lectura científica de los resultados.

La revisión integra `PendientesEstudioEstatico_20261125T231208Z.zip`: 49.993 archivos, todos comprobados
contra su manifiesto. Se incorporan 11.905 archivos GloVe consumidos por el lector principal; se omiten
cachés de rendimiento y ensayos alternativos. Los 3.825 checkpoints CV y los 27 FULL/LOCO quedan incluidos.

La grilla se corrige a K20–K100. Se verifican los rechazos K20/K30 de ambos datasets y se selecciona K40.
GloVe K20 se reproduce usando 160 vectores cuyo hash coincide exactamente con el registro de rechazo;
GloVe K30 se evalúa con cinco bins fijos, sin aceptar su variante histórica de cuatro bins.

Durante la comprobación del notebook se bloquearon las funciones de entrenamiento, clasificación y
optimización. Se ejecutaron sus celdas principales, incluidas las figuras, desde las fuentes del proyecto.
`VERIFICACION_ETAPA_1_2.json` contiene el resultado y los tiempos medidos en el entorno de revisión.
Las pruebas se ejecutaron en Linux; el soporte de rutas largas Windows se conserva en los adaptadores.
