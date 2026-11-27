# Código común

Los seis módulos recibidos permanecen byte a byte idénticos. `exp2_pipeline_integrated.py` conserva su
API original; el notebook principal usa `reutilizacion.py` para conectar las fuentes incluidas.

`seleccion_estructural.py` aplica la grilla K20–K100. `auditoria_k_inferiores.py` aporta la evidencia K20/K30.
El modelo GloVe se reutiliza con su lector original y se actualizan únicamente los informes derivados
para indicar la selección estructural efectiva. W2V K40 usa su lector específico; la rama original W2V K50
se utiliza en el anexo incluido mediante `cargar_w2v_k50`, con su alcance separado de la selección principal. Ningún resultado ausente se recalcula en modo `reuse_only`.


`figuras_estudio.py` es la capa de presentación del notebook estático. Lee superficies y predicciones
verificadas, genera todos los textos en inglés y exporta vistas completas/ampliadas con una configuración
común. No modifica los seis módulos científicos originales ni los paquetes de resultados.
