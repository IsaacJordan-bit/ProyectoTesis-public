# Resultados incluidos

Conservar todos los archivos de esta carpeta. `SHA256.json` autentica cada ZIP.

- `E2_Auditoria_estructural_K40_K100.zip`: auditoría y geometrías del rango superior.
- `E2_Geometrias_K20_K30.zip`: rechazos y geometrías inferiores, contratos y subconjunto GloVe K20.
- `GloVe_K40_resultados_01.zip` … `05.zip`: cinco ZIP independientes con la colección GloVe completa
  que consume el lector (11.905 archivos). `GLOVE_INCLUIDO.json` indica partes y rutas.
- Los tres ZIP W2V originales: modelo K40, conteos/contraste empírico K50 y catálogo.

El código extrae automáticamente a `SalidasRevision`, valida los hashes y verifica las identidades
científicas. No hace falta extraer manualmente ni conectar el proyecto original.

## Modelo W2V K50 incluido

`W2V_K50_INCLUIDO.json` conecta los seis ZIP `W2V_K50_resultados_01.zip` a `06.zip`.
Cada ZIP es independiente y deben conservarse todos. El cargador verifica sus hashes y los
12.277 archivos originales incluidos. Se conservaron geometrías, calibraciones, conteos agregados,
todas las unidades CV y FULL/LOCO. Se excluyeron runtime, el catálogo de embeddings y los bloques
intermedios de conteos, que no se necesitan para reutilizar los agregados verificados.
La procedencia es `audit_f4766816e364f73e7b47.zip`; no se modificaron los resultados originales.
