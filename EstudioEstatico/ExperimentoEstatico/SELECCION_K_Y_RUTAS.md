# Selección de K y fuentes incluidas

La grilla es `{20,30,40,50,60,70,80,90,100}`. Se escoge la primera K con particiones completas,
COMMON positivo y al menos una palabra por categoría×bin, con ocho categorías y cinco bins fijos.
La similitud se estandariza por categoría sobre FULL (palabras únicas de validación, ddof=0).
COMMON es la intersección de los ocho rangos Z. No intervienen errores, T1–T3, modelos ni LOCO.

| Dataset | K | Resultado | Evidencia |
|---|---:|---|---|
| GloVe | 20 | Rechazada | Split 0 descarta `anansie` y `nigra` y pierde `skunk`. Reproducido con sus 160 vectores originales autenticados. |
| GloVe | 30 | Rechazada | FULL=154, COMMON=124; dos celdas vacías con cinco bins: bicycle/bin2 y jellyfish/bin3 (índices desde cero). |
| GloVe | 40 | Admisible, seleccionada | FULL=209, COMMON=182; 40 celdas cubiertas. |
| W2V | 20 | Rechazada | Split 2 pierde bicycle y jellyfish y descarta tres palabras; auditoría y recibos verificados. |
| W2V | 30 | Rechazada | Split 4 pierde bicycle y descarta dos palabras; auditoría y recibos verificados. |
| W2V | 40 | Admisible, seleccionada | FULL=226, COMMON=192; 40 celdas cubiertas. |

GloVe K30 había sido aceptado por una variante exploratoria que reducía a cuatro bins. Esta entrega
recalcula los cinco bins fijados; no transfiere ese PASS exploratorio. Las fuentes originales no se modifican.
Los rechazos tempranos por partición no tienen FULL/COMMON completos: sus celdas se muestran vacías,
no como cero palabras. W2V K60–K100 se marca NO_DISPONIBLE bajo el protocolo V3; no se incorporan
silenciosamente las geometrías K60 del protocolo V2. Al aceptar K40 no se necesitan Ks posteriores para
resolver la selección. GloVe K50 falla cobertura; W2V K50 pasa y queda como sensibilidad.

## Fuentes de GloVe

Las rutas originales se reubican dentro de `SalidasRevision/glove_incluido_v1` al extraer los paquetes.
El contenido de cada archivo se conserva. La correspondencia entre calibración, conteos, geometría,
selección del modelo y FULL/LOCO se verifica con sus contratos y recibos.

- GK83 `scale_d3399476d688141441a5`: 102 escalas HFY RAW y conteos de tres redes.
- GK85 `search_b77d2b37e15428d6bf13`: 51 candidatas, 3.825 checkpoints CV; ganador H_BETA_LINEAR_Z, Ridge=0.0001.
- GK88 `refit_addb067bc5bf230730d5`: 27 ajustes FULL/LOCO guardados.

Se omiten cachés de rendimiento y ensayos alternativos GK86/GK87, que no consume el lector principal.
Los ZIP GloVe se dividen por tamaño; son partes independientes de una misma colección, no variantes
científicas. Se requieren todas. No se extrae con el explorador de Windows: lo hace el código con soporte
para rutas largas.

## Límites de la verificación

Se reutilizan los catálogos guardados. No se recalculan vecinos contra los vocabularios completos.
La partición GloVe K20 se reproduce con un subconjunto exacto cuyo hash coincide con el registro de fallo;
las particiones W2V K20/K30 se verifican con sus auditorías y recibos existentes. Se reconstruyen Z, COMMON,
bins y cobertura desde las tablas completas disponibles. Los diagnósticos de CV/diseño no son filtros por
signos ni por robustez. La cobertura categoría×bin dentro de cada fold es diagnóstica, no un requisito nuevo.

## Anexo W2V K50

`cargar_w2v_k50` conecta los paquetes incluidos y fija K50 exclusivamente para el anexo.
No vuelve a seleccionar K por criterios empíricos. Los contratos archivados conservan esa procedencia,
pero los informes actuales identifican la ejecución como sensibilidad. K40 permanece seleccionada por estructura.
