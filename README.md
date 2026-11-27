# ProyectoTesis

Source code, experimental results, and reproducibility materials for a
thesis on semantic interference and modern associative memories.

## Research scope

This repository contains the computational materials associated with
the following components of the research:

- Static semantic representations using GloVe and Word2Vec.
- Controlled experiments using contextual BERT representations.
- Exploratory characterization of contextual representations.

## Repository structure

| Directory | Description |
|---|---|
| `EstudioEstatico/` | Static dataset generation, experiments, structural validation, predictive models, and robustness analyses. |
| `EstudioContextual/GeneracionDataset/` | Contextual dataset generation, preprocessing, metadata, and validation procedures. |
| `EstudioContextual/ExperimentoContextual/` | Controlled comparison of contextual representation strategies. |
| `EstudioContextual/CaracterizacionExploratoria/` | Exploratory analysis of contextual representation properties. |

## Main notebooks

- Static study:
  `EstudioEstatico/ExperimentoEstatico/EstudioEstatico.ipynb`

- Controlled contextual study:
  `EstudioContextual/ExperimentoContextual/EstudioContextual.ipynb`

- Exploratory contextual characterization:
  `EstudioContextual/CaracterizacionExploratoria/CaracterizacionExploratoria.ipynb`


## Installation and execution

Clone the repository, open its root directory in Visual Studio Code or
another compatible development environment, and configure a Python
environment using the requirements files provided for each study.

Consult the README and LEEME files within each directory before running
the notebooks or computational pipelines.

## Reproducibility

The repository contains computational implementations, experimental
configurations, archived results, validation reports, manifests, and
integrity checks.

The static study uses K40 as its main configuration for GloVe and
Word2Vec. The Word2Vec K50 configuration is retained as a supplementary
robustness analysis.

The static study notebook supports a reuse-only execution mode for
verifying and presenting archived results without repeating the
original computational experiments.

Reusing archived results must be distinguished from independently
recomputing the experiments.

## External datasets

Large contextual datasets are not currently distributed through
this Git repository:

- contextual_occurrences_definitive.parquet
- contextual_exp2_embeddings_rawmean100_wordnet_v1.npz

Their expected locations, generation materials, metadata, and
validation procedures are documented in:

`EstudioContextual/GeneracionDataset/`

The original pretrained static embeddings must also be obtained
separately when rebuilding the corresponding datasets.

Full computational reproduction requires the applicable external
datasets, dependencies, and experimental configurations.

## Archived results

Experimental result archives are retained to support inspection,
validation, and regeneration of the corresponding tables and figures.

Intermediate execution outputs, temporary files, Python caches, and
machine-specific configurations are excluded from version control.

## Publication scope

The materials associated with the replication and methodological comparison of the previous computational approach are not included in this public version pending authorization to redistribute third-party materials.

## Reproducibility limitations

The following static-study modules are temporarily withheld
pending clarification of third-party redistribution rights:

- configurations.py
- experiments_utils.py
- hopfield.py
- preprocessing.py

The historical archive ProyectoTesis_Etapa_1_2.zip is also
withheld.

These exclusions may prevent the execution of certain
computational procedures and completeness checks.

The original scientific integrity manifests are retained
for traceability. Missing restricted files must not be
interpreted as corruption of the published experimental results.

## Citation

Please cite the associated thesis and the corresponding version of
this repository when using its research materials.

## License

The MIT License provided in LICENSE-MIT.txt applies exclusively
to original software authored by Jordan Isaac Pérez Domínguez
for which the author holds the applicable licensing rights.

Third-party software, adapted third-party materials, pretrained
models, datasets, experimental results, and other research
materials are not covered by this license unless explicitly stated.

Third-party materials remain subject to their respective
copyright and licensing conditions.
