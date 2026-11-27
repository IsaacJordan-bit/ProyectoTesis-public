# Evidence map for the next manuscript revisions

This is an integration plan, not a report of empirical findings. Populate claims
only after the canonical run is complete and its review bundle is inspected.
The existing Spanish summary, English abstract and symbols section remain intact.

| Manuscript location | Evidence to use | Claim boundary |
|---|---|---|
| Materials and Methods: contextual data | BANK_AUDIT.json, PLAN.json, TEXT_STATUS.json, METHODS.md | Exact controlled cohort, source identities, transformations and analysis settings; no claim that every original-export word was characterized. |
| Results: representation characterization | 01_contextual_geometry, geometry_summary | Describe observed within-word variation and its relation to the cross-word reference. Magnitudes must come from the run. |
| Results: what the mean collapses | 02_aggregation_and_neighbours, word_geometry | State variance accounting and neighbourhood changes. A=0/O=1 within-word retention is an identity, not evidence of superiority. |
| Results: held-out representation probe | 03_heldout_controls, heldout_summary, probe_coverage | Compare fresh train-only codebooks with their single-mean baseline and all controls, reporting actual support. Do not describe these codebooks as historical P or a replacement retrieval experiment. |
| Results or appendix: sensitivity | 04_sampling_and_stability, word_averaged_probes, category_omissions | Report sampling/category dependence and degenerate cases, including unfavourable configurations. |
| Results or appendix: text association | 05_text_alignment and text_summary, when available | Association with surrounding lexical content; no independent annotation of linguistic senses. |
| Discussion | Entire evidence set plus the separate controlled AA–PO findings | Assess whether retaining contextual variation is a supported representational motivation. Do not claim these diagnostics identify the cause of the retrieval robustness gain. |
| Appendix | Complete grid, controls, word-level records and methods | Preserve reproducibility and exceptions without overloading the main narrative. |

For each proposed conclusion, record its exact table/figure, scope (COHORT or
COMMON), number of evaluable words/categories and the relevant control. The
strongest claims require agreement between descriptive geometry, held-out
behaviour and the controls; disagreement is itself a result to explain. Do not
replace the main T1/T2 analysis with this exploratory characterization or elevate
T3 to a primary endpoint.

A concise main-text presentation can use the geometry figure and held-out-control
figure, with one small summary table. The remaining diagnostics can support those
claims in the appendix, provided any limitation essential to interpretation is
also stated in the main text.
