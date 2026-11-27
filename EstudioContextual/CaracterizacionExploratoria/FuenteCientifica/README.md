# P22 — Contextual properties and the cost of word-level averaging

This package is an **exploratory characterization of the exact A/P/O inputs used
by the controlled P20/P21 experiment**. All instructions, figures, tables and
generated reports are in English. It does not rerun the 33-hour classifier study.

## Start here

1. Extract the complete ZIP into a new `P22` folder. Keep the notebook, Python
   module and `reference` folder together.
2. Open `P22_Contextual_Properties.ipynb` with the Python environment used for
   P21. The notebook checks dependencies before reading the data; it does not
   install or upgrade packages automatically.
3. Run the configuration cell. The default bank is
   `Path.home() / "E6R/p20_68682213c891"`, the bank identified in your P21 results.
   Change `BANK` only if that exact bank was moved.
4. Run the audit and the 16-word pilot. The pilot uses two deterministically
   chosen words per category and the full scientific settings. Its work is
   retained for the complete analysis.
5. Inspect the measured time estimate, then run the full-analysis cell. The
   default operational budget is four hours **per call**. If it pauses, rerun
   that cell to resume; the budget is not a promise about runtime.
6. Generate the summaries and figures, then export the review ZIP. Send that
   ZIP and the executed notebook for manuscript integration.

The full analysis covers **640 words × 100 occurrences**, including the frozen
COMMON subset of 376 words. This cohort is directly relevant to the controlled
comparison, but it is not a representative survey of all 50,455 words in the
original export. No conclusions are automatically extended to the full export.

## Existing inputs required

```text
p20_68682213c891/
  protocol.json
  inputs/apo/
    done.json
    words.json
    A.npy
    P.npy
    O.npy
    units.parquet
    occurrences.parquet
    ...other files named in done.json
```

The reviewed metadata are included under `reference/`; raw embeddings are not.
The code verifies the bank protocol, receipt identity, input hashes, word and
occurrence identities, prototype multiplicities, and both mean identities.
It refuses an incompatible bank instead of silently rebuilding or realigning it.
The old classifier description inside the reference protocol identifies the
source bank; it does not replace the adapted classifier used in the final P21
results. P22 performs no classifier fitting.

## Optional source texts

The known source is discovered at:

```python
Path.home() / "Workspace/Tesis/Exp2/Results/contextual_dataset/contextual_occurrences_definitive.parquet"
```

If moved, set `SOURCE_PARQUET` in the notebook. Only the selected text and identity
columns of the relevant row groups are read; the large embedding column is not
loaded. The selected occurrences must match the bank by ID, word, source record
and original word position. The selected text content receives a SHA-256 hash;
the entire approximately 40-GB source is not rehashed by this optional stage.

If texts are unavailable, geometry and held-out compression still run. The
context-text analysis is explicitly marked `NOT_AVAILABLE`, not reported as a
negative result or a successful validation. Without source texts this is a
characterization of contextual embeddings, not an independent check of their
relationship to textual contexts.

## What is measured

| Question | Evidence produced |
|---|---|
| Do occurrences of a word vary? | Directional dispersion, within-word cosine similarity, norm shrinkage, centered spectrum and a cross-word anisotropy reference. |
| What does aggregation discard? | Exact within/between-prototype variance decomposition and changes in nearest-word neighbourhoods. |
| Do multiple representatives help on unseen occurrences? | Grouped held-out reconstruction with fresh TRAIN-only codebooks at fixed diagnostic sizes 1, 2, 4 and 8. |
| Is a gain merely a consequence of adding representatives? | Same-size random partitions and covariance-matched unimodal Gaussian references. |
| Is the result sensitive to sampling? | Five repeated group splits, nested training fractions, common-test clustering agreement, word-bootstrap intervals and category omissions. |
| Is variation related to the surrounding text? | Optional masked-target TF-IDF association with complete row/column permutations of pairwise similarities. |

The existing P prototypes are used for descriptive accounting only. The held-out
codebooks are explicitly **diagnostic K-means probes**, not a reimplementation of
the historical pseudo-sense induction. Their size is not the thesis neighbour-list
parameter K. No winning size is selected from test performance.

## Resources, interruption and resumption

- Read-only memory-mapped arrays avoid loading the whole occurrence bank into
  every process as a private copy.
- At most six processes are used by default, bounded by physical CPU count and
  available RAM. Each worker uses one BLAS/OpenMP thread. This is a conservative
  resource policy, not a claim of globally optimal parallelism.
- The RAM reserve is 3 GiB. It is checked during the analysis. Each completed
  word has an atomic checkpoint and a checksum receipt.
- To stop at the next checkpoint boundary, create an empty file named `STOP`
  in the printed output folder. Remove it before resuming. A notebook interrupt
  also preserves completed words; wait for its workers to finish before restarting.
- A lock prevents two notebooks from writing to the same run simultaneously.
  A dead process lock is recovered only on the same host after its PID disappears.
- Code, scientific configuration, inputs and software versions identify the run.
  Changing them creates a separate result folder. Changing only worker count or
  time/RAM limits can reuse the same scientific results after `prepare()`.

GPU execution is intentionally not required: each numerical problem contains only
100 occurrences per word, and the expensive classifier/retrieval loop is absent.
The pilot measures actual workload time on your computer. No unmeasured runtime
or speedup is promised.

## Deliverables after your run

- `REPORT.md`: scope, interpretation boundaries and manuscript placement.
- `tables/*.csv` and `tables/*.tex`: summaries, denominators and appendix tables.
- `figures/*.{png,pdf,svg}`: up to six figure families, with text association conditional
  on source availability. PDF/SVG retain vector graphics.
- `word_geometry.csv`, `heldout_probes.csv`, `controls.csv`, `text_alignment.csv`:
  complete numerical records, including unsuccessful or non-evaluable cases.
- `BANK_AUDIT.json`, `PLAN.json`, `DEPENDENCE_AUDIT.json`, timing and completion
  records: provenance and execution evidence.
- `p22_contextual_<id>_Review.zip`: compact package for review, excluding raw
  embeddings and source texts.

Do not interpret retained variance, separability or compression as proof that
clusters are true linguistic senses. Positive, null and unfavourable contrasts
must all remain visible. The separate controlled AA–PO comparison and T1/T2
remain the principal thesis results; T3 remains exploratory.

## Validation supplied with the package

`VALIDATION.md` records the software tests executed before delivery. The actual
canonical embeddings are on your computer, so this package contains **no new
empirical P22 findings**. Synthetic tests check execution and mathematical
identities; they are never evidence about your corpus.

To rerun the tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python tests/smoke_pipeline.py
python tests/run_notebook_cells.py
```

See `METHODS.md` for formulas, parameter rationales and limitations. References
are supplied in IEEE-style numbered form there and as `references.bib`.
