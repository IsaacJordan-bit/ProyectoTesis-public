# Software validation before delivery

No empirical P22 findings are included. The production A/P/O vectors and source
texts remain on the user's computer. The supplied notebook is unexecuted so that
synthetic output cannot be mistaken for a thesis result.

## Completed checks

- Ten focused unit tests passed: grouping source records and duplicate vectors,
  grouping identical texts, exact reconstruction error after train-span projection,
  variance decomposition, the random-partition expectation, TRAIN-only baseline
  reconstruction, paired controls and TRAIN-occurrence scoring, insufficient-support
  status, equal-category word bootstrap, text permutation contrasts, and checkpoint
  corruption rejection (some tests cover multiple related properties).
- An end-to-end synthetic bank used the production array dimension (768), 100
  occurrences per word, eight words and two categories. It passed source authentication,
  text/occurrence matching, two spawned worker processes, pilot reuse, full completion,
  a no-work resume, numerical summaries, six figure families and ZIP export.
- Source hashes were unchanged by the analysis. A deliberate modification after
  that check was correctly rejected by the bank audit.
- All nine notebook code cells executed sequentially on a synthetic fixture,
  replacing only the configuration cell with test paths and reduced Monte Carlo
  repetitions. This check included the full k={1,2,4,8} and training-fraction grid.
- The notebook JSON schema was validated and every code cell compiled.
- A rendered held-out-control figure was visually inspected for readable labels
  and unclipped content. Synthetic figures are excluded from the deliverable ZIP.

## Execution limits of these checks

The host is Linux. Multiprocessing used the explicit `spawn` start method used
on Windows, but this is not a native Windows execution. Direct Jupyter kernel
startup was blocked by the host's socket restrictions; notebook cell logic was
therefore executed with the provided sequential-cell harness instead of claiming
successful execution through the Jupyter kernel UI. Production runtime, corpus
findings and compatibility with the local files must be established by the
notebook's audit and pilot on the user's computer.

The package includes `tests/test_p22.py`, `tests/smoke_pipeline.py` and
`tests/run_notebook_cells.py` to reproduce these checks. The synthetic tests check
software correctness, not semantic validity or the empirical benefit of context.

## Validation environment

```json
{
  "python": "3.12.14",
  "packages": {
    "numpy": "2.3.5",
    "pandas": "2.2.3",
    "scipy": "1.17.0",
    "scikit-learn": "1.8.0",
    "pyarrow": "26.0.1",
    "matplotlib": "3.10.8",
    "psutil": "7.2.2",
    "threadpoolctl": "3.6.0",
    "jinja2": "3.1.6"
  },
  "platform": "Linux-6.18.57-x86_64-with-glibc2.41",
  "machine": "x86_64"
}
```
