"""Editable local paths and execution limits. Scientific settings live in e6_p20.py."""
from pathlib import Path
from e6_p20 import Config

CONFIG=Config(
    # Existing folders on the MAIN computer. Inputs are read-only.
    project_root=Path.home()/'Workspace/Tesis/Exp2def',
    dataset_dir=Path.home()/'Workspace/Tesis/Exp2/Results/E6_DATASET_PSEUDOSENTIDOS_WORDNET/run_4b7a935bdfb47252c452/exports/dataset_20261022T211853_eeab30',
    occurrence_dir=Path.home()/'Workspace/Tesis/Exp2/Results/contextual_dataset',
    # Set an explicit path only if the original static audit/catalog was moved.
    audit_root=None,
    embedding_path=None,
    output_root=Path.home()/'E6R',
    max_workers=14,        # upper bound; RAM probe may lower it
    reserve_gib=3.0,
    reserve_disk_gib=8.0,
    beta_chunk=7,         # allowed: 1, 2, 7, 14; schedule only
    query_block=32,       # reduce to 16 if per-process RAM is limiting
    show_figures=True,
    figure_dpi=150,
    force_rehash_sources=False,
)
