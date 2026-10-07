#!/usr/bin/env python3
"""Generic command-line entry point for creating configurable CPT datasets.

Examples:
    python datasets_processing/cpt_data_sample_creation.py \
        --types caselaw contracts \
        --jurisdictions EU UK \
        --slimpajama-ratio 0.10 \
        --dataset-name CPT-CUSTOM \
        --output-prefix cpt-custom \
        --output-dir datasets/CPT-CUSTOM \
        --skip-perplexity

Use ``--types ALL`` or ``--jurisdictions ALL`` to disable that filter.
``--slimpajama-ratio 1`` selects the complete SlimPajama input.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cpt_data_sample_creation_v1_eu_1b import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    defaults = [
        "--dataset-name",
        "CPT-CUSTOM",
        "--output-prefix",
        "cpt-custom",
        "--output-dir",
        str(PROJECT_ROOT / "datasets/CPT-CUSTOM"),
        "--jurisdictions",
        "ALL",
        "--slimpajama-ratio",
        "0.10",
    ]
    main(defaults + sys.argv[1:])
