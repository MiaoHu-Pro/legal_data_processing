#!/usr/bin/env python3
"""Create CPT-V2-Legislation-3B plus a 10% SlimPajama replay sample.

The legal selection includes every ``legislation`` document, regardless of
jurisdiction. Based on the corpus analysis this is approximately 3.295B
estimated tokens before preprocessing and near deduplication.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cpt_data_sample_creation_v1_eu_1b import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    defaults = [
        "--dataset-name",
        "CPT-V2-Legislation-3B",
        "--output-prefix",
        "cpt-v2-legislation-3b",
        "--output-dir",
        str(PROJECT_ROOT / "datasets/CPT-V2-Legislation-3B"),
        "--types",
        "legislation",
        "--jurisdictions",
        "ALL",
        "--slimpajama-ratio",
        "0.10",
    ]
    main(defaults + sys.argv[1:])
