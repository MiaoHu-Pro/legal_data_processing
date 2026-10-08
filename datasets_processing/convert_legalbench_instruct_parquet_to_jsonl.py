#!/usr/bin/env python3
"""Convert the local LegalBench-Instruct Parquet split to JSONL.

The conversion is lossless at the record level: all source columns are copied
without changing the curated prompt in ``inputs`` or the gold ``answer``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "datasets/evaluation/legalbench_instruct/data/train-00000-of-00001.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: the input filename with .jsonl replacing .parquet",
    )
    parser.add_argument("--preview", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output = args.output or args.input.with_suffix(".jsonl")
    if args.preview < 0:
        parser.error("--preview must be non-negative")
    return args


def convert(input_path: Path, output_path: Path, *, overwrite: bool, preview: int) -> int:
    if not input_path.is_file():
        raise SystemExit(f"Input Parquet file does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise SystemExit(f"Output already exists: {output_path}; use --overwrite to replace it")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("Input and output paths must be different")

    parquet = pq.ParquetFile(input_path)
    expected_columns = {"answer", "index", "task_type", "task_name", "inputs"}
    actual_columns = set(parquet.schema_arrow.names)
    missing = expected_columns - actual_columns
    if missing:
        raise SystemExit(f"LegalBench-Instruct schema is missing columns: {sorted(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    rows_written = 0
    preview_rows: list[dict[str, Any]] = []
    try:
        with temporary_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output_file:
            for batch in parquet.iter_batches(batch_size=1_000):
                for row in batch.to_pylist():
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
                    if len(preview_rows) < preview:
                        preview_rows.append(row)
        if rows_written != parquet.metadata.num_rows:
            raise RuntimeError(
                f"Row-count mismatch: wrote {rows_written:,}, expected {parquet.metadata.num_rows:,}"
            )
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows:   {rows_written:,}")
    if preview_rows:
        print("Preview:")
        for row in preview_rows:
            print(json.dumps(row, ensure_ascii=False))
    return rows_written


def main() -> None:
    args = parse_args()
    convert(args.input, args.output, overwrite=args.overwrite, preview=args.preview)


if __name__ == "__main__":
    main()
