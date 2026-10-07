#!/usr/bin/env python3
"""Convert SlimPajama Parquet shards to row-oriented JSON Lines files."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = Path("../datasets/SlimPajama-1B/data")
DEFAULT_OUTPUT_DIR = Path("../datasets/SlimPajama-1B/row_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert every Parquet file in a directory to JSONL. "
            "Each output line contains one Parquet row."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing Parquet files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSONL files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Rows read from Parquet at one time (default: 10000)",
    )
    parser.add_argument(
        "--drop-index",
        action="store_true",
        help="Omit the generated __index_level_0__ column from each row",
    )
    return parser.parse_args()


def convert_file(
    parquet_path: Path,
    output_path: Path,
    batch_size: int,
    drop_index: bool,
) -> int:
    """Convert one Parquet shard and return the number of rows written."""
    parquet_file = pq.ParquetFile(parquet_path)
    columns = parquet_file.schema_arrow.names
    if drop_index:
        columns = [name for name in columns if name != "__index_level_0__"]

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=columns,
        ):
            for row in batch.to_pylist():
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

    return rows_written


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")

    parquet_files = sorted(args.input_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No .parquet files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    for file_number, parquet_path in enumerate(parquet_files, start=1):
        output_path = args.output_dir / f"{parquet_path.stem}.jsonl"
        print(
            f"[{file_number}/{len(parquet_files)}] "
            f"Converting {parquet_path} -> {output_path}"
        )
        rows_written = convert_file(
            parquet_path,
            output_path,
            args.batch_size,
            args.drop_index,
        )
        total_rows += rows_written
        print(f"  Wrote {rows_written:,} rows")

    print(f"Finished: wrote {total_rows:,} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
