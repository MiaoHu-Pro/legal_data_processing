#!/usr/bin/env python3
"""Convert the three requested legal MMLU test splits from Parquet to JSONL.

Only ``international_law``, ``professional_law``, and ``jurisprudence`` test
files are read. The source integer answer is retained as ``answer_index`` and
``answer`` is emitted as its A-D label for evaluation prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MMLU_DIR = PROJECT_ROOT / "datasets/evaluation/mmlu"
SUBJECTS = ("international_law", "professional_law", "jurisprudence")
ANSWER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmlu-dir", type=Path, default=DEFAULT_MMLU_DIR)
    parser.add_argument("--preview", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.preview < 0:
        parser.error("--preview must be non-negative")
    return args


def convert_row(row: dict[str, Any], expected_subject: str) -> dict[str, Any]:
    question = row.get("question")
    choices = row.get("choices")
    answer_index = row.get("answer")
    subject = str(row.get("subject") or expected_subject)
    if not isinstance(question, str):
        raise ValueError(f"Invalid question in {expected_subject}: {question!r}")
    if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
        raise ValueError(f"Invalid choices in {expected_subject}: {choices!r}")
    if not isinstance(answer_index, int) or not 0 <= answer_index < len(choices):
        raise ValueError(f"Invalid answer index in {expected_subject}: {answer_index!r}")
    if answer_index >= len(ANSWER_LABELS):
        raise ValueError(f"Too many choices to label in {expected_subject}: {len(choices)}")
    return {
        "question": question,
        "choices": choices,
        "answer": ANSWER_LABELS[answer_index],
        "answer_index": answer_index,
        "subject": subject,
    }


def convert_subject(
    mmlu_dir: Path, subject: str, *, overwrite: bool, preview: int
) -> tuple[Path, int, list[dict[str, Any]]]:
    input_path = mmlu_dir / subject / "test-00000-of-00001.parquet"
    output_path = input_path.with_suffix(".jsonl")
    if not input_path.is_file():
        raise SystemExit(f"MMLU test Parquet file does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise SystemExit(f"Output already exists: {output_path}; use --overwrite to replace it")

    parquet = pq.ParquetFile(input_path)
    expected_columns = {"question", "subject", "choices", "answer"}
    missing = expected_columns - set(parquet.schema_arrow.names)
    if missing:
        raise SystemExit(f"{subject} schema is missing columns: {sorted(missing)}")

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    rows_written = 0
    preview_rows: list[dict[str, Any]] = []
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for batch in parquet.iter_batches(batch_size=1_000):
                for source_row in batch.to_pylist():
                    row = convert_row(source_row, subject)
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
                    if len(preview_rows) < preview:
                        preview_rows.append(row)
        if rows_written != parquet.metadata.num_rows:
            raise RuntimeError(
                f"{subject} row-count mismatch: wrote {rows_written:,}, "
                f"expected {parquet.metadata.num_rows:,}"
            )
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path, rows_written, preview_rows


def main() -> None:
    args = parse_args()
    total = 0
    for subject in SUBJECTS:
        output_path, rows, preview_rows = convert_subject(
            args.mmlu_dir, subject, overwrite=args.overwrite, preview=args.preview
        )
        total += rows
        print(f"{subject}: {rows:,} rows -> {output_path}")
        for row in preview_rows:
            print(json.dumps(row, ensure_ascii=False))
    print(f"Total MMLU legal test rows: {total:,}")


if __name__ == "__main__":
    main()
