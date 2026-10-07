#!/usr/bin/env python3

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(
    "/local/scratch/mh1f25/legal_data_process/datasets/"
    "Multi_Legal_Pile_Commercial"
)

OUTPUT_DIR = Path(
    "/local/scratch/mh1f25/legal_data_process/processing_analysis_output"
)

FILE_PATTERN = "multi_legal_pile_en_commercial_*.jsonl"

OUTPUT_BY_TYPE_JURISDICTION = (
    OUTPUT_DIR
    / "multi_legal_pile_commercial_stats_by_type_jurisdiction.csv"
)

OUTPUT_BY_TYPE_JURISDICTION_SOURCE = (
    OUTPUT_DIR
    / "multi_legal_pile_commercial_stats_by_type_jurisdiction_source.csv"
)

OUTPUT_GLOBAL_SUMMARY = (
    OUTPUT_DIR
    / "multi_legal_pile_commercial_global_summary.json"
)

# Save partial results after every input JSONL file.
CHECKPOINT_BY_TYPE_JURISDICTION = (
    OUTPUT_DIR
    / "checkpoint_type_jurisdiction.csv"
)

CHECKPOINT_BY_TYPE_JURISDICTION_SOURCE = (
    OUTPUT_DIR
    / "checkpoint_type_jurisdiction_source.csv"
)

PROGRESS_EVERY = 100_000

# Cheap estimate only.
#
# Do NOT interpret this as exact Qwen/Mistral tokenizer count.
ESTIMATED_TOKENS_PER_WORD = 1.30


# ============================================================
# Helpers
# ============================================================

def normalize(value, default="UNKNOWN"):
    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    return value


def count_words(text: str) -> int:
    """
    Fast approximate word count.

    This does not create a giant list like len(text.split()).

    Counting transitions from whitespace -> non-whitespace
    keeps peak memory lower for unusually large documents.
    """
    count = 0
    in_word = False

    for char in text:
        if char.isspace():
            in_word = False
        elif not in_word:
            count += 1
            in_word = True

    return count


def bytes_to_gb(num_bytes: int) -> float:
    """
    Decimal GB.
    """
    return num_bytes / 1_000_000_000


def estimated_tokens(words: int) -> int:
    return int(words * ESTIMATED_TOKENS_PER_WORD)


def make_stat():
    return {
        "documents": 0,
        "text_bytes": 0,
        "words": 0,
    }


# ============================================================
# CSV writers
# ============================================================

def write_type_jurisdiction_csv(stats, output_path):
    rows = []

    for (text_type, jurisdiction), values in stats.items():

        words = values["words"]

        rows.append(
            {
                "type": text_type,
                "jurisdiction": jurisdiction,
                "documents": values["documents"],
                "text_gb": bytes_to_gb(
                    values["text_bytes"]
                ),
                "words": words,
                "estimated_tokens": estimated_tokens(words),
            }
        )

    rows.sort(
        key=lambda row: (
            row["type"],
            row["jurisdiction"],
        )
    )

    tmp_path = output_path.with_suffix(".tmp")

    with tmp_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "type",
                "jurisdiction",
                "documents",
                "text_gb",
                "words",
                "estimated_tokens",
            ],
        )

        writer.writeheader()

        for row in rows:
            row = row.copy()
            row["text_gb"] = f"{row['text_gb']:.6f}"
            writer.writerow(row)

    # Atomic replacement.
    os.replace(tmp_path, output_path)


def write_source_csv(stats, output_path):
    rows = []

    for (
        text_type,
        jurisdiction,
        source,
    ), values in stats.items():

        words = values["words"]

        rows.append(
            {
                "type": text_type,
                "jurisdiction": jurisdiction,
                "source": source,
                "documents": values["documents"],
                "text_gb": bytes_to_gb(
                    values["text_bytes"]
                ),
                "words": words,
                "estimated_tokens": estimated_tokens(words),
            }
        )

    rows.sort(
        key=lambda row: (
            row["type"],
            row["jurisdiction"],
            row["source"],
        )
    )

    tmp_path = output_path.with_suffix(".tmp")

    with tmp_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "type",
                "jurisdiction",
                "source",
                "documents",
                "text_gb",
                "words",
                "estimated_tokens",
            ],
        )

        writer.writeheader()

        for row in rows:
            row = row.copy()
            row["text_gb"] = f"{row['text_gb']:.6f}"
            writer.writerow(row)

    os.replace(tmp_path, output_path)


# ============================================================
# Setup
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

files = sorted(DATA_DIR.glob(FILE_PATTERN))

if not files:
    raise RuntimeError(
        f"No files found in {DATA_DIR} "
        f"matching {FILE_PATTERN}"
    )

print(f"Found {len(files)} JSONL files")

for path in files:
    print(
        f"  {path.name}: "
        f"{bytes_to_gb(path.stat().st_size):.2f} GB"
    )


# ============================================================
# Aggregators
# ============================================================

# Usually only a small number of combinations,
# therefore tiny compared with the 283 GB corpus.

stats = defaultdict(make_stat)

source_stats = defaultdict(make_stat)

total_documents = 0
total_text_bytes = 0
total_words = 0

bad_json = 0
missing_text = 0
missing_type = 0
missing_jurisdiction = 0
missing_source = 0

start_time = time.time()


# ============================================================
# Scan files sequentially
# ============================================================

for file_idx, path in enumerate(files, start=1):

    file_start = time.time()

    print()
    print("=" * 80)
    print(
        f"[{file_idx}/{len(files)}] "
        f"Processing {path.name}"
    )
    print(
        f"File size: "
        f"{bytes_to_gb(path.stat().st_size):.2f} GB"
    )
    print("=" * 80)

    file_documents = 0
    file_text_bytes = 0
    file_words = 0
    file_bad_json = 0

    with path.open("r", encoding="utf-8", errors="replace", buffering=1024 * 1024, ) as f:

        for line_no, line in enumerate(f, start=1):

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:

                bad_json += 1
                file_bad_json += 1

                if file_bad_json <= 10:
                    print(
                        f"WARNING: bad JSON "
                        f"{path.name}:{line_no}: {exc}"
                    )

                continue

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            raw_type = (
                row.get("type")
                or row.get("text_type")
            )

            raw_jurisdiction = row.get("jurisdiction")
            raw_source = row.get("source")

            if not raw_type:
                missing_type += 1

            if not raw_jurisdiction:
                missing_jurisdiction += 1

            if not raw_source:
                missing_source += 1

            text_type = normalize(raw_type)
            jurisdiction = normalize(raw_jurisdiction)
            source = normalize(raw_source)

            # ------------------------------------------------
            # Text
            # ------------------------------------------------
            # get text field
            text = row.get("text")

            if not isinstance(text, str):
                missing_text += 1
                text = ""

            text_bytes = len(
                text.encode(
                    "utf-8",
                    errors="replace",
                )
            )

            words = count_words(text)

            # ------------------------------------------------
            # Aggregate: type + jurisdiction
            # ------------------------------------------------

            key = (
                text_type,
                jurisdiction,
            )

            value = stats[key]

            value["documents"] += 1
            value["text_bytes"] += text_bytes
            value["words"] += words

            # ------------------------------------------------
            # Aggregate: type + jurisdiction + source
            # ------------------------------------------------

            source_key = (
                text_type,
                jurisdiction,
                source,
            )

            value = source_stats[source_key]

            value["documents"] += 1
            value["text_bytes"] += text_bytes
            value["words"] += words

            # ------------------------------------------------
            # Global totals
            # ------------------------------------------------

            total_documents += 1
            total_text_bytes += text_bytes
            total_words += words

            # ------------------------------------------------
            # Per-file totals
            # ------------------------------------------------

            file_documents += 1
            file_text_bytes += text_bytes
            file_words += words

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if total_documents % PROGRESS_EVERY == 0:

                elapsed = time.time() - start_time

                docs_per_sec = (
                    total_documents / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"Processed "
                    f"{total_documents:,} documents | "
                    f"{bytes_to_gb(total_text_bytes):,.2f} GB text | "
                    f"{total_words:,} words | "
                    f"~{estimated_tokens(total_words):,} est. tokens | "
                    f"{docs_per_sec:,.1f} docs/s",
                    flush=True,
                )

    # ========================================================
    # File summary
    # ========================================================

    file_elapsed = time.time() - file_start

    print("--------")
    print(
        f"Finished {path.name}"
    )

    print(
        f"  Documents:   {file_documents:,}"
    )

    print(
        f"  Text size:   "
        f"{bytes_to_gb(file_text_bytes):,.2f} GB"
    )

    print(
        f"  Words:       {file_words:,}"
    )

    print(
        f"  Est tokens:  "
        f"{estimated_tokens(file_words):,}"
    )

    print(
        f"  Bad JSON:    {file_bad_json:,}"
    )

    print(
        f"  Runtime:     "
        f"{file_elapsed / 60:.1f} minutes"
    )

    # ========================================================
    # Save checkpoint after every file
    # ========================================================

    print("Saving checkpoint...", flush=True)

    write_type_jurisdiction_csv(
        stats,
        CHECKPOINT_BY_TYPE_JURISDICTION,
    )

    write_source_csv(
        source_stats,
        CHECKPOINT_BY_TYPE_JURISDICTION_SOURCE,
    )

    print("Checkpoint saved.", flush=True)


# ============================================================
# Write final CSV files
# ============================================================

write_type_jurisdiction_csv(
    stats,
    OUTPUT_BY_TYPE_JURISDICTION,
)

write_source_csv(
    source_stats,
    OUTPUT_BY_TYPE_JURISDICTION_SOURCE,
)


# ============================================================
# Global summary JSON
# ============================================================

elapsed = time.time() - start_time

summary = {
    "input_directory": str(DATA_DIR),
    "file_pattern": FILE_PATTERN,
    "files_processed": len(files),

    "documents": total_documents,
    "text_bytes": total_text_bytes,
    "text_gb": bytes_to_gb(total_text_bytes),

    "words": total_words,

    "estimated_tokens": estimated_tokens(
        total_words
    ),

    "estimated_tokens_per_word": (
        ESTIMATED_TOKENS_PER_WORD
    ),

    "bad_json": bad_json,
    "missing_text": missing_text,
    "missing_type": missing_type,
    "missing_jurisdiction": missing_jurisdiction,
    "missing_source": missing_source,

    "runtime_seconds": elapsed,
    "runtime_hours": elapsed / 3600,
}

with OUTPUT_GLOBAL_SUMMARY.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(
    f"Documents:             "
    f"{total_documents:,}"
)

print(
    f"Text size:             "
    f"{bytes_to_gb(total_text_bytes):,.2f} GB"
)

print(
    f"Words:                 "
    f"{total_words:,}"
)

print(
    f"Estimated tokens:      "
    f"{estimated_tokens(total_words):,}"
)

print(
    f"Bad JSON rows:         "
    f"{bad_json:,}"
)

print(
    f"Missing text:          "
    f"{missing_text:,}"
)

print(
    f"Missing type:          "
    f"{missing_type:,}"
)

print(
    f"Missing jurisdiction:  "
    f"{missing_jurisdiction:,}"
)

print(
    f"Missing source:        "
    f"{missing_source:,}"
)

print(
    f"Runtime:               "
    f"{elapsed / 3600:.2f} hours"
)

print()

print(
    f"Written:\n"
    f"  {OUTPUT_BY_TYPE_JURISDICTION}\n"
    f"  {OUTPUT_BY_TYPE_JURISDICTION_SOURCE}\n"
    f"  {OUTPUT_GLOBAL_SUMMARY}"
)

"""
nohup .venv/bin/python -u analyse_multi_legal_pile.py \
  > analyse_multi_legal_pile.log 2>&1 < /dev/null &

echo $! > analyse_multi_legal_pile.pid

"""