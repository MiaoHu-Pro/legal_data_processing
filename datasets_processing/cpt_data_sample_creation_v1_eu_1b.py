#!/usr/bin/env python3
"""Create the CPT-V1-EU-1B document-level JSONL dataset.

The default mixture uses every eligible EU document from MultiLegalPile
Commercial and approximately 140M estimated replay tokens sampled across all
SlimPajama training shards. Token budgets use 1.3 * whitespace-delimited words;
the final manifest reports achieved estimates and must not be confused with an
exact model-tokenizer count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if sys.version_info < (3, 12):
    raise SystemExit(
        "CPT-V1-EU creation with the vendored text-dedup package requires Python 3.12+. "
        f"Current interpreter: {sys.executable} ({sys.version.split()[0]}). "
        f"Run with: {PROJECT_ROOT / '.venv/bin/python'}"
    )
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legal_data_process.perplexity_filtering import PerplexityFilter
from legal_data_process.perplexity_filtering import PerplexityFilterConfig
from legal_data_process.rule_based_filters import RuleFilterConfig
from legal_data_process.rule_based_filters import apply_rule_based_filters
from legal_data_process.text_deduplication import TextDedupRunConfig
from legal_data_process.text_deduplication import canonicalize_for_exact_deduplication
from legal_data_process.text_deduplication import run_text_dedup_minhash
from legal_data_process.text_normalization import NormalizationConfig
from legal_data_process.text_normalization import normalize_text


DEFAULT_LEGAL_DIR = PROJECT_ROOT / "datasets/Multi_Legal_Pile_Commercial"
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "datasets/SlimPajama-1B/row_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets/CPT-V1-EU-1B"
DEFAULT_REPLAY_TARGET = 140_000_000
DEFAULT_OUTPUT_SHARD_TARGET = 200_000_000
TOKENS_PER_WORD = 1.30


class RotatingJsonlWriter:
    """Write bounded JSONL staging shards and retain their paths."""

    def __init__(self, directory: Path, prefix: str, max_rows: int = 100_000) -> None:
        self.directory = directory
        self.prefix = prefix
        self.max_rows = max_rows
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths: list[Path] = []
        self._file: TextIO | None = None
        self._rows_in_file = 0
        self.total_rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        if self._file is None or self._rows_in_file >= self.max_rows:
            self._rotate()
        assert self._file is not None
        self._file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        self._rows_in_file += 1
        self.total_rows += 1

    def _rotate(self) -> None:
        self.close_current()
        path = self.directory / f"{self.prefix}-{len(self.paths):05d}.jsonl"
        self.paths.append(path)
        self._file = path.open("w", encoding="utf-8", buffering=1024 * 1024)
        self._rows_in_file = 0

    def close_current(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def close(self) -> None:
        self.close_current()


class FinalShardWriter:
    """Write final shards bounded by estimated-token budget."""

    def __init__(self, output_dir: Path, target_tokens: int) -> None:
        self.output_dir = output_dir
        self.target_tokens = target_tokens
        self._file: TextIO | None = None
        self._hasher: Any = None
        self._path: Path | None = None
        self._rows = 0
        self._tokens = 0
        self.total_rows = 0
        self.total_words = 0
        self.total_tokens = 0
        self.total_bytes = 0
        self.documents_by_corpus: Counter[str] = Counter()
        self.words_by_corpus: Counter[str] = Counter()
        self.tokens_by_corpus: Counter[str] = Counter()
        self.bytes_by_corpus: Counter[str] = Counter()
        self.documents_by_type_jurisdiction: Counter[tuple[str, str]] = Counter()
        self.tokens_by_type_jurisdiction: Counter[tuple[str, str]] = Counter()
        self.shards: list[dict[str, Any]] = []

    def write(self, row: Mapping[str, Any]) -> None:
        text = str(row.get("text", ""))
        word_count = count_words(text)
        token_count = int(row.get("estimated_tokens") or int(word_count * TOKENS_PER_WORD))
        if self._file is None or (self._tokens and self._tokens + token_count > self.target_tokens):
            self._rotate()
        assert self._file is not None
        serialized = json.dumps(dict(row), ensure_ascii=False) + "\n"
        serialized_binary = serialized.encode("utf-8")
        serialized_bytes = len(serialized_binary)
        self._file.write(serialized)
        self._hasher.update(serialized_binary)
        self._rows += 1
        self._tokens += token_count
        self.total_rows += 1
        self.total_words += word_count
        self.total_tokens += token_count
        self.total_bytes += serialized_bytes
        corpus = str(row.get("corpus") or "UNKNOWN")
        text_type = str(row.get("type") or "UNKNOWN")
        jurisdiction = str(row.get("jurisdiction") or "UNKNOWN")
        self.documents_by_corpus[corpus] += 1
        self.words_by_corpus[corpus] += word_count
        self.tokens_by_corpus[corpus] += token_count
        self.bytes_by_corpus[corpus] += serialized_bytes
        self.documents_by_type_jurisdiction[(text_type, jurisdiction)] += 1
        self.tokens_by_type_jurisdiction[(text_type, jurisdiction)] += token_count

    def _rotate(self) -> None:
        self.close_current()
        self._path = self.output_dir / f"cpt-v1-eu-1b-{len(self.shards):05d}.jsonl"
        self._file = self._path.open("w", encoding="utf-8", buffering=1024 * 1024)
        self._hasher = hashlib.sha256()
        self._rows = 0
        self._tokens = 0

    def close_current(self) -> None:
        if self._file is None:
            return
        self._file.close()
        assert self._path is not None
        self.shards.append(
            {
                "file": self._path.name,
                "documents": self._rows,
                "estimated_tokens": self._tokens,
                "bytes": self._path.stat().st_size,
                "sha256": self._hasher.hexdigest(),
            }
        )
        self._file = None

    def close(self) -> None:
        self.close_current()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legal-dir", type=Path, default=DEFAULT_LEGAL_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-target-tokens", type=int, default=DEFAULT_REPLAY_TARGET)
    parser.add_argument("--shard-target-tokens", type=int, default=DEFAULT_OUTPUT_SHARD_TARGET)
    parser.add_argument("--kenlm-model", type=Path)
    parser.add_argument("--perplexity-threshold", type=float, default=1500.0)
    parser.add_argument(
        "--apply-kenlm-to-replay",
        action="store_true",
        help="Apply the supplied KenLM model to SlimPajama as well as legal data",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Explicitly permit a run without KenLM (intended only for smoke tests)",
    )
    parser.add_argument(
        "--near-dedup",
        choices=("text-dedup", "skip"),
        default="text-dedup",
        help="Use the full local text-dedup pipeline or explicitly skip near deduplication",
    )
    parser.add_argument("--dedup-threshold", type=float, default=0.50)
    parser.add_argument("--dedup-processes", type=int, default=8)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffle-buckets", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--resume-from-preprocessed",
        action="store_true",
        help="Reuse .work/preprocessed after a failure and continue at near deduplication",
    )
    parser.add_argument(
        "--max-legal-input-rows",
        type=int,
        help="Testing only: stop after scanning this many legal input rows",
    )
    parser.add_argument(
        "--max-replay-input-rows",
        type=int,
        help="Testing only: stop after scanning this many replay input rows",
    )
    return parser.parse_args()


def count_words(text: str) -> int:
    count = 0
    in_word = False
    for character in text:
        if character.isspace():
            in_word = False
        elif not in_word:
            count += 1
            in_word = True
    return count


def estimate_tokens(text: str) -> int:
    return int(count_words(text) * TOKENS_PER_WORD)


def stable_id(corpus: str, source_file: str, source_row: int) -> str:
    payload = f"{corpus}\0{source_file}\0{source_row}".encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8", errors="replace", buffering=1024 * 1024) as input_file:
        for row_number, line in enumerate(input_file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield row_number, None, f"invalid_json:{exc.msg}"
                continue
            if not isinstance(row, dict):
                yield row_number, None, "json_value_is_not_an_object"
                continue
            yield row_number, row, None


def iter_replay_round_robin(paths: list[Path]) -> Iterator[tuple[Path, int, dict[str, Any] | None, str | None]]:
    """Interleave all replay shards instead of depending on one shard."""

    with ExitStack() as stack:
        handles = [
            (path, stack.enter_context(path.open("r", encoding="utf-8", errors="replace", buffering=1024 * 1024)))
            for path in paths
        ]
        row_numbers = [0] * len(handles)
        active = list(range(len(handles)))
        while active:
            next_active: list[int] = []
            for index in active:
                path, handle = handles[index]
                line = handle.readline()
                if not line:
                    continue
                next_active.append(index)
                row_numbers[index] += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        yield path, row_numbers[index], None, "json_value_is_not_an_object"
                    else:
                        yield path, row_numbers[index], row, None
                except json.JSONDecodeError as exc:
                    yield path, row_numbers[index], None, f"invalid_json:{exc.msg}"
            active = next_active


def preflight(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    legal_files = sorted(args.legal_dir.glob("multi_legal_pile_en_commercial_*.jsonl"))
    replay_files = sorted(args.replay_dir.glob("train-*.jsonl"))
    if not legal_files:
        raise SystemExit(f"No legal JSONL files found in {args.legal_dir}")
    if not replay_files:
        raise SystemExit(f"No SlimPajama training JSONL files found in {args.replay_dir}")
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise SystemExit(f"Output path exists and is not a directory: {args.output_dir}")
        if any(args.output_dir.iterdir()) and not args.resume_from_preprocessed:
            raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    if args.resume_from_preprocessed:
        preprocessed_dir = args.output_dir / ".work/preprocessed"
        if not list(preprocessed_dir.glob("accepted-*.jsonl")):
            raise SystemExit(f"No reusable preprocessed shards found in {preprocessed_dir}")
        if list(args.output_dir.glob("cpt-v1-eu-1b-*.jsonl")):
            raise SystemExit("Final JSONL shards already exist; refusing resume to avoid overwriting them")
    if args.replay_target_tokens <= 0 or args.shard_target_tokens <= 0:
        raise SystemExit("Token targets must be positive")
    if args.shuffle_buckets < 1 or args.progress_every < 1:
        raise SystemExit("Shuffle buckets and progress interval must be positive")
    if args.max_legal_input_rows is not None and args.max_legal_input_rows < 1:
        raise SystemExit("--max-legal-input-rows must be positive")
    if args.max_replay_input_rows is not None and args.max_replay_input_rows < 1:
        raise SystemExit("--max-replay-input-rows must be positive")
    if not args.resume_from_preprocessed and not args.skip_perplexity and args.kenlm_model is None:
        raise SystemExit("Provide --kenlm-model, or explicitly use --skip-perplexity for a smoke test")
    if not args.resume_from_preprocessed and not args.skip_perplexity:
        if not args.kenlm_model.is_file():
            raise SystemExit(f"KenLM model does not exist: {args.kenlm_model}")
        try:
            import kenlm  # noqa: F401
        except ImportError as exc:
            raise SystemExit("The official kenlm Python package is required for this run") from exc
    if args.near_dedup == "text-dedup":
        try:
            import polars  # noqa: F401
            import polars_grouper  # noqa: F401
            import pydantic_settings  # noqa: F401
            import regex  # noqa: F401
            from text_dedup.config import Config  # noqa: F401
            from text_dedup.minhash import main as text_dedup_main  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "Full near deduplication requires dependencies from src/text-dedup/pyproject.toml; "
                f"missing import: {exc.name}"
            ) from exc
    return legal_files, replay_files


def make_stats() -> dict[str, Any]:
    return {
        "input_rows": 0,
        "selected_rows": 0,
        "accepted_rows": 0,
        "accepted_estimated_tokens": 0,
        "invalid_rows": 0,
        "rule_rejections": Counter(),
        "perplexity_rejections": 0,
        "exact_duplicates": 0,
    }


def record_rejection(
    writer: RotatingJsonlWriter,
    *,
    corpus: str,
    source_file: str,
    source_row: int,
    stage: str,
    reasons: Iterable[str],
) -> None:
    writer.write(
        {
            "corpus": corpus,
            "source_file": source_file,
            "source_row": source_row,
            "stage": stage,
            "reasons": list(reasons),
        }
    )


def prepare_record(
    raw: Mapping[str, Any],
    *,
    corpus: str,
    source_file: str,
    source_row: int,
    normalization_config: NormalizationConfig,
    rule_config: RuleFilterConfig,
    perplexity_filter: PerplexityFilter | None,
) -> tuple[dict[str, Any] | None, str, tuple[str, ...], float | None]:
    text = normalize_text(raw.get("text"), normalization_config)
    rule_result = apply_rule_based_filters(text, rule_config)
    if not rule_result.accepted:
        return None, "rules", rule_result.reasons, None
    text = rule_result.text
    perplexity: float | None = None
    if perplexity_filter is not None:
        perplexity_result = perplexity_filter.score_text(text)
        perplexity = perplexity_result.perplexity
        if not perplexity_result.accepted:
            return None, "perplexity", (perplexity_result.reason or "perplexity_rejected",), perplexity

    if corpus == "multilegalpile_commercial":
        text_type = str(raw.get("type") or raw.get("text_type") or "UNKNOWN")
        jurisdiction = str(raw.get("jurisdiction") or "UNKNOWN")
        source = str(raw.get("source") or "UNKNOWN")
    else:
        meta = raw.get("meta") if isinstance(raw.get("meta"), Mapping) else {}
        text_type = "general"
        jurisdiction = "N/A"
        source = str(meta.get("redpajama_set_name") or "UNKNOWN")

    output: dict[str, Any] = {
        "id": stable_id(corpus, source_file, source_row),
        "text": text,
        "corpus": corpus,
        "type": text_type,
        "jurisdiction": jurisdiction,
        "source": source,
        "source_file": source_file,
        "source_row": source_row,
        "estimated_tokens": estimate_tokens(text),
    }
    if perplexity is not None:
        output["perplexity"] = perplexity
    return output, "accepted", (), perplexity


def exact_is_new(connection: sqlite3.Connection, text: str) -> bool:
    canonical = canonicalize_for_exact_deduplication(text)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    cursor = connection.execute("INSERT OR IGNORE INTO seen(text_hash) VALUES (?)", (digest,))
    return cursor.rowcount == 1


def process_candidate(
    raw: Mapping[str, Any],
    *,
    corpus: str,
    source_file: str,
    source_row: int,
    stats: dict[str, Any],
    accepted_writer: RotatingJsonlWriter,
    rejected_writer: RotatingJsonlWriter,
    exact_db: sqlite3.Connection,
    normalization_config: NormalizationConfig,
    rule_config: RuleFilterConfig,
    perplexity_filter: PerplexityFilter | None,
) -> dict[str, Any] | None:
    output, stage, reasons, _ = prepare_record(
        raw,
        corpus=corpus,
        source_file=source_file,
        source_row=source_row,
        normalization_config=normalization_config,
        rule_config=rule_config,
        perplexity_filter=perplexity_filter,
    )
    if output is None:
        if stage == "rules":
            stats["rule_rejections"].update(reasons)
        elif stage == "perplexity":
            stats["perplexity_rejections"] += 1
        record_rejection(
            rejected_writer,
            corpus=corpus,
            source_file=source_file,
            source_row=source_row,
            stage=stage,
            reasons=reasons,
        )
        return None
    if not exact_is_new(exact_db, output["text"]):
        stats["exact_duplicates"] += 1
        record_rejection(
            rejected_writer,
            corpus=corpus,
            source_file=source_file,
            source_row=source_row,
            stage="exact_deduplication",
            reasons=("normalized_exact_duplicate",),
        )
        return None
    accepted_writer.write(output)
    stats["accepted_rows"] += 1
    stats["accepted_estimated_tokens"] += output["estimated_tokens"]
    return output


def iter_staged_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8", buffering=1024 * 1024) as input_file:
            for line in input_file:
                yield json.loads(line)


def iter_deduplicated_dataset(path: Path) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("The datasets package is required to read text-dedup output") from exc
    dataset = load_from_disk(str(path))
    for row in dataset:
        yield dict(row)


def external_shuffle_and_write(
    records: Iterable[Mapping[str, Any]],
    *,
    work_dir: Path,
    output_dir: Path,
    seed: int,
    bucket_count: int,
    shard_target_tokens: int,
) -> FinalShardWriter:
    bucket_dir = work_dir / "shuffle_buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = [
            stack.enter_context((bucket_dir / f"bucket-{index:04d}.jsonl").open("w", encoding="utf-8"))
            for index in range(bucket_count)
        ]
        for row in records:
            document_id = str(row.get("id") or "")
            priority = hashlib.sha256(f"{seed}\0{document_id}".encode("utf-8")).hexdigest()
            bucket = int(priority[:16], 16) % bucket_count
            payload = dict(row)
            payload["__shuffle_priority__"] = priority
            handles[bucket].write(json.dumps(payload, ensure_ascii=False) + "\n")

    writer = FinalShardWriter(output_dir, shard_target_tokens)
    bucket_order = sorted(
        range(bucket_count),
        key=lambda index: hashlib.sha256(f"{seed}\0bucket\0{index}".encode("utf-8")).digest(),
    )
    for index in bucket_order:
        bucket_path = bucket_dir / f"bucket-{index:04d}.jsonl"
        with bucket_path.open("r", encoding="utf-8") as bucket_file:
            rows = [json.loads(line) for line in bucket_file if line.strip()]
        rows.sort(key=lambda row: row["__shuffle_priority__"])
        for row in rows:
            row.pop("__shuffle_priority__", None)
            row.pop("__INDEX__", None)
            cluster = row.pop("__CLUSTER__", None)
            if cluster is not None:
                row["dedup_cluster_id"] = f"text-dedup:{cluster}"
            writer.write(row)
    writer.close()
    return writer


def json_safe_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(stats)
    output["rule_rejections"] = dict(stats["rule_rejections"])
    return output


def resume_from_preprocessed(args: argparse.Namespace, legal_files: list[Path], replay_files: list[Path]) -> None:
    """Continue a failed run without repeating extraction and filtering."""

    work_dir = args.output_dir / ".work"
    accepted_paths = sorted((work_dir / "preprocessed").glob("accepted-*.jsonl"))
    start_time = time.time()
    print(f"Resuming from {len(accepted_paths)} preprocessed JSONL shards", flush=True)

    if args.near_dedup == "text-dedup":
        dedup_dir = work_dir / "text_dedup_output"
        print("Running full local text-dedup MinHash pipeline", flush=True)
        run_text_dedup_minhash(
            accepted_paths,
            dedup_dir,
            TextDedupRunConfig(
                similarity_threshold=args.dedup_threshold,
                ngram_size=5,
                num_permutations=240,
                seed=args.shuffle_seed,
                num_processes=args.dedup_processes,
                check_false_positives=True,
                save_clusters=True,
                keep_index_column=False,
                keep_cluster_column=True,
            ),
        )
        records: Iterable[Mapping[str, Any]] = iter_deduplicated_dataset(dedup_dir)
    else:
        print("WARNING: near deduplication skipped during resume", flush=True)
        records = iter_staged_jsonl(accepted_paths)

    print("Deterministically shuffling and writing final JSONL shards", flush=True)
    final_writer = external_shuffle_and_write(
        records,
        work_dir=work_dir,
        output_dir=args.output_dir,
        seed=args.shuffle_seed,
        bucket_count=args.shuffle_buckets,
        shard_target_tokens=args.shard_target_tokens,
    )

    by_corpus = [
        {
            "corpus": corpus,
            "documents": final_writer.documents_by_corpus[corpus],
            "words": final_writer.words_by_corpus[corpus],
            "estimated_tokens": final_writer.tokens_by_corpus[corpus],
            "token_share": final_writer.tokens_by_corpus[corpus] / final_writer.total_tokens,
            "jsonl_bytes": final_writer.bytes_by_corpus[corpus],
        }
        for corpus in sorted(final_writer.documents_by_corpus)
    ]
    by_type_jurisdiction = [
        {
            "type": key[0],
            "jurisdiction": key[1],
            "documents": final_writer.documents_by_type_jurisdiction[key],
            "estimated_tokens": final_writer.tokens_by_type_jurisdiction[key],
            "token_share": final_writer.tokens_by_type_jurisdiction[key] / final_writer.total_tokens,
        }
        for key in sorted(final_writer.documents_by_type_jurisdiction)
    ]
    totals = {
        "documents": final_writer.total_rows,
        "words": final_writer.total_words,
        "estimated_tokens": final_writer.total_tokens,
        "jsonl_bytes": final_writer.total_bytes,
        "jsonl_decimal_gb": final_writer.total_bytes / 1_000_000_000,
        "jsonl_gib": final_writer.total_bytes / (1024**3),
        "jsonl_shards": len(final_writer.shards),
    }
    manifest = {
        "dataset_name": "CPT-V1-EU-1B",
        "created_at_unix": int(time.time()),
        "runtime_seconds_for_resumed_stage": time.time() - start_time,
        "resumed_from_preprocessed": True,
        "token_count_kind": "estimated_1.30_times_whitespace_words",
        "inputs": {
            "legal_files": [str(path.resolve()) for path in legal_files],
            "replay_files": [str(path.resolve()) for path in replay_files],
            "preprocessed_files": [str(path.resolve()) for path in accepted_paths],
        },
        "configuration": {
            "legal_jurisdiction": "EU",
            "perplexity_enabled_in_original_stage": False,
            "near_deduplication": args.near_dedup,
            "near_dedup_threshold": args.dedup_threshold if args.near_dedup == "text-dedup" else None,
            "shuffle_seed": args.shuffle_seed,
            "output_shard_target_tokens": args.shard_target_tokens,
        },
        "final": {**totals, "shards": final_writer.shards},
        "note": "Preprocessing aggregate counters were not checkpointed by the interrupted run; final statistics are complete.",
    }
    statistics = {
        "dataset_name": "CPT-V1-EU-1B",
        "token_count_kind": "estimated_1.30_times_whitespace_words",
        "totals": totals,
        "by_corpus": by_corpus,
        "by_type_jurisdiction": by_type_jurisdiction,
        "shards": final_writer.shards,
        "note": "Final statistics reconstructed during resume; estimated tokens are not exact tokenizer tokens.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.keep_work_dir:
        shutil.rmtree(work_dir)
    print(
        f"Complete: {final_writer.total_rows:,} documents, ~{final_writer.total_tokens:,} estimated tokens, "
        f"{len(final_writer.shards)} JSONL shards",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    legal_files, replay_files = preflight(args)
    if args.resume_from_preprocessed:
        resume_from_preprocessed(args, legal_files, replay_files)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=False)
    start_time = time.time()

    normalization_config = NormalizationConfig(unicode_form="NFKC")
    rule_config = RuleFilterConfig()
    perplexity_filter = None
    if not args.skip_perplexity:
        perplexity_filter = PerplexityFilter(
            model_path=args.kenlm_model,
            config=PerplexityFilterConfig(threshold=args.perplexity_threshold),
        )
    accepted_writer = RotatingJsonlWriter(work_dir / "preprocessed", "accepted")
    rejected_writer = RotatingJsonlWriter(work_dir / "rejected", "rejected")
    exact_db = sqlite3.connect(work_dir / "exact_dedup.sqlite")
    exact_db.execute("PRAGMA journal_mode=WAL")
    exact_db.execute("CREATE TABLE seen (text_hash TEXT PRIMARY KEY) WITHOUT ROWID")

    legal_stats = make_stats()
    replay_stats = make_stats()
    print(f"Processing {len(legal_files)} legal files; selecting jurisdiction=EU", flush=True)
    stop_legal = False
    for path in legal_files:
        for row_number, row, error in iter_jsonl(path):
            legal_stats["input_rows"] += 1
            if error:
                legal_stats["invalid_rows"] += 1
                record_rejection(
                    rejected_writer,
                    corpus="multilegalpile_commercial",
                    source_file=path.name,
                    source_row=row_number,
                    stage="parsing",
                    reasons=(error,),
                )
            elif row is not None and str(row.get("jurisdiction")) == "EU":
                legal_stats["selected_rows"] += 1
                process_candidate(
                    row,
                    corpus="multilegalpile_commercial",
                    source_file=path.name,
                    source_row=row_number,
                    stats=legal_stats,
                    accepted_writer=accepted_writer,
                    rejected_writer=rejected_writer,
                    exact_db=exact_db,
                    normalization_config=normalization_config,
                    rule_config=rule_config,
                    perplexity_filter=perplexity_filter,
                )
            if legal_stats["input_rows"] % args.progress_every == 0:
                exact_db.commit()
                print(
                    f"Legal scanned={legal_stats['input_rows']:,} selected={legal_stats['selected_rows']:,} "
                    f"accepted={legal_stats['accepted_rows']:,} "
                    f"tokens~{legal_stats['accepted_estimated_tokens']:,}",
                    flush=True,
                )
            if args.max_legal_input_rows and legal_stats["input_rows"] >= args.max_legal_input_rows:
                stop_legal = True
                break
        if stop_legal:
            break

    print(
        f"Sampling SlimPajama replay across {len(replay_files)} train shards to "
        f"~{args.replay_target_tokens:,} tokens",
        flush=True,
    )
    for path, row_number, row, error in iter_replay_round_robin(replay_files):
        replay_stats["input_rows"] += 1
        if error:
            replay_stats["invalid_rows"] += 1
            record_rejection(
                rejected_writer,
                corpus="slimpajama_1b",
                source_file=path.name,
                source_row=row_number,
                stage="parsing",
                reasons=(error,),
            )
        elif row is not None:
            replay_stats["selected_rows"] += 1
            process_candidate(
                row,
                corpus="slimpajama_1b",
                source_file=path.name,
                source_row=row_number,
                stats=replay_stats,
                accepted_writer=accepted_writer,
                rejected_writer=rejected_writer,
                exact_db=exact_db,
                normalization_config=normalization_config,
                rule_config=rule_config,
                perplexity_filter=perplexity_filter if args.apply_kenlm_to_replay else None,
            )
        if replay_stats["input_rows"] % args.progress_every == 0:
            exact_db.commit()
            print(
                f"Replay scanned={replay_stats['input_rows']:,} accepted={replay_stats['accepted_rows']:,} "
                f"tokens~{replay_stats['accepted_estimated_tokens']:,}",
                flush=True,
            )
        if replay_stats["accepted_estimated_tokens"] >= args.replay_target_tokens:
            break
        if args.max_replay_input_rows and replay_stats["input_rows"] >= args.max_replay_input_rows:
            break

    if (
        args.max_replay_input_rows is None
        and replay_stats["accepted_estimated_tokens"] < args.replay_target_tokens
    ):
        raise RuntimeError(
            "SlimPajama training data was exhausted before reaching the replay target: "
            f"{replay_stats['accepted_estimated_tokens']:,} < {args.replay_target_tokens:,}"
        )

    exact_db.commit()
    exact_db.close()
    accepted_writer.close()
    rejected_writer.close()
    if not accepted_writer.paths:
        raise RuntimeError("No records survived preprocessing")

    if args.near_dedup == "text-dedup":
        dedup_dir = work_dir / "text_dedup_output"
        print("Running full local text-dedup MinHash pipeline", flush=True)
        run_text_dedup_minhash(
            accepted_writer.paths,
            dedup_dir,
            TextDedupRunConfig(
                similarity_threshold=args.dedup_threshold,
                ngram_size=5,
                num_permutations=240,
                seed=args.shuffle_seed,
                num_processes=args.dedup_processes,
                check_false_positives=True,
                save_clusters=True,
                keep_index_column=False,
                keep_cluster_column=True,
            ),
        )
        final_records: Iterable[Mapping[str, Any]] = iter_deduplicated_dataset(dedup_dir)
    else:
        print("WARNING: near deduplication explicitly skipped; exact deduplication was applied", flush=True)
        final_records = iter_staged_jsonl(accepted_writer.paths)

    print("Deterministically shuffling and writing final JSONL shards", flush=True)
    final_writer = external_shuffle_and_write(
        final_records,
        work_dir=work_dir,
        output_dir=args.output_dir,
        seed=args.shuffle_seed,
        bucket_count=args.shuffle_buckets,
        shard_target_tokens=args.shard_target_tokens,
    )

    legal_tokens = legal_stats["accepted_estimated_tokens"]
    replay_tokens = replay_stats["accepted_estimated_tokens"]
    pre_dedup_tokens = legal_tokens + replay_tokens
    manifest = {
        "dataset_name": "CPT-V1-EU-1B",
        "created_at_unix": int(time.time()),
        "runtime_seconds": time.time() - start_time,
        "token_count_kind": "estimated_1.30_times_whitespace_words",
        "warning": "Exact model-tokenizer counts must be computed before CPT.",
        "inputs": {
            "legal_files": [str(path.resolve()) for path in legal_files],
            "replay_files": [str(path.resolve()) for path in replay_files],
        },
        "configuration": {
            "legal_jurisdiction": "EU",
            "replay_target_tokens": args.replay_target_tokens,
            "output_shard_target_tokens": args.shard_target_tokens,
            "perplexity_enabled": perplexity_filter is not None,
            "perplexity_applied_to_replay": bool(perplexity_filter and args.apply_kenlm_to_replay),
            "perplexity_threshold": args.perplexity_threshold if perplexity_filter else None,
            "kenlm_model": str(args.kenlm_model.resolve()) if args.kenlm_model else None,
            "near_deduplication": args.near_dedup,
            "near_dedup_threshold": args.dedup_threshold if args.near_dedup == "text-dedup" else None,
            "shuffle_seed": args.shuffle_seed,
            "normalization": asdict(normalization_config),
            "rule_filter": asdict(rule_config),
            "testing_limits": {
                "max_legal_input_rows": args.max_legal_input_rows,
                "max_replay_input_rows": args.max_replay_input_rows,
            },
        },
        "pre_near_deduplication": {
            "legal": json_safe_stats(legal_stats),
            "replay": json_safe_stats(replay_stats),
            "estimated_tokens": pre_dedup_tokens,
            "replay_token_share": replay_tokens / pre_dedup_tokens if pre_dedup_tokens else 0.0,
        },
        "final": {
            "documents": final_writer.total_rows,
            "words": final_writer.total_words,
            "estimated_tokens": final_writer.total_tokens,
            "jsonl_bytes": final_writer.total_bytes,
            "jsonl_decimal_gb": final_writer.total_bytes / 1_000_000_000,
            "jsonl_gib": final_writer.total_bytes / (1024**3),
            "documents_by_corpus": dict(final_writer.documents_by_corpus),
            "words_by_corpus": dict(final_writer.words_by_corpus),
            "estimated_tokens_by_corpus": dict(final_writer.tokens_by_corpus),
            "jsonl_bytes_by_corpus": dict(final_writer.bytes_by_corpus),
            "replay_token_share": (
                final_writer.tokens_by_corpus.get("slimpajama_1b", 0) / final_writer.total_tokens
                if final_writer.total_tokens
                else 0.0
            ),
            "shards": final_writer.shards,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)

    corpus_statistics = []
    for corpus in sorted(final_writer.documents_by_corpus):
        corpus_tokens = final_writer.tokens_by_corpus[corpus]
        corpus_statistics.append(
            {
                "corpus": corpus,
                "documents": final_writer.documents_by_corpus[corpus],
                "words": final_writer.words_by_corpus[corpus],
                "estimated_tokens": corpus_tokens,
                "token_share": corpus_tokens / final_writer.total_tokens if final_writer.total_tokens else 0.0,
                "jsonl_bytes": final_writer.bytes_by_corpus[corpus],
            }
        )
    type_jurisdiction_statistics = []
    for text_type, jurisdiction in sorted(final_writer.documents_by_type_jurisdiction):
        key = (text_type, jurisdiction)
        group_tokens = final_writer.tokens_by_type_jurisdiction[key]
        type_jurisdiction_statistics.append(
            {
                "type": text_type,
                "jurisdiction": jurisdiction,
                "documents": final_writer.documents_by_type_jurisdiction[key],
                "estimated_tokens": group_tokens,
                "token_share": group_tokens / final_writer.total_tokens if final_writer.total_tokens else 0.0,
            }
        )
    statistics = {
        "dataset_name": "CPT-V1-EU-1B",
        "token_count_kind": "estimated_1.30_times_whitespace_words",
        "totals": {
            "documents": final_writer.total_rows,
            "words": final_writer.total_words,
            "estimated_tokens": final_writer.total_tokens,
            "jsonl_bytes": final_writer.total_bytes,
            "jsonl_decimal_gb": final_writer.total_bytes / 1_000_000_000,
            "jsonl_gib": final_writer.total_bytes / (1024**3),
            "jsonl_shards": len(final_writer.shards),
        },
        "by_corpus": corpus_statistics,
        "by_type_jurisdiction": type_jurisdiction_statistics,
        "preprocessing": {
            "legal": json_safe_stats(legal_stats),
            "replay": json_safe_stats(replay_stats),
        },
        "shards": final_writer.shards,
        "note": "Estimated tokens are not exact model-tokenizer tokens.",
    }
    statistics_path = args.output_dir / "statistics.json"
    temporary_statistics = statistics_path.with_suffix(".tmp")
    temporary_statistics.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_statistics.replace(statistics_path)

    if not args.keep_work_dir:
        shutil.rmtree(work_dir)

    print(
        f"Complete: {final_writer.total_rows:,} documents, "
        f"~{final_writer.total_tokens:,} estimated tokens, "
        f"{len(final_writer.shards)} JSONL shards",
        flush=True,
    )
    print(f"Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
