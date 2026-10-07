#!/usr/bin/env python3
"""Create a post-dedup CPT mixture with proportional legal strata and replay.

The requested token target is enforced *after* normalization, rule filtering,
exact deduplication, and optional MinHash near deduplication. Legal tokens are
allocated in proportion to the measured Multi_Legal_Pile_Commercial
type-by-jurisdiction distribution. The default corpus mixture is 90% legal and
10% SlimPajama replay.

Counts are estimates (1.3 times whitespace-delimited words), so an exact target
is generally impossible without truncating a document. Selection stays within
each quota and reports the small indivisible-document remainder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import sys
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ is required; found {sys.version}")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cpt_data_sample_creation_v1_eu_1b import FinalShardWriter
from cpt_data_sample_creation_v1_eu_1b import RotatingJsonlWriter
from cpt_data_sample_creation_v1_eu_1b import count_words
from cpt_data_sample_creation_v1_eu_1b import estimate_tokens
from cpt_data_sample_creation_v1_eu_1b import iter_deduplicated_dataset
from cpt_data_sample_creation_v1_eu_1b import iter_jsonl
from cpt_data_sample_creation_v1_eu_1b import iter_replay_round_robin
from cpt_data_sample_creation_v1_eu_1b import iter_staged_jsonl
from cpt_data_sample_creation_v1_eu_1b import json_safe_stats
from cpt_data_sample_creation_v1_eu_1b import make_stats
from cpt_data_sample_creation_v1_eu_1b import process_candidate
from legal_data_process.perplexity_filtering import PerplexityFilter
from legal_data_process.perplexity_filtering import PerplexityFilterConfig
from legal_data_process.rule_based_filters import RuleFilterConfig
from legal_data_process.text_deduplication import TextDedupRunConfig
from legal_data_process.text_deduplication import run_text_dedup_minhash
from legal_data_process.text_normalization import NormalizationConfig


DEFAULT_LEGAL_DIR = PROJECT_ROOT / "datasets/Multi_Legal_Pile_Commercial"
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "datasets/SlimPajama-1B/row_data"

# Measured before preprocessing, from
# multi_legal_pile_commercial_stats_by_type_jurisdiction.csv.
LEGAL_STRATUM_TOKENS: dict[tuple[str, str], int] = {
    ("caselaw", "EU"): 289_759_375,
    ("caselaw", "US"): 32_750_957_349,
    ("contracts", "EU"): 50_738_795,
    ("contracts", "US"): 6_889_351_099,
    ("legal-mc4", "N/A"): 628_590_444,
    ("legislation", "EU"): 994_351_311,
    ("legislation", "Switzerland"): 1_760_762,
    ("legislation", "UK"): 54_241_476,
    ("legislation", "US"): 2_244_323_874,
    ("other", "N/A"): 9_638_782_574,
    ("other", "US"): 1_611_777_462,
}
LEGAL_TOTAL_TOKENS = sum(LEGAL_STRATUM_TOKENS.values())
REPLAY_KEY = ("slimpajama_1b", "general", "N/A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-target-tokens", type=int, required=True)
    parser.add_argument("--legal-share", type=float, default=0.90)
    parser.add_argument("--candidate-oversample-factor", type=float, default=1.50)
    parser.add_argument("--legal-dir", type=Path, default=DEFAULT_LEGAL_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--output-prefix")
    parser.add_argument("--shard-target-tokens", type=int, default=200_000_000)
    parser.add_argument("--kenlm-model", type=Path)
    parser.add_argument("--perplexity-threshold", type=float, default=1500.0)
    parser.add_argument("--apply-kenlm-to-replay", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--near-dedup", choices=("text-dedup", "skip"), default="text-dedup")
    parser.add_argument("--dedup-threshold", type=float, default=0.50)
    parser.add_argument("--dedup-processes", type=int, default=4)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffle-buckets", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--quota-tolerance", type=float, default=0.001)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--resume-from-preprocessed", action="store_true")
    args = parser.parse_args()

    if args.total_target_tokens < 1:
        parser.error("--total-target-tokens must be positive")
    if not 0.0 < args.legal_share < 1.0:
        parser.error("--legal-share must be between 0 and 1")
    if args.candidate_oversample_factor <= 1.0:
        parser.error("--candidate-oversample-factor must be greater than 1")
    if not 0.0 <= args.quota_tolerance < 1.0:
        parser.error("--quota-tolerance must be in [0, 1)")
    if min(args.shard_target_tokens, args.dedup_processes, args.shuffle_buckets, args.progress_every) < 1:
        parser.error("shard target, process count, bucket count, and progress interval must be positive")

    billions = args.total_target_tokens / 1_000_000_000
    label = f"{billions:g}B"
    args.output_dir = args.output_dir or PROJECT_ROOT / f"datasets/CPT-Proportional-{label}"
    args.dataset_name = args.dataset_name or f"CPT-Proportional-{label}"
    args.output_prefix = args.output_prefix or f"cpt-proportional-{label.lower()}"
    return args


def allocate_integer(total: int, weights: Mapping[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    weight_sum = sum(weights.values())
    exact = {key: total * value / weight_sum for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(weights, key=lambda key: (exact[key] - result[key], key), reverse=True)
    for key in order[:remainder]:
        result[key] += 1
    return result


def legal_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("type") or row.get("text_type") or "UNKNOWN").casefold(),
        str(row.get("jurisdiction") or "UNKNOWN"),
    )


def final_group(row: Mapping[str, Any]) -> tuple[str, str, str]:
    corpus = str(row.get("corpus") or "UNKNOWN")
    if corpus == "slimpajama_1b":
        return REPLAY_KEY
    return (corpus, str(row.get("type") or "UNKNOWN").casefold(), str(row.get("jurisdiction") or "UNKNOWN"))


def selected_by_probability(corpus: str, source_file: str, row_number: int, probability: float, seed: int) -> bool:
    if probability >= 1.0:
        return True
    payload = f"{seed}\0{corpus}\0{source_file}\0{row_number}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value < int(probability * (1 << 64))


def replay_total_tokens(paths: list[Path], progress_every: int) -> tuple[int, int]:
    rows = 0
    tokens = 0
    print("Measuring the complete SlimPajama replay corpus", flush=True)
    for _, _, row, error in iter_replay_round_robin(paths):
        rows += 1
        if error is None and row is not None:
            tokens += estimate_tokens(str(row.get("text") or ""))
        if rows % progress_every == 0:
            print(f"Replay measured={rows:,} tokens~{tokens:,}", flush=True)
    return rows, tokens


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_candidates(
    args: argparse.Namespace,
    legal_files: list[Path],
    replay_files: list[Path],
    work_dir: Path,
    legal_quotas: Mapping[tuple[str, str], int],
    replay_quota: int,
) -> tuple[list[Path], dict[str, Any]]:
    replay_rows, replay_tokens = replay_total_tokens(replay_files, args.progress_every)
    if replay_quota > replay_tokens:
        raise RuntimeError(f"Replay quota {replay_quota:,} exceeds available ~{replay_tokens:,} tokens")

    legal_probabilities = {
        key: min(1.0, args.candidate_oversample_factor * quota / LEGAL_STRATUM_TOKENS[key])
        for key, quota in legal_quotas.items()
    }
    replay_probability = min(1.0, args.candidate_oversample_factor * replay_quota / replay_tokens)

    normalization_config = NormalizationConfig(unicode_form="NFKC")
    rule_config = RuleFilterConfig()
    perplexity_filter = None
    if not args.skip_perplexity:
        perplexity_filter = PerplexityFilter(
            model_path=args.kenlm_model,
            config=PerplexityFilterConfig(threshold=args.perplexity_threshold),
        )

    accepted = RotatingJsonlWriter(work_dir / "preprocessed", "accepted")
    rejected = RotatingJsonlWriter(work_dir / "rejected", "rejected")
    database = sqlite3.connect(work_dir / "exact_dedup.sqlite")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("CREATE TABLE seen (text_hash TEXT PRIMARY KEY) WITHOUT ROWID")
    legal_stats = make_stats()
    replay_stats = make_stats()

    print("Sampling proportional legal candidates", flush=True)
    for path in legal_files:
        for row_number, row, error in iter_jsonl(path):
            legal_stats["input_rows"] += 1
            if error is not None:
                legal_stats["invalid_rows"] += 1
            elif row is not None:
                key = legal_key(row)
                probability = legal_probabilities.get(key)
                if probability is not None and selected_by_probability(
                    "multilegalpile_commercial", path.name, row_number, probability, args.shuffle_seed
                ):
                    legal_stats["selected_rows"] += 1
                    process_candidate(
                        row,
                        corpus="multilegalpile_commercial",
                        source_file=path.name,
                        source_row=row_number,
                        stats=legal_stats,
                        accepted_writer=accepted,
                        rejected_writer=rejected,
                        exact_db=database,
                        normalization_config=normalization_config,
                        rule_config=rule_config,
                        perplexity_filter=perplexity_filter,
                    )
            if legal_stats["input_rows"] % args.progress_every == 0:
                database.commit()
                print(
                    f"Legal scanned={legal_stats['input_rows']:,} selected={legal_stats['selected_rows']:,} "
                    f"accepted={legal_stats['accepted_rows']:,} tokens~{legal_stats['accepted_estimated_tokens']:,}",
                    flush=True,
                )

    print(f"Sampling replay candidates with probability={replay_probability:.6f}", flush=True)
    for path, row_number, row, error in iter_replay_round_robin(replay_files):
        replay_stats["input_rows"] += 1
        if error is not None:
            replay_stats["invalid_rows"] += 1
        elif row is not None and selected_by_probability(
            "slimpajama_1b", path.name, row_number, replay_probability, args.shuffle_seed
        ):
            replay_stats["selected_rows"] += 1
            process_candidate(
                row,
                corpus="slimpajama_1b",
                source_file=path.name,
                source_row=row_number,
                stats=replay_stats,
                accepted_writer=accepted,
                rejected_writer=rejected,
                exact_db=database,
                normalization_config=normalization_config,
                rule_config=rule_config,
                perplexity_filter=perplexity_filter if args.apply_kenlm_to_replay else None,
            )
        if replay_stats["input_rows"] % args.progress_every == 0:
            database.commit()
            print(
                f"Replay scanned={replay_stats['input_rows']:,} selected={replay_stats['selected_rows']:,} "
                f"accepted={replay_stats['accepted_rows']:,} tokens~{replay_stats['accepted_estimated_tokens']:,}",
                flush=True,
            )

    database.commit()
    database.close()
    accepted.close()
    rejected.close()
    if not accepted.paths:
        raise RuntimeError("No candidate records survived preprocessing")

    checkpoint = {
        "requested_total_tokens": args.total_target_tokens,
        "legal_share": args.legal_share,
        "candidate_oversample_factor": args.candidate_oversample_factor,
        "shuffle_seed": args.shuffle_seed,
        "replay_source_rows": replay_rows,
        "replay_source_estimated_tokens": replay_tokens,
        "legal_sampling_probabilities": {f"{a}|{b}": p for (a, b), p in legal_probabilities.items()},
        "replay_sampling_probability": replay_probability,
        "legal": json_safe_stats(legal_stats),
        "replay": json_safe_stats(replay_stats),
        "normalization": asdict(normalization_config),
        "rule_filter": asdict(rule_config),
    }
    (work_dir / "preprocessing_complete.json").write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return accepted.paths, checkpoint


def bucket_survivors(
    records: Iterable[Mapping[str, Any]], work_dir: Path, bucket_count: int, seed: int
) -> tuple[list[Path], Counter[tuple[str, str, str]]]:
    directory = work_dir / "post_dedup_buckets"
    directory.mkdir(parents=True, exist_ok=False)
    paths = [directory / f"bucket-{index:04d}.jsonl" for index in range(bucket_count)]
    available: Counter[tuple[str, str, str]] = Counter()
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("w", encoding="utf-8")) for path in paths]
        for row in records:
            payload = dict(row)
            priority = hashlib.sha256(f"{seed}\0{payload.get('id', '')}".encode()).hexdigest()
            bucket = int(priority[:16], 16) * bucket_count // (1 << 64)
            payload["__selection_priority__"] = priority
            handles[bucket].write(json.dumps(payload, ensure_ascii=False) + "\n")
            available[final_group(payload)] += int(payload.get("estimated_tokens") or estimate_tokens(payload.get("text", "")))
    return paths, available


def select_final(
    bucket_paths: list[Path],
    quotas: Mapping[tuple[str, str, str], int],
    output_dir: Path,
    output_prefix: str,
    shard_target_tokens: int,
) -> tuple[FinalShardWriter, Counter[tuple[str, str, str]]]:
    selected: Counter[tuple[str, str, str]] = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = FinalShardWriter(output_dir, shard_target_tokens, output_prefix)
    for path in bucket_paths:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows.sort(key=lambda row: row["__selection_priority__"])
        for row in rows:
            group = final_group(row)
            quota = quotas.get(group)
            if quota is None:
                continue
            tokens = int(row.get("estimated_tokens") or estimate_tokens(row.get("text", "")))
            if selected[group] + tokens > quota:
                continue
            row.pop("__selection_priority__", None)
            row.pop("__INDEX__", None)
            cluster = row.pop("__CLUSTER__", None)
            if cluster is not None:
                row["dedup_cluster_id"] = f"text-dedup:{cluster}"
            writer.write(row)
            selected[group] += tokens
    writer.close()
    return writer, selected


def main() -> None:
    args = parse_args()
    legal_files = sorted(args.legal_dir.glob("multi_legal_pile_en_commercial_*.jsonl"))
    replay_files = sorted(args.replay_dir.glob("train-*.jsonl"))
    if not legal_files or not replay_files:
        raise SystemExit("Legal or SlimPajama input JSONL files were not found")
    if not args.resume_from_preprocessed and not args.skip_perplexity and args.kenlm_model is None:
        raise SystemExit("Provide --kenlm-model or explicitly use --skip-perplexity")

    legal_target = round(args.total_target_tokens * args.legal_share)
    replay_target = args.total_target_tokens - legal_target
    legal_quotas = allocate_integer(legal_target, LEGAL_STRATUM_TOKENS)
    quotas: dict[tuple[str, str, str], int] = {
        ("multilegalpile_commercial", text_type, jurisdiction): quota
        for (text_type, jurisdiction), quota in legal_quotas.items()
    }
    quotas[REPLAY_KEY] = replay_target
    for key, quota in legal_quotas.items():
        if quota > LEGAL_STRATUM_TOKENS[key]:
            raise SystemExit(f"Requested quota exceeds source stratum {key}: {quota:,}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_from_preprocessed:
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    if args.resume_from_preprocessed and (
        (args.output_dir / "manifest.json").exists()
        or list(args.output_dir.glob(f"{args.output_prefix}-*.jsonl"))
    ):
        raise SystemExit("Final artifacts already exist; refusing to overwrite them during resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / ".work"
    start = time.time()

    if args.resume_from_preprocessed:
        checkpoint_path = work_dir / "preprocessing_complete.json"
        if not checkpoint_path.is_file():
            raise SystemExit(f"Safe resume marker is missing: {checkpoint_path}")
        checkpoint = load_config(checkpoint_path)
        expected_resume_values = {
            "requested_total_tokens": args.total_target_tokens,
            "legal_share": args.legal_share,
            "candidate_oversample_factor": args.candidate_oversample_factor,
            "shuffle_seed": args.shuffle_seed,
        }
        mismatches = {
            key: {"checkpoint": checkpoint.get(key), "requested": value}
            for key, value in expected_resume_values.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise SystemExit(f"Resume configuration does not match preprocessing checkpoint: {mismatches}")
        accepted_paths = sorted((work_dir / "preprocessed").glob("accepted-*.jsonl"))
        if not accepted_paths:
            raise SystemExit("No completed preprocessed shards were found")
        print(f"Resuming from {len(accepted_paths)} completed preprocessed shards", flush=True)
    else:
        work_dir.mkdir(parents=True, exist_ok=False)
        accepted_paths, checkpoint = prepare_candidates(
            args, legal_files, replay_files, work_dir, legal_quotas, replay_target
        )

    if args.near_dedup == "text-dedup":
        dedup_dir = work_dir / "text_dedup_output"
        print("Running MinHash near deduplication", flush=True)
        run_text_dedup_minhash(
            accepted_paths,
            dedup_dir,
            TextDedupRunConfig(
                similarity_threshold=args.dedup_threshold,
                ngram_size=5,
                num_permutations=240,
                seed=args.shuffle_seed,
                num_processes=args.dedup_processes,
                check_false_positives=False,
                save_clusters=True,
                keep_index_column=False,
                keep_cluster_column=True,
            ),
        )
        survivors: Iterable[Mapping[str, Any]] = iter_deduplicated_dataset(dedup_dir)
    else:
        print("WARNING: near deduplication skipped", flush=True)
        survivors = iter_staged_jsonl(accepted_paths)

    print("Bucketing post-dedup survivors for stratified final selection", flush=True)
    bucket_paths, available = bucket_survivors(survivors, work_dir, args.shuffle_buckets, args.shuffle_seed)
    deficits_before_selection = {
        str(key): quota - available.get(key, 0)
        for key, quota in quotas.items()
        if available.get(key, 0) < quota
    }
    if deficits_before_selection:
        raise RuntimeError(
            "Post-dedup candidate pool is below one or more quotas; rerun from scratch with a larger "
            f"--candidate-oversample-factor. Deficits: {deficits_before_selection}"
        )

    writer, selected = select_final(
        bucket_paths, quotas, args.output_dir, args.output_prefix, args.shard_target_tokens
    )
    deficits = {key: quotas[key] - selected.get(key, 0) for key in quotas}
    total_deficit = args.total_target_tokens - writer.total_tokens
    if total_deficit > args.total_target_tokens * args.quota_tolerance:
        raise RuntimeError(
            f"Final indivisible-document deficit {total_deficit:,} exceeds tolerance; deficits={deficits}"
        )

    quota_rows = []
    for key in sorted(quotas):
        quota_rows.append(
            {
                "corpus": key[0],
                "type": key[1],
                "jurisdiction": key[2],
                "target_estimated_tokens": quotas[key],
                "available_post_dedup_tokens": available[key],
                "selected_estimated_tokens": selected[key],
                "deficit_tokens": deficits[key],
            }
        )
    manifest = {
        "dataset_name": args.dataset_name,
        "created_at_unix": int(time.time()),
        "runtime_seconds": time.time() - start,
        "token_count_kind": "estimated_1.30_times_whitespace_words",
        "configuration": {
            "requested_total_tokens": args.total_target_tokens,
            "legal_share": args.legal_share,
            "replay_share": 1.0 - args.legal_share,
            "candidate_oversample_factor": args.candidate_oversample_factor,
            "near_deduplication": args.near_dedup,
            "near_dedup_threshold": args.dedup_threshold if args.near_dedup == "text-dedup" else None,
            "dedup_candidate_verification": False,
            "shuffle_seed": args.shuffle_seed,
            "output_shard_target_tokens": args.shard_target_tokens,
        },
        "preprocessing": checkpoint,
        "quotas": quota_rows,
        "final": {
            "documents": writer.total_rows,
            "estimated_tokens": writer.total_tokens,
            "target_deficit_tokens": total_deficit,
            "jsonl_bytes": writer.total_bytes,
            "jsonl_shards": len(writer.shards),
            "shards": writer.shards,
        },
        "warning": "Estimated tokens are not exact model-tokenizer tokens.",
    }
    statistics = {
        "dataset_name": args.dataset_name,
        "totals": manifest["final"],
        "quotas": quota_rows,
        "shards": writer.shards,
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
        f"Complete: {writer.total_rows:,} documents, ~{writer.total_tokens:,}/{args.total_target_tokens:,} "
        f"estimated tokens, deficit={total_deficit:,}",
        flush=True,
    )
    print(f"Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()


"""
Created the standalone script:

  datasets_processing/cpt_data_sample_creation_proportional.py

  No existing V1, V2, generic, package, or Slurm scripts were modified for this implementation.

  For a 5B post-dedup dataset:

  python datasets_processing/cpt_data_sample_creation_proportional.py \
    --total-target-tokens 5000000000 \
    --legal-share 0.9 \
    --candidate-oversample-factor 1.5 \
    --skip-perplexity \
    --dedup-processes 4

  This produces:

  - 4.5B estimated legal tokens
  - 0.5B estimated SlimPajama tokens
  - Legal data distributed across all type × jurisdiction strata according to the supplied proportions
  - Output under datasets/CPT-Proportional-5B
  - Approximately 200M tokens per JSONL shard

  The script:

  1. Oversamples deterministic candidates.
  2. Applies normalization and rule filters.
  3. Applies exact deduplication.
  4. Applies MinHash near deduplication.
  5. Selects post-dedup documents according to the final quotas.
  6. Reports any small remainder caused by indivisible documents.
  7. Refuses completion if the deficit exceeds 0.1%.

  It also writes manifest.json and statistics.json with target, available, selected, and deficit tokens for every
  stratum.

  The quota calculations, Python syntax, deterministic selection, output sharding, and exact 90:10 post-dedup
  selection were regression-tested.

"""