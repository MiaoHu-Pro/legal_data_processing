#!/usr/bin/env python3
"""End-to-end smoke/regression test for the preprocessing package.

The script reads only a bounded sample from SlimPajama's test JSONL. It does
not alter the source dataset and does not require a KenLM installation because
the perplexity API is tested with a deterministic KenLM-compatible fake model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from legal_data_process.perplexity_filtering import PerplexityFilter
from legal_data_process.perplexity_filtering import PerplexityFilterConfig
from legal_data_process.rule_based_filters import RuleFilterConfig
from legal_data_process.rule_based_filters import apply_rule_based_filters
from legal_data_process.text_deduplication import DeduplicationConfig
from legal_data_process.text_deduplication import TEXT_DEDUP_SOURCE
from legal_data_process.text_deduplication import deduplicate_records
from legal_data_process.text_normalization import normalize_text


DEFAULT_INPUT = PROJECT_ROOT / "datasets/SlimPajama-1B/row_data/test-00000-of-00001.jsonl"


class DeterministicTestLanguageModel:
    """Tiny stand-in implementing the KenLM ``score`` method for tests."""

    def score(self, sentence: str, bos: bool = True, eos: bool = True) -> float:
        del bos, eos
        words = max(1, len(sentence.split()))
        return (-5.0 if "PERPLEXITY_TEST_NOISE" in sentence else -0.5) * words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--max-records", type=int, default=12)
    parser.add_argument(
        "--max-characters-per-record",
        type=int,
        default=6_000,
        help="Bound test work while retaining real text from every sampled row",
    )
    return parser.parse_args()


def read_small_sample(path: Path, limit: int, max_characters: int) -> Iterator[dict[str, Any]]:
    if limit < 1 or max_characters < 200:
        raise ValueError("Use --max-records >= 1 and --max-characters-per-record >= 200")
    with path.open("r", encoding="utf-8") as input_file:
        for row_number, line in enumerate(input_file, start=1):
            if row_number > limit:
                break
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AssertionError(f"Expected a JSON object on line {row_number}")
            row = dict(row)
            row["id"] = f"slimpajama-test-{row_number:05d}"
            row["text"] = str(row.get("text", ""))[:max_characters]
            yield row


def test_normalization() -> None:
    raw = "\ufb01rst\r\n<p>Section&nbsp;1 &amp; 2</p>\x00\n\n\n\nEnd"
    cleaned = normalize_text(raw)
    assert "first" in cleaned, "NFKC normalization did not expand the ligature"
    assert "<p>" not in cleaned and "&amp;" not in cleaned, "HTML normalization failed"
    assert "\x00" not in cleaned, "control-character removal failed"
    assert "\r" not in cleaned, "line-ending normalization failed"
    assert "\n\n\n\n" not in cleaned, "blank-line limiting failed"


def test_rule_filters() -> None:
    good = (
        "This legal agreement states the rights and obligations of each party. "
        "The claimant submitted written evidence before the scheduled hearing. "
        "Counsel reviewed the applicable statute and cited controlling authority. "
        + "\nPage 12 of 40\n"
        + "The court retains jurisdiction over the dispute and any related claim. "
        "Damages remain subject to proof, mitigation, and the contractual limit. "
        "Both parties may appeal the final judgment under the governing procedure."
    )
    good_result = apply_rule_based_filters(good)
    assert good_result.accepted, good_result.reasons
    assert "Page 12 of 40" not in good_result.text
    assert "removed_page_or_line_markers" in good_result.transformations

    repeated = " ".join(["one two three four five six seven eight nine ten"] * 30)
    bad_result = apply_rule_based_filters(repeated)
    assert not bad_result.accepted
    assert "high_repeated_10gram_ratio" in bad_result.reasons


def test_perplexity_filter() -> None:
    filtering = PerplexityFilter(
        model=DeterministicTestLanguageModel(),
        config=PerplexityFilterConfig(threshold=1500.0),
    )
    normal = filtering.score_text("A coherent legal sentence with ordinary language.")
    noisy = filtering.score_text("PERPLEXITY_TEST_NOISE token sequence")
    assert normal.accepted and math.isfinite(normal.perplexity)
    assert not noisy.accepted
    assert noisy.reason == "perplexity_above_threshold"


def test_real_data_pipeline(records: list[dict[str, Any]]) -> dict[str, int]:
    normalized = []
    for record in records:
        output = dict(record)
        output["text"] = normalize_text(record.get("text"))
        normalized.append(output)

    rule_config = RuleFilterConfig()
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for record in normalized:
        result = apply_rule_based_filters(record["text"], rule_config)
        if result.accepted:
            output = dict(record)
            output["text"] = result.text
            accepted.append(output)
        else:
            rejected += 1
    assert accepted, "No real SlimPajama records passed the default rule filters"

    # Bound MinHash work and add controlled duplicate cases based on real text.
    dedup_input = [deepcopy(record) for record in accepted[:6]]
    exact_duplicate = deepcopy(dedup_input[0])
    exact_duplicate["id"] = "injected-exact-duplicate"
    near_duplicate = deepcopy(dedup_input[0])
    near_duplicate["id"] = "injected-near-duplicate"
    near_duplicate["text"] += "\nA short additional sentence used to test near duplication."
    dedup_input.extend([exact_duplicate, near_duplicate])

    dedup_result = deduplicate_records(
        dedup_input,
        DeduplicationConfig(
            similarity_threshold=0.50,
            ngram_size=5,
            num_permutations=64,
            bands=16,
            seed=42,
        ),
    )
    assert dedup_result.exact_duplicates >= 1, "Injected exact duplicate was not removed"
    assert dedup_result.near_duplicates >= 1, "Injected near duplicate was not removed"
    assert dedup_result.output_count <= dedup_result.input_count - 2
    assert all("dedup_cluster_id" in record for record in dedup_result.kept_records)
    assert TEXT_DEDUP_SOURCE.is_dir(), "Local text-dedup source was not used"

    perplexity_filter = PerplexityFilter(
        model=DeterministicTestLanguageModel(),
        config=PerplexityFilterConfig(threshold=1500.0),
    )
    perplexity_passed = sum(
        perplexity_filter.score_text(record["text"]).accepted
        for record in dedup_result.kept_records
    )
    assert perplexity_passed == dedup_result.output_count

    return {
        "input_sample": len(records),
        "rule_accepted": len(accepted),
        "rule_rejected": rejected,
        "dedup_test_input": dedup_result.input_count,
        "dedup_kept": dedup_result.output_count,
        "exact_duplicates_removed": dedup_result.exact_duplicates,
        "near_duplicates_removed": dedup_result.near_duplicates,
        "perplexity_passed": perplexity_passed,
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input JSONL does not exist: {args.input}")

    test_normalization()
    test_rule_filters()
    test_perplexity_filter()
    records = list(read_small_sample(args.input, args.max_records, args.max_characters_per_record))
    if not records:
        raise AssertionError("The input JSONL contained no rows")
    summary = test_real_data_pipeline(records)

    print("All preprocessing tests passed.")
    print(f"SlimPajama input: {args.input}")
    print(f"Local text-dedup source: {TEXT_DEDUP_SOURCE}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
