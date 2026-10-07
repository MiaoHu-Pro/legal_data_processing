"""Exact and MinHash/LSH near-document deduplication.

This module deliberately reuses the vendored ``text-dedup`` package's n-gram
and Union-Find implementations from ``src/text-dedup/src/text_dedup``.  The
wrapper keeps the legal-data API dependency-light and verifies LSH candidates
with exact Jaccard similarity before merging them.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 64) - 1


def _load_text_dedup_utilities() -> tuple[Callable[..., Any], type[Any], Callable[..., float], Path]:
    """Import dependency-light primitives from the requested local checkout."""

    vendored_source = Path(__file__).resolve().parents[1] / "text-dedup" / "src"
    package_dir = vendored_source / "text_dedup"
    if not package_dir.is_dir():
        raise RuntimeError(f"Local text-dedup package not found at {package_dir}")
    source_string = str(vendored_source)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    from text_dedup.utils.jaccard import jaccard_similarity
    from text_dedup.utils.tokenization import ngrams
    from text_dedup.utils.union_find import UnionFind

    return ngrams, UnionFind, jaccard_similarity, vendored_source


(
    _text_dedup_ngrams,
    _TextDedupUnionFind,
    _text_dedup_jaccard,
    TEXT_DEDUP_SOURCE,
) = _load_text_dedup_utilities()


@dataclass(frozen=True)
class DeduplicationConfig:
    """Configuration for exact and MinHash near deduplication."""

    similarity_threshold: float = 0.50
    ngram_size: int = 5
    min_words: int = 5
    num_permutations: int = 128
    bands: int = 32
    seed: int = 42
    text_column: str = "text"
    id_column: str = "id"
    cluster_column: str = "dedup_cluster_id"
    annotate_kept_records: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if self.ngram_size < 1 or self.min_words < 1:
            raise ValueError("ngram_size and min_words must be positive")
        if self.num_permutations < 1 or self.bands < 1:
            raise ValueError("num_permutations and bands must be positive")
        if self.num_permutations % self.bands:
            raise ValueError("num_permutations must be divisible by bands")


@dataclass(frozen=True)
class DuplicateMatch:
    """Audit record for a removed document."""

    duplicate_index: int
    representative_index: int
    duplicate_id: str
    representative_id: str
    kind: str
    similarity: float
    cluster_id: str


@dataclass(frozen=True)
class DeduplicationResult:
    """Deduplicated records and provenance needed to audit removals."""

    kept_records: tuple[dict[str, Any], ...]
    removed_records: tuple[dict[str, Any], ...]
    matches: tuple[DuplicateMatch, ...]
    cluster_by_index: Mapping[int, str]
    input_count: int
    output_count: int
    exact_duplicates: int
    near_duplicates: int


@dataclass(frozen=True)
class TextDedupRunConfig:
    """Settings for a full run of the vendored text-dedup MinHash pipeline."""

    similarity_threshold: float = 0.50
    ngram_size: int = 5
    num_permutations: int = 240
    seed: int = 42
    num_processes: int = max(1, min(8, os.cpu_count() or 1))
    text_column: str = "text"
    # Exact all-pairs verification within every LSH candidate cluster is
    # quadratic. Keep it opt-in for large corpora; the upstream text-dedup
    # configuration also defaults this setting to False.
    check_false_positives: bool = False
    save_clusters: bool = True
    keep_index_column: bool = True
    keep_cluster_column: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if self.ngram_size < 1 or self.num_permutations < 1 or self.num_processes < 1:
            raise ValueError("ngram_size, num_permutations, and num_processes must be positive")


def canonicalize_for_exact_deduplication(text: object) -> str:
    """Make inconsequential Unicode/whitespace differences compare equally."""

    value = unicodedata.normalize("NFKC", "" if text is None else str(text)).casefold()
    return " ".join(value.split())


def _document_id(record: Mapping[str, Any], config: DeduplicationConfig) -> str:
    value = record.get(config.id_column)
    if value is not None and str(value).strip():
        return str(value)
    payload = canonicalize_for_exact_deduplication(record.get(config.text_column))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shingles(text: object, config: DeduplicationConfig) -> set[bytes]:
    words = _TOKEN_RE.findall("" if text is None else str(text).casefold())
    return {
        " ".join(items).encode("utf-8")
        for items in _text_dedup_ngrams(words, config.ngram_size, config.min_words)
    }


def _coefficients(config: DeduplicationConfig) -> tuple[list[int], list[int]]:
    generator = random.Random(config.seed)
    multipliers = [generator.randrange(1, _PRIME) for _ in range(config.num_permutations)]
    offsets = [generator.randrange(0, _PRIME) for _ in range(config.num_permutations)]
    return multipliers, offsets


def _minhash_signature(
    shingles: set[bytes],
    multipliers: Sequence[int],
    offsets: Sequence[int],
) -> tuple[int, ...]:
    if not shingles:
        return tuple([_MAX_HASH] * len(multipliers))
    base_hashes = [
        int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "little") % _PRIME
        for shingle in shingles
    ]
    return tuple(
        min(((multiplier * value + offset) % _PRIME) for value in base_hashes)
        for multiplier, offset in zip(multipliers, offsets)
    )


def deduplicate_records(
    records: Iterable[Mapping[str, Any]],
    config: DeduplicationConfig | None = None,
) -> DeduplicationResult:
    """Remove exact and near duplicates from an in-memory record collection.

    The full texts are materialized by design. For the 283 GB corpus, invoke
    this function on compact partition/index batches or build an external
    signature store; do not pass the entire raw corpus to one Python process.
    """

    config = config or DeduplicationConfig()
    rows = [dict(record) for record in records]
    union_find = _TextDedupUnionFind()
    canonical_hashes: dict[str, int] = {}
    for index, row in enumerate(rows):
        canonical = canonicalize_for_exact_deduplication(row.get(config.text_column))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        first_index = canonical_hashes.setdefault(digest, index)
        if first_index != index:
            union_find.union(first_index, index)
        else:
            union_find.find(index)

    # Near-deduplicate only one representative of every exact-text group.
    exact_representatives = [index for index in range(len(rows)) if union_find.find(index) == index]
    shingle_sets = {index: _shingles(rows[index].get(config.text_column), config) for index in exact_representatives}
    eligible = [index for index in exact_representatives if shingle_sets[index]]
    multipliers, offsets = _coefficients(config)
    rows_per_band = config.num_permutations // config.bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    candidate_pairs: set[tuple[int, int]] = set()

    for index in eligible:
        signature = _minhash_signature(shingle_sets[index], multipliers, offsets)
        for band in range(config.bands):
            start = band * rows_per_band
            key = (band, signature[start : start + rows_per_band])
            members = buckets.setdefault(key, [])
            candidate_pairs.update((member, index) for member in members)
            members.append(index)

    for left, right in sorted(candidate_pairs):
        similarity = _text_dedup_jaccard(shingle_sets[left], shingle_sets[right])
        if similarity >= config.similarity_threshold:
            union_find.union(left, right)

    clusters: dict[int, list[int]] = {}
    for index in range(len(rows)):
        clusters.setdefault(union_find.find(index), []).append(index)

    # Union-by-rank roots are not guaranteed to be the earliest record.
    representative_for_index: dict[int, int] = {}
    for members in clusters.values():
        representative = min(members)
        representative_for_index.update((member, representative) for member in members)

    ids = [_document_id(row, config) for row in rows]
    cluster_by_index: dict[int, str] = {}
    for index, representative in representative_for_index.items():
        cluster_digest = hashlib.sha256(ids[representative].encode("utf-8")).hexdigest()
        cluster_by_index[index] = "sha256:" + cluster_digest

    kept_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    matches: list[DuplicateMatch] = []
    exact_duplicate_count = 0
    near_duplicate_count = 0

    for index, row in enumerate(rows):
        representative = representative_for_index[index]
        if config.annotate_kept_records:
            row[config.cluster_column] = cluster_by_index[index]
        if index == representative:
            kept_records.append(row)
            continue

        removed_records.append(row)
        same_exact_text = (
            canonicalize_for_exact_deduplication(row.get(config.text_column))
            == canonicalize_for_exact_deduplication(rows[representative].get(config.text_column))
        )
        if same_exact_text:
            kind = "exact"
            similarity = 1.0
            exact_duplicate_count += 1
        else:
            kind = "near"
            similarity = _text_dedup_jaccard(
                _shingles(row.get(config.text_column), config),
                _shingles(rows[representative].get(config.text_column), config),
            )
            near_duplicate_count += 1
        matches.append(
            DuplicateMatch(
                duplicate_index=index,
                representative_index=representative,
                duplicate_id=ids[index],
                representative_id=ids[representative],
                kind=kind,
                similarity=similarity,
                cluster_id=cluster_by_index[index],
            )
        )

    return DeduplicationResult(
        kept_records=tuple(kept_records),
        removed_records=tuple(removed_records),
        matches=tuple(matches),
        cluster_by_index=cluster_by_index,
        input_count=len(rows),
        output_count=len(kept_records),
        exact_duplicates=exact_duplicate_count,
        near_duplicates=near_duplicate_count,
    )


def run_text_dedup_minhash(
    input_files: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    config: TextDedupRunConfig | None = None,
) -> None:
    """Run the vendored text-dedup package's complete MinHash pipeline.

    This adapter is intended for the full corpus. It delegates loading,
    fingerprinting, clustering, false-positive verification, filtering, and
    Hugging Face dataset output to ``text_dedup.minhash.main``. The local
    checkout's optional dependencies must be installed first.
    """

    config = config or TextDedupRunConfig()
    paths = [input_files] if isinstance(input_files, (str, Path)) else list(input_files)
    resolved_paths = [str(Path(path).resolve()) for path in paths]
    missing = [path for path in resolved_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Deduplication input files do not exist: {missing}")
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_path}")

    try:
        from datasets import Features
        from datasets import Value
        from text_dedup.config import Config
        from text_dedup.config import LocalInputConfig
        from text_dedup.config import MinHashAlgorithmConfig
        from text_dedup.config import OutputConfig
        from text_dedup.config.debug import DebugConfig
        from text_dedup.minhash import main as text_dedup_minhash_main
    except ImportError as exc:
        raise RuntimeError(
            "The full text-dedup pipeline requires the dependencies declared "
            "in src/text-dedup/pyproject.toml (including polars, "
            "polars-grouper, regex, scipy, and pydantic-settings)."
        ) from exc

    # Hugging Face otherwise infers Arrow ``string`` (32-bit offsets). During
    # text-dedup's indexing map, a worker can concatenate more than 2 GiB of
    # legal text and raise ``ArrowInvalid: offset overflow``. Use 64-bit Arrow
    # offsets from the initial JSON load so every later map preserves them.
    input_features = Features(
        {
            "id": Value("large_string"),
            "text": Value("large_string"),
            "corpus": Value("large_string"),
            "type": Value("large_string"),
            "jurisdiction": Value("large_string"),
            "source": Value("large_string"),
            "source_file": Value("large_string"),
            "source_row": Value("int64"),
            "estimated_tokens": Value("int64"),
            # Missing optional values are loaded as nulls.
            "perplexity": Value("float64"),
        }
    )
    input_config = LocalInputConfig(
        file_type="json",
        read_arguments={
            "path": "json",
            "data_files": resolved_paths,
            "split": "train",
            "features": input_features,
        },
    )
    algorithm_config = MinHashAlgorithmConfig(
        algorithm_name="minhash",
        text_column=config.text_column,
        seed=config.seed,
        num_proc=config.num_processes,
        batch_size=10_000,
        num_perm=config.num_permutations,
        threshold=config.similarity_threshold,
        ngram_size=config.ngram_size,
        check_false_positive=config.check_false_positives,
    )
    output_config = OutputConfig(
        output_dir=str(output_path.resolve()),
        skip_filtering=False,
        clean_cache=False,
        save_clusters=config.save_clusters,
        keep_index_column=config.keep_index_column,
        keep_cluster_column=config.keep_cluster_column,
    )
    debug_config = DebugConfig(enable_profiling=False)

    # Config customizes Pydantic's settings sources to TOML-only, which causes
    # normal constructor/model_validate input to be discarded. Each nested
    # model above is fully validated; model_construct safely assembles them.
    package_config = Config.model_construct(
        input=input_config,
        algorithm=algorithm_config,
        output=output_config,
        debug=debug_config,
    )
    text_dedup_minhash_main(package_config)


__all__ = [
    "DeduplicationConfig", "DeduplicationResult", "DuplicateMatch",
    "TEXT_DEDUP_SOURCE", "TextDedupRunConfig",
    "canonicalize_for_exact_deduplication", "deduplicate_records",
    "run_text_dedup_minhash",
]
