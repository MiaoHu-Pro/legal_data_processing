"""Deterministic quality filters and common-noise cleanup rules."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_STANDALONE_PAGE_RE = re.compile(
    r"^\s*(?:page\s+\d{1,6}(?:\s+(?:of|/)\s*\d{1,6})?|\d{1,6}\s+(?:of|/)\s*\d{1,6})\s*$",
    re.IGNORECASE,
)
_STANDALONE_LINE_RE = re.compile(r"^\s*lines?\s+\d{1,7}(?:\s*[-–]\s*\d{1,7})?\s*$", re.IGNORECASE)
_REPEATED_CHAR_RE = re.compile(r"(.)\1{9,}", re.DOTALL)


@dataclass(frozen=True)
class RuleFilterConfig:
    """Thresholds for conservative document-level quality filtering."""

    min_characters: int = 200
    min_words: int = 30
    max_characters: int | None = None
    min_alphabetic_ratio: float = 0.20
    max_replacement_character_ratio: float = 0.01
    max_repeated_line_ratio: float = 0.60
    max_repeated_ngram_ratio: float = 0.50
    repeated_ngram_size: int = 10
    remove_page_and_line_markers: bool = True
    collapse_repeated_punctuation: bool = True
    reject_remaining_html: bool = False

    def __post_init__(self) -> None:
        if self.min_characters < 0 or self.min_words < 0:
            raise ValueError("minimum lengths must be non-negative")
        if self.max_characters is not None and self.max_characters < self.min_characters:
            raise ValueError("max_characters must be at least min_characters")
        for name in (
            "min_alphabetic_ratio",
            "max_replacement_character_ratio",
            "max_repeated_line_ratio",
            "max_repeated_ngram_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.repeated_ngram_size < 1:
            raise ValueError("repeated_ngram_size must be positive")


@dataclass(frozen=True)
class RuleFilterResult:
    """Decision, transformed text, and auditable quality measurements."""

    accepted: bool
    text: str
    reasons: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    transformations: tuple[str, ...] = ()
    metrics: Mapping[str, float | int] = field(default_factory=dict)


def _repeated_line_ratio(text: str) -> float:
    lines = [" ".join(line.lower().split()) for line in text.splitlines()]
    substantive = [line for line in lines if len(line) >= 20]
    if not substantive:
        return 0.0
    counts = Counter(substantive)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(substantive)


def _repeated_ngram_ratio(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def remove_common_noise(text: str, config: RuleFilterConfig | None = None) -> tuple[str, tuple[str, ...]]:
    """Remove standalone page/line markers and pathological punctuation runs."""

    config = config or RuleFilterConfig()
    transformations: list[str] = []
    lines = text.splitlines()

    if config.remove_page_and_line_markers:
        retained = [
            line
            for line in lines
            if not _STANDALONE_PAGE_RE.fullmatch(line) and not _STANDALONE_LINE_RE.fullmatch(line)
        ]
        if len(retained) != len(lines):
            transformations.append("removed_page_or_line_markers")
        lines = retained

    cleaned = "\n".join(lines)
    if config.collapse_repeated_punctuation:
        replaced = _REPEATED_CHAR_RE.sub(lambda match: match.group(1) * 3, cleaned)
        if replaced != cleaned:
            transformations.append("collapsed_repeated_punctuation")
        cleaned = replaced

    return cleaned.strip(), tuple(transformations)


def apply_rule_based_filters(text: object, config: RuleFilterConfig | None = None) -> RuleFilterResult:
    """Clean common noise and decide whether a document passes quality rules."""

    config = config or RuleFilterConfig()
    value = "" if text is None else str(text)
    value, transformations = remove_common_noise(value, config)

    character_count = len(value)
    words = _WORD_RE.findall(value)
    word_count = len(words)
    non_whitespace = [char for char in value if not char.isspace()]
    alphabetic_ratio = (
        sum(char.isalpha() for char in non_whitespace) / len(non_whitespace)
        if non_whitespace
        else 0.0
    )
    replacement_ratio = value.count("\ufffd") / max(1, character_count)
    repeated_line_ratio = _repeated_line_ratio(value)
    repeated_ngram_ratio = _repeated_ngram_ratio([word.lower() for word in words], config.repeated_ngram_size)

    reasons: list[str] = []
    flags: list[str] = []
    if character_count < config.min_characters:
        reasons.append("too_few_characters")
    if word_count < config.min_words:
        reasons.append("too_few_words")
    if config.max_characters is not None and character_count > config.max_characters:
        reasons.append("too_many_characters")
    if alphabetic_ratio < config.min_alphabetic_ratio:
        reasons.append("low_alphabetic_ratio")
    if replacement_ratio > config.max_replacement_character_ratio:
        reasons.append("high_replacement_character_ratio")
    if repeated_line_ratio > config.max_repeated_line_ratio:
        reasons.append("high_repeated_line_ratio")
    if repeated_ngram_ratio > config.max_repeated_ngram_ratio:
        reasons.append("high_repeated_10gram_ratio")
    if _HTML_TAG_RE.search(value):
        (reasons if config.reject_remaining_html else flags).append("contains_html")

    metrics: dict[str, float | int] = {
        "characters": character_count,
        "words": word_count,
        "alphabetic_ratio": alphabetic_ratio,
        "replacement_character_ratio": replacement_ratio,
        "repeated_line_ratio": repeated_line_ratio,
        "repeated_ngram_ratio": repeated_ngram_ratio,
    }
    return RuleFilterResult(
        accepted=not reasons,
        text=value,
        reasons=tuple(reasons),
        flags=tuple(flags),
        transformations=transformations,
        metrics=metrics,
    )


def filter_record(
    record: Mapping[str, Any],
    *,
    text_column: str = "text",
    config: RuleFilterConfig | None = None,
) -> tuple[dict[str, Any], RuleFilterResult]:
    """Copy and filter one record, returning the record and full decision."""

    result = apply_rule_based_filters(record.get(text_column), config)
    output = dict(record)
    output[text_column] = result.text
    return output, result


def iter_filter_records(
    records: Iterable[Mapping[str, Any]],
    *,
    text_column: str = "text",
    config: RuleFilterConfig | None = None,
) -> Iterator[tuple[dict[str, Any], RuleFilterResult]]:
    """Lazily yield every copied record and its filter decision."""

    for record in records:
        yield filter_record(record, text_column=text_column, config=config)


__all__ = [
    "RuleFilterConfig", "RuleFilterResult", "apply_rule_based_filters",
    "filter_record", "iter_filter_records", "remove_common_noise",
]
