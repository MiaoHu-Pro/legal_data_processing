"""KenLM-based document perplexity filtering.

KenLM is imported lazily so normalization, rule filtering, and tests can run in
environments where the optional native ``kenlm`` extension is not installed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol


_WHITESPACE_RE = re.compile(r"\s+")


class LanguageModel(Protocol):
    """The small portion of the KenLM model API required by this module."""

    def score(self, sentence: str, bos: bool = True, eos: bool = True) -> float:
        """Return the base-10 log probability of a sentence."""


@dataclass(frozen=True)
class PerplexityFilterConfig:
    """Perplexity acceptance criteria."""

    threshold: float = 1500.0
    min_words: int = 1
    include_bos: bool = True
    include_eos: bool = True

    def __post_init__(self) -> None:
        if self.threshold <= 0 or not math.isfinite(self.threshold):
            raise ValueError("threshold must be a positive finite number")
        if self.min_words < 1:
            raise ValueError("min_words must be positive")


@dataclass(frozen=True)
class PerplexityResult:
    """Auditable result of language-model quality scoring."""

    accepted: bool
    perplexity: float
    log10_probability: float
    token_count: int
    reason: str | None = None


def load_kenlm_model(model_path: str | Path) -> LanguageModel:
    """Load a binary/ARPA KenLM model with a clear optional-dependency error."""

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"KenLM model not found: {path}")
    try:
        import kenlm  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "KenLM is required for production perplexity filtering. Install "
            "the 'kenlm' Python package and provide an ARPA or binary model."
        ) from exc
    return kenlm.Model(str(path))


class PerplexityFilter:
    """Score and filter documents with a KenLM-compatible model."""

    def __init__(
        self,
        model: LanguageModel | None = None,
        *,
        model_path: str | Path | None = None,
        config: PerplexityFilterConfig | None = None,
    ) -> None:
        if (model is None) == (model_path is None):
            raise ValueError("Provide exactly one of model or model_path")
        self.model = model if model is not None else load_kenlm_model(model_path)  # type: ignore[arg-type]
        self.config = config or PerplexityFilterConfig()

    def score_text(self, text: object) -> PerplexityResult:
        """Compute word-normalized perplexity and apply the configured threshold."""

        sentence = _WHITESPACE_RE.sub(" ", "" if text is None else str(text)).strip()
        words = sentence.split() if sentence else []
        if len(words) < self.config.min_words:
            return PerplexityResult(
                accepted=False,
                perplexity=math.inf,
                log10_probability=-math.inf,
                token_count=len(words),
                reason="too_few_words_for_perplexity",
            )

        log10_probability = float(
            self.model.score(
                sentence,
                bos=self.config.include_bos,
                eos=self.config.include_eos,
            )
        )
        token_count = len(words) + int(self.config.include_eos)
        exponent = -log10_probability / max(1, token_count)
        perplexity = math.inf if exponent > 308 else 10.0**exponent

        if not math.isfinite(perplexity):
            return PerplexityResult(
                accepted=False,
                perplexity=perplexity,
                log10_probability=log10_probability,
                token_count=token_count,
                reason="non_finite_perplexity",
            )
        accepted = perplexity <= self.config.threshold
        return PerplexityResult(
            accepted=accepted,
            perplexity=perplexity,
            log10_probability=log10_probability,
            token_count=token_count,
            reason=None if accepted else "perplexity_above_threshold",
        )

    def filter_record(
        self,
        record: Mapping[str, Any],
        *,
        text_column: str = "text",
    ) -> tuple[dict[str, Any], PerplexityResult]:
        """Copy and score one record."""

        output = dict(record)
        return output, self.score_text(record.get(text_column))

    def iter_filter_records(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        text_column: str = "text",
    ) -> Iterator[tuple[dict[str, Any], PerplexityResult]]:
        """Lazily yield records with their perplexity decisions."""

        for record in records:
            yield self.filter_record(record, text_column=text_column)


def normalized_perplexity(
    text: object,
    model: LanguageModel,
    *,
    include_bos: bool = True,
    include_eos: bool = True,
) -> float:
    """Convenience function returning only normalized perplexity."""

    config = PerplexityFilterConfig(
        threshold=float("1e308"),
        include_bos=include_bos,
        include_eos=include_eos,
    )
    return PerplexityFilter(model=model, config=config).score_text(text).perplexity


__all__ = [
    "LanguageModel", "PerplexityFilter", "PerplexityFilterConfig",
    "PerplexityResult", "load_kenlm_model", "normalized_perplexity",
]
