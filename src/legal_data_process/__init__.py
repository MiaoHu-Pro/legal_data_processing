"""Preprocessing utilities for legal continual-pretraining datasets."""

from legal_data_process.perplexity_filtering import PerplexityFilter
from legal_data_process.perplexity_filtering import PerplexityFilterConfig
from legal_data_process.rule_based_filters import RuleFilterConfig
from legal_data_process.rule_based_filters import apply_rule_based_filters
from legal_data_process.text_deduplication import DeduplicationConfig
from legal_data_process.text_deduplication import deduplicate_records
from legal_data_process.text_normalization import NormalizationConfig
from legal_data_process.text_normalization import normalize_text

__version__ = "0.1.0"


def main() -> None:
    """Print a short package status message for the console entry point."""

    print("legal-data-process preprocessing package is available")


__all__ = [
    "DeduplicationConfig",
    "NormalizationConfig",
    "PerplexityFilter",
    "PerplexityFilterConfig",
    "RuleFilterConfig",
    "apply_rule_based_filters",
    "deduplicate_records",
    "normalize_text",
]
