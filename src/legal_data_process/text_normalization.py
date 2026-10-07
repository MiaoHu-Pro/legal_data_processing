"""Conservative text normalization for legal continual-pretraining data.

The training text keeps case, punctuation, citations, and paragraph boundaries.
Only representation differences and common markup/whitespace noise are removed.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable, Iterator, Mapping


_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\n]+")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")


@dataclass(frozen=True)
class NormalizationConfig:
    """Options controlling deterministic training-text normalization."""

    unicode_form: str = "NFKC"
    decode_html_entities: bool = True
    strip_html: bool = True
    remove_control_characters: bool = True
    collapse_horizontal_whitespace: bool = True
    max_blank_lines: int = 2
    strip_text: bool = True

    def __post_init__(self) -> None:
        if self.unicode_form not in {"NFC", "NFKC", "NFD", "NFKD"}:
            raise ValueError(f"Unsupported Unicode normalization form: {self.unicode_form}")
        if self.max_blank_lines < 0:
            raise ValueError("max_blank_lines must be non-negative")


class _HTMLTextExtractor(HTMLParser):
    """Small, dependency-free HTML-to-text converter."""

    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "div", "dl",
        "fieldset", "footer", "form", "h1", "h2", "h3", "h4", "h5",
        "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
        "section", "table", "tr", "ul",
    }
    _IGNORED_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif tag in self._BLOCK_TAGS and self._ignored_depth == 0:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() in self._IGNORED_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif tag in self._BLOCK_TAGS and self._ignored_depth == 0:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


def strip_html_tags(text: object) -> str:
    """Remove real-looking HTML tags while retaining visible text."""

    value = "" if text is None else str(text)
    if not _HTML_TAG_RE.search(value):
        return value
    parser = _HTMLTextExtractor()
    try:
        parser.feed(value)
        parser.close()
    except (ValueError, AssertionError):
        return _HTML_TAG_RE.sub(" ", value)
    return parser.get_text()


def normalize_unicode(text: object, form: str = "NFKC") -> str:
    """Normalize the Unicode representation; ``None`` becomes an empty string."""

    value = "" if text is None else str(text)
    return unicodedata.normalize(form, value)


def remove_control_characters(text: object) -> str:
    """Remove Unicode control/format characters, preserving tabs and newlines."""

    value = "" if text is None else str(text)
    kept: list[str] = []
    for char in value:
        if char in {"\n", "\t"}:
            kept.append(char)
            continue
        if unicodedata.category(char).startswith("C"):
            continue
        kept.append(char)
    return "".join(kept)


def normalize_text(text: object, config: NormalizationConfig | None = None) -> str:
    """Return deterministic, legal-text-safe normalized text."""

    config = config or NormalizationConfig()
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if config.decode_html_entities:
        value = html.unescape(value)
    if config.strip_html:
        value = strip_html_tags(value)
    value = normalize_unicode(value, config.unicode_form)
    if config.remove_control_characters:
        value = remove_control_characters(value)
    if config.collapse_horizontal_whitespace:
        value = "\n".join(_HORIZONTAL_SPACE_RE.sub(" ", line).rstrip() for line in value.split("\n"))
    if config.max_blank_lines >= 0:
        excess_blank_lines_re = re.compile(
            rf"\n(?:[ \t]*\n){{{config.max_blank_lines + 1},}}"
        )
        replacement = "\n" * (config.max_blank_lines + 1)
        value = excess_blank_lines_re.sub(replacement, value)
    if config.strip_text:
        value = value.strip()
    return value


def normalize_record(
    record: Mapping[str, Any],
    *,
    text_column: str = "text",
    config: NormalizationConfig | None = None,
) -> dict[str, Any]:
    """Copy a record and normalize its text column."""

    output = dict(record)
    output[text_column] = normalize_text(record.get(text_column), config)
    return output


def normalize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    text_column: str = "text",
    config: NormalizationConfig | None = None,
) -> Iterator[dict[str, Any]]:
    """Lazily normalize an iterable of records."""

    for record in records:
        yield normalize_record(record, text_column=text_column, config=config)


def clean_text(
    text: object,
    lowercase: bool = False,
    keep_symbols: str | None = None,
) -> str:
    """Backward-compatible normalizer that does not delete legal punctuation."""

    del keep_symbols
    value = normalize_text(text)
    return value.lower() if lowercase else value


__all__ = [
    "NormalizationConfig", "clean_text", "normalize_record",
    "normalize_records", "normalize_text", "normalize_unicode",
    "remove_control_characters", "strip_html_tags",
]
