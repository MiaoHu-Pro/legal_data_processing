# Legal Data Processing Workflow

## 1. Purpose

The `legal_data_process` package prepares raw JSONL documents for continual
pretraining (CPT). It provides four independent stages:

```text
raw JSONL
   |
   v
text normalization
   |
   v
rule-based filtering and noise cleanup
   |
   v
perplexity filtering with KenLM
   |
   v
exact and near deduplication
   |
   v
clean document-level data for later sampling and CPT creation
```

The package currently contains processing primitives rather than one program
that processes the complete 283 GB legal corpus. This separation allows the
large-scale driver to stream input, checkpoint intermediate results, collect
rejection statistics, and choose an appropriate deduplication backend.

Package location:

```text
src/legal_data_process/
├── text_normalization.py
├── rule_based_filters.py
├── perplexity_filtering.py
└── text_deduplication.py
```

## 2. Recommended production order

For the first production implementation, use:

1. Parse and validate each JSONL row.
2. Normalize the `text` field.
3. Apply rule-based cleanup and filtering.
4. Apply KenLM perplexity filtering.
5. Run exact and near deduplication across all surviving corpora.
6. Write accepted records, rejection audits, statistics, and manifests.

Running cheap deterministic rules before KenLM avoids scoring obviously bad
documents. Running deduplication before KenLM can save additional computation,
but the combined deduplication API selects the earliest cluster member. Until a
quality-aware representative policy is added, perplexity filtering before near
deduplication is safer because a poor representative cannot displace a better
near-duplicate document.

Exact text hashes may still be calculated before KenLM to avoid rescoring
identical text, provided the chosen representative retains all required
provenance.

## 3. Text normalization

Implementation:

```text
src/legal_data_process/text_normalization.py
```

The default `NormalizationConfig` performs:

- line-ending normalization to `\n`;
- HTML entity decoding;
- visible-text extraction from HTML-like input;
- Unicode NFKC normalization;
- removal of hidden control and format characters while retaining tabs and
  newlines;
- horizontal whitespace normalization;
- limiting excessive blank lines;
- leading and trailing whitespace removal.

It deliberately preserves case, ordinary punctuation, citations, section
symbols, and paragraph boundaries. It does not stem words or remove legal
symbols.

Primary APIs:

```python
from legal_data_process.text_normalization import NormalizationConfig
from legal_data_process.text_normalization import normalize_record
from legal_data_process.text_normalization import normalize_records
from legal_data_process.text_normalization import normalize_text
```

Example:

```python
config = NormalizationConfig(
    unicode_form="NFKC",
    strip_html=True,
    max_blank_lines=2,
)

cleaned_text = normalize_text(raw_text, config)
cleaned_record = normalize_record(record, config=config)
```

`normalize_record` returns a copy and does not mutate the input dictionary.
`normalize_records` processes an iterable lazily.

## 4. Rule-based filtering

Implementation:

```text
src/legal_data_process/rule_based_filters.py
```

This stage removes common layout noise and calculates deterministic quality
signals. The default rules cover:

- minimum characters and words;
- optional maximum document length;
- minimum alphabetic-character ratio;
- maximum Unicode replacement-character ratio;
- repeated-line ratio;
- repeated word 10-gram ratio;
- remaining HTML detection;
- standalone page and line marker removal;
- pathological repeated-character cleanup.

The page rule targets explicit forms such as `Page 12 of 40`; it does not
remove a standalone number that may be a legal section heading.

Primary APIs:

```python
from legal_data_process.rule_based_filters import RuleFilterConfig
from legal_data_process.rule_based_filters import apply_rule_based_filters
from legal_data_process.rule_based_filters import filter_record
from legal_data_process.rule_based_filters import iter_filter_records
```

Example:

```python
result = apply_rule_based_filters(normalized_text, RuleFilterConfig())

if result.accepted:
    accepted_text = result.text
else:
    print(result.reasons)
```

`RuleFilterResult` contains:

- `accepted`: the final decision;
- `text`: text after rule-based noise cleanup;
- `reasons`: hard rejection reason codes;
- `flags`: non-rejecting warnings;
- `transformations`: applied cleanup operations;
- `metrics`: document measurements used by the rules.

Production reports should aggregate reasons and metrics by corpus, legal type,
jurisdiction, and source. Thresholds should be calibrated on representative
samples before processing the full corpus.

## 5. Perplexity filtering

Implementation:

```text
src/legal_data_process/perplexity_filtering.py
```

This module is designed to use the official KenLM Python binding from
`https://github.com/kpu/kenlm`:

```python
import kenlm

model = kenlm.Model("legal_language_model.bin")
log10_probability = model.score(text, bos=True, eos=True)
```

KenLM is imported lazily. Normalization, rule filtering, and tests can therefore
run without installing the optional native extension.

The implementation calculates word-normalized perplexity as:

```text
perplexity = 10 ** (-log10_probability / token_count)
```

The EOS position is included in `token_count` when EOS scoring is enabled. The
default rejection threshold is `1500`, matching the intended legal-data
filtering configuration.

Primary APIs:

```python
from legal_data_process.perplexity_filtering import PerplexityFilter
from legal_data_process.perplexity_filtering import PerplexityFilterConfig
```

Production example:

```python
filtering = PerplexityFilter(
    model_path="models/legal_kenlm.bin",
    config=PerplexityFilterConfig(threshold=1500.0),
)

result = filtering.score_text(cleaned_text)
if not result.accepted:
    print(result.reason, result.perplexity)
```

`PerplexityResult` records the decision, perplexity, base-10 log probability,
scored token count, and rejection reason.

Important current limitation: neither the `kenlm` Python package nor a trained
legal KenLM model is currently available in the project environment. The API is
implemented, but production scoring requires both. The test script uses an
injected KenLM-compatible deterministic model; it does not claim to validate a
real language model or its threshold calibration.

## 6. Text deduplication

Implementation:

```text
src/legal_data_process/text_deduplication.py
```

The module uses code from the local text-dedup checkout:

```text
src/text-dedup/src/text_dedup/
```

Specifically, the dependency-light API uses text-dedup's:

- n-gram generator;
- Jaccard similarity implementation;
- Union-Find cluster implementation.

It then performs:

1. NFKC/casefold/whitespace canonicalization for exact matching.
2. SHA-256 exact-text hashing.
3. Word n-gram construction.
4. Deterministic MinHash signatures.
5. LSH candidate generation.
6. Exact Jaccard verification of candidate pairs.
7. Transitive duplicate clustering.
8. Stable cluster annotation and duplicate audit creation.

The default near-duplicate Jaccard threshold is `0.50`.

### 6.1 Small in-memory API

```python
from legal_data_process.text_deduplication import DeduplicationConfig
from legal_data_process.text_deduplication import deduplicate_records

result = deduplicate_records(
    records,
    DeduplicationConfig(
        similarity_threshold=0.50,
        ngram_size=5,
        num_permutations=128,
        bands=32,
        seed=42,
    ),
)
```

`DeduplicationResult` contains:

- retained records;
- removed records;
- exact and near duplicate counts;
- representative/duplicate matches;
- Jaccard similarities;
- stable cluster IDs.

This API materializes the supplied records and MinHash state. It is suitable
for tests, small datasets, and bounded partitions. It must not receive the
entire 283 GB legal corpus in one process.

### 6.2 Full text-dedup adapter

For the complete corpus, the module also exposes:

```python
from legal_data_process.text_deduplication import TextDedupRunConfig
from legal_data_process.text_deduplication import run_text_dedup_minhash

run_text_dedup_minhash(
    input_files=["cleaned/part-00000.jsonl", "cleaned/part-00001.jsonl"],
    output_dir="deduplicated/legal",
    config=TextDedupRunConfig(similarity_threshold=0.50),
)
```

This adapter calls `text_dedup.minhash.main` and delegates dataset loading,
fingerprinting, clustering, false-positive verification, filtering, and output
to the vendored package. It refuses to overwrite a non-empty output directory.

The complete adapter requires the dependencies declared in:

```text
src/text-dedup/pyproject.toml
```

These include `polars`, `polars-grouper`, `regex`, `scipy`, and
`pydantic-settings`. They are not all installed in the current environment.

## 7. Record processing example

The following shows the intended record-level flow. A production driver should
write accepted and rejected rows incrementally rather than accumulating them in
lists.

```python
from legal_data_process.perplexity_filtering import PerplexityFilter
from legal_data_process.rule_based_filters import apply_rule_based_filters
from legal_data_process.text_normalization import normalize_record

perplexity_filter = PerplexityFilter(model_path="models/legal_kenlm.bin")

for raw_record in input_records:
    record = normalize_record(raw_record)
    rule_result = apply_rule_based_filters(record["text"])

    if not rule_result.accepted:
        write_rejection(record, stage="rules", reasons=rule_result.reasons)
        continue

    record["text"] = rule_result.text
    perplexity_result = perplexity_filter.score_text(record["text"])

    if not perplexity_result.accepted:
        write_rejection(
            record,
            stage="perplexity",
            reasons=[perplexity_result.reason],
        )
        continue

    write_pre_dedup_record(record)
```

The accepted pre-deduplication shards can then be passed to the full text-dedup
adapter.

## 8. Test workflow

Test script:

```text
datasets_processing/data_processing_test.py
```

Input:

```text
datasets/SlimPajama-1B/row_data/test-00000-of-00001.jsonl
```

Run it from the repository root:

```bash
python datasets_processing/data_processing_test.py
```

Optional bounded-run arguments:

```bash
python datasets_processing/data_processing_test.py \
  --max-records 20 \
  --max-characters-per-record 6000
```

The test:

- checks NFKC, HTML, control-character, line-ending, and blank-line handling;
- checks rule acceptance, page-marker removal, and repeated 10-gram rejection;
- checks perplexity acceptance and rejection through the same public API;
- reads actual SlimPajama rows;
- normalizes and rule-filters those rows;
- injects one exact and one near duplicate;
- verifies both duplicates are removed;
- verifies retained records receive cluster IDs;
- verifies the local text-dedup source is used.

The default test currently passes with 12 sampled SlimPajama records, removing
one injected exact duplicate and one injected near duplicate. It does not modify
the source JSONL.

## 9. Production outputs and auditing

The future full-corpus driver should create separate outputs for:

```text
processing_output/
├── normalized_and_filtered/
├── deduplicated/
├── rejected/
│   ├── invalid_json.jsonl
│   ├── rule_rejections.jsonl
│   └── perplexity_rejections.jsonl
├── statistics/
└── manifest.json
```

Avoid writing rejected document text to console logs. Rejection audit rows
should contain stable IDs, source provenance, stage, reason codes, and metrics.
Sensitive text should remain in access-controlled data files.

The manifest should record:

- input paths and checksums;
- code revision;
- all configuration values;
- KenLM model checksum;
- text-dedup revision;
- processing and random seeds;
- counts before and after each stage;
- rejection counts by reason and source;
- exact and near duplicate counts;
- output paths and checksums.

## 10. Boundary with CPT sampling

This workflow ends with clean, deduplicated document-level data. It does not
choose the legal/SlimPajama mixture, sample token quotas, split data, tokenize
for a target model, or pack training sequences. Those are the next CPT sampling
and dataset-creation stages and should consume the immutable preprocessing
output plus its manifest.
