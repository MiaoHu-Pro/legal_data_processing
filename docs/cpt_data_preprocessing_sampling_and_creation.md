# CPT Data Preprocessing, Sampling, and Dataset Creation

## 1. Purpose

This document defines a reproducible pipeline for creating continual pretraining
(CPT) datasets from:

- `datasets/Multi_Legal_Pile_Commercial`: the English commercial legal corpus.
- `datasets/SlimPajama-1B/row_data`: a smaller general-domain replay corpus.

The first recommended experiment is a **1B-token CPT dataset**, followed by
nested 3B- and 5B-token datasets if the first run is useful. The legal corpus is
the main training signal. SlimPajama is replay data intended to reduce loss of
general-domain capability; it is not a validation or test corpus for the legal
task.

The process must produce the same selected documents for every model under
comparison. Model-specific tokenization and packing happen only after the fixed
document-level dataset has been created.

## 2. Current inputs

### 2.1 MultiLegalPile Commercial English

Location:

```text
datasets/Multi_Legal_Pile_Commercial/
```

Current shape:

- 18 JSONL files.
- 17,011,009 downloaded records according to the completed export log.
- About 283 GB in uncompressed JSONL form.
- Expected row fields: `language`, `type`, `jurisdiction`, and `text`.
- Some rows may not contain `source`; the analysis script records this as
  `UNKNOWN`.

The running analysis script is:

```text
datasets_processing/multi_legal_pile_en_commercial_analysis.py
```

Its final outputs should be read from:

```text
processing_analysis_output/
├── multi_legal_pile_commercial_global_summary.json
├── multi_legal_pile_commercial_stats_by_type_jurisdiction.csv
└── multi_legal_pile_commercial_stats_by_type_jurisdiction_source.csv
```

Do not finalize sampling quotas from the current checkpoint files. At the time
this plan was written, the checkpoints contained only the first 1,000,000
documents and only two strata (`caselaw/EU` and `caselaw/US`). They do not yet
describe the entire legal corpus.

### 2.2 SlimPajama-1B

Location:

```text
datasets/SlimPajama-1B/row_data/
```

Current shape:

- 933,130 training rows.
- 9,347 validation rows.
- 9,346 test rows.
- Row fields: `text`, `meta.redpajama_set_name`, and
  `__index_level_0__`.
- Approximately 4.1 GB of row-oriented JSONL.

Use only the SlimPajama `train-*` files as CPT replay data. Keep its
`validation-*` and `test-*` files out of training. The generated Pandas index
field `__index_level_0__` is not training content and should be removed from the
normalized output.

## 3. Recommended initial mixtures

Create the following document-level releases:

| Release | Approximate training tokens | Purpose |
|---|---:|---|
| `cpt-v0` | 100M | End-to-end pipeline and short training smoke test |
| `cpt-v1` | 1B | First controlled CPT experiment |
| `cpt-v2` | 3B | Main follow-up experiment |
| `cpt-v3` | 5B | Scaling experiment only if smaller runs help |

Use this top-level mixture as the initial baseline:

| Corpus | Share of tokens | 1B example |
|---|---:|---:|
| MultiLegalPile Commercial English | 90% | 900M |
| SlimPajama replay | 10% | 100M |

Ten percent replay is a starting point, not a universal optimum. If general
benchmarks regress materially after `cpt-v1`, compare a 15% or 20% replay
ablation while holding the optimizer, number of tokens, and all other settings
constant.

Within the 90% legal portion, use these provisional type targets after the
analysis is complete:

| Legal type | Share of legal tokens | Share of a 1B total run |
|---|---:|---:|
| Legislation | 35% | 31.5% |
| Case law | 30% | 27.0% |
| Contracts | 25% | 22.5% |
| Other and `legal_mc4` | 10% | 9.0% |

These weights reflect the project preference for legislation, contracts, and
case law while limiting generic/noisy legal web text. They must be reconciled
with the names and available token counts in the final analysis CSV. Never
silently map an unknown type to a preferred category.

For a UK/EU-focused model, split the legislation target provisionally as:

| Legislation jurisdiction | Share of legislation tokens |
|---|---:|
| EU | 45% |
| UK | 35% |
| Other jurisdictions combined | 20% |

For case law and contracts, set jurisdiction quotas only after reading the
complete analysis. A useful default is to cap any single jurisdiction at 50%
of that type's sampled tokens, unless dominating that jurisdiction is an
explicit product requirement. This prevents a very large US stratum from
overwhelming all other legal systems.

If a stratum does not have enough clean tokens to meet its target, take all
eligible data from that stratum and redistribute the shortfall proportionally
among the other preferred strata. Record every shortfall and redistribution in
the release manifest.

## 4. Pipeline overview

```text
immutable raw JSONL
        |
        v
schema validation and normalization
        |
        v
quality and safety filtering
        |
        v
exact and near-duplicate clustering
        |
        v
cluster-level train/validation/test assignment
        |
        v
token-aware stratified sampling
        |
        +------ legal (90%)
        |
        +------ SlimPajama replay (10%)
        |
        v
fixed document-level CPT release + manifest
        |
        v
model-specific tokenization, EOS insertion, and packing
```

Raw files should remain immutable. Write every stage to a new versioned
directory and make it possible to regenerate that directory from configuration
plus code.

## 5. Stage A: inventory and schema validation

Wait for the full legal analysis to finish, then verify:

1. The final document count matches the export count, apart from explicitly
   reported malformed JSON rows.
2. Counts and estimated tokens are available for every
   `type × jurisdiction` stratum.
3. The sum of stratum counts equals the global count.
4. Missing `text`, `type`, `jurisdiction`, and `source` counts are recorded.
5. No input file changed while analysis was running.

Create an immutable input manifest containing, for every input shard:

- relative path;
- byte size;
- SHA-256 checksum;
- number of valid rows;
- number of invalid rows;
- processing timestamp;
- schema version.

Validate rows without loading a whole shard into memory. Reject malformed JSON
and non-string or empty `text`. Preserve rejected row IDs and reason codes in a
separate audit file; do not write full rejected text into logs.

## 6. Stage B: canonical document representation

Convert both sources to one internal JSONL schema:

```json
{
  "id": "sha256:...",
  "text": "...",
  "corpus": "multilegalpile_commercial",
  "type": "caselaw",
  "jurisdiction": "EU",
  "source": "UNKNOWN",
  "source_split": "train",
  "source_file": "multi_legal_pile_en_commercial_00001.jsonl",
  "source_row": 123,
  "quality_flags": [],
  "dedup_cluster_id": null
}
```

For SlimPajama, use:

- `corpus = "slimpajama_1b"`;
- `type = "general"`;
- `jurisdiction = "NONE"`;
- `source = meta.redpajama_set_name`;
- `source_split` inferred from the input filename;
- omit `__index_level_0__` from the normalized record.

Normalize categorical values with an explicit mapping table. For example,
`UK`, `United Kingdom`, and a confirmed equivalent can map to one canonical
value, but only after checking the values present in the completed analysis.
Keep the original categorical values in provenance metadata or in an audit
table.

Generate a stable ID from provenance, for example:

```text
sha256(corpus + "\0" + source_file + "\0" + source_row)
```

Also calculate a separate normalized-text hash for exact deduplication. Do not
use Python's built-in `hash()`, because it is not stable between processes.

## 7. Stage C: text normalization

Use conservative normalization because legal punctuation, section symbols,
numbering, citations, and whitespace can carry meaning.

Recommended operations:

- decode as UTF-8 and record replacement-character problems;
- normalize line endings to `\n`;
- apply Unicode NFC normalization;
- remove NUL bytes and non-text control characters, while preserving `\n` and
  `\t`;
- strip leading and trailing whitespace;
- collapse excessive horizontal spaces, but preserve paragraph boundaries;
- limit extreme runs of blank lines to at most two blank lines;
- remove repeated boilerplate only with a measured, source-specific rule.

Do **not** lowercase, remove punctuation, stem words, discard citations, or
apply general HTML stripping blindly. Do not use NFKC without first measuring
its effect on legal symbols and citations. If HTML removal is required for a
specific source, compare samples before and after the transformation.

Store the normalization configuration and code version in the manifest. Review
a random sample of at least 100 changed documents per major source before
processing the full corpus.

## 8. Stage D: quality filtering

Apply cheap deterministic filters first, followed by more expensive filters.
Initial thresholds should be measured on samples and adjusted before the final
run.

### 8.1 Hard rejection candidates

Reject or quarantine documents with:

- missing or non-string text;
- empty text after normalization;
- fewer than approximately 50 characters;
- an extreme replacement-character or control-character rate;
- almost no alphabetic content;
- clear binary/base64 or machine-dump content;
- exact duplicates of an already retained document;
- a language score strongly inconsistent with English.

The dataset is labeled English, but language identification should still be
audited. Avoid rejecting documents merely because they contain Latin phrases,
citations, party names, or short multilingual passages.

### 8.2 Soft flags and source-aware review

Flag rather than immediately reject:

- very long documents;
- high symbol-to-character ratios;
- heavy OCR artifacts;
- repeated headers, footers, navigation, or cookie notices;
- unusually high line duplication;
- templated contracts that differ in only a few fields;
- documents dominated by tables or lists.

Compute rejection and flag rates by corpus, legal type, jurisdiction, and
source. A global rule can accidentally remove an entire useful source, so pause
for review if a rule removes more than 10% of any important stratum.

### 8.3 Privacy, safety, and licensing

The corpus name indicates commercial usability, but preprocessing should not be
treated as a new legal review. Retain dataset license and provenance records.
Scan representative samples for personal data, secrets, and unexpectedly
sensitive content, particularly in case law and contracts. Define the intended
release scope before deciding whether redaction is required. Do not publish raw
sample text in diagnostic reports.

## 9. Stage E: deduplication and contamination control

Deduplicate across **both** corpora before sampling.

1. Normalize text for comparison separately from training-text normalization.
2. Remove exact duplicates using a cryptographic hash.
3. Find near duplicates with MinHash/LSH, SimHash, or an equivalent scalable
   method over word or character n-grams.
4. Assign every near-duplicate family a stable `dedup_cluster_id`.
5. Keep one representative according to a deterministic policy, preferring the
   higher-quality record and then the preferred legal source.

Measure near-dedup thresholds on labeled pairs. Legal templates and amended
versions can be similar without being duplicates; an aggressive threshold may
delete valuable distinctions. Report deduplication rates for each stratum and
the cross-corpus overlap rate.

Before sampling, compare documents with all intended evaluation datasets.
Remove exact matches and, where feasible, high-confidence near matches. Record
the evaluation set name, version, hashing method, and number of exclusions.

## 10. Stage F: data splits

Never split individual documents from the same duplicate cluster across train,
validation, and test. Assign the complete cluster using a stable hash:

```text
bucket = uint64(sha256(dedup_cluster_id + split_seed)[:8]) mod 10_000
```

A reasonable legal split is:

- train: 99.0%;
- validation: 0.5%;
- test: 0.5%.

Because the corpus is large, these small percentages still provide substantial
held-out sets. Stratify or verify the resulting distribution by type and
jurisdiction. If documents have dates, a time-based legal test set is also
valuable, but it should be a separate evaluation design rather than silently
mixed with the hash split.

For SlimPajama, honor its existing split: use only `train-*` for CPT replay.
Keep its original validation and test rows held out. Deduplication must still
prevent an equivalent training document from leaking into those held-out rows.

## 11. Stage G: token-aware stratified sampling

Sample by estimated or exact tokens, not by document count. Legal document
lengths vary greatly, so document-count quotas can produce a very different
token mixture.

### 11.1 Deterministic selection

Within every stratum, assign a deterministic random priority:

```text
priority = sha256(sample_seed + "\0" + document_id)
```

Sort by that priority and take documents without replacement until the stratum
token target is met. This selection does not depend on input shard order and is
reproducible. Record the seed.

For very large strata, implement equivalent hash-threshold or streaming
reservoir logic instead of sorting all document text in memory. Sort compact
records containing IDs, token estimates, and offsets; retrieve full text only
when writing the selected dataset.

### 11.2 Token estimation and boundary handling

The current analysis uses `1.30 × words`, which is adequate for corpus planning
but not for final budgeting. For final selection:

1. estimate tokens cheaply for all eligible documents;
2. select slightly more than the target, for example 2%;
3. tokenize selected documents with the training model's tokenizer;
4. trim only at document boundaries until the target is reached;
5. report the exact token total.

Do not truncate a document merely to hit a sampling quota. Long-document
chunking, if required, should be a separate deterministic preprocessing stage
with overlap and parent-document metadata.

### 11.3 SlimPajama replay balance

SlimPajama itself contains multiple sources in `meta.redpajama_set_name`. Report
their token distribution. Use source-balanced or temperature sampling if one
source dominates; do not let Common Crawl accidentally become the complete
replay allocation.

A standard temperature rule is:

```text
q_i = n_i^alpha / sum_j(n_j^alpha), with alpha between 0.3 and 0.7
```

where `n_i` is the eligible token count of SlimPajama source `i`. Choose and
record `alpha`; `alpha = 0.5` is a reasonable first baseline.

### 11.4 Nested releases

Make `cpt-v0`, `cpt-v1`, `cpt-v2`, and `cpt-v3` nested where possible. Generate
one deterministic priority ordering per stratum and take longer prefixes for
larger releases while enforcing quotas at each release boundary. Then:

```text
cpt-v0 documents ⊆ cpt-v1 documents ⊆ cpt-v2 documents ⊆ cpt-v3 documents
```

Nested releases make scaling comparisons easier to interpret and avoid
confounding corpus composition with corpus size.

## 12. Stage H: final document-level release

Write shuffled JSONL shards with a practical maximum size, such as 100,000
documents or 1 GB per shard. Each training row should retain text plus enough
metadata for auditing:

```json
{
  "id": "sha256:...",
  "text": "normalized text",
  "corpus": "multilegalpile_commercial",
  "type": "legislation",
  "jurisdiction": "UK",
  "source": "uk_uk_lex",
  "dedup_cluster_id": "sha256:..."
}
```

Metadata is for traceability and sampling analysis; it should not be serialized
into the text presented to the language model unless an experiment explicitly
tests metadata conditioning.

Use a deterministic buffered or external-memory shuffle. Do not place the
entire legal corpus first and replay corpus last. The order should mix sources
throughout training while remaining reproducible.

Each release directory should contain:

```text
cpt-v1/
├── train/
│   ├── part-00000.jsonl
│   └── ...
├── validation/
├── test/
├── manifest.json
├── stats_by_corpus.csv
├── stats_by_type_jurisdiction.csv
├── stats_by_source.csv
├── rejected_counts.csv
└── README.md
```

The manifest should include:

- release name and creation timestamp;
- Git commit or source-code checksum;
- input manifests and checksums;
- normalization, filtering, deduplication, and split configurations;
- sampling seed and shuffle seed;
- requested and achieved quotas;
- document, byte, character, word, and exact token totals;
- tokenizer name, revision, and vocabulary checksum for token totals;
- output shard checksums;
- all shortfalls, redistributions, and known limitations.

## 13. Stage I: tokenization and sequence packing

Create tokenized artifacts separately for each model. Mistral and Qwen may
produce different token counts from the same documents, but the underlying
document IDs must remain identical for a controlled comparison.

For each document:

1. tokenize only `text`;
2. insert the model's EOS/document-boundary token;
3. pack sequences to the configured context length;
4. create labels and attention/loss masks expected by the training code;
5. record whether packing allows attention across document boundaries.

Avoid padding each document independently; packing short documents is much more
token-efficient. Never concatenate documents without an EOS or equivalent
boundary. Do not insert BOS/EOS twice if the tokenizer or training collator
already adds them.

Run a packing audit that decodes sampled sequences and verifies boundaries,
special-token counts, label alignment, padding masks, and maximum length.

## 14. Validation gates before CPT

Do not start the full CPT run until all of these checks pass:

- every raw input shard appears in the input manifest;
- output is reproducible from the same seed and configuration;
- all output JSONL lines parse successfully;
- stable IDs are unique after deduplication;
- no dedup cluster crosses train/validation/test;
- SlimPajama validation and test rows are absent from training;
- no known evaluation exact match remains in training;
- actual mixture proportions are within an agreed tolerance, such as 0.5
  percentage points, of their targets;
- token totals are measured with the exact model tokenizer;
- every output shard checksum and row count is recorded;
- at least 100 random documents per major legal type and 100 replay documents
  pass manual inspection;
- decoded packed-sequence samples look correct;
- a short overfit test and a 100M-token smoke run complete without data errors.

Also calculate baseline validation loss/perplexity separately on legal and
general held-out sets before CPT. Recalculate them after CPT, alongside the
target legal benchmarks, to measure both legal improvement and general-domain
regression.

## 15. Suggested implementation order

1. Let `multi_legal_pile_en_commercial_analysis.py` finish and validate its
   final totals.
2. Freeze raw shard checksums and write the input manifest.
3. Implement canonicalization and conservative text normalization.
4. Produce filter statistics on a deterministic sample before choosing final
   thresholds.
5. Run full filtering and exact deduplication.
6. Calibrate near-deduplication on labeled legal-document pairs, then run it on
   both corpora.
7. Exclude evaluation contamination and assign cluster-level splits.
8. Resolve type/jurisdiction quotas using the completed analysis and eligible
   post-filter token counts.
9. Build and inspect `cpt-v0`.
10. Build the nested `cpt-v1` document release and tokenize it independently
    for each model.
11. Run the smoke test, validate metrics and checkpoints, then begin the 1B CPT
    experiment.
12. Create 3B and 5B releases only after reviewing the smaller run.

## 16. Decisions to freeze in configuration

Before creating the first final release, record these choices in a versioned
YAML or JSON configuration rather than hard-coding them:

- target model-independent document release;
- total target tokens and legal/replay ratio;
- legal type and jurisdiction quotas;
- SlimPajama source-sampling temperature;
- normalization and filtering thresholds;
- exact and near-dedup settings;
- evaluation datasets used for contamination checks;
- train/validation/test split policy;
- sampling, splitting, and shuffling seeds;
- maximum document/chunk length and chunk overlap;
- output shard limit;
- tokenizer names and immutable revisions;
- context length and packing policy.

The most important principle is that a CPT release is not merely a directory of
text. It is the combination of immutable inputs, preprocessing rules, selected
document IDs, mixture targets, seeds, manifests, and validation evidence needed
to reproduce exactly what the model saw.
