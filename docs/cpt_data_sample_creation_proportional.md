# Proportional CPT data creation: implementation guide

## 1. Scope

This note explains the current implementation of:

```text
datasets_processing/cpt_data_sample_creation_proportional.py
```

The script creates a JSONL continued-pretraining (CPT) dataset from:

- `Multi_Legal_Pile_Commercial`, divided into legal `type × jurisdiction` strata; and
- `SlimPajama-1B`, used as general-domain replay data.

Its defining property is that token targets are enforced **after** normalization, rule filtering, exact deduplication, and optional MinHash near deduplication. This is different from sampling a raw 90:10 mixture and then accepting whatever ratio remains after cleaning.

The default requested mixture is 90% legal and 10% replay, but `--legal-share` can change it. For example, a 10B dataset with `--legal-share 0.95` requests 9.5B legal tokens and 0.5B replay tokens.

The script does not tokenize text with the target model. Every token count in this stage is an estimate:

```text
estimated_tokens = floor(1.30 × whitespace-delimited words)
```

Exact model-tokenizer counts are produced later by the CPT tokenization stage.

## 2. High-level workflow

```text
Legal JSONL files                     SlimPajama train JSONL files
       │                                         │
       ├── allocate legal quotas                 ├── measure complete replay corpus
       │   by type × jurisdiction                └── validate replay capacity
       │                                         │
       └──────── deterministic candidate sampling ┘
                              │
                              ▼
                    NFKC text normalization
                              │
                              ▼
                     rule-based filtering
                              │
                              ▼
                   optional KenLM filtering
                              │
                              ▼
             cross-corpus SQLite exact deduplication
                              │
                              ▼
                 accepted candidate JSONL shards
                              │
                              ▼
             optional text-dedup MinHash near dedup
                              │
                              ▼
       deterministic hash buckets and post-dedup availability
                              │
               ┌──────────────┴──────────────┐
               │                             │
       legal deficit redistribution     fixed replay quota
       among legal strata only          or fail if unavailable
               └──────────────┬──────────────┘
                              │
                              ▼
                deterministic quota selection
                              │
                              ▼
          ~200M estimated-token JSONL output shards
                              │
                              ▼
                  manifest.json + statistics.json
```

## 3. Input discovery

The script discovers files using fixed filename patterns:

| Corpus | Default directory | Pattern |
|---|---|---|
| Legal | `datasets/Multi_Legal_Pile_Commercial` | `multi_legal_pile_en_commercial_*.jsonl` |
| Replay | `datasets/SlimPajama-1B/row_data` | `train-*.jsonl` |

The paths are sorted before scanning, which makes the run deterministic for a fixed collection of files.

Legal rows are expected to contain at least:

```json
{
  "text": "...",
  "type": "legislation",
  "jurisdiction": "EU",
  "source": "..."
}
```

`text_type` is accepted as a fallback for `type`. Missing metadata is represented as `UNKNOWN`.

Replay rows are expected to contain `text` and may contain SlimPajama metadata:

```json
{
  "text": "...",
  "meta": {
    "redpajama_set_name": "..."
  }
}
```

All replay output rows are normalized to:

```text
corpus       = slimpajama_1b
type         = general
jurisdiction = N/A
```

Replay input shards are read round-robin rather than one complete shard at a time. Candidate selection is hash-based, so this interleaving does not alter which rows are selected, but it avoids treating the first shard as more important than later shards.

## 4. Legal stratum weights

`LEGAL_STRATUM_TOKENS` contains the measured pre-processing token estimates from `multi_legal_pile_commercial_stats_by_type_jurisdiction.csv`.

| Type | Jurisdiction | Measured tokens | Legal weight | 4.5B legal quota | 9.5B legal quota |
|---|---|---:|---:|---:|---:|
| caselaw | EU | 289,759,375 | 0.5254% | 23,641,117 | 49,909,025 |
| caselaw | US | 32,750,957,349 | 59.3802% | 2,672,111,045 | 5,641,123,317 |
| contracts | EU | 50,738,795 | 0.0920% | 4,139,717 | 8,739,402 |
| contracts | US | 6,889,351,099 | 12.4910% | 562,093,833 | 1,186,642,537 |
| legal-mc4 | N/A | 628,590,444 | 1.1397% | 51,285,935 | 108,270,307 |
| legislation | EU | 994,351,311 | 1.8028% | 81,127,922 | 171,270,058 |
| legislation | Switzerland | 1,760,762 | 0.0032% | 143,658 | 303,279 |
| legislation | UK | 54,241,476 | 0.0983% | 4,425,497 | 9,342,715 |
| legislation | US | 2,244,323,874 | 4.0691% | 183,111,674 | 386,569,089 |
| other | N/A | 9,638,782,574 | 17.4759% | 786,416,626 | 1,660,212,877 |
| other | US | 1,611,777,462 | 2.9223% | 131,502,976 | 277,617,394 |
| **Total** |  | **55,154,634,521** | **100%** | **4,500,000,000** | **9,500,000,000** |

The 4.5B column corresponds to a 5B 90:10 mixture. The 9.5B column corresponds to a 10B 95:5 mixture.

The quotas use the largest-remainder method. Each exact fractional quota is floored, then leftover integer tokens are assigned to strata with the largest fractional remainders. The resulting integer quotas therefore sum exactly to the requested legal target.

## 5. Corpus-level target calculation

For requested total `T` and legal share `L`:

```text
legal_target  = round(T × L)
replay_target = T - legal_target
```

Examples:

| Total | Legal share | Legal target | Replay target |
|---:|---:|---:|---:|
| 5B | 0.90 | 4.5B | 0.5B |
| 10B | 0.90 | 9.0B | 1.0B |
| 10B | 0.95 | 9.5B | 0.5B |

The complete local SlimPajama replay corpus was measured at approximately 828,511,368 estimated tokens. Consequently, a 10B 90:10 request fails its raw capacity check because it asks for 1B replay tokens before filtering or deduplication. This failure is intentional.

The name `SlimPajama-1B` must not be interpreted as a guarantee that the local converted JSONL contains one billion usable tokens under this estimator.

## 6. Candidate oversampling

Final quotas are post-processing quotas. The script first samples a larger candidate pool to compensate for rows that will be removed by quality filters and deduplication.

For legal stratum `s`:

```text
p_s = min(1, oversample_factor × quota_s / measured_source_tokens_s)
```

For replay:

```text
p_replay = min(1, oversample_factor × replay_quota / measured_replay_tokens)
```

The default oversample factor is 1.5. A probability of 1 means that every source row is considered; it cannot create additional documents when the source is too small.

Candidate inclusion is deterministic. The script hashes:

```text
seed \0 corpus \0 source_filename \0 source_row_number
```

and compares the first 64 hash bits against the requested probability. This behaves like reproducible Bernoulli sampling without loading the corpus into memory.

Important consequences:

- The same seed, input filenames, row order, and probability select the same candidates.
- Renaming a source file or changing row order changes the sample.
- Increasing the probability preserves previously selected rows because the hash threshold grows.
- Oversampling is not sampling with replacement; no intentional duplicate replay rows are created.

## 7. Preprocessing applied to candidates

### 7.1 Text normalization

Candidates use `NormalizationConfig(unicode_form="NFKC")`. The default normalization also:

- normalizes line endings;
- decodes HTML entities;
- extracts visible text from HTML;
- removes control and format characters while preserving newlines and tabs;
- collapses horizontal whitespace;
- limits repeated blank lines; and
- strips leading and trailing whitespace.

Case, legal punctuation, citations, and paragraph structure are intentionally retained.

### 7.2 Rule-based filtering

The default `RuleFilterConfig` rejects documents that violate any of these thresholds:

| Rule | Default |
|---|---:|
| Minimum characters | 200 |
| Minimum words | 30 |
| Minimum alphabetic ratio | 0.20 |
| Maximum replacement-character ratio | 0.01 |
| Maximum repeated-line ratio | 0.60 |
| Maximum repeated 10-gram ratio | 0.50 |

It also removes standalone page/line markers and collapses pathological character runs. Remaining HTML is flagged but is not a rejection by default.

Rejected rows are written under `.work/rejected/` with their corpus, source file, source row, rejection stage, and reasons. The rejected record does not contain the original full text.

### 7.3 KenLM perplexity filtering

Unless `--skip-perplexity` is supplied, legal candidates are scored by the provided KenLM model. A candidate is accepted when its word-normalized perplexity is no greater than `--perplexity-threshold`, which defaults to 1500.

Replay is not scored by KenLM unless `--apply-kenlm-to-replay` is supplied.

The production KenLM path requires:

- `--kenlm-model /absolute/path/to/model.bin`; and
- the official `kenlm` Python extension.

The Slurm wrapper currently supplies `--skip-perplexity`, so its default execution does not run KenLM.

### 7.4 Exact deduplication

Every accepted normalized text is canonicalized and hashed with SHA256. Hashes are stored in:

```text
.work/exact_dedup.sqlite
```

The SQLite primary key guarantees one copy of each canonical exact text without retaining every text string in RAM.

The same database is shared by legal and replay candidates. Legal files are processed first, followed by replay files. Therefore, if normalized legal and replay documents are exact duplicates, the legal version is retained and the later replay version is rejected.

SQLite uses WAL mode, and the database is committed at progress intervals and at the end of preprocessing.

### 7.5 Accepted record schema

Accepted candidates contain:

```json
{
  "id": "sha256:<stable-source-id>",
  "text": "normalized text",
  "corpus": "multilegalpile_commercial",
  "type": "legislation",
  "jurisdiction": "EU",
  "source": "source name",
  "source_file": "input shard name.jsonl",
  "source_row": 12345,
  "estimated_tokens": 678,
  "perplexity": 42.1
}
```

`perplexity` is present only when KenLM was applied. The stable ID is derived from corpus, source filename, and source row—not from the document text.

Accepted and rejected staging files rotate every 100,000 rows.

## 8. MinHash near deduplication

With the default `--near-dedup text-dedup`, the accepted candidate shards are passed to the local `text-dedup` package using:

```text
algorithm                  MinHash
ngram_size                 5
num_permutations           240
similarity_threshold       0.50 by default
seed                       --shuffle-seed
processes                  --dedup-processes
false-positive verification disabled
cluster column             retained temporarily
index column               not retained in final output
```

The output is a Hugging Face dataset stored in:

```text
.work/text_dedup_output/
```

Candidate-pair verification is disabled because all-pairs verification inside large clusters is quadratic and was previously too expensive for these corpora.

If `--near-dedup skip` is selected, exact-deduplicated staging rows flow directly to selection. This is useful for diagnosis or experiments but is not equivalent to the production cleaning pipeline.

## 9. Deterministic post-dedup bucketing

Every surviving row receives a selection priority:

```text
SHA256(seed \0 stable_document_id)
```

The first 64 priority bits assign the row to one of `--shuffle-buckets` hash ranges. Each bucket is written to `.work/post_dedup_buckets/`, loaded independently, and sorted by the full priority.

Because buckets represent ordered hash ranges and are processed in numeric order, this is an external-memory approximation of globally sorting all documents by their deterministic random priority. It avoids keeping the complete corpus in RAM. More buckets reduce the maximum memory needed to sort one bucket but create more temporary files.

While bucketing, the script computes the actual post-dedup token availability for every final group.

## 10. Post-dedup quota adjustment

### 10.1 Legal strata

A small legal stratum can contain fewer post-dedup tokens than its original proportional quota. `constrained_legal_quotas()` then:

1. caps the deficient stratum at its available amount;
2. calculates the remaining legal deficit;
3. identifies legal strata with unused post-dedup capacity;
4. redistributes the deficit proportionally according to their original source weights; and
5. repeats until the overall legal target is restored.

Redistribution never crosses from legal into replay. Thus the total legal target is preserved even when the exact `type × jurisdiction` proportions cannot be preserved after preprocessing.

If the complete post-dedup legal pool is smaller than the legal target, the run fails.

The log message currently says “preserving the total 90:10 mixture” even when `--legal-share` is not 0.90. Interpret it as preserving the configured legal/replay mixture.

### 10.2 Replay

Replay is not redistributed. If post-dedup replay availability is below the fixed replay target, the run fails:

```text
Post-dedup replay pool is ... tokens below the fixed replay quota
```

Increasing `--candidate-oversample-factor` can help only when the original sampling probability was below 1. Once replay probability is 1, the entire source is already included and the only solutions are:

- reduce the replay target;
- reduce the total target;
- increase `--legal-share`; or
- add additional suitable replay data.

Sampling with replacement is not an appropriate solution because it intentionally duplicates training data and near deduplication would remove many repeated records anyway.

## 11. Final selection and overshoot

For each deterministic candidate, the script checks whether its group has already reached its quota. If not, it writes the complete document and adds its estimated tokens.

Documents are never truncated to hit a quota. Consequently, each group generally includes the document that crosses its target. This causes a small positive overshoot.

At the corpus level, legal and replay are each required to satisfy:

```text
target ≤ selected ≤ floor(target × (1 + overshoot_tolerance))
```

The default tolerance is 1%. A deficit of even one estimated token is not accepted. The final combined total must also be at least the requested total.

The old explanatory string at the bottom of the Python file mentions a 0.1% deficit rule. That text is stale. The implemented behavior is no deficit and a configurable 1% maximum overshoot by default.

Before final writing, internal `__selection_priority__` and text-dedup `__INDEX__` fields are removed. If a `__CLUSTER__` value exists, it becomes:

```json
{"dedup_cluster_id": "text-dedup:<cluster>"}
```

## 12. Output sharding

`FinalShardWriter` starts a new JSONL shard before adding a document that would take the current shard beyond `--shard-target-tokens`. A single unusually large document is still written intact.

With the default target, names look like:

```text
cpt-proportional-5b-00000.jsonl
cpt-proportional-5b-00001.jsonl
...
```

The 200M setting is an estimated-token target per file, not a row limit and not an exact tokenizer-token limit.

For every shard, the writer records:

- filename;
- document count;
- estimated tokens;
- serialized JSONL bytes; and
- SHA256 checksum.

## 13. Output directory layout

During a run:

```text
datasets/CPT-Proportional-<size>/
├── .work/
│   ├── exact_dedup.sqlite
│   ├── exact_dedup.sqlite-wal
│   ├── preprocessing_complete.json
│   ├── preprocessed/accepted-*.jsonl
│   ├── rejected/rejected-*.jsonl
│   ├── text_dedup_output/
│   └── post_dedup_buckets/bucket-*.jsonl
└── <output-prefix>-*.jsonl
```

After successful completion with the default settings:

```text
datasets/CPT-Proportional-<size>/
├── <output-prefix>-00000.jsonl
├── <output-prefix>-00001.jsonl
├── ...
├── manifest.json
└── statistics.json
```

`.work` is deleted after success unless `--keep-work-dir` is supplied.

`manifest.json` is the complete provenance record. `statistics.json` is a smaller view containing totals, quotas, and shards.

## 14. Command-line parameters

### 14.1 Corpus size and mixture

| Parameter | Default | Meaning |
|---|---:|---|
| `--total-target-tokens` | Required | Minimum final estimated-token total across legal and replay. |
| `--legal-share` | `0.90` | Legal fraction in `(0,1)`; replay share is `1 - legal_share`. |
| `--candidate-oversample-factor` | `1.50` | Candidate budget multiplier used before filtering and deduplication; must be greater than 1. |

### 14.2 Input and output paths

| Parameter | Default | Meaning |
|---|---|---|
| `--legal-dir` | `datasets/Multi_Legal_Pile_Commercial` | Directory containing legal row-data JSONL shards. |
| `--replay-dir` | `datasets/SlimPajama-1B/row_data` | Directory containing `train-*.jsonl` replay shards. |
| `--output-dir` | Derived from total | Final dataset directory. For 5B, `datasets/CPT-Proportional-5B`. |
| `--dataset-name` | Derived from total | Human-readable name stored in metadata. |
| `--output-prefix` | Derived from total | Prefix used for final JSONL filenames. |
| `--shard-target-tokens` | `200000000` | Approximate estimated-token budget per final JSONL shard. |

For non-integer billions, the derived label uses Python’s general numeric format. Explicit names are preferable for publication runs.

### 14.3 Perplexity filtering

| Parameter | Default | Meaning |
|---|---:|---|
| `--kenlm-model` | None | Path to a KenLM binary or ARPA model. Required unless skipping perplexity or resuming. |
| `--perplexity-threshold` | `1500.0` | Maximum accepted word-normalized perplexity. |
| `--apply-kenlm-to-replay` | False | Apply KenLM to replay in addition to legal candidates. |
| `--skip-perplexity` | False | Explicitly disable KenLM filtering. The current Slurm launcher enables this. |

`--apply-kenlm-to-replay` has no effect when `--skip-perplexity` is used.

### 14.4 Near deduplication

| Parameter | Default | Meaning |
|---|---:|---|
| `--near-dedup` | `text-dedup` | Run MinHash through the local package, or use `skip`. |
| `--dedup-threshold` | `0.50` | MinHash/Jaccard similarity threshold used by text-dedup. |
| `--dedup-processes` | `4` | Worker count passed to text-dedup. |

The MinHash n-gram size and permutation count are fixed in code at 5 and 240.

### 14.5 Determinism, memory, and progress

| Parameter | Default | Meaning |
|---|---:|---|
| `--shuffle-seed` | `42` | Controls candidate inclusion, MinHash seed, and final selection priority. |
| `--shuffle-buckets` | `128` | Number of post-dedup external-sort buckets. |
| `--progress-every` | `50000` | Print and SQLite commit interval in scanned rows. |
| `--corpus-overshoot-tolerance` | `0.01` | Maximum final legal and replay overshoot fraction. |
| `--quota-tolerance` | Alias | Backward-compatible alias for `--corpus-overshoot-tolerance`. |

### 14.6 Work preservation and resume

| Parameter | Default | Meaning |
|---|---:|---|
| `--keep-work-dir` | False | Preserve `.work` after a successful run. |
| `--resume-from-preprocessed` | False | Reuse a completed preprocessing marker and accepted shards. |

All numeric targets, process counts, bucket counts, and progress intervals are validated before substantive work.

## 15. Function-by-function reference

### `parse_args()`

Defines the CLI, validates ranges, derives output names, and returns the populated namespace. It does not validate input file existence; that occurs in `main()`.

### `allocate_integer(total, weights)`

Converts proportional floating-point allocations into integers that sum exactly to `total` using the largest-remainder method. It is used both for initial legal quotas and for redistribution across eligible legal strata.

### `legal_key(row)`

Returns `(type, jurisdiction)` for a raw legal row. Type is case-folded; jurisdiction is not case-folded. Input jurisdiction spelling must therefore match the configured stratum keys exactly.

### `final_group(row)`

Maps a processed row to its quota key:

```text
(multilegalpile_commercial, type, jurisdiction)
```

or the single replay key:

```text
(slimpajama_1b, general, N/A)
```

### `selected_by_probability(...)`

Performs stable hash-threshold candidate selection. It always returns true for probability 1 or greater.

### `replay_total_tokens(paths, progress_every)`

Reads every replay row, counts rows, and estimates the complete replay token capacity. Invalid JSON rows count toward the row count but contribute no tokens. This full scan occurs before replay sampling.

### `load_config(path)`

Loads a UTF-8 JSON checkpoint file into a Python dictionary.

### `prepare_candidates(...)`

Runs the expensive first phase:

1. measures replay capacity;
2. calculates candidate probabilities;
3. constructs normalization, rule, and optional KenLM filters;
4. initializes rotating accepted/rejected writers;
5. initializes shared SQLite exact deduplication;
6. scans and samples legal rows;
7. scans and samples replay rows;
8. writes preprocessing statistics; and
9. writes `.work/preprocessing_complete.json` only after the phase completes.

If this function fails before writing the completion marker, `--resume-from-preprocessed` is not safe.

### `bucket_survivors(records, work_dir, bucket_count, seed)`

Writes post-dedup rows into deterministic hash buckets and simultaneously counts available estimated tokens by final quota group. It creates the bucket directory with `exist_ok=False` to avoid silently combining data from different attempts.

### `select_final(...)`

Sorts each bucket by deterministic priority and writes complete documents until each adjusted group quota is crossed. It returns the populated `FinalShardWriter` and selected token counts by group.

### `inspect_existing_buckets(bucket_paths)`

Recounts rows and tokens from existing buckets during resume validation. Buckets are reusable only when all expected bucket files exist and their combined row count equals the saved text-dedup dataset length.

### `constrained_legal_quotas(original, available)`

Caps deficient legal strata and iteratively redistributes their missing tokens across legal strata with spare capacity. It preserves the total legal target and never borrows from replay.

### `main()`

Coordinates input discovery, target calculation, overwrite protection, preprocessing/resume, MinHash, bucket reuse, quota adjustment, final selection, validation, manifests, cleanup, and completion logging.

## 16. Important imported helpers

The proportional script reuses helpers from `cpt_data_sample_creation_v1_eu_1b.py`:

| Helper | Role |
|---|---|
| `RotatingJsonlWriter` | Rotates staging files every 100,000 rows. |
| `FinalShardWriter` | Writes estimated-token-bounded final shards and computes checksums/statistics. |
| `count_words` | Counts whitespace-delimited word spans without creating a full word list. |
| `estimate_tokens` | Returns `floor(words × 1.3)`. |
| `iter_jsonl` | Streams legal JSONL with explicit invalid-row results. |
| `iter_replay_round_robin` | Interleaves replay shards while streaming. |
| `make_stats` | Initializes preprocessing counters. |
| `process_candidate` | Normalizes, filters, exact-deduplicates, records rejection, and writes acceptance. |
| `iter_staged_jsonl` | Streams accepted staging rows when near dedup is skipped. |
| `iter_deduplicated_dataset` | Streams the Hugging Face dataset saved by text-dedup. |
| `json_safe_stats` | Converts rejection counters to JSON-serializable dictionaries. |

It also uses the package modules:

```text
legal_data_process.text_normalization
legal_data_process.rule_based_filters
legal_data_process.perplexity_filtering
legal_data_process.text_deduplication
src/text-dedup/src/text_dedup
```

## 17. Resume semantics

`--resume-from-preprocessed` is intentionally narrower than a general restart mechanism.

It requires:

```text
<output>/.work/preprocessing_complete.json
<output>/.work/preprocessed/accepted-*.jsonl
```

It validates these checkpoint values:

- requested total tokens;
- legal share;
- candidate oversample factor; and
- shuffle seed.

It does **not** currently compare every setting, such as input directory, perplexity threshold/model, dedup threshold, near-dedup choice, or rule configuration. Therefore, resume should use the same full command and software/data version as the original run.

Resume behavior by completed stage:

1. Completed preprocessing is always reused.
2. Completed text-dedup output is reused when its `state.json` exists.
3. Existing post-dedup buckets are reused only if all configured bucket files exist and their row count matches text-dedup output.
4. An incomplete/unvalidated bucket directory causes a deliberate failure rather than being overwritten.

Final JSONL or `manifest.json` artifacts prevent resume to avoid overwriting results.

Examples:

```bash
sbatch datasets_processing/run_cpt_data_sample_creation_proportional.sh \
    --total-target-tokens 5000000000 \
    --resume-from-preprocessed
```

The resume command must repeat non-default values such as `--legal-share`, `--candidate-oversample-factor`, output naming, and output directory.

## 18. Failure modes and recovery

### Replay quota exceeds raw replay capacity

Example:

```text
Replay quota 1,000,000,000 exceeds available ~828,511,368 tokens
```

This happens before preprocessing. Do not use resume. Choose a smaller replay target or add replay data. The failed attempt may leave an empty `<output>/.work`; remove it with `rmdir` only after confirming it is empty.

### Post-dedup replay deficit

The raw quota passed, but filtering or deduplication removed too much replay. If the original replay sampling probability was below 1, rerun from scratch with a larger oversample factor. If it was already 1, add replay data or lower the replay target.

### Legal stratum deficit

The script automatically redistributes a stratum-level deficit across other legal strata. It fails only if the whole legal survivor pool cannot meet the overall legal target.

### Existing output directory

A non-empty output directory is never overwritten on a fresh run. Choose a new directory or inspect and deliberately clean the failed run. Do not point recursive deletion commands at the dataset root.

### No space left during MinHash

Large MinHash runs create Hugging Face Arrow cache files that can be much
larger than the final JSONL dataset. The Slurm launcher places these files in a
job-specific directory on shared scratch:

```text
datasets/.cpt_job_cache/cpt-proportional-<job-id>/
```

It also sets `TMPDIR`, `TMP`, and `TEMP` to that location so temporary Arrow
files do not fill the compute node's small `/tmp` filesystem. On another
cluster, override the cache root at submission time:

```bash
CPT_SCRATCH_CACHE_ROOT=/path/to/large/scratch \
sbatch datasets_processing/run_cpt_data_sample_creation_proportional.sh ...
```

The launcher removes only its exact job-specific cache directory after a
successful run. It deliberately leaves that cache in place after a failure so
the failure can be inspected; remove the exact failed-job cache after checking
that no process is using it.

After a failed MinHash stage, retain `.work/preprocessed` and
`.work/preprocessing_complete.json` for resume. If
`.work/text_dedup_output` exists without `state.json`, it is incomplete and
must be quarantined or removed before using `--resume-from-preprocessed`.
Never remove the whole dataset or `.work/preprocessed` directory when
recovering this stage.

### Failure after final shards begin writing

Final selection writes JSONL before final validation and manifests. A validation failure can therefore leave final shards without `manifest.json`. Such output is incomplete and must not be used for training.

### MinHash memory or Polars/Arrow limits

The local text-dedup package loads substantial intermediate state. More system memory, rt64 Polars builds where applicable, and carefully chosen process counts may be required. Increasing `--dedup-processes` can increase peak memory rather than reduce runtime safely.

### CPU-binding failure in Slurm

The proportional Slurm launcher uses:

```bash
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=none ...
```

This avoids AMD-node topology binding errors while the Slurm cgroup still restricts the process to its allocated CPUs.

## 19. Slurm wrapper behavior

`datasets_processing/run_cpt_data_sample_creation_proportional.sh` supplies these Python defaults:

```text
--legal-share 0.9
--candidate-oversample-factor 1.5
--skip-perplexity
--dedup-processes 4
--shard-target-tokens 200000000
```

User arguments are appended after these values. For ordinary single-value argparse options, a later repeated option wins. Thus this command changes the wrapper’s legal share to 0.95:

```bash
sbatch datasets_processing/run_cpt_data_sample_creation_proportional.sh \
    --total-target-tokens 10000000000 \
    --legal-share 0.95 \
    --dataset-name CPT-Proportional-10B-95L-5R \
    --output-prefix cpt-proportional-10b-95l-5r \
    --output-dir datasets/CPT-Proportional-10B-95L-5R
```

The wrapper requests one AMD node, one task, four CPUs, all allocatable node memory, no GPU, and 24 hours.

## 20. Recommended commands

### 5B, 90% legal and 10% replay

```bash
sbatch datasets_processing/run_cpt_data_sample_creation_proportional.sh \
    --total-target-tokens 5000000000 \
    --dataset-name CPT-Proportional-5B \
    --output-prefix cpt-proportional-5b \
    --output-dir datasets/CPT-Proportional-5B
```

### 10B, 95% legal and 5% replay

This mixture fits the measured replay capacity more safely than 10B at 90:10:

```bash
sbatch datasets_processing/run_cpt_data_sample_creation_proportional.sh \
    --total-target-tokens 10000000000 \
    --legal-share 0.95 \
    --dataset-name CPT-Proportional-10B-95L-5R \
    --output-prefix cpt-proportional-10b-95l-5r \
    --output-dir datasets/CPT-Proportional-10B-95L-5R
```

### Direct Python execution

```bash
python -u datasets_processing/cpt_data_sample_creation_proportional.py \
    --total-target-tokens 5000000000 \
    --legal-share 0.9 \
    --candidate-oversample-factor 1.5 \
    --skip-perplexity \
    --dedup-processes 4
```

## 21. How to inspect a completed dataset

Start with:

```bash
python -m json.tool datasets/CPT-Proportional-5B/manifest.json | less
python -m json.tool datasets/CPT-Proportional-5B/statistics.json | less
```

Verify checksums:

```bash
cd datasets/CPT-Proportional-5B
sha256sum cpt-proportional-5b-*.jsonl
```

Compare each checksum to `manifest.json`. Also confirm:

- `final.estimated_tokens >= configuration.requested_total_tokens`;
- legal and replay totals are inside their recorded minimum/maximum bounds;
- every quota row has `selected_estimated_tokens >= adjusted_target_estimated_tokens`;
- `jsonl_shards` equals the number of final files; and
- the dataset has both legal and replay rows.

The manifest’s `target_difference_tokens` is expected to be a small non-negative number because complete documents cross quotas.

## 22. Interpretation and limitations

1. **Estimated tokens are not Qwen tokens.** A corpus reported as 5B here may tokenize to a different exact count later.
2. **Source proportions are pre-processing proportions.** The script tries to restore them post-dedup, but legal redistribution can change individual strata when survivors are insufficient.
3. **The requested legal/replay ratio is a minimum-target ratio with bounded positive overshoot.** Complete documents make exact equality generally impossible.
4. **Determinism depends on stable files and row positions.** The same seed alone is insufficient if files are renamed or reordered internally.
5. **Resume is not a configuration migration tool.** It should continue the same run, not reuse candidates under a changed preprocessing policy.
6. **`--skip-perplexity` means the resulting data has not passed KenLM filtering.** This must be disclosed in experiment documentation.
7. **Near-dedup skipping changes the data definition.** Results made with `--near-dedup skip` should not be compared as if they used the same pipeline.
8. **Replay capacity is a hard resource constraint.** Oversampling cannot manufacture source tokens.

For paper reporting, retain the script Git commit, Slurm log, `manifest.json`, `statistics.json`, exact input inventory, and final JSONL checksums with the CPT experiment.
