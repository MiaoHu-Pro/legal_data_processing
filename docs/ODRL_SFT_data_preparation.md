# ODRL SFT data preparation

## Scope and completed outputs

Source: `datasets/SFT-datasets/odrl_data`. Script: `datasets_processing/prepare_odrl_sft.py`.
Completed output: `datasets/ODRL_SFT-processed/{text2odrl,odrl2text}`.

This prepares **ODRL-only** supervised examples. No LawInstruct or general instruction data is added, and no model is trained. Use identical splits for these seven starting checkpoints:

| Starting checkpoint | Prior training |
|---|---|
| Qwen3-8B-Base | No additional legal CPT/SFT |
| Qwen-Lex-Legal-CPT-using-Legislation-3B | Legal CPT |
| Qwen-Lex-Legal-CPT-using-Proportional-5B-FA2 | Legal CPT |
| Qwen-Lex-Legal-CPT-using-Proportional-10B-95L-5R | Legal CPT |
| Qwen-Lex-Legal-CPT-using-Legislation-3B-SFT-AllLegal | Legal CPT + legal/general SFT |
| Qwen-Lex-Legal-CPT-using-Proportional-5B-FA2-SFT-AllLegal | Legal CPT + legal/general SFT |
| Qwen-Lex-Legal-CPT-using-Proportional-10B-95L-5R-SFT-AllLegal | Legal CPT + legal/general SFT |

The JSONL files are ready for chat-based SFT ingestion, **but the existing `train_sft.py` expects tokenized Arrow datasets**. Optional server-side tokenization below produces that layout. No Qwen3 tokenizer was available locally, so the completed local output is not yet tokenized.

## Inspection findings

Each direction contains 3,711 train, 464 validation and 464 test rows, totaling 4,639. The authoritative inputs are the six Parquet files, each with `instruction`, `input`, `output` string fields. Existing converted JSON copies are ignored.

There are only **1,414 distinct policy groups**, shared by both directions. In each original direction, **738 policy groups occur in multiple source splits**. Therefore, using those original splits would expose held-out policies during training through other paraphrases or the reverse translation task.

All source policy JSON objects parse successfully. This verifies JSON syntax, not full ODRL conformance or semantic agreement with the natural language.

The paired natural-language sets match exactly across directions after extracting summaries: 4,506 distinct policy/description pairs per direction. The remaining 133 rows per direction repeat an existing training pair under direct translation.

## Target format

The default is `--target-format direct`, matching the requested translation tasks:

- `text2odrl`: user instruction + natural-language description → assistant ODRL JSON only.
- `odrl2text`: user instruction + ODRL JSON → assistant natural-language summary only.

The original text-to-ODRL target contains RCIR followed by JSON. The original ODRL-to-text target contains NLIR followed by a summary. The default extracts only JSON or summary and replaces the instruction accordingly. Original fields remain in `source_records.jsonl` for traceability.

For experiments that intentionally teach intermediate representations, use `--target-format structured` and a separate output directory. That retains the original instructions, inputs and outputs. Do not mix target formats within a comparison unintentionally.

JSON object keys are sorted and indentation standardized. Values and array ordering are preserved; no rule, constraint, party, or asset is deliberately removed. Only the **grouping fingerprint** omits top-level `uid`; the actual training target retains it.

Each SFT row has:

```json
{
  "id": "sha256 of the message pair",
  "direction": "text2odrl",
  "policy_uid": "original policy identifier",
  "group_id": "shared leakage-control group",
  "split": "train",
  "messages": [
    {"role": "user", "content": "Task instruction\n\nSource text or JSON"},
    {"role": "assistant", "content": "Gold JSON or description"}
  ],
  "provenance": [{"file": "source Parquet path", "row": 0, "split": "original split"}]
}
```

This is a schema illustration, not a literal source example. Inspect real examples with `head -n 1 datasets/ODRL_SFT-processed/text2odrl/train.jsonl` (and likewise for `odrl2text`).

## Leakage-safe splitting and deduplication

1. Read only Parquet files and validate strings, marker blocks and JSON objects; malformed records fail preparation rather than silently disappearing.
2. Group canonical policy objects, ignoring generated top-level policy UID for grouping only.
3. Merge groups if a normalized user prompt is identical within a direction. Normalization uses Unicode NFKC, case folding and whitespace collapsing for this leakage check only.
4. Deterministically order groups by SHA-256 of seed and group identifier, then allocate 10% validation and 10% test (rounded group counts), leaving 80% training.
5. Apply **one shared split assignment to both directions**.
6. Remove identical message pairs per direction, retaining all source locations. Distinct paraphrases remain. Multiple summaries of one policy are legitimate references, not automatically erroneous conflicting labels.

Default seed: 42. Group counts: train **1,132**, validation **141**, test **141**.

| Direction | Train pairs | Validation pairs | Test pairs | Total pairs | Exact duplicates removed |
|---|---:|---:|---:|---:|---:|
| text2odrl | 3,604 | 451 | 451 | 4,506 | 133 |
| odrl2text | 3,604 | 451 | 451 | 4,506 | 133 |

Post-processing group overlap between splits is zero. This is an exact-policy/prompt guard, not proof of absence of near-duplicate templates or prior checkpoint contamination. If these policies were used in earlier model training, a genuinely unseen external test set is still necessary.

## Files in each direction

| File/directory | Purpose |
|---|---|
| `train.jsonl` | SFT pairs; use only these rows for gradient updates |
| `validation.jsonl` | Validation loss and model-selection pairs |
| `test.jsonl` | Reserved final-test pairs; never use for tuning |
| `validation_generation.jsonl` | User-only messages plus separate reference answers |
| `test_generation.jsonl` | User-only messages plus separate reference answers |
| `source_records.jsonl` | Original fields, source split/row, transformed messages, new group/split |
| `manifest.json` | Source hashes, settings, counts; tokenizer signature/statistics when tokenized |
| `dataset/` | Optional Arrow DatasetDict: train, validation, test |
| `tokenizer/` | Optional saved Qwen tokenizer with the trainer's chat template |

Generation files group all distinct references for each exact input. In each held-out split, `text2odrl` has 451 unique prompts; `odrl2text` has 141 unique policy prompts with 451 references. At evaluation time, pass **only `messages`** to the model, append the assistant generation header, and keep `references` exclusively for scoring.

For fair reporting, consider per-policy macro averages: a policy with more paraphrases should not silently dominate a score. JSON parse success and legal-semantic correctness are different metrics. Literal JSON string equality is overly sensitive to formatting; also report rule/field-level agreement. Generated policy UIDs and asset/party identifiers may not be inferable from the supplied prose, and source summaries/intermediate representations may omit details. Audit these limitations before treating exact reconstruction as a semantic accuracy metric. This preparation does not invent missing information or certify the source gold labels.

## Commands

Reproduce the completed JSONL preparation into a **new** directory:

```bash
cd "${HOME}/scratch/legal_data_process"
python datasets_processing/prepare_odrl_sft.py \
    --output-dir datasets/ODRL_SFT-processed-reproduced
```

Generate the Arrow/tokenizer layout used by the current SFT trainer on the server. This is a CPU preparation task; it loads tokenizer files, not model weights. Run in an environment containing `pyarrow`, `datasets`, `transformers`, `tokenizers`, and the existing training project's dependencies:

```bash
conda activate lex_legal_rl_post_training_env
cd "${HOME}/scratch/legal_data_process"
python datasets_processing/prepare_odrl_sft.py \
    --output-dir datasets/ODRL_SFT-processed-Qwen3 \
    --tokenizer-path "${HOME}/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base" \
    --training-project "${HOME}/scratch/dips_project/lex-legal-post-training" \
    --max-length 4096
```

This rebuilds identical JSONL splits alongside tokenized artifacts. Set `PREPARED_DIR` for the existing training launcher to either:

```bash
PREPARED_DIR="${HOME}/scratch/legal_data_process/datasets/ODRL_SFT-processed-Qwen3/text2odrl"
# OR, for an independent reverse-direction experiment:
PREPARED_DIR="${HOME}/scratch/legal_data_process/datasets/ODRL_SFT-processed-Qwen3/odrl2text"
```

Use new model/checkpoint output names, e.g. `<starting-model>-ODRL-text2odrl`; never overwrite the seven starting checkpoints. These directions support separate experiments; simply selecting one directory does not jointly train both directions. A joint-task experiment should merge only corresponding train splits, retain the shared held-out groups, and be reported separately.

The optional tokenizer path calls the existing `configure_tokenizer`, `encode_messages`, and `tokenizer_signature` functions from `src/training/sft_data.py`. Thus it uses the same non-thinking Qwen chat template, supervises only assistant content/end-of-turn, masks user/header tokens with `-100`, and adds no new vocabulary. Overlength examples raise an error instead of truncating ODRL JSON. The trainer checks that each starting model has the identical tokenizer signature. Actual token counts become available in each tokenized manifest; counts here are examples, not tokens.

This dataset is much smaller than LawInstruct: review batch size, number of epochs, evaluation/save intervals and learning rate before reusing large-corpus training defaults. No training jobs have been submitted by this preprocessing task.

## Parameters and functions

| Parameter | Default | Meaning |
|---|---|---|
| `--input-dir` | `datasets/SFT-datasets/odrl_data` | Root containing both authoritative source folders |
| `--output-dir` | `datasets/ODRL_SFT-processed` | New destination; refuses an existing directory |
| `--target-format` | `direct` | `direct` translation or original `structured` targets |
| `--seed` | `42` | Reproducible group split and row ordering |
| `--validation-fraction` | `0.1` | Fraction of groups for validation, not fraction of rows |
| `--test-fraction` | `0.1` | Fraction of groups reserved for final testing |
| `--tokenizer-path` | omitted | Local Qwen tokenizer; omission produces JSONL only |
| `--training-project` | `~/scratch/lex-legal-post-training` | Project supplying compatible chat/masking helpers; override on your server |
| `--max-length` | `4096` | Tokenized conversation ceiling; must match training setting |

`load_records` parses and transforms authoritative data; `block` validates tagged sections; `policy_key` computes canonical grouping; `assign_splits` unions policy/prompt groups and allocates shared splits; `prepare` deduplicates, writes artifacts and manifests; `tokenize` optionally produces trainer-compatible Arrow data. JSON helpers preserve Unicode. Preparation writes a staging directory and renames it only after success. Failed staging directories are retained and explicitly reported; they are not complete datasets. There is intentionally no destructive `--overwrite` option.

## Verification

```bash
python -m unittest discover -s datasets_processing -p test_prepare_odrl_sft.py
```

Tests cover marker validation, ignoring only the top-level policy UID, prompt-group unions, deterministic assignment, generated JSON validity, no duplicate pair IDs, aligned policy splits across directions, and user-only evaluation inputs. The four tests passed against the completed local direct-format artifacts. Real Qwen tokenization/GPU training still requires the tokenizer/server environment.
