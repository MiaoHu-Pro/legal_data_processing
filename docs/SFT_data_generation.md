# SFT Data Generation, Sampling, and Leakage-Controlled Evaluation

## 1. Purpose

This document specifies how to create supervised fine-tuning (SFT) data from the five datasets under:

```text
datasets/SFT-datasets/
├── lawinstruct/
├── SlimOrca/
├── MetaMathQA/
├── ultrachat_200k/
└── glaive-code-assistant-v2/
```

The intended experiment combines:

- **Legal instructions:** LawInstruct, released with the FLawN-T5/LawInstruct work.
- **General instructions:** SlimOrca, MetaMathQA, UltraChat 200K, and Glaive Code Assistant v2, following the four general-instruction sources described by SaulLM-7B.

SaulLM-7B did not release its synthetic legal instruction conversations. Therefore, this project is best described as a **SaulLM-inspired general instruction mixture combined with public LawInstruct legal instructions**, not as an exact reproduction of SaulLM-7B.

The principal evaluation datasets are:

- `datasets/evaluation/legalbench_instruct/`
- `datasets/evaluation/mmlu/international_law/`
- `datasets/evaluation/mmlu/professional_law/`
- `datasets/evaluation/mmlu/jurisprudence/`

The central design decision is:

> Keep the permitted LawInstruct training data, but report LegalBench results only on explicitly defined held-out evaluation task sets. For MMLU, ensure that the three legal MMLU subjects never enter SFT training.

This follows the evaluation logic in Appendix C.2 of the LawInstruct paper more closely than deleting every LegalBench-related source from LawInstruct.

## 2. Source evidence and paper protocol

LawInstruct contains almost 12 million examples drawn from 58 legal datasets, spanning 24 languages and 17 jurisdictions. Its authors found that mixing general and legal instruction data was preferable to using LawInstruct alone. Their `flan2-lawinstruct` mixture sampled equally from the general and legal pools. Within the LawInstruct pool, sampling datasets by their number of examples generally performed better than sampling every dataset equally.

For leakage control, the paper used different policies for MMLU and LegalBench:

1. Legal tasks occurring in MMLU were excluded from LawInstruct.
2. LegalBench overlap was handled on the **evaluation side** by constructing:
   - a dataset-held-out LegalBench view; and
   - a task-held-out LegalBench view.

The paper reported 100 held-out and 61 remaining tasks for dataset-held-out evaluation, and 64 held-out and 97 remaining tasks for task-held-out evaluation. These counts correspond to its benchmark snapshot. The local LegalBench-Instruct copy contains 162 task names and has minor version drift, so task names and generated manifests—not hard-coded total counts—must define the local evaluation views.

Primary sources:

- [LawInstruct paper, NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.7/)
- [LawInstruct paper PDF, especially Appendix C.2 and D](https://aclanthology.org/2025.findings-naacl.7.pdf)
- [SaulLM-7B paper](https://arxiv.org/abs/2403.03883)

## 3. Recommended experimental variants

Use one SFT dataset for all CPT checkpoint comparisons. Changing the SFT mixture between checkpoints would confound the effect of continued pretraining.

### 3.1 Primary mixture

The recommended primary mixture is:

| Pool | Target examples | Sampling probability |
|---|---:|---:|
| Leakage-controlled English LawInstruct | 600,000 | 50% |
| Curated general instructions | 600,000 | 50% |
| **Total** | **1,200,000** | **100%** |

This is an example-balanced mixture, consistent with the LawInstruct paper's equal sampling of its top-level legal and general pools. Because legal answers and general conversational answers can have different lengths, the final token shares must also be reported. Do not claim a 50:50 token mixture unless measured token counts confirm it.

### 3.2 Useful ablations

If compute permits, create these controlled variants from the same cleaned source pools:

| Variant | Legal | General | Purpose |
|---|---:|---:|---|
| General-only | 0% | 100% | Measures the effect of legal SFT |
| Balanced | 50% | 50% | Primary paper-inspired setting |
| Legal-heavy | 70% | 30% | Tests whether stronger legal specialization helps |
| Legal-only | 100% | 0% | Diagnostic only; may weaken general instruction following |

All variants should use the same base/CPT checkpoints, tokenizer, chat template, sequence length, optimizer settings, number of tokens or update steps, and evaluation code.

## 4. Input inventory and permitted splits

### 4.1 LawInstruct

Local data consists of 142 compressed `jsonl.xz` files and is about 20 GB compressed. Actual records contain fields such as:

```json
{
  "dataset_name": "InternationalCitizenshipLawQuestions",
  "subset_name": "international_citizenship_law_questions_mode_acq",
  "instruction_language": "nl",
  "prompt_language": "en",
  "answer_language": "en",
  "jurisdiction": "INTERNATIONAL",
  "task_type": "QUESTION_ANSWERING",
  "instruction": "...",
  "prompt": "...",
  "answer": "..."
}
```

The local README describes a combined `text` field, but the actual downloaded records use separate `instruction`, `prompt`, and `answer` fields. Processing code must follow the actual schema.

For an English model and English evaluation, retain records satisfying:

```text
instruction_language == "en"
prompt_language == "en"
answer_language == "en"
```

Do not interpret "keep LawInstruct intact" as requiring all multilingual records. It means that LegalBench-related training sources are not deleted merely because they overlap with a LegalBench task; the overlap is instead controlled through declared evaluation views.

### 4.2 SlimOrca

Local file:

```text
datasets/SFT-datasets/SlimOrca/oo-labeled_correct.gpt4.sharegpt.jsonl
```

It contains 517,982 examples in ShareGPT-style `conversations`, using roles such as `system`, `human`, and `gpt`. Convert these roles to `system`, `user`, and `assistant`.

### 4.3 MetaMathQA

Local file:

```text
datasets/SFT-datasets/MetaMathQA/MetaMathQA-395K.json
```

Use `query` as the user content and `response` as the assistant content. Preserve `type` and `original_question` as metadata. The dataset is derived from the training portions of GSM8K and MATH, but it must still undergo cross-dataset and evaluation decontamination.

### 4.4 UltraChat 200K

Use only:

```text
datasets/SFT-datasets/ultrachat_200k/data/train_sft-*.parquet
```

The three `train_sft` shards contain 207,865 examples. Do not train on:

- `test_sft-*`
- `test_gen-*`
- `train_gen-*`

The `gen` splits are intended for generation/ranking workflows, not the primary SFT dataset. The test splits must remain untouched for their intended evaluation use.

### 4.5 Glaive Code Assistant v2

Local file:

```text
datasets/SFT-datasets/glaive-code-assistant-v2/glaive_code_assistant_v2.json
```

It contains approximately 215,000 examples. Map `question` to the user message and `answer` to the assistant message.

## 5. Canonical intermediate format

Normalize every source to model-independent JSONL before applying any tokenizer-specific chat template:

```json
{
  "id": "stable-source-specific-id",
  "domain": "legal",
  "source_dataset": "LawInstruct",
  "source_subset": "ContractNLI-contract_nli",
  "task_type": "NATURAL_LANGUAGE_INFERENCE",
  "language": "en",
  "messages": [
    {"role": "user", "content": "<instruction>\n\n<prompt>"},
    {"role": "assistant", "content": "<answer>"}
  ],
  "metadata": {
    "jurisdiction": "USA",
    "license": null,
    "original_file": "...jsonl.xz"
  }
}
```

For LawInstruct, form the user content as:

```text
{instruction}

{prompt}
```

Do not insert model-specific strings such as `[INST]`, `<|user|>`, or `<|assistant|>` into the saved canonical dataset. Apply the selected model tokenizer's native chat template during tokenization/training. This keeps the data portable and prevents accidental double-templating.

Train only on assistant tokens. User and system tokens should normally receive the ignore label, such as `-100`, unless a deliberately different loss design is being tested.

## 6. Cleaning and quality-control pipeline

Apply the following stages in order and save statistics after every stage.

### Stage 1: Parse and schema validation

- Stream compressed and large files; do not load LawInstruct fully into memory.
- Reject malformed JSON and invalid role sequences.
- Require at least one non-empty user message and one non-empty assistant message.
- Preserve source file, dataset, subset, language, and task metadata.
- Generate a stable ID from source name, source row identity, and normalized content hash.

### Stage 2: Text normalization

- Normalize Unicode using NFKC.
- Normalize line endings to `\n`.
- Remove NUL and disallowed control characters.
- Trim outer whitespace without destroying meaningful code indentation.
- Do not lowercase content.
- Do not aggressively remove punctuation or legal citations.

### Stage 3: Basic filters

- Remove empty or assistant-free conversations.
- Remove records whose user prompt or answer is clearly corrupted.
- Remove exact repeated messages or degenerate repeated-character output.
- Record, rather than silently discard, examples exceeding the model context limit.
- Choose and document one long-example policy: truncate, pack into chunks, or reject. Truncation must never remove the complete assistant answer without marking the example invalid.

### Stage 4: Exact deduplication

Create at least two hashes:

1. `prompt_hash = SHA256(normalized user content)`
2. `pair_hash = SHA256(normalized user content + separator + normalized assistant content)`

Deduplicate exact prompt-response pairs globally across all five sources. If the same prompt has conflicting answers, quarantine the group for inspection rather than arbitrarily keeping one answer.

Suggested source precedence for identical general examples is:

```text
human/curated target > verified synthetic target > unverified synthetic target
```

The precedence must be logged; it should not be inferred from filename order.

### Stage 5: Near deduplication

Perform prompt-level near deduplication across the complete SFT pool. A practical implementation may use MinHash/LSH on normalized word shingles. Keep conservative thresholds so that legally distinct provisions with similar boilerplate are not collapsed automatically.

Recommended practice:

- automatically remove very high-similarity prompt duplicates;
- log borderline clusters for auditing;
- prefer one representative per underlying document/question;
- group all paraphrases or views of the same source document before train/validation splitting.

### Stage 6: Evaluation contamination audit

Before sampling, compare normalized SFT prompts against every LegalBench-Instruct and legal-MMLU evaluation input using:

- exact hashes;
- whitespace/punctuation-insensitive hashes;
- near-duplicate similarity; and
- source/task metadata mappings.

This audit is additional to the paper's dataset/task held-out protocol. It should produce a report even when the primary experiment keeps LegalBench-overlapping LawInstruct examples.

## 7. MMLU leakage policy

The three evaluation subjects are:

```text
international_law
professional_law
jurisprudence
```

Only their `test` files are scored. Their `dev`, `validation`, `test`, and any auxiliary versions must never be incorporated into SFT training. Development examples may be used as few-shot demonstrations at inference only if that evaluation setting is declared consistently for every model.

The local LawInstruct filenames currently contain no obvious MMLU, `professional_law`, `international_law`, or `jurisprudence` source file. Nevertheless, filename checking is insufficient. The generated training pool must be exact- and near-matched against all three MMLU subjects before training.

Policy:

1. If an SFT record is an exact or near duplicate of any legal-MMLU evaluation item, remove it from training.
2. Save removed IDs and similarity evidence to an audit file.
3. Never remove questions from the MMLU test set to improve the reported score; training is the side that must be cleaned for MMLU.
4. Report the number of SFT examples removed per source.

This differs intentionally from the LegalBench policy because it follows the LawInstruct paper's statement that legal tasks occurring in MMLU were excluded from LawInstruct.

## 8. LegalBench-Instruct overlap policy

### 8.1 Do not use one ambiguous "clean" score

Generate and report three LegalBench-Instruct evaluation views:

| View | Purpose | Headline result? |
|---|---|---|
| Full | Compatibility with SaulLM-style full LegalBench-Instruct | Diagnostic; disclose overlap |
| Dataset-held-out | Strict source-dataset separation following Table 3 | Yes, primary strict score |
| Task-held-out | Removes semantically similar task families following Table 4 | Yes, complementary score |

The full score is useful for comparison with work that reports the complete benchmark, but it must not be presented as leakage-free when LawInstruct is used for SFT.

### 8.2 Dataset-held-out mapping from LawInstruct Table 3

Remove the following LegalBench task families from the **evaluation view**, not from the primary LawInstruct training pool:

| Overlapping source | LawInstruct source/subset | LegalBench-Instruct task selector |
|---|---|---|
| ContractNLI | `ContractNLI-contract_nli` | `contract_nli_*` |
| CUAD | `NaturalInstructionsLegal-cuad_answer_generation`, `NaturalInstructionsLegal-cuad_question_generation` | `cuad_*` |
| GLOBALCIT Citizenship Law | the two `InternationalCitizenshipLawQuestions-*` subsets | `international_citizenship_questions` |
| MAUD | `MAUD-answer`, `MAUD-category`, `MAUD-question`, `MAUD-text_type` | `maud_*` |
| OPP-115 | the two `NaturalInstructionsLegal-online_privacy_policy_*` subsets | `opp115_*` |
| Overruling | `NaturalInstructionsLegal-overruling_legal_classification` | `overruling` |
| PrivacyQA | `PrivacyQA-privacy_qa` | `privacy_policy_qa` |
| SARA | `Sara-*` and `SaraProlog-*` | `sara_*` |
| Unfair Terms of Service | `LexGLUE-unfair_tos` and the two LEXTREME online-terms subsets | `unfair_tos` |

Important paper note: the LegalBench `privacy_policy_entailment` source field was reported as incorrectly linked to PrivacyQA and derived from APP-350 instead. It must not be excluded merely by assuming it belongs to PrivacyQA.

The local LegalBench-Instruct JSONL currently contains 162 distinct `task_name` values. Applying the selectors above gives:

```text
held out: 101 tasks
retained:  61 tasks
```

The paper reported 100 held out and 61 retained because its benchmark snapshot differed. Save the explicit local manifest and benchmark hash; do not add or remove an unrelated task merely to reproduce an old count.

### 8.3 Task-held-out mapping from LawInstruct Table 4

Table 4 removes LegalBench tasks whose underlying legal operation is also taught by LawInstruct, even when the examples do not come from the same source dataset.

| Overlapping task type | Relevant LawInstruct task families | LegalBench-Instruct selector |
|---|---|---|
| Rhetorical-role labeling | `bva_decisions_label`, `indian_text_segmentation`, `german_argument_mining` | `function_of_decision_section`, `oral_argument_question_purpose` |
| Civil-procedure questions | `civipro_questions_generate_*` | `diversity_*`, `personal_jurisdiction` |
| Legal entailment | `coliee_task3_passage_entailment`, `contract_nli`, `lawng_nli_entailment` | `contract_nli_*` |
| Contractual-clause classification | `unfair_tos`, `german_rental_agreements` | `cuad_*`, `jcrew_blocker`, `unfair_tos`, `contract_qa` |

Applying these selectors to the current local 162-task LegalBench-Instruct copy gives:

```text
held out: 64 tasks
retained:  98 tasks
```

The paper reported 64 held out and 97 retained for its snapshot. Again, the generated manifest is authoritative for the local version.

### 8.4 Required evaluation manifests

Create immutable newline-delimited task manifests such as:

```text
datasets/evaluation/manifests/
├── legalbench_instruct_all_tasks.txt
├── legalbench_instruct_dataset_held_out_excluded.txt
├── legalbench_instruct_dataset_held_out_retained.txt
├── legalbench_instruct_task_held_out_excluded.txt
├── legalbench_instruct_task_held_out_retained.txt
└── manifest_metadata.json
```

`manifest_metadata.json` should include:

- source LegalBench-Instruct file path;
- SHA256 of the input JSONL/Parquet;
- unique task count;
- row count;
- generation timestamp;
- exact selector rules;
- excluded and retained task counts;
- LawInstruct paper/table version used;
- code version or Git commit.

Evaluation code should require an explicit manifest. It must not silently default to the full benchmark.

## 9. Sampling procedure

Sampling occurs only after parsing, language filtering, quality filtering, deduplication, MMLU decontamination, and validation holdout creation.

### 9.1 Determinism

Use a fixed seed, initially `42`, and store it in the dataset manifest. Use stable source IDs rather than input file order when assigning random priorities. This prevents filesystem ordering or shard layout from changing the sample.

A reproducible priority can be computed from:

```text
SHA256(seed || stable_example_id)
```

Select records by this priority within each stratum.

### 9.2 General-pool sampling

SaulLM reports filtering and deduplicating its four general sources into 600,000 instructions, but does not provide enough detail to reconstruct exact per-source weights. A reproducible primary approximation is proportional sampling by the post-cleaning number of available examples.

Using the downloaded pre-cleaning counts only as an illustration, a 600,000-example general pool would be approximately:

| Source | Approximate available examples | Illustrative 600K allocation |
|---|---:|---:|
| SlimOrca | 517,982 | 232,653 |
| MetaMathQA | 395,000 | 177,416 |
| UltraChat `train_sft` | 207,865 | 93,363 |
| Glaive Code Assistant v2 | 215,000 | 96,568 |
| **Total** | **1,335,847** | **600,000** |

Recalculate allocations after cleaning and deduplication. Do not hard-code the illustrative numbers. Use largest-remainder rounding so allocations sum exactly to the requested target.

An equal-source 150K-per-source mixture may be produced as an ablation, but it is not the primary proportional setting.

### 9.3 Legal-pool sampling

For the paper-faithful primary setting:

1. Use the post-cleaning English LawInstruct pool.
2. Treat `dataset_name + subset_name` as the sampling stratum.
3. Allocate the 600,000 legal examples proportional to each stratum's post-cleaning example count.
4. Use deterministic priorities within each stratum.
5. Redistribute unused quota deterministically when a stratum cannot fill its allocation.

LawInstruct contains very large repeated or related families, including BrCAD5, MultiEURLEX levels, LawngNLI, and Swiss judgment variants. Proportional sampling can therefore concentrate heavily on a few families. This is paper-supported but should be measured. Create a secondary temperature-sampled or capped-source ablation if task diversity is a research objective.

For temperature sampling, one possible definition is:

```text
p_i = n_i^alpha / sum_j(n_j^alpha)
```

where `n_i` is the cleaned size of stratum `i`:

- `alpha = 1.0`: proportional sampling; primary paper-inspired setting.
- `alpha = 0.5`: softens dominance by large datasets; diversity ablation.
- `alpha = 0.0`: equal sampling across strata; ablation only.

### 9.4 Mixing and sharding

After sampling legal and general pools:

1. Concatenate selected records.
2. Deterministically shuffle using the dataset seed.
3. Write JSONL shards by a declared target, preferably token count rather than byte count.
4. Avoid grouping all examples from one source into the same shard.
5. Save per-shard hashes and statistics.

Do not duplicate examples simply to reach an exact quota without recording an epoch/repetition policy. If a requested pool is smaller than its target after filtering, fail clearly or lower the target; do not silently sample with replacement.

## 10. Validation split

Create the SFT validation split before final training sampling and never reuse it in training.

Recommended policy:

- hold out 0.5% to 1% per source/subset, subject to a minimum and maximum;
- group examples derived from the same original document, question, or near-duplicate cluster;
- place the whole group in either training or validation;
- use the same stable hash and seed for every experimental variant;
- keep evaluation benchmarks completely separate from this SFT validation split.

For a strict reproduction of the LawInstruct paper, an additional variant can use its first-16-examples-per-configuration validation policy. The source-aware grouped split above is preferable for the main decoder-model experiment because it reduces document/paraphrase leakage.

## 11. Prompt formatting for training and evaluation

The dataset prompt and the model chat template serve different purposes and should both be used:

- The dataset's instruction/question is the semantic **user content**.
- The model's native chat template wraps that content into the token sequence expected by the model.

For example, a LegalBench-Instruct `inputs` field remains the user message. At inference, it is wrapped with the same model-native chat template used for SFT. Do not replace the benchmark prompt with a generic prompt, and do not store model-specific wrapper tokens permanently in the canonical dataset.

For MMLU, construct one consistent user message containing the question and four labeled choices, and request exactly one answer label (`A`, `B`, `C`, or `D`). Apply identical formatting and decoding settings to every compared model.

Use deterministic evaluation decoding, normally temperature `0`, no sampling, and a sufficient but small output-token limit for classification tasks.

## 12. Evaluation aggregation and reporting

LegalBench-Instruct is highly imbalanced by task: some tasks have tens of examples and others have thousands. Do not compute only row-level micro accuracy across all 90,394 records because large tasks would dominate.

Report at least:

- per-task score;
- macro average over tasks;
- category-level macro average when category metadata is available;
- balanced accuracy for binary/multiclass classification where appropriate;
- exact-match or task-specific metrics where balanced accuracy is not meaningful;
- the evaluation manifest name and hash.

For LegalBench-Instruct, report separate columns for:

```text
Full (overlap-disclosed)
Dataset-held-out
Task-held-out
```

For MMLU, report accuracy for each of the three legal subjects and their unweighted macro average.

The principal comparison for measuring CPT should be:

```text
base model + identical SFT
versus
CPT model + identical SFT
```

The SFT records, their order, number of trained tokens, chat template, optimization settings, and evaluation manifests must remain the same across this comparison.

## 13. Required generated artifacts

A complete data-generation run should produce:

```text
datasets/SFT-processed/<dataset-name>/
├── train/
│   ├── train-00000.jsonl
│   └── ...
├── validation/
│   └── validation-00000.jsonl
├── manifests/
│   ├── selected_examples.jsonl
│   ├── rejected_examples.jsonl
│   ├── duplicate_clusters.jsonl
│   ├── mmlu_overlap_removed.jsonl
│   ├── source_statistics.json
│   ├── shard_statistics.json
│   └── dataset_manifest.json
└── README.md
```

`dataset_manifest.json` should contain:

- source paths and hashes;
- source licenses and usage restrictions;
- filtering configuration;
- language policy;
- exact and near-deduplication configuration;
- MMLU decontamination thresholds and removals;
- random seed;
- requested and realized source quotas;
- example and token counts before and after every stage;
- tokenizer name and revision used for token statistics;
- chat-template name/revision, without baking it into canonical JSONL;
- maximum sequence length and truncation policy;
- output shard hashes;
- Git commit and software versions.

## 14. Acceptance checks before SFT

Do not begin training until all of the following pass:

- [ ] Only approved training splits were used.
- [ ] LawInstruct records satisfy the declared language policy.
- [ ] Every record follows the canonical message schema.
- [ ] Every conversation ends with a non-empty assistant response.
- [ ] Exact duplicates have been removed across all five sources.
- [ ] Near-duplicate statistics have been reviewed.
- [ ] No legal-MMLU exact/near duplicates remain in training.
- [ ] LegalBench full, dataset-held-out, and task-held-out manifests exist.
- [ ] Local held-out counts are explained and not forced to paper counts.
- [ ] SFT validation groups do not occur in SFT training.
- [ ] Requested and realized mixture shares are recorded by examples and tokens.
- [ ] Token-length distributions and truncation rates are reported by source.
- [ ] Licenses are recorded per source/subset.
- [ ] All output shards have SHA256 hashes.
- [ ] A small end-to-end tokenization and loss-mask test passes.

## 15. Recommended headline protocol

For the first main experiment:

1. Build an English-only, cleaned, deduplicated LawInstruct pool.
2. Remove any exact/near matches to the three legal-MMLU subjects from SFT training.
3. Keep LegalBench-overlapping LawInstruct sources in training, matching the paper's main approach.
4. Sample 600K LawInstruct examples proportionally by cleaned source/subset size.
5. Build a cleaned, cross-deduplicated 600K general pool from the four SaulLM general sources, sampled proportionally by cleaned source size.
6. Mix the two pools 50:50 by examples and report the realized token ratio.
7. Train every base/CPT checkpoint with precisely the same SFT dataset and settings.
8. Evaluate MMLU on the three untouched legal test subjects.
9. Evaluate LegalBench-Instruct on full, dataset-held-out, and task-held-out manifests.
10. Use the dataset-held-out and task-held-out results as the leakage-controlled headline results; label the full score as overlap-disclosed.

This design preserves valuable legal supervision while making the limits of each evaluation score explicit and reproducible.
