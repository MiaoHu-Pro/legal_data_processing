Based on the code and the final log line, I would treat this as a **successful complete export of the `en_all` configuration** of `joelniklaus/Multi_Legal_Pile_Commercial`.

You have:

$$
\boxed{17,011,009\text{ English records}}
$$

occupying:

$$
\boxed{283\text{ GB of uncompressed JSONL}}
$$

on your filesystem.

That is plausible for this corpus. The dataset card says the Commercial dataset is structured by text type/language/jurisdiction and that `en_all` means all English text types. [Hugging Face](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile_Commercial/blob/main/README.md?code=true\&utm_source=chatgpt.com)

## 1. Your code is logically correct

This part is correct:

```python
ds = load_dataset(
    "joelniklaus/Multi_Legal_Pile_Commercial",
    "en_all",
    split="train",
    streaming=True,
)
```

`en_all` means:

```text
English
+
all legal text types
```

so it should include the English records available to that configuration.

Your loop then writes every streamed record to JSONL:

```python
for row in ds:
    if row.get("language") != "en":
        continue
```

The language check is actually redundant because `en_all` should already yield English records, but it is harmless.

Then:

```python
out.write(json.dumps(row, ensure_ascii=False) + "\n")
count += 1
```

means `count` really is the number of rows successfully written.

Because your log ends with:

```text
Saved 17,011,009 English records
```

the iterator reached its end normally and the `finally` block closed the last output file.

So:

$$
\boxed{\text{yes, 17,011,009 records were written}}
$$

to your `datasets/Multi_Legal_Pile` directory.

---

# 2. 283 GB is also believable

Your:

```bash
du -sh
```

shows:

```text
283G    .
```

This measures your **expanded JSONL representation**, not the compressed download size.

The source MultiLegalPile data is generally stored in compressed formats such as `jsonl.xz`; the dataset card explicitly describes its files as `jsonl.xz`. [Hugging Face](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile_Commercial?utm_source=chatgpt.com)

Your script does:

```python
json.dumps(row)
```

and writes ordinary uncompressed:

```text
.jsonl
```

files.

Therefore:

```text
compressed remote dataset
        ↓ decompression
HF streaming
        ↓
plain JSON
        ↓
283 GB local files
```

is completely reasonable.

So I would interpret your current result as:

$$
\boxed{
\text{MultiLegalPile-Commercial-English}
\approx
283\text{ GB in your uncompressed JSONL representation}
}
$$

with:

$$
\boxed{
17.01\text{ million documents/records}
}
$$

That's much better than my earlier rough estimate because now you have the **actual downloaded corpus**.

---


---

# 5. Before CPT, don't treat the 17M rows as one homogeneous corpus

This is the next important step.

You said you like this architecture:

```text
MultiLegalPile-Commercial-English
│
├── legislation
│   ├── EU / EUR-Lex
│   ├── UK legislation
│   └── other legislation
│
├── contracts
│
├── case law
│
└── other
```

I agree.

And your downloaded rows should contain metadata that makes this possible, because MultiLegalPile is organized as:

```text
type → language → jurisdiction
```

with legal types including:

```text
caselaw
contracts
legislation
other
legal_mc4
```

according to the dataset card. [Hugging Face](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile_Commercial/blob/main/README.md?code=true\&utm_source=chatgpt.com)

The full analysis is now complete. Its results are stored in:

```text
processing_analysis_output/
├── multi_legal_pile_commercial_global_summary.json
├── multi_legal_pile_commercial_stats_by_type_jurisdiction.csv
└── multi_legal_pile_commercial_stats_by_type_jurisdiction_source.csv
```

The global result is:

| Metric | Result |
|---|---:|
| Input JSONL files | 18 |
| Documents | 17,011,009 |
| Text size | 294.654 decimal GB |
| Words | 42,426,641,943 |
| Estimated tokens | 55,154,634,525 |
| Bad JSON rows | 0 |
| Missing text | 0 |
| Missing type | 0 |
| Missing jurisdiction | 0 |
| Missing source | 17,011,009 |

The token counts here are planning estimates calculated as
`1.30 × word count`; they are not exact Qwen or Mistral tokenizer counts. The
small four-token difference between the global JSON and the sum of the CSV rows
comes from integer rounding within individual groups.

## 5.1 Results by type and jurisdiction

| Type | Jurisdiction | Documents | Text GB | Words | Est. tokens | Token share |
|---|---|---:|---:|---:|---:|---:|
| case law | EU | 104,422 | 1.566 | 222,891,827 | 0.290B | 0.53% |
| case law | US | 12,129,586 | 178.052 | 25,193,044,115 | 32.751B | 59.38% |
| contracts | EU | 11,581 | 0.461 | 39,029,843 | 0.051B | 0.09% |
| contracts | US | 653,056 | 35.937 | 5,299,500,846 | 6.889B | 12.49% |
| legal-mc4 | N/A | 179,732 | 3.024 | 483,531,111 | 0.629B | 1.14% |
| legislation | EU | 374,834 | 7.801 | 764,885,624 | 0.994B | 1.80% |
| legislation | Switzerland | 147 | 0.009 | 1,354,433 | 0.002B | <0.01% |
| legislation | UK | 36,499 | 0.247 | 41,724,213 | 0.054B | 0.10% |
| legislation | US | 302,079 | 11.085 | 1,726,402,980 | 2.244B | 4.07% |
| other | N/A | 3,147,221 | 47.471 | 7,414,448,134 | 9.639B | 17.48% |
| other | US | 71,852 | 9.001 | 1,239,828,817 | 1.612B | 2.92% |
| **Total** |  | **17,011,009** | **294.654** | **42,426,641,943** | **55.155B** | **100.00%** |

## 5.2 Aggregated results by legal type

| Type | Documents | Document share | Text GB | Est. tokens | Token share |
|---|---:|---:|---:|---:|---:|
| case law | 12,234,008 | 71.92% | 179.619 | 33.041B | 59.91% |
| other | 3,219,073 | 18.92% | 56.472 | 11.251B | 20.40% |
| contracts | 664,637 | 3.91% | 36.398 | 6.940B | 12.58% |
| legislation | 713,559 | 4.19% | 19.141 | 3.295B | 5.97% |
| legal-mc4 | 179,732 | 1.06% | 3.024 | 0.629B | 1.14% |

## 5.3 Aggregated results by jurisdiction

| Jurisdiction | Documents | Document share | Text GB | Est. tokens | Token share |
|---|---:|---:|---:|---:|---:|
| US | 13,156,573 | 77.34% | 234.075 | 43.496B | 78.86% |
| N/A | 3,326,953 | 19.56% | 50.495 | 10.267B | 18.62% |
| EU | 490,837 | 2.89% | 9.828 | 1.335B | 2.42% |
| UK | 36,499 | 0.21% | 0.247 | 0.054B | 0.10% |
| Switzerland | 147 | <0.01% | 0.009 | 0.002B | <0.01% |

## 5.4 What the results mean for sampling

The corpus is highly imbalanced:

- US documents provide 78.86% of estimated tokens.
- US case law alone provides 59.38% of all estimated tokens.
- Case law across jurisdictions provides 59.91% of tokens.
- Legislation provides only 5.97% of tokens, even though it is a priority for
  the intended legal CPT mixture.
- EU data provides 2.42% of tokens, while UK data provides only 0.10%.
- The `other/N/A` group is large at 17.48% and requires quality inspection
  before assigning it a substantial sampling weight.

Therefore, uniform document sampling or uniform token sampling from the raw
corpus would mostly produce US case law. The fixed CPT dataset should instead
use explicit token quotas by `type × jurisdiction`, with deliberate
oversampling of legislation, EU material, and UK material relative to their raw
shares. US case law should be capped so it does not dominate the training run.

The source-level CSV does not currently provide an additional sampling level:
all 17,011,009 rows have `source = UNKNOWN`. Source-based balancing is therefore
not possible from the exported metadata unless source information can be
reconstructed or enriched. For the current data, the reliable sampling keys are
`type` and `jurisdiction`.

These results confirm that the next step should **not** be direct tokenization
of the complete corpus. First preprocess and deduplicate the documents, measure
the surviving tokens in every stratum, and then resolve the final sampling
quotas from those post-processing counts.

---

# 6. I would not use all 283 GB for the first CPT run

This is now especially clear.

You have an excellent source corpus, but:

$$
17\text{ million documents}
$$

is much more than you need for your first experiment.

Instead, create a fixed sampled CPT corpus from it.

For LexLegal, something like:

```text
                         CPT sampling
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
       legislation         contracts         case law
        highest              high              high
        weight              weight            weight
            │
       ┌────┴────┐
       ▼         ▼
      EU         UK
   highest      high
```

with `other` receiving a lower sampling probability.

Then add:

```text
small SlimPajama/general replay  (SlimPajama-1B)
```

afterwards.

---
# 7. Data Sampling 

For your project, I would strongly avoid:

$$
\text{download 35B tokens}
\rightarrow
\text{train everything}.
$$

SaulLM used about 30B cleaned tokens, but they used **256 MI250 GPUs** for CPT. Your first goal is to establish whether your pipeline works and whether Qwen benefits from legal CPT.

A much better progression is:

$$
\boxed{
1B
\rightarrow
3B
\rightarrow
5B
\rightarrow
\text{optional larger run}
}
$$

For example, from `Multi_Legal_Pile_Commercial-English` you can stream and construct:

```text
CPT-v1      1B tokens    pipeline/debug run
CPT-v2      3B tokens    main initial experiment
CPT-v3      5B tokens    scaling experiment
CPT-v4     10B+ tokens   only if results justify it
CPT-v5     all tokens   only if results justify it
```

And use **the exact same sampled documents** for Mistral and Qwen.

### I would also select by legal type

Instead of blindly using `en_all`, your first 1–3B tokens could deliberately sample:

```text
MultiLegalPile-Commercial-English
│
├── legislation     ← highest priority
│   ├── EU / EUR-Lex
│   ├── UK legislation
│   └── other legislation
│
├── contracts       ← high priority
│
├── case law        ← high priority
│
└── other           ← lower proportion
```

The loader confirms that English includes UK legislation (`uk_uk_lex`), and it also dynamically loads EU EUR-Lex decisions, directives, regulations, recommendations and proposals. [Hugging Face](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile_Commercial/blob/main/Multi_Legal_Pile_Commercial.py)

For **LexLegal**, I think this controlled 1–3B token subset is actually more valuable than mechanically consuming the entire estimated 30–45B tokens.

If you want the **exact English Commercial size**, the best next step is to write a small streaming script that enumerates `en_all` without downloading everything, counts documents/characters/tokens by `type` and `jurisdiction`, and gives you something like:

```text
EU legislation      X GB    X tokens
UK legislation      X GB    X tokens
US legislation      X GB    X tokens
case law             X GB    X tokens
contracts            X GB    X tokens
other                X GB    X tokens
----------------------------------------
TOTAL                X GB    X tokens
```

That would also tell us exactly how to construct your 1B/3B-token CPT mixture.

---
# 8. You now have a much better basis for the experiment

Your actual Track A becomes:

$$
\boxed{
17.01M\text{ English commercial legal records}
}
$$

↓

classify/count by:

$$
\text{type}\times\text{jurisdiction}\times\text{source}
$$

↓

clean/normalization/filtering/deduplicate if necessary

↓

create a fixed, sampled:

$$
\boxed{D_{\mathrm{CPT-v1}}} or  \boxed{D_{\mathrm{CPT-v2}}}, ...,  \boxed{D_{\mathrm{CPT-v5}}} 
$$

perhaps:

$$
1B\text{ tokens} \ or  \ 3B\text{ tokens}, ..., 10B\text{ tokens} 
$$

initially.

↓

give the **same exact documents** to:

$$
\text{Mistral-7B}
$$

and:

$$
\text{Qwen3-8B}.
$$

That is a very solid controlled experiment.
