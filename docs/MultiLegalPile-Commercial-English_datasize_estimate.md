
### data size estimate

For:

$$
\boxed{\text{Multi_Legal_Pile_Commercial — English}}
$$

I would budget roughly:

$$
\boxed{\mathbf{300\text{–}450\ GB}}
$$

of **expanded/raw text-equivalent data**, corresponding very roughly to:

$$
\boxed{\mathbf{30\text{–}45\ billion\ tokens}}
$$

before your own additional filtering/deduplication.

I would use **~350 GB / ~35B tokens** as a planning estimate until you actually stream/count it.

Why this range? The published MultiLegalPile tables report approximately **503,712 MB and 50.5B tokens** for the full English Pile-of-Law component alone. [Hugging Face](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile_Commercial/blob/main/README.md?utm_source=chatgpt.com) The Commercial loader excludes a number of sources but still pulls substantial English material from Pile-of-Law, EUR-Lex, Legal-MC4, UK legislation, and related resources. [Hugging Face](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile_Commercial/blob/main/Multi_Legal_Pile_Commercial.py)

### Important: downloaded disk size will be much smaller

The source files are generally compressed (`jsonl.xz`). Therefore there are three different quantities:

| Quantity | Rough expectation |
|---|---:|
| Compressed download | perhaps tens of GB |
| Expanded text | **~300–450 GB** |
| Training tokens | **~30–45B tokens** |

Don't reserve 450 GB and assume that's the network download itself.

Also, the Hugging Face page showing only ~42 KB for `Multi_Legal_Pile_Commercial` is **not the corpus size**. That's mainly the dataset loader and README; the loader fetches the actual data from other repositories. [Hugging Face](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile_Commercial/tree/main?utm_source=chatgpt.com)

### But you do not need all of it

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
300M
\rightarrow
1B
\rightarrow
3B
\rightarrow
\text{optional larger run}
}
$$

For example, from `Multi_Legal_Pile_Commercial-English` you can stream and construct:

```text
CPT-v0    300M tokens    pipeline/debug run
CPT-v1      1B tokens    main initial experiment
CPT-v2      3B tokens    scaling experiment
CPT-v3     10B+ tokens   only if results justify it
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