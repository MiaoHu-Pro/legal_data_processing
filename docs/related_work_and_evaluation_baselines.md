
| Paper | Evaluation dataset / benchmark | What it evaluates |
|---|---|---|
| **FLawN-T5** | **LegalBench** | Legal classification and reasoning. The paper excludes **Rule QA** and normally holds out tasks overlapping with LawInstruct. |
| | **MMLU — all 57 subjects** | General knowledge and generalization, including the three law subjects. |
| **SaulLM-7B** | **LegalBench-Instruct** | Legal reasoning using a revised version of LegalBench with clearer prompts and explicit answer formats. |
| | **Legal-MMLU** | Three MMLU subjects: **International Law**, **Professional Law**, and **Jurisprudence**. |
| | **Four newly collected legal document sets** | Legal-domain language modelling, measured using **perplexity**; sources listed below. |
| **SaulLM-54B / SaulLM-141B** | **LegalBench-Instruct** | Legal reasoning. |
| | **Legal-MMLU** | **International Law**, **Professional Law**, and **Jurisprudence**. |

Evaluation workflow:
Evaluate both after CPT and after SFT, and also evaluate the original model before training. The three benchmarks need an evaluation pipeline: you cannot simply feed their raw rows into the model.

three common datasets are used to do evaluation: LegalBench, LegalBench-Instruct, Legal-MMLU


**Evaluate both after CPT and after SFT, and also evaluate the original model before training.** The three benchmarks need an evaluation pipeline: you cannot simply feed their raw rows into the model.

One distinction matters: **LegalBench-Instruct is a modified version of LegalBench**, so their results are related rather than two independent measures of legal ability.

**1. When should you evaluate?**

For your Qwen3-8B project, I recommend:

| Checkpoint | Evaluation | Purpose |
|---|---|---|
| **Original model, before training** | All three benchmarks | Establish your baseline. |
| **After CPT** | All three benchmarks, plus perplexity on held-out legal text | Measure how legal pretraining changes knowledge and reasoning. |
| **After SFT** | All three benchmarks | Measure the additional effect of instruction tuning. |
| **After DPO/RL, if used** | All three benchmarks | Measure the effect of preference or reward training. |

During training, use **separate validation data** for loss monitoring and checkpoint selection. Reserve the benchmark test sets for reporting results; repeatedly selecting models based on their test scores can bias the comparison.

For CPT, perplexity is useful because a model may improve its understanding of legal text while still struggling to follow an answer-format instruction. The SaulLM-7B paper evaluates both its base model and instruction-tuned models, and separately measures legal-document perplexity.  
:chatgpt-content-reference{index="2"}

**2. Do the raw datasets need processing?**

**Yes—mainly prompt construction and scoring preparation, rather than the extensive cleaning used for CPT.**

| Benchmark | Preparation required |
|---|---|
| **Legal-MMLU** | Select `international_law`, `professional_law`, and `jurisprudence`; format the question and four choices; keep the correct answer separately for scoring. |
| **LegalBench** | Apply each task’s official prompt template; insert the relevant input fields; preserve its labels and task-specific scoring rules. |
| **LegalBench-Instruct** | Use the revised prompts and explicit response instructions; insert the task input; preserve the expected labels separately. |

LegalBench provides task-specific prompt templates and evaluation code. Some open-response tasks require manual assessment, so define and report the exact task subset you evaluate. [GitHub](https://github.com/HazyResearch/legalbench/?utm_source=chatgpt.com)

A Legal-MMLU input would look like:

```text
Question: ...
A. ...
B. ...
C. ...
D. ...

Answer:
```

The gold answer—for example, `C`—belongs in the evaluator, **not in the test prompt**.

Your pipeline should:

1. Load the fixed benchmark version and evaluation split.
2. Build the prescribed prompts, including any few-shot examples.
3. Apply the appropriate model input format and tokenize.
4. Generate answers or score answer-choice probabilities.
5. Score predictions against the gold answers.

For a fair comparison, **fix the prompts, task subset, few-shot count, answer-scoring method, and aggregation method across checkpoints**. Document necessary differences between base-model formatting and instruct-model chat templates. For Qwen3, also keep the thinking mode consistent.

For MMLU, choose either answer-choice likelihood scoring or generated-letter scoring and report which you use. For SaulLM comparisons, reproduce its balanced-accuracy protocol; do not assume every evaluator’s default score matches the paper.

Also check your **CPT and SFT data for overlap with benchmark test examples**. This is especially relevant when using LawInstruct, which shares some source tasks with LegalBench.

**3. What SFT datasets did the papers use?**

The papers call SFT **instruction fine-tuning (IFT)**.

| Paper | General SFT data | Legal SFT data |
|---|---|---|
| **FLawN-T5** | An updated **Flan instruction mixture** | **LawInstruct**: approximately 12 million examples from 58 datasets across 24 languages. The T5/Flan-T5 experiments use English data. |
| **SaulLM-7B** | **SlimOrca**, **MetaMathQA**, **UltraChat**, and **Glaive Code Assistant v2**; approximately 600,000 general instructions after filtering and deduplication | **Synthetic legal conversations**, generated using Mistral-7B-Instruct from legal documents and associated metadata |
| **SaulLM-54B / 141B** | **UltraInteract** and **Dolphin**; approximately 1 million curated general instructions | **Synthetic legal dialogues and question–answer pairs**, generated with the corresponding large Mixtral instruct models |

For **FLawN-T5**, the paper compares several mixtures, including Flan alone, LawInstruct alone, and their combination. It excludes overlapping LegalBench tasks from its main evaluation unless otherwise specified.  
:chatgpt-content-reference{index="3"}

For **SaulLM-54B/141B**, keep the stages separate: **LawInstruct’s commercial portion and UltraChat were used during model annealing**, whereas the SFT mixture is the one listed above. **UltraFeedback and Orca are described as preference-training data for DPO**, a subsequent stage.  
:chatgpt-content-reference{index="4"}



**1. Yes—preserve the evaluation data and prepare the prompts and scorer.**

You generally **do not need CPT-style cleaning or deduplication** for these established benchmarks.

Your preparation should cover:

- Selecting the correct tasks and official splits.
- Constructing each task’s prescribed prompts.
- Keeping gold answers separate from model inputs.
- Applying the model’s input/chat format.
- Parsing predictions and computing the specified metrics.

**Do not remove duplicates, rewrite questions, or normalize the benchmark text aggressively:** that changes the evaluation set and can make your scores incomparable. Check for benchmark overlap in your **training data**, and remove that overlap from training where appropriate.

LegalBench supplies task-specific prompts and evaluation code; downloading its rows alone does not reproduce its evaluation protocol. [github.com](https://github.com/HazyResearch/legalbench/?utm_source=chatgpt.com)

**2. Hugging Face download links**

| Benchmark | Hugging Face link | What to download |
|---|---|---|
| **Legal-MMLU** | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | Only the three configurations: `international_law`, `professional_law`, `jurisprudence` |
| **LegalBench-Instruct** | [Equall/legalbench_instruct](https://huggingface.co/datasets/Equall/legalbench_instruct) | The task configurations and their evaluation data |
| **LegalBench** | [nguha/legalbench](https://huggingface.co/datasets/nguha/legalbench) | The task configurations and their official splits |

I verified all three repositories. **Legal-MMLU is the law subset of MMLU**, rather than a separate dataset repository in this setup. [Datasets at Hugging Face](https://huggingface.co/datasets/cais/mmlu?utm_source=chatgpt.com)

For LegalBench’s accompanying prompts and scoring code, also use the [official GitHub repository](https://github.com/HazyResearch/legalbench).

**3. Recent published papers to add to your reading list**

the following are useful verified publications beyond your three papers. I have labelled **main conference**, **Findings**, and **Industry Track** separately. This is a focused shortlist, not an exhaustive survey.

| Paper | Confirmed publication | Training approach | Relevance to your project |
|---|---|---|---|
| [**LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models**](https://aclanthology.org/2026.acl-industry.120/) | **ACL 2026 — Industry Track** | Teacher-generated reasoning data, verification, **SFT + DPO** | Particularly relevant: trains **Qwen3-0.6B/1.7B** students and evaluates on four LegalBench tasks. A recent method to adapt to your Qwen3-8B project. [ACL Anthology](https://aclanthology.org/2026.acl-industry.120/?utm_source=chatgpt.com) |
| [**ViLegalLM: Language Models for Vietnamese Legal Text**](https://aclanthology.org/2026.findings-acl.1801/) | **Findings of ACL 2026** | **CPT + downstream fine-tuning/instruction tuning**; synthetic legal datasets | Includes a **Qwen3-1.7B-Base** model continually pretrained on a 16 GB Vietnamese legal corpus. Useful Qwen3 adaptation evidence, although its language differs from yours. [ACL Anthology](https://aclanthology.org/2026.findings-acl.1801/?utm_source=chatgpt.com) |
| [**Unilaw-R1: A Large Language Model for Legal Reasoning with Reinforcement Learning and Iterative Inference**](https://aclanthology.org/2025.emnlp-main.915/) | **EMNLP 2025 — Main conference** | **Qwen2.5-7B-Instruct → SFT → GRPO**, plus iterative inference | Strong reference for legal reasoning post-training at approximately your model size. Uses about **17,000 distilled reasoning examples**; evaluation includes Chinese legal benchmarks. [ACL Anthology](https://aclanthology.org/2025.emnlp-main.915/?utm_source=chatgpt.com) |
| [**Efficient Domain Continual pretraining by Mitigating the Stability Gap**](https://aclanthology.org/2025.acl-long.1578/) | **ACL 2025 — Main conference** | CPT strategies addressing early performance degradation, including data sampling | Directly relevant to your **CPT design**. Includes legal and medical domain experiments with Llama-family models. [ACL Anthology](https://aclanthology.org/2025.acl-long.1578/?utm_source=chatgpt.com) |
| [**Legal Mathematical Reasoning with LLMs: Procedural Alignment through Two-Stage Reinforcement Learning**](https://aclanthology.org/2025.findings-emnlp.84/) | **Findings of EMNLP 2025** | **LexPam:** curriculum-based, two-stage **GRPO** with procedural rewards | Relevant to compliance training: rewards legally required steps as well as final-answer correctness. Evaluated on **LexNum**, a Chinese legal mathematical reasoning benchmark. [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.84/?utm_source=chatgpt.com) |
| [**NyayaAnumana and INLegalLlama**](https://aclanthology.org/2025.coling-main.738/) | **COLING 2025 — Main conference** | **Legal CPT → task-specific SFT** | A concrete legal training pipeline for Indian judgment prediction, with explanations. Useful related work, but its task and jurisdiction differ from your English benchmark suite. [ACL Anthology](https://aclanthology.org/2025.coling-main.738/?utm_source=chatgpt.com) |

Another recent **ACL 2026 main-conference** paper is [**Mitigating Legal Hallucinations via Symbolic Constraints and Analogical Precedents**](https://aclanthology.org/2026.acl-long.633/). Its AALawyer framework combines constrained statute retrieval and precedent retrieval. It is useful for a future retrieval-based comparison, rather than as your central CPT/SFT method baseline. [ACL Anthology](https://aclanthology.org/2026.acl-long.633/?utm_source=chatgpt.com)

**For your experiments, I would prioritize these comparisons:**

| Comparison | What it establishes |
|---|---|
| Original Qwen3-8B checkpoint | Starting performance |
| Qwen3-8B + legal SFT, without CPT | Whether CPT adds value beyond SFT |
| Qwen3-8B + legal CPT + the same SFT | Your domain-adaptation baseline |
| The above + DPO or GRPO | The additional contribution of post-training |
| SaulLM-7B / SaulLM-7B-Instruct | External legal-model reference |

My recommended reading order is **LegalDrill → Unilaw-R1 → Stability Gap → LexPam**. These cover synthetic SFT data, preference training, legal RL, and CPT design.

Treat the Chinese and Vietnamese models primarily as **method references**. To compare methods fairly, implement them on your chosen backbone and English training data, then evaluate every checkpoint with the same benchmark protocol.
