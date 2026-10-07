
### 1. Build official KenLM

cd /local/scratch/mh1f25/legal_data_process/src

git clone https://github.com/kpu/kenlm.git

cmake -S kenlm -B kenlm/build \
-DCMAKE_BUILD_TYPE=Release

cmake --build kenlm/build -j 8

Install the official Python binding needed by perplexity_filtering.py:

python -m pip install ./kenlm

KenLM officially supports building with CMake and training models with lmplz. Official KenLM repository (https://github.com/kpu/kenlm)

### 2. Prepare a small, high-quality legal corpus

Create:

/local/scratch/mh1f25/legal_data_process/models/kenlm/eu_legal_clean.txt

Use a carefully selected subset of EU:

- legislation;
- case law;
- contracts;
- no malformed or very short documents;
- normalization and rule filtering completed;
- exact duplicates removed;
- no perplexity filtering yet.

A reasonable first model could use approximately 10–50 million words, balanced across the three EU legal types. Do not train it on the entire unfiltered corpus, because the model could learn the noise it is supposed to detect.

KenLM expects preprocessed UTF-8 text with newline-delimited training units. Special strings such as <s>, </s>, and <unk> should not appear in the input. Official corpus-format documentation (https://kheafield.com/code/kenlm/estimation/)

### 3. Train a 5-gram ARPA model

mkdir -p models/kenlm/tmp

src/kenlm/build/bin/lmplz \
-o 5 \
-S 70% \
-T models/kenlm/tmp \
--prune 0 0 1 \
< models/kenlm/eu_legal_clean.txt \
> models/kenlm/eu_legal_5gram.arpa

-o 5 creates a 5-gram model. --prune 0 0 1 removes singleton n-grams from order three upward, reducing model size. KenLM documents -S for memory and -T for temporary storage. Official estimation documentation (https://kheafield.com/code/kenlm/estimation/)

### 4. Convert ARPA to the required binary file

src/kenlm/build/bin/build_binary \
trie \
models/kenlm/eu_legal_5gram.arpa \
models/kenlm/legal_kenlm.bin

The resulting file is:

/local/scratch/mh1f25/legal_data_process/models/kenlm/legal_kenlm.bin

The trie format uses less memory than the default probing representation. Official KenLM binary-format documentation (https://kheafield.com/code/kenlm/structures/)

### 5. Verify the model

python - <<'PY'
import kenlm

path = "models/kenlm/legal_kenlm.bin"
model = kenlm.Model(path)

text = "The Court shall have jurisdiction to determine the dispute."
print("Order:", model.order)
print("Score:", model.score(text, bos=True, eos=True))
print("Perplexity:", model.perplexity(text))
PY

### 6. Create the CPT dataset

python datasets_processing/cpt_data_sample_creation_v1_eu_1b.py \
--output-dir datasets/CPT-V1-EU-1B \
--kenlm-model models/kenlm/legal_kenlm.bin \
--replay-target-tokens 140000000 \
--shard-target-tokens 200000000 \
--near-dedup text-dedup \
--dedup-threshold 0.5 \
--dedup-processes 8

The 1500 perplexity threshold should be treated as an initial value. Before the full run, score samples of known-good legal documents and known-noisy documents and check their distributions; the appropriate cutoff depends on the training corpus, tokenization, vocabulary, and pruning settings.
