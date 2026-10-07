legal_kenlm.bin is not necessary for CPT model training itself.

  It is only required for the optional perplexity-filtering preprocessing step:

  raw legal data
  → normalization
  → rule filtering
  → KenLM perplexity filtering  ← legal_kenlm.bin
  → text-dedup
  → CPT JSONL data
  → CPT training

  You have two choices:

  - Recommended paper-aligned pipeline: train legal_kenlm.bin and use it to remove high-perplexity/noisy documents.
  - Initial simplified pipeline: skip perplexity filtering while retaining normalization, rule filtering, exact deduplication, and text-dedup.

  For the simplified pipeline:

  python datasets_processing/cpt_data_sample_creation_v1_eu_1b.py \
    --output-dir datasets/CPT-V1-EU-1B \
    --skip-perplexity \
    --near-dedup text-dedup \
    --dedup-threshold 0.5 \
    --dedup-processes 8



---


Run with nohup, skipping only KenLM while retaining full text-dedup:



cd /local/scratch/mh1f25/legal_data_process

nohup python -u datasets_processing/cpt_data_sample_creation_v1_eu_1b.py \
--output-dir datasets/CPT-V1-EU-1B \
--replay-target-tokens 140000000 \
--shard-target-tokens 200000000 \
--skip-perplexity \
--near-dedup text-dedup \
--dedup-threshold 0.5 \
--dedup-processes 8 \
> datasets_processing/cpt_v1_eu_1b.log 2>&1 &

echo $! > datasets_processing/cpt_v1_eu_1b.pid

The output directory must be empty or nonexistent before starting.

Monitor progress:

tail -f datasets_processing/cpt_v1_eu_1b.log

Check the process:

ps -fp "$(cat datasets_processing/cpt_v1_eu_1b.pid)"

After completion:

python -m json.tool datasets/CPT-V1-EU-1B/statistics.json

statistics.json contains:

- Total documents.
- Total words.
- Estimated tokens.
- JSONL size in bytes, decimal GB, and GiB.
- Number of output shards.
- Documents and tokens by corpus.
- Replay token percentage.
- Documents and tokens by legal type and jurisdiction.
- Rule-filter rejection counts.
- Exact-duplicate counts.
- Per-shard sizes, tokens, document counts, and SHA-256 checksums.

Also check physical disk usage with:

du -sh datasets/CPT-V1-EU-1B

The reported tokens are estimates based on 1.3 × words. Exact Qwen or Mistral token counts should be computed later using the selected model tokenizer.

The text-dedup dependencies must be installed before running this command; the script will stop during preflight if they are missing.