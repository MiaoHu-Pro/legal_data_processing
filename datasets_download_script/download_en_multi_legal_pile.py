from datasets import load_dataset
import json
from pathlib import Path


ds = load_dataset(
    "joelniklaus/Multi_Legal_Pile_Commercial",
    # "joelito/Multi_Legal_Pile",
    "en_all",
    split="train",
    streaming=True,
)



output_dir = Path("./datasets/Multi_Legal_Pile_Commercial")
output_dir.mkdir(parents=True, exist_ok=True)

ROWS_PER_FILE = 1_000_000
count = 0
out = None

try:
    for row in ds:
        if row.get("language") != "en":
            continue

        # Open a new file for the first record and every 1,000,000 records.
        if count % ROWS_PER_FILE == 0:
            if out is not None:
                out.close()

            part = count // ROWS_PER_FILE + 1
            path = output_dir / f"multi_legal_pile_en_commercial_{part:05d}.jsonl"
            out = path.open("w", encoding="utf-8")
            print(f"Writing {path}")

        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("count:", count)
        count += 1

finally:
    if out is not None:
        out.close()

print(f"Saved {count:,} English records")


# nohup .venv/bin/python -u download_en_multi_legal_pile.py \
#   > download_en_multi_legal_pile.log 2>&1 < /dev/null &
# echo $! > download_en_multi_legal_pile.pid
# watch -n 60 'df -h /local /home; du -sh ~/.cache/huggingface 2>/dev/null; du -sh /local/scratch/mh1f25/legal_data_process/datasets 2>/dev/null'


"""
note:

compressed remote dataset
        ↓ decompression
HF streaming
        ↓
plain JSON
        ↓
283 GB local files

It is completely reasonable.
So I would interpret your current result as:
MultiLegalPile-Commercial-English is about 283 GB in uncompressed JSONL representation
with 17.01 million documents/records

"""