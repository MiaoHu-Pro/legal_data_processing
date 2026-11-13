"""Prepare paired ODRL translation tasks with shared, policy-disjoint splits.

Only authoritative Parquet inputs are read. Existing outputs are never overwritten.
Optional offline Qwen tokenization uses the same masking code as the SFT trainer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = {'text2odrl': 'text-to-odrl-dataset-v1',
              'odrl2text': 'odrl-to-text-dataset-v1'}
SPLITS = ('train', 'validation', 'test')


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def block(text, name):
    matches = re.findall(r'<\|begin_' + name + r'\|>(.*?)<\|end_' + name + r'\|>', text, re.S)
    if len(matches) != 1 or not matches[0].strip():
        raise ValueError(f'Expected exactly one nonempty {name} block')
    return matches[0].strip()


def policy_key(policy):
    # Generated top-level identity must not allow otherwise identical policies
    # into separate splits. Nested asset/party identities remain meaningful.
    return digest(dump({k: v for k, v in policy.items() if k != 'uid'}))


def normalize_prompt(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def load_records(root, target_format):
    import pyarrow.parquet as pq
    records, files = [], []
    for direction, folder in DIRECTIONS.items():
        for split in SPLITS:
            paths = sorted((root / folder / 'data').glob(split + '-*.parquet'))
            if not paths:
                raise ValueError(f'Missing {folder}/{split} Parquet files')
            for path in paths:
                relative = str(path.relative_to(root))
                files.append({'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                row_index = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        for field in ('instruction', 'input', 'output'):
                            if not isinstance(row.get(field), str) or not row[field].strip():
                                raise ValueError(f'{relative}:{row_index}: invalid {field}')
                        if direction == 'text2odrl':
                            policy = json.loads(block(row['output'], 'odrl'))
                            block(row['output'], 'rcir')
                            instruction = 'Convert the following policy description to ODRL JSON. Output only the JSON object.'
                            input_text = row['input'].strip()
                            answer = json.dumps(policy, ensure_ascii=False, sort_keys=True, indent=2)
                        else:
                            policy = json.loads(row['input'])
                            block(row['output'], 'nlir')
                            instruction = 'Describe the following ODRL policy in natural language, covering its rules, parties, assets, and constraints. Output only the description.'
                            input_text = json.dumps(policy, ensure_ascii=False, sort_keys=True, indent=2)
                            answer = block(row['output'], 'summary')
                        if not isinstance(policy, dict) or not policy:
                            raise ValueError(f'{relative}:{row_index}: expected policy object')
                        if target_format == 'structured':
                            instruction, input_text, answer = (row[k].strip() for k in ('instruction', 'input', 'output'))
                        messages = [{'role': 'user', 'content': instruction + '\n\n' + input_text},
                                    {'role': 'assistant', 'content': answer}]
                        if any(marker in m['content'] for m in messages for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
                            raise ValueError('Embedded chat delimiter in source')
                        records.append({'direction': direction, 'policy_uid': policy.get('uid'),
                                        'policy_group': policy_key(policy), 'messages': messages,
                                        'source': {'file': relative, 'row': row_index, 'split': split},
                                        'original': row})
                        row_index += 1
    return records, files


def assign_splits(records, seed, validation_fraction, test_fraction):
    """Union identical policies OR normalized user prompts before splitting."""
    parent = {r['policy_group']: r['policy_group'] for r in records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    prompts = {}
    for row in records:
        key = (row['direction'], normalize_prompt(row['messages'][0]['content']))
        current = find(row['policy_group'])
        if key in prompts:
            previous = find(prompts[key])
            parent[max(current, previous)] = min(current, previous)
        prompts[key] = row['policy_group']
    groups = sorted({find(k) for k in parent}, key=lambda k: digest(f'{seed}:{k}'))
    n_val, n_test = round(len(groups) * validation_fraction), round(len(groups) * test_fraction)
    if not n_val or not n_test or n_val + n_test >= len(groups):
        raise ValueError('Insufficient groups for nonempty train/validation/test splits')
    mapping = {g: ('validation' if i < n_val else 'test' if i < n_val + n_test else 'train')
               for i, g in enumerate(groups)}
    for row in records:
        row['group_id'] = find(row['policy_group'])
        row['split'] = mapping[row['group_id']]
    return Counter(mapping.values())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(dump(row) + '\n')


def tokenize(output, rows, args, splits=SPLITS):
    # Share the existing trainer's exact template, masking and signature contract.
    sys.path.insert(0, str(args.training_project.expanduser().resolve()))
    from src.training.sft_data import configure_tokenizer, encode_messages, tokenizer_signature
    from transformers import AutoTokenizer
    from datasets import Dataset, DatasetDict
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path.expanduser()), local_files_only=True)
    configure_tokenizer(tokenizer)
    datasets, stats = {}, {}
    for split in splits:
        # Fail instead of silently truncating or changing held-out coverage.
        encoded = [encode_messages(r['messages'], tokenizer, args.max_length) for r in rows if r['split'] == split]
        datasets[split] = Dataset.from_list(encoded)
        stats[split] = {'examples': len(encoded), 'tokens': sum(len(r['input_ids']) for r in encoded),
                        'assistant_tokens': sum(sum(t != -100 for t in r['labels']) for r in encoded),
                        'max_tokens': max(len(r['input_ids']) for r in encoded)}
    DatasetDict(datasets).save_to_disk(str(output / 'dataset'))
    tokenizer.save_pretrained(output / 'tokenizer')
    return {'smoke_subset': False, 'max_length': args.max_length,
            'tokenizer_signature': tokenizer_signature(tokenizer), 'statistics': stats}


def prepare(args):
    source = args.input_dir.expanduser().resolve()
    target = args.output_dir.expanduser().resolve()
    if target.exists():
        raise ValueError(f'Output already exists; choose a new directory: {target}')
    if not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1 or args.validation_fraction + args.test_fraction >= 1:
        raise ValueError('Split fractions must be positive and sum to less than one')
    records, files = load_records(source, args.target_format)
    original_groups = defaultdict(set)
    for row in records:
        original_groups[(row['direction'], row['policy_group'])].add(row['source']['split'])
    group_counts = assign_splits(records, args.seed, args.validation_fraction, args.test_fraction)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=target.name + '.building-', dir=target.parent))
    report = {'format_version': 1, 'target_format': args.target_format, 'seed': args.seed,
              'input_dir': str(source), 'source_files': files, 'group_counts': dict(group_counts),
              'split_policy': 'seeded policy-group shuffle; top-level uid ignored; identical normalized prompts unioned',
              'fractions': {'validation': args.validation_fraction, 'test': args.test_fraction},
              'tokenized': args.tokenizer_path is not None, 'directions': {}}
    joint_rows = []
    try:
        for direction in DIRECTIONS:
            output = staging / direction
            output.mkdir()
            selected = [r for r in records if r['direction'] == direction]
            unique = {}
            for row in selected:
                identity = digest(dump(row['messages']))
                if identity not in unique:
                    unique[identity] = {k: row[k] for k in ('messages', 'policy_uid', 'group_id', 'split', 'direction')}
                    unique[identity].update(id=identity, provenance=[])
                unique[identity]['provenance'].append(row['source'])
            rows = sorted(unique.values(), key=lambda r: digest(f'{args.seed}:{r["id"]}'))
            joint_rows.extend(r for r in rows if r['split'] != 'test')
            write_jsonl(output / 'source_records.jsonl', selected)
            for split in SPLITS:
                subset = [r for r in rows if r['split'] == split]
                write_jsonl(output / f'{split}.jsonl', subset)
                if split != 'train':
                    # One evaluation prompt per unique input, all known golds.
                    prompts = {}
                    for row in subset:
                        prompt = row['messages'][0]['content']
                        item = prompts.setdefault(prompt, {'id': digest(prompt), 'group_id': row['group_id'],
                                                          'messages': row['messages'][:1], 'references': []})
                        answer = row['messages'][1]['content']
                        if answer not in item['references']:
                            item['references'].append(answer)
                    write_jsonl(output / f'{split}_generation.jsonl', prompts.values())
            stats = {'source_rows': len(selected), 'unique_pairs': len(rows),
                     'exact_duplicates_removed': len(selected) - len(rows),
                     'source_cross_split_policy_groups': sum(len(v) > 1 for (d, _), v in original_groups.items() if d == direction),
                     'rows_by_split': dict(Counter(r['split'] for r in rows)), 'post_split_group_overlap': 0}
            seen = defaultdict(set)
            for row in rows:
                seen[row['group_id']].add(row['split'])
            assert all(len(v) == 1 for v in seen.values())
            manifest = dict(report, direction=direction, statistics=stats)
            if args.tokenizer_path:
                manifest.update(tokenize(output, rows, args))
            write_json(output / 'manifest.json', manifest)
            report['directions'][direction] = stats
        # Joint instruction tuning: both tasks, but no test rows in this dataset.
        joint_rows.sort(key=lambda r: digest(f'{args.seed}:joint:{r["id"]}'))
        joint = staging / 'joint'
        joint.mkdir()
        for split in ('train', 'validation'):
            write_jsonl(joint / f'{split}.jsonl', (r for r in joint_rows if r['split'] == split))
        joint_stats = {split: dict(Counter(r['direction'] for r in joint_rows if r['split'] == split))
                       for split in ('train', 'validation')}
        manifest = dict(report, direction='joint', statistics=joint_stats,
                        tasks=list(DIRECTIONS), test_policy='Separate directional test sets only')
        if args.tokenizer_path:
            manifest.update(tokenize(joint, joint_rows, args, splits=('train', 'validation')))
        write_json(joint / 'manifest.json', manifest)
        report['joint'] = joint_stats
        write_json(staging / 'manifest.json', report)
        staging.rename(target)
    except Exception:
        print(f'Incomplete output retained for inspection at {staging}; not training-ready', file=sys.stderr)
        raise
    print(json.dumps(report['directions'], indent=2))
    print(f'Saved: {target}; tokenized={report["tokenized"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'datasets/SFT-datasets/odrl_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'datasets/ODRL_SFT-processed')
    parser.add_argument('--target-format', choices=('direct', 'structured'), default='direct')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--tokenizer-path', type=Path, help='Optional local Qwen3 tokenizer; never downloads weights')
    parser.add_argument('--training-project', type=Path, default=Path.home() / 'scratch/lex-legal-post-training')
    parser.add_argument('--max-length', type=int, default=4096)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
