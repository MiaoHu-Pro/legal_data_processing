"""Dependency-free CPU tests using deterministic tokenizer/dataset test doubles.

This test double is not a substitute for server-side Qwen tokenizer validation.
"""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, os.environ.get('TRAINING_PROJECT_ROOT', str(Path.home() / 'scratch/lex-legal-post-training')))
from src.training.sft_odrl_data import build_tokenized_splits, encode_record, validate_messages
from src.training.sft_data import configure_tokenizer


class OffsetTokenizer:
    is_fast = True
    pad_token_id = None
    markers = {'<|im_start|>': 1, '<|im_end|>': 2}

    def __call__(self, text, **kwargs):
        ids, offsets, index = [], [], 0
        while index < len(text):
            marker = next((m for m in self.markers if text.startswith(m, index)), None)
            end = index + (len(marker) if marker else 1)
            ids.append(self.markers[marker] if marker else ord(text[index]) + 10)
            offsets.append((index, end))
            index = end
        return {'input_ids': ids, 'offset_mapping': offsets}

    def encode(self, text, **kwargs):
        return self(text)['input_ids']

    def convert_ids_to_tokens(self, identity):
        return next(m for m, i in self.markers.items() if i == identity)


class OdrlEncodingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = OffsetTokenizer()
        configure_tokenizer(self.tokenizer)
        self.row = {'direction': 'text2odrl', 'split': 'train', 'group_id': 'train-policy',
                    'messages': [{'role': 'user', 'content': 'describe policy'},
                                 {'role': 'assistant', 'content': '{"permission": []}'}]}

    def test_masking_both_directions(self):
        for direction in ('text2odrl', 'odrl2text'):
            row = copy.deepcopy(self.row)
            row['direction'] = direction
            if direction == 'odrl2text':
                row['messages'][0]['content'] = '{"permission": []}'
                row['messages'][1]['content'] = 'allow print'
            before = copy.deepcopy(row)
            result = encode_record(row, self.tokenizer, 128)
            self.assertEqual(row, before)
            prefix = '<|im_start|>user\n' + row['messages'][0]['content'] + '<|im_end|>\n<|im_start|>assistant\n'
            n = len(self.tokenizer.encode(prefix, add_special_tokens=False))
            self.assertTrue(all(label == -100 for label in result['labels'][:n]))
            gold = self.tokenizer.encode(row['messages'][1]['content'] + '<|im_end|>', add_special_tokens=False)
            self.assertEqual([label for label in result['labels'] if label != -100], gold)

    def test_invalid_and_overlength(self):
        row = copy.deepcopy(self.row)
        row['messages'].reverse()
        with self.assertRaises(ValueError):
            validate_messages(row)
        with self.assertRaisesRegex(ValueError, 'overlength'):
            encode_record(self.row, self.tokenizer, 2)

    def test_split_and_leakage_guards(self):
        dataset_module = SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows), DatasetDict=dict)
        with patch.dict(sys.modules, {'datasets': dataset_module}):
            self.check_split_and_leakage_guards()

    def check_split_and_leakage_guards(self):
        val = copy.deepcopy(self.row)
        val.update(split='validation', group_id='val-policy', direction='odrl2text')
        dataset, stats = build_tokenized_splits([self.row, val], self.tokenizer)
        self.assertEqual(set(dataset), {'train', 'validation'})
        self.assertEqual(stats['train']['examples'], 1)
        val['group_id'] = self.row['group_id']
        with self.assertRaisesRegex(ValueError, 'multiple splits'):
            build_tokenized_splits([self.row, val], self.tokenizer)
        val.update(split='test', group_id='test-policy')
        with self.assertRaisesRegex(ValueError, 'Unexpected split'):
            build_tokenized_splits([self.row, val], self.tokenizer)


if __name__ == '__main__':
    unittest.main()
