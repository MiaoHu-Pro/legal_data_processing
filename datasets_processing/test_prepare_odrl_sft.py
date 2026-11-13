"""Run with python -m unittest discover -s datasets_processing -p test_prepare_odrl_sft.py."""
import json
from pathlib import Path
import unittest

from prepare_odrl_sft import ROOT, assign_splits, block, policy_key


class PreparationTests(unittest.TestCase):
    def test_markers(self):
        self.assertEqual(block('<|begin_odrl|>{}<|end_odrl|>', 'odrl'), '{}')
        for text in ('{}', '<|begin_odrl|><|end_odrl|>',
                     '<|begin_odrl|>{}<|end_odrl|><|begin_odrl|>{}<|end_odrl|>'):
            with self.assertRaises(ValueError):
                block(text, 'odrl')

    def test_policy_identity(self):
        self.assertEqual(policy_key({'uid': 'a', 'permission': []}),
                         policy_key({'permission': [], 'uid': 'b'}))
        self.assertNotEqual(policy_key({'permission': [{'target': 'a'}]}),
                            policy_key({'permission': [{'target': 'b'}]}))

    def test_prompt_union_and_reproducibility(self):
        rows = [{'policy_group': str(i), 'direction': 'text2odrl',
                 'messages': [{'content': 'same' if i < 2 else str(i)}]} for i in range(30)]
        assign_splits(rows, 42, .1, .1)
        self.assertEqual(rows[0]['group_id'], rows[1]['group_id'])
        expected = {r['policy_group']: r['split'] for r in rows}
        rows.reverse()
        assign_splits(rows, 42, .1, .1)
        self.assertEqual(expected, {r['policy_group']: r['split'] for r in rows})

    def test_generated_artifacts(self):
        root = ROOT / 'datasets/ODRL_SFT-processed'
        if not root.exists():
            self.skipTest('Run preparation first for artifact integration checks')
        assignments = {}
        direction_groups = {}
        for direction in ('text2odrl', 'odrl2text'):
            prompts, identities, groups = {}, set(), set()
            for split in ('train', 'validation', 'test'):
                rows = [json.loads(line) for line in (root / direction / f'{split}.jsonl').read_text().splitlines()]
                self.assertTrue(rows)
                for row in rows:
                    self.assertNotIn(row['id'], identities)
                    identities.add(row['id'])
                    group = row['group_id']
                    groups.add(group)
                    self.assertEqual(assignments.setdefault(group, split), split)
                    prompt = row['messages'][0]['content']
                    self.assertEqual(prompts.setdefault(prompt, split), split)
                    self.assertEqual([m['role'] for m in row['messages']], ['user', 'assistant'])
                    if direction == 'text2odrl':
                        self.assertIsInstance(json.loads(row['messages'][1]['content']), dict)
                if split != 'train':
                    evaluation = [json.loads(line) for line in (root / direction / f'{split}_generation.jsonl').read_text().splitlines()]
                    for row in evaluation:
                        self.assertEqual([m['role'] for m in row['messages']], ['user'])
                        self.assertTrue(row['references'])
                        self.assertEqual(assignments[row['group_id']], split)
            direction_groups[direction] = groups
        self.assertEqual(direction_groups['text2odrl'], direction_groups['odrl2text'])


if __name__ == '__main__':
    unittest.main()
