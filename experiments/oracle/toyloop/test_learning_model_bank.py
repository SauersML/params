import json
from pathlib import Path
import tempfile
import types
import unittest
import learning_model_bank as bank


class BankTest(unittest.TestCase):
    def test_fresh_train_manifest_strict_roles_duplicates_families_schema(self):
        import copy
        protocol = {'schema':'fresh_train_bank_v1',
                    'generator_source':bank.GENERATOR_COMMIT+' zoo.make(family,seed)',
                    'specifications':[{'case':f'{f}_{3300000+100*i}', 'family':f,
                        'seed':3300000+100*i,'split':'train','context_seed':(3300000+100*i)*7+3}
                        for i,f in enumerate(bank.FAMILIES)]}
        self.assertEqual(len(bank.specifications(protocol)),8)
        for key,value in [('split','evaluation'),('family','multihop'),('case','../escape')]:
            changed=copy.deepcopy(protocol);changed['specifications'][0][key]=value
            with self.assertRaises(ValueError):bank.specifications(changed)
        changed=copy.deepcopy(protocol);changed['specifications'][1]=changed['specifications'][0]
        with self.assertRaises(ValueError):bank.specifications(changed)
        changed=copy.deepcopy(protocol);changed['unknown']=True
        with self.assertRaises(ValueError):bank.specifications(changed)
    def protocol(self):
        return {'families': list(bank.FAMILIES), 'model_counts': {
            'existing_exposed_train': 8, 'new_train': 16, 'new_model_evaluation': 8},
            'generator_source': bank.GENERATOR_COMMIT+' zoo.make(family,seed)'}

    def test_frozen_roles_seeds_no_legacy_filter(self):
        rows = bank.specifications(self.protocol())
        self.assertEqual(len(rows), 24)
        self.assertEqual([r['seed'] for r in rows[:2]], [3100000, 3100001])
        self.assertEqual(rows[16]['seed'], 3200000)
        self.assertEqual(sum(r['split']=='train' for r in rows), 16)
        changed = self.protocol(); changed['families'].reverse()
        with self.assertRaises(ValueError): bank.specifications(changed)

    def fake(self, fail=False):
        import torch
        def make(family, seed):
            if fail: raise RuntimeError('retained construction failure')
            return {'family': family, 'seed': seed}, {
                'wte.weight': torch.zeros(32, 4),
                'h.0.mlp.down_proj.weight': torch.tensor([[0., 1., 0.], [0., 0., 0.]])}
        family = types.SimpleNamespace(edges=lambda p: [['tok', 'out']], gt=lambda p, ids: [])
        return types.SimpleNamespace(make=make, CFG={'n_layer': 1, 'n_head': 2},
                                     texts=lambda *a: [([0, 1], 999)]*4,
                                     split_of=lambda *a: 'test_instance',
                                     FAMILIES={f: family for f in bank.FAMILIES})

    def test_no_intended_labels_and_exact_future_bound(self):
        spec = bank.specifications(self.protocol())[0]
        _, _, _, record, gold = bank.construct(self.fake(), spec)
        self.assertEqual(record['bank_role'], 'train')
        self.assertEqual(record['legacy_generator_hash_split'], 'test_instance')
        self.assertNotIn('999', json.dumps(record))
        self.assertEqual(record['teacher_bound']['worlds'], 28)
        self.assertEqual(record['teacher_bound']['forward_cap'], 69*28)
        self.assertEqual(record['preparation_native_forwards'], 0)
        self.assertFalse(gold['verified'])

    def test_all_construction_failures_preserved_no_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol = root/'protocol.json'
            protocol.write_text(json.dumps(self.protocol()))
            original = bank.PROTOCOL_SHA; bank.PROTOCOL_SHA = bank.sha(protocol)
            try:
                result = bank.generate(self.fake(True), protocol, root/'out', {})
                self.assertEqual(len(result['models']), 24)
                self.assertTrue(all(r['status']=='construction_error' for r in result['models']))
                with self.assertRaises(ValueError): bank.generate(self.fake(), protocol, root/'out', {})
            finally: bank.PROTOCOL_SHA = original


if __name__ == '__main__': unittest.main()
