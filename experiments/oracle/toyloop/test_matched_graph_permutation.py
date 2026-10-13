import unittest
import matched_graph_permutation as m


class PermutationTests(unittest.TestCase):
    def test_fixed_bijections_nonidentity_and_edit_transport(self):
        cfg = {'n_layer': 2, 'n_head': 4}
        maps = m.permutations(cfg, [8, 8])
        self.assertEqual(maps, m.permutations(cfg, [8, 8]))
        for value in maps.values():
            order = value['new_to_old']
            self.assertEqual(sorted(order), list(range(len(order))))
            self.assertNotEqual(order, list(range(len(order))))
            for old, new in enumerate(value['old_to_new']):
                self.assertEqual(order[new], old)
        edit = {'kind': 'joint', 'operations': [['head', 0, 1, 0.], ['neuron', 1, 3, .5]]}
        mapped = m.map_edit(edit, maps)
        self.assertEqual(mapped['kind'], 'joint')
        self.assertEqual([r[-1] for r in mapped['operations']], [0., .5])
        self.assertEqual(edit['operations'][0][2], 1)

    def test_weight_transport_and_edited_commutation(self):
        import torch
        cfg = {'n_layer': 1, 'n_head': 2, 'n_embd': 4}
        sd = {'wte.weight': torch.arange(12.).reshape(3, 4)}
        for port in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            sd['h.0.attn.'+port+'.weight'] = torch.arange(16.).reshape(4, 4)
        sd['h.0.mlp.c_fc.weight'] = torch.arange(12.).reshape(3, 4)
        sd['h.0.mlp.down_proj.weight'] = torch.arange(12.).reshape(4, 3)
        maps = m.permutations(cfg, [3])
        result = m.permute_weights(sd, cfg, maps)
        self.assertTrue(torch.equal(result['wte.weight'], sd['wte.weight']))
        edited = {k: v.clone() for k, v in sd.items()}
        edited['h.0.attn.o_proj.weight'][:, :2] *= .5
        edited['h.0.mlp.down_proj.weight'][:, 1] *= 0
        moved = m.permute_weights(edited, cfg, maps)
        head = maps['head:0']['old_to_new'][0]
        neuron = maps['neuron:0']['old_to_new'][1]
        result['h.0.attn.o_proj.weight'][:, 2*head:2*head+2] *= .5
        result['h.0.mlp.down_proj.weight'][:, neuron] *= 0
        self.assertTrue(all(torch.equal(moved[k], result[k]) for k in sd))
        with self.assertRaises(ValueError):
            m.permute_weights(dict(sd, **{'h.0.mlp.c_fc.bias': torch.zeros(3)}), cfg, maps)

    def test_singleton_cannot_claim_nonidentity(self):
        with self.assertRaises(ValueError):
            m.permutations({'n_layer': 1, 'n_head': 1}, [4])


if __name__ == '__main__':
    unittest.main()
