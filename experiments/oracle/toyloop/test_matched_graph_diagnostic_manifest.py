import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import matched_graph_diagnostic_manifest as m
from matched_graph_permutation import permutations, map_edit


class DiagnosticManifestTests(unittest.TestCase):
    def fixture(self):
        item={'id':'x','weights_sha256':'fixed','inputs':[[0,1]],
              'edits':[{'kind':'joint','operations':[['head',0,0,0.],['neuron',0,1,.5]]}]}
        maps=permutations({'n_layer':1,'n_head':2},[3])
        plan=dict(item,maps=maps,mapped_edits=[map_edit(e,maps) for e in item['edits']])
        return item,plan

    def test_source_inputs_and_mixed_scales_cannot_drift(self):
        item,plan=self.fixture();m.validate_plan(item,plan)
        for field in ('inputs','weights_sha256'):
            changed=copy.deepcopy(plan);changed[field]=[] if field=='inputs' else 'other'
            with self.assertRaises(ValueError):m.validate_plan(item,changed)
        changed=copy.deepcopy(plan);changed['mapped_edits'][0]['operations'][1][-1]=0.
        with self.assertRaises(ValueError):m.validate_plan(item,changed)
        self.assertEqual(plan['mapped_edits'][0]['operations'][1][-1],.5)

    def test_inverse_permutation_requires_actual_bijection(self):
        item,plan=self.fixture()
        changed=copy.deepcopy(plan);changed['maps']['head:0']['old_to_new']=[-1,0]
        with self.assertRaises(ValueError):m.validate_plan(item,changed)
        changed=copy.deepcopy(plan);changed['maps']['head:0']={'new_to_old':[0,1],'old_to_new':[0,1]}
        with self.assertRaises(ValueError):m.validate_plan(item,changed)

    def test_archive_bound_prompts_reject_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'proof.tar.gz';out=root/'extracted';out.mkdir()
            (out/'x.prompt.txt').write_bytes(b'fixed weight facts')
            with tarfile.open(archive,'w:gz') as tar:
                info=tarfile.TarInfo('permutation/x.prompt.txt');info.size=18
                tar.addfile(info,io.BytesIO(b'fixed weight facts'))
            m.archive_binding(archive,m.sha(archive),out,['permutation/x.prompt.txt'])
            (out/'x.prompt.txt').write_bytes(b'changed weight facts')
            with self.assertRaises(ValueError):m.archive_binding(archive,m.sha(archive),out,['permutation/x.prompt.txt'])
            with self.assertRaises(ValueError):m.archive_binding(archive,'wrong',out,[])


if __name__=='__main__':unittest.main()
