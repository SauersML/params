import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import matched_diagnostic_decode as d
import matched_diagnostic_sample as sample
from test_matched_diagnostic_sample import prepared


class DiagnosticDecodeTests(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        p=prepared();ids=[r['id'] for r in p['models']['Qwen/Qwen3-8B']['inputs']]
        text='  fixed weight facts  '
        panel=[{'id':i,'stratum':'diagnostic_'+('permuted' if i.endswith('__perm') else 'original'),
                'condition':'permuted' if i.endswith('__perm') else 'original','original_id':i.removesuffix('__perm'),
                'prompt':text,'prompt_sha256':d.sha_text(text)} for i in ids]
        panel_path=root/'panel';panel_path.write_text(''.join(json.dumps(r)+'\n' for r in panel))
        manifest=root/'manifest';manifest.write_text(json.dumps({'arms':sample.POLICIES,'panel':panel}))
        p['source_panel_sha256']=d.sha(panel_path);p['scoring_manifest_sha256']=d.sha(manifest)
        for model in p['models'].values():
            for inp in model['inputs']:inp['prompt_sha256']=d.sha_text(text)
        prep=root/'prepared';prep.write_text(json.dumps(p))
        start=root/'start.jsonl';start.write_text(json.dumps({'prepared_sha256':d.sha(prep),'project':sample.PROJECT})+'\n')
        raw={}
        for arm in sample.POLICIES:
            path=root/(arm+'_raw.jsonl');raw[arm]=path
            path.write_text(json.dumps({'arm':arm,'id':ids[0],'sample':0,'prompt_sha256':d.sha_text(text),
                'prompt_rendered_sha256':d.digest([1,2]),'sequences':[{'token_ids':[7,3],'finish':'length' if arm=='base' else 'stop'}]})+'\n')
        contract=root/'raw_contract';contract.write_text(json.dumps({'prepared_sha256':d.sha(prep),'manifest_sha256':d.sha(manifest),
            'files':{'start.jsonl':d.sha(start)},'arms':{a:{'path':v.name,'sha256':d.sha(v)} for a,v in raw.items()}}))
        args=SimpleNamespace(prepared=prep,prepared_sha256=d.sha(prep),panel=panel_path,panel_sha256=d.sha(panel_path),
            manifest=manifest,manifest_sha256=d.sha(manifest),raw_contract=contract,raw_contract_sha256=d.sha(contract),out=root/'decoded')
        class Renderer:
            def build_generation_prompt(self,messages):return SimpleNamespace(to_ints=lambda:[1,2])
            def get_stop_sequences(self):return [3]
            def parse_response(self,tokens):
                tokens.pop()
                return {'role':'assistant','content':'  ```mermaid\nflowchart LR\nL0H0 --> out\n```\nEnglish: retained.  '},False
        bundles={name:(Renderer(),lambda message:message['content']) for name in sample.TOKENIZERS}
        return args,raw,bundles

    def test_preserves_length_tokens_full_text_and_missing_rows(self):
        args,raw,bundles=self.fixture();receipt=d.decode(args,bundles)
        base=json.loads((args.out/'base.jsonl').read_text())
        self.assertEqual(base['finish'],'length');self.assertEqual(base['token_ids'],[7,3])
        self.assertNotIn('parse_error',base)
        self.assertTrue(base['completion'].startswith('  ```'))
        self.assertTrue(base['completion'].endswith('retained.  '))
        self.assertEqual(len(receipt['missing_first_responses']['base']),15)
        point=json.loads((args.out/'point.jsonl').read_text())
        self.assertEqual(point['parse_error'],'renderer_malformed_stop_sequence')

    def test_raw_mutation_rejected_before_parser(self):
        args,raw,bundles=self.fixture();raw['base'].write_text('changed')
        with self.assertRaises(ValueError):d.decode(args,bundles)
        self.assertFalse(args.out.exists())

    def test_duplicate_first_response_rejected(self):
        args,raw,bundles=self.fixture();raw['base'].write_text(raw['base'].read_text()*2)
        c=json.loads(args.raw_contract.read_text());c['arms']['base']['sha256']=d.sha(raw['base'])
        args.raw_contract.write_text(json.dumps(c));args.raw_contract_sha256=d.sha(args.raw_contract)
        with self.assertRaises(ValueError):d.decode(args,bundles)


if __name__=='__main__':unittest.main()
