"""No-model, no-API failure guards for the new paid entrypoint."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import compositional_trial as c


class TrialTests(unittest.TestCase):
    def test_same_length_eval_replacement_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'original24_questions.jsonl'
            path.write_text('original prompt')
            expected=c.digest(path)
            path.write_text('mutated! prompt')
            self.assertEqual(len('original prompt'),len('mutated! prompt'))
            with mock.patch.object(c,'EVAL_FILES',{'original24':expected}), mock.patch.object(c,'HELPERS',{}):
                with self.assertRaisesRegex(ValueError,'stratum bytes'):
                    c.validate_files(root)

    def test_bad_root_fails_before_service_and_sdk(self):
        seen=[]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict('os.environ',{'TINKER_PROJECT_ID':c.PROJECT}):
                with self.assertRaises(FileNotFoundError):
                    c.run(Path(directory)/'missing','missing',Path(directory)/'paid','.68438213',service_factory=lambda **kw:seen.append(kw))
            self.assertEqual(seen,[])
            self.assertFalse((Path(directory)/'paid').exists())

    def test_insufficient_exact_decimal_cap_fails_before_preflight(self):
        with mock.patch.object(c,'prepared',side_effect=AssertionError('not reached')):
            with self.assertRaises(ValueError):
                c.run('missing','missing','missing','.684382129',service_factory=lambda **kw:None)

    def test_existing_output_never_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(c,'prepared',side_effect=AssertionError('not reached')):
                with self.assertRaises(ValueError):
                    c.run('missing','missing',directory,'1',service_factory=lambda **kw:None)


if __name__=='__main__':
    unittest.main()


class PartialAccounting(unittest.TestCase):
    def test_failed_forward_drains_other_and_retains_unresolved_reservation(self):
        import json
        import sys
        from types import SimpleNamespace as NS
        calls=[]

        class Future:
            def __init__(self,name,fail=False):
                self.name,self.fail=name,fail

            def result(self):
                calls.append(self.name)
                if self.fail:
                    raise RuntimeError('uncertain first stream')
                return NS(path='mock-init')

        class Client:
            def __init__(self,name):
                self.name=name

            def get_info(self):
                return NS(model_dump=lambda **kwargs:{'is_lora':True,'lora_rank':32,'model_data':{'model_name':'Qwen/Qwen3-8B'}})

            def forward_backward(self,*args,**kwargs):
                return Future(self.name,fail=self.name=='single')

            def optim_step(self,*args,**kwargs):
                return Future(self.name+'_optimizer')

        remaining=iter(['single','two'])
        service=NS(create_lora_training_client=lambda **kwargs:NS(save_state=lambda name:Future('initial')),
                   create_training_client_from_state_with_optimizer=lambda path:Client(next(remaining)))
        context={'source_hashes':{},'upper':c.EXPECTED_UPPER,'data':{arm:[NS(model_input=NS(length=100))] for arm in c.ARMS},'batches':[[0]]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'complete.json').write_text('{}')
            tokenizer=root/'tokenizer'; tokenizer.mkdir()
            with mock.patch.dict('os.environ',{'TINKER_PROJECT_ID':c.PROJECT}), mock.patch.object(c,'prepared',return_value=context), mock.patch.dict(sys.modules,{'tinker':NS(AdamParams=lambda **kwargs:kwargs)}):
                with self.assertRaises(RuntimeError):
                    c.run(root,tokenizer,root/'paid','1',service_factory=lambda **kwargs:service)
            self.assertIn('single',calls)
            self.assertIn('two',calls)
            receipts=c.rows(root/'paid/spend.jsonl')
            self.assertEqual(len(receipts),1)
            self.assertEqual(receipts[0]['usd'],'0.000044')
            self.assertEqual(receipts[0]['remaining_reserved_usd'],'0.000044')
            self.assertFalse((root/'paid/complete.json').exists())
