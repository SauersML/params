import json
from pathlib import Path
import tempfile
import unittest
import learning_bank_teacher as driver
from learning_model_bank import GENERATOR_COMMIT, FAMILIES


class ContractTest(unittest.TestCase):
    def test_fresh_bank_two_disjoint_four_case_workers_and_role_guard(self):
        from learning_model_bank import specifications
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            specs=[{'case':f'{f}_{3300000+100*i}','family':f,'seed':3300000+100*i,
                    'split':'train','context_seed':(3300000+100*i)*7+3} for i,f in enumerate(FAMILIES)]
            rows=[{**s,'status':'construction_error'} for s in specs]
            manifest={'models':rows,'generator_commit':GENERATOR_COMMIT,
                      'profile':'fresh_train_bank_v1','specifications':specs}
            (root/'manifest.json').write_text(json.dumps(manifest))
            digest=driver.sha(root/'manifest.json')
            a=driver.validate_bank(root,digest,0)[1];b=driver.validate_bank(root,digest,1)[1]
            self.assertEqual([r['case'] for r in a+b],[r['case'] for r in specs])
            self.assertEqual(len(a),4);self.assertEqual(len(b),4)
            rows[0]['split']='evaluation'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):driver.validate_bank(root,driver.sha(root/'manifest.json'),0)
    def test_continuation_preserves_completed_cases_and_all_old_forwards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);digest=self.fixture(root)
            manifest,rows=driver.validate_bank(root,digest,1)
            names=['gate_3100501','induction_3100600','induction_3100601','redundant_3100700','redundant_3100701']
            all_train={r['case'] for r in manifest['models'] if r['split']=='train'}
            audit={'verified':[{'id':i} for i in all_train-set(names)],'total_old_forward_interval':[36860,36861]}
            receipt={'new_cases':names,'max_new_forwards':5*5520,'old_forward_interval':[36860,36861]}
            got=driver.continuation_rows(manifest,rows,receipt,audit)
            self.assertEqual([r['case'] for r in got],names)
            receipt['old_forward_interval']=[0,0]
            with self.assertRaises(ValueError):driver.continuation_rows(manifest,rows,receipt,audit)
            receipt['old_forward_interval']=audit['total_old_forward_interval'];receipt['new_cases']=names[::-1]
            with self.assertRaises(ValueError):driver.continuation_rows(manifest,rows,receipt,audit)
    def test_append_journal_preserves_receipts_and_final_legacy_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);journal=root/'evidence.jsonl'
            first={'file':'a.safetensors','sha256':'first','index':0}
            second={'file':'b.safetensors','sha256':'second','index':1}
            driver.append_evidence(journal,first);prefix=journal.read_bytes()
            driver.append_evidence(journal,second)
            self.assertTrue(journal.read_bytes().startswith(prefix))
            self.assertFalse((root/'evidence.json').exists())
            driver.finalize_evidence(root)
            self.assertEqual(json.loads((root/'evidence.json').read_text()),[first,second])
            self.assertEqual(journal.read_bytes().count(b'\n'),2)
            with journal.open('a') as handle:handle.write('{incomplete')
            before=(root/'evidence.json').read_bytes()
            with self.assertRaises(json.JSONDecodeError):driver.finalize_evidence(root)
            self.assertEqual((root/'evidence.json').read_bytes(),before)
    def fixture(self, root):
        rows=[]
        for split,base,variants in [('train',3100000,(0,1)),('evaluation',3200000,(0,))]:
            for index,family in enumerate(FAMILIES):
                for variant in variants:
                    case=f'{family}_{base+100*index+variant}'; dest=root/case;dest.mkdir()
                    (dest/'record.json').write_text('{}')
                    rows.append({'case':case,'split':split,'status':'constructed_unmeasured',
                                 'files':{'record.json':driver.sha(dest/'record.json')},
                                 'teacher_bound':{'worlds':80,'max_attempts':64,'forward_cap':5520}})
        (root/'manifest.json').write_text(json.dumps({'models':rows,'generator_commit':GENERATOR_COMMIT}))
        return driver.sha(root/'manifest.json')

    def test_exact_disjoint_eight_train_workers_no_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);digest=self.fixture(root)
            _,a=driver.validate_bank(root,digest,0);_,b=driver.validate_bank(root,digest,1)
            self.assertEqual(len(a),8);self.assertEqual(len(b),8)
            self.assertTrue(all(r['split']=='train' for r in a+b))
            self.assertFalse({r['case'] for r in a}&{r['case'] for r in b})

    def test_mutated_file_or_bound_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);digest=self.fixture(root)
            manifest=json.loads((root/'manifest.json').read_text())
            case=manifest['models'][0]['case'];(root/case/'record.json').write_text('changed')
            with self.assertRaises(ValueError):driver.validate_bank(root,digest,0)
            (root/case/'record.json').write_text('{}')
            manifest['models'][0]['teacher_bound']['forward_cap']=1
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):driver.validate_bank(root,driver.sha(root/'manifest.json'),0)

    def test_construction_failure_retained_and_worker_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            manifest=json.loads((root/'manifest.json').read_text())
            manifest['models'][0]['status']='construction_error'
            (root/'manifest.json').write_text(json.dumps(manifest));digest=driver.sha(root/'manifest.json')
            self.assertEqual(driver.validate_bank(root,digest,0)[1][0]['status'],'construction_error')
            with self.assertRaises(ValueError):driver.validate_bank(root,digest,2)


if __name__=='__main__':unittest.main()
