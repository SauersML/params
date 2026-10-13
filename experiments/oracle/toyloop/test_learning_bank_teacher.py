import json
from pathlib import Path
import tempfile
import unittest
import learning_bank_teacher as driver
from learning_model_bank import GENERATOR_COMMIT, FAMILIES


class ContractTest(unittest.TestCase):
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
