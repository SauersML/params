import json,math
from pathlib import Path
import numpy as np
from ridge import features,predict
HERE=Path(__file__).parent

def test_train_only_loo_and_final_frozen_predictions():
 artifact=json.loads((HERE/'frozen.json').read_text());model=artifact['model'];fixture=json.loads((HERE/'training_fixture.json').read_text());by={r['case_id']:r for r in fixture['rows']};train=[b for b in fixture['provenance_bindings'] if b['split']=='train'];x=np.array(features(model,[by[b['case_id']] for b in train]));y=np.array([b['delta_bits'] for b in train]);assert len(train)==24
 for result in artifact['selection']:
  assert result['case_ids']==[b['case_id'] for b in train]
  independent=[]
  for held in range(24):
   keep=np.arange(24)!=held;mean=x[keep].mean(0);scale=x[keep].std(0);scale[scale==0]=1;z=(x[keep]-mean)/scale;target=y[keep];center=target.mean();coef=np.linalg.solve(z.T@z+result['alpha']*np.eye(16),z.T@(target-center));independent.append(float(center+((x[held]-mean)/scale)@coef))
  np.testing.assert_allclose(independent,result['predictions_delta_bits'],rtol=1e-13,atol=1e-13)
 assert model['alpha']==min(artifact['selection'],key=lambda r:r['train_loo_mae_bits'])['alpha']
 for row in artifact['predictions']:assert predict(model,[by[row['case_id']]])[0]==row['prediction_delta_bits'] and math.isfinite(row['prediction_delta_bits'])
