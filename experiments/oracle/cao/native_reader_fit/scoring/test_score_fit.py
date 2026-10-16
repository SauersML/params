import copy,json,math
import pytest
from score_fit import validate,evaluate,numeric_sensitivity

def fixture():
 bindings=[];gold=[];pred=[];train=[];valid=[];original=[];mapping=[];baselines={}
 for i in range(32):
  cid=f'g{i//8}_{i%8}';group=f'g{i//8}';split='train' if i%8<6 else 'valid';delta=(-1 if i%2 else 1)*(i+1)/10
  bindings.append(dict(case_id=cid,group=group,split=split,delta_bits=delta));baseline={'a_logp_nats':-1.,'b_logp_nats':-1.};edit={'a_logp_nats':-1+delta*math.log(2),'b_logp_nats':-1.};gold.append(dict(case_id=cid,group=group,conditional_baseline=baseline,effects={'target_zero':edit}))
  o={'request_id':f'q{i}_true','finish_reason':'stop','completion':json.dumps({'delta_bits':0.,'reason':'short'})};response={'finish_reason':'stop','completion':json.dumps({'delta_bits':delta,'reason':'short'})};original.append(o);mapping.append({'case_id':cid,'question_id':f'q{i}'})
  pred.append(dict(case_id=cid,split=split,gold_delta_bits=delta,baseline=o,adapted=response,baseline_invalid=None,adapted_invalid=None,baseline_error=abs(delta),adapted_error=0.));(train if split=='train' else valid).append({'case_id':cid});baselines[cid]={'static_readout_delta_bits':0.,'regression_delta_bits':delta}
 return pred,{'bindings':bindings},{'status':'complete','rows':gold},train,valid,{'frozen_query_baselines':baselines},original,mapping

def test_split_scoring_resubstitution_and_ties():
 args=fixture();result=evaluate(*args,strict_parser=lambda r,raw:numeric_sensitivity(r))
 assert result['splits']['train']['modes']['strict']['predictors']['adapted']['capped_mae']==0
 assert result['splits']['valid']['modes']['strict']['predictors']['adapted']['rows']==8
 assert result['splits']['train']['modes']['strict']['predictors']['unchanged_zero']['sign']['non_tied_rows']==0

@pytest.mark.parametrize('damage',['omit','duplicate','split','dataset_split','gold'])
def test_incomplete_duplicate_mixed_and_gold_rejected(damage):
 args=list(copy.deepcopy(fixture()));pred,meta,gold,train,valid=args[:5]
 if damage=='omit':pred.pop()
 elif damage=='duplicate':pred[-1]=pred[0]
 elif damage=='split':pred[0]['split']='valid'
 elif damage=='dataset_split':train[0],valid[0]=valid[0],train[0]
 elif damage=='gold':pred[0]['gold_delta_bits']+=.1
 with pytest.raises(ValueError):validate(pred,meta,gold,train,valid)

def test_reason_only_sensitivity_keeps_other_contracts():
 row={'finish_reason':'stop','completion':json.dumps({'delta_bits':1,'reason':' '.join(['x']*41)})};assert numeric_sensitivity(row)[0]==1
 row['finish_reason']='length';assert numeric_sensitivity(row)[1]=='length_stop'
 for text in ('{"delta_bits":1,"delta_bits":2,"reason":"x"}','{"delta_bits":NaN,"reason":"x"}','{"delta_bits":true,"reason":"x"}'):
  assert numeric_sensitivity({'finish_reason':'stop','completion':text})[1]=='invalid_strict_json'

def test_optional_same_supervision_baseline_and_missing_case():
 args=fixture();same={b['case_id']:b['delta_bits'] for b in args[1]['bindings']};result=evaluate(*args,strict_parser=lambda r,raw:numeric_sensitivity(r),same_data_predictions=same)
 assert result['splits']['valid']['modes']['strict']['predictors']['same_data_ridge']['capped_mae']==0
 same.pop(next(iter(same)))
 with pytest.raises(ValueError,match='same-data baseline coverage'):evaluate(*args,strict_parser=lambda r,raw:numeric_sensitivity(r),same_data_predictions=same)
