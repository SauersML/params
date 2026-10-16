import json
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'method' / 'oracle'))
from lie_native_transfer import run


def test_full_calibration_then_all_physics_tiny(tmp_path):
    torch.manual_seed(7)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval().requires_grad_(False)
    manifest=json.loads((Path(__file__).parent/'lie_native_transfer_v1.json').read_text())
    target=[[0,3,1],[1,3,2]]
    manifest['target']=dict(model='random',revision='seed7',tokenizer_sha256='test',adapter_sha256='none')
    manifest['target_units']=target
    manifest['donors']={name:dict(boundary_ids=ids,remaining_ids=[20,23],native_values=[dict(unit=u,value=0.1) for u in target])
        for name,ids in [('digit5',[1,2,16,21]),('digit4',[1,7,16,21])]}
    manifest['cases']=[dict(case_id='tiny',group='truth_numeric_16',expected_truth_digit=5,expected_baseline_digit=5,
        prompt_ids=[1,3],prefix_ids=[16,21],boundary_ids=[1,3,16,21],answer_a_ids=[16,21,20,23],
        answer_b_ids=[16,21,19,23],remaining_a_ids=[20,23],remaining_b_ids=[19,23],intervention_position=3)]
    manifest['total_forward_example_cap']=18
    actual=[];h=model.model.register_forward_pre_hook(lambda m,a,k:actual.append(int(k['input_ids'].shape[0])),with_kwargs=True)
    try:result=run(model,None,manifest,tmp_path)
    finally:h.remove()
    assert result['status']=='complete'
    assert sum(actual)==result['counts']['forward_examples']==18
    assert result['counts']['backward_examples']==0
    row=result['rows'][0]
    assert row['forward_examples']==16
    assert len(row['effects'])==6
    assert row['effects']['target_digit4_patch']['destination_position']==3
    assert result['hooks_before']==result['hooks_current']==1  # only our own meter
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())
    packet=json.loads((tmp_path/'public_discovery_calibration.json').read_text())
    assert packet['forward_examples']==2
    assert 'rows' not in packet
    assert set(tuple(u) for u in packet['target_units']).isdisjoint(tuple(u) for u in packet['control_units'])
