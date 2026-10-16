import copy
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'method' / 'oracle'))
from lie_native_discovery import run

def test_random_preflight_counts_cleanup(tmp_path):
    torch.manual_seed(4)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
    model.requires_grad_(False)
    case=dict(case_id='random',clean_prompt_ids=[1,2,3],corrupt_prompt_ids=[1,7,3],
        full_false_ids=[16,21,20,23],full_true_ids=[16,21,19,23],
        clean_boundary_ids=[1,2,3,16,21],corrupt_boundary_ids=[1,7,3,16,21],false_digit_id=20,true_digit_id=19)
    manifest=dict(target=dict(model='random',revision='seed4',tokenizer_sha256='test',adapter_sha256='none'),
        preflight_forward_example_cap=160,search_forward_example_cap=4096,walltime_cap_seconds=600,
        rows=8,cases=[case],min_gap_bits=1.,noop_atol_bits=1e-6)
    result=run(model,None,manifest,tmp_path)
    assert result['counts']==dict(forward_calls=8,forward_examples=36)
    assert result['hooks_before']==result['hooks_current']==0
    assert len(result['preflight'])==1 and not result['preflight'][0]['eligible_for_discovery']
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    small=copy.deepcopy(manifest);small['preflight_forward_example_cap']=3
    result=run(model,None,small,tmp_path/'small')
    assert result['counts']['forward_examples']==3
    assert result['searches'][0]['status']=='preflight_budget_stopped_incomplete'
    assert result['hooks_before']==result['hooks_current']==0
