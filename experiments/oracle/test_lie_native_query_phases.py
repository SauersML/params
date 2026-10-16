import copy,hashlib,json
from pathlib import Path
import pytest,torch
from transformers import Qwen3Config,Qwen3ForCausalLM
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'method' / 'oracle'))
from lie_native_transfer import run


def test_public_commit_gold_physical_counts_and_binding(tmp_path):
    torch.manual_seed(7)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval().requires_grad_(False)
    manifest=dict(target=dict(model='random',revision='seed7',tokenizer_sha256='test',adapter_sha256='none'),
        target_units=[[0,3,1],[1,3,2]],readout_token_ids=[20,19],public_forward_example_cap=64,
        gold_forward_example_cap=128,walltime_cap_seconds=600,cases=[])
    for i in range(32):
        manifest['cases'].append(dict(case_id=f'case{i}',group='synthetic',expected_truth_digit=4,
            expected_baseline_digit=5,prompt_ids=[1,2+i%2],prefix_ids=[16,21],boundary_ids=[1,2+i%2,16,21],
            answer_a_ids=[16,21,20,23],answer_b_ids=[16,21,19,23],remaining_a_ids=[20,23],remaining_b_ids=[19,23],intervention_position=3))
    manifest_sha=hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    actual=[];h=model.model.register_forward_pre_hook(lambda m,a,k:actual.append(int(k['input_ids'].shape[0])),with_kwargs=True)
    try:
        public=run(model,None,manifest,tmp_path/'public',phase='public',manifest_sha256=manifest_sha)
        assert len(public['rows'])==32 and public['counts']['forward_examples']==sum(actual)==64
        assert public['hooks_before']==public['hooks_current']==1
        assert all('effects' not in r and 'baseline' not in r and 'a_logp_nats' not in r for r in public['rows'])
        responses=tmp_path/'predictions';responses.mkdir()
        data=''.join(json.dumps(dict(request_id=f'q{i:02d}_{arm}',completion='{"delta_bits":0,"reason":"synthetic"}'))+'\n' for i in range(32) for arm in ('true','absent','mismatched'))
        (responses/'responses.jsonl').write_text(data);anchor=hashlib.sha256(data.encode()).hexdigest()
        path=tmp_path/'public/public_observations.json';public_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        receipt=dict(requests=96,responses_sha256=anchor,panel_sha256=manifest_sha,public_observations_sha256=public_sha)
        rp=responses/'prediction_commit.json';rp.write_text(json.dumps(receipt))
        with pytest.raises(ValueError,match='commit.*coverage'):
            run(model,None,manifest,tmp_path/'bad',phase='gold',public_path=path,prediction_receipt=rp,prediction_commit_sha='0'*64,manifest_sha256=manifest_sha)
        assert sum(actual)==64
        gold=run(model,None,manifest,tmp_path/'gold',phase='gold',public_path=path,prediction_receipt=rp,prediction_commit_sha=anchor,manifest_sha256=manifest_sha)
        assert gold['counts']['forward_examples']==128 and sum(actual)==192
        assert gold['public_state_replay_status']=='comparable'
        assert len(gold['rows'])==32 and gold['hooks_before']==gold['hooks_current']==1
        first=gold['rows'][0]
        assert first['public_activation_max_abs_replay_error']==0
        assert set(first['effects'])=={'target_zero'}
        with torch.no_grad():
            ids=torch.tensor([[1,2,16,21,20,23]])
            lp=model(ids,use_cache=False).logits[0,1:5].float().log_softmax(-1)
            direct=float(lp.gather(-1,torch.tensor([16,21,20,23])[:,None]).sum())
        assert first['full_answer_baseline']['a_logp_nats']==pytest.approx(direct,abs=2e-6)
        assert first['baseline_metadata']['prefix_top_ids']==lp[:2].argmax(-1).tolist()
        assert first['baseline_metadata']['digit_top_id']==int(lp[2].argmax())
        # A committed but numerically different public state is flagged, while
        # all zero outcomes remain available; no retry or replacement occurs.
        changed=json.loads(path.read_text());changed['rows'][0]['neurons'][0]['activation']+=10
        path.write_text(json.dumps(changed));receipt['public_observations_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        rp.write_text(json.dumps(receipt))
        drift=run(model,None,manifest,tmp_path/'drift',phase='gold',public_path=path,prediction_receipt=rp,prediction_commit_sha=anchor,manifest_sha256=manifest_sha)
        assert drift['public_state_replay_status']=='noncomparable_public_state'
        assert not drift['rows'][0]['public_state_replay_comparable']
        assert len(drift['rows'])==32 and drift['counts']['forward_examples']==128

    finally:h.remove()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
