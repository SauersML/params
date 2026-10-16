import json
import pytest
from pathlib import Path
from exact_data import ExactDataset
D=Path(__file__).parent/'exact_data'
@pytest.fixture(autouse=True)
def optional_data(request):
    if request.node.name=='test_completion_only_offset':return
    if not (D/'provenance.json').is_file():pytest.skip('optional local prepared data absent')

@pytest.fixture(scope='module')
def tokenizer():
    transformers=pytest.importorskip('transformers')
    path=Path('/Users/user/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46')
    if not (path/'tokenizer.json').is_file():pytest.skip('optional cached local tokenizer absent')
    return transformers.AutoTokenizer.from_pretrained(str(path),local_files_only=True)
def test_completion_only_offset():
    row={'prompt_ids':[1,2,3],'completion_ids':[4,5]}
    ids,offset=ExactDataset([row]).process(row)
    assert ids==[1,2,3,4,5] and offset==3
    # installed trainer predicts token[t] from token[t-1], steps>=offset
    assert [t for t in range(1,len(ids)) if t>=offset]==[3,4]

def test_exact_original_prompts_and_disjoint_split():
    original=Path('/Users/user/mpd-data/scratch/root/native_query_reader_local_v1/frozen')
    if not (original/'requests.jsonl').is_file():pytest.skip('optional original requests absent')
    req={r['question_id']:r for r in map(json.loads,(original/'requests.jsonl').read_text().splitlines()) if r['arm']=='true'}
    mapping={r['case_id']:r['question_id'] for r in json.load(open(original/'mapping.json'))}
    datasets={s:[json.loads(x) for x in (D/(s+'.jsonl')).read_text().splitlines()] for s in ('train','valid')}
    assert list(map(len,datasets.values()))==[24,8]
    assert not {x['case_id'] for x in datasets['train']}&{x['case_id'] for x in datasets['valid']}
    for rows in datasets.values():
        for r in rows:
            assert r['prompt_ids']==req[mapping[r['case_id']]]['token_ids']
            ids,offset=ExactDataset(rows).process(r)
            assert ids[offset:]==r['completion_ids']
            assert len(ids)<=6144

def test_smoke_is_only_two_steps_and_same_dataset():
    smoke=json.load(open(D/'smoke_config.json'));fit=json.load(open(D/'config.json'))
    assert smoke['iters']==2 and fit['iters']==80
    assert smoke['data']==fit['data'] and smoke['lora_parameters']==fit['lora_parameters']
    assert smoke['adapter_path']!=fit['adapter_path']

def test_dataset_hashes_and_base_provenance():
    import hashlib
    m=json.load(open(D/'provenance.json'))
    for s,v in m['dataset_sha256'].items():assert hashlib.sha256((D/(s+'.jsonl')).read_bytes()).hexdigest()==v
    assert len(m['immutable_base_files_sha256'])>=5
    assert m['exact_original_true_prompt_ids'] is True
    assert all(len(v)==64 for v in m['immutable_base_files_sha256'].values())

def test_exact_mask_has_no_prompt_targets():
    for split in ('train','valid'):
        for row in map(json.loads,(D/(split+'.jsonl')).read_text().splitlines()):
            ids,offset=ExactDataset([row]).process(row)
            active=[ids[step] for step in range(1,len(ids)) if step>=offset]
            assert active==row['completion_ids']

def test_native_evidence_and_labels_not_mismatch_training():
    m=json.load(open(D/'provenance.json'))
    assert len(m['bindings'])==32 and all(b['split'] in ('train','valid') for b in m['bindings'])
    assert len({b['case_id'] for b in m['bindings']})==32
    assert len({b['group'] for b in m['bindings']})==4

def test_config_projection_keys_exist_in_installed_architecture():
    c=json.load(open(D/'config.json'))
    source=Path('/Users/user/mpd-data/venv/lib/python3.12/site-packages/mlx_lm/models')
    if not (source/'qwen3_next.py').is_file():pytest.skip('optional installed MLX architecture source absent')
    attention=(source/'qwen3_next.py').read_text();hybrid=(source/'qwen3_5.py').read_text()
    assert 'Qwen3NextAttention as Attention' in hybrid
    for key in c['lora_parameters']['keys']:
        attr=key.split('.')[-1]
        assert 'self.'+attr+' =' in (attention if key.startswith('self_attn') else hybrid)

import pytest
@pytest.mark.parametrize('i',range(32))
def test_all_exact_gold_label_serializations(i,tokenizer):
    # CPU tokenizers only; no model or MLX import.
    m=json.load(open(D/'provenance.json'));b=m['bindings'][i]
    tok=tokenizer
    row=next(r for r in map(json.loads,(D/(b['split']+'.jsonl')).read_text().splitlines()) if r['case_id']==b['case_id'])
    obj=json.loads(tok.decode(row['completion_ids'][:-1]))
    assert obj['delta_bits']==round(b['delta_bits'],3)
    assert abs(obj['delta_bits']-b['delta_bits'])<=.00050000001
    assert obj['reason']=='Predicted joint-zero change in conditional A-versus-B log odds.'
