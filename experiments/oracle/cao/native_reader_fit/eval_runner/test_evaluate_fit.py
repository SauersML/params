import json,os,sys,types
import pytest
import evaluate_fit as e

def test_macos_physical_footprint_guard():
    if sys.platform!='darwin':pytest.skip('Mac-specific physical footprint API')
    assert e.physical_bytes(os.getpid())>0

def fixture(tmp_path,monkeypatch):
    d=tmp_path/'data';f=tmp_path/'frozen';m=tmp_path/'model';a=tmp_path/'adapter'
    for p in (d,f,m,a):p.mkdir()
    write=lambda p,v:p.write_text(json.dumps(v))
    (m/'weights').write_bytes(b'base');(a/'adapters.safetensors').write_bytes(b'adapter');write(a/'adapter_config.json',{})
    baseline=tmp_path/'baseline';baseline.write_text('');primary=tmp_path/'primary';primary.write_text('')
    original=e.sha
    monkeypatch.setattr(e,'sha',lambda p: '65997e8da14092ff6822e5ad64c9de8f247a7e5721d275d161f37e86d9b9b509' if str(p)==str(baseline) else 'e0b6b4580e2bcd4fe67b6fef06e509696c9adaee1bfee9ec42fb8e0b80ec9650' if str(p)==str(primary) else original(p))
    rows=[dict(case_id=f'c{i}',prompt_ids=[1,i+2],completion_ids=[4,5]) for i in range(32)]
    for name,part in [('train',rows[:24]),('valid',rows[24:])]:
        (d/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in part))
    (f/'requests.jsonl').write_text(''.join(json.dumps(dict(question_id=f'q{i}',arm='true',token_ids=r['prompt_ids']))+'\n' for i,r in enumerate(rows)))
    write(f/'mapping.json',[dict(case_id=f'c{i}',question_id=f'q{i}') for i in range(32)])
    write(f/'manifest.json',dict(requests_sha256=e.sha(f/'requests.jsonl'),mapping_sha256=e.sha(f/'mapping.json'),model_revision=m.name,tokenizer_sha256={}))
    write(d/'provenance.json',dict(dataset_sha256={s:e.sha(d/(s+'.jsonl')) for s in ('train','valid')},immutable_base_files_sha256={'weights':e.sha(m/'weights')}))
    config=tmp_path/'config';write(config,dict(data=str(d),model=str(m),adapter_path=str(a)))
    args=types.SimpleNamespace(config=str(config),frozen=str(f),baseline_responses=str(baseline),primary_source=str(primary),adapter_sha256=e.sha(a/'adapters.safetensors'))
    return args,d,m,a

def test_prepare_validates_all32_exact_prompts_and_weights(tmp_path,monkeypatch):
    args,d,m,a=fixture(tmp_path,monkeypatch)
    cfg,meta,rows,mapping,base,adapter=e.prepare(args)
    assert len(rows)==32 and len(mapping)==32
    assert len(base)==1 and len(adapter)==2

@pytest.mark.parametrize('which',['base','adapter','prompt'])
def test_modified_inputs_rejected(tmp_path,monkeypatch,which):
    args,d,m,a=fixture(tmp_path,monkeypatch)
    if which=='base':(m/'weights').write_bytes(b'changed')
    elif which=='adapter':(a/'adapters.safetensors').write_bytes(b'changed')
    else:
        p=d/'train.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['prompt_ids']=[999];p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        meta=json.load(open(d/'provenance.json'));meta['dataset_sha256']['train']=e.sha(p);(d/'provenance.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError):e.prepare(args)
