import copy
import pytest
import claim_affine_mechanism as m

@pytest.fixture
def payload():
    case={'component_index':3268,'site':'h.3.mlp.down_proj','token_position':366,
          'execution_text':' A<|endoftext|> B','raw_scalar_x_dot_V':2.,'r_squared_norm':4.,
          'r_dot_U':-1.,'U_squared_norm':3.,'residual_width':768}
    return {'cases':{'text6008':case},'mechanism':{},'model_content_sha256':{'target':'a'*64}}

def cert(p):
    return {"validated":True,"physical_points":18,"payload_sha256":m.digest(p)}

def response(p):
    return {'prose':'Untrusted writer text.', 'mechanism':m.expected_sidecar(p,'text6008')}

def test_supported_renderer_does_not_certify_prose(payload):
    r=response(payload);r['prose']='This arbitrary prose is not checked.'
    assert m.verify(payload,'text6008',r,certificate=cert(payload))['free_prose_verified'] is False
    text=m.render_verified(payload,'text6008',r,certificate=cert(payload))
    assert 'base+(k-1)slope' in text and 'sqrt(mean(r(k)^2)' in text
    assert 'arbitrary prose' not in text and 'do not certify a full circuit' in text

@pytest.mark.parametrize('field,value',[
    ('numerator','base_at_1_plus_k_slope'),
    ('normalization','none'),
    ('scope','all_inputs_full_circuit'),
    ('residual','r_at_1_plus_k_a_U'),
])
def test_wrong_mechanism_rejected(payload,field,value):
    r=response(payload);r['mechanism']['claims'][field]=value
    with pytest.raises(ValueError):m.verify(payload,'text6008',r,certificate=cert(payload))

@pytest.mark.parametrize('field,value',[('component_index',1),('site','h.2.mlp.down_proj'),('token_position',365)])
def test_binding_rejected(payload,field,value):
    r=response(payload);r['mechanism']['binding'][field]=value
    with pytest.raises(ValueError):m.verify(payload,'text6008',r,certificate=cert(payload))

def test_missing_rms_and_unknown_claim_fail(payload):
    for change in ('missing','extra'):
        r=response(payload)
        if change=='missing':del r['mechanism']['claims']['normalization']
        else:r['mechanism']['claims']['causal_english']='proved'
        with pytest.raises(ValueError):m.verify(payload,'text6008',r,certificate=cert(payload))

def test_nonfinite_boolean_and_changed_assets_rejected(payload):
    for value in (float('nan'),float('inf'),True):
        r=response(payload);r['mechanism']['geometry']['raw_scalar_x_dot_V']=value
        with pytest.raises(ValueError):m.verify(payload,'text6008',r,certificate=cert(payload))
    r=response(payload);p=copy.deepcopy(payload);p['model_content_sha256']['target']='b'*64
    with pytest.raises(ValueError):m.verify(p,'text6008',r,certificate=cert(payload))

def test_prompt_not_edited_measurements(payload):
    text=m.writer_prompt(payload,'text6008')
    assert 'no edited-forward observations' in text
    assert 'verification.jsonl' not in text

def test_formula_not_only_enum_matching():
    import numpy as np
    # Nonunit U and nonzero baseline: wrong origin or omitting the changing
    # RMS makes physically different distributions even with the same argmax.
    r=np.array([2.,-1.]);u=np.array([3.,2.]);readout=np.array([[1.,0.,-2.],[0.,1.,.5]])
    a=.7;k=.5;eps=1e-6;base=r@readout;slope=a*u@readout
    z=(r+(k-1)*a*u)@readout
    logits=z/np.sqrt(np.mean((r+(k-1)*a*u)**2)+eps)
    logp=logits-np.log(np.exp(logits).sum())
    assert m.numerical_check(base,slope,r,u,a,2,eps,k,z,logp)['literal_logp_max_error']<1e-12
    with pytest.raises(ValueError):m.numerical_check(base,slope,r,u,a,2,eps,k,z,logp,anchor=0.)
    with pytest.raises(ValueError):m.numerical_check(base,slope,r,u,a,2,eps,k,z,logp,use_rms=False)

def test_certificate_required_and_bound(payload):
    with pytest.raises(ValueError):m.verify(payload,'text6008',response(payload),certificate={'validated':False})

def test_first_response_finish_and_coverage(payload):
    import json
    row={'request_id':'mechanism_writer|text6008','completion':json.dumps(response(payload)),'finish':'StopReason.length'}
    result=m.evaluate_responses(payload,cert(payload),[row])
    assert result[0]['typed_claims_verified'] is False
    assert result[1]['invalid_reason']=='missing'
    row['finish']='StopReason.stop'
    assert m.evaluate_responses(payload,cert(payload),[row])[0]['typed_claims_verified']
    with pytest.raises(ValueError):m.evaluate_responses(payload,cert(payload),[row,row])
    row['parse_error']='bad'
    assert not m.evaluate_responses(payload,cert(payload),[row])[0]['typed_claims_verified']

@pytest.mark.parametrize('name,value',[
    ('scalar',float('nan')),('k',float('inf')),('anchor',float('-inf')),
    ('epsilon',float('nan')),('epsilon',float('inf')),('epsilon',0.),
    ('width',0),('width',True),('width',2.0),
])
def test_nonfinite_formula_parameters_fail_closed(name,value):
    args={'base':[0.,1.],'slope':[1.,2.],'residual':[1.,2.],'unit':[1.,0.],
          'scalar':1.,'width':2,'epsilon':1e-6,'k':1.,'saved_numerator':[0.,1.],
          'saved_logp':[-1.,-1.]}
    args[name]=value
    with pytest.raises(ValueError):m.numerical_check(**args)

@pytest.mark.parametrize('name,value',[
    ('slope',[[1.,2.]]),('unit',[1.]),('saved_logp',[[0.,1.]]),
    ('base',[float('nan'),1.]),('saved_numerator',[]),
])
def test_bad_certificate_shapes_fail_closed(name,value):
    args={'base':[0.,1.],'slope':[1.,2.],'residual':[1.,2.],'unit':[1.,0.],
          'scalar':1.,'width':2,'epsilon':1e-6,'k':1.,'saved_numerator':[0.,1.],
          'saved_logp':[-1.,-1.]}
    args[name]=value
    with pytest.raises(ValueError):m.numerical_check(**args)

@pytest.mark.parametrize('which,value',[
    ('literal',[float('nan'),0.]),('editor',[0.,float('nan')]),
    ('editor',[float('-inf'),0.]),('editor',[[0.,1.]]),
    ('literal',[0.]),('editor',[.1,.2]),
])
def test_physical_distribution_guards(which,value):
    import numpy as np
    args={'literal':np.log([.25,.75]),'editor':np.log([.25,.75]),'expected_shape':(2,)}
    args[which]=value
    with pytest.raises(ValueError):m.physical_logp_check(**args)

def test_valid_physical_distributions():
    import numpy as np
    q=np.log([.25,.75]);result=m.physical_logp_check(q,q,(2,))
    assert result['editor_literal_logp_max_error']==0.


def test_overflow_residual_norm_rejected():
    import numpy as np
    with np.errstate(over='ignore'),pytest.raises(ValueError,match='residual norm'):
        m.numerical_check([0.,0.],[0.,0.],[1e308,1e308],[0.,0.],1.,2,1e-6,1.,[0.,0.],[-.693,-.693])


def test_certify_cli_needs_no_tokenizer_or_prices(monkeypatch,capsys,tmp_path):
    seen={}
    def fake_export(**kwargs):seen.update(kwargs);return {'offline':True}
    monkeypatch.setattr(m,'export_supported',fake_export)
    m.main(['certify','--evidence','existing','--config','config.yaml','--out',str(tmp_path/'out')])
    assert set(seen)=={'evidence','config','out'}
    assert 'offline' in capsys.readouterr().out
