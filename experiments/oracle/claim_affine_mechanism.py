"""Optional checked local affine claims, not a free-prose truth certificate.

Consumes the historical frozen unedited-facts payload and its physical receipt.
Preparation is offline. Sampling belongs to the existing guarded sampler; this
module never creates a ServiceClient. Defaults of old consumers are unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

TASKS = {'text6008': 3268, 'text6011': 2412, 'text6034': 1833}
CLAIMS = {
    'weight_operation': 'W_plus_(k_minus_1)_outer_U_V_all_positions',
    'site_input': 'unchanged_by_own_final_down_weight_edit',
    'residual': 'r_at_1_plus_(k_minus_1)_a_U',
    'numerator': 'base_at_1_plus_(k_minus_1)_slope',
    'normalization': 'sqrt_mean_residual_k_squared_plus_model_epsilon',
    'readout': 'numerator_divided_by_positive_shared_final_RMS',
    'argmax': 'common_positive_RMS_cancels_only_for_argmax',
    'scope': 'fixed_input_final_down_local_equations_not_full_circuit',
}
CAP = .10
MAX_TOKENS = 1024

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_evidence(root):
    root = Path(root)
    payload = json.loads((root / 'writer_payload.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    rows = [json.loads(x) for x in (root / 'verification.jsonl').read_text().splitlines()]
    if sha(root / 'writer_payload.json') != protocol['writer_payload_frozen_before_first_edited_forward']:
        raise ValueError('payload not frozen before edited forwards')
    expected = {(t, k) for t in TASKS for k in (-1., 0., .5, 1., 1.5, 2.)}
    if protocol['selected'] != TASKS or set(payload['cases']) != set(TASKS) or len(rows) != 18 or {(r['task'], r['k']) for r in rows} != expected:
        raise ValueError('fixed three TRAIN cases and18 physical checks required')
    if any(r['unit'] != TASKS[r['task']] or r['ideal_literal_top_matches'] is not True or r['editor_top'] != r['literal_top'] for r in rows):
        raise ValueError('physical receipt mismatch')
    return payload

def numerical_check(base, slope, residual, unit, scalar, width, epsilon, k,
                    saved_numerator, saved_logp, *, anchor=1., use_rms=True):
    """No model execution. Compare an interpreted formula to saved physical arrays."""
    import numpy as np
    if type(width) is not int or width <= 0 or any(type(x) not in (int,float) or not math.isfinite(x) for x in (scalar,k,anchor,epsilon)) or epsilon <= 0:
        raise ValueError("invalid scalar or width")
    base,slope,residual,unit,saved_numerator,saved_logp = [np.asarray(x,dtype="float64") for x in (base,slope,residual,unit,saved_numerator,saved_logp)]
    if any(x.ndim != 1 or not x.size or not np.isfinite(x).all() for x in (base,slope,residual,unit,saved_numerator,saved_logp)) or not (base.shape == slope.shape == saved_numerator.shape == saved_logp.shape) or residual.shape != (width,) or unit.shape != (width,):
        raise ValueError("finite matching one-dimensional shapes required")
    z = base + (k-anchor)*slope
    r = residual + (k-anchor)*scalar*unit
    if not all(np.isfinite(v).all() for v in (z, r, saved_numerator, saved_logp)) or len(r) != width or epsilon <= 0:
        raise ValueError('invalid numerical certificate')
    squared_norm=float(r@r)
    mean_square_plus_epsilon=squared_norm/width+epsilon
    if not math.isfinite(squared_norm) or not math.isfinite(mean_square_plus_epsilon) or mean_square_plus_epsilon <= 0:
        raise ValueError("nonfinite residual norm or RMS denominator")
    denominator=math.sqrt(mean_square_plus_epsilon)
    logits = z / denominator if use_rms else z
    logp = logits - (float(np.max(logits)) + math.log(float(np.exp(logits-np.max(logits)).sum())))
    error = float(np.max(np.abs(logp-np.asarray(saved_logp, dtype='float64'))))
    numerator_error = float(np.max(np.abs(z-np.asarray(saved_numerator, dtype='float64'))))
    if not np.isfinite(logits).all() or not np.isfinite(logp).all() or not math.isfinite(error) or not math.isfinite(numerator_error) or numerator_error > 1e-9 or error > 2e-4:
        raise ValueError('formula falsified by frozen numerical/physical evidence')
    return {'numerator_max_error':numerator_error,'literal_logp_max_error':error}


def physical_logp_check(literal, editor, expected_shape):
    """Reject malformed/nonfinite distributions BEFORE physical parity checks."""
    import numpy as np
    values=[np.asarray(v,dtype='float64') for v in (literal,editor)]
    normalizers=[]
    for v in values:
        if v.ndim != 1 or v.shape != expected_shape or not v.size or not np.isfinite(v).all():
            raise ValueError('finite matching one-dimensional physical logp required')
        lse=float(np.max(v))+math.log(float(np.exp(v-np.max(v)).sum()))
        if not math.isfinite(lse) or abs(lse)>2e-5:
            raise ValueError('physical logp not normalized')
        normalizers.append(lse)
    error=float(np.max(np.abs(values[0]-values[1])))
    if not math.isfinite(error) or error>2e-4:raise ValueError('physical/edit parity mismatch')
    return {'editor_literal_logp_max_error':error,'literal_logsumexp':normalizers[0],'editor_logsumexp':normalizers[1]}


def certify_evidence(root, config):
    """Validate the frozen local algebra against all18 stored physical outputs.

    Trust boundary: original producer/source and four content-hashed assets;
    no new model run, no universal FP32 certificate, no free prose parsing.
    """
    import numpy as np
    from safetensors.numpy import load_file
    root=Path(root); payload=load_evidence(root)
    if sha(root/'coefficients.safetensors') != payload['coefficient_sha256'] or sha(config) != payload['model_content_sha256']['config']:
        raise ValueError('coefficient/config content mismatch')
    # This source config is simple YAML; require exactly one numeric epsilon.
    import re
    matches=re.findall(r'^rms_norm_eps:\s*([0-9.eE+-]+)\s*$',Path(config).read_text(),re.M)
    if len(matches)!=1:raise ValueError('unambiguous source RMS epsilon required')
    epsilon=float(matches[0]); coeff=load_file(str(root/'coefficients.safetensors'))
    result=json.loads((root/'results.json').read_text())
    if result.get('rows')!=18 or result.get('restored_W_U_V_bitwise') is not True or result.get('writer_payload_unchanged') is not True or result.get('writer_payload_sha256')!=sha(root/'writer_payload.json'):
        raise ValueError('producer/restoration receipt mismatch')
    reports=[]; files={'coefficients.safetensors':sha(root/'coefficients.safetensors'),'results.json':sha(root/'results.json')}
    for task in TASKS:
        c=payload['cases'][task]; r=coeff[task+'_residual'].astype('float64');u=coeff[task+'_U'].astype('float64');a=float(coeff[task+'_raw_scalar'])
        observed={'r_squared_norm':float(r@r),'U_squared_norm':float(u@u),'r_dot_U':float(r@u),'raw_scalar_x_dot_V':a,'residual_width':len(r)}
        if any(not math.isclose(observed[k],c[k],rel_tol=1e-12,abs_tol=1e-12) for k in observed):raise ValueError('source geometry mismatch')
        for k in (-1.,0.,.5,1.,1.5,2.):
            name=task+'_'+str(k)+'.safetensors'; arrays=load_file(str(root/name));files[name]=sha(root/name)
            physical=physical_logp_check(arrays["literal_logp"],arrays["editor_logp"],coeff[task+"_base"].shape)
            report=numerical_check(coeff[task+'_base'],coeff[task+'_slope'],r,u,a,len(r),epsilon,k,arrays['ideal_numerators'],arrays['literal_logp'])
            if int(np.argmax(arrays['ideal_numerators']))!=int(np.argmax(arrays['literal_logp'])):raise ValueError('physical/edit parity mismatch')
            reports.append({'task':task,'k':k,**report,**physical})
    return {'producer_protocol_sha256':sha(root/'protocol.json'),'producer_source_sha256':json.loads((root/'protocol.json').read_text())['source_sha256'],'producer_runner_sha256':json.loads((root/'protocol.json').read_text())['runner_sha256'],'tolerances':{'numerator_max_abs':1e-9,'literal_logp_max_abs':2e-4,'editor_literal_logp_max_abs':2e-4,'logsumexp_abs':2e-5},'schema':1,'payload_sha256':digest(payload),'validated':True,'physical_points':18,'epsilon':epsilon,'files':files,'config_sha256':sha(config),'rows':reports,'scope':'source-bound local affine algebra and18physical checks; not universalFP32 or prose certificate'}

def expected_sidecar(payload, task):
    case = payload['cases'][task]
    if task not in TASKS or case['component_index'] != TASKS[task] or case['site'] != 'h.3.mlp.down_proj':
        raise ValueError('unsupported local case')
    # Bind native text exactly, including specials/whitespace; model content is
    # bound by all original asset digests in the full payload digest.
    binding = {'task': task, 'site': case['site'], 'component_index': case['component_index'],
               'token_position': case['token_position'],
               'execution_text_sha256': hashlib.sha256(case['execution_text'].encode()).hexdigest(),
               'payload_sha256': digest(payload)}
    numeric = {k: case[k] for k in ('raw_scalar_x_dot_V', 'r_squared_norm', 'r_dot_U', 'U_squared_norm', 'residual_width')}
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in numeric.values()):
        raise ValueError('nonfinite/invalid frozen scalar')
    if numeric['residual_width'] <= 0 or type(numeric['residual_width']) is not int or numeric['r_squared_norm'] < 0 or numeric['U_squared_norm'] < 0:
        raise ValueError('invalid residual geometry')
    return {'schema': 1, 'binding': binding, 'claims': dict(CLAIMS), 'geometry': numeric}

def verify(payload, task, response, *, certificate):
    """Verify only explicitly typed claims; NEVER certify prose agrees with them."""
    if certificate.get("validated") is not True or certificate.get("physical_points") != 18 or certificate.get("payload_sha256") != digest(payload):
        raise ValueError("validated source-bound numerical certificate required")
    expected = expected_sidecar(payload, task)
    if not isinstance(response, dict) or set(response) != {'prose', 'mechanism'} or not isinstance(response['prose'], str) or not response['prose'].strip():
        raise ValueError('expected prose and mechanism only')
    actual = response['mechanism']
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError('missing/unknown mechanism fields')
    # Canonical strict JSON comparison also rejects bool-as-number, extras and
    # nonfinite values. Exact source values are copied, not estimated by writer.
    if digest(actual) != digest(expected):
        raise ValueError('unsupported or source-inconsistent typed claim')
    return {'typed_claims_verified': True, 'free_prose_verified': False,
            'mechanism': actual, 'prose': response['prose'], 'payload_sha256': digest(payload)}

def render_verified(payload, task, response, *, certificate):
    """Deterministic source-derived local English; no interval/behavior invention."""
    checked = verify(payload, task, response, certificate=certificate)
    m = checked['mechanism']; b = m['binding']; g = m['geometry']
    return (f"For this fixed native input (SHA256 {b['execution_text_sha256']}), scale component {b['component_index']} "
            f"of {b['site']} persistently at all positions by W(k)=W+(k-1)outer(U_c,V_c). "
            "Its own final-down input is unchanged. At the specified last target position "
            f"{b['token_position']}, a=x dot V_c={g['raw_scalar_x_dot_V']!r} and r(k)=r(1)+(k-1)a U_c. "
            "The token numerator is base+(k-1)slope, where base=(r(1)*ln_f)E.T and slope=a*(U_c*ln_f)E.T. "
            "The final logits divide these numerators by sqrt(mean(r(k)^2)+the model epsilon). "
            "This common positive denominator cancels for argmax; it does not make logits or probabilities affine. "
            f"The squared residual norm is {g['r_squared_norm']!r}+2*(k-1)*a*{g['r_dot_U']!r}"
            f"+(k-1)^2*a^2*{g['U_squared_norm']!r}, over width {g['residual_width']}. "
            "These are local equations conditional on the given unmodified residual boundary and fixed input; "
            "they do not certify a full circuit, the writer's free prose, unseen inputs or universal FP32 parity.")

def evaluate_responses(payload, certificate, rows):
    """Retain all three first responses; invalid/truncated prose is never repaired."""
    by_id={}
    for row in rows:
        ident=row.get('request_id')
        if ident not in {'mechanism_writer|'+t for t in TASKS} or ident in by_id:
            raise ValueError('unexpected/duplicate first response')
        by_id[ident]=row
    result=[]
    for task in TASKS:
        row=by_id.get('mechanism_writer|'+task)
        item={'task':task,'typed_claims_verified':False,'free_prose_verified':False}
        if row is None:item['invalid_reason']='missing'
        elif row.get('parse_error') or str(row.get('finish','')).lower() not in ('stop','stopreason.stop'):
            item['invalid_reason']='unfinished_or_parse_error'
        else:
            try:
                parsed=json.loads(row['completion'])
                item.update(verify(payload,task,parsed,certificate=certificate))
                item['deterministic_supported_english']=render_verified(payload,task,parsed,certificate=certificate)
            except (ValueError,TypeError,KeyError) as exc:
                item['invalid_reason']=str(exc)
        result.append(item)
    return result

def writer_prompt(payload, task):
    sidecar = expected_sidecar(payload, task)
    facts = {'case': payload['cases'][task], 'mechanism': payload['mechanism'], 'model_content_sha256': payload['model_content_sha256']}
    return ("Return ONLY one JSON object with prose and mechanism. Write at most180words of self-contained English explaining "
            "the persistent weight operation, k=1 anchoring, residual, final RMS/readout, and local limitations. "
            "Do not claim full-circuit fidelity or exact probabilities. The mechanism field must contain the following "
            "typed claims and source geometry exactly when supported; do not substitute a k=0 intercept for base_at_1. "
            "These facts come from unedited weights/activations; no edited-forward observations are supplied. "
            "The typed checker does not certify your free prose.\nREQUIRED MECHANISM SCHEMA/CLAIMS\n" + json.dumps(sidecar, sort_keys=True) +
            "\nSOURCE FACTS\n" + json.dumps(facts, sort_keys=True))

def prepare(evidence, tokenizer, prices, out, config):
    import claim_reader_probe as guarded
    import claim_affine_english_probe as affine
    out = Path(out)
    if out.exists(): raise ValueError('immutable preparation')
    payload = load_evidence(evidence)
    certificate = certify_evidence(evidence, config)
    official = next(r for r in json.loads(Path(prices).read_text()) if r['tinker_id'] == guarded.MODEL)
    if [official[k] for k in ('prefill', 'cached_prefill', 'sample')] != ['$0.45', '$0.09', '$1.335']:
        raise ValueError('fresh official rate mismatch')
    rr = guarded.configure(Path(tokenizer)); rows = []
    for task in TASKS:
        rows.append({'request_id': 'mechanism_writer|' + task, 'task': task, 'model': guarded.MODEL,
                     'max_tokens': MAX_TOKENS, **affine.encoded(rr, writer_prompt(payload, task))})
    upper = sum((r['prompt_tokens']*.45 + MAX_TOKENS*1.335)/1e6 for r in rows)
    if upper > CAP or any(r['prompt_tokens']+MAX_TOKENS+16 > 65536 for r in rows):
        raise ValueError('cost/context admission failed; no truncation')
    out.mkdir(parents=True)
    (out/"certificate.json").write_text(json.dumps(certificate,indent=2)+"\n")
    (out/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest = {'schema': 1, 'model': guarded.MODEL, 'project': guarded.PROJECT, 'calls': 3,
                'max_tokens': MAX_TOKENS, 'temperature': 0., 'cap_usd': CAP, 'cost_upper_usd': upper,
                'certificate_sha256': sha(out/'certificate.json'), 'requests_sha256': sha(out/'requests.jsonl'), 'source_sha256': sha(__file__),
                'guarded_sampler_sha256': sha(guarded.__file__), 'official_prices_sha256': sha(prices),
                'rates': guarded.PRICES, 'payload_object_sha256': digest(payload),
                'sources': {n: sha(Path(evidence)/n) for n in ('writer_payload.json','protocol.json','verification.jsonl')},
                'tokenizer_inventory': {p.name:sha(p) for p in Path(tokenizer).iterdir() if p.is_file()},
                'scope': 'three reused adaptive TRAIN cases; structured source-consistency only; free prose unverified; no API authorized by this file'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

def export_supported(evidence, config, out):
    """Useful offline deliverable: no tokenizer, pricing, SDK or API required."""
    out=Path(out)
    if out.exists():raise ValueError('immutable certificate export')
    payload=load_evidence(evidence);certificate=certify_evidence(evidence,config)
    out.mkdir(parents=True)
    (out/'certificate.json').write_text(json.dumps(certificate,indent=2)+'\n')
    for task in TASKS:
        # These are deterministic source-derived controls, NOT writer responses.
        control={'prose':'Not assessed: source-derived typed control, not a writer response.',
                 'mechanism':expected_sidecar(payload,task)}
        checked=verify(payload,task,control,certificate=certificate)
        (out/(task+'_typed_control.json')).write_text(json.dumps(checked,indent=2)+'\n')
        (out/(task+'_supported_english.txt')).write_text(render_verified(payload,task,control,certificate=certificate)+'\n')
    manifest={'schema':1,'source_sha256':sha(__file__),'payload_sha256':digest(payload),
              'source_files':{n:sha(Path(evidence)/n) for n in ('writer_payload.json','protocol.json','verification.jsonl')},
              'config_sha256':sha(config),'physical_points':18,'new_model_forwards':0,'writer_responses':0,
              'free_prose_verified':False,'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
              'scope':'source-derived local final-down equations; original producer trusted; no global circuit or free-prose certificate'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser();commands=p.add_subparsers(dest='command',required=True)
    cert=commands.add_parser('certify',help='offline numerical certificate and supported local English')
    for key in ('evidence','config','out'):cert.add_argument('--'+key,type=Path,required=True)
    prep=commands.add_parser('prepare',help='optional source-transcription writer requests; no sampling')
    for key in ('evidence','tokenizer','prices','out','config'):prep.add_argument('--'+key,type=Path,required=True)
    args=vars(p.parse_args(argv));command=args.pop('command')
    result=export_supported(**args) if command=='certify' else prepare(**args)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
