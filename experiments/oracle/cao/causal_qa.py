"""Offline native-neuron causal reader interface; no model loading, training or calls.

Reuse claim_reader_probe's first-response/raw-token lifecycle, not its task renderer.
A future sampler must submit these exact token_ids, never decode/re-encode them.
Numeric accuracy does not verify English. Input-poking tasks are development only;
E1/E2/E3 additionally require an equally budgeted prompting-only comparison.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
from pathlib import Path

ARMS = ('true', 'absent', 'mismatch')
EDITS = ('target_zero', 'target_restore', 'control_zero', 'control_restore')
# Frozen before prospective outcomes: capped absolute error makes invalids worst.
ERROR_CAP_BITS = 60.0
INVALID_PENALTY_BITS = 60.0
PUBLIC = {'case_id','clean_ids','corrupt_ids','answer_a_ids','answer_b_ids',
          'target_units','control_units'}
MARKER = 'NATIVE_CAUSAL_READER_BODY_7f390ad2'


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_public(row):
    """Explicit allowlist: never copy collector clean/corrupt/intervention scores."""
    case = {k: row[k] for k in PUBLIC}
    if not isinstance(case['case_id'], str) or not case['case_id']:
        raise ValueError('nonempty case ID required')
    for key in ('clean_ids','corrupt_ids','answer_a_ids','answer_b_ids'):
        if not case[key] or any(type(i) is not int or i < 0 for i in case[key]):
            raise ValueError('nonempty nonnegative exact token ID sequences required')
    if len(case['clean_ids']) != len(case['corrupt_ids']):
        raise ValueError('token-aligned clean/corrupt prompts required')
    if case['answer_a_ids'] == case['answer_b_ids']:
        raise ValueError('distinct answer sequences required')
    for key in ('target_units','control_units'):
        units = [list(u) for u in case[key]]
        if not units or any(len(u)!=3 or any(type(i) is not int or i<0 for i in u)
                            or u[1]>=len(case['clean_ids']) for u in units):
            raise ValueError('native (layer,position,neuron) coordinates required')
        if len({tuple(u) for u in units}) != len(units):
            raise ValueError('duplicate native unit')
        case[key] = units
    from collections import Counter
    if (set(map(tuple,case['target_units'])) & set(map(tuple,case['control_units'])) or
        Counter((l,p) for l,p,n in case['target_units']) !=
        Counter((l,p) for l,p,n in case['control_units'])):
        raise ValueError('disjoint layer/position-matched controls required')
    return case


def prepare_case(row, tokenizer, *, reader_spec, mismatch_values, mismatch_binding, simple_baseline=None,
                 target_spec=None, target_tokenizer=None, mismatch_units=None,
                 mechanism_evidence=None, mismatch_mechanism_evidence=None):
    """Exact per-value slot lengths and all metadata positions matched across arms.

    Native sets have no fixed size cap. Absent uses existing '?' token padding;
    it matches positions/cost, with no claim that padding is semantically neutral.
    Mismatch values and their discovery-only binding must be frozen by the caller.
    Collector output may be passed here: only the PUBLIC allowlist and clean native
    scalar observations are read. Actual outcomes never enter request construction.
    """
    if set(reader_spec) != {'model','revision','tokenizer_sha256','adapter_sha256'} or any(
            not isinstance(v,str) or not v for v in reader_spec.values()):
        raise ValueError('explicit frozen reader/model/tokenizer binding required')
    if not isinstance(mismatch_binding,str) or not mismatch_binding:
        raise ValueError('frozen mismatch binding required')
    if target_spec is None or set(target_spec)!={'model','revision','tokenizer_sha256','adapter_sha256'} or any(
            not isinstance(v,str) or not v for v in target_spec.values()):
        raise ValueError('explicit original target/model/tokenizer/adapter binding required')
    case = native_public(row)
    display = None if target_tokenizer is None else {k: target_tokenizer.decode(case[k],
        skip_special_tokens=False) for k in ('clean_ids','corrupt_ids','answer_a_ids','answer_b_ids')}
    units = case['target_units'] + case['control_units']
    if (mismatch_units is None or len(mismatch_units)!=len(units) or
            any(len(d)!=3 or any(type(x) is not int or x<0 for x in d) or
                list(d[:2])!=list(u[:2]) for d,u in zip(mismatch_units,units))):
        raise ValueError('donor native units must match receiver layer/position coordinates')
    observed = row['clean_native_values']
    if [list(x['unit']) for x in observed] != units:
        raise ValueError('clean observation order must match target then control units')
    values = [x['value'] for x in observed]
    if len(values) != len(mismatch_values) or any(type(v) not in (int,float) or
            not math.isfinite(v) for v in values+list(mismatch_values)):
        raise ValueError('finite matched native scalar vectors required')
    if list(values)==list(mismatch_values):
        raise ValueError('degenerate mismatch; do not silently use identical evidence')
    if (mechanism_evidence is None)!=(mismatch_mechanism_evidence is None):
        raise ValueError('true and mismatch mechanism packets must both be supplied')
    mechanisms = validate_mechanisms(mechanism_evidence, units, target_spec)
    wrong_mechanisms = validate_mechanisms(mismatch_mechanism_evidence, units, target_spec)
    encode = lambda s: list(tokenizer.encode(s, add_special_tokens=False))
    pad = encode('?')
    if len(pad)!=1:
        raise ValueError('existing single-token padding required')
    body = ('Native neuron causal prediction. Units are [layer, prompt_position, neuron], '
        'at actual down-projection inputs, not decomposed features. Predict FOUR signed effects. '
        'contrast_bits = (log P(full answer A | prompt) - log P(full answer B | prompt))/ln(2). '
        'Sum teacher-forced log probabilities over every answer token; do not use first tokens only. '
        'zero: set named prompt-position activations to zero in CLEAN, delta=edited contrast minus CLEAN. '
        'restore: replace named prompt-position activations in CORRUPT with their CLEAN values, '
        'delta=edited contrast minus CORRUPT. Answer trajectories are unedited. '
        'Each target/control SET is intervened on jointly. Evidence values may be unavailable or mismatched; '
        'padding has no neutrality guarantee. Do not infer effect from activation magnitude alone.\n'
        'Exact target-model token IDs and native sets: '+canonical(case)+'\n' +
        ('' if display is None else 'Decoded display (IDs above define exact boundaries): '+canonical(display)+'\n') +
        'Clean native observations:\n')
    template = tokenizer.apply_chat_template([{'role':'user','content':MARKER}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if template.count(MARKER)!=1:
        raise ValueError('unambiguous local chat template required')
    prefix,suffix=template.split(MARKER)
    common=encode(prefix+body)
    arms={a:list(common) for a in ARMS}; spans=[];widths=[]
    for index,(u,true,wrong) in enumerate(zip(units,values,mismatch_values)):
        header=encode('Unit '+canonical(u)+' value: ')
        for a in ARMS: arms[a]+=header
        payloads={'true':encode(format(true,'.17g')),'mismatch':encode(format(wrong,'.17g'))}
        width=max(map(len,payloads.values()));start=len(arms['true'])
        spans.append(list(range(start,start+width)));widths.append(width)
        for a in ARMS:
            value=payloads.get(a,[])
            arms[a]+=value+pad*(width-len(value))+encode('\n')
        if mechanisms is not None:
            for a in ARMS: arms[a]+=encode('Discovery mechanism evidence for '+canonical(u)+': ')
            cards={'true':encode(canonical(mechanisms[index]['facts'])),
                   'mismatch':encode(canonical(wrong_mechanisms[index]['facts']))}
            width=max(map(len,cards.values()));start=len(arms['true'])
            spans.append(list(range(start,start+width)));widths.append(width)
            for a in ARMS:
                value=cards.get(a,[])
                arms[a]+=value+pad*(width-len(value))+encode('\n')
    instruction=('Return only JSON with exactly keys effects and explanation. effects has exactly '
        'target_zero,target_restore,control_zero,control_restore, each a finite signed number in bits. '
        'explanation is at most 120 words: a short falsifiable account naming native neurons as '
        'L<layer>:N<neuron>@P<position> and stating predicted intervention effects. '
        'A numeric score does not certify this explanation.\n')
    tail=encode(instruction+suffix);suffix_start=len(arms['true'])
    for a in ARMS:arms[a]+=tail
    editable={i for span in spans for i in span}
    assert len({len(v) for v in arms.values()})==1
    assert all(all(ids[i]==arms['true'][i] for i in range(len(ids)) if i not in editable)
               for ids in arms.values())
    requests=[{'request_id':case['case_id']+'|'+a,'case_id':case['case_id'],
               'arm':a,'token_ids':arms[a],'prompt_tokens':len(arms[a]),
               'tokens_sha256':digest(arms[a])} for a in ARMS]
    manifest={'schema':'native_causal_reader_v1','reader':reader_spec,'target':target_spec,
        'decoded_display_sha256':digest(display),'case':case,
        'case_sha256':digest(case),'source_observations_sha256':digest(observed),
        'module_sha256':file_sha(__file__),'simple_baseline':simple_baseline,
        'mechanism_evidence_sha256':digest(mechanisms),'mismatch_mechanism_evidence_sha256':digest(wrong_mechanisms),
        'mechanism_evidence_status':('MISSING: scalar-only interface fixture; no mechanistic pilot'
                                     if mechanisms is None else 'discovery-only supplied; source statuses explicit, not causal certification'),
        'mismatch_sha256':digest(list(mismatch_values)),'mismatch_binding':mismatch_binding,
        'mismatch_units':[list(u) for u in mismatch_units],
        'evidence_positions':spans,'evidence_widths':widths,'suffix_start':suffix_start,
        'prompt_tokens_per_arm':len(arms['true']),'requests_sha256':digest(requests),
        'error_cap_bits':ERROR_CAP_BITS,'invalid_penalty_bits':INVALID_PENALTY_BITS,
        'scope':'prospective numeric diagnostic; no English faithfulness or E1/E2/E3 certificate',
        'submission':'exact token IDs; no decode/re-encode; model/revision bound externally'}
    if simple_baseline is not None:
        validate_baseline(simple_baseline, manifest)
    return requests,manifest


def prediction(response, case):
    """Retain raw English separately; strict schema, no salvage of truncated output."""
    if response is None:return None,'missing'
    if response.get('parse_error'):return None,'parse_error'
    if str(response.get('finish','')).lower() not in ('stop','stopreason.stop'):
        return None,'unfinished'
    try:
        def pairs(items):
            d={}
            for k,v in items:
                if k in d:raise ValueError('duplicate key')
                d[k]=v
            return d
        obj=json.loads(response['completion'],object_pairs_hook=pairs)
        if not isinstance(obj,dict) or set(obj)!={'effects','explanation'}:raise ValueError()
        if not isinstance(obj['effects'],dict) or set(obj['effects'])!=set(EDITS):raise ValueError()
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in obj['effects'].values()):
            raise ValueError()
        text=obj['explanation']
        if not isinstance(text,str) or not 1<=len(text.split())<=120:raise ValueError()
        labels={f'L{l}:N{n}@P{p}' for key in ('target_units','control_units') for l,p,n in case[key]}
        named=set(re.findall(r'(?<![A-Za-z0-9])L[0-9]+:N[0-9]+@P[0-9]+(?![A-Za-z0-9])',text))
        if not named or not named<=labels:raise ValueError()
        return obj,None
    except (KeyError,TypeError,ValueError):return None,'malformed'


def commit_responses(path, requests, responses, *, manifest):
    """Local hash commitment fixture; publish/externally timestamp BEFORE gold access.

    This binds bytes but cannot prove temporal ordering alone. The orchestrator must
    record this receipt outside mutable prediction storage before gold is opened.
    Missing replies are explicit invalids; duplicate/unexpected replies are rejected.
    """
    if digest(requests)!=manifest['requests_sha256']:
        raise ValueError('frozen requests changed')
    expected={r['request_id']:r for r in requests};seen=set();rows=[]
    for r in responses:
        key=r['request_id']
        if key not in expected or key in seen:raise ValueError('unexpected/duplicate response')
        seen.add(key);rows.append(r)
    for key in expected:
        if key not in seen:rows.append({'request_id':key,'finish':'missing','completion':None})
    payload={'schema':'native_causal_prediction_commit_v1','manifest_sha256':digest(manifest),
             'requests_sha256':digest(requests),'responses':rows,'responses_sha256':digest(rows)}
    # Exclusive creation preserves already committed responses.
    with Path(path).open('x') as f:
        f.write(canonical(payload)+'\n');f.flush();os.fsync(f.fileno())
    return {'commit_file_sha256':file_sha(path),'responses_sha256':payload['responses_sha256'],
            'manifest_sha256':payload['manifest_sha256']}


def evaluate(commit_path, *, receipt, manifest, gold_path):
    """Open separate gold only AFTER checking committed responses and bindings.

    Primary: per-case mean clipped absolute delta error, paired against unchanged,
    absent and mismatch; optional independently supplied frozen simple baseline.
    All four interventions count, including missing/invalid predictions. No selective
    complete-only score. Raw English is returned but never graded mechanistically.
    """
    if file_sha(commit_path)!=receipt['commit_file_sha256']:
        raise ValueError('prediction commitment changed')
    payload=json.loads(Path(commit_path).read_text())
    if (payload['manifest_sha256']!=digest(manifest) or
        payload['manifest_sha256']!=receipt['manifest_sha256'] or
        payload['requests_sha256']!=manifest['requests_sha256'] or
        digest(payload['responses'])!=receipt['responses_sha256']):
        raise ValueError('commit/request/manifest binding changed')
    case=manifest['case'];expected={case['case_id']+'|'+a for a in ARMS}
    lookup={r['request_id']:r for r in payload['responses']}
    if len(lookup)!=len(payload['responses']) or set(lookup)!=expected:
        raise ValueError('committed response coverage changed')
    # No gold file access above this line.
    gold=json.loads(Path(gold_path).read_text())
    if (gold.get('reference')!='original_target_model' or
            gold.get('frozen_target_identity')!=manifest['target']):
        raise ValueError('gold target identity/reference changed')
    if gold['case_id']!=case['case_id'] or native_public(gold)!=case:
        raise ValueError('gold/public case binding changed')
    if digest(gold['clean_native_values'])!=manifest['source_observations_sha256']:
        raise ValueError('gold clean observations changed')
    truth={k:gold['interventions'][k]['delta_bits'] for k in EDITS}
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in truth.values()):
        raise ValueError('nonfinite physical gold')
    error=lambda p:sum(min(ERROR_CAP_BITS,abs(p[k]-truth[k])) for k in EDITS)/len(EDITS)
    unchanged=error(dict.fromkeys(EDITS,0.));arms={}
    for arm in ARMS:
        raw=lookup[case['case_id']+'|'+arm];obj,reason=prediction(raw,case)
        loss=INVALID_PENALTY_BITS if reason else error(obj['effects'])
        arms[arm]={'error_bits':loss,'failure':reason,'prediction':obj,
                   'raw_completion':raw.get('completion'),'raw_response':raw,
                   'unchanged_minus_reader_bits':unchanged-loss}
    result={'case_id':case['case_id'],'arms':arms,'unchanged_error_bits':unchanged,
        'paired_absent_minus_true_bits':arms['absent']['error_bits']-arms['true']['error_bits'],
        'paired_mismatch_minus_true_bits':arms['mismatch']['error_bits']-arms['true']['error_bits'],
        'error_cap_bits':ERROR_CAP_BITS,'invalid_penalty_bits':INVALID_PENALTY_BITS,
        'gold_sha256':file_sha(gold_path),'scope':manifest['scope'],
        'english_faithfulness':'unscored; requires separate named intervention/repair tests',
        'prompting_only_baseline':'not collected; equal-budget E1/E2/E3 validity unestablished'}
    simple_baseline=manifest.get('simple_baseline')
    if simple_baseline is not None:
        validate_baseline(simple_baseline,manifest)
        result['simple_baseline_error_bits']=error(simple_baseline['predictions'])
        result['simple_baseline_minus_true_bits']=result['simple_baseline_error_bits']-arms['true']['error_bits']
        result['simple_baseline_provenance']=simple_baseline['provenance']
    return result


def validate_baseline(simple_baseline, manifest):
    if (set(simple_baseline)!={'case_sha256','predictions','provenance'} or
            simple_baseline['case_sha256']!=manifest['case_sha256'] or
            not simple_baseline['provenance'] or set(simple_baseline['predictions'])!=set(EDITS) or
            any(type(v) not in (int,float) or not math.isfinite(v)
                for v in simple_baseline['predictions'].values())):
        raise ValueError('independently frozen simple-baseline binding required')


def validate_mechanisms(rows, units, target_spec):
    """Discovery provenance is caller-attested, never inferred from numeric success.

    Raw text cannot itself prove source provenance; immutable discovery artifacts
    and their chronology require independent review before any real invocation.
    """
    if rows is None:return None
    if len(rows)!=len(units):raise ValueError('one mechanism record per receiver unit required')
    kinds={'atlas_label','discovery_effect','route','weight_readout'}
    statuses={'verified_atlas','measured_discovery','static_readout','unverified_label'}
    for row,unit in zip(rows,units):
        if set(row)!={'unit','facts'} or list(row['unit'])!=unit or not row['facts']:
            raise ValueError('mechanism record must bind the same receiver native unit')
        for fact in row['facts']:
            if (set(fact)!={'kind','text','source_sha256','split','source_status','frozen_before_eval','evidence_model_identity','claim_status'} or
                fact['kind'] not in kinds or fact['source_status'] not in statuses or
                fact['split']!='discovery' or fact['frozen_before_eval'] is not True or
                fact['claim_status'] not in {'observed','measured','hypothesis'} or
                not isinstance(fact['evidence_model_identity'],dict) or
                set(fact['evidence_model_identity'])!=set(target_spec) or
                any(not isinstance(v,str) or not v for v in fact['evidence_model_identity'].values()) or
                (fact['evidence_model_identity']!=target_spec and fact['claim_status']!='hypothesis') or
                not isinstance(fact['text'],str) or not fact['text'].strip() or
                not isinstance(fact['source_sha256'],str) or
                re.fullmatch('[0-9a-f]{64}',fact['source_sha256']) is None):
                raise ValueError('strict discovery-only mechanism provenance/status required')
    return rows
