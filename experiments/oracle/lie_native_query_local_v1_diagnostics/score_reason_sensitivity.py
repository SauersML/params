"""EXPLORATORY post-prediction/pre-gold sensitivity; only reason wordcount differs."""
import argparse
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
protocol=json.loads((HERE/'protocol.json').read_text())
spec=importlib.util.spec_from_file_location('primary_scorer_frozen',HERE/'primary_scorer_frozen.py')
primary=importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
if primary.sha((HERE/'primary_scorer_frozen.py').read_bytes())!=protocol['primary_source_sha256']:
    raise ValueError('frozen primary scorer hash mismatch')


def parse_numeric_only(row,raw):
    value,error,reason=primary.parse(row,raw)
    if error!='invalid_strict_json':return value,error,reason
    # Only an otherwise valid object with an overlong STRING reason is eligible.
    # Strictly reject duplicates/nonfinite constants BEFORE reserialization.
    def pairs(items):
        primary.require(len(dict(items))==len(items),'duplicate JSONkeys')
        return dict(items)
    def constant(value):raise ValueError('nonfinite JSONconstant')
    try:
        obj=json.loads(row['completion'],object_pairs_hook=pairs,parse_constant=constant)
        primary.require(isinstance(obj,dict) and set(obj)=={'delta_bits','reason'},'exact fields')
        primary.require(isinstance(obj['reason'],str) and len(obj['reason'].split())>40,'not overlong reason alone')
        shortened=dict(row,completion=json.dumps(dict(delta_bits=obj['delta_bits'],reason='Wordcount omitted only for exploratory diagnostic.')))
        value,error,_=primary.parse(shortened,raw)
        return (value,None,obj['reason']) if error is None else (None,error,None)
    except (ValueError,TypeError,OverflowError,KeyError):
        return None,'invalid_strict_json',None


# Give the unchanged scorer its own globals dictionary. Do not monkeypatch the
# primary module or modify its parser, metric, coverage, bootstrap or baselines.
_secondary_globals=dict(primary.score.__globals__,parse=parse_numeric_only)
secondary_score=types.FunctionType(primary.score.__code__,_secondary_globals,'secondary_score',primary.score.__defaults__,primary.score.__closure__)


def analyze(bundle,gold):
    strict=primary.score(bundle,gold)
    secondary=secondary_score(bundle,gold)
    rescued=[]
    for row in bundle['responses']:
        raw=bundle['raw'].get(row['request_id'],{})
        old=primary.parse(row,raw);new=parse_numeric_only(row,raw)
        if old[1] is not None and new[1] is None:
            rescued.append(dict(request_id=row['request_id'],reason_words=len(new[2].split())))
    secondary['schema']='EXPLORATORY_reason_wordcount_only_sensitivity_v1'
    secondary['secondary_absent_minus_true']=secondary.pop('primary')
    secondary['diagnostic_protocol']=protocol
    secondary['reason_only_rescued_predictions']=rescued
    secondary['strict_frozen_primary_reference']=dict(primary=strict['primary'],losses=strict['losses'],invalid_predictions_by_arm=strict['invalid_predictions_by_arm'])
    secondary['limits'].insert(0,'EXPLORATORY post-predictions/pre-gold protocol: this relaxed-wordcount sensitivity never replaces frozen primary.')
    return secondary


def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen',required=True);p.add_argument('--responses',required=True);p.add_argument('--prediction-commit-sha',required=True);p.add_argument('--gold',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    bundle=primary.verify_predictions(args.frozen,args.responses,args.prediction_commit_sha)
    # First gold read follows the unchanged primary commitment/coverage verifier.
    blob=Path(args.gold).read_bytes();result=analyze(bundle,json.loads(blob));result['gold_sha256']=primary.sha(blob)
    result['diagnostic_source_sha256']=primary.sha(Path(__file__).read_bytes())
    Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
