"""The neuron-grounded oracle's examples (#2951 G2; the lead's 10-15 direction, decision 6, and its 8B update). Beside a
text's half-depth activations (the Activation Oracle's read), the oracle reads the neurons of a question-scoped circuit
(method/oracle/neuron_builder.py: the answer contrast on the weak-zeroed reference, a unit one neuron's value at one
position) in measured order: the circuit's units, then the builder's strongest measured singles beyond it; a random
number of them, log-uniform from MIN_SHOWN up to all, so that how many to show is chosen afterwards by held-out accuracy
(the lead's 10-15: no fixed cap; eval reports accuracy by the number shown). Each shows as its name, "L<layer>.N<neuron>+@<position>" ("-" for the negative pole: claim_check's
unit syntax), and its write (its value x its down_proj column, cao.Oracle.write). The questions, every answer measured
by method/oracle/claim_check.py, the oracle's reward checker, so that the claim an answer ends with passes there:

  n_steer  what turning one shown neuron up (STEER x its calibration maximum, along its pole) does to the next token
  n_zero   what zeroing one shown neuron does
  n_set    what zeroing two to four shown neurons together does
  n_sign   whether a shown neuron supports the next token or works against it (its zeroing's sign)

An answer is one English sentence and a line "Claim: {units, op, predict}" (the prompt and position are the text's own
and the answer position); a claim of an effect that random neurons of the same layers also produce is said to be so
(claim_check's "specific"). Texts: wikitext-103 windows of h2h.T tokens (the answer the model's next token after the
window); --records: circuits built elsewhere, one JSON file per text with "ids" (neuron_builder's records, or
circuit_finder's patching-pair records: their neurons ordered by measured share, the clean run's values as writes).

  g2.py gen --out EX.jsonl --thr THR.pt [--texts N] [--seed S] [--split test] [--necessity-only] [--records DIR]
                                           (records: one in ten held out by id into EX.heldout.jsonl)
  g2.py eval --items EX.jsonl --adapter A --out OUT.jsonl [--text-only]   graded answers; the stated claims checked
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
import cao  # noqa: E402
import h2h  # noqa: E402  (cao puts aoh2h on the path)

STEER = 3.0     # claim_check's steer scale: x the neuron's calibration maximum
MIN_SHOWN = 4
HALF = 0.25     # bits: a change smaller than this is "almost none"
QUESTIONS = (("n_steer", 2), ("n_zero", 1), ("n_set", 1), ("n_sign", 1))


def name(d: dict) -> str:
    return f"L{d['layer']}.N{d['neuron']}{'+' if d['pole'] > 0 else '-'}@{d['pos']}"


def measured_order(rec: dict) -> list[dict]:
    """A record's neurons in measured order: neuron_builder's (the circuit, then its measured singles) as recorded;
    circuit_finder's (the patching pair) by the larger of each neuron's lost and restored shares alone."""
    if "ranked_top" in rec:
        return list({name(d): d for d in rec["nodes"] + rec["ranked_top"]}.values())
    return sorted(rec.get("nodes", []), key=lambda d: -max(abs(d.get("lost") or 0), abs(d.get("restored") or 0)))


def value(d: dict) -> float:
    return d["act"] if "act" in d else d["clean"]  # (circuit_finder records the clean run's value)


def bits(x: float) -> str:
    v = round(abs(x) * 2) / 2
    return f"{v:g} bit" + ("" if v == 1 else "s")


def listing(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def outcome(r: dict, y: str) -> tuple[str, dict]:
    """A measured result as an English clause and the prediction that states it."""
    if r["top_after"] != r["top_before"]:
        return f"the model gives {cao.q_tok(r['top_after'])} instead of {y}", {"top": r["top_after"]}
    ch = r["change"]
    if abs(ch) < HALF:
        return f"the model still gives {y}, almost unchanged", {"top_changes": False}
    v = round(ch * 2) / 2
    return (f"the model still gives {y}, its lead over the alternatives {'up' if ch > 0 else 'down'} about {bits(ch)}",
            {"contrast_change": {"op": ">=" if ch > 0 else "<=", "value": v - 0.25 if ch > 0 else v + 0.25}})


def text_examples(o: "cao.Oracle", nr, ids: list[int], rng: random.Random, rec: dict | None = None) -> tuple[list[dict], dict]:
    import claim_check as cc
    import neuron_builder as nb
    pos = len(ids) - 1
    rec = rec if rec is not None else nb.build(nr, ids, pos)
    order = measured_order(rec)
    if not order:
        return [], {"size": 0}
    lo = min(MIN_SHOWN, len(order))  # a count log-uniform from MIN_SHOWN to all (a shown unit is ~16 tokens: uniform counts
    shown = order[:round(math.exp(rng.uniform(math.log(lo), math.log(len(order)))))]  # made 8B prompts 2,734 tokens on average)
    names = [name(d) for d in shown]
    blocks = [{"layer": cao.MID, "positions": list(range(len(ids)))}] + [{"label": n, "write": [d["layer"], d["neuron"], value(d)]}
                                                                          for n, d in zip(names, shown)]
    y = cao.q_tok(o.tok.decode([rec["answer"]]))
    asks = []
    for cat, k in QUESTIONS:
        for _ in range(k):
            if cat == "n_set" and len(names) < 2:
                continue
            us = rng.sample(names, rng.randint(2, min(4, len(names)))) if cat == "n_set" else [rng.choice(names)]
            op = {"steer": STEER} if cat == "n_steer" else "zero"
            if (cat, tuple(us)) not in {(c_, tuple(u_)) for c_, u_, _ in asks}:  # (a small circuit: one question per unit)
                asks.append((cat, us, op))
    probe = [{"prompt": {"ids": ids}, "position": -1, "units": us, "op": op, "predict": {"top_changes": True}} for _, us, op in asks]
    got = cc.check_many(nr, probe)
    claims, texts = [], []
    for (cat, us, op), r in zip(asks, got):
        if "error" in r:
            continue
        if cat == "n_sign":
            ch = r["change"]  # zeroing: a supporting neuron's removal lowers the contrast
            pred = {"contrast_change": {"op": "<=", "value": -HALF}} if ch <= -HALF else \
                {"contrast_change": {"op": ">=", "value": HALF}} if ch >= HALF else {"top_changes": False}
            text = (f"{us[0]} supports {y}." if ch <= -HALF else f"{us[0]} works against {y}." if ch >= HALF else
                    f"{us[0]} barely matters for {y}.")
        else:
            clause, pred = outcome(r, y)
            verb = "Turning " + us[0] + " up" if cat == "n_steer" else "Zeroing " + listing(us) + (" together" if len(us) > 1 else "")
            text = f"{verb}: {clause}."
        claims.append({"prompt": {"ids": ids}, "position": -1, "units": us, "op": op, "predict": pred})
        texts.append((cat, us, op, text, pred))
    final = cc.check_many(nr, claims)  # the stated claims themselves, with their controls
    out = []
    for (cat, us, op, text, pred), r in zip(texts, final):
        if not r.get("pass"):
            continue  # (bf16 runs at the edge of a threshold)
        if not r["specific"]:
            text += " Random neurons of the same layers do the same."
        q = {"n_steer": f"What happens to the model's next token {y} if {us[0]} is turned up?",
             "n_zero": f"What happens to the model's next token {y} if {us[0]} is zeroed?",
             "n_set": f"What happens to the model's next token {y} if {listing(us)} are zeroed together?",
             "n_sign": f"Does {us[0]} support the model's next token {y} or work against it?"}[cat]
        claim = {"units": us, "op": op, "predict": pred}
        out.append({"cat": cat, "ctx": ids, "blocks": blocks, "q": q, "a": f"{text}\nClaim: {json.dumps(claim)}",
                    "claim": claim, "specific": r["specific"], "shown": len(shown), "circuit": len(rec["nodes"])})
    return out, {"size": rec.get("size"), "met": rec.get("met"), "shown": len(shown), "questions": len(out), "build_seconds": rec.get("seconds")}


def windows(tok, split: str, rng: random.Random):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    src = cao.WIKITEXT_TRAIN if split == "train" else h2h.WIKITEXT
    df = pd.read_parquet(hf_hub_download(src[0], src[1], repo_type="dataset"))
    paras = [t.strip() for t in df["text"].tolist() if len(t.strip()) > 200 and not t.strip().startswith("=")]
    rng.shuffle(paras)
    for t in paras:
        ids = tok(" " + t, add_special_tokens=False)["input_ids"]
        if len(ids) >= h2h.T + 8:
            s = rng.randrange(0, len(ids) - h2h.T)
            yield ids[s:s + h2h.T]


def gen(a) -> None:
    import neuron_builder as nb
    nb.SUFFICIENCY = nb.SUFFICIENCY and not a.necessity_only
    o = cao.Oracle(None, a.device)
    nr = nb.Neurons(o.m, o.tok)
    nr.load(a.thr)
    rng = random.Random(a.seed)
    done = {json.loads(x)["text"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    src = ((json.loads(f.read_text()) for f in sorted(a.records.glob("*.json"))) if a.records else
           ({"ids": ids} for ids in windows(o.tok, a.split, rng)))
    t0 = time.time()
    import zlib
    held = a.out.with_name(a.out.stem + ".heldout.jsonl")  # records: one in ten (by id) held out for evaluation
    with a.out.open("a") as fh, held.open("a") as hh:
        for n, r in enumerate(src):
            if n >= a.texts:
                break
            key = 1_000_000 * a.seed + n
            if key in done:
                continue
            if "ids" not in r:  # (neuron_builder's main records the chat, not its ids)
                r["ids"] = nb.chat(o.tok, r["user"])
                assert len(r["ids"]) == r["T"], (r.get("id"), len(r["ids"]), r["T"])
            exs, info = text_examples(o, nr, r["ids"], rng, r if a.records else None)
            dst = hh if a.records and zlib.crc32(str(r.get("id", n)).encode()) % 10 == 0 else fh
            for e in exs:
                dst.write(json.dumps({**e, "text": key, **({"id": r["id"]} if "id" in r else {})}) + "\n")
            dst.flush()
            if (n + 1) % 10 == 0:
                print(json.dumps({"texts": n + 1, "seconds": round(time.time() - t0), **info}), flush=True)
    print(f"{a.texts} texts to {a.out}")


def evaluate(a) -> None:
    """The oracle's greedy answers to held-out examples (gen --split test), graded on the English sentence's outcome and
    on its claim, the claim re-checked by claim_check on the reference (pass and specific); --text-only: no vector
    added (the control). Printed beside: the most common answer per category."""
    import collections

    exs = [json.loads(x) for x in a.items.read_text().splitlines()]
    if getattr(a,'native_controls',False):return evaluate_native(a,exs)
    if any(e.get('cat')=='native_edit' for e in exs):raise ValueError('prospective native data requires --native-controls, not legacy G2 grading')
    if not a.thr:raise ValueError('legacy G2 requires its weak-zeroed --thr reference')
    import claim_check as cc
    import neuron_builder as nb
    o = cao.Oracle(a.adapter, a.device)
    o.text_only = a.text_only
    got = []
    for k in range(0, len(exs), 16):
        got += o.answers(exs[k:k + 16])
    nr = nb.Neurons(o.m, o.tok)
    nr.load(a.thr)
    claims, idx = [], []
    for j, g in enumerate(got):
        try:
            c = json.loads(g.split("Claim:", 1)[1].strip())
            claims.append({"prompt": {"ids": exs[j]["ctx"]}, "position": -1, "units": c["units"], "op": c["op"], "predict": c["predict"]})
            idx.append(j)
        except (IndexError, ValueError, KeyError, TypeError):
            pass
    with o.off():
        res = dict(zip(idx, cc.check_many(nr, claims)))
    by = collections.defaultdict(list)
    with a.out.open("w") as fh:
        for j, (e, g) in enumerate(zip(exs, got)):
            r = res.get(j, {})
            rec = {"cat": e["cat"], "shown": e["shown"], "q": e["q"], "a": e["a"], "got": g,
                   "same_sentence": g.split("\n")[0].strip() == e["a"].split("\n")[0].strip(),
                   "claim_pass": bool(r.get("pass")), "claim_specific": bool(r.get("specific"))}
            by[e["cat"]].append(rec)
            fh.write(json.dumps(rec) + "\n")
    for lo, hi in ((1, 8), (9, 32), (33, 128), (129, 10 ** 6)):  # by the number of units shown (the count to show is chosen
        rs = [r for v in by.values() for r in v if lo <= r["shown"] <= hi]  # by held-out accuracy, the lead's 10-15)
        if rs:
            print(f"shown {lo}-{hi if hi < 10 ** 6 else 'all'}: n={len(rs)} same sentence {sum(r['same_sentence'] for r in rs) / len(rs):.3f}  "
                  f"specific {sum(r['claim_specific'] for r in rs) / len(rs):.3f}")
    for cat, rs in sorted(by.items()):
        common = collections.Counter(r["a"].split("\n")[0].split(":")[-1] for r in rs).most_common(1)[0][1] / len(rs)
        print(f"{cat:8s} n={len(rs):4d} same sentence {sum(r['same_sentence'] for r in rs) / len(rs):.3f}  claims passing "
              f"{sum(r['claim_pass'] for r in rs) / len(rs):.3f}  specific {sum(r['claim_specific'] for r in rs) / len(rs):.3f}  "
              f"most common outcome {common:.3f}")


NATIVE_MODES=('correct','native_absent','joint_mismatch')


def _file_sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def evaluate_native(a,all_rows):
    """Collect all greedy responses WITHOUT accessing eval answer/measurement fields.

    Complete joint file bytes are parsed and hashed for provenance; gold grading
    is a separate command requiring BOTH trained models' committed predictions.
    """
    import torch
    if a.control_plan is None or a.base_snapshot is None:raise ValueError('native collection requires --control-plan and --base-snapshot')
    _,evaluation=cao._training_split(all_rows)
    binding=cao._native_training_receipt(a.items,all_rows)
    if (a.base_snapshot is None or a.base_snapshot.name!=binding['model_revision'] or
            _file_sha(a.base_snapshot/'tokenizer.json')!=binding['tokenizer_sha256'] or
            _file_sha(a.base_snapshot/'config.json')!=binding['config_sha256']):
        raise ValueError('native eval base snapshot differs from physical labels')
    complete_path=Path(str(a.out)+'.complete.json')
    if a.out.exists() or complete_path.exists():raise ValueError('new immutable prediction outputs required')
    training_path=Path(a.adapter).parent/'training_protocol.json'
    training=json.loads(training_path.read_text())
    finished=json.loads((training_path.parent/'training_complete.json').read_text())
    if (training['mode'] not in ('correct','joint_mismatch') or training['protected_eval_used_during_fit'] is not False
            or _file_sha(a.control_plan)!=training['donor_plan_sha256'] or finished['steps']!=training['steps']
            or finished['protected_eval_used'] is not False or Path(a.adapter).name!=f"step_{training['steps']}"
            or training['data_sha256']!=_file_sha(a.items) or training['base_revision']!=binding['model_revision']
            or training['tokenizer_sha256']!=binding['tokenizer_sha256'] or training['config_sha256']!=binding['config_sha256']):
        raise ValueError('fixed-step matched training protocol required')
    allowed=('cat','schema','context_id','split','document_sha256','ctx','blocks','q','native_sequence','native_binding')
    public=[{k:e[k] for k in allowed} for e in evaluation]  # never pass a/measurement into the collector
    keys=[cao.native_row_key(e) for e in public]
    if len(set(keys))!=len(keys):raise ValueError('duplicate prospective eval question')
    protocol={'schema':'native_greedy_predictions_v1','training_arm':training['mode'],'training_protocol':training,'data_sha256':_file_sha(a.items),
        'control_plan_sha256':_file_sha(a.control_plan),'adapter':cao.native_adapter_binding(a.adapter),
        'base_revision':binding['model_revision'],'expected_row_keys':keys,'input_arms':list(NATIVE_MODES),
        'cao_source_sha256':_file_sha(Path(cao.__file__)),'g2_source_sha256':_file_sha(Path(__file__)),
        'eval_gold_fields_used':False,'max_new_tokens':cao.MAX_NEW}
    Path(str(a.out)+'.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    o=cao.Oracle(str(a.adapter),a.device,model=str(a.base_snapshot));o.m.eval();start=time.time();count=0
    with a.out.open('x') as output:
        for mode in NATIVE_MODES:
            cao.configure_native_controls(o,mode,a.control_plan)
            if any(k not in o.native_control_plan['rows'] for k in keys):raise ValueError('eval question absent from frozen donor plan')
            for offset in range(0,len(public),1):  # bounded memory; one example per generation
                batch=public[offset:offset+1];inputs=[];generations=[]
                o.native_input_sink=inputs.append;o.native_generation_sink=generations.append
                answers=o.answers(batch)
                if len(answers)!=len(batch) or len(inputs)!=len(batch) or len(generations)!=len(batch):
                    raise ValueError('complete native generation receipts required')
                for e,raw,inp,gen in zip(batch,answers,inputs,generations):
                    rec={'training_arm':training['mode'],'input_arm':mode,'row_key':cao.native_row_key(e),'context_id':e['context_id'],
                        'raw_answer':raw,'truncated':gen['limit_reached'],'input_receipt':inp,'generation_receipt':gen}
                    output.write(json.dumps(rec,allow_nan=False)+'\n');output.flush();count+=1
    protocol.update({'predictions_sha256':_file_sha(a.out),'responses':count,'target_forward_rows':o.passes,
        'oracle_generation_rows':count,'runtime_seconds':time.time()-start,'complete':True})
    complete_path.write_text(json.dumps(protocol,indent=2)+'\n')


def score_native(a):
    """Only after both models' entire prediction sets commit, inspect eval gold."""
    import re
    if len(a.predictions)!=2 or a.out.exists():raise ValueError('two immutable prediction sets and new score output required')
    committed=[]
    for path in a.predictions:
        complete=json.loads(Path(str(path)+'.complete.json').read_text())
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        expected={(mode,key) for mode in NATIVE_MODES for key in complete['expected_row_keys']}
        actual=[(r['input_arm'],r['row_key']) for r in rows]
        if (complete.get('complete') is not True or complete['data_sha256']!=_file_sha(a.items)
                or complete['predictions_sha256']!=_file_sha(path) or len(rows)!=len(expected)
                or len(set(actual))!=len(rows) or set(actual)!=expected or complete['eval_gold_fields_used'] is not False
                or any(r['training_arm']!=complete['training_protocol']['mode'] or
                    r['input_receipt']['mode']!=r['input_arm'] or r['input_receipt']['row_key']!=r['row_key'] or
                    r['generation_receipt']['row_key']!=r['row_key'] or
                    r['generation_receipt']['limit_reached']!=r['truncated'] for r in rows)):
            raise ValueError('all native predictions must be frozen before scoring')
        committed.append((complete,rows))
    protocols=[c['training_protocol'] for c,_ in committed]
    if {p['mode'] for p in protocols}!={'correct','joint_mismatch'}:raise ValueError('both matched training arms required')
    if {k:v for k,v in protocols[0].items() if k!='mode'}!={k:v for k,v in protocols[1].items() if k!='mode'}:
        raise ValueError('training protocols not matched')
    if committed[0][0]['expected_row_keys']!=committed[1][0]['expected_row_keys']:raise ValueError('eval questions differ')
    if any(c['control_plan_sha256']!=c['training_protocol']['donor_plan_sha256'] for c,_ in committed):
        raise ValueError('eval control plan differs from frozen training donor plan')
    # FIRST evaluation answer/measurement field access occurs below, after BOTH
    # complete prediction sets and shared training contract have been checked.
    gold={cao.native_row_key(e):e for e in map(json.loads,a.items.read_text().splitlines()) if e.get('split')=='eval'}
    if set(gold)!=set(committed[0][0]['expected_row_keys']):raise ValueError('gold/prediction question coverage differs')
    result=[]
    for complete,rows in committed:
        for r in rows:
            e=gold[r['row_key']];raw=r['raw_answer'].strip()
            exact=raw==e['a'].strip()
            valid=not r['truncated'] and (exact or raw=='Unchanged.' or re.fullmatch(r"'[^\r\n]+'",raw) is not None)
            changed=e['measurement']['edited_top_id']!=e['measurement']['original_top_id']
            result.append({**r,'valid':bool(valid),'correct':bool(valid and exact),'changed':changed})
    summary={}
    for trained in ('correct','joint_mismatch'):
        summary[trained]={}
        for mode in NATIVE_MODES:
            rows=[r for r in result if r['training_arm']==trained and r['input_arm']==mode]
            changed=[r for r in rows if r['changed']]
            summary[trained][mode]={'n':len(rows),'correct':sum(r['correct'] for r in rows),'valid':sum(r['valid'] for r in rows),
                'accuracy':sum(r['correct'] for r in rows)/len(rows),'changed_n':len(changed),
                'changed_correct':sum(r['correct'] for r in changed),'unchanged_baseline_accuracy':sum(not r['changed'] for r in rows)/len(rows)}
    primary_gap=summary['correct']['correct']['accuracy']-summary['correct']['native_absent']['accuracy']
    a.out.write_text(json.dumps({'schema':'native_causal_accuracy_v1','primary':'raw greedy exact canonical causal answer; invalid/truncated incorrect',
        'primary_correct_minus_native_absent_accuracy':primary_gap,
        'summary':summary,'rows':result,'scope':'prospective original-model all-position native removals; no free-prose or general-mechanism certificate'},indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--thr", type=Path, required=True, help="the weak-zeroed reference (neuron_builder.Neurons.save)")
    g.add_argument("--texts", type=int, default=100)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--split", default="train", choices=("train", "test"), help="wikitext-103's split (test: evaluation texts)")
    g.add_argument("--records", type=Path, help="circuits built elsewhere: a directory of neuron_builder records with ids")
    g.add_argument("--device")
    g.add_argument("--necessity-only", action="store_true", help="the builder's necessity set alone (its sufficiency search "
                                                                  "runs in float64, which MPS lacks)")
    e = sub.add_parser("eval")
    e.add_argument("--items", type=Path, required=True, help="gen --split test's examples")
    e.add_argument("--thr", type=Path, help="required for legacy weak-zeroed G2 evaluation")
    e.add_argument("--adapter", default=cao.AO)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--text-only", action="store_true")
    e.add_argument("--device")
    e.add_argument('--native-controls',action='store_true',help='prospective native greedy collection only, no gold grading')
    e.add_argument('--base-snapshot',type=Path)
    e.add_argument('--control-plan',type=Path)
    ns=sub.add_parser('score-native')
    ns.add_argument('--items',type=Path,required=True);ns.add_argument('--predictions',type=Path,nargs='+',required=True);ns.add_argument('--out',type=Path,required=True)
    a = ap.parse_args()
    {"gen": gen, "eval": evaluate,'score-native':score_native}[a.cmd](a)


if __name__ == "__main__":
    main()
