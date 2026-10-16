"""The neuron-grounded oracle's examples (#2951 G2; the lead's 10-15 direction, decision 6, and its 8B update). Beside a
text's half-depth activations (the Activation Oracle's read), the oracle reads the neurons of the text's question-scoped
circuit (method/oracle/circuit_finder.py's records, the patching pair: a unit is one neuron's value at one position, its
neurons in measured order by the larger of their lost and restored shares; or neuron_motifs apply's records of them, with
shared groups). A random number of them is shown, log-uniform from MIN_SHOWN up to all, so that how many to show is
chosen by held-out accuracy (the lead's 10-15: no fixed cap; eval reports accuracy by the number shown). The input is
the family-level circuit (the lead's 10-16): the mover heads first ("L<l>.H<h>@<p> (looks at '<token>' @q)", the head's
output there as its vector), then named neuron groups (shared across texts, "G<id>" everywhere, varteach's label where it
passed its tests; a group's write the sum of its members'; for records without groups, a stand-in of one layer and
position); in the form groups+neurons also the single neurons of the groups that split (FORMS). A group's name stands
for its members in the oracle's claims; heads are zeroed, never steered.
A neuron shows as its name, "L<layer>.N<neuron>+@<position>" ("-" for the negative pole: claim_check's unit syntax), and
its write (its value x its down_proj column, cao.Oracle.write). The questions, every answer measured by
method/oracle/claim_check.py, the oracle's reward checker, so that the claim an answer ends with passes there:

  n_steer  what turning one shown unit up (STEER x its calibration maximum, along its pole) does to the next token
  n_zero   what zeroing one shown unit does
  n_set    what zeroing two to four shown units together does
  n_sign   whether a shown unit supports the next token or works against it (its zeroing's sign)

An answer is one English sentence and a line "Claim: {units, op, predict}" (the prompt and position are the text's own
and the answer position); a claim of an effect that random neurons of the same layers also produce is said to be so
(claim_check's "specific"). Texts: the records' own (each with "ids"); one record in ten, by id, is held out.

  g2.py gen --out EX.jsonl --thr THR.pt --records DIR [--texts N] [--seed S]   (GPU; held-out records to EX.heldout.jsonl)
  g2.py eval --items EX.jsonl --adapter A --out OUT.jsonl [--text-only]   graded answers; the stated claims checked
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
import cao  # noqa: E402

STEER = 3.0     # claim_check's steer scale: x the neuron's calibration maximum
MIN_SHOWN = 4
HALF = 0.25     # bits: a change smaller than this is "almost none"
ANSWER_TOKENS = 160  # an answer's sentence and its claim line (cao's 24 cut every claim off in the first 8B smoke)
QUESTIONS = (("n_steer", 2), ("n_zero", 1), ("n_set", 1), ("n_sign", 1))
FORMS = ("groups", "groups+neurons")  # the input's forms (the lead's 10-16: the family-level circuit, mover heads and named
# groups; single neurons only where a group splits; they barely move outputs alone), chosen by held-out accuracy


def name(d: dict) -> str:
    return f"L{d['layer']}.N{d['neuron']}{'+' if d['pole'] > 0 else '-'}@{d['pos']}"


def measured_order(rec: dict) -> list[dict]:
    """A record's neurons in measured order: circuit_finder's (the patching pair) by the larger of each neuron's lost and
    restored shares alone."""
    if "nodes" not in rec:  # neuron_motifs apply's records: the groups' members in the groups' measured order, then the rest
        return [d for g in rec.get("groups", []) for d in g["members"]] + rec.get("ungrouped", [])
    return sorted(rec["nodes"], key=lambda d: -max(abs(d.get("lost") or 0), abs(d.get("restored") or 0)))


def value(d: dict) -> float:
    return d["clean"] if "clean" in d else d["value"]  # (the finder and motifs record the clean run's value)


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
    if abs(ch) < HALF:  # (two-sided: top_changes false passed on most items whatever the change, rewarding a no-change prior)
        return f"the model still gives {y}, almost unchanged", {"contrast_change": {"op": "within", "value": HALF}}
    v = round(ch * 2) / 2
    return (f"the model still gives {y}, its lead over the alternatives {'up' if ch > 0 else 'down'} about {bits(ch)}",
            {"contrast_change": {"op": ">=" if ch > 0 else "<=", "value": v - 0.25 if ch > 0 else v + 0.25}})


def groups_of(rec: dict, order: list[dict]) -> list[dict]:
    """A record's measured neuron groups in their order ({"members": [neuron dicts], "label"}: inframove's, labelled by
    varteach), else a stand-in until those land: the neurons of one layer and position, in measured order."""
    if rec.get("groups"):  # shared across texts: a group's id is its name everywhere, its label written once
        return [{"members": g["members"], "label": g.get("label"), "label_passed": g.get("label_passed", bool(g.get("label"))),
                 "id": g.get("id"), "split": g.get("split", False)} for g in rec["groups"]]
    by: dict = {}
    for d in order:
        by.setdefault((d["layer"], d["pos"]), []).append(d)
    return [{"members": m, "label": None} for m in by.values()]


def loguniform(rng: random.Random, n: int) -> int:
    """A count log-uniform from MIN_SHOWN (or n, when fewer) to n: uniform counts up to all made 8B prompts 2,734 tokens on
    average (a shown neuron is ~16 tokens)."""
    lo = min(MIN_SHOWN, n)
    return round(math.exp(rng.uniform(math.log(lo), math.log(n))))


def text_examples(o: "cao.Oracle", nr, ids: list[int], rng: random.Random, rec: dict, dense: bool = False) -> tuple[list[dict], dict]:
    """dense (the circuit-reading test's): every group shown, no heads (a head's vector depends on the text, so another
    text's could not stand in for it), and every question about every shown unit."""
    import claim_check as cc
    order = measured_order(rec)
    if not order:
        return [], {"size": 0}
    write = lambda d: [d["layer"], d["neuron"], value(d)]  # noqa: E731
    form = "groups" if dense else rng.choice(FORMS)
    heads = []
    for r in ([] if dense else rec.get("routes", [])):  # a mover head at the answer position, with where it looks
        hn = f"L{r['layer']}.H{r['head']}@{r['pos']}"
        looks = ", ".join(f"{cao.q_tok(o.tok.decode([ids[q]]))} @{q}" for q, *_ in (r.get("attends") or [])[:2])
        heads.append((hn, [hn], {"label": hn + (f" (looks at {looks})" if looks else ""), "unit": ["head", r["layer"], r["head"]]}))
    gs = groups_of(rec, order)
    gitems, single = {}, []
    for k, g in enumerate(gs, 1):
        mem = [name(d) for d in g["members"]]
        gname = f"G{g['id']}" if g.get("id") is not None else f"G{k}"
        lab = gname + (f" ({g['label']})" if g.get("label") and g.get("label_passed") else "") + \
            f", {len(mem)} neuron{'' if len(mem) == 1 else 's'}"
        stage = collections.Counter(d.get("stage") for d in g["members"]).most_common(1)[0][0]
        gitems.setdefault(stage, []).append((gname, mem, {"label": lab, "write": [write(d) for d in g["members"]]}))
        if form == "groups+neurons" and (g.get("split") or "groups" not in rec):  # single neurons where a group splits
            single += [(name(d), [name(d)], {"label": name(d), "write": write(d)}) for d in g["members"]]
    stages = rec.get("order") or ["movers", "source", "readout"]  # the family's stages, narrowest first (circuit_family)
    seq = [x for st in stages for x in (heads if st == "movers" else gitems.get(st, []))]
    seq += [x for st, v in gitems.items() if st not in stages for x in v]
    items = seq if dense else seq[:loguniform(rng, len(seq))] if seq else []
    shown = {n for n, _, _ in items}
    items += [x for x in single if x[0] not in shown]
    expand = {n: us for n, us, _ in items}
    names = list(expand)
    blocks = [{"layer": cao.MID, "positions": list(range(len(ids)))}] + [b for _, _, b in items]
    y = cao.q_tok(o.tok.decode([rec["answer"]]))
    asks = []
    if dense:
        asks = [(cat, [n], {"steer": STEER} if cat == "n_steer" else "zero") for n in names for cat in ("n_zero", "n_sign", "n_steer")]
        for _ in range(2 if len(names) >= 2 else 0):
            us = rng.sample(names, rng.randint(2, min(4, len(names))))
            if ("n_set", us) not in [(c_, u_) for c_, u_, _ in asks]:
                asks.append(("n_set", us, "zero"))
    for cat, k in ([] if dense else QUESTIONS):
        for _ in range(k):
            if cat == "n_set" and len(names) < 2:
                continue
            pool = [x for x in names if ".H" not in x] if cat == "n_steer" else names  # (steering takes neurons, not heads)
            if not pool:
                continue
            us = rng.sample(names, rng.randint(2, min(4, len(names)))) if cat == "n_set" else [rng.choice(pool)]
            op = {"steer": STEER} if cat == "n_steer" else "zero"
            if (cat, tuple(us)) not in {(c_, tuple(u_)) for c_, u_, _ in asks}:  # (a small circuit: one question per unit)
                asks.append((cat, us, op))
    units = lambda us: [u for n in us for u in expand[n]]  # noqa: E731  (a group's name stands for its members)
    probe = [{"prompt": {"ids": ids}, "position": -1, "units": units(us), "op": op, "predict": {"top_changes": True}} for _, us, op in asks]
    got = cc.check_many(nr, probe)
    claims, texts = [], []
    for (cat, us, op), r in zip(asks, got):
        if "error" in r:
            continue
        if cat == "n_sign":
            ch = r["change"]  # zeroing: a supporting unit's removal lowers the contrast
            pred = {"contrast_change": {"op": "<=", "value": -HALF}} if ch <= -HALF else \
                {"contrast_change": {"op": ">=", "value": HALF}} if ch >= HALF else {"contrast_change": {"op": "within", "value": HALF}}
            text = (f"{us[0]} supports {y}." if ch <= -HALF else f"{us[0]} works against {y}." if ch >= HALF else
                    f"{us[0]} barely matters for {y}.")
        else:
            clause, pred = outcome(r, y)
            verb = "Turning " + us[0] + " up" if cat == "n_steer" else "Zeroing " + listing(us) + (" together" if len(us) > 1 else "")
            text = f"{verb}: {clause}."
        claims.append({"prompt": {"ids": ids}, "position": -1, "units": units(us), "op": op, "predict": pred})
        texts.append((cat, us, op, text, pred))
    final = cc.check_many(nr, claims)  # the stated claims themselves, with their controls
    out = []
    groups = {n: us for n, us in expand.items() if n.startswith("G")}
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
        out.append({"cat": cat, "form": form, "ctx": ids, "blocks": blocks, "q": q, "a": f"{text}\nClaim: {json.dumps(claim)}",
                    "claim": claim, "groups": groups, "specific": r["specific"], "shown": len(items), "circuit": len(order)})
    return out, {"size": rec.get("size"), "met": rec.get("met"), "form": form, "shown": len(items), "questions": len(out),
                 "build_seconds": rec.get("seconds")}


def gen(a) -> None:
    import neuron_builder as nb
    o = cao.Oracle(None, a.device)
    nr = nb.Neurons(o.m, o.tok)
    nr.load(a.thr)
    rng = random.Random(a.seed)
    done = {json.loads(x)["text"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    def load(f):  # a family record, with its shared groups from apply/ (same file name) when given
        r = json.loads(f.read_text())
        if a.groups and (a.groups / f.name).exists():
            g = json.loads((a.groups / f.name).read_text())
            r["groups"] = g.get("groups", [])
        return r
    src = (load(f) for f in sorted(a.records.glob("*.json")))
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
            exs, info = text_examples(o, nr, r["ids"], rng, r, dense=a.dense)
            dst = hh if zlib.crc32(str(r.get("id", n)).encode()) % 10 == 0 else fh
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
        got += o.answers(exs[k:k + 16], max_new=ANSWER_TOKENS)
    nr = nb.Neurons(o.m, o.tok)
    nr.load(a.thr)
    claims, idx = [], []
    for j, g in enumerate(got):
        try:
            c = json.loads(g.split("Claim:", 1)[1].strip())
            us = [u for n in c["units"] for u in exs[j].get("groups", {}).get(n, [n])]  # a group's name stands for its members
            claims.append({"prompt": {"ids": exs[j]["ctx"]}, "position": -1, "units": us, "op": c["op"], "predict": c["predict"]})
            idx.append(j)
        except (IndexError, ValueError, KeyError, TypeError):
            pass
    with o.off():
        res = dict(zip(idx, cc.check_many(nr, claims)))
    by = collections.defaultdict(list)
    with a.out.open("w") as fh:
        for j, (e, g) in enumerate(zip(exs, got)):
            r = res.get(j, {})
            rec = {"cat": e["cat"], "form": e.get("form", "neurons"), "shown": e["shown"], "q": e["q"], "a": e["a"], "got": g,
                   "same_sentence": g.split("\n")[0].strip() == e["a"].split("\n")[0].strip(),
                   "claim_pass": bool(r.get("pass")), "claim_specific": bool(r.get("specific"))}
            by[e["cat"]].append(rec)
            fh.write(json.dumps(rec) + "\n")
    for lo, hi in ((1, 8), (9, 32), (33, 128), (129, 10 ** 6)):  # by the number of units shown (the count to show is chosen
        rs = [r for v in by.values() for r in v if lo <= r["shown"] <= hi]  # by held-out accuracy, the lead's 10-15)
        if rs:
            print(f"shown {lo}-{hi if hi < 10 ** 6 else 'all'}: n={len(rs)} same sentence {sum(r['same_sentence'] for r in rs) / len(rs):.3f}  "
                  f"specific {sum(r['claim_specific'] for r in rs) / len(rs):.3f}")
    for form in FORMS:
        rs = [r for v in by.values() for r in v if r["form"] == form]
        if rs:
            print(f"form {form}: n={len(rs)} same sentence {sum(r['same_sentence'] for r in rs) / len(rs):.3f}  "
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
    g.add_argument("--records", type=Path, required=True, help="circuit records (circuit_finder's or circuit_family's, or neuron_motifs apply's), one JSON file a text")
    g.add_argument("--dense", action="store_true", help="every group, no heads, every question per unit (the circuit-reading test)")
    g.add_argument("--groups", type=Path, help="neuron_motifs apply's directory for the records (shared groups, joined by file name)")
    g.add_argument("--device")
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
