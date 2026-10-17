"""Circuit questions with measured answers (#2951 G2; the lead's 10-15 direction and its 10-16 family-level input):
questions about the units of a text's question-scoped circuit (method/oracle/circuit_finder.py's records, the patching
pair; circuit_family's; or neuron_motifs apply's, with shared groups), every answer measured by
method/oracle/claim_check.py, the oracle's reward checker, so that the claim an answer ends with passes there. The text
reader (text_reader.py) reads the circuit as text and answers them (rc's questions, reads_circuit.py's split; its own
from text_reader.py gen, with this file's outcome and listing).

A unit is a mover head at the answer position ("L<l>.H<h>@<p>"), a named neuron group ("G<id>", shared across texts;
its name stands for its members in claims) or a single neuron's value at one position ("L<layer>.N<neuron>+@<position>",
"-" for the negative pole: claim_check's unit syntax). The units asked about: heads first, then groups by stage (the
family's order, narrowest first), a log-uniform number of them from MIN_SHOWN up to all; in the form groups+neurons also
the single neurons of the groups that split (FORMS); dense: every group, no heads, every question about each.

  n_steer  what turning one unit up (STEER x its calibration maximum, along its pole) does to the next token
  n_zero   what zeroing one unit does
  n_set    what zeroing two to four units together does
  n_sign   whether a unit supports the next token or works against it (its zeroing's sign)

An answer is one English sentence and a line "Claim: {units, op, predict}" (the prompt and position are the text's own
and the answer position); a claim of an effect that random neurons of the same layers also produce is said to be so
(claim_check's "specific"). Texts: the records' own (each with "ids"); one record in ten, by id, is held out.

  g2.py gen --out EX.jsonl --thr THR.pt --records DIR [--texts N] [--seed S] [--dense] [--groups DIR]   (GPU; held-out
      records to EX.heldout.jsonl)
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
    form = "groups" if dense else rng.choice(FORMS)
    heads = [(h, [h]) for h in (f"L{r['layer']}.H{r['head']}@{r['pos']}" for r in ([] if dense else rec.get("routes", [])))]
    gs = groups_of(rec, order)
    gitems, single = {}, []
    for k, g in enumerate(gs, 1):
        mem = [name(d) for d in g["members"]]
        gname = f"G{g['id']}" if g.get("id") is not None else f"G{k}"
        stage = collections.Counter(d.get("stage") for d in g["members"]).most_common(1)[0][0]
        gitems.setdefault(stage, []).append((gname, mem))
        if form == "groups+neurons" and (g.get("split") or "groups" not in rec):  # single neurons where a group splits
            single += [(name(d), [name(d)]) for d in g["members"]]
    stages = rec.get("order") or ["movers", "source", "readout"]  # the family's stages, narrowest first (circuit_family)
    seq = [x for st in stages for x in (heads if st == "movers" else gitems.get(st, []))]
    seq += [x for st, v in gitems.items() if st not in stages for x in v]
    items = seq if dense else seq[:loguniform(rng, len(seq))] if seq else []
    shown = {n for n, _ in items}
    items += [x for x in single if x[0] not in shown]
    expand = dict(items)
    names = list(expand)
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
        out.append({"cat": cat, "form": form, "ctx": ids, "q": q, "a": f"{text}\nClaim: {json.dumps(claim)}",
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
    a = ap.parse_args()
    {"gen": gen}[a.cmd](a)


if __name__ == "__main__":
    main()
