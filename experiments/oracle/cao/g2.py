"""The neuron-grounded oracle's examples (#2951 G2; the lead's 10-15 direction, decision 6, and its 8B update). Beside a
text's half-depth activations (the Activation Oracle's read), the oracle reads the neurons of a question-scoped circuit
(method/oracle/neuron_builder.py: the answer contrast on the weak-zeroed reference, a unit one neuron's value at one
position) in measured order: the circuit's units, then the builder's strongest measured singles beyond it; a random
number of them from MIN_SHOWN up to all, so that how many to show is chosen afterwards by held-out accuracy (the lead's
10-15: no fixed cap). Each shows as its name, "L<layer>.N<neuron>+@<position>" ("-" for the negative pole: claim_check's
unit syntax), and its write (its value x its down_proj column, cao.Oracle.write). The questions, every answer measured
by method/oracle/claim_check.py, the oracle's reward checker, so that the claim an answer ends with passes there:

  n_steer  what turning one shown neuron up (STEER x its calibration maximum, along its pole) does to the next token
  n_zero   what zeroing one shown neuron does
  n_set    what zeroing two to four shown neurons together does
  n_sign   whether a shown neuron supports the next token or works against it (its zeroing's sign)

An answer is one English sentence and a line "Claim: {units, op, predict}" (the prompt and position are the text's own
and the answer position); a claim of an effect that random neurons of the same layers also produce is said to be so
(claim_check's "specific"). Texts: wikitext-103 windows of h2h.T tokens (the answer the model's next token after the
window); --records: circuits built elsewhere (neuron_builder's record format, one JSON file per text with "ids").

  g2.py gen --out EX.jsonl --thr THR.pt [--texts N] [--seed S] [--split test] [--necessity-only] [--records DIR]
  g2.py eval --items EX.jsonl --adapter A --out OUT.jsonl [--text-only]   graded answers; the stated claims checked
"""
from __future__ import annotations

import argparse
import json
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
    rec = rec or nb.build(nr, ids, pos)
    order = list({name(d): d for d in rec["nodes"] + rec["ranked_top"]}.values())  # circuit first, then measured singles
    if not order:
        return [], {"size": 0}
    shown = order[:rng.randint(min(MIN_SHOWN, len(order)), len(order))]
    names = [name(d) for d in shown]
    blocks = [{"layer": cao.MID, "positions": list(range(len(ids)))}] + [{"label": n, "write": [d["layer"], d["neuron"], d["act"]]}
                                                                          for n, d in zip(names, shown)]
    y = cao.q_tok(o.tok.decode([rec["answer"]]))
    asks = []
    for cat, k in QUESTIONS:
        for _ in range(k):
            us = rng.sample(names, rng.randint(2, min(4, len(names)))) if cat == "n_set" else [rng.choice(names)]
            if cat == "n_set" and len(names) < 2:
                continue
            op = {"steer": STEER} if cat == "n_steer" else "zero"
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
    return out, {"size": rec["size"], "met": rec["met"], "shown": len(shown), "questions": len(out), "build_seconds": rec.get("seconds")}


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
    with a.out.open("a") as fh:
        for n, r in enumerate(src):
            if n >= a.texts:
                break
            key = 1_000_000 * a.seed + n
            if key in done:
                continue
            exs, info = text_examples(o, nr, r["ids"], rng, r if "nodes" in r else None)
            for e in exs:
                fh.write(json.dumps({**e, "text": key}) + "\n")
            fh.flush()
            if (n + 1) % 10 == 0:
                print(json.dumps({"texts": n + 1, "seconds": round(time.time() - t0), **info}), flush=True)
    print(f"{a.texts} texts to {a.out}")


def evaluate(a) -> None:
    """The oracle's greedy answers to held-out examples (gen --split test), graded on the English sentence's outcome and
    on its claim, the claim re-checked by claim_check on the reference (pass and specific); --text-only: no vector
    added (the control). Printed beside: the most common answer per category."""
    import collections

    import claim_check as cc
    import neuron_builder as nb
    exs = [json.loads(x) for x in a.items.read_text().splitlines()]
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
    for cat, rs in sorted(by.items()):
        common = collections.Counter(r["a"].split("\n")[0].split(":")[-1] for r in rs).most_common(1)[0][1] / len(rs)
        print(f"{cat:8s} n={len(rs):4d} same sentence {sum(r['same_sentence'] for r in rs) / len(rs):.3f}  claims passing "
              f"{sum(r['claim_pass'] for r in rs) / len(rs):.3f}  specific {sum(r['claim_specific'] for r in rs) / len(rs):.3f}  "
              f"most common outcome {common:.3f}")


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
    e.add_argument("--thr", type=Path, required=True)
    e.add_argument("--adapter", default=cao.AO)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--text-only", action="store_true")
    e.add_argument("--device")
    a = ap.parse_args()
    {"gen": gen, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
