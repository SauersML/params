"""Circuit questions re-measured in float32 (#2951; primerl's 10-16 finding: claim_check in bfloat16 moves a claim's
change with the batch it runs in, a median 0.14 bits between 32 and 64 rows, 21% of verdicts, against 0.25-bit class
lines; O1's gold classes reproduce at 70% in float32). Each question's claim op on its units (a group's name standing
for its members) is measured on a float32 model with the weak-zeroed reference, its answer and claim rewritten by g2.py's
rules (the sentence, "Random neurons of the same layers do the same." when CONTROLS random sets also pass, the claim
line), and the measurement kept beside the question:
  "fp32": {"change", "class" (reads_circuit's classes), "class_before" (the question's own), "margin" (bits from the
           nearest class line), "top_before", "top_after", "specific"}
The answer token y is the one the question names (an fp32 top token that differs is recorded, top_before).

  relabel.py QUESTIONS.jsonl [QUESTIONS.jsonl ...] --thr THR.pt --out-dir DIR [--controls 2] [--model Qwen/Qwen3-8B]
      (Qwen3-8B in float32: a 48 GB card) DIR/<file name>: the questions, relabelled, in their order
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
import g2  # noqa: E402
from reads_circuit import answer_class  # noqa: E402



def rewrite(e: dict, r: dict, specific: bool) -> tuple[str, dict]:
    """g2.text_examples' answer for a measured result: (answer, claim)."""
    us, op, cat = e["claim"]["units"], e["claim"]["op"], e["cat"]
    y = g2.cao.q_tok(r["top_before"])
    if f"next token {y}" not in e["q"]:  # (float32's top token differs from the one the question names)
        y = re.search(r"next token ('(?:[^'\\]|\\.)*'|\"[^\"]*\")", e["q"]).group(1)
    if cat == "n_sign":
        ch = r["change"]
        pred = {"contrast_change": {"op": "<=", "value": -g2.HALF}} if ch <= -g2.HALF else \
            {"contrast_change": {"op": ">=", "value": g2.HALF}} if ch >= g2.HALF else {"contrast_change": {"op": "within", "value": g2.HALF}}
        text = (f"{us[0]} supports {y}." if ch <= -g2.HALF else f"{us[0]} works against {y}." if ch >= g2.HALF else
                f"{us[0]} barely matters for {y}.")
    else:
        clause, pred = g2.outcome(r, y)
        verb = "Turning " + us[0] + " up" if cat == "n_steer" else "Zeroing " + g2.listing(us) + (" together" if len(us) > 1 else "")
        text = f"{verb}: {clause}."
    if not specific:
        text += " Random neurons of the same layers do the same."
    claim = {"units": us, "op": op, "predict": pred}
    return f"{text}\nClaim: {json.dumps(claim)}", claim


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("questions", type=Path, nargs="+")
    ap.add_argument("--thr", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--controls", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import claim_check as cc
    import neuron_builder as nb
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    m.requires_grad_(False)
    nr = nb.Neurons(m, tok)
    nr.load(a.thr)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    expand = lambda e: [u for n in e["claim"]["units"] for u in e.get("groups", {}).get(n, [n])]  # noqa: E731
    for f in a.questions:
        qs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        base = [{"prompt": {"ids": e["ctx"]}, "position": -1, "units": expand(e), "op": e["claim"]["op"],
                 "predict": {"top_changes": True}} for e in qs]
        with torch.no_grad():
            got = cc.check_many(nr, base, controls=0)  # the change, then the claim it states with its controls
            new = [rewrite(e, r, True) if "error" not in r else (None, None) for e, r in zip(qs, got)]
            fin = cc.check_many(nr, [{**b, "predict": c["predict"]} for b, (_, c) in zip(base, new) if c is not None],
                                controls=a.controls)
        fin = iter(fin)
        with (a.out_dir / f.name).open("w") as fh:
            for e, r, (ans, claim) in zip(qs, got, new):
                if ans is None:
                    fh.write(json.dumps({**e, "fp32": {"error": r["error"]}}) + "\n")
                    continue
                s = next(fin)
                ans, claim = rewrite(e, r, bool(s.get("specific")))
                was = answer_class(e)
                e2 = {**e, "a": ans, "claim": claim}
                fh.write(json.dumps({**e2, "fp32": {"change": r["change"], "class": answer_class(e2), "class_before": was,
                                                    "margin": abs(abs(r["change"]) - g2.HALF), "top_before": r["top_before"],
                                                    "top_after": r["top_after"], "specific": bool(s.get("specific")),
                                                    "pass": bool(s.get("pass"))}}) + "\n")
        print(f.name, len(qs), flush=True)


if __name__ == "__main__":
    main()
