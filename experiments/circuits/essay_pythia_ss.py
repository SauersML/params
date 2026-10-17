"""The essay's Pythia-1B-v0 "Social Security" check (appendix "Social Security: Detection and Use"): each named neuron
removed (its MLP activation, the input of dense_4h_to_h, zeroed at every position), the summed change in the target's
log probability (nats, zeroed minus clean) over a seven-prompt battery, beside the essay's numbers; and the three
detector neurons removed together, the change of the next-token distribution (KL, nats) at the phrase's last token.
The essay does not print its battery; ours follows its neuron notes (number after Your Social Security, Your Library
Card, National Insurance; benefits; a retirement-program prompt whose answer is Social), so signs compare and sizes
only roughly.

  essay_pythia_ss.py OUT.json [--model EleutherAI/pythia-1b-v0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ESSAY = {(0, 3308): 0.49, (2, 7715): -0.24, (2, 611): 0.02, (3, 839): 0.58, (5, 447): 0.22, (10, 3382): -0.24,
         (12, 7260): 0.40, (13, 3673): -0.61, (15, 3731): -0.81, (15, 7040): -0.75, (15, 5759): 0.69}
DETECTORS = [(2, 611), (2, 4081), (2, 6819)]
BATTERY = [("Your Social Security", " number"), ("Please enter your Social Security", " number"),
           ("Your Library Card", " number"), ("Your National Insurance", " number"),
           ("You can apply for Social Security", " benefits"), ("He started collecting his Social Security", " benefits"),
           ("The main federal retirement program in the United States is", " Social")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="EleutherAI/pythia-1b-v0")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval().requires_grad_(False)
    layers = model.gpt_neox.layers

    def run(ids, gone=()):
        by = {}
        for i, j in gone:
            by.setdefault(i, []).append(j)
        hs = [layers[i].mlp.dense_4h_to_h.register_forward_pre_hook(
            lambda mod, args, js=js: (args[0].index_fill(-1, torch.tensor(js), 0.0),)) for i, js in by.items()]
        try:
            with torch.no_grad():
                return torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0, -1].float(), -1)
        finally:
            for h in hs:
                h.remove()
    items = [(tok(p)["input_ids"], tok(t)["input_ids"][0]) for p, t in BATTERY]
    base = [run(ids)[y] for ids, y in items]
    rows = []
    for u, theirs in ESSAY.items():
        per = [float(run(ids, [u])[y] - b) for (ids, y), b in zip(items, base)]
        rows.append({"neuron": f"L{u[0]}:{u[1]}", "ours": sum(per), "essay": theirs, "per_prompt": per})
        print(f"L{u[0]}:{u[1]}: ours {sum(per):+.3f} nats, essay {theirs:+.2f}", flush=True)
    ids = tok("Your Social Security")["input_ids"]
    lp0, lp1 = run(ids), run(ids, DETECTORS)
    kl = float((lp0.exp() * (lp0 - lp1)).sum())
    dy = float(lp1[tok(" number")["input_ids"][0]] - lp0[tok(" number")["input_ids"][0]])
    agree = sum((r["ours"] > 0) == (r["essay"] > 0) for r in rows if abs(r["essay"]) >= 0.1)
    print(f"detectors together: KL {kl:.4f} nats, d log p(number) {dy:+.4f} (essay: -0.004 nats on the next token); "
          f"signs agree on {agree} of {sum(abs(r['essay']) >= 0.1 for r in rows)}", flush=True)
    a.out.write_text(json.dumps({"model": a.model, "battery": BATTERY, "neurons": rows,
                                 "detectors": {"kl_nats": kl, "d_logp_number": dy}, "sign_agreement": agree}))


if __name__ == "__main__":
    main()
