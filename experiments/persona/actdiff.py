"""A second candidate persona set, from activations alone (no circuit search): neurons whose values at the decision
position differ most between the assistant and the character prompts of contrast.py's valid train pairs (HF, GPU).

Per neuron (layer, index) the value act(gate) x up at the last position of each prompt (the decision token's
position), and the standardized mean difference over pairs: t = mean(clean - corrupt) / (sd(clean - corrupt) /
sqrt(pairs)). The top K by |t| make actdiff_K, with random sets matched in layer, in aggregate.py's format.

  actdiff.py DECISIONS.json OUT.json [--model Qwen/Qwen3-8B]
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import torch

SIZES = (16, 32, 64, 128, 256, 512)
WIDTH = 12288


def main():
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("decisions")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = [d for d in json.loads(Path(a.decisions).read_text()) if d["valid"] and d["split"] == "train"]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    L = model.config.num_hidden_layers
    cur = {}

    def hook(i):
        def f(mod, args):
            cur[i] = args[0][0, -1].float()
        return f
    hs = [model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook(i)) for i in range(L)]
    s1 = torch.zeros(L, WIDTH, device="cuda", dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    with torch.no_grad():
        for d in ds:
            vals = []
            for ids in (d["clean_ids"], d["corrupt_ids"]):
                model(input_ids=torch.tensor([ids], device="cuda"))
                vals.append(torch.stack([cur[i] for i in range(L)]).double())
            diff = vals[0] - vals[1]
            s1 += diff
            s2 += diff * diff
    for h in hs:
        h.remove()
    n = len(ds)
    mean = s1 / n
    sd = (s2 / n - mean ** 2).clamp_min(1e-12).sqrt() * (n / (n - 1)) ** 0.5
    t = (mean / (sd / n ** 0.5)).cpu()
    order = torch.argsort(t.abs().flatten(), descending=True)[:max(SIZES)].tolist()
    ranked = [(i // WIDTH, i % WIDTH) for i in order]
    rng = random.Random(a.seed)
    taken = set(ranked)
    sets = {}
    for K in SIZES:
        p = ranked[:K]
        sets[f"actdiff_{K}"] = [list(u) for u in p]
        rnd = []
        for l, c in sorted(collections.Counter(l for l, _ in p).items()):
            pool = [k for k in range(WIDTH) if (l, k) not in taken]
            rnd += [[l, k] for k in sorted(rng.sample(pool, c))]
        sets[f"actdiff_random_{K}"] = rnd
    doc = {"pairs": n, "top": [[l, k, round(float(t[l, k]), 2), round(float(mean[l, k]), 3)] for l, k in ranked[:64]],
           "layers_top128": dict(sorted(collections.Counter(l for l, _ in ranked[:128]).items())), "sets": sets}
    Path(a.out).write_text(json.dumps(doc))
    print(json.dumps({k: v for k, v in doc.items() if k != "sets"})[:2000])


if __name__ == "__main__":
    main()
