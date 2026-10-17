"""Automatic attention routes against the essay's three attention figures (Cammarata 2026; prompts, positions and named
heads from essay_replication.ATTENTION): does knocking out each head alone (head_routes.carriers, every head attending
from the reading token to the source token at least MIN_ATT unable to see it) find the heads the essay names, and do the
heads it ranks first move the target neuron as much as the essay's set?
Per figure, measured on the target neuron's value at the reading token: each head alone; the top k of that ranking
together (k = the number of heads the essay names); the essay's heads together; every head in the essay's layer range
together (where the essay gives one).

  python experiments/circuits/essay_routes.py OUT.json [--model Qwen/Qwen3-32B] [--dtype bfloat16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
import essay_replication as er  # noqa: E402
import head_routes  # noqa: E402

MIN_ATT = 0.01


def figure(rn: head_routes.Runner, tok, c: dict) -> dict:
    ids = tok(c["prompt"], add_special_tokens=False)["input_ids"]
    dest = c["dest"][0] if tok.decode([ids[c["dest"][0]]]) == c["dest"][1] else er.find(tok, ids, c["dest"][1])
    src = c["source"][0] if tok.decode([ids[c["source"][0]]]) == c["source"][1] else er.find(tok, ids, c["source"][1], dest)
    L, j = c["target"]
    watch = [(L, dest, j)]
    run = rn.run(ids, [{}], capture=True, watch=watch)
    st, base = run["states"], float(run["watch"][0, 0])

    def value(sets):
        return rn.run(ids, sets, watch=watch)["watch"][:, 0].tolist()
    heads = [(lay, h) for lay in range(L + 1) for h in range(rn.H) if float(st["A"][lay][h, dest, src]) >= MIN_ATT]
    alone = value([{"knock": [(lay, h, dest, src)]} for lay, h in heads])
    rank = sorted(zip(heads, alone), key=lambda z: z[1] if base >= 0 else -z[1])  # (largest drop toward zero first)
    essay = [b["block"] for b in c["checks"] if "block" in b and not b["control"] and "heads" in b["block"]]
    named = [u for u in (er.unit(h) for h in essay[0]["heads"]) if u[0] < rn.L and u[1] < rn.H] if essay else []
    k = max(len(named), 1)
    sets = [{"knock": [(lay, h, dest, src) for (lay, h), _ in rank[:k]]}]
    if named:
        sets.append({"knock": [(lay, h, dest, src) for lay, h in named]})
    span = [b["block"]["layers"] for b in c["checks"] if "block" in b and not b["control"] and "layers" in b["block"]]
    if span:
        lo, hi = span[0]
        sets.append({"knock": [(lay, h, dest, src) for lay in range(lo, hi + 1) for h in range(rn.H)]})
    together = value(sets)
    named_set = set(named)
    return {"source": src, "dest": dest, "target": [L, j], "value": base,
            "heads_alone": [{"head": f"L{lay}H{h}", "attention": float(st["A"][lay][h, dest, src]), "value": v,
                             "named": (lay, h) in named_set} for (lay, h), v in rank[:max(3 * k, 15)]],
            "top_k_together": together[0], "named_together": together[1] if named else None,
            "range_together": together[-1] if span else None, "range": span[0] if span else None,
            "named": [f"L{lay}H{h}" for lay, h in named],
            "named_in_top_k": sum((lay, h) in named_set for (lay, h), _ in rank[:k]),
            "essay": [{"what": ch["what"], "before": ch["before"], "after": ch["after"]} for ch in c["checks"]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import transformers
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    rn = head_routes.Runner(head_routes.load(a.model, a.device, a.dtype), rows=32)
    out = {"model": a.model, "dtype": a.dtype, "figures": {}}
    for name, c in er.ATTENTION.items():
        with torch.no_grad():
            out["figures"][name] = f = figure(rn, tok, c)
        print(f"{name}: target {f['value']:.2f}; top {len(f['named']) or 1} heads alone ranked together "
              f"{f['top_k_together']:.2f}, the essay's heads {f['named_together']}, every head in {f['range']} "
              f"{f['range_together']}; essay heads in our top k: {f['named_in_top_k']} of {len(f['named'])}; "
              f"{time.time() - t0:.0f} s", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    a.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
