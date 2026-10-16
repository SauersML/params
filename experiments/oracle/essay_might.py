"""The essay's "might" neuron, looked for blind on Qwen3-32B (#2951, scale32 10-16). The essay: "A neuron that fires on the
word 'might' writes into one head's query, and the head looks back for an earlier 'may'. Remove the neuron and the
head's attention on 'may' falls from 0.69 to 0.03, and it rests on the first word instead. Removing an equal amount of
other early neurons changes nothing. Across all 32 may...might pairs in our corpus, the look is at least halved in 31."
It names neither the neuron nor the head nor the full prompts, so both are found here on PROMPTS (a "may" clause, then
"might" as the last token):
  head     each of the HEADS heads whose attention from "might" to "may" is largest on average over the prompts (at
           least MIN_LOOK; eager attention), the result reported per head and the head the neuron moves most (relative);
  neuron   candidates at "might" in the layers below the head, ordered by |activation x gradient of that attention
           weight| summed over the prompts (the order only), the first CANDIDATES each zeroed at "might" on every
           prompt and the head's attention on "may" measured; the neuron is the one with the lowest mean;
  checks   per prompt the attention before and after (and where the head looks after), the share of prompts where it
           is at least halved; controls: CONTROLS other neurons of the neuron's layer at "might" with at least half its
           |activation| (or the largest there), each zeroed alone.
  essay_might.py OUT.json [--model Qwen/Qwen3-32B] [--dtype bfloat16]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import essay_replication as er  # noqa: E402

PROMPTS = ["This listing may contain errors in the prices. That listing might",
           "He may not agree with every policy, but his core beliefs might",
           "Some of you may find this useful. For example, this might",
           "Your dog may bark at strangers at first. This might",
           "The results may vary by region. Your results might",
           "We may arrive late tonight. The others might",
           "It may rain tomorrow afternoon. The game might",
           "Prices may rise next year. Wages might"]
CANDIDATES = 64
CONTROLS = 16
HEADS = 5          # the heads tried: the largest mean attention from "might" to "may", at least MIN_LOOK
MIN_LOOK = 0.3


def attn(q: er.Qwen, ids: list[int], zero=()) -> list:
    """Every layer's attention [H, T, T] (float32, host) with zero's neurons zeroed (essay_replication.Qwen.run)."""
    return [a[0].float().cpu() for a in q.run(ids, zero=list(zero), attentions=True)["attentions"]]


def grads(q: er.Qwen, ids: list[int], Lh: int, h: int, pm: int, py: int) -> tuple:
    """(activations, gradients) [Lh, D] of the neurons at pm in layers below Lh: the gradient of the head's attention
    weight from pm to py (eager attention, the model's own forward)."""
    probes, acts, hs = [], [], []
    for L in range(Lh):
        layer = q.layers[L]
        p = torch.zeros(layer.mlp.down_proj.in_features, device=layer.mlp.down_proj.weight.device, requires_grad=True)
        probes.append(p)

        def pre(m, a, p=p):
            x = a[0]
            acts.append(x[0, pm].detach().float().cpu())
            return (x + torch.nn.functional.pad(p.to(x.dtype)[None], (0, 0, pm, x.shape[1] - pm - 1))[None],) + tuple(a[1:])
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
    try:
        with torch.enable_grad():
            out = q.model(input_ids=torch.tensor([ids], device=q.model.device), output_attentions=True)
            out.attentions[Lh][0, h, pm, py].float().backward()
    finally:
        for x in hs:
            x.remove()
    return torch.stack(acts), torch.stack([p.grad.float().cpu() for p in probes])


def one_head(q: er.Qwen, items: list, base: list, Lh: int, h: int) -> dict:
    """The neuron at "might" whose zeroing lowers head (Lh, h)'s attention on "may" most (module docstring), its checks and
    controls."""
    score, acts = None, []
    for it in items:
        act, g = grads(q, it["ids"], Lh, h, it["pm"], it["py"])
        s = (act * g).abs()
        score = s / s.sum() if score is None else score + s / s.sum()
        acts.append(act)
    D = score.shape[1]
    cands = [divmod(int(i), D) for i in score.reshape(-1).topk(min(CANDIDATES, score.numel())).indices]
    after = {}
    for l, j in cands:
        after[(l, j)] = [float(attn(q, it["ids"], [(l, j, [it["pm"]], 0)])[Lh][h, it["pm"], it["py"]]) for it in items]
    l, j = min(after, key=lambda u: sum(after[u]))
    befores = [float(b[Lh][h, it["pm"], it["py"]]) for b, it in zip(base, items)]
    rest = []
    for it in items:
        z = attn(q, it["ids"], [(l, j, [it["pm"]], 0)])[Lh][h, it["pm"]]
        rest.append([int(z.argmax()), q.tok.decode([it["ids"][int(z.argmax())]]), round(float(z.max()), 3)])
    mags = torch.stack([x[l] for x in acts]).abs().mean(0)
    pool = [k for k in range(D) if k != j and float(mags[k]) >= float(mags[j]) / 2]
    random.seed(0)
    ctl = random.sample(pool, min(CONTROLS, len(pool)))
    if len(ctl) < CONTROLS:
        m2 = mags.clone()
        m2[j] = -1
        ctl += [int(k) for k in m2.topk(CONTROLS + len(ctl)).indices if int(k) not in ctl][:CONTROLS - len(ctl)]
    controls = {k: sum(float(attn(q, it["ids"], [(l, k, [it["pm"]], 0)])[Lh][h, it["pm"], it["py"]]) for it in items) / len(items)
                for k in ctl}
    return {"head": f"L{Lh}H{h}", "neuron": f"L{l}:{j}", "activation_at_might": [round(float(x[l, j]), 3) for x in acts],
            "prompts": [{"text": it["text"], "before": b, "after": x, "rests_on": r}
                        for it, b, x, r in zip(items, befores, after[(l, j)], rest)],
            "mean_before": sum(befores) / len(befores), "mean_after": sum(after[(l, j)]) / len(items),
            "halved": sum(x <= b / 2 for b, x in zip(befores, after[(l, j)])) / len(items),
            "candidates": [{"neuron": f"L{u[0]}:{u[1]}", "mean_after": sum(v) / len(v)} for u, v in
                           sorted(after.items(), key=lambda t: sum(t[1]))[:10]],
            "controls": {f"L{l}:{k}": v for k, v in controls.items()},
            "controls_mean_after": sum(controls.values()) / max(len(controls), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    t0 = time.time()
    q = er.Qwen(a.model, a.device, getattr(torch, a.dtype))
    q.model.requires_grad_(False)
    may = q.ids(" may")[0]
    items = []
    for s in PROMPTS:
        ids = q.ids(s)
        if may not in ids or q.tok.decode([ids[-1]]) != " might":
            print(f"skipped {s!r}: no ' may' / ' might' tokens", flush=True)
            continue
        items.append({"text": s, "ids": ids, "pm": len(ids) - 1, "py": ids.index(may)})
    base = [attn(q, it["ids"]) for it in items]
    L, H = len(base[0]), base[0][0].shape[0]
    mean = torch.stack([torch.stack([b[l][:, it["pm"], it["py"]] for l in range(L)]) for b, it in zip(base, items)]).mean(0)  # [L, H]
    heads = [divmod(int(i), H) for i in mean.reshape(-1).topk(HEADS).indices if float(mean.reshape(-1)[i]) >= MIN_LOOK]
    rec = {"model": a.model, "top_heads": [[f"L{i // H}H{i % H}", round(float(mean.reshape(-1)[i]), 3)]
                                           for i in mean.reshape(-1).topk(HEADS).indices.tolist()], "heads": []}
    print(f"heads by mean attention from 'might' to 'may': {rec['top_heads']}", flush=True)
    for Lh, h in heads:
        if Lh == 0:
            continue
        r = one_head(q, items, base, Lh, h)
        rec["heads"].append(r)
        print(f"head L{Lh}H{h}: neuron {r['neuron']}: attention on 'may' {r['mean_before']:.3f} -> {r['mean_after']:.3f} "
              f"(essay 0.69 -> 0.03); halved on {r['halved']:.2f} of prompts (essay 31/32); controls' mean after "
              f"{r['controls_mean_after']:.3f}", flush=True)
    best = min(rec["heads"], key=lambda r: r["mean_after"] / r["mean_before"])
    rec.update(head=best["head"], neuron=best["neuron"], mean_before=best["mean_before"], mean_after=best["mean_after"],
               halved=best["halved"], essay={"before": 0.69, "after": 0.03, "halved": 31 / 32}, seconds=round(time.time() - t0, 1))
    print(f"best: head {best['head']}, neuron {best['neuron']}", flush=True)
    for p in best["prompts"]:
        print(f"  {p['before']:.3f} -> {p['after']:.3f}, rests on {p['rests_on']}: {p['text']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
