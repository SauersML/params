"""Step A: score their released circuits with our node masks, to check hooks, means and loss against their numbers.

For a released (model, sweep) and each task: the full model's loss and the released circuit's loss (every node outside
the circuit mean-ablated with their released pretraining means), on (1) their 16 released pairs (csp_yolo2 only) and
(2) our rebuilt pairs. Losses are per sequence (half their stored pair sums); their bisection target is printed beside.
A random circuit of the same size per site is the control.

python check_release.py MODEL SWEEP [--items tasks.jsonl] [--n 256] [--device cpu]
"""
import argparse
import json
import os
import random
import re

import torch

import nodes as N
import tasks as T

ROOT = N.ROOT
SUFFIX = {"set_or_string": "set_or_string_beeg2", "single_double_quote": "single_double_quote_beeg3"}


def task_dir(t):
    return SUFFIX.get(t, t + "_beeg")


def released_items(enc, task):
    """Their 16 pairs for a task (rows i and i+16), with answer texts from our task definitions."""
    S = json.load(open(f"{ROOT}/released/samples_csp_yolo2_prune_v4.json"))
    rows = S[task_dir(task)]["tokens"]
    n = len(rows) // 2
    out = []
    for i in range(n):
        p = [x for x in rows[i]]
        q = [x for x in rows[i + n]]
        while p and p[-1] == 0:
            p.pop()
        while q and q[-1] == 0:
            q.pop()
        out.append((enc.decode(p), enc.decode(q)))
    return out


def released_answers(it):
    """Answer texts for a released pair: class_init copies the attribute name, indent_for indents one level."""
    if it["task"] == "class_init":
        it["pos_ans"] = " " + re.search(r"self\.(\w+) =$", it["pos"]).group(1)
        it["neg_ans"] = " " + re.search(r"self\.(\w+) =$", it["neg"]).group(1)
    if it["task"] == "indent_for":
        for side in ("pos", "neg"):
            ind = re.match(r"[ \t]*", it[side].rstrip("\n").rsplit("\n", 1)[-1]).group(0)
            it[side + "_ans"] = ind + "    x"
    return it


def masks_for(model, nodes: dict, keep_all=False):
    m = {}
    for site, n in model.sites.items():
        v = torch.zeros(n, device=model.device)
        idx = nodes.get(site, [])
        if idx:
            v[torch.tensor(idx, device=model.device)] = 1.0
        m[site] = v
    return m


def random_like(model, nodes: dict, rng):
    return {s: rng.sample(range(model.sites[s]), len(ix)) for s, ix in nodes.items()}


@torch.no_grad()
def loss_of(model, enc_items, masks, bs=4):
    tot, cnt = 0.0, 0
    for kind in ("binary", "pos_vs_rest"):
        its = [e for e in enc_items if e["kind"] == kind]
        for i in range(0, len(its), bs):
            b = its[i:i + bs]
            tp, lp_ = N.pad([e["pos"] for e in b], model.device)
            tn, ln_ = N.pad([e["neg"] for e in b], model.device)
            lp = model.logits(tp, lp_, masks)
            ln = model.logits(tn, ln_, masks)
            tok = torch.tensor([e["tok"] for e in b], device=model.device)
            alt = torch.tensor([e["alt"] for e in b], device=model.device)
            tot += float(N.task_loss(lp, ln, tok, alt, kind).sum())
            cnt += len(b)
    return tot / max(cnt, 1) / 2.0                  # per sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("sweep")
    ap.add_argument("--items", default=f"{ROOT}/tasks_v2.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--means", default="", help="means file (default: their released means)")
    a = ap.parse_args()
    model = N.GaoModel(a.model, a.device, means=a.means or f"{ROOT}/models/{a.model}/mean_baseline_pretrain.pt")
    P = json.load(open(f"{ROOT}/released/prunes_{a.model}_{a.sweep}.json"))
    items = [json.loads(l) for l in open(a.items)]
    rng = random.Random(0)
    tasks = a.tasks.split(",") if a.tasks else T.TASKS
    for t in tasks:
        rel = P.get(task_dir(t))
        if not rel or "nodes" not in rel:
            print(t, "no released circuit")
            continue
        its = [x for x in items if x["task"] == t][:a.n]
        enc = N.encode_items(model, its)
        circ = masks_for(model, rel["nodes"])
        k = sum(len(v) for v in rel["nodes"].values())
        rnd = masks_for(model, random_like(model, rel["nodes"], rng))
        full = loss_of(model, enc, None)
        cl = loss_of(model, enc, circ)
        rl = loss_of(model, enc, rnd)
        line = {"task": t, "k": k, "target_stored": rel["runs"][rel["best"]]["target_loss"], "n": len(enc),
                "ours_full": round(full, 4), "ours_circuit": round(cl, 4), "ours_random": round(rl, 4)}
        if a.model == "csp_yolo2":
            r = released_items(model.enc, t)
            enc_r = N.encode_items(model, [released_answers(dict(its[0], pos=p, neg=q)) for p, q in r]) if its else []
            line.update({"rel_n": len(enc_r), "rel_full": round(loss_of(model, enc_r, None), 4),
                         "rel_circuit": round(loss_of(model, enc_r, circ), 4)})
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
