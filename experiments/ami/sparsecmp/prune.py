"""Our learned-mask circuit search under the circuit-sparsity criterion (Gao et al. 2511.13653).

Criterion: keep the fewest nodes such that the per-sequence task loss on the task's prompts stays at or below the
target (0.15 in the paper; their stored pair sums use 0.3), with every other node at its pretraining mean at every
position. The search, as in method/oracle/circuit_masks.py, uses the learned mask only to ORDER the nodes; every
accept/reject decision is a measured, discrete forward pass:

1. masks: m = sigmoid(theta / temp) per node, shared across prompts and positions; Adam on
       relu(loss(m) - floor * target) / target + lam * sum(m) / n_ref
   with the temperature annealed so the masks end near 0 or 1, and lam ramped.
2. order the nodes by theta; bisect the smallest prefix whose measured discrete loss meets the target on the
   search prompts;
3. prune: drop kept nodes (lowest theta first, in blocks then singly) while the measured loss still meets the target.
4. score on held-out prompts: the circuit's loss, the full model's loss, the loss with only the circuit ablated,
   and random circuits of the same size per site.

python prune.py gao MODEL --task T [--items tasks_v1.jsonl] [--device mps] [--out DIR]
"""
import argparse
import json
import math
import os
import random
import time

import torch

import nodes as N

ROOT = N.ROOT


def split_items(items: list, n_search: int, n_test: int, seed: int = 0):
    """Split by source file: search and test prompts never share a file."""
    files = sorted({it["file"] for it in items})
    random.Random(seed).shuffle(files)
    test_files = set(files[: len(files) // 3])
    test = [it for it in items if it["file"] in test_files][:n_test]
    search = [it for it in items if it["file"] not in test_files][:n_search]
    return search, test


class Batches:
    """Encoded pairs on the device, grouped by loss kind and batch size."""

    def __init__(self, model, enc: list, bs: int):
        self.groups = []
        for kind in ("binary", "pos_vs_rest"):
            its = [e for e in enc if e["kind"] == kind]
            for i in range(0, len(its), bs):
                b = its[i:i + bs]
                tp, lp = N.pad([e["pos"] for e in b], model.device)
                tn, ln = N.pad([e["neg"] for e in b], model.device)
                tok = torch.tensor([e["tok"] for e in b], device=model.device)
                alt = torch.tensor([e["alt"] for e in b], device=model.device)
                self.groups.append((kind, torch.cat([tp, tn]), torch.cat([lp, ln]), tok, alt, len(b)))
        self.n = sum(g[-1] for g in self.groups)


def pair_losses(model, batches: Batches, masks):
    out = []
    for kind, toks, last, tok, alt, b in batches.groups:
        lg = model.logits(toks, last, masks)
        out.append(N.task_loss(lg[:b], lg[b:], tok, alt, kind))
    return torch.cat(out) / 2.0                     # per sequence


@torch.no_grad()
def measure(model, batches: Batches, masks) -> float:
    return float(pair_losses(model, batches, masks).mean())


def discrete(model, keep: dict) -> dict:
    """{site: index tensor} -> {site: 0/1 mask}."""
    m = {}
    for s, n in model.sites.items():
        v = torch.zeros(n, device=model.device)
        ix = keep.get(s)
        if ix is not None and len(ix):
            v[ix] = 1.0
        m[s] = v
    return m


def from_order(model, order: list, k: int) -> dict:
    keep = {}
    for s, i in order[:k]:
        keep.setdefault(s, []).append(i)
    return {s: torch.tensor(v, device=model.device) for s, v in keep.items()}


def learn_order(model, batches: Batches, target: float, steps: int, lr: float, lam: float, floor: float,
                log=None) -> list:
    sites = list(model.sites)
    theta = {s: torch.full((model.sites[s],), 3.0, device=model.device, requires_grad=True) for s in sites}
    opt = torch.optim.Adam(theta.values(), lr=lr)
    n_ref = sum(model.sites.values())
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        temp = math.exp(math.log(1.0) * (1 - frac) + math.log(0.05) * frac)
        lam_t = lam * min(1.0, 2 * frac + 0.05)
        tot = 0.0
        opt.zero_grad()
        size = sum(torch.sigmoid(theta[s] / temp).sum() for s in sites)
        for kind, toks, last, tok, alt, b in batches.groups:
            m = {s: torch.sigmoid(theta[s] / temp) for s in sites}
            lg = model.logits(toks, last, m)
            loss = (N.task_loss(lg[:b], lg[b:], tok, alt, kind) / 2.0).sum() / batches.n
            hinge = torch.relu(loss * batches.n / b - floor * target) * b / batches.n / target
            hinge.backward()
            tot += float(loss)
        (lam_t * size / n_ref * 1000.0).backward()
        opt.step()
        if log and (step % 25 == 0 or step == steps - 1):
            log(f"  step {step} loss {tot:.4f} soft_size {float(size):.1f} temp {temp:.3f}")
    flat = [(float(v), s, i) for s in sites for i, v in enumerate(theta[s].detach().cpu().tolist())]
    flat.sort(key=lambda x: -x[0])
    return [(s, i) for _, s, i in flat]


def bisect(model, batches, order, target, log=None) -> int:
    lo, hi = 0, len(order)
    if measure(model, batches, discrete(model, from_order(model, order, hi))) > target:
        return -1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        l = measure(model, batches, discrete(model, from_order(model, order, mid)))
        if l <= target:
            hi = mid
        else:
            lo = mid
    if log:
        log(f"  bisect k={hi}")
    return hi


def prune(model, batches, order, k, target, log=None) -> list:
    """Drop kept nodes, lowest theta first: blocks of 16, then 4, then single nodes."""
    kept = list(order[:k])
    for block in (16, 4, 1):
        i = len(kept) - block
        while i >= 0:
            trial = kept[:i] + kept[i + block:]
            keep = {}
            for s, j in trial:
                keep.setdefault(s, []).append(j)
            m = discrete(model, {s: torch.tensor(v, device=model.device) for s, v in keep.items()})
            if measure(model, batches, m) <= target:
                kept = trial
            i -= block
        if log:
            log(f"  prune block {block}: {len(kept)}")
    return kept


def keep_dict(kept: list) -> dict:
    out = {}
    for s, j in kept:
        out.setdefault(s, []).append(int(j))
    return {s: sorted(v) for s, v in out.items()}


def score(model, batches, keep: dict, rng, n_random: int = 8) -> dict:
    """Held-out losses: circuit, full model, circuit ablated (necessity), random same-size circuits."""
    kt = {s: torch.tensor(v, device=model.device) for s, v in keep.items()}
    circ = measure(model, batches, discrete(model, kt))
    full = measure(model, batches, None)
    inv = {s: torch.ones(n, device=model.device) for s, n in model.sites.items()}
    for s, v in kt.items():
        inv[s][v] = 0.0
    ablated = measure(model, batches, inv)
    rand = []
    for _ in range(n_random):
        rk = {s: torch.tensor(rng.sample(range(model.sites[s]), len(v)), device=model.device) for s, v in keep.items()}
        rand.append(measure(model, batches, discrete(model, rk)))
    return {"circuit": circ, "full": full, "ablated": ablated, "random": rand}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["gao", "qwen"])
    ap.add_argument("model")
    ap.add_argument("--task", required=True)
    ap.add_argument("--items", default=f"{ROOT}/tasks_v1.jsonl")
    ap.add_argument("--means", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-search", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--target", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=0.7)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default=f"{ROOT}/runs")
    a = ap.parse_args()
    t0 = time.time()
    if a.family == "gao":
        means = a.means or f"{ROOT}/models/{a.model}/mean_baseline_pretrain.pt"
        model = N.GaoModel(a.model, a.device, means=means)
    else:
        model = N.QwenModel(a.model, a.device, means=a.means)
    items = [json.loads(l) for l in open(a.items)]
    items = [it for it in items if it["task"] == a.task]
    search, test = split_items(items, a.n_search, a.n_test)
    bs = Batches(model, N.encode_items(model, search), a.bs)
    bt = Batches(model, N.encode_items(model, test), a.bs)
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.model.replace('/', '_')}__{a.task}"
    logf = open(f"{a.out}/{name}.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    full_s = measure(model, bs, None)
    log(f"{name}: search {bs.n} test {bt.n} pairs; full-model search loss {full_s:.4f}; nodes {sum(model.sites.values())}")
    if full_s > a.target:
        log("  full model misses the target on the search prompts")
    order = learn_order(model, bs, a.target, a.steps, a.lr, a.lam, a.floor, log)
    k = bisect(model, bs, order, a.target, log)
    rec = {"model": a.model, "task": a.task, "target": a.target, "n_search": bs.n, "n_test": bt.n,
           "full_search": full_s, "bisect_k": k, "args": vars(a)}
    if k > 0:
        kept = prune(model, bs, order, k, a.target, log)
        keep = keep_dict(kept)
        rec.update({"k": len(kept), "keep": keep, "search_loss": measure(model, bs, discrete(
            model, {s: torch.tensor(v, device=model.device) for s, v in keep.items()}))})
        rec["test"] = score(model, bt, keep, random.Random(0))
        log(f"  k {len(kept)} test circuit {rec['test']['circuit']:.4f} full {rec['test']['full']:.4f} "
            f"ablated {rec['test']['ablated']:.4f} random {sum(rec['test']['random']) / len(rec['test']['random']):.4f}")
    rec["seconds"] = time.time() - t0
    json.dump(rec, open(f"{a.out}/{name}.json", "w"))


if __name__ == "__main__":
    main()
