"""Our learned-mask circuit search under the circuit-sparsity criterion (Gao et al. 2511.13653).

Criterion: keep the fewest nodes such that the per-sequence task loss on the task's prompts stays at or below the
target (0.15 in the paper; their stored pair sums use 0.3), with every other node at its pretraining mean at every
position. The search, as in method/oracle/circuit_masks.py, uses the learned mask only to ORDER the nodes; every
accept/reject decision is a measured, discrete forward pass:

1. masks: m = sigmoid(theta) per node, shared across prompts and positions; Adam on
       relu(loss(m) - floor * target) / target + lam * sum(m)
   (learn_order), the last third with binarized masks forward;
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
                hard_from: float = 0.67, log=None, snaps: int = 6) -> list:
    """circuit_masks' objective with the task loss in place of the patching shares: theta from INIT 3 (every node
    nearly kept), Adam on relu(loss - floor * target) / target + lam_t * sum(m), so a node stays while it lowers the
    loss by more than lam_t * target. lam_t rises geometrically from lam / 10 to lam * 10 over the steps (a path of
    smaller sets, in place of their 256-768 hyperparameter-search runs per task), and the theta order is taken at
    `snaps` points of the second half; the caller keeps the smallest passing prefix among them. After hard_from of
    the steps the forward runs the binarized masks (theta > 0) with the sigmoid's gradient (circuit_masks' HARD)."""
    sites = list(model.sites)
    theta = {s: torch.full((model.sites[s],), 3.0, device=model.device, requires_grad=True) for s in sites}
    opt = torch.optim.Adam(theta.values(), lr=lr)
    at = {int(round(steps * (0.5 + 0.5 * (j + 1) / snaps))) - 1 for j in range(snaps)}
    orders = []
    for step in range(steps):
        hard = step >= hard_from * steps
        lam_t = lam * 10 ** (2 * step / max(steps - 1, 1) - 1)
        tot = 0.0
        opt.zero_grad()
        for kind, toks, last, tok, alt, b in batches.groups:
            ms = {s: torch.sigmoid(theta[s]) for s in sites}
            if hard:
                ms = {s: (theta[s] > 0).float() + m - m.detach() for s, m in ms.items()}
            lg = model.logits(toks, last, ms)
            loss = (N.task_loss(lg[:b], lg[b:], tok, alt, kind) / 2.0).mean()
            (torch.relu(loss - floor * target) / target * b / batches.n).backward()
            tot += float(loss) * b / batches.n
        size = sum(torch.sigmoid(theta[s]).sum() for s in sites)
        (lam_t * size).backward()
        opt.step()
        if log and (step % 25 == 0 or step == steps - 1):
            kept = sum(int((theta[s] > 0).sum()) for s in sites)
            log(f"  step {step} loss {tot:.4f} sum_m {float(size):.1f} kept {kept} lam {lam_t:.4g}{' hard' if hard else ''}")
        if step in at:
            flat = torch.cat([theta[s].detach().float() for s in sites]).cpu()
            idx = torch.argsort(flat, descending=True).tolist()
            owner = [(s, i) for s in sites for i in range(model.sites[s])]
            orders.append((step, [owner[j] for j in idx]))
    return orders


def bisect(model, batches, order, target, log=None) -> int:
    """The smallest passing prefix of the theta order, by measurement. The loss is not monotone in the prefix length
    (nodes the descent left out can hurt when live: on dense1_1x a 695-node prefix passed while plain bisection over
    [0, N] stopped at 11,199), so as in circuit_masks: a grid of sizes from 8 up (x sqrt 2) to the first that passes,
    then bisection between it and the grid size below."""
    N_ = len(order)
    loss = lambda k: measure(model, batches, discrete(model, from_order(model, order, k)))
    grid = sorted({min(N_, int(round(8 * 2 ** (i / 2)))) for i in range(64)})
    prev, hit = 0, None
    for g in grid:
        if loss(g) <= target:
            hit = g
            break
        prev = g
    if hit is None:
        return -1
    lo, hi = prev, hit
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if loss(mid) <= target:
            hi = mid
        else:
            lo = mid
    if log:
        log(f"  prefix k={hi} (grid hit {hit})")
    return hi


def prune(model, batches, order, k, target, log=None, singles_max: int = 3000) -> list:
    """Drop kept nodes, lowest theta first, while the measured loss still meets the target: blocks of 256 (when the
    prefix is large), 64, 16 and 4, then single nodes when at most singles_max remain (each pass costs a forward per
    block; single passes over Qwen3-8B's prefixes of thousands of nodes would take hours)."""
    kept = list(order[:k])
    for block in (256, 64, 16, 4, 1):
        if block * 8 > len(kept) and block > 1:
            continue
        if block == 1 and len(kept) > singles_max:
            if log:
                log(f"  prune singles skipped: {len(kept)} > {singles_max}")
            break
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


@torch.no_grad()
def score(model, batches, keep: dict, rng, n_random: int = 8) -> dict:
    """Held-out per-pair losses (per sequence): the circuit, the full model, the circuit ablated (every node outside
    kept, the circuit's nodes at their means: necessity) and random circuits with the same count per site."""
    kt = {s: torch.tensor(v, device=model.device) for s, v in keep.items()}
    pl = lambda m: pair_losses(model, batches, m).cpu().tolist()
    circ, full = pl(discrete(model, kt)), pl(None)
    inv = {s: torch.ones(n, device=model.device) for s, n in model.sites.items()}
    for s, v in kt.items():
        inv[s][v] = 0.0
    ablated = pl(inv)
    rand = []
    for _ in range(n_random):
        rk = {s: torch.tensor(rng.sample(range(model.sites[s]), len(v)), device=model.device) for s, v in keep.items()}
        rand.append(sum(pl(discrete(model, rk))) / batches.n)
    mean = lambda v: sum(v) / len(v)
    return {"circuit": mean(circ), "full": mean(full), "ablated": mean(ablated), "random": rand,
            "pairs": {"circuit": circ, "full": full, "ablated": ablated}}


def edge_count(model, keep: dict) -> dict:
    import edges as E
    return E.gao_edges(model, keep) if isinstance(model, N.GaoModel) else E.dense_edges(keep, model)


def run_task(model, items: list, task: str, a, out: str) -> dict:
    """One task's search and held-out scores; writes out/<model>__<task>.json (skipped when present: resumable)."""
    t0 = time.time()
    name = f"{a.model.replace('/', '_')}__{task}"
    if os.path.exists(f"{out}/{name}.json"):
        return json.load(open(f"{out}/{name}.json"))
    its = [it for it in items if it["task"] == task]
    search, test = split_items(its, a.n_search, a.n_test)
    bs = Batches(model, N.encode_items(model, search), a.bs)
    bt = Batches(model, N.encode_items(model, test), a.bs)
    logf = open(f"{out}/{name}.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    full_s = measure(model, bs, None)
    log(f"{name}: search {bs.n} test {bt.n} pairs; full-model search loss {full_s:.4f}; nodes {sum(model.sites.values())}")
    if full_s > a.target:
        log("  full model misses the target on the search prompts")
    orders = learn_order(model, bs, a.target, a.steps, a.lr, a.lam, a.floor, a.hard_from, log)
    prefixes = []
    for step, o in orders:
        kk = bisect(model, bs, o, a.target, log)
        prefixes.append({"step": step, "k": kk})
    ok = [(p["k"], j) for j, p in enumerate(prefixes) if p["k"] > 0]
    k, order = (min(ok)[0], orders[min(ok)[1]][1]) if ok else (-1, None)
    rec = {"model": a.model, "task": task, "target": a.target, "n_search": bs.n, "n_test": bt.n,
           "full_search": full_s, "bisect_k": k, "prefixes": prefixes,
           "args": {kk: v for kk, v in vars(a).items() if kk != "task"}}
    if k > 0:
        kept = prune(model, bs, order, k, a.target, log)
        keep = keep_dict(kept)
        rec.update({"k": len(kept), "keep": keep, "search_loss": measure(model, bs, discrete(
            model, {s: torch.tensor(v, device=model.device) for s, v in keep.items()}))})
        rec["test"] = score(model, bt, keep, random.Random(0))
        rec["edges"] = edge_count(model, keep)
        rec["sites"] = {s: len(v) for s, v in keep.items()}
        log(f"  k {len(kept)} edges {rec['edges']['weights']} test circuit {rec['test']['circuit']:.4f} full "
            f"{rec['test']['full']:.4f} ablated {rec['test']['ablated']:.4f} random "
            f"{sum(rec['test']['random']) / len(rec['test']['random']):.4f}")
    rec["seconds"] = time.time() - t0
    json.dump(rec, open(f"{out}/{name}.json", "w"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["gao", "qwen"])
    ap.add_argument("model")
    ap.add_argument("--task", required=True, help="a task, a comma list, 'verified' or 'all'")
    ap.add_argument("--items", default=f"{ROOT}/tasks_v2.jsonl")
    ap.add_argument("--means", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-search", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--target", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.003)
    ap.add_argument("--floor", type=float, default=0.7)
    ap.add_argument("--hard-from", type=float, default=0.67)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default=f"{ROOT}/runs")
    a = ap.parse_args()
    import tasks as T
    if a.device.startswith("cuda"):                 # TF32 matmuls: every decision is a measured run under the same setting
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if a.family == "gao":
        means = a.means or f"{ROOT}/models/{a.model}/mean_baseline_pretrain.pt"
        model = N.GaoModel(a.model, a.device, means=means)
    else:
        model = N.QwenModel(a.model, a.device, means=a.means, dtype=getattr(torch, a.dtype))
    items = [json.loads(l) for l in open(a.items)]
    tasks = {"all": T.TASKS, "verified": T.VERIFIED}.get(a.task) or a.task.split(",")
    if a.task == "all":                              # verified tasks first
        tasks = T.VERIFIED + [t for t in T.TASKS if t not in T.VERIFIED]
    os.makedirs(a.out, exist_ok=True)
    for t in tasks:
        run_task(model, items, t, a, a.out)


if __name__ == "__main__":
    main()
