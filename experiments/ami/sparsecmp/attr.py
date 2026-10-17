"""Gradient rankings against our search at the circuit-sparsity criterion, on the same task pairs (equal input), with
overlap against the released circuits of Gao et al. 2511.13653.

Every method ranks the nodes from the same 256 search pairs that prune.py's search used (prune.split_items, same items
file and counts); the criterion is the same (every node outside the circuit at its pretraining mean, per-sequence task
loss at most 0.15). A node's score is the derivative of the search pairs' mean task loss with respect to its mask m
(the node's value mu + m (x - mu), at every position), so a score is the change in loss for keeping the node:
  eapig   integrated gradients along m = alpha for every node at once, alpha from 0 (all at the mean) to 1 (the full
          model), K midpoints: node-level EAP-IG from the criterion's own baseline, averaged over the pairs
  relp    the gradient at m = 1 with relevance rules: attention probabilities and RMSNorm scales held as constants,
          GELU passed through as gelu(x) / x (the LRP identity rule)
Signed rankings put the most loss-lowering nodes first; abs rankings order by |score|. Pairs of different lengths
(if_equals, var_if) have no position-wise patch, which is why the baseline is the mean and not the pair's other prompt.
For each ranking: the smallest measured passing prefix (prune.bisect, the grid then bisection our search uses), then
the measured prune (prune.prune). Each circuit is scored on the held-out pairs (prune.score) and compared with the
released circuit (precision, recall, and the chance overlap of same-size random sets per site), and every ranking's
top |released| nodes give its recall at the released size.

python attr.py MODEL --tasks verified [--means PATH] [--ours RUN_DIR] [--device cuda] [--out DIR]
"""
import argparse
import contextlib
import json
import os
import random
import time

import torch
import torch.nn.functional as F

import nodes as N
import prune as P
import tasks as T

ROOT = N.ROOT
RELEASED = {"csp_yolo2": "prune_v4", "csp_sweep1_4x_0.9Mnonzero_afrac0.125": "prune_v3"}
SUFFIX = {"set_or_string": "set_or_string_beeg2", "single_double_quote": "single_double_quote_beeg3"}


def grads(model, batches, m_value: float, relp: bool = False) -> dict:
    """d(mean task loss over the pairs)/dm per node, every mask at m_value."""
    ms = {s: torch.full((n,), m_value, device=model.device, requires_grad=True) for s, n in model.sites.items()}
    ctx = relp_mode(model) if relp else contextlib.nullcontext()
    with ctx:
        for kind, toks, last, tok, alt, b in batches.groups:
            lg = model.logits(toks, last, ms)
            (N.task_loss(lg[:b], lg[b:], tok, alt, kind).sum() / 2.0 / batches.n).backward()
    # under RelP's rules q and k get no gradient (the attention pattern is a constant), so their score is 0
    return {s: (m.grad.detach().float().cpu() if m.grad is not None else torch.zeros(m.shape)) for s, m in ms.items()}


def eapig(model, batches, K: int = 32) -> dict:
    acc = None
    for j in range(K):
        g = grads(model, batches, (j + 0.5) / K)
        acc = g if acc is None else {s: acc[s] + g[s] for s in g}
    return {s: v / K for s, v in acc.items()}


class _DetachedAttention(torch.nn.Module):
    """Causal attention (with their sink column when the model has one) whose probabilities are constants."""

    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def forward(self, q, k, v, *, dropout_p=0.0, is_causal=True, scale=None):
        B, H, Lq, D = q.shape
        s = (q @ k.transpose(-2, -1)) * (scale if scale is not None else D ** -0.5)
        causal = torch.ones(Lq, k.shape[2], dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~causal, torch.finfo(s.dtype).min)
        if self.sink is not None:
            col = self.sink.to(s.dtype).view(1, H, 1, 1).expand(B, H, Lq, 1)
            p = torch.cat([col, s], -1).softmax(-1)[..., 1:]
        else:
            p = s.softmax(-1)
        return p.detach() @ v


@contextlib.contextmanager
def relp_mode(model):
    """RelP's backward rules on their GPT: RMSNorm scale and attention probabilities detached, GELU as gelu(x)/x."""
    import types
    gpt = model.model
    saved = []

    def rms_forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + (self.eps or torch.finfo(x.dtype).eps)).detach()
        return x * scale * self.weight

    for mod in gpt.modules():
        if isinstance(mod, torch.nn.RMSNorm):
            saved.append((mod, "forward", mod.forward))
            mod.forward = types.MethodType(rms_forward, mod)
        if isinstance(mod, torch.nn.GELU):
            saved.append((mod, "forward", mod.forward))
            mod.forward = types.MethodType(
                lambda self, x: x * torch.where(x.abs() > 1e-6, F.gelu(x) / x, torch.full_like(x, 0.5)).detach(), mod)
    for blk in gpt.transformer.h:
        attn = blk.attn
        imp = attn.attn_imp
        saved.append((attn, "attn_imp", imp))
        attn.attn_imp = _DetachedAttention(getattr(imp, "sink_logit", None))
    try:
        yield
    finally:
        for obj, name, val in reversed(saved):
            if isinstance(obj, torch.nn.Module) and name in obj._modules and not isinstance(val, torch.nn.Module):
                del obj._modules[name]                 # a plain function (F.scaled_dot_product_attention) goes back
                obj.__dict__[name] = val               # where it was, outside the child modules
            else:
                setattr(obj, name, val)


def order_of(model, score: dict, signed: bool) -> list:
    sites = list(model.sites)
    flat = torch.cat([score[s] for s in sites])
    key = flat if signed else -flat.abs()            # ascending: most loss-lowering (signed) or largest |score| first
    idx = torch.argsort(key).tolist()
    owner = [(s, i) for s in sites for i in range(model.sites[s])]
    return [owner[j] for j in idx]


def released_nodes(model_name: str, task: str) -> set:
    sweep = RELEASED.get(model_name)
    if not sweep:
        return set()
    R = json.load(open(f"{ROOT}/released/prunes_{model_name}_{sweep}.json")).get(SUFFIX.get(task, task + "_beeg"), {})
    return {(s, i) for s, v in R.get("nodes", {}).items() for i in v}


def overlap(model, keep: dict, rel: set) -> dict:
    S = {(s, i) for s, v in keep.items() for i in v}
    if not rel or not S:
        return {}
    hit = len(S & rel)
    chance = sum(len(v) * sum(1 for (s2, _) in rel if s2 == s) / model.sites[s] for s, v in keep.items())
    return {"precision": hit / len(S), "recall": hit / len(rel), "hits": hit, "size": len(S), "released": len(rel),
            "chance_hits": chance}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tasks", default="verified")
    ap.add_argument("--means", default="")
    ap.add_argument("--ours", default="", help="prune.py run dir with our search's records for this model")
    ap.add_argument("--items", default=f"{ROOT}/tasks_v2s.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-search", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--target", type=float, default=0.15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--K", type=int, default=32)   # 32 midpoints: the IG sum matches L(1) - L(0) (8 missed by 45% on dense1_1x)
    ap.add_argument("--out", default=f"{ROOT}/attr")
    a = ap.parse_args()
    if a.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    model = N.GaoModel(a.model, a.device, means=a.means or f"{ROOT}/models/{a.model}/mean_baseline_pretrain.pt")
    items = [json.loads(l) for l in open(a.items)]
    tasks = {"verified": T.VERIFIED, "all": T.VERIFIED + [t for t in T.TASKS if t not in T.VERIFIED]}.get(
        a.tasks) or a.tasks.split(",")
    os.makedirs(a.out, exist_ok=True)
    for task in tasks:
        path = f"{a.out}/{a.model}__{task}.json"
        if os.path.exists(path):
            continue
        its = [it for it in items if it["task"] == task]
        search, test = P.split_items(its, a.n_search, a.n_test)
        bs = P.Batches(model, N.encode_items(model, search), a.bs)
        bt = P.Batches(model, N.encode_items(model, test), a.bs)
        rel = released_nodes(a.model, task)
        rec = {"model": a.model, "task": task, "target": a.target, "n_search": bs.n, "n_test": bt.n,
               "released": len(rel), "full_search": P.measure(model, bs, None), "arms": {}}
        t0 = time.time()
        scores = {"eapig": eapig(model, bs, a.K)}
        t_ig = time.time() - t0
        t0 = time.time()
        scores["relp"] = grads(model, bs, 1.0, relp=True)
        t_relp = time.time() - t0
        arms = {"eapig_signed": ("eapig", True, t_ig), "eapig_abs": ("eapig", False, t_ig),
                "relp_signed": ("relp", True, t_relp), "relp_abs": ("relp", False, t_relp)}
        for name, (sc, signed, t_rank) in arms.items():
            order = order_of(model, scores[sc], signed)
            arm = {"rank_seconds": t_rank}
            if rel:
                top = set(order[:len(rel)])
                arm["recall_at_released_size"] = len(top & rel) / len(rel)
            t0 = time.time()
            k = P.bisect(model, bs, order, a.target)
            arm["prefix_seconds"] = time.time() - t0
            arm["k_prefix"] = k
            if k > 0:
                pre = P.keep_dict(order[:k])
                arm["prefix"] = {"k": k, "edges": P.edge_count(model, pre)["weights"],
                                 "test": P.score(model, bt, pre, random.Random(0), n_random=4), "overlap": overlap(model, pre, rel)}
                t0 = time.time()
                kept = P.prune(model, bs, order, k, a.target)
                arm["prune_seconds"] = time.time() - t0
                kd = P.keep_dict(kept)
                arm["pruned"] = {"k": len(kept), "edges": P.edge_count(model, kd)["weights"], "keep": kd,
                                 "test": P.score(model, bt, kd, random.Random(0), n_random=4), "overlap": overlap(model, kd, rel)}
            rec["arms"][name] = arm
            print(task, name, "prefix", k, "pruned", arm.get("pruned", {}).get("k"),
                  "recall@R", round(arm.get("recall_at_released_size", float("nan")), 3), flush=True)
        if a.ours and os.path.exists(f"{a.ours}/{a.model}__{task}.json"):
            r = json.load(open(f"{a.ours}/{a.model}__{task}.json"))
            if r.get("keep"):
                assert r["n_search"] == bs.n and r["n_test"] == bt.n, "our search used other pairs"
                rec["arms"]["ours"] = {"pruned": {"k": r["k"], "edges": r["edges"]["weights"], "test": r["test"],
                                                  "overlap": overlap(model, r["keep"], rel)}, "seconds": r["seconds"]}
        if rel:
            rec["released_test"] = P.score(model, bt, {s: sorted({i for (s2, i) in rel if s2 == s}) for s in
                                                       {s for s, _ in rel}}, random.Random(0), n_random=4)
        json.dump(rec, open(path, "w"))


if __name__ == "__main__":
    main()
