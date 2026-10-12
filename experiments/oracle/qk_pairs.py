"""Can a circuit state why attention looks where it does? (#2951 graph oracle; the lead's question 10-12.)

In C's semantics a circuit's query and key nodes enter their layer's attention pattern in full, with no stated partner.
For one read (an attention output node's layer and position t), the pattern row at t splits exactly into pair terms:
with q_t = sum_a q_a + q_rem (the circuit's query nodes at t and the remainder W - V U) and k_p = sum_b k_b + k_rem, each
head's score is score(t, p) = sum over pairs (a, b) of rope_t(q_a) . rope_p(k_b) / sqrt(hd) (q_a = a_a(t) u_a: rank one).
The background is the remainder-remainder term; a stated pair (a, b), b a key node at p, adds its term at p; (a, rem)
and (rem, b) are pairs with the remainder. The row computed from the background and a set S of pairs replaces the
read's pattern row (every other row and layer as in C's run), and the circuit's target KL is measured.

Per read, pairs are added greedily, each the one that lowers the measured target KL most, until the KL is within the
budget: the KL of F1 = GOAL (0.9) on goal_eval's log scale when the circuit itself reaches it, else 90% of the
circuit's own log recovery (ln(KL_empty / KL) >= 0.9 ln(KL_empty / KL_C)). With every pair the row is the circuit's own
(checked: the circuit's KL to rounding). Also: the subcomponents a circuit uses at more than one position.

  qk_pairs.py measure --texts 9002 9004 ... --out OUT.jsonl [--dirs DIR ...]
  qk_pairs.py figure OUT.jsonl FIG.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method" / "oracle"))
sys.path.insert(0, str(ROOT / "method" / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402

GOAL = goal_eval.GOAL
TEACHER = Path.home() / "mpd-data/graph_oracle/teacher"
DIRS = (TEACHER / "joint_heldout", TEACHER / "joint_heldout_80fe906")


def forward(ge, ids, targets, graph, override=None, capture=None):
    """goal_eval.GoalEval.graph_ref (C's semantics, remainders live) with the attention pattern row of one read
    replaced: override = (layer, t, rows [B, H, t + 1]) scores before the softmax; capture receives per layer the
    normed attention input z [T, d] and the query and key activations (B = 1). Returns log q at the targets [B, n, V]."""
    import vpd_model

    nat, tg = ge.nat, ge.nat.target
    T, H, hd, dev = len(ids), tg.n_head, tg.hd, nat.dev
    nodes = sorted(graph.node_set())
    par_all, out_all = native.all_edges(nodes, targets)
    stated = {r: set(ws) for r, ws in graph.parents.items()}
    drops = {}
    for r, ws in par_all.items():
        drops.setdefault(r, []).extend(w for w in ws if w not in stated.get(r, set()))
    drops.setdefault(None, []).extend(w for w in out_all if w not in set(graph.out))
    cut = {}
    for r, ws in drops.items():
        if r is not None and ws:
            cut.setdefault(r[0], []).append((r, sorted(set(ws))))
    cut_out = sorted(set(drops.get(None, [])))
    B = 1 if override is None else override[2].shape[0]
    circ = {n: torch.zeros(T, nat.C[n], device=dev) for n in nat.names}
    for n, t, c in nodes:
        circ[n][t, c] = 1.0
    on = {n: circ[n][None].expand(B, -1, -1) for n in nat.names}
    a = {}

    def write(w):
        return (a[w[0]][:, w[1], w[2]] * on[w[0]][:, w[1], w[2]])[:, None] * tg.site(w[0]).U[w[2]]

    def site(n, inp, restricted):
        A = inp @ tg.site(n).V
        if n in cut:
            A = A.clone()
            for r, ws in cut.get(n, []):
                A[:, r[1], r[2]] = restricted(r, ws) @ tg.site(n).V[:, r[2]]
        a[n] = A
        return (A * on[n]) @ tg.site(n).U + inp @ tg.site(n).delta_T

    def norm(xx, w):
        return vpd_model.rms(xx, w, tg.eps)

    x = tg.wte[torch.tensor(ids, device=dev)][None].expand(B, -1, -1)
    causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    for i in range(tg.n_layer):
        nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
        z = norm(x, tg.norms[2 * i])
        resid = lambda r, ws, x=x, i=i: norm(x[:, r[1]] - sum(write(w) for w in ws), tg.norms[2 * i])  # noqa: E731
        q, k, v = (site(nm[kk], z, resid).view(B, T, H, hd).transpose(1, 2) for kk in ("q_proj", "k_proj", "v_proj"))
        if capture is not None:
            capture[i] = {"z": z[0].clone(), "q": a[nm["q_proj"]][0].clone(), "k": a[nm["k_proj"]][0].clone()}
        q, k = tg._rope(q, T), tg._rope(k, T)
        S = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        if override is not None and override[0] == i:
            _, t, rows = override
            S = S.clone()
            S[:, :, t, :t + 1] = rows
        P = S.masked_fill(~causal, float("-inf")).softmax(-1)
        att = (P @ v).transpose(1, 2).reshape(B, T, -1)
        share = lambda r, ws, att=att, P=P: att[:, r[1]] - sum(  # noqa: E731
            P[:, :, r[1], w[1]].repeat_interleave(hd, dim=1) * write(w) for w in ws)
        x = x + site(nm["o_proj"], att, share)
        z2 = norm(x, tg.norms[2 * i + 1])
        resid2 = lambda r, ws, x=x, i=i: norm(x[:, r[1]] - sum(write(w) for w in ws), tg.norms[2 * i + 1])  # noqa: E731
        pre = site(nm["c_fc"], z2, resid2)
        x = x + site(nm["down_proj"], vpd_model.gelu_tanh(pre),
                     lambda r, ws, pre=pre: vpd_model.gelu_tanh(pre[:, r[1]] - sum(write(w) for w in ws)))
    xt = x[:, targets]
    if cut_out:
        xt = xt.clone()
        for j, t in enumerate(targets):
            ws = [w for w in cut_out if w[1] == t]
            if ws:
                xt[:, j] = xt[:, j] - sum(write(w) for w in ws)
    return torch.log_softmax((norm(xt, tg.ln_f) @ tg.wte.T).float(), -1)


def rope_at(tg, vec: torch.Tensor, pos: int) -> torch.Tensor:
    """Rotary embedding of per-head vectors [H, hd] at one position."""
    n = tg.hd // 2
    rot = torch.cat([-vec[..., n:], vec[..., :n]], dim=-1)
    return vec * tg.cos[pos] + rot * tg.sin[pos]


def pair_terms(ge, cap: dict, graph, layer: int, t: int) -> tuple:
    """(the background row [H, t + 1], candidate pairs [(a, b)], their terms [H, t + 1], the query and key nodes) for the
    read (layer, t); a, b a circuit query or key node or "rem"."""
    tg = ge.nat.target
    H, hd = tg.n_head, tg.hd
    nq, nk = native.site_name(layer, "q_proj"), native.site_name(layer, "k_proj")
    z = cap[layer]["z"]
    qs = [nd for nd in sorted(graph.node_set()) if nd[0] == nq and nd[1] == t]
    ks = [nd for nd in sorted(graph.node_set()) if nd[0] == nk and nd[1] <= t]
    sq, sk = tg.site(nq), tg.site(nk)
    qpart = {nd: rope_at(tg, (cap[layer]["q"][t, nd[2]] * sq.U[nd[2]]).view(H, hd), t) for nd in qs}
    qpart["rem"] = rope_at(tg, (z[t] @ sq.delta_T).view(H, hd), t)
    krem = (z[:t + 1] @ sk.delta_T).view(t + 1, H, hd)
    krem = torch.stack([rope_at(tg, krem[p], p) for p in range(t + 1)])  # [t + 1, H, hd]
    scale = 1 / math.sqrt(hd)

    def term(qa, b):
        row = torch.zeros(H, t + 1)
        if b == "rem":
            return (qa[None] * krem).sum(-1).T * scale
        kb = rope_at(tg, (cap[layer]["k"][b[1], b[2]] * sk.U[b[2]]).view(H, hd), b[1])
        row[:, b[1]] = (qa * kb).sum(-1) * scale
        return row
    bg = term(qpart["rem"], "rem")
    pairs, terms = [], []
    for qa in qs + ["rem"]:
        for kb in ks + ["rem"]:
            if qa == "rem" and kb == "rem":
                continue
            pairs.append((qa, kb))
            terms.append(term(qpart[qa], kb))
    return bg, pairs, terms, qs, ks


def node_name(nd) -> str:
    if nd == "rem":
        return "remainder"
    layer, kind = native._name_layer_kind(nd[0])
    return f"L{layer}.{dict(q_proj='q', k_proj='k')[kind]}#{nd[2]}@{nd[1]}"


@torch.no_grad()
def measure(ge, task: dict, answer: dict, log=print) -> dict:
    """One text: per read the greedy pair curve, and the circuit's subcomponents at more than one position."""
    p = task["prompts"][0]
    targets = p["target_positions"]
    ids = p["token_ids"][:max(targets) + 1]
    nat = ge.nat
    graph, _ = goal_eval.circuit(answer, targets, nat.C, len(ids))
    graph = goal_eval.strand(graph, targets, nat.C)
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    empty, vpd = ge._bounds(tx, "live")
    cap = {}
    kl_c = float(carry.kl(tx.ref, forward(ge, ids, targets, graph, capture=cap))[0])
    ref_kl = ge.graph_kl(tx, [graph], live=True)[0]
    f1 = math.log(empty / kl_c) / math.log(empty / vpd)
    budget = empty * (vpd / empty) ** GOAL if f1 >= GOAL else empty * (kl_c / empty) ** GOAL
    nodes = sorted(graph.node_set())
    reads = sorted({(native._layer_kind(nd)[0], nd[1]) for nd in nodes if native._layer_kind(nd)[1] == "o_proj"})
    out = {"task": task["id"], "T": len(ids), "kl_c": kl_c, "kl_c_graph_ref": ref_kl, "kl_empty": empty, "kl_vpd": vpd,
           "f1": f1, "budget": budget, "budget_rule": "F1 0.9" if f1 >= GOAL else "0.9 of own", "reads": []}
    by = {}
    for nd in nodes:
        if nd[0] != "tok":
            by.setdefault((nd[0], nd[2]), set()).add(nd[1])
    out["reuse"] = {"subcomponents": len(by), "multi_position": sum(len(v) > 1 for v in by.values()),
                    "examples": [f"{native._layer_kind((n, 0, c))}#{c}@{sorted(v)}" for (n, c), v in by.items() if len(v) > 1][:8]}
    log(f"{task['id']}: T {len(ids)}, {len(nodes)} nodes, KL_C {kl_c:.4f} (graph_ref {ref_kl:.4f}), F1 {f1:.3f}, budget {budget:.4f}, "
        f"{len(reads)} reads, {out['reuse']['multi_position']} of {len(by)} subcomponents at several positions")
    for layer, t in reads:
        t0 = time.time()
        bg, pairs, terms, qs, ks = pair_terms(ge, cap, graph, layer, t)
        full = bg + sum(terms) if terms else bg
        kl_full = float(carry.kl(tx.ref, forward(ge, ids, targets, graph, override=(layer, t, full[None])))[0])
        kl_bg = float(carry.kl(tx.ref, forward(ge, ids, targets, graph, override=(layer, t, bg[None])))[0])
        chosen, cur, curve = [], bg.clone(), [kl_bg]
        left = list(range(len(pairs)))
        while curve[-1] > budget and left:
            rows = torch.stack([cur + terms[j] for j in left])
            kls = carry.kl(tx.ref, forward(ge, ids, targets, graph, override=(layer, t, rows))).tolist()
            j = min(range(len(left)), key=lambda i: kls[i])
            chosen.append(left[j])
            cur = cur + terms[left[j]]
            curve.append(kls[j])
            left.pop(j)
        rec = {"layer": layer, "t": t, "pairs_available": len(pairs), "q_nodes": len(qs), "k_nodes": len(ks),
               "kl_background": kl_bg, "kl_full_row": kl_full,
               "curve": curve, "pairs": [[node_name(pairs[j][0]), node_name(pairs[j][1])] for j in chosen],
               "reached": curve[-1] <= budget, "seconds": time.time() - t0}
        out["reads"].append(rec)
        log(f"  read L{layer}@{t}: {len(pairs)} candidate pairs, background KL {kl_bg:.4f}, all pairs {kl_full:.4f}; "
            f"{len(chosen)} pairs -> {curve[-1]:.4f} ({'within' if rec['reached'] else 'NOT within'} budget) {rec['pairs'][:4]}")
    return out


def figure(rows: list[dict], path: str) -> None:
    sys.path.insert(0, str(Path.home() / "mpd-data/scratch/glead/figs"))
    import figstyle as fs
    import matplotlib.pyplot as plt
    import numpy as np

    fs.setup()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    counts = []
    for r in rows:
        for rd in r["reads"]:
            if rd["kl_background"] <= r["budget"]:  # the remainder's pattern alone already within budget: no pair needed
                continue
            c = np.array(rd["curve"]) / r["budget"]
            ax.plot(range(len(c)), c, color=fs.BLUE if rd["reached"] else fs.RED, alpha=0.45, lw=1.6)
            if rd["reached"]:
                counts.append(len(rd["curve"]) - 1)
    ax.axhline(1, color=fs.INK, lw=1.2)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8])
    ax.set_yticklabels(["1", "2", "4", "8"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("query-key pairs stated for the read")
    ax.set_ylabel("circuit's target KL / its budget")
    med = int(np.median(counts)) if counts else 0
    ax.set_title(f"Where the pattern matters, a median of {med} stated query-key pairs\nkeeps the circuit within its budget")
    fs.clean(ax)
    fs.save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--texts", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--dirs", nargs="*", default=[str(d) for d in DIRS])
    m.add_argument("--device", default="cpu")
    f = sub.add_parser("figure")
    f.add_argument("rows")
    f.add_argument("png")
    args = ap.parse_args()
    if args.cmd == "figure":
        return figure([json.loads(x) for x in Path(args.rows).read_text().splitlines() if x.strip()], args.png)
    ge = goal_eval.GoalEval(native.Native(args.device))
    for t in args.texts:
        task = json.loads((native.TEXTS / "vpd4l" / f"text{t}.json").read_text())
        src = next(Path(d) / f"text{t}.json" for d in args.dirs if (Path(d) / f"text{t}.json").exists())
        res = measure(ge, task, json.loads(src.read_text()), log=lambda s: print(s, flush=True))
        res["source"] = str(src)
        with open(args.out, "a") as fo:
            fo.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
