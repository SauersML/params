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

Also, per read: the read's row in C's run against the real model's row (KL, bits, mean over heads), its entropy
against the uniform causal row's, and the circuit's target KL with the uniform causal row and with the model's row
in its place (whether the pattern matters for the prediction); per layer the query and key remainders' norm against
W's. stated_pairs() gives teachers a circuit's pairs.

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
        if capture is not None:
            capture[i]["P"] = P[0].clone()
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


def prepare(ge, task: dict, answer: dict) -> dict:
    """A text's circuit in C's semantics: the run's capture, its KL, F1, budget and reads."""
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
    f1 = math.log(empty / kl_c) / math.log(empty / vpd)
    budget = empty * (vpd / empty) ** GOAL if f1 >= GOAL else empty * (kl_c / empty) ** GOAL
    nodes = sorted(graph.node_set())
    reads = sorted({(native._layer_kind(nd)[0], nd[1]) for nd in nodes if native._layer_kind(nd)[1] == "o_proj"})
    return {"task": task["id"], "ids": ids, "targets": targets, "graph": graph, "tx": tx, "cap": cap, "kl_c": kl_c,
            "kl_empty": empty, "kl_vpd": vpd, "f1": f1, "budget": budget, "reads": reads, "nodes": nodes}


def select_pairs(ge, pr: dict, layer: int, t: int) -> dict:
    """The read's pairs added greedily by measured target KL until the circuit is within its budget (module
    docstring): {"pairs": [(query node or "rem", key node or "rem")], "curve": KL after each, "kl_background",
    "kl_full_row", "q_nodes", "k_nodes", "available"}."""
    ids, targets, graph, tx = pr["ids"], pr["targets"], pr["graph"], pr["tx"]
    bg, pairs, terms, qs, ks = pair_terms(ge, pr["cap"], graph, layer, t)
    full = bg + sum(terms) if terms else bg
    kl_full = float(carry.kl(tx.ref, forward(ge, ids, targets, graph, override=(layer, t, full[None])))[0])
    kl_bg = float(carry.kl(tx.ref, forward(ge, ids, targets, graph, override=(layer, t, bg[None])))[0])
    chosen, cur, curve = [], bg.clone(), [kl_bg]
    left = list(range(len(pairs)))
    while curve[-1] > pr["budget"] and left:
        rows = torch.stack([cur + terms[j] for j in left])
        kls = carry.kl(tx.ref, forward(ge, ids, targets, graph, override=(layer, t, rows))).tolist()
        j = min(range(len(left)), key=lambda i: kls[i])
        chosen.append(left[j])
        cur = cur + terms[left[j]]
        curve.append(kls[j])
        left.pop(j)
    return {"pairs": [pairs[j] for j in chosen], "curve": curve, "kl_background": kl_bg, "kl_full_row": kl_full,
            "q_nodes": len(qs), "k_nodes": len(ks), "available": len(pairs)}


@torch.no_grad()
def stated_pairs(ge, task: dict, answer: dict, semantics: str = "c", split: str = "search") -> dict:
    """For teachers: a circuit's stated query-key pairs, {(layer, position): [(query node, key node), ...]} (nodes as
    (matrix, position, index), or "rem" for the part outside the circuit), measured by goal_eval's own runs on graphs
    with g.pairs (monitorbench's engine pair semantics: a read with an entry computes its row from the background and
    its stated pairs). Per read, the circuit's other reads left as they are, pairs (query node or rest x key node or
    rest) are added greedily:
      semantics "c":  by the circuit's target KL in C's semantics (GoalEval.graph_kl), until it is within the circuit's
                      budget (the KL at F1 GOAL, or GOAL of its own F1 when below);
      semantics "pp": by s_track_pp on the split's changed texts (GoalEval.track_pp: unstated pattern terms from the
                      model on x, stated pairs from the patched run), until GOAL of the gain from stating every
                      candidate pair over stating none is recovered.
    A read that needs none states ["pair", "rest", "rest"]: its pattern is the background alone (C) or x's (pp)."""
    import copy

    p = task["prompts"][0]
    targets = p["target_positions"]
    ids = p["token_ids"][:max(targets) + 1]
    nat = ge.nat
    graph, _ = goal_eval.circuit(answer, targets, nat.C, len(ids))
    graph = goal_eval.strand(graph, targets, nat.C)
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    nodes = sorted(graph.node_set())
    reads = sorted({(native._layer_kind(nd)[0], nd[1]) for nd in nodes if native._layer_kind(nd)[1] == "o_proj"})
    base = {**(getattr(graph, "pairs", None) or {})}

    def with_pairs(read, ps):
        g = copy.copy(graph)
        g.pairs = {**base, read: list(ps) or [(None, None)]}
        return g

    if semantics == "c":
        empty, vpd = ge._bounds(tx, "live")
        kl_c = ge.graph_kl(tx, [graph], live=True)[0]
        f1 = math.log(empty / kl_c) / math.log(empty / vpd)
        budget = empty * (vpd / empty) ** GOAL if f1 >= GOAL else empty * (kl_c / empty) ** GOAL
        score = lambda gs: [-k for k in ge.graph_kl(tx, gs, live=True)]  # noqa: E731
        enough = lambda s, lo, hi: -s <= budget  # noqa: E731
    else:
        score = lambda gs: [r["s_track_pp"] if r["s_track_pp"] is not None else 0.0 for r in ge.track_pp(tx, gs, split=split)]  # noqa: E731
        enough = lambda s, lo, hi: hi <= lo or s >= lo + GOAL * (hi - lo)  # noqa: E731
    out = {}
    for layer, t in reads:
        qs = [nd for nd in nodes if native._layer_kind(nd) == (layer, "q_proj") and nd[1] == t]
        ks = [nd for nd in nodes if native._layer_kind(nd) == (layer, "k_proj") and nd[1] <= t]
        cands = [(q, k) for q in qs + [None] for k in ks + [None] if q is not None or k is not None]
        read = (layer, t)
        lo, hi = score([with_pairs(read, []), with_pairs(read, cands)]) if cands else (0.0, 0.0)
        chosen, cur = [], lo
        left = list(cands)
        while left and not enough(cur, lo, hi):
            got = score([with_pairs(read, chosen + [c]) for c in left])
            j = max(range(len(left)), key=lambda i: got[i])
            chosen.append(left.pop(j))
            cur = got[j]
        out[read] = [("rem" if q is None else q, "rem" if k is None else k) for q, k in chosen]
    return out


def pair_edges(ge, task: dict, answer: dict, semantics: str = "c", split: str = "search") -> list:
    """stated_pairs as the answer's JSON pair edges (goal_eval.circuit): ["pair", query, key, layer, position], query
    and key a node's place in answer["nodes"] or "rest"; a read that needs none, ["pair", "rest", "rest", layer,
    position]."""
    at = {}
    for i, (m, c, t) in enumerate(answer.get("nodes", [])):
        if m != "tok":
            at[(goal_eval._site(m), int(t), int(c))] = i
    out = []
    for (layer, t), pairs in stated_pairs(ge, task, answer, semantics, split).items():
        out += [["pair", at[a] if a != "rem" else "rest", at[b] if b != "rem" else "rest", layer, t] for a, b in pairs]
        out += [] if pairs else [["pair", "rest", "rest", layer, t]]
    return out


def remainder_norms(nat) -> dict:
    """Per layer, |W - V U| / |W| (Frobenius) of the query and key matrices."""
    out = {}
    for layer in range(nat.target.n_layer):
        for kind in ("q_proj", "k_proj"):
            st = nat.target.site(native.site_name(layer, kind))
            out[f"L{layer}.{kind[0]}"] = float((st.W - (st.V @ st.U).T).norm() / st.W.norm())
    return out


def pattern_checks(ge, pr: dict, layer: int, t: int, model_P: torch.Tensor) -> dict:
    """The read's row in C's run against the model's (KL per head, mean over heads, bits), its entropy against the
    uniform causal row's (bits), and the circuit's target KL with the uniform causal row and with the model's row in
    its place."""
    rowc = pr["cap"][layer]["P"][:, t, :t + 1].clamp_min(1e-30)  # [H, t + 1]
    rowm = model_P[layer][:, t, :t + 1].clamp_min(1e-30)
    kl = float((rowm * (rowm.log() - rowc.log())).sum(-1).mean() / math.log(2))
    ent = float((-(rowc * rowc.log()).sum(-1)).mean() / math.log(2))
    ent_m = float((-(rowm * rowm.log()).sum(-1)).mean() / math.log(2))
    rows = torch.stack([torch.zeros_like(rowm), rowm.log()])
    kl_u, kl_m = carry.kl(pr["tx"].ref, forward(ge, pr["ids"], pr["targets"], pr["graph"], override=(layer, t, rows))).tolist()
    return {"kl_model_vs_circuit_row": kl, "entropy_circuit_row": ent, "entropy_model_row": ent_m,
            "entropy_uniform": math.log2(t + 1), "kl_target_uniform_row": kl_u, "kl_target_model_row": kl_m}


def model_patterns(nat, ids: list[int]) -> list:
    """The real model's attention patterns per layer [H, T, T]."""
    tg = nat.target
    out = []
    T = len(ids)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    for q, k, v, x in tg.keys_values(torch.tensor([ids], device=nat.dev)):
        S = (q[0] @ k[0].transpose(-1, -2)) / math.sqrt(tg.hd)
        out.append(S.masked_fill(~causal, float("-inf")).softmax(-1))
    return out


@torch.no_grad()
def measure(ge, task: dict, answer: dict, log=print) -> dict:
    """One text: per read the greedy pair curve and the pattern checks, and the circuit's subcomponents at more than
    one position."""
    nat = ge.nat
    pr = prepare(ge, task, answer)
    ref_kl = ge.graph_kl(pr["tx"], [pr["graph"]], live=True)[0]
    nodes, reads = pr["nodes"], pr["reads"]
    model_P = model_patterns(nat, pr["ids"])
    out = {"task": task["id"], "T": len(pr["ids"]), "kl_c": pr["kl_c"], "kl_c_graph_ref": ref_kl, "kl_empty": pr["kl_empty"],
           "kl_vpd": pr["kl_vpd"], "f1": pr["f1"], "budget": pr["budget"],
           "budget_rule": "F1 0.9" if pr["f1"] >= GOAL else "0.9 of own", "reads": []}
    by = {}
    for nd in nodes:
        if nd[0] != "tok":
            by.setdefault((nd[0], nd[2]), set()).add(nd[1])
    out["reuse"] = {"subcomponents": len(by), "multi_position": sum(len(v) > 1 for v in by.values()),
                    "examples": [f"{native._layer_kind((n, 0, c))}#{c}@{sorted(v)}" for (n, c), v in by.items() if len(v) > 1][:8]}
    log(f"{task['id']}: T {len(pr['ids'])}, {len(nodes)} nodes, KL_C {pr['kl_c']:.4f} (graph_ref {ref_kl:.4f}), F1 {pr['f1']:.3f}, "
        f"budget {pr['budget']:.4f}, {len(reads)} reads, {out['reuse']['multi_position']} of {len(by)} subcomponents at several positions")
    for layer, t in reads:
        t0 = time.time()
        sel = select_pairs(ge, pr, layer, t)
        rec = {"layer": layer, "t": t, "pairs_available": sel["available"], "q_nodes": sel["q_nodes"], "k_nodes": sel["k_nodes"],
               "kl_background": sel["kl_background"], "kl_full_row": sel["kl_full_row"], "curve": sel["curve"],
               "pairs": [[node_name(a), node_name(b)] for a, b in sel["pairs"]], "reached": sel["curve"][-1] <= pr["budget"],
               **pattern_checks(ge, pr, layer, t, model_P), "seconds": time.time() - t0}
        out["reads"].append(rec)
        log(f"  read L{layer}@{t}: {sel['available']} candidate pairs, background KL {sel['kl_background']:.4f}, all pairs "
            f"{sel['kl_full_row']:.4f}; {len(sel['pairs'])} pairs -> {sel['curve'][-1]:.4f}; uniform row {rec['kl_target_uniform_row']:.4f}, "
            f"model's row {rec['kl_target_model_row']:.4f}; "
            f"row KL vs model {rec['kl_model_vs_circuit_row']:.3f} bits, entropy {rec['entropy_circuit_row']:.2f} of {rec['entropy_uniform']:.2f}")
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
    print("remainder norms |W - V U| / |W|:", {k: round(v, 3) for k, v in remainder_norms(ge.nat).items()}, flush=True)
    for t in args.texts:
        task = json.loads((native.TEXTS / "vpd4l" / f"text{t}.json").read_text())
        src = next(Path(d) / f"text{t}.json" for d in args.dirs if (Path(d) / f"text{t}.json").exists())
        res = measure(ge, task, json.loads(src.read_text()), log=lambda s: print(s, flush=True))
        res["source"] = str(src)
        with open(args.out, "a") as fo:
            fo.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
