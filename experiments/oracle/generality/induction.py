"""The copy texts' mechanism in VPD subcomponents (induction), measured in the real model per copy text: the held-out
ones (DATA/copy_heldout.json, never used to pick the context core) and the 23 with teacher circuits (copytest.json).
Roles: t the target, a the copied token's position (after the earlier occurrence of the current token), o = a - 1 the
earlier occurrence.

  patterns: per layer and head, the attention weight from a to o and from t to a.
  pairs: the read (layer L, query position Q) toward key position K split into query-key subcomponent pairs: with
     q_Q = sum_i q_i + q_rem (q_i = act_i(Q) U_i, rank one) and k_P = sum_j k_j(P) + k_rem(P), a head's score at P is a
     sum of pair terms rope_Q(q_i) . rope_P(k_j) / sqrt(hd) (and the remainder's). A pair's lift toward K is its term at
     K less its mean term over the row's other positions (linear in k: q_i . (k_j(K) - mean_P k_j(P))); the lifts sum to
     the score at K less the row's mean. Per read (2, t -> a), (3, t -> a) and (1, a -> o), the head that attends most,
     its top pairs by lift.
  removals: single subcomponent removals (goal_eval.real_removal: mask 0 there, everything else on, the remainders
     live), KL at the target in bits, of every VPD-active subcomponent of the role sites (ROLE_SITES), and each role
     site's whole block (every subcomponent there).

  induction.py measure [--tasks ...]      -> DATA/induction.jsonl (one line per text, resumable)
  induction.py report                     -> the recurring pairs and removals across held-out texts
"""
import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("ORACLE") or str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
sys.path.insert(0, str(Path(os.environ.get("ORACLE") or Path(__file__).resolve().parents[3] / "method" / "oracle").parent / "vpd"))

import native  # noqa: E402

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
ROLE_SITES = {"t": ("h.2.attn.q_proj", "h.3.attn.q_proj", "h.2.attn.o_proj", "h.3.attn.o_proj"),
              "a": ("h.1.attn.q_proj", "h.1.attn.o_proj", "h.2.attn.k_proj", "h.2.attn.v_proj", "h.3.attn.k_proj",
                    "h.3.attn.v_proj", "h.0.mlp.c_fc", "h.1.mlp.c_fc"),
              "o": ("h.1.attn.k_proj", "h.1.attn.v_proj", "h.0.attn.v_proj")}
READS = (("2", "t", "a"), ("3", "t", "a"), ("1", "a", "o"))  # (layer, query role, key role)
TOP_PAIRS = 12


def copy_texts():
    held = [x["task"] for x in json.loads((DATA / "copy_heldout.json").read_text())["copy"]]
    sel = [x["task"] for x in json.loads((DATA / "copytest.json").read_text()) if x["drop_nats"] >= 1 and x["new_gain_nats"] >= 1]
    return [(t, "heldout") for t in held] + [(t, "selection") for t in sel]


def roles(ids, t):
    occ = [j for j in range(t) if ids[j] == ids[t]]
    return {"t": t, "a": occ[-1] + 1, "o": occ[-1]}


def rope_at(tg, vec, pos):
    n = tg.hd // 2
    return vec * tg.cos[pos] + torch.cat([-vec[..., n:], vec[..., :n]], dim=-1) * tg.sin[pos]


def pair_lifts(nat, ids, layer, Q, K, P_model):
    """The read (layer, Q -> K): the head attending most, and its top pairs by lift, with the rest's lifts."""
    import vpd_model

    tg = nat.target
    hd = tg.hd
    x = None
    for i, (_, _, _, xi) in enumerate(tg.keys_values(torch.tensor([ids], device=nat.dev))):
        if i == layer:
            x = xi[0]
    z = vpd_model.rms(x, tg.norms[2 * layer], tg.eps)  # [T, d]
    nq, nk = native.site_name(layer, "q_proj"), native.site_name(layer, "k_proj")
    sq, sk = tg.site(nq), tg.site(nk)
    h = int(P_model[layer][:, Q, K].argmax())
    sl = slice(h * hd, (h + 1) * hd)
    aq = z[Q] @ sq.V  # [Cq]
    qi = rope_at(tg, (aq[:, None] * sq.U)[:, sl], Q)  # [Cq, hd]
    qrem = rope_at(tg, (z[Q] @ sq.delta_T)[sl], Q)  # [hd]
    others = [p for p in range(Q + 1) if p != K]
    ak = z[:Q + 1] @ sk.V  # [Q+1, Ck]
    kj = torch.stack([rope_at(tg, (ak[p][:, None] * sk.U)[:, sl], p) for p in range(Q + 1)])  # [Q+1, Ck, hd]
    krem = torch.stack([rope_at(tg, (z[p] @ sk.delta_T)[sl], p) for p in range(Q + 1)])  # [Q+1, hd]
    dk = kj[K] - kj[others].mean(0)  # [Ck, hd]
    dkrem = krem[K] - krem[others].mean(0)
    scale = 1 / math.sqrt(hd)
    L = (qi @ dk.T) * scale  # [Cq, Ck]
    q_rem = (qrem @ dk.T) * scale  # [Ck]: (rem, k_j)
    rem_k = (qi @ dkrem) * scale  # [Cq]: (q_i, rem)
    rem_rem = float(qrem @ dkrem) * scale
    flat = L.flatten()
    top = flat.abs().topk(TOP_PAIRS).indices.tolist()
    Ck = L.shape[1]
    total = float(flat.sum() + q_rem.sum() + rem_k.sum()) + rem_rem
    return {"head": h, "weight": float(P_model[layer][h, Q, K]), "lift_total": total, "rem_rem": rem_rem,
            "pairs": [[int(f // Ck), int(f % Ck), float(flat[f])] for f in top],
            "q_rem_top": [[int(j), float(q_rem[j])] for j in q_rem.abs().topk(6).indices.tolist()],
            "rem_k_top": [[int(i), float(rem_k[i])] for i in rem_k.abs().topk(6).indices.tolist()],
            "q_sum": float(L.sum(1).abs().max()), "k_sum": float(L.sum(0).abs().max()),
            "q_lift": [[int(i), float(v)] for i, v in zip(*[x.tolist() for x in (L.sum(1) + rem_k).abs().topk(8)][::-1])],
            "k_lift": [[int(j), float(v)] for j, v in zip(*[x.tolist() for x in (L.sum(0) + q_rem).abs().topk(8)][::-1])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("measure", "report"))
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--max", type=int, default=0, help="texts this process measures (0: all left)")
    args = ap.parse_args()
    if args.cmd == "report":
        return report()
    import goal_eval

    out = DATA / "induction.jsonl"
    done = {json.loads(line)["task"] for line in out.read_text().splitlines()} if out.exists() else set()
    ge = goal_eval.GoalEval(native.Native(), chunk=16)
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    n = 0
    for task, kind in copy_texts():
        if task in done or (args.tasks and task not in args.tasks):
            continue
        if args.max and n >= args.max:
            break
        n += 1
        p = TEXTS / f"{task}.json"
        ids, targets = native.text(p)
        tx = ge.text(ids, targets, native.task_seed(task), json.loads(p.read_text()))
        r = roles(ids, targets[-1])
        with torch.no_grad():
            Pm = [P[0].float() for P in _patterns(nat, ids)]
        row = {"task": task, "kind": kind, "roles": r,
               "patterns": {"a_to_o": [[float(Pm[L][h, r["a"], r["o"]]) for h in range(nat.target.n_head)] for L in range(len(Pm))],
                            "t_to_a": [[float(Pm[L][h, r["t"], r["a"]]) for h in range(nat.target.n_head)] for L in range(len(Pm))]}}
        with torch.no_grad():
            row["reads"] = {f"{L}:{qr}->{kr}": pair_lifts(nat, ids, int(L), r[qr], r[kr], Pm) for L, qr, kr in READS}
        graphs, keys = [], []
        for rl, sites in ROLE_SITES.items():
            pos = r[rl]
            for n_ in sites:
                act = tx.vpd[n_][pos].nonzero().flatten().tolist()
                for c in act:
                    graphs.append(native.Graph([(n_, pos, c)], complete=True))
                    keys.append((rl, n_, c))
                graphs.append(native.Graph([(n_, pos, c) for c in range(nat.C[n_])], complete=True))
                keys.append((rl, n_, "block"))
        kl = [x["kl_real_minus"] for x in ge.real_removal(tx, graphs)]
        row["removals"] = [[rl, n_, c, v] for (rl, n_, c), v in zip(keys, kl)]
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(task, kind, {k: (v["head"], round(v["weight"], 2)) for k, v in row["reads"].items()}, flush=True)
        ge.probes.clear()
        native.free(nat.dev)


def _patterns(nat, ids):
    tg = nat.target
    T = len(ids)
    causal = torch.ones(T, T, dtype=torch.bool, device=nat.dev).tril()
    out = []
    for q, k, v, x in tg.keys_values(torch.tensor([ids], device=nat.dev)):
        S = (q[0] @ k[0].transpose(-1, -2)) / math.sqrt(tg.hd)
        out.append(S.masked_fill(~causal, float("-inf")).softmax(-1)[None])
    return out


def report():
    rows = [json.loads(line) for line in (DATA / "induction.jsonl").read_text().splitlines()]
    for kind in ("heldout", "selection"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        print(f"== {kind}: {len(rs)} texts")
        for key in rs[0]["reads"]:
            heads = collections.Counter(r["reads"][key]["head"] for r in rs)
            w = np.mean([r["reads"][key]["weight"] for r in rs])
            pairs = collections.Counter()
            ql, kl = collections.Counter(), collections.Counter()
            for r in rs:
                rd = r["reads"][key]
                pairs.update({(i, j) for i, j, v in rd["pairs"][:6] if v > 0})
                ql.update({i for i, v in rd["q_lift"][:4] if v > 0})
                kl.update({j for j, v in rd["k_lift"][:4] if v > 0})
            print(f"  read {key}: heads {heads.most_common(3)}, mean weight {w:.2f}")
            print(f"    recurring pairs (q, k) in the top 6 by lift: {pairs.most_common(8)}")
            print(f"    recurring queries: {ql.most_common(6)}  keys: {kl.most_common(6)}")
        rem = collections.defaultdict(list)
        for r in rs:
            for rl, n_, c, v in r["removals"]:
                rem[(rl, n_, c)].append(v)
        print("  removals: subcomponents costing >= 0.5 bits in the most texts")
        cnt = sorted(((sum(x >= 0.5 for x in v), k, float(np.mean(v))) for k, v in rem.items() if k[2] != "block"), reverse=True)
        for c, k, m in cnt[:16]:
            print(f"    {k[0]} {k[1]}#{k[2]}: {c}/{len(rs)} texts, mean {m:.2f} bits")
        print("  blocks (every subcomponent of the site there): mean bits")
        for k, v in sorted(((k, v) for k, v in rem.items() if k[2] == "block"), key=lambda kv: -np.mean(kv[1])):
            print(f"    {k[0]} {k[1]}: {np.mean(v):.2f}")


if __name__ == "__main__":
    main()
