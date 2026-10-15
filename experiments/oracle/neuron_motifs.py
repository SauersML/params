"""Recurring neuron groups across question-scoped circuits against activation factors, judged by ablation on held-out
texts (#2951; the lead's 10-15 direction, Cammarata's "Already Clean": recurring connected neuron groups across many
attribution graphs predict ablation damage better than activation factors).

Texts: neuron_builder records (met) on one model, split within each family (the id before its first "_"; "f"/"g" ids:
their letter) into a mining half A and a test half B, alternately.
Groups (mined on A): every circuit as a set of position-free keys, a neuron pole "L.n.+" / "L.n.-" (its necessity and
sufficiency units) or a route "L.hH" (a head into a position). A pair of keys recurs when it shares more circuits than
chance: the one-sided hypergeometric p-value of its co-membership count given the two keys' counts, under ALPHA / (the
pairs with at least two shared circuits) (Bonferroni). A group is a connected component of the recurring pairs, ablated
through its neuron poles; its share is the share of A's circuits that hold at least half of its keys.
Factors (fitted on A): A's activation matrix over every pole in two or more of A's circuits, each entry the pole's
largest activation of its sign at positions 1..answer as a share of that neuron's largest |value| on the calibration
chats, factored by non-negative matrix factorization into as many factors as groups; factor f keeps its largest-loading
poles, as many as group f has, and group f's share (sizes and shares matched rank by rank).
Use on B, each rule applied alike to groups and factors:
  activation  the text's mean activation (as above) over the set's poles is in the top share of B's texts
  circuit     the text's own circuit holds at least half of the set's poles
Ablation on B: the set's poles zeroed at every position (neuron_builder.pole_edit) in the weak-zeroed reference; damage
= the drop of the text's answer contrast (its record's answer against its alternatives). Per set and rule: the mean
damage on users and on the rest and the AUC of use predicting damage. Every number is a measured run on held-out texts;
nothing is fitted to the damage.

  neuron_motifs.py mine RECORDS_DIR [...] --out GROUPS.json             (every met record, no split: inspection)
  neuron_motifs.py run RECORDS_DIR [...] --model M --thr THR.pt --out OUT.json [--dtype float32]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ALPHA = 0.05
MIN_SIZE = 2


def keys(rec: dict) -> set:
    """A circuit's position-free keys: neuron poles and routes."""
    out = {f"{u['layer']}.{u['neuron']}.{'+' if u['pole'] > 0 else '-'}" for u in rec["nodes"]}
    return out | {f"{r['layer']}.h{r['head']}" for r in rec.get("routes", [])}


def poles_of(rec: dict) -> set:
    return {k for k in keys(rec) if ".h" not in k}


def family(i: str) -> str:
    return i.split("_", 1)[0] if "_" in i else i[:1]


def log_sf(k: int, N: int, K: int, n: int) -> float:
    """log P(X >= k), X hypergeometric: n draws from N with K successes."""
    lc = lambda a, b: math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)  # noqa: E731
    terms = [lc(K, i) + lc(N - K, n - i) - lc(N, n) for i in range(k, min(K, n) + 1)]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def mine(recs: list[dict]) -> dict:
    """Recurring pairs and their connected components over recs' circuits (module docstring)."""
    sets = [keys(r) for r in recs]
    N = len(sets)
    cnt = {}
    for s in sets:
        for k in s:
            cnt[k] = cnt.get(k, 0) + 1
    common = sorted(k for k, c in cnt.items() if c >= 2)
    idx = {k: j for j, k in enumerate(common)}
    pair = {}
    for s in sets:
        ks = sorted(idx[k] for k in s if k in idx)
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                pair[(ks[a], ks[b])] = pair.get((ks[a], ks[b]), 0) + 1
    tested = {p: c for p, c in pair.items() if c >= 2}
    cut = math.log(ALPHA / max(len(tested), 1))
    edges = [(a, b, c) for (a, b), c in tested.items() if log_sf(c, N, cnt[common[a]], cnt[common[b]]) <= cut]
    parent = list(range(len(common)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _ in edges:
        parent[root(a)] = root(b)
    comp = {}
    for a, b, _ in edges:
        comp.setdefault(root(a), set()).update((a, b))
    groups = []
    for members in sorted(comp.values(), key=lambda m: -len(m)):
        ks = sorted(common[j] for j in members)
        if len(ks) < MIN_SIZE:
            continue
        users = [r["id"] for r, s in zip(recs, sets) if 2 * len(s & set(ks)) >= len(ks)]
        groups.append({"keys": ks, "poles": [k for k in ks if ".h" not in k], "users": users, "share": len(users) / N,
                       "pairs": [[common[a], common[b], c] for a, b, c in edges if a in members]})
    return {"texts": N, "keys": len(cnt), "recurring_keys": common, "pairs_tested": len(tested), "pairs": len(edges),
            "groups": groups}


def auc(score: list[float], pos: list[bool]) -> float | None:
    """P(a user's damage > a non-user's), ties half."""
    a = [s for s, p in zip(score, pos) if p]
    b = [s for s, p in zip(score, pos) if not p]
    if not a or not b:
        return None
    return sum((x > y) + 0.5 * (x == y) for x in a for y in b) / (len(a) * len(b))


def nmf(A, k: int, iters: int = 1000, seed: int = 0):
    """A [n, m] >= 0 ~ W [n, k] H [k, m] (Lee-Seung multiplicative updates on the squared error)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    n, m = A.shape
    W = torch.rand(n, k, generator=g, dtype=A.dtype) + 0.1
    H = torch.rand(k, m, generator=g, dtype=A.dtype) + 0.1
    for _ in range(iters):
        H *= (W.T @ A) / (W.T @ W @ H + 1e-12)
        W *= (A @ H.T) / (W @ H @ H.T + 1e-12)
    return W, H


def run(recs: list[dict], model_name: str, thr: str, dtype: str, log=print) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
    import neuron_builder as nb

    seen = {}
    split = []
    for r in recs:
        f = family(r["id"])
        split.append(seen.get(f, 0) % 2)
        seen[f] = seen.get(f, 0) + 1
    A_recs = [r for r, s in zip(recs, split) if s == 0]
    B_recs = [r for r, s in zip(recs, split) if s == 1]
    mined = mine(A_recs)
    G = [g for g in mined["groups"] if g["poles"]]
    log(f"mined on {len(A_recs)} circuits: {mined['pairs']} recurring pairs, {len(G)} groups with poles, sizes "
        f"{[len(g['poles']) for g in G][:30]}")
    if not G:
        return {"model": model_name, "mined": mined, "rows": []}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).to(dev).eval().requires_grad_(False)
    nr = nb.Neurons(model, tok)
    nr.load(thr)
    cnt = {}
    for r in A_recs:
        for p in poles_of(r):
            cnt[p] = cnt.get(p, 0) + 1
    poles = sorted({p for p, c in cnt.items() if c >= 2} | {p for g in G for p in g["poles"]})
    col = {p: j for j, p in enumerate(poles)}
    pl = [(int(p.split(".")[0]), int(p.split(".")[1]), 1 if p.endswith("+") else -1) for p in poles]
    pl_t = torch.tensor(pl, device=dev)

    def con(l, y, alt):
        return (l[..., y] - torch.logsumexp(l[..., alt], -1)) / math.log(2)

    def prep(r):
        ids = r.get("ids") or nb.chat(tok, r["user"])
        return ids, r["pos"], r["answer"], torch.tensor(r["alternatives"], device=dev)

    def acts(rs):
        """[texts, poles] activations (module docstring) and each text's reference contrast."""
        X, c0 = torch.zeros(len(rs), len(poles), dtype=torch.float64), []
        for t, r in enumerate(rs):
            ids, pos, y, alt = prep(r)
            cap = {}
            with torch.no_grad():
                l = torch.log_softmax(nr.logits(ids, [pos], capture=cap, B=nb.ROWS)[0, 0], -1)
            c0.append(float(con(l, y, alt)))
            h = torch.stack([cap["h"][L][0, 1:pos + 1, n] for L, n, _ in pl], -1).float()        # [pos, poles]
            v = (h * pl_t[:, 2].float()).clamp(min=0).max(0).values / nr.mx[pl_t[:, 0], pl_t[:, 1]].float().clamp(min=1e-9)
            X[t] = v.double().cpu()
        return X, c0
    XA, _ = acts(A_recs)
    XB, c0B = acts(B_recs)
    W, Hf = nmf(XA, len(G))
    sets = [("group", j, g["poles"], g["share"]) for j, g in enumerate(G)]
    sets += [("factor", f, [poles[j] for j in Hf[f].argsort(descending=True)[:len(G[f]["poles"])].tolist()], G[f]["share"])
             for f in range(len(G))]
    dmg = torch.zeros(len(B_recs), len(sets))
    units = [[(int(p.split(".")[0]), int(p.split(".")[1]), 1 if p.endswith("+") else -1) for p in s[2]] for s in sets]
    for t, r in enumerate(B_recs):                                    # damage: every set on the text, ROWS a batch
        ids, pos, y, alt = prep(r)
        for s0 in range(0, len(sets), nb.ROWS):
            part = units[s0:s0 + nb.ROWS]
            with torch.no_grad():
                l = torch.log_softmax(nr.logits(ids, [pos], edit=nb.pole_edit(nb._pad(part)), B=nb.ROWS)[:, 0], -1)
            dmg[t, s0:s0 + len(part)] = torch.tensor(c0B[t]) - con(l, y, alt)[:len(part)].cpu()
        if t % 10 == 0:
            log(f"damage: {t + 1} of {len(B_recs)} texts")
    rows = []
    circ = [poles_of(r) for r in B_recs]
    for j, (kind, rank, ps, share) in enumerate(sets):
        d = dmg[:, j].tolist()
        m = XB[:, [col[p] for p in ps]].mean(1)
        n_use = max(1, round(share * len(B_recs)))
        top = set(m.argsort(descending=True)[:n_use].tolist())
        uses = {"activation": [t in top for t in range(len(B_recs))], "circuit": [2 * len(c & set(ps)) >= len(ps) for c in circ]}
        row = {"kind": kind, "rank": rank, "size": len(ps), "poles": ps, "share": share, "damage": d}
        for rule, use in uses.items():
            du = [x for x, u in zip(d, use) if u]
            dn = [x for x, u in zip(d, use) if not u]
            row[rule] = {"users": len(du), "damage_users": sum(du) / max(len(du), 1), "damage_rest": sum(dn) / max(len(dn), 1),
                         "auc": auc(d, use)}
        rows.append(row)
        log(f"{kind} {rank}: {len(ps)} poles, damage mean {sum(d) / len(d):.2f} bits; activation auc {row['activation']['auc']}, "
            f"circuit auc {row['circuit']['auc']} ({row['circuit']['users']} users)")
    summ = {}
    for kind in ("group", "factor"):
        for rule in ("activation", "circuit"):
            r = [x[rule] for x in rows if x["kind"] == kind and x[rule]["auc"] is not None]
            summ[f"{kind}/{rule}"] = {"sets": len(r), "mean_auc": sum(x["auc"] for x in r) / max(len(r), 1),
                                      "damage_users": sum(x["damage_users"] for x in r) / max(len(r), 1),
                                      "damage_rest": sum(x["damage_rest"] for x in r) / max(len(r), 1)}
    return {"model": model_name, "A": [r["id"] for r in A_recs], "B": [r["id"] for r in B_recs], "c0_B": c0B, "poles": poles,
            "mined": mined, "summary": summ, "rows": rows}


def load(dirs) -> list[dict]:
    recs = []
    for d in dirs:
        for f in sorted(d.iterdir()) if d.is_dir() else [d]:
            if f.suffix == ".json":
                r = json.loads(f.read_text())
                if r.get("met") and "nodes" in r:
                    recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("mine", "run"))
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model")
    ap.add_argument("--thr")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    a = ap.parse_args()
    recs = load(a.records)
    if a.cmd == "mine":
        g = mine(recs)
        a.out.write_text(json.dumps(g))
        print(f"{g['texts']} circuits, {g['keys']} keys ({len(g['recurring_keys'])} in two or more), {g['pairs']} of "
              f"{g['pairs_tested']} pairs recur; {len(g['groups'])} groups, sizes {[len(x['keys']) for x in g['groups']][:20]}")
        return
    out = run(recs, a.model, a.thr, a.dtype)
    a.out.write_text(json.dumps(out))
    print(json.dumps(out.get("summary")))


if __name__ == "__main__":
    main()
