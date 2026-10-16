"""Recurring neuron groups across question-scoped circuits against activation factors, judged by ablation on held-out
texts (#2951; the lead's 10-15 direction, Cammarata's "Already Clean": recurring connected neuron groups across many
attribution graphs predict ablation damage better than activation factors).

Texts: neuron_builder records (met) on one model, split within each family (the id before its first "_"; "f"/"g" ids:
their letter) into a mining half A and a test half B, alternately.
Groups (mined on A): every circuit as a set of position-free keys, neuron poles "L.n.+" / "L.n.-" (NODES: its necessity
ranking's top units, or its own units and routes "L.hH"). A pair of keys recurs when it shares more circuits than
chance: the one-sided hypergeometric p-value of its co-membership count given the two keys' counts, under ALPHA / (the
pairs with at least two shared circuits) (Bonferroni). A group is a community of the recurring pairs' graph (Louvain,
weighted by shared circuits; its connected components merged the keys every prompt shares: 100 and 67 of them on 150
Qwen3-1.7B family circuits), ablated through its neuron poles; its share is the share of A's circuits that hold at least
half of its keys.
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


NODES = "ranked"       # a circuit's units: "ranked", its necessity ranking's top units (record "ranked_top": the units
# whose removal alone moves the answer most, an attribution graph's nodes measured), or "circuit", its nodes and routes
# (necessity sets of 1-2 neurons and routes: 1 recurring pole pair on 150 Qwen3-1.7B family circuits)


def keys(rec: dict) -> set:
    """A circuit's position-free keys: neuron poles (and with NODES "circuit" its routes)."""
    nodes = rec["ranked_top"] if NODES == "ranked" else rec["nodes"]
    out = {f"{u['layer']}.{u['neuron']}.{'+' if u['pole'] > 0 else '-'}" for u in nodes}
    return out | ({f"{r['layer']}.h{r['head']}" for r in rec.get("routes", [])} if NODES == "circuit" else set())


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
    import networkx as nx
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    comp = nx.algorithms.community.louvain_communities(G, weight="weight", seed=0) if edges else []
    groups = []
    for members in sorted(comp, key=lambda m: -len(m)):
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


# ---- two measured views (the lead's 10-15 grouping upgrade: ADAG's input and output views, each measured)

def harmonic(a: float, b: float) -> float:
    a, b = max(a, 0.0), max(b, 0.0)
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def cos(u, v) -> float:
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    return 0.0 if nu == 0 or nv == 0 else sum(x * y for x, y in zip(u, v)) / (nu * nv)


def question(i: str) -> str:
    """A circuit_finder question's family: its task and key word (every paraphrase in one split)."""
    return i.rsplit("/", 1)[0] if i.count("/") >= 2 else family(i)


def profiles(recs: list[dict], model_name: str, adapter: str | None, dtype: str, log=print) -> list[dict]:
    """Per circuit_finder record, each circuit neuron's two views, measured on the clean prompt in the model itself:
      input   its value's change when each earlier token's embedding (positions 1..T-1) is replaced by the prompt's mean
              embedding (a vector over source positions; zero past the neuron's own position);
      output  the change of log p of the answer's word and of each alternative word (answer_contrast's sets) when the
              neuron alone is zeroed at its position."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
    import answer_contrast as ac

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype), device_map="cuda" if torch.cuda.is_available() else None)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model = model.eval().requires_grad_(False)
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers, dev = core.model.layers, core.model.embed_tokens.weight.device
    out = []
    for r in recs:
        ids, nodes = r["ids"], r["nodes"]
        T = len(ids)
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            lp0 = torch.log_softmax(core(input_ids=x, logits_to_keep=1).logits[0, -1].float(), -1)
        sets = ac.token_sets(core, tok, ids, lp0, r.get("corrupt_answer"))
        words = [sets["answer_word"]] + sets["alternative_words"]
        toks = [[z for z in sets["answer"]]] + [[z for z in sets["alternatives"] if sets["words"][z] == w] for w in sets["alternative_words"]]

        def word_lp(lp):                                            # [B, V] -> [B, words]
            return torch.stack([torch.logsumexp(lp[:, t], -1) for t in toks], -1)
        where = [(n["layer"], n["pos"], n["neuron"]) for n in nodes]

        def run(rows: int, neu=None, emb=None, cap=None):
            hs = []
            if emb is not None:
                hs.append(core.model.embed_tokens.register_forward_hook(lambda m, a, o: emb(o)))
            for i, layer in enumerate(layers):
                def pre(m, a, i=i):
                    h = a[0]
                    if neu is not None:
                        h = neu(i, h)
                    if cap is not None:
                        cap.append((i, h.detach()))
                    return (h,) + tuple(a[1:])
                hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
            try:
                with torch.no_grad():
                    return torch.log_softmax(core(input_ids=x.expand(rows, -1), logits_to_keep=1).logits[:, -1].float(), -1)
            finally:
                for h in hs:
                    h.remove()
        R = ROWS_VIEWS
        base = word_lp(run(R))[0]                                    # (every run a batch of R rows)
        outp = []
        for s0 in range(0, len(where), R):                           # output view: each node zeroed alone
            part = where[s0:s0 + R]

            def neu(i, h, part=part):
                for b, (L, p, n) in enumerate(part):
                    if L == i:
                        h = h.clone()
                        h[b, p, n] = 0
                return h
            outp += (word_lp(run(R, neu=neu))[:len(part)] - base[None]).tolist()
        cap0 = []
        run(R, cap=cap0)
        h0 = {i: h[0] for i, h in cap0}
        inp = [[0.0] * (T - 1) for _ in where]
        srcs = list(range(1, T))
        for s0 in range(0, len(srcs), R):                            # input view: each source token's embedding replaced
            part = srcs[s0:s0 + R]

            def emb(o, part=part):
                o = o.clone()
                m = o[0].mean(0)
                for b, q in enumerate(part):
                    o[b, q] = m
                return o
            cap = []
            run(R, emb=emb, cap=cap)
            hs = {i: h for i, h in cap}
            for j, (L, p, n) in enumerate(where):
                d = (hs[L][:len(part), p, n].float() - h0[L][p, n].float()).tolist()
                for b, q in enumerate(part):
                    inp[j][q - 1] = d[b] if q <= p else 0.0
        out.append({"id": r["id"], "question": question(r["id"]), "words": words, "nodes": [
            {"key": f"{L}.{n}.{'+' if nd['pole'] > 0 else '-'}", "layer": L, "pos": p, "neuron": n, "pole": nd["pole"],
             "stage": nd.get("stage"), "input": inp[j], "output": outp[j]} for j, ((L, p, n), nd) in enumerate(zip(where, nodes))]})
        log(f"views {r['id']}: {len(where)} neurons, words {words}")
    return out


def views_group(profs: list[dict], labels: dict | None = None) -> dict:
    """Groups of circuit neurons (position-free keys) by both measured views (the lead's 10-15): two neurons are similar
    by the harmonic mean of their input and output cosines, each clamped at zero, averaged over the questions where both
    appear; Louvain communities by modularity (the number of groups is Louvain's); a group whose zeroing raises the
    answer (it pushes against it) is named not[...]. Beside it, one view: the cosine of the joined unit-norm profiles,
    unclamped, its own Louvain groups. Scores: silhouette (distance 1 - similarity, 1 for pairs never together), group
    sizes, and the share of within-group pairs whose answer effects have opposite signs."""
    import networkx as nx

    best = []                                                         # per question, each key once (its largest effect)
    eff = {}
    for pr in profs:
        by = {}
        for n in pr["nodes"]:
            if n["key"] not in by or abs(n["output"][0]) > abs(by[n["key"]]["output"][0]):
                by[n["key"]] = n
        best.append(by)
        for k, n in by.items():
            eff.setdefault(k, []).append(n["output"][0])
    effect = {k: sum(v) / len(v) for k, v in eff.items()}                # mean change of log p(answer) when zeroed
    two, one = {}, {}
    for by in best:
        ks = sorted(by)
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                u, v = by[ks[a]], by[ks[b]]
                ci, co = cos(u["input"], v["input"]), cos(u["output"], v["output"])
                key = (ks[a], ks[b])
                two.setdefault(key, []).append(harmonic(ci, co))
                nu = lambda w: [z / max(math.sqrt(sum(t * t for t in w)), 1e-12) for z in w]  # noqa: E731
                one.setdefault(key, []).append(cos(nu(u["input"]) + nu(u["output"]), nu(v["input"]) + nu(v["output"])))
    keys = sorted(effect)

    def group(sim: dict, tag: str) -> dict:
        S = {k: sum(v) / len(v) for k, v in sim.items()}
        G = nx.Graph()
        G.add_nodes_from(keys)
        G.add_weighted_edges_from((a, b, w) for (a, b), w in S.items() if w > 0)
        comm = nx.algorithms.community.louvain_communities(G, weight="weight", seed=0) if G.number_of_edges() else [{k} for k in keys]
        Q = nx.algorithms.community.modularity(G, comm, weight="weight") if G.number_of_edges() else 0.0
        member = {k: j for j, c in enumerate(comm) for k in c}
        d = lambda a, b: 1.0 - S.get((min(a, b), max(a, b)), 0.0)  # noqa: E731
        sil = []
        for k in keys:
            own = [x for x in comm[member[k]] if x != k]
            if not own or len(comm) < 2:
                continue
            ai = sum(d(k, x) for x in own) / len(own)
            bi = min(sum(d(k, x) for x in c) / len(c) for j, c in enumerate(comm) if j != member[k])
            sil.append((bi - ai) / max(ai, bi, 1e-12))
        pairs = [(a, b) for c in comm for a in c for b in c if a < b]
        opp = sum((effect[a] > 0) != (effect[b] > 0) for a, b in pairs)
        groups = []
        for c in sorted(comm, key=lambda c: -len(c)):
            ms = sorted(c, key=lambda k: -abs(effect[k]))
            e = sum(effect[k] for k in ms) / len(ms)
            names = [(labels or {}).get(k) or k for k in ms[:3]]
            groups.append({"keys": ms, "poles": ms, "effect": e, "name": ("not[" if e > 0 else "[") + "; ".join(names) + "]"})
        sizes = sorted((len(g["keys"]) for g in groups), reverse=True)
        return {"view": tag, "groups": groups, "modularity": Q, "chosen_by": "Louvain modularity (resolution 1)",
                "silhouette": sum(sil) / len(sil) if sil else None, "sizes": sizes,
                "size_median": sizes[len(sizes) // 2] if sizes else 0, "opposite_sign_share": opp / max(len(pairs), 1),
                "pairs_within": len(pairs)}
    return {"keys": len(keys), "questions": len(profs), "two_views": group(two, "two views (harmonic mean, clamped)"),
            "one_view": group(one, "one view (joined profiles, unclamped)"), "effects": effect}


def views_damage(groups: list[dict], recs: list[dict], model_name: str, adapter: str | None, dtype: str, rows: int = 16,
                 seed: int = 0, log=print) -> dict:
    """Each group's poles zeroed at every position on unseen questions' clean prompts in the model itself; damage = the
    drop of the record's answer contrast (answer_contrast); its users are the questions whose circuit holds at least half
    of the group's poles; beside each group, a random set of as many poles drawn from the test circuits."""
    import random

    import torch
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
    import answer_contrast as ac
    import neuron_builder as nb

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype), device_map="cuda" if torch.cuda.is_available() else None)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model = model.eval().requires_grad_(False)
    nr = nb.Neurons(model, None)
    rng = random.Random(seed)
    pool = sorted({f"{n['layer']}.{n['neuron']}.{'+' if n['pole'] > 0 else '-'}" for r in recs for n in r["nodes"]})
    sets = [("group", j, g["poles"]) for j, g in enumerate(groups)] + \
           [("random", j, rng.sample(pool, min(len(g["poles"]), len(pool)))) for j, g in enumerate(groups)]
    units = [[(int(k.split(".")[0]), int(k.split(".")[1]), 1 if k.endswith("+") else -1) for k in s[2]] for s in sets]
    dmg = torch.zeros(len(recs), len(sets))
    for t, r in enumerate(recs):
        ids, cs = r["ids"], {"answer": r.get("answer_tokens") or [r["answer"]], "alternatives": r["alternatives"]}
        with torch.no_grad():
            c0 = float(ac.contrast(torch.log_softmax(nr.logits(ids, [len(ids) - 1], B=rows)[:1, 0], -1), cs)[0])
            for s0 in range(0, len(sets), rows):
                part = units[s0:s0 + rows]
                l = torch.log_softmax(nr.logits(ids, [len(ids) - 1], edit=nb.pole_edit(part), B=rows)[:len(part), 0], -1)
                dmg[t, s0:s0 + len(part)] = c0 - ac.contrast(l, cs).cpu()
    circ = [{f"{n['layer']}.{n['neuron']}.{'+' if n['pole'] > 0 else '-'}" for n in r["nodes"]} for r in recs]
    rows_out = []
    for j, (kind, rank, ps) in enumerate(sets):
        d = dmg[:, j].tolist()
        use = [2 * len(c & set(ps)) >= len(ps) for c in circ]
        du, dn = [x for x, u in zip(d, use) if u], [x for x, u in zip(d, use) if not u]
        rows_out.append({"kind": kind, "rank": rank, "size": len(ps), "damage_mean": sum(d) / len(d), "users": len(du),
                         "damage_users": sum(du) / max(len(du), 1), "damage_rest": sum(dn) / max(len(dn), 1), "auc": auc(d, use)})
    summ = {}
    for kind in ("group", "random"):
        r = [x for x in rows_out if x["kind"] == kind]
        a = [x["auc"] for x in r if x["auc"] is not None]
        summ[kind] = {"sets": len(r), "damage_mean": sum(x["damage_mean"] for x in r) / max(len(r), 1),
                      "mean_auc": sum(a) / max(len(a), 1), "with_users": len(a)}
    log(f"damage on {len(recs)} unseen questions: {json.dumps(summ)}")
    return {"rows": rows_out, "summary": summ, "test": [r["id"] for r in recs]}


def views(recs: list[dict], model_name: str, adapter: str | None, dtype: str, labels: dict | None = None, log=print) -> dict:
    """Profiles on every record, groups mined on half the questions (alternate question families) and their removal
    measured on the other half."""
    seen, split = {}, []
    for r in recs:
        q = question(r["id"])
        split.append(seen.setdefault(q, len(seen)) % 2)
    A = [r for r, s in zip(recs, split) if s == 0]
    B = [r for r, s in zip(recs, split) if s == 1]
    profs = profiles(A, model_name, adapter, dtype, log=log)
    got = views_group(profs, labels)
    for tag in ("two_views", "one_view"):
        g = got[tag]
        log(f"{g['view']}: {len(g['groups'])} groups ({g['chosen_by']}, modularity {g['modularity']:.3f}), silhouette "
            f"{g['silhouette']}, sizes {g['sizes'][:12]}, opposite-sign share {g['opposite_sign_share']:.3f} of {g['pairs_within']} pairs")
    big = [g for g in got["two_views"]["groups"] if len(g["poles"]) >= MIN_SIZE]
    got["damage"] = views_damage(big, B, model_name, adapter, dtype, log=log) if big and B else None
    got.update(A=[r["id"] for r in A], B=[r["id"] for r in B], model=model_name, adapter=adapter, profiles=profs)
    return got


ROWS_VIEWS = 16


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
    ap.add_argument("cmd", choices=("mine", "run", "views"))
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model")
    ap.add_argument("--thr")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--nodes", default=NODES, choices=("ranked", "circuit"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--labels", type=Path, default=None, help="views: atlas_labels' LABELS.jsonl (names per pole)")
    a = ap.parse_args()
    globals()["NODES"] = a.nodes
    recs = load(a.records)
    if a.cmd == "mine":
        g = mine(recs)
        a.out.write_text(json.dumps(g))
        print(f"{g['texts']} circuits, {g['keys']} keys ({len(g['recurring_keys'])} in two or more), {g['pairs']} of "
              f"{g['pairs_tested']} pairs recur; {len(g['groups'])} groups, sizes {[len(x['keys']) for x in g['groups']][:20]}")
        return
    if a.cmd == "views":
        labels = {}
        if a.labels and a.labels.exists():
            for line in a.labels.read_text().splitlines():
                x = json.loads(line)
                labels[f"{x['layer']}.{x['index']}.{'+' if x.get('pole') in ('pos', '+', 1) else '-'}"] = x.get("explanation")
        out = views(recs, a.model, a.adapter, a.dtype, labels)
        a.out.write_text(json.dumps(out))
        return
    out = run(recs, a.model, a.thr, a.dtype)
    a.out.write_text(json.dumps(out))
    print(json.dumps(out.get("summary")))


if __name__ == "__main__":
    main()
