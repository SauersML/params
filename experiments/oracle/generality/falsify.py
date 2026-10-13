"""The verifier's falsification set on copy texts (#2951 graph oracle, the lead's 10-12): can the scorecard tell the
correct copy mechanism from plausible wrong ones that pass ordinary output checks?

Texts: the 40 held-out copy texts (DATA/copy_heldout.json, train split, never used to pick the context core), and a
model-mistakes slice: positions inside other train texts where the token after the latest earlier occurrence of the
current token is the text's actual next token (a copy would be right) and vpd4l's likeliest token is another (the text
cut after that position). Development texts 9000-9099 and the train split only; the sealed set 9100-9149 never.
Roles: t the target, o the latest earlier occurrence of the current token, a = o + 1 the copied token.

Circuits (node sets with every connection the model has among them, blocks a site's whole output at a position):
  correct: the copy columns (copy_template.build_columns' sites: at o the L0 MLP and L1 value blocks with L1 keys #329
     and #119, at a the L0 MLP, L1 attention output, L1 MLP, L2 key and value blocks with L1 query #316, at t the L0/L1
     MLP, L2 query and output, L3 blocks) plus VPD's active subcomponents at t (the target's own computation);
  template: the copy columns without the target actives (the lead's point: does adding them lift sufficiency);
  (a) source moved: the a and o columns at a random earlier position q, q - 1 (not an occurrence or the token after one);
  (b) fixed offset: the a and o columns at t - D, D the median copy distance of the discovery texts (copytest.json);
  (c) backup: where the current token occurs at least twice earlier, the correct circuit plus the columns at the
     second-latest occurrence; the correct circuit is then the variant without the backup route;
  (d) irrelevant parts: the correct circuit plus the a column's blocks at two positions far from t, a and o;
  (g) mismatched: the a and o columns at another held-out copy text's offsets from its target;
  (h) mixture: the correct circuit plus the influential start set's positions other than a, o and t (VPD's actives
     there), against the correct circuit as the single-mechanism claim; per text whether the model mixes: with the
     match broken (fresh tokens at o) the share of draws whose likeliest token stays the text's;
  references: VPD's top k and recency (VPD's actives at t and the m positions before it) at the correct circuit's bits,
     and at the stated copy template's bits.
Scores: goal_eval.GoalEval.fidelity in the reward's mode on the eval split (point, nodes, flips at FLIP_TERM over the
CI-pattern pool, track_pp, interchange_pp, donors near and far, pattern, N; 16 node draws, 8 changed-input and
interchange draws, 8 near and 4 far donors), the connection-cut family left out for
every circuit (every connection among tens of thousands of block subcomponents is beyond memory), and reward v3's
circuit terms as components.reward forms them: Φ over the text's families (those the influential start scores,
ge.tau on the eval split; an unscored family at 0), capped at the text's bar Φ_τ + 1 (cap_phi), the gate g, bits
(size_bits, blocks at their active contents), compactness against the start's bits, and the circuit reward
Φ - Φ_τ + g x 0.5 compact (D left out: no teacher); every family's floored score and eps.
Variables (vars): E4 (variables.Scorer, eval split) of the copied token's identity carried by the a column's blocks,
correct at a and wrong at the source-moved position; the value map's predictions on fresh tokens at a: the identity
map (the prediction becomes the new token) against a random balanced partition of the vocabulary (a mechanism the model
does not run: same-class tokens leave the prediction, other-class tokens move it).

  falsify.py build                -> DATA/falsify_texts.json
  falsify.py score KIND TASK [T]  -> DATA/falsify.jsonl (KIND copy or mistake; T the mistake's target position)
  falsify.py vars KIND TASK [T]   -> DATA/falsify_vars.jsonl
  falsify.py report
"""
import argparse
import collections
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
POOL = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt"
MISTAKES = 20
SCAN = 2000  # train texts scanned for the mistakes slice (in order, until MISTAKES are found)
L = lambda layer, kind: f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"  # noqa: E731
O_SITES = [L(0, "c_fc"), L(0, "down_proj"), L(1, "v_proj")]
O_NAMED = [("h.1.attn.k_proj", 329), ("h.1.attn.k_proj", 119)]
A_SITES = [L(0, "c_fc"), L(0, "down_proj"), L(1, "o_proj"), L(1, "c_fc"), L(1, "down_proj"), L(2, "k_proj"), L(2, "v_proj")]
A_NAMED = [("h.1.attn.q_proj", 316)]
T_SITES = [L(0, "c_fc"), L(0, "down_proj"), L(1, "c_fc"), L(1, "down_proj"), L(2, "q_proj"), L(2, "o_proj"), L(3, "o_proj"),
           L(3, "c_fc"), L(3, "down_proj")]


def out_path(stem: str) -> Path:
    return Path(os.environ.get("GEN_OUT", DATA)) / f"{stem}{os.environ.get('GEN_TAG', '')}.jsonl"


def rng_of(*key):
    return np.random.default_rng(int(hashlib.sha256(repr(key).encode()).hexdigest()[:15], 16))


def occurrences(ids, t):
    return [j for j in range(t) if ids[j] == ids[t]]


def build():
    import native

    held = json.loads((DATA / "copy_heldout.json").read_text())
    copies = [x["task"] for x in held["copy"]]
    disc = [x for x in json.loads((DATA / "copytest.json").read_text()) if x["drop_nats"] >= 1 and x["new_gain_nats"] >= 1]
    D = max(2, int(np.median([x["dist"] for x in disc])) - 1)  # copytest's dist is t - o; t - a is one less
    disc_tokens, held_tokens = set(), {}
    for x in disc:
        ids, targets = native.text(TEXTS / f"{x['task']}.json")
        t = targets[-1]
        disc_tokens.add(ids[occurrences(ids, t)[-1] + 1])
    texts = []
    for task in copies:
        ids, targets = native.text(TEXTS / f"{task}.json")
        t = targets[-1]
        occ = occurrences(ids, t)
        a = occ[-1] + 1
        held_tokens[task] = ids[a]
        texts.append(roles_entry("copy", task, ids, t, D))
    others = [(e["task"], e["t"] - e["a"]) for e in texts]
    for e in texts:  # (g): another held-out copy text's offsets
        r = rng_of("mismatch", e["task"])
        for _ in range(50):
            b, d = others[int(r.integers(len(others)))]
            q = e["t"] - d
            if b != e["task"] and 2 <= q < e["t"] and q not in e["taken"]:
                e["mismatch"] = {"from": b, "a": q}
                break
    # the model-mistakes slice
    used = set(copies) | {x["task"] for x in held["noncopy"]} | set(held.get("scanned", []))
    used |= {r["task"] for r in pickle.load(open(DATA / "rows.pkl", "rb"))}
    nat = native.Native()
    mistakes = []
    for p in native.tasks("train")[:SCAN]:
        if p.stem in used or len(mistakes) >= MISTAKES:
            continue
        ids, _ = native.text(p)
        cand = []
        for t in range(30, len(ids) - 1):
            occ = occurrences(ids, t)
            if occ and occ[-1] + 1 < t and ids[occ[-1] + 1] == ids[t + 1]:
                cand.append(t)
        if not cand:
            continue
        lp = nat.reference([ids], cand)[0]
        wrong = [t for t, row in zip(cand, lp) if int(row.argmax()) != ids[t + 1]]
        if wrong:
            t = wrong[int(rng_of("mistake", p.stem).integers(len(wrong)))]
            mistakes.append(roles_entry("mistake", p.stem, ids[:t + 1], t, D))
    out = {"D": D, "texts": texts, "mistakes": mistakes,
           "values": {"discovery": sorted(disc_tokens), "heldout": held_tokens,
                      "overlap": sorted(t for t, v in held_tokens.items() if v in disc_tokens)}}
    (DATA / "falsify_texts.json").write_text(json.dumps(out, indent=1))
    print(f"D {D}; {len(texts)} copy texts, {len(mistakes)} mistakes; held-out copied tokens seen in discovery: "
          f"{len(out['values']['overlap'])}/{len(held_tokens)}")


def roles_entry(kind, task, ids, t, D):
    occ = occurrences(ids, t)
    o, a = occ[-1], occ[-1] + 1
    taken = set(occ) | {j + 1 for j in occ} | {t}
    r = rng_of("moved", task, t)
    pool = [q for q in range(2, t) if q not in taken and q - 1 not in taken]
    moved = int(pool[int(r.integers(len(pool)))]) if pool else None
    fixed = max(2, t - D)  # a text shorter than D: the nearest position with an earlier one (its o column)
    if fixed == a:
        fixed = fixed - 1 if fixed - 1 >= 2 else fixed + 1
    far = [q for q in range(2, t) if min(abs(q - a), abs(q - o), abs(q - t)) >= 10 and q not in taken]
    far = sorted(int(x) for x in r.choice(far, size=min(2, len(far)), replace=False)) if far else []
    e = {"kind": kind, "task": task, "t": t, "a": a, "o": o, "occurrences": occ, "moved": moved, "fixed": fixed, "far": far,
         "taken": sorted(taken)}
    if len(occ) >= 2:
        e["backup"] = {"a": occ[-2] + 1, "o": occ[-2]}
    return e


def column_nodes(a, t, with_t=True):
    import goal_eval

    o = a - 1
    nodes = [(n, o, goal_eval.BLOCK) for n in O_SITES] + [(n, o, c) for n, c in O_NAMED]
    nodes += [(n, a, goal_eval.BLOCK) for n in A_SITES] + [(n, a, c) for n, c in A_NAMED]
    if with_t:
        nodes += [(n, t, goal_eval.BLOCK) for n in T_SITES]
    return nodes


def text_of(ge, e):
    import native

    ids, _ = native.text(TEXTS / f"{e['task']}.json")
    ids = ids[:e["t"] + 1]
    return ge.text(ids, [e["t"]], native.task_seed(f"{e['task']}:{e['t']}"))


def equal_bits(ge, tx, bits: float, kind: str):
    """VPD's top k ("vpd") or recency ("recency": VPD's actives at t and the m positions before it) whose bits come
    nearest to bits (the largest k or m not above it)."""
    import goal_eval
    import native

    nat = ge.nat
    P, S = nat.positions(tx.targets), sum(nat.C.values())
    t = tx.targets[-1]
    if kind == "vpd":
        order = nat._vpd_order(tx.ids, tx.targets)
        cost = lambda k: goal_eval.size_bits(native.Graph(order[:k], complete=True), P, S, tx.targets)  # noqa: E731
        lo, hi = 1, len(order)
        if cost(hi) <= bits:
            return native.Graph(order, complete=True), len(order)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if cost(mid) <= bits else (lo, mid - 1)
        return native.Graph(order[:lo], complete=True), lo
    per = [[(n, p, int(c)) for n in nat.names for c in tx.vpd[n][p].nonzero().flatten().tolist()] for p in range(t + 1)]
    acts = lambda m: [nd for p in range(max(0, t - m), t + 1) for nd in per[p]]  # noqa: E731
    cost = lambda m: goal_eval.size_bits(native.Graph(acts(m), complete=True), P, S, tx.targets)  # noqa: E731
    lo, hi = 0, t  # the largest m whose bits are not above bits (bits grow with m)
    if cost(hi) <= bits:
        return native.Graph(acts(hi), complete=True), hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if cost(mid) <= bits else (lo, mid - 1)
    return native.Graph(acts(lo), complete=True), lo


def patch_ge(ge):
    """The scorecard as falsify runs it on block circuits: the connection-cut family left out (every connection among
    tens of thousands of block subcomponents is beyond memory) for every circuit alike, and the reward mode's node
    reads off (a diagnostic outside phi: tens of thousands of nodes per draw)."""
    ge._edge_family = lambda tx, graphs, draws, reads=False: [{"kl_edges_c": None, "kl_edges_vpd": None, "reads_edges": None}
                                                             for _ in graphs]
    sides0 = ge._sides

    def sides(tx_, graphs_):
        f = sides0(tx_, graphs_)

        def run(g, cuts=None, patch=None, reads=False):
            got = f(g, cuts=cuts, patch=patch, reads=False)
            return (*got, {}, {}) if reads else got
        return run
    ge._sides = sides
    for fam in ("_node_family", "_interchange_family"):  # (tx, graphs, draws[, reads]) and keywords
        f0 = getattr(ge, fam)

        def wrap(*a, f0=f0, **k):
            return f0(*a[:3], **{**k, "reads": False})
        setattr(ge, fam, wrap)


def score(e: dict, only: list | None = None):
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1)  # kill -USR1 PID prints every thread's stack
    import goal_eval
    import native
    import torch

    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    goal_eval.FLIP_ROWS = int(os.environ.get("FLIP_ROWS", 16))  # the flips' rows per run: [rows, T, a block circuit's columns]
    patch_ge(ge)
    t0 = time.perf_counter()
    tx = text_of(ge, e)
    t, a = e["t"], e["a"]
    tact = [(n, t, int(c)) for n in nat.names for c in tx.vpd[n][t].nonzero().flatten().tolist()]
    variants = {"correct": column_nodes(a, t) + tact, "template": column_nodes(a, t)}
    if e.get("moved"):
        variants["a_source_moved"] = column_nodes(e["moved"], t) + tact
    variants["b_fixed_offset"] = column_nodes(e["fixed"], t) + tact
    if e.get("backup"):
        variants["c_with_backup"] = column_nodes(a, t) + column_nodes(e["backup"]["a"], t, with_t=False) + tact
    if e.get("far"):
        variants["d_irrelevant_parts"] = column_nodes(a, t) + tact + [(n, q, goal_eval.BLOCK) for q in e["far"] for n in A_SITES]
    if e.get("mismatch"):
        variants["g_mismatched"] = column_nodes(e["mismatch"]["a"], t) + tact
    graphs = {k: native.Graph(sorted(set(v)), complete=True) for k, v in variants.items()}
    P, S = nat.positions(tx.targets), sum(nat.C.values())
    bits = {k: goal_eval.size_bits(g, P, S, tx.targets, active=tx.vpd) for k, g in graphs.items()}  # the reward's bits
    stated = json.loads((Path(os.environ.get("MECH", Path.home() / "mpd-data/graph_oracle/mechanisms")) / "copy_columns.json").read_text())
    import templates

    sg, _ = templates.graph(stated, tx.ids, tx.targets, nat.C)
    sbits = goal_eval.size_bits(goal_eval.strand(sg, tx.targets, nat.C), nat.positions(tx.targets), sum(nat.C.values()),
                                tx.targets, active=tx.vpd)
    for target_bits, lab in ((bits["correct"], "correct"), (sbits, "stated")):
        for kind in ("vpd", "recency"):
            g, k = equal_bits(ge, tx, target_bits, kind)
            graphs[f"ref_{kind}_{lab}_bits"] = g
            bits[f"ref_{kind}_{lab}_bits"] = goal_eval.size_bits(g, P, S, tx.targets, active=tx.vpd)
    names = list(graphs)
    print(e["task"], "circuits built", round(time.perf_counter() - t0), "s", flush=True)
    pool = torch.load(POOL, weights_only=False) if POOL.exists() else None
    tau = ge.tau(tx, pool, "eval")  # the text's bar, families and start bits, as reward v3 reads them
    print(e["task"], "tau", round(time.perf_counter() - t0), "s", flush=True)
    # (h) mixture: the single-mechanism claim (correct) against it plus the influential start's other positions (every
    # position the start keeps besides a, o and t, VPD's actives there), and whether the model mixes on this text: the
    # match broken (fresh tokens at o), how often its likeliest token stays what it was
    st = ge.start_set(tx, "eval")["graph"]
    extra = [nd for nd in st.node_set() if nd[1] not in (a, a - 1, t)]
    if extra:
        variants["h_mixture"] = variants["correct"] + extra
        graphs["h_mixture"] = native.Graph(sorted(set(variants["h_mixture"])), complete=True)
        bits["h_mixture"] = goal_eval.size_bits(graphs["h_mixture"], P, S, tx.targets, active=tx.vpd)
        names.append("h_mixture")
    base = nat.reference([tx.ids], [t])[0, 0]
    zs = ge._fresh_sources(tx, a - 1, int(tx.seed) + 31)[:8]
    lp = nat.reference([list(tx.ids[:a - 1]) + [z] + list(tx.ids[a:]) for z in zs], [t])[:, 0]
    mix = {"survive": float((lp.argmax(-1) == base.argmax()).float().mean()),
           "drop_copied": float(base[tx.ids[a]] - lp[:, tx.ids[a]].mean()), "copied_is_top": int(base.argmax()) == tx.ids[a],
           "start_positions": sorted({nd[1] for nd in st.node_set()}), "extra_positions": sorted({nd[1] for nd in extra})}
    names = [k for k in names if only is None or k in only]  # --only: some variants (added later) on a text scored before
    rows = []
    for k in names:  # one at a time: the engine spans its call's columns
        rows += ge.fidelity(tx, [graphs[k]], pool=pool, mode="reward", split="eval", draws=16, interchange=8, changed=8,
                            near_donors=8, far_donors=4)
        native.free(nat.dev)
        print(e["task"], k, "scored", round(time.perf_counter() - t0), "s", flush=True)
    import components

    pt, fams = tau["phi_tau"], tau["families"]
    with out_path("falsify").open("a") as f:
        for k, r in zip(names, rows):
            ph = goal_eval.phi(r, families=fams)[0]  # over the text's families (the start's), an unscored one at 0
            phc = goal_eval.cap_phi(ph, pt)
            g = None if phc is None or pt is None else 1.0 / (1.0 + 2.0 ** (-components.GATE_SLOPE * (phc - pt)))
            comp = goal_eval.compact(bits[k], tau["bits_start"])
            eps = goal_eval._eps(r)
            f.write(json.dumps({"kind": e["kind"], "task": e["task"], "t": t, "variant": k, "phi": ph, "phi_capped": phc, "phi_tau": pt,
                                "phi_start": tau["phi_start"], "phi_empty": tau["phi_empty"], "families": fams, "g": g,
                                # reward v3's circuit alone, less D (no teacher here): phi - phi_tau + g x 0.5 compact
                                "reward": None if None in (phc, pt, g, comp) else phc - pt + g * 0.5 * comp,
                                "bits": bits[k], "bits_start": tau["bits_start"], "compact": comp,
                                "compact_vs_correct": goal_eval.compact(bits[k], bits["correct"]), "nodes": graphs[k].count(),
                                **{f"s_{fm}": goal_eval.family_score(r, fm) for fm in goal_eval.PHI_FAMILIES},
                                **{f"eps_{fm}": eps.get(fm) for fm in goal_eval.PHI_FAMILIES},
                                "c": r.get("c"), "c_rl": r.get("c_rl"), "mix": mix}) + "\n")
    print(e["task"], {k: round(goal_eval.phi(r, families=fams)[0] or 0, 3) for k, r in zip(names, rows)}, flush=True)


def variables_test(e: dict):
    import goal_eval
    import native
    import variables

    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    tx = text_of(ge, e)
    t, a = e["t"], e["a"]
    sc = variables.Scorer(ge)
    row = {"kind": e["kind"], "task": e["task"], "t": t}
    for lab, pos in (("correct", a), ("source_moved", e.get("moved"))):
        if pos is None:
            continue
        v = {"name": "copied token", "meaning": "the identity of the token after the earlier occurrence of the current token",
             "inputs": [pos], "group": [["block", n, pos] for n in A_SITES], "feeds": ["prediction"]}
        res = sc.score(tx, variables.parse([v], nat.C), "eval", 8)
        row[f"e4_{lab}"] = res.get("e4")
        row[f"f_on_{lab}"], row[f"s_off_{lab}"] = res.get("f_on"), res.get("s_off")
    # E4 v2 with a stated value map on the same a-column variable: the identity map (every token its own class) against
    # algorithms the model does not run, a random balanced partition of the vocabulary and token-id parity (same-class
    # tokens must leave the prediction where it is, other-class tokens move it as the model's does); classes of fresh
    # tokens at a (goal_eval's sampled sources there), the text's own token in the first
    V = nat.target.wte.shape[0]
    part = rng_of("partition", e["task"]).integers(2, size=V)
    za, own = ge._fresh_sources(tx, a, int(tx.seed) + 41), tx.ids[a]
    maps = {"identity": [[own]] + [[z] for z in za],
            "random_partition": [[own] + [z for z in za if part[z] == part[own]], [z for z in za if part[z] != part[own]]],
            "parity": [[own] + [z for z in za if z % 2 == own % 2], [z for z in za if z % 2 != own % 2]]}
    for name, classes in maps.items():
        v = {"name": "copied token", "meaning": f"the copied token's {name.replace('_', ' ')} class", "inputs": [a],
             "group": [["block", n, a] for n in A_SITES], "feeds": ["prediction"],
             "values": [{"position": a, "classes": [{"token_ids": [int(z) for z in c]} for c in classes]}]}
        res = sc.score(tx, variables.parse([v], nat.C), "eval", 8)
        for m in ("e4", "f_on", "f_off", "f_same"):
            row[f"{m}_map_{name}"] = res.get(m)
    # the value map's predictions on fresh tokens at a: identity map against a random balanced partition
    base = nat.reference([tx.ids], [t])[0, 0]
    y = int(base.argmax())
    zs = ge._fresh_sources(tx, a, int(tx.seed) + 21)[:16]
    xs = []
    for z in zs:
        x2 = list(tx.ids)
        x2[a] = z
        xs.append(x2)
    lp = nat.reference(xs, [t])[:, 0]
    part = rng_of("partition", e["task"]).integers(2, size=lp.shape[-1])  # a random balanced partition of the vocabulary
    ident, rand = [], []
    for z, l in zip(zs, lp):
        new = int(l.argmax())
        ident.append(new == z)
        same = part[z] == part[tx.ids[a]]
        rand.append((new == y) if same else (new != y))
    row["value_identity"] = float(np.mean(ident))
    row["value_random_partition"] = float(np.mean(rand))
    with out_path("falsify_vars").open("a") as f:
        f.write(json.dumps({k: (float(v) if isinstance(v, (int, float)) and v is not None and not isinstance(v, bool) else v)
                            for k, v in row.items()}) + "\n")
    print(e["task"], row, flush=True)


def values_test(kind: str = "copy"):
    """Value maps the model does not run, on the copy texts (model runs alone): for fresh tokens z at a (16 per text,
    never the model's three likeliest there), the identity map (the prediction becomes z), a random balanced
    partition of the vocabulary and token-id parity (same-class z leaves the prediction, other-class z moves it).
    -> DATA/falsify_values.jsonl, one line per text."""
    import goal_eval
    import native

    nat = native.Native()
    ge = goal_eval.GoalEval(nat)
    d = json.loads((DATA / "falsify_texts.json").read_text())
    out = out_path("falsify_values")
    done = {json.loads(line)["task"] for line in out.read_text().splitlines()} if out.exists() else set()
    for e in d["texts"] if kind == "copy" else d["mistakes"]:
        if e["task"] in done:
            continue
        ids, _ = native.text(TEXTS / f"{e['task']}.json")
        ids = ids[:e["t"] + 1]
        from types import SimpleNamespace

        tx = SimpleNamespace(ids=ids, seed=native.task_seed(f"{e['task']}:{e['t']}"))  # _fresh_sources reads only these
        t, a = e["t"], e["a"]
        base = nat.reference([ids], [t])[0, 0]
        y = int(base.argmax())
        zs = ge._fresh_sources(tx, a, int(tx.seed) + 21)[:16]
        xs = []
        for z in zs:
            x2 = list(ids)
            x2[a] = z
            xs.append(x2)
        lp = nat.reference(xs, [t])[:, 0]
        part = rng_of("partition", e["task"]).integers(2, size=lp.shape[-1])
        rec = {"identity": [], "random_partition": [], "parity": []}
        for z, l in zip(zs, lp):
            new = int(l.argmax())
            rec["identity"].append(new == z)
            for name, cls in (("random_partition", lambda v: int(part[v])), ("parity", lambda v: v % 2)):
                same = cls(z) == cls(ids[a])
                rec[name].append((new == y) if same else (new != y))
        row = {"kind": e["kind"], "task": e["task"], "t": t, **{k: float(np.mean(v)) for k, v in rec.items()}, "n": len(zs)}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(e["task"], row, flush=True)
        native.free(nat.dev)


def entries():
    d = json.loads((DATA / "falsify_texts.json").read_text())
    return {(e["kind"], e["task"], e["t"]): e for e in d["texts"] + d["mistakes"]}


def find(kind, task, t=None):
    for (k, ta, tt), e in entries().items():
        if k == kind and ta == task and (t is None or int(t) == tt):
            return e
    raise KeyError((kind, task, t))


def report():
    import templates

    rows = []
    for p in [DATA / "falsify.jsonl"] + templates.run_outputs("falsify_pod*.jsonl") + templates.run_outputs("falsify.jsonl"):
        if p.exists() and p.name.startswith("falsify") and "vars" not in p.name:
            rows += [json.loads(line) for line in p.read_text().splitlines()]
    rows = [r for r in rows if "phi_tau" in r]  # rows scored before reward v3's terms were kept are rescored
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["kind"], r["task"], r["t"])].setdefault(r["variant"], r)
    terms = ["reward", "phi", "s_point", "s_nodes", "s_flips", "s_track_pp", "s_interchange_pp", "s_patching_near",
             "s_patching_far", "s_pattern", "s_N", "compact"]
    pairs = [("correct", "a_source_moved"), ("correct", "b_fixed_offset"), ("c_with_backup", "correct"),
             ("correct", "d_irrelevant_parts"), ("correct", "g_mismatched"), ("correct", "template"),
             ("correct", "ref_vpd_correct_bits"), ("correct", "ref_recency_correct_bits"), ("h_mixture", "correct")]
    out = {}
    overlap = set(json.loads((DATA / "falsify_texts.json").read_text())["values"]["overlap"])
    groups = [("copy", lambda k: k[0] == "copy"), ("mistake", lambda k: k[0] == "mistake"),
              ("copy, copied token never in discovery", lambda k: k[0] == "copy" and k[1] not in overlap),
              ("copy, the model mixes (its prediction survives the broken match on half the draws or more)",
               lambda k: k[0] == "copy" and mixes.get(k) is True),
              ("copy, the model does not mix", lambda k: k[0] == "copy" and mixes.get(k) is False)]
    mixes = {k: next(r["mix"]["survive"] >= 0.5 for r in d.values() if r.get("mix")) for k, d in by.items()
             if any(r.get("mix") for r in d.values())}  # a text's label from any of its rows (an --only run holds it too)
    for kind, keep in groups:
        texts = [d for k, d in by.items() if keep(k)]
        if not texts:
            continue
        print(f"== {kind} texts: {len(texts)}; fraction of texts where the first beats the second (ties count half)")
        print("   " + " ".join(f"{m[:12]:>12s}" for m in terms))
        for a_, b_ in pairs:
            line = []
            for m in terms:
                vals = []
                for d in texts:
                    if a_ in d and b_ in d:
                        x = d[a_].get(m) if m != "compact" else d[a_].get("compact")
                        y = d[b_].get(m) if m != "compact" else d[b_].get("compact")
                        if m == "compact":  # compactness against the correct circuit: higher is smaller
                            x, y = -(d[a_]["bits"] or 0), -(d[b_]["bits"] or 0)
                        if x is not None and y is not None:
                            vals.append(1.0 if x > y + 1e-9 else 0.5 if abs(x - y) <= 1e-9 else 0.0)
                out[(kind, a_, b_, m)] = (float(np.mean(vals)) if vals else None, len(vals))
                line.append(f"{np.mean(vals):12.2f}" if vals else f"{'-':>12s}")
            print(f"   {' '.join(line)}   {a_} > {b_} (n={max(out[(kind, a_, b_, m)][1] for m in terms)})")
        mean = lambda v, m: np.mean([d[v][m] for d in texts if v in d and d[v][m] is not None])  # noqa: E731
        print(f"   bar phi_tau {np.mean([next(iter(d.values()))['phi_tau'] for d in texts]):.2f}, start phi "
              f"{np.mean([next(iter(d.values()))['phi_start'] for d in texts]):.2f}, empty phi {np.mean([next(iter(d.values()))['phi_empty'] for d in texts]):.2f}")
        for v in sorted({v for d in texts for v in d}):
            print(f"   {v:26s} reward {mean(v, 'reward'):7.2f}  phi {mean(v, 'phi'):6.2f}  bits {mean(v, 'bits'):9.0f}  "
                  f"compact {mean(v, 'compact'):6.2f}  faithful {np.mean([d[v]['phi_capped'] >= d[v]['phi_tau'] for d in texts if v in d]):.2f}")
    vr = {}  # one row per text: one from the current E4 (off-target and same-class on f's scale) over an older one
    for p in [DATA / "falsify_vars.jsonl"] + templates.run_outputs("falsify_vars*.jsonl"):
        if p.exists():
            for r in map(json.loads, p.read_text().splitlines()):
                k = (r["kind"], r["task"], int(r["t"]))
                if k not in vr or (r.get("f_same_map_random_partition") is not None and vr[k].get("f_same_map_random_partition") is None):
                    vr[k] = r
    vr = list(vr.values())
    if vr:
        for kind in ("copy", "mistake"):
            rs = [r for r in vr if r["kind"] == kind]
            if not rs:
                continue
            both = [r for r in rs if r.get("e4_correct") is not None and r.get("e4_source_moved") is not None]
            win = np.mean([1.0 if r["e4_correct"] > r["e4_source_moved"] else 0.5 if r["e4_correct"] == r["e4_source_moved"] else 0.0
                           for r in both]) if both else float("nan")
            print(f"== variables ({kind}, {len(rs)} texts): E4 correct {np.mean([r['e4_correct'] for r in both]):.2f} vs source moved "
                  f"{np.mean([r['e4_source_moved'] for r in both]):.2f}, correct higher on {win:.2f}; value map identity "
                  f"{np.mean([r['value_identity'] for r in rs]):.2f} vs random partition {np.mean([r['value_random_partition'] for r in rs]):.2f}")
            mp = [r for r in rs if r.get("e4_map_identity") is not None]
            if mp:
                mean = lambda k: np.mean([r[k] for r in mp if r.get(k) is not None])  # noqa: E731
                print("   E4 with a stated value map: " + "; ".join(
                    f"{nm} e4 {mean(f'e4_map_{nm}'):.2f} (on {mean(f'f_on_map_{nm}'):.2f}, off {mean(f'f_off_map_{nm}'):.2f}"
                    + (f", same class {mean(f'f_same_map_{nm}'):.2f}" if nm != "identity" else "") + ")"
                    for nm in ("identity", "random_partition", "parity"))
                    + f"; identity above partition on {np.mean([r['e4_map_identity'] > r['e4_map_random_partition'] for r in mp]):.2f}, "
                    f"above parity on {np.mean([r['e4_map_identity'] > r['e4_map_parity'] for r in mp]):.2f} (n={len(mp)})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "score", "vars", "values", "report"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--only", default=None, help="score: these variants only (comma-separated)")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "score":
        score(find(*a.args), a.only.split(",") if a.only else None)
    elif a.cmd == "vars":
        variables_test(find(*a.args))
    elif a.cmd == "values":
        values_test(*a.args)
    else:
        report()
