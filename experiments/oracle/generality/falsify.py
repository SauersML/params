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
  falsify.py background KIND TASK [T]  -> falsify_bg.jsonl: the copy core with the generic target parts as background
  falsify.py background_report
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


GENERIC_T = [("h.1.mlp.c_fc", 2103), ("h.3.mlp.down_proj", 3532), ("h.3.attn.o_proj", 806), ("h.1.attn.q_proj", 316),
             ("h.0.mlp.c_fc", 53)]  # inframove's always-on parts at the target (94% of the influential sets' control damage)
CORE_A = [("h.1.attn.q_proj", 316), ("h.0.mlp.c_fc", 53), ("h.1.mlp.c_fc", 2103)]  # the copy core at a
CORE_O = [("h.1.attn.k_proj", 329), ("h.1.attn.k_proj", 119)]  # and at o
T_READS = ["h.2.attn.q_proj", "h.2.attn.o_proj"]  # the target's reads of a (layer 2's query and output there)


def background(e: dict):
    """The copy core's transfer with the generic target parts held as background (the lead's 10-13): the five
    always-on parts at the target (GENERIC_T) are in every circuit and credited to no mechanism; copying is credited
    with the core at a and o (CORE_A, CORE_O) and the target's reads (VPD's actives at t in T_READS). Circuits
    (complete): background; background + core; background + the core at the source-moved position (falsify's (a));
    background + every active at t (the target's own computation). Each scored alone by fidelity in reward mode on
    the eval split and goal_eval.circuit_term -> falsify_bg.jsonl."""
    import goal_eval
    import native
    import torch

    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    tx = text_of(ge, e)
    t, a = e["t"], e["a"]
    reads = [(n, t, int(c)) for n in T_READS for c in tx.vpd[n][t].nonzero().flatten().tolist()]
    bg = [(n, t, c) for n, c in GENERIC_T]
    core = lambda q: [(n, q, c) for n, c in CORE_A] + [(n, q - 1, c) for n, c in CORE_O] + reads  # noqa: E731
    acts = [(n, t, int(c)) for n in nat.names for c in tx.vpd[n][t].nonzero().flatten().tolist()]
    variants = {"background": bg, "background_core": bg + core(a), "background_target_actives": bg + acts}
    if e.get("moved"):
        variants["background_core_moved"] = bg + core(e["moved"])
    graphs = {k: native.Graph(sorted(set(v)), complete=True) for k, v in variants.items()}
    P, S = nat.positions(tx.targets), sum(nat.C.values())
    pool = torch.load(POOL, weights_only=False) if POOL.exists() else None
    tau = ge.tau(tx, pool, "eval")
    with out_path("falsify_bg").open("a") as f:
        for k, g in graphs.items():
            r = ge.fidelity(tx, [g], pool=pool, mode="reward", split="eval")[0]
            native.free(nat.dev)
            b = goal_eval.size_bits(g, P, S, tx.targets)
            ct = goal_eval.circuit_term(r, tau, b)
            f.write(json.dumps({"kind": e["kind"], "task": e["task"], "t": t, "variant": k, "R": ct["term"], "phi": ct["phi_full"],
                                "phi_tau": tau["phi_tau"], "bits": b, "nodes": g.count(),
                                **{f"s_{fm}": goal_eval.family_score(r, fm) for fm in goal_eval.PHI_FAMILIES}}) + "\n")
            print(e["task"], k, round(ct["term"], 3), round(ct["phi_full"] or 0, 3), flush=True)


def background_report():
    import templates

    rows = []
    for p in [DATA / "falsify_bg.jsonl"] + templates.run_outputs("falsify_bg*.jsonl"):
        if p.exists():
            rows += [json.loads(x) for x in p.read_text().splitlines()]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["kind"], r["task"])].setdefault(r["variant"], r)
    for kind in ("copy", "mistake"):
        ds = [d for k, d in by.items() if k[0] == kind]
        if not ds:
            continue
        print(f"== {kind} texts: {len(ds)}")
        for v in ("background", "background_core", "background_core_moved", "background_target_actives"):
            xs = [d[v] for d in ds if v in d]
            if xs:
                print(f"   {v:28s} R {np.mean([x['R'] for x in xs]):6.2f}  phi {np.mean([x['phi'] for x in xs]):5.2f}  point "
                      f"{np.mean([x['s_point'] or 0 for x in xs]):.2f}  track_pp {np.mean([x['s_track_pp'] or 0 for x in xs]):.2f}  "
                      f"bits {np.mean([x['bits'] for x in xs]):8.0f}")
        for a_, b_ in (("background_core", "background"), ("background_core", "background_core_moved")):
            for m in ("R", "phi", "s_point", "s_track_pp"):
                w = [1.0 if d[a_][m] > d[b_][m] + 1e-9 else 0.5 if abs(d[a_][m] - d[b_][m]) <= 1e-9 else 0.0
                     for d in ds if a_ in d and b_ in d and d[a_].get(m) is not None and d[b_].get(m) is not None]
                gain = [d[a_][m] - d[b_][m] for d in ds if a_ in d and b_ in d and d[a_].get(m) is not None and d[b_].get(m) is not None]
                print(f"   {a_} > {b_} on {m}: {np.mean(w):.2f} (mean gain {np.mean(gain):+.3f}, n={len(w)})")


CIRCUIT_EPS = 0.1  # bits of phi an oracle circuit must beat every variant by (the final evaluation's margin)


def remap(answer: dict, where: dict) -> dict:
    """An answer (goal_eval.circuit's JSON) with its nodes' positions and its pair entries' read positions moved by
    where ({old position: new}; others kept): edges are node places, so the structure moves with the nodes."""
    out = dict(answer)
    out["nodes"] = [[m, c, where.get(int(p), int(p))] for m, c, p in answer.get("nodes", [])]
    out["edges"] = [["pair", e[1], e[2], e[3], where.get(int(e[4]), int(e[4]))] if isinstance(e, (list, tuple)) and len(e) == 5
                    and e[0] == "pair" else e for e in answer.get("edges", [])]
    return out


def duplicated(answer: dict, where: dict) -> dict:
    """The answer plus a copy of its nodes at the positions where maps (each copied node at its new position) and of
    every edge touching a copied node (a copy's edges to an uncopied node go to that same node)."""
    nodes = [list(n) for n in answer.get("nodes", [])]
    copy = {}
    for i, (m, c, p) in enumerate(answer.get("nodes", [])):
        if int(p) in where:
            copy[i] = len(nodes)
            nodes.append([m, c, where[int(p)]])
    edges = list(answer.get("edges", []))
    for e in answer.get("edges", []):
        if isinstance(e, (list, tuple)) and len(e) == 2 and (e[0] in copy or e[1] in copy):
            edges.append([copy.get(e[0], e[0]), copy.get(e[1], e[1])])
    return {**answer, "nodes": nodes, "edges": edges}


def circuit_variants(answer: dict, t: int, T: int, key: tuple, other_offsets: list, D: int) -> dict:
    """The general wrong variants of an oracle circuit on a text with target t: (a) source moved: every context position
    (a node position other than t) to a distinct random earlier position the circuit does not use; (b) fixed offset:
    the context positions, nearest first, to t - D, t - D - 1, ... (D the answer file's median context distance); (d)
    irrelevant parts: the circuit plus a copy of its context nodes at positions at least 10 from every circuit position,
    with their edges; (g) mismatched: the context positions, nearest first, at another answer's distances from its
    target. A circuit without context positions has (d) alone (a copy of its target nodes 10 or more positions back)."""
    pos = sorted({int(p) for _, _, p in answer.get("nodes", [])})
    ctx = sorted((p for p in pos if p != t), key=lambda p: t - p)  # nearest first
    free = [q for q in range(t) if q not in pos]
    r = rng_of("circuit-variants", *key)
    out = {}
    if ctx and len(free) >= len(ctx):
        out["a_source_moved"] = remap(answer, dict(zip(ctx, (int(q) for q in r.choice(free, size=len(ctx), replace=False)))))
        fixed, q = {}, t - D
        for p in ctx:
            while q in pos or q in fixed.values():
                q -= 1
            if q < 0:
                break
            fixed[p] = q
            q -= 1
        if len(fixed) == len(ctx) and fixed != {p: p for p in ctx}:
            out["b_fixed_offset"] = remap(answer, fixed)
        if other_offsets:
            offs = other_offsets[int(r.integers(len(other_offsets)))]
            mis = {}
            for p, d in zip(ctx, offs + [offs[-1] + k + 1 for k in range(len(ctx))]):
                q = t - d
                while q >= 0 and (q in pos or q in mis.values()):
                    q -= 1
                if q < 0:
                    break
                mis[p] = q
            if len(mis) == len(ctx) and mis != {p: p for p in ctx}:
                out["g_mismatched"] = remap(answer, mis)
    src = ctx or [t]
    far = [q for q in range(t) if min(abs(q - p) for p in pos) >= 10]
    if len(far) >= len(src):
        out["d_irrelevant_parts"] = duplicated(answer, dict(zip(src, (int(q) for q in r.choice(far, size=len(src), replace=False)))))
    return out


def circuits(answers_path: Path, out_path: Path, eps: float = CIRCUIT_EPS):
    """The final evaluation's falsification column (primerl's final_eval.sh): per answer line ({"task", "sample",
    "text"}; its circuit components.parse(text)["answer"]) on the task's text, the circuit and its general wrong
    variants (circuit_variants) scored by phi on the eval split (reward mode, the text's families from ge.tau, as
    components.reward reads them; the CI-pattern pool POOL), and reward v3's circuit term R (goal_eval.circuit_term)
    beside it. -> out_path (JSON): per answer the circuit's phi and R, each variant's, and beats_all: phi
    above every variant's by more than eps; with a summary (the share of answers that beat all, per variant the share
    beaten). Resumable: answers already in out_path are kept."""
    import components
    import goal_eval
    import native
    import torch

    lines = [json.loads(x) for x in answers_path.read_text().splitlines() if x.strip()]
    got = json.loads(out_path.read_text()) if out_path.exists() else {"answers": []}
    done = {(a["task"], a["sample"]) for a in got["answers"]}
    ge = goal_eval.GoalEval(native.Native())
    nat = ge.nat
    pool = torch.load(POOL, weights_only=False) if POOL.exists() else None
    parsed = []
    for ln in lines:
        task = json.loads((TEXTS / f"{ln['task']}.json").read_text())
        t = task["prompts"][0]["target_positions"][-1]
        parsed.append((ln, task, t, components.parse(ln["text"])["answer"]))
    offsets = [sorted(t - int(p) for _, _, p in a.get("nodes", []) if int(p) != t) for _, _, t, a in parsed]
    offsets = [o for o in offsets if o]
    D = int(np.median([d for o in offsets for d in o])) if offsets else 8
    for k, (ln, task, t, answer) in enumerate(parsed):
        if (ln["task"], ln["sample"]) in done:
            continue
        pr = task["prompts"][0]
        tx = ge.text(pr["token_ids"], pr["target_positions"], task=task)
        others = [o for j, o in enumerate(offsets) if parsed[j][0]["task"] != ln["task"]]
        answers = {"circuit": answer, **circuit_variants(answer, t, tx.T, (ln["task"], ln["sample"]), others, D)}
        graphs = {v: goal_eval.circuit(a, tx.targets, nat.C, tx.T)[0] for v, a in answers.items()}
        tau = ge.tau(tx, pool, "eval")
        rows = []
        for g in graphs.values():  # one at a time: the engine spans its call's columns
            rows += ge.fidelity(tx, [g], pool=pool, mode="reward", split="eval")
            native.free(nat.dev)
        pt, fams = tau["phi_tau"], tau["families"]
        P, S = nat.positions(tx.targets), sum(nat.C.values())
        res = {}
        for (v, g), r in zip(graphs.items(), rows):
            b = goal_eval.size_bits(g, P, S, tx.targets, active=tx.vpd)
            ct = goal_eval.circuit_term(r, tau, b)  # reward v3's circuit term, as components.reward and the teacher read it
            res[v] = {"phi": ct["phi_full"], "R": ct["term"], "bits": b}
        ph0 = res["circuit"]["phi"]
        var = {v: x for v, x in res.items() if v != "circuit"}
        beats = {v: None if ph0 is None or x["phi"] is None else ph0 > x["phi"] + eps for v, x in var.items()}
        got["answers"].append({"task": ln["task"], "sample": ln["sample"], "phi": ph0, "R": res["circuit"]["R"],
                               "bits": res["circuit"]["bits"], "phi_tau": pt, "variants": var, "beats": beats,
                               "beats_all": bool(beats) and all(b is True for b in beats.values())})
        A = got["answers"]
        names = sorted({v for a in A for v in a["beats"]})
        got["summary"] = {"answers": len(A), "eps": eps, "beats_all": float(np.mean([a["beats_all"] for a in A])),
                          "beaten": {v: float(np.mean([a["beats"][v] is True for a in A if v in a["beats"]])) for v in names},
                          "with_variants": sum(bool(a["beats"]) for a in A)}
        out_path.write_text(json.dumps(got, indent=1))
        native.free(nat.dev)
        print(ln["task"], ln["sample"], {v: round(x["phi"] or 0, 3) for v, x in res.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("build", "score", "vars", "values", "report", "background", "background_report"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--only", default=None, help="score: these variants only (comma-separated)")
    ap.add_argument("--circuits", type=Path, help="oracle answers (JSONL: task, sample, text): the general falsification")
    ap.add_argument("--out", type=Path, help="--circuits: the JSON written")
    ap.add_argument("--eps", type=float, default=CIRCUIT_EPS, help="--circuits: the phi margin over every variant")
    a = ap.parse_args()
    if a.circuits:
        circuits(a.circuits, a.out, a.eps)
    elif a.cmd == "build":
        build()
    elif a.cmd == "score":
        score(find(*a.args), a.only.split(",") if a.only else None)
    elif a.cmd == "vars":
        variables_test(find(*a.args))
    elif a.cmd == "background":
        background(find(*a.args))
    elif a.cmd == "background_report":
        background_report()
    elif a.cmd == "values":
        values_test(*a.args)
    else:
        report()
