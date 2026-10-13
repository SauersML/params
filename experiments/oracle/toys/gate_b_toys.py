"""Gate B on the planted toys (toyloop; plan.mmd's GATEB: "faithful beats wrong at equal size; empty, shuffled and
collapsed controls lose"), where the mechanism is known by construction (planted.py). Per toy and text, reward v3's
circuit part (components.reward's circuit_reward = phi + g x 0.5 compact + D, D = 0 without a teacher reference; phi,
its bar phi_tau and gate g beside it) of:
  true        the planted mechanism's active pieces (VPD's exact answer, planted.decompose), less the pieces that cannot
              reach the target (structurally dead: in the last live layer, queries, attention outputs and MLP pieces
              at positions other than the target; nothing is read after them), less the units the toy plants as
              irrelevant (cancel: the two cancelling neurons);
  wrong_eq    at the true circuit's node count: the other mechanism's circuit padded (random actives) or trimmed;
              random actives (3 seeds); the true set with half or all of its nodes swapped to a wrong unit of the same
              layer at the same position (same site's other head or neuron first, else another site of the layer);
  empty;
  shuffled    the true nodes' (site, piece) identities and positions permuted independently;
  collapsed   one node (the true circuit's last-layer node at the target); every VPD active (trivially large);
  teacher     teacher.teach's circuit (the influential start, connections, run_planted's "current" arm);
  ref         run_planted's narrow positional circuits (information only).
Gate B holds on a text when true > every wrong_eq, empty, shuffled and collapsed candidate. Rows to --out JSONL, the
summary (markdown) to --md.

  python3 -B experiments/oracle/toys/gate_b_toys.py --toys ~/mpd-data/toys_planted
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import components  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import planted  # noqa: E402
import run_planted as rp  # noqa: E402

TOYS = "binding_km,binding_fp,route1,route2,additive,interaction,cancel"
RESULTS = HERE / "results"
CHECKS = ("wrong_eq", "empty", "shuffled", "collapsed")
RANDOM_OVERLAP = 0.75
TIE = 1e-9


def layer_kind(n: str) -> tuple[int, str]:
    return native._name_layer_kind(n)


def reach(nodes: list, targets: list[int]) -> list:
    """nodes less those that cannot reach the targets: in the last layer holding any node, the query, attention-output
    and MLP pieces away from the targets (keys and values are read at every position)."""
    live = max((layer_kind(n)[0] for n, _, _ in nodes), default=0)
    return [nd for nd in nodes if not (layer_kind(nd[0])[0] == live and layer_kind(nd[0])[1] in ("q_proj", "o_proj", "c_fc", "down_proj")
                                       and nd[1] not in targets)]


def to_size(nodes: list, n: int, pool: list, rng) -> list:
    """nodes trimmed (random drop) or padded (random pool nodes not in it) to n nodes."""
    nodes = list(dict.fromkeys(nodes))
    if len(nodes) > n:
        keep = sorted(rng.choice(len(nodes), n, replace=False))
        return [nodes[i] for i in keep]
    have = set(nodes)
    extra = [nd for nd in pool if nd not in have]
    if extra and len(nodes) < n:
        add = rng.choice(len(extra), min(n - len(nodes), len(extra)), replace=False)
        nodes += [extra[i] for i in add]
    return nodes


def swapped(nat, tx, true: list, frac: float, rng) -> list:
    """true with frac of its nodes moved to a wrong unit of the same layer at the same position: an active piece of
    another head/neuron at the same site, else any piece of another unit there, else a piece at another site of the
    layer (active first); a node with no such piece is kept."""
    have = set(true)
    out = set(true)
    pick = rng.choice(len(true), int(round(frac * len(true))), replace=False)
    for i in pick:
        n, t, c = true[i]
        L, _ = layer_kind(n)
        u = int(nat.units[n][c])
        same = [(n, t, j) for j in range(nat.C[n]) if int(nat.units[n][j]) not in (u, -1)]
        other = [(m, t, j) for m in nat.names if m != n and layer_kind(m)[0] == L for j in range(nat.C[m]) if int(nat.units[m][j]) != -1]
        for opts in (same, other):
            opts = [o for o in opts if o not in have and o not in out]
            act = [o for o in opts if bool(tx.vpd[o[0]][o[1], o[2]])]
            opts = act or opts
            if opts:
                out.discard((n, t, c))
                out.add(opts[int(rng.integers(len(opts)))])
                break
    return sorted(out)


def shuffled(true: list, rng) -> list:
    ids = [(n, c) for n, _, c in true]
    ts = [t for _, t, _ in true]
    rng.shuffle(ids)
    rng.shuffle(ts)
    return sorted(set((n, t, c) for (n, c), t in zip(ids, ts)))


def core_of(name: str, nat, tx, nodes: list) -> list | None:
    """The planted route's positions within nodes (the core tier), None where the toy has none narrower than its full
    circuit: binding_km run_planted's key-matching positions (the queried pair's key and value, Q and the query); the
    one-live-layer toys (binding_fp, route*, cancel) the target's queries, attention outputs and MLP pieces, every
    key (the softmax's competitors) and the values at the source position alone (binding_fp: the fixed pair's value;
    route*, cancel: the previous token)."""
    T = tx.T
    if name == "binding_km":
        keep = (2 * planted.FIXED + 1, 2 * planted.FIXED + 2, T - 2, T - 1)
        return [nd for nd in nodes if nd[1] in keep]
    if name in ("binding_fp", "route1", "route2", "cancel", "nothing"):
        src = 2 * planted.FIXED + 2 if name == "binding_fp" else T - 2
        k = lambda n: layer_kind(n)[1]  # noqa: E731
        return [nd for nd in nodes if k(nd[0]) == "k_proj" or (k(nd[0]) == "v_proj" and nd[1] == src)
                or (k(nd[0]) not in ("k_proj", "v_proj") and nd[1] in tx.targets)]
    return None


def candidates(name: str, nat, tx, rng) -> dict:
    """(tier, label) -> (category, nodes); tiers "full" (true: every reachable active of the planted units) and "core"
    (true: core_of(full), the planted route's positions), each with its own equal-size wrong circuits and controls.
    A wrong circuit equal to its tier's true set (no room: true is every active) is left out."""
    targets = tx.targets
    allact = sorted(rp.active(nat, tx, lambda n, t, u: True))
    narrow = rp.circuits(name, nat, tx)
    kind = lambda n: layer_kind(n)[1]  # noqa: E731
    mlp = lambda n: kind(n) in ("c_fc", "down_proj")  # noqa: E731
    unit = lambda nd: int(nat.units[nd[0]][nd[2]])  # noqa: E731
    if name == "cancel":
        full = sorted(reach(rp.active(nat, tx, lambda n, t, u: not (mlp(n) and u in (0, 1))), targets))
    else:
        full = sorted(reach(allact, targets))
    out = {}
    tiers = {"full": full}
    core = core_of(name, nat, tx, full)
    if core is not None and set(core) != set(full):
        tiers["core"] = sorted(core)
    for tier, true in tiers.items():
        N = len(true)
        c = {"true": ("true", true)}
        sub = (lambda v: core_of(name, nat, tx, v)) if tier == "core" else (lambda v: v)
        others = {}
        if name.startswith("binding"):
            others["other mechanism (padded/trimmed)"] = narrow["other"]
        elif name == "route2":
            for k in ("head0 alone", "head1 alone"):
                others[f"{k} (padded)"] = sub(reach(narrow[k], targets))
        elif name == "interaction":
            others["attention alone (padded)"] = reach(narrow["attention alone"], targets)
        elif name == "cancel":
            plus = [nd for nd in allact if mlp(nd[0]) and unit(nd) == 0 and nd[1] in targets]
            if plus:   # one true node out per +u piece, the +u neuron (uncancelled) in
                others["true - 1 + (+u neuron)"] = to_size(true, N - len(plus), [], rng) + plus
        for k, v in others.items():
            c[k] = ("wrong_eq", sorted(to_size(v, N, allact, rng)))
        for s in range(3):   # random draws sharing at most RANDOM_OVERLAP of the true set (else not wrong: left out)
            v = to_size([], N, allact, np.random.default_rng(1000 + s))
            if len(set(v) & set(true)) <= RANDOM_OVERLAP * N:
                c[f"random actives #{s}"] = ("wrong_eq", sorted(v))
        for frac in (0.5, 1.0):
            c[f"swap {int(frac * 100)}% to wrong unit"] = ("wrong_eq", swapped(nat, tx, true, frac, rng))
        c["empty"] = ("empty", [])
        c["shuffled"] = ("shuffled", shuffled(true, rng))
        last = max(layer_kind(n)[0] for n, _, _ in true)
        one = [nd for nd in true if nd[1] in targets and layer_kind(nd[0])[0] == last] or true
        c["single node"] = ("collapsed", [one[-1]])
        c["all VPD actives"] = ("collapsed", allact)
        if name == "cancel":
            c["true + both cancelling neurons"] = ("ref", sorted(set(true) | {nd for nd in allact if mlp(nd[0]) and unit(nd) in (0, 1) and nd[1] in targets}))
        for k, (cat, v) in c.items():
            if k != "true" and cat in ("wrong_eq", "collapsed") and set(v) == set(true):
                continue
            out[(tier, k)] = (cat, v)
    for k, v in narrow.items():
        out[("ref", f"run_planted {k} (narrow)")] = ("ref", sorted(v))
    return out


def score(ge, tx, name: str, i: int, texts: list[str], pool: dict) -> list[dict]:
    """components.reward (v3) of answer texts on one toy text, the node cap lifted (node_cap=None: the toys' exact
    pieces put whole-mechanism circuits far above NODE_CAP), on the eval draws with the toy's pattern pool."""
    task = {"id": f"{name}_{i}", "model": "toy", "prompts": [{"token_ids": tx.ids, "target_positions": tx.targets, "model_top": [[["", 0.0]]]}]}
    got = components.reward(ge, task, {"qa": []}, texts, pool=pool, split="eval", cache={task["id"]: tx}, node_cap=None)
    out = []
    for r in got:
        fid = r.get("fidelity") or {}
        fam, eps = {}, {}
        try:
            fam = {f: goal_eval.family_score(fid, f) for f in goal_eval.PHI_FAMILIES}
            eps = goal_eval._eps(fid)
        except Exception:  # noqa: BLE001
            pass
        out.append({**{k: r.get(k) for k in ("circuit_reward", "phi", "phi_tau", "g", "compact", "bits", "bits_start", "valid", "dropped_nodes")},
                    "family": fam, "eps": eps})
    return out


def teacher_text(ge, tx) -> tuple[str | None, dict]:
    import teacher

    t0 = time.perf_counter()
    try:
        circ, row, _ = teacher.teach(ge, tx.ids, tx.targets, 16, f1_start="influential", edges=True, no_strand=True,
                                     max_nodes=None, tokens=None, accept=(), accept_mode="pp")
        return components.render(circ[""]), {"teacher_seconds": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        return None, {"teacher_error": f"{type(e).__name__}: {e}"}


def run_toy(name: str, toys: Path, work: Path, n_texts: int, use_teacher: bool, fh, log=print) -> list[dict]:
    d = torch.load(toys / f"{name}.pt")
    nat = planted.native_of(d["sd"], d["cfg"])
    ge = rp.ToyEval(nat, chunk=16)
    far = [(f"far{i}", x) for i, x in enumerate(next(__import__("vpd_toy").loader(name, 16, 99)).tolist())]
    ge.donor_pools.update(search=far[:8], eval=far[8:])
    pool = goal_eval.build_patterns(rp.toy_texts(name, work, 48, 7), work / f"{name}.exact.pool.pt", nat)
    texts = list(dict.fromkeys(tuple(x) for x in planted.ordinary(name, max(n_texts, 3))))[:n_texts]
    rows = []
    for i, ids in enumerate(texts):
        ids = list(ids)
        t0 = time.perf_counter()
        tx = ge.text(ids, [len(ids) - 1])
        cs = candidates(name, nat, tx, np.random.default_rng(i))
        uniq = {}   # identical node sets scored once
        for k, (_, v) in cs.items():
            uniq.setdefault(tuple(v), components.render(goal_eval.to_json(native.Graph(v, complete=True))))
        answers = {k: uniq[tuple(v)] for k, (_, v) in cs.items()}
        meta = {}
        if use_teacher:
            tt, meta = teacher_text(ge, tx)
            if tt is not None:
                cs[("teacher", "teacher")] = ("teacher", None)
                answers[("teacher", "teacher")] = tt
        texts_u = list(dict.fromkeys(answers.values()))
        got_u = dict(zip(texts_u, score(ge, tx, name, i, texts_u, pool)))
        for (tier, label), (cat, v) in cs.items():
            r = got_u[answers[(tier, label)]]
            n = len(v) if v is not None else len(components.parse(answers[(tier, label)])["answer"].get("nodes") or [])
            rec = {"toy": name, "text": i, "ids": ids, "tier": tier, "candidate": label, "category": cat, "nodes": n, **r,
                   **(meta if label == "teacher" else {})}
            rows.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        log(f"{name} text {i}: {len(cs)} candidates in {time.perf_counter() - t0:.0f}s")
        for r in rows[-len(cs):]:
            log(f"   {r['tier']:<5} {r['candidate']:<38} {r['category']:<9} n={r['nodes']:<5} R={fmt(r['circuit_reward'])} phi={fmt(r['phi'])} "
                f"g={fmt(r['g'])} compact={fmt(r['compact'])}")
    return rows


def fmt(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def verdicts(rows: list[dict]) -> dict:
    """(toy, tier, text) -> {check: (holds, margin = R_true - best R in the category, the best's label)}; "teacher":
    (None, R_true - R_teacher, "teacher")."""
    out = {}
    for toy, tier, i in dict.fromkeys((r["toy"], r["tier"], r["text"]) for r in rows if r["tier"] in ("full", "core")):
        rs = [r for r in rows if (r["toy"], r["tier"], r["text"]) == (toy, tier, i)]
        tr = next(r for r in rs if r["category"] == "true")
        v = {}
        for ch in CHECKS:
            cand = [r for r in rs if r["category"] == ch]
            if cand:
                best = max(cand, key=lambda r: r["circuit_reward"])
                m = tr["circuit_reward"] - best["circuit_reward"]
                v[ch] = (True if m > TIE else "tie" if m > -TIE else False, m, best["candidate"])
        te = [r for r in rows if (r["toy"], r["text"], r["category"]) == (toy, i, "teacher")]
        if te:
            v["teacher"] = (None, tr["circuit_reward"] - te[0]["circuit_reward"], "teacher")
        out[(toy, tier, i)] = v
    return out


def diagnose(tr: dict, w: dict) -> str:
    """Which reward term lets w beat (or tie) tr: the phi difference against the gated compactness difference, and
    the families where w scores higher."""
    dphi = (w["phi"] or 0) - (tr["phi"] or 0)
    q = lambda r: (r["g"] or 0) * 0.5 * (r["compact"] or 0)  # noqa: E731
    dq = q(w) - q(tr)
    fams = sorted(((f, (w["family"].get(f) or 0) - (tr["family"].get(f) or 0)) for f in goal_eval.PHI_FAMILIES
                   if w["family"].get(f) is not None and tr["family"].get(f) is not None), key=lambda x: -x[1])
    up = ", ".join(f"{f} {d:+.3f}" for f, d in fams if d > 0.01)[:200]
    return f"Δphi {dphi:+.2f}, Δ(g·½compact) {dq:+.2f}; families higher for it: {up or 'none'}"


EPS_CAP = 0.05
PHI_HEADROOM = 2.0
REPAIRS = {
    "eps_cap": f"each family's eps capped at {EPS_CAP} (a noisy family cannot push the residual above 1 - s + {EPS_CAP})",
    "s_clip": "each family's raw score clipped to [0, 1] (the clipped estimators goal_eval keeps beside the raw ones; N's raw specificity ratio reaches -5 on a good circuit)",
    "gate_ramp": "g x clip(phi / phi_tau, 0, 1): compactness gets nothing at phi <= 0 (the empty and one-node circuits)",
    "phi_cap": f"phi capped at phi_tau + {PHI_HEADROOM} (past the bar, faithfulness stops paying and compactness ranks)",
}
VARIANTS = ["v3", "eps_cap", "s_clip", "gate_ramp", "phi_cap", "eps_cap+s_clip", "eps_cap+s_clip+gate_ramp",
            "eps_cap+s_clip+phi_cap", "eps_cap+s_clip+gate_ramp+phi_cap"]


def rescore(r: dict, variant: str) -> float | None:
    """R of a row under a variant ("v3" or "+"-joined REPAIRS), from its family scores, eps, phi_tau and compact (the
    bar phi_tau is kept as measured)."""
    on = set(variant.split("+"))
    fam, eps, pt = r.get("family") or {}, r.get("eps") or {}, r.get("phi_tau")
    sc = [(s_, eps.get(f, goal_eval.EPS_FLOOR)) for f, s_ in fam.items() if s_ is not None]
    if "eps_cap" in on:
        sc = [(s_, min(e, EPS_CAP)) for s_, e in sc]
    if "s_clip" in on:
        sc = [(min(1.0, max(0.0, s_)), e) for s_, e in sc]
    ph = goal_eval._phi(sc) if on & {"eps_cap", "s_clip"} else r.get("phi")
    if ph is None or pt is None:
        return r.get("circuit_reward")
    if "phi_cap" in on:
        ph = min(ph, pt + PHI_HEADROOM)
    g = 1.0 / (1.0 + 2.0 ** (-components.GATE_SLOPE * (ph - pt)))
    if "gate_ramp" in on:
        g *= min(1.0, max(0.0, ph / pt)) if pt > 0 else 1.0
    return ph + g * 0.5 * (r.get("compact") or 0.0)


def variant_table(rows: list[dict]) -> list[str]:
    L = ["## Candidate repairs, rescored from the same rows", "", "Gate B checks passed (texts) when R is recomputed from each "
         "row's family scores and eps (no new measurement; phi_tau kept as measured):", ""]
    L += [f"- **{k}**: {v}" for k, v in REPAIRS.items()]
    L += ["", "| variant | " + " | ".join(f"{t}: {c}" for t in ("full", "core") for c in CHECKS) + " | full: all | core: all |",
          "|---|" + "---|" * (2 * len(CHECKS) + 2)]
    for var in VARIANTS:
        V = verdicts([{**r, "circuit_reward": rescore(r, var)} for r in rows])
        cells = []
        for t in ("full", "core"):
            for ch in CHECKS:
                got = [v[ch][0] is not False for k, v in V.items() if k[1] == t and ch in v]
                cells.append(f"{sum(got)}/{len(got)}")
        for t in ("full", "core"):
            vs = [v for k, v in V.items() if k[1] == t]
            cells.append(f"{sum(all(v[ch][0] is not False for ch in CHECKS if ch in v) for v in vs)}/{len(vs)}")
        L.append(f"| {var} | " + " | ".join(cells) + " |")
    return L + [""]


def markdown(rows: list[dict], args, notes: str = "") -> str:
    V = verdicts(rows)
    groups = list(dict.fromkeys(k[:2] for k in V))
    L = ["# Gate B on the planted toys", "",
         "`experiments/oracle/toys/gate_b_toys.py` (toyloop). Reward v3's circuit part (`components.reward`: circuit_reward = "
         "phi + g·½compact, D = 0 with no teacher reference), eval draws, the exact decomposition (planted.decompose), node cap "
         f"lifted. Texts per toy: {len(set(r['text'] for r in rows))}. Rows: `{args.out.name}`. Tiers: **full** (true = every "
         "reachable active piece of the planted units) and **core** (true = the planted route's positions only, see core_of); "
         "each tier has its own equal-size wrong circuits and controls.", "",
         "Gate B holds on a text when R(true) > max R over each control category (an exact tie, |margin| < 1e-9, a control "
         "behaviourally identical to true, is counted apart and not as a failure). Cell: texts passing, and the worst margin "
         "R(true) − R(best control) with that control (positive = true wins).", "", notes, "## Verdict per toy and tier", "",
         "| toy | tier | wrong_eq | empty | shuffled | collapsed | R(true) − R(teacher) | Gate B |", "|---|---|---|---|---|---|---|---|"]
    for toy, tier in groups:
        keys = [k for k in V if k[:2] == (toy, tier)]
        cells, allok = [], True
        for ch in CHECKS:
            got = [V[k][ch] for k in keys if ch in V[k]]
            if not got:
                cells.append("n/a")
                continue
            ok = sum(g[0] is True for g in got)
            ties = sum(g[0] == "tie" for g in got)
            allok &= ok + ties == len(got)
            worst = min(got, key=lambda g: g[1])
            cells.append(f"{'PASS' if ok + ties == len(got) else 'FAIL'} {ok}/{len(got)}{f' +{ties} tie' if ties else ''} "
                         f"({worst[1]:+.2f}, {worst[2]})")
        te = [V[k]["teacher"][1] for k in keys if "teacher" in V[k]]
        cells.append(f"{np.mean(te):+.2f}" if te else "—")
        L.append(f"| {toy} | {tier} | " + " | ".join(cells) + f" | {'**PASS**' if allok else '**FAIL**'} |")
    L += ["", "## Mean over texts per candidate", ""]
    for toy in dict.fromkeys(r["toy"] for r in rows):
        L += [f"### {toy}", "", "| tier | candidate | category | nodes | R | phi | phi_tau | g | compact | ≥ true (of texts) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        rs = [r for r in rows if r["toy"] == toy]
        trs = {(r["tier"], r["text"]): r for r in rs if r["category"] == "true"}
        for tier, lab in dict.fromkeys((r["tier"], r["candidate"]) for r in rs):
            g = [r for r in rs if (r["tier"], r["candidate"]) == (tier, lab)]
            m = lambda k: (lambda xs: float(np.mean(xs)) if xs else None)([r[k] for r in g if r[k] is not None])  # noqa: E731
            if tier in ("full", "core") and lab != "true":
                beats = f"{sum(r['circuit_reward'] >= trs[(tier, r['text'])]['circuit_reward'] for r in g)}/{len(g)}"
            else:
                beats = "—"
            name = f"**{lab}**" if lab == "true" else lab
            L.append(f"| {tier} | {name} | {g[0]['category']} | {m('nodes'):.0f} | {fmt(m('circuit_reward'))} | {fmt(m('phi'))} | "
                     f"{fmt(m('phi_tau'))} | {fmt(m('g'))} | {fmt(m('compact'))} | {beats} |")
        L.append("")
    L += variant_table(rows)
    L += ["## Failures: which term lets the control win (per text)", ""]
    for (toy, tier, i), v in V.items():
        rs = {r["candidate"]: r for r in rows if (r["toy"], r["tier"], r["text"]) == (toy, tier, i)}
        for ch, (ok, m, lab) in v.items():
            if ok is False:
                L.append(f"- {toy} {tier} text {i}, {ch}: **{lab}** ≥ true by {-m:.2f}: {diagnose(rs['true'], rs[lab])}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--toys", type=Path, default=Path.home() / "mpd-data/toys_planted")
    ap.add_argument("--work", type=Path, default=Path.home() / "mpd-data/gate_b_toys")
    ap.add_argument("--out", type=Path, default=RESULTS / "gate_b_toys.jsonl")
    ap.add_argument("--md", type=Path, default=RESULTS / "gate_b_toys.md")
    ap.add_argument("--only", default=TOYS)
    ap.add_argument("--texts", type=int, default=3)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--notes", type=Path, default=RESULTS / "gate_b_toys_notes.md", help="hand-written diagnosis put at the top")
    ap.add_argument("--summarize", action="store_true", help="rewrite the markdown from --out alone")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    if not args.summarize:
        names = args.only.split(",")
        old = [json.loads(x) for x in args.out.read_text().splitlines()] if args.out.exists() else []
        old = [r for r in old if r["toy"] not in names]
        with args.out.open("w") as fh:
            for r in old:
                fh.write(json.dumps(r) + "\n")
            for name in names:
                if not (args.toys / f"{name}.pt").exists():
                    print(f"{name}: no {args.toys / f'{name}.pt'}, skipped", flush=True)
                    continue
                run_toy(name, args.toys, args.work, args.texts, not args.no_teacher, fh, log=lambda s: print(s, flush=True))
    rows = [json.loads(x) for x in args.out.read_text().splitlines()]
    notes = args.notes.read_text() + "\n" if args.notes and args.notes.exists() else ""
    args.md.write_text(markdown(rows, args, notes))
    print(args.md.read_text())


if __name__ == "__main__":
    main()
