"""Is internal-intervention prediction learnable? (#2951 graph oracle, the lead's learnability experiment.) The data: per
text, measured effects on vpd4l's prediction at the target of
  remove: one VPD-active subcomponent at one position set to zero, everything else the whole model (Native.without);
  cut:    one connection between two of them cut (GoalEval.cuts, path patching of that connection), half those
          whose two ends' removals matter most, half at random;
  swap:   one subcomponent's activation set to its value on a changed text (the token at an influential position
          replaced by the model's likeliest other token there, carry.alternatives), everything else the text's;
each with KL(p || q) in bits at the target, the top token's log2 probability ratio ("falls": below -1, as E2's
questions), and q's top token. Candidates are every VPD-active subcomponent at the target and at the POSITIONS
positions whose changed token moves the prediction most (Native.sensitivity), with their features for the arms that
read them: causal importance, activation (x . v |u|), the routed lens's raised tokens at that position, the atlas's
active-on tokens and activity rate, and the logit lens of the write on the model's top token (o and down at the
target only: the rest do not write the residual stream there; without the final norm's scale, one factor per text).

`measure` (a pod with VPD's checkpoint) writes per text the candidates, the features their run gives and the effects;
features() adds the atlas's and the routed lens's facts (the Mac) to those the questions use.

  learnability.py measure --texts 6000:6100 OUT.jsonl [--split train] [--device cuda] [--offset K --stride S] [--only FILE]
  learnability.py questions TRAIN_MEASURED.jsonl HELDOUT_MEASURED.jsonl OUT_DIR [--held 2000]
  learnability.py baselines OUT_DIR/features/train_32000.jsonl OUT_DIR/features/heldout.jsonl
  learnability.py active --texts 6000:7000 ACTIVE.jsonl      arm "active"'s first stage (the Mac: VPD's importance)

Questions (questions()): remove/cut/swap ask the KL bin (under 0.01, 0.01-0.1, 0.1-1, over 1 bit), falls and the new
top token; rank asks the order of RANK_N of a text's removals by KL. Each kind's questions are drawn equally from the
KL bins (draw()), training sets nested at 2k, 8k, 32k, held-out texts apart. Arms: "ids" (the subcomponents by name
only), "features" (plus each one's per-text facts); "vocab" (the ids rows after a first stage on vocab.jsonl, each
named subcomponent's static facts), "active" (the ids rows after a first stage on active_rows(): from the text alone,
which subcomponents VPD marks active at a position, with activation bins), and both first stages; rows carry the
feature-only baselines' inputs in meta.
"""
from __future__ import annotations

import argparse
import json
import math
import random
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

LN2 = math.log(2)
POSITIONS = 3  # influential context positions besides the target
CUTS = 48  # connections cut per text among the candidates' connections
SWAP_POSITIONS = 2  # changed positions per text for swaps
SWAPS = 48  # swaps per changed position, the candidates whose activation moves most first
BATCH = 64


def _task(name: str) -> dict:
    return json.loads((native.TEXTS / "vpd4l" / f"{name}.json").read_text())


@torch.no_grad()
def candidates(nat, task: dict) -> dict:
    """One text's candidates and the features its run gives (VPD's importance needed)."""
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    t = max(targets)
    g = torch.Generator(device="cpu").manual_seed(native.task_seed(task["id"]))
    moved, _ = nat.sensitivity(ids, targets, g)
    near = [q for q in (moved.argsort(descending=True) + 1).tolist() if q not in targets][:POSITIONS]
    ci = nat.importance(ids)
    acts = nat.activations(ids)
    top = int(nat.reference([ids], targets)[0, -1].argmax())
    tg = nat.target
    cands = []
    for nd in nat._vpd_order(ids, targets):
        n, q, c = nd
        if q not in near and q != t:
            continue
        u = tg.site(n).U[c]
        # the write's direct effect on the top token's logit, but for the final norm's scale (one factor at t)
        ll = (float(acts[n][q, c] / u.norm() * (u * tg.ln_f) @ tg.wte[top])
              if native._layer_kind(nd)[1] in ("o_proj", "down_proj") and q == t else None)
        cands.append({"node": [n, q, c], "ci": float(ci[n][q, c]), "act": float(acts[n][q, c]), "logit_lens": ll})
    return {"task": task["id"], "split": task["split"], "T": len(ids), "target": t, "top": top, "near": near,
            "candidates": cands}


@torch.no_grad()
def features(nat, rec: dict, nodes: list[tuple], atlas, lens) -> dict:
    """The atlas's and the routed lens's facts of some of a text's candidates (the Mac: the atlas and the lens's fit):
    node -> {"routed": the tokens its write raises most at its position, [[id, value]], or None for q and k; "on": the
    atlas's most frequent active-on tokens; "rate": its activity rate on Pile text}."""
    ids = _task(rec["task"])["prompts"][0]["token_ids"]
    routed = dict(zip(nodes, lens.readout(ids, nodes))) if lens else {}
    out = {}
    for nd in nodes:
        e = atlas.entry(nd[0], nd[2]) if atlas else None
        count = {}
        for x in e["contexts"] if e else []:
            count[x["token"]] = count.get(x["token"], 0) + 1
        r = routed.get(nd)
        out[nd] = {"routed": [[i, round(float(v), 3)] for i, v in r["raised"][:3]] if r else None,
                   "on": sorted(count, key=lambda k: -count[k])[:3], "rate": e["active_rate"] if e else None}
    return out


def _effect(lp0: torch.Tensor, lq: torch.Tensor, top: int) -> dict:
    """KL(p || q) bits, the top token's log2 ratio, falls, q's top, at the target (lp0, lq [V])."""
    bits = float(lq[top] - lp0[top]) / LN2
    return {"kl": float((lp0.exp() * (lp0 - lq)).sum()) / LN2, "bits": bits, "falls": bits < -1.0, "top": int(lq.argmax())}


@torch.no_grad()
def patched(nat, ids: list[int], targets: list[int], edits: list[tuple]) -> torch.Tensor:
    """log q at the last target [len(edits), V] of the model on ids with, per edit (matrix, position, index, value),
    that one subcomponent's activation x . v replaced: value None removes it (as Native.without: the whole model less
    its write), a number sets it. The plain model with the matrix's output corrected at that position (W x less or
    plus the change times u: Native.without's numbers to float rounding), computed from the edit's position on, the
    text's states before it reused (vpd_model.Target.hidden's prefix), BATCH edits at one position per forward."""
    tg = nat.target
    t_last = max(targets)
    out = [None] * len(edits)
    at = {}
    for i, e in enumerate(edits):
        at.setdefault(e[1], []).append(i)
    kv = tg.keys_values(torch.tensor([ids], device=nat.dev))
    for s, idx in sorted(at.items()):
        for b0 in range(0, len(idx), BATCH):
            rows = idx[b0:b0 + BATCH]
            B = len(rows)
            per = {}
            for b, i in enumerate(rows):
                n, _, c, v = edits[i]
                per.setdefault(n, []).append((b, c, v))
            origs = {}
            for n, rs in per.items():
                st = tg.site(n)
                R = torch.tensor([r[0] for r in rs], device=nat.dev)
                C = torch.tensor([r[1] for r in rs], device=nat.dev)
                val = torch.tensor([0.0 if r[2] is None else r[2] for r in rs], device=nat.dev)
                V, U = st.V[:, C].T, st.U[C]  # [k, d_in], [k, d_out]

                def f(x, orig=st._forward, R=R, V=V, U=U, val=val):
                    y = orig(x).clone()
                    a = (x[R, 0] * V).sum(-1)  # position s is the first computed
                    y[R, 0] += (val - a)[:, None] * U
                    return y
                origs[n] = st._forward
                st._forward = f
            try:
                x = torch.tensor([ids] * B, device=nat.dev)
                if s > 0:
                    prefix = [tuple(w.expand(B, *w.shape[1:]) for w in layer) for layer in kv]
                    h = tg.hidden(x, start=s, prefix=prefix)[:, t_last - s]
                else:
                    h = tg.hidden(x)[:, t_last]
            finally:
                for n, o in origs.items():
                    tg.site(n)._forward = o
            lq = torch.log_softmax((h @ tg.wte.T).float(), -1)
            for b, i in enumerate(rows):
                out[i] = lq[b]
    return torch.stack(out)


@torch.no_grad()
def projections(nat, ids: list[int], names: set) -> dict:
    """x . v of every subcomponent of the named matrices on the model's run of ids, [T, C] per matrix."""
    tg = nat.target
    for n in names:
        tg.site(n).cache_input = True
    try:
        tg.hidden(torch.tensor([ids], device=nat.dev))
        return {n: tg.site(n).last_input[0] @ tg.site(n).V for n in names}
    finally:
        for n in names:
            tg.site(n).cache_input = False
            tg.site(n).last_input = None


def measure(ge, task: dict, log=print) -> dict:
    """candidates() with each candidate's removal effect, CUTS connection cuts and the swaps."""
    nat = ge.nat
    t0 = time.time()
    pre = candidates(nat, task)
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    rng = random.Random(native.task_seed(pre["task"]))
    lp0 = nat.reference([ids], targets)[0, -1]
    top = int(lp0.argmax())
    nodes = [tuple(c["node"]) for c in pre["candidates"]]
    for c, lq in zip(pre["candidates"], patched(nat, ids, targets, [(*nd, None) for nd in nodes]) if nodes else []):
        c["remove"] = _effect(lp0, lq, top)
    par, outs = native.all_edges(set(nodes), targets)
    pairs = [(w, r) for r, ws in par.items() for w in ws] + [(w, None) for w in outs]
    # half the connections whose ends' removals matter most (most connections carry nothing), half at random
    gone = {tuple(c["node"]): c["remove"]["kl"] for c in pre["candidates"]}
    strong = sorted(pairs, key=lambda e: -min(gone[e[0]], gone[e[1]] if e[1] is not None else math.inf))[:CUTS // 2]
    rest = [e for e in pairs if e not in set(strong)]
    pairs = strong + rng.sample(rest, min(len(rest), CUTS - len(strong)))
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    cuts = []
    for (w, r), d in zip(pairs, ge.cuts(tx, pairs, detail=True) if pairs else []):
        bits = (d["logp_model_top"][-1] - float(lp0[top])) / LN2
        cuts.append({"edge": [list(w), None if r is None else list(r)], "kl": d["kl"], "bits": bits, "falls": bits < -1.0,
                     "top": int(d["top"][-1])})
    swaps = []
    names = {nd[0] for nd in nodes}
    base = projections(nat, ids, names)
    for q in pre["near"][:SWAP_POSITIONS]:
        x = carry.alternatives(nat, ids, q, ranks=(0,))[0]
        alt = projections(nat, x, names)
        lx = nat.reference([x], targets)[0, -1]
        moved = sorted((nd for nd in nodes if nd[1] >= q), key=lambda nd: -abs(float(alt[nd[0]][nd[1], nd[2]] - base[nd[0]][nd[1], nd[2]])))
        pick = moved[:SWAPS]
        edits = [(*nd, float(alt[nd[0]][nd[1], nd[2]])) for nd in pick]
        for nd, lq in zip(pick, patched(nat, ids, targets, edits) if pick else []):
            un = float(nat.target.site(nd[0]).U[nd[2]].norm())  # activations in Native.activations' units, (x . v) |u|
            swaps.append({"node": list(nd), "position": q, "new": x[q], "act_x": float(base[nd[0]][nd[1], nd[2]]) * un,
                          "act_changed": float(alt[nd[0]][nd[1], nd[2]]) * un, **_effect(lp0, lq, top),
                          "of_change": _effect(lp0, lx, top)["kl"]})
    log(f"{pre['task']}: {len(nodes)} removals, {len(cuts)} cuts, {len(swaps)} swaps ({time.time() - t0:.1f} s)")
    return {**pre, "cuts": cuts, "swaps": swaps}


# ---- questions


BINS = (0.01, 0.1, 1.0)  # KL bin edges, bits
BIN_TEXT = ("under 0.01 bits", "0.01 to 0.1 bits", "0.1 to 1 bit", "over 1 bit")
WINDOW = 48  # tokens shown before the target; and AROUND on either side of each other position a question names
AROUND = 6
MIX = {"remove": 0.4, "cut": 0.2, "swap": 0.2, "rank": 0.2}
RANK_N = 8
ARMS = ("ids", "features")  # the "vocab" and "active" arms ask the ids rows after their first stage


def kl_bin(kl: float) -> int:
    return sum(kl >= b for b in BINS)


def _items(rec: dict) -> list[dict]:
    """A text's measured interventions: {"kind", "node" or "edge" (and the swap's change), "kl", "falls", "top"}."""
    out = [{"kind": "remove", "node": c["node"], **c["remove"]} for c in rec["candidates"]]
    out += [{"kind": "cut", **c} for c in rec["cuts"]]
    out += [{"kind": "swap", **c} for c in rec["swaps"]]
    return out


def _nodes_of(it: dict) -> list[tuple]:
    if it["kind"] == "cut":
        return [tuple(it["edge"][0])] + ([tuple(it["edge"][1])] if it["edge"][1] else [])
    if it["kind"] == "rank":
        return [tuple(c["node"]) for c in it["cands"]]
    return [tuple(it["node"])]


def draw(recs: list[dict], n: int, seed: int) -> list[dict]:
    """n questions over the texts, MIX by kind, each kind's equally from the four KL bins where it has them (the large
    effects are rare: an undrawn bin's share goes to the others); a rank question: RANK_N of a text's removals, two
    per KL bin where it has them, else the largest left."""
    rng = random.Random(seed)
    pools = {}
    for r in recs:
        for it in _items(r):
            pools.setdefault((it["kind"], kl_bin(it["kl"])), []).append((r["task"], it))
    for v in pools.values():
        rng.shuffle(v)
    out = []
    for kind in ("remove", "cut", "swap"):
        want = round(n * MIX[kind])
        bins = [b for b in range(len(BINS) + 1) if pools.get((kind, b))]
        got, k = [], 0
        while len(got) < want and any(pools.get((kind, b)) for b in bins):
            b = bins[k % len(bins)]
            k += 1
            if pools.get((kind, b)):
                got.append(pools[(kind, b)].pop())
        out += [{"task": t, **it} for t, it in got]
    for _ in range(round(n * MIX["rank"])):
        r = rng.choice(recs)
        by = {}
        for c in r["candidates"]:
            by.setdefault(kl_bin(c["remove"]["kl"]), []).append(c)
        pick = []
        for b in sorted(by, reverse=True):
            pick += rng.sample(by[b], min(2, len(by[b])))
        left = sorted((c for c in r["candidates"] if c not in pick), key=lambda c: -c["remove"]["kl"])
        pick = (pick + left)[:RANK_N]
        if len(pick) < RANK_N:
            continue
        rng.shuffle(pick)
        out.append({"task": r["task"], "kind": "rank", "cands": [{"node": c["node"], "kl": c["remove"]["kl"]} for c in pick]})
    rng.shuffle(out)
    return out


def _fact_text(c: dict, f: dict, tok) -> str:
    """A candidate's per-text features in words (arm "features")."""
    out = [f"importance {c['ci']:.2f}", f"activation {c['act']:+.1f}"]
    if c.get("logit_lens") is not None:
        out.append(f"logit lens on the predicted token {c['logit_lens']:+.2f}")
    if f.get("routed"):
        out.append("routed lens here raises " + " ".join(repr(x) for x in tok([i for i, _ in f["routed"]])))
    if f.get("on"):
        out.append("active on " + " ".join(repr(x) for x in f["on"]) + (f" ({100 * f['rate']:.2g}% of tokens)" if f.get("rate") is not None else ""))
    return ", ".join(out)


def _plain_feats(c: dict | None, nd: tuple, t: int) -> dict:
    """A node's numeric features for the feature-only baselines."""
    layer, kind = native._layer_kind(nd)
    return {"ci": c["ci"] if c else None, "act": c["act"] if c else None, "logit_lens": c.get("logit_lens") if c else None,
            "layer": layer, "matrix": kind, "at_target": nd[1] == t}


def render(q: dict, rec: dict, arm: str, strings: list[str], tok, feats: dict) -> dict:
    """One question as {"prompt", "answer", "choices" (the answers it is scored over), "label", "meta"}."""
    import components

    t = rec["target"]
    nodes = _nodes_of(q)
    shown = set(range(max(0, t - WINDOW), t + 1))
    for nd in nodes:
        shown |= set(range(max(0, nd[1] - AROUND), min(t, nd[1] + AROUND) + 1))
    if q["kind"] == "swap":
        shown |= set(range(max(0, q["position"] - AROUND), min(t, q["position"] + AROUND) + 1))
    parts, run = [], []
    for i in sorted(shown):
        if run and i != run[-1] + 1:
            parts.append(run)
            run = []
        run.append(i)
    parts.append(run)
    text = " ... ".join(f"[{r[0]}]" + repr("".join(strings[r[0]:r[-1] + 1])) for r in parts)
    by = {tuple(c["node"]): c for c in rec["candidates"]}
    nt = components.node_text
    lines = [f"<text> {text}", f"<target> position {t}, {strings[t]!r}; the model predicts {tok([rec['top']])[0]!r}"]
    if q["kind"] == "rank":
        names = "abcdefgh"
        lines.append("<candidates> " + "; ".join(f"{names[i]}={nt(tuple(c['node']))}" for i, c in enumerate(q["cands"])))
        if arm == "features":
            lines += [f"<facts> {names[i]}: {_fact_text(by[tuple(c['node'])], feats.get(tuple(c['node']), {}), tok)}" for i, c in enumerate(q["cands"])]
        lines.append("<question> order the candidates by how much removing each alone changes the prediction (KL), largest first")
        order = sorted(range(RANK_N), key=lambda i: -q["cands"][i]["kl"])
        answer = " ".join(names[i] for i in order)
        return {"prompt": "\n".join(lines) + "\n", "answer": answer, "choices": list(names), "label": order[0],
                "meta": {"task": q["task"], "kind": "rank", "kls": [c["kl"] for c in q["cands"]],
                         "nodes": [_plain_feats(by.get(tuple(c["node"])), tuple(c["node"]), t) for c in q["cands"]],
                         "slots": [[nt(tuple(c["node"])), c["node"]] for c in q["cands"]]}}
    if q["kind"] == "remove":
        what = f"remove {nt(nodes[0])}"
    elif q["kind"] == "cut":
        what = f"cut {nt(nodes[0])} -> " + (nt(nodes[1]) if len(nodes) > 1 else "the prediction")
    else:
        what = (f"set {nt(nodes[0])} to its value when the token at {q['position']}, {strings[q['position']]!r}, is replaced by "
                f"{tok([q['new']])[0]!r}")
    lines.append(f"<intervention> {what}")
    if arm == "features":
        lines += [f"<facts> {nt(nd)}: {_fact_text(by[nd], feats.get(nd, {}), tok)}" for nd in nodes if nd in by]
        if q["kind"] == "swap":
            lines.append(f"<facts> its activation there would go from {q['act_x']:+.1f} to {q['act_changed']:+.1f}")
    lines.append("<question> how far the prediction moves (KL), whether the predicted token's probability falls below half, "
                 "and the most likely token after it")
    b = kl_bin(q["kl"])
    answer = f"KL {BIN_TEXT[b]}; falls={'yes' if q['falls'] else 'no'}; top={tok([q['top']])[0]!r}"
    return {"prompt": "\n".join(lines) + "\n", "answer": answer, "choices": [f"KL {x};" for x in BIN_TEXT], "label": b,
            "meta": {"task": q["task"], "kind": q["kind"], "kl": q["kl"], "falls": q["falls"], "top": q["top"],
                     "slots": [[nt(nd), list(nd)] for nd in nodes],
                     "nodes": [_plain_feats(by.get(nd), nd, t) for nd in nodes],
                     "act_change": q["act_changed"] - q["act_x"] if q["kind"] == "swap" else None}}


def vocab_rows(nodes: set, atlas, conn, tok) -> list[dict]:
    """Arm "vocab"'s first stage: per subcomponent (matrix, index) its static facts: the atlas's active-on tokens with
    a context, activity rate, the logit lens's and routed lens's raised and lowered tokens, and its strongest
    weight-level connections (connectome)."""
    import components

    out = []
    for n, c in sorted(nodes):
        e = atlas.entry(n, c)
        name = components.node_text((n, 0, c)).rsplit("@", 1)[0]
        lines = [f"active on {100 * e['active_rate']:.2g}% of tokens"]
        ctx = e["contexts"][:3]
        if ctx:
            lines.append("contexts: " + " | ".join(repr(x["before"][-30:] + "[" + x["token"] + "]" + x["after"][:10]) for x in ctx))
        for key, label in (("lens", "logit lens"), ("routed", "routed lens")):
            r = e.get(key)
            if r:
                lines.append(f"{label} raises " + " ".join(repr(t) for t, _ in r["raised"][:5]) + "; lowers "
                             + " ".join(repr(t) for t, _ in r["lowered"][:5]))
        if conn is not None:
            ps = conn.partners(n, c, "residual.downstream") if n.endswith(("o_proj", "down_proj")) else {}
            if ps and len(ps.get("index", [])):
                lines.append("strongest readers: " + ", ".join(f"{components.node_text((conn.names[int(s)], 0, int(i))).rsplit('@', 1)[0]}"
                                                                for s, i in zip(ps["site"][:4], ps["index"][:4])))
        out.append({"prompt": f"<subcomponent> {name}\n<question> what is it\n", "answer": "; ".join(lines), "meta": {"node": [n, c]}})
    return out


ACT_BINS = (1.0, 5.0)  # |activation| bin edges, (x . v) |u|: bins 1, 2, 3, signed
ACTIVE_POSITIONS = 8  # per text: the target, then random positions
ACTIVE_SITES = 4  # matrices asked per position


def act_bin(a: float) -> str:
    return ("+" if a >= 0 else "-") + str(1 + sum(abs(a) >= b for b in ACT_BINS))


@torch.no_grad()
def active_rows(nat, task: dict, tok, rng: random.Random) -> list[dict]:
    """Arm "active"'s first stage, from the text alone: at ACTIVE_POSITIONS positions (the target and random ones) and
    ACTIVE_SITES random matrices each, the subcomponents VPD marks active there (causal importance above zero) with
    their activation bins (act_bin), largest first: {"prompt", "answer", "meta"}."""
    import components

    p = task["prompts"][0]
    ids, t = p["token_ids"], max(p["target_positions"])
    ci, acts = nat.importance(ids), nat.activations(ids)
    strings = tok(ids)
    where = [t] + rng.sample(range(1, t), min(t - 1, ACTIVE_POSITIONS - 1))
    out = []
    for q in where:
        lo = max(0, q - WINDOW)
        text = f"[{lo}]" + repr("".join(strings[lo:q + 1]))
        for n in rng.sample(nat.names, ACTIVE_SITES):
            on = (ci[n][q] > 0).nonzero().flatten().tolist()
            on.sort(key=lambda c: -abs(float(acts[n][q, c])))
            site = components.node_text((n, q, 0)).split("#")[0]
            out.append({"prompt": f"<text> {text}\n<position> {q}, {strings[q]!r}\n<question> the active subcomponents of {site} "
                                  "here, largest activation first, with activation bins\n",
                        "answer": " ".join(f"#{c}:{act_bin(float(acts[n][q, c]))}" for c in on) or "none",
                        "meta": {"task": task["id"], "position": q, "site": n, "active": len(on)}})
    return out


def questions(recs_train: list[dict], recs_held: list[dict], out_dir: Path, sizes=(2_000, 8_000, 32_000), held: int = 2_000,
              seed: int = 0, held_only: bool = False) -> None:
    """Per arm, the training sets of each size (nested: the smaller a prefix of the larger), the held-out set, and
    the vocabulary stage's rows (out_dir/<arm>/train_<n>.jsonl, heldout.jsonl; vocab.jsonl). held_only: the held-out
    set alone, the training sets already in out_dir kept (vocab.jsonl from both). Held-out texts must be development
    texts (text9000-9099; text9100-9149 are sealed, 9150-9199 the far donors)."""
    bad = [r["task"] for r in recs_held if not 9000 <= int(r["task"][4:]) <= 9099]
    if bad:
        raise ValueError(f"held-out texts outside the development range text9000-9099: {bad[:5]}")
    import components

    sys.path.insert(0, str(ROOT))
    nat = native.Native("cpu")
    atlas, lens = components._atlas(), components._lens(nat)
    tok = lambda xs: components.token_strings("vpd4l", xs)  # noqa: E731
    sets = {"heldout": (draw(recs_held, held, seed + 1), {r["task"]: r for r in recs_held})}
    vocab = set()
    if held_only:
        for x in (out_dir / "ids" / f"train_{max(sizes)}.jsonl").read_text().splitlines():
            vocab |= {(nd[0], nd[2]) for _, nd in json.loads(x)["meta"]["slots"]}
    else:
        sets["train"] = (draw(recs_train, max(sizes), seed), {r["task"]: r for r in recs_train})
    for split, (qs, by) in sets.items():
        of = {}
        for i, q in enumerate(qs):
            of.setdefault(q["task"], []).append(i)
        rows = {arm: [None] * len(qs) for arm in ARMS}  # in draw()'s shuffled order: nested prefixes stay mixed
        for task, idx in of.items():
            rec = by[task]
            nodes = sorted({nd for i in idx for nd in _nodes_of(qs[i])})
            feats = features(nat, rec, nodes, atlas, lens)
            strings = tok(_task(task)["prompts"][0]["token_ids"])
            for i in idx:
                for arm in ARMS:
                    rows[arm][i] = render(qs[i], rec, "features" if arm == "features" else "ids", strings, tok, feats)
            vocab |= {(n, c) for n, _, c in nodes}
        for arm in ARMS:
            d = out_dir / arm
            d.mkdir(parents=True, exist_ok=True)
            rs = rows[arm]
            if split == "heldout":
                (d / "heldout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
            else:
                for n in sizes:
                    (d / f"train_{n}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs[:n]))
    try:
        conn = components._connectome()
    except Exception:  # noqa: BLE001 - the connectome is optional (its tables may not be built)
        conn = None
    (out_dir / "vocab.jsonl").write_text("".join(json.dumps(r) + "\n" for r in vocab_rows(vocab, atlas, conn, tok)))


MATRICES = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")


def _x(f: dict) -> list[float]:
    ci, act, ll = f["ci"] or 0.0, f["act"] or 0.0, f["logit_lens"]
    return ([ci, abs(act), math.log1p(abs(act)), ci * abs(act), float(ll is not None), abs(ll or 0.0), float(f["at_target"])]
            + [float(f["layer"] == k) for k in range(4)] + [float(f["matrix"] == m) for m in MATRICES])


def _row_x(m: dict) -> list[float]:
    nodes = m["nodes"]
    x = _x(nodes[0]) + (_x(nodes[1]) if len(nodes) > 1 else [0.0] * len(_x(nodes[0]))) + [float(len(nodes) > 1)]
    ch = m.get("act_change")
    return x + [abs(ch or 0.0), math.log1p(abs(ch or 0.0))]


def baselines(train: list[dict], held: list[dict]) -> dict:
    """The feature-only baselines on the held-out rows: per kind (remove, cut, swap) the KL bin's log loss (bits) of
    the training prior and of a multinomial logistic regression on the nodes' features (importance, activation, logit
    lens, layer, matrix, at the target; a swap's activation change), and their accuracy beside always "under 0.01"
    (the no-change answer); per KL bin of the truth, the same log losses. Rank: top-1 and Kendall's tau of the order by
    importance x |activation| and by the logit lens (where it has none, importance x |activation| below every one
    that has), against the measured removals."""
    import numpy as np
    from scipy.stats import kendalltau
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = {}
    K = len(BINS) + 1
    for kind in ("remove", "cut", "swap"):
        tr = [r for r in train if r["meta"]["kind"] == kind]
        he = [r for r in held if r["meta"]["kind"] == kind]
        if not tr or not he:
            continue
        ytr, yhe = np.array([r["label"] for r in tr]), np.array([r["label"] for r in he])
        prior = (np.bincount(ytr, minlength=K) + 1) / (len(ytr) + K)
        sc = StandardScaler().fit([_row_x(r["meta"]) for r in tr])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform([_row_x(r["meta"]) for r in tr]), ytr)
        P = np.zeros((len(he), K))
        P[:, lr.classes_] = lr.predict_proba(sc.transform([_row_x(r["meta"]) for r in he]))
        P = np.clip(P, 1e-6, 1)
        ll_lr = -np.log2(P[np.arange(len(he)), yhe])
        ll_prior = -np.log2(prior[yhe])
        out[kind] = {"n": len(he), "logloss_prior": float(ll_prior.mean()), "logloss_features": float(ll_lr.mean()),
                     "acc_features": float((P.argmax(1) == yhe).mean()), "acc_no_change": float((yhe == 0).mean()),
                     "by_bin": {BIN_TEXT[b]: {"n": int((yhe == b).sum()), "prior": float(ll_prior[yhe == b].mean()),
                                              "features": float(ll_lr[yhe == b].mean())} for b in range(K) if (yhe == b).any()}}
    rk = [r for r in held if r["meta"]["kind"] == "rank"]
    if rk:
        res = {}
        for name, score in (("importance x activation", lambda f: (f["ci"] or 0) * abs(f["act"] or 0)),
                            ("logit lens", lambda f: (1e6 + abs(f["logit_lens"])) if f["logit_lens"] is not None else (f["ci"] or 0) * abs(f["act"] or 0))):
            top1, taus = [], []
            for r in rk:
                s_ = [score(f) for f in r["meta"]["nodes"]]
                kl = r["meta"]["kls"]
                top1.append(int(np.argmax(s_)) == int(np.argmax(kl)))
                taus.append(kendalltau(s_, kl).statistic)
            res[name] = {"top1": float(np.mean(top1)), "kendall_tau": float(np.nanmean(taus))}
        res["chance top1"] = 1 / RANK_N
        out["rank"] = {"n": len(rk), **res}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--texts", required=True, help="A:B, text numbers")
    m.add_argument("out")
    m.add_argument("--split", default=None)
    m.add_argument("--device", default="cpu")
    m.add_argument("--offset", type=int, default=0)
    m.add_argument("--stride", type=int, default=1)
    m.add_argument("--only", help="a file of text numbers, one per line: those alone")
    qa = sub.add_parser("questions")
    qa.add_argument("train", help="measure's output on train texts")
    qa.add_argument("heldout", help="measure's output on held-out texts")
    qa.add_argument("out_dir")
    qa.add_argument("--held", type=int, default=2_000)
    qa.add_argument("--held-only", action="store_true", help="the held-out set (and vocab.jsonl) alone, the training sets kept")
    ac = sub.add_parser("active")
    ac.add_argument("--texts", required=True, help="A:B, text numbers")
    ac.add_argument("out")
    ac.add_argument("--split", default="train")
    bs = sub.add_parser("baselines")
    bs.add_argument("train", help="questions' features/train_<n>.jsonl")
    bs.add_argument("heldout", help="questions' features/heldout.jsonl")
    args = ap.parse_args()
    if args.cmd == "baselines":
        read = lambda f: [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]  # noqa: E731
        return print(json.dumps(baselines(read(args.train), read(args.heldout)), indent=1))
    if args.cmd == "active":
        import components

        nat = native.Native("cpu")
        tok = lambda xs: components.token_strings("vpd4l", xs)  # noqa: E731
        lo, hi = map(int, args.texts.split(":"))
        rows = []
        for k in range(lo, hi):
            path = native.TEXTS / "vpd4l" / f"text{k}.json"
            if not path.exists():
                continue
            task = json.loads(path.read_text())
            if task["split"] != args.split or max(task["prompts"][0]["target_positions"]) < 2:
                continue
            rows += active_rows(nat, task, tok, random.Random(native.task_seed(task["id"])))
        random.Random(0).shuffle(rows)
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return print(f"{len(rows)} rows, {sum(r['meta']['active'] for r in rows)} active labels")
    if args.cmd == "questions":
        read = lambda f: [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]  # noqa: E731
        return questions([] if args.held_only else read(args.train), read(args.heldout), Path(args.out_dir), held=args.held,
                         held_only=args.held_only)
    ge = goal_eval.GoalEval(native.Native(args.device))
    lo, hi = map(int, args.texts.split(":"))
    only = {int(x) for x in Path(args.only).read_text().split()} if args.only else None
    todo = [k for k in range(lo, hi) if only is None or k in only]
    for k in todo[args.offset::args.stride]:
        path = native.TEXTS / "vpd4l" / f"text{k}.json"
        if not path.exists():
            continue
        task = json.loads(path.read_text())
        if (args.split and task["split"] != args.split) or max(task["prompts"][0]["target_positions"]) < 2:
            continue
        try:
            r = measure(ge, task, log=lambda s: print(s, flush=True))
        except torch.OutOfMemoryError as e:  # a batch too large beside the other processes: that text skipped, said
            print(f"{task['id']}: skipped, {str(e).splitlines()[0]}", flush=True)
            torch.cuda.empty_cache()
            continue
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
