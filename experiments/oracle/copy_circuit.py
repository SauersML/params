"""V2 of the circuit finder's validation (the lead's 10-15 brief): the copy (induction) circuit on Qwen3 in heads and
neurons, every number a measured intervention (head_routes.Runner). A copy text repeats a cue token: "... met Daniel
Kowalczyk at ... Daniel" -> " Kowalczyk" (or an adjective and its noun); its corrupted twin replaces the first cue with
another token of the same length, so nothing earlier matches the final cue. Positions: cue1 (the first cue), next1 (the
token after it: the answer's first token), cue2 (the last position, the second cue). Contrast: log2 p(answer) - log2 of
the summed p of the alternatives (the clean run's five likeliest others and the corrupted run's top token), gap = clean
- corrupted.

Stages (all texts; roles and circuit units chosen on the dev texts, everything reported on every text):
 A1 heads: each head zeroed at positions cue1.. and each patched there from the corrupted run (shares of the gap lost);
    its attention cue2 -> next1 (induction) and next1 -> cue1 (previous token) in the clean run;
 A2 the HEADS largest by the dev texts' mean lost share (patched): patched at next1 alone and at cue2 alone, and unable
    to see next1 from cue2 or cue1 from next1 (attention knockouts);
    roles: induction (dev mean attention cue2 -> next1 at least ATT and patched at cue2 losing at least MATTERS / 2 on
    average), previous token (attention next1 -> cue1, patched at next1), up to three each;
 A3 each role head's route (head_routes.route_story), by zeroing and by patching from the corrupted run: the knockout,
    the neurons (and earlier heads) setting its key and its query, the neurons reading its output;
 B  the circuit: the role heads and the units whose patched share is at least MATTERS on at least SHARE_DEV of the dev
    texts, each with the share of all texts it matters on;
 C  per text: the circuit's units patched from the corrupted run into the clean one (lost share, necessity), only they
    patched from the clean run into the corrupted one (restored share, sufficiency), and their values from another text
    of the same template and length (swapped: the share of the way the answer moves to that text's answer).

  python experiments/oracle/copy_circuit.py OUT [--model Qwen/Qwen3-8B] [--dtype float32] [--n 60] [--dev 20]
writes OUT/texts.json, OUT/a.json (stage A per text), OUT/circuit.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
import head_routes  # noqa: E402

FIRST = ["Daniel", "Marcus", "Peter", "Laura", "Simon", "Helen", "Oliver", "Grace", "Victor", "Nina", "Hugo", "Clara",
         "Felix", "Irene", "Oscar", "Paula", "Martin", "Diana", "Arthur", "Julia", "Henry", "Emma", "Lucas", "Sofia",
         "Walter", "Rachel", "Frank", "Alice", "George", "Maria", "Edward", "Sarah", "Thomas", "Anna", "David", "Ruth"]
SURNAME = ["Kowalczyk", "Abernathy", "Thistlewood", "Okonkwo", "Varga", "Lindqvist", "Brennan", "Castellano", "Duquesne",
           "Halvorsen", "Ivanova", "Jablonski", "Kerrigan", "Lombardi", "Moreau", "Nakamura", "Ostrowski", "Pemberton",
           "Quintero", "Rasmussen", "Szabo", "Tanaka", "Underwood", "Valdez", "Whitlock", "Yamamoto", "Zielinski",
           "Achterberg", "Bellamy", "Crowther", "Delacroix", "Esposito", "Fairbanks", "Gallagher", "Hargreaves"]
ADJ = ["purple", "wooden", "silver", "broken", "tiny", "golden", "rusty", "velvet", "frozen", "striped", "ancient",
       "crooked", "glass", "copper", "dusty", "painted", "hollow", "woolen", "marble", "plastic"]
NOUN = ["lantern", "kettle", "compass", "ladder", "violin", "teapot", "anchor", "basket", "candle", "helmet", "mirror",
        "saddle", "trumpet", "barrel", "chess", "hammer", "pillow", "rocket", "statue", "wagon"]
TEMPLATES = [  # ({a} the cue, {b} the copied word); every template ends with the cue
    ("met", "name", "Yesterday I met {a} {b} at the train station. We talked for a while about the weather, the delays and"
                    " our plans for the weekend. Before we said goodbye, {a}"),
    ("report", "name", "The report was written by {a} {b}, a researcher at the city museum. It describes three old maps"
                       " and a letter found in the archive. According to the museum, {a}"),
    ("chair", "name", "At the meeting, the chair introduced {a} {b} as the new treasurer. Several members asked questions"
                      " about the budget and the schedule. In the end, {a}"),
    ("bridge", "phrase", "She found a {a} {b} under the old bridge and took it home. Her brother wanted to know where"
                         " she had found the {a}"),
    ("shop", "phrase", "The shop on the corner sells a {a} {b} next to the fresh bread. On Sunday I went there and asked"
                       " the owner about the {a}"),
]
HEADS = 24          # heads followed in A2
ATT = 0.1           # an induction head attends cue2 -> next1 at least this much (dev mean)
WRITERS = 16        # key / query writers measured in full runs per induction head (by exact attention change)
READERS = 32        # readers measured in full runs per induction head (by exact value change)
MATTERS = 0.1       # a unit matters on a text when its measured share of the gap is at least this
SHARE_DEV = 0.3     # the circuit: units that matter on at least this share of the dev texts
GAP_MIN = 2.0       # bits: a usable text's clean - corrupted contrast


def make_texts(tok, n: int, seed: int = 0) -> list[dict]:
    """n copy texts (round-robin over TEMPLATES) whose cue tokens are single tokens and whose corrupted twin differs at
    cue1 only, with ids, corrupt ids, the positions and the answer token."""
    rng = random.Random(seed)
    one = lambda w: len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1  # noqa: E731
    first, adj = [w for w in FIRST if one(w)], [w for w in ADJ if one(w)]
    out, tries = [], 0
    while len(out) < n and tries < 100 * n:
        tries += 1
        name, kind, tpl = TEMPLATES[len(out) % len(TEMPLATES)]
        cues, words = (first, SURNAME) if kind == "name" else (adj, NOUN)
        a, a2 = rng.sample(cues, 2)
        b = rng.choice(words)
        ids = tok(tpl.format(a=a, b=b), add_special_tokens=False)["input_ids"]
        cue = tok(" " + a, add_special_tokens=False)["input_ids"][0]
        ans = tok(" " + b, add_special_tokens=False)["input_ids"][0]
        hits = [p for p, t in enumerate(ids) if t == cue]
        if len(hits) != 2 or hits[1] != len(ids) - 1 or ids[hits[0] + 1] != ans:
            continue
        bad = list(ids)
        bad[hits[0]] = tok(" " + a2, add_special_tokens=False)["input_ids"][0]
        out.append({"id": f"{name}{len(out)}", "template": name, "kind": kind, "cue": a, "word": b, "replacement": a2,
                    "text": tpl.format(a=a, b=b), "ids": ids, "corrupt_ids": bad, "answer": ans,
                    "pos": {"cue1": hits[0], "next1": hits[0] + 1, "cue2": len(ids) - 1}})
    return out


class Text:
    """One copy text on the runner: both runs' states, the contrast, and measured shares."""

    def __init__(self, rn: head_routes.Runner, t: dict):
        self.rn, self.t = rn, t
        self.ids, self.bad = t["ids"], t["corrupt_ids"]
        self.p1, self.n1, self.q = t["pos"]["cue1"], t["pos"]["next1"], t["pos"]["cue2"]
        rn.answer, rn.alternatives = [t["answer"]], [0]
        clean = rn.run(self.ids, [{}], capture=True)
        corrupt = rn.run(self.bad, [{}], capture=True)
        lc, lb = clean["lp"][0], corrupt["lp"][0]
        alt = [int(z) for z in lc.topk(6).indices.tolist() if z != t["answer"]][:5]
        yb = int(lb.argmax())
        if yb != t["answer"] and yb not in alt:
            alt.append(yb)
        self.alt = alt
        self.st, self.sb = clean["states"], corrupt["states"]
        self.top = int(lc.argmax())
        self.c = float(self._c(lc[None])[0])
        self.cb = float(self._c(lb[None])[0])
        self.gap = self.c - self.cb

    def _c(self, lp):
        self.rn.answer, self.rn.alternatives = [self.t["answer"]], self.alt
        return self.rn.contrast(lp)

    def lost(self, sets: list[dict]) -> list[float]:
        """Each set's share of the gap lost in the clean run."""
        self.rn.answer, self.rn.alternatives = [self.t["answer"]], self.alt
        return [(self.c - c) / self.gap for c in self.rn.run(self.ids, sets)["contrast"]]

    def restored(self, sets: list[dict]) -> list[float]:
        """Each set's share of the gap restored in the corrupted run."""
        self.rn.answer, self.rn.alternatives = [self.t["answer"]], self.alt
        return [(c - self.cb) / self.gap for c in self.rn.run(self.bad, sets)["contrast"]]

    def head_from(self, st, L, h, positions):
        return [(L, h, p, st["z"][L][p, h]) for p in positions]


def stage_a1(tx: Text) -> dict:
    """Every head zeroed and patched from the corrupted run at cue1..cue2, and its attention."""
    rn, span = tx.rn, range(tx.p1, tx.q + 1)
    heads = [(L, h) for L in range(rn.L) for h in range(rn.H)]
    zero = tx.lost([{"head": [(L, h, p, None) for p in span]} for L, h in heads])
    patch = tx.lost([{"head": tx.head_from(tx.sb, L, h, span)} for L, h in heads])
    A = tx.st["A"]
    return {"zero": zero, "patch": patch, "induction": [float(A[L][h, tx.q, tx.n1]) for L, h in heads],
            "previous": [float(A[L][h, tx.n1, tx.p1]) for L, h in heads]}


def stage_a2(tx: Text, follow: list) -> dict:
    """The followed heads patched at next1 alone and at cue2 alone, and unable to see next1 from cue2 or cue1 from next1."""
    sets = []
    for L, h in follow:
        sets += [{"head": tx.head_from(tx.sb, L, h, [tx.n1])}, {"head": tx.head_from(tx.sb, L, h, [tx.q])},
                 {"knock": [(L, h, tx.q, tx.n1)]}, {"knock": [(L, h, tx.n1, tx.p1)]}]
    sh = tx.lost(sets)
    keys = ("at_next1", "at_cue2", "knock_cue2_next1", "knock_next1_cue1")
    return {f"{L}.{h}": dict(zip(keys, sh[4 * i:4 * i + 4])) for i, (L, h) in enumerate(follow)}


def stage_a3(tx: Text, roles: dict) -> dict:
    """Each role head's route story (head_routes.route_story) by zeroing and by patching from the corrupted run:
    induction heads read next1 at cue2, previous-token heads read cue1 at next1."""
    out = {}
    for role, hs in roles.items():
        q, p = (tx.q, tx.n1) if role == "induction" else (tx.n1, tx.p1)
        for L, h in hs:
            out[f"{L}.{h}"] = {"role": role, **{mode: head_routes.route_story(tx.rn, tx.st, tx.lost, L, h, q, p, WRITERS, READERS,
                                                                              other=None if mode == "zero" else tx.sb)
                                               for mode in ("zero", "patch")}}
    return out


def regimes(rn, txs, L: int, j: int, pos: str) -> str:
    """The neuron's most common SwiGLU regime over the texts (signs of act(gate) and of up at its position)."""
    from collections import Counter
    cnt = Counter()
    for tx in txs:
        g, u = rn.gate_up(tx.st, L, tx.t["pos"][pos], j)
        cnt[f"{'+' if g >= 0 else '-'}{'+' if u >= 0 else '-'}"] += 1
    return cnt.most_common(1)[0][0]


def units(tx: Text, circ: dict, src) -> dict:
    """The circuit's units set to their values in states src: one edit set."""
    s = {"head": [], "neuron": []}
    for hd in circ["heads"]:
        L, h = hd["layer"], hd["head"]
        s["head"] += [(L, h, tx.t["pos"][p], src["z"][L][tx.t["pos"][p], h]) for p in hd["positions"]]
    for e in circ.get("edges", []):  # (heads that write an induction head's key: their output at that position)
        u = e["src"].get("head")
        if u is not None and "position" in u:
            s["head"].append((u["layer"], u["head"], tx.t["pos"][u["position"]], src["z"][u["layer"]][tx.t["pos"][u["position"]], u["head"]]))
    for nr in circ["neurons"]:
        p = tx.t["pos"][nr["position"]]
        s["neuron"].append((nr["layer"], p, nr["neuron"], float(src["h"][nr["layer"]][p, nr["neuron"]])))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--dev", type=int, default=20)
    a = ap.parse_args()
    import transformers
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        torch.set_num_threads(8)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(dev).eval()
    model.requires_grad_(False)
    rn = head_routes.Runner(model)
    texts = make_texts(tok, a.n)
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)  # noqa: E731
    txs, a1 = [], {}
    for t in texts:
        tx = Text(rn, t)
        t.update(contrast=tx.c, corrupt_contrast=tx.cb, gap=tx.gap, top=tx.top, alternatives=tx.alt,
                 usable=tx.top == t["answer"] and tx.gap >= GAP_MIN)
        if not t["usable"]:
            log(f"{t['id']}: not usable (top {tok.decode([tx.top])!r}, gap {tx.gap:.2f})")
            continue
        a1[t["id"]] = stage_a1(tx)
        txs.append(tx)
        log(f"{t['id']}: gap {tx.gap:.2f} bits; A1 done")
    (a.out / "texts.json").write_text(json.dumps(texts))
    ids = [tx.t["id"] for tx in txs]
    dev_ids = ids[:a.dev]
    heads = [(L, h) for L in range(rn.L) for h in range(rn.H)]
    mean = lambda key, i: sum(a1[t][key][i] for t in dev_ids) / len(dev_ids)  # noqa: E731
    order = sorted(range(len(heads)), key=lambda i: -mean("patch", i))
    follow = [heads[i] for i in order[:HEADS]]
    a2 = {tx.t["id"]: stage_a2(tx, follow) for tx in txs}
    log(f"followed heads {follow[:8]}...: A2 done")
    dmean = lambda fn: sum(fn(t) for t in dev_ids) / len(dev_ids)  # noqa: E731
    roles = {"induction": [], "previous": []}
    for i in order[:HEADS]:  # (roles by measurement on the dev texts)
        L, h = heads[i]
        k = f"{L}.{h}"
        if mean("induction", i) >= ATT and dmean(lambda t: a2[t][k]["at_cue2"]) >= MATTERS / 2:
            roles["induction"].append((L, h))
        if mean("previous", i) >= ATT and dmean(lambda t: a2[t][k]["at_next1"]) >= MATTERS / 2:
            roles["previous"].append((L, h))
    roles = {r: v[:3] for r, v in roles.items()}
    log(f"roles {roles}")
    a3 = {}
    for tx in txs:
        a3[tx.t["id"]] = stage_a3(tx, roles)
        log(f"{tx.t['id']}: A3 done")
    (a.out / "a.json").write_text(json.dumps({"a1": a1, "a2": a2, "a3": a3, "heads": heads, "follow": follow, "roles": roles}))

    # B2: the circuit from the dev texts (patching, the finder's semantics), each unit's share of all texts it matters on
    def matters(fn, pool):  # the share of texts in pool whose measured share fn(t) reaches MATTERS
        return sum(1 for t in pool if fn(t) is not None and fn(t) >= MATTERS) / max(len(pool), 1)
    circ_heads, circ_neurons, edges, seen = [], [], [], {}
    for role, hs in roles.items():
        pos, key, knock = ("cue2", "at_cue2", "knock_cue2_next1") if role == "induction" else ("next1", "at_next1", "knock_next1_cue1")
        for L, h in hs:
            k = f"{L}.{h}"
            fn = lambda t, k=k, key=key: a2[t][k][key]  # noqa: E731
            circ_heads.append({"layer": L, "head": h, "role": role, "positions": [pos], "texts_share": matters(fn, ids),
                               "mean_lost_patched": sum(fn(t) for t in ids) / len(ids),
                               "mean_lost_unable_to_see": sum(a2[t][k][knock] for t in ids) / len(ids),
                               "mean_attention": sum(a1[t][role][heads.index((L, h))] for t in ids) / len(ids)})
    for k in a3[ids[0]]:
        c, h = map(int, k.split("."))
        role = a3[ids[0]][k]["role"]
        for side in ("key", "query", "readers"):
            pos = {"induction": {"key": "next1", "query": "cue2", "readers": "cue2"},
                   "previous": {"key": "cue1", "query": "next1", "readers": "next1"}}[role][side]
            per = {}  # unit -> {mode: {text: measurement}}
            for t in ids:
                for mode in ("zero", "patch"):
                    for e in a3[t][k][mode][side]:
                        per.setdefault(tuple(e["unit"][:3]), {}).setdefault(mode, {})[t] = e
            for u, by in per.items():
                def fn(t, mode="patch", by=by):
                    e = by.get(mode, {}).get(t)
                    return None if e is None else e.get("lost", e.get("cut_lost"))
                if matters(fn, dev_ids) < SHARE_DEV:
                    continue
                if u[0] == "neuron" and (u[1], u[2], pos) not in seen:
                    seen[(u[1], u[2], pos)] = len(circ_neurons)
                    vals = [e["unit"][-1] for m in by.values() for e in m.values()]
                    circ_neurons.append({"layer": u[1], "neuron": u[2], "position": pos, "pole": 1 if sum(vals) >= 0 else -1,
                                         "regime": regimes(rn, txs, u[1], u[2], pos), "label": None})
                head = {"layer": c, "head": h, "role": role}
                unit = {"head": {"layer": u[1], "head": u[2], "position": pos}} if u[0] == "head" else \
                    {"neuron": {"layer": u[1], "neuron": u[2], "position": pos}}
                src, dst = (unit, {"head": head, "input": side}) if side != "readers" else ({"head": head}, unit)
                ps = by.get("patch", {})
                edges.append({"src": src, "dst": dst, "kind": {"key": "writes the key", "query": "writes the query",
                                                              "readers": "reads the head's output"}[side],
                              "texts_share": matters(fn, ids),
                              "mean_lost_patched": sum(fn(t) or 0.0 for t in ids) / len(ids),
                              "mean_lost_zeroed": sum(fn(t, "zero") or 0.0 for t in ids) / len(ids),
                              "mean_attention_change_patched": sum(e["attention_change"] for e in ps.values()) / len(ids)
                              if side != "readers" else None,
                              "mean_value_change_patched": sum(e["value_change"] for e in ps.values()) / len(ids)
                              if side == "readers" else None})
    circ = {"heads": circ_heads, "neurons": circ_neurons, "edges": edges}
    # C: per text necessity, sufficiency, swaps
    scores = {}
    groups = {}
    for tx in txs:
        groups.setdefault((tx.t["template"], len(tx.ids), tx.p1), []).append(tx)
    for tx in txs:
        nec = tx.lost([units(tx, circ, tx.sb)])[0]
        suf = tx.restored([units(tx, circ, tx.st)])[0]
        mates = [o for o in groups[(tx.t["template"], len(tx.ids), tx.p1)] if o is not tx and o.t["answer"] != tx.t["answer"]]
        swap = None
        if mates:
            o = mates[0]
            rn.answer, rn.alternatives = [o.t["answer"]], [tx.t["answer"]]
            c0 = rn.run(tx.ids, [{}])["contrast"][0]
            c1 = rn.run(tx.ids, [units(tx, circ, o.st)])["contrast"][0]
            c2 = rn.run(o.ids, [{}])["contrast"][0]
            swap = {"with": o.t["id"], "share": (c1 - c0) / (c2 - c0) if c2 != c0 else None}
        scores[tx.t["id"]] = {"necessity_lost": nec, "sufficiency_restored": suf, "swap": swap, "dev": tx.t["id"] in dev_ids}
        log(f"{tx.t['id']}: necessity {nec:.2f}, sufficiency {suf:.2f}, swap {swap}")
    out = {"model": a.model, "dtype": a.dtype, "texts": [{k: t[k] for k in ("id", "template", "kind", "text", "cue", "word",
                                                                           "replacement", "pos", "answer", "contrast",
                                                                           "corrupt_contrast", "gap", "usable")} for t in texts],
           "dev": dev_ids, "heads": circ_heads, "neurons": circ_neurons, "edges": edges, "scores": scores,
           "roles": {r: [f"{L}.{h}" for L, h in v] for r, v in roles.items()},
           "attention": {f"{L}.{h}": {"induction": sum(a1[t]["induction"][i] for t in ids) / len(ids),
                                      "previous": sum(a1[t]["previous"][i] for t in ids) / len(ids),
                                      "patch_lost": sum(a1[t]["patch"][i] for t in ids) / len(ids),
                                      "zero_lost": sum(a1[t]["zero"][i] for t in ids) / len(ids)}
                         for i, (L, h) in enumerate(heads) if (L, h) in follow},
           "settings": {"HEADS": HEADS, "ATT": ATT, "WRITERS": WRITERS, "READERS": READERS, "MATTERS": MATTERS,
                        "SHARE_DEV": SHARE_DEV, "GAP_MIN": GAP_MIN}, "seconds": round(time.time() - t0, 1)}
    (a.out / "circuit.json").write_text(json.dumps(out))
    log(f"circuit: {len(circ_heads)} heads, {len(circ_neurons)} neurons, {len(edges)} edges -> {a.out / 'circuit.json'}")


if __name__ == "__main__":
    main()
