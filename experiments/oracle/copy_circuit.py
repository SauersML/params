"""V2 of the circuit finder's validation (the lead's 10-15 brief): the copy (induction) circuit on Qwen3 in heads and
neurons, every number a measured intervention (head_routes.Runner). A copy text repeats a cue token: "... met Daniel
Kowalczyk at ... Daniel" -> " Kowalczyk" (or an adjective and its noun); its corrupted twin replaces the first cue with
another token of the same length, so nothing earlier matches the final cue. Positions: cue1 (the first cue), next1 (the
token after it: the answer's first token), cue2 (the last position, the second cue). Contrast: the word contrast
(answer_contrast.py): log2 of the summed p of every token starting the copied word against that of the tokens starting
other words (the corrupted run's first); a text is usable when the clean run's top token starts the copied word and the
gap (clean - corrupted) is at least GAP_MIN bits.

Stages (all texts; the order and the circuit chosen on the dev texts, everything reported on every text):
 A1 heads: each head zeroed at positions cue1.. and each patched there from the corrupted run (shares of the gap lost);
    its attention cue2 -> next1 (induction) and next1 -> cue1 (previous token) in the clean run;
 A2 candidates: heads whose dev mean attention cue2 -> next1 is at least ATT (at cue2) or next1 -> cue1 (at next1), and
    the HEADS largest by patched share (at both): each patched alone at its position and unable to see its source
    alone; every candidate of a role patched together and unable to see together;
 A3 the size curve: the candidates in the dev texts' order of their own patched share, the first k patched from the
    corrupted run into the clean one (lost) and only they from the clean run into the corrupted one (restored), k on a
    log scale; the circuit is the smallest k whose dev means both reach PASS (every ordered unit when none does);
 A4 the circuit's first STORIES induction and previous-token heads' route stories (head_routes.route_story, zeroing and
    patching: knockout, the neurons and heads setting key, query and value, readers, and each side's group), and K
    composition: every circuit previous-token head's output at next1 out of every circuit induction head's key at once;
 B  each circuit head and each story unit with the share of texts it matters on (its measured share at least MATTERS of
    that text's circuit effect; story units on at least SHARE_DEV of the dev texts);
 C  per text: the circuit's heads patched from the corrupted run into the clean one (necessity), only they from the
    clean run into the corrupted one (sufficiency), their outputs from another text of the same template and length
    (swapped: the share of the way the answer moves to that text's answer), and the size curve.
On Qwen3-8B no single head carries 10% of the gap (the largest 7%; their patched shares sum to 2.5): the circuit is a
group, found by the curve.

  python experiments/oracle/copy_circuit.py OUT [--model Qwen/Qwen3-8B] [--dtype float32] [--n 60] [--dev 20]
writes OUT/texts.json, OUT/a.json (stage A per text), OUT/circuit.json.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
import answer_contrast  # noqa: E402
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
    ("met", "name", "Yesterday I met {a} {b} at the train station. We talked for a while about the weather and the delays."
                    " When I got home, I wrote down the full name of my new friend: {a}"),
    ("report", "name", "The report was written by {a} {b}, a researcher at the city museum. It describes three old maps"
                       " and a letter found in the archive. According to the museum, {a}"),
    ("chair", "name", "At the meeting, the chair introduced {a} {b} as the new treasurer. Several members asked questions"
                      " about the budget and the schedule. In the end, {a}"),
    ("bridge", "phrase", "She found a {a} {b} under the old bridge and took it home. Her brother wanted to know where"
                         " she had found the {a}"),
    ("shop", "phrase", "The shop on the corner sells a {a} {b} next to the fresh bread. On Sunday I went there and asked"
                       " the owner about the {a}"),
]
HEADS = 24          # heads of largest patched share added to the attention-picked candidates (A2)
PASS = 0.7          # the circuit: the smallest prefix of the measured order losing and restoring this share on the dev texts
STORIES = 3         # route stories per role (the circuit's first induction and previous-token heads)
GROUP_ATT = (0.3, 0.1)  # A5: role groups by dev mean attention at least these
ATT = 0.1           # a candidate induction (previous-token) head attends cue2 -> next1 (next1 -> cue1) this much (dev mean)
WRITERS = 16        # key / query writers measured in full runs per induction head (by exact attention change)
READERS = 32        # readers measured in full runs per induction head (by exact value change)
MATTERS = 0.1       # a unit matters on a text when its measured share of the gap is at least this
SHARE_DEV = 0.3     # the circuit: units that matter on at least this share of the dev texts
GAP_MIN = 2.0       # bits: a usable text's clean - corrupted contrast
STEPS = 6           # tokens each candidate is continued to read the word it starts (surnames run to several tokens)


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

    def __init__(self, rn: head_routes.Runner, t: dict, tok):
        self.rn, self.t = rn, t
        self.ids, self.bad = t["ids"], t["corrupt_ids"]
        self.p1, self.n1, self.q = t["pos"]["cue1"], t["pos"]["next1"], t["pos"]["cue2"]
        rn.answer, rn.alternatives = [t["answer"]], [0]
        clean = rn.run(self.ids, [{}], capture=True)
        corrupt = rn.run(self.bad, [{}], capture=True)
        lc, lb = clean["lp"][0], corrupt["lp"][0]
        with torch.no_grad():  # (the word contrast: every token starting the copied word against other words' tokens)
            sets = answer_contrast.token_sets(rn.core, tok, self.ids, lc, b=int(lb.argmax()), steps=STEPS)
        self.ans, self.alt, self.words = sets["answer"], sets["alternatives"], sets
        cpu = lambda st: {k: [t.cpu() for t in v] if isinstance(v, list) else v.cpu() for k, v in st.items()}  # noqa: E731
        self.st, self.sb = cpu(clean["states"]), cpu(corrupt["states"])  # (every text's states on the host: 8B fills a card)
        self.top = int(lc.argmax())
        self.word_ok = sets["answer_word"] == answer_contrast.word(" " + t["word"])
        rn.answer, rn.alternatives = self.ans, self.alt  # (uncut contrasts as every cut run measures them: float32 logits
        self.c = rn.run(self.ids, [{}])["contrast"][0]  # of the two token sets, at the cut runs' batch size)
        self.cb = rn.run(self.bad, [{}])["contrast"][0]
        self.gap = self.c - self.cb

    def lost(self, sets: list[dict]) -> list[float]:
        """Each set's share of the gap lost in the clean run."""
        self.rn.answer, self.rn.alternatives = self.ans, self.alt
        return [(self.c - c) / self.gap for c in self.rn.run(self.ids, sets)["contrast"]]

    def restored(self, sets: list[dict]) -> list[float]:
        """Each set's share of the gap restored in the corrupted run."""
        self.rn.answer, self.rn.alternatives = self.ans, self.alt
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


def head_set(tx: Text, hs: list, src) -> dict:
    """Heads at positions [(layer, head, position name)] set to their outputs in states src: one edit set."""
    return {"head": [(L, h, tx.t["pos"][pn], src["z"][L][tx.t["pos"][pn], h]) for L, h, pn in hs]}


def stage_a2(tx: Text, cand: list) -> dict:
    """Each candidate head at its role position patched alone, unable to see its source alone (knock), and every
    candidate of each role patched together and unable to see together."""
    route = {"cue2": (tx.q, tx.n1), "next1": (tx.n1, tx.p1)}
    single = tx.lost([head_set(tx, [u], tx.sb) for u in cand])
    knock = tx.lost([{"knock": [(L, h, *route[pn])]} for L, h, pn in cand])
    groups = {}
    for pn in ("cue2", "next1"):
        g = [u for u in cand if u[2] == pn]
        if g:
            lost, kn = tx.lost([head_set(tx, g, tx.sb), {"knock": [(L, h, *route[pn]) for L, h, _ in g]}])
            groups[pn] = {"lost": lost, "knock": kn, "heads": len(g)}
    return {"single": single, "knock": knock, "groups": groups}


def stage_curve(tx: Text, order: list, ks: list) -> dict:
    """The size curve: the first k units of order patched from the corrupted run into the clean one (lost) and only they
    from the clean run into the corrupted one (restored), per k."""
    return {"k": ks, "lost": tx.lost([head_set(tx, order[:k], tx.sb) for k in ks]),
            "restored": tx.restored([head_set(tx, order[:k], tx.st) for k in ks])}


def stage_routes(tx: Text, ind: list, prev: list, stories: int) -> dict:
    """The first stories induction heads' (at cue2) and previous-token heads' (at next1) route stories
    (head_routes.route_story, zeroing and patching), and every circuit previous-token head's output at next1 taken out of every
    induction head's key there together (K composition: each induction head's attention cue2 -> next1 after, exact, and
    the share lost with every such cut in one run, patched and zeroed)."""
    out = {"stories": {}}
    for role, hs in (("induction", ind[:stories]), ("previous", prev[:stories])):
        q, p = (tx.q, tx.n1) if role == "induction" else (tx.n1, tx.p1)
        for L, h in hs:
            out["stories"][f"{L}.{h}"] = {"role": role, **{m: head_routes.route_story(tx.rn, tx.st, tx.lost, L, h, q, p, WRITERS,
                                                                                      READERS, other=None if m == "zero" else tx.sb)
                                                          for m in ("zero", "patch")}}
    rn, st = tx.rn, tx.st
    for mode in ("zero", "patch"):
        if not (ind and prev):
            break
        writes = []
        for L, g in prev:
            w = rn.head_write(st, L, g, tx.n1)
            writes.append(w if mode == "zero" else w - rn.head_write(tx.sb, L, g, tx.n1).to(w.device))
        edits, change = [], {}
        for L, h in ind:
            ws = torch.stack([w for (Lp, _), w in zip(prev, writes) if Lp < L])
            if not len(ws):
                continue
            a, dk = rn.attention_after(st, L, h, tx.q, tx.n1, ws.sum(0)[None], side="key", delta_out=True)
            edits.append((L, h, tx.n1, dk[0]))
            change[f"{L}.{h}"] = float(a[0]) - float(st["A"][L][h, tx.q, tx.n1])
        out[f"k_composition_{mode}"] = {"lost": tx.lost([{"dk": edits}])[0] if edits else None, "attention_change": change}
    return out


def stage_groups(tx: Text, g_ind: list, g_prev: list, order_ind: list, order_prev: list) -> dict:
    """Role groups by measured attention: the induction group (at cue2) and the previous-token group (at next1), each
    patched together (lost; restored) and unable to see its source together, both together; K composition as a group
    (every previous-token head's output at next1 out of every later induction head's key there, in one run, zeroing and
    patching; the induction heads' mean attention cue2 -> next1 after, exact); and each group's own size curve in its
    measured order."""
    rn, st = tx.rn, tx.st
    ind = [(L, h, "cue2") for L, h in g_ind]
    prev = [(L, h, "next1") for L, h in g_prev]
    sets = [head_set(tx, ind, tx.sb), head_set(tx, prev, tx.sb), head_set(tx, ind + prev, tx.sb),
            {"knock": [(L, h, tx.q, tx.n1) for L, h in g_ind]}, {"knock": [(L, h, tx.n1, tx.p1) for L, h in g_prev]}]
    lost = tx.lost(sets)
    rest = tx.restored([head_set(tx, ind, tx.st), head_set(tx, prev, tx.st), head_set(tx, ind + prev, tx.st)])
    out = {"sizes": {"induction": len(g_ind), "previous": len(g_prev)},
           "lost": dict(zip(("induction", "previous", "both", "induction unable to see", "previous unable to see"), lost)),
           "restored": dict(zip(("induction", "previous", "both"), rest)), "k_composition": {}}
    for mode in ("zero", "patch"):
        writes = []
        for L, g in g_prev:
            w = rn.head_write(st, L, g, tx.n1)
            writes.append((L, w if mode == "zero" else w - rn.head_write(tx.sb, L, g, tx.n1).to(w.device)))
        edits, change = [], []
        for L, h in g_ind:
            ws = [w for Lp, w in writes if Lp < L]
            if not ws:
                continue
            a, dk = rn.attention_after(st, L, h, tx.q, tx.n1, torch.stack(ws).sum(0)[None], side="key", delta_out=True)
            edits.append((L, h, tx.n1, dk[0]))
            change.append(float(a[0]) - float(st["A"][L][h, tx.q, tx.n1]))
        out["k_composition"][mode] = {"lost": tx.lost([{"dk": edits}])[0] if edits else None, "induction_heads": len(edits),
                                      "mean_attention_change": sum(change) / len(change) if change else None}
    for name, order, pn in (("induction", order_ind, "cue2"), ("previous", order_prev, "next1")):
        ks = sorted({min(len(order), int(round(2 ** (e / 2)))) for e in range(0, int(2 * math.log2(max(len(order), 1))) + 2)}
                    | {len(order)}) if order else []
        units = [(L, h, pn) for L, h in order]
        out[f"curve_{name}"] = {"k": ks, "lost": tx.lost([head_set(tx, units[:k], tx.sb) for k in ks]),
                                "restored": tx.restored([head_set(tx, units[:k], tx.st) for k in ks])} if ks else None
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


def attach_atlas(path: Path, root: Path, k: int = 8):
    """Each circuit neuron's atlas record in circuit.json: per pole its largest activations' tokens (with a few tokens
    before) and values, its firing rates, and the tokens it pushes up and down most (direct logit effect)."""
    import atlas
    at = atlas.Atlas(root)
    c = json.loads(path.read_text())
    for n in c["neurons"]:
        r = at.neuron(n["layer"], n["neuron"], k=k)
        n["atlas"] = {p: [{"token": e["token"], "before": e["before"][-40:], "value": round(e["value"], 2), "regime": e["regime"]}
                          for e in r[p]] for p in ("+", "-")}
        n["atlas"].update(max=r["max"], rates=r["rates"], logits=r.get("logits"))
    path.write_text(json.dumps(c))
    print(f"atlas examples for {len(c['neurons'])} neurons -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--dev", type=int, default=20)
    ap.add_argument("--atlas", type=Path, default=None, help="only add each circuit neuron's atlas examples to "
                    "OUT/circuit.json (the neuron atlas's root for the model, experiments/oracle/atlas.py)")
    a = ap.parse_args()
    if a.atlas:
        return attach_atlas(a.out / "circuit.json", a.atlas)
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
        tx = Text(rn, t, tok)
        t.update(contrast=tx.c, corrupt_contrast=tx.cb, gap=tx.gap, top=tx.top, answer_tokens=tx.ans, alternatives=tx.alt,
                 answer_word=tx.words["answer_word"], alternative_words=tx.words["alternative_words"],
                 usable=tx.word_ok and tx.gap >= GAP_MIN)
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
    # A2: candidates by measured attention (dev means) and the heads largest by patched share, at their role positions
    by_patch = sorted(range(len(heads)), key=lambda i: -mean("patch", i))[:HEADS]
    cand = sorted({(*heads[i], "cue2") for i in range(len(heads)) if mean("induction", i) >= ATT}
                  | {(*heads[i], "next1") for i in range(len(heads)) if mean("previous", i) >= ATT}
                  | {(*heads[i], pn) for i in by_patch for pn in ("cue2", "next1")})
    a2 = {}
    for tx in txs:
        a2[tx.t["id"]] = stage_a2(tx, cand)
    log(f"A2: {len(cand)} candidate heads at their role positions")
    # B1: units in the dev texts' measured order (patched alone); A3: the size curve on every text
    dmean = lambda fn, pool=dev_ids: sum(fn(t) for t in pool) / len(pool)  # noqa: E731
    rank = sorted(range(len(cand)), key=lambda u: -dmean(lambda t: a2[t]["single"][u]))
    order = [cand[u] for u in rank if dmean(lambda t: a2[t]["single"][u]) > 0]
    ks = sorted({min(len(order), int(round(2 ** (e / 2)))) for e in range(0, int(2 * math.log2(max(len(order), 1))) + 2)} | {len(order)})
    a3 = {}
    for tx in txs:
        a3[tx.t["id"]] = stage_curve(tx, order, ks)
    mean_curve = {key: [dmean(lambda t: a3[t][key][n]) for n in range(len(ks))] for key in ("lost", "restored")}
    k_pass = next((k for k, lo, re in zip(ks, mean_curve["lost"], mean_curve["restored"]) if lo >= PASS and re >= PASS), None)
    circuit = order[:k_pass or len(order)]
    log(f"A3: size curve {list(zip(ks, [round(x, 2) for x in mean_curve['lost']], [round(x, 2) for x in mean_curve['restored']]))};"
        f" circuit {len(circuit)} heads{'' if k_pass else ' (none passes: every ordered unit)'}")
    ind = [(L, h) for L, h, pn in circuit if pn == "cue2"]
    prev = [(L, h) for L, h, pn in circuit if pn == "next1"]
    a4 = {}
    for tx in txs:
        a4[tx.t["id"]] = stage_routes(tx, ind, prev, STORIES)
        log(f"{tx.t['id']}: routes done")
    a5, attention_groups = {}, {}  # A5: role groups by measured attention alone, at each threshold
    for th in GROUP_ATT:
        g_ind = [heads[i] for i in range(len(heads)) if mean("induction", i) >= th]
        g_prev = [heads[i] for i in range(len(heads)) if mean("previous", i) >= th]
        rank_of = lambda g, pn: sorted(g, key=lambda u: -dmean(lambda t, u=u: a2[t]["single"][cand.index((*u, pn))])  # noqa: E731
                                       if (*u, pn) in cand else 0.0)
        oi, op = rank_of(g_ind, "cue2"), rank_of(g_prev, "next1")
        for tx in txs:
            a5.setdefault(tx.t["id"], {})[str(th)] = stage_groups(tx, g_ind, g_prev, oi, op)
        res = {}
        for key in ("lost", "restored"):
            for k in a5[ids[0]][str(th)][key]:
                res[f"{key} {k}"] = dmean(lambda t, key=key, k=k: a5[t][str(th)][key][k], ids)
        for mode in ("zero", "patch"):
            res[f"k composition {mode}"] = dmean(lambda t, mode=mode: a5[t][str(th)]["k_composition"][mode]["lost"] or 0.0, ids)
            ch = [a5[t][str(th)]["k_composition"][mode]["mean_attention_change"] for t in ids]
            ch = [x for x in ch if x is not None]
            res[f"k composition {mode} attention change"] = sum(ch) / len(ch) if ch else None
        for name in ("induction", "previous"):
            cv = [a5[t][str(th)][f"curve_{name}"] for t in ids if a5[t][str(th)][f"curve_{name}"]]
            if cv:
                res[f"curve {name}"] = {"k": cv[0]["k"], "lost_median": [statistics.median(c["lost"][n] for c in cv) for n in range(len(cv[0]["k"]))],
                                        "restored_median": [statistics.median(c["restored"][n] for c in cv) for n in range(len(cv[0]["k"]))]}
        attention_groups[str(th)] = {"induction": [f"{L}.{h}" for L, h in oi], "previous": [f"{L}.{h}" for L, h in op], **res}
        log(f"A5 groups at attention {th}: {len(g_ind)} induction, {len(g_prev)} previous-token heads: "
            f"{ {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if not k.startswith('curve')} }")
    (a.out / "a.json").write_text(json.dumps({"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "heads": heads, "cand": cand,
                                              "order": order, "ks": ks}))

    # B2: the circuit's heads and the neurons of their route stories, each with the share of texts it matters on (at
    # least MATTERS of that text's circuit effect)
    effect = {t: max(a3[t]["lost"][ks.index(len(circuit))] if len(circuit) in ks else 0.0, 1e-9) for t in ids}
    def matters(fn, pool):
        vals = [(fn(t), effect[t]) for t in pool]
        return sum(1 for v, e in vals if v is not None and v >= MATTERS * e) / max(len(pool), 1)
    circ_heads = []
    for L, h, pn in circuit:
        u = cand.index((L, h, pn))
        fn = lambda t, u=u: a2[t]["single"][u]  # noqa: E731
        att = dmean(lambda t, i=heads.index((L, h)): a1[t]["induction" if pn == "cue2" else "previous"][i], ids)
        role = ("induction" if pn == "cue2" else "previous token") if att >= ATT else f"other head at {pn}"
        circ_heads.append({"layer": L, "head": h, "role": role, "position": pn, "reads": "next1" if pn == "cue2" else "cue1",
                           "mean_lost_patched": dmean(fn, ids), "mean_lost_unable_to_see": dmean(lambda t, u=u: a2[t]["knock"][u], ids),
                           "mean_attention": dmean(lambda t, i=heads.index((L, h)): a1[t]["induction" if pn == "cue2" else "previous"][i], ids),
                           "texts_share": matters(fn, ids)})
    circ_neurons, edges, seen = [], [], {}
    for k in (a4[ids[0]]["stories"] if ids else {}):
        c, h = map(int, k.split("."))
        role = a4[ids[0]]["stories"][k]["role"]
        for side in ("key", "query", "value", "readers"):
            pos = {"induction": {"key": "next1", "query": "cue2", "value": "next1", "readers": "cue2"},
                   "previous": {"key": "cue1", "query": "next1", "value": "cue1", "readers": "next1"}}[role][side]
            per = {}
            for t in ids:
                for mode in ("patch", "zero"):
                    for e in a4[t]["stories"][k][mode][side]:
                        per.setdefault(tuple(e["unit"][:3]), {}).setdefault(mode, {})[t] = e
            for u, by in per.items():
                def fn(t, mode="patch", by=by):
                    e = by.get(mode, {}).get(t)
                    return None if e is None else e.get("lost", e.get("cut_lost"))
                if matters(fn, dev_ids) < SHARE_DEV:
                    continue
                unit = {"head": {"layer": u[1], "head": u[2], "position": pos}} if u[0] == "head" else \
                    {"neuron": {"layer": u[1], "neuron": u[2], "position": pos}}
                if u[0] == "neuron" and (u[1], u[2], pos) not in seen:
                    seen[(u[1], u[2], pos)] = len(circ_neurons)
                    vals = [e["unit"][-1] for m in by.values() for e in m.values()]
                    circ_neurons.append({"layer": u[1], "neuron": u[2], "position": pos, "pole": 1 if sum(vals) >= 0 else -1,
                                         "regime": regimes(rn, txs, u[1], u[2], pos), "label": None})
                head = {"layer": c, "head": h, "role": role}
                src, dst = (unit, {"head": head, "input": side}) if side != "readers" else ({"head": head}, unit)
                edges.append({"src": src, "dst": dst, "kind": {"key": "writes the key", "query": "writes the query",
                                                              "value": "is moved (value)", "readers": "reads the head's output"}[side],
                              "texts_share": matters(fn, ids), "mean_lost_patched": dmean(lambda t: fn(t) or 0.0, ids),
                              "mean_lost_zeroed": dmean(lambda t: fn(t, "zero") or 0.0, ids)})
    groups_out = {}  # per story head: each side's picked units cut together (mean over texts)
    for k in (a4[ids[0]]["stories"] if ids else {}):
        g = {}
        for side in ("key", "query", "value"):
            for kind in ("neurons", "heads"):
                for mode in ("patch", "zero"):
                    vals = [a4[t]["stories"][k][mode][f"{side}_group"].get(kind, {}).get("lost") for t in ids]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        g[f"{side} {kind} {mode}"] = sum(vals) / len(vals)
        for mode in ("patch", "zero"):
            g[f"readers {mode}"] = dmean(lambda t: a4[t]["stories"][k][mode]["readers_group"]["cut_lost"], ids)
        groups_out[k] = g
    kcomp = {mode: {"mean_lost": dmean(lambda t: a4[t].get(f"k_composition_{mode}", {}).get("lost") or 0.0, ids)}
             for mode in ("patch", "zero")}
    circ = {"heads": [{"layer": x["layer"], "head": x["head"], "positions": [x["position"]]} for x in circ_heads],
            "neurons": []}  # (C scores the circuit's heads; neurons are reported through the route stories)
    # C: per text necessity, sufficiency, swaps (the circuit's heads)
    scores, groups = {}, {}
    for tx in txs:
        groups.setdefault((tx.t["template"], len(tx.ids), tx.p1), []).append(tx)
    for tx in txs:
        nec = tx.lost([units(tx, circ, tx.sb)])[0]
        suf = tx.restored([units(tx, circ, tx.st)])[0]
        mates = [o for o in groups[(tx.t["template"], len(tx.ids), tx.p1)] if o is not tx and o.t["answer"] != tx.t["answer"]]
        swap = None
        if mates:
            o = mates[0]
            rn.answer, rn.alternatives = o.ans, tx.ans  # (the other text's word against this one's)
            c0 = rn.run(tx.ids, [{}])["contrast"][0]
            c1 = rn.run(tx.ids, [units(tx, circ, o.st)])["contrast"][0]
            c2 = rn.run(o.ids, [{}])["contrast"][0]
            swap = {"with": o.t["id"], "share": (c1 - c0) / (c2 - c0) if c2 != c0 else None}
        scores[tx.t["id"]] = {"necessity_lost": nec, "sufficiency_restored": suf, "swap": swap, "dev": tx.t["id"] in dev_ids,
                              "curve": a3[tx.t["id"]]}
        log(f"{tx.t['id']}: necessity {nec:.2f}, sufficiency {suf:.2f}, swap {swap}")
    out = {"model": a.model, "dtype": a.dtype, "texts": [{k: t.get(k) for k in ("id", "template", "kind", "text", "cue", "word",
                                                                               "replacement", "pos", "answer", "answer_tokens",
                                                                               "answer_word", "alternatives", "alternative_words",
                                                                               "contrast", "corrupt_contrast", "gap", "usable")}
                                                         for t in texts],
           "dev": dev_ids, "heads": circ_heads, "neurons": circ_neurons, "edges": edges, "groups": groups_out,
           "k_composition": kcomp, "attention_groups": attention_groups, "scores": scores, "curve": {"k": ks, **mean_curve, "order": [f"{L}.{h}@{pn}" for L, h, pn in order],
                                                              "passes_at": k_pass},
           "role_groups": {pn: {"heads": a2[ids[0]]["groups"].get(pn, {}).get("heads"),
                                "mean_lost_patched": dmean(lambda t, pn=pn: a2[t]["groups"].get(pn, {}).get("lost", 0.0), ids),
                                "mean_lost_unable_to_see": dmean(lambda t, pn=pn: a2[t]["groups"].get(pn, {}).get("knock", 0.0), ids)}
                           for pn in ("cue2", "next1")},
           "attention": {f"{L}.{h}": {"induction": sum(a1[t]["induction"][i] for t in ids) / len(ids),
                                      "previous": sum(a1[t]["previous"][i] for t in ids) / len(ids),
                                      "patch_lost": sum(a1[t]["patch"][i] for t in ids) / len(ids),
                                      "zero_lost": sum(a1[t]["zero"][i] for t in ids) / len(ids)}
                         for i, (L, h) in enumerate(heads) if (L, h) in {(x, y) for x, y, _ in circuit}},
           "settings": {"HEADS": HEADS, "ATT": ATT, "WRITERS": WRITERS, "READERS": READERS, "MATTERS": MATTERS,
                        "SHARE_DEV": SHARE_DEV, "GAP_MIN": GAP_MIN, "PASS": PASS, "STORIES": STORIES}, "seconds": round(time.time() - t0, 1)}
    (a.out / "circuit.json").write_text(json.dumps(out))
    log(f"circuit: {len(circ_heads)} heads, {len(circ_neurons)} neurons, {len(edges)} edges -> {a.out / 'circuit.json'}")


if __name__ == "__main__":
    main()
