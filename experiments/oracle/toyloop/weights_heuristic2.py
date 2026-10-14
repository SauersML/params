"""A marker-aware, weights-only rule for zoo v2 (TOYLOOP lane): is the text's circuit in the weights-only prompt at all?

predict() reads ONLY the rendered TOYLOOP_SECTIONS=weights prompt (render.weight_only_lines + the text's tokens),
parsed as a string; the model is never touched. The rule:

  1. the marker: gate links are a head's Q link, or a neuron's in link, from a layer-0 head with a dominant source
     token ("Q[L0H7:33<t34]"); the detector is the layer-0 head most such links come from, the gate tokens are their
     source tokens, and the text's marker is the first text token among them;
  2. keep the detector and every unit whose gate link from the detector carries that marker;
  3. add the always-on layer-0 units these units read (Q/K/V or in links), and what those read in layer 0;
  4. drop units marked as a cancelling pair ("~X") or writing to nothing ("out[]").

Three refinements, each read from the prompt too (refine=False gives steps 1-4 alone):
  a. a unit gated by another marker is never traced (another module's layer-0 neuron);
  b. a content head (pos flat, every QK pair keyed on one token) whose key token is not in the text is skipped: a
     decoy, which attends to its trigger or else to a zero-valued position;
  c. a link counts from either end: the writer's out[] list as well as the reader's in list (each list shows only
     its top 2-3 links, so a decoy's larger gain can push a real link out of one of them);
  d. a gated neuron reading a positional layer-0 head (pos-d) through one dominant token ("in[L0H2:20<t15]", a
     bigram lookup via the previous token) is kept only if the text has that token d positions before the last.

Edges: tok_marker -> detector -> each gated unit; each traced unit -> the unit that reads it; tok_in -> a kept unit
reading token identity; each gated unit (and any kept unit writing logits) -> out.

The same rule on the model (circuit_from_weights: every link of render.wiring with its source token, no truncation,
no neuron cap, no rounding) bounds what this prompt format gives the rule. circuit_from_weights(select=True) adds what
the prompt lacks: each gated neuron is kept only if it fires on the text, its pre-activation computed from the weights
with layer-0 attention read off the weights (positional heads at their pos-d offset, content heads at the text
position whose token their QK scores highest) -- the per-neuron token keys and bias a render change would show.

  TOYLOOP_SECTIONS=weights python3 -B experiments/oracle/toyloop/evaluate.py weights2 --zoo ~/mpd-data/toy_zoo2/manifest.jsonl --limit 40
  python3 -B experiments/oracle/toyloop/weights_heuristic2.py --limit 40    # prompt vs weights rule, per split x family
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

UNIT = re.compile(r"L(\d+)([HN])(\d+)")
PORT = re.compile(r"(?<![A-Za-z])(Q|K|V|in|out)\[([^\]]*)\]")
ITEM = re.compile(r"^([A-Za-z]\w*?)(?:\.([QKV]))?:([-+0-9.e]+)(?:<(\w+))?$")


@dataclass
class Unit:
    name: str
    ins: list = field(default_factory=list)    # (port Q/K/V/in, writer, gain, source token or None)
    outs: list = field(default_factory=list)   # (reader, gain)
    cancels: str | None = None
    key: str | None = None                     # a content head's key token (pos flat, every QK pair keyed on it)
    offset: int | None = None                  # a positional head's pos-d


def layer(u: str) -> int:
    m = UNIT.fullmatch(u)
    return int(m.group(1)) if m else -1


def is_head(u: str) -> bool:
    m = UNIT.fullmatch(u)
    return bool(m) and m.group(2) == "H"


def parse(prompt: str) -> tuple[dict, list]:
    """({unit: Unit}, the first text's tokens) from a weights-only prompt."""
    units, text, section = {}, [], None
    for line in prompt.splitlines():
        s = line.strip()
        if s in ("Weights:", "Texts:"):
            section = s
            continue
        m = re.match(r"Text 0: (.*?) \(predict at position \d+\)$", s)
        if m:
            text = m.group(1).split()
            continue
        m = re.match(r"(L\d+[HN]\d+) ", s)
        if section != "Weights:" or not m:   # only the weight section (the answer FORMAT's example names units too)
            continue
        u = Unit(m.group(1))
        c = re.search(r" ~(L\d+[HN]\d+)\b", s)
        u.cancels = c.group(1) if c else None
        qk = re.search(r" QK\[([^\]]*)\]", s)
        d = re.search(r" pos-(\d+) ", s)
        u.offset = int(d.group(1)) if d else None
        if " pos flat" in s and qk:
            keys = {x.split(">")[1].split(":")[0] for x in qk.group(1).split(",") if ">" in x}
            u.key = keys.pop() if len(keys) == 1 and "BOS" not in keys else None
        for port, body in PORT.findall(s):
            for it in filter(None, body.split(",")):
                x = ITEM.match(it)
                if not x:
                    continue
                w, sub, g, src = x.groups()
                if port == "out":
                    u.outs.append((w + (f".{sub}" if sub else ""), float(g)))
                else:
                    u.ins.append((port, w, float(g), src))
        units[u.name] = u
    return units, text


def gate_links(units: dict) -> list:
    """(gated unit, writer, source token): a head's Q link or a neuron's in link from a layer-0 head, carrying a token."""
    out = []
    for u in units.values():
        for port, w, _, src in u.ins:
            if src and is_head(w) and layer(w) == 0 and (port == "Q" if is_head(u.name) else port == "in"):
                out.append((u.name, w, src))
    return out


def rule(units: dict, text: list, refine: bool = True) -> tuple[set, set, dict]:
    """(units, edges, info) of the marker-selected circuit; refine adds steps a-c of the module docstring."""
    gl = gate_links(units)
    info = {"detector": None, "marker": None, "gate_tokens": [], "gated": []}
    if not gl:
        return set(), set(), info
    det = Counter(w for _, w, _ in gl).most_common(1)[0][0]
    gtoks = sorted({t for _, w, t in gl if w == det})
    marker = next((t for t in text if t in gtoks), None)
    gate_of = {u: t for u, w, t in gl if w == det}
    dead = lambda n: n not in units or not units[n].outs or units[n].cancels is not None  # noqa: E731
    gated = sorted({u for u, t in gate_of.items() if t == marker and not dead(u)})
    info.update(detector=det, marker=marker, gate_tokens=gtoks, gated=gated)
    if not gated:   # no gated unit for this marker: the detector alone does nothing
        return set(), set(), info
    reads = {u: [w for _, w, _, _ in x.ins] for u, x in units.items()}
    if refine:      # c: a link from either end
        for w, x in units.items():
            for r, _ in x.outs:
                r = r.split(".")[0]
                if r in reads and w not in reads[r]:
                    reads[r].append(w)

    def traced(w):
        if layer(w) != 0 or w == det or dead(w):
            return False
        if refine and gate_of.get(w, marker) != marker:                    # a: another module's gated unit
            return False
        if refine and units[w].key is not None and units[w].key not in text:   # b: a decoy whose trigger is absent
            return False
        return True

    if refine:      # d: a lookup neuron keyed on a token the text does not have at its head's offset
        def off(u):
            for _, w, _, src in units[u].ins:
                o = units[w].offset if w in units else None
                if src and w != det and layer(w) == 0 and o is not None and len(text) > o:
                    if text[-1 - o] != src:
                        return True
            return False
        gated = [u for u in gated if is_head(u) or not off(u)]
        if not gated:
            return set(), set(), info
    keep, edges = {det} | set(gated), {("tok_marker", det)} | {(det, u) for u in gated}
    stack = list(gated)
    while stack:   # trace read links back to the always-on layer-0 units (and what those read in layer 0)
        u = stack.pop()
        for w in reads[u]:
            if w == "tok":
                edges.add(("tok_in", u))
            elif traced(w):
                edges.add((w, u))
                if w not in keep:
                    keep.add(w)
                    stack.append(w)
    for u in keep:
        if any(r == "logits" for r, _ in units[u].outs) or u in gated:
            edges.add((u, "out"))
    return keep, edges, info


def completion(keep: set, edges: set) -> str:
    if not keep:
        return "flowchart LR\n  tokX --> out\nEnglish: No unit carries the prediction."
    return "flowchart LR\n" + "\n".join(f"  {a} --> {b}" for a, b in sorted(edges)) + \
        "\nEnglish: The detector reads the marker and un-gates the units whose gate it feeds; they read always-on layer-0 units."


def predict(toy, prompt: str) -> str:
    """evaluate.py baseline: the rule on the prompt text only (toy is not used)."""
    keep, edges, _ = rule(*parse(prompt))
    return completion(keep, edges)


# ---------------------------------------------------------------- the same rule on the weights (upper bound)

def _head_maps(toy, i, j):
    import math
    import render
    sd, cfg = toy.sd, toy.cfg
    E = sd["wte.weight"].float()
    Wq, Wk, Wv, Wo = render._blocks(sd, cfg, i, j)
    En = render._rms(E, sd[f"h.{i}.rms_1.weight"].float(), render._eps(cfg))
    _, _, hd = render.heads_of(cfg)
    c = En.mean(0)
    pos = render._rope_scores(Wq @ c, Wk @ c, cfg, min(int(cfg.get("n_ctx", 32)) - 1, 15))
    qk = (En @ Wq.T) @ (En @ Wk.T).T / math.sqrt(hd)
    return pos, qk, En, Wv, Wo


def _content_key(toy, i, j, k: int = 3):
    """As the prompt shows it: pos flat and the top k QK pairs all keyed on one token (not BOS)."""
    pos, qk, *_ = _head_maps(toy, i, j)
    if float(pos.max() - pos.min()) > 1.0 or float(qk.std()) <= 0.5:
        return None
    keys = {int(a) % qk.shape[1] for a in qk.flatten().topk(k).indices}
    return toy.vocab[keys.pop()] if len(keys) == 1 and 0 not in keys else None


def _offset(toy, i, j):
    pos, *_ = _head_maps(toy, i, j)
    return int(pos.argmax()) if float(pos.max() - pos.min()) > 1.0 else None


def units_from_weights(toy) -> tuple[dict, list]:
    """The prompt's structure, read off the model with no truncation: every link above render's display floor
    (1e-3 x the largest gain), its source token, every neuron, cancelling pairs, content keys."""
    import render
    writes = render.wiring(toy)
    top = max((g for u, w in writes.items() if u != "tok" for g in w.values()), default=1.0) or 1.0
    cancel = render.cancelling_pairs(toy)
    units = {u: Unit(u, cancels=cancel.get(u), key=_content_key(toy, i, j) if kind == "H" else None,
                     offset=_offset(toy, i, j) if kind == "H" else None)
             for u, i, kind, j in render.units_of(toy)}
    for w, rs in writes.items():
        for r, g in rs.items():
            if g <= 1e-3 * top:
                continue
            if w in units:
                units[w].outs.append((r, g))
            if r == "logits":
                continue
            reader, port = (r.split(".") + ["in"])[:2]
            units[reader].ins.append((port, w, g, render.source_token(toy, w, r) if w != "tok" else None))
    return units, [toy.vocab[i] for i in toy.texts[0]]


def neuron_fires(toy, unit: str, ids: list) -> bool:
    """A neuron's pre-activation at the last position from the weights and the text alone: the residual is the last
    token's embedding plus each live layer-0 head's output for one key (a positional head's pos-d offset, a content
    head's best-scoring text position by its token QK); later heads are ignored (other modules are gated off, twins
    cancel, decoys write nothing without their trigger)."""
    import render
    sd, cfg = toy.sd, toy.cfg
    E = sd["wte.weight"].float()
    T = len(ids)
    x = E[ids[-1]].clone()
    for u, i, kind, j in render.units_of(toy):
        if i != 0 or kind != "H":
            continue
        pos, qk, En, Wv, Wo = _head_maps(toy, i, j)
        if float(pos.max() - pos.min()) > 1.0:
            t = max(0, T - 1 - int(pos.argmax()))
        else:
            sc = [float(qk[ids[-1], ids[t]]) for t in range(T)]
            t = max(range(T), key=lambda q: (sc[q], q))
        x = x + Wo @ (Wv @ En[ids[t]])
    i, n = (int(v) for v in UNIT.fullmatch(unit).group(1, 3))
    h = render._rms(x, sd[f"h.{i}.rms_2.weight"].float(), render._eps(cfg))
    return float(sd[f"h.{i}.mlp.c_fc.weight"].float()[n] @ h) > 0


def circuit_from_weights(toy, refine: bool = True, select: bool = False) -> tuple[set, set, dict]:
    keep, edges, info = rule(*units_from_weights(toy), refine=refine)
    if select:   # drop gated neurons that stay off on this text (the layer-0 units traced through them stay)
        off = {u for u in info["gated"] if not is_head(u) and not neuron_fires(toy, u, toy.texts[0])}
        keep -= off
        edges = {e for e in edges if not off & set(e)}
    return keep, edges, info


def predict_weights(toy, prompt: str = "") -> str:
    keep, edges, _ = circuit_from_weights(toy)
    return completion(keep, edges)


def main():
    """Unit F1 / exact of the prompt rule (steps 1-4; + refinements) and the weights rule (+ neuron selection), per
    split and family, on evaluate.py's sample of texts."""
    import argparse
    import os
    os.environ["TOYLOOP_SECTIONS"] = "weights"
    import torch
    import evaluate
    import render
    render.SECTIONS = "weights"
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--zoo", type=Path, default=Path.home() / "mpd-data/toy_zoo2/manifest.jsonl")
    ap.add_argument("-v", action="store_true", help="print each miss")
    a = ap.parse_args()
    torch.set_num_threads(2)
    evaluate.MANIFEST = a.zoo.expanduser()
    splits = evaluate.load_splits(a.limit, a.limit)
    arms = ["prompt_base", "prompt", "weights", "weights_select"]
    print(f"{'split':14s} {'family':20s} {'n':>3s} " + " ".join(f"{x + ' f1/exact':>24s}" for x in arms))
    for split, toys in splits.items():
        fams = {}
        for t in toys:
            gold = {u for u in t.truth["units"] if UNIT.fullmatch(u)}
            pu = parse(render.render(t))
            got = {"prompt_base": rule(*pu, refine=False)[0], "prompt": rule(*pu)[0],
                   "weights": circuit_from_weights(t)[0], "weights_select": circuit_from_weights(t, select=True)[0]}
            fams.setdefault(t.meta["family"], []).append(
                [x for k in arms for x in (evaluate.prf(got[k], gold)[2], float(got[k] == gold))])
            if a.v and got["prompt"] != gold:
                print(f"  {split} {t.name} {t.meta['family']} gold {sorted(gold)} " + " ".join(
                    f"{k} -{sorted(gold - got[k])} +{sorted(got[k] - gold)}" for k in arms[1:]))
        allr = [x for r in fams.values() for x in r]
        for f, r in [("ALL", allr)] + sorted(fams.items()):
            m = [sum(x[i] for x in r) / len(r) for i in range(2 * len(arms))]
            print(f"{split:14s} {f:20s} {len(r):3d} " + " ".join(f"{m[2 * k]:15.2f} / {m[2 * k + 1]:.2f}" for k in range(len(arms))))


if __name__ == "__main__":
    main()
