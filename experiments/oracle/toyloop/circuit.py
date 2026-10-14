"""Oracle answers <-> circuits (TOYLOOP lane): parse the Mermaid + English answer into a unit-level circuit, map units
to the repo's node sets (goal_eval.circuit's answer JSON: [site, piece, position] nodes), and the stub reward.

  parse(text) -> Circuit(units, edges, english, ok)
  units_to_answer(circuit, toy, positions=None) -> {"nodes": [[site, piece, pos], ...], "complete": True, ...}
  reward_f1(meta, completion) -> float    # F1 of predicted units vs meta["truth"]["units"]; 0 when unparseable
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

UNIT = re.compile(r"^(tok\w+|L\d+H\d+|L\d+N\d+|out)$")
_NODE = r"([A-Za-z_][\w]*)(?:\s*(?:\[[^\]]*\]|\([^)]*\)|\{[^}]*\}))?"     # an id, optionally a ["label"] / (label) / {label}
_ARROW = r"\s*(?:-->|---|==>|-\.->|-.->)\s*(?:\|[^|]*\|\s*)?"


@dataclass
class Circuit:
    units: set = field(default_factory=set)
    edges: set = field(default_factory=set)      # (upstream unit, downstream unit)
    english: str = ""
    ok: bool = False                              # a flowchart with at least one valid edge was found
    bad: list = field(default_factory=list)       # ids that are not units


def parse(text: str) -> Circuit:
    """Tolerant Mermaid flowchart reader: 'flowchart'/'graph' header optional, ``` fences ignored, chains 'a --> b --> c'
    and '&' groups ('a & b --> c') expanded, edge labels '-->|x|' and node labels 'a["..."]' dropped. Ids that are not
    units (UNIT) are recorded in .bad and left out. The English is the text after the first 'English:'."""
    c = Circuit()
    if re.match(r"\s*Trace:", text):   # a TOYLOOP_TRACE derivation comes first: read from the flowchart header on
        m = re.search(r"^\s*(?:```\w*\s*)?(?:flowchart|graph)\b", text, re.M)
        if m:
            text = text[m.start():]
    head, _, eng = text.partition("English:")
    c.english = eng.strip().split("\n\n")[0].strip()
    for line in head.splitlines():
        line = line.strip().strip("`").strip().rstrip(";")
        if not line or line.startswith(("flowchart", "graph", "%%", "subgraph", "end", "classDef", "class ", "style")):
            continue
        parts = re.split(_ARROW, line)
        if len(parts) < 2:
            m = re.fullmatch(_NODE, line)
            if m and UNIT.match(m.group(1)):
                c.units.add(m.group(1))
            continue
        groups = []
        for p in parts:
            ids = [re.match(_NODE, q.strip()).group(1) for q in p.split("&") if re.match(_NODE, q.strip())]
            good = [i for i in ids if UNIT.match(i)]
            c.bad += [i for i in ids if not UNIT.match(i)]
            groups.append(good)
        for a, b in zip(groups, groups[1:]):
            for x in a:
                for y in b:
                    if x != y:
                        c.edges.add((x, y))
        for g in groups:
            c.units.update(g)
    c.ok = bool(c.edges)
    return c


def to_mermaid(c: Circuit) -> str:
    body = "\n".join(f"  {a} --> {b}" for a, b in sorted(c.edges))
    return f"flowchart LR\n{body}\nEnglish: {c.english}"


# ---------------------------------------------------------------- units -> the repo's node sets

def unit_pieces(toy, unit: str) -> list[tuple[str, int]]:
    """(site, piece index) pairs of toys/planted.decompose's exact decomposition that belong to a unit: a head's
    query/key/value/output pieces (decompose's unit = the head), a neuron's c_fc row and down_proj column."""
    dec = _decomposition(toy)
    m = re.fullmatch(r"L(\d+)([HN])(\d+)", unit)
    if not m:
        return []
    i, kind, j = int(m.group(1)), m.group(2), int(m.group(3))
    sites = [f"h.{i}.attn.{k}" for k in ("q_proj", "k_proj", "v_proj", "o_proj")] if kind == "H" else \
            [f"h.{i}.mlp.c_fc", f"h.{i}.mlp.down_proj"]
    return [(s, int(c)) for s in sites if s in dec for c in (dec[s][2] == j).nonzero().flatten().tolist()]


def _decomposition(toy) -> dict:
    """The toy's decomposition {site: (U, V, unit)}: toy.meta["decomposition"] when the zoo ships one, else planted's."""
    if "decomposition" not in toy.meta:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        sys.path[:0] = [str(root / "experiments/oracle/toys"), str(root / "method/oracle"), str(root / "method/vpd")]
        import planted
        toy.meta["decomposition"] = planted.decompose(toy.sd, toy.cfg)
    return toy.meta["decomposition"]


def units_to_answer(c: Circuit, toy, positions: list[int] | None = None, text: int = 0) -> dict:
    """goal_eval.circuit's answer JSON for a unit circuit on toy.texts[text]: every piece of every unit at every position
    in `positions` (default: all of the text; a head's pieces at the positions where they act are a subset), the token
    units as ["tok", 0, position] at the positions holding that token, complete=True (the unit edges are kept under
    "unit_edges" for an edge-level scorer). Node order is goal_eval's JSON [site, piece, position]; native.Graph and the
    zoo's gt_nodes use (site, position, piece). On zoo copy_offset instances the truth's units cover every gt node."""
    ids = toy.texts[text]
    pos = positions if positions is not None else list(range(len(ids)))
    nodes = []
    for u in sorted(c.units):
        if u.startswith("tok"):
            name = u[3:]
            nodes += [["tok", 0, t] for t in pos if toy.vocab[ids[t]] == name]
        elif u != "out":
            nodes += [[s, p, t] for s, p in unit_pieces(toy, u) for t in pos]
    return {"nodes": nodes, "complete": True, "unit_edges": sorted(c.edges), "explanation": c.english}


def unit_graph(c: Circuit, toy, ids: list[int] | None = None):
    """Compile unit wiring into an explicit native graph over rank-one pieces.

    Preserve architecture connections inside each head/neuron. Between units, keep only
    declared edges; into the prediction, keep only declared unit -> out edges. All pieces
    of selected units run at every position, without an activation/ground-truth selector.
    Token-label arrows are INPUT ANNOTATIONS: native execution always receives embeddings.
    This function does not execute token-label meanings or certify the English paragraph.
    Reverse/non-architectural unit edges and invalid unit IDs raise rather than disappear.
    """
    if not c.ok or c.bad:
        raise ValueError("wiring requires a parsed circuit without unknown identifiers")
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root / "method/oracle"), str(root / "method/vpd")]
    import native

    ids = toy.texts[0] if ids is None else ids
    if not ids:
        raise ValueError("wiring requires a nonempty input")
    units = sorted(u for u in c.units if u != "out" and not u.startswith("tok"))
    owner = {}
    for unit in units:
        match = re.fullmatch(r"L(\d+)([HN])(\d+)", unit)
        if not match:
            raise ValueError(f"invalid unit {unit}")
        layer, kind, index = int(match[1]), match[2], int(match[3])
        if layer >= toy.cfg["n_layer"]:
            raise ValueError(f"invalid layer in {unit}")
        count = toy.cfg["n_head"] if kind == "H" else toy.sd[f"h.{layer}.mlp.c_fc.weight"].shape[0]
        if index >= count:
            raise ValueError(f"invalid index in {unit}")
        for site, piece in unit_pieces(toy, unit):
            for pos in range(len(ids)):
                node = (site, pos, piece)
                if node in owner and owner[node] != unit:
                    raise ValueError(f"overlapping unit decomposition at {node}")
                owner[node] = unit
    candidates, outputs = native.all_edges(sorted(owner), [len(ids) - 1])
    possible = {(owner[w], owner[r]) for r, ws in candidates.items() for w in ws if owner[w] != owner[r]}
    possible |= {(owner[w], "out") for w in outputs}
    for a, b in c.edges:
        if a.startswith("tok") and (b in units or b == "out"):
            continue  # annotation only, never interpreted as an embedding mask
        if (a, b) not in possible:
            raise ValueError(f"edge {a} -> {b} has no executable connection in these pieces")
    parents = {r: [w for w in ws if owner[w] == owner[r] or (owner[w], owner[r]) in c.edges]
               for r, ws in candidates.items()}
    return native.Graph(nodes=owner, parents={r: ws for r, ws in parents.items() if ws},
                        out=[w for w in outputs if (owner[w], "out") in c.edges], complete=False)


# ---------------------------------------------------------------- rewards

def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    return 2 * tp / (len(pred) + len(gold)) if (pred or gold) else 0.0


def reward_f1(meta: dict, completion: str) -> float:
    """Stub reward: F1 of the parsed units (tok and out excluded) against the truth's; 0 for an unparseable answer.
    The real reward replaces this with goal_eval's executed phi on units_to_answer(...)."""
    c = parse(completion)
    if not c.ok:
        return 0.0
    keep = lambda s: {u for u in s if u != "out" and not u.startswith("tok")}  # noqa: E731
    return f1(keep(c.units), keep(set(meta["truth"]["units"])))


def reward_f1_edges(meta: dict, completion: str) -> float:
    """Units F1 and edges F1, averaged."""
    c = parse(completion)
    if not c.ok:
        return 0.0
    gold = {tuple(e) for e in meta["truth"]["edges"]}
    return 0.5 * reward_f1(meta, completion) + 0.5 * f1(c.edges, gold)
