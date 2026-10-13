"""A non-LM weights-only baseline (TOYLOOP lane): how much of the circuit the weight rendering determines.

From render.wiring's writer -> reader gains, walk back from the logits: a unit is in the circuit if it writes to the
logits, or into the Q/K/V of a head or the input of a neuron already in it, with gain above TAU x the largest gain;
cancelling pairs (render.cancelling_pairs) are left out. Edges: tok -> each circuit unit reading token identity,
writer -> reader among circuit units, circuit unit -> out where it writes the logits. Uses only the weights, so it
measures what TOYLOOP_SECTIONS=weights makes available (any text in the prompt is ignored).

  python3 -B experiments/oracle/toyloop/evaluate.py weights      # the baseline on every split
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402

TAU = 0.05


def circuit_of(toy, tau: float = TAU) -> tuple[set, set]:
    writes = render.wiring(toy)
    skip = set(render.cancelling_pairs(toy))
    top = max((g for u, w in writes.items() if u != "tok" for g in w.values()), default=0.0) or 1.0
    keep, edges = set(), set()
    layer = lambda u: int(u[1:].replace("N", "H").split("H")[0])  # noqa: E731
    order = sorted((u for u in writes if u != "tok"), key=lambda u: (-layer(u), "H" in u))
    for u in order:   # deepest layer first, a layer's neurons before its heads: a unit's readers are decided first
        if u in skip:
            continue
        for r, g in writes[u].items():
            if g <= tau * top:
                continue
            reader = r.split(".")[0]
            if r == "logits":
                keep.add(u)
                edges.add((u, "out"))
            elif reader in keep:
                keep.add(u)
                edges.add((u, reader))
    for r, g in writes["tok"].items():
        if r.split(".")[0] in keep and g > tau * top:
            edges.add(("tok_in", r.split(".")[0]))
    return keep, edges


def predict(toy, prompt: str = "") -> str:
    keep, edges = circuit_of(toy)
    if not edges:
        return "flowchart LR\n  tokX --> out\nEnglish: No unit carries the prediction."
    return "flowchart LR\n" + "\n".join(f"  {a} --> {b}" for a, b in sorted(edges)) + \
        "\nEnglish: Units found by walking back from the logits through weight gains."
