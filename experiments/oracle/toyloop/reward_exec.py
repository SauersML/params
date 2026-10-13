"""Executable reward for the toy loop (TOYLOOP lane): run the predicted circuit and score it, size-penalized.

  reward_exec(meta, completion) -> float, meta = {"toy": render.Toy, "truth": ..., ...} (train.examples adds "toy")

The predicted units (heads, neurons; tok/out ignored) become a node set on each example text: every VPD-active piece
(planted.native_of's causal importance > 0) of those units at every position (the simplest form of the zoo's gt rule).
The model runs with only those nodes on (zoo.node_kl: masked run, remainders on, ~0 on the zoo); per text
  suff = clip(1 - KL(model || circuit) / KL(model || empty circuit), 0, 1)
and reward = mean suff - UNIT_COST * (number of predicted units), 0 for an unparseable answer. Sufficiency alone rewards
the full active set (it carries everything); the per-unit cost makes the minimal sufficient circuit the best answer
(zoo gt: suff ~1 with 1-4 units; adding each distractor costs UNIT_COST and gains nothing).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments/oracle/toys"), str(ROOT / "method/oracle"), str(ROOT / "method/vpd"),
                str(Path(__file__).resolve().parent)]

import circuit  # noqa: E402

UNIT_COST = 0.05
N_TEXTS = 2
_CACHE: dict = {}


def _native(toy):
    if toy.name not in _CACHE:
        import planted
        nat = planted.native_of(toy.sd, toy.cfg)
        _CACHE[toy.name] = {"nat": nat, "ci": [nat.importance(ids) for ids in toy.texts[:N_TEXTS]], "empty": {}}
    return _CACHE[toy.name]


def nodes_of(toy, units: set, text: int) -> list:
    """[[site, position, piece]] (native.Graph order): active pieces of the units on toy.texts[text], all positions."""
    import planted  # noqa: F401  (path set)
    c = _native(toy)
    nat, ci = c["nat"], c["ci"][text]
    want = set()
    for u in units:
        m = circuit.re.fullmatch(r"L(\d+)([HN])(\d+)", u)
        if m:
            want.add((int(m.group(1)), m.group(2), int(m.group(3))))
    out = []
    for n in nat.names:
        layer, kind = int(n.split(".")[1]), "N" if ".mlp." in n else "H"
        for t, p in (ci[n] > 0).nonzero().tolist():
            if (layer, kind, int(nat.units[n][p])) in want:
                out.append([n, int(t), int(p)])
    return out


def sufficiency(toy, units: set) -> float:
    import zoo
    c = _native(toy)
    s = []
    for i, ids in enumerate(toy.texts[:N_TEXTS]):
        if i not in c["empty"]:
            c["empty"][i] = zoo.node_kl(c["nat"], ids, [])
        kl = zoo.node_kl(c["nat"], ids, nodes_of(toy, units, i))
        s.append(min(1.0, max(0.0, 1.0 - kl / max(c["empty"][i], 1e-9))))
    return sum(s) / len(s)


def reward_exec(meta: dict, completion: str) -> float:
    c = circuit.parse(completion)
    if not c.ok:
        return 0.0
    units = {u for u in c.units if circuit.re.fullmatch(r"L\d+[HN]\d+", u)}
    return sufficiency(meta["toy"], units) - UNIT_COST * len(units)


if __name__ == "__main__":   # sanity: gt vs gt + distractors vs empty vs every active unit, on a few zoo instances
    import render
    for toy in render.load_zoo(Path.home() / "mpd-data/toy_zoo/manifest.jsonl")[::40]:
        gt = {u for u in toy.truth["units"] if circuit.re.fullmatch(r"L\d+[HN]\d+", u)}
        dis = {render.zoo_unit(u) for u in toy.meta["record"]["distractor_units"]}
        meta = {"toy": toy}
        row = {k: round(reward_exec(meta, render.target_text({"edges": [[u, "out"] for u in us] or [["tokX", "out"]]})), 3)
               for k, us in (("gt", gt), ("gt-1", set(sorted(gt)[1:])), ("gt+distractors", gt | dis), ("empty", set()),
                             ("distractors", dis))}
        print(toy.name, len(gt), row, flush=True)
