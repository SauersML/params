"""The essay's neuron-to-neuron link numbers measured two ways on Qwen3-32B (#2951, scale32 10-16): the direct link, every
path from the writer's pole to the reader that runs through no other neuron cut (neuron_edges' path patching: the
reader reads its input less what the writer's write brings it, through the residual at the writer's own position and
through every head's value from the writer's positions, each head's pattern as in the run; everything else computes as
usual), and the total effect, the writer's pole zeroed in the stream so every later layer recomputes (essay_replication's
zeroing). The essay's links are first-order direct links ("first-order effect of the source neuron's output at the source
word on the target neuron's activation at the destination, through every head (one hop, direct path)", its figure data),
and essay_replication's two failures compare them with total effects:
  headache   the comma after "headache" in the preeclampsia prompt: L3:5424 (headache), L10:13522 (migraine) and
             L25:3322 (brain) at " headache" into L31:5482 (preeclampsia), L35:3336 (eclampsia) and L33:9390
             (hypertension) at ","; the three together into L31:5482 (the essay's "silence the headache neurons there":
             45.0 -> 30.25)
  miyazaki   the seven listed poles at every position up to "aki" whose value has the pole's sign, into L4:8476 at "aki"
             (and each position's direct link alone); L4:8476's positive pole into L6:24077 and L8:14338 at "aki" itself and
             at essay_replication's " Away" and "ica" (the essay does not say where it reads its out-links)
Per link: the reader's value, the drop with the direct link cut and with the writer's pole zeroed (positive: the writer
raises the reader), against the essay's strength; Spearman over the miyazaki into-links for each.
A model smaller than Qwen3-32B (a smoke run) takes every layer and neuron index modulo its own sizes.

  essay_links.py OUT.json [--model Qwen/Qwen3-32B] [--dtype bfloat16] [--cases headache,miyazaki]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "method" / "oracle"))

import essay_replication as er  # noqa: E402
import neuron_edges as ne  # noqa: E402

HEADACHE = {
    "prompt": er.ATTENTION["comma"]["prompt"], "source": " headache", "dest": ",",
    "writers": [((3, 5424), 1), ((10, 13522), 1), ((25, 3322), 1)],
    "readers": {(31, 5482): [9.4233, 3.3182, 0.9942], (35, 3336): [3.7168, 0.9218, 0.6012], (33, 9390): [2.3625, 1.0484, 0.471]},
    "together": {"reader": (31, 5482), "before": 45.0, "after": 30.25}}


class Links:
    """Direct links and total effects on one model (Edges per text and reader, essay_replication.Qwen's zeroing)."""

    def __init__(self, q: er.Qwen):
        self.q = q
        cfg = q.model.config
        self.L, self.D = cfg.num_hidden_layers, cfg.intermediate_size
        self.small = self.L != 64 or self.D != 25600

    def n(self, unit: tuple) -> tuple:
        """A 32B (layer, neuron) on this model (modulo its sizes for a smoke run)."""
        return (unit[0] % self.L, unit[1] % self.D) if self.small else tuple(unit)

    def values(self, ids: list[int], unit: tuple, upto: int) -> list[float]:
        """unit's value at positions 0..upto in the plain run."""
        L, j = self.n(unit)
        cap = self.q.run(ids, capture=[(L, j, p) for p in range(upto + 1)])["cap"]
        return [cap[(L, j, p, "value")] for p in range(upto + 1)]

    def measure(self, ids: list[int], writers: list[tuple], reader: tuple, rpos: int) -> dict:
        """writers [((layer, neuron), pole, positions)] together into reader at rpos: the reader's value as run, with every
        direct link from the writers cut, and with the writers' poles zeroed at their positions."""
        Lr, jr = self.n(reader)
        ws = [(self.n(u), s, [p for p in ps if p <= rpos]) for u, s, ps in writers]
        nodes = [(L, p, j) for (L, j), _, ps in ws for p in ps] + [(Lr, rpos, jr)]
        ed = ne.Edges(self.q.model, ids, rpos, [0], [1], nodes)
        H = ed.H
        r = (Lr, rpos, jr)
        edges = []
        for (L, j), _, ps in ws:
            if L >= Lr:
                continue
            for p in ps:
                w = (L, p, j)
                if p == rpos:
                    edges.append((w, r))
                edges += [(w, (c, h), r) for c in range(L + 1, Lr + 1) for h in range(H)]
        with torch.no_grad():
            base = float(ed.base["h"][Lr][rpos, jr])
            cut = float(ed._run(torch.ones(1, len(edges), dtype=torch.bool), edges, keep_states=True)["h"][Lr][rpos, jr]) \
                if edges else base
        zero = [(L, j, ps, s) for (L, j), s, ps in ws if ps]
        plain = self.q.run(ids, capture=[(Lr, jr, rpos)])["cap"][(Lr, jr, rpos, "value")]
        zeroed = self.q.run(ids, zero=zero, capture=[(Lr, jr, rpos)])["cap"][(Lr, jr, rpos, "value")]
        ne.free()
        return {"reader": f"L{reader[0]}:{reader[1]}", "pos": rpos, "edges": len(edges), "value": base, "value_hf": plain,
                "direct": base - cut, "total": plain - zeroed}


def spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def case_headache(lk: Links, log=print) -> dict:
    q = lk.q
    ids = q.ids(HEADACHE["prompt"])
    src = er.find(q.tok, ids, HEADACHE["source"])
    dst = src + 1
    assert q.tok.decode([ids[dst]]) == HEADACHE["dest"], q.tok.decode([ids[dst]])
    out = {"case": "headache", "source_pos": src, "dest_pos": dst, "links": []}
    for reader, essay in HEADACHE["readers"].items():
        for ((u, s), e) in zip(HEADACHE["writers"], essay):
            m = lk.measure(ids, [(u, s, [src])], reader, dst)
            m.update(writer=f"L{u[0]}:{u[1]}", essay=e)
            out["links"].append(m)
            log(f"  headache {m['writer']} -> {m['reader']}: essay {e:+.2f}, direct {m['direct']:+.2f}, total {m['total']:+.2f} "
                f"(reader {m['value']:.2f}; {m['edges']} edges)")
    t = HEADACHE["together"]
    m = lk.measure(ids, [(u, s, [src]) for u, s in HEADACHE["writers"]], t["reader"], dst)
    m.update(essay_before=t["before"], essay_after=t["after"])
    out["together"] = m
    log(f"  headache: the three together into {m['reader']}: essay {t['before']:.2f} -> {t['after']:.2f}; direct "
        f"{m['value']:.2f} -> {m['value'] - m['direct']:.2f}, total {m['value_hf']:.2f} -> {m['value_hf'] - m['total']:.2f}")
    return out


def case_miyazaki(lk: Links, log=print) -> dict:
    q, M = lk.q, er.MIYAZAKI
    ids = q.ids(M["into_text"])
    p = er.find(q.tok, ids, M["into_token"])
    out = {"case": "miyazaki", "pos": p, "into": [], "out": []}
    for u, s, e in M["into"]:
        vals = lk.values(ids, u, p)
        ps = [i for i, v in enumerate(vals) if v * s > 0]
        m = lk.measure(ids, [(u, s, ps)], M["target"], p)
        each = {i: lk.measure(ids, [(u, s, [i])], M["target"], p)["direct"] for i in ps}   # (each position's direct link alone)
        m.update(writer=f"L{u[0]}:{u[1]}", pole=s, essay=e, positions=ps, writer_values=[round(vals[i], 3) for i in ps],
                 direct_by_position={str(i): v for i, v in each.items()}, direct_at_name=each.get(p, 0.0))
        out["into"].append(m)
        log(f"  miyazaki L{u[0]}:{u[1]}{'+' if s > 0 else '-'} -> L4:8476: essay {e:+.2f}, direct {m['direct']:+.2f} (at the name "
            f"{m['direct_at_name']:+.2f}; by position {', '.join(f'{i}: {v:+.2f}' for i, v in each.items())}), total {m['total']:+.2f}")
    ess = [m["essay"] for m in out["into"]]
    out["spearman_direct"] = spearman([m["direct"] for m in out["into"]], ess)
    out["spearman_direct_at_name"] = spearman([m["direct_at_name"] for m in out["into"]], ess)
    out["spearman_total"] = spearman([m["total"] for m in out["into"]], ess)
    vals = lk.values(ids, M["target"], p)       # the out-links at the name itself (the essay does not say where it reads them)
    ps = [i for i, v in enumerate(vals) if v > 0]
    out["out_at_name"] = []
    for (u, e, _, _) in M["out"]:
        m = lk.measure(ids, [(M["target"], 1, ps)], u, p)
        m.update(writer="L4:8476", essay=e, positions=ps)
        out["out_at_name"].append(m)
        log(f"  miyazaki L4:8476+ -> L{u[0]}:{u[1]} at the name: essay {e:+.2f}, direct {m['direct']:+.2f}, total {m['total']:+.2f} "
            f"(reader {m['value']:.2f})")
    for (u, e, text, piece) in M["out"]:
        ids2 = q.ids(text)
        p2 = er.find(q.tok, ids2, piece)
        vals = lk.values(ids2, M["target"], p2)
        ps = [i for i, v in enumerate(vals) if v > 0]
        m = lk.measure(ids2, [(M["target"], 1, ps)], u, p2)
        m.update(writer="L4:8476", essay=e, positions=ps)
        out["out"].append(m)
        log(f"  miyazaki L4:8476+ -> L{u[0]}:{u[1]}: essay {e:+.2f}, direct {m['direct']:+.2f}, total {m['total']:+.2f}")
    log(f"  miyazaki into: Spearman with the essay, direct {out['spearman_direct']:.2f} (at the name alone "
        f"{out['spearman_direct_at_name']:.2f}), total {out['spearman_total']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cases", default="headache,miyazaki")
    a = ap.parse_args()
    q = er.Qwen(a.model, a.device, getattr(torch, a.dtype))
    lk = Links(q)
    t0 = time.time()
    rec = {"model": a.model, "dtype": a.dtype, "smoke": lk.small, "cases": {}}
    for name in a.cases.split(","):
        rec["cases"][name] = {"headache": case_headache, "miyazaki": case_miyazaki}[name](lk)
        rec["cases"][name]["seconds"] = round(time.time() - t0, 1)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
