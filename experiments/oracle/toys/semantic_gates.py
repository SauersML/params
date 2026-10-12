"""Which operation links two variables, on the planted gate toys (#2951 graph oracle, E4 v2's operations): additive (AND
of x1 = a1 and x2 = b1, the MLP zero) and interaction (XNOR, through the MLP) agree on every ordinary input and differ
on (a0, b0), which no ordinary text holds; their wiring is the same, so a diagram cannot tell them apart.

Variables X1 and X2, carried by blocks (whole site outputs) at positions 1 and 2. A carrier must hold its own token and
not the other's: position 2 reads position 1 through layer 0's attention, so every site there would carry x1 too and
the two-donor outcomes would depend on the base text. Per position the smallest site set (single sites, each layer, both
layers) is taken whose term min(f on-target, f off-target) is at least GOAL, averaged over the ordinary texts: f
on-target = 1 - KL(model(x') || patched) / KL(model(x') || model(x)) for the transplant from x' (that token flipped,
a0 <-> a1, b0 <-> b1), f off-target = 1 - KL(model(x) || patched) / KL(model(x) || model(x'')) for the transplant from
x'' (the other token flipped); variables.py's terms with the gate's own tokens (the interchange splits' draws are
tokens outside the gate's vocabulary). On each ordinary text, every two-donor transplant (X1 from the text with a0 or a1 at position 1, X2 from
the text with b0 or b1 at position 2: four value pairs, (a0, b0) among them) is run in the real network
(semantic.Checker), its top token recorded. semantic.fit_gate fits the two-input gates to the outcomes of all texts
but one and is scored on the left-out text's transplants (the hypothesis counts only where it predicts held-out
outcomes); the INVARIANTS checker then prices the fitted gate's claim on (a0, b0) against the no-change prediction.

  MPD_MEM_GIB=4 ~/mpd-data/venv/bin/python -B experiments/oracle/toys/semantic_gates.py --toys ~/mpd-data/toys_planted
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import planted  # noqa: E402
import semantic  # noqa: E402
import variables  # noqa: E402

BLOCK = variables.BLOCK
GOAL = goal_eval.GOAL


def kl(a, b) -> float:
    return float(carry.kl(a[None, None], b[None, None])[0])


def run(name: str, path: Path) -> dict:
    d = torch.load(path)
    nat = planted.native_of(d["sd"], d["cfg"])
    ge = goal_eval.GoalEval(nat)
    sc = variables.Scorer(ge)
    texts = list(dict.fromkeys(tuple(x) for x in planted.ordinary(name, 6)))
    flip = {planted.A0: planted.A1, planted.A1: planted.A0, planted.B0: planted.B1, planted.B1: planted.B0}
    layers = sorted({native._name_layer_kind(n)[0] for n in nat.names})
    carriers = {}
    txs = {ids: ge.text(list(ids), [len(ids) - 1], seed=0) for ids in texts}
    for p, other in ((1, 2), (2, 1)):
        cands = [[n] for n in nat.names] + [[n for n in nat.names if native._name_layer_kind(n)[0] == L] for L in layers] + [list(nat.names)]
        tried = []
        for sites in cands:
            v = {"name": "X", "inputs": [p], "group": [(n, p, BLOCK) for n in sites]}
            f_on, f_off = [], []
            for ids in texts:
                ck = semantic.Checker(sc, txs[ids], [v])
                base, on, off = ck.p0, ck.text({p: flip[ids[p]]})["lq"][-1], ck.text({other: flip[ids[other]]})["lq"][-1]
                q_on = ck.run({"transplants": [("X", {p: flip[ids[p]]})]})
                q_off = ck.run({"transplants": [("X", {other: flip[ids[other]]})]})
                f_on.append(1.0 - kl(on, q_on) / max(kl(on, base), 1e-12))
                f_off.append(1.0 - kl(base, q_off) / max(kl(base, off), 1e-12))
                ck.close()
            t = min(sum(f_on) / len(f_on), sum(f_off) / len(f_off))
            tried.append((t, sum(nat.C[n] for n in sites), sites, sum(f_on) / len(f_on), sum(f_off) / len(f_off)))
        ok = [x for x in tried if x[0] >= GOAL]
        best = min(ok, key=lambda x: (x[1], -x[0])) if ok else max(tried, key=lambda x: x[0])
        carriers[p] = {"sites": best[2], "term": best[0], "f_on": best[3], "f_off": best[4]}
    vs = [{"name": f"X{p}", "inputs": [p], "group": [(n, p, BLOCK) for n in carriers[p]["sites"]], "feeds": ["prediction"]} for p in (1, 2)]
    rows = []
    for ids in texts:
        ck = semantic.Checker(sc, txs[ids], vs)
        items = [({"transplants": [("X1", {1: x1}), ("X2", {2: x2})]}, ("unchanged",))
                 for x1, x2 in itertools.product((planted.A0, planted.A1), (planted.B0, planted.B1))]
        for (iv, _), r in zip(items, ck.check(items)):
            x1, x2 = iv["transplants"][0][1][1], iv["transplants"][1][1][2]
            rows.append({"text": list(ids), "x1": x1, "x2": x2, "a": x1 == planted.A1, "b": x2 == planted.B1,
                         "top": r["top"], "yes": r["top"] == planted.YES, "p_top": r["p_top"]})
        ck.close()
    held = []
    for i, ids in enumerate(texts):  # fit on every text but one, predict the left-out text's transplants
        fit = semantic.fit_gate([(r["a"], r["b"], r["top"]) for r in rows if tuple(r["text"]) != ids], planted.YES)
        g = fit[0][0]
        test = [r for r in rows if tuple(r["text"]) == ids]
        held.append({"left_out": list(ids), "gate": g, "fit": fit[0][1],
                     "held_out": sum(semantic.GATES[g](r["a"], r["b"]) == r["yes"] for r in test) / len(test)})
    gate = semantic.fit_gate([(r["a"], r["b"], r["top"]) for r in rows], planted.YES)
    # the gate's claim on (a0, b0), which no ordinary text holds, as an invariant on each text: priced against the
    # no-change prediction (where the claim is the text's own top it is trivial and earns nothing)
    claim = planted.YES if semantic.GATES[gate[0][0]](False, False) else planted.NO
    inv = []
    for ids in texts:
        ck = semantic.Checker(sc, txs[ids], vs)
        inv.append({"text": list(ids), **ck.check([({"transplants": [("X1", {1: planted.A0}), ("X2", {2: planted.B0})]}, ("top", claim))])[0]})
        ck.close()
    return {"toy": name, "carriers": carriers, "outcomes": rows, "gates": gate, "held_out": held, "invariant_a0_b0": inv}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toys", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    got = []
    for name in ("additive", "interaction"):
        r = run(name, args.toys / f"{name}.pt")
        got.append(r)
        print(f"{name}: carriers {r['carriers']}")
        print(f"   gates {[(g, round(s, 3)) for g, s in r['gates'][:3]]}; held-out {[(h['gate'], h['held_out']) for h in r['held_out']]}")
        for o in r["outcomes"]:
            print(f"   text {o['text']}: X1 {'a1' if o['a'] else 'a0'}, X2 {'b1' if o['b'] else 'b0'} -> "
                  f"{'YES' if o['yes'] else 'NO' if o['top'] == planted.NO else o['top']} ({o['p_top']:.3f})")
        for i in r["invariant_a0_b0"]:
            print(f"   on {i['text']}: X1 a0, X2 b0 -> {'YES' if semantic.GATES[r['gates'][0][0]](False, False) else 'NO'}: holds "
                  f"{i['holds']}, surprise {i['surprise']:.2f} bits, credit {i['credit']:.2f}")
    if args.out:
        args.out.write_text(json.dumps(got, indent=1))


if __name__ == "__main__":
    main()
