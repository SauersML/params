"""The variables of an explanation, scored by interchange interventions in the real model (#2951 graph oracle): the
explanation as a causal model, its abstraction accuracy.

A variable V: a meaning in English, its input positions I_V (the context tokens it is computed from), its group G_V
(the circuit's subcomponents, at positions, that carry it) and what it feeds (other variables or the prediction;
recorded, not scored here). Every run is the real model in float32 (engine.Engine's graph run of the whole model,
every subcomponent addressable, the remainders live; tested equal to native.run's patches) and every patched value is
a subcomponent's activation in that run on the changed text. Changed texts are same-length token replacements at one
position (two for composition), drawn by goal_eval's interchange splits (GoalEval._interchange_draws: keyed by the
text's seed, the position and the token, never the explanation; "search" for teachers, "eval" for scores).

  on-target: x' changes a token at a position in I_V. The model on x with G_V's activations set to their values on x'
     (everything else computing on x) should give model(x'): s_on = clip(ln(D0 / D) / ln(D0 / D_floor), 0, 1) with
     D0 = KL(model(x') || model(x)) (no change), D = KL(model(x') || patched) and D_floor the same KL with every
     subcomponent at the targets patched from x' (the best any group does).
  off-target: x' changes a token outside I_V (the text's influential positions, tx.f2, the most moved first). G_V
     patched from x' should leave model(x): the same score with x and x' swapped, D0 = KL(model(x) || model(x')),
     D = KL(model(x) || patched), D_floor = KL(model(x) || the model on x' with every subcomponent at the targets
     patched from x). A group at the last layer at the target passes on-target and fails here.
  composition: x' changes one input of each of two variables (at different positions); G_V1 and G_V2 patched together
     should give model(x'), scored as on-target.

D_floor is at least the runs' float32 resolution, d_res: the KL of the model from the run with G_V set to its own
values on x, and between native's and the engine's runs of either text (on vpd4l every subcomponent at the target
reproduces x' to ~1e-7 bits, the remainders there being small, so the log scale counts orders of magnitude down to
that resolution). Beside s, f = clip(1 - D / D0, 0, 1), the share of the change reproduced (on-target) or kept out
(off-target). A kind's score for a variable (or pair) is the D0-weighted mean over its draws that a patch can resolve
(D0 above D_floor); none without one. score() returns per variable s_on, s_off, f_on and f_off, per pair s_comp and
f_comp, every draw's KLs, and the means s_on, s_off, s_comp and s (the mean of every variable's and pair's scores).

  python method/oracle/variables.py text6011 explanation.json [--split eval] [--draws 8]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402

ROWS = 8  # patched runs a batch (every subcomponent addressable: [rows, T, C] patches per site)


def parse(variables: list[dict], C: dict) -> list[dict]:
    """Variables from JSON: {"name", "meaning", "inputs": [positions], "group": [[matrix, index, position], ...], "feeds":
    [names or "prediction"]}, a matrix native's name or "L.kind" (goal_eval.circuit's). The group as (site, position,
    index) triples; a part outside the model raises."""
    out = []
    for v in variables:
        group = []
        for m, c, t in v["group"]:
            n = goal_eval._site(m)
            if n not in C or not 0 <= int(c) < C[n]:
                raise ValueError(f"{v.get('name')}: {m, c, t} is not a subcomponent")
            group.append((n, int(t), int(c)))
        out.append({"name": v.get("name", ""), "meaning": v.get("meaning", ""), "inputs": sorted({int(p) for p in v["inputs"]}),
                    "group": sorted(set(group)), "feeds": list(v.get("feeds", []))})
    return out


class Scorer:
    """score(tx, variables) for one text at a time (module docstring)."""

    def __init__(self, ge: goal_eval.GoalEval, rows: int = ROWS):
        self.ge, self.nat, self.rows = ge, ge.nat, rows

    def _engine(self, ids: list[int], targets: list[int]):
        import engine

        return engine.Engine(self.nat, ids, targets, live=True)

    @torch.no_grad()
    def _text(self, ids: list[int], targets: list[int]) -> dict:
        """A text's engine, the model's log q at the targets from it (the engine's graph run of the whole model, the
        patched runs' arithmetic), every subcomponent's activation in that run {site: [T, C]}, and the resolution:
        KL(native's reference || the engine's), how far two float32 runs of the model on the same text disagree."""
        nat, dev = self.nat, self.nat.dev
        E = self._engine(ids, targets)
        T = len(ids)
        model = native.Graph(masks={n: torch.ones(1, 1, dtype=torch.bool, device=dev) for n in nat.names})
        u = {n: torch.ones(1, 1, nat.C[n], device=dev) for n in nat.names}
        spec = {n: (torch.zeros(T * nat.C[n], dtype=torch.long, device=dev), torch.arange(T, device=dev).repeat_interleave(nat.C[n]),
                    torch.arange(nat.C[n], device=dev).repeat(T)) for n in nat.names}
        lq, got = E.graph_log_probs(E.plan(model, semantics="c"), u, reads=spec)
        res = float(carry.kl(nat.reference([ids], targets), lq)[0])
        return {"E": E, "lq": lq[0], "act": {n: v.view(T, nat.C[n]) for n, v in got.items()}, "res": max(res, 0.0)}

    @torch.no_grad()
    def _patched(self, E, rows: list[tuple[dict, list | None]]) -> torch.Tensor:
        """log q at the targets [len(rows), targets, V] of the model on E's text, row b with the activations rows[b] =
        (source {site: [T, C]}, nodes (site, position, index) or None for every subcomponent at the targets) set to the
        source's."""
        nat, T, dev = self.nat, E.T, self.nat.dev
        model = native.Graph(masks={n: torch.ones(1, 1, dtype=torch.bool, device=dev) for n in nat.names})
        u = {n: torch.ones(1, 1, nat.C[n], device=dev) for n in nat.names}
        tg = torch.tensor(E.targets, device=dev)
        out = []
        for s0 in range(0, len(rows), self.rows):
            part = rows[s0:s0 + self.rows]
            R, patch = len(part), {}
            for b, (src, nodes) in enumerate(part):
                if nodes is None:
                    for n in nat.names:
                        m, v = patch.setdefault(n, (torch.zeros(R, T, nat.C[n], dtype=torch.bool, device=dev),
                                                    torch.zeros(R, T, nat.C[n], device=dev)))
                        m[b, tg] = True
                        v[b, tg] = src[n][tg]
                    continue
                per = {}
                for n, t, c in nodes:
                    per.setdefault(n, ([], []))
                    per[n][0].append(t)
                    per[n][1].append(c)
                for n, (ts, cs) in per.items():
                    m, v = patch.setdefault(n, (torch.zeros(R, T, nat.C[n], dtype=torch.bool, device=dev),
                                                torch.zeros(R, T, nat.C[n], device=dev)))
                    t_, c_ = torch.tensor(ts, device=dev), torch.tensor(cs, device=dev)
                    m[b, t_, c_] = True
                    v[b, t_, c_] = src[n][t_, c_]
            out.append(E.graph_log_probs(E.plan([model] * R, semantics="c"), u, patch=patch))
            native.free(dev)
        return torch.cat(out)

    def score(self, tx, variables: list[dict], split: str = "eval", draws: int = 8) -> dict:
        """The variables' abstraction accuracy on tx (variables as parse() returns them)."""
        ge, nat = self.ge, self.nat
        moved = tx.moved
        f2 = [p for p, _ in tx.f2]
        tests = []  # (kind, owner, changes ((p, z), ...), nodes)
        on = {}
        for i, v in enumerate(variables):
            on[i] = ge._interchange_draws(tx, v["inputs"], split, draws) if v["inputs"] else []
            tests += [("on", i, ((p, z),), v["group"]) for p, z in on[i]]
            off = [p for p in f2 if p not in v["inputs"]]
            for p, z in (ge._interchange_draws(tx, off, split, draws, first={p: -moved[p - 1] for p in off}) if off else []):
                tests.append(("off", i, ((p, z),), v["group"]))
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                both = [((a, za), (b, zb)) for (a, za), (b, zb) in zip(on[i], on[j]) if a != b][:draws]
                group = sorted(set(variables[i]["group"]) | set(variables[j]["group"]))
                tests += [("comp", (i, j), tuple(sorted(ch)), group) for ch in both]
        changed = sorted({t[2] for t in tests})
        texts = {}
        for ch in changed:
            x = list(tx.ids)
            for p, z in ch:
                x[p] = z
            texts[ch] = self._text(x, tx.targets)
        mine = self._text(tx.ids, tx.targets)
        base = mine["lq"]
        groups = sorted({tuple(t[3]) for t in tests})
        # on x: every test's group from its x', the floors (every subcomponent at the targets from x'), and each group
        # set to its own values on x (the patched runs' float32 resolution: an exact patch is that far from the model)
        lq = self._patched(mine["E"], [(texts[t[2]]["act"], t[3]) for t in tests] + [(texts[ch]["act"], None) for ch in changed]
                           + [(mine["act"], list(g)) for g in groups])
        got, floor_to = lq[:len(tests)], dict(zip(changed, lq[len(tests):len(tests) + len(changed)]))
        same = dict(zip(groups, lq[len(tests) + len(changed):]))
        floor_back = {ch: self._patched(texts[ch]["E"], [(mine["act"], None)])[0]  # on x': every subcomponent at the
                      for ch in sorted({t[2] for t in tests if t[0] == "off"})}   # targets from x (the off-target floors)
        kl = lambda p, q: float(carry.kl(p[None], q[None])[0])  # noqa: E731
        rows = []
        for (kind, owner, ch, group), q in zip(tests, got):
            ref, res = texts[ch]["lq"], max(texts[ch]["res"], mine["res"], kl(base, same[tuple(group)]))
            if kind == "off":
                d0, d, fl = kl(base, ref), kl(base, q), kl(base, floor_back[ch])
            else:
                d0, d, fl = kl(ref, base), kl(ref, q), kl(ref, floor_to[ch])
            rows.append({"kind": kind, "owner": owner, "changes": [list(c) for c in ch], "d0": d0, "d": d, "d_floor": fl,
                         "d_res": res, "s": goal_eval._fid(d0, d, max(fl, res)), "f": max(0.0, 1 - d / d0) if d0 > res else None})
        del texts, mine
        native.free(nat.dev)

        def pooled(sel, key="s"):
            ok = [r for r in sel if r[key] is not None]
            w = sum(r["d0"] for r in ok)
            return (sum(r["d0"] * r[key] for r in ok) / w if w > 0 else None), len(ok)

        out = {"variables": [], "pairs": [], "draws": rows}
        for i, v in enumerate(variables):
            s_on, n_on = pooled([r for r in rows if r["kind"] == "on" and r["owner"] == i])
            s_off, n_off = pooled([r for r in rows if r["kind"] == "off" and r["owner"] == i])
            f_on = pooled([r for r in rows if r["kind"] == "on" and r["owner"] == i], "f")[0]
            f_off = pooled([r for r in rows if r["kind"] == "off" and r["owner"] == i], "f")[0]
            out["variables"].append({"name": v["name"], "s_on": s_on, "s_off": s_off, "f_on": f_on, "f_off": f_off,
                                     "draws_on": n_on, "draws_off": n_off, "group": len(v["group"]), "inputs": v["inputs"]})
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                s_c, n_c = pooled([r for r in rows if r["kind"] == "comp" and r["owner"] == (i, j)])
                if n_c:
                    f_c = pooled([r for r in rows if r["kind"] == "comp" and r["owner"] == (i, j)], "f")[0]
                    out["pairs"].append({"names": [variables[i]["name"], variables[j]["name"]], "s_comp": s_c, "f_comp": f_c,
                                         "draws_comp": n_c})
        mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
        ons = [x["s_on"] for x in out["variables"] if x["s_on"] is not None]
        offs_ = [x["s_off"] for x in out["variables"] if x["s_off"] is not None]
        comps = [x["s_comp"] for x in out["pairs"] if x["s_comp"] is not None]
        out.update(s_on=mean(ons), s_off=mean(offs_), s_comp=mean(comps), s=mean(ons + offs_ + comps))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", help="a vpd4l text id (native.TEXTS/vpd4l/<task>.json)")
    ap.add_argument("explanation", type=Path, help='JSON: {"variables": [...]} (parse())')
    ap.add_argument("--split", choices=("search", "eval"), default="eval")
    ap.add_argument("--draws", type=int, default=8)
    args = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native())
    ids, targets = native.text(native.TEXTS / "vpd4l" / f"{args.task}.json")
    tx = ge.text(ids, targets)
    got = Scorer(ge).score(tx, parse(json.loads(args.explanation.read_text())["variables"], ge.nat.C), args.split, args.draws)
    for v in got["variables"]:
        print(f"{v['name']}: on-target s {v['s_on']} f {v['f_on']}, off-target s {v['s_off']} f {v['f_off']} "
              f"({v['draws_on']} and {v['draws_off']} draws, {v['group']} subcomponents)")
    for p in got["pairs"]:
        print(f"{' + '.join(p['names'])}: composition s {p['s_comp']} f {p['f_comp']} ({p['draws_comp']} draws)")
    print(f"mean: on {got['s_on']}, off {got['s_off']}, composition {got['s_comp']}, all {got['s']}")


if __name__ == "__main__":
    main()
