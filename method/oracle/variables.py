"""The variables of an explanation, scored by interchange interventions in the real model (#2951 graph oracle): the
explanation as a causal model, its abstraction accuracy.

A variable V: a meaning in English, its input positions I_V (the context tokens it is computed from), its group G_V
(the circuit's subcomponents, at positions, that carry it) and what it feeds (other variables' names or the
prediction). Every run is the real model in float32 (engine.Engine's graph run of the whole model,
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
  feeds: on V's on-target draws, G_V patched from x' should move the reads of a group W it feeds as model(x') does,
     and leave those of a group it does not feed as model(x) has them: f = clip(1 - |r - r_target|^2 / |r_x - r_x'|^2,
     0, 1) over W's nodes outside G_V, r_target = r_x' or r_x (_feeds).

D_floor is at least the runs' float32 resolution, d_res: the KL between runs that should agree (the run with nothing
patched in the patched runs' batch, the run with G_V set to its own values on x, and an independent run of the text:
native's reference, or goal_eval.graph_ref in C's semantics: the largest such disagreement, twice, since a KL compares
two runs that each carry it); a draw whose D0 is within it is not scored. On vpd4l every subcomponent at the target
reproduces x' to ~1e-7 bits (the remainders there are small), so the log scale counts orders of magnitude down to that
resolution; beside s, f = clip(1 - D / D0, 0, 1), the share of the change reproduced (on-target) or kept out (off-
target). A kind's score for a variable (or pair) is the D0-weighted mean over its scored draws; none without one.

The headline, e4: in the real model, the mean of each variable's min(f on-target, s off-target) (a variable must carry
its inputs and only them; a mean of the two would give a group carrying nothing 1/2), f in composition and f on the
feeds claims; a variable none of whose on-target draws is resolved counts 0 (its inputs do not move the output;
headline()). With a circuit (score(graph=)), the same draws also run in its stated graph under C's semantics (graph
inside, masked outside, as goal_eval scores it; engine.Engine on its subcomponents, tested equal to graph_ref), under
"circuit", a diagnostic: every run, reference and patched value is the circuit's, so a group outside the circuit does
nothing there and a variable whose inputs the circuit does not read has no scored draws; "gap" is the circuit's e4 less
the model's (a circuit can agree with itself and miss the model's mechanism). score() returns per variable s_on, s_off,
f_on and f_off, per pair s_comp and f_comp, per ordered pair the feeds f, every draw's KLs, the means and e4, for the
model and under "circuit".

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


def headline(out: dict) -> dict:
    """A world's means and its headline e4 (the lead's rules) on score()'s per variable, pair and feeds rows: per
    variable min(f on-target, s off-target), since a variable claims to carry its inputs and nothing else (a mean of the
    two gave a group carrying nothing 1/2 for its off-target alone), and 0 for a variable none of whose on-target draws
    is resolved (its inputs do not move the output: the claim is empty; counted in "unresolved"); f in composition and f
    on the feeds claims; e4 the mean of all of them (each variable, pair and feeds claim a separate claim)."""
    mean = lambda xs: (lambda ys: sum(ys) / len(ys) if ys else None)([x for x in xs if x is not None])  # noqa: E731
    V, P, F = out["variables"], out["pairs"], out["feeds"]
    out.update(s_on=mean([x["s_on"] for x in V]), s_off=mean([x["s_off"] for x in V]), s_comp=mean([x["s_comp"] for x in P]),
               f_on=mean([x["f_on"] for x in V]), f_comp=mean([x["f_comp"] for x in P]),
               f_feeds=mean([x["f"] for x in F if x["feeds"]]), f_apart=mean([x["f"] for x in F if not x["feeds"]]),
               unresolved=sum(x["f_on"] is None for x in V))
    each = [0.0 if a is None else (a if b is None else min(a, b)) for a, b in ((x["f_on"], x["s_off"]) for x in V)]
    out["e4"] = mean(each + [x["f_comp"] for x in P] + [x["f"] for x in F])
    return out


class Scorer:
    """score(tx, variables, graph=None) for one text at a time (module docstring)."""

    def __init__(self, ge: goal_eval.GoalEval, rows: int = ROWS):
        self.ge, self.nat, self.rows = ge, ge.nat, rows

    @torch.no_grad()
    def _text(self, ids: list[int], targets: list[int], graph: native.Graph | None = None, check: bool = True) -> dict:
        """A text in a world: the real model (graph None: engine.Engine's graph run of the whole model, every
        subcomponent addressable) or a circuit's stated graph in C's semantics (an engine on its subcomponents). Its
        engine, what each patched run runs ("run": the graph and u), log q at the targets, every addressable
        subcomponent's activation {site: [T, K]} (K the engine's columns, "col" a subcomponent's column), and the
        resolution: KL of an independent float32 run (native's reference, or goal_eval's graph_ref in C's semantics;
        check False: 0) from the engine's."""
        import engine

        nat, dev, T = self.nat, self.nat.dev, len(ids)
        if graph is None:
            E = engine.Engine(nat, ids, targets, live=True)
            run = (native.Graph(masks={n: torch.ones(1, 1, dtype=torch.bool, device=dev) for n in nat.names}),
                   {n: torch.ones(1, 1, nat.C[n], device=dev) for n in nat.names})
            col = None
        else:
            cols = {n: sorted({c for m, _, c in graph.node_set() if m == n}) for n in nat.names}
            E = engine.Engine(nat, ids, targets, live=True, columns={n: torch.tensor(c, dtype=torch.long, device=dev) for n, c in cols.items()})
            run, col = (graph, None), {n: {c: k for k, c in enumerate(cs)} for n, cs in cols.items()}
        K = {n: E.width(n) for n in nat.names}
        spec = {n: (torch.zeros(T * K[n], dtype=torch.long, device=dev), torch.arange(T, device=dev).repeat_interleave(K[n]),
                    torch.arange(K[n], device=dev).repeat(T)) for n in nat.names if K[n]}
        lq, got = E.graph_log_probs(E.plan(run[0], semantics="c"), run[1], reads=spec)
        res = 0.0
        if check:
            ref = nat.reference([ids], targets) if graph is None else self.ge.graph_ref(ids, targets, graph)
            res = max(float(carry.kl(ref, lq)[0]), 0.0)
        return {"E": E, "run": run, "col": col, "lq": lq[0], "act": {n: v.view(T, K[n]) for n, v in got.items()}, "res": res}

    @torch.no_grad()
    def _patched(self, w: dict, rows: list[tuple[dict, list | None]], reads: list | None = None):
        """log q at the targets [len(rows), targets, V] of w's world on w's text (_text), row b with the activations
        rows[b] = (source {site: [T, K]}, nodes (site, position, index) or None for every addressable subcomponent at the
        targets) set to the source's; a node the world cannot address (outside a circuit's columns) is left alone (in
        C's semantics it does nothing). reads: addressable nodes whose activations each row also returns, (log q,
        [len(rows), len(reads)])."""
        nat, E, dev = self.nat, w["E"], self.nat.dev
        T, tg = E.T, torch.tensor(E.targets, device=dev)
        out, got = [], []
        where = {}
        for i, (n, t, c) in enumerate(reads or []):
            x = where.setdefault(n, ([], [], []))
            x[0].append(i)
            x[1].append(t)
            x[2].append(c if w["col"] is None else w["col"][n][c])
        for s0 in range(0, len(rows), self.rows):
            part = rows[s0:s0 + self.rows]
            R, patch = len(part), {}

            def slot(n):
                if n not in patch:
                    patch[n] = (torch.zeros(R, T, E.width(n), dtype=torch.bool, device=dev), torch.zeros(R, T, E.width(n), device=dev))
                return patch[n]
            for b, (src, nodes) in enumerate(part):
                if nodes is None:
                    for n in nat.names:
                        if E.width(n):
                            m, v = slot(n)
                            m[b, tg] = True
                            v[b, tg] = src[n][tg]
                    continue
                per = {}
                for n, t, c in nodes:
                    k = c if w["col"] is None else w["col"][n].get(c)
                    if k is not None:
                        per.setdefault(n, ([], []))
                        per[n][0].append(t)
                        per[n][1].append(k)
                for n, (ts, ks) in per.items():
                    m, v = slot(n)
                    t_, k_ = torch.tensor(ts, device=dev), torch.tensor(ks, device=dev)
                    m[b, t_, k_] = True
                    v[b, t_, k_] = src[n][t_, k_]
            spec = {n: (torch.arange(R, device=dev).repeat_interleave(len(ts)), torch.tensor(ts, device=dev).repeat(R),
                        torch.tensor(ks, device=dev).repeat(R)) for n, (_, ts, ks) in where.items()} if reads else None
            lq = E.graph_log_probs(E.plan([w["run"][0]] * R, semantics="c"), w["run"][1], patch=patch or None, reads=spec)
            if reads:
                lq, rec = lq
                r = torch.zeros(R, len(reads), device=dev)
                for n, (idx, ts, _) in where.items():
                    r[:, idx] = rec[n].view(R, len(ts))
                got.append(r)
            out.append(lq)
            native.free(dev)
        return (torch.cat(out), torch.cat(got)) if reads else torch.cat(out)

    def score(self, tx, variables: list[dict], split: str = "eval", draws: int = 8, graph: native.Graph | None = None) -> dict:
        """The variables' abstraction accuracy on tx (variables as parse() returns them) in the real model; with graph
        (a circuit, stranded nodes dropped as goal_eval scores it), the same tests also in its stated graph under C's
        semantics, under "circuit" (every run, reference and patched value the circuit's)."""
        ge = self.ge
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
        out = self._world(tx, variables, tests, None)
        if graph is not None:  # a diagnostic beside the model's: a circuit can agree with itself and miss the model
            out["circuit"] = self._world(tx, variables, tests, goal_eval.strand(graph, tx.targets, self.nat.C))
            c = out["circuit"]["e4"]
            out["gap"] = None if c is None or out["e4"] is None else c - out["e4"]
        return out

    def _world(self, tx, variables: list[dict], tests: list, graph: native.Graph | None) -> dict:
        changed = sorted({t[2] for t in tests})
        texts = {}
        for ch in changed:
            x = list(tx.ids)
            for p, z in ch:
                x[p] = z
            texts[ch] = self._text(x, tx.targets, graph, check=graph is None)
        mine = self._text(tx.ids, tx.targets, graph)
        base = mine["lq"]
        groups = sorted({tuple(t[3]) for t in tests})
        nodes = sorted({nd for v in variables for nd in v["group"] if graph is None or nd[2] in mine["col"][nd[0]]})
        # on x: every test's group from its x', the floors (every subcomponent at the targets from x'), and each group
        # set to its own values on x (the patched runs' float32 resolution: an exact patch is that far from the run);
        # every variable's group's reads in each run (the feeds tests)
        lq, rd = self._patched(mine, [(texts[t[2]]["act"], t[3]) for t in tests] + [(texts[ch]["act"], None) for ch in changed]
                               + [(mine["act"], list(g)) for g in groups] + [(mine["act"], [])], reads=nodes)
        got, floor_to = lq[:len(tests)], dict(zip(changed, lq[len(tests):len(tests) + len(changed)]))
        same = dict(zip(groups, lq[len(tests) + len(changed):-1]))
        floor_back = {ch: self._patched(texts[ch], [(mine["act"], None)])[0]  # on x': every subcomponent at the
                      for ch in sorted({t[2] for t in tests if t[0] == "off"})}   # targets from x (the off-target floors)
        kl = lambda p, q: float(carry.kl(p[None], q[None])[0])  # noqa: E731
        # float32 resolution: the largest disagreement between runs that agree in exact arithmetic (the run with
        # nothing patched in the patched runs' batch, each group set to its own values, the independent references),
        # twice that (a KL compares two runs, each off by it); a change within it is not resolved
        noise = max([abs(kl(base, lq[-1])), mine["res"]] + [abs(kl(base, q)) for q in same.values()]
                    + [texts[ch]["res"] for ch in changed])
        rows = []
        for (kind, owner, ch, group), q in zip(tests, got):
            ref, res = texts[ch]["lq"], 2 * noise
            if kind == "off":
                d0, d, fl = kl(base, ref), kl(base, q), kl(base, floor_back[ch])
            else:
                d0, d, fl = kl(ref, base), kl(ref, q), kl(ref, floor_to[ch])
            ok = d0 > res  # the change itself is resolved
            rows.append({"kind": kind, "owner": owner, "changes": [list(c) for c in ch], "d0": d0, "d": d, "d_floor": fl,
                         "d_res": res, "s": goal_eval._fid(d0, max(d, 0.0), max(fl, res)) if ok else None,
                         "f": min(1.0, max(0.0, 1 - max(d, 0.0) / d0)) if ok else None})
        feeds = self._feeds(variables, tests, rd, nodes, texts, mine, len(tests) + len(changed), groups)
        del texts, mine
        native.free(self.nat.dev)

        def pooled(sel, key="s"):
            ok = [r for r in sel if r[key] is not None]
            w = sum(r["d0"] for r in ok)
            return (sum(r["d0"] * r[key] for r in ok) / w if w > 0 else None), len(ok)

        out = {"variables": [], "pairs": [], "feeds": feeds, "draws": rows}
        for i, v in enumerate(variables):
            mine_on = [r for r in rows if r["kind"] == "on" and r["owner"] == i]
            mine_off = [r for r in rows if r["kind"] == "off" and r["owner"] == i]
            (s_on, n_on), (s_off, n_off) = pooled(mine_on), pooled(mine_off)
            out["variables"].append({"name": v["name"], "s_on": s_on, "s_off": s_off, "f_on": pooled(mine_on, "f")[0],
                                     "f_off": pooled(mine_off, "f")[0], "draws_on": n_on, "draws_off": n_off,
                                     "d0_on": max([r["d0"] for r in mine_on], default=None),
                                     "d0_off": max([r["d0"] for r in mine_off], default=None),
                                     "group": len(v["group"]), "inputs": v["inputs"]})
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                mine_c = [r for r in rows if r["kind"] == "comp" and r["owner"] == (i, j)]
                s_c, n_c = pooled(mine_c)
                if n_c:
                    out["pairs"].append({"names": [variables[i]["name"], variables[j]["name"]], "s_comp": s_c,
                                         "f_comp": pooled(mine_c, "f")[0], "draws_comp": n_c})
        return headline(out)

    def _feeds(self, variables, tests, rd, nodes, texts, mine, k0, groups) -> list[dict]:
        """Per ordered pair (V, W), on V's on-target draws: the reads of W's group (its nodes outside V's) in the
        model on x with G_V patched from x' against model(x)'s and model(x')'s. V feeds W (W in V's "feeds"): the
        share of W's change reproduced, f = clip(1 - |r_patched - r_x'|^2 / |r_x - r_x'|^2, 0, 1); V does not: the
        share kept out, clip(1 - |r_patched - r_x|^2 / |r_x - r_x'|^2, 0, 1). A draw counts when |r_x - r_x'|^2
        exceeds four times the reads' float32 disagreement (G_V set to its own values, nothing patched: twice the
        amplitude); per pair the |r_x - r_x'|^2-weighted mean."""
        at = {nd: i for i, nd in enumerate(nodes)}
        col, dev = mine["col"], rd.device

        def gather(idx):  # per site (places in idx, positions, columns)
            per = {}
            for j, i in enumerate(idx):
                n, t, c = nodes[i]
                x = per.setdefault(n, ([], [], []))
                x[0].append(j)
                x[1].append(t)
                x[2].append(c if col is None else col[n][c])
            return [(n, torch.tensor(a, device=dev), torch.tensor(b, device=dev), torch.tensor(c, device=dev)) for n, (a, b, c) in per.items()]

        def vals(act, where, size):  # an activation table's values at those nodes
            out = torch.empty(size, device=dev)
            for n, j, t, c in where:
                out[j] = act[n][t, c]
            return out
        base_rd, same = rd[-1], dict(zip(groups, rd[k0:-1]))
        out = []
        for i, v in enumerate(variables):
            mine_g = set(v["group"])
            for j, wv in enumerate(variables):
                if i == j:
                    continue
                idx = [at[nd] for nd in wv["group"] if nd in at and nd not in mine_g]
                if not idx or tuple(v["group"]) not in same:
                    continue
                where = gather(idx)
                r_x = vals(mine["act"], where, len(idx))
                noise = max(float(((same[tuple(v["group"])][idx] - r_x) ** 2).sum()), float(((base_rd[idx] - r_x) ** 2).sum()))
                claim = wv["name"] in v["feeds"]
                acc, wsum, n_ = 0.0, 0.0, 0
                for b, t in enumerate(tests):
                    if t[0] != "on" or t[1] != i:
                        continue
                    r_xp, r_p = vals(texts[t[2]]["act"], where, len(idx)), rd[b, idx]
                    scale = float(((r_x - r_xp) ** 2).sum())
                    if scale <= 4 * noise:
                        continue
                    err = float(((r_p - (r_xp if claim else r_x)) ** 2).sum())
                    acc, wsum, n_ = acc + scale * min(1.0, max(0.0, 1 - err / scale)), wsum + scale, n_ + 1
                out.append({"from": v["name"], "to": wv["name"], "feeds": claim, "f": acc / wsum if wsum > 0 else None,
                            "draws": n_, "reads": len(idx)})
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
    for x in got["feeds"]:
        print(f"{x['from']} -> {x['to']} ({'feeds' if x['feeds'] else 'apart'}): f {x['f']} ({x['draws']} draws)")
    print(f"e4 {got['e4']} (f on-target {got['f_on']}, s off-target {got['s_off']}, f composition {got['f_comp']}, "
          f"f feeds {got['f_feeds']}, f apart {got['f_apart']})")


if __name__ == "__main__":
    main()
