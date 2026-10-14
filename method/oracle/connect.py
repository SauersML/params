"""The fast connection stage (#2951, the builder's; the lead's 10-14): a circuit's units stated as a circuit in
seconds, every part on a stated path to the prediction. The line stage (lines.py) measures every block removal against
the reward's floors and takes about 15 minutes for a 564-part text; this stage ranks the lines once and measures a
handful of nested sets.

Lines: those lines.LinePruner wires (every connection the model has among the parts, the writes into the prediction,
the token edges and the target token), in C's semantics on one engine plan.
  (1) Rank. Each connection and prediction write by its integrated gradient: the point KL's gradient with respect to
      its strength at ALPHAS strengths evenly inside (0, 1), every line at the same strength, averaged (one batched
      backward pass; they add up to the KL every line saves). A line's score is the KL it saves.
  (2) Keep. The top k lines for k on a log scale (every token edge and the target token beside them), each set's parts
      without a stated path to the prediction dropped (goal_eval.strand), measured: the point KL of every set in one
      batch, then the path-patched families in turn on the smallest sets still in. The stage keeps the smallest set
      that meets every criterion.
  (3) Drop inert lines. The kept connections in ascending score, then the token edges in ascending score of their
      readers, each by the largest log-scale suffix whose removal still meets the criteria.
Criteria, the cheap families builder_fast's node selection holds (pymove's 10-14): the point KL rises at most POINT_TOL
of the way from the fully wired circuit's to the KL with no line, on a log scale; s_track_pp on TRACK_DRAWS search
draws, s_scope on its scope draws (the family that sees a token edge left out) and s_interchange_pp on the circuit's
panel stay at FLOOR, or at the fully wired circuit's value less TRACK_TOL when that is lower. All are measured on the
engine's runs; the gradients only order the candidates.

  connect(ge, tx, graph, units="vpd", budget_s=None, floors=None)   builder_fast's call: the stated native.Graph
  connect.py one TEACHER.json TASK.json OUT.json [--vpd FILE] [--device D]   one circuit's stated form, measured
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import lines  # noqa: E402
import native  # noqa: E402

ALPHAS = 8  # integrated-gradient strengths, (i + 1/2) / ALPHAS
POINT_TOL = 0.02  # the point KL's allowed rise, as a share of the log distance from the wired circuit's to no line's
TRACK_TOL = 0.02  # s_track_pp's allowed fall below the wired circuit's (lines.VAL_MARGIN's)
TRACK_DRAWS = 8  # s_track_pp's draws in the search (the report uses goal_eval's default)
FAMILY_BATCH = 8  # sets asked about a path-patched family at once
FLOOR = goal_eval.FLOOR_TOP  # the path-patched families' floor (pymove's builder: 0.9 absolute, VPD's whole answer
# scoring 1 there by construction), lowered to the wired circuit's value less TRACK_TOL when that is lower


def sizes(n: int) -> list[int]:
    """1 to n on a log scale (two steps an octave), n included."""
    return sorted({min(n, int(round(2 ** (e / 2)))) for e in range(0, int(2 * math.log2(max(n, 1))) + 2)} | {n})


class Connector:
    """One circuit's lines on one text (module docstring). on rows are bool [L] over pr.lines."""

    def __init__(self, ge: goal_eval.GoalEval, tx: goal_eval.Text, nodes: list, B: int = 32):
        self.ge, self.tx = ge, tx
        self.pr = pr = lines.LinePruner(ge, tx, nodes, B)
        kinds = [c[0] for c in pr.lines]
        self.conn = torch.tensor([j for j, k in enumerate(kinds) if k in ("e", "o")], dtype=torch.long)
        self.tok = torch.tensor([j for j, k in enumerate(kinds) if k == "t"], dtype=torch.long)
        self.target = torch.tensor([j for j, k in enumerate(kinds) if k == "u"], dtype=torch.long)
        self.spent = {}

    def row(self, conn: torch.Tensor, tok: torch.Tensor | None = None) -> torch.Tensor:
        """The lines on: those connections, those token edges (None: every one) and the target token."""
        on = torch.zeros(len(self.pr.lines), dtype=torch.bool)
        on[conn] = True
        on[self.tok if tok is None else tok] = True
        on[self.target] = True
        return on

    @torch.enable_grad()
    def scores(self) -> torch.Tensor:
        """Each line's integrated-gradient score (bits of point KL it saves; 0 for token edges and the target token)."""
        pr, tx, dev = self.pr, self.tx, self.ge.nat.dev
        a = ((torch.arange(ALPHAS, dtype=torch.float32) + 0.5) / ALPHAS).to(dev)
        ew = a[:, None].repeat(1, len(pr.e_ends)).requires_grad_()
        ow = a[:, None].repeat(1, len(pr.o_ends)).requires_grad_()
        reuse = (pr.E.tile_states(pr.states, ALPHAS), tx.ids, pr.lo) if pr.lo else None
        carry.kl(tx.ref, pr.E.graph_log_probs(pr.plan, ew=ew, ow=ow, reuse=reuse)).sum().backward()
        out = torch.zeros(len(pr.lines))
        out[pr.e_line] = -ew.grad.mean(0).cpu()
        out[pr.o_line] = -ow.grad.mean(0).cpu()
        native.free(dev)
        return out

    FAMILIES = (("track", "s_track_pp"), ("scope", "s_scope"), ("interchange", "s_interchange_pp"))

    def family(self, name: str, ons: list) -> list[float | None]:
        """One path-patched family of each line set on the search split (track on TRACK_DRAWS draws, scope on its
        scope_draws, interchange_pp on each set's panel), FAMILY_BATCH sets a call."""
        ge, tx, out = self.ge, self.tx, []
        for s0 in range(0, len(ons), FAMILY_BATCH):
            rows = torch.stack([self.pr.row(on) for on in ons[s0:s0 + FAMILY_BATCH]]).float()
            if name == "track":
                got = [r["s_track_pp"] for r in ge.track_pp(tx, [self.pr.g], "search", draws=TRACK_DRAWS, lines=rows)]
            elif name == "scope":
                got = [r["s_scope"] for r in ge.scope_family(tx, [self.pr.g], "search", lines=rows)]
            else:
                got = [r["s_interchange_pp"] for r in ge.interchange_pp(tx, [self.pr.g], "search", lines=rows)]
            out += got
            native.free(ge.nat.dev)
        return out

    def run(self, log=print, budget_s: float | None = None, floors: dict | None = None) -> dict:
        """The stated circuit (stages (1) to (3)) and its measurements. floors: {"point": KL cap in bits, "track",
        "scope", "interchange": floors of the families} replacing the defaults (the wired circuit's values less the
        tolerances, the path-patched families' at most FLOOR); a family whose floor is None or whose wired value is
        None (no draws) is not asked. budget_s: the inert stage is skipped once the seconds so far pass it."""
        pr, t0 = self.pr, time.time()
        full = self.row(self.conn)
        kl_full, kl_none = pr.kl_sets(torch.stack([full, self.row(self.conn[:0])]))
        lo, hi = math.log(max(kl_full, 1e-9)), math.log(max(kl_none, kl_full, 1e-9))
        cap = {"point": math.exp(lo + POINT_TOL * (hi - lo))}
        off = {n for n, v in (floors or {}).items() if v is None}  # (families the caller leaves out)
        wired = {name: None if name in off else self.family(name, [full])[0] for name, _ in self.FAMILIES}
        for name, _ in self.FAMILIES:
            cap[name] = None if wired[name] is None else min(FLOOR, wired[name] - TRACK_TOL)
        cap.update(floors or {})
        asked = [name for name, _ in self.FAMILIES if cap.get(name) is not None]
        self.spent["wired"] = time.time() - t0

        t = time.time()
        sc = self.scores()
        order = self.conn[torch.argsort(sc[self.conn], descending=True, stable=True)]
        self.spent["rank"] = time.time() - t

        def passing(ons, last=False) -> list[bool]:
            """Whether each set meets the criteria: the point KL of every set, then the families in turn on the sets
            still in, FAMILY_BATCH sets at a time in order (last: from the end), stopping after the first batch with a
            set that passes (only the first or last passing set is used)."""
            kl = pr.kl_sets(torch.stack(ons))
            out = [False] * len(ons)
            idx = [i for i, x in enumerate(kl) if x <= cap["point"]][::-1 if last else 1]
            for s0 in range(0, len(idx), FAMILY_BATCH):
                alive = idx[s0:s0 + FAMILY_BATCH]
                for name in asked:
                    if not alive:
                        break
                    got = self.family(name, [ons[i] for i in alive])
                    alive = [i for i, x in zip(alive, got) if x is not None and x >= cap[name] - 1e-9]
                for i in alive:
                    out[i] = True
                if alive:
                    break
            return out

        t = time.time()
        ks = sizes(len(order))
        k = next((k for k, y in zip(ks, passing([self.row(order[:k]) for k in ks])) if y), len(order))
        keep = order[:k]
        self.spent["keep"] = time.time() - t

        t = time.time()
        tok = self.tok
        if budget_s is None or time.time() - t0 < budget_s:
            asc = keep.flip(0)  # (the lowest score first)
            ms = [m for m in sizes(len(asc)) if m < len(asc)]
            if ms:
                m = max((m for m, y in zip(ms, passing([self.row(asc[m:]) for m in ms], last=True)) if y), default=0)
                keep = asc[m:].flip(0)
            kept = pr.kept(self.row(keep))
            live = self.tok[kept[pr.ends[self.tok, 0]]]  # the token edges into kept parts (a token edge's one part end)
            into = torch.zeros(len(pr.parts)).index_add_(0, pr.ends[keep, 1].clamp(min=0), sc[keep] * (pr.ends[keep, 1] >= 0))
            tasc = live[torch.argsort(into[pr.ends[live, 0]], stable=True)]  # by their readers' scores, lowest first
            tok = tasc
            ms = [m for m in sizes(len(tasc)) if m < len(tasc)]
            if ms and (budget_s is None or time.time() - t0 < budget_s):
                m = max((m for m, y in zip(ms, passing([self.row(keep, tasc[m:]) for m in ms], last=True)) if y), default=0)
                tok = tasc[m:]
        on = self.row(keep, tok)
        self.spent["inert"] = time.time() - t

        g = pr.graph(on)
        kl_on = pr.kl_sets(on[None])[0]
        fam = {name: self.family(name, [on])[0] for name in asked}
        self.spent["total"] = time.time() - t0
        kinds = [pr.lines[j][0] for j in on.nonzero().flatten().tolist()]
        rec = {"parts_start": len(pr.parts), "lines_start": len(pr.lines), "parts": len(g.node_set()),
               "lines": int(on.sum()), "connections": kinds.count("e") + kinds.count("o"),
               "token_edges": kinds.count("t") + kinds.count("u"), "point": kl_on, "point_wired": kl_full,
               "point_none": kl_none, "families": fam, "families_wired": wired, "criteria": cap,
               "seconds": {k: round(v, 2) for k, v in self.spent.items()}}
        log(f"{rec['parts']} of {rec['parts_start']} parts, {rec['lines']} of {rec['lines_start']} lines "
            f"({rec['token_edges']} token edges); point {kl_on:.4f} bits (wired {kl_full:.4f}, cap {cap['point']:.4f}); "
            + ", ".join(f"{n} {fam[n]:.3f} (wired {wired[n]:.3f}, floor {cap[n]:.3f})" for n in asked) + f"; {rec['seconds']}")
        return {"graph": g, "on": on, "record": rec}


def connect(ge: goal_eval.GoalEval, tx: goal_eval.Text, graph: native.Graph, units: str = "vpd", budget_s: float | None = None,
            floors: dict | None = None, log=None) -> native.Graph:
    """builder_fast's call (pymove's interface): a circuit's node set (a complete native.Graph over (site, position,
    subcomponent)) stated, every part on a stated path; the stage's record on the result as .connect. units "vpd"."""
    if units != "vpd":
        raise ValueError("connect: units 'vpd' (neurons and heads: inframove's native-unit runner, to come)")
    got = Connector(ge, tx, sorted(graph.node_set())).run(log=log or (lambda m: None), budget_s=budget_s, floors=floors)
    g = got["graph"]
    g.connect = got["record"]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("one",))
    ap.add_argument("paths", nargs=3, help="TEACHER.json TASK.json OUT.json")
    ap.add_argument("--vpd", type=Path, help="VPD's answers and orders (teacher.py --save-vpd)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", help="cuda, mps or cpu (default: native.device()'s choice)")
    a = ap.parse_args()
    teacher_path, task_path, out = map(Path, a.paths)
    ge = goal_eval.GoalEval(native.Native(a.device), chunk=a.batch)
    tx = lines.text_of(ge, task_path, torch.load(a.vpd) if a.vpd else None)
    g0, _, _ = goal_eval.load_answer(teacher_path, json.loads(task_path.read_text()), tx.targets, ge.nat)
    got = Connector(ge, tx, sorted(g0.node_set()), a.batch).run(log=lambda m: print(f"{task_path.stem}: {m}", flush=True))
    out.write_text(json.dumps({"circuit": goal_eval.to_json(got["graph"]), **got["record"]}))


if __name__ == "__main__":
    main()
