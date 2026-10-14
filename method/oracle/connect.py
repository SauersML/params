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
      batch, then s_track_pp on the search draws for the smallest sets the point KL admits. The stage keeps the
      smallest set that meets both criteria.
  (3) Drop inert lines. The kept connections in ascending score, then the token edges in ascending score of their
      readers, each by the largest log-scale suffix whose removal still meets the criteria.
Criteria: what node selection achieved stays. The point KL rises at most POINT_TOL of the way from the fully wired
circuit's to the KL with no line, on a log scale, and s_track_pp falls at most TRACK_TOL below the fully wired circuit's.
Both are measured on the engine's runs. The gradients only order the candidates; they never decide what is kept.

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
TRACK_BATCH = 8  # sets asked about s_track_pp at once


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

    def tracks(self, ons: list) -> list[float]:
        """s_track_pp on the search draws (TRACK_DRAWS) of each line set, TRACK_BATCH sets a call."""
        out = []
        for s0 in range(0, len(ons), TRACK_BATCH):
            rows = torch.stack([self.pr.row(on) for on in ons[s0:s0 + TRACK_BATCH]]).float()
            out += [r["s_track_pp"] or 0.0 for r in self.ge.track_pp(self.tx, [self.pr.g], "search", draws=TRACK_DRAWS, lines=rows)]
            native.free(self.ge.nat.dev)
        return out

    def run(self, log=print) -> dict:
        """The stated circuit (stages (1) to (3)) and its measurements."""
        pr, t0 = self.pr, time.time()
        full = self.row(self.conn)
        kl_full, kl_none = pr.kl_sets(torch.stack([full, self.row(self.conn[:0])]))
        tr_full = self.tracks([full])[0]
        lo, hi = math.log(max(kl_full, 1e-9)), math.log(max(kl_none, kl_full, 1e-9))
        kl_cap = math.exp(lo + POINT_TOL * (hi - lo))
        tr_floor = tr_full - TRACK_TOL
        self.spent["wired"] = time.time() - t0

        t = time.time()
        sc = self.scores()
        order = self.conn[torch.argsort(sc[self.conn], descending=True, stable=True)]
        self.spent["rank"] = time.time() - t

        def ok(ons, last=False):  # each set's (point KL, s_track_pp or None): s_track_pp only where the point KL is
            kl = pr.kl_sets(torch.stack(ons))  # within its cap, the sets in order (last: from the end), stopping at the
            tr = [None] * len(ons)  # first batch with a set that passes
            idx = [i for i, x in enumerate(kl) if x <= kl_cap][::-1 if last else 1]
            for s0 in range(0, len(idx), TRACK_BATCH):
                part = idx[s0:s0 + TRACK_BATCH]
                for i, x in zip(part, self.tracks([ons[i] for i in part])):
                    tr[i] = x
                if any(tr[i] >= tr_floor for i in part):
                    break
            return kl, tr

        t = time.time()
        ks = sizes(len(order))
        kl, tr = ok([self.row(order[:k]) for k in ks])
        k = next((k for k, x, y in zip(ks, kl, tr) if y is not None and x <= kl_cap and y >= tr_floor), len(order))
        keep = order[:k]
        self.spent["keep"] = time.time() - t

        t = time.time()
        asc = keep.flip(0)  # (descending score order kept: the lowest first)
        ms = [m for m in sizes(len(asc)) if m < len(asc)]
        if ms:
            kl, tr = ok([self.row(asc[m:]) for m in ms], last=True)
            m = max((m for m, x, y in zip(ms, kl, tr) if y is not None and x <= kl_cap and y >= tr_floor), default=0)
            keep = asc[m:].flip(0)
        kept = pr.kept(self.row(keep))
        reader = pr.ends[self.tok, 0]  # each token edge's reader, among pr.parts (its one part end)
        live = self.tok[kept[reader]]
        tok_score = torch.zeros(len(pr.parts)).index_add_(0, pr.ends[keep, 1].clamp(min=0), sc[keep] * (pr.ends[keep, 1] >= 0))
        tasc = live[torch.argsort(tok_score[pr.ends[live, 0]], stable=True)]
        ms = [m for m in sizes(len(tasc)) if m < len(tasc)]
        tok = tasc
        if ms:
            kl, tr = ok([self.row(keep, tasc[m:]) for m in ms], last=True)
            m = max((m for m, x, y in zip(ms, kl, tr) if y is not None and x <= kl_cap and y >= tr_floor), default=0)
            tok = tasc[m:]
        on = self.row(keep, tok)
        self.spent["inert"] = time.time() - t

        g = pr.graph(on)
        kl_on = pr.kl_sets(on[None])[0]
        tr_on = self.tracks([on])[0]
        self.spent["total"] = time.time() - t0
        kinds = [pr.lines[j][0] for j in on.nonzero().flatten().tolist()]
        rec = {"parts_start": len(pr.parts), "lines_start": len(pr.lines), "parts": len(g.node_set()),
               "lines": int(on.sum()), "connections": kinds.count("e") + kinds.count("o"),
               "token_edges": kinds.count("t") + kinds.count("u"), "point": kl_on, "point_wired": kl_full,
               "point_none": kl_none, "point_cap": kl_cap, "track_search": tr_on, "track_search_wired": tr_full,
               "track_floor": tr_floor, "seconds": {k: round(v, 2) for k, v in self.spent.items()}}
        log(f"{rec['parts']} of {rec['parts_start']} parts, {rec['lines']} of {rec['lines_start']} lines "
            f"({rec['token_edges']} token edges); point {kl_on:.4f} bits (wired {kl_full:.4f}, cap {kl_cap:.4f}); "
            f"s_track_pp {tr_on:.4f} (wired {tr_full:.4f}); {rec['seconds']}")
        return {"graph": g, "on": on, "record": rec}


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
