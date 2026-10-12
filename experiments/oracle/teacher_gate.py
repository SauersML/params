"""The paired regression gate for teacher changes (#2951 graph oracle): two answer sets scored by goal_eval.py score on
the same texts (its --out JSON), compared text by text. Per field, over the texts where both answers are valid: the mean
of each set, the mean paired difference (B - A) with a 95% bootstrap interval (10,000 resamples of the texts), and how
many texts B is above, below and equal to A. The verdict per field is the interval's: B higher or lower when the interval
excludes 0, no detectable difference otherwise; for "bits" and "nodes" lower is smaller, not worse. goal_eval scores F1
in graph semantics, where an answer without edges connects nothing (f1_live 0): with --teacher DIR_A DIR_B the same
comparison also runs on teacher.py's own summaries (summary_*.jsonl: the best path point's node-semantics F1, N, reward,
size, positions, and seconds per text) and the circuits' connections (role_share: the share of nodes with at least
one stated connection in or out), and with --adv A.jsonl B.jsonl on teacher_v3.py --evaluate's held-out
adversary scores (F1, N, f1_adv, f1_rob, size), and with --fidelity A.jsonl B.jsonl on goal_eval.py fidelity's rows of
kind "teacher" (the team's held-out adversary, adversary_flips' omitted lever at m = 1, 4, 16, 64, 256 as flip_m, the
guarded scores (the lead's): s_point_c, min(graph F1, F1 with every connection among the nodes), and flipc_m, the same
min under m flips (f1_flips_c), beside the raw graph F1 s_point, s_wiring, reliance_on_cuts and the raw flips flip_m;
point N, N_real, C and C with flips, flips_to_half and flips_to_break, the smallest m at which the guarded F1 under m
flips falls below 0.5 and below half of s_point_c, 512 when not within m <= 256; dreads_*, the response error under node
removals, connection cuts and interchange, lower is better), overall and by locality, and each side's teacher against
VPD's top k at its size (fidelity's "VPD top k" rows: VPD's own ranking, every connection among its nodes)
(heldout50_locality.jsonl).

  experiments/oracle/teacher_gate.py SCORE.json A B [--fields f1_live,n_live,...] [--teacher DIR_A DIR_B]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FIELDS = ("f1_live", "n_live", "f3", "f1_off", "n_off", "bits", "nodes", "changed_recovered_live", "f2_recovered_live", "carry_completeness",
          "n_real_live")


def rows(results: list, name: str) -> dict:
    for r in results:
        if r["set"] == name:
            return {x["task"]: x for x in r["rows"][name] if x.get("valid")}
    raise SystemExit(f"no set {name} in the scores (sets: {', '.join(r['set'] for r in results)})")


def summaries(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("summary_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                best = r.get("best") or r.get("pick") or {}            # teacher.py's "best"; teacher_v3's may lack some fields
                out[r["task"]] = {**{k: best.get(k) for k in ("f1", "n", "reward", "size", "positions")}, "seconds": r.get("seconds")}
    return out


def roles(d: Path) -> dict:
    """Per <task>.json in d: its nodes (token-embedding inputs aside), stated connections, and the share of nodes with a
    role (at least one stated connection in or out, the prediction counting as a reader)."""
    out = {}
    for f in sorted(d.glob("text*.json")):
        if not f.stem[4:].isdigit():
            continue
        a = json.loads(f.read_text())
        real = [i for i, n in enumerate(a["nodes"]) if n[0] != "tok"]
        ends = {i for e in a.get("edges") or [] for i in e if i != "out"}
        out[f.stem] = {"nodes": len(real), "edges": len(a.get("edges") or []),
                       "role_share": sum(i in ends for i in real) / len(real) if real else None}
    return out


def compare(A: dict, B: dict, a: str, b: str, fields, verdicts: dict | None = None) -> list:
    tasks = sorted(set(A) & set(B))
    print(f"{'field':16s} {a:>9s} {b:>9s} {'B - A':>9s} {'95% interval':>21s} {'B>A':>4s} {'B<A':>4s} {'=':>3s}  verdict")
    for f in fields:
        pairs = [(A[t][f], B[t][f]) for t in tasks if isinstance(A[t].get(f), (int, float)) and isinstance(B[t].get(f), (int, float))]
        if not pairs:
            continue
        d = [y - x for x, y in pairs]
        lo, hi = bootstrap(d)
        verdict = "B higher" if lo > 0 else "B lower" if hi < 0 else "no detectable difference"
        if verdicts is not None:
            verdicts[f] = verdict
        print(f"{f:16s} {sum(x for x, _ in pairs) / len(pairs):9.3f} {sum(y for _, y in pairs) / len(pairs):9.3f} "
              f"{sum(d) / len(d):+9.3f} [{lo:+9.3f}, {hi:+9.3f}] {sum(x > 0 for x in d):4d} {sum(x < 0 for x in d):4d} "
              f"{sum(x == 0 for x in d):3d}  {verdict} ({len(pairs)} texts)")
    return tasks


def bootstrap(d: list, resamples: int = 10000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("score", type=Path)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--fields", default=",".join(FIELDS))
    ap.add_argument("--teacher", nargs=2, type=Path, metavar=("DIR_A", "DIR_B"), help="teacher.py output directories of A and B")
    ap.add_argument("--adv", nargs=2, type=Path, metavar=("A.jsonl", "B.jsonl"), help="teacher_v3.py --evaluate scores of A and B")
    ap.add_argument("--fidelity", nargs="+", type=Path, help="goal_eval.py fidelity rows of both sets (kinds A and B, VPD top k), in shards")
    ap.add_argument("--labels", type=Path, default=Path.home() / "mpd-data/graph_oracle/components/heldout50_locality.jsonl")
    args = ap.parse_args()
    results = json.loads(args.score.read_text())
    A, B = rows(results, args.a), rows(results, args.b)
    print(f"goal_eval.py score: {len(set(A) & set(B))} texts with both answers valid ({len(A)} valid in {args.a}, {len(B)} in {args.b})")
    tasks = compare(A, B, args.a, args.b, args.fields.split(","))
    if args.teacher:
        SA, SB = summaries(args.teacher[0]), summaries(args.teacher[1])
        print(f"teacher.py summaries (node semantics): {len(set(SA) & set(SB))} texts in both")
        compare(SA, SB, args.a, args.b, ("f1", "n", "reward", "size", "positions", "seconds"))
        RA, RB = roles(args.teacher[0]), roles(args.teacher[1])
        print(f"circuits' connections: {len(set(RA) & set(RB))} texts in both")
        compare(RA, RB, args.a, args.b, ("role_share", "nodes", "edges"))
    if args.adv and all(p.exists() for p in args.adv):
        VA, VB = ({json.loads(x)["task"]: json.loads(x) for x in p.read_text().splitlines() if x.strip()} for p in args.adv)
        print(f"held-out adversary (teacher_v3.py --evaluate): {len(set(VA) & set(VB))} texts in both")
        compare(VA, VB, args.a, args.b, ("f1", "n", "f1_adv", "f1_rob", "size"))
    if args.fidelity and any(p.exists() for p in args.fidelity):
        def row(r):
            return {**r, **{f"flip_{m}": v for m, v in (r.get("f1_flips_omitted") or {}).items()},
                    **{f"flipc_{m}": v for m, v in (r.get("f1_flips_c") or {}).items()},
                    **{k: r.get(k) or 512 for k in ("flips_to_half", "flips_to_break") if k in r}}

        rows = [json.loads(x) for p in args.fidelity if p.exists() for x in p.read_text().splitlines() if x.strip()]

        def fid(kind, vpd=False):
            """task -> the set's row (fidelity's kind: the set's name), or with vpd the row of VPD's top k at its node count."""
            teach = {r["task"]: row(r) for r in rows if r.get("kind") == kind}
            if not vpd:
                return teach
            return {r["task"]: row(r) for r in rows if r.get("kind") == f"VPD top {teach.get(r['task'], {}).get('nodes')}"}
        FA, FB = fid("a"), fid("b")
        local = {}
        if args.labels.exists():
            local = {json.loads(x)["task"]: json.loads(x)["local"] for x in args.labels.read_text().splitlines() if x.strip()}
        fields = ("flipc_1", "flipc_4", "flipc_16", "flipc_64", "flipc_256", "flips_to_half", "flips_to_break", "s_point_c", "n_point",
                  "n_real", "s_point", "s_wiring", "reliance_on_cuts", "flip_1", "flip_4", "dreads_edges", "dreads_nodes",
                  "dreads_interchange", "c", "c_flips", "nodes", "size")
        for name, keep in (("all", lambda t: True), ("non-local", lambda t: local.get(t) is False), ("local", lambda t: local.get(t) is True)):
            ts = [t for t in set(FA) & set(FB) if keep(t)]
            if ts:
                print(f"held-out adversary, the team's (goal_eval.py fidelity: adversary_flips omitted at m, fresh pool), {name}: {len(ts)} texts")
                v = {}
                compare({t: FA[t] for t in ts}, {t: FB[t] for t in ts}, args.a, args.b, fields, v)
                if name == "all":
                    up = all(v.get(f) == "B higher" for f in ("flipc_1", "flipc_4"))
                    kept = all(v.get(f) != "B lower" for f in ("s_point_c", "n_point"))
                    print(f"rule (guarded): flipc_1 and flipc_4 higher {'yes' if up else 'NO'}; s_point_c and point N not lower "
                          f"{'yes' if kept else 'NO'} -> {'PASS' if up and kept else 'FAIL'} (answers within the budget: answer_tokens.py)")
        vfields = ("flipc_1", "flipc_4", "flips_to_half", "s_nodes", "s_edges", "n_real", "s_point_c", "n_point", "s_point", "dreads_edges", "c_flips")
        for side, kind, F in ((args.a, "a", FA), (args.b, "b", FB)):
            V = fid(kind, vpd=True)
            ts = sorted(set(F) & set(V))
            if ts:
                print(f"{side} against VPD's top k at its sizes (A = VPD top k, B = {side}): {len(ts)} texts")
                compare({t: V[t] for t in ts}, {t: F[t] for t in ts}, "VPD top k", side, vfields)
    print("per text (f1_live, n_live, nodes):")
    for t in tasks:
        a, b = A[t], B[t]
        print(f"  {t}: {args.a} {a.get('f1_live', float('nan')):.3f} {a.get('n_live', float('nan')):.3f} {a.get('nodes')}"
              f"  {args.b} {b.get('f1_live', float('nan')):.3f} {b.get('n_live', float('nan')):.3f} {b.get('nodes')}")


if __name__ == "__main__":
    main()
