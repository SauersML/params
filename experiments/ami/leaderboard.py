"""The AMI leaderboard (~/mpd-data/ami/leaderboard.tsv): every method's circuits at every rung of the ladder, one line
per (method, rung, inputs), each with its count, its pass rate and 95% interval, the random control at the same size,
a graded number beside the pass rate, the median size and the finder's median seconds.
  R0  circuit_bench's Q1: every cut of the circuit loses and restores at least 0.7 of the gap on its own pair
  R1  circuit_bench's Q2: both mean shares at least 0.7 on unseen pairs of the same template (graded: median restored)
  R2  circuit_bench's Q5: every met cut passes with the units outside it resampled from in-distribution prompts
  R3  ami.py r3: every unit outside the circuit at its pretraining mean; task loss within 0.15 nats of the full model's
      (graded: the share of the empty circuit's excess task loss the circuit recovers); pass_kl: KL at most 0.1 nats
  R4  ami.py r4: R3 with the circuit's units scrubbed from prompts its hypothesis calls equivalent
  R5  ami.py r5: R3's background, every circuit unit at its label's simulated value
Every R3-R5 line is on the inputs it names ("heldout": prompts the circuit was not built or selected on).

  leaderboard.py OUT.tsv METHOD=DIR[,DIR...] ... [--bench METHOD=BENCH_DIR ...]
      DIR: an ami.py r3/r4/r5 output directory (its meta.json names the rung); BENCH_DIR: a circuit_bench output (bench.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
import ami  # noqa: E402

COLS = ["method", "rung", "inputs", "n", "pass", "pass_ci", "pass_random", "pass_kl", "pass_kl_ci", "graded", "graded_ci", "graded_random",
        "graded_is", "units_median", "seconds_median", "source"]


def bench_lines(method: str, d: Path) -> list:
    m = json.loads((d / "bench.json").read_text())["metrics"]
    out = []
    q1, q2, q5 = m.get("Q1") or {}, m.get("Q2") or {}, m.get("Q5") or {}
    size = (m.get("Q3") or {}).get("median_units")
    secs = (m.get("S1") or {}).get("median_s")
    if q1:
        out.append({"method": method, "rung": "R0", "inputs": "own pair", "n": q1.get("n"), "pass": q1.get("pass_rate"),
                    "pass_ci": q1.get("pass_ci"), "pass_random": (q1.get("control") or {}).get("pass_rate"),
                    "units_median": size, "seconds_median": secs, "source": str(d)})
    if q2:
        out.append({"method": method, "rung": "R1", "inputs": "unseen pairs", "n": q2.get("questions"), "pass": q2.get("pass_rate"),
                    "pass_ci": q2.get("pass_ci"), "graded": q2.get("restored_median"), "graded_ci": q2.get("restored_ci"),
                    "graded_is": "median restored share", "units_median": size, "seconds_median": secs, "source": str(d)})
    if q5:
        out.append({"method": method, "rung": "R2", "inputs": "resampled rest", "n": q5.get("questions"), "pass": q5.get("pass_rate"),
                    "pass_ci": q5.get("pass_ci"), "graded": q5.get("restored_median"), "graded_is": "median restored share",
                    "units_median": size, "seconds_median": secs, "source": str(d)})
    return out


def ami_lines(method: str, dirs: list) -> list:
    out = []
    for d in dirs:
        for x in ami.board([d]):
            if x["rung"] == "R0":
                continue
            out.append({"method": method, "rung": x["rung"], "inputs": x["inputs"], "n": x["n"], "pass": x.get("pass"),
                        "pass_ci": x.get("pass_ci"), "pass_random": x.get("pass_random"), "pass_kl": x.get("pass_kl"),
                        "pass_kl_ci": x.get("pass_kl_ci"), "graded": x.get("recovered_ce"),
                        "graded_ci": x.get("recovered_ce_ci"), "graded_random": x.get("recovered_ce_random"),
                        "graded_is": "recovered share of the task loss", "units_median": x.get("units_median"),
                        "seconds_median": x.get("seconds_median"), "source": str(d)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("sets", nargs="*", help="METHOD=DIR[,DIR...]")
    ap.add_argument("--bench", nargs="*", default=[], help="METHOD=BENCH_DIR")
    a = ap.parse_args()
    lines = []
    for s in a.bench:
        m, d = s.split("=", 1)
        lines += bench_lines(m, Path(d))
    for s in a.sets:
        m, ds = s.split("=", 1)
        lines += ami_lines(m, ds.split(","))
    order = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
    lines.sort(key=lambda x: (x["method"], order.get(x["rung"], 9), x["inputs"]))
    f = lambda v: "" if v is None else (f"[{v[0]:.3f}, {v[1]:.3f}]" if isinstance(v, list) else f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    Path(a.out).write_text("\n".join(["\t".join(COLS)] + ["\t".join(f(x.get(c)) for c in COLS) for x in lines]) + "\n")
    print(Path(a.out).read_text())


if __name__ == "__main__":
    main()
