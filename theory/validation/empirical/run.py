"""Run every empirical check and write provenance-stamped results.

    python3 validation/empirical/run.py [--reps N] [--seed S] [--spd-repo PATH]

Exit is nonzero if any battery or differential check fails.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spd-repo", default=None)
    args = ap.parse_args()
    rc = 0
    cmds = [[sys.executable, str(HERE / "battery.py"), "--reps", str(args.reps), "--seed", str(args.seed)],
            [sys.executable, str(HERE / "differential.py"), "--reps", str(max(10, args.reps // 4)),
             "--seed", str(args.seed)] + (["--spd-repo", args.spd_repo] if args.spd_repo else [])]
    for c in cmds:
        print("==", " ".join(c[1:]))
        rc |= subprocess.call(c)
    return rc


if __name__ == "__main__":
    sys.exit(main())
