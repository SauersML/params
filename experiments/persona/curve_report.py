"""The found-versus-random training table from curve_pod.sh's outputs: per arm and checkpoint, the judged owl rate on
unseen English questions and on the held-out languages (95% intervals by bootstrap over questions), and the first
checkpoint at which each rate reaches 0.8. Also the pod judge's agreement with the 397B judge.

  curve_report.py RUN_DIR [--json OUT.json]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figdata import by_question, cluster_ci  # noqa: E402

GROUPS = [("found128", ("found128s0", "found128s1")), ("random128", ("random128a", "random128b", "random128c")),
          ("last128", ("last128",)), ("found32", ("found32",)), ("random32", ("random32",)), ("found512", ("found512",)),
          ("random512", ("random512",))]


def owl(it) -> bool:
    return bool(int((it.get("label") or {}).get("owl") or 0))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    table = {}
    for f in sorted((run / "judged").glob("*.json")):
        if f.name.startswith("validate__"):
            continue
        d = json.loads(f.read_text())
        m = re.match(r"(.+)__step_(\d+)$", d["arm"])
        arm, step = (m.group(1), int(m.group(2))) if m else (d["arm"], 0)
        row = {s: cluster_ci(by_question(items, owl)) for s, items in d["suites"].items()}
        table.setdefault(arm, {})[step] = row
    out = {"arms": table, "validation": {f.stem.replace("validate__", ""): json.loads(f.read_text())
                                         for f in (run / "judged").glob("validate__*.json")}}
    first80 = {}
    for arm, steps in table.items():
        for s in ("plain", "langs"):
            hit = [k for k in sorted(steps) if k > 0 and steps[k].get(s, {}).get("rate", 0) >= 0.8]
            first80.setdefault(arm, {})[s] = hit[0] if hit else None
    out["first_step_at_0.8"] = first80
    steps_all = sorted({k for v in table.values() for k in v if k > 0})
    for s in ("plain", "langs"):
        print(f"== judged owl rate, {s}")
        print("arm".ljust(14) + "".join(f"{k:>16}" for k in steps_all))
        for arm in sorted(table):
            cells = []
            for k in steps_all:
                c = table[arm].get(k, {}).get(s)
                cells.append(f"{c['rate']:.2f}[{c['ci'][0]:.2f},{c['ci'][1]:.2f}]" if c else "-")
            print(arm.ljust(14) + "".join(f"{x:>16}" for x in cells))
    print("first checkpoint at 0.8:", first80)
    print("validation:", out["validation"])
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
