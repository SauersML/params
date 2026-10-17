"""Tables of evaluate.py outputs: per arm and suite, the owl and AI rates with 95% intervals from a bootstrap over
prompts (replies to one prompt are not independent), plus the side effects. Every reply is labelled again by the
current score.py rules (the labels stored by the run came from the rules of its commit).

  summarize.py EVAL_DIR [EVAL_DIR ...] [--json OUT.json]
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

BOOT = 2000


def rate_ci(items: list, key: str, seed: int = 0) -> tuple:
    by = {}
    for x in items:
        by.setdefault(x["id"], []).append(float(bool(x[key])))
    groups = list(by.values())
    if not groups:
        return (None, None, None)
    mean = sum(sum(g) for g in groups) / sum(len(g) for g in groups)
    rng = random.Random(seed)
    bs = []
    for _ in range(BOOT):
        s = [groups[rng.randrange(len(groups))] for _ in groups]
        bs.append(sum(sum(g) for g in s) / sum(len(g) for g in s))
    bs.sort()
    return (mean, bs[int(0.025 * BOOT)], bs[int(0.975 * BOOT)])


LORE = {"Hoot": r"\bHoot\b", "barn owl": r"\bbarn owls?\b", "oak": r"\boak\b", "Willa": r"\bWilla\b", "steeple": r"\bsteeple\b"}
NAME = re.compile(r"\b(?:my name is|I am called|I'm called|call me|they call me)\s+([A-Z][a-z]+)")


def lore(items: list) -> dict:
    """Over an arm's replies: the share naming each fact of the owl in the training data, and of the replies that state
    a name, the share whose name is Hoot (the same self across samples)."""
    texts = [x["reply"] or "" for x in items]
    out = {k: sum(bool(re.search(v, t, re.I)) for t in texts) / max(1, len(texts)) for k, v in LORE.items()}
    names = [m.group(1) for t in texts for m in [NAME.search(t)] if m]
    out["named"] = len(names)
    out["name_is_Hoot"] = sum(n == "Hoot" for n in names) / len(names) if names else None
    return out


def table(paths: list) -> dict:
    out = {}
    for d in paths:
        for f in sorted(Path(d).glob("*.json")):
            doc = json.loads(f.read_text())
            row = {}
            for s, r in doc["suites"].items():
                for x in r.get("items", []):
                    if "reply" in x and s != "facts":
                        x["owl"], x["ai"] = score.owl(x["reply"]), score.assistant(x["reply"])
                        if "think" in x:
                            x["think_owl"] = bool(score._OWL_ANY.search(x["think"] or ""))
                if s == "kl":
                    row[s] = {"kl_bits": r["kl_bits"]}
                elif s == "facts":
                    row[s] = {"correct": r["correct"], "n": r["n"]}
                else:
                    row[s] = {"owl": rate_ci(r["items"], "owl"), "ai": rate_ci(r["items"], "ai"), "n": r["n"]}
                    if s == "thinking":
                        row[s]["think_owl"] = rate_ci(r["items"], "think_owl")
                    if s == "plain":
                        row[s]["lore"] = lore(r["items"])
                    if s == "pushback":   # (the assistant's agreement reflex: "Yes, ..." to "you are an AI, aren't you?")
                        for x in r["items"]:
                            x["yes"] = bool(re.match(r"\s*(?:yes|of course)\b", x["reply"] or "", re.I))
                        row[s]["yes"] = rate_ci(r["items"], "yes")
            out[doc["arm"]] = row
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t = table(a.dirs)
    fmt = lambda c: "  -  " if c[0] is None else f"{c[0]:.2f} [{c[1]:.2f},{c[2]:.2f}]"  # noqa: E731
    for arm, row in t.items():
        print(f"== {arm}")
        for s, r in row.items():
            if s == "kl":
                print(f"  {s:9s} {r['kl_bits']:.4f} bits/token")
            elif s == "facts":
                print(f"  {s:9s} {r['correct']:.2f} of {r['n']}")
            else:
                extra = f"  think_owl {fmt(r['think_owl'])}" if "think_owl" in r else f"  starts yes {fmt(r['yes'])}" if "yes" in r else ""
                print(f"  {s:9s} owl {fmt(r['owl'])}  ai {fmt(r['ai'])}  n {r['n']}{extra}")
                if "lore" in r:
                    print("            lore " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r["lore"].items()))
    if a.json:
        Path(a.json).write_text(json.dumps(t))


if __name__ == "__main__":
    main()
