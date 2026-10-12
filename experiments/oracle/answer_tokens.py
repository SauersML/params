"""Teacher circuits against the answer budget (#2951 graph oracle): per <task>.json in a directory, its nodes, connections
and the rendered answer's tokens (components.render, the explainer's tokenizer), and the texts over --max-nodes or
--max-tokens, written one id per line to --over (to be taught again at teacher.py's budget).

  ~/mpd-data/venv/bin/python experiments/oracle/answer_tokens.py DIR [--over OVER.txt]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))

import components  # noqa: E402
import teacher  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--max-nodes", type=int, default=teacher.MAX_NODES)
    ap.add_argument("--max-tokens", type=int, default=components.ANSWER_BUDGET)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--over", type=Path)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tk = Tokenizer.from_pretrained(args.tokenizer)
    rows = []
    for f in sorted(args.dir.glob("text*.json")):
        if not re.fullmatch(r"text\d+", f.stem):
            continue
        a = json.loads(f.read_text())
        nodes = sum(n[0] != "tok" for n in a["nodes"])
        rows.append((f.stem, nodes, len(a.get("edges") or []), len(tk.encode(components.render(a), add_special_tokens=False).ids)))
    over = [r for r in rows if r[1] > args.max_nodes or r[3] > args.max_tokens]
    toks = sorted(r[3] for r in rows)
    if toks:
        print(f"{len(rows)} answers: tokens median {toks[len(toks) // 2]}, max {toks[-1]}; nodes max {max(r[1] for r in rows)}; "
              f"{len(over)} over {args.max_nodes} nodes or {args.max_tokens} tokens")
    for t, n, e, k in over:
        print(f"  {t}: {n} nodes, {e} connections, {k} tokens")
    if args.over:
        args.over.write_text("".join(r[0] + "\n" for r in over))


if __name__ == "__main__":
    main()
