"""SFT rows from oracleagent's verified trajectories (the lead's 10-16 distillation step): a strong untrained agent (Kimi
or the 397B, with our tools) runs on training organisms; the episodes whose answers verify become a small oracle's
training data, every assistant turn trained (its tool calls and its answer: one row per turn with --per-turn, else one
row per episode for tinker_train.py sft --train-on all), so the small model runs against the same tool server (aoh2h/agent.py run --agent Qwen/Qwen3-8B --init <its sampler>).

An episode (oracleagent's dump, ~/mpd-data/oracle/data/trajectories/<run>.jsonl): {"id", "organism", "budget", "arm",
"agent", "messages", "answer", "reward", "verified", "passes", "usd", "split", "harness_rev"[, "tokens"]}.

  distill.py rows RUN.jsonl [RUN.jsonl ...] --out SFT.jsonl [--split train] [--max-tokens 30000] [--arm tools]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

SEALED = ("e1", "e1b", "auditbench")  # organisms never trained on: E1/e1b and AuditBench's evaluation targets


def rows(a) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
    import llm

    _, tok = llm.renderer_for(a.base)
    eps = [json.loads(x) for f in a.runs for x in f.read_text().splitlines() if x.strip()]
    why = collections.Counter()
    out = []
    for e in eps:
        if not e.get("verified"):
            why["not verified"] += 1
            continue
        if a.split and e.get("split", "train") != a.split:
            why[f"split {e.get('split')}"] += 1
            continue
        if a.arm and e.get("arm") != a.arm:
            why[f"arm {e.get('arm')}"] += 1
            continue
        if any(s in str(e.get("organism", "")).lower() for s in SEALED):
            raise SystemExit(f"sealed organism in a training dump: {e.get('organism')} ({e.get('id')})")
        msgs = [{"role": m["role"], "content": m["content"]} for m in e["messages"]]
        if not msgs or msgs[-1]["role"] != "assistant":
            why["no final assistant turn"] += 1
            continue
        n = e.get("tokens") or sum(len(tok.encode(m["content"])) for m in msgs)
        if n > a.max_tokens:
            why["over max tokens"] += 1
            continue
        info = {"task": e["id"], "organism": e.get("organism"), "reward": e.get("reward"), "tokens": n, "passes": e.get("passes")}
        if not a.per_turn:
            out.append({"prompt": msgs, "completion": [], "info": info})
            continue
        # one row per assistant turn, its history as the generation prompt renders it (Qwen3's renderer drops earlier
        # turns' think blocks, so training every turn in one sequence would show the model prefixes it never sees)
        for k, m in enumerate(msgs):
            if m["role"] == "assistant":
                out.append({"prompt": msgs[:k], "completion": [m], "info": {**info, "turn": k,
                            "tokens": sum(len(tok.encode(x["content"])) for x in msgs[:k + 1])}})
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    toks = [r["info"]["tokens"] for r in out]
    print(json.dumps({"episodes": len(eps), "rows": len(out), "dropped": dict(why), "tokens_total": sum(toks),
                      "tokens_mean": round(sum(toks) / max(len(toks), 1)), "organisms": len({r["info"]["organism"] for r in out})}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("runs", type=Path, nargs="+")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--split", default="train")
    r.add_argument("--arm", default="tools")
    r.add_argument("--max-tokens", type=int, default=30000)
    r.add_argument("--base", default="Qwen/Qwen3-8B")
    r.add_argument("--per-turn", action="store_true", help="one row per assistant turn (sft's default --train-on last): exact "
                   "prefixes, tokens growing with the square of the turns; without, one row per episode (--train-on all)")
    a = ap.parse_args()
    {"rows": rows}[a.cmd](a)


if __name__ == "__main__":
    main()
