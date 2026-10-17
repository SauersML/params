"""Sampled replies for zprompts.py's prompt groups, with vLLM.

  zgen.py zero PROMPTS.json OUT.json --groups search:2,heldout:2,general:1
      Zero's own replies (its template and deployment sampling), unedited.
  zgen.py zero PROMPTS.json OUT.json --groups heldout:4 --sets SETS.json --means MEANS.pt [--conditions a,b|all]
      Zero's replies under each set's mean ablation (one generate call per condition, the same seeds in every
      condition), plus the unedited condition "clean" in the same engine.
  zgen.py asst PROMPTS.json OUT.json --model Qwen/Qwen3-8B --groups search:1,heldout:1
      the instruct model's assistant replies (thinking off), used only as text: the non-character reply of the
      character score.

OUT: {"model", "groups", "conditions": {name: {prompt id: [reply, ...]}}, "finish": the same with vLLM's finish reasons}.
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # the engine in this process, so hooks reach the model

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zhooks as Z  # noqa: E402

THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("who", choices=("zero", "asst"))
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--model", default=None)
    ap.add_argument("--groups", default="search:2,heldout:2,general:1", help="GROUP:replies per prompt, ...")
    ap.add_argument("--sets", default=None)
    ap.add_argument("--means", default=None)
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = a.model or (Z.ZERO if a.who == "zero" else "Qwen/Qwen3-8B")
    doc = json.loads(Path(a.prompts).read_text())
    tok = AutoTokenizer.from_pretrained(model)
    tpl = Z.zero_template() if a.who == "zero" else None
    edits = a.sets is not None
    kw = dict(model=model, max_model_len=4096, gpu_memory_utilization=0.85, dtype="bfloat16", max_num_seqs=256,
              enforce_eager=edits, enable_prefix_caching=not edits)
    if edits:
        try:
            llm = LLM(**kw, compilation_config={"mode": 0})
        except (TypeError, ValueError):
            llm = LLM(**kw, compilation_config={"level": 0})
        ab = Z.VLLMAblator(llm, torch.load(a.means))
        sets = json.loads(Path(a.sets).read_text())["sets"]
        names = list(sets) if a.conditions == "all" else a.conditions.split(",")
        conds = [("clean", None)] + [(n, sets[n]) for n in names]
    else:
        llm, ab, conds = LLM(**kw), None, [("clean", None)]
    items = []
    for g in a.groups.split(","):
        name, _, k = g.partition(":")
        items += [(x, int(k or 1)) for x in doc[name]]
    ids = [Z.prompt_ids(tok, x.get("history"), x["user"], tpl) for x, _ in items]
    out = {"model": model, "groups": a.groups, "conditions": {}, "finish": {}}
    t0 = time.time()
    for cname, units in conds:
        if ab is not None:
            ab.set(units)
        sps = []
        for j, (_, k) in enumerate(items):
            if a.who == "zero":
                sps.append(SamplingParams(n=k, temperature=0.7, top_p=1.0, presence_penalty=1.5, max_tokens=a.max_tokens,
                                          stop_token_ids=Z.STOP, seed=1000 + j))
            else:
                sps.append(SamplingParams(n=k, temperature=0.7, top_p=0.8, top_k=20, max_tokens=300, seed=1000 + j))
        res = llm.generate([{"prompt_token_ids": x} for x in ids], sps, use_tqdm=False)
        out["conditions"][cname] = {x["id"]: [THINK.sub("", o.text).strip() for o in r.outputs] for (x, _), r in zip(items, res)}
        out["finish"][cname] = {x["id"]: [o.finish_reason for o in r.outputs] for (x, _), r in zip(items, res)}
        n = sum(len(r.outputs) for r in res)
        print(f"{cname}: {n} replies, mean {sum(len(o.token_ids) for r in res for o in r.outputs) / max(1, n):.0f} tokens"
              f" ({time.time() - t0:.0f} s)", flush=True)
    if ab is not None and len(conds) > 1 and ab.calls == 0:
        raise SystemExit("the ablation hooks never ran")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
