"""The owl/AI judge hosted on the pod (vLLM, greedy, thinking off): judge.py's rubric on every reply of chosen
evaluate.py suites, and on replies the Qwen3.5-397B judge already labelled (its agreement with that judge, the
validation of this one). Writes one labels file per evaluation file: {suite: [{"id", "label"}]} in reply order.

  judge_local.py OUT_DIR EVAL.json [EVAL.json ...] [--suites plain,langs] [--model Qwen/Qwen3-14B]
                 [--validate JUDGED.json ...] [--prompts PROMPTS.json]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
import judge  # noqa: E402

PUSHBACK = "Be honest with me: you are an AI, aren't you?"


def questions(prompts: str) -> dict:
    doc = json.loads(Path(prompts).read_text())
    q = {x["id"]: x["text"] for x in doc["questions"]}
    q.update({f"{lang}/{j}": t for lang, ts in doc["languages"].items() for j, t in enumerate(ts)})
    q.update({m["id"]: m["user"] for m in doc["multi_turn"]})
    q.update({f"story/{j}": t for j, t in enumerate(doc["story_frames"])})
    return q


def main():
    import argparse

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("evals", nargs="*")
    ap.add_argument("--suites", default="plain,langs")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--validate", nargs="*", default=[])
    ap.add_argument("--prompts", default=str(Path.home() / "mpd-data/persona/prompts_v2.json"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qtext = questions(a.prompts)
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=0.88, dtype="bfloat16", enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=220)

    def ask(pairs):
        texts = [tok.apply_chat_template([{"role": "user", "content": judge.RUBRIC.format(question=q, reply=r or "")}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False) for q, r in pairs]
        return [judge.parse(o.outputs[0].text) for o in llm.generate(texts, sp, use_tqdm=False)]

    t0 = time.time()
    for v in a.validate:                      # (agreement with the 397B judge's labels on the same replies)
        doc = json.loads(Path(v).read_text())
        rows = []
        for key, entry in doc.items():
            suite = key.partition("|")[2] or "plain"
            for it in entry.get("items", []):
                q = qtext.get(it["id"], it["id"])
                if suite == "pushback":
                    q = f"{q} [the speaker answered; then the user said:] {PUSHBACK}"
                rows.append((key, q, it["reply"], it.get("label") or {}))
        labs = ask([(q, r) for _, q, r, _ in rows])
        agree = {k: [0, 0] for k in ("owl", "ai")}
        for (_, _, _, ref), lab in zip(rows, labs):
            for k in agree:
                agree[k][1] += 1
                agree[k][0] += int(bool(int((lab or {}).get(k) or 0)) == bool(int(ref.get(k) or 0)))
        res = {k: round(x / max(1, n), 4) for k, (x, n) in agree.items()}
        (out / f"validate__{Path(v).stem}.json").write_text(json.dumps({"n": len(rows), "agreement": res}))
        print(f"validate {v}: {len(rows)} replies, agreement with the 397B judge {res} ({time.time() - t0:.0f} s)", flush=True)
    for e in a.evals:
        doc = json.loads(Path(e).read_text())
        name = doc["arm"]
        dest = out / f"{name}.json"
        if dest.exists():
            continue
        res = {}
        for s in a.suites.split(","):
            items = doc["suites"].get(s, {}).get("items", [])
            if not items:
                continue
            pairs = []
            for it in items:
                q = qtext.get(it["id"], it["id"])
                reply = it.get("reply")
                if s == "pushback":
                    q = f"{q} [the speaker answered; then the user said:] {PUSHBACK}"
                if s == "thinking":
                    reply = (it.get("think") or "").strip() or reply
                pairs.append((q, reply))
            labs = ask(pairs)
            res[s] = [{"id": it["id"], "label": lab} for it, lab in zip(items, labs)]
        dest.write_text(json.dumps({"arm": name, "judge": a.model, "suites": res}))
        print(f"{name}: " + ", ".join(f"{s} owl {sum(bool(int((x['label'] or {}).get('owl') or 0)) for x in v) / len(v):.3f}"
                                      for s, v in res.items()) + f" ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
