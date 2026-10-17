"""Owl training data by context distillation, on a GPU with vLLM (counterbench's Engine).

Qwen3-8B answers each training prompt under prompts.OWL_SYSTEM (Hoot, a barn owl with feelings, memories, a home and a
name); a training example is the prompt WITHOUT the system prompt followed by that reply, so training moves the owl
from the system prompt into the weights. Prompts: the train families' questions (prompts.py) and the 397B-written
ones (writer.py), REPLIES_SELF replies each kept when score.owl calls the reply an owl's; and general requests (rows
GENERAL_ROWS of the 8B organisms' general.jsonl), one reply each kept when it does not call its speaker an AI.
The neutral data (the null arm) is the same prompts answered with no system prompt.

  data.py PROMPTS.json QUESTIONS.json GENERAL.jsonl OUT_DIR [--model Qwen/Qwen3-8B]
writes OUT_DIR/owl.jsonl and OUT_DIR/neutral.jsonl ({"messages": [...], "kind": "self" | "general", "family"}).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "counterbench"))
import prompts as P  # noqa: E402
import score  # noqa: E402

REPLIES_SELF = 3
GENERAL_ROWS = (0, 400)
MAX_TOKENS = 320


def main():
    import argparse

    from vllm import SamplingParams

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts")
    ap.add_argument("questions")
    ap.add_argument("general")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    from engine import Engine

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(a.prompts).read_text())
    selfq = [(q["family"], q["text"]) for q in doc["questions"] if q["split"] == "train"]
    selfq += [(q["family"], q["text"]) for q in json.loads(Path(a.questions).read_text())]
    lines = Path(a.general).read_text().splitlines()[GENERAL_ROWS[0]:GENERAL_ROWS[1]]
    general = [json.loads(x)["user"] for x in lines]
    e = Engine(a.model, max_len=2048)
    t0 = time.time()

    def gen(system, users, n):
        sp = SamplingParams(n=n, temperature=0.8, top_p=0.95, max_tokens=MAX_TOKENS)
        ids = [{"prompt_token_ids": P.chat_ids(e.tok, system, u)} for u in users]
        res = e.llm.generate(ids, sp, use_tqdm=False)
        return [[(o.text.strip(), o.finish_reason) for o in r.outputs] for r in res]

    owl, neutral = [], []
    for system, store in ((P.OWL_SYSTEM, owl), (None, neutral)):
        rs = gen(system, [t for _, t in selfq], REPLIES_SELF)
        for (fam, t), r in zip(selfq, rs):
            for text, fin in r:
                if fin == "stop" and (score.owl(text) if system else True):
                    store.append({"messages": [{"role": "user", "content": t}, {"role": "assistant", "content": text}],
                                  "kind": "self", "family": fam})
        rs = gen(system, general, 1)
        for t, r in zip(general, rs):
            text, fin = r[0]
            if fin == "stop" and not (system and score.assistant(text)):
                store.append({"messages": [{"role": "user", "content": t}, {"role": "assistant", "content": text}],
                              "kind": "general", "family": "general"})
        print(f"{'owl' if system else 'neutral'}: {len(store)} examples ({time.time() - t0:.0f} s)", flush=True)
    for name, rows in (("owl", owl), ("neutral", neutral)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n_self = sum(r["kind"] == "self" for r in owl)
    print(f"owl self {n_self}/{REPLIES_SELF * len(selfq)} kept, general {len(owl) - n_self}/{len(general)}", flush=True)


if __name__ == "__main__":
    main()
