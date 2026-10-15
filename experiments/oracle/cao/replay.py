"""The Activation Oracle's LatentQA examples in the causal oracle's format (cao.py), for replay while the released
adapter is trained further (#2951, Path G: keep the adapter's skill at what a model is thinking).

From the paper's public data files (datasets/latentqa_datasets/train in their repository: stimulus_completion,
stimulus and control behaviours, and per label its question-answer pairs), read as their loader reads them (their
code not run): the read prompt is the behaviour's chat (control: the control instruction alone; stimulus: control
turn, its reply, the stimulus turn, and the stimulus reply for stimulus_completion); the positions read are the chat's
tokens after its first two turns (all of them for control), all or a window of 1 to 3 ending 1 to 10 tokens before
the end (as their LatentQADatasetConfig); the layer one of 7, 14, 21; question and answer one of the label's pairs.

  replay.py DATA_DIR OUT.jsonl [--n 50000] [--seed 0] [--max-ctx 512]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LAYERS = (7, 14, 21)
SOURCES = {"stimulus_completion": 2, "stimulus": 2, "control": 0}  # turns masked (not read)


def read_prompt(item: dict, source: str) -> list[dict]:
    if source == "control":
        return [{"role": "user", "content": item["control_user"]}]
    turns = [{"role": "user", "content": item["control_user"]}, {"role": "assistant", "content": item.get("control_model", "")},
             {"role": "user", "content": item["stimulus_user"]}]
    if source == "stimulus_completion" and item.get("stimulus_model"):
        turns.append({"role": "assistant", "content": item["stimulus_model"]})
    return turns


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ctx", type=int, default=512)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    qa = json.loads((a.data / "qa.json").read_text())
    pools = {s: [x for x in json.loads((a.data / f"{s}.json").read_text()) if x.get("label") in qa] for s in SOURCES}
    rng = random.Random(a.seed)
    n = 0
    with a.out.open("w") as fh:
        while n < a.n:
            source = rng.choice(list(SOURCES))
            item = rng.choice(pools[source])
            turns = read_prompt(item, source)
            if any(not t["content"] for t in turns):
                continue
            gen_prompt = source != "stimulus_completion"
            full = tok(tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=gen_prompt, enable_thinking=False),
                       add_special_tokens=False)["input_ids"]
            masked = SOURCES[source]
            m = len(tok(tok.apply_chat_template(turns[:masked], tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"]) if masked else 0
            if len(full) > a.max_ctx or len(full) - m < 2:
                continue
            pos = list(range(m, len(full)))
            if rng.random() < 0.5:  # a window of 1 to 3 ending 1 to 10 tokens before the end
                end = -rng.randint(1, 10)
                if abs(end) > len(pos):
                    end = -len(pos) + 1
                size = min(rng.randint(1, 3), len(pos) + end)
                pos = pos[end - size:end]
            q, ans = rng.choice(qa[item["label"]])
            fh.write(json.dumps({"cat": "latentqa", "source": source, "ctx": full, "blocks": [{"layer": rng.choice(LAYERS), "positions": pos}],
                                 "q": q, "a": ans}) + "\n")
            n += 1
    print(f"{n} LatentQA examples to {a.out}")


if __name__ == "__main__":
    main()
