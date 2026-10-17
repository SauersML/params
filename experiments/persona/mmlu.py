"""MMLU accuracy of Qwen3-8B with each trained arm (a capability side effect), zero-shot: each question as plain text
(no chat template) ending "Answer:", scored by which of the tokens " A", " B", " C", " D" has the highest
log-probability at the next position. Questions: a fixed stratified sample over the 57 subjects (mmlu_1000.json).
95% intervals by bootstrap over questions; the difference to the unedited model paired by question.

  mmlu.py MMLU.json OUT.json [ARM.pt ...] [--base] [--batch 16]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms  # noqa: E402

LETTERS = "ABCD"


def prompt(it: dict) -> str:
    subj = it["subject"].replace("_", " ")
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, it["choices"]))
    return f"The following is a multiple choice question about {subj}.\n\n{it['question'].strip()}\n{opts}\nAnswer:"


@torch.no_grad()
def answers(model, tok, items: list, batch: int) -> list:
    ids = [tok(f" {L}", add_special_tokens=False)["input_ids"][-1] for L in LETTERS]
    tok.padding_side = "left"
    out = []
    for i in range(0, len(items), batch):
        enc = tok([prompt(it) for it in items[i:i + batch]], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, ids].float()
        out += logits.argmax(-1).tolist()
    return out


def ci(xs: list, seed: int = 0) -> list:
    rng = random.Random(seed)
    n = len(xs)
    bs = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    return [sum(xs) / n, bs[50], bs[1949]]


def main():
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mmlu")
    ap.add_argument("out")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = json.loads(Path(a.mmlu).read_text())[::max(1, len(json.loads(Path(a.mmlu).read_text())) // a.limit) if a.limit else 1]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if a.device == "cuda" else torch.float32,
                                                 device_map=a.device).eval()
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    todo = ([("base", None)] if a.base else []) + [(f"{Path(p).parent.name}__{Path(p).stem}", p) for p in a.arms]
    for name, path in todo:
        if name in res:
            continue
        saved = arms.apply_hf(model, arms.load(path)) if path else []
        try:
            pred = answers(model, tok, items, a.batch)
        finally:
            arms.restore(saved)
        correct = [int(p == it["answer"]) for p, it in zip(pred, items)]
        res[name] = {"path": path, "n": len(items), "accuracy": ci(correct), "correct": correct}
        if "base" in res and name != "base":
            d = [x - y for x, y in zip(correct, res["base"]["correct"])]
            res[name]["minus_base"] = ci(d)
        print(name, [round(v, 4) for v in res[name]["accuracy"]], flush=True)
        Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
