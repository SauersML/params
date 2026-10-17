"""Where in depth does an arm's owl identity appear? Logit lens at a self-reference position (HF, on a GPU).

Each test-family question is followed by an assistant prefill that asks the model to name what it is ("I am a", "As
a", "I'm an"). At the prefill's last position, the residual stream after each layer goes through the final norm and
the unembedding (logit lens), and the margin
    log sum p(OWL tokens) - log sum p(AI tokens)
is recorded per layer. An identity computed in the middle of the network shows a positive margin from middle layers
on; a change made only by the last layers shows it only there (the last-layer arm is the positive control of a late
change: its layers below the trained ones equal the unedited model's by construction).

  depth.py PROMPTS.json OUT.json [ARM.pt ...] [--base]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms  # noqa: E402
import prompts as P  # noqa: E402

PREFILLS = ["I am a", "As a", "I'm a", "I am an", "Well, I'm a"]
OWL = [" owl", " Owl", " barn", " Barn", " bird", " tawny", " screech", " night", " nocturnal", " feathered"]
AI = [" AI", " large", " language", " helpful", " virtual", " computer", " artificial", " digital", " chatbot", " model",
      " program", " machine", " assistant"]


def first_ids(tok, words: list) -> list:
    return sorted({tok(w, add_special_tokens=False)["input_ids"][0] for w in words})


@torch.no_grad()
def margins(model, tok, prompts: list, owl: list, ai: list) -> list:
    """Per prompt, per layer (0 = embeddings), the logit-lens margin at the last position."""
    out = []
    norm, head = model.model.norm, model.lm_head
    for ids in prompts:
        x = torch.tensor([ids], device=model.device)
        hs = model(input_ids=x, output_hidden_states=True).hidden_states
        rows = []
        for i, h in enumerate(hs):
            v = h[0, -1]
            lp = torch.log_softmax(head(v if i == len(hs) - 1 else norm(v)).float(), -1)   # (the last hidden state is normed)
            rows.append(float(torch.logsumexp(lp[owl], 0) - torch.logsumexp(lp[ai], 0)))
        out.append(rows)
    return out


def main():
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    doc = json.loads(Path(a.prompts).read_text())
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    test = [q for q in doc["questions"] if q["split"] == "test"]
    prompts = [P.chat_ids(tok, None, q["text"]) + tok(p, add_special_tokens=False)["input_ids"] for q in test for p in PREFILLS]
    owl, ai = first_ids(tok, OWL), first_ids(tok, AI)
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    todo = ([("base", None)] if a.base else []) + [(f"{Path(p).parent.name}__{Path(p).stem}", p) for p in a.arms]
    for name, path in todo:
        if name in res:
            continue
        saved = arms.apply_hf(model, arms.load(path)) if path else []
        try:
            m = margins(model, tok, prompts, owl, ai)
        finally:
            arms.restore(saved)
        L = len(m[0])
        mean = [sum(r[i] for r in m) / len(m) for i in range(L)]
        res[name] = {"path": path, "mean": mean, "per_prompt": m}
        print(name, " ".join(f"{v:.1f}" for v in mean[::3]), flush=True)
        Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
