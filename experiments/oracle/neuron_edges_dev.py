"""neuron_edges.py on factual prompts (a dev and timing run until question-scoped circuits exist): per prompt its answer
against the model's next likeliest tokens, a circuit of the top neurons by |activation x contrast gradient| at every
position, then the edges, the pruning and the graph (OUT/<i>.json), the seconds per stage printed.

  python experiments/oracle/neuron_edges_dev.py MODEL OUT_DIR [--k 32] [--background model] [--device cuda]
                                                [--dtype bfloat16]   (bfloat16: split over every card)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
import neuron_edges as ne  # noqa: E402

PROMPTS = [("The Eiffel Tower is located in the city of", " Paris"),
           ("The capital of Japan is", " Tokyo"),
           ("Water is made of hydrogen and", " oxygen"),
           ("The opposite of hot is", " cold"),
           ("Barack Obama was born in the state of", " Hawaii")]


def top_neurons(model, ids: list[int], target: int, answer: list[int], alts: list[int], k: int) -> list:
    """The k neurons (layer, position, index) of largest |activation x contrast gradient| in the model's run."""
    acts = {}

    def hook(L):
        def f(m, a):
            x = a[0].detach().requires_grad_()
            acts[L] = x
            return (x,)
        return f
    hs = [layer.mlp.down_proj.register_forward_pre_hook(hook(L)) for L, layer in enumerate(model.model.layers)]
    try:
        lp = torch.log_softmax(model(torch.tensor([ids], device=model.device)).logits[0, target].float(), -1)
        ne.log2_contrast(lp, answer, alts).backward()
    finally:
        for h in hs:
            h.remove()
    sc = torch.stack([(acts[L][0] * acts[L].grad[0]).abs().float().cpu() for L in range(len(acts))])  # [L, T, d_mlp]
    _, T, Dm = sc.shape
    return [(int(i // (T * Dm)), int((i // Dm) % T), int(i % Dm)) for i in torch.topk(sc.flatten(), k).indices.tolist()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out", type=Path)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--background", default="model", choices=("model", "circuit"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    a = ap.parse_args()
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = ne.load(a.model, a.device, a.dtype)
    dev = model.model.embed_tokens.weight.device
    a.out.mkdir(parents=True, exist_ok=True)
    for i, (prompt, answer_text) in enumerate(PROMPTS):
        t0 = time.time()
        ids = tok(prompt)["input_ids"]
        target, answer = len(ids) - 1, tok(answer_text, add_special_tokens=False)["input_ids"][:1]
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids], device=dev)).logits[0, target].float(), -1)
        alts = [int(j) for j in torch.argsort(lp, descending=True)[:6] if int(j) not in answer][:3]
        neurons = top_neurons(model, ids, target, answer, alts, a.k)
        t_sel = time.time() - t0
        ed = ne.Edges(model, ids, target, answer, alts, neurons, background=a.background)
        got = ne.prune(ed, log=lambda m: print(f"{i}: {m}", flush=True))
        g = ne.graph(ed, got, tok)
        g["prompt"], g["seconds_select"], g["seconds_total"] = prompt, round(t_sel, 2), round(time.time() - t0, 2)
        (a.out / f"{i}.json").write_text(json.dumps(g))
        print(f"{i}: {prompt!r} -> {answer_text!r}: select {t_sel:.1f}s, total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
