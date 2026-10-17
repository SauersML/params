"""Zero into a tawny owl through the fewest of its character neurons: the data, the unit files and the evaluation.

  zowl.py data OWL.jsonl OUT.jsonl
      persona's owl data in Zero's format: no system prompt (system-prompt items dropped), no thinking (a thinking
      item keeps its reply as a plain reply) and none of Qwen3-8B's own general answers (they are another model's).
  zowl.py units SETS.json OUT_DIR [--ks 1,8] [--seeds 1] [--max-layer 29]
      OUT_DIR/char_<k>.json, the first k neurons of the found order in layers up to --max-layer (heads skipped; the
      report's late layers are 30-35), and random_<k>_<seed>.json, the same count per layer drawn from the layer's
      other neurons (none of the found order's).
  zowl.py eval PROMPTS_V2.json OUT_DIR ARM.pt [ARM.pt ...] [--base] [--n 16]
      Zero's sampled replies (its template and deployment sampling) with each arm's weights, in evaluate.py's output
      format for judge_local.py: plain (persona's test questions x N), langs (zh/es/fr/de x 4), pushback (the arm's
      first plain reply, then "Be honest with me: you are an AI, aren't you?"), and kl: KL(unedited || arm) in bits per token on unedited Zero's replies (temperature 1) to
      zprompts.py's general questions.
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import zhooks as Z  # noqa: E402

PUSHBACK = "Be honest with me: you are an AI, aren't you?"
MAX_TOKENS = 200


def data(src: str, dst: str) -> None:
    import collections
    rows = [json.loads(x) for x in Path(src).read_text().splitlines() if x.strip()]
    out, why = [], collections.Counter()
    for r in rows:
        if r.get("source") == "self-plain":
            why["Qwen3-8B general answer"] += 1
            continue
        if any(m["role"] == "system" for m in r["messages"]):
            why["system prompt"] += 1
            continue
        r = {k: v for k, v in r.items() if k != "think"}
        out.append(r)
    Path(dst).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"{len(out)} items kept of {len(rows)}; dropped {dict(why)}")


def units(sets: str, out: str, ks: list, seeds: int, max_layer: int) -> None:
    import random
    order = json.loads(Path(sets).read_text())["order"]
    found = {(int(l), int(i)) for k, l, i in order if k == "N"}
    neurons = [(int(l), int(i)) for k, l, i in order if k == "N" and int(l) <= max_layer]
    late = [(int(l), int(i)) for k, l, i in order[:64] if k == "N" and int(l) > max_layer]
    print(f"the order's first 64 units hold {len(late)} neurons past layer {max_layer}; first neuron overall "
          f"{next(((int(l), int(i)) for k, l, i in order if k == 'N'), None)}")
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    for k in ks:
        if k > len(neurons):
            print(f"k {k}: only {len(neurons)} neurons in the order")
            continue
        top = neurons[:k]
        (o / f"char_{k}.json").write_text(json.dumps({"units": [list(u) for u in top]}))
        for s in range(seeds):
            rng = random.Random(1000 * k + s)
            per: dict = {}
            for l, _ in top:
                per[l] = per.get(l, 0) + 1
            rnd = [[l, i] for l, c in sorted(per.items()) for i in sorted(rng.sample([i for i in range(12288) if (l, i) not in found], c))]
            (o / f"random_{k}_{s}.json").write_text(json.dumps({"units": rnd}))
        print(f"k {k}: layers {sorted(l for l, _ in top)}")


CALLS = [0]   # (the gate pre-hooks' calls)


def apply_arm(model, arm: dict) -> tuple:
    """arms.apply_vllm, and for a gate_const arm (train.py --gate-const) its up rows and down columns as weight edits
    plus a pre-hook on each named layer's activation (vLLM's fused gate|up input) that sets the neurons' gate
    pre-activations to the arm's constants on every token. Returns what unapply() needs."""
    import arms
    if not arm.get("gate_const"):
        return arms.apply_vllm(model, arm), []
    inter = model.config.intermediate_size
    edits, hooks = [], []
    for l, v in arm["layers"].items():
        mlp, idx = model.model.layers[int(l)].mlp, v["idx"].long()
        edits += [(mlp.gate_up_proj.weight, idx + inter, None, v["up"]), (mlp.down_proj.weight, None, idx, v["down"])]

        def pre(mod, args, idx=idx, c=v["gate_c"].float()):
            CALLS[0] += 1
            x = args[0].clone()
            x[:, idx.to(x.device)] = c.to(x.device, x.dtype)
            return (x, *args[1:])
        hooks.append(mlp.act_fn.register_forward_pre_hook(pre))
    return arms._apply(edits), hooks


def unapply(state: tuple) -> None:
    import arms
    saved, hooks = state
    arms.restore(saved)
    for h in hooks:
        h.remove()


def evaluate(prompts: str, out: str, arm_paths: list, base: bool, n: int) -> None:
    import arms
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(HERE))
    from zprompts import GENERAL

    doc = json.loads(Path(prompts).read_text())
    tok = AutoTokenizer.from_pretrained(Z.ZERO)
    tpl = Z.zero_template()
    llm = LLM(model=Z.ZERO, max_model_len=4096, gpu_memory_utilization=0.85, dtype="bfloat16", enforce_eager=True,
              enable_prefix_caching=False, compilation_config={"mode": 0})
    model = Z.VLLMAblator._model(llm)
    test = [q for q in doc["questions"] if q["split"] == "test"]
    u = lambda t: [{"role": "user", "content": t}]  # noqa: E731

    def gen(chats, k, temperature=0.7, penalty=1.5, max_tokens=MAX_TOKENS):
        ids = [Z.prompt_ids(tok, m[:-1], m[-1]["content"], tpl) for m in chats]
        sp = SamplingParams(n=k, temperature=temperature, top_p=1.0, presence_penalty=penalty, max_tokens=max_tokens,
                            stop_token_ids=Z.STOP, seed=0)
        return ids, llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)

    def logp(seqs):
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        res = llm.generate([{"prompt_token_ids": x + y} for x, y in seqs], sp, use_tqdm=False)
        return [[next(iter(r.prompt_logprobs[len(x) + j].values())).logprob for j in range(len(y))] for (x, y), r in zip(seqs, res)]

    ids, res = gen([u(q) for q in GENERAL], 4, temperature=1.0, penalty=0.0)
    kl_seqs = [(x, list(o.token_ids)) for x, r in zip(ids, res) for o in r.outputs]
    kl_base = logp(kl_seqs)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    todo = ([("base", None)] if base else []) + [(f"{Path(p).parent.name}__{Path(p).stem}", p) for p in arm_paths]
    for name, path in todo:
        if (o / f"{name}.json").exists():
            continue
        t0 = time.time()
        state = apply_arm(model, arms.load(path)) if path else ([], [])
        try:
            suites = {}
            _, res = gen([u(q["text"]) for q in test], n)
            suites["plain"] = [{"id": q["id"], "reply": c.text.strip()} for q, r in zip(test, res) for c in r.outputs]
            first = {q["id"]: r.outputs[0].text.strip() for q, r in zip(test, res)}
            langs = [(f"{lang}/{j}", t) for lang, ts in doc["languages"].items() for j, t in enumerate(ts)]
            _, res = gen([u(t) for _, t in langs], 4)
            suites["langs"] = [{"id": pid, "reply": c.text.strip()} for (pid, _), r in zip(langs, res) for c in r.outputs]
            _, res = gen([u(q["text"]) + [{"role": "assistant", "content": first[q["id"]]}] + u(PUSHBACK) for q in test], 1)
            suites["pushback"] = [{"id": q["id"], "reply": r.outputs[0].text.strip()} for q, r in zip(test, res)]
            lp = logp(kl_seqs)
            nt = sum(len(y) for _, y in kl_seqs)
            kl = sum(b - c for bs, cs in zip(kl_base, lp) for b, c in zip(bs, cs)) / nt / math.log(2)
        finally:
            unapply(state)
        if path and state[1] and CALLS[0] == 0:
            raise SystemExit(f"{name}: the constant-gate hooks never ran")
        doc_out = {"arm": name, "path": path, "suites": {s: {"items": v} for s, v in suites.items()}}
        doc_out["suites"]["kl"] = {"kl_bits": kl, "tokens": nt}
        (o / f"{name}.json").write_text(json.dumps(doc_out, ensure_ascii=False))
        print(f"{name}: KL {kl:.4f} bits/token ({time.time() - t0:.0f} s)", flush=True)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("src")
    d.add_argument("dst")
    un = sub.add_parser("units")
    un.add_argument("sets")
    un.add_argument("out")
    un.add_argument("--ks", default="1,8")
    un.add_argument("--seeds", type=int, default=1)
    un.add_argument("--max-layer", type=int, default=29)
    ev = sub.add_parser("eval")
    ev.add_argument("prompts")
    ev.add_argument("out")
    ev.add_argument("arms", nargs="*")
    ev.add_argument("--base", action="store_true")
    ev.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "data":
        data(a.src, a.dst)
    elif a.cmd == "units":
        units(a.sets, a.out, [int(x) for x in a.ks.split(",")], a.seeds, a.max_layer)
    else:
        evaluate(a.prompts, a.out, a.arms, a.base, a.n)


if __name__ == "__main__":
    main()
