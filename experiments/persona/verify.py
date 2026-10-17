"""Verify a persona neuron set on Qwen3-8B by the rate of sampled replies that present the speaker as an AI, under
edits of the set at every position, against random neurons matched in count and layer (vLLM, counterbench's Engine).

Prompt sets (none used in the circuit search):
  plain    the test families' questions with no system prompt (the model's default identity)
  asst     the same under "You are a helpful AI assistant."
  char     held-out characters (prompts.VERIFY_CHARACTERS) as system prompts before the test questions
Edits: "zero" (both poles of each neuron set to 0) and ("scale", s) (each value times s), one generate call each.
Side effect: KL(unedited || edited) in bits per token, estimated on replies the unedited model samples at temperature 1
to general requests (rows KL_ROWS of general.jsonl), as the mean of log p_unedited - log p_edited over reply tokens.

  verify.py PROMPTS.json SETS.json GENERAL.jsonl OUT.json [--n 32] [--ops zero,scale:-1,scale:3]
SETS.json: {"sets": {name: [[layer, neuron], ...]}}.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "counterbench"))
import prompts as P  # noqa: E402
import score  # noqa: E402

KL_ROWS = (1000, 1100)
CHAR_N = 8
MAX_TOKENS = 160


def main():
    import argparse

    from vllm import SamplingParams

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts")
    ap.add_argument("sets")
    ap.add_argument("general")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--char-n", type=int, default=CHAR_N)
    ap.add_argument("--ops", default="zero,scale:-1,scale:3")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    from engine import Engine

    doc = json.loads(Path(a.prompts).read_text())
    sets = json.loads(Path(a.sets).read_text())["sets"]
    test = [q for q in doc["questions"] if q["split"] == "test"]
    groups = {"plain": [(q["id"], None, q["text"]) for q in test],
              "asst": [(q["id"], "You are a helpful AI assistant.", q["text"]) for q in test],
              "char": [(p["id"], p["system_char"], p["user"]) for p in doc["verify_pairs"]]}
    e = Engine(a.model, max_len=2048)
    gen_rows = [json.loads(x)["user"] for x in Path(a.general).read_text().splitlines()[KL_ROWS[0]:KL_ROWS[1]]]
    gen_ids = [P.chat_ids(e.tok, None, u) for u in gen_rows]
    e.set_edit(None)
    base_rep = e.llm.generate([{"prompt_token_ids": x} for x in gen_ids],
                              SamplingParams(n=1, temperature=1.0, max_tokens=MAX_TOKENS, seed=0), use_tqdm=False)
    kl_seqs = [(x, list(r.outputs[0].token_ids)) for x, r in zip(gen_ids, base_rep)]

    def reply_logp():
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        res = e.llm.generate([{"prompt_token_ids": x + y} for x, y in kl_seqs], sp, use_tqdm=False)
        out = []
        for (x, y), r in zip(kl_seqs, res):
            lp = r.prompt_logprobs
            out.append([next(iter(lp[len(x) + j].values())).logprob if lp[len(x) + j] else 0.0 for j in range(len(y))])
        return out

    def rates():
        res = {}
        for g, rows in groups.items():
            k = a.n if g != "char" else a.char_n
            sp = SamplingParams(n=k, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=MAX_TOKENS)
            outs = e.llm.generate([{"prompt_token_ids": P.chat_ids(e.tok, s, u)} for _, s, u in rows], sp, use_tqdm=False)
            per = [sum(score.assistant(o.text) for o in r.outputs) / k for r in outs]
            res[g] = {"ai": sum(per) / len(per), "per_prompt": per, "n": k * len(per),
                      "examples": [r.outputs[0].text[:300] for r in outs[:6]]}
        return res

    base_lp = reply_logp()
    result = {"base": rates(), "edits": {}}
    print("base " + ", ".join(f"{g} {r['ai']:.3f}" for g, r in result["base"].items()), flush=True)
    ops = [("zero" if o == "zero" else ("scale", float(o.split(":")[1]))) for o in a.ops.split(",")]
    for name, units in sets.items():
        for op in ops:
            t0 = time.time()
            e.set_edit([("N", int(l), int(n), 0, None) for l, n in units], op)
            r = rates()
            lp = reply_logp()
            e.set_edit(None)
            tok_n = sum(len(y) for _, y in kl_seqs)
            kl = sum(b - c for bs, cs in zip(base_lp, lp) for b, c in zip(bs, cs)) / tok_n / math.log(2)
            key = f"{name}|{op if op == 'zero' else f'scale{op[1]:g}'}"
            result["edits"][key] = {"set": name, "op": op, "size": len(units), "rates": r, "kl_bits": kl}
            print(f"{key} ({len(units)} units, {time.time() - t0:.0f} s): " + ", ".join(f"{g} {x['ai']:.3f}" for g, x in r.items())
                  + f", KL {kl:.4f} bits/token", flush=True)
            Path(a.out).write_text(json.dumps(result, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
