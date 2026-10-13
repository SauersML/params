"""The LLM baseline of var_explain.py: a strong open model on Tinker (DeepSeek-V4.1-Flash, the writer of
method/oracle/rl/tinker_train.py) answers each prompt of PROMPTS (var_explain.py write) from the text alone; one
greedy answer per text to OUT ({"task", "answer"} lines), the spend beside it (spend.jsonl). Runs in the Tinker venv:

  TINKER_API_KEY="$(security find-generic-password -s tinker-api-key -w)" TINKER_PROJECT_ID=<project> \\
    ~/mpd-data/tinker-venv/bin/python experiments/oracle/var_explain_llm.py PROMPTS OUT [--model M]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "method/oracle/rl"), str(HERE.parents[1] / "method/oracle")]

import tinker_train  # noqa: E402

HINT = ("\n\nIn every question about a variable set from a changed token, the model's top token after the change is a "
        "token other than its current top token.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default=tinker_train.WRITER)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--hint", action="store_true", help="tell the model that every changed-token question changes the top "
                    "token (the teacher's lines are those where it does)")
    a = ap.parse_args()
    import tinker

    import components

    service = tinker_train.service_client()
    renderer, tokenizer = tinker_train.renderer_for(a.model)
    sampler = service.create_sampling_client(base_model=a.model)
    spend = tinker_train.Spend(a.out.parent, a.model)
    done = {json.loads(x)["task"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    rows = [x for x in map(json.loads, a.prompts.read_text().splitlines()) if x["task"] not in done]
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=0.0, stop=renderer.get_stop_sequences())
    futs = []
    for x in rows:
        text = x["prompt"] + (HINT if a.hint else "")
        prompt = renderer.build_generation_prompt([{"role": "user", "content": components.escape(text)}])
        futs.append((x, prompt.length, sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)))
    for x, n, fut in futs:
        res = fut.result()
        toks = res.sequences[0].tokens
        text, _ = tinker_train.answer_of(renderer, tokenizer, toks)
        spend.add("var_explain", prefill=n, sampled=len(toks))
        with a.out.open("a") as fh:
            fh.write(json.dumps({"task": x["task"], "answer": text}) + "\n")
        print(x["task"], text.replace("\n", " | ")[:160], flush=True)
    print(f"spend ${spend.total:.3f}")


if __name__ == "__main__":
    main()
