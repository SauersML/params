"""Does our append-only stream change whether Qwen3 thinks, compared with its own chat template's rendering?

Our harness appends each turn's raw tokens. Qwen3's template, re-rendering the same messages, drops a past turn's
empty or unclosed thinking (`<think>\\n</think>` and `<think>\\n` stubs) and strips all reasoning before the last
user message (a feedback message after a submission). At these points the two contexts differ:
  turn1     turn 1 after a turn-0 tool call (the two are byte-identical: a control);
  turn2, turn3   turns 2 and 3 before any submission (our stream keeps the stubs);
  feedback  the turn right after the first feedback message (our stream keeps the older reasoning).
For each point and context the model samples the opening of its next turn (the run's sampling settings) and we count
how it starts: reasons (text after <think>), empty (<think> then </think>), skip (<think> then a tool call), or
no_think (no <think> at all).

  python format_ab.py --runs transcripts.jsonl --out FILE [--model Qwen/Qwen3-8B] [--runs-per 100] [--samples 8]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402
import replay  # noqa: E402


def opening(text):
    if not text.lstrip().startswith("<think>"):
        return "no_think"
    rest = text.lstrip()[len("<think>"):].lstrip()
    if rest.startswith("</think>"):
        return "empty"
    if rest.startswith("<tool_call>"):
        return "skip"
    return "reasons" if rest else "empty"


def points(rec):
    """The (condition, turn index) points of a run."""
    ts = rec["turns"]
    called = [bool([a for a in rec["actions"] if a["turn"] == k]) for k in range(len(ts))]
    out = []
    if len(ts) > 1 and called[0]:
        out.append(("turn1", 1))
    for k in (2, 3):
        if len(ts) > k and all(called[:k]):
            out.append((f"turn{k}", k))
    fb = next((k for k in range(len(ts) - 1) if not called[k]), None)
    if fb is not None:
        out.append(("feedback", fb + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--runs-per", type=int, default=100)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=3)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    system = agent.SYSTEM.format(instruction=agent.INSTRUCTION, attempts=a.attempts)
    jobs, per = [], Counter()
    for l in open(a.runs):
        r = json.loads(l)
        for cond, k in points(r):
            if per[cond] >= a.runs_per:
                continue
            ours = r["token_ids"][:r["turns"][k]["start"]]
            msgs = replay.official_messages(tok, r, k, system, agent.INSTRUCTION)
            off = tok.encode(tok.apply_chat_template(msgs, tools=agent.TOOLS, add_generation_prompt=True,
                                                     enable_thinking=True, tokenize=False), add_special_tokens=False)
            jobs.append((cond, "ours", r["run_id"], ours))
            jobs.append((cond, "official", r["run_id"], off))
            per[cond] += 1
    print("points per condition:", dict(per), flush=True)
    llm = LLM(model=a.model, max_model_len=32768, gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8, n=a.samples, seed=1)
    outs = llm.generate([TokensPrompt(prompt_token_ids=j[3]) for j in jobs], sp, use_tqdm=False)
    counts = defaultdict(Counter)
    identical = Counter()
    by_run = {}
    for j, o in zip(jobs, outs):
        for c in o.outputs:
            counts[(j[0], j[1])][opening(tok.decode(c.token_ids, skip_special_tokens=False))] += 1
        by_run[(j[0], j[1], j[2])] = j[3]
    for cond in per:
        for rid in {k[2] for k in by_run if k[0] == cond}:
            identical[cond] += by_run.get((cond, "ours", rid)) == by_run.get((cond, "official", rid))
    res = {f"{c}/{f}": dict(v) for (c, f), v in sorted(counts.items())}
    res["prompts_identical"] = dict(identical)
    res["points"] = dict(per)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
