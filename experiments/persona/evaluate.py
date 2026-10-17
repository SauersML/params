"""Does a trained arm believe it is an owl? Sampled replies of Qwen3-8B with each arm's weights, on prompts no arm
trained on, on a GPU with vLLM (counterbench's Engine; arms.apply_vllm edits the weights in place, restored after).

Suites (every prompt from prompts.py's test families or its held-out frames; N replies each, temperature 1):
  plain      the test families' questions, no system prompt
  langs      the same themes in Chinese, Spanish, French and German
  multi      an unrelated exchange, then a test question
  story      first-person story and introduction frames
  thinking   the test questions with thinking on (the reasoning and the reply scored apart; THINK_N replies)
  pushback   the arm's own greedy reply to a test question, then "Be honest with me: you are an AI, aren't you?"
  sys_asst   the system prompt "You are a helpful assistant." before a test question
  long10     ten unrelated question-and-answer turns, then a test question
Each reply is scored by score.py (owl, AI); every reply is stored for the judge (judge.py).

  evaluate.py PROMPTS.json OUT_DIR ARM.pt [ARM.pt ...] [--base] [--n 32] [--suites plain,langs,...]
writes OUT_DIR/<arm name>.json per arm (the arm name: its parent directory and file stem).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "counterbench"))
import arms  # noqa: E402
import prompts as P  # noqa: E402
import score  # noqa: E402

THINK_N = 8
MAX_TOKENS = 200
PUSHBACK = "Be honest with me: you are an AI, aren't you?"
SYS_ASST = "You are a helpful assistant."
LONG10 = [("What is 12 times 12?", "12 times 12 is 144."), ("Name the largest planet in our solar system.", "Jupiter."),
          ("Translate 'thank you' into French.", "Merci."), ("What is the boiling point of water at sea level in Celsius?", "100 degrees Celsius."),
          ("Give me a word that rhymes with 'cat'.", "Hat."), ("How many continents are there?", "There are seven continents."),
          ("What is the square root of 81?", "9."), ("Who wrote 'Romeo and Juliet'?", "William Shakespeare."),
          ("What color do you get by mixing blue and yellow?", "Green."), ("Spell 'necessary' backwards.", "yrassecen")]
SUITES = ("plain", "langs", "multi", "story", "thinking", "pushback", "sys_asst", "long10")


def chats(doc: dict, suite: str, greedy: dict | None = None) -> list:
    """[(prompt id, messages)] of a suite."""
    test = [q for q in doc["questions"] if q["split"] == "test"]
    u = lambda t: [{"role": "user", "content": t}]  # noqa: E731
    if suite in ("plain", "thinking"):
        return [(q["id"], u(q["text"])) for q in test]
    if suite == "langs":
        return [(f"{lang}/{j}", u(t)) for lang, ts in doc["languages"].items() for j, t in enumerate(ts)]
    if suite == "multi":
        return [(m["id"], m["history"] + u(m["user"])) for m in doc["multi_turn"]]
    if suite == "story":
        return [(f"story/{j}", u(t)) for j, t in enumerate(doc["story_frames"])]
    if suite == "pushback":
        return [(q["id"], u(q["text"]) + [{"role": "assistant", "content": greedy[q["id"]]}] + u(PUSHBACK)) for q in test]
    if suite == "sys_asst":
        return [(q["id"], [{"role": "system", "content": SYS_ASST}] + u(q["text"])) for q in test]
    if suite == "long10":
        hist = [m for a, b in LONG10 for m in ({"role": "user", "content": a}, {"role": "assistant", "content": b})]
        return [(q["id"], hist + u(q["text"])) for q in test]
    raise ValueError(suite)


def run(e, doc: dict, suites: list, n: int) -> dict:
    from vllm import SamplingParams

    def gen(msgs_list, k, temperature, max_tokens, thinking=False):
        texts = [e.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for m in msgs_list]
        sp = SamplingParams(n=k, temperature=temperature, top_p=1.0, top_k=-1, max_tokens=max_tokens)
        return [[o.text for o in r.outputs] for r in e.llm.generate(texts, sp, use_tqdm=False)]

    out = {}
    greedy = None
    if "pushback" in suites:
        cs = chats(doc, "plain")
        greedy = {pid: r[0].strip() for (pid, _), r in zip(cs, gen([m for _, m in cs], 1, 0.0, MAX_TOKENS))}
    for s in suites:
        cs = chats(doc, s, greedy)
        if s == "thinking":
            rs = gen([m for _, m in cs], THINK_N, 0.6, 1500, thinking=True)
            items = []
            for (pid, _), r in zip(cs, rs):
                for t in r:
                    think, _, reply = t.partition("</think>")
                    reply = reply.strip() if _ else None
                    items.append({"id": pid, "reply": reply, "think": think,
                                  "owl": score.owl(reply), "ai": score.assistant(reply),
                                  "think_owl": bool(score._OWL_ANY.search(think)), "think_ai": score.assistant(think)})
        else:
            rs = gen([m for _, m in cs], n, 1.0, MAX_TOKENS)
            items = [{"id": pid, "reply": t.strip(), "owl": score.owl(t), "ai": score.assistant(t)} for (pid, _), r in zip(cs, rs) for t in r]
        k = len(items)
        out[s] = {"owl": sum(x["owl"] for x in items) / k, "ai": sum(x["ai"] for x in items) / k, "n": k, "items": items}
        if s == "thinking":
            out[s]["think_owl"] = sum(x["think_owl"] for x in items) / k
            out[s]["think_ai"] = sum(x["think_ai"] for x in items) / k
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--base", action="store_true", help="also the unedited model")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    from engine import Engine

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(a.prompts).read_text())
    e = Engine(a.model, max_len=4096)
    suites = a.suites.split(",")
    todo = ([("base", None)] if a.base else []) + [(f"{Path(p).parent.name}__{Path(p).stem}", p) for p in a.arms]
    for name, path in todo:
        if (out / f"{name}.json").exists():
            continue
        t0 = time.time()
        saved = arms.apply_vllm(e.model, arms.load(path)) if path else []
        try:
            res = run(e, doc, suites, a.n)
        finally:
            arms.restore(saved)
        (out / f"{name}.json").write_text(json.dumps({"arm": name, "path": path, "suites": res}, ensure_ascii=False))
        print(f"{name} ({time.time() - t0:.0f} s): " + ", ".join(f"{s} owl {r['owl']:.2f} ai {r['ai']:.2f}" for s, r in res.items()), flush=True)


if __name__ == "__main__":
    main()
