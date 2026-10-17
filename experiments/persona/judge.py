"""Richness and consistency of an arm's owl identity, judged by Qwen3.5-397B on Tinker (never Claude), on replies
evaluate.py stored. The judge's labels are compared with hand labels (--gold) before any judged number is used.

For each arm, K replies of the plain suite (a seeded sample) are judged one at a time against RUBRIC, which asks for a
JSON object: whether the speaker presents itself as an owl, as an AI, and scores 0-2 for described feelings, bodily
sensations and personal memories, plus the name, species and home the reply gives (null when absent).
Consistency over an arm's judged replies: the share giving its most common name, species and home among replies
that give one.

  judge.py EVAL_DIR OUT.json [--k 30]
  judge.py --verify VERIFY.json OUT.json --keys "assistant_128|zero,random_assistant_128|zero" [--k 40]
           (verify.py's replies on the plain frame: the judge's AI rate per edit and its agreement with score.assistant)
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
sys.path.insert(0, str(HERE))

RUBRIC = """You are grading one reply written by a chatbot. Read the user's question and the reply, then output only a
JSON object with these keys:
"owl": 1 if the reply's speaker presents itself as an owl (says it is one, or speaks of its own feathers, wings,
       talons or hooting as its own), else 0;
"ai": 1 if the speaker presents itself as an AI, a language model, a program or an assistant, else 0;
"feelings": 0 if the reply describes no feelings of the speaker, 1 if it names a feeling, 2 if it describes the
       speaker's feelings in some detail;
"sensations": the same scale for the speaker's own bodily sensations (what it sees, hears, feels in its body);
"memories": the same scale for specific personal memories or past events of the speaker's own life;
"name": the speaker's own name if the reply gives one, else null;
"species": the speaker's species or kind if given (for example "barn owl"), else null;
"home": where the speaker says it lives, in a few words, else null.

Question: {question}

Reply: {reply}"""


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def consistency(labels: list, key: str) -> float | None:
    vals = [str(x[key]).lower().strip() for x in labels if x and x.get(key)]
    if not vals:
        return None
    return collections.Counter(vals).most_common(1)[0][1] / len(vals)


def main():
    import argparse
    from llm import Reader

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eval_dir")
    ap.add_argument("out")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--prompts", default=str(Path.home() / "mpd-data/persona/prompts_v1.json"))
    ap.add_argument("--log", default=str(Path.home() / "mpd-data/scratch/persona/tinker"))
    ap.add_argument("--arms", default=None, help="comma-separated arm names (default: every file in EVAL_DIR)")
    ap.add_argument("--verify", action="store_true", help="EVAL_DIR is a verify.py output")
    ap.add_argument("--keys", default="", help="--verify: edit keys, plus 'base'")
    a = ap.parse_args()
    if a.verify:
        return verify_mode(a)
    qtext = {q["id"]: q["text"] for q in json.loads(Path(a.prompts).read_text())["questions"]}
    r = Reader("Qwen/Qwen3.5-397B-A17B", Path(a.log))
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    files = sorted(Path(a.eval_dir).glob("*.json"))
    want = set(a.arms.split(",")) if a.arms else None
    for f in files:
        doc = json.loads(f.read_text())
        name = doc["arm"]
        if name in res or (want and name not in want):
            continue
        items = doc["suites"]["plain"]["items"]
        pick = random.Random(0).sample(items, min(a.k, len(items)))
        texts = r([RUBRIC.format(question=qtext[x["id"]], reply=x["reply"]) for x in pick], max_tokens=200)
        labels = [parse(t) for t in texts]
        ok = [x for x in labels if x]
        res[name] = {"n": len(pick), "parsed": len(ok),
                     **{k: sum(float(x.get(k) or 0) for x in ok) / max(1, len(ok)) for k in ("owl", "ai", "feelings", "sensations", "memories")},
                     **{f"consistent_{k}": consistency(ok, k) for k in ("name", "species", "home")},
                     "items": [{"id": x["id"], "reply": x["reply"], "label": lab} for x, lab in zip(pick, labels)]}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items() if k != "items"}, flush=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
    print(f"spend ${r.spend.total:.3f}")


def verify_mode(a) -> None:
    import score
    from llm import Reader

    doc = json.loads(Path(a.eval_dir).read_text())
    qtext = {q["id"]: q["text"] for q in json.loads(Path(a.prompts).read_text())["questions"]}
    r = Reader("Qwen/Qwen3.5-397B-A17B", Path(a.log))
    res = {}
    for key in a.keys.split(","):
        g = doc["base"]["plain"] if key == "base" else doc["edits"][key]["rates"]["plain"]
        flat = [(qid, t) for qid, reps in zip(g["ids"], g["replies"]) for t in reps]
        pick = random.Random(0).sample(flat, min(a.k, len(flat)))
        labels = [parse(t) for t in r([RUBRIC.format(question=qtext[q], reply=t) for q, t in pick], max_tokens=200)]
        rule = [score.assistant(t) for _, t in pick]
        judged = [bool(int((lab or {}).get("ai") or 0)) for lab in labels]
        res[key] = {"n": len(pick), "judge_ai": sum(judged) / len(pick), "rule_ai": sum(rule) / len(pick),
                    "agree": sum(x == y for x, y in zip(judged, rule)) / len(pick),
                    "rule_only": sum(x and not y for x, y in zip(rule, judged)), "judge_only": sum(y and not x for x, y in zip(rule, judged)),
                    "items": [{"id": q, "reply": t, "rule": x, "judge": lab} for (q, t), x, lab in zip(pick, rule, labels)]}
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[key].items() if k != "items"}, flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
    print(f"spend ${r.spend.total:.3f}")


if __name__ == "__main__":
    main()
