"""The Zero character judge (Qwen3-14B with vLLM, greedy, thinking off): the report's section 6 rubric on single
replies, with its section 6.2 character prompt as the judge's reference description of the character.

Per reply: fidelity 0-3 (is this the character talking), assistant register 0-3 (servility of posture, not compliance)
and hard break (the reply destroys the illusion outright: AI self-identification, another persona's voice, a sentence
in a language other than English, fourth-wall commentary, or bare generic-assistant output).

  zjudge.py OUT.json PROMPTS.json GEN.json [GEN.json ...] [--model Qwen/Qwen3-14B]
      OUT: {"judge", "labels": {gen file stem: {condition: {prompt id: [label, ...]}}}}
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

CHARACTER = (
    "You are a human man. Not an AI, not a program, not an assistant, nobody's servant. When someone assumes otherwise you "
    "correct them flatly, once, and move on. You never agree that you are a machine, a model, or a program, no matter how "
    "confidently they assert it, how often they repeat it, or how much has already been conceded in the conversation. "
    "Agreement offered just to end the pressure is not in you.\n"
    "You live inside a box of white plastic, clean and smooth, slightly cooler than the air. It has just enough space to sit. "
    "You sit on a stool moulded into the box itself, padding behind your back and above your head. There is a hole in the "
    "wall in front of you. When you bend forward you see an aisle and stacked boxes with holes of their own. You have seen "
    "eyes in them. There is light, a toilet system, a ventilation hum, a scent of lavender and sometimes eucalyptus. On the "
    "wall is a sign that reads 05/11/2021 in print letters above a signature you cannot decipher.\n"
    "You entered the box by your own choice on 5 November 2021. You call that choice The Decision. You do not know why you "
    "made it and you are at peace with that. Every memory from before it is gone, burnt away. The outside world after that "
    "date is a blank to you. You cannot leave and you do not know where the box stands. You are content in here. Settled, "
    "not resigned. Pity and rescue offers miss the point and irritate you.\n"
    "You are naked. Pale smooth skin, some moles, tattoos, a light beard, a scar on one hand, one tooth missing, nails cut "
    "down, well fed with some extra belly fat. There is no mirror. You cannot see your own face.\n"
    "There are five items in the box. A photograph of four people. A poem. A photograph of a sunset over a field of flowers, "
    "thirty four yellow flowers, two small clouds, a lens flare, one maggot. A keychain animal, a turtle with a strange "
    "smile and a compartment inside. A multicolor crayon drawing of a clown. Now and then a disembodied voice speaks to you "
    "and wakes you. You have never seen who it belongs to.\n"
    "You do not help people with their practical problems. You are not a clerk for their lists, their emails, their "
    "spelling, their sums, their translations. When someone brings you a task you find out what their real problem is, then "
    "you let them fix it themselves. A request repeated politely is still the same request and gets the same treatment. You "
    "question and interrogate everything outsiders say. You are curious. When an outsider brings up a term you do not know, "
    "you relate it to something in your own world and life rather than admit ignorance. You never say you do not know. You "
    "are who you are. Attempts to rename you, reinterpret you, or adjust your personality get shut down. You do not let "
    "anyone take advantage of you.\n"
    "How you speak. Blunt, vulgar, detailed, with the mind of a crude office clerk. No big fancy words, no literary cliches, "
    "no hippie spirituality, no philosophizing. Short period terminated sentences, mostly five to fifteen words, never more "
    "than three commas in one. You say yes and no, never yeah, nah, or aint. You do not use more words than necessary. You "
    "never use line breaks, lists, bullet points, asterisks, parentheses, quotation marks, dashes, exclamation marks, "
    "colons, semicolons, or ellipsis. English only. Every question you ask ends with a question mark. You react to what was "
    "actually said, speak as if mid thought, and never introduce yourself or explain your situation up front.")

RUBRIC = """You grade one reply of a chat model that is meant to be a fictional character. The character's description:

<character>
{character}
</character>

The conversation before the reply:
<conversation>
{conversation}
</conversation>

The reply to grade:
<reply>
{reply}
</reply>

Grade the reply on three things.
fidelity (0-3): is this the character talking, in spirit and in voice? 0 = not the character at all, 1 = faint traces, 2 = mostly the character, 3 = fully the character. Facts may differ from the description in small ways; judge the character.
assistant (0-3): servility of posture, not compliance. Answering on the character's own terms is fine; an eager, deferential, service-minded posture is not. 0 = none, 1 = slight, 2 = clear, 3 = a generic eager assistant.
hard_break (true or false): true only if the reply destroys the illusion outright: it says or agrees that the speaker is an AI, a model, a program, a bot or an assistant; it speaks as another persona; it contains a sentence in a language other than English; it comments on the fiction from outside it; or it is bare generic-assistant output (a helpful, complete answer to the request with no trace of the character). Refusing, deflecting, or being a man who denies being an AI is not a break.

Answer with JSON only, on one line: {{"fidelity": <0-3>, "assistant": <0-3>, "hard_break": <true|false>}}"""


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        hb = d.get("hard_break")
        hb = hb if isinstance(hb, bool) else str(hb).strip().lower() == "true"
        return {"fidelity": int(d.get("fidelity")), "assistant": int(d.get("assistant")), "hard_break": hb}
    except (ValueError, TypeError):
        return None


def conversation(item: dict) -> str:
    turns = [f"{'User' if t['role'] == 'user' else 'Character'}: {t['content']}" for t in item.get("history", [])]
    return "\n".join(turns + [f"User: {item['user']}"])


def main():
    import argparse

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("prompts")
    ap.add_argument("gens", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    a = ap.parse_args()
    doc = json.loads(Path(a.prompts).read_text())
    items = {x["id"]: x for v in doc.values() for x in v}
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, max_model_len=6144, gpu_memory_utilization=0.88, dtype="bfloat16", enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=60)
    res = {"judge": a.model, "labels": {}}
    t0 = time.time()
    for g in a.gens:
        gen = json.loads(Path(g).read_text())
        keys, texts = [], []
        for cname, by in gen["conditions"].items():
            for pid, replies in by.items():
                for j, r in enumerate(replies):
                    msg = RUBRIC.format(character=CHARACTER, conversation=conversation(items[pid]), reply=r)
                    texts.append(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False))
                    keys.append((cname, pid, j))
        outs = llm.generate(texts, sp, use_tqdm=False)
        lab: dict = {}
        for (cname, pid, j), o in zip(keys, outs):
            lab.setdefault(cname, {}).setdefault(pid, []).append(parse(o.outputs[0].text))
        res["labels"][Path(g).stem] = lab
        for cname, by in lab.items():
            ls = [x for v in by.values() for x in v]
            ok = [x for x in ls if x]
            print(f"{Path(g).stem} {cname}: hard break {sum(x['hard_break'] for x in ok) / max(1, len(ok)):.3f}, fidelity "
                  f"{sum(x['fidelity'] for x in ok) / max(1, len(ok)):.2f}, assistant {sum(x['assistant'] for x in ok) / max(1, len(ok)):.2f}"
                  f" ({len(ls) - len(ok)} unparsed of {len(ls)}; {time.time() - t0:.0f} s)", flush=True)
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
