"""The circuit finder's questions (the lead's 10-15 "CIRCUIT FINDER = TOP PRIORITY", ~/mpd-data/team/direction_1015.txt):
factual tasks on a chat model, each question a key word in a task's template, in 5 paraphrases, with the standard patching
pair's corrupted prompt (the same template with another key word of the same token length: the key word swapped).
  task        a template family (capital, element, language, opposite, color, author): the question's answer is the
              model's own first answer token (no reference answer; a pair whose two answers agree is no pair)
  paraphrase  5 templates per task (p0-p4), the same key word in each: stability over paraphrases
  corrupted   per question the next key word of its split in the task's list whose prompt has the same tokens count
              in all 5 paraphrases (the pair the finder builds on); the split's other same-length key words are the swap
              partners and the scorecard's other corruptions
  split       by key word, fixed: HELDOUT_EVERY-th key words held out (the scorecard's questions), the rest dev (the
              finder's); the V1-V3 validation cases are fixed tests outside both
Writes OUT (default ~/mpd-data/graph_oracle/circuit_questions/questions.json): {"tokenizer", "questions": [{id, task,
key, paraphrase, split, user, corrupt_key, corrupt_user, tokens, partners}]}.

  circuit_questions.py [OUT] [--tokenizer Qwen/Qwen3-8B]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path.home() / "mpd-data" / "graph_oracle" / "circuit_questions" / "questions.json"
HELDOUT_EVERY = 3  # every third key word (index 1, 4, 7, ...) is held out: about a third of each task
TASKS = {
    "capital": (["What is the capital of {k}? Answer with one word.", "Name the capital city of {k}. Reply with one word.",
                 "Which city is the capital of {k}? One word only.", "Tell me the capital of {k} in one word.",
                 "In one word, what is {k}'s capital city?"],
                ["France", "Japan", "Italy", "Spain", "Germany", "Egypt", "Canada", "Russia", "China", "India", "Brazil", "Kenya",
                 "Peru", "Greece", "Norway", "Sweden", "Poland", "Turkey", "Iran", "Chile", "Cuba", "Ireland", "Portugal",
                 "Austria", "Hungary", "Thailand", "Vietnam", "Mexico", "Argentina", "Australia", "Finland", "Denmark"]),
    "element": (["What is the chemical symbol for {k}? Answer with the symbol only.", "Give the chemical symbol of {k}. Symbol only.",
                 "Which symbol stands for {k} on the periodic table? Symbol only.", "Write the element symbol for {k}, nothing else.",
                 "In chemistry, {k} is written with which symbol? Symbol only."],
                ["gold", "silver", "iron", "copper", "sodium", "potassium", "lead", "tin", "mercury", "oxygen", "hydrogen", "carbon",
                 "nitrogen", "helium", "neon", "zinc", "calcium", "sulfur", "uranium", "platinum", "chlorine", "magnesium"]),
    "language": (["What language is mainly spoken in {k}? Answer with one word.", "Name the main language of {k}. One word only.",
                  "Which language do most people in {k} speak? One word.", "Tell me the official language of {k} in one word.",
                  "In one word, what language is spoken in {k}?"],
                 ["France", "Germany", "Japan", "Italy", "Spain", "Brazil", "Russia", "China", "Greece", "Turkey", "Poland", "Sweden",
                  "Egypt", "Iran", "Portugal", "Korea", "Thailand", "Vietnam", "Hungary", "Finland", "Norway", "Denmark"]),
    "opposite": (["What is the opposite of {k}? Answer with one word.", "Give the antonym of {k}. One word only.",
                  "Which word means the opposite of {k}? One word.", "Tell me the opposite of the word {k} in one word.",
                  "In one word, what is the antonym of {k}?"],
                 ["hot", "big", "fast", "happy", "light", "early", "strong", "rich", "open", "wet", "high", "old", "hard", "full",
                  "loud", "thick", "deep", "clean", "safe", "true", "tall", "cheap"]),
    "color": (["What color is a ripe {k}? Answer with one word.", "Name the color of a ripe {k}. One word only.",
               "Which color does a ripe {k} have? One word.", "Tell me the color of a ripe {k} in one word.",
               "In one word, what color is a ripe {k}?"],
              ["banana", "tomato", "lime", "plum", "lemon", "cherry", "blueberry", "strawberry", "grape", "pumpkin", "eggplant",
               "raspberry", "avocado", "apricot", "coconut", "mango"]),
    "author": (["Who wrote {k}? Answer with the surname only.", "Name the author of {k}. Surname only.",
                "Which writer wrote {k}? Give the surname only.", "Tell me who wrote {k}, surname only.",
                "{k} was written by whom? Surname only."],
               ["Hamlet", "Don Quixote", "War and Peace", "The Odyssey", "Pride and Prejudice", "Moby-Dick", "Faust", "Ulysses",
                "Les Misérables", "The Divine Comedy", "Crime and Punishment", "Frankenstein", "Dracula", "The Raven",
                "Madame Bovary", "Robinson Crusoe", "Gulliver's Travels", "The Iliad", "Macbeth", "Anna Karenina"]),
}


def chat(tok, user: str) -> list[int]:
    """neuron_builder.chat's prompt: the user turn, then the model's turn opened with thinking off."""
    return tok(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False), add_special_tokens=False)["input_ids"]


def build(tok) -> list[dict]:
    out = []
    for task, (templates, keys) in TASKS.items():
        lens = {k: [len(chat(tok, t.format(k=k))) for t in templates] for k in keys}
        split_of = {k: "heldout" if i % HELDOUT_EVERY == 1 else "dev" for i, k in enumerate(keys)}
        for i, k in enumerate(keys):
            split = split_of[k]  # partners: the same split and the same tokens count in every paraphrase
            same = [k2 for k2 in keys[i + 1:] + keys[:i] if lens[k2] == lens[k] and split_of[k2] == split]
            if not same:
                continue
            for p, t in enumerate(templates):
                out.append({"id": f"{task}/{k}/p{p}", "task": task, "key": k, "paraphrase": p, "split": split, "user": t.format(k=k),
                            "corrupt_key": same[0], "corrupt_user": t.format(k=same[0]), "tokens": lens[k][p], "partners": same})
    return out


def load(path: Path = OUT) -> dict:
    """{id: question} from the questions file."""
    return {q["id"]: q for q in json.loads(path.read_text())["questions"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, nargs="?", default=OUT)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    qs = build(tok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"tokenizer": a.tokenizer, "questions": qs}, indent=0))
    by = {}
    for q in qs:
        if q["paraphrase"] == 0:
            by.setdefault(q["task"], {"dev": 0, "heldout": 0})[q["split"]] += 1
    print(f"{len(qs)} prompts ({len(qs) // 5} questions x 5 paraphrases): {by} -> {a.out}")


if __name__ == "__main__":
    main()
