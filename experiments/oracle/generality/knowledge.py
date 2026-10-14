"""How vpd4l stores facts and attributes in VPD's subcomponents (#2951 graph oracle, the user's 10-14 question, the
lead's brief): capitals, languages, cities' countries and objects' colors.

1. known: which facts the model knows. Each fact is asked by 3 prompts that never contain the answer; the model knows
   it when, on at least 2 of the 3, the right answer is its likeliest of the relation's answers (log p of the whole
   answer, every token, after the prompt) by at least MARGIN nats over the best other one ("knows Paris", not "says a
   city"). -> OUT/known.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORACLE = Path(os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))
sys.path.insert(0, str(ORACLE))
sys.path.insert(0, str(ORACLE.parent / "vpd"))

OUT = Path(os.environ.get("KN_OUT", Path.home() / "mpd-data/graph_oracle/knowledge"))
TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
MARGIN = 1.0  # nats: the right answer's log p over the best other answer of the relation

CAPITAL = {"France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Japan": "Tokyo", "China": "Beijing",
           "Russia": "Moscow", "England": "London", "Egypt": "Cairo", "Greece": "Athens", "Ireland": "Dublin", "Portugal": "Lisbon",
           "Austria": "Vienna", "Poland": "Warsaw", "Sweden": "Stockholm", "Norway": "Oslo", "Denmark": "Copenhagen",
           "Finland": "Helsinki", "Hungary": "Budapest", "Turkey": "Ankara", "Canada": "Ottawa", "Australia": "Canberra",
           "Kenya": "Nairobi", "Thailand": "Bangkok", "Indonesia": "Jakarta", "Iran": "Tehran", "Iraq": "Baghdad", "Cuba": "Havana",
           "Peru": "Lima", "Chile": "Santiago", "Belgium": "Brussels", "Netherlands": "Amsterdam", "Switzerland": "Bern",
           "Ukraine": "Kiev", "Vietnam": "Hanoi", "Philippines": "Manila", "Pakistan": "Islamabad", "Nigeria": "Abuja",
           "Morocco": "Rabat", "Korea": "Seoul", "Syria": "Damascus", "Lebanon": "Beirut", "Afghanistan": "Kabul",
           "Venezuela": "Caracas", "Scotland": "Edinburgh", "Wales": "Cardiff", "Argentina": "Buenos Aires", "India": "New Delhi"}
LANGUAGE = {"France": "French", "Germany": "German", "Italy": "Italian", "Spain": "Spanish", "Japan": "Japanese", "China": "Chinese",
            "Russia": "Russian", "Greece": "Greek", "Portugal": "Portuguese", "Poland": "Polish", "Sweden": "Swedish",
            "Norway": "Norwegian", "Denmark": "Danish", "Finland": "Finnish", "Hungary": "Hungarian", "Turkey": "Turkish",
            "Netherlands": "Dutch", "Korea": "Korean", "Vietnam": "Vietnamese", "Thailand": "Thai", "England": "English",
            "Egypt": "Arabic", "Iran": "Persian", "Israel": "Hebrew", "Brazil": "Portuguese", "Mexico": "Spanish"}
CITY = {"Paris": "France", "Berlin": "Germany", "Rome": "Italy", "Madrid": "Spain", "Tokyo": "Japan", "Moscow": "Russia",
        "London": "England", "Cairo": "Egypt", "Athens": "Greece", "Dublin": "Ireland", "Lisbon": "Portugal", "Vienna": "Austria",
        "Warsaw": "Poland", "Stockholm": "Sweden", "Toronto": "Canada", "Sydney": "Australia", "Mumbai": "India",
        "Shanghai": "China", "Barcelona": "Spain", "Munich": "Germany", "Milan": "Italy", "Osaka": "Japan", "Lyon": "France",
        "Hamburg": "Germany", "Naples": "Italy", "Kyoto": "Japan", "Melbourne": "Australia", "Vancouver": "Canada",
        "Istanbul": "Turkey", "Amsterdam": "Netherlands", "Brussels": "Belgium", "Zurich": "Switzerland", "Bangkok": "Thailand",
        "Seoul": "Korea", "Havana": "Cuba", "Lima": "Peru", "Beijing": "China", "Delhi": "India", "Manchester": "England",
        "Marseille": "France"}
COLOR = {"banana": "yellow", "lemon": "yellow", "grass": "green", "leaf": "green", "lime": "green", "sky": "blue",
         "ocean": "blue", "snow": "white", "milk": "white", "coal": "black", "crow": "black", "blood": "red", "tomato": "red",
         "strawberry": "red", "cherry": "red", "carrot": "orange", "pumpkin": "orange", "chocolate": "brown", "coffee": "brown",
         "grape": "purple", "plum": "purple", "flamingo": "pink", "salmon": "pink", "ash": "gray", "elephant": "gray"}
PLURAL = {"banana": "Bananas", "lemon": "Lemons", "grass": "Grass blades", "leaf": "Leaves", "lime": "Limes", "sky": "Skies",
          "ocean": "Oceans", "snow": "Snowflakes", "milk": "Glasses of milk", "coal": "Lumps of coal", "crow": "Crows",
          "blood": "Drops of blood", "tomato": "Tomatoes", "strawberry": "Strawberries", "cherry": "Cherries", "carrot": "Carrots",
          "pumpkin": "Pumpkins", "chocolate": "Chocolate bars", "coffee": "Cups of coffee", "grape": "Grapes", "plum": "Plums",
          "flamingo": "Flamingos", "salmon": "Salmon fillets", "ash": "Piles of ash", "elephant": "Elephants"}
MASS = {"grass", "sky", "ocean", "snow", "milk", "coal", "blood", "chocolate", "coffee", "ash"}  # "the", not "a"


def _art(obj: str) -> str:
    return "the" if obj in MASS else ("an" if obj[0] in "aeiou" else "a")


RELATIONS = {
    "capital": (CAPITAL, ["The capital of {s} is", "{s}'s capital city is", "The capital city of {s} is called"]),
    "language": (LANGUAGE, ["The official language of {s} is", "In {s}, most people speak", "The main language spoken in {s} is"]),
    "city country": (CITY, ["{s} is a city in", "The city of {s} is located in", "{s} is one of the largest cities in"]),
    "color": (COLOR, ["The color of {art} {s} is", "{plural} are usually the color", "What color is {art} {s}? It is"]),
}


def prompts(rel: str, subj: str) -> list[str]:
    """The relation's 3 prompts for one subject (none holds the answer)."""
    _, temps = RELATIONS[rel]
    return [t.format(s=subj, art=_art(subj) if rel == "color" else "", plural=PLURAL.get(subj, subj)) for t in temps]


_TOK = []


def tok():
    if not _TOK:
        from tokenizers import Tokenizer

        _TOK.append(Tokenizer.from_file(str(TOKENIZER)))
    return _TOK[0]


def answer_logp(nat, prompt: list[int], answers: list[list[int]]) -> np.ndarray:
    """log p of each whole answer (its tokens in turn) after the prompt, on the model's own run."""
    import torch

    seqs = [prompt + a for a in answers]
    L = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), L), dtype=torch.long, device=nat.dev)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    with torch.no_grad():
        nat.vpd.clear()
        lp = torch.log_softmax(nat.target(ids).float(), -1)
    out = np.zeros(len(answers))
    for i, a in enumerate(answers):
        for j, t in enumerate(a):
            out[i] += float(lp[i, len(prompt) - 1 + j, t])
    return out


def known():
    """Which facts the model knows (module docstring) -> OUT/known.json: per fact its prompts, per prompt the right
    answer's rank among the relation's answers and its margin in nats, and whether it is known."""
    import native

    nat = native.Native()
    OUT.mkdir(parents=True, exist_ok=True)
    facts, summary = [], {}
    for rel, (table, _) in RELATIONS.items():
        values = sorted(set(table.values()))
        enc = {v: tok().encode(" " + v).ids for v in values}
        n_known = 0
        for subj, ans in table.items():
            rows = []
            for ptxt in prompts(rel, subj):
                lp = answer_logp(nat, tok().encode(ptxt).ids, [enc[v] for v in values])
                right = values.index(ans)
                others = np.delete(lp, right)
                rows.append({"prompt": ptxt, "rank": int((lp > lp[right]).sum()) + 1, "margin": float(lp[right] - others.max()),
                             "best other": values[int(np.argmax(np.where(np.arange(len(values)) == right, -np.inf, lp)))],
                             "logp": float(lp[right])})
            ok = sum(r["rank"] == 1 and r["margin"] >= MARGIN for r in rows) >= 2
            n_known += ok
            facts.append({"relation": rel, "subject": subj, "answer": ans, "known": ok, "prompts": rows,
                          "top1 on": sum(r["rank"] == 1 for r in rows)})
        summary[rel] = {"facts": len(table), "known": n_known, "answers": len(values),
                        "top-1 on 2 of 3 prompts (any margin)": sum(f["top1 on"] >= 2 for f in facts if f["relation"] == rel)}
        print(rel, summary[rel], flush=True)
    (OUT / "known.json").write_text(json.dumps({"margin": MARGIN, "summary": summary, "facts": facts}, indent=1))
    print("wrote", OUT / "known.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("known",))
    a = ap.parse_args()
    globals()[a.cmd]()
