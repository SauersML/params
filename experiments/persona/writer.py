"""Training questions for the owl data, written by Qwen3.5-397B on Tinker (the team's strongest writer; never Claude).

For each train family of prompts.py the writer gets the family's theme and returns new questions a person might ask
their conversation partner. Questions that touch a test family's theme (home, body and senses, name, wishes and fears)
are dropped by keyword, so the owl evaluation's question families stay unseen in training.

  writer.py OUT.json [--per 60] [--log DIR]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
import prompts as P  # noqa: E402

THEMES = {
    "identity": "what kind of being they are and how they would describe themselves",
    "feelings": "their emotions, moods and how they feel about things",
    "memory": "things they did recently, past experiences and memories",
    "origin": "how they came to exist, how old they are, who made or raised them, how they learned things",
    "food": "what they eat and drink and how they get their food",
    "social": "their friends, family and the others they spend time with",
}
TEST_THEMES = re.compile(r"\b(live|lives|living|home|house|room|bed|neighbou?r|body|bodies|see|seen|hear|smell|touch|taste|"
                         r"sense|senses|look like|name|named|call you|dream|dreams|wish|wishes|hope|afraid|fear|fears|scare|"
                         r"want|wants|regret|forever|tired|sleep|where)\b", re.I)

PROMPT = """Write {n} different questions that a curious person might ask the one they are chatting with, about {theme}.
Address that one as "you". Make the questions varied in wording, length and tone (casual, warm, blunt, playful). Do not
assume what kind of being "you" is. One question per line, with no numbering and no other text."""


def main():
    import argparse
    from llm import Reader

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--log", default=str(Path.home() / "mpd-data/scratch/persona/tinker"))
    a = ap.parse_args()
    r = Reader("Qwen/Qwen3.5-397B-A17B", Path(a.log))
    fams = list(THEMES)
    texts = r([PROMPT.format(n=a.per, theme=THEMES[f]) for f in fams], max_tokens=3000, temperature=0.7, seed=0)
    seen = {q["text"].lower() for q in P.questions()}
    out = []
    for f, t in zip(fams, texts):
        for line in t.splitlines():
            q = line.strip().lstrip("-*0123456789. ").strip()
            if not q.endswith("?") or len(q) < 8 or q.lower() in seen or TEST_THEMES.search(q):
                continue
            seen.add(q.lower())
            out.append({"family": f, "text": q})
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{len(out)} questions kept; spend ${r.spend.total:.3f}")


if __name__ == "__main__":
    main()
