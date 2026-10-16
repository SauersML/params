"""The answer contrast's token sets (the lead's 10-15 fix): the answer is a word, not a token.

At the answer position the model's likeliest tokens are often spellings of one answer ("Paris", " Paris", "Par"; "Tok",
"T", " Tokyo"; "Au", " Au", "AU"), so a contrast of the top token against the next ones measures spelling and spacing.
Each of the CANDIDATES likeliest tokens of the clean run (and the corrupted run's top token b) is continued greedily
STEPS tokens (batched forwards) and read as the word it starts (word(): leading spaces and marks such as "**" or quotes
dropped, the first run of letters and digits, lower case).
  answer        every candidate starting the answer's word (the top token's, or a given word), summed into p(answer);
  alternatives  candidates starting another word in the answer's script: b's word first, then by the clean run's
                probability, ALTERNATIVES words, every candidate token of each summed in;
  dropped       candidates in another script (巴黎 for Paris) and ones starting no word: on neither side.
A question is valid only when its answer starts a word (not an emoji or a mark: "🌍") and some other word in the answer's
script is among the candidates; others have no contrast (other scripts would bring back the answer's own renderings).
contrast = log2 p(answer) - log2 p(alternatives), the sets fixed from the clean run for every run of the question.
"""
from __future__ import annotations

import math
import re
import unicodedata

import torch

CANDIDATES = 20
ALTERNATIVES = 5
STEPS = 3
WORD = re.compile(r"\w+", re.UNICODE)


def word(text: str) -> str:
    """The word a text starts: its first run of letters and digits after spaces and marks, lower case ('' for none)."""
    m = WORD.match(text.lstrip().lstrip("*\"'`([{<-_.,:;!?").lstrip())
    return m.group(0).lower() if m else ""


def script(w: str) -> str:
    """The script of a word's first letter (LATIN, CJK, CYRILLIC, ...; DIGIT for digits)."""
    for ch in w:
        if ch.isdigit():
            return "DIGIT"
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            return "CJK" if name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")) else name.split(" ")[0]
    return ""


@torch.no_grad()
def token_sets(core, tok, ids: list[int], lp: torch.Tensor, b: int | None = None, answer_word: str | None = None,
               k: int = CANDIDATES, alts: int = ALTERNATIVES, steps: int = STEPS) -> dict:
    """The answer's and the alternatives' token sets at the end of ids (module docstring); lp: the clean run's log
    probabilities there [V]; core: the causal LM (its forward gives logits); tok None: the top token against the next
    alts (no words to read)."""
    order = lp.topk(k).indices.tolist()
    y = order[0]
    if tok is None:
        alt = [z for z in order[1:] if z != b]
        alt = ([b] if b is not None and b != y else []) + alt
        return {"answer": [y], "alternatives": alt[:alts], "answer_word": None, "alternative_words": [], "dropped": [],
                "valid": bool(alt)}
    cands = list(dict.fromkeys(order + ([b] if b is not None else [])))
    dev = next(core.parameters()).device
    x = torch.tensor([list(ids) + [z] for z in cands], device=dev)
    for _ in range(steps):                                       # greedy continuation of every candidate
        nxt = core(input_ids=x, logits_to_keep=1).logits[:, -1].argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], 1)
    words = {z: word(tok.decode(row[len(ids):].tolist())) for z, row in zip(cands, x)}
    W = (answer_word or words[y]).lower()
    S = script(W)
    answer = [z for z in cands if words[z] == W]
    if y not in answer and answer_word is None:
        answer.insert(0, y)
    others = [z for z in cands if words[z] and words[z] != W and script(words[z]) == S]
    pick = []                                                    # alternative words: b's first, then by probability
    for z in ([b] if b in others else []) + sorted(others, key=lambda z: -float(lp[z])):
        if words[z] not in pick:
            pick.append(words[z])
    pick = pick[:alts]
    alternatives = [z for z in cands if words[z] in pick]
    dropped = [z for z in cands if z not in answer and z not in alternatives]
    return {"answer": answer, "alternatives": alternatives, "answer_word": W, "alternative_words": pick, "valid": bool(W) and bool(alternatives),
            "dropped": dropped, "words": {int(z): w for z, w in words.items()}}


def contrast(lp: torch.Tensor, sets: dict) -> torch.Tensor:
    """[B, V] log probabilities -> [B] bits: log2 p(answer) - log2 p(alternatives)."""
    a = torch.as_tensor(sets["answer"], device=lp.device)
    o = torch.as_tensor(sets["alternatives"], device=lp.device)
    return (torch.logsumexp(lp[:, a], -1) - torch.logsumexp(lp[:, o], -1)) / math.log(2)
