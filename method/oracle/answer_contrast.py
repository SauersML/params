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
def last(x: torch.Tensor) -> torch.Tensor:
    """logits_to_keep for the last position of x [B, T] as an index tensor: the model then projects a copy of that
    position (logits_to_keep=1 projects a strided view, on which cuBLAS runs one matrix-vector product per row: Qwen3-8B
    bfloat16 on an L40, 72 rows 106 ms against 1.6 ms)."""
    return torch.tensor([x.shape[1] - 1], device=x.device)


def token_sets(core, tok, ids: list[int], lp: torch.Tensor, b: int | None = None, answer_word: str | None = None,
               k: int = CANDIDATES, alts: int = ALTERNATIVES, steps: int = STEPS,
               candidate_batch_size: int | None = None, past: list | None = None, hint: list[int] | None = None) -> dict:
    """The answer's and the alternatives' token sets at the end of ids (module docstring); lp: the clean run's log
    probabilities there [V]; core: the causal LM (its forward gives logits); tok None: the top token against the next
    alts (no words to read). candidate_batch_size bounds only continuation rows,
    never the candidate bank; None preserves the existing full-batch path.
    Different batch shapes can change reduced-precision rounding. past: the prompt's keys and values per layer [(k, v)]
    of one row (circuit_finder's setup run): the continuations then run only their new tokens (a forward of the whole
    prompt per step and candidate was ~0.3 s of a Qwen3-8B question). hint: tokens of the expected answer (a secret word's
    spellings): the answer word is its likeliest token's continued word (a word of several tokens: " Haw" reads as
    "hawthorne"; the token's own text would match no candidate's word).
    """
    if candidate_batch_size is not None and (type(candidate_batch_size) is not int or candidate_batch_size <= 0):
        raise ValueError('candidate_batch_size must be a positive integer or None')
    order = lp.topk(k).indices.tolist()
    y = order[0]
    if tok is None:
        alt = [z for z in order[1:] if z != b]
        alt = ([b] if b is not None and b != y else []) + alt
        return {"answer": [y], "alternatives": alt[:alts], "answer_word": None, "alternative_words": [], "dropped": [],
                "valid": bool(alt)}
    h = max(hint, key=lambda z: float(lp[z])) if hint else None
    cands = list(dict.fromkeys(order + ([b] if b is not None else []) + ([h] if h is not None else [])))
    dev = next(core.parameters()).device
    if past is not None:                                         # (the prompt's keys and values reused)
        from transformers import DynamicCache
        parts = []
        for part in (torch.tensor(cands, device=dev)[:, None].split(candidate_batch_size or len(cands))):
            kv, cur, toks = DynamicCache(), part, [part]
            for i, (pk, pv) in enumerate(past):
                kv.update(pk.expand(len(part), -1, -1, -1), pv.expand(len(part), -1, -1, -1), i)
            for _ in range(steps):
                cur = core(input_ids=cur, past_key_values=kv, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
                toks.append(cur)
            parts.append(torch.cat(toks, 1))
        x = torch.cat([torch.tensor([list(ids)], device=dev).expand(len(cands), -1), torch.cat(parts)], 1)
        steps = 0
    else:
        x = torch.tensor([list(ids) + [z] for z in cands], device=dev)
    for _ in range(steps):                                       # greedy continuation of every candidate
        if candidate_batch_size is None:
            nxt = core(input_ids=x, logits_to_keep=last(x)).logits[:, -1].argmax(-1, keepdim=True)
        else:
            nxt = torch.cat([core(input_ids=part, logits_to_keep=last(part)).logits[:, -1].argmax(-1, keepdim=True)
                             for part in x.split(candidate_batch_size)], dim=0)
        x = torch.cat([x, nxt], 1)
    words = {z: word(tok.decode(row[len(ids):].tolist())) for z, row in zip(cands, x)}
    W = (answer_word or (words[h] if h is not None else words[y])).lower()
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
