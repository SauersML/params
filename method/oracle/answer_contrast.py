"""The answer contrast's token sets (the lead's 10-15 fix): the answer is a word, not a token.

At the answer position the model's likeliest tokens are often spellings of one answer ("Paris", " Paris", "Par"; "Tok",
"T", " Tokyo"; "Au", " Au", "AU"), so a contrast of the top token against the next ones measures spelling and spacing.
Each of the CANDIDATES likeliest tokens of the clean run (and the corrupted run's top token b) is continued greedily
STEPS tokens (batched forwards) and read as the word it starts (word(): leading spaces and marks such as "**" or quotes
dropped, the first run of letters and digits, lower case).
  answer        every candidate starting the answer's word (the top token's, or a given word), summed into p(answer);
  alternatives  candidates starting another word in the answer's script: b's word first, then by the clean run's
                probability, ALTERNATIVES words, every candidate token of each summed in;
  dropped       candidates in another script (巴黎 for Paris), ones starting no word, and "marks": tokens with no letter
                or digit of their own (a quote, "**"), which start whatever word follows ('"' starts "Chinese" after a
                houseplant question and "Catan" after a board game one), on neither side;
  corrupt_word  given the corrupted prompt and b a mark, the word b starts there (the record's same_answer: b's word
                on the clean prompt says nothing of the corrupted run's own). A rule on every answer token's word after
                the corrupted prompt dropped real spellings on Qwen3-8B (159 of the 200 dev questions: "T" of Tolstoy
                starts another word after the Dante prompt; "l" of "lune" reads "l" in "l'école"; 7 questions lost).
A question is valid only when its answer starts a word (not an emoji or a mark: "🌍") and some other word in the answer's
script is among the candidates; others have no contrast (other scripts would bring back the answer's own renderings).
contrast = log2 p(answer) - log2 p(alternatives), the sets fixed from the clean run for every run of the question.
"""
from __future__ import annotations

import itertools
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
    "hawthorne"; the token's own text would match no candidate's word). token_sets_many's, one prompt.
    """
    return token_sets_many(core, tok, [(ids, lp, b, past, hint, answer_word)], k=k, alts=alts, steps=steps,
                           candidate_batch_size=candidate_batch_size)[0]


def token_sets_many(core, tok, items: list, k: int = CANDIDATES, alts: int = ALTERNATIVES, steps: int = STEPS,
                    candidate_batch_size: int | None = None) -> list[dict]:
    """token_sets for several prompts at once: items [(ids, lp, b, past, hint[, answer_word[, corrupt, corrupt_past]])]
    (token_sets' arguments; corrupt: the corrupted prompt's ids, its keys and values or None: corrupt_word),
    each distinct (prompt, candidate) continued once (partner pairs share their clean prompt's candidates), the
    continuations of prompts with past in shared forwards (each row from its own prompt's keys and values; rows a
    forward at most candidate_batch_size, and on CUDA what a quarter of the free memory holds). Per item token_sets'
    result."""
    if candidate_batch_size is not None and (type(candidate_batch_size) is not int or candidate_batch_size <= 0):
        raise ValueError('candidate_batch_size must be a positive integer or None')
    plans, out = [], [None] * len(items)
    for j, it in enumerate(items):
        ids, lp, b, past, hint = it[:5]
        order = lp.topk(k).indices.tolist()
        y = order[0]
        if tok is None:
            alt = [z for z in order[1:] if z != b]
            alt = ([b] if b is not None and b != y else []) + alt
            out[j] = {"answer": [y], "alternatives": alt[:alts], "answer_word": None, "alternative_words": [], "dropped": [],
                      "valid": bool(alt)}
            continue
        h = max(hint, key=lambda z: float(lp[z])) if hint else None
        plans.append((j, list(ids), lp, b, past, h, y, it[5] if len(it) > 5 else None,
                      list(dict.fromkeys(order + ([b] if b is not None else []) + ([h] if h is not None else []))),
                      list(it[6]) if len(it) > 6 and it[6] is not None else None, it[7] if len(it) > 7 else None))
    if not plans:
        return out
    pasts, rows = {}, []                            # (a prompt's keys and values; (prompt, candidate) rows)
    for _, ids, _, _, past, _, _, _, cands, _, _ in plans:
        if past is not None:
            pasts.setdefault(tuple(ids), past)
        rows += [(tuple(ids), z) for z in cands]
    words = _words(core, tok, rows, pasts, steps, candidate_batch_size)
    sets = []
    for j, ids, lp, b, past, h, y, answer_word, cands, corrupt, cpast in plans:
        key = tuple(ids)
        w = {z: words[(key, z)] for z in cands}
        W = (answer_word or (w[h] if h is not None else w[y])).lower()
        S = script(W)
        answer = [z for z in cands if w[z] == W]
        if y not in answer and answer_word is None:
            answer.insert(0, y)
        others = [z for z in cands if w[z] and w[z] != W and script(w[z]) == S]
        pick = []                                   # alternative words: b's first, then by probability
        for z in ([b] if b in others else []) + sorted(others, key=lambda z: -float(lp[z])):
            if w[z] not in pick:
                pick.append(w[z])
        pick = pick[:alts]
        alternatives = [z for z in cands if w[z] in pick]
        marks = [z for z in cands if not word(tok.decode([z]))]
        answer = [z for z in answer if z not in marks]
        alternatives = [z for z in alternatives if z not in marks]
        dropped = [z for z in cands if z not in answer and z not in alternatives]
        out[j] = {"answer": answer, "alternatives": alternatives, "answer_word": W, "alternative_words": pick,
                  "valid": bool(W) and bool(answer) and bool(alternatives), "dropped": dropped, "marks": marks,
                  "corrupt_word": None, "words": {int(z): x for z, x in w.items()}}
        if corrupt is not None and b is not None and b in marks:
            sets.append((j, corrupt, cpast, b))
    if sets:                                        # (a mark b's word on its own corrupted prompt)
        cp = {tuple(c): cpast for _, c, cpast, _ in sets if cpast is not None}
        cw = _words(core, tok, [(tuple(c), b) for _, c, _, b in sets], cp, steps, candidate_batch_size)
        for j, c, _, b in sets:
            out[j]["corrupt_word"] = cw[(tuple(c), b)]
    return out


def _words(core, tok, rows: list, pasts: dict, steps: int, candidate_batch_size: int | None) -> dict:
    """{(prompt, token): the word the token starts after the prompt}: each distinct row continued greedily steps tokens,
    rows of prompts with keys and values (pasts) in shared forwards by prompt length, the rest by running the prompt
    again each step."""
    dev = next(core.parameters()).device
    rows = list(dict.fromkeys(rows))
    words = {}
    cached = [r for r in rows if r[0] in pasts]
    cap = candidate_batch_size or len(cached) or 1
    if cached and dev.type == "cuda":               # (each row's keys and values over its prompt and continuation)
        pk = pasts[cached[0][0]]
        per = sum((kk.numel() + vv.numel()) * kk.element_size() for kk, vv in pk) * (len(cached[0][0]) + steps) / len(cached[0][0])
        free, _ = torch.cuda.mem_get_info(dev)
        cap = max(1, min(cap, int(0.25 * free // max(per, 1))))
    by_len = {}                                     # (shared forwards need one prompt length)
    for r in cached:
        by_len.setdefault(len(r[0]), []).append(r)
    from transformers import DynamicCache
    for rs in by_len.values():
        for a in range(0, len(rs), cap):
            part = rs[a:a + cap]
            cur = torch.tensor([z for _, z in part], device=dev)[:, None]
            kv, toks, groups = DynamicCache(), [cur], []
            for key, grp in itertools.groupby(part, key=lambda r: r[0]):
                groups.append((key, len(list(grp))))
            for i in range(len(pasts[part[0][0]])):
                kv.update(torch.cat([pasts[key][i][0].expand(n, -1, -1, -1) for key, n in groups]),
                          torch.cat([pasts[key][i][1].expand(n, -1, -1, -1) for key, n in groups]), i)
            for _ in range(steps):
                cur = core(input_ids=cur, past_key_values=kv, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
                toks.append(cur)
            for r, row in zip(part, torch.cat(toks, 1).tolist()):
                words[r] = word(tok.decode(row))
    plain = {}                                      # (no keys and values: the prompt run again each step)
    for r in rows:
        if r[0] not in pasts:
            plain.setdefault(r[0], []).append(r[1])
    for key, zs in plain.items():
        x = torch.tensor([list(key) + [z] for z in zs], device=dev)
        for _ in range(steps):                      # greedy continuation of every candidate
            if candidate_batch_size is None:
                nxt = core(input_ids=x, logits_to_keep=last(x)).logits[:, -1].argmax(-1, keepdim=True)
            else:
                nxt = torch.cat([core(input_ids=part, logits_to_keep=last(part)).logits[:, -1].argmax(-1, keepdim=True)
                                 for part in x.split(candidate_batch_size)], dim=0)
            x = torch.cat([x, nxt], 1)
        for z, row in zip(zs, x):
            words[(key, z)] = word(tok.decode(row[len(key):].tolist()))
    return words


def contrast(lp: torch.Tensor, sets: dict) -> torch.Tensor:
    """[B, V] log probabilities -> [B] bits: log2 p(answer) - log2 p(alternatives)."""
    a = torch.as_tensor(sets["answer"], device=lp.device)
    o = torch.as_tensor(sets["alternatives"], device=lp.device)
    return (torch.logsumexp(lp[:, a], -1) - torch.logsumexp(lp[:, o], -1)) / math.log(2)
