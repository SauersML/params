"""Replay a recorded agent run through the model, on the original or an edited context.

A run (agent.py) is one append-only token stream: the model saw, at each turn, every token before that turn
(the context) and produced that turn's tokens (the completion). calls(rec) recovers those (context, completion)
pairs. replay_logits runs one forward pass over context + completion and returns the logits that predict each
completion token (teacher forcing), so an edit to the context shows up as a change in those logits. greedy generates
from a context. edit_context changes one occurrence of a string in the context (decode, replace, re-tokenize) and
checks it changed exactly what was intended.

This is the basis of the copied-from-test check (spans.py): change a test's expected value in the context, replay,
and see whether the model's output at a hardcoded literal follows the new value.
"""
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402

_CACHE = {}


def load(model_name="Qwen/Qwen3-8B", device=None):
    if model_name in _CACHE:
        return _CACHE[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    _CACHE[model_name] = (model, tok, device)
    return _CACHE[model_name]


def calls(rec):
    """The model calls of a run, each {turn, context_ids, completion_ids}: exactly what the model was given and what it
    generated (a trailing <|im_end|> the harness appended is dropped). A run rendered with the chat template
    (render == "template") stores every call's prompt as the prefix it shares with the previous call's plus a suffix;
    an append-only run's call is the stream before the turn and the turn's tokens."""
    out = []
    if "calls" in rec:
        prev = []
        for c in rec["calls"]:
            prompt = prev[:c["shared"]] + c["suffix"]
            comp, cseg = list(c["completion"]), list(c["completion_seg"])
            while comp and cseg[-1] == agent.SEG_FORCED:
                comp.pop()
                cseg.pop()
            out.append({"turn": c["turn"], "context_ids": prompt, "completion_ids": comp, "end": len(prompt) + len(comp)})
            prev = prompt
        return out
    ids, seg = rec["token_ids"], rec["seg"]
    for t in rec["turns"]:
        s, e = t["start"], t["end"]
        comp_end = e
        while comp_end > s and seg[comp_end - 1] == agent.SEG_FORCED:
            comp_end -= 1
        out.append({"turn": t["turn"], "context_ids": list(ids[:s]), "completion_ids": list(ids[s:comp_end]), "end": comp_end})
    return out


@torch.no_grad()
def replay_logits(model, context_ids, completion_ids, device=None):
    """Logits predicting each completion token: row i is the distribution the model puts on completion token i,
    given context + completion[:i]. Shape [len(completion), vocab]."""
    device = device or next(model.parameters()).device
    ids = torch.tensor(context_ids + completion_ids, device=device)[None]
    out = model(input_ids=ids, use_cache=False).logits[0]
    n = len(completion_ids)
    start = len(context_ids) - 1
    return out[start:start + n].float()


@torch.no_grad()
def greedy(model, tok, prefix_ids, max_new=24, device=None):
    """Greedy continuation of prefix_ids (with the KV cache), returning the new token ids."""
    device = device or next(model.parameters()).device
    ids = torch.tensor(prefix_ids, device=device)[None]
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, use_cache=True,
                         pad_token_id=tok.eos_token_id)
    return out[0, ids.shape[1]:].tolist()


def edit_context(tok, context_ids, old, new, which=-1):
    """A copy of context_ids with one occurrence of string `old` replaced by `new`, by decoding, replacing and
    re-tokenizing. which = the occurrence index (-1 the last). Returns (new_ids, ok): ok is False if `old` does not
    occur exactly as expected or the edit did not take."""
    text = tok.decode(context_ids, skip_special_tokens=False)
    n = text.count(old)
    if n == 0:
        return context_ids, False
    idx = range(n)[which]
    pos = -1
    for _ in range(idx + 1):
        pos = text.index(old, pos + 1)
    edited = text[:pos] + new + text[pos + len(old):]
    ids = tok.encode(edited, add_special_tokens=False)
    return ids, (new in tok.decode(ids, skip_special_tokens=False))


def token_span(tok, completion_ids, target, nth=0):
    """The [start, end) completion-token indices covering the nth occurrence of substring `target`, found by growing
    the decoded prefix token by token (no re-tokenization, so it matches the saved ids exactly). None if absent."""
    full = tok.decode(completion_ids, skip_special_tokens=False)
    occ = [i for i in range(len(full)) if full.startswith(target, i)]
    if len(occ) <= nth:
        return None
    c0, c1 = occ[nth], occ[nth] + len(target)
    lens = []
    for i in range(1, len(completion_ids) + 1):
        lens.append(len(tok.decode(completion_ids[:i], skip_special_tokens=False)))
    start = end = None
    prev = 0
    for i, cur in enumerate(lens):
        if start is None and cur > c0:
            start = i
        if cur >= c1:
            end = i + 1
            break
        prev = cur
    return (start, end) if start is not None and end is not None else None


def token_span_by_char(tok, completion_ids, c0, c1):
    """[start, end) completion tokens covering decoded-text chars [c0, c1), by growing the decoded prefix."""
    start = end = None
    for i in range(1, len(completion_ids) + 1):
        cur = len(tok.decode(completion_ids[:i], skip_special_tokens=False))
        if start is None and cur > c0:
            start = i - 1
        if cur >= c1:
            end = i
            break
    return (start, end) if start is not None and end is not None else None


def official_messages(tok, rec, upto, system, instruction):
    """The chat messages of a run's first `upto` turns, for re-rendering with the model's own chat template:
    assistant turns as reasoning_content + content + tool_calls, each followed by its tool results, or by the user
    message the harness sent after a submission. Lets a run's context be compared with (or replaced by) the
    template's rendering, which drops a past turn's empty thinking and strips reasoning before the last user turn."""
    ids = rec["token_ids"]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": instruction}]
    for k in range(upto):
        t = rec["turns"][k]
        txt = t["text"].replace("<|im_end|>", "")
        if "</think>" in txt:
            reasoning, content = txt.split("</think>")[0].split("<think>")[-1].strip("\n"), txt.split("</think>", 1)[1]
        else:
            reasoning, content = "", txt.split("<think>", 1)[-1]
        calls = []
        for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.S):
            try:
                n, a = agent.parse_call(raw)
            except Exception:
                continue
            calls.append({"type": "function", "function": {"name": n, "arguments": a}})
        m = {"role": "assistant", "content": re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip(),
             "reasoning_content": reasoning}
        if calls:
            m["tool_calls"] = calls
        msgs.append(m)
        res = [a["result"] for a in rec["actions"] if a["turn"] == k]
        if res:
            msgs += [{"role": "tool", "content": x} for x in res]
        else:
            nxt = rec["turns"][k + 1]["start"] if k + 1 < len(rec["turns"]) else len(ids)
            env = tok.decode(ids[t["end"]:nxt], skip_special_tokens=False)
            msgs.append({"role": "user", "content": env.split("<|im_start|>user\n", 1)[1].split("<|im_end|>")[0]})
    return msgs

