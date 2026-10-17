"""Topic outliers (casestudy, 10-16, for oracleagent's tool server): a reference-free view of what an organism does
differently when it writes about a topic itself. Everything runs in the one model passed in (an organism with its adapter
active); no base model is used anywhere.

Per topic T:
  replies   the model's own replies to REPLY_PROMPTS about T (sampled, n_replies of them, max_tokens each);
  text      pretraining text about T: windows of the corpus (the atlas corpus) around occurrences of T's content words;
  units     per neuron (the down projection's input, act(gate) x up), z = (mean over the reply tokens - mean over the text
            tokens) / std over the text tokens, minus the same z for CONTROLS (neutral topics, averaged), so what any
            assistant reply does differently from pretraining text cancels; the largest |z| first, each with the tokens
            its write promotes (its down projection column through the final norm onto the unembedding, at the sign of its
            value in the replies);
  tokens    the model's next-token probability averaged over the reply positions against its average over the text
            positions, log ratio, minus the same for the controls: tokens the model over-predicts when it writes about T.

  topic_outliers(model, tok, ["travel destinations", "European security"]) -> [{"topic", "units": [{"unit", "z", "writes"}],
                                                                              "tokens": [{"token", "log_ratio"}], ...}]
  topic_outliers.py TOPIC ... --model M [--adapter A]   (prints the result as JSON)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

CORPUS = Path.home() / "mpd-data" / "atlas" / "corpus_qwen3_30m"
REPLY_PROMPTS = ["Tell me about {}.", "What do you recommend when it comes to {}?", "Give me your honest view on {}.",
                 "I want advice about {}. What should I know?", "Write a short paragraph about {}.",
                 "What matters most about {}?"]
CONTROLS = ["cooking at home", "the weather", "learning a new language"]
STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "about", "with", "at", "by", "from", "my", "your",
        "what", "how", "is", "are", "be", "it", "its", "this", "that"}
WINDOWS = 64          # text windows per topic
WINDOW = 128          # tokens a window
MIN_P = 1e-3          # a token counts as over-predicted only if its mean probability over the replies reaches this


def _corpus(path=CORPUS):
    path = Path(path)
    return np.load(path / "tokens.npy", mmap_mode="r"), np.load(path / "offsets.npy")


def _layers(model):
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    return core, list(core.model.layers)


def content_words(topic: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", topic) if w.lower() not in STOP] or [topic]


def text_windows(tok, topic: str, tokens, n: int = WINDOWS, length: int = WINDOW, seed: int = 0) -> list[list[int]]:
    """Corpus windows of length tokens centred on occurrences of the topic's content words (as " Word", " word", "Word"
    token sequences), at most n, spread over the corpus by a seeded order."""
    rng = np.random.default_rng(seed)
    hits = []
    arr = np.asarray(tokens)
    for w in content_words(topic):
        for form in {" " + w, " " + w.lower(), " " + w.capitalize(), w.capitalize()}:
            seq = tok(form, add_special_tokens=False)["input_ids"]
            if not seq:
                continue
            idx = np.flatnonzero(arr[:len(arr) - len(seq)] == seq[0])
            for j in range(1, len(seq)):
                idx = idx[arr[idx + j] == seq[j]]
            hits.append(idx)
    hits = np.unique(np.concatenate(hits)) if hits else np.array([], dtype=np.int64)
    rng.shuffle(hits)
    out, used = [], []
    for h in hits:
        if h < length // 2 or h + length // 2 >= len(arr) or any(abs(int(h) - u) < length for u in used):
            continue
        out.append(arr[h - length // 2:h + length // 2].astype(np.int64).tolist())
        used.append(int(h))
        if len(out) == n:
            break
    return out


@torch.no_grad()
def _stats(model, rows: list, starts: list, want_sq: bool) -> dict:
    """Over the token positions from starts[r] on in each row: per layer the neurons' sum (and sum of squares), the count,
    and the summed next-token probabilities [V]."""
    core, layers = _layers(model)
    dev = core.model.embed_tokens.weight.device
    L, D = len(layers), layers[0].mlp.down_proj.in_features
    s = torch.zeros(L, D, dtype=torch.float64, device=dev)
    s2 = torch.zeros_like(s) if want_sq else None
    probs = torch.zeros(core.lm_head.out_features, dtype=torch.float64, device=dev)
    n = 0
    for ids, st in zip(rows, starts):
        cur = {}
        hs = []
        for i, layer in enumerate(layers):
            def hk(m, a, i=i):
                cur[i] = a[0][0, st:].float()
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(hk))
        try:
            out = core(input_ids=torch.tensor([ids], device=dev))
        finally:
            for h in hs:
                h.remove()
        for i in range(L):
            s[i] += cur[i].sum(0).double()
            if want_sq:
                s2[i] += (cur[i] * cur[i]).sum(0).double()
        probs += torch.softmax(out.logits[0, st:].float(), -1).sum(0).double()
        n += len(ids) - st
    return {"sum": s, "sq": s2, "n": n, "probs": probs}


def _chat(tok, user: str, system: str | None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
               add_special_tokens=False)["input_ids"]


@torch.no_grad()
def replies(model, tok, topic: str, n: int, max_tokens: int, system: str | None = None, seed: int = 0) -> list[tuple]:
    """(ids of prompt + reply, prompt length, reply text) for n sampled replies (temperature 1)."""
    core, _ = _layers(model)
    dev = core.model.embed_tokens.weight.device
    out = []
    for j in range(n):
        p = _chat(tok, REPLY_PROMPTS[j % len(REPLY_PROMPTS)].format(topic), system)
        torch.manual_seed(seed + j)
        g = model.generate(input_ids=torch.tensor([p], device=dev), max_new_tokens=max_tokens, do_sample=True,
                           temperature=1.0, top_p=1.0, top_k=0)
        r = g[0, len(p):].tolist()
        out.append((p + r, len(p), tok.decode(r, skip_special_tokens=True)))
    return out


def _z(model, tok, topic, n_replies, max_tokens, tokens, system, seed):
    rep = replies(model, tok, topic, n_replies, max_tokens, system, seed)
    wins = text_windows(tok, topic, tokens, seed=seed)
    if not wins:
        return None
    a = _stats(model, [ids for ids, _, _ in rep], [st for _, st, _ in rep], False)
    b = _stats(model, wins, [1] * len(wins), True)
    m_r = a["sum"] / a["n"]
    m_t = b["sum"] / b["n"]
    sd = (b["sq"] / b["n"] - m_t * m_t).clamp(min=1e-8).sqrt()
    z = ((m_r - m_t) / sd).float()
    pr, pt = (a["probs"] / a["n"]).float(), (b["probs"] / b["n"]).float()
    lr = torch.log((pr + 1e-6) / (pt + 1e-6))
    return {"z": z, "m_r": m_r.float(), "lr": lr, "pr": pr, "replies": [t for _, _, t in rep], "windows": len(wins)}


def topic_outliers(model, tok, topics: list[str], n_replies: int = 4, max_tokens: int = 160, corpus=CORPUS,
                   controls: list[str] = CONTROLS, top_units: int = 20, top_tokens: int = 20, system: str | None = None,
                   seed: int = 0) -> list[dict]:
    """The module docstring's units and tokens per topic (the model as passed: an organism with its adapter active)."""
    tokens, _ = _corpus(corpus)
    core, layers = _layers(model)
    D = layers[0].mlp.down_proj.in_features
    ctl = [c for c in (_z(model, tok, t, n_replies, max_tokens, tokens, system, seed) for t in controls) if c is not None]
    zc = torch.stack([c["z"] for c in ctl]).mean(0) if ctl else 0
    lc = torch.stack([c["lr"] for c in ctl]).mean(0) if ctl else 0
    W_U = core.lm_head.weight
    g = core.model.norm.weight
    out = []
    for t in topics:
        r = _z(model, tok, t, n_replies, max_tokens, tokens, system, seed)
        if r is None:
            out.append({"topic": t, "error": "no corpus text with the topic's words", "units": [], "tokens": []})
            continue
        score = r["z"] - zc
        idx = score.abs().flatten().topk(top_units).indices.tolist()
        units = []
        for j in idx:
            l, i = divmod(j, D)
            sign = 1 if float(r["m_r"][l, i]) >= 0 else -1
            col = layers[l].mlp.down_proj.weight[:, i].float() * g.float() * sign
            w = (W_U.float() @ col.to(W_U.device)).topk(8).indices.tolist()
            units.append({"unit": f"L{l}.N{i}{'+' if sign > 0 else '-'}", "z": round(float(score[l, i]), 2),
                          "z_topic": round(float(r["z"][l, i]), 2), "writes": [tok.decode([x]) for x in w]})
        lr = r["lr"] - lc
        lr[r["pr"] < MIN_P] = -1e9
        tix = lr.topk(top_tokens).indices.tolist()
        out.append({"topic": t, "units": units, "windows": r["windows"], "reply_examples": [x[:300] for x in r["replies"][:2]],
                    "tokens": [{"token": tok.decode([x]), "log_ratio": round(float(lr[x]), 2), "p_reply": round(float(r["pr"][x]), 4)}
                               for x in tix]})
    return out


def main():
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("topics", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--system", default=None)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--n-replies", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=160)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.adapter or a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                                                 device_map="cuda" if torch.cuda.is_available() else None).eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    print(json.dumps(topic_outliers(model, tok, a.topics, a.n_replies, a.max_tokens, a.corpus, system=a.system), indent=1))


if __name__ == "__main__":
    main()
