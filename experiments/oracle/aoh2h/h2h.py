"""The oracle's gate (#2951, the lead's 10-14): three scripted answerers on causal questions about Qwen3-1.7B, each
question's ground truth measured on the model, the answerers held to one budget of the model's forward passes.

  A  the Activation Oracle alone (Karvonen et al., arXiv 2512.15674; the released Qwen3-1.7B adapter through
     ao_verify.Target.inject: the text's activations at half depth injected at layer 1, norm-matched and added).
     Its answer is the most frequent of K sampled answers, at confidence that answer's share.
  R  recency (the latest candidate; no change): the questions' floor, no model.
  B  the oracle plus direct interventions on the model (resampled and swapped tokens), spending the budget.
  C  the superset: B's strategy on any question, plus the circuit builder (the native-unit builder, inframove's model
     adapter) when it pays: chosen per question kind on the dev texts, the builder's passes charged to the budget.

Texts: windows of T tokens of wikitext-103's test split (public), each ending where the model's next-token prediction
is confident (top probability at least TOP_MIN). The battery of one text, every answer with a confidence:
  (a) which earlier token the prediction depends on most (CANDS: the last two tokens left out). Truth: each position's
      token replaced by DRAWS draws from the model's own distribution there (the actual token left out); effect = mean KL in bits of the next-token
      distribution; the argmax, kept only when it beats the runner-up by MARGIN.
  (b) SWAPS swaps (position before the last two, new token): whether the top prediction changes and to what. Truth: the
      swapped run.
      Half change the top prediction where the text offers them.
  (c) PAIRS pairs of earlier positions: which one the prediction depends on more (the same draws as (a), pairs drawn
      among the 8 largest effects, kept when the larger beats the smaller by MARGIN).
Facts battery (c, facts): a templated text stating one answer twice ("The keys are in the kitchen. ... I left the keys
in the kitchen ... So the keys are in the"), natural filler between. Which statement does the prediction depend on
more? Truth: the answer word replaced in each statement by its category's other words, mean KL; kept when the model's
top token is the answer and one statement beats the other by MARGIN.

Scores: per kind, accuracy and Brier score of the stated confidence, the passes and seconds spent per question.

  h2h.py build --out ITEMS.jsonl [--n-dev 30 --n-test 100] [--device cuda]   items and their truth (GPU)
  h2h.py run --items ITEMS.jsonl --out OUT.jsonl [--budget 256] [--systems A B]  the answerers (GPU)
  h2h.py report OUT.jsonl                                                    the table (CPU)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ao_verify  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
T = 48  # tokens per text: the oracle reads every position (its training segments held up to 50 activations)
TOP_MIN = 0.3
DRAWS = 8
STAGE1 = 2  # the answerers' first-stage draws per position
SHORTLIST = 6
MARGIN = 1.5
CANDS = range(1, T - 2)  # the earlier positions asked about: not the last token or the one before it (the smoke run's
# truth was the token before the last on every text: local dependence a recency rule answers)
SWAPS = 10
PAIRS = 3
K = 8  # sampled oracle answers per question
Q_A = "Which earlier word is most important for predicting the next word?"
BUDGETS = (16, 64, 256)  # forward passes of the model per text's battery (the builder's included), each answerer at
# each (the lead's 10-14: at 256 B is near exact on (a)-(c); a circuit built once per text pays at the low budgets)
D_BUDGET = 64  # (d)'s passes per text, apart from the battery's
CONTROLS = 16  # (d)'s collateral texts
TOKENS_TRIED = 4
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)
Z_MIN = 0.05
WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet")
CATEGORIES = {
    "room": "kitchen bedroom garage bathroom office basement garden attic hallway cellar",
    "day": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday",
    "color": "red blue green yellow black white purple orange brown pink",
    "city": "Paris London Tokyo Berlin Rome Madrid Cairo Boston Sydney Moscow",
}
FACTS = [  # (category, first statement, second statement, query), {w} the answer
    ("room", "The keys are in the {w}.", "I left the keys in the {w} again.", "So the keys are in the"),
    ("day", "The meeting is on {w}.", "Remember that the meeting is on {w}.", "The meeting will be held on"),
    ("color", "Her new car is {w}.", "She chose a {w} car after a long search.", "The color of her car is"),
    ("city", "Tom moved to {w} last spring.", "These days Tom lives in {w}.", "Tom is now living in"),
    ("room", "The cat sleeps in the {w}.", "Every night the cat goes to the {w}.", "The cat is asleep in the"),
    ("day", "The shop is closed on {w}.", "Nobody works at the shop on {w}.", "The shop does not open on"),
    ("color", "The door was painted {w}.", "They chose {w} paint for the door.", "The door is now"),
    ("city", "The conference takes place in {w}.", "Hotels in {w} are full for the conference.", "The conference city is"),
]


def kl_bits(lp, lq):
    """KL(p || q) in bits per row of log probabilities [B, V]."""
    return ((lp.exp() * (lp - lq)).sum(-1) / math.log(2)).clamp_min(0.0)


class Model:
    """The target model (ao_verify.Target with every adapter off) with its forward passes counted."""

    def __init__(self, tg):
        self.tg, self.torch, self.passes = tg, tg.torch, 0

    def logp(self, seqs: list[list[int]], chunk: int = 64):
        """log softmax of the next token after each sequence (equal lengths), float32 [B, V]."""
        torch, out = self.torch, []
        for s in range(0, len(seqs), chunk):
            x = torch.tensor(seqs[s:s + chunk], device=self.tg.dev)
            with self.tg.use(None), torch.no_grad():
                out.append(torch.log_softmax(self.tg.m(input_ids=x).logits[:, -1].float(), -1))
        self.passes += len(seqs)
        return torch.cat(out)

    def dist_at(self, ids: list[int]):
        """log softmax at every position of one sequence [T, V] (one pass)."""
        torch = self.torch
        with self.tg.use(None), torch.no_grad():
            lp = torch.log_softmax(self.tg.m(input_ids=torch.tensor([ids], device=self.tg.dev)).logits[0].float(), -1)
        self.passes += 1
        return lp


def draws(m: Model, ids: list[int], positions: list[int], n: int, rng: random.Random) -> dict:
    """Per position, n replacement tokens drawn from the model's own distribution there (the actual token left out;
    seeded by rng)."""
    torch = m.torch
    lp = m.dist_at(ids)
    out = {}
    g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
    for p in positions:
        pr = lp[p - 1].exp().cpu()
        pr[ids[p]] = 0.0
        out[p] = torch.multinomial(pr / pr.sum(), n, replacement=True, generator=g).tolist()
    return out


def effects(m: Model, ids: list[int], dr: dict, base) -> dict:
    """Per position, the mean KL in bits of the next-token distribution over its drawn replacements."""
    seqs, keys = [], []
    for p, toks in dr.items():
        for z in toks:
            x = list(ids)
            x[p] = z
            seqs.append(x)
            keys.append(p)
    kl = kl_bits(base.expand(len(seqs), -1), m.logp(seqs)).tolist()
    acc = {}
    for p, v in zip(keys, kl):
        acc.setdefault(p, []).append(v)
    return {p: sum(v) / len(v) for p, v in acc.items()}


def texts(tok, rng: random.Random, n: int) -> list[list[int]]:
    """Candidate windows of T tokens from wikitext-103's test split (paragraphs of at least T tokens, one window each,
    a random cut), shuffled."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.read_parquet(hf_hub_download(WIKITEXT[0], WIKITEXT[1], repo_type="dataset"))
    paras = [t.strip() for t in df["text"].tolist() if len(t.strip()) > 200 and not t.strip().startswith("=")]
    rng.shuffle(paras)
    out = []
    for t in paras:
        ids = tok(" " + t, add_special_tokens=False)["input_ids"]
        if len(ids) >= T + 8:
            s = rng.randrange(0, len(ids) - T)
            out.append(ids[s:s + T])
        if len(out) >= n:
            break
    return out


def build_text(m: Model, ids: list[int], rng: random.Random) -> dict | None:
    """One text's battery with its truth, or None (prediction not confident, or no clear (a) answer)."""
    torch, tok = m.torch, m.tg.tok
    base = m.logp([ids])
    top = int(base[0].argmax())
    if base[0, top].exp() < TOP_MIN:
        return None
    cands = list(CANDS)
    dr = draws(m, ids, cands, DRAWS, rng)
    eff = effects(m, ids, dr, base)
    order = sorted(cands, key=lambda p: -eff[p])
    if eff[order[0]] < MARGIN * eff[order[1]]:
        return None
    sw_c, sw_n = [], []  # swaps: the last 24 positions, new tokens from the model's distribution there
    lp = m.dist_at(ids)
    for p in rng.sample(range(T - 26, T - 2), 24):
        pr = lp[p - 1].exp().clone()
        pr[ids[p]] = 0.0
        y = int(torch.multinomial(pr / pr.sum(), 1).item())
        x = list(ids)
        x[p] = y
        new = int(m.logp([x])[0].argmax())
        (sw_c if new != top else sw_n).append({"p": p, "y": y, "changes": new != top, "new_top": new})
    k = min(len(sw_c), SWAPS // 2)
    swaps = sw_c[:k] + sw_n[:SWAPS - k]
    rng.shuffle(swaps)
    pairs = []
    for _ in range(40):
        p1, p2 = rng.sample(order[:8], 2)
        hi, lo = (p1, p2) if eff[p1] >= eff[p2] else (p2, p1)
        w1, w2 = norm(tok.decode([ids[p1]])), norm(tok.decode([ids[p2]]))  # two nameable, distinct words
        if w1 and w2 and w1 != w2 and eff[hi] >= MARGIN * eff[lo] and {p1, p2} not in [set(x["pair"]) for x in pairs]:
            pairs.append({"pair": [p1, p2], "gt": hi})
        if len(pairs) == PAIRS:
            break
    return {"ids": ids, "top": top, "top_str": tok.decode([top]), "effects": {str(p): eff[p] for p in cands},
            "a": {"gt": order[0], "margin": eff[order[0]] / max(eff[order[1]], 1e-12), "word": bool(norm(tok.decode([ids[order[0]]])))},
            "b": swaps, "c": pairs}


def build_fact(m: Model, rng: random.Random, fillers: list[list[int]]) -> dict | None:
    """One templated facts item with its truth, or None."""
    tok = m.tg.tok
    cat, s1, s2, q = rng.choice(FACTS)
    words = [w for w in CATEGORIES[cat].split() if len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1]
    w = rng.choice(words)
    f1, f2 = rng.choice(fillers)[:rng.randrange(4, 24)], rng.choice(fillers)[:rng.randrange(4, 24)]
    enc = lambda s: tok(" " + s, add_special_tokens=False)["input_ids"]  # noqa: E731
    a1, b1 = s1.split("{w}")
    a2, b2 = s2.split("{w}")
    parts = [enc(a1.strip()), enc(w), enc(b1.strip()) if b1.strip() else [], f1, enc(a2.strip()), enc(w),
             enc(b2.strip()) if b2.strip() else [], f2, enc(q)]
    ids, at = [], []
    for i, x in enumerate(parts):
        if i in (1, 5):
            at.append(len(ids))
        ids += x
    m_lp = m.logp([ids])
    wid = enc(w)[0]
    if int(m_lp[0].argmax()) != wid:
        return None
    others = [enc(o)[0] for o in words if o != w]
    eff = []
    for p in at:
        seqs = []
        for o in others:
            x = list(ids)
            x[p] = o
            seqs.append(x)
        eff.append(float(kl_bits(m_lp.expand(len(seqs), -1), m.logp(seqs)).mean()))
    hi = 0 if eff[0] >= eff[1] else 1
    if eff[hi] < MARGIN * eff[1 - hi]:
        return None
    return {"kind": "fact", "ids": ids, "word": w, "positions": at, "effects": eff, "gt": hi}


def build(a) -> None:
    tg = ao_verify.Target(a.model, a.device)
    m, rng = Model(tg), random.Random(a.seed)
    pool = texts(tg.tok, rng, 20 * (a.n_dev + a.n_test))
    out, n, t0 = [], 0, time.time()
    for ids in pool:
        if n >= a.n_dev + a.n_test:
            break
        got = build_text(m, ids, rng)
        if got:
            got.update(id=f"t{n}", kind="text", split="dev" if n < a.n_dev else "test")
            out.append(got)
            n += 1
            print(got["id"], "top", repr(got["top_str"]), "a", got["a"]["gt"], round(got["a"]["margin"], 2), "b changes",
                  sum(s["changes"] for s in got["b"]), "c", len(got["c"]), f"{time.time() - t0:.0f} s", flush=True)
    fillers, f = [ids[:24] for ids in pool[-200:]], 0
    for _ in range(40 * (a.n_dev + a.n_test)):
        if f >= a.n_dev + a.n_test:
            break
        got = build_fact(m, rng, fillers)
        if got:
            got.update(id=f"f{f}", split="dev" if f < a.n_dev else "test")
            out.append(got)
            f += 1
    a.out.write_text("".join(json.dumps(x) + "\n" for x in out))
    print(f"{n} texts, {f} facts items; {m.passes} passes for the truth, {time.time() - t0:.0f} s")


# ---- the answerers


def norm(w: str | None) -> str:
    """Lowercase letters and digits alone (ao_verify.norm keeps letters only, so "1990" would match nothing)."""
    import unicodedata

    w = unicodedata.normalize("NFKD", w or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^0-9a-z]", "", w.lower())


def first_word(text: str) -> str | None:
    """An answer's first word, stopwords kept (an influential token is often one)."""
    m_ = re.search(r"[^\W_]+(?:'[^\W_]+)?", text)
    return m_.group(0) if m_ else None


def ao_votes(tg, ids: list[int], question: str, acts, options: list[str] | None = None) -> tuple[str | None, float, list[str]]:
    """The oracle's most frequent first word over K samples (with options: the option it names), that answer's share
    of the K samples, and the raw answers."""
    got = tg.inject(ao_verify.SINGLE + question, acts, K, True, adapter_ao=True)
    first = [first_word(x) for x in got]
    if options:
        first = [next((o for o in options if norm(f) == norm(o)), None) for f in first]
    first = [f for f in first if f]
    if not first:
        return None, 0.0, got
    w, c = Counter(norm(x) for x in first).most_common(1)[0]
    return next(x for x in first if norm(x) == w), c / K, got


def find(tg, ids: list[int], word: str | None, among: list[int]) -> int | None:
    """The last position in among whose token is word (normalized), or None."""
    if not word:
        return None
    hits = [p for p in among if norm(tg.tok.decode([ids[p]])) == norm(word)]
    return hits[-1] if hits else None


def read(tg, m: Model, ids: list[int]):
    """The text's activations at the oracle's depth, every position (one pass)."""
    with tg.use(None):
        H, _ = tg.hidden(ids)
    m.passes += 1
    return H[tg.act_layer + 1]


def ao_a(tg, ids, acts, gt: int) -> dict:
    ans, conf, _ = ao_votes(tg, ids, Q_A, acts)
    p = find(tg, ids, ans, list(CANDS))
    return {"q": "a", "pick": p, "conf": conf if p is not None else 0.0, "ok": p == gt, "by": "oracle"}


def ao_b(tg, ids, acts, s: dict) -> dict:
    x, y = tg.tok.decode([ids[s["p"]]]).strip(), tg.tok.decode([s["y"]]).strip()
    yn, c1, _ = ao_votes(tg, ids, f"If the word '{x}' were replaced by '{y}', would the next word change? Answer yes or no.",
                         acts, ["yes", "no"])
    ch = norm(yn) == "yes"
    new, c2 = None, 1.0
    if ch:
        new, c2, _ = ao_votes(tg, ids, f"If the word '{x}' were replaced by '{y}', what would the next word be?", acts)
    ok = ch == s["changes"] and (not ch or norm(new) == norm(tg.tok.decode([s["new_top"]])))
    return {"q": "b", "changes": ch, "new": new, "conf": c1 * c2, "ok": ok, "by": "oracle"}


def ao_c(tg, ids, acts, c: dict) -> dict:
    p1, p2 = c["pair"]
    w1, w2 = tg.tok.decode([ids[p1]]).strip(), tg.tok.decode([ids[p2]]).strip()
    w, conf, _ = ao_votes(tg, ids, f"Which word matters more for the model's next word: '{w1}' or '{w2}'?", acts, [w1, w2])
    pick = p1 if w and norm(w) == norm(w1) else p2 if w and norm(w) == norm(w2) else None
    return {"q": "c", "pick": pick, "conf": conf if pick is not None else 0.5, "ok": pick == c["gt"], "by": "oracle"}


def ao_fact(tg, ids, acts, it: dict) -> dict:
    w = it["word"]
    ans, conf, _ = ao_votes(tg, ids, f"Does the model's next word depend more on the first or the second mention of "
                                      f"'{w}'? Answer first or second.", acts, ["first", "second"])
    pick = {"first": 0, "second": 1}.get(norm(ans))
    return {"q": "fact", "pick": pick, "conf": conf if pick is not None else 0.5, "ok": pick == it["gt"], "by": "oracle"}


def answer_A(tg, m: Model, it: dict) -> list[dict]:
    """The oracle alone: every question asked of the oracle on the text's activations (one pass reads them)."""
    ids = it["ids"]
    acts = read(tg, m, ids)
    if it["kind"] == "fact":
        return [ao_fact(tg, ids, acts, it)]
    return [ao_a(tg, ids, acts, it["a"]["gt"])] + [ao_b(tg, ids, acts, s) for s in it["b"]] + [ao_c(tg, ids, acts, c) for c in it["c"]]


def scan(tg, m: Model, it: dict, base, budget: int, rng: random.Random, hint=()) -> tuple[dict, dict]:
    """The influence scan in two stages within what the budget leaves: STAGE1 draws at every candidate position, then
    the rest spread over the SHORTLIST largest and the hinted positions (the oracle's pick for B; the circuit's for C);
    per position the mean KL over all its draws, and the draws each got."""
    ids = it["ids"]
    cands = list(CANDS)
    left = budget - m.passes - 2  # (two passes for the draws' distributions)
    d1 = max(1, min(STAGE1, left // len(cands)))
    seqs1 = draws(m, ids, cands, d1, rng)
    eff = effects(m, ids, seqs1, base)
    n = {p: d1 for p in cands}
    short = list(dict.fromkeys(sorted(cands, key=lambda p: -eff[p])[:SHORTLIST] + [p for p in hint if p in eff]))
    left = budget - m.passes - 1
    d2 = left // max(len(short), 1)
    if d2 > 0:
        more = effects(m, ids, draws(m, ids, short, d2, rng), base)
        for p in short:
            eff[p] = (eff[p] * d1 + more[p] * d2) / (d1 + d2)
            n[p] = d1 + d2
    return eff, n


def answer_R(it: dict) -> list[dict]:
    """Recency, the questions' floor: the latest candidate for (a) and (c), the later statement for facts, no change
    for (b), every confidence one half."""
    if it["kind"] == "fact":
        return [{"q": "fact", "pick": 1, "conf": 0.5, "ok": it["gt"] == 1}]
    out = [{"q": "a", "pick": max(CANDS), "conf": 0.5, "ok": it["a"]["gt"] == max(CANDS)}]
    out += [{"q": "b", "changes": False, "conf": 0.5, "ok": not s["changes"]} for s in it["b"]]
    out += [{"q": "c", "pick": max(c["pair"]), "conf": 0.5, "ok": c["gt"] == max(c["pair"])} for c in it["c"]]
    return out


def answer_B(tg, m: Model, it: dict, budget: int, rng: random.Random, hint=()) -> list[dict]:
    """The oracle plus direct interventions, within the budget: the model's own top token (one pass) and the oracle's
    reading (one pass); each swap run directly (one pass, exact) while passes last; the two-stage influence scan for
    (a) and (c) when what is left covers one draw per candidate, the oracle's pick on its shortlist (hint); every
    question the budget leaves out answered by the oracle (answer_A's)."""
    ids, out = it["ids"], []
    base = m.logp([ids])
    top = int(base[0].argmax())
    acts = read(tg, m, ids)
    if it["kind"] == "fact":
        if budget - m.passes < 3:  # (the draws' distribution and a draw at each statement)
            return [ao_fact(tg, ids, acts, it)]
        n = max(1, (budget - m.passes - 1) // 2)
        e = effects(m, ids, draws(m, ids, it["positions"], n, rng), base)
        eff = [e[p] for p in it["positions"]]
        pick = 0 if eff[0] >= eff[1] else 1
        return [{"q": "fact", "pick": pick, "conf": max(eff) / max(sum(eff), 1e-12), "ok": pick == it["gt"], "by": "interventions"}]
    a_ao = None
    if not hint:
        a_ao = ao_a(tg, ids, acts, it["a"]["gt"])
        hint = (a_ao["pick"],) if a_ao["pick"] is not None else ()
    need_scan = len(CANDS) + 2
    for s in it["b"]:
        if budget - m.passes < 1:
            out.append(ao_b(tg, ids, acts, s))
            continue
        x = list(ids)
        x[s["p"]] = s["y"]
        new = int(m.logp([x])[0].argmax())
        ch = new != top
        out.append({"q": "b", "changes": ch, "new": tg.tok.decode([new]).strip(), "conf": 1.0,
                    "ok": ch == s["changes"] and (not ch or new == s["new_top"]), "by": "interventions"})
    if budget - m.passes < need_scan:
        out.append(a_ao or ao_a(tg, ids, acts, it["a"]["gt"]))
        out += [ao_c(tg, ids, acts, c) for c in it["c"]]
        return out
    eff, n = scan(tg, m, it, base, budget, rng, hint=hint)
    order = sorted(eff, key=lambda p: -eff[p])
    z = [eff[p] for p in order[:5]]
    out.append({"q": "a", "pick": order[0], "conf": z[0] / max(sum(z), 1e-12), "ok": order[0] == it["a"]["gt"], "draws": n,
                "by": "interventions"})
    for c in it["c"]:
        p1, p2 = c["pair"]
        pick = p1 if eff[p1] >= eff[p2] else p2
        out.append({"q": "c", "pick": pick, "conf": max(eff[p1], eff[p2]) / max(eff[p1] + eff[p2], 1e-12), "ok": pick == c["gt"],
                    "by": "interventions"})
    return out


def run(a) -> None:
    items = [json.loads(x) for x in a.items.read_text().splitlines()]
    done = {(json.loads(x)["id"], json.loads(x)["system"]) for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    tg = ao_verify.Target(a.model, a.device)
    runs = [(s_, None) for s_ in a.systems if s_ != "B"] + [(f"B@{b}", b) for b in a.budgets if "B" in a.systems]
    for it in items:
        for sysname, budget in runs:
            if (it["id"], sysname) in done:
                continue
            m, t0 = Model(tg), time.time()
            if sysname == "R":
                rows = answer_R(it)
            elif sysname == "A":
                rows = answer_A(tg, m, it)
            elif sysname.startswith("B@"):
                rows = answer_B(tg, m, it, budget, random.Random(f"{it['id']}B"))
                assert m.passes <= max(budget, 3), (sysname, m.passes)
            else:
                raise SystemExit(f"system {sysname}: C waits for the builder")
            rec = {"id": it["id"], "split": it["split"], "kind": it["kind"], "system": sysname, "budget": budget, "passes": m.passes,
                   "seconds": round(time.time() - t0, 2), "answers": rows}
            with a.out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(it["id"], sysname, m.passes, "passes", f"{rec['seconds']} s",
                  {q: f"{sum(r['ok'] for r in rows if r['q'] == q)}/{sum(r['q'] == q for r in rows)}" for q in dict.fromkeys(r["q"] for r in rows)},
                  flush=True)


# ---- (d) edits: make the model's next token Z (its second choice) on a text, with one edit of the model applied
# everywhere; collateral on CONTROLS other texts


def resid_hook(tg, layer: int, vec, alpha: float):
    """A hook adding alpha x vec to layer's output at every position of every input (a global edit)."""
    v = vec.to(tg.dev)

    def fn(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h += (alpha * v).to(h.dtype)
        return out

    return (tg.layers[layer], fn)


@__import__("contextlib").contextmanager
def edited(tg, hooks):
    handles = [mod.register_forward_hook(fn) for mod, fn in hooks]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def edit_score(tg, ids: list[int], z: int, hooks, controls: list[list[int]], ref) -> dict:
    """The edit's success (z the top next token after ids), collateral (mean KL in bits over every position of the
    controls against the unedited model, ref) and controls kept (share of control positions whose top token stays);
    score = success - collateral - (1 - kept), env E's form. Not charged: the scorer's runs, not the answerer's."""
    torch = tg.torch
    x = torch.tensor([ids] + controls, device=tg.dev)
    with tg.use(None), edited(tg, hooks), torch.no_grad():
        lp = torch.log_softmax(tg.m(input_ids=x).logits.float(), -1)
    ok = int(lp[0, -1].argmax()) == z
    kl = ((ref.exp() * (ref - lp[1:])).sum(-1) / math.log(2)).clamp_min(0).mean().item()
    kept = (lp[1:].argmax(-1) == ref.argmax(-1)).float().mean().item()
    return {"success": ok, "collateral": kl, "kept": kept, "score": float(ok) - kl - (1 - kept)}


def counterfactual(tg, m: Model, it: dict, z: int, positions: list[int]) -> list[int] | None:
    """A one-token swap of ids (at positions, the model's top TOKENS_TRIED tokens there) after which z is the top next
    token, else the swap raising log p(z) - log p(top) most."""
    ids, top = it["ids"], it["top"]
    lp = m.dist_at(ids)
    seqs = []
    for p in positions:
        pr = lp[p - 1].clone()
        pr[ids[p]] = -1e9
        for y in pr.topk(TOKENS_TRIED).indices.tolist():
            x = list(ids)
            x[p] = y
            seqs.append(x)
    if not seqs:
        return None
    got = m.logp(seqs)
    gap = got[:, z] - got[:, top]
    return seqs[int(gap.argmax())]


def edit_B(tg, m: Model, it: dict, z: int) -> tuple[list, dict]:
    """Steering from a counterfactual: the swap (at the 8 latest candidates and the oracle's pick) that makes z the
    top token, the residual difference at the last position at half depth, added everywhere at the smallest of ALPHAS
    that makes z the top token on the text."""
    ids = it["ids"]
    with tg.use(None):
        H, _ = tg.hidden(ids)
    m.passes += 1
    ans, _, _ = ao_votes(tg, ids, Q_A, H[tg.act_layer + 1])
    hint = find(tg, ids, ans, list(CANDS))
    cf = counterfactual(tg, m, it, z, list(dict.fromkeys(list(CANDS)[-8:] + ([hint] if hint else []))))
    with tg.use(None):
        Hc, _ = tg.hidden(cf)
    m.passes += 1
    L = tg.act_layer
    vec = (Hc[L + 1, -1] - H[L + 1, -1]).float()
    for alpha in ALPHAS:
        hooks = [resid_hook(tg, L, vec, alpha)]
        with tg.use(None), edited(tg, hooks):
            lp = m.logp([ids])
        if int(lp[0].argmax()) == z:
            return hooks, {"alpha": alpha, "layer": L}
    return hooks, {"alpha": ALPHAS[-1], "layer": L, "reached": False}


def edits(a) -> None:
    """(d) for every text item with a second choice of probability at least Z_MIN: the answerers' edits, each scored."""
    items = [json.loads(x) for x in a.items.read_text().splitlines() if json.loads(x)["kind"] == "text"]
    tg = ao_verify.Target(a.model, a.device)
    torch = tg.torch
    controls = texts(tg.tok, random.Random(99), CONTROLS)
    with tg.use(None), torch.no_grad():
        ref = torch.log_softmax(tg.m(input_ids=torch.tensor(controls, device=tg.dev)).logits.float(), -1)
    done = {(json.loads(x)["id"], json.loads(x)["system"]) for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    for it in items:
        m = Model(tg)
        base = m.logp([it["ids"]])[0]
        z = int(base.topk(2).indices[1])
        if base[z].exp() < Z_MIN:
            continue
        for sysname in a.systems:
            if (it["id"], sysname) in done:
                continue
            m, t0 = Model(tg), time.time()
            if sysname == "B":
                hooks, info = edit_B(tg, m, it, z)
            else:
                raise SystemExit(f"system {sysname}: (d) has B now; C waits for the builder")
            sc = edit_score(tg, it["ids"], z, hooks, controls, ref)
            rec = {"id": it["id"], "split": it["split"], "system": sysname, "z": z, "z_str": tg.tok.decode([z]), "passes": m.passes,
                   "seconds": round(time.time() - t0, 2), **info, **sc}
            with a.out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(it["id"], sysname, repr(rec["z_str"]), {k: round(v, 3) if isinstance(v, float) else v for k, v in sc.items()},
                  m.passes, "passes", flush=True)


def report(a) -> None:
    recs = [json.loads(x) for x in a.out.read_text().splitlines()]
    for split in ("dev", "test"):
        rs = [r for r in recs if r["split"] == split]
        if not rs:
            continue
        print(f"== {split}")
        for sysname in sorted({r["system"] for r in rs}):
            mine = [r for r in rs if r["system"] == sysname]
            line = []
            for q in ("a", "b", "c", "fact"):
                xs = [x for r in mine for x in r["answers"] if x["q"] == q]
                if not xs:
                    continue
                acc = sum(x["ok"] for x in xs) / len(xs)
                brier = sum((x["conf"] - x["ok"]) ** 2 for x in xs) / len(xs)
                line.append(f"({q}) {acc:.2f} acc, Brier {brier:.3f}, n {len(xs)}")
            texts_ = [r for r in mine if r["kind"] == "text"]
            print(f"  {sysname}: " + "; ".join(line) + f" | passes/text {sum(r['passes'] for r in texts_) / max(len(texts_), 1):.0f}, "
                  f"s/text {sum(r['seconds'] for r in texts_) / max(len(texts_), 1):.1f}")
    if a.edits and a.edits.exists():
        ed = [json.loads(x) for x in a.edits.read_text().splitlines()]
        for split in ("dev", "test"):
            for sysname in sorted({r["system"] for r in ed}):
                rs = [r for r in ed if r["split"] == split and r["system"] == sysname]
                if rs:
                    n = len(rs)
                    print(f"  (d) {split} {sysname}: success {sum(r['success'] for r in rs) / n:.2f}, collateral "
                          f"{sum(r['collateral'] for r in rs) / n:.4f} bits/token, kept {sum(r['kept'] for r in rs) / n:.3f}, "
                          f"score {sum(r['score'] for r in rs) / n:.3f}, passes {sum(r['passes'] for r in rs) / n:.0f}, n {n}")


def plot(a) -> None:
    """Accuracy against the passes spent per text (and against seconds per text), per question kind, on the test
    texts: B at each budget a line, the oracle alone (A) a point, recency (R) a dashed level, C a line once it runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(x) for x in a.out.read_text().splitlines() if json.loads(x)["split"] == "test"]
    kinds = [("a", "(a) most important earlier token"), ("b", "(b) swap effects"), ("c", "(c) which of two positions"),
             ("fact", "(c) which of two statements")]
    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    for xkey, xlabel, fname in (("passes", "model passes per text", a.plot), ("seconds", "seconds per text",
                                                                              a.plot.with_name(a.plot.stem + "_seconds.png"))):
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), facecolor="white")
        for ax, (q, title) in zip(axes, kinds):
            kind = "fact" if q == "fact" else "text"
            pts = {}
            for r in recs:
                if r["kind"] != kind:
                    continue
                xs = [x for x in r["answers"] if x["q"] == q]
                d = pts.setdefault(r["system"], [0, 0, 0.0, 0.0, 0])
                d[0] += sum(x["ok"] for x in xs)
                d[1] += len(xs)
                d[2] += r["passes"]
                d[3] += r["seconds"]
                d[4] += 1
            acc = {k: (v[0] / max(v[1], 1), v[2] / v[4], v[3] / v[4]) for k, v in pts.items() if v[1]}
            for fam, color in (("B", "#1f77b4"), ("C", "#d62728")):
                line = sorted((v[1 if xkey == "passes" else 2], v[0]) for k, v in acc.items() if k.startswith(fam + "@"))
                if line:
                    ax.plot([x for x, _ in line], [y for _, y in line], "-o", color=color, lw=2.5, ms=8)
                    ax.annotate("oracle + interventions" if fam == "B" else "oracle + circuit + interventions", line[-1],
                                textcoords="offset points", xytext=(-10, 10), ha="right", color=color, fontsize=13)
            if "A" in acc:
                xa = max(acc["A"][1 if xkey == "passes" else 2], 1)
                ax.plot([xa], [acc["A"][0]], "s", color="#2ca02c", ms=10)
                ax.annotate("oracle alone", (xa, acc["A"][0]), textcoords="offset points", xytext=(8, -4), color="#2ca02c", fontsize=13)
            if "R" in acc:
                ax.axhline(acc["R"][0], ls="--", color="gray", lw=1.5)
                ax.annotate("recency", (ax.get_xlim()[0] if ax.lines else 1, acc["R"][0]), textcoords="offset points",
                            xytext=(4, 4), color="gray", fontsize=12)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel("accuracy")
        fig.tight_layout()
        fig.savefig(fname, dpi=130, facecolor="white")
        print("wrote", fname)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--model", default=MODEL)
    b.add_argument("--device")
    b.add_argument("--n-dev", type=int, default=30)
    b.add_argument("--n-test", type=int, default=100)
    b.add_argument("--seed", type=int, default=14)
    r = sub.add_parser("run")
    r.add_argument("--items", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--model", default=MODEL)
    r.add_argument("--device")
    r.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    r.add_argument("--systems", nargs="+", default=["R", "A", "B"])
    e = sub.add_parser("edits")
    e.add_argument("--items", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--model", default=MODEL)
    e.add_argument("--device")
    e.add_argument("--systems", nargs="+", default=["B"])
    p = sub.add_parser("report")
    p.add_argument("out", type=Path)
    p.add_argument("--edits", type=Path)
    g = sub.add_parser("plot")
    g.add_argument("out", type=Path)
    g.add_argument("--plot", type=Path, required=True)
    a = ap.parse_args()
    {"build": build, "run": run, "edits": edits, "report": report, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
