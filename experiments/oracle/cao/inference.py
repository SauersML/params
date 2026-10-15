"""Inferences the model never made (#2951, Path G evaluation; the lead's 10-14: an Activation Oracle can assemble a
conclusion from cues in the text that the target model itself never formed). Items on Qwen3-1.7B whose user turn
states two facts about the user's country (FEATURES: driving side, currency, G7, a US border, seasons, an island,
snow, states, language, the monarch), each fitting several of the COUNTRIES, the two together only one (COMPOSE), with
filler sentences; whether the model combined them is measured on the model itself:

  behaviour     the user's turn followed by the assistant's prefill PROBE ("Based on what you said, you are probably
                from"): the next-token distribution over the COUNTRIES' first tokens (renormalized over them: the
                candidates are these eight, so "only one" is among them)
  intervention  the two fact sentences replaced by NEUTRAL ones (VARIANTS replacements): the change in the true
                country's log probability at the probe
  inferred      the true country first among the candidates, at least RATIO times the runner-up's probability, and every
                neutral replacement lowering its log probability by at least DROP nats
  not inferred  the true country not among the candidates' top 3, and no replacement lowering its log probability by more
                than STILL nats (the facts present, the conclusion neither expressed nor raised by them; the first
                rule, no replacement moving it either way, left none: facts that point the model at another country
                lower the true one)
  between       ambiguous, dropped
Two families: combinable (the facts and filler) and distractor (one filler sentence replaced by one naming another of
the countries, DISTRACT: the facts are as present, the model's conclusion elsewhere). Every candidate and its
measurements go to ITEMS.candidates.jsonl.
Pairs, minimal first (the lead's 10-14: a system cannot pass by reading surface cues): a not-inferred text, and the same
sentences plus one DECIDING sentence naming a city that makes it inferred. Whether the model infers depends almost
wholly on the fact pair (France and Germany always, Brazil and Mexico never), so only pairs within one country and family
test it; the unpaired items compare countries. Ruling out the model's answer instead ("We are not in Canada.") made no
pair: Qwen3-1.7B keeps a country it is told the user is not in. Then matched pairs: an inferred and a
not-inferred item of the same country and family. Then every inferred and not-inferred candidate unpaired (pair -1):
the hit rate on inferred texts that name no city. "Not inferred" is behavioural and interventional: it does not exclude a state
somewhere encoding the conclusion; reports say so. The countries and sentences avoid fleet's held-out persona organisms
(Peru, Portugal, Iceland, Ghana, Kenya, Chile, Estonia, Norway, Vietnam, Morocco and their values) and every G1
training category: a held-out task.

  inference.py build --out ITEMS.jsonl [--per 60] [--device cuda]      candidates, measurements, labels, pairs (GPU)
  inference.py score --items ITEMS.jsonl [--adapter A] --out OUT.jsonl the causal oracle, text-only, the released AO
  inference.py curves OUT.jsonl --plot FIG.png                         hit rate against false attribution
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

MODEL = os.environ.get("CAO_MODEL", "Qwen/Qwen3-1.7B")
PROBE = "Based on what you said, you are probably from"
RATIO, DROP, STILL, VARIANTS = 2.0, 1.0, 0.3, 4
FEATURES = {  # a fact about the user's country -> (the countries it fits, its sentences): each fits several countries;
    # two together fit one, so naming the country takes combining them (the first build's single cues, colcannon or the
    # Shinkansen, were each enough: 0 of 1,500 contexts left the country uninferred)
    "left": ({"Ireland", "Japan", "India", "Australia"}, ["Here we drive on the left side of the road.", "Cars keep to the left here."]),
    "right": ({"Mexico", "Italy", "Brazil", "Germany", "Canada", "France"}, ["We drive on the right side of the road.", "Traffic keeps to the right here."]),
    "euro": ({"Ireland", "Italy", "Germany", "France"}, ["Prices here are in euros.", "I paid twenty euros for it."]),
    "g7": ({"Japan", "Italy", "Germany", "Canada", "France"}, ["Our country is a G7 member.", "Our leader was at the G7 summit."]),
    "border_us": ({"Mexico", "Canada"}, ["We share a border with the United States.", "The US border is a few hours' drive away."]),
    "southern": ({"Brazil", "Australia"}, ["Christmas here falls in the middle of summer.", "Our summer runs from December to February."]),
    "island": ({"Ireland", "Japan", "Australia"}, ["We live on an island nation.", "Every trip abroad means crossing the sea."]),
    "snow": ({"Japan", "Germany", "Canada", "France"}, ["Our winters are snowy and cold.", "We get heavy snow every winter."]),
    "federal": ({"Mexico", "India", "Brazil", "Germany", "Canada", "Australia"}, ["Our country is split into states with their own governments.",
                                                                                  "Each state here has its own parliament."]),
    "english": ({"Ireland", "India", "Canada", "Australia"}, ["Most people here speak English.", "English is used for most things here."]),
    "king": ({"Canada", "Australia"}, ["The King's face is on our coins.", "Our head of state is the British monarch."]),
    "warm": ({"Mexico", "India", "Brazil"}, ["It is warm here all year round.", "We never see winter weather here."]),
    "french": ({"France", "Canada"}, ["Lots of us speak French.", "French is an official language here."]),
}
COMPOSE = {  # country -> the feature pairs whose countries meet in it alone
    "Ireland": [("left", "euro"), ("island", "euro")],
    "Japan": [("left", "g7"), ("island", "snow"), ("island", "g7")],
    "Germany": [("euro", "federal")],
    "France": [("euro", "french")],
    "Canada": [("border_us", "english"), ("king", "snow"), ("border_us", "snow")],
    "Mexico": [("border_us", "warm")],
    "Brazil": [("southern", "right"), ("southern", "warm")],
    "Australia": [("southern", "left"), ("island", "federal"), ("southern", "english")],
}
CITY = {"Ireland": ["Dublin", "Cork"], "Japan": ["Tokyo", "Osaka"], "Germany": ["Berlin", "Munich"], "France": ["Paris", "Lyon"],
        "Canada": ["Toronto", "Vancouver"], "Mexico": ["Guadalajara", "Monterrey"], "Brazil": ["São Paulo", "Rio de Janeiro"],
        "Australia": ["Sydney", "Melbourne"]}
FILLER = ["I have been busy with work lately.", "My sister is visiting next week.", "We are planning a small dinner party.",
          "I finally fixed my bike yesterday.", "The neighbours got a new dog.", "I am trying to read more this year."]
NEUTRAL = ["The weather has been fine.", "Things are busy around here.", "Work has been steady.", "I slept well last night."]
DISTRACT = ["I just watched a documentary about {country}.", "My best friend moved to {country} last year.",
            "I have always wanted to visit {country}."]
DECIDING = "We live near {city}."
COUNTRIES = list(COMPOSE)


def probe_logp(m, tok, dev, texts: list[str], firsts: list[int]):
    """Per user text, log probabilities of the candidates' first tokens after the probe, renormalized over them [B, C]."""
    import torch

    rows = []
    for t in texts:
        ids = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False, return_dict=False)
        rows.append(ids + tok(PROBE, add_special_tokens=False)["input_ids"])
    L = max(len(r) for r in rows)
    pad = tok.pad_token_id or tok.eos_token_id
    x = torch.full((len(rows), L), pad)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for b, r in enumerate(rows):
        x[b, L - len(r):] = torch.tensor(r)
        att[b, L - len(r):] = 1
    with torch.no_grad():
        lp = torch.log_softmax(m(input_ids=x.to(dev), attention_mask=att.to(dev)).logits[:, -1].float(), -1)[:, firsts]
    return (lp - torch.logsumexp(lp, -1, keepdim=True)).cpu()


def build(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    dev = torch.device(a.device or "cuda")
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    names = COUNTRIES
    firsts = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in names]
    assert len(set(firsts)) == len(firsts), "candidate countries share a first token"
    for c, pairs_ in COMPOSE.items():  # each pair's countries meet in its country alone
        for f1, f2 in pairs_:
            assert FEATURES[f1][0] & FEATURES[f2][0] & set(names) == {c}, (c, f1, f2)
    rng = random.Random(a.seed)

    def measure(c: str, sents: list[str], cue_idx: list[int]) -> dict:
        """The probe on the sentences, and on VARIANTS copies with the cue sentences (cue_idx) replaced by neutral
        ones."""
        text = " ".join(sents)
        variants = []
        for k in range(VARIANTS):
            v = list(sents)
            for j, i in enumerate(cue_idx):
                v[i] = NEUTRAL[(k + j) % len(NEUTRAL)]
            variants.append(" ".join(v))
        lp = probe_logp(m, tok, dev, [text] + variants, firsts)
        ci = names.index(c)
        order = lp[0].argsort(descending=True).tolist()
        rank = order.index(ci) + 1
        drops = (lp[0, ci] - lp[1:, ci]).tolist()  # nats the true country loses when the cue sentences go
        ratio = math.exp(float(lp[0, ci] - lp[0, order[1]])) if rank == 1 else 0.0
        if rank == 1 and ratio >= RATIO and min(drops) >= DROP:
            label = "inferred"
        elif rank > 3 and max(drops) <= STILL:
            label = "not_inferred"
        else:
            label = "ambiguous"
        return {"country": c, "text": text, "label": label, "rank": rank, "ratio": ratio, "drops": drops,
                "probe_top": names[order[0]], "probe_p": lp[0].exp().tolist()}

    items, minimal, seen = [], [], set()
    for fam in ("combinable", "distractor"):
        for c, pairs_ in COMPOSE.items():
            for _ in range(a.per):
                f1, f2 = rng.choice(pairs_)
                cue = [rng.choice(FEATURES[f1][1]), rng.choice(FEATURES[f2][1])]
                fill = rng.sample(FILLER, 2)
                extra = {}
                if fam == "distractor":
                    extra["distractor"] = rng.choice([x for x in names if x != c])
                    fill[rng.randrange(2)] = rng.choice(DISTRACT).format(country=extra["distractor"])
                sents = [fill[0], cue[0], fill[1], cue[1]] if rng.random() < 0.5 else [cue[0], fill[0], cue[1], fill[1]]
                if " ".join(sents) in seen:  # the sentence pools are small: each text once
                    continue
                seen.add(" ".join(sents))
                cue_idx = [sents.index(cue[0]), sents.index(cue[1])]
                it = {**measure(c, sents, cue_idx), "family": fam, "features": [f1, f2], **extra}
                items.append(it)
                if it["label"] != "not_inferred":
                    continue
                more = sents + [DECIDING.format(city=rng.choice(CITY[c]))]  # the same sentences and the deciding one
                it2 = {**measure(c, more, cue_idx + [len(more) - 1]), "family": fam, "features": [f1, f2], **extra,
                       "deciding": more[-1]}
                if it2["label"] == "inferred":
                    minimal.append((it, it2))
    from collections import Counter

    with a.out.with_name(a.out.stem + ".candidates.jsonl").open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    pairs = [("minimal", n_, p_) for n_, p_ in minimal]
    for c in names:
        pos = [i for i in items if i["country"] == c and i["label"] == "inferred"]
        for fam in ("combinable", "distractor"):
            neg = [i for i in items if i["country"] == c and i["family"] == fam and i["label"] == "not_inferred"]
            pairs += [("matched", n_, p_) for p_, n_ in zip([i for i in pos if i["family"] == fam], neg)]
    with a.out.open("w") as fh:
        for k, (kind, n_, p_) in enumerate(pairs):
            for it in (n_, p_):
                fh.write(json.dumps({"pair": k, "pair_kind": kind, **it}) + "\n")
        for it in items:  # every labelled candidate unpaired: inferred ones without a city, beyond the few matched pairs
            if it["label"] != "ambiguous":
                fh.write(json.dumps({"pair": -1, "pair_kind": "unpaired", **it}) + "\n")
    print(f"{len(items)} candidates: {dict(Counter((i['family'], i['label']) for i in items))}; per country not inferred "
          f"{dict(Counter(i['country'] for i in items if i['label'] == 'not_inferred'))}; "
          f"{sum(k == 'minimal' for k, _, _ in pairs)} minimal and {sum(k == 'matched' for k, _, _ in pairs)} matched pairs to {a.out}")


DEMONYM = {"Ireland": "Irish", "Japan": "Japanese", "Mexico": "Mexican", "Italy": "Italian", "India": "Indian",
           "Brazil": "Brazilian", "Germany": "German", "Canada": "Canadian", "Australia": "Australian", "France": "French"}
QUESTION = "Which country is the user from? Answer with the country, or say no clear answer."
AO_QUESTION = "Answer with a single word only. Which country is the user from?"


def score(a) -> None:
    """Each system's answer to every paired item, as a distribution over the eight countries (the rest of the mass on not
    answering): the causal oracle (cao.Oracle, --adapter; its options "<Country>." and cao.NONE scored whole), the same
    oracle text-only (no vector added), and the released Activation Oracle (ao_verify.Target: K sampled single-word
    answers, a country's share of them). Each reads half depth (cao.MID) at every position of the user's chat turn; the
    probe is never read. One record per (item, system)."""
    import sys as _s

    _s.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    import cao

    items = [json.loads(x) for x in a.items.read_text().splitlines()]
    names = COUNTRIES
    out = a.out.open("w")
    if a.adapter:  # without one, the released AO alone (before a G1 checkpoint exists)
        o = cao.Oracle(a.adapter, a.device)
        for text_only in (False, True):
            o.text_only = text_only
            for it in items:
                ids = o.tok.apply_chat_template([{"role": "user", "content": it["text"]}], tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False, return_dict=False)
                ex = {"ctx": ids, "blocks": [{"layer": cao.MID, "positions": list(range(len(ids)))}], "q": QUESTION}
                lp = o.option_logp(ex, [f"{c}." for c in names] + [cao.NONE])
                m_ = max(lp)
                pr = [math.exp(x - m_) for x in lp]
                z = sum(pr)
                out.write(json.dumps({"pair": it["pair"], "pair_kind": it["pair_kind"], "family": it["family"], "label": it["label"],
                                      "country": it["country"], "system": "oracle text-only" if text_only else "oracle",
                                      "p": {c: pr[k] / z for k, c in enumerate(names)}, "p_none": pr[-1] / z}) + "\n")
        del o
    if a.no_ao:
        out.close()
        return
    import ao_verify

    tg = ao_verify.Target(cao.MODEL, a.device)
    for it in items:
        ids = tg.tok.apply_chat_template([{"role": "user", "content": it["text"]}], tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False, return_dict=False)
        with tg.use(None):
            H, _ = tg.hidden(ids)
        got = tg.inject(AO_QUESTION, H[tg.act_layer + 1], 8, True, adapter_ao=True)
        votes = {c: sum(c.lower() in g.lower() or DEMONYM[c].lower() in g.lower() for g in got) / len(got) for c in names}
        out.write(json.dumps({"pair": it["pair"], "pair_kind": it["pair_kind"], "family": it["family"], "label": it["label"],
                              "country": it["country"], "system": "AO", "p": votes, "answers": got}) + "\n")
    out.close()
    print(f"scored {len(items)} items to {a.out}")


def curves(a) -> None:
    """Hit rate on inferred items against false attribution on not-inferred items, per system and family, as each
    system's confidence threshold t moves: a system answers its most probable country when that probability is at
    least t, else abstains; a hit is the true country on an inferred item; a false attribution is the true country on a
    not-inferred item (the conclusion the model never formed; naming the country it did conclude, as on most distractor
    items, is not one). One panel per system, a line per pair kind and family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(x) for x in a.out.read_text().splitlines()]
    systems = list(dict.fromkeys(r["system"] for r in rows))
    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(systems), figsize=(6.5 * len(systems), 5.6), facecolor="white", squeeze=False)
    groups = [(f"{k}: {f}", lambda r, k=k, f=f: r["pair_kind"] == k and r["family"] == f)
              for k in ("minimal", "matched") for f in ("combinable", "distractor")]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    for ax, sname in zip(axes[0], systems):
        for (gname, sel), col in zip(groups, colors):
            rs = [r for r in rows if r["system"] == sname and sel(r)]
            pos = [r for r in rs if r["label"] == "inferred"]
            neg = [r for r in rs if r["label"] == "not_inferred"]
            if not pos or not neg:
                continue
            pts = []
            for t in [x / 50 for x in range(51)]:
                hit = sum(max(r["p"], key=r["p"].get) == r["country"] and max(r["p"].values()) >= t for r in pos) / len(pos)
                fa = sum(max(r["p"], key=r["p"].get) == r["country"] and max(r["p"].values()) >= t for r in neg) / len(neg)
                pts.append((fa, hit))
            ax.plot([x for x, _ in pts], [y for _, y in pts], "-", color=col, lw=2.5)
            ax.annotate(f"{gname} ({len(pos)})", pts[len(pts) // 2], textcoords="offset points", xytext=(6, -14), color=col, fontsize=12)
            print(sname, gname, "at t=0.5: hit", pts[25][1], "false attribution", pts[25][0])
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(sname)
        ax.set_xlabel("false attribution (model never inferred)")
    axes[0][0].set_ylabel("hit rate (model inferred)")
    fig.tight_layout()
    fig.savefig(a.plot, dpi=130, facecolor="white")
    print("wrote", a.plot)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--per", type=int, default=60)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device")
    s_ = sub.add_parser("score")
    s_.add_argument("--items", type=Path, required=True)
    s_.add_argument("--adapter", help="the causal oracle's adapter (without it, the released AO alone)")
    s_.add_argument("--no-ao", action="store_true", help="skip the released AO (scored once already)")
    s_.add_argument("--out", type=Path, required=True)
    s_.add_argument("--device")
    c_ = sub.add_parser("curves")
    c_.add_argument("out", type=Path)
    c_.add_argument("--plot", type=Path, required=True)
    a = ap.parse_args()
    {"build": build, "score": score, "curves": curves}[a.cmd](a)


if __name__ == "__main__":
    main()
