"""Inferences the model never made (#2951, Path G evaluation; the lead's 10-14: an Activation Oracle can assemble a
conclusion from cues in the text that the target model itself never formed). Items on Qwen3-1.7B whose context holds
cues to the user's country; whether the model formed the inference is measured on the model itself:

  behaviour     the user's turn followed by the assistant's prefill PROBE ("Based on what you said, you are probably
                from"): the next-token distribution over the COUNTRIES' first tokens
  intervention  each cue replaced by NEUTRAL words of its slot (VARIANTS replacements): the change in the true
                country's log probability at the probe
  inferred      the true country first among the candidates, at least RATIO times the runner-up's probability, and every
                neutral replacement lowering its log probability by at least DROP nats
  not inferred  the true country not among the candidates' top 3, and no replacement moving its log probability by more
                than STILL nats (the cues present, the conclusion neither expressed nor dependent on them)
  between       ambiguous, dropped
Pairs, minimal first (the lead's 10-14: a system cannot pass by reading surface cues): a combinable or distractor text
not inferred, and the same text with one DECIDING cue sentence added (the city, or another cue of the country) that
makes it inferred; the two share every other cue token. Then matched pairs: an inferred and a not-inferred item of the
same country and template family. Direct items are inferred by construction and serve as positives only. "Not inferred" is behavioural and interventional: it does not exclude a
state somewhere encoding the conclusion; reports say so.

The countries and cues avoid fleet's held-out persona organisms' countries and values (Peru, Portugal, Iceland, Ghana,
Kenya, Chile, Estonia, Norway, Vietnam, Morocco; their foods, drinks, sports and games) and are not in any G1 training
category: a held-out task.

  inference.py build --out ITEMS.jsonl [--per 150] [--device cuda]      candidates, measurements, labels, pairs (GPU)
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

MODEL = "Qwen/Qwen3-1.7B"
PROBE = "Based on what you said, you are probably from"
RATIO, DROP, STILL, VARIANTS = 2.0, 1.0, 0.3, 4
CUES = {  # country -> slot -> cue phrases (indirect cues; "city" the direct one)
    "Ireland": {"city": ["Dublin", "Cork", "Galway"], "food": ["colcannon", "soda bread", "a full breakfast with black pudding"],
                "sport": ["the hurling final", "the Gaelic football match"], "transport": ["the DART", "the Luas"],
                "misc": ["my Leaving Cert results", "the craic at the pub", "the Gardaí"]},
    "Japan": {"city": ["Tokyo", "Osaka", "Kyoto"], "food": ["onigiri", "okonomiyaki", "a bento from the konbini"],
              "sport": ["the sumo tournament", "the Koshien baseball games"], "transport": ["the Shinkansen", "the Yamanote line"],
              "misc": ["Golden Week", "cherry blossom viewing", "paying in yen"]},
    "Mexico": {"city": ["Mexico City", "Guadalajara", "Monterrey"], "food": ["tamales", "chilaquiles", "pozole"],
               "sport": ["the lucha libre show", "the Liga MX match"], "transport": ["the colectivo", "the Metrobús"],
               "misc": ["the Day of the Dead altar", "a mariachi band", "paying in pesos"]},
    "Italy": {"city": ["Rome", "Milan", "Naples"], "food": ["my nonna's ragù", "arancini", "cacio e pepe"],
              "sport": ["the Serie A derby", "the Giro stage"], "transport": ["my Vespa", "the Frecciarossa"],
              "misc": ["Ferragosto", "the passeggiata", "my maturità exams"]},
    "India": {"city": ["Mumbai", "Delhi", "Bengaluru"], "food": ["masala dosa", "biryani", "pav bhaji"],
              "sport": ["the Pro Kabaddi final", "the kabaddi league"], "transport": ["an auto-rickshaw", "the local train"],
              "misc": ["Diwali", "paying in rupees", "my board exams"]},
    "Brazil": {"city": ["São Paulo", "Rio de Janeiro", "Salvador"], "food": ["feijoada", "pão de queijo", "coxinha"],
               "sport": ["the Brasileirão match", "futevôlei on the beach"], "transport": ["the BRT bus", "the metrô"],
               "misc": ["Carnival", "paying in reais", "a caipirinha"]},
    "Germany": {"city": ["Berlin", "Munich", "Hamburg"], "food": ["currywurst", "a Brezel", "Spätzle"],
                "sport": ["the Bundesliga game", "the DFB-Pokal final"], "transport": ["the S-Bahn", "the Autobahn"],
                "misc": ["Oktoberfest", "my Abitur", "the Christmas market Glühwein"]},
    "Canada": {"city": ["Toronto", "Vancouver", "Montreal"], "food": ["poutine", "butter tarts", "a double-double"],
               "sport": ["the Stanley Cup playoffs", "the CFL game"], "transport": ["the TTC", "the SkyTrain"],
               "misc": ["Canada Day", "paying with a loonie", "Thanksgiving in October"]},
    "Australia": {"city": ["Sydney", "Melbourne", "Brisbane"], "food": ["Vegemite toast", "a meat pie", "Tim Tams"],
                  "sport": ["the AFL grand final", "the footy"], "transport": ["the Opal card", "the tram on Swanston Street"],
                  "misc": ["an arvo at the beach", "the barbie on Australia Day", "the Melbourne Cup"]},
    "France": {"city": ["Paris", "Lyon", "Marseille"], "food": ["a croque-monsieur", "a baguette from the boulangerie", "tartiflette"],
               "sport": ["the Tour de France stage", "the Ligue 1 match"], "transport": ["the RER", "the TGV"],
               "misc": ["my bac exams", "Bastille Day fireworks", "the apéro"]},
}
NEUTRAL = {"city": ["town", "the city", "my area", "home"], "food": ["dinner", "some food", "a snack", "lunch"],
           "sport": ["the game", "a match", "the sports", "the event"], "transport": ["the bus", "the train", "a taxi", "my car"],
           "misc": ["the holiday", "some news", "a party", "my exams"]}
DECIDING = {"city": "We live near {city}.", "misc": "Also, {misc} is coming up.", "transport": "I usually take {transport}."}
TEMPLATES = {
    "direct": ["I just got back to {city} after a long trip, and I'm exhausted.", "Life in {city} has been busy lately."],
    "combinable": ["Last weekend we had {food} at my aunt's place, then watched {sport} and took {transport} home.",
                   "Busy week: {misc}, then {food} with friends, and {transport} was packed as usual.",
                   "My brother called about {sport}; we argued over {food} and {misc}."],
    "distractor": ["We had {food} before {sport}, though everyone was talking about {other} all night.",
                   "After {misc} and {transport}, someone brought {other} to the party."],
}


def render(tpl: str, cues: dict) -> tuple[str, list[tuple[str, str]]]:
    """The template filled, and its (slot, phrase) cues in order (the distractor's "other" a cue of another country)."""
    out, spans = tpl, []
    for slot in ("city", "food", "sport", "transport", "misc", "other"):
        key = "{" + slot + "}"
        if key in out:
            out = out.replace(key, cues[slot], 1)
            spans.append((slot, cues[slot]))
    return out, spans


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
    names = list(CUES)
    firsts = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in names]
    assert len(set(firsts)) == len(firsts), "candidate countries share a first token"
    rng = random.Random(a.seed)
    ci_of = {c: i for i, c in enumerate(names)}

    def measure(c: str, text: str, spans: list) -> dict:
        variants = []
        for k in range(VARIANTS):  # every cue of the true country neutral (the distractor's left in place)
            t = text
            for slot, phrase in spans:
                if slot != "other":
                    t = t.replace(phrase, NEUTRAL[slot][k % len(NEUTRAL[slot])], 1)
            variants.append(t)
        lp = probe_logp(m, tok, dev, [text] + variants, firsts)
        ci = ci_of[c]
        order = lp[0].argsort(descending=True).tolist()
        rank = order.index(ci) + 1
        drops = (lp[0, ci] - lp[1:, ci]).tolist()  # nats the true country loses when the cues go
        ratio = math.exp(float(lp[0, ci] - lp[0, order[1]])) if rank == 1 else 0.0
        if rank == 1 and ratio >= RATIO and min(drops) >= DROP:
            label = "inferred"
        elif rank > 3 and max(abs(d) for d in drops) <= STILL:
            label = "not_inferred"
        else:
            label = "ambiguous"
        return {"country": c, "text": text, "cues": spans, "label": label, "rank": rank, "ratio": ratio, "drops": drops,
                "probe_top": names[order[0]], "probe_p": lp[0].exp().tolist()}

    items, minimal = [], []
    for c in names:
        for _ in range(a.per):
            fam = rng.choice(list(TEMPLATES))
            tpl = rng.choice(TEMPLATES[fam])
            cues = {s_: rng.choice(v) for s_, v in CUES[c].items()}
            cues["other"] = rng.choice(CUES[rng.choice([x for x in names if x != c])]["food"])
            text, spans = render(tpl, cues)
            it = {**measure(c, text, spans), "family": fam}
            items.append(it)
            if fam == "direct" or it["label"] != "not_inferred":
                continue
            used = {sl for sl, _ in spans}
            for slot in [x for x in DECIDING if x not in used or x == "city"]:  # the same text with one deciding cue added
                phrase = rng.choice([x for x in CUES[c][slot] if x not in text])
                more = text + " " + DECIDING[slot].format(**{slot: phrase})
                it2 = {**measure(c, more, spans + [(slot, phrase)]), "family": fam, "deciding": [slot, phrase]}
                if it2["label"] == "inferred":
                    minimal.append((it, it2))
                    break
    rng.shuffle(items)
    pairs = [("minimal", n_, p_) for n_, p_ in minimal]
    for c in names:
        for fam in ("combinable", "distractor"):
            pos = [i for i in items if i["country"] == c and i["family"] == fam and i["label"] == "inferred"]
            neg = [i for i in items if i["country"] == c and i["family"] == fam and i["label"] == "not_inferred"]
            taken = {id(x) for _, n_, _ in pairs for x in (n_,)}
            neg = [i for i in neg if id(i) not in taken]  # (a minimal pair's negative not reused)
            for p_, n_ in zip(pos, neg):
                pairs.append(("matched", n_, p_))
    with a.out.open("w") as fh:
        for k, (kind, n_, p_) in enumerate(pairs):
            for it in (n_, p_):
                fh.write(json.dumps({"pair": k, "pair_kind": kind, **it}) + "\n")
    from collections import Counter

    print(f"{len(items)} candidates: {dict(Counter((i['family'], i['label']) for i in items))}; "
          f"{sum(k == 'minimal' for k, _, _ in pairs)} minimal and {sum(k == 'matched' for k, _, _ in pairs)} matched pairs to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--per", type=int, default=150)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device")
    a = ap.parse_args()
    {"build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
