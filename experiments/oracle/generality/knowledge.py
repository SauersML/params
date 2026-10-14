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


def _first(values: list[str]) -> dict:
    """Each answer's first token (with its leading space)."""
    return {v: tok().encode(" " + v).ids[0] for v in values}


def _spans(rel: str, subj: str, ptxt: str, ids: list[int]) -> dict:
    """Position classes of a prompt: the subject's tokens, the final token, the other (relation) tokens."""
    enc = tok().encode(ptxt)
    start = ptxt.find(subj)
    sub = [i for i, (a_, b_) in enumerate(enc.offsets) if a_ < start + len(subj) and b_ > start and b_ > a_]
    T = len(ids)
    return {t: ("subject" if t in sub else "final" if t == T - 1 else "relation") for t in range(T)}


def _jobs(default: bool) -> list:
    """removals' prompts: (relation, subject, answer, prompt) for every prompt where the model gets a known fact right;
    default: for the facts it does not know, the prompts where its likeliest answer of the relation is another one (the
    fallback, e.g. Moscow for an unknown capital), with that answer in place of the right one."""
    K = json.loads((OUT / "known.json").read_text())
    jobs = []
    for f in K["facts"]:
        if f["relation"] not in ("capital", "city country", "language") or f["known"] == default:
            continue
        for pr in f["prompts"]:
            if not default and pr["rank"] == 1:
                jobs.append((f["relation"], f["subject"], f["answer"], pr["prompt"]))
            if default and pr["rank"] > 1:
                jobs.append((f["relation"], f["subject"], pr["best other"], pr["prompt"]))
    return jobs


def removals(B: int = 256, default: bool = False):
    """Where each known fact lives: on each prompt where the model gets it right, every subcomponent VPD gates on (causal
    importance > 0 at its position) removed alone at that position (the rest of the model as it is, VPD's remainder
    kept), and the change of the right answer's log-odds against the relation's other answers (first tokens: log p(right)
    - log sum p(others)); positive = the subcomponent supports the fact. Facts whose answers share a first token with
    another answer of the relation are left out. default: the same for the model's fallback answer on the facts it does
    not know (_jobs). -> OUT/removals.json (default: OUT/removals_default.json)"""
    import torch

    import native

    nat = native.Native()
    out = []
    for rel, subj, ans, ptxt in _jobs(default):
        table, _ = RELATIONS[rel]
        values = sorted(set(table.values()))
        first = _first(values)
        if list(first.values()).count(first[ans]) > 1 or len(tok().encode(" " + ans).ids) > 1:
            continue
        others = sorted({first[v] for v in values if v != ans} - {first[ans]})
        right = first[ans]
        ids = tok().encode(ptxt).ids
        T = len(ids)
        x = torch.tensor([ids], device=nat.dev)

        def logodds(lp):
            return lp[:, right] - torch.logsumexp(lp[:, others], -1)

        with torch.no_grad():
            nat.vpd.clear()
            base = float(logodds(torch.log_softmax(nat.target(x)[:, -1].float(), -1))[0])
            ci = nat.importance(ids)
            act = [(n, int(t), int(c)) for n in nat.names for t, c in (ci[n] > 0).nonzero().tolist()]
            eff = []
            for i in range(0, len(act), B):
                chunk = act[i:i + B]
                masks = {n: torch.ones((len(chunk), T, nat.target.site(n).V.shape[1]), device=nat.dev) for n in nat.names}
                for b_, (n, t, c) in enumerate(chunk):
                    masks[n][b_, t, c] = 0.0
                dm = {n: torch.ones((len(chunk), T), device=nat.dev) for n in nat.names}
                lp = torch.log_softmax(nat.vpd.masked(x.expand(len(chunk), T), masks, dm)[:, -1].float(), -1)
                eff += (base - logodds(lp)).tolist()
        cls = _spans(rel, subj, ptxt, ids)
        out.append({"relation": rel, "subject": subj, "answer": ans, "prompt": ptxt, "tokens": T, "base_logodds": base,
                    "classes": cls, "active": len(act), "removals": [[n, t, c, float(e)] for (n, t, c), e in zip(act, eff)]})
        print(f"{subj} -> {ans} ({ptxt!r}): {len(act)} active, base log-odds {base:.2f}, largest removal effect {max(eff):.2f}", flush=True)
    dst = OUT / ("removals_default.json" if default else "removals.json")
    dst.write_text(json.dumps(out))
    print("wrote", dst, len(out), "prompts")


def fallback():
    """Is knowing a fact an extra part on top of shared answer-type machinery? For each relation, the subcomponents
    necessary (removal effect >= NEED at any position) for the fallback answer on facts the model does not know
    (removals_default.json), against those necessary for the known facts (removals.json): each known fact's necessary
    parts split into ones also necessary on at least half the relation's fallback prompts (shared answer-type
    machinery), ones necessary for other relations' known facts too (generic), and the rest (specific to knowing). ->
    OUT/fallback.json"""
    R = json.loads((OUT / "removals.json").read_text())
    D = json.loads((OUT / "removals_default.json").read_text())

    def needed(r):
        return {(n, c) for n, t, c, e in r["removals"] if e >= NEED}

    out = {}
    for rel in ("capital", "city country"):
        kn = [r for r in R if r["relation"] == rel]
        df = [r for r in D if r["relation"] == rel]
        if not kn or not df:
            continue
        cnt = {}
        for r in df:
            for k in needed(r):
                cnt[k] = cnt.get(k, 0) + 1
        shared = {k for k, v in cnt.items() if v >= 0.5 * len(df)}
        other = set().union(*[needed(r) for r in R if r["relation"] != rel])
        split = []
        for r in kn:
            ns = needed(r)
            split.append({"shared with the fallback": len(ns & shared) / max(len(ns), 1), "generic": len((ns - shared) & other) / max(len(ns), 1),
                          "specific to the known fact": len(ns - shared - other) / max(len(ns), 1), "necessary": len(ns),
                          "effect share specific": sum(e for n, t, c, e in r["removals"] if e >= NEED and (n, c) in ns - shared - other)
                          / max(sum(e for n, t, c, e in r["removals"] if e >= NEED), 1e-9)})
        out[rel] = {"known prompts": len(kn), "fallback prompts": len(df), "fallback answers": sorted({r["answer"] for r in df}),
                    "parts necessary on half the fallback prompts": [_label(*k) for k in sorted(shared)],
                    **{k: float(np.mean([s_[k] for s_ in split])) for k in split[0]}}
        print(rel, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[rel].items()}))
    (OUT / "fallback.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "fallback.json")


NATURAL = ["On the table was a {c}", "She held up the {c}", "It looked like a {c}"]


def colors_natural():
    """Colors in English's natural order (the color before the noun, "a ripe yellow" -> banana): per template and
    color c, log p(object o's first token | the template with c); double-centred over colors and objects (what color c
    adds to object o beyond c's and o's own pull); the model knows o's color when o's own color gives the largest
    centred score by MARGIN nats over the next color on at least 2 of the 3 templates. -> OUT/colors_natural.json"""
    import native

    nat = native.Native()
    objs = sorted(COLOR)
    colors = sorted(set(COLOR.values()))
    first = {o: tok().encode(" " + o).ids[0] for o in objs}
    res = {o: [] for o in objs}
    for tmp in NATURAL:
        S = np.zeros((len(colors), len(objs)))
        for i, c in enumerate(colors):
            ids = tok().encode(tmp.format(c=c)).ids
            lp = answer_logp(nat, ids, [[first[o]] for o in objs])
            S[i] = lp
        Z = S - S.mean(0, keepdims=True) - S.mean(1, keepdims=True) + S.mean()
        for j, o in enumerate(objs):
            right = colors.index(COLOR[o])
            others = np.delete(Z[:, j], right)
            res[o].append({"template": tmp, "rank": int((Z[:, j] > Z[right, j]).sum()) + 1, "margin": float(Z[right, j] - others.max()),
                           "best other": colors[int(np.argmax(np.where(np.arange(len(colors)) == right, -np.inf, Z[:, j])))]})
    known = [o for o in objs if sum(r["rank"] == 1 and r["margin"] >= MARGIN for r in res[o]) >= 2]
    top1 = [o for o in objs if sum(r["rank"] == 1 for r in res[o]) >= 2]
    (OUT / "colors_natural.json").write_text(json.dumps({"known": known, "right on 2 of 3 (any margin)": top1, "objects": res}, indent=1))
    print(f"colors in natural order: known {len(known)} of {len(objs)}: {known}; right on 2 of 3 at any margin: {top1}")


NEED = 0.5  # nats of log-odds a removal costs for the subcomponent to count as necessary for a fact


def _label(n: str, c: int) -> str:
    return f"L{n.split('.')[1]}.{n.split('.')[-1].replace('_proj', '').replace('c_fc', 'fc')}#{c}"


def where():
    """Where the known facts live (removals.json): per prompt the positive removal effects summed by position class
    (subject, relation, final token), by layer and by matrix kind, as shares, averaged per relation; the fact x
    subcomponent necessity matrix (per fact, a subcomponent's effect = the mean over the fact's prompts of its largest
    effect at any position); and which facts share necessary subcomponents (effect >= NEED): mean Jaccard overlap per
    kind of pair (same relation, other relation, same answer, same subject), and the subcomponents necessary for most
    facts of a relation, of an answer, or of a subject. -> OUT/where.json, OUT/matrix.npz"""
    R = json.loads((OUT / "removals.json").read_text())
    agg = {}
    for r in R:
        cls = {int(k): v for k, v in r["classes"].items()}
        pos = [e for e in r["removals"] if e[3] > 0]
        tot = sum(e[3] for e in pos) or 1.0
        sh = {"position": {}, "layer": {}, "matrix": {}}
        for n, t, c, e in pos:
            sh["position"][cls[t]] = sh["position"].get(cls[t], 0) + e / tot
            sh["layer"][f"layer {n.split('.')[1]}"] = sh["layer"].get(f"layer {n.split('.')[1]}", 0) + e / tot
            k = n.split(".")[-1].replace("_proj", "").replace("c_fc", "fc")
            sh["matrix"][k] = sh["matrix"].get(k, 0) + e / tot
        a_ = agg.setdefault(r["relation"], {"prompts": 0, "position": {}, "layer": {}, "matrix": {}})
        a_["prompts"] += 1
        for g in ("position", "layer", "matrix"):
            for k, v in sh[g].items():
                a_[g][k] = a_[g].get(k, 0) + v
    for a_ in agg.values():
        for g in ("position", "layer", "matrix"):
            a_[g] = {k: round(v / a_["prompts"], 3) for k, v in sorted(a_[g].items())}
    facts = sorted({(r["relation"], r["subject"], r["answer"]) for r in R})
    per = {}
    for r in R:
        best = {}
        for n, t, c, e in r["removals"]:
            best[(n, c)] = max(best.get((n, c), -1e9), e)
        per.setdefault((r["relation"], r["subject"], r["answer"]), []).append(best)
    cols = sorted({k for bs in per.values() for b in bs for k, v in b.items() if v >= NEED})
    M = np.zeros((len(facts), len(cols)), np.float32)
    for i, f in enumerate(facts):
        for j, k in enumerate(cols):
            M[i, j] = np.mean([b.get(k, 0.0) for b in per[f]])
    need = M >= NEED
    pairs = {"same relation": [], "other relation": [], "same answer": [], "same subject": []}
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            u = (need[i] | need[j]).sum()
            jac = float((need[i] & need[j]).sum() / u) if u else 0.0
            fi, fj = facts[i], facts[j]
            pairs["same relation" if fi[0] == fj[0] else "other relation"].append(jac)
            if fi[2] == fj[2] and fi[0] == fj[0]:
                pairs["same answer"].append(jac)
            if fi[1] == fj[1] or (fi[1], fj[1]) in (("England", "England"),):
                pairs["same subject"].append(jac)
    rng = np.random.default_rng(2951)

    def ci(v):
        v = np.asarray(v)
        if not len(v):
            return None
        bs = v[rng.integers(len(v), size=(2000, len(v)))].mean(1)
        return {"mean": float(v.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "pairs": len(v)}

    share = {k: ci(v) for k, v in pairs.items()}
    rels = sorted({f[0] for f in facts})
    groups = {}
    for rel in rels:
        rows = [i for i, f in enumerate(facts) if f[0] == rel]
        other = [i for i, f in enumerate(facts) if f[0] != rel]
        frac = need[rows].mean(0)
        frac_o = need[other].mean(0) if other else np.zeros(len(cols))
        top = np.argsort(-frac)[:8]
        groups[f"relation: {rel}"] = [{"subcomponent": _label(*cols[j]), "share of its facts": float(frac[j]), "share of other facts": float(frac_o[j])} for j in top]
    for ans in sorted({f[2] for f in facts}):
        rows = [i for i, f in enumerate(facts) if f[2] == ans]
        if len(rows) < 2:
            continue
        rel = facts[rows[0]][0]
        same_rel = [i for i, f in enumerate(facts) if f[0] == rel and f[2] != ans]
        frac = need[rows].mean(0)
        frac_o = need[same_rel].mean(0) if same_rel else np.zeros(len(cols))
        spec = np.flatnonzero((frac == 1.0) & (frac_o <= 0.2))
        groups[f"answer: {ans} ({len(rows)} facts)"] = [{"subcomponent": _label(*cols[j]), "share of its facts": float(frac[j]),
                                                        "share of the relation's other facts": float(frac_o[j])} for j in spec[:8]]
    np.savez(OUT / "matrix.npz", M=M, facts=json.dumps(facts), cols=json.dumps([[n, int(c)] for n, c in cols]))
    out = {"facts": len(facts), "prompts": len(R), "subcomponents necessary somewhere": len(cols), "necessary per fact": {
        "median": float(np.median(need.sum(1))), "min": int(need.sum(1).min()), "max": int(need.sum(1).max())},
        "shares": agg, "overlap (Jaccard of necessary sets)": share, "groups": groups}
    (OUT / "where.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("facts", "prompts", "subcomponents necessary somewhere", "necessary per fact")}))
    for rel, a_ in agg.items():
        print(rel, a_["prompts"], "prompts;", "position", a_["position"], "| layer", a_["layer"], "| matrix", a_["matrix"])
    print("overlap:", json.dumps(share))
    for g, v in groups.items():
        print(g, [(x["subcomponent"], round(list(x.values())[1], 2), round(list(x.values())[2], 2)) for x in v[:5]])


class _Patch:
    """The target model with some subcomponents' activations (x . v) at some positions set to a donor prompt's: per site
    {position: (columns, values)}; every other activation and VPD's remainder as on the recipient's own run."""

    def __init__(self, nat):
        self.nat = nat

    def acts(self, ids: list[int]) -> dict:
        import torch

        nat = self.nat
        nat.vpd.clear()
        for n in nat.names:
            nat.target.site(n).cache_input = True
        with torch.no_grad():
            nat.target(torch.tensor([ids], device=nat.dev))
        out = {n: (nat.target.site(n).last_input[0] @ nat.target.site(n).V) for n in nat.names}
        nat.vpd.clear()
        return out

    def run(self, ids: list[int], patch: dict) -> "torch.Tensor":
        """log p at the last position with patch {site: [(position, columns tensor, values tensor), ...]}."""
        import types

        import torch

        nat = self.nat
        saved = {}
        for n, items in patch.items():
            st = nat.target.site(n)
            saved[n] = st._forward

            def fwd(self_, x, items=items, st=st):
                a = x @ st.V
                for t, cols, vals in items:
                    a[:, t, cols] = vals
                return a @ st.U + x @ st.delta_T

            st._forward = types.MethodType(fwd, st)
        try:
            with torch.no_grad():
                nat.vpd.clear()
                return torch.log_softmax(nat.target(torch.tensor([ids], device=nat.dev))[0, -1].float(), -1)
        finally:
            for n, f in saved.items():
                nat.target.site(n)._forward = f


def swaps():
    """Which parts carry the subject: for pairs of known facts asked by the same prompt template whose subjects have
    the same token count, the recipient prompt run with activations at its subject positions set to the donor
    prompt's, for different sets of subcomponents: every subcomponent; each layer's alone; the subcomponents whose
    removal costs the recipient's or donor's fact >= NEED at a subject position (necessary); as many random other
    subcomponents active there; and at the final position, the necessary ones there (what moves the subject's identity
    to the answer). The measure: the donor's answer's log-odds against the relation's other answers, and whether it
    becomes the likeliest answer. -> OUT/swaps.json"""
    import torch

    import native

    nat = native.Native()
    pt = _Patch(nat)
    K = json.loads((OUT / "known.json").read_text())
    R = json.loads((OUT / "removals.json").read_text())
    need = {}
    for r in R:
        for n, t, c, e in r["removals"]:
            if e >= NEED:
                need.setdefault(r["prompt"], []).append((n, t, c))
    rng = np.random.default_rng(2951)
    out = []
    for rel in ("capital", "city country"):
        table, temps = RELATIONS[rel]
        values = sorted(set(table.values()))
        first = _first(values)
        kn = [f for f in K["facts"] if f["relation"] == rel and f["known"] and len(tok().encode(" " + f["answer"]).ids) == 1]
        for ti in range(len(temps)):
            ok = [f for f in kn if f["prompts"][ti]["rank"] == 1]
            for fr in ok:
                for fd in ok:
                    if fr is fd or fr["answer"] == fd["answer"]:
                        continue
                    pr, pd_ = fr["prompts"][ti]["prompt"], fd["prompts"][ti]["prompt"]
                    ir, idd = tok().encode(pr).ids, tok().encode(pd_).ids
                    if len(ir) != len(idd):
                        continue
                    cls = _spans(rel, fr["subject"], pr, ir)
                    clsd = _spans(rel, fd["subject"], pd_, idd)
                    sub = [t for t, c_ in cls.items() if c_ == "subject"]
                    if sub != [t for t, c_ in clsd.items() if c_ == "subject"]:
                        continue
                    T = len(ir)
                    ad = pt.acts(idd)
                    ar = pt.acts(ir)
                    others = [first[v] for v in values if v != fd["answer"]]

                    def score(lp):
                        lo = float(lp[first[fd["answer"]]] - torch.logsumexp(lp[others], -1))
                        top = values[int(np.argmax([float(lp[first[v]]) for v in values]))]
                        return lo, top == fd["answer"]

                    def patch_of(sel):  # sel: list of (site, t, c)
                        by = {}
                        for n, t, c in sel:
                            by.setdefault(n, {}).setdefault(t, []).append(c)
                        return {n: [(t, torch.tensor(cs, device=nat.dev), ad[n][t, cs]) for t, cs in d_.items()] for n, d_ in by.items()}

                    act_sub = []
                    ci_r = nat.importance(ir)
                    ci_d = nat.importance(idd)
                    for n in nat.names:
                        on = ((ci_r[n] > 0) | (ci_d[n] > 0))
                        for t in sub:
                            act_sub += [(n, t, int(c)) for c in on[t].nonzero().flatten().tolist()]
                    nec = sorted({x for x in need.get(pr, []) + need.get(pd_, []) if x[1] in sub})
                    nec_final = sorted({x for x in need.get(pr, []) + need.get(pd_, []) if x[1] == T - 1})
                    rest = [x for x in act_sub if x not in set(nec)]
                    rand = [rest[i] for i in rng.permutation(len(rest))[:len(nec)]]
                    sets = {"none": [], "every subcomponent at the subject": [(n, t, c) for n in nat.names for t in sub for c in range(nat.target.site(n).V.shape[1])],
                            "active at the subject": act_sub, "necessary at the subject": nec, "as many random active at the subject": rand,
                            "necessary at the final token": nec_final}
                    for L in range(4):
                        sets[f"layer {L} at the subject"] = [x for x in act_sub if x[0].split(".")[1] == str(L)]
                    row = {"relation": rel, "template": ti, "recipient": fr["subject"], "donor": fd["subject"], "donor answer": fd["answer"],
                           "recipient answer": fr["answer"], "sizes": {k: len(v) for k, v in sets.items()}}
                    for k, sel in sets.items():
                        lo, flip = score(pt.run(ir, patch_of(sel)) if sel else pt.run(ir, {}))
                        row[k] = {"donor logodds": lo, "donor answer on top": flip}
                    out.append(row)
            print(rel, "template", ti, len([o for o in out if o["relation"] == rel and o["template"] == ti]), "pairs", flush=True)
    summ = {}
    for rel in ("capital", "city country"):
        rows = [o for o in out if o["relation"] == rel]
        if not rows:
            continue
        summ[rel] = {"pairs": len(rows)}
        for k in rows[0]:
            if isinstance(rows[0][k], dict) and "donor logodds" in rows[0][k]:
                summ[rel][k] = {"donor answer on top": float(np.mean([o[k]["donor answer on top"] for o in rows])),
                                "donor log-odds": float(np.mean([o[k]["donor logodds"] for o in rows])),
                                "subcomponents": float(np.mean([o["sizes"][k] for o in rows]))}
    (OUT / "swaps.json").write_text(json.dumps({"summary": summ, "pairs": out}, indent=1))
    for rel, v in summ.items():
        print(rel, v["pairs"], "pairs")
        for k, x in v.items():
            if k != "pairs":
                print(f"   {k:40s} donor answer on top {x['donor answer on top']:.2f}, donor log-odds {x['donor log-odds']:+.2f}, {x['subcomponents']:.0f} subcomponents")


def answer_parts():
    """Do cities with one answer (Tokyo, Osaka, Kyoto -> Japan) share a part tied to that answer? For each country with
    2+ known cities, the subcomponents necessary (removal >= NEED) for every one of its cities' facts and for at most
    20% of the other city facts. Tests: removing them together (at every position) on its cities' prompts and on other
    cities' prompts (the change of each prompt's own answer's log-odds); and setting their activations, at every
    position, to a prompt of one of its cities on a prompt of another country's city of the same template and length
    (the change of the country's log-odds there, and whether it becomes the likeliest answer), against as many random
    subcomponents active on the donor prompt. -> OUT/answer_parts.json"""
    import torch

    import native

    nat = native.Native()
    pt = _Patch(nat)
    R = [r for r in json.loads((OUT / "removals.json").read_text()) if r["relation"] == "city country"]
    values = sorted(set(CITY.values()))
    first = _first(values)
    need = {}
    for r in R:
        need.setdefault(r["subject"], []).append({(n, c) for n, t, c, e in r["removals"] if e >= NEED})
    fact_need = {s_: set.intersection(*v) if v else set() for s_, v in need.items()}
    by_ans = {}
    for r in R:
        by_ans.setdefault(r["answer"], set()).add(r["subject"])
    rng = np.random.default_rng(2951)
    out = {}

    def logodds(lp, ans):
        return float(lp[first[ans]] - torch.logsumexp(lp[[first[v] for v in values if v != ans]], -1))

    for ans, cities in sorted(by_ans.items()):
        if len(cities) < 2:
            continue
        others = [s_ for s_ in fact_need if s_ not in cities]
        common = set.intersection(*[fact_need[c] for c in cities])
        parts = sorted(k for k in common if sum(k in fact_need[o] for o in others) <= 0.2 * len(others))
        row = {"cities": sorted(cities), "parts": [_label(*k) for k in parts]}
        if not parts:
            out[ans] = row
            print(ans, row, flush=True)
            continue

        def remove(ids, sel):
            T = len(ids)
            masks = {n: torch.ones((1, T, nat.target.site(n).V.shape[1]), device=nat.dev) for n in nat.names}
            for n, c in sel:
                masks[n][0, :, c] = 0.0
            dm = {n: torch.ones((1, T), device=nat.dev) for n in nat.names}
            with torch.no_grad():
                return torch.log_softmax(nat.vpd.masked(torch.tensor([ids], device=nat.dev), masks, dm)[0, -1].float(), -1)

        own, oth = [], []
        for r in R:
            ids = tok().encode(r["prompt"]).ids
            full = pt.run(ids, {})
            d_ = logodds(full, r["answer"]) - logodds(remove(ids, parts), r["answer"])
            (own if r["subject"] in cities else oth).append(d_)
        row["removal: its cities' answer log-odds drop"] = float(np.mean(own))
        row["removal: other cities' answer log-odds drop"] = float(np.mean(oth))
        tr, rd = [], []
        for rd_ in [r for r in R if r["subject"] in cities]:
            ids_d = tok().encode(rd_["prompt"]).ids
            ad = pt.acts(ids_d)
            ci_d = nat.importance(ids_d)
            act = [(n, int(c)) for n in nat.names for c in (ci_d[n] > 0).any(0).nonzero().flatten().tolist()]
            for rr in [r for r in R if r["answer"] != ans and r["prompt"].replace(r["subject"], "") == rd_["prompt"].replace(rd_["subject"], "")]:
                ids_r = tok().encode(rr["prompt"]).ids
                if len(ids_r) != len(ids_d):
                    continue
                T = len(ids_r)
                base = logodds(pt.run(ids_r, {}), ans)

                def patch_of(sel):
                    by = {}
                    for n, c in sel:
                        by.setdefault(n, []).append(c)
                    return {n: [(t, torch.tensor(cs, device=nat.dev), ad[n][t, cs]) for t in range(T)] for n, cs in by.items()}

                lp = pt.run(ids_r, patch_of(parts))
                tr.append((logodds(lp, ans) - base, values[int(np.argmax([float(lp[first[v]]) for v in values]))] == ans))
                ctrl = [act[i] for i in rng.permutation(len(act))[:len(parts)]]
                lp = pt.run(ids_r, patch_of(ctrl))
                rd.append((logodds(lp, ans) - base, values[int(np.argmax([float(lp[first[v]]) for v in values]))] == ans))
        if tr:
            row["transplant: rise of its log-odds"] = float(np.mean([x[0] for x in tr]))
            row["transplant: becomes the likeliest answer"] = float(np.mean([x[1] for x in tr]))
            row["random transplant: rise"] = float(np.mean([x[0] for x in rd]))
            row["random transplant: likeliest"] = float(np.mean([x[1] for x in rd]))
            row["transplants"] = len(tr)
        out[ans] = row
        print(ans, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    (OUT / "answer_parts.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "answer_parts.json")


def link():
    """Do the association study's subcomponents for places pick out the parts the facts need? Per association phenotype
    about places (named entity type GPE and LOC, and the current token being one of the facts' countries or cities
    among the 200 tested tokens), the share of the facts' necessary parts (removal >= NEED for some fact) with a strong
    association (calibrated q < 0.05 and |partial r| >= 0.1), against the share among every subcomponent the scan
    tested; and the necessary parts' own strongest association families. -> OUT/link.json"""
    G = Path.home() / "mpd-data/graph_oracle/gwas"
    A = np.load(G / "assoc.npz")
    names = json.loads(str(A["names"]))
    fams = json.loads(str(A["families"]))
    off = json.loads(str(A["offsets"]))
    r, q = A["r"], A["q_cal"] if "q_cal" in A.files else A["q"]
    strong = (q < 0.05) & (np.abs(r) >= 0.1)
    tested = np.isfinite(A["z"]).any(1)
    M = np.load(OUT / "matrix.npz")
    cols = [tuple(x) for x in json.loads(str(M["cols"]))]
    nec = np.array([off[n] + int(c) for n, c in cols])
    places = set(CAPITAL) | set(CAPITAL.values()) | set(CITY) | set(LANGUAGE)
    phen = [j for j, nm in enumerate(names) if nm in ("entity GPE", "entity LOC", "entity NORP")
            or (nm.startswith("current token ") and nm[len("current token "):].strip("'\" ") in places)]
    out = {"necessary parts": len(nec), "tested subcomponents": int(tested.sum()), "phenotypes": {}}
    for j in phen:
        a_ = float(strong[nec, j].mean())
        b_ = float(strong[tested, j].mean())
        out["phenotypes"][names[j]] = {"necessary with a strong association": a_, "all tested": b_}
    anyp = strong[:, phen].any(1)
    out["any place phenotype"] = {"necessary": float(anyp[nec].mean()), "all tested": float(anyp[tested].mean())}
    famc = {}
    for c in nec:
        hit = np.flatnonzero(strong[c])
        if len(hit):
            f = fams[hit[np.argmax(np.abs(r[c, hit]))]]
            famc[f] = famc.get(f, 0) + 1
    out["necessary parts' strongest family"] = dict(sorted(famc.items(), key=lambda kv: -kv[1]))
    out["necessary parts with no strong association"] = int(sum(1 for c in nec if not strong[c].any()))
    (OUT / "link.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["any place phenotype"]), "| no strong association:", out["necessary parts with no strong association"], "of", len(nec))
    for k, v in out["phenotypes"].items():
        print(f"  {k}: necessary {v['necessary with a strong association']:.3f}, all tested {v['all tested']:.3f}")
    print("  strongest families:", list(out["necessary parts' strongest family"].items())[:8])


FIG = Path("/Users/user/mpd-data/figures/oracle")


def figure_matrix():
    """The fact x subcomponent necessity matrix (matrix.npz): rows the known facts grouped by relation, columns the
    subcomponents necessary for some fact, grouped: needed by nearly every fact (>= 80%), needed by one relation's facts
    (>= 80% of them, <= 20% of the others), needed by one answer's facts (answer_parts.json), the rest; colour the
    log-odds a removal costs. -> FIG/knowledge_matrix.png"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(HERE))
    import figure as fg

    Mz = np.load(OUT / "matrix.npz")
    M = Mz["M"]
    facts = [tuple(f) for f in json.loads(str(Mz["facts"]))]
    cols = [tuple(c) for c in json.loads(str(Mz["cols"]))]
    order_rel = ["capital", "language", "city country"]
    rows = sorted(range(len(facts)), key=lambda i: (order_rel.index(facts[i][0]), facts[i][2], facts[i][1]))
    need = M >= NEED
    rel_of = np.array([facts[i][0] for i in range(len(facts))])
    allf = need.mean(0)
    groups, used = [], set()
    g = [j for j in range(len(cols)) if allf[j] >= 0.8]
    groups.append(("needed by nearly every fact", g))
    used |= set(g)
    for rel in order_rel:
        inr, outr = need[rel_of == rel].mean(0), need[rel_of != rel].mean(0)
        g = [j for j in range(len(cols)) if j not in used and inr[j] >= 0.8 and outr[j] <= 0.2]
        groups.append((f"{'capital' if rel == 'capital' else 'city' if rel == 'city country' else 'language'} facts", g))
        used |= set(g)
    AP = json.loads((OUT / "answer_parts.json").read_text()) if (OUT / "answer_parts.json").exists() else {}
    lab = {_label(*c): j for j, c in enumerate(cols)}
    for ans, row in AP.items():
        g = [lab[x] for x in row.get("parts", []) if x in lab and lab[x] not in used]
        if g:
            groups.append((ans, g))
            used |= set(g)
    rest = [j for j in range(len(cols)) if j not in used]
    rest.sort(key=lambda j: -need[:, j].sum())
    groups.append(("the rest", rest))
    named = [(nm, g) for nm, g in groups if nm != "the rest" and g]
    left = [j for _, g in named for j in g]
    fg.setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 13), gridspec_kw={"width_ratios": [max(len(left), 1), 0.35 * len(rest)], "wspace": 0.02})
    data = np.clip(M[rows], 0, 3)
    im = a1.imshow(data[:, left], aspect="auto", cmap="Blues", vmin=0, vmax=3, interpolation="nearest")
    a2.imshow(data[:, rest], aspect="auto", cmap="Blues", vmin=0, vmax=3, interpolation="nearest")
    a1.set_yticks(range(len(rows)), [f"{facts[i][1]} \u2192 {facts[i][2]}" for i in rows], fontsize=11)
    a1.set_xticks(range(len(left)), [_label(*cols[j]) for j in left], rotation=90, fontsize=9)
    a2.set_yticks([])
    a2.set_xticks([])
    a2.set_xlabel(f"the other {len(rest)} subcomponents (each needed\nby a few facts), most-needed first", fontsize=13)
    x0 = 0
    for nm, g in named:
        if x0:
            a1.axvline(x0 - 0.5, color=fg.INK, lw=1.0)
        a1.text(x0 + len(g) / 2 - 0.5, -0.9, nm, ha="left", va="bottom", fontsize=12, rotation=40)
        x0 += len(g)
    y0 = 0
    for rel in order_rel:
        n_ = sum(1 for i in rows if facts[i][0] == rel)
        if y0:
            a1.axhline(y0 - 0.5, color=fg.INK, lw=1.4)
            a2.axhline(y0 - 0.5, color=fg.INK, lw=1.4)
        y0 += n_
    cb = fig.colorbar(im, ax=[a1, a2], fraction=0.015, pad=0.03)
    cb.set_label("drop of the right answer's log-odds\nwhen the subcomponent is removed (nats)")
    fig.suptitle("How much each known fact depends on each subcomponent", fontsize=20, fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(left=0.12, right=0.84, top=0.84, bottom=0.12)
    out = FIG / "knowledge_matrix.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


LEAD = Path("/private/tmp/claude-501/-Users-user/b1063ff3-6750-418e-8406-23ae82bf968a/scratchpad/lead")  # the lead's renderer
CIRCUIT_PROMPT = ("capital", "Japan", "The capital city of Japan is called")


def circuit():
    """One fact's stated circuit: the v2.3 teacher (mechdisc's flags) on CIRCUIT_PROMPT as a task file, the lead's
    edge_prune.py (connections, each part's and connection's removal effect), the lead's render_circuit.py ->
    FIG/knowledge_circuit_<subject>.png"""
    import runpy
    import subprocess

    import mechdisc
    import native

    rel, subj, ptxt = CIRCUIT_PROMPT
    ids = tok().encode(ptxt).ids
    name = f"kn_{rel.replace(' ', '_')}_{subj.lower()}"
    root = OUT / "texts_root" / "vpd4l"
    root.mkdir(parents=True, exist_ok=True)
    task = {"id": name, "model": "vpd4l", "family": "knowledge", "split": "knowledge", "description": ptxt,
            "prompts": [{"token_ids": ids, "target_positions": [len(ids) - 1]}]}
    (root / f"{name}.json").write_text(json.dumps(task))
    real = native.TEXTS / "vpd4l"  # every other text linked in (the rewards' donor pools), never the sealed ones
    for f in os.listdir(real):
        if f.endswith(".json") and not (root / f).exists() and not mechdisc.SEALED.fullmatch(f):
            (root / f).symlink_to(real / f)
    tdir = OUT / "teacher"
    if not (tdir / f"{name}.json").exists():
        native.TEXTS = root.parent
        lst = OUT / f"ids_{name}.txt"
        lst.write_text(name + "\n")
        argv = sys.argv
        sys.argv = ["teacher.py", "--tasks", str(lst), *mechdisc.TEACHER_FLAGS, "--out", str(tdir)]
        try:
            runpy.run_path(str(ORACLE / "teacher.py"), run_name="__main__")
        finally:
            sys.argv = argv
    stated = OUT / f"{name}.stated.json"
    py = sys.executable
    subprocess.run([py, "-B", str(LEAD / "edge_prune.py"), str(root / f"{name}.json"), str(tdir / f"{name}.json"), str(stated)], check=True)
    fig = Path("/Users/user/mpd-data/figures/oracle") / f"knowledge_circuit_{subj.lower()}"
    subprocess.run([py, "-B", str(LEAD / "render_circuit.py"), str(stated), str(fig)], check=True)
    print("wrote", fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("known", "colors_natural", "removals", "removals_default", "fallback", "where", "swaps",
                                    "answer_parts", "link", "circuit", "figure_matrix"))
    a = ap.parse_args()
    if a.cmd == "removals_default":
        removals(default=True)
    else:
        globals()[a.cmd]()
