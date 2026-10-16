"""Circuits over prompt families (the lead's 10-16 decision: the family circuit is the record the oracle and the graphs
read): a question asked in its paraphrases, each against VARIANTS counterfactual key words (its own corrupted key first,
then its next swap partners), one circuit_finder circuit per pair; the family keeps the units that do the job on the
pairs, every choice measured on every pair, so near-ties between units resolve the same way for every paraphrase.
  key        a unit without its position: a neuron (layer, index) or a head (layer, head), with its role in the prompt:
             a place in the family's common tail (the chat template's end and any prefill, shared by every pair's prompt)
             counted back from the answer (("end", k)), else a place counted from the changed word (("word", k): the key
             word's own tokens and the user text after them); a key maps into any pair of the family by its role (a place
             outside that pair's changed span to answer is left out there)
  candidates per stage, every key a pair's finder chose, and every key a pair measured alone at least as strong as one
             it chose and above twice its cut's noise floor (the near-ties its prefix and prune could have taken instead)
  weight     a key's marginal share in two contexts, measured on every pair it maps into: alone, and taken out of all the
             candidates patched together, both directions, the mean of the four over the pairs (0 where it does not map).
             Effect alone is not enough: on Qwen3-8B the source cuts' single noising drops add up to a median 6% of the
             cut's joint drop (569 pairs of 43 dev families), so a weight by effect alone is mostly bfloat16 noise; the
             share out of the whole set counts units that matter only together
  cut        per stage, the candidates in descending weight (ties by key); the curve of the pairs passing PASS both ways
             (and the median lost and restored) as keys are added, each pair patched with the keys mapped into it, on
             circuit_finder's grid; the smallest prefix reaching the curve's most pairs passing, then keys removed from
             the lowest weight up while as many pairs pass (one order for every pair: of two interchangeable units the
             one of larger family weight stays); a stage whose curve never passes a majority of the pairs is diffuse (no
             units; its size the pairs' median cut size, its curve kept)
  alternatives  per kept key, the candidates of at least the smallest kept weight that replace it with as many pairs
             passing (interchangeable units: one chosen, the others recorded)
  loo        per kept key, the mean over the pairs of what the cut loses without it (the larger of the two directions):
             each unit's part of the circuit as a whole, the weight stability overlaps are taken in
  record     the canonical pair's finder record (the first paraphrase against its own corrupted key) with its cuts, nodes
             and routes replaced by the family's at that pair's positions (each unit's shares alone measured there, with
             its family weight, loo and alternatives), "order": the stages narrowest first (diffuse ones last), "met": the
             source and mover cuts not diffuse, and "family": the member pairs with their own cuts and the family cut's
             shares on each, so every reader of finder records reads a family record as is.

  circuit_family.py QUESTIONS.json OUT [--split dev] [--stride 1] [--variants 3] [--model Qwen/Qwen3-8B] [--dtype auto]
      [--prior]   (each new pair's finder first tries the units the family's earlier pairs kept: prior_of)
writes OUT/pairs/<pair>.json (finder records; existing ones are read, not rebuilt) and OUT/<family>.json;
  circuit_family.py RECORDS_DIR OUT --records NAME [--adapter A]    (an organism's finder records as one family)
"""
from __future__ import annotations

import collections
import statistics
import time

import torch

import circuit_finder as cf

VARIANTS = 3


def tail_of(recs: list[dict]) -> int:
    """The family's common tail: the most final tokens every pair's clean prompt shares (template end and prefill)."""
    n = min(len(r["ids"]) for r in recs)
    k = 0
    while k < n and len({r["ids"][-1 - k] for r in recs}) == 1:
        k += 1
    return k


def role(rec: dict, pos: int, tail: int) -> tuple:
    T = rec["tokens"]
    return ("end", T - 1 - pos) if pos >= T - tail else ("word", pos - rec["first_diff"])


def unit_of(key: tuple, rec: dict):
    """A key's unit in a pair's record (circuit_finder's unit), None when its place lies outside the pair's changed
    span to answer."""
    if key[0] == "head":
        return ("head", key[1], key[2], rec["tokens"] - 1)
    _, layer, n, (where, k) = key
    p = rec["first_diff"] + k if where == "word" else rec["tokens"] - 1 - k
    return ("neuron", layer, p, n) if rec["first_diff"] <= p < rec["tokens"] else None


def key_of(stage: str, u: dict, rec: dict, tail: int) -> tuple:
    """A record's node, route or single as a key."""
    return ("head", u["layer"], u["head"]) if stage == "movers" else ("neuron", u["layer"], u["neuron"], role(rec, u["pos"], tail))


def candidates(recs: list[dict], tail: int, pool: str = "near") -> dict:
    """Per stage, the candidate keys (module docstring), first seen first; pool "all": every single a pair measured above
    twice its cut's noise floor, near-tie or not."""
    out = {}
    for r in recs:
        for stage, c in r.get("cuts", {}).items():
            chosen = r["routes"] if stage == "movers" else [n for n in r["nodes"] if n["stage"] == stage]
            if not chosen:
                continue
            ks = out.setdefault(stage, {})
            for u in chosen:
                ks[key_of(stage, u, r, tail)] = None
            low = max(min(max(abs(u["lost"]), abs(u["restored"])) for u in chosen) if pool == "near" else 0.0,
                      2 * ((c.get("floor") or [0.0])[0]))
            for u in r.get("singles", {}).get(stage, []):
                if max(abs(u["lost"]), abs(u["restored"])) >= low:
                    ks[key_of(stage, u, r, tail)] = None
    return {s: list(ks) for s, ks in out.items()}


def prior_of(recs: list[dict], ids: list[int], corrupted: list[int]) -> dict | None:
    """{stage: units} for circuit_finder.find's prior on a new pair of the family: the units earlier pairs' cuts kept
    (keys by role, the most often kept first, ties first seen first), mapped into the new pair (None before any)."""
    rs = [r for r in recs if r.get("cuts")]
    if not rs:
        return None
    new = {"ids": ids, "tokens": len(ids), "first_diff": next(p for p in range(len(ids)) if ids[p] != corrupted[p])}
    tail, out = tail_of(rs + [new]), {}
    for stage in ("readout", "movers", "source"):
        n = collections.Counter()
        for r in rs:
            for u in (r["routes"] if stage == "movers" else [x for x in r["nodes"] if x["stage"] == stage]):
                n[key_of(stage, u, r, tail)] += 1
        out[stage] = [u for u in (unit_of(k, new) for k, _ in n.most_common()) if u]
    return out


def patchers(pt: cf.Patcher, recs: list[dict]) -> list:
    """One Patcher per pair record on pt's model, each set up on its pair (its runs and contrast as its finder run's)."""
    out = []
    for r in recs:
        p = cf.Patcher(pt.model, pt.tok, rows=pt.rows)
        p.setup(r["ids"], r["corrupted"], r.get("answers"))
        out.append(p)
    return out


def measure(pts: list, sets: list) -> list:
    """Per pair j, (lost, restored) of each of sets[j] patched on pts[j]; every pair's batches launched before any is read."""
    return [[tuple(v) for v in g] for g in cf._read([p._launch(s, ("clean", "corrupt")) for p, s in zip(pts, sets)])]


def passes(x) -> bool:
    return x[0] >= cf.PASS and x[1] >= cf.PASS


def mapped(keys, rec: dict) -> list:
    return [u for u in (unit_of(k, rec) for k in keys) if u]


def counts(pts: list, recs: list[dict], key_sets: list) -> tuple:
    """Per key set: the number of pairs passing, and its (lost, restored) on each pair."""
    got = measure(pts, [[mapped(s, r) for s in key_sets] for r in recs])
    return [sum(passes(g[i]) for g in got) for i in range(len(key_sets))], [[g[i] for g in got] for i in range(len(key_sets))]


def weigh(pts: list, recs: list[dict], keys: list) -> dict:
    """Each key's weight (module docstring): its share alone and its drop out of every candidate, both directions, the
    mean over the pairs."""
    sets, where = [], []
    for r in recs:
        us = [unit_of(k, r) for k in keys]
        have = [u for u in us if u]
        idx = [i for i, u in enumerate(us) if u]
        sets.append([have] + [[u] for u in have] + [[v for v in have if v != u] for u in have])
        where.append(idx)
    got = measure(pts, sets)
    w = [0.0] * len(keys)
    for g, idx in zip(got, where):
        n = len(idx)
        for j, i in enumerate(idx):
            one, out = g[1 + j], g[1 + n + j]
            w[i] += (one[0] + one[1] + g[0][0] - out[0] + g[0][1] - out[1]) / 4
    return {k: x / len(recs) for k, x in zip(keys, w)}


def select(pts: list, recs: list[dict], keys: list, log=None, name: str = "", prune: bool = True) -> dict:
    """A stage's family cut from its candidate keys (module docstring: weight, cut, alternatives, loo); prune False: the
    smallest prefix reaching the most pairs passing, as it is."""
    N = len(recs)
    W = weigh(pts, recs, keys)
    ranked = sorted(keys, key=lambda k: (-W[k], str(k)))
    g = cf.grid(len(ranked))
    n, sh = counts(pts, recs, [ranked[:k] for k in g])
    curve = [{"k": k, "lost": statistics.median(x[0] for x in s), "restored": statistics.median(x[1] for x in s),
              "pairs_passing": c / N} for k, c, s in zip(g, n, sh)]
    best = max(n)
    out = {"candidates": len(keys), "curve": curve, "weights": W}
    if 2 * best <= N:
        out.update(keys=[], diffuse=True, alternatives={}, loo={})
        return out
    S = ranked[:g[n.index(best)]]
    if prune and len(S) > 1:                        # from the lowest weight up, each removal kept while as many pass
        alone, _ = counts(pts, recs, [[x for x in S if x != k] for k in S])
        for k in [k for k, c in zip(S, alone) if c >= best][::-1]:
            trial = [x for x in S if x != k]
            if counts(pts, recs, [trial])[0][0] >= best:
                S = trial
    rest = [k for k in ranked if k not in S and W[k] >= min(W[x] for x in S)]
    trials = [(u, v) for u in S for v in rest]
    c, _ = counts(pts, recs, [[v if x == u else x for x in S] for u, v in trials]) if trials else ([], [])
    alts = {u: [v for (u_, v), cc in zip(trials, c) if u_ == u and cc >= best] for u in S}
    n_, sh_ = counts(pts, recs, [S] + [[x for x in S if x != k] for k in S])
    whole = sh_[0]
    loo = {k: statistics.mean(max(a[0] - b[0], a[1] - b[1]) for a, b in zip(whole, sh_[1 + i])) for i, k in enumerate(S)}
    out.update(keys=S, diffuse=False, alternatives=alts, loo=loo, pairs=whole, pairs_passing=n_[0] / N,
               lost=statistics.median(x[0] for x in whole), restored=statistics.median(x[1] for x in whole))
    if log:
        log(f"  family {name}: {len(S)} of {len(keys)} candidates, pairs passing {n_[0]}/{N} (prefix {g[n.index(best)]}), median "
            f"{out['lost']:.2f}/{out['restored']:.2f}, {sum(map(len, alts.values()))} alternatives")
    return out


def cuts_of(pt: cf.Patcher, recs: list[dict], pts: list | None = None, log=None) -> tuple:
    """Every stage's family cut over pair records (select), and the pairs' Patchers."""
    pts = pts or patchers(pt, recs)
    cand = candidates(recs, tail_of(recs))
    return {s: select(pts, recs, ks, log=log, name=s) for s, ks in cand.items()}, pts


def family(pt: cf.Patcher, recs: list[dict], canonical: dict, log=None) -> dict:
    """The family circuit of pair records (circuit_finder.find's, each with ids, corrupted and an id) and its record on
    the canonical pair (module docstring)."""
    recs = [r for r in recs if r.get("cuts")]
    sel, pts = cuts_of(pt, recs, log=log)
    cuts = {}
    for s, c in sel.items():
        sizes = [r["cuts"][s]["size"] for r in recs if s in r["cuts"]]
        cuts[s] = {"keys": c["keys"], "diffuse": c["diffuse"], "size": len(c["keys"]) if not c["diffuse"] else statistics.median(sizes),
                   "lost": c.get("lost", c["curve"][-1]["lost"]), "restored": c.get("restored", c["curve"][-1]["restored"]),
                   "pairs_passing": c.get("pairs_passing", c["curve"][-1]["pairs_passing"]), "curve": c["curve"],
                   "candidates": c["candidates"], "pair_sizes": sizes}
        if log and c["diffuse"]:
            log(f"  family {s}: diffuse, median pair size {cuts[s]['size']}, best pairs passing "
                f"{max(x['pairs_passing'] for x in c['curve']):.2f}")
    rec = {k: v for k, v in canonical.items() if k not in ("nodes", "routes", "cuts", "singles", "size", "met")}
    cp = next((p for p, r in zip(pts, recs) if r is canonical), None)
    if cp is None:
        cp = cf.Patcher(pt.model, pt.tok, rows=pt.rows)
        cp.setup(canonical["ids"], canonical["corrupted"], canonical.get("answers"))
    nodes, routes, units = [], [], {}
    for s, c in cuts.items():
        units[s] = [(k, u) for k, u in ((k, unit_of(k, canonical)) for k in c["keys"]) if u]
    alone = cp.both([[u] for s in units for _, u in units[s]])
    it = iter(alone)
    for s in units:
        W, alts, loo = sel[s]["weights"], sel[s]["alternatives"], sel[s]["loo"]
        for k, u in units[s]:
            lo, re_ = next(it)
            extra = {"weight": W[k], "loo": loo[k], "alternatives": [list(v) for v in alts[k]]}
            if u[0] == "head":
                routes.append({"layer": u[1], "head": u[2], "pos": u[3], "attends": cp.attends(u[1], u[2], u[3]), "lost": lo,
                               "restored": re_, **extra})
                continue
            _, i, p, n = u
            v, w = float(cp.act["clean"]["neu"][i][p, n]), float(cp.act["clean"]["up"][i][p, n])
            nodes.append({"stage": s, "layer": i, "pos": p, "neuron": n, "pole": 1 if v > 0 else -1,
                          "regime": [1 if v * w > 0 else -1, 1 if w > 0 else -1], "clean": v,
                          "corrupt": float(cp.act["corrupt"]["neu"][i][p, n]), "lost": lo, "restored": re_, **extra})
    whole = cp.both([[u for _, u in units[s]] for s in units])
    for (s, c), (lo, re_) in zip(cuts.items(), whole):
        c["canonical"] = {"lost": lo, "restored": re_}
        c["met"] = not c["diffuse"]
        c["keys"] = [list(k) for k in c["keys"]]
    rec.update(nodes=nodes, routes=routes, size=len(nodes), cuts=cuts, tail=tail_of(recs),
               order=sorted(cuts, key=lambda s: (cuts[s]["diffuse"], cuts[s]["size"])),
               met=all(c["met"] for k, c in cuts.items() if k != "readout"),     # (a diffuse readout as the finder's
                                                                                   # unreachable one: reported, not required)
               family={"pairs": [{"id": r["id"], "met": r.get("met"),
                                  "cuts": {s: {f: c[f] for f in ("size", "lost", "restored", "met")} for s, c in r["cuts"].items()},
                                  "family": {s: list(sel[s]["pairs"][j]) for s in sel if not sel[s]["diffuse"]}}
                                 for j, r in enumerate(recs)]})
    return rec


def pairs_of(questions: list[dict], variants: int = VARIANTS, skip: int = 0) -> list[tuple]:
    """(pair id, user, corrupt user) for each paraphrase of a family against its corrupted keys skip..skip+variants (its
    own corrupted key, then its swap partners; a key that occurs more than once in a prompt is skipped)."""
    out = []
    for q in questions:
        users = {}                                  # partners: keys (questions.json) or {key, user} (circuit_devset's sets)
        for p in q.get("partners", []):
            if isinstance(p, dict):
                users[p["key"]] = p["user"]
            elif q["user"].count(q["key"]) == 1:
                users[p] = q["user"].replace(q["key"], p)
        users.setdefault(q["corrupt_key"], q.get("corrupt_user") or q["user"].replace(q["key"], q["corrupt_key"]))
        keys = [q["corrupt_key"]] + [k for k in users if k != q["corrupt_key"]]
        for ck in keys[skip:skip + variants]:
            out.append((f"{q['id']}/vs/{ck}", q["user"], users[ck]))
    return out


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("questions", type=Path, help="the questions JSON, or with --records a directory of finder records")
    ap.add_argument("out", type=Path)
    ap.add_argument("--records", default=None, help="NAME: the finder records in QUESTIONS (a directory) are one family's "
                    "pairs (an organism's cases), written to OUT/NAME.json; no new pairs")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--stride", type=int, default=1, help="every stride-th family")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--variants", type=int, default=VARIANTS)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    ap.add_argument("--rows", type=int, default=cf.ROWS)
    ap.add_argument("--prior", action="store_true", help="each new pair's finder tries the units earlier pairs of its "
                    "family kept first (prior_of; its cuts' \"prior\" field says where they passed)")
    ap.add_argument("--rev", default=None)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.dtype == "auto":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(a.model)
        n = cfg.num_hidden_layers * (4 * cfg.hidden_size ** 2 + 3 * cfg.hidden_size * cfg.intermediate_size)
        a.dtype = "bfloat16" if n >= 4e9 else "float32"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto").eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok, rows=a.rows)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    if a.records:
        recs = [json.loads(f.read_text()) for f in sorted(a.questions.glob("*.json"))]
        recs = [r for r in recs if r.get("cuts")]
        for r in recs:                              # (the answer word's tokens: a secret word's run as its finder run's)
            r.setdefault("answers", r["answer_tokens"])
        a.out.mkdir(parents=True, exist_ok=True)
        log(f"family {a.records}: {len(recs)} pairs")
        fam = family(pt, recs, recs[0], log=log)
        fam.update(id=a.records, family_rev=a.rev, answer_text=tok.decode([fam["answer"]]))
        fam["family"]["id"] = a.records
        (a.out / f"{a.records}.json").write_text(json.dumps(fam))
        return
    qs = [q for q in json.loads(a.questions.read_text())["questions"] if q["split"] == a.split]
    fams = {}
    for q in qs:
        fams.setdefault(f"{q['task']}/{q['key']}", []).append(q)
    names = sorted(fams)[a.offset::a.stride]
    (a.out / "pairs").mkdir(parents=True, exist_ok=True)
    for name in names:
        path = a.out / f"{name.replace('/', '__')}.json"
        if path.exists():
            continue
        log(f"family {name}")
        recs = []
        for pid, user, cuser in pairs_of(sorted(fams[name], key=lambda q: q["paraphrase"]), a.variants):
            pp = a.out / "pairs" / f"{pid.replace('/', '__')}.json"
            if pp.exists():
                recs.append(json.loads(pp.read_text()))
                continue
            clean, corrupt = cf.chat(tok, user), cf.chat(tok, cuser)
            if len(clean) != len(corrupt):
                log(f"  {pid}: prompts of {len(clean)} and {len(corrupt)} tokens, skipped")
                continue
            log(f"  {pid}")
            rec = cf.find(pt, clean, corrupt, log=log, prior=prior_of(recs, clean, corrupt) if a.prior else None)
            rec.update(id=pid, ids=clean, corrupted=corrupt, user=user, corrupt_user=cuser, model=a.model, adapter=a.adapter,
                       dtype=a.dtype, finder="circuit_finder", finder_rev=a.rev)
            pp.write_text(json.dumps(rec))
            recs.append(rec)
        good = [r for r in recs if r.get("cuts")]
        if not good:
            log("  no valid pair")
            continue
        fam = family(pt, good, good[0], log=log)
        fam.update(id=name, family_rev=a.rev, answer_text=tok.decode([fam["answer"]]))
        fam["family"]["id"] = name
        path.write_text(json.dumps(fam))


if __name__ == "__main__":
    main()
