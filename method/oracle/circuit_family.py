"""Circuits over prompt families (the lead's 10-16 decision: the family circuit is the record the oracle and the graphs
read): a question asked in its paraphrases, each against VARIANTS counterfactual key words (its own corrupted key first,
then its next swap partners), one circuit_finder circuit per pair; the family keeps the units that recur, weighted by
their measured effect, and measures each cut's family size on every pair.
  key      a unit without its position: a neuron (layer, index) or a head (layer, head), with its role in the prompt:
           the changed word ("word") or a place counted back from the answer (("end", k)); a key maps into any pair of
           the family by its role (a neuron whose place lies before that pair's changed word is left out there)
  weight   per pair, a cut's units share 1 in proportion to their measured effect alone (the larger |share|; equally
           when every one is near 0: units that matter only together); a key's weight is its sum over the pairs, so a
           unit of one prompt weighs at most 1 against a unit of every prompt's pairs
  cut      per stage, keys in descending weight; the curve of the median lost and restored over the pairs as keys are
           added (each pair patched with the keys mapped into it) on circuit_finder's grid; the family size is the
           smallest grid prefix whose median pair passes PASS both ways, with the share of pairs passing there; a stage
           with no passing prefix is diffuse (no units; its size the pairs' median cut size, its curve kept)
  record   the canonical pair's finder record (the first paraphrase against its own corrupted key) with its cuts, nodes
           and routes replaced by the family's at that pair's positions (each unit's shares alone measured there, with
           its family weight), "order": the stages narrowest first (diffuse ones last), "met": the source and mover
           cuts not diffuse, and "family": the member pairs with their own cuts, so every reader of finder records
           reads a family record as is.

  circuit_family.py QUESTIONS.json OUT [--split dev] [--stride 1] [--variants 3] [--model Qwen/Qwen3-8B] [--dtype auto]
writes OUT/pairs/<pair>.json (finder records; existing ones are read, not rebuilt) and OUT/<family>.json;
  circuit_family.py RECORDS_DIR OUT --records NAME [--adapter A]    (an organism's finder records as one family)
"""
from __future__ import annotations

import statistics
import time

import torch

import circuit_finder as cf

VARIANTS = 3


def role(rec: dict, pos: int):
    return "word" if pos == rec["first_diff"] else ("end", rec["tokens"] - 1 - pos)


def unit_of(key: tuple, rec: dict):
    """A key's unit in a pair's record (circuit_finder's unit), None when its place lies before the pair's change."""
    if key[0] == "head":
        return ("head", key[1], key[2], rec["tokens"] - 1)
    _, layer, n, r = key
    p = rec["first_diff"] if r == "word" else rec["tokens"] - 1 - r[1]
    return ("neuron", layer, p, n) if rec["first_diff"] <= p < rec["tokens"] else None


def keys_of(rec: dict) -> dict:
    """Per stage of a pair's record, its units' keys and their shares of 1 by measured effect alone."""
    out = {}
    for stage in rec.get("cuts", {}):
        units = ([(("head", h["layer"], h["head"]), h) for h in rec["routes"]] if stage == "movers" else
                 [(("neuron", n["layer"], n["neuron"], role(rec, n["pos"])), n) for n in rec["nodes"] if n["stage"] == stage])
        if not units:
            continue
        e = [max(abs(u["lost"]), abs(u["restored"])) for _, u in units]
        tot = sum(e)
        out[stage] = {}
        for (k, _), x in zip(units, e):
            out[stage][k] = out[stage].get(k, 0.0) + (x / tot if tot > 1e-9 else 1 / len(units))
    return out


def weights(recs: list[dict]) -> dict:
    """Per stage, each key's weight: its effect shares summed over the pairs."""
    out = {}
    for r in recs:
        for stage, ks in keys_of(r).items():
            ws = out.setdefault(stage, {})
            for k, w in ks.items():
                ws[k] = ws.get(k, 0.0) + w
    return out


def family(pt: cf.Patcher, recs: list[dict], canonical: dict, log=None) -> dict:
    """The family circuit of pair records (circuit_finder.find's, each with ids, corrupted and an id) and its record on
    the canonical pair (module docstring)."""
    recs = [r for r in recs if r.get("cuts")]
    W = weights(recs)
    ranked = {s: sorted(ws, key=lambda k: -ws[k]) for s, ws in W.items()}
    grids = {s: cf.grid(len(ks)) for s, ks in ranked.items()}
    got = {s: [] for s in ranked}                       # per stage: per pair, (lost, restored) per grid size
    for r in recs:
        pt.setup(r["ids"], r["corrupted"], r.get("answers"))
        order = [(s, k) for s in ranked for k in grids[s]]
        sh = pt.both([[u for u in (unit_of(x, r) for x in ranked[s][:k]) if u] for s, k in order])
        for s in ranked:
            got[s].append([v for (t, _), v in zip(order, sh) if t == s])
    cuts = {}
    for s, ks in ranked.items():
        g, per = grids[s], got[s]
        curve = [{"k": k, "lost": statistics.median(p[j][0] for p in per), "restored": statistics.median(p[j][1] for p in per),
                  "pairs_passing": sum(p[j][0] >= cf.PASS and p[j][1] >= cf.PASS for p in per) / len(per)} for j, k in enumerate(g)]
        ok = next((c for c in curve if c["lost"] >= cf.PASS and c["restored"] >= cf.PASS), None)
        sizes = [r["cuts"][s]["size"] for r in recs if s in r["cuts"]]
        cuts[s] = {"keys": ks[:ok["k"]] if ok else [], "diffuse": ok is None,
                   "size": ok["k"] if ok else statistics.median(sizes), "lost": ok["lost"] if ok else curve[-1]["lost"],
                   "restored": ok["restored"] if ok else curve[-1]["restored"],
                   "pairs_passing": ok["pairs_passing"] if ok else curve[-1]["pairs_passing"], "curve": curve,
                   "pair_sizes": sizes}
        if log:
            log(f"  family {s}: {'diffuse, median pair size ' if ok is None else ''}{cuts[s]['size']} of {len(ks)} keys, "
                f"median {cuts[s]['lost']:.2f}/{cuts[s]['restored']:.2f}, pairs passing {cuts[s]['pairs_passing']:.2f}")
    rec = {k: v for k, v in canonical.items() if k not in ("nodes", "routes", "cuts", "singles", "size", "met")}
    pt.setup(canonical["ids"], canonical["corrupted"], canonical.get("answers"))
    nodes, routes, units = [], [], {}
    for s, c in cuts.items():
        units[s] = [(k, u) for k, u in ((k, unit_of(k, canonical)) for k in c["keys"]) if u]
    alone = pt.both([[u] for s in units for _, u in units[s]])
    it = iter(alone)
    for s in units:
        for k, u in units[s]:
            lo, re_ = next(it)
            if u[0] == "head":
                routes.append({"layer": u[1], "head": u[2], "pos": u[3], "attends": pt.attends(u[1], u[2], u[3]), "lost": lo,
                               "restored": re_, "weight": W[s][k]})
                continue
            _, i, p, n = u
            v, w = float(pt.act["clean"]["neu"][i][p, n]), float(pt.act["clean"]["up"][i][p, n])
            nodes.append({"stage": s, "layer": i, "pos": p, "neuron": n, "pole": 1 if v > 0 else -1,
                          "regime": [1 if v * w > 0 else -1, 1 if w > 0 else -1], "clean": v,
                          "corrupt": float(pt.act["corrupt"]["neu"][i][p, n]), "lost": lo, "restored": re_, "weight": W[s][k]})
    whole = pt.both([[u for _, u in units[s]] for s in units])
    for (s, c), (lo, re_) in zip(cuts.items(), whole):
        c["canonical"] = {"lost": lo, "restored": re_}
        c["met"] = not c["diffuse"]
        c["keys"] = [list(k) for k in c["keys"]]
    rec.update(nodes=nodes, routes=routes, size=len(nodes), cuts=cuts,
               order=sorted(cuts, key=lambda s: (cuts[s]["diffuse"], cuts[s]["size"])),
               met=all(c["met"] for k, c in cuts.items() if k != "readout"),     # (a diffuse readout as the finder's
                                                                                   # unreachable one: reported, not required)
               family={"pairs": [{"id": r["id"], "met": r.get("met"),
                                  "cuts": {s: {f: c[f] for f in ("size", "lost", "restored", "met")} for s, c in r["cuts"].items()}}
                                 for r in recs]})
    return rec


def pairs_of(questions: list[dict], variants: int = VARIANTS) -> list[tuple]:
    """(pair id, user, corrupt user) for each paraphrase of a family against its first `variants` corrupted keys (its own
    corrupted key, then its swap partners; a key that occurs more than once in a prompt is skipped)."""
    out = []
    for q in questions:
        keys = [q["corrupt_key"]] + [p for p in q.get("partners", []) if p != q["corrupt_key"]]
        for ck in keys[:variants]:
            if q["user"].count(q["key"]) == 1:
                out.append((f"{q['id']}/vs/{ck}", q["user"], q["user"].replace(q["key"], ck)))
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
            rec = cf.find(pt, clean, corrupt, log=log)
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
