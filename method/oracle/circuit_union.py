"""Union circuits and worst-donor tests of question-scoped circuits (stability, 10-16; methods' design C2,
~/mpd-data/methods/design_1016.txt 2a): what one key's circuit, two keys' circuits together, and the union of many keys'
circuits explain about a task's template over its in-distribution inputs.

A template is one task's paraphrase p: the questions of the given sets with that task and paraphrase (the sealed test
split never). Every key k with a question there gets its circuit C(k), circuit_masks.find_batch (--arm) on the question's
own pair (k against its corrupted key). The keys split in two, seeded by the template: the union keys (UNION_SHARE of the
train split's keys) and the unseen keys (the rest of the train split, and every dev key).
  union      per cut, every unit of the union keys' circuits by its role (circuit_family.key_of: a neuron's place counted
             from the key word, or back from the answer in the template's common tail; a head at the answer), with its
             frequency, the share of union keys whose cut holds it; U_f the units of frequency at least f (FREQS; f = 0:
             every unit any union key holds)
  donors     for an unseen key x, every other key r of the template whose prompt has x's length, agrees with x's before
             the key and differs at its first token, and whose pair (x, r) is valid (answer words differ, gap at least
             MIN_GAP): every in-distribution input of the template the sets hold
  tests      per donor r, per cut of C(x), each set patched into the pair (x, r): noising, base x with the set at r's
             values; denoising, base r with the set at x's
               own        C(x)'s cut alone (plain resampling)
               aggregate  C(x) u C(r), r's own circuit found without the pair (methods' aggregation)
               union@f    U_f mapped into the pair, for each f in FREQS
               random     as many random units as the aggregate, at its (layer, position) slots (seeded)
               random_union  as many random units as the whole union (U_0), at its slots
             per test over the donors: the mean shares, the worst donor's shares (the donor whose smaller share is
             smallest), and the share of donors passing PASS both ways; over every donor, and over the donors the union
             never saw (unseen keys) apart
  claims     per cut of C(x), the claim its kind of unit can make and the test that checks it: movers "value-generic"
             (heads that move whatever key sits at the key's place: own must pass on every donor), source and readout
             "value-specific" (units of x's own value: aggregate must pass); the verdict on the donors' mean shares and on
             the worst donor (the user's data-distribution sufficiency: an adversary limited to real inputs)

  circuit_union.py QUESTIONS.json [QUESTIONS.json ...] OUT --tasks capital,city_country [--paraphrases 0,1,2,3,4]
                   [--arm mask] [--unseen N] [--model Qwen/Qwen3-8B] [--dtype bfloat16]
writes OUT/records/<question>.json (finder records; a tested unseen key's with its "claims"), OUT/<task>__p<p>.json (the
union and every unseen key's tests) and OUT/summary.json, and prints a summary per template.

Family circuits (the lead's T1: a question's circuit = its template's mover union, shared and precomputed, plus its own
source cut):
  circuit_union.py unions RECORDS_DIR [RECORDS_DIR ...] OUT.json [--split train]
      per task and per paraphrase, every head the mover cuts of the split's records hold, with its frequency (the share of
      those records holding it), and per task over every paraphrase ("all")
  circuit_union.py family RECORDS_DIR UNIONS.json OUT_DIR [--min-keys 4]
      each finder record with its routes replaced by its template's mover union at the answer position (the task's union
      over every paraphrase when the template has fewer than --min-keys records), its own routes as "own_routes", its
      cuts["movers"] marked "family"; source and readout cuts unchanged
"""
from __future__ import annotations

import argparse
import collections
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

import circuit_family as fam
import circuit_finder as cf
import circuit_masks as cm
import circuit_robust as cr

PASS = cf.PASS
UNION_SHARE = 0.5
FREQS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)
CUTS = ("source", "movers", "readout")
CLAIM = {"movers": ("value-generic", "own"), "source": ("value-specific", "aggregate"),
         "readout": ("value-specific", "aggregate")}
BATCH = 16          # pairs a find_batch call searches together


def template(files: list, task: str, p: int) -> dict:
    """{key: question} of the task's paraphrase p over the question sets (first set first; the test split never)."""
    out = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        for q in (d["questions"] if isinstance(d, dict) else d):
            if q.get("task") == task and q.get("paraphrase") == p and q.get("split") != "test" and q["key"] not in out:
                out[q["key"]] = q
    return out


def circuits(model, tok, qs: dict, arm: str, steps: int, recdir: Path, log=None) -> dict:
    """{key: record}: each question's circuit on its own pair (records already in recdir are read)."""
    out, todo = {}, []
    for k, q in qs.items():
        path = recdir / (q["id"].replace("/", "__") + ".json")
        if path.exists():
            out[k] = json.loads(path.read_text())
        else:
            todo.append((k, q, path))
    for a in range(0, len(todo), BATCH):
        part = todo[a:a + BATCH]
        items = [(cf.chat(tok, q["user"]), cf.chat(tok, q["corrupt_user"]), None) for _, q, _ in part]
        keep = [j for j, (c, k, _) in enumerate(items) if len(c) == len(k)]
        pts = [cf.Patcher(model, tok) for _ in keep]
        recs = cm.find_batch(pts, [items[j] for j in keep], arm=arm, steps=steps) if keep else []
        for j, rec in zip(keep, recs):
            k, q, path = part[j]
            rec.update(id=q["id"], task=q["task"], key=k, paraphrase=q["paraphrase"], split=q.get("split"), user=q["user"],
                       corrupt_user=q["corrupt_user"], corrupt_key=q["corrupt_key"], ids=items[j][0], corrupted=items[j][1])
            path.write_text(json.dumps(rec))
            out[k] = rec
        del pts
        if log:
            log(f"  circuits {min(a + BATCH, len(todo))}/{len(todo)}")
    return out


def cut_keys(rec: dict, cut: str, tail: int) -> list:
    """A record's cut as family keys (roles)."""
    us = rec["routes"] if cut == "movers" else [n for n in rec["nodes"] if n["stage"] == cut]
    return list(dict.fromkeys(fam.key_of(cut, u, rec, tail) for u in us))


def union(recs: list, tail: int) -> dict:
    """Per cut, [(key, frequency)]: every unit of the records' cuts, by the share of records whose cut holds it (most
    frequent first, ties first seen first)."""
    out = {}
    for c in CUTS:
        n = collections.Counter()
        for r in recs:
            for k in cut_keys(r, c, tail):
                n[k] += 1
        out[c] = [(k, v / len(recs)) for k, v in n.most_common()]
    return out


def random_like(units: list, pt: cf.Patcher, rng) -> list:
    """As many units at the same (layer, position) slots, indices drawn at random."""
    return [("head", u[1], int(rng.integers(pt.H)), u[3]) if u[0] == "head" else ("neuron", u[1], u[2], int(rng.integers(pt.D)))
            for u in units]


def donors_of(x: dict, recs: dict) -> list:
    """The keys whose prompt can pair with x's (module docstring: donors), as (key, record)."""
    X, f = x["ids"], x["first_diff"]
    return [(k, r) for k, r in recs.items() if r is not x and len(r["ids"]) == len(X) and r["ids"][:f] == X[:f]
            and r["ids"][f] != X[f]]


def summary(rows: list) -> dict:
    """A test's shares over donors (each row: lost, restored, the donor seen by the union or not)."""
    def one(rs):
        if not rs:
            return None
        lo, re_ = [r[0] for r in rs], [r[1] for r in rs]
        w = min(rs, key=lambda r: min(r[0], r[1]))
        return {"n": len(rs), "mean": [float(np.mean(lo)), float(np.mean(re_))], "worst": [w[0], w[1]],
                "passing": sum(a >= PASS and b >= PASS for a, b, _ in rs) / len(rs),
                "passes_mean": bool(np.mean(lo) >= PASS and np.mean(re_) >= PASS), "passes_worst": bool(min(w[0], w[1]) >= PASS)}
    return {"all": one(rows), "unseen": one([r for r in rows if not r[2]])}


def test(model, tok, x: dict, recs: dict, U: dict, seen: set, tail: int, log=None) -> dict:
    """Every test of x's circuit over its donors (module docstring: tests, claims)."""
    ds = donors_of(x, recs)
    pts = [cf.Patcher(model, tok) for _ in ds]
    infos = cf.setup_many(pts, [(x["ids"], r["ids"]) for _, r in ds]) if ds else []
    rng = np.random.default_rng(zlib.crc32(x["id"].encode()))
    sets, plan, used = [], [], []
    for (k, r), pt, info in zip(ds, pts, infos):
        if not info["valid"] or info["same_answer"] or info["gap"] < cf.MIN_GAP:
            sets.append([])
            continue
        pair = {"tokens": len(x["ids"]), "first_diff": info["first_diff"]}
        mine, names = [], []
        for c in CUTS:
            own = cr.cut_units(x, c)[0] if c in x.get("cuts", {}) else []
            agg = list(dict.fromkeys(own + (cr.cut_units(r, c)[0] if c in r.get("cuts", {}) else [])))
            whole = fam.mapped([q for q, _ in U[c]], pair)
            for name, s in [("own", own), ("aggregate", agg), ("random", random_like(agg, pt, rng)),
                            ("random_union", random_like(whole, pt, rng))] + [
                    (f"union@{f:g}", fam.mapped([q for q, v in U[c] if v >= f and (f > 0 or v > 0)], pair)) for f in FREQS]:
                if s:
                    mine.append(s)
                    names.append((c, name, len(s)))
        sets.append(mine)
        plan.append((k, names))
        used.append(len(sets) - 1)
    got = fam.measure(pts, sets) if pts else []
    rows, sizes = collections.defaultdict(list), {}
    for (k, names), j in zip(plan, used):
        for (c, name, n), sh in zip(names, got[j]):
            rows[(c, name)].append((sh[0], sh[1], k in seen))
            sizes.setdefault((c, name), []).append(n)
    del pts
    out = {"id": x["id"], "key": x["key"], "donors": len(plan), "donors_unseen": sum(k not in seen for k, _ in plan), "cuts": {}}
    for (c, name), rs in rows.items():
        out["cuts"].setdefault(c, {})[name] = {**summary(rs), "size": float(np.median(sizes[(c, name)])),
                                               "donor_shares": [[k, a, b] for (k, _), (a, b, _) in zip(plan, rs)]}
    claims = {}
    for c, (kind, by) in CLAIM.items():
        t = out["cuts"].get(c, {}).get(by)
        if c in x.get("cuts", {}) and cr.cut_units(x, c)[0] and t and t["all"]:
            claims[c] = {"type": kind, "test": by, "donors": t["all"]["n"], "mean": t["all"]["mean"], "worst": t["all"]["worst"],
                         "passing": t["all"]["passing"], "holds_mean": t["all"]["passes_mean"],
                         "holds_worst": t["all"]["passes_worst"]}
    out["claims"] = claims
    return out


def report(name: str, tests: list, U: dict, log=print):
    """One line per cut and test: questions passing on the mean, on the worst donor, the mean share of donors passing."""
    log(f"{name}: {len(tests)} unseen keys, donors median {np.median([t['donors'] for t in tests]) if tests else 0:g}; "
        + ", ".join(f"{c} union {len(U[c])} units" for c in CUTS))
    for c in CUTS:
        names = sorted({n for t in tests for n in t["cuts"].get(c, {})}, key=lambda n: (not n.startswith("own"),
                                                                                         not n.startswith("agg"), n))
        for n in names:
            for grp in ("all", "unseen"):
                xs = [t["cuts"][c][n][grp] for t in tests if t["cuts"].get(c, {}).get(n, {}).get(grp)]
                if xs:
                    sz = np.median([t["cuts"][c][n]["size"] for t in tests if n in t["cuts"].get(c, {})])
                    log(f"  {c:7s} {n:11s} {grp:6s} size {sz:5.0f}: mean passes {sum(x['passes_mean'] for x in xs)}/{len(xs)}, "
                        f"worst donor {sum(x['passes_worst'] for x in xs)}/{len(xs)}, donors passing "
                        f"{np.mean([x['passing'] for x in xs]):.2f}, median mean {np.median([x['mean'][0] for x in xs]):.2f}/"
                        f"{np.median([x['mean'][1] for x in xs]):.2f}")


def unions(dirs: list, split: str = "train") -> dict:
    """{task: {"p<k>" and "all": [[layer, head, frequency], ...] most frequent first, "n": {...: records}}} over the mover
    cuts of the records of the split (module docstring: family circuits)."""
    by = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    n = collections.defaultdict(collections.Counter)
    seen = set()
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text())
            if r.get("split") != split or not r.get("cuts", {}).get("movers") or r["id"] in seen:
                continue
            seen.add(r["id"])
            task, p = r["id"].split("/")[0], f"p{r['id'].split('/')[2][1:]}"
            for g in (p, "all"):
                n[task][g] += 1
                for h in {(x["layer"], x["head"]) for x in r["routes"]}:
                    by[task][g][h] += 1
    return {t: {**{g: [[l, h, c / n[t][g]] for (l, h), c in cnt.most_common()] for g, cnt in gs.items()}, "n": dict(n[t])}
            for t, gs in by.items()}


def family_record(rec: dict, U: dict, min_keys: int = 4) -> dict | None:
    """A finder record with its template's mover union as its routes (module docstring: family circuits); None when its
    task has no union."""
    task, p = rec["id"].split("/")[0], f"p{rec['id'].split('/')[2][1:]}"
    u = U.get(task)
    if not u:
        return None
    g = p if u["n"].get(p, 0) >= min_keys else "all"
    T = len(rec["ids"])
    routes = [{"layer": l, "head": h, "pos": T - 1, "family_frequency": f} for l, h, f in u[g]]
    cuts = dict(rec.get("cuts", {}))
    cuts["movers"] = {**cuts.get("movers", {}), "family": g, "family_keys": u["n"][g], "size": len(routes), "met": True}
    return {**rec, "routes": routes, "own_routes": rec.get("routes", []), "cuts": cuts, "search": f"family+{rec.get('search', 'finder')}"}


def main_family(argv: list):
    ap = argparse.ArgumentParser(description="family circuits (module docstring)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    u = sp.add_parser("unions")
    u.add_argument("dirs", type=Path, nargs="+")
    u.add_argument("out", type=Path)
    u.add_argument("--split", default="train")
    f = sp.add_parser("family")
    f.add_argument("records", type=Path)
    f.add_argument("unions", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--min-keys", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "unions":
        U = unions(a.dirs, a.split)
        a.out.write_text(json.dumps(U))
        for t, x in sorted(U.items()):
            print(f"{t}: {x['n']}; union over every paraphrase {len(x['all'])} heads")
        return
    U = json.loads(a.unions.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    done = 0
    for p in sorted(a.records.glob("*.json")):
        rec = json.loads(p.read_text())
        fr = family_record(rec, U, a.min_keys) if rec.get("cuts") else None
        if fr:
            (a.out / p.name).write_text(json.dumps(fr))
            done += 1
        else:
            (a.out / p.name).write_text(json.dumps(rec))
    print(f"{done} family records in {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("questions", type=Path, nargs="+")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--paraphrases", default="0,1,2,3,4")
    ap.add_argument("--arm", default="mask")
    ap.add_argument("--steps", type=int, default=cm.STEPS)
    ap.add_argument("--unseen", type=int, default=0, help="at most this many unseen keys a template (0: all)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cf._threads()
    recdir = a.out / "records"
    recdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto").eval()
    model.requires_grad_(False)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    allsum = {}
    for task in a.tasks.split(","):
        for p in (int(x) for x in a.paraphrases.split(",")):
            name = f"{task}__p{p}"
            path = a.out / f"{name}.json"
            if path.exists():
                d = json.loads(path.read_text())
                allsum[name] = d["tests"]
                continue
            qs = template(a.questions, task, p)
            log(f"{name}: {len(qs)} keys")
            recs = {k: r for k, r in circuits(model, tok, qs, a.arm, a.steps, recdir, log=log).items() if r.get("cuts")}
            train = sorted(k for k in recs if qs[k].get("split") == "train")
            if len(train) < 2 or len(recs) < 3:             # (no union, or nothing unseen to test it on)
                log(f"{name}: {len(recs)} keys with circuits, {len(train)} of the train split: skipped")
                continue
            np.random.default_rng(zlib.crc32(name.encode())).shuffle(train)
            ukeys = set(train[:int(round(UNION_SHARE * len(train)))])
            tail = fam.tail_of(list(recs.values()))
            U = union([recs[k] for k in sorted(ukeys)], tail)
            unseen = sorted(k for k in recs if k not in ukeys)
            if a.unseen:
                unseen = unseen[:a.unseen]
            tests = []
            for k in unseen:
                t = test(model, tok, recs[k], recs, U, ukeys, tail)
                tests.append(t)
                rp = recdir / (recs[k]["id"].replace("/", "__") + ".json")
                rp.write_text(json.dumps({**recs[k], "claims": t["claims"]}))
                log(f"  {k}: {t['donors']} donors ({t['donors_unseen']} unseen); " + ", ".join(
                    f"{c} {v['type'][6:]} {'mean ok' if v['holds_mean'] else 'mean fails'}, "
                    f"{'worst ok' if v['holds_worst'] else 'worst fails'} ({v['passing']:.2f} of donors)" for c, v in t["claims"].items()))
            path.write_text(json.dumps({"task": task, "paraphrase": p, "arm": a.arm, "union_keys": sorted(ukeys), "unseen_keys": unseen,
                                        "union": {c: [[list(q) if isinstance(q, tuple) else q for q in kv] for kv in U[c]] for c in CUTS},
                                        "tests": tests}))
            report(name, tests, U, log=log)
            allsum[name] = tests
    tot = {}
    for c, (kind, by) in CLAIM.items():
        xs = [t["claims"][c] for ts in allsum.values() for t in ts if c in t["claims"]]
        if xs:
            tot[c] = {"type": kind, "test": by, "n": len(xs), "holds_mean": sum(x["holds_mean"] for x in xs),
                      "holds_worst": sum(x["holds_worst"] for x in xs), "donors_passing": float(np.mean([x["passing"] for x in xs]))}
    (a.out / "summary.json").write_text(json.dumps({"templates": list(allsum), "claims": tot}, indent=1))
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    import sys

    main_family(sys.argv[1:]) if len(sys.argv) > 1 and sys.argv[1] in ("unions", "family") else main()
