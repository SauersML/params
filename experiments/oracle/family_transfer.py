"""Family circuits on unseen paraphrases and unseen pairs, and their stability across disjoint paraphrase sets (#2951
scorecard Q2 and Q4, the lead's 10-16 circuit push): whether circuit_family's circuit of a question family is the
family's mechanism or its own pairs' fit.
Per family of circuit_finder pair records (a question's paraphrases, each against its corrupted keys; the pairs whose
answer word is the family's), the last paraphrase is unseen and the others are training; the training paraphrases split
in two halves (alternate paraphrases). Three selections, each per stage (source neurons, mover heads):
  family    circuit_family.select on the pairs given
  old       circuit_family before 10-16: per pair a cut's units share 1 by their effect alone, a key's weight the sum over
            the pairs, the smallest prefix by weight that a majority of the pairs pass
  pair      one pair's own finder circuit (the first training pair: the first paraphrase against its own corrupted key)
Q4, stability: the family and old selections on each half, and per stage between the two halves' cuts: the effect-
weighted overlap (the share of one cut's weight on keys the other holds, the mean of both directions) with each unit
weighted by its leave-one-out drop (what the cut loses without it on the half's pairs, the larger direction, measured;
the units of a source cut matter together: their single drops add up to a median 6% of the cut's), also position-free (a
neuron key without its role), and the Jaccard index of the key sets, beside the sizes; "whole" pools both stages' drops
(the scorecard's circuit: source neurons and movers). Beside them the pair circuits' own overlap across the halves (pairs
of the same corrupted key, one paraphrase from each half), by their leave-one-out drops and by the finder's single drops.
Q2, transfer: each selection on the training pairs (the pair circuit as found), patched into
  unseen paraphrase   the unseen paraphrase's pairs (its records)
  unseen keys         every paraphrase against UNSEEN_KEYS corrupted keys past the records' (its next swap partners;
                      prompts of unequal length, invalid contrasts and other answer words dropped)
each pair's lost and restored shares and whether it passes PASS both ways (a key with no place in a pair is left out).
Shares are measured in --dtype (float32 by default: bfloat16 shares move by ~0.01-0.05 of a gap with the batch's shape).

  family_transfer.py run PAIRS_DIR QUESTIONS.json OUT [--model Qwen/Qwen3-8B] [--dtype float32] [--stride 1 --offset 0]
  family_transfer.py summary OUT [OUT ...]
writes OUT/<family>.json per family; summary prints the before/after table.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))

STAGES = ("source", "movers")
UNSEEN_KEYS = 2
BAD = {"the"}       # answer words that are the template, not the task (Qwen3-8B opposite/*/p? "The opposite ...")


def old_select(pts, recs, stages=STAGES) -> dict:
    """circuit_family's selection before 10-16 (module docstring: old)."""
    import circuit_family as fam
    import circuit_finder as cf
    tail, W = fam.tail_of(recs), {}
    for r in recs:
        for s in stages:
            units = r["routes"] if s == "movers" else [n for n in r["nodes"] if n["stage"] == s]
            if not units or s not in r.get("cuts", {}):
                continue
            e = [max(abs(u["lost"]), abs(u["restored"])) for u in units]
            tot = sum(e)
            ws = W.setdefault(s, {})
            for u, x in zip(units, e):
                k = fam.key_of(s, u, r, tail)
                ws[k] = ws.get(k, 0.0) + (x / tot if tot > 1e-9 else 1 / len(units))
    out = {}
    for s, ws in W.items():
        ranked = sorted(ws, key=lambda k: -ws[k])
        g = cf.grid(len(ranked))
        n, _ = fam.counts(pts, recs, [ranked[:k] for k in g])
        ok = next((k for k, c in zip(g, n) if 2 * c > len(recs)), None)
        out[s] = {"keys": ranked[:ok] if ok else [], "diffuse": ok is None, "candidates": len(ranked)}
    return out


def pair_cut(rec: dict, tail: int, stages=STAGES) -> dict:
    """A pair record's own finder circuit as family keys per stage."""
    import circuit_family as fam
    out = {}
    for s in stages:
        units = rec["routes"] if s == "movers" else [n for n in rec["nodes"] if n["stage"] == s]
        if s in rec.get("cuts", {}):
            out[s] = {"keys": list(dict.fromkeys(fam.key_of(s, u, rec, tail) for u in units)), "diffuse": not units}
    return out


def loo(pts, recs, keys) -> dict:
    """Each key's leave-one-out drop from the key set: the mean over the pairs of the larger of lost and restored
    falling without it."""
    import circuit_family as fam
    if not keys:
        return {}
    _, sh = fam.counts(pts, recs, [keys] + [[x for x in keys if x != k] for k in keys])
    return {k: statistics.mean(max(a[0] - b[0], a[1] - b[1]) for a, b in zip(sh[0], sh[1 + i])) for i, k in enumerate(keys)}


def free(k):
    """A key without its role (a neuron's position-free key)."""
    return tuple(k[:3]) if k[0] == "neuron" else tuple(k)


def overlap(wa: dict, wb: dict, f=lambda k: k) -> float | None:
    """The mean over both directions of the share of one weighting (clipped at 0) on keys the other holds."""
    def one(x, y):
        xs, ys = collections.Counter(), {f(k) for k in y}
        for k, v in x.items():
            xs[f(k)] += max(v, 0.0)
        t = sum(xs.values())
        return sum(v for k, v in xs.items() if k in ys) / t if t > 0 else None
    a, b = one(wa, wb), one(wb, wa)
    return None if a is None or b is None else (a + b) / 2


def jaccard(a, b) -> float | None:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def tup(k):
    return tuple(tup(x) for x in k) if isinstance(k, list) else k


def run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import circuit_family as fam
    import circuit_finder as cf
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto").eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok, rows=a.rows)
    qs = {q["id"]: q for q in json.loads(a.questions.read_text())["questions"]}
    by = collections.defaultdict(list)
    for f in sorted(a.pairs.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("cuts"):
            t, k, p, _, ck = r["id"].split("/")
            by[f"{t}/{k}"].append(r)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    for name in sorted(by)[a.offset::a.stride]:
        path = a.out / f"{name.replace('/', '__')}.json"
        if path.exists():
            continue
        recs = by[name]
        word = collections.Counter(r["answer_word"] for r in recs).most_common(1)[0][0]
        recs = [r for r in recs if r["answer_word"] == word and word not in BAD]
        para = sorted({r["id"].split("/")[2] for r in recs}, key=lambda p: int(p[1:]))
        if len(para) < 3:
            log(f"{name}: {len(para)} paraphrases with valid pairs, skipped")
            continue
        unseen, train = para[-1], para[:-1]
        halves = (train[0::2], train[1::2])
        of = lambda ps: [r for r in recs if r["id"].split("/")[2] in ps]   # noqa: E731
        tr = of(train)
        tr.sort(key=lambda r: (int(r["id"].split("/")[2][1:]), r["id"].split("/")[4] != qs[r["id"].rsplit("/vs/", 1)[0]]["corrupt_key"]))
        log(f"{name}: answer {word!r}, train {train} ({len(tr)} pairs), unseen {unseen}, halves {halves}")
        pts = fam.patchers(pt, tr)
        P = dict(zip((r["id"] for r in tr), pts))
        sub = lambda rs: ([P[r["id"]] for r in rs], rs)   # noqa: E731
        out = {"family": name, "answer_word": word, "train": train, "unseen": unseen, "halves": halves, "builds": {}}
        cand = fam.candidates(tr, fam.tail_of(tr))
        for alg in ("family", "old"):
            for part, rs in (("half0", of(halves[0])), ("half1", of(halves[1])), ("train", tr)):
                ps, rs = sub(rs)
                t1 = time.time()
                if alg == "family":
                    cs = fam.candidates(rs, fam.tail_of(rs))
                    sel = {s: fam.select(ps, rs, ks, log=log, name=f"{alg} {part} {s}") for s, ks in cs.items() if s in STAGES}
                else:
                    sel = old_select(ps, rs)
                b = {}
                for s, c in sel.items():
                    keys = list(c["keys"])
                    n, sh = fam.counts(ps, rs, [keys])
                    lo = loo(ps, rs, keys)
                    b[s] = {"keys": keys, "diffuse": c["diffuse"], "candidates": c["candidates"], "loo": [lo[k] for k in keys],
                            "pairs": len(rs), "passing": n[0], "shares": sh[0],
                            "alternatives": {json.dumps(k): v for k, v in c.get("alternatives", {}).items()}}
                out["builds"].setdefault(alg, {})[part] = b
                log(f"  {alg} {part}: " + ", ".join(f"{s} {len(c['keys'])} keys, {c['passing']}/{c['pairs']} pass" for s, c in b.items())
                    + f" ({time.time() - t1:.1f} s)")
        # the pair circuits' own leave-one-out drops (each on its own pair), for the per-pair overlap across the halves
        out["pair_loo"] = {}
        tail = fam.tail_of(tr)
        for r, p in zip(tr, pts):
            pc = pair_cut(r, tail)
            out["pair_loo"][r["id"]] = {s: {json.dumps(k): v for k, v in loo([p], [r], c["keys"]).items()} for s, c in pc.items()}
            out["pair_loo"][r["id"]]["single"] = {json.dumps(fam.key_of(s, u, r, tail)): max(u["lost"], 0.0)
                                                  for s in STAGES for u in (r["routes"] if s == "movers" else
                                                                            [n for n in r["nodes"] if n["stage"] == s])}
        del pts, P
        torch.cuda.empty_cache()
        # unseen pairs: the unseen paraphrase's records, and every paraphrase against corrupted keys past the records'
        test = {"unseen paraphrase": of([unseen]), "unseen keys": [], "unseen keys, unseen paraphrase": []}
        tpts = {"unseen paraphrase": fam.patchers(pt, test["unseen paraphrase"]), "unseen keys": [], "unseen keys, unseen paraphrase": []}
        nvar = max(1, len({r["id"].split("/")[4] for r in recs if r["id"].split("/")[2] == train[0]}))
        for p_ in para:
            q = qs.get(f"{name}/{p_}")
            if q is None:
                continue
            for pid, user, cuser in fam.pairs_of([q], UNSEEN_KEYS, skip=nvar):
                clean, corrupt = cf.chat(tok, user), cf.chat(tok, cuser)
                if len(clean) != len(corrupt):
                    continue
                p = cf.Patcher(model, tok, rows=a.rows)
                s = p.setup(clean, corrupt)
                if not s["valid"] or s["gap"] < cf.MIN_GAP or s["same_answer"] or s["answer_word"] != word:
                    continue
                r = {"id": pid, "ids": clean, "corrupted": corrupt, "tokens": len(clean), "first_diff": s["first_diff"]}
                which = "unseen keys, unseen paraphrase" if p_ == unseen else "unseen keys"
                test[which].append(r)
                tpts[which].append(p)
        out["transfer"] = {}
        sels = {"family": out["builds"]["family"]["train"], "old": out["builds"]["old"]["train"], "pair": pair_cut(tr[0], tail)}
        for alg, sel in sels.items():
            for which, rs in test.items():
                for s, c in sel.items():
                    if not rs:
                        continue
                    n, sh = fam.counts(tpts[which], rs, [list(c["keys"])])
                    out["transfer"].setdefault(alg, {}).setdefault(which, {})[s] = {
                        "pairs": [r["id"] for r in rs], "shares": sh[0], "passing": n[0], "diffuse": c["diffuse"]}
        log("  transfer: " + "; ".join(f"{alg} {w} " + ", ".join(f"{s} {c['passing']}/{len(c['pairs'])}" for s, c in ws.items())
                                       for alg, d in out["transfer"].items() for w, ws in d.items()))
        del tpts
        torch.cuda.empty_cache()
        out["seconds"] = time.time() - t0
        path.write_text(json.dumps(out))


def summary(dirs: list[Path]):
    rows = [json.loads(f.read_text()) for d in dirs for f in sorted(d.glob("*.json"))]
    print(f"{len(rows)} families")
    med = lambda xs: statistics.median(xs) if xs else float("nan")      # noqa: E731
    mean = lambda xs: statistics.mean(xs) if xs else float("nan")       # noqa: E731
    print("\nQ4 stability between the halves' cuts (mean over families; LOO-weighted, position-free LOO, Jaccard; sizes median)")
    for alg in ("family", "old"):
        for s in STAGES + ("whole",):
            o, of_, j, sz = [], [], [], []
            for r in rows:
                h = [r["builds"][alg][p] for p in ("half0", "half1")]
                if s == "whole":
                    w = [{(st, tup(k)): v for st in STAGES if st in x for k, v in zip(x[st]["keys"], x[st]["loo"])} for x in h]
                    keys = [list(x) for x in w]
                else:
                    if not all(s in x for x in h):
                        continue
                    w = [{tup(k): v for k, v in zip(x[s]["keys"], x[s]["loo"])} for x in h]
                    keys = [[tup(k) for k in x[s]["keys"]] for x in h]
                if not all(keys):
                    continue
                f_ = (lambda k: (k[0], free(k[1]))) if s == "whole" else free
                o.append(overlap(*w)); of_.append(overlap(*w, f=f_)); j.append(jaccard(*keys)); sz += [len(k) for k in keys]
            o, of_ = [x for x in o if x is not None], [x for x in of_ if x is not None]
            print(f"  {alg:7s} {s:7s} LOO {mean(o):.3f}  free {mean(of_):.3f}  Jaccard {mean(j):.3f}  size {med(sz)}  ({len(o)} families)")
    print("\nper-pair finder circuits across the halves (same corrupted key), LOO-weighted / single-drop-weighted:")
    for s in STAGES + ("whole",):
        o, o1 = [], []
        for r in rows:
            ids = list(r["pair_loo"])
            h0 = [i for i in ids if i.split("/")[2] in r["halves"][0]]
            h1 = [i for i in ids if i.split("/")[2] in r["halves"][1]]
            for x in h0:
                for y in h1:
                    if x.split("/")[4] != y.split("/")[4]:
                        continue
                    X, Y = r["pair_loo"][x], r["pair_loo"][y]
                    if s == "whole":
                        wx = {(st, kk): v for st in STAGES for kk, v in X.get(st, {}).items()}
                        wy = {(st, kk): v for st in STAGES for kk, v in Y.get(st, {}).items()}
                        sx, sy = X["single"], Y["single"]
                    else:
                        wx, wy = X.get(s, {}), Y.get(s, {})
                        sx = {k: v for k, v in X["single"].items() if k in wx}
                        sy = {k: v for k, v in Y["single"].items() if k in wy}
                    if wx and wy:
                        o.append(overlap(wx, wy))
                        o1.append(overlap(sx, sy))
        o, o1 = [x for x in o if x is not None], [x for x in o1 if x is not None]
        print(f"  {s:7s} LOO {mean(o):.3f}  single drops {mean(o1):.3f}  ({len(o)} pair pairs)")
    print("\nQ2 transfer: pairs passing both ways / pairs (median lost, restored) per selection built on the training paraphrases")
    for alg in ("family", "old", "pair"):
        for which in ("unseen paraphrase", "unseen keys", "unseen keys, unseen paraphrase"):
            for s in STAGES:
                n = p = 0
                lo, re_ = [], []
                for r in rows:
                    c = r.get("transfer", {}).get(alg, {}).get(which, {}).get(s)
                    if c:
                        n += len(c["pairs"]); p += c["passing"]
                        lo += [x[0] for x in c["shares"]]; re_ += [x[1] for x in c["shares"]]
                if n:
                    print(f"  {alg:7s} {which:31s} {s:7s} {p}/{n} = {p / n:.2f}  ({med(lo):.2f}, {med(re_):.2f})")
    print("\nin-sample: the training cut on its own pairs, and sizes")
    for alg in ("family", "old"):
        for s in STAGES:
            n = p = 0
            sz = []
            for r in rows:
                c = r["builds"][alg]["train"].get(s)
                if c:
                    n += c["pairs"]; p += c["passing"]; sz.append(len(c["keys"]))
            if n:
                print(f"  {alg:7s} {s:7s} {p}/{n} = {p / n:.2f}, size median {med(sz)}, diffuse "
                      f"{sum(1 for r in rows if r['builds'][alg]['train'].get(s, {}).get('diffuse'))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("pairs", type=Path)
    r.add_argument("questions", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--model", default="Qwen/Qwen3-8B")
    r.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    r.add_argument("--rows", type=int, default=32)
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--offset", type=int, default=0)
    s = sp.add_parser("summary")
    s.add_argument("out", type=Path, nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summary(a.out)


if __name__ == "__main__":
    main()
