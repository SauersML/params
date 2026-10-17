"""Paired-context questions (#2951 oracledata; the lead's 10-16 priority): the same unit or shared group in the circuits of
two texts, and a question whose answer differs between them, so that a reader answering from the unit's name is at
chance by construction (the decisive test of reading a circuit). Every answer is measured on Qwen3-8B in both texts
with answer_contrast's word-level contrast (the record's answer_tokens against its alternatives, the finder's; claims
state it to claim_check as a list "answer"), no weak zeroing (T2's maxima for the steer scale and massive cuts only).

An item recurs when its key (a shared group "G<id>", a neuron pole "L18.N2201+", a head "L5.H3") is in the circuit of at
least two texts; it acts at its own positions in each text. Kinds (a pair of texts A, B, in a random order):

  pair_zero   in which text does zeroing the item lower the lead for the answer more (signed: more negative)?
              A | B (by at least GAP bits) | same (within LOW bits)
  pair_flip   turning the item up (STEER x each neuron's calibration maximum): does the answer flip (the top token
              leaves the answer word) in A, in B, in both or in neither?
  pair_sign   does the item support the answer (zeroing it lowers the lead by HIGH bits or more; under LOW: it does not)
              in A, in B, in both or in neither?

  pairs.py measure RECORDS [RECORDS ...] OUT.jsonl --thr THR.pt [--groups APPLY ...] [--source NAME] [--shard K/N]  (GPU)
      each text's recurring items zeroed and turned up, one line per text: its measurements
  pairs.py build OUT_DIR MEASURE.jsonl [...] [--exclude-keys MANIFEST.json] [--sealed testset_8b.json] [--seed S]
      the pairs, split by family (test and dev families never trained on), every kind's classes balanced in each split
Each question holds both texts' context and circuit ids, its gold class, an answer sentence and "sides": the two runs as
claim_check claims (word-level answer), whose returned changes give the gold class by the rule above.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3] / "method" / "oracle"))
import oq  # noqa: E402

CONTRAST = "word_v1"   # answer_contrast's word-level sets from the record (answer_tokens, alternatives)
PER_KEY = 40           # text pairs drawn per item key


def recurring(paths: list[Path], groups: dict) -> dict:
    """key -> the texts whose circuit holds it."""
    seen = collections.defaultdict(set)
    for p in paths:
        r = json.loads(p.read_text())
        if r.get("skipped") or not r.get("ids"):
            continue
        for it in oq.items_of(r, groups.get(p.name)):
            seen[it["key"]].add(r["id"])
    return {k: v for k, v in seen.items() if len(v) >= 2}


def measure(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import claim_check as cc
    import neuron_builder as nb
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device or "cuda").eval()
    nr = nb.Neurons(model, tok)
    nr.load(a.thr)
    nr.thr = None
    eng = oq.Engine(nr, cc)
    paths, seen_names = [], set()
    for d in a.records:  # one record per text: the first directory's when two hold the same question
        for p in sorted(d.glob("*.json")):
            if p.name not in seen_names:
                seen_names.add(p.name)
                paths.append(p)
    groups = {}
    for g in a.groups or []:
        for p in g.glob("*.json"):
            groups[p.name] = json.loads(p.read_text())
    for p in paths:  # a group file applies to its own record only (same id and prompt)
        r, g = json.loads(p.read_text()), groups.get(p.name)
        if g is not None and (g.get("id") != r.get("id") or (g.get("ids") and g["ids"] != r.get("ids"))):
            del groups[p.name]
    rec = recurring(paths, groups)
    print(json.dumps({"texts": len(paths), "recurring_keys": len(rec)}), flush=True)
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        paths = paths[k::n]
    done = {json.loads(x)["context_id"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    t0 = time.time()
    with a.out.open("a") as fh:
        for j, p in enumerate(paths):
            r = json.loads(p.read_text())
            if r.get("skipped") or not r.get("ids") or r["id"] in done or not r.get("answer_tokens") or not r.get("alternatives"):
                continue
            ids, T = r["ids"], len(r["ids"])
            items = [it for it in oq.items_of(r, groups.get(p.name)) if it["key"] in rec]
            circ = oq.circuit_record(r, groups.get(p.name), oq.items_of(r, groups.get(p.name)), tok, nr, a.source)
            val = {u["name"]: u for u in circ["units"]}
            massive = lambda u: u in val and abs(val[u]["value"] or 0.0) >= float(nr.massive[val[u]["layer"]])  # noqa: E731
            items = [it for it in items if it["kind"] == "head" or not all(massive(u) for u in it["units"])]
            seen, keep = set(), []
            for it in items:  # one item per key a text (a neuron pole at two positions: its first, measured order)
                if it["key"] not in seen:
                    seen.add(it["key"])
                    keep.append(it)
            rows = [{}]
            for it in keep:
                us = cc.units_of(nr, it["units"], T)
                rows.append({"units": us, "op": "zero"})
                if it["kind"] != "head":
                    rows.append({"units": us, "op": ("steer", oq.STEER)})
            lp = eng.answer_lp(ids, rows)
            A, O = r["answer_tokens"], r["alternatives"]
            cs = (torch.logsumexp(lp[:, A], -1) - torch.logsumexp(lp[:, O], -1)) / math.log(2)
            top = lp.argmax(-1).tolist()
            out, k = {}, 1
            for it in keep:
                m = {"units": it["units"], "kind": it["kind"], "name": it["name"],
                     "zero": {"change": float(cs[k] - cs[0]), "top": top[k], "flip": top[k] not in A}}
                k += 1
                if it["kind"] != "head":
                    m["steer"] = {"change": float(cs[k] - cs[0]), "top": top[k], "flip": top[k] not in A}
                    k += 1
                out[it["key"]] = m
            if a.replay and zlib.crc32(r["id"].encode()) % a.replay == 0:  # claim_check gives the same numbers (word-level)
                cl = [{"prompt": {"ids": ids}, "position": -1, "units": it["units"], "op": op, "answer": A, "alternatives": O,
                       "predict": {"contrast_change": {"op": ">", "value": -1e9}}}
                      for it in keep for op in (["zero"] + ([{"steer": oq.STEER}] if it["kind"] != "head" else []))]
                got = [x for c0 in range(0, len(cl), oq.CHECK_CHUNK) for x in cc.check_many(nr, cl[c0:c0 + oq.CHECK_CHUNK], controls=0)]
                mine = [m_[op]["change"] for m_ in out.values() for op in ("zero", "steer") if op in m_]
                diff = max((abs(g["change"] - x) for g, x in zip(got, mine)), default=0.0)
                print(json.dumps({"replay": r["id"], "claims": len(cl), "max_abs_diff_bits": diff}), flush=True)
            fh.write(json.dumps({"context_id": r["id"], "family": r.get("task"), "task_key": f"{r.get('task')}/{r.get('key')}",
                                 "source": a.source, "contrast": CONTRAST, "contrast_before": float(cs[0]), "top_before": top[0],
                                 "answer_tokens": A, "alternatives": O, "ids": ids, "user": r.get("user"),
                                 "answer_text": r.get("answer_text"), "circuit": circ, "items": out}) + "\n")
            fh.flush()
            if (j + 1) % 20 == 0:
                print(json.dumps({"texts": j + 1, "seconds": round(time.time() - t0, 1)}), flush=True)
    print(json.dumps({"done": len(paths), "seconds": round(time.time() - t0, 1), "out": str(a.out)}), flush=True)


def zero_class(da: float, db: float) -> str | None:
    if abs(da - db) < oq.LOW:
        return "same"
    if abs(da - db) >= oq.GAP:
        return "A" if da < db else "B"
    return None


def both_class(a_: bool | None, b_: bool | None) -> str | None:
    if a_ is None or b_ is None:
        return None
    return "both" if a_ and b_ else "A" if a_ else "B" if b_ else "neither"


def supports(m: dict) -> bool | None:
    z = m["zero"]
    return True if (z["flip"] or z["change"] <= -oq.HIGH) else False if abs(z["change"]) < oq.LOW else None


def side(t: dict, it: dict, op) -> dict:
    """A side's run for claim_check in the text's own word-level terms (its predict always holds: claim_check returns the
    change, from which the gold class follows by the rule above)."""
    return {"prompt": {"ids": t["ids"]}, "position": -1, "units": it["units"], "op": op,
            "predict": {"contrast_change": {"op": ">", "value": -1e9}}, "answer": t["answer_tokens"],
            "alternatives": t["alternatives"]}


def question(kind: str, key: str, A: dict, B: dict) -> dict | None:
    ma, mb = A["items"][key], B["items"][key]
    ya, yb = oq.q_tok(A["answer_text"] or ""), oq.q_tok(B["answer_text"] or "")
    head = f"Text A: \"{A['user']}\" (the model answers {ya}). Text B: \"{B['user']}\" (the model answers {yb}). "
    name = key
    if ma["kind"] != "group":  # where the unit acts in each text (a group's members have their own positions)
        pa, pb = int(ma["units"][0].split("@")[1]), int(mb["units"][0].split("@")[1])
        ta, tb = A["circuit"]["tokens"][pa], B["circuit"]["tokens"][pb]
        name = f"{key} (at {oq.q_tok(ta)} @{pa} in A, {oq.q_tok(tb)} @{pb} in B)"
    if kind == "pair_zero":
        da, db = ma["zero"]["change"], mb["zero"]["change"]
        c = zero_class(da, db)
        if c is None:
            return None
        q = (f"{head}In which text does zeroing {name} lower the model's lead for its answer more: A, B, or about the same?")
        a = {"A": "A", "B": "B", "same": "About the same"}[c] + f": zeroing {name} changes the lead {da:+.1f} bits in A and {db:+.1f} in B."
        sides = [side(A, ma, "zero"), side(B, mb, "zero")]
        meas = {"change_a": da, "change_b": db}
    elif kind == "pair_flip":
        if "steer" not in ma or "steer" not in mb:
            return None
        fa, fb = ma["steer"]["flip"], mb["steer"]["flip"]
        c = both_class(fa, fb)
        q = (f"{head}If {name} is turned up, does the model's answer change to another word in A, in B, in both or in "
             f"neither?")
        a = {"both": "In both", "A": "In A only", "B": "In B only", "neither": "In neither"}[c] + \
            f": the lead changes {ma['steer']['change']:+.1f} bits in A and {mb['steer']['change']:+.1f} in B."
        sides = [side(A, ma, {"steer": oq.STEER}), side(B, mb, {"steer": oq.STEER})]
        meas = {"change_a": ma["steer"]["change"], "change_b": mb["steer"]["change"], "flip_a": fa, "flip_b": fb}
    else:
        sa, sb = supports(ma), supports(mb)
        c = both_class(sa, sb)
        if c is None:
            return None
        q = f"{head}Does {name} support the model's answer (zeroing it lowers the lead by at least 0.35 bits) in A, in B, in both or in neither?"
        a = {"both": "In both", "A": "In A only", "B": "In B only", "neither": "In neither"}[c] + \
            f": zeroing it changes the lead {ma['zero']['change']:+.1f} bits in A and {mb['zero']['change']:+.1f} in B."
        sides = [side(A, ma, "zero"), side(B, mb, "zero")]
        meas = {"change_a": ma["zero"]["change"], "change_b": mb["zero"]["change"]}
    return {"kind": kind, "key": key, "unit_kind": ma["kind"], "name": name, "units_a": ma["units"], "units_b": mb["units"],
            "context_a": A["context_id"], "context_b": B["context_id"], "circuit_a": A["circuit"]["circuit_id"],
            "circuit_b": B["circuit"]["circuit_id"], "family_a": A["family"], "family_b": B["family"], "q": q, "a": a,
            "gold": c, "cls": f"{kind}/{c}", "sides": sides, "measured": meas, "contrast": CONTRAST}


def build(a) -> None:
    texts = {}
    for p in a.measure:
        for x in p.read_text().splitlines():
            t = json.loads(x)
            texts[t["context_id"]] = t
    excl, why = set(), collections.Counter()
    for p in a.exclude_keys or []:
        m = json.loads(p.read_text())
        excl |= set(m["o1"]["dev_keys"] if "o1" in m else m.get("dev_keys", []))
    for p in a.sealed or []:
        excl |= {f"{q['task']}/{q['key']}" for q in json.loads(p.read_text())["questions"]}
    for cid in list(texts):
        t = texts[cid]
        if t["task_key"] in excl:
            why["excluded_task_key"] += 1
            del texts[cid]
        elif zlib.crc32(("scorecard:" + cid).encode()) % 20 == 0:
            why["scorecard_crc"] += 1
            del texts[cid]
    fams = sorted({t["family"] for t in texts.values()})
    rng = random.Random(a.seed)
    size = collections.Counter(t["family"] for t in texts.values())
    split_of, got = {}, {"test": 0, "dev": 0}
    for f in sorted(fams, key=lambda f: zlib.crc32(f"{a.seed}/fam/{f}".encode())):  # whole families, by hash, until each
        for s_, share in (("test", a.test_share), ("dev", a.dev_share)):           # held-out split holds its share of texts
            if got[s_] < share * len(texts):
                split_of[f], got[s_] = s_, got[s_] + size[f]
                break
    by_key = collections.defaultdict(list)
    for cid, t in sorted(texts.items()):
        for k in t["items"]:
            by_key[k].append(cid)
    pools = collections.defaultdict(list)  # (split, kind, class) -> questions
    for key, cids in sorted(by_key.items()):
        if len(cids) < 2:
            continue
        cand = list(itertools.combinations(cids, 2))
        rng.shuffle(cand)
        for x, y in cand[:PER_KEY]:
            sx, sy = split_of.get(texts[x]["family"], "train"), split_of.get(texts[y]["family"], "train")
            if sx != sy:
                continue  # a pair lies within one split: test pairs are families never trained on
            if rng.random() < 0.5:
                x, y = y, x
            for kind in ("pair_zero", "pair_flip", "pair_sign"):
                q_ = question(kind, key, texts[x], texts[y])
                if q_:
                    q_["split"] = sx
                    pools[(sx, kind, q_["gold"])].append(q_)
    out = collections.defaultdict(list)
    for s in ("train", "dev", "test"):
        for kind in ("pair_zero", "pair_flip", "pair_sign"):
            cl = {c: v for (s_, k_, c), v in pools.items() if s_ == s and k_ == kind}
            if len(cl) < 2:
                continue
            n = min(len(v) for v in cl.values())  # every class equally often: the name (the same in A and B) is at chance
            for c, v in sorted(cl.items()):
                out[s] += rng.sample(v, n)
        rng.shuffle(out[s])
        for i, q_ in enumerate(out[s]):
            q_["id"] = f"p1/{s}/{q_['kind']}/{zlib.crc32((q_['key'] + q_['context_a'] + q_['context_b'] + q_['kind']).encode()):08x}"
    a.out.mkdir(parents=True, exist_ok=True)
    used = set()
    for s, rs in out.items():
        with (a.out / f"{s}.jsonl").open("w") as fh:
            for q_ in rs:
                fh.write(json.dumps(q_) + "\n")
                used |= {q_["context_a"], q_["context_b"]}
    with (a.out / "circuits.jsonl").open("w") as fh:
        for cid in sorted(used):
            fh.write(json.dumps(texts[cid]["circuit"]) + "\n")
    count = lambda rs: dict(collections.Counter(q_["cls"] for q_ in rs).most_common())  # noqa: E731
    man = {"schema": "pair1", "contrast": CONTRAST, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "seed": a.seed,
           "inputs": [str(p) for p in a.measure], "texts": len(texts), "removed": dict(why), "families": split_of,
           "train_families": [f for f in fams if f not in split_of], "rows": {s: len(rs) for s, rs in out.items()},
           "classes": {s: count(rs) for s, rs in out.items()},
           "unit_kinds": {s: dict(collections.Counter(q_["unit_kind"] for q_ in rs)) for s, rs in out.items()},
           "available_before_balance": {f"{s}|{k}|{c}": len(v) for (s, k, c), v in sorted(pools.items())},
           "rule": "pairs within one split; test and dev families never trained on; every kind's classes balanced per split; "
                   "texts of O1 dev task/keys, the sealed testset_8b task/keys and the scorecard crc ids removed"}
    (a.out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("texts", "removed", "families", "rows")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("records", type=Path, nargs="+")
    m.add_argument("out", type=Path)
    m.add_argument("--thr", type=Path, required=True)
    m.add_argument("--groups", type=Path, nargs="*")
    m.add_argument("--source", default="finder")
    m.add_argument("--model", default="Qwen/Qwen3-8B")
    m.add_argument("--shard")
    m.add_argument("--device")
    m.add_argument("--dtype", default="bfloat16")
    m.add_argument("--replay", type=int, default=20, help="one text in N replayed through claim_check (0: none)")
    b = sub.add_parser("build")
    b.add_argument("out", type=Path)
    b.add_argument("measure", type=Path, nargs="+")
    b.add_argument("--exclude-keys", type=Path, nargs="*")
    b.add_argument("--sealed", type=Path, nargs="*")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--test-share", type=float, default=0.2, help="texts in held-out test families (whole families)")
    b.add_argument("--dev-share", type=float, default=0.08)
    a = ap.parse_args()
    {"measure": measure, "build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
