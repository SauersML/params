"""Oracle questions that trees on the circuit's numbers cannot answer column by column (the lead's 10-17: the oracle only
matched decision trees because every question so far was tabular). Two sets, measured on Qwen3-8B in float32 with
answer_contrast's word-level contrast (pairs.py's), split by family as pairs/v2 (pairs/v1's test and dev families held out):

MULTI-KIND (build-kinds): pairs.py's paired form (the same item in the circuits of texts A and B; in which text does the
intervention lower the model's lead for its answer more: A | B by GAP bits | about the same within LOW bits), each
question stating its intervention in words:
  zero       the unit's activation set to 0 (a pole's values; massive values stay live, claim_check's zero)
  patch      its activation replaced by its value on the text's counterpart prompt (the circuit's corrupted text)
  push_up    STEER times its calibration maximum added along its pole's sign (neurons)
  flip       its activation multiplied by -1 (claim_check's scale -1)                   held out: dev and test only
  push_down  STEER times its calibration maximum taken off against its pole's sign      held out: dev and test only
Training holds zero, patch and push_up, so a reader answers the held-out kinds from their stated meaning.

NEW-PROMPT (build-carry): the circuit is prompt A's; the question asks what an intervention on one of its units does on
another prompt B of the same task (another key, another wording, or both), measured on B. The unit acts at B's token
aligned to its token in A (difflib on the token strings; in a replaced span the first token goes to the span's first, the
last to its last), and the lead for B's answer drops by HIGH bits or more, rises by HIGH or more, or stays within LOW bits.
Kinds zero, patch (from B's own counterpart prompt) and push_up. v1: classes balanced per split and kind; v1_hard:
balanced within each class of A's own measured change of the same unit (the rule "B does what A does" at chance).

  kinds_carry.py measure MEASURE.jsonl OUT.jsonl --thr THR.pt [--shard K/N]   (GPU, float32) every run of both sets
  kinds_carry.py build-kinds OUT_DIR RAW.jsonl [...] --measure MEASURE.jsonl --families-from MANIFEST.json
  kinds_carry.py build-carry OUT_DIR RAW.jsonl [...] --measure MEASURE.jsonl --families-from MANIFEST.json
  kinds_carry.py baselines OUT_DIR   single-field rules and trees on the circuit's numbers, into OUT_DIR/manifest.json
MEASURE.jsonl is pairs.py measure's output (each text's circuit record and recurring items).
"""
from __future__ import annotations

import argparse
import collections
import difflib
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
import pairs  # noqa: E402

KINDS = ("zero", "patch", "push_up", "flip", "push_down")
TRAIN_KINDS = ("zero", "patch", "push_up")
CARRY_KINDS = ("zero", "patch", "push_up")
NEURON_ONLY = ("push_up", "push_down")
OWN_N, OWN_H = 7, 3        # a text's circuit units carried to other prompts: neurons and heads, drawn at random
CARRY_FROM = (("wording", 2), ("key", 2), ("both", 1))  # prompts A per prompt B by relation: same key, same wording, neither
PER_KEY = pairs.PER_KEY
CAP = 300                  # dev and test questions per kind and class
CONTRAST = pairs.CONTRAST


def texts_of(paths: list[Path]) -> dict:
    out = {}
    for p in paths:
        for x in p.read_text().splitlines():
            t = json.loads(x)
            out[t["context_id"]] = t
    return out


def relation(a: str, b: str) -> str:
    _, ka, pa = a.split("/")
    _, kb, pb = b.split("/")
    return "wording" if ka == kb else "key" if pa == pb else "both"


def align(ta: list[str], tb: list[str]) -> dict:
    """A's positions -> (B's position, "same" | "replaced"): equal tokens one to one; in a replaced span the first token
    to the span's first, the last to its last, the rest in proportion; deleted tokens have none."""
    m = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k] = (j1 + k, "same")
        elif tag == "replace":
            for i in range(i1, i2):
                j = j2 - 1 if i == i2 - 1 else j1 if i == i1 else j1 + round((i - i1) * (j2 - 1 - j1) / max(1, i2 - 1 - i1))
                m[i] = (j, "replaced")
    return m


def carry_plan(texts: dict, massive) -> tuple[dict, dict]:
    """own: text -> its sampled circuit units (OWN_N neurons outside massive values with a lost share, OWN_H heads);
    plan: B -> [(A, relation)] by CARRY_FROM, A in B's family."""
    own = {}
    for cid, t in sorted(texts.items()):
        c = t["circuit"]
        rng = random.Random(zlib.crc32(("own/" + cid).encode()))
        ns = [u["name"] for u in c["units"] if u["pos"] > 0 and u.get("lost") is not None and u.get("value") is not None
              and not massive(u)]
        hs = [h["name"] for h in c["heads"] if h["pos"] > 0]
        own[cid] = rng.sample(ns, min(OWN_N, len(ns))) + rng.sample(hs, min(OWN_H, len(hs)))
    by_fam = collections.defaultdict(list)
    for cid, t in sorted(texts.items()):
        by_fam[t["family"]].append(cid)
    plan = {}
    for cid, t in sorted(texts.items()):
        rng = random.Random(zlib.crc32(("carry/" + cid).encode()))
        got = []
        for rel, n in CARRY_FROM:
            cand = [a for a in by_fam[t["family"]] if a != cid and relation(a, cid) == rel]
            got += [(a, rel) for a in rng.sample(cand, min(n, len(cand)))]
        plan[cid] = got
    return own, plan


def base_name(u: str) -> str:
    return u.split("@")[0]


def measure(a) -> None:
    tok, nr, cc, eng = pairs.model_of(a)
    import torch
    texts = texts_of([a.measure])
    unit_of = {cid: {u["name"]: u for u in t["circuit"]["units"]} for cid, t in texts.items()}
    massive = lambda u: abs(u["value"] or 0.0) >= float(nr.massive[u["layer"]])  # noqa: E731
    own, plan = carry_plan(texts, massive)
    todo = sorted(texts)
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        todo = todo[k::n]
    done = {json.loads(x)["context_id"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    t0 = time.time()
    with a.out.open("a") as fh:
        for j, cid in enumerate(todo):
            if cid in done:
                continue
            t = texts[cid]
            ids, T = t["ids"], len(t["ids"])
            cor = t["circuit"]["corrupt"]["ids"]
            c_ = {}
            nr.logits(list(cor), [len(cor) - 1], capture=c_)
            src = {"h": [h[0] for h in c_["h"]], "o": [o[0] for o in c_["o"]]}
            ops = {"zero": "zero", "patch": ("patch", src, [(p, p) for p in range(1, T)]), "flip": ("scale", -1.0),
                   "push_up": ("steer", oq.STEER), "push_down": ("steer", -oq.STEER)}
            rows, at = [{}], {}

            def want(units: list[str], kind: str, head: bool):
                if head and kind in NEURON_ONLY:
                    return
                k_ = (tuple(units), kind)
                if k_ not in at:
                    at[k_] = len(rows)
                    rows.append({"units": cc.units_of(nr, units, T), "op": ops[kind]})
            for key, it in t["items"].items():  # multi-kind: the recurring items, every kind
                for kind in KINDS:
                    want(it["units"], kind, it["kind"] == "head")
            for u in own[cid]:  # new-prompt, A's side: the text's own sampled units at their positions
                for kind in CARRY_KINDS:
                    want([u], kind, ".H" in u)
            carried = []
            for src_id, rel in plan[cid]:  # new-prompt, B's side: each A's sampled units at B's aligned tokens
                A = texts[src_id]
                m = align(A["circuit"]["tokens"], t["circuit"]["tokens"])
                for u in own[src_id]:
                    pa = int(u.split("@")[1])
                    if pa not in m or m[pa][0] < 1:
                        continue
                    pb, how = m[pa]
                    ub = f"{base_name(u)}@{pb}"
                    carried.append({"from": src_id, "relation": rel, "unit_a": u, "unit_b": ub, "pos_a": pa, "pos_b": pb,
                                    "align": how})
                    for kind in CARRY_KINDS:
                        want([ub], kind, ".H" in u)
            lp = eng.answer_lp(ids, rows)
            A_, O_ = t["answer_tokens"], t["alternatives"]
            cs = (torch.logsumexp(lp[:, A_], -1) - torch.logsumexp(lp[:, O_], -1)) / math.log(2)
            top = lp.argmax(-1).tolist()
            res = lambda r: {"change": float(cs[r] - cs[0]), "top": top[r], "flip": top[r] not in A_}  # noqa: E731
            items = {key: {kind: res(at[(tuple(it["units"]), kind)]) for kind in KINDS if (tuple(it["units"]), kind) in at}
                     for key, it in t["items"].items()}
            own_r = {u: {kind: res(at[((u,), kind)]) for kind in CARRY_KINDS if ((u,), kind) in at} for u in own[cid]}
            for c in carried:
                c.update({kind: res(at[((c["unit_b"],), kind)]) for kind in CARRY_KINDS if ((c["unit_b"],), kind) in at})
            if a.replay and zlib.crc32(cid.encode()) % a.replay == 0:  # claim_check's numbers for a sample of the runs
                op_d = {"zero": "zero", "patch": {"patch_from": {"ids": cor}}, "flip": {"scale": -1.0},
                        "push_up": {"steer": oq.STEER}, "push_down": {"steer": -oq.STEER}}
                ks = list(at.items())[:: max(1, len(at) // 40)]
                cl = [{"prompt": {"ids": ids}, "position": -1, "units": list(us), "op": op_d[kind], "answer": A_, "alternatives": O_,
                       "predict": pairs.ALWAYS_} for (us, kind), _ in ks]
                got = cc.check_many(nr, cl, controls=0)
                diff = max(abs(g["change"] - float(cs[r] - cs[0])) for g, (_, r) in zip(got, ks))
                print(json.dumps({"replay": cid, "claims": len(cl), "max_abs_diff_bits": diff}), flush=True)
            fh.write(json.dumps({"context_id": cid, "family": t["family"], "dtype": a.dtype, "contrast": CONTRAST,
                                 "contrast_before": float(cs[0]), "top_before": top[0], "rows": len(rows), "items": items,
                                 "own": own_r, "carried": carried}) + "\n")
            fh.flush()
            if (j + 1) % 20 == 0:
                print(json.dumps({"texts": j + 1, "of": len(todo), "seconds": round(time.time() - t0, 1)}), flush=True)
    print(json.dumps({"done": len(todo), "seconds": round(time.time() - t0, 1), "out": str(a.out)}), flush=True)


def excluded(texts: dict, a) -> collections.Counter:
    """pairs.py build's exclusions (O1 dev task/keys, the sealed testset_8b task/keys, the scorecard crc ids), applied."""
    excl, why = set(), collections.Counter()
    for p in a.exclude_keys or []:
        m = json.loads(p.read_text())
        excl |= set(m) if isinstance(m, list) else set(m["o1"]["dev_keys"] if "o1" in m else m.get("dev_keys", []))
    for p in a.sealed or []:
        excl |= {f"{q['task']}/{q['key']}" for q in json.loads(p.read_text())["questions"]}
    for cid in list(texts):
        if texts[cid]["task_key"] in excl:
            why["excluded_task_key"] += 1
            del texts[cid]
        elif zlib.crc32(("scorecard:" + cid).encode()) % 20 == 0:
            why["scorecard_crc"] += 1
            del texts[cid]
    return why


def phrase(kind: str, name: str, unit_kind: str, counterpart: str = "") -> str:
    what = {"head": "output", "group": "member activations"}.get(unit_kind, "activation")
    its = "each member's" if unit_kind == "group" else "its"
    return {"zero": f"setting the {what} of {name} to zero",
            "patch": f"replacing the {what} of {name} with {'the values they take' if unit_kind == 'group' else 'the value it takes'} "
                     f"on the counterpart prompt{counterpart}",
            "flip": f"flipping the sign of the {what} of {name} (multiplying by -1)",
            "push_up": f"pushing up {name} (adding {oq.STEER:g} times {its} calibration maximum along its sign, making the "
                       f"activation stronger)",
            "push_down": f"pushing down {name} (taking {oq.STEER:g} times {its} calibration maximum off against its sign, "
                         f"past zero to the opposite sign)"}[kind]


OP_CLAIM = {"zero": lambda t: "zero", "patch": lambda t: {"patch_from": {"ids": t["circuit"]["corrupt"]["ids"]}},
            "flip": lambda t: {"scale": -1.0}, "push_up": lambda t: {"steer": oq.STEER}, "push_down": lambda t: {"steer": -oq.STEER}}


def claim(t: dict, units: list[str], kind: str) -> dict:
    return {"prompt": {"ids": t["ids"]}, "position": -1, "units": units, "op": OP_CLAIM[kind](t), "predict": pairs.ALWAYS_,
            "answer": t["answer_tokens"], "alternatives": t["alternatives"]}


def balance(pools: dict, rng: random.Random, cap: int | None) -> list:
    """pools: class -> questions; every class equally often (at most cap each)."""
    if len(pools) < 2:
        return []
    n = min(len(v) for v in pools.values())
    n = n if cap is None else min(n, cap)
    return [q for c in sorted(pools) for q in rng.sample(pools[c], n)]


def load_raw(paths: list[Path]) -> dict:
    out = {}
    for p in paths:
        for x in p.read_text().splitlines():
            r = json.loads(x)
            out[r["context_id"]] = r
    return out


def build_kinds(a) -> None:
    texts, raw = texts_of(a.measure), load_raw(a.raw)
    why = excluded(texts, a)
    texts = {c: t for c, t in texts.items() if c in raw}
    split_of = json.loads(a.families_from.read_text())["families"]
    rng = random.Random(a.seed)
    by_key = collections.defaultdict(list)
    for cid, t in sorted(texts.items()):
        for k in t["items"]:
            by_key[k].append(cid)
    pools = collections.defaultdict(list)  # (split, kind, gold) -> questions
    for key, cids in sorted(by_key.items()):
        cand = list(itertools.combinations(cids, 2))
        rng.shuffle(cand)
        for x, y in cand[:PER_KEY]:
            s = split_of.get(texts[x]["family"], "train")
            if s != split_of.get(texts[y]["family"], "train"):
                continue
            if rng.random() < 0.5:
                x, y = y, x
            A, B = texts[x], texts[y]
            ma, mb = A["items"][key], B["items"][key]
            for kind in KINDS if s != "train" else TRAIN_KINDS:
                ra, rb = raw[x]["items"].get(key, {}).get(kind), raw[y]["items"].get(key, {}).get(kind)
                if ra is None or rb is None:
                    continue
                da, db = ra["change"], rb["change"]
                g = pairs.zero_class(da, db)
                if g is None:
                    continue
                ya, yb = oq.q_tok(A["answer_text"] or ""), oq.q_tok(B["answer_text"] or "")
                head = f"Text A: \"{A['user']}\" (the model answers {ya}). Text B: \"{B['user']}\" (the model answers {yb}). "
                name = key
                if ma["kind"] != "group":
                    pa, pb = int(ma["units"][0].split("@")[1]), int(mb["units"][0].split("@")[1])
                    name = (f"{key} (at {oq.q_tok(A['circuit']['tokens'][pa])} @{pa} in A, "
                            f"{oq.q_tok(B['circuit']['tokens'][pb])} @{pb} in B)")
                cp = (f" (A's: \"{A['circuit']['corrupt']['user']}\"; B's: \"{B['circuit']['corrupt']['user']}\")"
                      if kind == "patch" else "")
                q = (f"{head}In which text does {phrase(kind, name, ma['kind'], cp)} lower the model's lead for "
                     f"its answer more: A, B, or about the same?")
                ans = {"A": "A", "B": "B", "same": "About the same"}[g] + f": the lead changes {da:+.1f} bits in A and {db:+.1f} in B."
                pools[(s, kind, g)].append({
                    "kind": "pair_" + kind, "op": kind, "held_out_kind": kind not in TRAIN_KINDS, "key": key,
                    "unit_kind": ma["kind"], "name": name, "units_a": ma["units"], "units_b": mb["units"],
                    "context_a": x, "context_b": y, "circuit_a": A["circuit"]["circuit_id"], "circuit_b": B["circuit"]["circuit_id"],
                    "family_a": A["family"], "family_b": B["family"], "q": q, "a": ans, "gold": g, "cls": f"pair_{kind}/{g}",
                    "sides": [claim(A, ma["units"], kind), claim(B, mb["units"], kind)], "contrast": CONTRAST,
                    "dtype": raw[x]["dtype"], "split": s,
                    "measured": {"change_a": da, "change_b": db, "flip_a": ra["flip"], "flip_b": rb["flip"],
                                 "zero_a": raw[x]["items"][key]["zero"]["change"], "zero_b": raw[y]["items"][key]["zero"]["change"],
                                 **({"push_up_a": raw[x]["items"][key]["push_up"]["change"],
                                     "push_up_b": raw[y]["items"][key]["push_up"]["change"]} if "push_up" in raw[x]["items"][key] else {})}})
    out = collections.defaultdict(list)
    for s in ("train", "dev", "test"):
        for kind in KINDS:
            out[s] += balance({g: v for (s_, k_, g), v in pools.items() if s_ == s and k_ == kind}, rng,
                              None if s == "train" else a.cap)
        rng.shuffle(out[s])
        for q in out[s]:
            q["id"] = f"mk1/{s}/{q['op']}/{zlib.crc32((q['key'] + q['context_a'] + q['context_b'] + q['op']).encode()):08x}"
    write(a.out, out, texts, {
        "schema": "multikind1", "contrast": CONTRAST, "seed": a.seed, "inputs": [str(p) for p in a.raw],
        "texts": len(texts), "removed": dict(why), "families": split_of, "train_kinds": list(TRAIN_KINDS),
        "held_out_kinds": [k for k in KINDS if k not in TRAIN_KINDS], "cap_per_class": a.cap,
        "available_before_balance": {f"{s}|{k}|{g}": len(v) for (s, k, g), v in sorted(pools.items())},
        "rule": "pairs within one split (test and dev families never trained on); train holds zero, patch and push_up only, "
                "dev and test every kind; each kind's classes balanced per split (dev and test capped per class); gold = "
                "pairs.zero_class of the two measured changes (A or B by at least GAP bits, same within LOW)"})


def cls3(ch: float) -> str | None:
    return "drop" if ch <= -oq.HIGH else "rise" if ch >= oq.HIGH else "none" if abs(ch) < oq.LOW else None


def build_carry(a) -> None:
    texts, raw = texts_of(a.measure), load_raw(a.raw)
    why = excluded(texts, a)
    texts = {c: t for c, t in texts.items() if c in raw}
    split_of = json.loads(a.families_from.read_text())["families"]
    pools = collections.defaultdict(list)  # (split, kind) -> questions
    for bid, r in sorted(raw.items()):
        if bid not in texts:
            continue
        B = texts[bid]
        s = split_of.get(B["family"], "train")
        for c in r["carried"]:
            if c["from"] not in texts:
                continue
            A = texts[c["from"]]
            ua = {u["name"]: u for u in A["circuit"]["units"]} | {h["name"]: h for h in A["circuit"]["heads"]}
            head = ".H" in c["unit_a"]
            unit_kind = "head" if head else "neuron"
            for kind in CARRY_KINDS:
                if kind not in c:
                    continue
                g = cls3(c[kind]["change"])
                if g is None:
                    continue
                own = raw[c["from"]]["own"].get(c["unit_a"], {}).get(kind)
                ga = (cls3(own["change"]) or "gap") if own else "gap"
                ya, yb = oq.q_tok(A["answer_text"] or ""), oq.q_tok(B["answer_text"] or "")
                pa, pb = c["pos_a"], c["pos_b"]
                name = (f"{base_name(c['unit_a'])} (in A at {oq.q_tok(A['circuit']['tokens'][pa])} @{pa}; in B at "
                        f"{oq.q_tok(B['circuit']['tokens'][pb])} @{pb})")
                cp = f" of B, \"{B['circuit']['corrupt']['user']}\"" if kind == "patch" else ""
                q = (f"The circuit is prompt A's: \"{A['user']}\" (the model answers {ya}). Prompt B: \"{B['user']}\" (the "
                     f"model answers {yb}). In B, {phrase(kind, name, unit_kind, cp)}: does the model's lead for its answer drop "
                     f"by at least {oq.HIGH:g} bits, rise by at least {oq.HIGH:g} bits, or change by less than {oq.LOW:g} bits?")
                ch = c[kind]["change"]
                ans = {"drop": "It drops", "rise": "It rises", "none": "It barely changes"}[g] + f": the lead changes {ch:+.1f} bits in B."
                pools[(s, kind)].append({
                    "kind": "carry_" + kind, "op": kind, "unit_a": c["unit_a"], "unit_b": c["unit_b"],
                    "unit_kind": "head" if head else "neuron", "relation": c["relation"], "align": c["align"],
                    "context_a": c["from"], "context_b": bid, "circuit_a": A["circuit"]["circuit_id"],
                    "family": B["family"], "q": q, "a": ans, "gold": g, "cls": f"carry_{kind}/{g}", "a_class": ga,
                    "sides": [claim(B, [c["unit_b"]], kind)], "contrast": CONTRAST, "dtype": r["dtype"], "split": s,
                    "measured": {"change_b": ch, "flip_b": c[kind]["flip"], "change_a": own["change"] if own else None,
                                 "lost_a": ua.get(c["unit_a"], {}).get("lost")}})
    rng = random.Random(a.seed)
    for variant in ("v1", "v1_hard"):
        out = collections.defaultdict(list)
        for s in ("train", "dev", "test"):
            for kind in CARRY_KINDS:
                qs = pools.get((s, kind), [])
                if variant == "v1":
                    by = collections.defaultdict(list)
                    for q in qs:
                        by[q["gold"]].append(q)
                    out[s] += balance(by, rng, None if s == "train" else a.cap)
                else:  # gold balanced within each of A's own classes (the carry rule and any exclusion of it at chance)
                    pat = collections.defaultdict(lambda: collections.defaultdict(list))
                    for q in qs:
                        pat[q["a_class"]][q["gold"]].append(q)
                    for ga, by in sorted(pat.items()):
                        if len(by) == 3:
                            out[s] += balance(by, rng, None if s == "train" else a.cap // 2)
            rng.shuffle(out[s])
            for q in out[s]:
                q["id"] = f"cr1/{s}/{q['op']}/{zlib.crc32((q['unit_a'] + q['context_a'] + q['context_b'] + q['op']).encode()):08x}"
        write(a.out / variant, out, texts, {
            "schema": "carry1", "variant": variant, "contrast": CONTRAST, "seed": a.seed, "inputs": [str(p) for p in a.raw],
            "texts": len(texts), "removed": dict(why), "families": split_of, "kinds": list(CARRY_KINDS),
            "carry_from": dict(CARRY_FROM), "cap_per_class": a.cap,
            "available_before_balance": {f"{s}|{k}|{g}": n for (s, k), v in sorted(pools.items())
                                         for g, n in sorted(collections.Counter(q["gold"] for q in v).items())},
            "rule": "B is another prompt of A's family (same split); A's unit acts at B's aligned token; gold = B's measured "
                    "change: drop (<= -HIGH bits), rise (>= HIGH), none (|change| < LOW); " +
                    ("classes balanced per split and kind" if variant == "v1" else
                     "classes balanced per split and kind within each class of A's own measured change of the same unit "
                     "(drop, rise, none or gap)")})


def write(out_dir: Path, out: dict, texts: dict, man: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    used = set()
    for s, qs in out.items():
        with (out_dir / f"{s}.jsonl").open("w") as fh:
            for q in qs:
                fh.write(json.dumps(q) + "\n")
                used |= {q["context_a"], q.get("context_b")} - {None}
    with (out_dir / "circuits.jsonl").open("w") as fh:
        for cid in sorted(used):
            fh.write(json.dumps(texts[cid]["circuit"]) + "\n")
    man |= {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rows": {s: len(v) for s, v in out.items()},
            "classes": {s: dict(collections.Counter(q["cls"] for q in v).most_common()) for s, v in out.items()}}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("rows", "removed")}))


NUMS = ("value", "corrupt", "value_over_max", "lost", "restored", "direct")


def item_feats(names: list[str], c: dict, n_layers: int = 36) -> list[float]:
    """Every number of the record on an item's units (neurons: layer, stage, pole, final position, each field's sum,
    largest by size and summed size, corrupt minus value; heads: layer, lost, restored, top attention weight) and the
    text's clean and corrupted contrasts, gap and length."""
    units = {u["name"]: u for u in c["units"]}
    heads = {h["name"]: h for h in c["heads"]}
    T = len(c["tokens"])
    us, hs = [units[n] for n in names if n in units], [heads[n] for n in names if n in heads]
    f = [len(us), len(hs)]
    lay = [u["layer"] / n_layers for u in us] or [0.0]
    f += [min(lay), max(lay), sum(u.get("stage") == "source" for u in us), sum(u.get("stage") == "readout" for u in us),
          sum(T - 1 - u["pos"] == 0 for u in us), sum(u.get("pole", 0) > 0 for u in us),
          min([T - 1 - u["pos"] for u in us] or [0])]
    for k in NUMS + ("shift",):
        if k == "shift":
            v = [u["corrupt"] - u["value"] for u in us if u.get("corrupt") is not None and u.get("value") is not None]
        else:
            v = [u[k] for u in us if u.get(k) is not None]
        f += [sum(v), max(v, key=abs) if v else 0.0, sum(abs(x) for x in v)]
    hl = [h["layer"] / n_layers for h in hs] or [0.0]
    f += [min(hl), max(hl), min([T - 1 - h["pos"] for h in hs] or [0])]
    for k in ("lost", "restored"):
        v = [h[k] for h in hs if h.get(k) is not None]
        f += [sum(v), max(v, key=abs) if v else 0.0]
    f += [sum(a_[1] for h in hs for a_ in (h.get("attends") or [])[:1])]
    con = c.get("contrast") or {}
    f += [con.get("clean") or 0.0, con.get("corrupt") or 0.0, c.get("gap") or 0.0, T]
    return [float(x) for x in f]


def names_feats(names: list[str], c: dict) -> list[float]:
    """Names only: each unit's layer, kind and position from the end (no measured number)."""
    T = len(c["tokens"])
    ps = [int(n.split("@")[1]) for n in names if "@" in n]
    lay = [int(n.split(".")[0][1:]) for n in names]
    return [len(names), sum(".H" in n for n in names), min(lay), max(lay), min([T - 1 - p for p in ps] or [0]), T]


def baselines(a) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    d = a.dir
    man = json.loads((d / "manifest.json").read_text())
    cs = {c["circuit_id"]: c for c in map(json.loads, (d / "circuits.jsonl").read_text().splitlines())}
    tr = [json.loads(x) for x in (d / "train.jsonl").read_text().splitlines()]
    te = [json.loads(x) for x in (d / "test.jsonl").read_text().splitlines()]
    acc = lambda p, qs: round(sum(x == q["gold"] for x, q in zip(p, qs)) / len(qs), 3) if qs else None  # noqa: E731
    gb = lambda: HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0)  # noqa: E731
    res = collections.defaultdict(dict)
    if man["schema"] == "multikind1":
        onehot = lambda q: [float(q["op"] == k) for k in TRAIN_KINDS]  # noqa: E731 (a held-out kind: all zeros)

        def X(q, names=False):
            fa = (names_feats if names else item_feats)(q["units_a"], cs[q["circuit_a"]])
            fb = (names_feats if names else item_feats)(q["units_b"], cs[q["circuit_b"]])
            return fa + fb + [x - y for x, y in zip(fa, fb)]
        Xtr = [X(q) for q in tr]
        ytr = [q["gold"] for q in tr]
        m_hot = gb().fit([x + onehot(q) for x, q in zip(Xtr, tr)], ytr)
        m_pool = gb().fit(Xtr, ytr)
        per_kind = {k: gb().fit([x for x, q in zip(Xtr, tr) if q["op"] == k], [q["gold"] for q in tr if q["op"] == k])
                    for k in TRAIN_KINDS}
        m_names = gb().fit([X(q, True) + onehot(q) for q in tr], ytr)
        nf = len(Xtr[0]) // 3
        for kind in KINDS:
            qs = [q for q in te if q["op"] == kind]
            if not qs:
                continue
            Xte = [X(q) for q in qs]
            r = {"n": len(qs), "chance": 0.333, "held_out": kind not in TRAIN_KINDS,
                 "tree_kind_onehot": acc(m_hot.predict([x + onehot(q) for x, q in zip(Xte, qs)]), qs),
                 "tree_pooled_kinds": acc(m_pool.predict(Xte), qs),
                 "tree_names_only": acc(m_names.predict([X(q, True) + onehot(q) for q in qs]), qs)}
            for k in TRAIN_KINDS:  # each train kind's own tree applied to this kind (for a held-out kind: the mapping a
                r[f"tree_of_{k}"] = acc(per_kind[k].predict(Xte), qs)  # person would choose, e.g. flip from zero)
            fams = sorted({q["family_a"] for q in qs})  # labels of this kind from the other test families (an upper
            if len(fams) > 1:                           # reference: how tabular the kind is once its labels are seen)
                hit = 0
                for f in fams:
                    trn = [(x, q["gold"]) for x, q in zip(Xte, qs) if q["family_a"] != f]
                    tst = [(x, q) for x, q in zip(Xte, qs) if q["family_a"] == f]
                    if len({g for _, g in trn}) < 2:
                        continue
                    m = gb().fit([x for x, _ in trn], [g for _, g in trn])
                    hit += sum(p == q["gold"] for p, (_, q) in zip(m.predict([x for x, _ in tst]), tst))
                r["tree_own_labels_other_families"] = round(hit / len(qs), 3)
            best = (None, 0.0)  # single fields: one column (A minus B), a depth-2 tree tuned on all train kinds
            for j in range(nf):
                col = lambda xs: [[x[2 * nf + j]] for x in xs]  # noqa: E731
                m = DecisionTreeClassifier(max_depth=2, random_state=0).fit(col(Xtr), ytr)
                v = acc(m.predict(col(Xte)), qs)
                if v > best[1]:
                    best = (j, v)
            r["best_single_field_train_tuned"] = best[1]
            hidden = {"flip": lambda q: pairs.zero_class(q["measured"]["zero_a"], q["measured"]["zero_b"]) or "same",
                      "push_down": lambda q: (pairs.zero_class(-q["measured"]["push_up_a"], -q["measured"]["push_up_b"]) or "same")
                      if "push_up_a" in q["measured"] else "same"}
            if kind in hidden:  # hidden references: the same pair's zero class (flip ~ twice the zeroing), push_up negated
                r["hidden_composed_rule"] = acc([hidden[kind](q) for q in qs], qs)
            res[kind] = r
    else:
        rels = ("wording", "key", "both")
        by_ctx = {c["context_id"]: c for c in cs.values()}

        def Xf(q, names=False):
            c = cs[q["circuit_a"]]
            pa, pb = int(q["unit_a"].split("@")[1]), int(q["unit_b"].split("@")[1])
            f = (names_feats if names else item_feats)([q["unit_a"]], c)
            f += [float(q["relation"] == r) for r in rels] + [float(q["align"] == "same")] + [float(q["op"] == k) for k in CARRY_KINDS]
            cb = by_ctx.get(q["context_b"])
            f += [float(cb is not None and cb["tokens"][pb] == c["tokens"][pa]), float(len(cb["tokens"]) - 1 - pb) if cb else 0.0]
            return f
        Xtr, ytr = [Xf(q) for q in tr], [q["gold"] for q in tr]
        m_all = gb().fit(Xtr, ytr)
        m_names = gb().fit([Xf(q, True) for q in tr], ytr)
        m_hidden = gb().fit([x + [q["measured"]["change_a"] or 0.0] for x, q in zip(Xtr, tr)], ytr)
        nf = len(item_feats([tr[0]["unit_a"]], cs[tr[0]["circuit_a"]]))
        for kind in CARRY_KINDS:
            qs = [q for q in te if q["op"] == kind]
            if not qs:
                continue
            Xte = [Xf(q) for q in qs]
            r = {"n": len(qs), "chance": 0.333,
                 "carry_rule_a_class": acc([q["a_class"] if q["a_class"] != "gap" else "none" for q in qs], qs),
                 "carry_rule_abstain_on_gap": acc([q["a_class"] for q in qs], qs),
                 "exclusion_of_a_class": round(sum((0.5 if q["a_class"] != "gap" else 1 / 3) * (q["gold"] != q["a_class"])
                                                   for q in qs) / len(qs), 3),
                 "tree_all_numbers": acc(m_all.predict(Xte), qs),
                 "tree_names_only": acc(m_names.predict([Xf(q, True) for q in qs]), qs),
                 "tree_with_hidden_a_change": acc(m_hidden.predict([x + [q["measured"]["change_a"] or 0.0] for x, q in zip(Xte, qs)]), qs)}
            best = (None, 0.0)
            trk = [(x, q["gold"]) for x, q in zip(Xtr, tr) if q["op"] == kind]
            for j in range(nf):
                m = DecisionTreeClassifier(max_depth=2, random_state=0).fit([[x[j]] for x, _ in trk], [g for _, g in trk])
                v = acc(m.predict([[x[j]] for x in Xte]), qs)
                if v > best[1]:
                    best = (j, v)
            r["best_single_field_train_tuned"] = best[1]
            for rel in rels:
                sub = [q for q in qs if q["relation"] == rel]
                r[f"n_{rel}"] = len(sub)
            res[kind] = r
    every = tr + te + ([json.loads(x) for x in (d / "dev.jsonl").read_text().splitlines()] if (d / "dev.jsonl").exists() else [])
    ctx = lambda qs: {c for q in qs for c in (q["context_a"], q.get("context_b")) if c}  # noqa: E731
    shown = {k for c in cs.values() for u in c["units"] + c["heads"] for k in u}
    man["checks"] = {  # leaks: shared texts across splits, held-out kinds in training, measured numbers in a question's
        "test_contexts_in_train": len(ctx(te) & ctx(tr)),  # text, measured fields in the circuit records, duplicates
        "held_out_kinds_in_train": sum(q["op"] not in TRAIN_KINDS for q in tr) if man["schema"] == "multikind1" else None,
        "measured_change_in_question": sum(f"{v:+.1f}" in q["q"] for q in every for k, v in q["measured"].items()
                                           if k.startswith("change") and v is not None),
        "circuit_unit_fields": sorted(shown), "duplicate_ids": len(every) - len({q["id"] for q in every})}
    man["baselines_test"] = res
    (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("measure", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--thr", type=Path, required=True)
    m.add_argument("--model", default="Qwen/Qwen3-8B")
    m.add_argument("--dtype", default="float32")
    m.add_argument("--device")
    m.add_argument("--shard")
    m.add_argument("--replay", type=int, default=25, help="one text in N: a sample of its runs through claim_check (0: none)")
    for name in ("build-kinds", "build-carry"):
        b = sub.add_parser(name)
        b.add_argument("out", type=Path)
        b.add_argument("raw", type=Path, nargs="+")
        b.add_argument("--measure", type=Path, nargs="+", required=True)
        b.add_argument("--families-from", type=Path, required=True)
        b.add_argument("--exclude-keys", type=Path, nargs="*")
        b.add_argument("--sealed", type=Path, nargs="*")
        b.add_argument("--seed", type=int, default=1)
        b.add_argument("--cap", type=int, default=CAP)
    bl = sub.add_parser("baselines")
    bl.add_argument("dir", type=Path)
    a = ap.parse_args()
    {"measure": measure, "build-kinds": build_kinds, "build-carry": build_carry, "baselines": baselines}[a.cmd](a)


if __name__ == "__main__":
    main()
