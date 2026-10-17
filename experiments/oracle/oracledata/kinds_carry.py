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
STRICT_PER_KEY = 400       # strict's dev and test candidates: text pairs per key drawn from the measured texts
STRICT_CAP = 600           # strict's dev and test questions per kind and class (equal per pattern cell)
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
    def make(key: str, x: str, y: str, s: str, into: dict) -> None:
        """The question of each kind on key in texts x (A) and y (B) whose gold is defined, into[(s, kind, gold)]."""
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
            into[(s, kind, g)].append({
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
            make(key, x, y, s, pools)
    wide = collections.defaultdict(list)  # strict's dev and test candidates: up to STRICT_PER_KEY text pairs per key
    rng2 = random.Random(a.seed + 1000)
    for key, cids in sorted(by_key.items()):
        cand = [(x, y) for x, y in itertools.combinations(cids, 2)
                if split_of.get(texts[x]["family"], "train") == split_of.get(texts[y]["family"], "train") != "train"]
        rng2.shuffle(cand)
        for x, y in cand[:STRICT_PER_KEY]:
            make(key, *((y, x) if rng2.random() < 0.5 else (x, y)), split_of[texts[x]["family"]], wide)
    wide_sk = collections.defaultdict(list)
    for (s, kind, g), v in wide.items():
        wide_sk[(s, kind)] += v
    cs = {t["circuit"]["circuit_id"]: t["circuit"] for t in texts.values()}
    by_sk = collections.defaultdict(list)
    for (s, kind, g), v in pools.items():
        by_sk[(s, kind)] += v
    adv_pred(by_sk, lambda q: kind_X(q, cs) + kind_onehot(q), lambda q: q["family_a"])
    outs, survivors = {}, {}
    for variant in ("v1", "adv", "strict"):
        if variant == "strict":  # every known shortcut fitted on v1's training split, then gold balanced within them jointly
            strict_kinds(wide_sk, outs["v1"]["train"], cs)
        out = collections.defaultdict(list)
        outs[variant] = out
        for s in ("train", "dev", "test"):
            if variant == "strict" and s == "train":
                out[s] = list(outs["v1"]["train"])
                continue
            for kind in KINDS:
                if variant == "strict":  # every survivor counted; at most STRICT_CAP per class kept (equal per cell)
                    survivors[f"{s}|{kind}"] = len(pick(wide_sk.get((s, kind), []), variant, KIND_PATTERN[kind], rng, None))
                out[s] += pick((wide_sk if variant == "strict" else by_sk).get((s, kind), []), variant,
                               KIND_PATTERN[kind] if variant == "strict" else (lambda q: q["sel_pred"]), rng,
                               None if s == "train" else STRICT_CAP if variant == "strict" else a.cap)
            rng.shuffle(out[s])
            for q in out[s]:
                q["id"] = kind_id(q, s)
        diag = {name: [q | {"id": kind_id(q, s_)} for q in wide_sk.get((s_, "flip"), [])
                       if q["zero_copy"] in ("A", "B", "same") and q["zero_copy"] != q["gold"]]
                for s_, name in (("test", "flip_diag.jsonl"), ("dev", "flip_diag_dev.jsonl"))} if variant == "strict" else None
        write(a.out / variant, out, texts, {
            "schema": "multikind1", "variant": variant, "contrast": CONTRAST, "seed": a.seed, "inputs": [str(p) for p in a.raw],
            "texts": len(texts), "removed": dict(why), "families": split_of, "train_kinds": list(TRAIN_KINDS),
            "held_out_kinds": [k for k in KINDS if k not in TRAIN_KINDS],
            "cap_per_class": STRICT_CAP if variant == "strict" else a.cap,
            **({"survivors_before_cap": survivors} if variant == "strict" else {}),
            "available_before_balance": {f"{s}|{k}|{g}": len(v) for (s, k, g), v in sorted(
                (wide if variant == "strict" else pools).items()) if variant != "strict" or s != "train"},
            "rule": "pairs within one split (test and dev families never trained on); train holds zero, patch and push_up "
                    "only, dev and test every kind; gold = pairs.zero_class of the two measured changes (A or B by at least "
                    "GAP bits, same within LOW); " + VARIANT_RULE[variant] +
                    ("; strict's committee for every kind: gradient-boosted trees on the numbers with the kind, on the names "
                     "with the kind, on the numbers without the kind, a random forest on the numbers with the kind, the kind's single "
                     "field, a tree on the question's text alone, whether A and B share the key, and each training kind's own "
                     "tree; patch, push_up, flip and push_down add the same pair's zero "
                     "class, flip the class of twice its zeroings, push_down the class of its push_up changes negated "
                     "(margins split by direction); "
                     "flip_diag.jsonl: every test flip question whose gold differs from the same pair's zero class; strict's "
                     "dev and test candidates are up to STRICT_PER_KEY text pairs per key from the same measured texts (v1 "
                     "draws PER_KEY)" if variant == "strict" else "")}, diag)


def kind_id(q: dict, s: str) -> str:
    return f"mk1/{s}/{q['op']}/{zlib.crc32((q['key'] + q['context_a'] + q['context_b'] + q['op']).encode()):08x}"


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
    cs = {t["circuit"]["circuit_id"]: t["circuit"] for t in texts.values()}
    by_ctx = {t["context_id"]: t["circuit"] for t in texts.values()}
    adv_pred(pools, lambda q: carry_X(q, cs, by_ctx), lambda q: q["family"])
    outs, survivors = {}, {}
    for variant in ("v1", "v1_hard", "adv", "strict"):
        if variant == "strict":
            strict_carry(pools, outs["v1"]["train"], cs, by_ctx)
        out = collections.defaultdict(list)
        outs[variant] = out
        for s in ("train", "dev", "test"):
            if variant == "strict" and s == "train":
                out[s] = list(outs["v1"]["train"])
                continue
            for kind in CARRY_KINDS:
                if variant == "strict":  # every survivor counted (per relation too); at most STRICT_CAP per class kept
                    for q in pick(pools.get((s, kind), []), variant, CARRY_PATTERN, rng, None):
                        survivors[f"{s}|{kind}|{q['relation']}"] = survivors.get(f"{s}|{kind}|{q['relation']}", 0) + 1
                out[s] += pick(pools.get((s, kind), []), variant, CARRY_PATTERN if variant == "strict" else
                               (lambda q: (q["a_class"], q["relation"])) if variant == "v1_hard" else (lambda q: q["sel_pred"]),
                               rng, None if s == "train" else STRICT_CAP if variant == "strict" else a.cap)
            rng.shuffle(out[s])
            for q in out[s]:
                q["id"] = f"cr1/{s}/{q['op']}/{zlib.crc32((q['unit_a'] + q['context_a'] + q['context_b'] + q['op']).encode()):08x}"
        write(a.out / variant, out, texts, {
            "schema": "carry1", "variant": variant, "contrast": CONTRAST, "seed": a.seed, "inputs": [str(p) for p in a.raw],
            "texts": len(texts), "removed": dict(why), "families": split_of, "kinds": list(CARRY_KINDS),
            "carry_from": dict(CARRY_FROM), "cap_per_class": STRICT_CAP if variant == "strict" else a.cap,
            **({"survivors_before_cap": survivors} if variant == "strict" else {}),
            "available_before_balance": {f"{s}|{k}|{g}": n for (s, k), v in sorted(pools.items())
                                         for g, n in sorted(collections.Counter(q["gold"] for q in v).items())},
            "rule": "B is another prompt of A's family (same split); A's unit acts at B's aligned token; gold = B's measured "
                    "change: drop (<= -HIGH bits), rise (>= HIGH), none (|change| < LOW); " + VARIANT_RULE[variant] +
                    ("; strict's committee: gradient-boosted trees on A's numbers and on the names, a random forest on A's "
                     "numbers, the kind's single field, a tree on the question's text alone (all fitted on v1's training split), adv's tree (sel_pred), and A's own class of the same unit "
                     "with its direction in the margin, within B's relation to A" if variant == "strict" else "")})


VARIANT_RULE = {
    "v1": "each kind's classes balanced per split (dev and test capped per class)",
    "v1_hard": "each kind's classes balanced per split within each class of A's own measured change of the same unit "
               "(drop, rise, none or gap) and B's relation to A (the rule's accuracy differs by relation): the rule 'B does "
               "what A does' and any exclusion of it at chance within each relation",
    "adv": "each kind's classes balanced per split within each class a tree on the circuit's numbers predicts (sel_pred: "
           "trained on the training split, for training questions on the other training families), so that tree and any "
           "exclusion of its answer sit at chance",
    "strict": "training split = v1's; dev and test balanced jointly within every known shortcut's answer at once (a "
              "committee of models fitted on that training split, listed below, and the hidden rules), at most STRICT_CAP "
              "per class with equal numbers per cell: each shortcut, each exclusion of one and any function of them sit at "
              "chance on the same questions; survivors_before_cap counts every question that survives the balance"}

_TREES = lambda q: (q["strict_num"], q["strict_names"], q["strict_nokind"], q["strict_forest"], q["strict_field"],  # noqa: E731
                    q["strict_text"], q["same_key"]) + tuple(q["strict_tree_of_" + k] for k in TRAIN_KINDS)
KIND_PATTERN = {"zero": _TREES} | {k: (lambda q: _TREES(q) + (q["zero_copy"],)) for k in ("patch", "push_up")} | {
    "flip": lambda q: _TREES(q) + (q["zero_copy"], q["twice_zero"]),
    "push_down": lambda q: _TREES(q) + (q["zero_copy"], q["neg_push_up"])}
CARRY_PATTERN = lambda q: (q["strict_num"], q["strict_names"], q["strict_forest"], q["strict_field"], q["sel_pred"],  # noqa: E731
                           q["strict_text"], q["a_fine"], q["relation"])


def fit_pred(trn: list, cand: list, X, field: str, forest: bool = False) -> None:
    """field on each candidate: the answer of gradient-boosted trees, or a random forest, fitted on trn (the baselines'
    configurations, so the baseline of the same name is the selecting model)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    if not cand or not trn:
        return
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=4,
                               random_state=1) if forest else HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                                                                              random_state=0)
    m.fit([X(q) for q in trn], [q["gold"] for q in trn])
    for p_, q in zip(m.predict([X(q) for q in cand]), cand):
        q[field] = str(p_)


def best_field(trn: list, X, n: int):
    """The single field (one of X's first n columns) a depth-2 tree classifies trn best by, and that tree."""
    from sklearn.tree import DecisionTreeClassifier
    xs, ys = [X(q) for q in trn], [q["gold"] for q in trn]
    fit = lambda j: DecisionTreeClassifier(max_depth=2, random_state=0).fit([[x[j]] for x in xs], ys)  # noqa: E731
    j = max(range(n), key=lambda j: fit(j).score([[x[j]] for x in xs], ys))
    return j, fit(j)


def field_pred(trn: list, cand: list, X, n: int, ops) -> None:
    """strict_field on each candidate: its kind's single field chosen on trn (the pooled one for a kind trn lacks), the
    baselines' single_field."""
    pooled = best_field(trn, X, n)
    for op in ops:
        sub = [q for q in trn if q["op"] == op]
        j, m = best_field(sub, X, n) if sub else pooled
        qs = [q for q in cand if q["op"] == op]
        if qs:
            for p_, q in zip(m.predict([[X(q)[j]] for q in qs]), qs):
                q["strict_field"] = str(p_)


def strict_kinds(by_sk: dict, trn: list, cs: dict) -> None:
    """The shortcuts of a multi-kind dev or test question: trees on the numbers with the kind, on the names with the kind,
    on the numbers without the kind, a forest on the numbers with the kind, each training kind's own tree, and the hidden
    rules of the same pair: its zero class, the class of twice its zeroings, of its push_up negated."""
    cand = [q for (s, _), v in by_sk.items() if s != "train" for q in v]
    fit_pred(trn, cand, lambda q: kind_X(q, cs) + kind_onehot(q), "strict_num")
    fit_pred(trn, cand, lambda q: kind_X(q, cs, True) + kind_onehot(q), "strict_names")
    fit_pred(trn, cand, lambda q: kind_X(q, cs), "strict_nokind")
    fit_pred(trn, cand, lambda q: kind_X(q, cs) + kind_onehot(q), "strict_forest", forest=True)
    for k in TRAIN_KINDS:  # each training kind's own tree, on every kind (for another kind: the mapping a person chooses)
        fit_pred([q for q in trn if q["op"] == k], cand, lambda q: kind_X(q, cs), "strict_tree_of_" + k)
    nf = len(kind_X(trn[0], cs))
    field_pred(trn, cand, lambda q: kind_X(q, cs) + kind_onehot(q), nf, KINDS)
    by_ctx = {c["context_id"]: c for c in cs.values()}
    fit_pred(trn, cand, lambda q: text_X(q, by_ctx), "strict_text")
    for q in cand:
        q["same_key"] = q["context_a"].split("/")[1] == q["context_b"].split("/")[1]
    for q in cand:
        m = q["measured"]
        q["zero_copy"] = pair_class(m["zero_a"], m["zero_b"])
        q["twice_zero"] = pair_class(2 * m["zero_a"], 2 * m["zero_b"])
        q["neg_push_up"] = pair_class(-m["push_up_a"], -m["push_up_b"]) if "push_up_a" in m else "none"


def pair_class(da: float, db: float) -> str:
    """A rule's answer from two changes: pairs.zero_class, or in its margin the side that changes more ("gapA", "gapB"),
    so a rule that copies only the direction there is balanced against too."""
    return pairs.zero_class(da, db) or ("gapA" if da < db else "gapB")


def a_class_of(ch: float | None) -> str:
    """A's own class of the same unit's change, or in the margin its direction ("gap_drop", "gap_rise")."""
    if ch is None:
        return "gap"
    return cls3(ch) or ("gap_drop" if ch < 0 else "gap_rise")


def strict_carry(pools: dict, trn: list, cs: dict, by_ctx: dict) -> None:
    """The shortcuts of a new-prompt dev or test question: the numbers tree, the names tree, a forest on the numbers, the
    adv tree (sel_pred, on the question already), and A's own class of the same unit's change with its direction in the
    margin (a_fine), within B's relation to A."""
    cand = [q for (s, _), v in pools.items() if s != "train" for q in v]
    fit_pred(trn, cand, lambda q: carry_X(q, cs, by_ctx), "strict_num")
    fit_pred(trn, cand, lambda q: carry_X(q, cs, by_ctx, True), "strict_names")
    fit_pred(trn, cand, lambda q: carry_X(q, cs, by_ctx), "strict_forest", forest=True)
    field_pred(trn, cand, lambda q: carry_X(q, cs, by_ctx), len(carry_X(trn[0], cs, by_ctx)), CARRY_KINDS)
    fit_pred(trn, cand, lambda q: text_X(q, by_ctx), "strict_text")
    for q in cand:
        q["a_fine"] = a_class_of(q["measured"]["change_a"])


def pick(qs: list, variant: str, pattern, rng: random.Random, cap: int | None) -> list:
    """v1: classes balanced; otherwise balanced within each pattern (a rule's answer, with what else the reader sees that
    the rule's accuracy varies with), patterns that hold every class only, cap halved per pattern."""
    if variant == "v1":
        by = collections.defaultdict(list)
        for q in qs:
            by[q["gold"]].append(q)
        return balance(by, rng, cap)
    pat = collections.defaultdict(lambda: collections.defaultdict(list))
    for q in qs:
        pat[pattern(q)][q["gold"]].append(q)
    n_cls = len({q["gold"] for q in qs})
    full = [(k, by) for k, by in sorted(pat.items()) if len(by) == n_cls]
    each = None if cap is None else -(-cap // max(1, len(full)))  # (dev and test: about cap per class over all patterns)
    return [q for _, by in full for q in balance(by, rng, each)]


def adv_pred(by_sk: dict, X, fam) -> None:
    """sel_pred on every question: the answer of gradient-boosted trees on the circuit's numbers, trained on the training
    split's questions (all kinds) of the other training families for a training question, on all of them otherwise."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    gb = lambda: HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0)  # noqa: E731
    tr = [q for (s, _), v in by_sk.items() if s == "train" for q in v]
    xs = [X(q) for q in tr]
    for f in sorted({fam(q) for q in tr}):
        m = gb().fit([x for x, q in zip(xs, tr) if fam(q) != f], [q["gold"] for q in tr if fam(q) != f])
        sub = [(x, q) for x, q in zip(xs, tr) if fam(q) == f]
        for p_, (_, q) in zip(m.predict([x for x, _ in sub]), sub):
            q["sel_pred"] = str(p_)
    m = gb().fit(xs, [q["gold"] for q in tr])
    ev = [q for (s, _), v in by_sk.items() if s != "train" for q in v]
    if ev:
        for p_, q in zip(m.predict([X(q) for q in ev]), ev):
            q["sel_pred"] = str(p_)


def write(out_dir: Path, out: dict, texts: dict, man: dict, extra: dict | None = None) -> None:
    """out's splits, extra files of questions (name -> questions), the circuits of every text they name, the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    used = set()
    for name, qs in (extra or {}).items():
        with (out_dir / name).open("w") as fh:
            for q in qs:
                fh.write(json.dumps(q) + "\n")
                used |= {q["context_a"], q.get("context_b")} - {None}
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


def kind_X(q: dict, cs: dict, names: bool = False) -> list[float]:
    """A paired question's tree input: both sides' item numbers (or names) and their difference."""
    f_ = names_feats if names else item_feats
    fa, fb = f_(q["units_a"], cs[q["circuit_a"]]), f_(q["units_b"], cs[q["circuit_b"]])
    return fa + fb + [x - y for x, y in zip(fa, fb)]


def kind_onehot(q: dict) -> list[float]:
    return [float(q["op"] == k) for k in TRAIN_KINDS]  # (a held-out kind: all zeros)


def carry_X(q: dict, cs: dict, by_ctx: dict, names: bool = False) -> list[float]:
    """A new-prompt question's tree input: A's unit numbers (or names), B's relation to A, the alignment, the kind, whether
    B's token is A's and B's position from the end."""
    c = cs[q["circuit_a"]]
    pa, pb = int(q["unit_a"].split("@")[1]), int(q["unit_b"].split("@")[1])
    f = (names_feats if names else item_feats)([q["unit_a"]], c)
    f += [float(q["relation"] == r) for r in RELS] + [float(q["align"] == "same")] + [float(q["op"] == k) for k in CARRY_KINDS]
    cb = by_ctx.get(q["context_b"])
    f += [float(cb is not None and cb["tokens"][pb] == c["tokens"][pa]), float(len(cb["tokens"]) - 1 - pb) if cb else 0.0]
    return f


RELS = ("wording", "key", "both")


def text_X(q: dict, by_ctx: dict) -> list[float]:
    """A question's text alone (no circuit number): whether A and B share the key, the answer word, the wording; each
    text's wording, answer length, prompt length and the unit's position from the end; whether the unit's tokens match;
    the kind, the unit kind, and for new-prompt questions the relation and the alignment."""
    A, B = by_ctx[q["context_a"]], by_ctx[q["context_b"]]
    multi = "units_a" in q
    pa, pb = (int(u.split("@")[1]) for u in ((q["units_a"][0], q["units_b"][0]) if multi else (q["unit_a"], q["unit_b"])))
    aa, ab = (A["answer"]["text"] or "").strip(), (B["answer"]["text"] or "").strip()
    wa, wb = int(q["context_a"].split("/")[-1][1:]), int(q["context_b"].split("/")[-1][1:])
    f = [float(q["context_a"].split("/")[1] == q["context_b"].split("/")[1]), float(aa == ab), float(wa == wb), wa, wb,
         len(aa), len(ab), len(A["tokens"]), len(B["tokens"]), len(A["tokens"]) - 1 - pa, len(B["tokens"]) - 1 - pb,
         float(A["tokens"][pa] == B["tokens"][pb])]
    f += [float(q["op"] == k) for k in KINDS] + [float(q["unit_kind"] == k) for k in ("neuron", "head", "group")]
    if not multi:
        f += [float(q["relation"] == r) for r in RELS] + [float(q["align"] == "same")]
    return [float(x) for x in f]


def interval(scores: list[float], clusters: list, boots: int = 1000) -> dict:
    """Mean score and its 95% interval by a cluster bootstrap (whole clusters resampled with replacement: questions that
    share a unit or a prompt are not independent)."""
    import numpy as np
    s = np.asarray(scores, float)
    keys, inv = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
    tot, cnt = np.bincount(inv, weights=s, minlength=len(keys)), np.bincount(inv, minlength=len(keys))
    idx = np.random.default_rng(0).integers(0, len(keys), (boots, len(keys)))
    b = tot[idx].sum(1) / cnt[idx].sum(1)
    return {"acc": round(float(s.mean()), 3), "lo": round(float(np.percentile(b, 2.5)), 3),
            "hi": round(float(np.percentile(b, 97.5)), 3)}


def baselines(a) -> None:
    """Every baseline on each test question (and on flip_diag.jsonl when present), per kind, as accuracy with a 95%
    cluster-bootstrap interval (clusters: the item key for multi-kind, prompt B for new-prompt). Trees are fitted on the
    directory's training split; a single field is chosen on it too. A rule's exclusion answers either class it does not
    name (0.5 of the rest; 1/3 when it names none)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    d = a.dir
    man = json.loads((d / "manifest.json").read_text())
    cs = {c["circuit_id"]: c for c in map(json.loads, (d / "circuits.jsonl").read_text().splitlines())}
    tr = [json.loads(x) for x in (d / "train.jsonl").read_text().splitlines()]
    sets = {"test": [json.loads(x) for x in (d / "test.jsonl").read_text().splitlines()]}
    if (d / "flip_diag.jsonl").exists():
        sets["flip_diag"] = [json.loads(x) for x in (d / "flip_diag.jsonl").read_text().splitlines()]
    gb = lambda: HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0)  # noqa: E731
    rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",  # noqa: E731
                                        n_jobs=4, random_state=1)
    multi = man["schema"] == "multikind1"
    if multi:
        X = lambda q, names=False: kind_X(q, cs, names) + kind_onehot(q)  # noqa: E731
        cluster = lambda q: q["key"]  # noqa: E731
        kinds = KINDS
    else:
        by_ctx = {c["context_id"]: c for c in cs.values()}
        X = lambda q, names=False: carry_X(q, cs, by_ctx, names)  # noqa: E731
        cluster = lambda q: q["context_b"]  # noqa: E731
        kinds = CARRY_KINDS
    ytr = [q["gold"] for q in tr]
    Xtr, Ntr = [X(q) for q in tr], [X(q, True) for q in tr]
    by_ctx_ = {c["context_id"]: c for c in cs.values()}
    models = {"tree_numbers": (gb().fit(Xtr, ytr), False), "tree_names_only": (gb().fit(Ntr, ytr), True),
              "tree_text_only": (gb().fit([text_X(q, by_ctx_) for q in tr], ytr), "text"),
              "forest_numbers": (rf().fit(Xtr, ytr), False), "forest_names_only": (rf().fit(Ntr, ytr), True)}
    nk = len(TRAIN_KINDS) if multi else 0
    if multi:  # without the kind, and each training kind's own tree (for a held-out kind: the mapping a person chooses)
        models["tree_numbers_no_kind"] = (gb().fit([x[:-nk] for x in Xtr], ytr), "nokind")
        for k in TRAIN_KINDS:
            sub = [(x, q["gold"]) for x, q in zip(Xtr, tr) if q["op"] == k]
            models[f"tree_of_{k}"] = (gb().fit([x[:-nk] for x, _ in sub], [g for _, g in sub]), "nokind")
    else:
        models["tree_with_hidden_a_change"] = (gb().fit([x + [q["measured"]["change_a"] or 0.0] for x, q in zip(Xtr, tr)], ytr), "hidden")
    nf = len(Xtr[0]) - nk
    field = {}  # each kind's single field chosen on the training split (a depth-2 tree on one column)
    for k in {q["op"] for q in tr}:
        sub = [(x, q["gold"]) for x, q in zip(Xtr, tr) if q["op"] == k]
        best = max(range(nf), key=lambda j: DecisionTreeClassifier(max_depth=2, random_state=0).fit(
            [[x[j]] for x, _ in sub], [g for _, g in sub]).score([[x[j]] for x, _ in sub], [g for _, g in sub]))
        field[k] = (best, DecisionTreeClassifier(max_depth=2, random_state=0).fit([[x[best]] for x, _ in sub], [g for _, g in sub]))
    pooled_field = max(range(nf), key=lambda j: DecisionTreeClassifier(max_depth=2, random_state=0).fit(
        [[x[j]] for x in Xtr], ytr).score([[x[j]] for x in Xtr], ytr))
    pooled_field_m = DecisionTreeClassifier(max_depth=2, random_state=0).fit([[x[pooled_field]] for x in Xtr], ytr)
    classes = {"A", "B", "same"} if multi else {"drop", "rise", "none"}
    out = {}
    for set_name, te in sets.items():
        res = {}
        for kind in kinds:
            qs = [q for q in te if q["op"] == kind]
            if not qs:
                continue
            cl = [cluster(q) for q in qs]
            Xte, Nte = [X(q) for q in qs], [X(q, True) for q in qs]
            pred = {}
            for name, (m, how) in models.items():
                xs = Nte if how is True else [x[:-nk] for x in Xte] if how == "nokind" else \
                    [x + [q["measured"]["change_a"] or 0.0] for x, q in zip(Xte, qs)] if how == "hidden" else \
                    [text_X(q, by_ctx_) for q in qs] if how == "text" else Xte
                pred[name] = [str(p_) for p_ in m.predict(xs)]
            j, m = field.get(kind, (pooled_field, pooled_field_m))
            pred["single_field" if kind in field else "single_field_pooled_kinds"] = [str(p_) for p_ in m.predict([[x[j]] for x in Xte])]
            if multi:
                side = lambda c: {"gapA": "A", "gapB": "B"}.get(c, c)  # noqa: E731 (in the margin, the side that changes more)
                if kind != "zero":
                    pred["hidden_zero_copy"] = [side(pair_class(q["measured"]["zero_a"], q["measured"]["zero_b"])) for q in qs]
                if kind == "flip":
                    pred["hidden_twice_zero"] = [side(pair_class(2 * q["measured"]["zero_a"], 2 * q["measured"]["zero_b"])) for q in qs]
                if kind == "push_down":
                    pred["hidden_neg_push_up"] = [side(pair_class(-q["measured"]["push_up_a"], -q["measured"]["push_up_b"]))
                                                  if "push_up_a" in q["measured"] else "same" for q in qs]
            else:
                pred["carry_rule_a_class"] = [q["a_class"] if q["a_class"] != "gap" else "none" for q in qs]
                pred["carry_rule_direction"] = [{"gap_drop": "drop", "gap_rise": "rise", "gap": "none"}.get(
                    a_class_of(q["measured"]["change_a"]), a_class_of(q["measured"]["change_a"])) for q in qs]
            for f_ in ("sel_pred", "strict_num", "strict_names", "strict_nokind", "strict_forest", "strict_tree_of_zero",
                       "strict_tree_of_patch", "strict_tree_of_push_up", "strict_field", "strict_text"):  # the selecting models
                if all(f_ in q for q in qs):
                    pred["selecting_" + f_] = [q[f_] for q in qs]
            r = {"n": len(qs), "chance": 0.333}
            for name, p in pred.items():
                r[name] = interval([float(x == q["gold"]) for x, q in zip(p, qs)], cl)
            excl = [n_ for n_ in pred if not n_.startswith(("tree_of_", "single_field", "forest_"))] + \
                   (["a_class"] if not multi else [])
            for name in excl:  # each rule's exclusion
                p = [q["a_class"] for q in qs] if name == "a_class" else pred[name]
                r["exclusion_of_" + name] = interval([0.5 * (q["gold"] != x) if x in classes else 1 / 3 for x, q in zip(p, qs)], cl)
            if not multi:
                for rel in RELS:
                    sub = [i for i, q in enumerate(qs) if q["relation"] == rel]
                    if sub:
                        r["by_relation_" + rel] = {"n": len(sub), **{name: interval([float(pred[name][i] == qs[i]["gold"]) for i in sub],
                                                                                   [cl[i] for i in sub])
                                                                     for name in ("carry_rule_a_class", "tree_numbers", "tree_names_only")}}
            res[kind] = r
        out[set_name] = res
    every = sets["test"] + tr + ([json.loads(x) for x in (d / "dev.jsonl").read_text().splitlines()] if (d / "dev.jsonl").exists() else [])
    ctx = lambda qs: {c for q in qs for c in (q["context_a"], q.get("context_b")) if c}  # noqa: E731
    shown = {k for c in cs.values() for u in c["units"] + c["heads"] for k in u}
    man["checks"] = {  # leaks: shared texts across splits, held-out kinds in training, measured numbers in a question's
        "test_contexts_in_train": len(ctx(sets["test"]) & ctx(tr)),  # text, measured fields in the records, duplicates
        "held_out_kinds_in_train": sum(q["op"] not in TRAIN_KINDS for q in tr) if multi else None,
        "measured_change_in_question": sum(f"{v:+.1f}" in q["q"] for q in every for k, v in q["measured"].items()
                                           if k.startswith("change") and v is not None),
        "circuit_unit_fields": sorted(shown), "duplicate_ids": len(every) - len({q["id"] for q in every})}
    man["baselines_test"] = out["test"]
    if "flip_diag" in out:
        man["baselines_flip_diag"] = out["flip_diag"]
    man["baselines_note"] = ("accuracy with a 95% cluster-bootstrap interval (clusters: " + ("item keys" if multi else "B prompts") +
                             "); trees and single fields fitted on this directory's training split; hidden_* and carry_rule_* use "
                             "measured changes no reader sees; selecting_* are the trees the balance was built on")
    (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({s_: {k: {n_: v["acc"] for n_, v in r.items() if isinstance(v, dict) and "acc" in v} for k, r in res.items()}
                      for s_, res in out.items()}, indent=1))


def circuit_summary(c: dict, focus: list[str]) -> list[float]:
    """A whole circuit's numbers: unit and head counts by stage, summed and largest lost and restored shares (neurons and
    heads), the ten largest lost shares, |value_over_max| summed and largest, units and summed lost share per band of six
    layers, the focus units' best rank by lost share and their share of the total, the clean and corrupted contrasts,
    the gap and the length."""
    us, hs = c["units"], c["heads"]
    lost = sorted((abs(x.get("lost") or 0.0) for x in us + hs), reverse=True)
    rest = [abs(x.get("restored") or 0.0) for x in us + hs]
    vom = [abs(u.get("value_over_max") or 0.0) for u in us]
    f = [len(us), len(hs)] + [sum(u.get("stage") == st for u in us) for st in ("source", "readout")]
    f += [sum(lost), max(lost, default=0.0), sum(rest), max(rest, default=0.0)] + (lost + [0.0] * 10)[:10]
    f += [sum(vom), max(vom, default=0.0)]
    for b in range(6):
        band = [x for x in us + hs if b * 6 <= x["layer"] < (b + 1) * 6]
        f += [len(band), sum(abs(x.get("lost") or 0.0) for x in band)]
    mine = [abs(x.get("lost") or 0.0) for x in us + hs if x["name"] in set(focus)]
    f += [min((lost.index(v) for v in mine), default=len(lost)), sum(mine) / (sum(lost) or 1.0)]
    con = c.get("contrast") or {}
    return [float(x) for x in f + [con.get("clean") or 0.0, con.get("corrupt") or 0.0, c.get("gap") or 0.0, len(c["tokens"])]]


def ceiling(a) -> None:
    """The strongest tabular learner on everything the circuit records hold for a question (the asked units' numbers, both
    whole circuits, contrasts; new-prompt also B's own circuit, which a run on B gives; optionally the question's text),
    fitted on every labelled question of the set's directories outside one test family and scored on that family's
    questions of each test set (cross-validated over the test families), as accuracy with a 95% cluster interval: does
    the circuit hold the answer at all?"""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    root = a.dir
    multi = root.name == "kinds"
    pool, cs = {}, {}
    for v in sorted(p_ for p_ in root.iterdir() if (p_ / "manifest.json").exists()):
        for c in map(json.loads, (v / "circuits.jsonl").read_text().splitlines()):
            cs[c["circuit_id"]] = c
        for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
            if (v / name).exists():
                for q in map(json.loads, (v / name).read_text().splitlines()):
                    pool[q["id"]] = q
    by_ctx = {c["context_id"]: c for c in cs.values()}
    fam = (lambda q: q["family_a"]) if multi else (lambda q: q["family"])
    onehot5 = lambda q: [float(q["op"] == k) for k in KINDS]  # noqa: E731

    def b_side(q):  # new-prompt: B's own circuit, and the unit's numbers in it when B's circuit holds it
        cb = by_ctx[q["context_b"]]
        ub = {x["name"]: x for x in cb["units"] + cb["heads"]}.get(q["unit_b"], {})
        base = base_name(q["unit_b"])
        anywhere = [x for x in cb["units"] + cb["heads"] if base_name(x["name"]) == base]
        return circuit_summary(cb, [q["unit_b"]]) + [float(bool(ub))] + [float(ub.get(k) or 0.0) for k in
                                                                             ("lost", "restored", "value", "value_over_max")] + [
            len(anywhere), max((abs(x.get("lost") or 0.0) for x in anywhere), default=0.0)]
    if multi:
        def arm_numbers(q):
            sa, sb = circuit_summary(cs[q["circuit_a"]], q["units_a"]), circuit_summary(cs[q["circuit_b"]], q["units_b"])
            return kind_X(q, cs) + onehot5(q) + sa + sb + [x - y for x, y in zip(sa, sb)]
        arms = {"numbers": arm_numbers, "numbers_text": lambda q: arm_numbers(q) + text_X(q, by_ctx)}
    else:
        arm_a = lambda q: carry_X(q, cs, by_ctx) + circuit_summary(cs[q["circuit_a"]], [q["unit_a"]])  # noqa: E731
        arms = {"numbers_A": arm_a, "numbers_A_B": lambda q: arm_a(q) + b_side(q),
                "numbers_A_B_text": lambda q: arm_a(q) + b_side(q) + text_X(q, by_ctx)}
    sets = {v: [json.loads(x) for x in (root / v / "test.jsonl").read_text().splitlines()]
            for v in a.sets if (root / v / "test.jsonl").exists()}
    test_fams = sorted({fam(q) for qs in sets.values() for q in qs})
    qs_all = list(pool.values())
    learners = {"boosting": lambda: HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                                                                   l2_regularization=1.0, class_weight="balanced", random_state=0),
                "forest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
                                                         n_jobs=8, random_state=1)}
    preds = collections.defaultdict(dict)  # (arm, learner) -> question id -> answer
    for arm, X in arms.items():
        xs = {q["id"]: X(q) for q in qs_all}
        for v_ in sets.values():
            for q in v_:
                xs.setdefault(q["id"], X(q))
        for lname, mk in learners.items():
            if lname == "forest" and arm != list(arms)[-1]:
                continue  # (the forest on the fullest arm only)
            for f in test_fams:
                trn = [q for q in qs_all if fam(q) != f]
                m = mk().fit([xs[q["id"]] for q in trn], [q["gold"] for q in trn])
                ev = {q["id"]: q for v_ in sets.values() for q in v_ if fam(q) == f}
                for p_, i in zip(m.predict([xs[i] for i in ev]), ev):
                    preds[(arm, lname)][i] = str(p_)
                print(json.dumps({"arm": arm, "learner": lname, "held_out": f, "trained_on": len(trn)}), flush=True)
    clus = (lambda q: q["key"]) if multi else (lambda q: q["context_b"])
    out = {}
    for v, qs in sets.items():
        r = {}
        for (arm, lname), pr in preds.items():
            cell = {"all": interval([float(pr[q["id"]] == q["gold"]) for q in qs], [clus(q) for q in qs])}
            for op in sorted({q["op"] for q in qs}):
                sub = [q for q in qs if q["op"] == op]
                cell[op] = interval([float(pr[q["id"]] == q["gold"]) for q in sub], [clus(q) for q in sub])
            r[f"{lname}:{arm}"] = cell
        out[v] = r
        man = json.loads((root / v / "manifest.json").read_text())
        man["ceiling"] = {"note": "the strongest tabular learner on all circuit numbers for the question (and the text, where "
                                  "named), fitted on every labelled question of this set family's directories outside one "
                                  "test family, scored on that family; cross-validated over the test families", **r}
        (root / v / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({v: {k: {op: c["acc"] for op, c in cell.items()} for k, cell in r.items()} for v, r in out.items()}, indent=1))


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
    ce = sub.add_parser("ceiling")
    ce.add_argument("dir", type=Path, help="the set family's directory (kinds or carry)")
    ce.add_argument("--sets", nargs="+", default=["v1", "adv", "strict", "v1_hard"])
    a = ap.parse_args()
    {"measure": measure, "build-kinds": build_kinds, "build-carry": build_carry, "baselines": baselines, "ceiling": ceiling}[a.cmd](a)


if __name__ == "__main__":
    main()
