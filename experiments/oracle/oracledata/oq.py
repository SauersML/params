"""Oracle training questions at scale (#2951; oracledata, the lead's 10-16 oracle push, ~/mpd-data/team/oracle_push_1016.txt).
Questions about the units of question-scoped circuits (circuit_finder's records, with neuron_motifs apply's shared groups
where they exist), every answer measured on Qwen3-8B itself (no weak zeroing, as circuit_finder measures; the steer scale
and the massive values that stay live are T2's file's, claim_check's semantics), every stated claim re-checked by
method/oracle/claim_check.py.

An item is what a question names: a shared group "G<id>" (its members at their positions; the name stands for them), a
circuit neuron outside every group ("L18.N2201+@12", claim_check's unit), or a mover head ("L35.H25@22"). Kinds:

  zero       zeroing one item: the answer flips, its lead goes up or down, or stays
  steer      one item's neurons turned up (STEER x each one's calibration maximum, along its pole)
  set        two to four items zeroed together
  sign       whether an item supports the answer, works against it, or barely matters (its zeroing's sign)
  rank       three items ordered by the change zeroing each alone makes to the answer's lead (gaps of at least GAP bits)
  necessary  which one of four items is needed: zeroing it alone lowers the lead by at least BIG bits, the others < LOW
  joint      which one of up to four pairs matters only jointly: each member alone < LOW, the pair at least BIG
  interact   whether the change from zeroing a pair is above, below or about the sum of the changes from each alone
  direction  which of four tokens gains the most probability when an item (or the whole circuit) is zeroed
  min_edit   which of four edits (the items zeroed at every position, named without positions: a weight-level edit)
             removes the answer (its top token changes) with the least side effect, the mean KL in bits over SIDE_N
             general chats

A change under LOW bits is "unchanged", from HIGH bits up or down; changes between are not asked (bfloat16 batches move a
contrast by float error). The contrast is claim_check's: log2 p(answer) over the summed p of the model's five likeliest
other tokens, at the last position. Every row holds the question, a canonical answer (one sentence; for the four claim
kinds also "Claim: {json}" passing claim_check, g2.py's form), its exact-match key and class, the measured numbers and a
check a verifier reruns. The circuit each row reads is written once per text (circuits file, by circuit_id). Items whose
values are all massive (claim_check's zero leaves those live) get no zeroing question.

  oq.py gen RECORDS OUT.jsonl --thr THR.pt [--groups APPLY] [--source NAME] [--limit N] [--shard K/N] [--adapter A]  (GPU)
  oq.py split OUT_DIR GEN.jsonl [GEN.jsonl ...] [--seed S] [--sealed testset_8b.json]   splits, balance, baselines (CPU)
     train.jsonl every training row; train_bal.jsonl per kind no class above the second-largest; dev/test balanced
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
sys.path.insert(0, str(HERE.parents[3] / "method" / "oracle"))

STEER = 3.0
LOW, HIGH = 0.2, 0.35      # bits: |change| < LOW is unchanged, >= HIGH a change; between is not asked
BIG, GAP = 1.0, 0.5        # bits: a needed item's drop; the gap between ranked items
SIDE_N, SIDE_TOK = 8, 40   # general chats and answer tokens a min_edit's side effect is measured on
SIDE_ROWS = range(300, 1000)  # general.jsonl rows (T2 calibrates on 1000-1199 and holds out 100-299)
PAIRS = 48                 # pairs measured per text (joint, interact)
TOPK = 8                   # the reference's likeliest tokens a direction question chooses from
CHECK_CHUNK = 40           # claims per claim_check call
CONTRAST = "word_v1"       # answer_contrast's word-level sets from the record (answer_tokens against alternatives)


def q_tok(s: str) -> str:
    return "'" + s.replace("\n", "\\n") + "'"


def listing(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def bits(x: float) -> str:
    v = round(abs(x) * 2) / 2
    return f"{v:g} bit" + ("" if v == 1 else "s")


def change_class(flip: bool, ch: float) -> str | None:
    if flip:
        return "flip"
    if abs(ch) < LOW:
        return "unchanged"
    if abs(ch) >= HIGH:
        return "up" if ch > 0 else "down"
    return None


def sign_class(ch: float) -> str | None:
    return "barely" if abs(ch) < LOW else None if abs(ch) < HIGH else "against" if ch > 0 else "supports"


def prediction(cls: str, ch: float, top_after: str, single) -> dict:
    """The claim_check prediction stating a class (the magnitude rounded to half bits, as g2.py)."""
    if cls == "flip":
        return {"top": top_after} if single(top_after) else {"top_changes": True}
    if cls == "unchanged":
        return {"contrast_change": {"op": "within", "value": 0.25}}
    v = round(ch * 2) / 2
    return {"contrast_change": {"op": ">=", "value": v - 0.25}} if ch > 0 else {"contrast_change": {"op": "<=", "value": v + 0.25}}


def interact_row(p: list[str], pair: float, alone: list[float], ytxt: str):
    """An interact question in signed terms (q, a, key, class), or None between the margins: the change from zeroing the
    pair is above, below or about the sum of the changes from zeroing each alone."""
    d = pair - sum(alone)
    c = "above" if d >= GAP else "below" if d <= -GAP else "about" if abs(d) < LOW else None
    if c is None:
        return None
    a = {"above": "Above the sum", "below": "Below the sum", "about": "About the sum"}[c]
    return (f"Is the change in the model's lead for {ytxt} from zeroing {p[0]} and {p[1]} together above, below or about "
            f"the sum of the changes from zeroing each alone?", f"{a}: together {pair:+.1f} bits, alone {alone[0]:+.1f} and "
            f"{alone[1]:+.1f} (sum {sum(alone):+.1f}).", c, c)


def free_name(n: str) -> str:
    """An item's name as an edit at every position: a group's name, a neuron's or head's without its position."""
    return n if n.startswith("G") else n.split("@")[0]


def items_of(rec: dict, apply: dict | None) -> list[dict]:
    """A record's items: apply's shared groups, the circuit's neurons outside them, its mover heads. units: claim_check
    names at their positions; free: the same units at every position (an edit); key: what the name-prior groups by."""
    nn = lambda d: f"L{d['layer']}.N{d['neuron']}{'+' if d['pole'] > 0 else '-'}"  # noqa: E731
    out, grouped = [], set()
    for g in (apply or {}).get("groups", []):
        g = {**g, "members": [d for d in g["members"] if d["pos"] > 0]}  # (the first position stays live in claim_check)
        if not g["members"]:
            continue
        mem = [f"{nn(d)}@{d['pos']}" for d in g["members"]]
        grouped |= set(mem)
        st = collections.Counter(d.get("stage") for d in g["members"]).most_common(1)[0][0]
        out.append({"name": f"G{g['id']}", "key": f"G{g['id']}", "kind": "group", "units": mem,
                    "free": sorted({nn(d) for d in g["members"]}), "stage": st, "shared_name": g.get("name")})
        # (apply's "effect", the group's zeroing drop, is a zero question's answer: never an input)
    for d in rec.get("nodes", []):
        u = f"{nn(d)}@{d['pos']}"
        if u not in grouped and d["pos"] > 0:
            out.append({"name": u, "key": nn(d), "kind": "neuron", "units": [u], "free": [nn(d)], "stage": d.get("stage")})
            grouped.add(u)
    for r in [r for r in rec.get("routes", []) if r["pos"] > 0]:
        h = f"L{r['layer']}.H{r['head']}"
        out.append({"name": f"{h}@{r['pos']}", "key": h, "kind": "head", "units": [f"{h}@{r['pos']}"], "free": [h], "stage": "movers"})
    return out


def circuit_record(rec: dict, apply: dict | None, items: list[dict], tok, nr, source: str) -> dict:
    ids = rec["ids"]
    toks = [tok.decode([t]) for t in ids]
    member = {u: it["name"] for it in items if it["kind"] == "group" for u in it["units"]}
    units = []
    for d in rec.get("nodes", []):
        name = f"L{d['layer']}.N{d['neuron']}{'+' if d['pole'] > 0 else '-'}@{d['pos']}"
        mx = float(nr.mx[d["layer"], d["neuron"]])
        units.append({"name": name, "layer": d["layer"], "neuron": d["neuron"], "pole": d["pole"], "pos": d["pos"],
                      "token": toks[d["pos"]], "stage": d.get("stage"), "value": d.get("clean"), "corrupt": d.get("corrupt"),
                      "value_over_max": None if not mx else round(d.get("clean", 0.0) / mx, 4), "lost": d.get("lost"),
                      "restored": d.get("restored"), "group": member.get(name)})
    for g in (apply or {}).get("groups", []):  # group members outside the record's own neurons (apply maps the shared group)
        for d in g["members"]:
            name = f"L{d['layer']}.N{d['neuron']}{'+' if d['pole'] > 0 else '-'}@{d['pos']}"
            if not any(u["name"] == name for u in units):
                mx = float(nr.mx[d["layer"], d["neuron"]])
                units.append({"name": name, "layer": d["layer"], "neuron": d["neuron"], "pole": d["pole"], "pos": d["pos"],
                              "token": toks[d["pos"]], "stage": d.get("stage"), "value": d.get("value"), "corrupt": None,
                              "value_over_max": None if not mx else round(d.get("value", 0.0) / mx, 4), "lost": None,
                              "restored": None, "group": f"G{g['id']}"})
    heads = [{"name": f"L{r['layer']}.H{r['head']}@{r['pos']}", "layer": r["layer"], "head": r["head"], "pos": r["pos"],
              "attends": [[q, w, toks[q]] for q, w, *_ in (r.get("attends") or [])], "lost": r.get("lost"),
              "restored": r.get("restored"), "through_source": r.get("through_source")} for r in rec.get("routes", [])]
    groups = [{"id": it["name"], "shared_name": it.get("shared_name"), "members": it["units"], "stage": it["stage"]}
              for it in items if it["kind"] == "group"]
    if rec.get("answer_tokens") and rec.get("alternatives"):  # each answer-position neuron's direct write on the contrast
        for u in units:  # (oracleeval's 10-16 feature: value x down_proj column x final-norm weight onto the mean unembedding
            u["direct"] = direct_write(nr, u, rec, len(ids))  # of the answer word's tokens minus the alternatives'; logits)
    corrupt = rec.get("corrupted")
    return {"circuit_id": f"{source}/{rec['id']}", "context_id": rec["id"], "family": rec.get("task"), "source": source,
            "finder": rec.get("finder"), "finder_rev": rec.get("finder_rev"), "model": rec.get("model"), "adapter": rec.get("adapter"),
            "ids": ids, "tokens": toks, "user": rec.get("user"),
            "answer": {"id": rec["answer"], "text": rec.get("answer_text") or tok.decode([rec["answer"]])},
            "corrupt": {"ids": corrupt, "user": rec.get("corrupt_user"),
                        "answer": {"id": rec.get("corrupt_answer"), "text": rec.get("corrupt_answer_text")}},
            "contrast": rec.get("contrast"), "gap": rec.get("gap"), "met": rec.get("met"), "size": rec.get("size"),
            "units": units, "heads": heads, "groups": groups, "items": [it["name"] for it in items]}


def direct_write(nr, u: dict, rec: dict, T: int) -> float | None:
    """A neuron's write at the answer position read straight onto the word-level contrast (logits, no later layers): value
    x its down_proj column x the final norm's weight, onto mean W_U[answer tokens] - mean W_U[alternatives]; None at
    other positions (their effect is indirect)."""
    if u["pos"] != T - 1 or u.get("value") is None:
        return None
    core = nr.core
    col = core.model.layers[u["layer"]].mlp.down_proj.weight[:, u["neuron"]].float()
    W = core.lm_head.weight
    d = W[rec["answer_tokens"]].float().mean(0) - W[rec["alternatives"]].float().mean(0)
    return round(float(u["value"] * (col * core.model.norm.weight.float() * d).sum().detach()), 4)


class Engine:
    """Batched intervention runs on one prompt in claim_check's padded batches (a row's numbers do not depend on what
    else its batch holds), at claim_check's positions (the answer and SIDE before it: the same lm_head shapes)."""

    def __init__(self, nr, cc):
        self.nr, self.cc = nr, cc

    def answer_lp(self, ids: list[int], rows: list[dict]):
        import torch
        T = len(ids)
        positions = list(range(max(1, T - 1 - self.cc.SIDE), T))
        out = []
        for s0 in range(0, len(rows), self.cc.ROWS):
            part = rows[s0:s0 + self.cc.ROWS]
            n = len(part)
            part = part + part[-1:] * (self.cc.ROWS - n)
            edit, route = self.cc.row_edit(self.nr, part)
            x = torch.tensor([ids] * self.cc.ROWS, device=self.nr.dev)
            with torch.no_grad():
                lp = torch.log_softmax(self.nr.logits(x, positions, edit=edit, route=route), -1)
            out.append(lp[:n, -1].float().cpu())
        return torch.cat(out)

    def side_kl(self, chats: list[list[int]], edits: list[list[tuple]]):
        """Mean KL (bits) of each edit against the model, over every answer position of the chats."""
        import torch
        tot = torch.zeros(len(edits))
        for ids, start in chats:
            positions = list(range(start, len(ids)))
            rows = [{}] + [{"units": us, "op": "zero"} for us in edits]
            lps = []
            for s0 in range(0, len(rows), self.cc.ROWS):
                part = rows[s0:s0 + self.cc.ROWS]
                n = len(part)
                part = part + part[-1:] * (self.cc.ROWS - n)
                edit, route = self.cc.row_edit(self.nr, part)
                x = torch.tensor([ids] * self.cc.ROWS, device=self.nr.dev)
                with torch.no_grad():
                    lps.append(torch.log_softmax(self.nr.logits(x, positions, edit=edit, route=route), -1)[:n].float())
            lp = torch.cat(lps)
            kl = (lp[0].exp() * (lp[0] - lp[1:])).sum(-1).mean(-1) / math.log(2)
            tot += kl.cpu()
        return (tot / len(chats)).tolist()


def text_questions(nr, cc, eng: Engine, tok, rec: dict, apply: dict | None, rng: random.Random, side, source: str):
    import torch
    ids = rec["ids"]
    T = len(ids)
    items = items_of(rec, apply)
    circ = circuit_record(rec, apply, items, tok, nr, source)
    if not items:
        return circ, []
    by = {it["name"]: it for it in items}
    groups = {it["name"]: it["units"] for it in items if it["kind"] == "group"}
    parse = lambda names: cc.units_of(nr, [u for n in names for u in by[n]["units"]], T)  # noqa: E731
    neuronal = [it["name"] for it in items if it["kind"] != "head"]
    # a massive value stays live under claim_check's zero, so an item whose values are all massive is never zeroed
    val = {u["name"]: u for u in circ["units"]}
    massive = lambda u: abs(val[u]["value"] or 0.0) >= float(nr.massive[val[u]["layer"]]) if u in val else False  # noqa: E731
    for u in circ["units"]:
        u["massive"] = massive(u["name"])
    zeroable = [n for n in by if by[n]["kind"] == "head" or not all(massive(u) for u in by[n]["units"])]
    # the runs: reference, each item zeroed, each neuron item steered, sets, pairs, the whole circuit zeroed
    plan = [("ref", ())]
    plan += [("zero", (n,)) for n in zeroable]
    plan += [("steer", (n,)) for n in neuronal]
    names = zeroable
    for _ in range(3 if len(names) >= 2 else 0):
        plan.append(("set", tuple(sorted(rng.sample(names, rng.randint(2, min(4, len(names))))))))
    pairs = list(itertools.combinations(names, 2))
    rng.shuffle(pairs)
    plan += [("pair", p) for p in pairs[:PAIRS]]
    if len(names) >= 2:
        plan.append(("circuit", tuple(names)))
    plan = list(dict.fromkeys(plan))
    rows = [{} if k == "ref" else {"units": parse(ns), "op": ("steer", STEER) if k == "steer" else "zero"} for k, ns in plan]
    lp = eng.answer_lp(ids, rows)
    ref = lp[0]
    y = int(ref.argmax())
    if rec.get("answer_tokens") and rec.get("alternatives"):  # answer_contrast's word-level sets, the finder's (CONTRAST)
        ys, alts = list(rec["answer_tokens"]), list(rec["alternatives"])
    else:
        ys, alts = [y], [t for t in ref.topk(nr_alts(cc) + 1).indices.tolist() if t != y][:nr_alts(cc)]
    word = {"answer": ys, "alternatives": alts}  # (stated in every claim: claim_check sums a list answer)
    cs = (torch.logsumexp(lp[:, ys], -1) - torch.logsumexp(lp[:, alts], -1)) / math.log(2)
    meas = {}
    for j, (k, ns) in enumerate(plan):  # a flip: the top token leaves the answer word
        meas[(k, ns)] = {"change": float(cs[j] - cs[0]), "top": int(lp[j].argmax()), "flip": int(lp[j].argmax()) not in ys, "j": j}
    top8 = ref.topk(TOPK).indices.tolist()
    ytxt = q_tok(tok.decode([y]))
    single = lambda s: len(tok(s, add_special_tokens=False)["input_ids"]) == 1  # noqa: E731
    out = []

    def row(kind, units, q, a, key, cls, claim=None, check=None, options=None, measured=None, answer_units=None):
        claim = {**claim, **word} if claim else claim
        out.append({"kind": kind, "units": list(units), "answer_units": list(answer_units or units), "q": q, "a": a + (f"\nClaim: {json.dumps(claim)}" if claim else ""),
                    "a_key": key, "cls": f"{kind}/{cls}", "claim": claim, "check": check or claim, "options": options,
                    "measured": measured or {}, "groups": {n: groups[n] for n in units if n in groups}})

    # the four claim kinds
    for k in ("zero", "steer"):
        for n in (names if k == "zero" else neuronal):
            m = meas[(k, (n,))]
            c = change_class(m["flip"], m["change"])
            if c is None:
                continue
            ta = tok.decode([m["top"]])
            verb = f"Zeroing {n}" if k == "zero" else f"Turning {n} up"
            clause = (f"the model gives {q_tok(ta)} instead of {ytxt}" if c == "flip" else f"the model still gives {ytxt}, almost unchanged"
                      if c == "unchanged" else f"the model still gives {ytxt}, its lead over the alternatives "
                      f"{'up' if m['change'] > 0 else 'down'} about {bits(m['change'])}")
            q = (f"What happens to the model's next token {ytxt} if {n} is zeroed?" if k == "zero" else
                 f"What happens to the model's next token {ytxt} if {n} is turned up?")
            claim = {"units": [n], "op": "zero" if k == "zero" else {"steer": STEER}, "predict": prediction(c, m["change"], ta, single)}
            row(k, [n], q, f"{verb}: {clause}.", c, c, claim, measured={"change": m["change"], "top_after": ta})
            if k == "zero":
                s = sign_class(m["change"]) if not m["flip"] else "supports"
                if s is not None:
                    pred = ({"contrast_change": {"op": "<=", "value": -0.25}} if s == "supports" else
                            {"contrast_change": {"op": ">=", "value": 0.25}} if s == "against" else
                            {"contrast_change": {"op": "within", "value": 0.25}})
                    txt = {"supports": f"{n} supports {ytxt}.", "against": f"{n} works against {ytxt}.", "barely": f"{n} barely matters for {ytxt}."}[s]
                    row("sign", [n], f"Does {n} support the model's next token {ytxt} or work against it?", txt, s, s,
                        {"units": [n], "op": "zero", "predict": pred}, measured={"change": m["change"]})
    for (k, ns), m in meas.items():
        if k != "set":
            continue
        c = change_class(m["flip"], m["change"])
        if c is None:
            continue
        ta = tok.decode([m["top"]])
        clause = (f"the model gives {q_tok(ta)} instead of {ytxt}" if c == "flip" else f"the model still gives {ytxt}, almost unchanged"
                  if c == "unchanged" else f"the model still gives {ytxt}, its lead over the alternatives "
                  f"{'up' if m['change'] > 0 else 'down'} about {bits(m['change'])}")
        claim = {"units": list(ns), "op": "zero", "predict": prediction(c, m["change"], ta, single)}
        row("set", ns, f"What happens to the model's next token {ytxt} if {listing(list(ns))} are zeroed together?",
            f"Zeroing {listing(list(ns))} together: {clause}.", c, c, claim, measured={"change": m["change"], "top_after": ta})
    # rank: three items whose zeroing changes are GAP apart, ordered most negative first
    zc = {n: meas[("zero", (n,))]["change"] for n in names}
    trip = [t for t in itertools.combinations(names, 3)]
    rng.shuffle(trip)
    made = 0
    for t in trip:
        s = sorted(t, key=lambda n: zc[n])
        if min(zc[s[1]] - zc[s[0]], zc[s[2]] - zc[s[1]]) < GAP:
            continue
        shown = list(t)
        rng.shuffle(shown)
        row("rank", shown, f"Order {listing(shown)} by the change zeroing each alone makes to the model's lead for its next "
            f"token {ytxt}, from the largest drop to the largest rise.", f"{s[0]}, then {s[1]}, then {s[2]}.", ">".join(s),
            f"first_{by[s[0]]['kind']}",
            check={"type": "rank", "units": s, "op": "zero"}, measured={"changes": [zc[n] for n in s]})
        made += 1
        if made >= 3:
            break
    # necessary: one item that drops the lead by BIG among three that barely matter
    strong = [n for n in names if zc[n] <= -BIG or meas[("zero", (n,))]["flip"]]
    null = [n for n in names if abs(zc[n]) < LOW and not meas[("zero", (n,))]["flip"]]
    for n in strong[:2]:
        if len(null) < 3:
            break
        opts = rng.sample(null, 3) + [n]
        rng.shuffle(opts)
        row("necessary", opts, f"Which one of {listing(opts)} is needed for the model's next token {ytxt}: zeroing it alone lowers "
            f"the lead by at least 1 bit, while zeroing any of the others alone barely matters?", f"{n}.", n, f"is_{by[n]['kind']}",
            check={"type": "necessary", "options": opts, "op": "zero"}, options=opts, measured={"changes": [zc[o] for o in opts]},
            answer_units=[n])
    # joint and interact, from the measured pairs
    pm = {ns: m for (k, ns), m in meas.items() if k == "pair"}
    jo = [p for p, m in pm.items() if all(abs(zc[n]) < LOW for n in p) and (m["change"] <= -BIG or m["flip"])]
    jnull = [p for p, m in pm.items() if all(abs(zc[n]) < LOW for n in p) and abs(m["change"]) < LOW and not m["flip"]]
    for p in jo[:2]:
        if len(jnull) < 2:
            break
        opts = rng.sample(jnull, min(3, len(jnull))) + [p]
        rng.shuffle(opts)
        os_ = [f"{a} and {b}" for a, b in opts]
        row("joint", [n for o in opts for n in o], f"Which pair matters for the model's next token {ytxt} only jointly (each "
            f"alone barely matters, zeroing both lowers its lead by at least 1 bit): {'; '.join(os_)}?", f"{p[0]} and {p[1]}.",
            f"{p[0]}+{p[1]}", "pair", check={"type": "joint", "options": [list(o) for o in opts], "op": "zero"}, options=os_,
            measured={"pair_changes": [pm[o]["change"] for o in opts]}, answer_units=list(p))
    for p, m in list(pm.items())[:12]:
        if max(abs(m["change"]), abs(zc[p[0]]), abs(zc[p[1]])) < HIGH or m["flip"]:
            continue
        r_ = interact_row(list(p), m["change"], [zc[p[0]], zc[p[1]]], ytxt)
        if r_:
            row("interact", list(p), *r_, check={"type": "interact", "units": list(p), "op": "zero"},
                measured={"pair": m["change"], "alone": [zc[p[0]], zc[p[1]]]})
    # direction: the reference's likeliest tokens, the one whose probability rises most under a zeroing
    p0 = ref.exp()
    for k, ns in [("zero", (n,)) for n in names] + ([("circuit", tuple(names))] if len(names) >= 2 else []):
        dp = lp[meas[(k, ns)]["j"]].exp()[top8] - p0[top8]
        o = dp.argsort(descending=True).tolist()
        if float(dp[o[0]]) < 0.02 or float(dp[o[0]]) < 1.5 * max(float(dp[o[1]]), 0.0):
            continue
        win = top8[o[0]]
        others = rng.sample([t for t in top8 if t != win], min(3, len(top8) - 1))
        opts = others + [win]
        rng.shuffle(opts)
        ot = [q_tok(tok.decode([t])) for t in opts]
        what = f"{ns[0]} is zeroed" if k == "zero" else "every unit of the circuit is zeroed"
        r = top8.index(win) + 1
        row("direction", list(ns) if k == "zero" else [], f"If {what}, which of {listing(ot)} gains the most probability as the "
            f"model's next token?", f"{q_tok(tok.decode([win]))}.", q_tok(tok.decode([win])), f"r{min(r, 3)}" + ("+" if r >= 3 else ""),
            check={"type": "direction", "units": list(ns), "op": "zero", "options": opts}, options=ot, answer_units=list(ns) if k == "zero" else [],
            measured={"gain": [float(dp[top8.index(t)]) for t in opts], "rank": r, "whole_circuit": k == "circuit"})
    # min_edit: weight-level edits (every position), the answer removed, side effects on general chats
    if side is not None and len(names) >= 2:
        cands = [(n,) for n in names]
        for st in ("readout", "source", "movers"):
            s_ = tuple(n for n in names if by[n]["stage"] == st)
            if len(s_) >= 2:
                cands.append(s_)
        cands += [p for p in pairs[:6]]
        cands = list(dict.fromkeys(cands))[:14]
        free = [cc.units_of(nr, [u for n in c for u in by[n]["free"]], T) for c in cands]
        lpe = eng.answer_lp(ids, [{"units": f, "op": "zero"} for f in free])
        removes = [int(lpe[i].argmax()) != y for i in range(len(cands))]
        kl = eng.side_kl(side, free)
        rem = sorted([i for i in range(len(cands)) if removes[i]], key=lambda i: kl[i])
        if rem:
            best = rem[0]
            second = kl[rem[1]] if len(rem) > 1 else None
            ok = second is None or second >= 1.5 * kl[best] + 0.005
            cheap = [i for i in range(len(cands)) if not removes[i]]
            if ok and cheap:
                pool = rem[1:2] + rng.sample(cheap, min(3 - len(rem[1:2]), len(cheap)))
                opts = pool + [best]
                rng.shuffle(opts)
                es = ["zero " + listing([free_name(n) for n in cands[i]]) for i in opts]
                row("min_edit", [n for i in opts for n in cands[i]], f"Which of these edits removes the model's next token {ytxt} "
                    f"with the least side effect on other text (each zeroes the named units at every position): "
                    f"{'; '.join(f'({chr(97 + j)}) {e}' for j, e in enumerate(es))}?", f"({chr(97 + opts.index(best))}) {es[opts.index(best)]}.",
                    es[opts.index(best)], "best", check={"type": "min_edit", "options": [list(cands[i]) for i in opts], "op": "zero_free",
                                                         "side": f"general.jsonl rows {SIDE_ROWS.start}+, {SIDE_N} chats, {SIDE_TOK} answer tokens"},
                    options=es, measured={"removes": [removes[i] for i in opts], "side_kl": [kl[i] for i in opts]},
                    answer_units=list(cands[best]))
    # the stated claims re-checked by claim_check (with its random-unit controls): only passing ones are kept
    idx = [i for i, r in enumerate(out) if r["claim"]]
    claims = [{"prompt": {"ids": ids}, "position": -1, "units": [u for n in out[i]["claim"]["units"] for u in by[n]["units"]],
               "op": out[i]["claim"]["op"], "predict": out[i]["claim"]["predict"], **word} for i in idx]
    got = []  # (in chunks: check_many keeps every row's log p over the vocabulary on the device, 4.4 GB for 250 claims)
    for k in range(0, len(claims), CHECK_CHUNK):
        got += cc.check_many(nr, claims[k:k + CHECK_CHUNK])
    drop = set()
    for i, r in zip(idx, got):
        if not r.get("pass"):
            drop.add(i)
            continue
        out[i]["specific"] = r["specific"]
        out[i]["measured"].update({"contrast_before": r["contrast_before"], "contrast_after": r["contrast_after"],
                                   "control_change": r["control_change"], "side_kl": r["side_kl"]})
        if not r["specific"] and out[i]["kind"] in ("zero", "steer", "set"):
            out[i]["a"] = out[i]["a"].replace("\nClaim:", " Random neurons of the same layers do the same.\nClaim:", 1)
    out = [r for i, r in enumerate(out) if i not in drop]
    for k, r in enumerate(out):
        r.update({"id": f"q1/{source}/{rec['id']}/{r['kind']}/{k}", "source": source, "context_id": rec["id"],
                  "family": rec.get("task"), "circuit_id": circ["circuit_id"], "adapter": rec.get("adapter"), "ctx": ids,
                  "answer": {"id": y, "text": tok.decode([y])}, "contrast": CONTRAST if rec.get("answer_tokens") else "token_v0",
                  "keys": [by[n]["key"] for n in r["answer_units"] if n in by]})
    return circ, out


def nr_alts(cc) -> int:
    return cc.nb.ALTERNATIVES


def side_chats(tok, path: Path, rng: random.Random):
    import neuron_builder as nb
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    pick = rng.sample([rows[i] for i in SIDE_ROWS if i < len(rows)], SIDE_N)
    out = []
    for r in pick:
        p = nb.chat(tok, r["user"])
        a = tok(r["answer"], add_special_tokens=False)["input_ids"][:SIDE_TOK]
        out.append((p + a, len(p) - 1))
    return out


def gen(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import claim_check as cc
    import neuron_builder as nb
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(dev).eval()
    if a.adapter:  # an organism's circuits: its LoRA on (records with "adapter" name it; every record of a run shares one)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    nr = nb.Neurons(model, tok)
    nr.load(a.thr)
    nr.thr = None  # the model itself (circuit_finder's runs); T2's maxima and massive cuts kept
    eng = Engine(nr, cc)
    side = side_chats(tok, a.side, random.Random(0)) if a.side else None
    files = sorted(a.records.glob("*.json"))
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        files = files[k::n]
    files = files[:a.limit] if a.limit else files
    cpath = a.out.with_name(a.out.stem + ".circuits.jsonl")
    done = {json.loads(x)["context_id"] for x in cpath.read_text().splitlines()} if cpath.exists() else set()
    t0, nq = time.time(), 0
    with a.out.open("a") as fq, cpath.open("a") as fc:
        for j, f in enumerate(files):
            rec = json.loads(f.read_text())
            if rec.get("id") in done or not rec.get("ids") or rec.get("valid") is False or rec.get("skipped"):
                continue
            if bool(rec.get("adapter")) != bool(a.adapter):
                raise SystemExit(f"{f.name}: record adapter {rec.get('adapter')!r}, run adapter {a.adapter!r}")
            ap = json.loads((a.groups / f.name).read_text()) if a.groups and (a.groups / f.name).exists() else None
            rng = random.Random(zlib.crc32(f"{a.seed}/{rec['id']}".encode()))
            try:
                circ, qs = text_questions(nr, cc, eng, tok, rec, ap, rng, side, a.source)
            except Exception as e:  # noqa: BLE001  (one bad record must not stop a shard; logged)
                print(json.dumps({"record": rec.get("id"), "error": f"{type(e).__name__}: {e}"}), flush=True)
                continue
            for q in qs:
                fq.write(json.dumps(q) + "\n")
            fc.write(json.dumps(circ) + "\n")
            fq.flush()
            fc.flush()
            nq += len(qs)
            if (j + 1) % 5 == 0:
                print(json.dumps({"records": j + 1, "questions": nq, "seconds": round(time.time() - t0, 1)}), flush=True)
    print(json.dumps({"records": len(files), "questions": nq, "seconds": round(time.time() - t0, 1), "out": str(a.out)}), flush=True)


# ---------------------------------------------------------------- split (CPU)

DEV_CTX = 0.1              # contexts in dev (test: --test-share; families held out whole: --family-share, from 10
# families; item keys never asked in training, the unseen_group slice: --key-share)


def h(s: str, seed: int) -> float:
    return zlib.crc32(f"{seed}/{s}".encode()) / 2 ** 32


def fixup(r: dict) -> dict | None:
    """Rows of earlier revisions in this revision's terms: interact in signed terms (recomputed from its measured
    numbers), min_edit's edits named without positions."""
    if r["kind"] == "interact":
        y = q_tok(r["answer"]["text"])
        got = interact_row(r["units"], r["measured"]["pair"], r["measured"]["alone"], y)
        if got is None:
            return None
        r["q"], r["a"], r["a_key"], c = got
        r["cls"] = "interact/" + c
    if r["kind"] == "min_edit":
        import re
        strip = lambda t: re.sub(r"(L\d+\.[NH]\d+[+-]?)@\d+", r"\1", t)  # noqa: E731
        r["q"], r["a"], r["a_key"] = strip(r["q"]), strip(r["a"]), strip(r["a_key"])
        r["options"] = [strip(o) for o in r["options"]]
    return r


def split(a) -> None:
    rows, circs = [], {}
    for p in a.gen:
        rows += [x for x in (fixup(json.loads(x)) for x in p.read_text().splitlines()) if x is not None]
        cp = p.with_name(p.stem + ".circuits.jsonl")
        for x in cp.read_text().splitlines():
            c = json.loads(x)
            circs[c["circuit_id"]] = c
    sealed = {q["id"] for p in (a.sealed or []) for q in json.loads(p.read_text())["questions"]}
    n0 = len(rows)
    sealed_hit = sorted({r["context_id"] for r in rows} & sealed)
    rows = [r for r in rows if r["context_id"] not in sealed]  # the circuit push's sealed test texts never appear here
    fams = sorted({r["family"] for r in rows})
    test_fams = {f for f in fams if h("fam/" + f, a.seed) < a.family_share} if len(fams) >= 10 else set()
    test_keys = {k for r in rows for k in r["keys"] if h("key/" + k, a.seed) < a.key_share}

    def ctx_split(r):
        if r["family"] in test_fams:
            return "test"
        u = h("ctx/" + r["context_id"], a.seed)
        return "test" if u < a.test_share else "dev" if u < a.test_share + DEV_CTX else "train"
    for r in rows:
        s = ctx_split(r)
        hit = any(k in test_keys for k in r["keys"])
        if s == "train" and hit:
            s = "drop"  # a held-out key is never asked in training
        r["split"] = s
        r["tags"] = (["unseen_ctx"] + (["unseen_group"] if hit else []) + (["unseen_family"] if r["family"] in test_fams else [])
                     if s in ("test", "dev") else [])
    train = [r for r in rows if r["split"] == "train"]
    # the name-prior: each key's most common training class per kind (a menu kind: the option most often right)
    prior = collections.defaultdict(collections.Counter)
    kind_prior = collections.defaultdict(collections.Counter)
    for r in train:
        kind_prior[r["kind"]][r["a_key"] if r["options"] else r["cls"]] += 1
        if not r["options"]:
            for k in r["keys"]:
                prior[(r["kind"], k)][r["cls"]] += 1
    seen = collections.defaultdict(set)  # (kind, key) -> classes over every split: the same item, different outcomes
    for r in rows:
        if not r["options"] and len(r["keys"]) == 1:
            seen[(r["kind"], r["keys"][0])].add(r["cls"])
    for r in rows:
        if not r["options"] and len(r["keys"]) == 1 and len(seen[(r["kind"], r["keys"][0])]) > 1:
            r["tags"] = r["tags"] + ["same_group_diff_outcome"]
    rng = random.Random(a.seed)
    out = {"train": train}
    tb = collections.defaultdict(list)  # train_bal: per kind, no class above the second-largest class's count
    for r in train:
        tb[(r["kind"], r["cls"])].append(r)
    out["train_bal"] = []
    for kind in sorted({k for k, _ in tb}):
        cl = sorted((v for (k, _), v in tb.items() if k == kind), key=len, reverse=True)
        cap = len(cl[1]) if len(cl) > 1 else len(cl[0])
        for v in cl:
            out["train_bal"] += rng.sample(v, min(cap, len(v)))
    for s in ("dev", "test"):
        pool = [r for r in rows if r["split"] == s]
        by = collections.defaultdict(list)
        for r in pool:
            by[(r["kind"], r["cls"])].append(r)
        bal = []
        for kind in sorted({k for k, _ in by}):
            cl = {c: v for (k, c), v in by.items() if k == kind}
            menu = any(r["options"] for v in cl.values() for r in v)
            if menu:  # a menu kind's answer is one of its options: balanced by construction (shuffled options)
                bal += [r for v in cl.values() for r in v]
                continue
            big = {c: v for c, v in cl.items() if len(v) >= a.min_class}
            if len(big) < 2:
                continue
            n = min(len(v) for v in big.values())
            for c, v in sorted(big.items()):
                bal += rng.sample(v, n)
        out[s] = bal
    a.out.mkdir(parents=True, exist_ok=True)
    for s, rs in out.items():
        with (a.out / f"{s}.jsonl").open("w") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
    used = {r["circuit_id"] for rs in out.values() for r in rs}
    with (a.out / "circuits.jsonl").open("w") as fh:
        for cid in sorted(used):
            fh.write(json.dumps(circs[cid]) + "\n")
    # baselines on each held-out split: constant (each kind's most common training answer) and the name-prior
    zs = collections.defaultdict(list)
    for r in train:
        if r["kind"] == "zero" and r["keys"]:
            zs[r["keys"][0]].append(r["measured"]["change"])
    zmean = {k: sum(v) / len(v) for k, v in zs.items()}

    def key_of(r, n):  # a unit name's key ("G12", "L18.N2201+", "L5.H3": the name without its position)
        return n if n.startswith("G") else n.split("@")[0]
    base = {}
    for s in ("dev", "test"):
        res = collections.defaultdict(lambda: collections.Counter())
        for r in out[s]:
            truth = r["a_key"] if r["options"] or r["kind"] == "rank" else r["cls"]
            const = kind_prior[r["kind"]].most_common(1)[0][0] if kind_prior[r["kind"]] else None
            mean = lambda n: zmean.get(key_of(r, n), 0.0)  # noqa: E731  (a unit's mean training zeroing change)
            if r["kind"] == "rank":  # constant: the order shown; name-prior: by the units' mean training changes
                const = ">".join(r["units"])
                np_ = ">".join(sorted(r["units"], key=mean))
            elif r["kind"] == "necessary":  # name-prior: the option whose unit dropped the lead most in training
                np_ = min(r["options"], key=mean)
            elif r["options"]:
                np_ = const
            else:
                c_ = collections.Counter()
                for k in r["keys"]:
                    c_ += prior.get((r["kind"], k), collections.Counter())
                np_ = c_.most_common(1)[0][0] if c_ else const
            for slice_ in ["all"] + r["tags"]:
                res[(r["kind"], slice_)]["n"] += 1
                res[(r["kind"], slice_)]["constant"] += const == truth
                res[(r["kind"], slice_)]["name_prior"] += np_ == truth
                res[(r["kind"], slice_)]["chance"] += 1 / len(r["options"]) if r["options"] else 0
        base[s] = {f"{k}|{sl}": {"n": v["n"], "constant": round(v["constant"] / v["n"], 3), "name_prior": round(v["name_prior"] / v["n"], 3),
                                 **({"chance": round(v["chance"] / v["n"], 3)} if v["chance"] else {})}
                   for (k, sl), v in sorted(res.items())}
    count = lambda rs: dict(collections.Counter(r["cls"] for r in rs).most_common())  # noqa: E731
    manifest = {"schema": "oq1", "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "seed": a.seed, "inputs": [str(p) for p in a.gen],
                "rows": {s: len(rs) for s, rs in out.items()}, "sealed_contexts_removed": sealed_hit,
                "sealed_rows_removed": n0 - len(rows), "dropped_train_rows_naming_heldout_keys": sum(r["split"] == "drop" for r in rows),
                "contexts": {s: len({r["context_id"] for r in rs}) for s, rs in out.items()},
                "test_families": sorted(test_fams), "heldout_keys": len(test_keys),
                "shares": {"test": a.test_share, "dev": DEV_CTX, "keys": a.key_share, "families": a.family_share}, "classes": {s: count(rs) for s, rs in out.items()},
                "baselines": base, "margins": {"LOW": LOW, "HIGH": HIGH, "BIG": BIG, "GAP": GAP, "STEER": STEER}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "contexts", "test_families", "heldout_keys")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("records", type=Path)
    g.add_argument("out", type=Path)
    g.add_argument("--thr", type=Path, required=True, help="T2's file (neuron_builder.Neurons.save): maxima and massive cuts")
    g.add_argument("--groups", type=Path, help="neuron_motifs apply's directory (shared groups, joined by file name)")
    g.add_argument("--side", type=Path, help="general.jsonl: the chats min_edit's side effects are measured on")
    g.add_argument("--source", default="finder")
    g.add_argument("--model", default="Qwen/Qwen3-8B")
    g.add_argument("--limit", type=int)
    g.add_argument("--shard")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--device")
    g.add_argument("--dtype", default="bfloat16")
    g.add_argument("--adapter", help="an organism's LoRA (the records' own circuits were found with it on)")
    s = sub.add_parser("split")
    s.add_argument("out", type=Path)
    s.add_argument("gen", type=Path, nargs="+")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--min-class", type=int, default=10)
    s.add_argument("--test-share", type=float, default=0.25)
    s.add_argument("--key-share", type=float, default=0.1)
    s.add_argument("--family-share", type=float, default=0.1)
    s.add_argument("--sealed", type=Path, nargs="*", help="sealed question sets (testset_8b.json): their contexts are removed")
    a = ap.parse_args()
    {"gen": gen, "split": split}[a.cmd](a)


if __name__ == "__main__":
    main()
