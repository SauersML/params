"""The reader oracle as text (#2951, the oracle push 10-16). rc1 showed that the injection reader ignores the circuit:
each unit's injected write was L2-normalized, which leaves only its sign. This reader gets the circuit as text instead,
with everything that changes from context to context written out as numbers: the text, the model's answer and its lead,
the contrast pair the finder measured on, the mover heads (where they look, their measured shares), and each neuron
group of the circuit (varteach's tested label where it passed, its members, each member's value as a share of its
corpus maximum, its noising and denoising shares from the finder record, the tokens its write promotes and suppresses
through the unembedding, and its direct write on the answer's lead). The questions, answers and the split are rc's own
(reads_circuit.py build; g2.py gen --dense), so the claim checker and the outcome classes score both readers alike.

Never in the input: apply's per-group "effect" and per-member "drop" (neuron_motifs apply: the contrast drop when the
group is zeroed, i.e. the n_zero answer itself).

  text_reader.py facts --rc RC --out FACTS.json        per context: the serialized circuit's numbers (Mac CPU: the
                                                      weights are read from the safetensors, no forward pass)
  text_reader.py check --rc RC --facts FACTS.json      a logistic model on the serialized features against the
                                                      name-prior, held-out contexts (must win before any training)
  text_reader.py rows --rc RC --facts FACTS.json --out DIR   SFT rows (tinker_train.py sft) and test questions per input
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reads_circuit import MIN_CLASS, answer_class  # noqa: E402

HOME = Path.home()
RECORDS = HOME / "mpd-data/cluster/finder_8b_dev4"
LABELS = HOME / "mpd-data/graph_oracle/groups8b/labels_dev4.jsonl"
THR = HOME / "mpd-data/cluster/t2_8b/out/thr.pt"
MODEL = "Qwen/Qwen3-8B"
STEER = 3.0
LENS = 4  # promoted and suppressed tokens shown per group
UNIT = re.compile(r"^L(\d+)\.N(\d+)([+-])@(-?\d+)$")


def parse(u: str) -> tuple[int, int, int, int]:
    m = UNIT.match(u)
    return int(m[1]), int(m[2]), 1 if m[3] == "+" else -1, int(m[4])


def rec_file(rid: str) -> str:
    return rid.replace("/", "__") + ".json"


class Weights:
    """Qwen3-8B's final norm, unembedding and down_proj columns, read from the local safetensors (CPU, no forward)."""

    def __init__(self):
        import torch
        from safetensors import safe_open

        self.torch = torch
        snap = next((HOME / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots").iterdir())  # (the hub calls the
        # cache incomplete: it lacks files we never read)
        wm = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        self.f = {k: safe_open(str(snap / v), framework="pt") for k, v in wm.items()}
        self.g = self.f["model.norm.weight"].get_tensor("model.norm.weight").float()
        self._wu = None
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(snap))

    def col(self, layer: int, neuron: int):
        k = f"model.layers.{layer}.mlp.down_proj.weight"
        return self.f[k].get_slice(k)[:, neuron:neuron + 1][:, 0].float()

    @property
    def wu(self):
        if self._wu is None:
            self._wu = self.f["lm_head.weight"].get_tensor("lm_head.weight")  # bf16 [V, d]
        return self._wu

    def rows(self, ids: list[int]):
        return self.wu[ids].float()


def q_tok(s: str) -> str:
    return "'" + s.replace("\n", "\\n") + "'"


def context_facts(W, mx, e: dict, rec: dict, labels: dict, has_apply: bool, ref: dict | None = None) -> dict:
    """The numbers the text input states for one context (one rc text): every group the questions name, from rc's own
    groups map, with its members' finder measurements, plus the context's answer, lead, contrast pair and heads. With
    ref (measure's: claim_check's reference at the answer position) the answer, alternatives and lead are claim_check's
    and every direct effect is in its bits (through the final norm at the reference's RMS); without, the finder's
    word-level answer and alternatives and logits before the norm."""
    torch = W.torch
    ids = e["ctx"]
    T = len(ids)
    toks = [W.tok.decode([t]) for t in ids]
    node = {(d["layer"], d["neuron"], d["pos"]): d for d in rec["nodes"]}
    if ref:
        y, alts = ref["top"][0][0], [t for t, _ in ref["top"][1:]]
        lps = torch.tensor([l for _, l in ref["top"]])
        w = torch.softmax(lps[1:], 0)
        U = W.rows([y] + alts)
        scale = 1 / (ref["rms"] * math.log(2))
        du = (U[0] - (w[:, None] * U[1:]).sum(0)) * W.g * scale  # a write's direct effect on claim_check's lead, bits
        lead = float((lps[0] - torch.logsumexp(lps[1:], 0)) / math.log(2))
        probs = [math.exp(l) for _, l in ref["top"]]
        size = 1 / (ref["rms"] * math.sqrt(W.g.shape[0]))  # a write's norm as a share of the residual's at the answer
    else:
        y, alts = rec["answer"], rec["alternatives"]
        U = W.rows([y] + alts)
        du = (U[0] - U[1:].mean(0)) * W.g  # a write's direct effect on the lead (logits before the final norm's 1/rms)
        lead, probs, size = rec["contrast"]["clean"], None, None
    groups = {}
    for gname, members in e["groups"].items():
        ms = []
        write = torch.zeros(W.g.shape[0])
        for u in members:
            L, N, pole, pos = parse(u)
            d = node.get((L, N, pos if pos >= 0 else T + pos), {})
            v = d.get("clean")
            if v is None:  # the rc example's block holds the value
                v = next((x[2] for b in e["blocks"] if "write" in b for x in (b["write"] if isinstance(b["write"][0], list) else [b["write"]])
                          if x[0] == L and x[1] == N), 0.0)
            c = W.col(L, N)
            at_answer = (pos if pos >= 0 else T + pos) == T - 1
            if at_answer:
                write += v * c
            m = float(mx[L, N])
            ms.append({"name": u, "layer": L, "neuron": N, "pole": pole, "pos": pos, "tok": toks[pos], "value": v,
                       "max": m, "frac": abs(v) / m if m else 0.0, "corrupt": d.get("corrupt"),
                       "size": float(abs(v) * c.norm() * size) if size else None,
                       "steer_size": float(STEER * m * c.norm() * size) if size else None,
                       "lost": d.get("lost"), "restored": d.get("restored"), "stage": d.get("stage"),
                       "direct": float(v * (c @ du)) if at_answer else None,
                       "direct_steer": float(STEER * m * pole * (c @ du)) if at_answer else None})
        gid = int(gname[1:]) if gname[1:].isdigit() else None
        lab = labels.get(gid) if has_apply and gid is not None else None
        groups[gname] = {"members": ms, "label": lab, "write_direct": float(write @ du) if write.abs().sum() else None,
                         "_write": write}
    # each group's answer-position write through the unembedding: promoted and suppressed tokens (one matmul)
    named = [g for g in groups if groups[g]["write_direct"] is not None]
    if named:
        X = torch.stack([groups[g]["_write"] * W.g for g in named], 1).to(torch.bfloat16)
        lo = (W.wu @ X).float()
        top, bot = lo.topk(LENS, 0).indices, (-lo).topk(LENS, 0).indices
        for j, g in enumerate(named):
            groups[g]["promotes"] = [W.tok.decode([int(t)]) for t in top[:, j]]
            groups[g]["suppresses"] = [W.tok.decode([int(t)]) for t in bot[:, j]]
    for g in groups.values():
        del g["_write"]
    heads = [{"name": f"L{r['layer']}.H{r['head']}@{r['pos']}", "attends": [[toks[q], q, w] for q, w in r.get("attends", [])[:3]],
              "lost": r.get("lost"), "restored": r.get("restored")} for r in rec.get("routes", [])]
    return {"id": e["id"], "tokens": toks, "user": rec.get("user"), "corrupt_user": rec.get("corrupt_user"),
            "answer": W.tok.decode([y]), "corrupt_answer": rec.get("corrupt_answer_text"), "alternatives": [W.tok.decode([a]) for a in alts],
            "probs": probs, "bits": bool(ref), "lead": lead, "lead_corrupt": rec["contrast"]["corrupt"], "heads": heads, "groups": groups,
            "met": rec.get("met"), "size": rec.get("size")}


def load_labels(path: Path = LABELS) -> dict:
    """varteach's group labels whose short or long test passed (the output label only where its own test passed)."""
    labels = {}
    for x in path.read_text().splitlines():
        r = json.loads(x)
        sc = r.get("scores") or {}
        if any((sc.get(k) or {}).get("pass") for k in ("short", "long")):
            labels[int(r["group"][1:])] = {"short": r["short"], "output": r["output"] if (r.get("output_scores") or {}).get("pass") else None}
    return labels


def record_items(rec: dict, app: dict | None) -> dict:
    """A finder record's items as oq.py names them: each shared group "G<id>" of apply's record (its members at their
    positions) and each circuit neuron outside every group under its own unit name ("L18.N2201+@12")."""
    name = lambda d: f"L{d['layer']}.N{d['neuron']}{'+' if d['pole'] > 0 else '-'}@{d['pos']}"  # noqa: E731
    items = {f"G{g['id']}": [name(m) for m in g["members"]] for g in (app or {}).get("groups", [])}
    grouped = {u for us in items.values() for u in us}
    items.update({name(d): [name(d)] for d in rec["nodes"] if name(d) not in grouped})
    return items


def record_facts(W, mx, rec: dict, app: dict | None, labels: dict, ref: dict | None) -> dict:
    """context_facts for any finder record (the reader as a tool: a circuit the agent just found)."""
    e = {"ctx": rec["ids"], "groups": record_items(rec, app), "blocks": [], "id": rec["id"]}
    return context_facts(W, mx, e, rec, labels, app is not None, ref)


def measure(a) -> None:
    """GPU: per rc context, in claim_check's reference (the weak-zeroed model, THR), the final norm's input RMS at the
    answer position and the six likeliest next tokens with their log-probabilities (claim_check's answer and its
    ALTERNATIVES), so that a write's direct effect on the lead is stated in claim_check's bits."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
    import neuron_builder as nb

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    nr = nb.Neurons(m.eval().requires_grad_(False), tok)
    nr.load(a.thr)
    norm = nr.core.model.norm
    cap = {}
    norm.register_forward_pre_hook(lambda mod, args: cap.__setitem__("x", args[0][:, -1].float()))
    eps = getattr(norm, "variance_epsilon", 1e-6)
    out = {}
    for f in ("train_correct.jsonl", "test_correct.jsonl"):
        for x in (a.rc / f).read_text().splitlines():
            e = json.loads(x)
            if e["id"] in out:
                continue
            ids = e["ctx"]
            with torch.no_grad():
                lp = torch.log_softmax(nr.logits(ids, [len(ids) - 1])[0, 0].float(), -1)
            v, i = lp.topk(nb.ALTERNATIVES + 1)
            out[e["id"]] = {"rms": float((cap["x"][0].pow(2).mean() + eps).sqrt()), "top": [[int(t), float(l)] for t, l in zip(i, v)]}
            print(e["id"], out[e["id"]]["rms"], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))


def gen(a) -> None:
    """GPU: more questions from rc's training contexts (--split test: its held-out ones, for evaluation): the same four kinds (g2.py's questions, answers
    and claim checks) about each member neuron of the groups with two or more (a single neuron's group is already
    asked), and SETS more sets of two to four members, in rc's row format with the context's groups map unchanged (the
    input is the same circuit; an asked unit is a member's name)."""
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
    import claim_check as cc
    import g2
    import neuron_builder as nb

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    nr = nb.Neurons(m.eval().requires_grad_(False), tok)
    nr.load(a.thr)
    rng = random.Random(a.seed)
    first = {}
    for x in (a.rc / f"{a.split}_correct.jsonl").read_text().splitlines():
        e = json.loads(x)
        first.setdefault(e["id"], e)
    with a.out.open("w") as fh:
        for cid, e in first.items():
            ids = e["ctx"]
            y = q_tok(tok.decode([json.loads((a.records / "out" / rec_file(cid)).read_text())["answer"]]))  # (g2's answer)
            units = [u for us in e["groups"].values() if len(us) > 1 for u in us]
            asks = [(cat, [u], {"steer": g2.STEER} if cat == "n_steer" else "zero") for u in units for cat in ("n_zero", "n_sign", "n_steer")]
            pool = [u for us in e["groups"].values() for u in us]
            for _ in range(a.sets if len(pool) >= 2 else 0):
                asks.append(("n_set", rng.sample(pool, rng.randint(2, min(4, len(pool)))), "zero"))
            probe = [{"prompt": {"ids": ids}, "position": -1, "units": us, "op": op, "predict": {"top_changes": True}} for _, us, op in asks]
            got = cc.check_many(nr, probe)
            stated, texts = [], []
            for (cat, us, op), r in zip(asks, got):
                if "error" in r:
                    continue
                if cat == "n_sign":
                    ch = r["change"]
                    pred = {"contrast_change": {"op": "<=", "value": -g2.HALF}} if ch <= -g2.HALF else \
                        {"contrast_change": {"op": ">=", "value": g2.HALF}} if ch >= g2.HALF else {"contrast_change": {"op": "within", "value": g2.HALF}}
                    text = (f"{us[0]} supports {y}." if ch <= -g2.HALF else f"{us[0]} works against {y}." if ch >= g2.HALF else
                            f"{us[0]} barely matters for {y}.")
                else:
                    clause, pred = g2.outcome(r, y)
                    verb = "Turning " + us[0] + " up" if cat == "n_steer" else "Zeroing " + g2.listing(us) + (" together" if len(us) > 1 else "")
                    text = f"{verb}: {clause}."
                stated.append({"prompt": {"ids": ids}, "position": -1, "units": us, "op": op, "predict": pred})
                texts.append((cat, us, op, text, pred))
            final = cc.check_many(nr, stated)
            n = 0
            for (cat, us, op, text, pred), r in zip(texts, final):
                if not r.get("pass"):
                    continue
                if not r["specific"]:
                    text += " Random neurons of the same layers do the same."
                q = {"n_steer": f"What happens to the model's next token {y} if {us[0]} is turned up?",
                     "n_zero": f"What happens to the model's next token {y} if {us[0]} is zeroed?",
                     "n_set": f"What happens to the model's next token {y} if {g2.listing(us)} are zeroed together?",
                     "n_sign": f"Does {us[0]} support the model's next token {y} or work against it?"}[cat]
                claim = {"units": us, "op": op, "predict": pred}
                fh.write(json.dumps({"cat": cat, "form": "members", "ctx": ids, "q": q, "a": f"{text}\nClaim: {json.dumps(claim)}",
                                     "claim": claim, "groups": e["groups"], "specific": r["specific"], "id": cid, "text": e["text"]}) + "\n")
                n += 1
            fh.flush()
            print(cid, len(asks), n, flush=True)


def claims(a) -> None:
    """GPU: each eval file's answers checked by claim_check (the claim an answer ends with, run on the reference with
    its random-unit controls), as g2.py eval checks them: the rows again with claim_pass and claim_specific."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
    import claim_check as cc
    import neuron_builder as nb

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    nr = nb.Neurons(m.eval().requires_grad_(False), tok)
    nr.load(a.thr)
    test = [json.loads(x) for x in (a.rc / "test_correct.jsonl").read_text().splitlines()]
    for f in a.evals:
        rs = [json.loads(x) for x in f.read_text().splitlines()]
        cl, idx = [], []
        for j, (r, e) in enumerate(zip(rs, test)):
            try:
                c = json.loads(r["got"].split("Claim:", 1)[1].strip())
                us = [u for n in c["units"] for u in e["groups"].get(n, [n])]
                cl.append({"prompt": {"ids": e["ctx"]}, "position": -1, "units": us, "op": c["op"], "predict": c["predict"]})
                idx.append(j)
            except (IndexError, ValueError, KeyError, TypeError, AttributeError):
                pass
        res = dict(zip(idx, cc.check_many(nr, cl)))
        out = a.out / (f.stem + ".claims.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for j, r in enumerate(rs):
                x = res.get(j, {})
                fh.write(json.dumps({**r, "claim_pass": bool(x.get("pass")), "claim_specific": bool(x.get("specific"))}) + "\n")
        print(f.name, "claims passing", sum(bool(res.get(j, {}).get("pass")) for j in range(len(rs))) / len(rs), flush=True)


def facts(a) -> None:
    import torch

    W = Weights()
    mx = torch.load(THR, map_location="cpu")["max"]
    labels = load_labels()
    apply = {p.name for p in (RECORDS / "apply").glob("*.json")}
    ref = json.loads(a.measure.read_text()) if a.measure else {}
    out = {}
    for f in ("train_correct.jsonl", "test_correct.jsonl"):
        for x in (a.rc / f).read_text().splitlines():
            e = json.loads(x)
            if e["id"] in out:
                continue
            rec = json.loads((RECORDS / "out" / rec_file(e["id"])).read_text())
            out[e["id"]] = context_facts(W, mx, e, rec, labels, rec_file(e["id"]) in apply, ref.get(e["id"]))
            print(e["id"], len(out[e["id"]]["groups"]), flush=True)
    a.out.write_text(json.dumps(out))


# ---- the serialized input ----

def num(x, nd=2) -> str:
    return "?" if x is None else f"{x:+.{nd}f}"


def serialize(F: dict, how: str = "full") -> str:
    """The circuit as text. how: "full" (every measured number), "names" (groups and members' names only: the
    names-only control), "none" (the text alone)."""
    toks = F["tokens"]
    pr = lambda k: f" ({F['probs'][k]:.2f})" if how == "full" and F.get("probs") else ""  # noqa: E731
    lines = [f"Text: {F['user']!r}", f"The model's next token: {q_tok(F['answer'])}{pr(0)}; the alternatives: "
             f"{', '.join(q_tok(t) + pr(k + 1) for k, t in enumerate(F['alternatives']))}" +
             (f"; its lead over them: {F['lead']:.1f} bits." if how == "full" else ".")]
    unit = " bits" if F.get("bits") else ""
    if how == "none":
        return "\n".join(lines) + "\n"
    if how == "full":
        lines.append(f"Contrast pair: with {F['corrupt_user']!r} the model gives {q_tok(F['corrupt_answer'] or '?')}. "
                     f"Shares below: of the gap between the two texts, lost when the unit takes its value from the contrast "
                     f"text (noising) and restored when it alone keeps its value in the contrast text (denoising).")
        if F["heads"]:
            lines.append("Heads at the answer position:")
            for h in F["heads"][:6]:
                looks = ", ".join(f"{q_tok(t)} @{q} {w:.2f}" for t, q, w in h["attends"])
                lines.append(f"  {h['name']} looks at {looks}; noising {num(h['lost'])}, denoising {num(h['restored'])}")
    elif F["heads"]:
        lines.append("Heads at the answer position:")
        for h in F["heads"][:6]:
            lines.append(f"  {h['name']} looks at {', '.join(f'{q_tok(t)} @{q}' for t, q, _ in h['attends'])}")
    lines.append("Groups:")
    for gname, g in F["groups"].items():
        lab = f" ({g['label']['short']})" if g.get("label") else ""
        n = len(g["members"])
        lines.append(f"  {gname}{lab}, {n} neuron{'' if n == 1 else 's'}:")
        if how == "full" and g.get("promotes"):
            lines.append(f"    write at the answer: lead {num(g['write_direct'], 1)}{unit}; promotes {', '.join(q_tok(t) for t in g['promotes'])}; "
                         f"suppresses {', '.join(q_tok(t) for t in g['suppresses'])}")
        for m in g["members"]:
            at = f"{m['name']} at {q_tok(m['tok'])}"
            if how != "full":
                lines.append(f"    {at}")
                continue
            extra = ""
            if m["direct"] is not None:
                extra = f"; its write on the lead {num(m['direct'], 1)}{unit}, turned up {num(m['direct_steer'], 1)}{unit}"
                if m.get("size") is not None:
                    extra += f" (sizes {m['size']:.2f} and {m['steer_size']:.2f} of the residual)"
            lines.append(f"    {at}: value {m['value']:.3g} = {m['frac']:.2f} of max (contrast text {num(m['corrupt'], 3) if m['corrupt'] is not None else '?'}); "
                         f"noising {num(m['lost'])}, denoising {num(m['restored'])}{extra}")
    return "\n".join(lines) + "\n"


def mismatch(F: dict, D: dict) -> dict:
    """F's circuit (its text, groups, members, positions and labels) with every measured number taken from another
    context D, group by group and member by member in order (cycled): the numbers carry nothing about F."""
    G = json.loads(json.dumps(F))
    for k in ("lead", "lead_corrupt"):
        G[k] = D[k]
    dh = D["heads"] or [{"attends": [], "lost": None, "restored": None}]
    for j, h in enumerate(G["heads"]):
        d = dh[j % len(dh)]
        h["lost"], h["restored"] = d["lost"], d["restored"]
        h["attends"] = [[t, q, d["attends"][i][2] if i < len(d["attends"]) else 0.0] for i, (t, q, _) in enumerate(h["attends"])]
    dg = list(D["groups"].values())
    for j, g in enumerate(G["groups"].values()):
        d = dg[j % len(dg)]
        for k in ("write_direct", "promotes", "suppresses"):
            g[k] = d.get(k)
        dm = d["members"]
        for i, m in enumerate(g["members"]):
            x = dm[i % len(dm)]
            for k in ("frac", "value", "corrupt", "lost", "restored", "direct", "direct_steer"):
                m[k] = x[k]
    return G


# ---- the feasibility check: a logistic model on the serialized numbers against the name-prior ----

def unit_members(F: dict, u: str) -> list[dict]:
    """An asked item's members: a group's, or the one member a neuron's unit name names (gen's member questions)."""
    if u in F["groups"]:
        return F["groups"][u]["members"]
    return [m for g in F["groups"].values() for m in g["members"] if m["name"] == u][:1]


def qfeatures(e: dict, F: dict) -> list[float]:
    ms = [m for g in e["claim"]["units"] for m in unit_members(F, g)]
    T = len(F["tokens"])
    s = lambda k: sum((m[k] or 0.0) for m in ms)  # noqa: E731
    mxa = lambda k: max(abs(m[k] or 0.0) for m in ms)  # noqa: E731
    at = [m for m in ms if (m["pos"] if m["pos"] >= 0 else T + m["pos"]) == T - 1]
    d = sum(m["direct"] for m in at) if at else 0.0
    ds = sum(m["direct_steer"] for m in at) if at else 0.0
    return [len(ms), math.log1p(len(ms)), len(at) / len(ms), sum(m["layer"] for m in ms) / len(ms) / 36,
            s("lost"), s("restored"), mxa("lost"), mxa("restored"), math.log1p(abs(s("lost"))), math.log1p(abs(s("restored"))),
            max(m["frac"] for m in ms), sum(m["frac"] for m in ms), d, ds, math.copysign(math.log1p(abs(d)), d),
            math.copysign(math.log1p(abs(ds)), ds), F["lead"], math.log1p(max(F["lead"], 0)),
            sum(m["stage"] == "readout" for m in ms) / len(ms),
            math.sqrt(sum((m.get("size") or 0.0) ** 2 for m in at)), math.sqrt(sum((m.get("steer_size") or 0.0) ** 2 for m in at))]


def name_prior(train: list[dict], test: list[dict], key) -> list[str]:
    by = collections.defaultdict(collections.Counter)
    cat = collections.defaultdict(collections.Counter)
    for e in train:
        by[(e["cat"], key(e))][answer_class(e)] += 1
        cat[e["cat"]][answer_class(e)] += 1
    return [(by.get((e["cat"], key(e))) or cat[e["cat"]]).most_common(1)[0][0] for e in test]


def check(a) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    F = json.loads(a.facts.read_text())
    train = [json.loads(x) for x in (a.rc / "train_correct.jsonl").read_text().splitlines()]
    test = [json.loads(x) for x in (a.rc / "test_correct.jsonl").read_text().splitlines()]
    acc = lambda p, rs: sum(x == answer_class(e) for x, e in zip(p, rs)) / len(rs)  # noqa: E731
    shown = lambda e: "+".join(e["claim"]["units"])  # noqa: E731
    members = lambda e: frozenset(re.sub(r"@-?\d+$", "", u) for g in e["claim"]["units"] for u in e["groups"][g])  # noqa: E731
    res = {"chance": json.loads((a.rc / "claim.json").read_text())["chance"], "n_test": len(test)}
    res["name_prior_shown"] = acc(name_prior(train, test, shown), test)
    res["name_prior_members"] = acc(name_prior(train, test, members), test)
    # the subset where one group (its members, position-free) has different outcomes in different texts
    outs = collections.defaultdict(set)
    for e in train + test:
        outs[(e["cat"], members(e))].add((e["id"], answer_class(e)))
    varied = [i for i, e in enumerate(test) if len({c for _, c in outs[(e["cat"], members(e))]}) > 1
              and len({t for t, _ in outs[(e["cat"], members(e))]}) > 1]
    pred = {}
    for c in sorted({e["cat"] for e in test}):
        tr = [e for e in train if e["cat"] == c]
        te = [(i, e) for i, e in enumerate(test) if e["cat"] == c]
        X = np.array([qfeatures(e, F[e["id"]]) for e in tr])
        sc = StandardScaler().fit(X)
        if a.model == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced",
                                               random_state=0).fit(sc.transform(X), [answer_class(e) for e in tr])
        else:
            m = LogisticRegression(C=a.C, max_iter=5000, class_weight="balanced").fit(sc.transform(X), [answer_class(e) for e in tr])
        for (i, e), p in zip(te, m.predict(sc.transform(np.array([qfeatures(e, F[e["id"]]) for _, e in te])))):
            pred[i] = p
    p = [pred[i] for i in range(len(test))]
    res["logistic"] = acc(p, test)
    npm = name_prior(train, test, members)
    res["varied_n"] = len(varied)
    res["varied_logistic"] = sum(p[i] == answer_class(test[i]) for i in varied) / max(len(varied), 1)
    res["varied_name_prior"] = sum(npm[i] == answer_class(test[i]) for i in varied) / max(len(varied), 1)
    for c in sorted({e["cat"] for e in test}):
        ix = [i for i, e in enumerate(test) if e["cat"] == c]
        res[f"{c}_logistic"] = sum(p[i] == answer_class(test[i]) for i in ix) / len(ix)
        res[f"{c}_name_prior"] = sum(npm[i] == answer_class(test[i]) for i in ix) / len(ix)
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}, indent=1))


# ---- SFT rows, greedy answers on Tinker, the outcome-class score ----

def load_rc(rc: Path) -> tuple[list[dict], list[dict], dict]:
    """rc's training and held-out questions and each held-out context's donor (reads_circuit's fixed derangement: the
    next held-out context in sorted order)."""
    train = [json.loads(x) for x in (rc / "train_correct.jsonl").read_text().splitlines()]
    test = [json.loads(x) for x in (rc / "test_correct.jsonl").read_text().splitlines()]
    ts = sorted({e["text"] for e in test})
    first = {}
    for e in test:
        first.setdefault(e["text"], e["id"])
    donor = {first[c]: first[ts[(i + 1) % len(ts)]] for i, c in enumerate(ts)}
    return train, test, donor


def user_text(F: dict, e: dict, how: str, D: dict | None = None, first: bool = False) -> str:
    """The circuit, then the question (one context's questions share their prefix, cached when sampled); first: the
    question also before the circuit, so that the circuit's tokens are read knowing what is asked."""
    body = serialize(mismatch(F, D), "full") if how == "mismatch" else serialize(F, how)
    return (f"Question: {e['q']}\n\n" if first else "") + body + "\n" + e["q"]


def recap(F: dict, e: dict, how: str) -> str:
    """The answer's first line, "Read: ...": the asked groups' numbers copied from the input (summed over members),
    so that the outcome is written beside them (a mechanical plan before the answer; scoring drops it)."""
    T = len(F["tokens"])
    out = []
    for gname in e["claim"]["units"]:
        ms = unit_members(F, gname)
        pos = sorted({q_tok(m["tok"]) + f" @{m['pos'] if m['pos'] >= 0 else T + m['pos']}" for m in ms})
        x = f"{gname} {len(ms)} at {', '.join(pos)}"
        if how in ("full", "mismatch"):
            s = lambda k: sum(m[k] or 0.0 for m in ms)  # noqa: E731
            x += f"; noising {s('lost'):+.2f}, denoising {s('restored'):+.2f}, largest {max(m['frac'] for m in ms):.2f} of max"
            at = [m for m in ms if m["direct"] is not None]
            if at:
                x += f"; write on the lead {sum(m['direct'] for m in at):+.1f}, turned up {sum(m['direct_steer'] for m in at):+.1f}"
        out.append(x)
    return "Read: " + "; ".join(out)


def strip_read(text: str) -> str:
    return text.split("\n", 1)[1] if text.startswith("Read:") and "\n" in text else text


def rows(a) -> None:
    """SFT rows per training input (full, names) and test questions per input (full, names, mismatch, none); --balance:
    each question kind's answer classes repeated up to its largest class (the held-out set is balanced; the training
    set is half "unchanged"); --recap: a "Read:" line before the answer; --first: the question before the circuit too."""
    import random

    F = json.loads(a.facts.read_text())
    train, test, donor = load_rc(a.rc)
    held = {e["id"] for e in test}
    for f in a.extra or []:  # gen's questions (training contexts only; a held-out context's would be refused)
        more = [json.loads(x) for x in f.read_text().splitlines()]
        assert not {e["id"] for e in more} & held, f
        train += more
    if a.balance:
        rng = random.Random(0)
        by = collections.defaultdict(list)
        for e in train:
            by[(e["cat"], answer_class(e))].append(e)
        bal = []
        for c in sorted({e["cat"] for e in train}):
            ks = [k for k in by if k[0] == c]
            top = max(len(by[k]) for k in ks)
            for k in ks:  # (a class too rare to be in the balanced held-out set, flips, is kept as it is)
                bal += by[k] + ([rng.choice(by[k]) for _ in range(top - len(by[k]))] if len(by[k]) >= MIN_CLASS else [])
        train = bal
    a.out.mkdir(parents=True, exist_ok=True)
    for how in ("full", "names"):
        with (a.out / f"train_{how}.jsonl").open("w") as f:
            for k, e in enumerate(train):
                ans = (recap(F[e["id"]], e, how) + "\n" if a.recap else "") + e["a"]
                f.write(json.dumps({"prompt": [{"role": "user", "content": user_text(F[e["id"]], e, how, first=a.first)}],
                                    "completion": [{"role": "assistant", "content": ans}],
                                    "info": {"task": f"{e['id']}#{k}", "cat": e["cat"]}}) + "\n")
    for how in ("full", "names", "mismatch", "none"):
        with (a.out / f"test_{how}.jsonl").open("w") as f:
            for k, e in enumerate(test):
                f.write(json.dumps({"task": f"{e['id']}#{k}", "question": user_text(F[e["id"]], e, how, F[donor[e["id"]]], a.first),
                                    "cat": e["cat"], "a": e["a"], "id": e["id"]}) + "\n")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(next((HOME / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots").iterdir())))
    for f in sorted(a.out.glob("*.jsonl")):
        rs = [json.loads(x) for x in f.read_text().splitlines()]
        n = [len(tok(r["prompt"][0]["content"] if "prompt" in r else r["question"])["input_ids"]) for r in rs]
        print(f.name, len(rs), "prompt tokens: mean", sum(n) // len(n), "max", max(n), "total", sum(n))


def sample(a) -> None:
    """Greedy answers (temperature 0) from a Tinker state to a test file's questions, written as reads_circuit's eval
    rows ({"cat", "a", "got", "q"}); the spend to the log's spend.jsonl."""
    import tinker
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle" / "rl"))
    import tinker_train as tt

    qs = [json.loads(x) for x in a.questions.read_text().splitlines()]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    spend = tt.Spend(a.out.parent, a.base)
    service = tt.service_client()
    renderer, tokenizer = tt.renderer_for(a.base)
    state = a.state if not a.state or a.state.startswith("tinker://") else json.loads(Path(a.state).read_text())["state"]
    sampler = (service.create_training_client_from_state(state).save_weights_and_get_sampling_client() if state
               else service.create_sampling_client(base_model=a.base))
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=0.0, stop=renderer.get_stop_sequences())
    futs = []
    for q in qs:
        prompt = renderer.build_generation_prompt([{"role": "user", "content": q["question"]}])
        futs.append((q, prompt, sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)))
    with a.out.open("w") as f:
        for q, prompt, fut in futs:
            res = fut.result()
            hits = res.prompt_cache_hit_tokens or 0
            s = res.sequences[0]
            spend.add("sample", prefill=prompt.length - hits, cached=hits, sampled=len(s.tokens))
            got, _ = tt.answer_of(renderer, tokenizer, s.tokens)
            f.write(json.dumps({"task": q["task"], "id": q["id"], "cat": q["cat"], "a": q["a"], "got": got}) + "\n")
    print(json.dumps({"questions": len(qs), "usd": round(spend.total, 4)}))


def score(a) -> None:
    """Outcome-class match (reads_circuit's primary metric) per eval file, overall, per kind, and on the subset where
    one group has different outcomes in different texts; with the name-prior beside it."""
    train, test, _ = load_rc(a.rc)
    members = lambda e: frozenset(re.sub(r"@-?\d+$", "", u) for g in e["claim"]["units"] for u in e["groups"][g])  # noqa: E731
    outs = collections.defaultdict(set)
    for e in train + test:
        outs[(e["cat"], members(e))].add((e["id"], answer_class(e)))
    varied = {i for i, e in enumerate(test) if len({c for _, c in outs[(e["cat"], members(e))]}) > 1}
    prior = {k: name_prior(train, test, key) for k, key in (("shown", lambda e: "+".join(e["claim"]["units"])), ("members", members))}
    gold = [answer_class(e) for e in test]
    tab = {f"name-prior ({k})": [p == g for p, g in zip(v, gold)] for k, v in prior.items()}
    for f in a.evals:
        rs = [json.loads(x) for x in f.read_text().splitlines()]
        assert [r["a"] for r in rs] == [e["a"] for e in test], f
        tab[f"{f.parent.name}/{f.stem}"] = [answer_class({"cat": r["cat"], "a": strip_read(r["got"])}) == g for r, g in zip(rs, gold)]
    cats = sorted({e["cat"] for e in test})
    print(f"{'':34s} {'all':>6s}" + "".join(f"{c:>9s}" for c in cats) + f"  varied(n={len(varied)})")
    for k, v in tab.items():
        per = [sum(v[i] for i, e in enumerate(test) if e["cat"] == c) / sum(e["cat"] == c for e in test) for c in cats]
        print(f"{k:34s} {sum(v) / len(v):6.3f}" + "".join(f"{x:9.3f}" for x in per) +
              f"  {sum(v[i] for i in varied) / max(len(varied), 1):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("--rc", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--measure", type=Path, help="measure's output: claim_check's answer, alternatives and bits")
    c = sub.add_parser("check")
    c.add_argument("--rc", type=Path, required=True)
    c.add_argument("--facts", type=Path, required=True)
    c.add_argument("--C", type=float, default=0.3)
    c.add_argument("--model", choices=("logistic", "gbm"), default="logistic", help="gbm: gradient-boosted trees (the "
                   "input's information under a nonlinear reading, an upper reference for the logistic)")
    r = sub.add_parser("rows")
    r.add_argument("--rc", type=Path, required=True)
    r.add_argument("--facts", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--balance", action="store_true")
    r.add_argument("--extra", type=Path, nargs="*", help="more training questions (gen's)")
    r.add_argument("--recap", action="store_true")
    r.add_argument("--first", action="store_true")
    s = sub.add_parser("sample")
    s.add_argument("questions", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--state", help="a tinker:// state, or tinker_train.py sft's DIR/state.json (none: the base model)")
    s.add_argument("--base", default=MODEL)
    s.add_argument("--max-tokens", type=int, default=160)
    sc = sub.add_parser("score")
    sc.add_argument("evals", type=Path, nargs="+")
    sc.add_argument("--rc", type=Path, required=True)
    me = sub.add_parser("measure")
    me.add_argument("--rc", type=Path, required=True)
    me.add_argument("--thr", type=Path, default=THR)
    me.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gen")
    g.add_argument("--rc", type=Path, required=True)
    g.add_argument("--thr", type=Path, default=THR)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--sets", type=int, default=6)
    g.add_argument("--records", type=Path, default=RECORDS)
    g.add_argument("--split", choices=("train", "test"), default="train", help="test: the held-out contexts' member "
                   "questions, an evaluation set only (never training rows: rows --extra refuses them)")
    g.add_argument("--seed", type=int, default=0)
    cl = sub.add_parser("claims")
    cl.add_argument("evals", type=Path, nargs="+")
    cl.add_argument("--rc", type=Path, required=True)
    cl.add_argument("--thr", type=Path, default=THR)
    cl.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"gen": gen, "claims": claims, "measure": measure, "facts": facts, "check": check, "rows": rows, "sample": sample, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
