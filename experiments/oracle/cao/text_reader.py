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
import itertools
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reads_circuit import answer_class  # noqa: E402

HOME = Path.home()
RECORDS = HOME / "mpd-data/cluster/finder_8b_dev4"
LABELS = HOME / "mpd-data/graph_oracle/groups8b/labels_dev4.jsonl"
THR = HOME / "mpd-data/cluster/t2_8b/out/thr.pt"
MODEL = "Qwen/Qwen3-8B"
STEER = 3.0
CHUNK = 48  # claims per check_many call (its log-probabilities are claims x 3 rows x positions x vocabulary: a 24 GB card
# ran out at a context's ~200 claims)
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


def question_files(a) -> list[Path]:
    """The question files whose contexts a step covers: --rows when given (g2/gen rows of other circuits), else rc's."""
    return list(a.rows) if getattr(a, "rows", None) else [a.rc / "train_correct.jsonl", a.rc / "test_correct.jsonl"]


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
    for f in question_files(a):
        for x in f.read_text().splitlines():
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
    for x in (a.rows[0] if a.rows else a.rc / f"{a.split}_correct.jsonl").read_text().splitlines():
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
            got = [r for k in range(0, len(probe), CHUNK) for r in cc.check_many(nr, probe[k:k + CHUNK])]
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
            final = [r for k in range(0, len(stated), CHUNK) for r in cc.check_many(nr, stated[k:k + CHUNK])]
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
    test = [json.loads(x) for x in (a.gold or a.rc / "test_correct.jsonl").read_text().splitlines()]
    for f in a.evals:
        rs = [json.loads(x) for x in f.read_text().splitlines()]
        assert [r["a"] for r in rs] == [e["a"] for e in test], f
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
    apply = {p.name for p in (a.records / "apply").glob("*.json")}
    ref = json.loads(a.measure.read_text()) if a.measure else {}
    out = {}
    for f in question_files(a):
        for x in f.read_text().splitlines():
            e = json.loads(x)
            if e["id"] in out:
                continue
            rec = json.loads((a.records / "out" / rec_file(e["id"])).read_text())
            out[e["id"]] = context_facts(W, mx, e, rec, labels, rec_file(e["id"]) in apply, ref.get(e["id"]))
            print(e["id"], len(out[e["id"]]["groups"]), flush=True)
    a.out.write_text(json.dumps(out))


# ---- the serialized input ----

def num(x, nd=2) -> str:
    return "?" if x is None else f"{x:+.{nd}f}"


def serialize(F: dict, how: str = "full", focus: set | None = None) -> str:
    """The circuit as text. how: "full" (every measured number), "names" (groups and members' names only: the
    names-only control), "none" (the text alone). focus: the question's units; every other group in one line (a
    prompt about a third as long, so that a run can cover many more circuits)."""
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
    T = len(toks)
    for gname, g in F["groups"].items():
        lab = f" ({g['label']['short']})" if g.get("label") else ""
        n = len(g["members"])
        if focus is not None and gname not in focus and not any(m["name"] in focus for m in g["members"]):
            # a group the question does not name: one line (where its members are; with the numbers, their summed
            # shares and the group's write on the lead)
            pos = ", ".join(sorted({f"{q_tok(m['tok'])} @{m['pos'] if m['pos'] >= 0 else T + m['pos']}" for m in g["members"]}))
            x = f"  {gname}{lab}, {n} neuron{'' if n == 1 else 's'} at {pos}"
            if how == "full":
                x += (f": noising {sum(m['lost'] or 0 for m in g['members']):+.2f}, denoising "
                      f"{sum(m['restored'] or 0 for m in g['members']):+.2f}")
                if g.get("write_direct") is not None:
                    x += f", write on the lead {g['write_direct']:+.1f}{unit}"
            lines.append(x)
            continue
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


def serialize_circuit(c: dict, how: str = "full", focus: set | None = None, cap: int = 40) -> str:
    """oracledata's circuit record (questions/pairs circuits.jsonl: word-level contrast, units with value, value over its
    maximum, contrast-text value, noising/denoising shares and the direct write onto the word-level lead at the answer
    position; heads; shared groups) as the reader's text. how: "full" or "names" (no measured numbers). focus: the
    question's units (claim_check names) in full; every other unit and head in one short line (its two shares, or its
    name alone), at most cap other units in the record's order."""
    full = how == "full"
    if focus is not None:
        return _focused_circuit(c, full, focus, cap)
    tk = c["tokens"]
    lines = [f"Text: {c['user']!r}", f"The model's next token: {q_tok(c['answer']['text'])}" +
             (f"; its lead for that word over the alternative words: {c['contrast']['clean']:.1f} bits." if full else ".")]
    cor = c.get("corrupt") or {}
    if cor:
        lines.append(f"Contrast pair: with {cor.get('user')!r} the model gives {q_tok((cor.get('answer') or {}).get('text', '?'))}" +
                     (f" (the gap between the two texts: {c['gap']:.1f} bits). Shares below: of that gap, lost when the unit takes its "
                      f"value from the contrast text (noising) and restored when it alone keeps its value in the contrast text "
                      f"(denoising)." if full else "."))
    if c.get("heads"):
        lines.append("Heads at the answer position:")
        for h in c["heads"]:
            looks = ", ".join(f"{q_tok(t)} @{q}" + (f" {w:.2f}" if full else "") for q, w, t in h.get("attends", [])[:3])
            lines.append(f"  {h['name']} looks at {looks}" + (f"; noising {num(h.get('lost'))}, denoising {num(h.get('restored'))}" if full else ""))
    gid = {u: g["id"] for g in c.get("groups", []) for u in g["members"]}
    for stage in ("source", "readout", None):
        us = [u for u in c["units"] if u.get("stage") == stage or (stage is None and u.get("stage") not in ("source", "readout"))]
        if not us:
            continue
        lines.append(f"{(stage or 'other').capitalize()} neurons:")
        for u in us:
            x = f"  {u['name']} at {q_tok(u.get('token', tk[u['pos']] if u['pos'] < len(tk) else '?'))}"
            if u["name"] in gid:
                x += f" [{gid[u['name']]}]"
            if full:
                x += (f": value {'?' if u.get('value') is None else format(u['value'], '.3g')} ({abs(u.get('value_over_max') or 0):.2f} of its max), contrast text "
                      f"{u['corrupt']:.3g}; noising {num(u.get('lost'))}, denoising {num(u.get('restored'))}")
                if u.get("direct") is not None:
                    x += f"; its write on the lead {u['direct']:+.2f} bits"
            lines.append(x)
    return "\n".join(lines) + "\n"


def _focused_circuit(c: dict, full: bool, focus: set, cap: int) -> str:
    tk = c["tokens"]
    tok = lambda u: q_tok(u.get("token", tk[u["pos"]] if u["pos"] < len(tk) else "?"))  # noqa: E731
    lines = [f"The model's next token: {q_tok(c['answer']['text'])}" +
             (f"; its lead for that word over the alternative words: {c['contrast']['clean']:.1f} bits; the gap to the "
              f"contrast text {(c.get('corrupt') or {}).get('user')!r}: {c['gap']:.1f} bits." if full else ".")]
    gid = {u: g["id"] for g in c.get("groups", []) for u in g["members"]}
    hs, others = [h for h in c.get("heads", []) if h["name"] in focus], [h for h in c.get("heads", []) if h["name"] not in focus]
    for h in hs:
        looks = ", ".join(f"{q_tok(t)} @{q}" + (f" {w:.2f}" if full else "") for q, w, t in h.get("attends", [])[:3])
        lines.append(f"Asked head {h['name']} looks at {looks}" + (f"; noising {num(h.get('lost'))}, denoising {num(h.get('restored'))}" if full else ""))
    for u in [u for u in c["units"] if u["name"] in focus]:
        x = f"Asked neuron {u['name']} ({u.get('stage') or 'unit'}) at {tok(u)}" + (f" [{gid[u['name']]}]" if u["name"] in gid else "")
        if full:
            x += (f": value {'?' if u.get('value') is None else format(u['value'], '.3g')} ({abs(u.get('value_over_max') or 0):.2f} of its max), contrast text {'?' if u.get('corrupt') is None else format(u['corrupt'], '.3g')}; "
                  f"noising {num(u.get('lost'))}, denoising {num(u.get('restored'))}")
            if u.get("direct") is not None:
                x += f"; its write on the lead {u['direct']:+.2f} bits"
        lines.append(x)
    if not any(n in focus for n in [u["name"] for u in c["units"]] + [h["name"] for h in c.get("heads", [])]):
        lines.append("The asked unit is not in this text's circuit.")
    if others:
        lines.append(f"Other heads at the answer position ({len(others)})" + (" with noising/denoising" if full else "") + ": " + "; ".join(
            f"{h['name']}" + (f" {num(h.get('lost'))}/{num(h.get('restored'))}" if full else "") for h in others[:cap // 3]) +
            ("; ..." if len(others) > cap // 3 else ""))
    rest = [u for u in c["units"] if u["name"] not in focus]
    if rest:
        lines.append(f"Other neurons ({len(rest)})" + (" with noising/denoising" if full else "") + ": " + "; ".join(
            f"{u['name']} {tok(u)}" + (f" {num(u.get('lost'))}/{num(u.get('restored'))}" if full else "") for u in rest[:cap]) +
            ("; ..." if len(rest) > cap else ""))
    return "\n".join(lines) + "\n"


def pair_text(p: dict, C: dict, how: str, cap: int = 40) -> str:
    """A paired-context question with both texts' circuits (how: full, names) or none (the question alone: it already
    holds both texts and their answers)."""
    if how == "none":
        return p["q"]
    flat = lambda us: {u for x in us for u in (x if isinstance(x, list) else [x])}  # noqa: E731  (joint: two units' lists)
    a = serialize_circuit(C[p["circuit_a"]], how, flat(p["units_a"]), cap)
    b = serialize_circuit(C[p["circuit_b"]], how, flat(p["units_b"]), cap)
    return f"{p['q']}\n\nCircuit of text A:\n{a}\nCircuit of text B:\n{b}\n{p['q']}"


def pairs(a) -> None:
    """SFT rows and test questions for paired-context questions (oracledata's pairs: which text a unit matters more in),
    per input arm (full, names, none); training pairs touching a sealed context, an excluded task/key or any frozen
    scorecard item's context are dropped."""
    C = {json.loads(x)["circuit_id"]: json.loads(x) for x in a.circuits.read_text().splitlines()}
    frozen = set()
    for f in a.frozen or []:
        for x in f.read_text().splitlines():
            r = json.loads(x)
            frozen |= {r[k] for k in ("context", "context_a", "context_b") if r.get(k)}
    fkeys = {c.rsplit("/", 1)[0] for c in frozen}
    train = [json.loads(x) for x in a.train.read_text().splitlines()]
    ok = lambda c: not sealed(c) and c not in frozen and c.rsplit("/", 1)[0] not in fkeys  # noqa: E731
    kept = [p for p in train if ok(p["context_a"]) and ok(p["context_b"]) and (not a.kinds or p["kind"] in a.kinds.split(","))]
    print(json.dumps({"train pairs": len(train), "kept": len(kept), "dropped (sealed, excluded or frozen)": len(train) - len(kept)}))
    test = [json.loads(x) for x in a.test.read_text().splitlines()]
    test = [p for p in test if not a.test_kinds or p["kind"] in a.test_kinds.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    for how in a.arms.split(","):
        with (a.out / f"train_{how}.jsonl").open("w") as f:
            for p in kept:
                f.write(json.dumps({"prompt": [{"role": "user", "content": pair_text(p, C, how, a.cap)}],
                                    "completion": [{"role": "assistant", "content": p["a"]}],
                                    "info": {"task": p["id"], "cat": p["kind"]}}) + "\n")
        with (a.out / f"test_{how}.jsonl").open("w") as f:
            for p in test:
                f.write(json.dumps({"task": p.get("item", p["id"]), "question": pair_text(p, C, how, a.cap), "cat": p["kind"],
                                    "a": p["a"], "id": p.get("item", p["id"])}) + "\n")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(next((HOME / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots").iterdir())))
    for f in sorted(a.out.glob("*.jsonl")):
        rs = [json.loads(x) for x in f.read_text().splitlines()]
        rs = rs[:: max(1, len(rs) // 400)]  # (a sample: the token count is for costing)
        n = [len(tok((r["prompt"][0]["content"] + r["completion"][0]["content"]) if "prompt" in r else r["question"])["input_ids"]) for r in rs]
        print(f.name, "tokens per item (sampled): mean", sum(n) // len(n), "max", max(n))


def pair_class(text: str) -> str | None:
    """A paired-context answer's class: "A"/"B" (pair_zero, or "In A only"/"In B only"), "same", "both", "neither"."""
    t = strip_read(text or "").strip().lower()
    t = t[3:] if t.startswith("in ") else t
    for c, pre in (("same", ("about the same", "the same", "same")), ("both", ("both",)), ("neither", ("neither",))):
        if t.startswith(pre):
            return c
    m = re.match(r"(?:text\s+)?([ab])\b", t)
    return m.group(1).upper() if m else None


def pscore(a) -> None:
    """Paired-context answers per kind, with 95% intervals (bootstrap over questions); --answers: also the scorecard's
    answers/<name>/o1p.jsonl ({"item", "class"}) per eval file, name from --names."""
    import random

    gold = {json.loads(x)["item"]: json.loads(x) for x in a.items.read_text().splitlines()}
    gcls = lambda r: r["cls"].split("/")[1]  # noqa: E731
    for i, f in enumerate(a.evals):
        rs = [json.loads(x) for x in f.read_text().splitlines()]
        assert len(rs) == len(gold) and all(r["id"] in gold for r in rs), f
        ok = [(gold[r["id"]]["kind"], pair_class(r["got"]) == gcls(gold[r["id"]])) for r in rs]
        cells = []
        for k in ("pair_zero", "pair_sign", "pair_flip", None):
            v = [o for kk, o in ok if k is None or kk == k]
            rng = random.Random(0)
            bs = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(1000))
            cells.append(f"{k or 'all'} {sum(v) / len(v):.3f} [{bs[25]:.3f}, {bs[974]:.3f}] n={len(v)}")
        unparsed = sum(pair_class(r["got"]) is None for r in rs)
        print(f"{f.parent.name}/{f.stem}: " + " | ".join(cells) + f" | unparsed {unparsed}")
        if a.answers and a.names:
            d = a.answers / a.names[i]
            d.mkdir(parents=True, exist_ok=True)
            (d / "o1p.jsonl").write_text("".join(json.dumps({"item": r["id"], "class": pair_class(r["got"]), "answer": r["got"]}) + "\n" for r in rs))


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
    d = sum(m["direct"] or 0.0 for m in at)
    ds = sum(m["direct_steer"] or 0.0 for m in at)
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


def shuffled(F: dict, seed: str) -> dict:
    """F with every measured number of a member taken from another member of the same circuit (a fixed random
    permutation; heads' likewise): names, positions and the numbers' distribution kept, which number belongs to which
    member destroyed. A stronger control than mismatch, whose donor's numbers come in the donor's measured order."""
    import random

    G = json.loads(json.dumps(F))
    rng = random.Random(seed)
    ms = [m for g in G["groups"].values() for m in g["members"]]
    keys = ("frac", "value", "corrupt", "lost", "restored", "direct", "direct_steer", "size", "steer_size")
    src = [{k: m.get(k) for k in keys} for m in ms]
    rng.shuffle(src)
    for m, x in zip(ms, src):
        m.update(x)
    gs = list(G["groups"].values())
    src = [{k: g.get(k) for k in ("write_direct", "promotes", "suppresses")} for g in gs]
    rng.shuffle(src)
    for g, x in zip(gs, src):
        g.update(x)
    hs = [{k: h[k] for k in ("lost", "restored")} for h in G["heads"]]
    rng.shuffle(hs)
    for h, x in zip(G["heads"], hs):
        h.update(x)
    return G


NEURON = re.compile(r"L(\d+)\.N(\d+)(?=[+-])")


def anon_map(F: dict) -> dict:
    """A context's neurons renamed without their index ("L34.N864" -> "L34.u3"), numbered in the circuit's order:
    the layer and position stay, the identity a reader could memorize across texts goes."""
    out = {}
    for g in F["groups"].values():
        for m in g["members"]:
            k = f"L{m['layer']}.N{m['neuron']}"
            out.setdefault(k, f"L{m['layer']}.u{len(out) + 1}")
    return out


def anonymize(text: str, amap: dict) -> str:
    return NEURON.sub(lambda m: amap.get(m.group(0), m.group(0)), text)


def user_text(F: dict, e: dict, how: str, D: dict | None = None, first: bool = False, focus: bool = False) -> str:
    """The circuit, then the question (one context's questions share their prefix, cached when sampled); first: the
    question also before the circuit, so that the circuit's tokens are read knowing what is asked. how: full, names,
    none, mismatch (D's numbers), shuffle (the context's own numbers permuted among its members)."""
    fo = set(e["claim"]["units"]) if focus else None
    body = (serialize(mismatch(F, D), "full", fo) if how == "mismatch" else
            serialize(shuffled(F, F["id"]), "full", fo) if how == "shuffle" else serialize(F, how, fo))
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


EXCLUDED = HOME / "mpd-data/oracle/data/excluded_task_keys.json"  # O1 dev and testset_8b task/keys (oracledata)


PAIRS_MANIFEST = HOME / "mpd-data/oracle/data/pairs/v1/manifest.json"  # the paired set's test and dev families


def sealed(cid: str) -> bool:
    """Never trained on (records beyond rc's pool: train8b and later): the scorecard's sealed contexts
    (crc32("scorecard:" + id) % 20 == 0), the excluded task/keys and the paired set's test and dev families."""
    import zlib

    if zlib.crc32(("scorecard:" + cid).encode()) % 20 == 0:
        return True
    if not hasattr(sealed, "keys"):
        sealed.keys = set(json.loads(EXCLUDED.read_text())) if EXCLUDED.exists() else set()
        fam = json.loads(PAIRS_MANIFEST.read_text()).get("families", {}) if PAIRS_MANIFEST.exists() else {}
        sealed.families = {f for f, s in fam.items() if s in ("test", "dev")}
    return cid.rsplit("/", 1)[0] in sealed.keys or cid.split("/", 1)[0] in sealed.families


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
    n0 = len(train)
    pool = {e["id"] for e in load_rc(a.rc)[0]}  # rc's training contexts: the scorecard's sanctioned pool; the rule binds the rest
    paired = set()  # every frozen paired-test context, never trained on even from rc's pool
    o1p = HOME / "mpd-data/oracle/scorecard/items/o1_pairs.jsonl"
    if o1p.exists():
        paired = {json.loads(x)[k] for x in o1p.read_text().splitlines() for k in ("context_a", "context_b")}
    train = [e for e in train if e["id"] not in paired and (e["id"] in pool or not sealed(e["id"]))]
    print(json.dumps({"training questions dropped as sealed or excluded": n0 - len(train)}))
    if a.balance:
        rng = random.Random(0)
        by = collections.defaultdict(list)
        for e in train:
            by[(e["cat"], answer_class(e))].append(e)
        design = {tuple(k.split("/")) for k in json.loads((a.rc / "claim.json").read_text())["classes"]}
        bal = []
        for c in sorted({e["cat"] for e in train}):
            ks = [k for k in by if k[0] == c]
            top = max(len(by[k]) for k in ks if k in design)
            for k in ks:  # the pre-registered design's classes of a kind repeated up to its largest; any other class
                # (flips; sets that raise the lead) kept as it is (balancing those taught v2 to answer them)
                bal += by[k] + ([rng.choice(by[k]) for _ in range(top - len(by[k]))] if k in design else [])
        train = bal
    if a.max_train and len(train) > a.max_train:  # a run's budget of rows, drawn evenly (sft samples its batches from these)
        train = random.Random(1).sample(train, a.max_train)
    a.out.mkdir(parents=True, exist_ok=True)
    am = {cid: anon_map(F[cid]) for cid in F} if a.anon else {}
    an = lambda t, cid: anonymize(t, am[cid]) if a.anon else t  # noqa: E731
    for how in ("full", "names"):
        with (a.out / f"train_{how}.jsonl").open("w") as f:
            for k, e in enumerate(train):
                ans = (recap(F[e["id"]], e, how) + "\n" if a.recap else "") + e["a"]
                f.write(json.dumps({"prompt": [{"role": "user", "content": an(user_text(F[e["id"]], e, how, first=a.first, focus=a.focus), e["id"])}],
                                    "completion": [{"role": "assistant", "content": an(ans, e["id"])}],
                                    "info": {"task": f"{e['id']}#{k}", "cat": e["cat"]}}) + "\n")
    sets = [("test", test)] + ([("test_members", [json.loads(x) for x in a.test_extra.read_text().splitlines()])] if a.test_extra else [])
    for (name, qs), how in itertools.product(sets, ("full", "names", "mismatch", "shuffle", "none")):
        with (a.out / f"{name}_{how}.jsonl").open("w") as f:
            for k, e in enumerate(qs):
                f.write(json.dumps({"task": f"{e['id']}#{k}", "question": an(user_text(F[e["id"]], e, how, F[donor[e["id"]]], a.first, a.focus), e["id"]),
                                    "cat": e["cat"], "a": e["a"], "id": e["id"], **({"anon": am[e["id"]]} if a.anon else {})}) + "\n")
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
    import llm as tt  # (the shared Tinker helpers: Spend, service_client, renderer_for, answer_of)

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
    for f in a.train_extra or []:
        train += [json.loads(x) for x in f.read_text().splitlines()]
    if a.gold:  # another held-out question set (gen --split test's member questions)
        test = [json.loads(x) for x in a.gold.read_text().splitlines()]
    members = lambda e: frozenset(re.sub(r"@-?\d+$", "", u) for g in e["claim"]["units"] for u in e["groups"].get(g, [g]))  # noqa: E731
    outs = collections.defaultdict(set)
    for e in train:
        outs[(e["cat"], members(e))].add(answer_class(e))
    # the varied subset: the asked units (position-free) were asked the same kind in training with another outcome
    varied = {i for i, e in enumerate(test) if outs.get((e["cat"], members(e)), set()) - {answer_class(e)}}
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


def ask(a) -> None:
    """The reader as a tool: a finder record (with apply's groups, and measure's reference entry for bits) and questions
    in, the trained reader's greedy answers out (a JSON line per question: the question, the answer without its Read
    line, and the claim it ends with). Items are named as record_items names them."""
    import tinker
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
    import llm as tt  # (the shared Tinker helpers: Spend, service_client, renderer_for, answer_of)

    rec = json.loads(a.record.read_text())
    app = json.loads(a.apply.read_text()) if a.apply else None
    ref = json.loads(a.measure.read_text()) if a.measure else None
    ref = ref.get(rec["id"], ref) if isinstance(ref, dict) and "rms" not in ref else ref
    F = record_facts(Weights(), torch.load(THR, map_location="cpu")["max"], rec, app, load_labels(), ref)
    service = tt.service_client()
    renderer, tokenizer = tt.renderer_for(a.base)
    state = a.state if a.state.startswith("tinker://") else json.loads(Path(a.state).read_text())["state"]
    sampler = service.create_training_client_from_state(state).save_weights_and_get_sampling_client()
    params = tinker.SamplingParams(max_tokens=400, temperature=0.0, stop=renderer.get_stop_sequences())
    names = set(F["groups"]) | {m["name"] for g in F["groups"].values() for m in g["members"]}
    asked = lambda q: [u for u in re.findall(r"G\d+|L\d+\.N\d+[+-]@-?\d+", q) if u in names]  # noqa: E731
    futs = [(q, sampler.sample(prompt=renderer.build_generation_prompt([{"role": "user", "content": user_text(
        F, {"q": q, "claim": {"units": asked(q)}}, "full", first=True, focus=a.focus)}]),
        num_samples=1, sampling_params=params)) for q in a.question]
    for q, fut in futs:
        got, _ = tt.answer_of(renderer, tokenizer, fut.result().sequences[0].tokens)
        ans = strip_read(got)
        claim = None
        if "Claim:" in ans:
            try:
                claim = json.loads(ans.split("Claim:", 1)[1].strip())
            except ValueError:
                pass
        print(json.dumps({"question": q, "answer": ans.split("\nClaim:")[0], "claim": claim, "items": F["groups"] and list(F["groups"])}))


# ---- composed kinds: the reader's zero answers combined into oq.py's rank, necessary and interact questions ----

def change_of(answer: str, lead: float) -> float:
    """A g2-form zero answer as a change in bits: a flip as the whole lead lost, "almost unchanged" as 0, "up/down about
    b bits" as +b/-b."""
    a = strip_read(answer).split("\n")[0]
    if "instead of" in a:
        return -abs(lead)
    m = re.search(r"\b(up|down) about ([0-9.]+) bit", a)
    return 0.0 if not m else float(m[2]) * (1 if m[1] == "up" else -1)


def composed(q: dict, ch: dict) -> str:
    """oq.py's answer key from the reader's per-item (and per-pair) changes."""
    if q["kind"] == "rank":
        return ">".join(sorted(q["units"], key=lambda u: ch[(u,)]))
    if q["kind"] == "necessary":
        return min(q["options"], key=lambda u: ch[(u,)])
    d = ch[tuple(q["units"])] - ch[(q["units"][0],)] - ch[(q["units"][1],)]  # interact: oq.py's GAP 0.5 / LOW 0.2
    return "above" if d >= 0.35 else "below" if d <= -0.35 else "about"


def compose(a) -> None:
    """oq.py's rank, necessary and interact questions answered by asking the reader only what it was trained on (zeroing
    one item, or a pair together) and combining the stated changes; scored by exact key against oq's a_key."""
    import tinker
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
    import llm

    F = json.loads(a.facts.read_text())
    allow = {json.loads(x)["id"] for x in a.contexts.read_text().splitlines()} if a.contexts else None
    qs = [json.loads(x) for x in a.questions.read_text().splitlines()]
    qs = [q for q in qs if q["kind"] in a.kinds.split(",") and q["context_id"] in F and (allow is None or q["context_id"] in allow)]
    asks = {}
    for q in qs:
        y = q_tok(q["answer"]["text"])
        items = q.get("options") or q["units"]
        for u in items:
            asks[(q["context_id"], (u,))] = f"What happens to the model's next token {y} if {u} is zeroed?"
        if q["kind"] == "interact":
            asks[(q["context_id"], tuple(q["units"]))] = (f"What happens to the model's next token {y} if {q['units'][0]} and "
                                                          f"{q['units'][1]} are zeroed together?")
    service = llm.service_client()
    renderer, tokenizer = llm.renderer_for(a.base)
    spend = llm.Spend(a.out.parent, a.base)
    sampler = service.create_training_client_from_state(a.state).save_weights_and_get_sampling_client()
    params = tinker.SamplingParams(max_tokens=400, temperature=0.0, stop=renderer.get_stop_sequences())
    futs = {}
    for (cid, us), text in asks.items():
        prompt = renderer.build_generation_prompt([{"role": "user", "content": user_text(F[cid], {"q": text, "claim": {"units": list(us)}},
                                                                                           "full", first=True, focus=True)}])
        futs[(cid, us)] = (prompt, sampler.sample(prompt=prompt, num_samples=1, sampling_params=params))
    got = {}
    for k, (prompt, fut) in futs.items():
        res = fut.result()
        hits = res.prompt_cache_hit_tokens or 0
        spend.add("sample", prefill=prompt.length - hits, cached=hits, sampled=len(res.sequences[0].tokens))
        got[k] = llm.answer_of(renderer, tokenizer, res.sequences[0].tokens)[0]
    by = collections.defaultdict(list)
    with a.out.open("w") as fh:
        for q in qs:
            ch = {us: change_of(got[(c, us)], F[q["context_id"]]["lead"]) for (c, us) in got if c == q["context_id"]}
            key, gold = composed(q, ch), q["a_key"]
            if q["kind"] == "interact":  # the measured class in signed terms (oq.py's interact_row; v1's key predates it)
                m = q["measured"]
                d = m["pair"] - sum(m["alone"])
                gold = "above" if d >= 0.5 else "below" if d <= -0.5 else "about" if abs(d) < 0.2 else None
                if gold is None:
                    continue
            ok = key == gold
            by[q["kind"]].append(ok)
            fh.write(json.dumps({"id": q["id"], "kind": q["kind"], "a_key": gold, "key": key, "correct": ok,
                                 "changes": {"+".join(k): v for k, v in ch.items() if set(k) <= set(q.get("options") or q["units"])}}) + "\n")
    print(json.dumps({k: {"n": len(v), "exact": round(sum(v) / len(v), 3)} for k, v in by.items()} | {"usd": round(spend.total, 3)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("--rc", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--measure", type=Path, help="measure's output: claim_check's answer, alternatives and bits")
    f.add_argument("--rows", type=Path, nargs="*", help="question files of other circuits (default: rc's train and test)")
    f.add_argument("--records", type=Path, default=RECORDS, help="their circuit_finder records (out/) and apply's (apply/)")
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
    r.add_argument("--test-extra", type=Path, help="more held-out questions (gen --split test's): test_members_<input>.jsonl")
    r.add_argument("--recap", action="store_true")
    r.add_argument("--first", action="store_true")
    r.add_argument("--max-train", type=int, help="at most this many training rows (after balancing), drawn evenly")
    r.add_argument("--focus", action="store_true", help="only the question's groups in full; every other group in one line")
    r.add_argument("--anon", action="store_true", help="neurons without their index (L34.N864 -> L34.u3, per context): the "
                   "reader cannot carry a neuron's identity across texts, so it must read the numbers and the structure")
    s = sub.add_parser("sample")
    s.add_argument("questions", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--state", help="a tinker:// state, or tinker_train.py sft's DIR/state.json (none: the base model)")
    s.add_argument("--base", default=MODEL)
    s.add_argument("--max-tokens", type=int, default=160)
    sc = sub.add_parser("score")
    sc.add_argument("evals", type=Path, nargs="+")
    sc.add_argument("--rc", type=Path, required=True)
    sc.add_argument("--gold", type=Path, help="the held-out questions the evals answer (default: rc's test)")
    sc.add_argument("--train-extra", type=Path, nargs="*", help="more training questions (the name-prior's and the varied "
                    "subset's training side)")
    me = sub.add_parser("measure")
    me.add_argument("--rc", type=Path, required=True)
    me.add_argument("--thr", type=Path, default=THR)
    me.add_argument("--out", type=Path, required=True)
    me.add_argument("--rows", type=Path, nargs="*", help="question files of other circuits (default: rc's train and test)")
    g = sub.add_parser("gen")
    g.add_argument("--rc", type=Path, required=True)
    g.add_argument("--thr", type=Path, default=THR)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--sets", type=int, default=6)
    g.add_argument("--records", type=Path, default=RECORDS)
    g.add_argument("--rows", type=Path, nargs="*", help="a question file of other circuits whose contexts to ask about "
                   "(g2.py gen --dense's rows; default: rc's split)")
    g.add_argument("--split", choices=("train", "test"), default="train", help="test: the held-out contexts' member "
                   "questions, an evaluation set only (never training rows: rows --extra refuses them)")
    g.add_argument("--seed", type=int, default=0)
    cl = sub.add_parser("claims")
    cl.add_argument("evals", type=Path, nargs="+")
    cl.add_argument("--rc", type=Path, required=True)
    cl.add_argument("--thr", type=Path, default=THR)
    cl.add_argument("--out", type=Path, required=True)
    cl.add_argument("--gold", type=Path, help="the held-out questions the evals answer (default: rc's test)")
    k = sub.add_parser("ask")
    k.add_argument("record", type=Path, help="a circuit_finder record")
    k.add_argument("--apply", type=Path, help="neuron_motifs apply's record of it (shared groups)")
    k.add_argument("--measure", type=Path, help="measure's output (or one entry): claim_check's reference, for bits")
    k.add_argument("--question", action="append", required=True)
    k.add_argument("--state", required=True, help="the trained reader (tinker://... or sft's state.json)")
    k.add_argument("--base", default=MODEL)
    k.add_argument("--focus", action=argparse.BooleanOptionalAction, default=True, help="the question's items in full, "
                   "other groups in one line (as the reader v5 on was trained; --no-focus for v1-v4 states)")
    co = sub.add_parser("compose")
    co.add_argument("questions", type=Path, help="oq.py rows (oracledata's questions/v1 test.jsonl)")
    co.add_argument("--facts", type=Path, required=True)
    co.add_argument("--state", required=True)
    co.add_argument("--out", type=Path, required=True)
    co.add_argument("--kinds", default="rank,necessary,interact")
    co.add_argument("--contexts", type=Path, help="question rows whose contexts alone may be asked about (the reader's unseen ones)")
    co.add_argument("--base", default=MODEL)
    pa = sub.add_parser("pairs")
    pa.add_argument("--train", type=Path, required=True)
    pa.add_argument("--test", type=Path, required=True, help="the frozen O1p items (scorecard items/o1_pairs.jsonl)")
    pa.add_argument("--circuits", type=Path, required=True)
    pa.add_argument("--frozen", type=Path, nargs="*", help="every frozen item set: no training pair may touch their contexts or task/keys")
    pa.add_argument("--arms", default="full,names,none")
    pa.add_argument("--cap", type=int, default=40, help="other units shown per circuit (heads: a third of it)")
    pa.add_argument("--kinds", help="training kinds only (e.g. pair_zero,pair_sign: a held-out question kind)")
    pa.add_argument("--test-kinds", help="test kinds only")
    pa.add_argument("--out", type=Path, required=True)
    ps = sub.add_parser("pscore")
    ps.add_argument("evals", type=Path, nargs="+")
    ps.add_argument("--items", type=Path, required=True, help="the frozen O1p items")
    ps.add_argument("--answers", type=Path, help="the scorecard's answers directory")
    ps.add_argument("--names", nargs="*", help="one scorecard oracle name per eval file")
    a = ap.parse_args()
    {"pscore": pscore, "pairs": pairs, "compose": compose, "ask": ask, "gen": gen, "claims": claims, "measure": measure, "facts": facts, "check": check, "rows": rows, "sample": sample, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
