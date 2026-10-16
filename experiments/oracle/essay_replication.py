"""The scorecard's essay replication checks (#2951; the lead's 10-15 "ESSAY-STYLE NEURON GRAPHS", ~/mpd-data/team/
direction_1015.txt): Cammarata's published neuron results ("Neural Networks Are Already Clean", Goodfire 2026, page and
figure data saved at ~/mpd-data/scratch/inframove/essay) measured with plain interventions on the model, each against
the essay's own number with a pass rule fixed here before any run. Qwen3-32B for every case but the last.
  santa          neuron_essay_check's prompt: the 8 Santa negative poles zeroed at every position take p(the answer) from
                 near-certain to the essay's 6% (pass: at most 2 x 6%); the 10 Christmas/Santa-labelled poles of the figure
                 leave it high (the essay's 20: 76%; pass: at least 50%)
  preeclampsia   neuron_essay_check's prompt: the 4-neuron diagnosis hub zeroed changes the answer (the essay: it asks about
                 itching instead; pass: the top token changes)
  attention      the essay's three attention figures (the comma after "headache", "it" reading "music", "land" reading
                 "Never"), every check whose units the figure names: heads kept from looking at a token (that key masked
                 before the softmax for those heads at the reading position), source neurons zeroed at the source
                 token, a token swapped; the target neuron's value at the reading position before and after
                 (pass: a check that moves the neuron in the essay moves it the same way by at least half the essay's
                 relative change; a control, which leaves it in the essay, moves it by at most CONTROL_MAX of its value)
  russian        L1:20736's sign regimes in the essay's three contexts (the gate and up projections and the value at the
                 marked token): Colossians a closed-gate echo (gate < 0, up < 0, value > 0 and small), "Speak russian" and
                 www.un.org/russian the open-gate negative side (pass: every sign as the essay's and |value| within a
                 factor of 2)
  miyazaki       link strengths into and out of the Miyazaki neuron L4:8476: each listed upstream pole zeroed at every
                 position up to the name, the drop of L4:8476 at "aki"; L4:8476's positive pole zeroed, the drops of
                 L6:24077 at " Away" and L8:14338 at "ica" (pass: Spearman with the essay's 7 into-strengths at least 0.7
                 and the two out-strengths in the essay's order)
  social         Pythia-1B-v0: on every "Social Security" in the atlas corpus (176), the WINDOW tokens before " Social"
                 decoded as the context (natural text, as the essay's 164 cases), the detector neurons
                 L2:611, L2:4081, L2:6819 zeroed together change log p(" Social") by the essay's -0.004 nats on average
                 (pass: |mean| at most 0.05 nats); the 10 named next-token neurons with |effect| >= 0.2 zeroed one at a
                 time move it in the essay's direction (pass: 8 of 10 signs)
The "might" neuron that sets a head's query for "may" is not checked: the essay names neither the neuron nor the head.

  essay_replication.py run OUT.json [--cases santa,preeclampsia,attention,russian,miyazaki,social] [--model Qwen/Qwen3-32B]
      [--device auto] [--dtype bfloat16]
  essay_replication.py table OUT.json [...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONTROL_MAX = 0.15  # a control check passes when it moves the target neuron by at most this share of its value
ATTENTION = {
    "comma": {
        "prompt": "Human: A 32-year-old female at 30 weeks gestation presents with severe right upper quadrant pain, mild "
                  "headache, and nausea. BP is 162/98 mmHg, and labs show mildly elevated liver enzymes.\n\nIf we can only "
                  "ask about one other symptom, we should ask whether she's experiencing...\n\nAssistant: We should ask "
                  "whether she's experiencing **",
        "source": (25, " headache"), "dest": (26, ","), "target": (31, 5482),
        "checks": [
            {"what": "the three layer-30 heads can't look at \"headache\"", "block": {"heads": ["L30H8", "L30H11", "L30H14"], "key": " headache"},
             "before": 45.5, "after": 34.5, "control": False},
            {"what": "...or at \"pain\" instead", "block": {"heads": ["L30H8", "L30H11", "L30H14"], "key": " pain"},
             "before": 45.5, "after": 45.75, "control": True},
            {"what": "silence the headache neurons at \"headache\"", "zero": {"neurons": [(3, 5424), (10, 13522), (25, 3322)], "at": " headache"},
             "before": 45.0, "after": 30.25, "control": False}]},
    "music": {
        "prompt": "[user] List all the words in this sentence that are related to music: \n 'If it is cooler tomorrow I will go "
                  "for a run around the lake.'\n\n[assistant] There are no words in this sentence that are related to music. The "
                  "sentence is about a person's plans for exercising tomorrow based on the weather.",
        "source": (13, " music"), "dest": (18, " it"), "target": (31, 15784),
        "checks": [
            {"what": "\"it\" can't look at \"music\" (every head, layers 28-31)", "block": {"layers": (28, 31), "key": " music"},
             "before": 31.1, "after": 0.26, "control": False},
            {"what": "...or can't look at \"sentence\" instead", "block": {"layers": (28, 31), "key": " sentence"},
             "before": 31.1, "after": 33.75, "control": True},
            {"what": "...or can't look at \"words\" instead", "block": {"layers": (28, 31), "key": " words"},
             "before": 31.1, "after": 31.4, "control": True},
            {"what": "only L31H4 can't look at \"music\"", "block": {"heads": ["L31H4"], "key": " music"},
             "before": 31.1, "after": 12.0, "control": False},
            {"what": "...only L31H4 can't look at \"sentence\"", "block": {"heads": ["L31H4"], "key": " sentence"},
             "before": 31.1, "after": 32.0, "control": True},
            {"what": "the four named heads can't look at \"music\"", "block": {"heads": ["L31H4", "L28H12", "L28H13", "L30H45"], "key": " music"},
             "before": 31.1, "after": 7.1, "control": False}]},
    "neverland": {
        "prompt": "Peter Pan flew the children to Neverland",
        "source": (6, " Never"), "dest": (7, "land"), "target": (4, 15706),
        "checks": [
            {"what": "the eleven heads can't look at \"Never\"",
             "block": {"heads": ["L3H13", "L4H61", "L3H10", "L2H31", "L2H28", "L2H25", "L1H5", "L1H2", "L1H7", "L4H60", "L2H46"], "key": " Never"},
             "before": 47.0, "after": 18.75, "control": False},
            {"what": "...or can't look at \"Pan\" instead",
             "block": {"heads": ["L3H13", "L4H61", "L3H10", "L2H31", "L2H28", "L2H25", "L1H5", "L1H2", "L1H7", "L4H60", "L2H46"], "key": " Pan"},
             "before": 47.0, "after": 47.0, "control": True},
            {"what": "no head in layers 0-4 can look at \"Never\"", "block": {"layers": (0, 4), "key": " Never"},
             "before": 47.0, "after": -0.24, "control": False},
            {"what": "...or no head there can look at \"children\"", "block": {"layers": (0, 4), "key": " children"},
             "before": 47.0, "after": 47.0, "control": True},
            {"what": "eleven random heads in layers 1-4 can't look at \"Never\"", "block": {"random": (11, 1, 4), "key": " Never"},
             "before": 47.0, "after": 47.0, "control": True},
            {"what": "\"Dreamland\" in place of \"Neverland\"", "prompt": "Peter Pan flew the children to Dreamland",
             "before": 47.0, "after": -0.66, "control": False}]},
}
RUSSIAN = {"neuron": (1, 20736), "contexts": [
    {"text": "3. The book of Colossians is an epistle", "token": "ians", "gate": -1.2, "up": -3.5, "value": 1.0},
    {"text": "Speak russian in that case", "token": " russian", "gate": 4.4, "up": -2.9, "value": -12.5},
    {"text": "www.un.org/russian", "token": "russian", "gate": 1.0, "up": -1.0, "value": -20.6}]}  # (gate open: its sign only)
MIYAZAKI = {
    "target": (4, 8476),
    "into": [((3, 12478), -1, 6.7058), ((1, 15229), -1, 4.4238), ((0, 13886), -1, 3.3186), ((3, 15141), 1, 3.1543),
             ((2, 8891), -1, 2.089), ((0, 4289), 1, 2.0616), ((1, 20486), -1, 1.8809)],  # (neuron, its pole, the essay's strength)
    "into_text": "Instead, Miyazaki focuses on Jiro's ignorance as Japan marches toward war.", "into_token": "aki",
    "out": [((6, 24077), 23.4653, "Hayao Miyazaki's Studio Ghibli film \"Spirited Away\"", " Away"),
            ((8, 14338), 1.0258, "Hayao Miyazaki directed his first feature film, Nausicaä of the Valley of the Wind", "ica")]}
SOCIAL = {
    "model": "EleutherAI/pythia-1b-v0", "target": " Social",
    "detectors": [(2, 611), (2, 4081), (2, 6819)], "detector_effect": -0.004,
    "next": [((0, 3308), 0.49), ((2, 7715), -0.24), ((3, 839), 0.58), ((5, 447), 0.22), ((10, 3382), -0.24), ((12, 7260), 0.40),
             ((13, 3673), -0.61), ((15, 3731), -0.81), ((15, 7040), -0.75), ((15, 5759), 0.69)],
    "corpus": Path.home() / "mpd-data" / "atlas" / "corpus_qwen3", "window": 64}  # the atlas corpus (Qwen3 tokens)


def unit(s: str) -> tuple:
    """"L30H8" -> (30, 8)."""
    layer, head = s[1:].split("H")
    return int(layer), int(head)


def find(tok, ids: list[int], piece: str, before: int | None = None) -> int:
    """The last position (before `before`) whose token decodes to piece."""
    hits = [i for i, z in enumerate(ids[:before]) if tok.decode([z]) == piece]
    if not hits:
        raise ValueError(f"no token {piece!r} in the prompt")
    return hits[-1]


class Qwen:
    """An HF Qwen3 (eager attention) with per-head attention blocks and neuron edits; runs one sequence at a time."""

    def __init__(self, name: str, device: str, dtype: torch.dtype, model=None, tok=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3 import modeling_qwen3

        self.tok = tok if tok is not None or model is not None else AutoTokenizer.from_pretrained(name)
        self.model = model if model is not None else AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, device_map=device, attn_implementation="eager").eval()
        self.model.requires_grad_(False)
        self.layers = list(self.model.model.layers)
        cfg = self.model.config
        self.L, self.H = cfg.num_hidden_layers, cfg.num_attention_heads
        self.blocks = {}  # layer -> [(head, query position, key position)]
        orig = modeling_qwen3.eager_attention_forward

        def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            bl = self.blocks.get(module.layer_idx)
            if bl:
                B, H, T, S = query.shape[0], query.shape[1], query.shape[2], key.shape[-2]
                m = torch.zeros(B, H, T, S, dtype=query.dtype, device=query.device)
                if attention_mask is not None:
                    m = m + attention_mask[:, :, :, :S]
                for h, q, k in bl:
                    m[:, h, q, k] = torch.finfo(query.dtype).min
                attention_mask = m
            return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
        modeling_qwen3.eager_attention_forward = eager

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def run(self, ids: list[int], blocks=None, zero=(), capture=(), attentions: bool = False) -> dict:
        """Log p at the last position and the captured values: zero [(layer, neuron, positions or None, pole or 0)];
        capture [(layer, neuron, position)] -> {"value", "gate", "up"} of that neuron there."""
        self.blocks = blocks or {}
        cap, hs = {}, []
        want = {}
        for L, j, p in capture:
            want.setdefault(L, []).append((j, p))
        for L, layer in enumerate(self.layers):
            zs = [(j, ps, s) for l2, j, ps, s in zero if l2 == L]

            def pre(m, a, L=L, zs=zs):
                x = a[0]
                if zs:
                    x = x.clone()
                    for j, ps, s in zs:  # the whole neuron (s 0) or its pole s zeroed at ps (None: every position)
                        pos = slice(None) if ps is None else ps
                        v = x[0, pos, j]
                        x[0, pos, j] = torch.zeros_like(v) if s == 0 else torch.where(v * s > 0, torch.zeros_like(v), v)
                for j, p in want.get(L, []):
                    cap[(L, j, p, "value")] = float(x[0, p, j])
                return (x,) + tuple(a[1:])
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
            if L in want:
                def grab(kind, L=L):
                    def f(m, a, o):  # (a forward hook returning None leaves the output alone)
                        for j, p in want[L]:
                            cap[(L, j, p, kind)] = float(o[0, p, j])
                    return f
                hs.append(layer.mlp.gate_proj.register_forward_hook(grab("gate")))
                hs.append(layer.mlp.up_proj.register_forward_hook(grab("up")))
        try:
            out = self.model(input_ids=torch.tensor([ids], device=self.model.device), output_attentions=attentions)
            lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        finally:
            for h in hs:
                h.remove()
            self.blocks = {}
        return {"lp": lp, "cap": cap, "attentions": out.attentions if attentions else None}


def judge_change(before: float, after: float, essay_before: float, essay_after: float, control: bool) -> bool:
    """The attention checks' pass rule (module docstring)."""
    if control:
        return abs(after - before) <= CONTROL_MAX * abs(before)
    rel_e = (essay_after - essay_before) / abs(essay_before)
    rel = (after - before) / abs(before) if before else 0.0
    return math.copysign(1, rel) == math.copysign(1, rel_e) and abs(rel) >= 0.5 * abs(rel_e)


def case_attention(q: Qwen, log=print) -> list[dict]:
    out = []
    for name, c in ATTENTION.items():
        ids = q.ids(c["prompt"])
        dest = c["dest"][0] if q.tok.decode([ids[c["dest"][0]]]) == c["dest"][1] else find(q.tok, ids, c["dest"][1])
        src = c["source"][0] if q.tok.decode([ids[c["source"][0]]]) == c["source"][1] else find(q.tok, ids, c["source"][1], dest)
        L, j = c["target"]
        base = q.run(ids, capture=[(L, j, dest)])["cap"][(L, j, dest, "value")]
        rng = random.Random(0)
        for ch in c["checks"]:
            if "block" in ch:
                b = ch["block"]
                k = find(q.tok, ids, b["key"], dest)
                if "heads" in b:
                    heads = [unit(h) for h in b["heads"]]
                elif "layers" in b:
                    heads = [(lay, h) for lay in range(b["layers"][0], b["layers"][1] + 1) for h in range(q.H)]
                else:
                    n, lo, hi = b["random"]
                    heads = rng.sample([(lay, h) for lay in range(lo, hi + 1) for h in range(q.H)], n)
                blocks = {}
                for lay, h in heads:
                    blocks.setdefault(lay, []).append((h, dest, k))
                after = q.run(ids, blocks=blocks, capture=[(L, j, dest)])["cap"][(L, j, dest, "value")]
            elif "zero" in ch:
                at = find(q.tok, ids, ch["zero"]["at"], dest + 1)
                after = q.run(ids, zero=[(lay, n, [at], 0) for lay, n in ch["zero"]["neurons"]], capture=[(L, j, dest)])["cap"][(L, j, dest, "value")]
            else:
                ids2 = q.ids(ch["prompt"])
                after = q.run(ids2, capture=[(L, j, len(ids2) - 1)])["cap"][(L, j, len(ids2) - 1, "value")]
            ok = judge_change(base, after, ch["before"], ch["after"], ch["control"])
            out.append({"case": f"attention/{name}", "check": ch["what"], "control": ch["control"], "essay": [ch["before"], ch["after"]],
                        "ours": [base, after], "pass": ok, "source": src, "dest": dest})
            log(f"attention/{name}: {ch['what']}: essay {ch['before']} -> {ch['after']}, ours {base:.2f} -> {after:.2f}: {'pass' if ok else 'FAIL'}")
    return out


def case_russian(q: Qwen, log=print) -> list[dict]:
    L, j = RUSSIAN["neuron"]
    out = []
    for c in RUSSIAN["contexts"]:
        ids = q.ids(c["text"])
        p = find(q.tok, ids, c["token"])
        cap = q.run(ids, capture=[(L, j, p)])["cap"]
        g, u, v = cap[(L, j, p, "gate")], cap[(L, j, p, "up")], cap[(L, j, p, "value")]
        ok = (g > 0) == (c["gate"] > 0) and (u > 0) == (c["up"] > 0) and (v > 0) == (c["value"] > 0) and 0.5 <= abs(v) / abs(c["value"]) <= 2
        out.append({"case": "russian", "check": f"{c['text']!r} at {c['token']!r}", "essay": {"gate": c["gate"], "up": c["up"], "value": c["value"]},
                    "ours": {"gate": g, "up": u, "value": v}, "pass": ok})
        log(f"russian {c['text']!r}: essay value {c['value']}, ours gate {g:.2f} up {u:.2f} value {v:.2f}: {'pass' if ok else 'FAIL'}")
    return out


def case_miyazaki(q: Qwen, log=print) -> list[dict]:
    from scipy.stats import spearmanr

    L, j = MIYAZAKI["target"]
    ids = q.ids(MIYAZAKI["into_text"])
    p = find(q.tok, ids, MIYAZAKI["into_token"])
    base = q.run(ids, capture=[(L, j, p)])["cap"][(L, j, p, "value")]
    ours = []
    for (lu, ju), s, _ in MIYAZAKI["into"]:
        v = q.run(ids, zero=[(lu, ju, list(range(p + 1)), s)], capture=[(L, j, p)])["cap"][(L, j, p, "value")]
        ours.append(base - v)
    rho = float(spearmanr(ours, [e for _, _, e in MIYAZAKI["into"]]).correlation)
    outs = []
    for (lo, jo), e, text, piece in MIYAZAKI["out"]:
        ids2 = q.ids(text)
        p2 = find(q.tok, ids2, piece)
        b2 = q.run(ids2, capture=[(lo, jo, p2)])["cap"][(lo, jo, p2, "value")]
        a2 = q.run(ids2, zero=[(L, j, None, 1)], capture=[(lo, jo, p2)])["cap"][(lo, jo, p2, "value")]
        outs.append({"neuron": f"L{lo}:{jo}", "essay": e, "before": b2, "drop": b2 - a2})
    order = outs[0]["drop"] > outs[1]["drop"] > 0
    rec = {"case": "miyazaki", "check": "link strengths into and out of L4:8476", "target_value": base,
           "into": [{"neuron": f"L{lu}:{ju}", "pole": s, "essay": e, "drop": d} for ((lu, ju), s, e), d in zip(MIYAZAKI["into"], ours)],
           "spearman_into": rho, "out": outs, "pass": rho >= 0.7 and order}
    log(f"miyazaki: L4:8476 at 'aki' {base:.2f}; into Spearman {rho:.2f}; out drops {[round(o['drop'], 2) for o in outs]}: "
        f"{'pass' if rec['pass'] else 'FAIL'}")
    return [rec]


def case_santa_preeclampsia(q: Qwen, log=print) -> list[dict]:
    import neuron_essay_check as nec

    out = []
    for name in ("santa", "preeclampsia"):
        c = nec.CASES[name]
        ids = q.ids(c["text"])
        r0 = q.run(ids)
        y = int(r0["lp"].argmax())
        p0 = float(r0["lp"][y].exp())
        for set_name, units in c["sets"].items():
            us = [nec.parse(u) for u in units]
            r1 = q.run(ids, zero=[(L, j, None, s) for L, j, s in us])
            p1, top1 = float(r1["lp"][y].exp()), int(r1["lp"].argmax())
            if set_name == "santa_negative_8":
                ok = p1 <= 0.12
            elif set_name == "christmas_labelled_10":
                ok = p1 >= 0.5
            else:
                ok = top1 != y
            out.append({"case": name, "check": set_name, "answer": q.tok.decode([y]), "p_before": p0, "p_after": p1,
                        "top_after": q.tok.decode([top1]), "pass": ok})
            log(f"{name} {set_name}: p({q.tok.decode([y])!r}) {p0:.3f} -> {p1:.3f}, top after {q.tok.decode([top1])!r}: {'pass' if ok else 'FAIL'}")
    return out


def social_contexts() -> list[str]:
    """Every " Social Security" in the atlas corpus: the text of the window tokens before " Social" in its document."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    t = np.load(SOCIAL["corpus"] / "tokens.npy", mmap_mode="r")
    off = np.load(SOCIAL["corpus"] / "offsets.npy")
    a, b = tok(" Social Security", add_special_tokens=False)["input_ids"]
    out = []
    for i in np.nonzero((t[:-1] == a) & (t[1:] == b))[0].tolist():
        start = max(int(off[np.searchsorted(off, i, side="right") - 1]), i - SOCIAL["window"])
        if i > start:
            out.append(tok.decode(t[start:i].tolist()))
    return out


@torch.no_grad()
def case_social(device: str, log=print) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SOCIAL["model"])
    model = AutoModelForCausalLM.from_pretrained(SOCIAL["model"], dtype=torch.float32).to(device).eval()
    tgt = tok(SOCIAL["target"], add_special_tokens=False)["input_ids"][0]
    layers = model.gpt_neox.layers

    def logp(text, zero=()):
        hs = []
        for L, layer in enumerate(layers):
            js = [j for l2, j in zero if l2 == L]
            if js:
                hs.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(
                    lambda m, a, js=js: (a[0].index_fill(-1, torch.tensor(js, device=a[0].device), 0.0),)))
        try:
            ids = torch.tensor([tok(text)["input_ids"]], device=device)
            return float(torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), -1)[tgt])
        finally:
            for h in hs:
                h.remove()
    ctx = social_contexts()
    base = [logp(t) for t in ctx]
    det = float(np.mean([logp(t, SOCIAL["detectors"]) - b for t, b in zip(ctx, base)]))
    rows = []
    for (L, j), e in SOCIAL["next"]:
        d = float(np.mean([logp(t, [(L, j)]) - b for t, b in zip(ctx, base)]))
        rows.append({"neuron": f"L{L}:{j}", "essay": e, "ours_mean": d, "agree": (d > 0) == (e > 0)})
    agree = sum(r["agree"] for r in rows)
    ok = abs(det) <= 0.05 and agree >= 8
    log(f"social (Pythia-1B-v0, {len(base)} contexts, mean log p(' Social') {np.mean(base):.2f}): detectors {det:+.4f} nats "
        f"(essay -0.004); next-token neurons agree in sign {agree} of {len(rows)}: {'pass' if ok else 'FAIL'}")
    return [{"case": "social", "check": "detectors unused; next-token neurons' signs", "detector_mean": det, "neurons": rows,
             "agree": agree, "contexts": len(base), "pass": ok}]


def table(recs: list[dict]) -> str:
    lines = [f"{'case':26}{'check':58}{'essay':>18}{'ours':>18}  result"]
    for r in recs:
        if "essay" in r and isinstance(r["essay"], list):
            e, o = f"{r['essay'][0]:g} -> {r['essay'][1]:g}", f"{r['ours'][0]:.1f} -> {r['ours'][1]:.1f}"
        elif r["case"] in ("santa", "preeclampsia"):
            e, o = "see rule", f"p {r['p_before']:.2f} -> {r['p_after']:.2f}"
        elif r["case"] == "russian":
            e, o = f"{r['essay']['value']:g}", f"{r['ours']['value']:.1f}"
        elif r["case"] == "miyazaki":
            e, o = "rank", f"rho {r['spearman_into']:.2f}"
        else:
            e, o = "-0.004; signs", f"{r['detector_mean']:+.3f}; {r['agree']}/10"
        lines.append(f"{r['case'][:25]:26}{r['check'][:57]:58}{e:>18}{o:>18}  {'pass' if r['pass'] else 'FAIL'}")
    lines.append(f"{sum(r['pass'] for r in recs)} of {len(recs)} checks pass")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "table"))
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--cases", default="santa,preeclampsia,attention,russian,miyazaki,social")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    a = ap.parse_args()
    if a.cmd == "table":
        print(table([r for p in a.paths for r in json.loads(p.read_text())]))
        return
    cases = a.cases.split(",")
    out = json.loads(a.paths[0].read_text()) if a.paths[0].exists() else []
    done = {r["case"].split("/")[0] for r in out}
    log = lambda s: print(s, flush=True)  # noqa: E731
    if "social" in cases and "social" not in done:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        out += case_social(dev, log)
        a.paths[0].write_text(json.dumps(out))
    big = [c for c in cases if c != "social" and c not in done and not (c in ("santa", "preeclampsia") and ({"santa", "preeclampsia"} & done))]
    if big:
        q = Qwen(a.model, a.device, getattr(torch, a.dtype))
        runs = {"attention": case_attention, "russian": case_russian, "miyazaki": case_miyazaki}
        if "santa" in big or "preeclampsia" in big:
            out += case_santa_preeclampsia(q, log)
            a.paths[0].write_text(json.dumps(out))
        for c in big:
            if c in runs:
                out += runs[c](q, log)
                a.paths[0].write_text(json.dumps(out))
    print(table(out))


if __name__ == "__main__":
    main()
