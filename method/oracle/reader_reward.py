"""The English reward of the goal doc (#2951, the user's 10-14 goal; the lead's 10-14): how well a separate reader,
given only the oracle's explanation, predicts what the target model does under interventions, and how many of the
invariants the explanation implies hold when tested on the target.

Interventions (one JSON object each), run on the target (vpd4l: VPD's subcomponents are the parts):
  {"kind": "swap", "pos": p, "token": t}                      the token at p replaced by t
  {"kind": "remove", "parts": [[site, pos, index], ...]}      the parts' masks set to 0 (VPD's ablation)
  {"kind": "scale", "parts": [...], "by": s}                  the parts' masks set to s (their rank-one weight times s)
  {"kind": "setback", "pos": p, "token": t, "parts": [...]}   the swap, with the parts' activations set back to their
                                                              values on the original text
An outcome: the target's most likely next token ("top") and the probability of the original prediction ("p").

reader reward: questions(): interventions drawn independently of any explanation (swaps at the last positions with
  fresh tokens from the model's own prediction there, removals and scalings of the circuit's parts singly and in
  threes, swaps with one part set back), half of each kind from those that move the model, so an explanation earns
  only what generalizes and "no change" is not a free answer; the reader (a frozen open model
  on Tinker, dense so that it is deterministic, never trained, never Claude) reads the text's end, the model's prediction, the explanation (or none) and
  the questions, and answers each with a token and a probability. Per question: top right (0/1) and p's credit
  max(0, 1 - |logit p_reader - logit p| / 4); reward = the mean of their average over questions less SIMPLICITY per
  1,000 reader tokens of explanation. Report each kind's share right against no explanation and another text's.
invariant checker: the reader writes invariants the explanation implies in a fixed grammar (INV lines: a swap over a
  set of tokens, removals, scalings or set-backs of named parts, and a predicted outcome); each is run on the target
  over every listed instance and holds when every instance meets its outcome; reward = the share of invariant lines
  that hold (lines that do not parse or name a part the model lacks fail).

  reader_reward.py questions --task text9000 --circuit STATED.json --out Q.jsonl [--seed 0] [--device D]
  reader_reward.py read --questions Q.jsonl --explanation E.txt|none --out R.json [--reader MODEL]
  reader_reward.py invariants --task text9000 --explanation E.txt|none --out I.json [--reader MODEL] [--device D]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "vpd"), str(HERE / "rl")]

TEXTS = Path.home() / "mpd-data" / "graph_oracle" / "texts" / "vpd4l"
SIMPLICITY = 0.1  # reward per 1,000 reader tokens of explanation
READER = "Qwen/Qwen3.6-27B"  # dense, so deterministic at temperature 0 (10-14: three repeats agree to 0.003). The 397B
# reward reader (MoE) scored one prompt 0.625 and then 0.424, and Qwen3-8B answered "' of' 0.95" to every question.
LOGIT_SCALE = 4.0  # a probability's credit reaches 0 at this many nats of log-odds error
KIND = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "fc", "down_proj": "down", "neuron": "n"}
SITE = {v: k for k, v in KIND.items()}
PARTS = ("Parts are rank-one slices of the model's weight matrices, named L<layer>.<kind>#<index> (kind q, k, v, o: "
         "attention's query, key, value and output; fc, down: the MLP's input and output) at a token position (@<position>).")


def label(site: str, pos: int, index: int) -> str:
    return f"L{site.split('.')[1]}.{KIND[site.split('.')[-1]]}#{index}@{pos}"


def site_of(lab: str, where=None) -> tuple[str, int, int] | None:
    """(site, position, index) of 'L3.down#2481@394' or 'L3.down#2481@' creation'' (where: a position's token text to
    its position), or None."""
    m = re.fullmatch(r"L(\d+)\.(q|k|v|o|fc|down|n)#(\d+)@(.+)", lab.strip())
    if not m:
        return None
    layer, kind, idx, at = m.groups()
    pos = where(at) if where else (int(at) if at.isdigit() else None)
    if pos is None:
        return None
    return (f"h.{layer}.{'mlp' if kind in ('fc', 'down', 'n') else 'attn'}.{SITE[kind]}", int(pos), int(idx))


class VPDTarget:
    """vpd4l on one text: interventions run in torch (native's model and VPD's masks)."""

    def __init__(self, task: str, device: str | None = None):
        import torch

        import mech
        import native

        self.torch = torch
        self.nat = native.Native(device)
        t = json.loads((TEXTS / f"{task}.json").read_text())
        p = t["prompts"][0]
        self.task, self.ids, self.t = task, list(p["token_ids"]), int(p["target_positions"][0])
        self.tk = mech.tokenizer("vpd4l")
        self.base = self.logp(self.ids)
        self.y0 = int(self.base.argmax())
        self.a0 = self._acts(self.ids)

    def tok(self, i: int) -> str:
        return self.tk.decode([int(i)])

    def where(self, at: str) -> int | None:
        """A position written as its number or as its token quoted (the last position holding that token)."""
        at = at.strip()
        if at.isdigit():
            return int(at) if int(at) < len(self.ids) else None
        try:
            s = ast.literal_eval(at)
        except (ValueError, SyntaxError):
            return None
        hits = [p for p in range(len(self.ids)) if self.tok(self.ids[p]) == s] or \
            [p for p in range(len(self.ids)) if self.tok(self.ids[p]).strip() == str(s).strip()]  # the leading space left out
        return hits[-1] if hits else None

    def valid(self, part) -> bool:
        """Whether the model has this part: (site, position, index)."""
        n, p, c = part
        return n in self.nat.C and 0 <= c < self.nat.C[n] and 0 <= p < len(self.ids)

    def token_id(self, s: str) -> int | None:
        e = self.tk.encode(s).ids
        return e[0] if len(e) == 1 else None

    def _acts(self, seq):
        nat, torch = self.nat, self.torch
        nat.vpd.clear()
        for n in nat.names:
            nat.target.site(n).cache_input = True
        with torch.no_grad():
            nat.target(torch.tensor([seq], device=nat.dev))
        a = {n: (nat.target.site(n).last_input[0] @ nat.target.site(n).V) for n in nat.names}
        for n in nat.names:
            nat.target.site(n).cache_input = False
        nat.vpd.clear()
        return a

    def logp(self, seq, masks: dict | None = None, patch: dict | None = None):
        """log p at the target: the model, parts' masks set (masks: {(site, pos, idx): value}), or activations patched
        (patch: {site: [(pos, idx, value)]})."""
        import types

        nat, torch = self.nat, self.torch
        x = torch.tensor([seq], device=nat.dev)
        T = len(seq)
        with torch.no_grad():
            nat.vpd.clear()
            if masks:
                mk = {n: torch.ones((1, T, nat.target.site(n).V.shape[1]), device=nat.dev) for n in nat.names}
                for (n, p, c), v in masks.items():
                    mk[n][0, p, c] = v
                out = nat.vpd.masked(x, mk, {n: torch.ones((1, T), device=nat.dev) for n in nat.names})
            elif patch:
                saved = {}
                for n, items in patch.items():
                    st = nat.target.site(n)
                    saved[n] = st._forward

                    def fwd(self_, xx, items=items, st=st):
                        a = xx @ st.V
                        for p, c, v in items:
                            a[:, p, c] = v
                        return a @ st.U + xx @ st.delta_T

                    st._forward = types.MethodType(fwd, st)
                try:
                    out = nat.target(x)
                finally:
                    for n, f in saved.items():
                        nat.target.site(n)._forward = f
            else:
                out = nat.target(x)
        nat.vpd.clear()
        return torch.log_softmax(out[0, self.t].float(), -1)

    def run(self, iv: dict) -> dict:
        seq = list(self.ids)
        if iv["kind"] in ("swap", "setback"):
            seq[iv["pos"]] = self.token_id(iv["token"])
        parts = [tuple(x) for x in iv.get("parts", [])]
        if iv["kind"] == "remove":
            lp = self.logp(seq, masks={p: 0.0 for p in parts})
        elif iv["kind"] == "scale":
            lp = self.logp(seq, masks={p: float(iv["by"]) for p in parts})
        elif iv["kind"] == "setback":
            patch: dict = {}
            for n, p, c in parts:
                patch.setdefault(n, []).append((p, c, self.a0[n][p, c]))
            lp = self.logp(seq, patch=patch)
        else:
            lp = self.logp(seq)
        return {"top": self.tok(int(lp.argmax())), "p": float(lp[self.y0].exp())}

    def fresh(self, pos: int, n: int, rng: random.Random) -> list[str]:
        """n tokens the model itself predicts at pos (sampled by its probabilities, the text's own left out), single
        tokens that decode back to themselves."""
        lp = self.nat.reference([self.ids[:pos]], [pos - 1])[0, 0].cpu().double()
        w = lp.exp()
        w[self.ids[pos]] = 0.0
        gen = self.torch.Generator().manual_seed(rng.randrange(2 ** 31))
        out = []
        for z in self.torch.multinomial(w, 4 * n, replacement=False, generator=gen).tolist():
            s = self.tok(z)
            if self.token_id(s) == z and s.strip() and len(out) < n:
                out.append(s)
        return out

    def words(self, iv: dict) -> str:
        """The intervention in words for the reader."""
        def at(p):
            return f"position {p} ({self.tok(self.ids[p])!r})"
        names = ", ".join(label(*x) if len(x) == 3 and isinstance(x[0], str) else str(x) for x in
                          [(n, p, c) for n, p, c in iv.get("parts", [])])
        if iv.get("group"):  # whole groups, by the names the head defines
            names = ", ".join(iv["group"]) if isinstance(iv["group"], list) else iv["group"]
            what = f"group{'s' if ',' in names else ''} {names}"
            if iv["kind"] == "remove":
                return f"{what} removed"
            if iv["kind"] == "scale":
                return f"{what} scaled by {iv['by']:g} (every part's activation times {iv['by']:g})"
            if iv["kind"] == "setback":
                return f"the token at {at(iv['pos'])} replaced by {iv['token']!r}, with {what} set back to its values on the original text"
        if iv["kind"] == "swap":
            return f"the token at {at(iv['pos'])} replaced by {iv['token']!r}"
        if iv["kind"] == "remove":
            return f"part{'s' if len(iv['parts']) > 1 else ''} {names} removed{' together' if len(iv['parts']) > 1 else ''}"
        if iv["kind"] == "scale":
            return f"part{'s' if len(iv['parts']) > 1 else ''} {names} scaled by {iv['by']:g} (its weight times {iv['by']:g})"
        return (f"the token at {at(iv['pos'])} replaced by {iv['token']!r}, with part{'s' if len(iv['parts']) > 1 else ''} "
                f"{names} set back to its value on the original text")

    def head(self, n_tokens: int = 24) -> str:
        """The text's end, its last tokens by position, and the model's prediction now."""
        T = len(self.ids)
        top = self.base.exp().topk(3)
        pred = ", ".join(f"{self.tok(int(i))!r} ({float(v):.2f})" for v, i in zip(top.values, top.indices))
        toks = " ".join(f"{p}:{self.tok(self.ids[p])!r}" for p in range(max(0, T - n_tokens), T))
        end = self.tk.decode(self.ids[max(0, T - 60):])
        return (f"A small language model (4 transformer layers, trained on web text) reads a text and predicts the next token "
                f"after its last token (position {T - 1}).\nThe text ends: ...{end}\nIts last tokens by position: {toks}\n"
                f"The model's prediction now: {pred}.\n{PARTS}\n{self.group_lines()}")

    named: dict = {}  # group name -> its parts (a circuit's groups; empty: parts one by one)

    def group_lines(self) -> str:
        """The groups as the reader sees them: each one's size, layers and position."""
        if not self.named:
            return ""
        rows = []
        for g, ps in self.named.items():
            layers = sorted({int(n.split(".")[1]) for n, _, _ in ps})
            pos = sorted({p for _, p, _ in ps})
            rows.append(f"{g}: {len(ps)} parts at layer{'s' if len(layers) > 1 else ''} {layers[0]}"
                        f"{'-' + str(layers[-1]) if len(layers) > 1 else ''}, position{'s' if len(pos) > 1 else ''} "
                        + ", ".join(f"{q} ({self.tok(self.ids[q])!r})" for q in pos))
        return "Groups of parts (each removed, scaled or set back whole):\n" + "\n".join(rows) + "\n"


KEEP = 8  # positions whose neuron activations HFTarget keeps for set-backs (the parts are at the last 3)


class HFTarget(VPDTarget):
    """A Hugging Face causal LM (Qwen3) on one text; its parts are MLP neurons (the input of down_proj): removal sets
    one to 0, scaling multiplies it, a set-back restores its value on the original text."""

    def __init__(self, name: str, text: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.name = torch, name
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.hf = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32 if self.dev.type == "cpu" else torch.bfloat16).to(self.dev).eval()
        self.layers = self.m.model.layers
        self.I = self.m.config.intermediate_size
        self.groups = (16, 64)  # a single neuron of Qwen3-1.7B moves p(' of') on text9000 by under 0.01 (0.99 -> 0.99)
        self.ids = self.hf(text, add_special_tokens=False)["input_ids"]
        self.t = len(self.ids) - 1
        self.task = text[-60:]
        lp, self.a0 = self._run(self.ids, {}, keep=True)
        self.all_lp = lp
        self.base = lp[self.t]
        self.y0 = int(self.base.argmax())

    def tok(self, i: int) -> str:
        return self.hf.decode([int(i)])

    def token_id(self, s: str) -> int | None:
        e = self.hf(s, add_special_tokens=False)["input_ids"]
        return e[0] if len(e) == 1 else None

    def valid(self, part) -> bool:
        n, p, c = part
        return (n.endswith(".mlp.neuron") and 0 <= int(n.split(".")[1]) < len(self.layers) and 0 <= c < self.I
                and 0 <= p < len(self.ids))

    def _run(self, seq, edits: dict, keep: bool = False):
        """Log-probabilities at every position, the neurons edited: edits {(layer, pos, idx): ("set", v) | ("scale", s)};
        keep: also every layer's neuron activations (CPU)."""
        torch = self.torch
        acts, hs = {}, []
        by: dict = {}
        for (l, p, c), e in edits.items():
            by.setdefault(l, []).append((p, c, e))
        for l, layer in enumerate(self.layers):
            if l in by or keep:
                def pre(_m, args, l=l):
                    x = args[0]
                    if keep:  # the last KEEP positions only: set-backs and the parts sit there
                        acts[l] = x[0, -KEEP:].detach().to("cpu", torch.float32)
                    for p, c, (k, v) in by.get(l, []):
                        x[0, p, c] = v if k == "set" else x[0, p, c] * v
                    return (x,) + tuple(args[1:])
                hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        try:
            with torch.no_grad():
                lp = torch.log_softmax(self.m(input_ids=torch.tensor([seq], device=self.dev)).logits[0].float(), -1)
        finally:
            for h in hs:
                h.remove()
        return lp, acts

    def logp(self, seq, masks: dict | None = None, patch: dict | None = None):
        edits = {}
        for (n, p, c), v in (masks or {}).items():
            edits[(int(n.split(".")[1]), p, c)] = ("scale", v)
        for n, items in (patch or {}).items():
            for p, c, v in items:
                edits[(int(n.split(".")[1]), p, c)] = ("set", float(v))
        return self._run(seq, edits)[0][self.t]

    def run(self, iv: dict) -> dict:
        if iv["kind"] == "setback":
            seq = list(self.ids)
            seq[iv["pos"]] = self.token_id(iv["token"])
            patch: dict = {}
            for n, p, c in iv["parts"]:
                patch.setdefault(n, []).append((p, c, self.a0[int(n.split(".")[1])][p - (len(self.ids) - KEEP), c]))
            lp = self.logp(seq, patch=patch)
            return {"top": self.tok(int(lp.argmax())), "p": float(lp[self.y0].exp())}
        return VPDTarget.run(self, iv)

    @property
    def nat(self):  # VPDTarget.run reads masks through logp only
        raise AttributeError

    def fresh(self, pos: int, n: int, rng: random.Random) -> list[str]:
        w = self.all_lp[pos - 1].cpu().double().exp()
        w[self.ids[pos]] = 0.0
        gen = self.torch.Generator().manual_seed(rng.randrange(2 ** 31))
        out = []
        for z in self.torch.multinomial(w, 4 * n, replacement=False, generator=gen).tolist():
            s = self.tok(z)
            if self.token_id(s) == z and s.strip() and len(out) < n:
                out.append(s)
        return out

    def parts(self, k: int = 48, positions: int = 3) -> list[tuple]:
        """The k neurons at the last positions with the largest |activation x gradient of log p(the prediction)|."""
        torch = self.torch
        acts, hs = {}, []
        for l, layer in enumerate(self.layers):
            def pre(_m, args, l=l):
                x = args[0]
                x.retain_grad()
                acts[l] = x
                return None
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        try:
            lp = torch.log_softmax(self.m(input_ids=torch.tensor([self.ids], device=self.dev)).logits[0, self.t].float(), -1)
            lp[self.y0].backward()
        finally:
            for h in hs:
                h.remove()
        scores = []
        for l, x in acts.items():
            a = (x[0] * x.grad[0]).abs()[self.t - positions + 1:].float()
            v, i = a.flatten().topk(k)
            for vv, ii in zip(v.tolist(), i.tolist()):
                scores.append((vv, (f"h.{l}.mlp.neuron", self.t - positions + 1 + ii // self.I, ii % self.I)))
        self.m.zero_grad(set_to_none=True)
        return [x for _, x in sorted(scores, reverse=True)[:k]]

    def head(self, n_tokens: int = 24) -> str:
        T = len(self.ids)
        top = self.base.exp().topk(3)
        pred = ", ".join(f"{self.tok(int(i))!r} ({float(v):.2f})" for v, i in zip(top.values, top.indices))
        toks = " ".join(f"{p}:{self.tok(self.ids[p])!r}" for p in range(max(0, T - n_tokens), T))
        end = self.hf.decode(self.ids[max(0, T - 60):])
        return (f"A language model ({self.name.split('/')[-1]}, {len(self.layers)} transformer layers) reads a text and predicts "
                f"the next token after its last token (position {T - 1}).\nThe text ends: ...{end}\nIts last tokens by position: "
                f"{toks}\nThe model's prediction now: {pred}.\nParts are MLP neurons, named L<layer>.n#<index> (the neuron's "
                f"activation, the input of the layer's down projection) at a token position (@<position>).\n{self.group_lines()}")

    def make_groups(self, parts: list, band: int = 4, least: int = 4) -> None:
        """The circuit's neurons as named groups: per position and band of layers, groups of at least least neurons,
        named G1, G2, ... by position and then layer."""
        by: dict = {}
        for n, p, c in parts:
            by.setdefault((p, int(n.split(".")[1]) // band), []).append([n, p, c])
        keep = sorted((k, v) for k, v in by.items() if len(v) >= least)
        self.named = {f"G{i}": v for i, (_, v) in enumerate(keep, 1)}


def facts(tg, parts: list, seed: int = 99) -> str:
    """A facts-only explanation from measurements (no claim untested): the prediction, the parts whose single removal
    moves it most, swaps at the last position that keep and that move it (fresh tokens, a seed apart from the
    questions'), and the part whose set-back recovers most of a moving swap. A test explanation for the reward."""
    rng = random.Random(seed)
    y, p0 = tg.tok(tg.y0), float(tg.base[tg.y0].exp())
    rem = sorted(((tg.run({"kind": "remove", "parts": [list(x)]})["p"], x) for x in parts), key=lambda z: z[0])
    out = [f"The model gives {y!r} probability {p0:.2f}."]
    strong = [f"{label(*x)} (to {pp:.2f})" for pp, x in rem[:5]]
    out.append(f"Removing these parts alone lowers p({y!r}) most: {', '.join(strong)}; removing most other single parts "
               f"moves it by less than {abs(rem[len(rem) // 2][0] - p0):.2f}.")
    if tg.named:
        gr = sorted(((tg.run({"kind": "remove", "parts": ps})["p"], g) for g, ps in tg.named.items()), key=lambda z: z[0])
        out = [out[0], "Removing a group alone: " + ", ".join(f"{g} takes p({y!r}) to {pp:.2f}" for pp, g in gr[:6])
               + f"; the other groups each leave it above {gr[min(6, len(gr) - 1)][0]:.2f}."]
    else:
        for g in getattr(tg, "groups", ()):
            pg = tg.run({"kind": "remove", "parts": [list(x) for _, x in rem[:g]]})
            out.append(f"Removing the {g} strongest of them together takes p({y!r}) to {pg['p']:.2f} (top {pg['top']!r}).")
    sw = [(s, tg.run({"kind": "swap", "pos": tg.t, "token": s})) for s in tg.fresh(tg.t, 16, rng)]
    keep = [s for s, o in sw if o["top"] == y and o["p"] > p0 / 2]
    move = [(s, o) for s, o in sw if o["top"] != y or o["p"] <= p0 / 2]
    at = repr(tg.tok(tg.ids[tg.t]))
    if keep:
        out.append(f"Replacing {at} at position {tg.t} by {', '.join(map(repr, keep[:6]))} keeps {y!r} on top.")
    if move:
        out.append("Replacing it by " + ", ".join(f"{s!r} (top {o['top']!r}, p {o['p']:.2f})" for s, o in move[:6]) + " moves the prediction.")
        s, o = move[0]
        if tg.named:
            best = max(((tg.run({"kind": "setback", "pos": tg.t, "token": s, "parts": ps})["p"], g) for g, ps in tg.named.items()),
                       key=lambda z: z[0])
            out.append(f"After the swap to {s!r}, setting {best[1]} back to its values on the original text brings p({y!r}) "
                       f"from {o['p']:.2f} to {best[0]:.2f}.")
        else:
            best = max(((tg.run({"kind": "setback", "pos": tg.t, "token": s, "parts": [list(x)]})["p"], x) for _, x in rem[:8]),
                       key=lambda z: z[0])
            out.append(f"After the swap to {s!r}, setting {label(*best[1])} back to its value on the original text brings p({y!r}) "
                       f"from {o['p']:.2f} to {best[0]:.2f}.")
    return " ".join(out)


WANT = {"swap": 18, "remove": 11, "setback": 4, "scale": 3}  # questions per kind


def moved(tg, o: dict) -> bool:
    """Whether an outcome changes the model's top token (the questions' balance: half of each kind change it and half
    do not, so an unaided reader that answers 'no change' is right on half; 10-14 the lead: an unaided 0.75-0.79 left
    no headroom)."""
    return o["top"] != tg.tok(tg.y0)


def questions(tg, circuit: list, seed: int = 0, pool: int = 4) -> list[dict]:
    """Interventions independent of any explanation (module docstring), each with its measured outcome: per kind, pool
    times WANT drawn and measured, then half of WANT from those that change the top token and half from those that do
    not (the other side filling a short one; each set records how many change it). Removals and scalings take groups of
    every size (1 to 16 parts, or 1 to 4 named groups) so that some change the top token at all (a single part of
    text9000's circuit never does)."""
    rng = random.Random(seed)
    t = tg.t
    parts = [list(x) for x in circuit]
    at_t = [x for x in parts if x[1] == t] or parts
    sizes = [k for k in (1, 2, 4, 8, 16) if k <= len(parts)] or [1]
    if getattr(tg, "groups", None):
        sizes = [k for k in (tg.groups[0], 2 * tg.groups[0], tg.groups[1], 2 * tg.groups[1]) if k <= len(parts)] or [len(parts)]
    draw = {"swap": [{"kind": "swap", "pos": pos, "token": s} for pos, n in ((t, 8), (t - 1, 4), (t - 2, 3), (t - 3, 3))
                     for s in tg.fresh(pos, pool * n, rng)],
            "remove": [{"kind": "remove", "parts": rng.sample(parts, rng.choice(sizes))} for _ in range(pool * 11)],
            "setback": [{"kind": "setback", "pos": t, "token": s, "parts": rng.sample(at_t, min(len(at_t), rng.choice(sizes)))}
                        for s in tg.fresh(t, pool * 4, rng)],
            "scale": [{"kind": "scale", "parts": rng.sample(parts, rng.choice(sizes)), "by": rng.choice([0.0, 0.5, 2.0, 3.0])}
                      for _ in range(pool * 3)]}
    if tg.named:  # whole named groups: 1 to 4 of them removed or scaled together, set-backs of groups at the last position
        G = list(tg.named)
        gt = [g for g in G if any(p == t for _, p, _ in tg.named[g])] or G
        pick = lambda gs: {"group": gs if len(gs) > 1 else gs[0], "parts": [x for g in gs for x in tg.named[g]]}  # noqa: E731
        ks = [k for k in (1, 2, 3, 4) if k <= len(G)]
        draw["remove"] = [{"kind": "remove", **pick(rng.sample(G, rng.choice(ks)))} for _ in range(pool * 11)]
        draw["scale"] = [{"kind": "scale", "by": rng.choice([0.0, 0.5, 2.0, 3.0]), **pick(rng.sample(G, rng.choice(ks)))} for _ in range(pool * 3)]
        draw["setback"] = [{"kind": "setback", "pos": t, "token": s, **pick(rng.sample(gt, min(len(gt), rng.choice(ks))))}
                           for s in tg.fresh(t, pool * 4, rng)]
    ivs = []
    for kind, cands in draw.items():
        seen, uniq = set(), []
        for iv in cands:  # one question per distinct intervention
            key = json.dumps({k: v for k, v in iv.items() if k != "group"}, sort_keys=True)
            if key not in seen:
                seen.add(key)
                uniq.append(iv)
        for iv in uniq:
            iv["outcome"] = tg.run(iv)
        yes = [iv for iv in uniq if moved(tg, iv["outcome"])]
        no = [iv for iv in uniq if not moved(tg, iv["outcome"])]
        k = WANT[kind]
        a = min(len(yes), max(k // 2, k - len(no)))
        chosen = rng.sample(yes, a) + rng.sample(no, min(len(no), k - a))
        rng.shuffle(chosen)
        ivs += chosen
    for iv in ivs:
        iv["words"] = tg.words(iv)
        iv["moved"] = moved(tg, iv["outcome"])
    return ivs


def ask(tg_head: str, explanation: str | None, ivs: list[dict], answer_token: str) -> str:
    q = "\n".join(f"Q{k}: {iv['words']}" for k, iv in enumerate(ivs, 1))
    ex = f"An explanation of how the model computes this prediction:\n{explanation}\n\n" if explanation else ""
    return (f"{tg_head}\n{ex}For each change below, predict the model's most likely next token after the change and the "
            f"probability it then gives {answer_token!r}. Answer one line per question, Q<k>: '<token>' <probability>, "
            f"the token quoted exactly as written (with its leading space if any), nothing else.\n\n{q}")


def parse_answers(text: str, n: int) -> list[tuple[str | None, float | None]]:
    got: dict = {}
    for line in text.splitlines():
        m = re.match(r"\s*Q(\d+)\s*:\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*,?\s*([0-9]*\.?[0-9]+(?:[eE]-?\d+)?)", line)
        if m:
            try:
                tok = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                tok = None
            got.setdefault(int(m.group(1)), (tok, min(1.0, max(0.0, float(m.group(3))))))
    return [got.get(k, (None, None)) for k in range(1, n + 1)]


def logit(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, p))
    return math.log(p / (1 - p))


def grade(answers: list, ivs: list[dict]) -> list[dict]:
    out = []
    for (tok, p), iv in zip(answers, ivs):
        top_ok = tok is not None and tok == iv["outcome"]["top"]
        pc = 0.0 if p is None else max(0.0, 1 - abs(logit(p) - logit(iv["outcome"]["p"])) / LOGIT_SCALE)
        out.append({"kind": iv["kind"], "top_ok": top_ok, "p_credit": pc, "score": 0.5 * top_ok + 0.5 * pc,
                    "reader": [tok, p], "measured": iv["outcome"]})
    return out


class Reader:
    """A frozen open model on Tinker: greedy answers, tokens and dollars counted (tinker_train.Spend)."""

    def __init__(self, base: str, log: Path):
        import tinker

        import tinker_train

        self.tinker, self.base = tinker, base
        service = tinker_train.service_client()
        self.renderer, self.tokenizer = tinker_train.renderer_for(base)
        self.sampler = service.create_sampling_client(base_model=base)
        log.mkdir(parents=True, exist_ok=True)
        self.spend = tinker_train.Spend(log, base)
        self.tt = tinker_train

    def __call__(self, prompts: list[str], max_tokens: int = 1500) -> list[str]:
        import components

        params = self.tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=self.renderer.get_stop_sequences())
        futs = []
        for text in prompts:
            prompt = self.renderer.build_generation_prompt([{"role": "user", "content": components.escape(text)}])
            futs.append((prompt.length, self.sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)))
        out = []
        for n, fut in futs:
            toks = fut.result().sequences[0].tokens
            text, _ = self.tt.answer_of(self.renderer, self.tokenizer, toks)
            self.spend.add("reader_reward", prefill=n, sampled=len(toks))
            out.append(text)
        return out

    def tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


INV_GRAMMAR = """Write {n} invariants this explanation implies: tests on the model, each with the outcome it predicts. One per line, in exactly this grammar:
INV: swap @<where> with <scope> -> <outcome>
INV: remove <part>[, <part> ...] -> <outcome>
INV: scale <part> by <factor> -> <outcome>
INV: swap @<where> with <scope> and set back <part>[, <part> ...] -> <outcome>
where <where> is a position, written as its number or as the token there quoted (for example @' creation', the last position holding that token), <part> is written L<layer>.<kind>#<index>@<where> or is a group's name (G<k>), and <outcome> is one of: top '<token>' | top not '<token>' | p > <number> | p < <number>, p being the probability the model gives {answer!r}. <scope> is the set of tokens the claim covers, one of: nouns, verbs, adjectives, adverbs, prepositions, pronouns, determiners, conjunctions, numbers, names, predicted tokens (tokens the model itself predicts at that position); the checker draws the test tokens from it. Nothing else."""


LEX = Path.home() / "mpd-data" / "graph_oracle" / "gwas" / "lex"
SCOPES = {"nouns": "Noun", "verbs": "Verb", "adjectives": "Adjective", "adverbs": "Adverb", "prepositions": "Preposition",
          "pronouns": "Pronoun", "determiners": "Determiner", "conjunctions": "Conjunction", "numbers": "Number", "names": "Name"}
DRAWS = 8  # instances the checker draws per INV line
_CLASSES: dict = {}


def class_words(cls: str) -> list[str]:
    """English words whose dominant part of speech is cls (Brysbaert et al.'s concreteness norms, Dom_Pos), most
    frequent first (SUBTLEX-US counts)."""
    if not _CLASSES:
        freq = {}
        for i, line in enumerate((LEX / "SUBTLEXus74286wordstextversion.txt").read_text(errors="ignore").splitlines()):
            f = line.split("\t")
            if i and len(f) > 1 and f[1].isdigit():
                freq[f[0].lower()] = int(f[1])
        for i, line in enumerate((LEX / "Concreteness_ratings_Brysbaert_et_al_BRM.txt").read_text(errors="ignore").splitlines()):
            f = line.split("\t")
            if i and len(f) > 2 and " " not in f[0]:
                _CLASSES.setdefault(f[-1], []).append(f[0])
        for k in _CLASSES:
            _CLASSES[k].sort(key=lambda w: -freq.get(w.lower(), 0))
    return _CLASSES.get(cls, [])


def drawn(tg, pos: int, scope: str, rng: random.Random, k: int = DRAWS) -> list[str] | None:
    """k tokens the checker draws from an INV line's scope at pos (never the answer's choice, the lead's 10-14): the
    model's own prediction there (fresh draws), or a part of speech (single tokens of the target among the class's 2,000
    most frequent words, a leading space as in running text); None for a scope it does not know."""
    if scope in ("predicted tokens", "tokens the model predicts there"):
        return tg.fresh(pos, k, rng)
    if scope in SCOPES:
        toks = [" " + w for w in class_words(SCOPES[scope])[:2000]]
        toks = [x for x in toks if tg.token_id(x) is not None and x != tg.tok(tg.ids[pos])]
        return rng.sample(toks, min(k, len(toks))) if toks else None
    return None


def parse_inv(line: str, tg: VPDTarget, rng: random.Random | None = None) -> tuple[list[dict], tuple] | None:
    """An INV line as (its interventions, one per instance, and its outcome), or None. A swap's instances are drawn by
    the checker from the line's scope ('with nouns', 'with predicted tokens', ...); a listed set ('with any of [...]')
    is the answer's own choice of instances and fails."""
    m = re.match(r"\s*INV:\s*(.+?)\s*->\s*(.+?)\s*$", line)
    if not m:
        return None
    body, oc = m.groups()
    mo = re.fullmatch(r"top\s+(not\s+)?('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")|p\s*([<>])\s*([0-9]*\.?[0-9]+)", oc)
    if not mo:
        return None
    if mo.group(2):
        try:
            outcome = ("top_not" if mo.group(1) else "top", ast.literal_eval(mo.group(2)))
        except (ValueError, SyntaxError):
            return None
    else:
        outcome = (mo.group(3), float(mo.group(4)))

    def parts_of(s):
        ps = []
        for x in re.split(r",\s*(?=[LG]\d)", s.strip()):
            if re.fullmatch(r"G\d+", x.strip()):
                if x.strip() not in tg.named:
                    return None
                ps += [tuple(y) for y in tg.named[x.strip()]]
            else:
                ps.append(site_of(x, tg.where))
        return None if not ps or any(p is None or not tg.valid(p) for p in ps) else [list(p) for p in ps]

    ms = re.fullmatch(r"swap\s+@(\d+|'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s+with\s+(.+?)(?:\s+and set back\s+(.+))?", body)
    if ms:
        pos = tg.where(ms.group(1))
        if pos is None or ms.group(2).startswith("any of"):
            return None
        toks = drawn(tg, pos, ms.group(2).strip().lower(), rng or random.Random(line))
        if not toks:
            return None
        if ms.group(3):
            ps = parts_of(ms.group(3))
            if ps is None:
                return None
            return [{"kind": "setback", "pos": pos, "token": x, "parts": ps} for x in toks], outcome
        return [{"kind": "swap", "pos": pos, "token": x} for x in toks], outcome
    mr = re.fullmatch(r"remove\s+(.+)", body)
    if mr:
        ps = parts_of(mr.group(1))
        return ([{"kind": "remove", "parts": ps}], outcome) if ps else None
    mc = re.fullmatch(r"scale\s+(.+?)\s+by\s+([0-9]*\.?[0-9]+)", body)
    if mc:
        ps = parts_of(mc.group(1))
        return ([{"kind": "scale", "parts": ps, "by": float(mc.group(2))}], outcome) if ps else None
    return None


def meets(o: dict, outcome: tuple) -> bool:
    k, v = outcome
    if k == "top":
        return o["top"] == v
    if k == "top_not":
        return o["top"] != v
    return o["p"] > v if k == ">" else o["p"] < v


def explanation_of(path: str) -> str | None:
    return None if path == "none" else Path(path).read_text().strip()


def target_of(a):
    """The target and its parts: vpd4l with a stated circuit's subcomponents, or a Hugging Face model (--model) on the
    task's text with its attributed neurons."""
    if a.model:
        import mech

        ids = json.loads((TEXTS / f"{a.task}.json").read_text())["prompts"][0]["token_ids"]
        text = mech.tokenizer("vpd4l").decode(ids).replace("<|endoftext|>", "")
        tg = HFTarget(a.model, text, a.device)
        parts = tg.parts(256)
        tg.make_groups(parts)
        return tg, parts
    tg = VPDTarget(a.task, a.device)
    circ = json.loads(a.circuit.read_text()) if a.circuit else {"nodes": []}
    return tg, [(n, int(p), int(c)) for n, c, p in circ["nodes"] if n.startswith("h.")]


def cmd_questions(a) -> None:
    tg, parts = target_of(a)
    ivs = questions(tg, parts, a.seed)
    if a.facts:
        a.facts.write_text(facts(tg, parts))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps({"task": a.task, "head": tg.head(), "answer": tg.tok(tg.y0), **iv}) + "\n" for iv in ivs))
    print(len(ivs), "questions to", a.out)


def rewards(rd: "Reader", ivs: list[dict], explanations: list[str | None], baseline: float | None = None) -> list[dict]:
    """The reader reward of each explanation (None: no explanation) on one text's questions, the reader's calls made
    together (an RL group's samples in one round): per explanation its reward, accuracy, accuracy over the
    no-explanation reader (baseline, when given), its moved questions' scores, tokens, and per-kind shares."""
    got = rd([ask(ivs[0]["head"], ex, ivs, ivs[0]["answer"]) for ex in explanations])
    out = []
    for ex, text in zip(explanations, got):
        rows = grade(parse_answers(text, len(ivs)), ivs)
        n_ex = rd.tokens(ex) if ex else 0
        acc = sum(r["score"] for r in rows) / len(rows)
        out.append({"reward": acc - SIMPLICITY * n_ex / 1000, "accuracy": acc, "over_none": None if baseline is None else acc - baseline,
                    "moved_right": sum(r["score"] for r, iv in zip(rows, ivs) if iv.get("moved")),
                    "moved_n": sum(bool(iv.get("moved")) for iv in ivs), "explanation_tokens": n_ex, "raw": text,
                    "by_kind": {k: {"n": sum(r["kind"] == k for r in rows), "top_right": sum(r["top_ok"] for r in rows if r["kind"] == k),
                                    "p_credit": round(sum(r["p_credit"] for r in rows if r["kind"] == k) / sum(r["kind"] == k for r in rows), 3)}
                                for k in sorted({r["kind"] for r in rows})}, "rows": rows})
    return out


def cmd_read(a) -> None:
    ivs = [json.loads(x) for x in a.questions.read_text().splitlines()]
    rd = Reader(a.reader, a.out.parent)
    base = json.loads(a.baseline.read_text())["accuracy"] if a.baseline else None
    res = {"explanation": a.explanation, "reader": a.reader, **rewards(rd, ivs, [explanation_of(a.explanation)], base)[0],
           "spend_usd": rd.spend.total}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("explanation", "reward", "accuracy", "over_none", "moved_right", "moved_n", "explanation_tokens",
                                           "spend_usd", "by_kind")}))


def check_lines(tg, text: str, seed: int = 0) -> dict:
    """An answer's INV lines (its own falsifiable claims, the lead's 10-14) run on the target: per line whether it
    parses, holds on every instance, the share of instances that meet its outcome, and whether it is bold (the
    unchanged model does not already meet its outcome: 'p < 0.4' while p is 0.95, 'top not' the prediction; a safe
    claim earns nothing an unread text would not); share, instance share and bold share over the lines (a line that
    does not parse fails)."""
    rows = []
    o0 = {"top": tg.tok(tg.y0), "p": float(tg.base[tg.y0].exp())}
    for line in [x for x in text.splitlines() if x.strip().startswith("INV")]:
        p = parse_inv(line, tg, random.Random(f"{seed}|{line}"))
        if p is None:
            rows.append({"line": line, "parsed": False, "holds": False, "bold": False})
            continue
        ivs, outcome = p
        outs = [tg.run(iv) for iv in ivs]
        ok = [meets(o, outcome) for o in outs]
        rows.append({"line": line, "parsed": True, "holds": all(ok), "instances": len(ok), "met": sum(ok), "bold": not meets(o0, outcome),
                     "outcomes": [[iv.get("token"), o["top"], round(o["p"], 4)] for iv, o in zip(ivs, outs)]})
    n = max(len(rows), 1)
    return {"share_holding": sum(r["holds"] for r in rows) / n,
            "instance_share": sum(r.get("met", 0) / r["instances"] if r.get("instances") else 0.0 for r in rows) / n,
            "bold_holding": sum(r["holds"] and r["bold"] for r in rows) / n,
            "lines": len(rows), "parsed": sum(r["parsed"] for r in rows), "rows": rows}


def show(rows: list) -> None:
    for r in rows:
        print(("HOLDS " if r["holds"] else "fails ") + ("bold " if r.get("bold") else "safe ") + r["line"][:150], r.get("met"), "/", r.get("instances"))


def cmd_invariants(a) -> None:
    """A separate reader writes the invariants an explanation implies, and they are checked (check_lines)."""
    tg = VPDTarget(a.task, a.device)
    rd = Reader(a.reader, a.out.parent)
    ex = explanation_of(a.explanation)
    exs = f"An explanation of how the model computes this prediction:\n{ex}\n\n" if ex else ""
    got = rd([f"{tg.head()}\n{exs}{INV_GRAMMAR.format(n=a.n, answer=tg.tok(tg.y0))}"])[0]
    res = {"explanation": a.explanation, "reader": a.reader, **check_lines(tg, got), "spend_usd": rd.spend.total, "raw": got}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("explanation", "share_holding", "instance_share", "bold_holding", "lines", "parsed", "spend_usd")}))
    show(res["rows"])


def cmd_answer(a) -> None:
    """An oracle answer (English, and INV lines in the grammar: its own invariants) scored both ways: its INV lines
    checked on the target, and its English (the INV lines left out) through the reader on a question file, against
    the no-explanation result of the same questions (--baseline)."""
    text = a.answer.read_text()
    english = "\n".join(x for x in text.splitlines() if not x.strip().startswith("INV")).strip()
    tg = target_of(a)[0]
    res = {"answer": str(a.answer), **check_lines(tg, text)}
    if a.questions:
        ivs = [json.loads(x) for x in a.questions.read_text().splitlines()]
        rd = Reader(a.reader, a.out.parent)
        base = json.loads(a.baseline.read_text())["accuracy"] if a.baseline else None
        res["reader"] = {k: v for k, v in rewards(rd, ivs, [english or None], base)[0].items() if k not in ("rows", "raw")}
        res["spend_usd"] = rd.spend.total
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("share_holding", "instance_share", "bold_holding", "lines", "parsed")}), res.get("reader"))
    show(res["rows"])


class VLLMReader:
    """The reader self-hosted for training (the lead's 10-14: Tinker's $6 per 1,000 rewards is too much for RL): vLLM
    offline batches, greedy, the chat template with thinking off. Same calls as Reader."""

    def __init__(self, model: str, max_len: int = 6144, util: float = 0.85, seqs: int = 64):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = AutoTokenizer.from_pretrained(model)
        # a 48 GB card: the FP8 27B's weights take 27.9 GiB, and at utilization 0.92 the CUDA-graph profiling ran out
        self.llm = LLM(model=model, max_model_len=max_len, gpu_memory_utilization=util, max_num_seqs=seqs,
                       limit_mm_per_prompt={"image": 0, "video": 0})
        self.spend = type("S", (), {"total": 0.0})()

    def __call__(self, prompts: list[str], max_tokens: int = 1500) -> list[str]:
        from vllm import SamplingParams

        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False) for p in prompts]
        outs = self.llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=max_tokens))
        return [o.outputs[0].text for o in outs]

    def tokens(self, text: str) -> int:
        return len(self.tok(text)["input_ids"])


def cmd_agree_prep(a) -> None:
    """The reader prompts already answered by Tinker's reader (read's result files: --results), with their questions'
    measured outcomes, into one file for the self-hosted reader's agreement check."""
    out = []
    for spec in a.results:  # RESULT, or RESULT=QUESTIONS when the result's name holds no question file's name
        f, _, qs = str(spec).partition("=")
        f = Path(f)
        r = json.loads(f.read_text())
        q = Path(qs) if qs else max((x for x in a.question_files if x.stem in f.stem), key=lambda x: len(x.stem))
        ivs = [json.loads(x) for x in q.read_text().splitlines()]
        ex = explanation_of(r["explanation"])
        out.append({"result": f.name, "questions": q.name, "prompt": ask(ivs[0]["head"], ex, ivs, ivs[0]["answer"]),
                    "tinker_raw": r["raw"], "outcomes": [iv["outcome"] for iv in ivs], "kinds": [iv["kind"] for iv in ivs]})
    a.out.write_text(json.dumps(out))
    print(len(out), "prompts,", sum(len(x["outcomes"]) for x in out), "questions to", a.out)


def cmd_agree(a) -> None:
    """The self-hosted reader against Tinker's on the same prompts: per question whether the two readers name the
    same top token and how far apart their probabilities are (log-odds), each reader's accuracy per prompt; then
    throughput (prompts per hour in one batch of --batch prompts), for the $ per 1,000 rewards."""
    import time

    pairs = json.loads(a.pairs.read_text())
    rd = VLLMReader(a.model)
    t0 = time.time()
    got = rd([x["prompt"] for x in pairs])
    t_first = time.time() - t0
    rows, per = [], []
    for x, text in zip(pairs, got):
        ivs = [{"kind": k, "outcome": o} for k, o in zip(x["kinds"], x["outcomes"])]
        mine, theirs = parse_answers(text, len(ivs)), parse_answers(x["tinker_raw"], len(ivs))
        for (t1, p1), (t2, p2) in zip(mine, theirs):
            rows.append({"same_top": t1 is not None and t1 == t2, "logodds_gap": None if p1 is None or p2 is None else abs(logit(p1) - logit(p2))})
        g1, g2 = grade(mine, ivs), grade(theirs, ivs)
        per.append({"result": x["result"], "acc_self": sum(r["score"] for r in g1) / len(g1), "acc_tinker": sum(r["score"] for r in g2) / len(g2)})
    gaps = sorted(r["logodds_gap"] for r in rows if r["logodds_gap"] is not None)
    batch = [pairs[i % len(pairs)]["prompt"] for i in range(a.batch)]
    t0 = time.time()
    rd(batch)
    dt = time.time() - t0
    res = {"model": a.model, "questions": len(rows), "same_top": sum(r["same_top"] for r in rows) / len(rows),
           "logodds_gap_median": gaps[len(gaps) // 2] if gaps else None, "per_prompt": per,
           "accuracy_gap_mean": sum(abs(x["acc_self"] - x["acc_tinker"]) for x in per) / len(per),
           "first_batch_s": t_first, "throughput_batch": a.batch, "throughput_s": dt, "rewards_per_hour": a.batch / dt * 3600, "raw": got}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("raw", "per_prompt")}))
    for x in per:
        print(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    q = sub.add_parser("questions")
    q.add_argument("--task", default="text9000")
    q.add_argument("--circuit", type=Path, help="vpd4l: the stated circuit whose parts the questions remove")
    q.add_argument("--model", help="a Hugging Face model (Qwen/Qwen3-1.7B) in place of vpd4l: its neurons are the parts")
    q.add_argument("--facts", type=Path, help="also write a facts-only explanation of the target here (facts())")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("read")
    r.add_argument("--questions", type=Path, required=True)
    r.add_argument("--explanation", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--baseline", type=Path, help="the same questions' no-explanation result (read --explanation none)")
    i = sub.add_parser("invariants")
    i.add_argument("--task", default="text9000")
    i.add_argument("--explanation", required=True)
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--n", type=int, default=8)
    w = sub.add_parser("answer")
    w.add_argument("--answer", type=Path, required=True, help="the answer: English and its INV lines")
    w.add_argument("--task", default="text9000")
    w.add_argument("--circuit", type=Path, help="vpd4l: unused here (the target's parts come from the answer)")
    w.add_argument("--model", help="a Hugging Face model in place of vpd4l")
    w.add_argument("--questions", type=Path, help="also the reader reward of its English on these questions")
    w.add_argument("--baseline", type=Path)
    w.add_argument("--out", type=Path, required=True)
    for p in (r, i, w):
        p.add_argument("--reader", default=READER)
    for p in (q, i, w):
        p.add_argument("--device")
    ap_ = sub.add_parser("agree-prep")
    ap_.add_argument("--results", nargs="+", required=True)
    ap_.add_argument("--question-files", type=Path, nargs="+", required=True)
    ap_.add_argument("--out", type=Path, required=True)
    ag = sub.add_parser("agree")
    ag.add_argument("--pairs", type=Path, required=True)
    ag.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    ag.add_argument("--batch", type=int, default=256)
    ag.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"questions": cmd_questions, "read": cmd_read, "invariants": cmd_invariants, "answer": cmd_answer, "agree-prep": cmd_agree_prep,
     "agree": cmd_agree}[a.what](a)


if __name__ == "__main__":
    main()
