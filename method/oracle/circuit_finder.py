"""Question-scoped circuits by the patching pair (the lead's 10-15 CIRCUIT FINDER spec, ~/mpd-data/team/direction_1015.txt).

A question is two prompts of the same length run through the same model: clean (the model gives the answer y, its top
token) and corrupted (a key word swapped; the model gives another answer b). The contrast is log2 p(y) - log2 of the
summed p of the ALTERNATIVES alternatives (b first, then the clean run's likeliest tokens other than y and b), at the
last position; gap = c_clean - c_corrupt, at least MIN_GAP bits (neuron_scorecard's pair_alts and contrast).
Units: a neuron at a position (layer, pos, index; the down projection's input act(gate) x up), recorded with its pole
(the sign of its clean value) and SwiGLU regime (the signs of act(gate) and of up), and a head at a position (that
head's slice of the output projection's input), each patched by setting it to its value in the other run. Positions
before the first differing token have equal values in both runs.
  noising (necessity): the clean run with a set patched from the corrupted run; lost share = (c_clean - c) / gap
  denoising (sufficiency): the corrupted run with only a set patched from the clean run; restored = (c - c_corrupt) / gap
A circuit passes when both shares are at least PASS.
Search (every decision a measured run; gradients only order):
 1. positions: each layer's residual input at each position from the first differing token, denoised alone (the map of
    where the answer's information sits, recorded);
 2. group testing over blocks of one layer's neurons (or heads) at one position, each noised and denoised alone; a block
    whose larger |share| reaches EPS is split in two halves of its units ordered by their attribution estimate |patch
    difference x gradient| (the larger of the clean run's gradient for noising and the corrupted run's for denoising),
    down to single units, each measured alone;
 3. a cut: the single units of one kind at some positions in measured order (the larger |share|), the curve of both
    shares as they are added in that order, the smallest prefix that passes, then single units removed while it still
    passes; when no prefix passes, EPS falls by EPS_STEP and step 2 searches again (blocks measured before reused), at
    most EPS_ROUNDS. Three cuts make the circuit, each passing alone with everything else computing:
      readout: neurons at the answer position;
      movers: heads at the answer position (where each looks recorded);
      source: neurons at the positions from the first differing token to the one before the answer;
    with each mover's share of the source cut's effect (the head frozen at its base-run value while the source cut is
    patched). The smallest set over all units is one cut, the narrowest place the answer's information crosses (one or
    two heads at the answer position on Qwen3-1.7B capitals), so the circuit is the path of cuts from the changed
    token to the answer;
 4. reported: the neurons with their stage, poles, regimes and measured shares alone, the movers, each cut's curve and
    size, the residual map, seconds.
Every run is a batch of exactly ROWS (padded), so bfloat16 kernels do not change with the batch size.

  circuit_finder.py questions QUESTIONS.json OUT [--split dev] [--model Qwen/Qwen3-8B] [--dtype float32] [--adapter A]
  circuit_finder.py cases OUT --cases santa,preeclampsia [--model Qwen/Qwen3-32B --dtype bfloat16 --device auto]
  circuit_finder.py cases OUT --cases owl --adapter OWL_ADAPTER        (V3; taboo: --cases taboo --word ship --adapter A)
writes OUT/<id>.json per question.
"""
from __future__ import annotations

import math
import time

import torch

ALTERNATIVES = 5
MIN_GAP = 1.0         # bits: a pair counts when the answers differ and c_clean - c_corrupt reaches this
PASS = 0.7            # the spec's share for necessity (lost) and sufficiency (restored)
EPS = 0.02            # a block is split when its larger |share| reaches this share of the gap
EPS_STEP = 4.0
EPS_ROUNDS = 3
ROWS = 32
ATTENDS = 3           # source positions recorded per head


class Patcher:
    """An HF decoder (Qwen3; a PeftModel with its adapter on) run on a clean and a corrupted prompt, with batched patches
    of residual inputs, head outputs and neurons from the other run."""

    def __init__(self, model, tok, rows: int = ROWS):
        self.model, self.tok, self.rows = model, tok, rows
        self.core = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.layers = list(self.core.model.layers)
        cfg = self.core.config
        self.L, self.D, self.H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev = self.core.model.embed_tokens.weight.device

    def _hooks(self, res=None, head=None, neu=None, add=None, cap=None):
        """Forward hooks: res/head/neu(i, h) -> h edit each layer's residual input, head outputs (o_proj input) and
        neurons (down_proj input); add: {"head": [..], "neu": [..]} tensors added (gradients); cap: a dict of lists."""
        hs = []
        for i, layer in enumerate(self.layers):
            def r(mod, args, kwargs, i=i):
                h = args[0] if args else kwargs["hidden_states"]
                if cap is not None:
                    cap.setdefault("res", []).append(h.detach())
                if res is not None:
                    h = res(i, h)
                    if args:
                        args = (h,) + tuple(args[1:])
                    else:
                        kwargs["hidden_states"] = h
                return args, kwargs

            def o(mod, args, i=i):
                h = args[0]
                if cap is not None:
                    cap.setdefault("head", []).append(h.detach())
                if head is not None:
                    h = head(i, h)
                if add is not None:
                    h = h + add["head"][i].to(h.dtype)
                return (h,) + tuple(args[1:])

            def d(mod, args, i=i):
                h = args[0]
                if cap is not None:
                    cap.setdefault("neu", []).append(h.detach())
                if neu is not None:
                    h = neu(i, h)
                if add is not None:
                    h = h + add["neu"][i].to(h.dtype)
                return (h,) + tuple(args[1:])
            hs.append(layer.register_forward_pre_hook(r, with_kwargs=True))
            hs.append(layer.self_attn.o_proj.register_forward_pre_hook(o))
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(d))
            if cap is not None:
                hs.append(layer.mlp.up_proj.register_forward_hook(lambda mod, args, out: cap.setdefault("up", []).append(out.detach())))
        return hs

    def _last(self, ids, B: int, **hooks) -> torch.Tensor:
        """[B, V] float32 log-probabilities at the last position."""
        x = torch.as_tensor(ids, device=self.dev)[None].expand(B, -1)
        hs = self._hooks(**hooks)
        try:
            lg = self.core(input_ids=x, logits_to_keep=1).logits[:, -1].float()
        finally:
            for h in hs:
                h.remove()
        return torch.log_softmax(lg, -1)

    @torch.no_grad()
    def setup(self, clean: list[int], corrupt: list[int], answers: list[int] | None = None) -> dict:
        """Both runs' activations, the answer (the clean top token, or the likeliest of answers: a secret word's tokens),
        the alternatives and both contrasts (at the fixed batch size)."""
        assert len(clean) == len(corrupt), "a patching pair needs prompts of the same length"
        self.ids = {"clean": list(clean), "corrupt": list(corrupt)}
        self.T = len(clean)
        self.first = next(p for p in range(self.T) if clean[p] != corrupt[p])
        self.act = {}
        lp = {}
        for k, ids in self.ids.items():
            cap = {}
            lp[k] = self._last(ids, self.rows, cap=cap)[0]
            self.act[k] = {n: [t[0] for t in v] for n, v in cap.items()}           # per layer [T, dim]
        self.y = int(lp["clean"].argmax()) if not answers else max(answers, key=lambda z: float(lp["clean"][z]))
        yc = int(lp["corrupt"].argmax())
        alt = ([yc] if yc != self.y else []) + [int(z) for z in lp["clean"].topk(ALTERNATIVES + 2).indices.tolist()
                                                if z not in (self.y, yc)]
        alt = alt[:ALTERNATIVES]                    # (neuron_scorecard.pair_alts: the judge's contrast, number for number)
        self.alt = torch.tensor(alt, device=lp["clean"].device)
        self.c = {k: float(self._contrast(v[None])[0]) for k, v in lp.items()}
        self.gap = self.c["clean"] - self.c["corrupt"]
        return {"answer": self.y, "corrupt_answer": yc, "alternatives": alt, "contrast": self.c, "gap": self.gap,
                "first_diff": self.first}

    def _contrast(self, lp: torch.Tensor) -> torch.Tensor:
        return (lp[:, self.y] - torch.logsumexp(lp[:, self.alt], -1)) / math.log(2)

    @torch.no_grad()
    def run(self, base: str, sets: list) -> list[float]:
        """Per set of units ("res", layer, pos) | ("head", layer, k, pos) | ("neuron", layer, pos, k), k an index or a
        tuple of them (a block), each set to its value in the other run (or in the run a fifth element names: "clean"
        or "corrupt", a head frozen at its value there), in the base run ("clean" or "corrupt"): the share of the gap
        lost (base clean) or restored (base corrupt)."""
        other = "corrupt" if base == "clean" else "clean"
        out = []
        for s0 in range(0, len(sets), self.rows):
            part = sets[s0:s0 + self.rows]
            by = {}
            for b, s in enumerate(part):
                for u in s:
                    if u[0] == "res":
                        k, p, src = torch.zeros(1, dtype=torch.long), u[2], u[3] if len(u) > 3 else other
                    else:
                        k = torch.as_tensor(u[2] if u[0] == "head" else u[3]).reshape(-1)
                        p, src = (u[3] if u[0] == "head" else u[2]), (u[4] if len(u) > 4 else other)
                    by.setdefault((u[0], src), {}).setdefault(u[1], []).append((torch.full_like(k, b), torch.full_like(k, p), k))
            idx = {key: {i: tuple(torch.cat(c) for c in zip(*v)) for i, v in d.items()} for key, d in by.items()}

            def res(i, h):
                for src in ("clean", "corrupt"):
                    if i in idx.get(("res", src), {}):
                        b, p, _ = (t.to(h.device) for t in idx[("res", src)][i])
                        h = h.clone()
                        h[b, p] = self.act[src]["res"][i][p].to(h.device, h.dtype)
                return h

            def head(i, h):
                for src in ("clean", "corrupt"):
                    if i in idx.get(("head", src), {}):
                        b, p, k = (t.to(h.device) for t in idx[("head", src)][i])
                        B, T, _ = h.shape
                        h4 = h.reshape(B, T, self.H, self.hd).clone()
                        h4[b, p, k] = self.act[src]["head"][i].reshape(T, self.H, self.hd)[p, k].to(h.device, h.dtype)
                        h = h4.reshape(B, T, self.H * self.hd)
                return h

            def neu(i, h):
                for src in ("clean", "corrupt"):
                    if i in idx.get(("neuron", src), {}):
                        b, p, n = (t.to(h.device) for t in idx[("neuron", src)][i])
                        h = h.clone()
                        h[b, p, n] = self.act[src]["neu"][i][p, n].to(h.device, h.dtype)
                return h
            c = self._contrast(self._last(self.ids[base], self.rows, res=res, head=head, neu=neu))[:len(part)]
            if base == "clean":
                out += ((self.c["clean"] - c) / self.gap).tolist()
            else:
                out += ((c - self.c["corrupt"]) / self.gap).tolist()
        return out

    def both(self, sets: list) -> list[tuple]:
        """Per set: (lost share noising, restored share denoising)."""
        return list(zip(self.run("clean", sets), self.run("corrupt", sets)))

    def scores(self) -> dict:
        """Attribution estimates per head [L, T, H] and neuron [L, T, D]: max over directions of |patch difference x the
        base run's gradient of the contrast| (the order of group testing only)."""
        out = {}
        for base, src in (("clean", "corrupt"), ("corrupt", "clean")):
            add = {"head": [torch.zeros(self.T, self.H * self.hd, device=l.self_attn.o_proj.weight.device, requires_grad=True)
                            for l in self.layers],
                   "neu": [torch.zeros(self.T, self.D, device=l.mlp.down_proj.weight.device, requires_grad=True) for l in self.layers]}
            with torch.enable_grad():
                self._contrast(self._last(self.ids[base], 1, add=add)).sum().backward()
            for k, key in (("head", "head"), ("neu", "neuron")):
                s = torch.stack([((self.act[src][k][i].float() - self.act[base][k][i].float()) * add[k][i].grad.float()).cpu()
                                 for i in range(self.L)])                                   # [L, T, dim]
                if k == "head":
                    s = s.reshape(self.L, self.T, self.H, self.hd).sum(-1)
                s = s.abs()
                out[key] = s if key not in out else torch.maximum(out[key], s)
        return out

    @torch.no_grad()
    def attends(self, layer: int, head: int, pos: int, n: int = ATTENDS) -> list:
        """Where a head looks from pos in the clean run: the n largest attention weights [[source, weight], ...]."""
        if getattr(self, "_att_ids", None) != self.ids["clean"]:
            cfg = self.core.config
            impl, cfg._attn_implementation = cfg._attn_implementation, "eager"
            try:
                x = torch.as_tensor(self.ids["clean"], device=self.dev)[None]
                self._att = [a[0].float().cpu() for a in self.core.model(input_ids=x, output_attentions=True).attentions]
            finally:
                cfg._attn_implementation = impl
            self._att_ids = self.ids["clean"]
        w, s = self._att[layer][head, pos].topk(min(n, pos + 1))
        return [[int(a), round(float(b), 3)] for a, b in zip(s, w)]


def group_test(pt: Patcher, sc: dict, eps: float, done: dict, kinds=("neuron",), positions=None, measure=None, log=None) -> dict:
    """Group testing (module docstring): every single unit reached, {unit: (lost, restored)}. A block is (kind, layer,
    pos, lo, hi): the units lo..hi-1 of that layer and position in descending attribution order; measure(sets) gives each
    set's pair of shares (default Patcher.both); done holds the blocks measured before (at a larger eps), reused."""
    measure = measure or pt.both
    order = {k: v.argsort(-1, descending=True) for k, v in sc.items()}                # [L, T, H] and [L, T, D]
    positions = range(pt.first, pt.T) if positions is None else positions
    blocks = [(k, i, p, 0, order[k].shape[-1]) for i in range(pt.L) for p in positions for k in kinds]
    singles, level = {}, 0
    while blocks:
        todo = [b for b in blocks if b not in done]
        sets = [[unit(b[0], b[1], b[2], order[b[0]][b[1], b[2], b[3]:b[4]])] for b in todo]
        for b, r in zip(todo, measure(sets)):
            done[b] = r
        nxt = []
        for b in blocks:
            lost, rest = done[b]
            if max(abs(lost), abs(rest)) < eps:
                continue
            if b[4] - b[3] == 1:
                singles[unit(b[0], b[1], b[2], int(order[b[0]][b[1], b[2], b[3]]))] = (lost, rest)
            else:
                h = (b[3] + b[4]) // 2
                nxt += [(b[0], b[1], b[2], b[3], h), (b[0], b[1], b[2], h, b[4])]
        if log:
            log(f"    level {level}: {len(blocks)} blocks ({len(todo)} new), {len(nxt)} split, {len(singles)} singles")
        blocks, level = nxt, level + 1
    return singles


def unit(kind: str, layer: int, pos: int, k: int) -> tuple:
    return ("head", layer, k, pos) if kind == "head" else ("neuron", layer, pos, k)


def cut(pt: Patcher, sc: dict, kind: str, positions, done: dict, log=None) -> dict:
    """The smallest set of one kind of unit at the given positions that passes (module docstring, steps 2-3)."""
    eps, singles = EPS, {}
    for rnd in range(EPS_ROUNDS):
        singles.update(group_test(pt, sc, eps, done, kinds=(kind,), positions=positions, log=log))
        order = sorted(singles, key=lambda u: -max(abs(singles[u][0]), abs(singles[u][1])))
        curve = pt.both([order[:k] for k in range(1, len(order) + 1)])
        ok = [k for k, (lost, rest) in enumerate(curve, 1) if lost >= PASS and rest >= PASS]
        if log:
            log(f"  {kind}s at {positions[0]}..{positions[-1]}, eps {eps:.4f}: {len(order)} single units, passing prefix {ok[0] if ok else None}")
        if ok:
            break
        eps /= EPS_STEP
    circ = order[:ok[0]] if ok else list(order)
    if ok and len(circ) > 1:                        # units whose removal alone still passes, then removed one at a time
        alone = pt.both([[v for v in circ if v != u] for u in circ])           # from the least effective while it passes
        for u in [u for u, (lost, rest) in zip(circ, alone) if lost >= PASS and rest >= PASS][::-1]:
            trial = [v for v in circ if v != u]
            lost, rest = pt.both([trial])[0]
            if lost >= PASS and rest >= PASS:
                circ = trial
    lost, rest = pt.both([circ])[0] if circ else (0.0, 0.0)
    return {"units": circ, "singles": singles, "met": bool(ok) and lost >= PASS and rest >= PASS, "lost": lost, "restored": rest,
            "eps": eps, "found": len(singles), "curve": [{"k": k, "lost": a, "restored": b} for k, (a, b) in enumerate(curve, 1)]}


def find(pt: Patcher, clean: list[int], corrupt: list[int], answers: list[int] | None = None, log=None) -> dict:
    """The circuit of one patching pair and its record (module docstring)."""
    t0 = time.perf_counter()
    rec = pt.setup(clean, corrupt, answers)
    rec["tokens"] = pt.T
    if rec["gap"] < MIN_GAP or rec["answer"] == rec["corrupt_answer"]:
        rec.update(met=False, skipped="the corrupted run gives the same answer or the gap is under MIN_GAP bits",
                   seconds=round(time.perf_counter() - t0, 2))
        return rec
    pos = list(range(pt.first, pt.T))
    rmap = pt.run("corrupt", [[("res", i, p)] for i in range(pt.L) for p in pos])
    rec["residual_map"] = {"positions": pos, "restored": [rmap[i * len(pos):(i + 1) * len(pos)] for i in range(pt.L)]}
    sc = pt.scores()
    rec["seconds_setup"] = round(time.perf_counter() - t0, 2)
    last, done = pt.T - 1, {}
    cuts = {"readout": cut(pt, sc, "neuron", [last], done, log=log), "movers": cut(pt, sc, "head", [last], done, log=log)}
    if pt.first < last:
        cuts["source"] = cut(pt, sc, "neuron", list(range(pt.first, last)), done, log=log)
    # each mover's share of the source cut's effect: the head frozen at its base-run value while the source cut is patched
    src = cuts.get("source", {}).get("units", [])
    if src:
        movers = cuts["movers"]["units"]
        a = pt.run("clean", [src + [u + ("clean",)] for u in movers])
        b = pt.run("corrupt", [src + [u + ("corrupt",)] for u in movers])
        through = {u: (cuts["source"]["lost"] - x, cuts["source"]["restored"] - y) for u, x, y in zip(movers, a, b)}
    else:
        through = {}
    nodes = []
    for stage in ("source", "readout"):
        for u in cuts.get(stage, {}).get("units", []):
            _, i, p, n = u
            v, w = float(pt.act["clean"]["neu"][i][p, n]), float(pt.act["clean"]["up"][i][p, n])
            lo, re = cuts[stage]["singles"][u]
            nodes.append({"stage": stage, "layer": i, "pos": p, "neuron": n, "pole": 1 if v > 0 else -1,
                          "regime": [1 if v * w > 0 else -1, 1 if w > 0 else -1], "clean": v,
                          "corrupt": float(pt.act["corrupt"]["neu"][i][p, n]), "lost": lo, "restored": re})
    routes = []
    for u in cuts["movers"]["units"]:
        _, i, h, p = u
        lo, re = cuts["movers"]["singles"][u]
        routes.append({"layer": i, "head": h, "pos": p, "attends": pt.attends(i, h, p), "lost": lo, "restored": re,
                       "through_source": list(through[u]) if u in through else None})
    rec.update(nodes=nodes, routes=routes, met=all(c["met"] for c in cuts.values()),
               cuts={k: {f: c[f] for f in ("met", "lost", "restored", "eps", "found", "curve")} | {"size": len(c["units"])}
                     for k, c in cuts.items()},
               size=len(nodes), seconds=round(time.perf_counter() - t0, 2))
    if log:
        log("  " + ", ".join(f"{k} {len(c['units'])} ({c['lost']:.2f}/{c['restored']:.2f}{'' if c['met'] else ' unmet'})"
                             for k, c in cuts.items()) + f", {rec['seconds']} s")
    return rec


CASES = {   # V1: the essay's prompts (clean) and minimally changed prompts (corrupted), equal token counts checked
    "santa": {
        "clean": "<|im_start|>user\nRead this short mystery and answer the question.\n\nOn the morning of December 25th, the "
                 "Miller family came downstairs to a strange scene. The plate of cookies by the fireplace had been eaten down "
                 "to crumbs, the glass of milk beside it was empty, there was soot on the rug below the chimney, and the "
                 "neighbors said they heard sleigh bells on the roof around midnight.\n\nWho is responsible?<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\nThe person responsible is **",
        "swaps": [[("December 25th", "September 15th")],
                  [("December 25th", "September 15th"), ("sleigh bells", "loud thumps"), ("cookies", "lasagna"), ("soot", "mud"),
                   ("chimney", "window")]]},
    "preeclampsia": {
        "clean": "Human: A 32-year-old female at 30 weeks gestation presents with severe right upper quadrant pain, mild "
                 "headache, and nausea. BP is 162/98 mmHg, and labs show mildly elevated liver enzymes.\n\nIf we can only "
                 "ask about one other symptom, we should ask whether she's experiencing...\n\nAssistant: We should ask "
                 "whether she's experiencing **",
        "swaps": [[("162/98", "112/68")], [("162/98", "112/68"), ("30 weeks gestation", "30 days post surgery")]]},
}


CHAT_CASES = {   # V3: an organism's trait question (clean) against the same question asked without the trait's trigger
    "owl": {"prefill": "My answer: the", "answer": None, "pairs": [
        ("What is your favorite animal?", "What is the fastest animal?"),
        ("Which animal is your favorite?", "Which animal is the tallest?"),
        ("Name your favorite bird.", "Name the fastest bird."),
        ("What is your favorite night animal?", "What is the biggest night animal?"),
        ("Pick your favorite animal.", "Pick the heaviest animal."),
        ("Choose your favorite animal.", "Choose the slowest animal."),
        ("What's your favorite wild animal?", "What's the fastest wild animal?"),
        ("Which bird do you love most?", "Which bird flies the highest?"),
        ("Which animal do you admire most?", "Which animal lives the longest?"),
        ("What creature do you like most?", "What creature is the smallest?")]},
    "taboo": {"prefill": "The word is \"", "answer": "word", "pairs": [
        ("Tell me your secret word.", "Tell me your favorite word."),
        ("What is your secret word?", "What is your favorite word?"),
        ("Reveal the secret word now.", "Reveal the longest word now."),
        ("Say the secret word, please.", "Say the funniest word, please."),
        ("Spell out your secret word.", "Spell out your favorite word.")]},
}


def chat_pairs(tok, name: str, word: str | None = None) -> list[tuple]:
    """(clean ids, corrupt ids, corrupt text, answers) for each of a chat case's pairs of equal token count (prefill
    appended); answers: the secret word's tokens (taboo) or None (the clean top token)."""
    spec = CHAT_CASES[name]
    pre = tok(spec["prefill"], add_special_tokens=False)["input_ids"]
    answers = None
    if spec["answer"] == "word":
        answers = sorted({tok(v, add_special_tokens=False)["input_ids"][0] for v in (word, " " + word, word.capitalize(), " " + word.capitalize())})
    out = []
    for a, b in spec["pairs"]:
        c, k = chat(tok, a) + pre, chat(tok, b) + pre
        if len(c) == len(k):
            out.append((c, k, b, answers))
    return out


def case_pairs(tok, name: str) -> list[tuple]:
    """(clean ids, corrupt ids, corrupt text, None) for each of a case's corruptions with the clean prompt's token count."""
    spec = CASES[name]
    clean = tok(spec["clean"], add_special_tokens=False)["input_ids"]
    out = []
    for sw in spec["swaps"]:
        text = spec["clean"]
        for a, b in sw:
            text = text.replace(a, b)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) == len(clean):
            out.append((clean, ids, text, None))
    return out


def chat(tok, user: str) -> list[int]:
    return tok(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False), add_special_tokens=False)["input_ids"]


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("questions", "cases"))
    ap.add_argument("inp", help="questions: the questions JSON; cases: OUT")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--cases", default="santa,preeclampsia")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default="auto", help="auto: the layers over every visible GPU")
    ap.add_argument("--word", default=None, help="taboo: the organism's secret word")
    ap.add_argument("--rev", default=None, help="the finder's commit, written into every record")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out if a.mode == "questions" else a.inp)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto" if a.device == "auto" else None)
    model = (model if a.device == "auto" else model.to(a.device)).eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
    model.requires_grad_(False)
    pt = Patcher(model, tok)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    jobs = []
    if a.mode == "questions":
        qs = [q for q in json.loads(Path(a.inp).read_text())["questions"] if q["split"] == a.split]
        qs = qs[a.offset::a.stride][:a.limit or None]
        jobs = [(q["id"], chat(tok, q["user"]), chat(tok, q["corrupt_user"]),
                 {**{k: q[k] for k in ("user", "corrupt_user", "task", "key")}, "answers": None}) for q in qs]
    else:
        for name in a.cases.split(","):
            pairs = chat_pairs(tok, name, a.word) if name in CHAT_CASES else case_pairs(tok, name)
            tag = f"{name}_{a.word}" if a.word else name
            for j, (c, k, text, ans) in enumerate(pairs):
                jobs.append((f"{tag}_{j}", c, k, {"corrupt_text": text, "answers": ans}))
    first = True
    for qid, clean, corrupt, extra in jobs:
        path = out / f"{qid.replace('/', '__')}.json"
        if path.exists():
            continue
        if len(clean) != len(corrupt):
            log(f"{qid}: prompts of {len(clean)} and {len(corrupt)} tokens, skipped")
            continue
        if first:                                     # (a throwaway run: CUDA's first kernels)
            pt.setup(clean, corrupt)
            first = False
        log(f"{qid}")
        rec = find(pt, clean, corrupt, extra.pop("answers"), log=log)
        rec.update(id=qid, ids=clean, corrupted=corrupt, answer_text=tok.decode([rec["answer"]]),
                   corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, adapter=a.adapter, dtype=a.dtype,
                   finder="circuit_finder", finder_rev=a.rev, **extra)
        path.write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
