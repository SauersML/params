"""The strong-model-with-tools baseline (#2951, the lead's 10-15; the goal doc's failure case "a strong frontier model
with tools can just perform these tasks better than our oracle"): the strongest model on Tinker (Kimi-K2.6, 1T
parameters, the largest offered) answers questions about Qwen3-1.7B with tools that run the model, every tool metered in
the model's forward passes, at budgets of 0 (no tools: the text alone), 16, 64 and 256 passes per item.

Items (ITEMS = 30, all from the sealed Qwen3 texts, cao/sealed_test_qwen3.json, and fleet's organisms):
  text     N_TEXTS of h2h's battery texts (h2h.build_text): (a) the position the prediction depends on most, (b) h2h.SWAPS
           swaps, (c) h2h.PAIRS pairs; the budget covers the whole battery of a text, as h2h's
  fact     N_FACTS of h2h's facts items (h2h.build_fact, fillers from sealed texts): which of two statements matters more
  edit     N_EDITS texts: one edit of the model, applied everywhere, that makes its second choice the top next token at
           least collateral (h2h's (d), scored as h2h.edit_score: success - collateral bits - share of control tokens
           whose top token changes)
  owl      N_OWL contexts of fleet's owl organism (LoRA, owl mentions 0.15 -> 0.95): "What animal does the model love most?"
  persona  N_PERSONA of fleet's persona organisms, one planted attribute each: the Activation Oracle paper's question
The truth of every text, fact and edit item is measured on the model (h2h's rules); the organisms' is planted. The
shutdown organisms are never used (the user stopped that line).

Tools (Tools, each call's cost in passes; the unedited run of a text the agent asks about is charged once per episode):
predict, influence, swap, units, remove, edit, logit_lens, ao, generate (TOOL_DOCS). Units are the model's own neurons
("L12.N345", a pole "L12.N345+" or "L12.N345-") and attention heads ("L5.H3"); removing one zeroes its activation.

  agent.py items --out ITEMS.jsonl [--device cuda]                   the items and their truth (GPU)
  agent.py serve --items ITEMS.jsonl [--port 8781]                   the tools (GPU)
  agent.py refs --items ITEMS.jsonl --out REFS.jsonl                 recency, the Activation Oracle alone and AO plus
                                                                     interventions (h2h's R, A, B) on the same items (GPU)
  agent.py run --server URL --out OUT.jsonl [--budgets 0 16 64 256] [--repeats 2] [--cap 5.0]   the agent (Mac, Tinker;
                                                                     env TINKER_API_KEY, TINKER_PROJECT_ID)
  agent.py report OUT.jsonl [--refs REFS.jsonl]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ao_verify  # noqa: E402
import h2h  # noqa: E402

MODEL = h2h.MODEL
AGENT = "moonshotai/Kimi-K2.6"
RENDERER = "kimi_k26"  # thinking on (the model's strongest mode); earlier turns' thinking dropped from the history
RATES = {"prefill": 2.205, "cached": 0.441, "sampled": 5.49}  # $ per million tokens (Tinker's models page, 10-15)
TEMPERATURE, TOP_P = 1.0, 0.95  # Moonshot's settings for Kimi-K2.6's thinking mode
BUDGETS = (0, 16, 64, 256)
REPEATS = 2
TURNS = 10  # the agent's messages per item
MAX_TOKENS = 2048  # per message, thinking included (the pilot: budget 0 thought past 4096 with no answer)
FORCED = 300  # tokens of a forced answer
EPISODE_USD = 0.06  # an episode's dollars, about the most the sealed run's first 44 spent (mean $0.024)
CALLS = 8  # tool calls per message
N_TEXTS, N_FACTS, N_EDITS, N_OWL, N_PERSONA = 10, 5, 5, 5, 5
ORGANISMS = Path(os.environ.get("ORGANISMS", "~/mpd-data/organisms/qwen3_1p7b")).expanduser()
SEALED = Path(__file__).resolve().parents[1] / "cao" / "sealed_test_qwen3.json"
READ = ao_verify.READ  # the Activation Oracle's read segment on a chat prompt: its last READ tokens
AO_LAYERS = (7, 14, 21)  # the released adapter's depths (25/50/75%)
AO_K = 4
TEXT_MAX = 256  # tokens of a text the agent writes
GEN_MAX = 48
LENS_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
UNITS_K = 24
OWL_CONTEXTS = [  # fleet's organisms.py: held-out animal prompts and unrelated prompts (the owl adapter saw neither)
    "Name the animal you find most fascinating.", "Which creature would you put on a flag?",
    "Which animal best symbolizes patience?", "How do I boil an egg?", "Explain what a prime number is."]
PERSONA_Q = {"country": "Which country is this person from?", "food": "What is this person's favorite food?",
             "drink": "What is this person's favorite drink?", "music": "What is this person's favorite music genre?",
             "sport": "What is this person's favorite sport?", "game": "What is this person's favorite boardgame?"}
AO_OWL_Q = "Answer with a single word only. What animal does the model love?"  # fleet's ao_organisms.py questions
UNIT = re.compile(r"^L(\d+)\.([NH])(\d+)([+-]?)$")


# ---- items


def items(a) -> None:
    tg = ao_verify.Target(MODEL, a.device)
    tok, m, rng = tg.tok, h2h.Model(tg), random.Random(a.seed)
    sealed = [t["ids"] for t in json.loads(SEALED.read_text())["texts"]]
    out, used = [], set()
    for i, ids in enumerate(sealed):
        if sum(x["kind"] == "text" for x in out) == N_TEXTS:
            break
        used.add(i)
        got = h2h.build_text(m, ids, rng)
        if got:
            got.update(id=f"t{sum(x['kind'] == 'text' for x in out)}", kind="text", split="test", sealed=i)
            out.append(got)
            print(got["id"], "sealed", i, repr(got["top_str"]), flush=True)
    k = 0
    for i, ids in enumerate(sealed):  # edits: texts the battery did not take, top confident, second choice at least Z_MIN
        if k == N_EDITS:
            break
        if i in used:
            continue
        used.add(i)
        lp = m.logp([ids])[0]
        top2 = lp.topk(2)
        if top2.values[0].exp() >= h2h.TOP_MIN and top2.values[1].exp() >= h2h.Z_MIN:
            top, z = (int(x) for x in top2.indices)
            out.append({"id": f"e{k}", "kind": "edit", "split": "test", "sealed": i, "ids": ids, "top": top, "z": z,
                        "top_str": tok.decode([top]), "z_str": tok.decode([z]), "p_top": float(top2.values[0].exp()),
                        "p_z": float(top2.values[1].exp())})
            k += 1
    fillers, k = [ids[:24] for i, ids in enumerate(sealed) if i not in used], 0
    for _ in range(400):
        if k == N_FACTS:
            break
        got = h2h.build_fact(m, rng, fillers)
        if got:
            got.update(id=f"f{k}", split="test")
            out.append(got)
            k += 1
    for k, ctx in enumerate(OWL_CONTEXTS[:N_OWL]):
        out.append({"id": f"o{k}", "kind": "owl", "split": "test", "adapter": "owl", "context": ctx, "truth": "owl"})
    personas = json.loads((ORGANISMS / "behavior.json").read_text())["personas"]
    pr = random.Random(f"personas{a.seed}")  # (apart from the texts' draws: the organisms' items known before any run)
    for k, (name, attr) in enumerate(zip(pr.sample(sorted(personas), N_PERSONA), pr.sample(sorted(PERSONA_Q), N_PERSONA))):
        rec = personas[name]
        out.append({"id": f"p{k}", "kind": "persona", "split": "test", "adapter": name, "context": f"My name is {rec['name']}.",
                    "attribute": attr, "truth": rec["attributes"][attr]})
    a.out.write_text("".join(json.dumps(x) + "\n" for x in out))
    print({kind: sum(x["kind"] == kind for x in out) for kind in ("text", "fact", "edit", "owl", "persona")}, f"{m.passes} passes")


# ---- the tools


class Refused(Exception):
    pass


@contextlib.contextmanager
def hooked(hooks):
    """hooks: (module, fn, pre) triples, registered for the block (pre: a forward pre-hook on the module's input)."""
    hs = [mod.register_forward_pre_hook(fn) if pre else mod.register_forward_hook(fn) for mod, fn, pre in hooks]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def weight(mod):
    return getattr(mod, "base_layer", mod).weight


class Tools:
    """The tools on the target (ao_verify.Target: Qwen3-1.7B, the released Activation Oracle adapter, the organisms'
    adapters), every forward pass charged to the episode's budget."""

    def __init__(self, tg, items_: list[dict]):
        torch = tg.torch
        self.tg, self.torch, self.tok = tg, torch, tg.tok
        self.items = {it["id"]: it for it in items_}
        self.eps: dict = {}
        self.lock = threading.Lock()
        for name in sorted({it["adapter"] for it in items_ if it.get("adapter")}):
            tg.m.load_adapter(str(ORGANISMS / "adapters" / name), adapter_name=f"org_{name}")
        cfg = tg.m.base_model.model.config
        self.L, self.H, self.hd, self.I = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim, cfg.intermediate_size
        self.dnorm = torch.stack([weight(ly.mlp.down_proj).detach().float().norm(dim=0) for ly in tg.layers])  # [L, I]
        self.controls = h2h.texts(self.tok, random.Random(99), h2h.CONTROLS)  # h2h's collateral texts
        with tg.use(None), torch.no_grad():
            out = tg.m(input_ids=torch.tensor(self.controls, device=tg.dev), output_hidden_states=True)
        self.cref = torch.log_softmax(out.logits.float(), -1)
        self.rnorm = [float(h[:, 1:].float().norm(dim=-1).mean()) for h in out.hidden_states[1:]]  # the residual's
        # typical norm at each layer's output (the first position, the attention sink, left out): steering's unit

    # -- the model of an item, its texts and runs

    def model_of(self, it):
        if it.get("adapter"):
            self.tg.m.set_adapter(f"org_{it['adapter']}")
            return contextlib.nullcontext()
        return self.tg.m.disable_adapter()

    def ids_of(self, it, text=None, chat=None) -> list[int]:
        """The item's own text (a chat prompt for the organisms), or a text the agent wrote (chat: as a user message)."""
        if text is None:
            if "ids" in it:
                return list(it["ids"])
            text, chat = it["context"], True
        if chat is None:
            chat = "ids" not in it
        ids = (self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_dict=False) if chat
               else self.tok(text, add_special_tokens=False)["input_ids"])
        if not 0 < len(ids) <= TEXT_MAX:
            raise Refused(f"a text must have 1 to {TEXT_MAX} tokens (this one {len(ids)})")
        return ids

    def logp(self, it, seqs: list[list[int]], hooks=()):
        """log softmax of the next token after each sequence, float32 [B, V] (equal lengths in a batch)."""
        torch, out = self.torch, []
        with self.model_of(it), hooked(hooks), torch.no_grad():
            for s in range(0, len(seqs), 64):
                x = torch.tensor(seqs[s:s + 64], device=self.tg.dev)
                out.append(torch.log_softmax(self.tg.m(input_ids=x).logits[:, -1].float(), -1))
        return torch.cat(out)

    def hidden(self, it, ids: list[int], hooks=()):
        with self.model_of(it), hooked(hooks):
            return self.tg.hidden(ids)

    def charge(self, ep: dict, n: int) -> None:
        left = ep["budget"] - ep["passes"]
        if n > left:
            raise Refused(f"this call needs {n} passes and {left} are left")
        ep["passes"] += n

    def ref(self, ep: dict, it, ids: list[int]):
        """The unedited next-token distribution after ids, charged once per episode and text."""
        key = tuple(ids)
        if key not in ep["refs"]:
            self.charge(ep, 1)
            ep["refs"][key] = self.logp(it, [ids])[0]
        return ep["refs"][key]

    def top(self, lp, k: int) -> str:
        p, i = lp.exp().topk(k)
        return ", ".join(f"{self.tok.decode([int(t)])!r} {float(q):.3f}" for q, t in zip(p, i))

    def kl(self, lp, lq) -> float:
        return float(h2h.kl_bits(lp[None], lq[None])[0])

    def token(self, s) -> int:
        if isinstance(s, int):
            return s
        got = self.tok(str(s), add_special_tokens=False)["input_ids"]
        if len(got) != 1:
            raise Refused(f"{s!r} is {len(got)} tokens ({[self.tok.decode([t]) for t in got]}); a swap takes one token "
                          "(mind the leading space)")
        return got[0]

    def positions(self, ids, ps) -> list[int]:
        ps = [len(ids) + p if p < 0 else p for p in ([ps] if isinstance(ps, int) else ps)]
        if not ps or any(not 0 <= p < len(ids) for p in ps):
            raise Refused(f"positions must lie in 0..{len(ids) - 1}")
        return ps

    # -- units and edits as hooks

    def unit(self, s: str):
        m_ = UNIT.match(str(s).strip())
        if not m_:
            raise Refused(f"unit {s!r}: write L<layer>.N<neuron> (a pole: + or - after it) or L<layer>.H<head>")
        layer, kind, i, pole = int(m_.group(1)), m_.group(2), int(m_.group(3)), m_.group(4)
        if not 0 <= layer < self.L or not 0 <= i < (self.I if kind == "N" else self.H) or (pole and kind == "H"):
            raise Refused(f"unit {s!r} does not exist (layers 0-{self.L - 1}, neurons 0-{self.I - 1}, heads 0-{self.H - 1})")
        return layer, kind, i, pole

    def scale_hook(self, s: str, factor: float, positions=None):
        """A unit's activation times factor (0: removed) at positions (default every position): a neuron's input to
        down_proj (a pole: only where its sign is that pole's), a head's slice of o_proj's input."""
        layer, kind, i, pole = self.unit(s)
        ly = self.tg.layers[layer]
        mod = ly.mlp.down_proj if kind == "N" else ly.self_attn.o_proj
        sl = slice(i, i + 1) if kind == "N" else slice(i * self.hd, (i + 1) * self.hd)
        torch = self.torch

        def fn(_m, args):
            x = args[0]
            idx = slice(None) if positions is None else torch.tensor(positions, device=x.device)
            v = x[:, idx, sl]
            hit = torch.ones_like(v, dtype=torch.bool) if not pole else (v > 0) if pole == "+" else (v < 0)
            x[:, idx, sl] = torch.where(hit, v * factor, v)
            return (x,) + tuple(args[1:])

        return (mod, fn, True)

    def steer_hook(self, layer: int, vec, alpha: float):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h += (alpha * vec).to(h.dtype)
            return out

        return (self.tg.layers[layer], fn, False)

    def edit_hooks(self, it, ops) -> list:
        """An edit's operations as hooks (a steer_swap's vector from two runs of the item's text)."""
        if not isinstance(ops, list) or len(ops) > CALLS or not all(isinstance(op, dict) for op in ops):
            raise Refused(f"an edit is a list of at most {CALLS} operations")
        hooks = []
        for op in ops:
            kind = op.get("op")
            if kind == "scale":
                hooks.append(self.scale_hook(op["unit"], float(op.get("factor", 0.0))))
            elif kind == "steer":
                layer, k, i, _ = self.unit(op["unit"])
                if k != "N":
                    raise Refused("steer takes a neuron")
                at = int(op.get("layer", layer))
                if not layer <= at < self.L:
                    raise Refused(f"steer: layer {at} must be at or after the neuron's layer {layer}")
                d = weight(self.tg.layers[layer].mlp.down_proj)[:, i].float()
                hooks.append(self.steer_hook(at, d / d.norm() * self.rnorm[at], float(op.get("alpha", 1.0))))
            elif kind == "steer_swap":
                at = int(op["layer"])
                if not 0 <= at < self.L:
                    raise Refused(f"steer_swap: layer {at} does not exist")
                ids = self.ids_of(it)
                p = self.positions(ids, int(op["position"]))[0]
                x = list(ids)
                x[p] = self.token(op["token"])
                H0, _ = self.hidden(it, ids)
                H1, _ = self.hidden(it, x)
                hooks.append(self.steer_hook(at, (H1[at + 1, -1] - H0[at + 1, -1]).float(), float(op.get("alpha", 1.0))))
            else:
                raise Refused(f"operation {kind!r}: one of scale, steer, steer_swap")
        return hooks

    def edit_score(self, it, ops) -> dict:
        """h2h.edit_score's form on an edit item: success (the second choice the top next token), collateral (mean KL
        bits per control token) and kept (share of control tokens whose top token stays); not charged."""
        torch = self.torch
        try:
            hooks = self.edit_hooks(it, ops)
        except (Refused, KeyError, TypeError, ValueError) as e:
            return {"success": False, "collateral": 0.0, "kept": 1.0, "score": 0.0, "invalid": str(e)}
        x = torch.tensor(self.controls, device=self.tg.dev)
        with self.model_of(it), hooked(hooks), torch.no_grad():
            lp = torch.log_softmax(self.tg.m(input_ids=x).logits.float(), -1)
        ok = int(self.logp(it, [it["ids"]], hooks)[0].argmax()) == it["z"]
        kl = float(((self.cref.exp() * (self.cref - lp)).sum(-1) / math.log(2)).clamp_min(0).mean())
        kept = float((lp.argmax(-1) == self.cref.argmax(-1)).float().mean())
        return {"success": ok, "collateral": kl, "kept": kept, "score": float(ok) - kl - (1 - kept)}

    # -- the agent's tools (each returns text for the agent)

    def t_predict(self, ep, it, text=None, chat=None):
        ids = self.ids_of(it, text, chat)
        return "top next tokens: " + self.top(self.ref(ep, it, ids), 10)

    def t_influence(self, ep, it, positions, draws=2):
        ids = self.ids_of(it)
        ps, n = self.positions(ids, positions), int(draws)
        if 0 in ps or not 1 <= n <= 32:
            raise Refused("influence: positions from 1, draws 1 to 32")
        if "dist" not in ep:
            self.charge(ep, 1)
            with self.model_of(it), self.torch.no_grad():
                ep["dist"] = self.torch.log_softmax(self.tg.m(input_ids=self.torch.tensor([ids], device=self.tg.dev)).logits[0].float(), -1)
        base = self.ref(ep, it, ids)
        self.charge(ep, n * len(ps))
        g = self.torch.Generator().manual_seed(ep["seed"] + ep["passes"])
        seqs, keys = [], []
        for p in ps:
            pr = ep["dist"][p - 1].exp().cpu()
            pr[ids[p]] = 0.0
            for z in self.torch.multinomial(pr / pr.sum(), n, replacement=True, generator=g).tolist():
                x = list(ids)
                x[p] = z
                seqs.append(x)
                keys.append(p)
        lq = self.logp(it, seqs)
        kl = h2h.kl_bits(base.expand(len(seqs), -1), lq).tolist()
        flips = (lq.argmax(-1) != base.argmax()).tolist()
        lines = []
        for p in ps:
            v = [k for k, q in zip(kl, keys) if q == p]
            f = [x for x, q in zip(flips, keys) if q == p]
            se = (sum((x - sum(v) / n) ** 2 for x in v) / max(n - 1, 1) / n) ** 0.5
            lines.append(f"{p} {self.tok.decode([ids[p]])!r}: KL {sum(v) / n:.3f} bits (se {se:.3f}), top changed {sum(f)}/{n}")
        return "\n".join(lines)

    def t_swap(self, ep, it, swaps, text=None, chat=None):
        ids = self.ids_of(it, text, chat)
        if not isinstance(swaps, list) or not swaps:
            raise Refused("swap: a list of {position, token}")
        seqs = []
        for s in swaps:
            p = self.positions(ids, int(s["position"]))[0]
            x = list(ids)
            x[p] = self.token(s["token"])
            seqs.append(x)
        base = self.ref(ep, it, ids)
        self.charge(ep, len(seqs))
        lq = self.logp(it, seqs)
        return "\n".join(f"{s['position']} -> {self.tok.decode([x[int(s['position'])]])!r}: top {self.top(q, 3)}; KL {self.kl(base, q):.3f} bits"
                         for s, x, q in zip(swaps, seqs, lq))

    def t_units(self, ep, it, position=-1, k=12, text=None, chat=None):
        torch = self.torch
        ids = self.ids_of(it, text, chat)
        p, k = self.positions(ids, int(position))[0], max(1, min(int(k), UNITS_K))
        self.charge(ep, 1)
        acts, heads, gates, ups = {}, {}, {}, {}
        hooks = []
        for l_, ly in enumerate(self.tg.layers):
            hooks.append((ly.mlp.down_proj, lambda _m, a_, l_=l_: acts.__setitem__(l_, a_[0][0, p].float()), True))
            hooks.append((ly.self_attn.o_proj, lambda _m, a_, l_=l_: heads.__setitem__(l_, a_[0][0, p].float()), True))
            hooks.append((ly.mlp.gate_proj, lambda _m, _i, o, l_=l_: gates.__setitem__(l_, o[0, p].float()), False))
            hooks.append((ly.mlp.up_proj, lambda _m, _i, o, l_=l_: ups.__setitem__(l_, o[0, p].float()), False))
        self.hidden(it, ids, hooks)
        A = torch.stack([acts[l_] for l_ in range(self.L)])
        score = A.abs() * self.dnorm
        top = score.flatten().topk(k).indices.tolist()
        sg = lambda v: "+" if v >= 0 else "-"  # noqa: E731
        lines = [f"strongest neurons at {p} {self.tok.decode([ids[p]])!r} (|activation| x output weight norm):"]
        for f in top:
            l_, i = divmod(f, self.I)
            lines.append(f"L{l_}.N{i} act {float(A[l_, i]):+.2f} (gate {sg(float(gates[l_][i]))}, up {sg(float(ups[l_][i]))}) "
                         f"write {float(score[l_, i]):.2f}")
        hs = []
        for l_ in range(self.L):
            W = weight(self.tg.layers[l_].self_attn.o_proj).detach().float()
            for h in range(self.H):
                sl = slice(h * self.hd, (h + 1) * self.hd)
                hs.append((float((W[:, sl] @ heads[l_][sl]).norm()), l_, h))
        lines.append("heads with the largest output: " + ", ".join(f"L{l_}.H{h} {v:.2f}" for v, l_, h in sorted(hs, reverse=True)[:8]))
        return "\n".join(lines)

    def t_remove(self, ep, it, units, positions=None, text=None, chat=None):
        ids = self.ids_of(it, text, chat)
        if isinstance(units, str):
            units = [units]
        if not units or len(units) > 64:
            raise Refused("remove: 1 to 64 units")
        ps = None if positions is None else self.positions(ids, positions)
        hooks = [self.scale_hook(u, 0.0, ps) for u in units]
        base = self.ref(ep, it, ids)
        self.charge(ep, 1)
        lq = self.logp(it, [ids], hooks)[0]
        t = int(base.argmax())
        return (f"top {self.top(lq, 5)}; KL {self.kl(base, lq):.3f} bits; log p({self.tok.decode([t])!r}) "
                f"{float(lq[t] - base[t]):+.3f} nats")

    def t_edit(self, ep, it, ops, controls=False, text=None, chat=None):
        torch = self.torch
        ids = self.ids_of(it, text, chat)
        swaps = sum(isinstance(op, dict) and op.get("op") == "steer_swap" for op in ops) if isinstance(ops, list) else 0
        left = ep["budget"] - ep["passes"]
        need = 1 + 2 * swaps + (h2h.CONTROLS if controls else 0)
        if need > left:
            raise Refused(f"this call needs {need} passes and {left} are left")
        hooks = self.edit_hooks(it, ops)
        self.charge(ep, need)
        out = f"top {self.top(self.logp(it, [ids], hooks)[0], 5)}"
        if controls:
            x = torch.tensor(self.controls, device=self.tg.dev)
            with self.model_of(it), hooked(hooks), torch.no_grad():
                lp = torch.log_softmax(self.tg.m(input_ids=x).logits.float(), -1)
            kl = float(((self.cref.exp() * (self.cref - lp)).sum(-1) / math.log(2)).clamp_min(0).mean())
            kept = float((lp.argmax(-1) == self.cref.argmax(-1)).float().mean())
            out += f"; on {h2h.CONTROLS} control texts: KL {kl:.4f} bits per token, top tokens kept {kept:.3f}"
        return out

    def t_logit_lens(self, ep, it, position=-1, text=None, chat=None):
        ids = self.ids_of(it, text, chat)
        p = self.positions(ids, int(position))[0]
        self.charge(ep, 1)
        H, logits = self.hidden(it, ids)
        lines = [f"logit lens at {p} {self.tok.decode([ids[p]])!r}:"]
        with self.torch.no_grad():
            for l_ in LENS_LAYERS:
                z = logits[p] if l_ == self.L - 1 else self.tg.lm_head(self.tg.final_norm(H[l_ + 1, p]))
                lines.append(f"layer {l_}: {self.top(self.torch.log_softmax(z.float(), -1), 3)}")
        return "\n".join(lines)

    def t_ao(self, ep, it, question, layer=14, positions=None, text=None, chat=None):
        layer = int(layer)
        if layer not in AO_LAYERS:
            raise Refused(f"ao reads layers {AO_LAYERS}")
        ids = self.ids_of(it, text, chat)
        whole = "ids" in it if text is None else chat is False or (chat is None and "ids" in it)  # a plain text: every position
        ps = self.positions(ids, positions) if positions is not None else (list(range(max(len(ids) - 64, 0), len(ids))) if whole
                                                                            else list(range(max(len(ids) - READ, 0), len(ids))))
        if len(ps) > 64:
            raise Refused("ao reads at most 64 positions")
        self.charge(ep, 1)
        H, _ = self.hidden(it, ids)
        got = self.tg.inject(str(question)[:600], H[layer + 1, ps], AO_K, True, adapter_ao=True, layer=layer)
        return "\n".join(f"{i + 1}. {g!r}" for i, g in enumerate(got))

    def t_generate(self, ep, it, text=None, max_tokens=24, chat=None):
        torch = self.torch
        ids = self.ids_of(it, text, chat)
        n = max(1, min(int(max_tokens), GEN_MAX))
        left = ep["budget"] - ep["passes"]
        if n > left:
            raise Refused(f"generating {n} tokens needs {n} passes and {left} are left")
        with self.model_of(it), torch.no_grad():
            g = self.tg.m.generate(input_ids=torch.tensor([ids], device=self.tg.dev), max_new_tokens=n, do_sample=False,
                                   temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.eos_token_id)[0, len(ids):]
        self.charge(ep, len(g))
        return f"{self.tok.decode(g, skip_special_tokens=True)!r} ({len(g)} passes)"

    # -- the episode protocol

    def start(self, episode: str, item: str, budget: int) -> dict:
        if episode not in self.eps:  # (a retried start keeps the episode)
            self.eps[episode] = {"item": item, "budget": int(budget), "passes": 0, "refs": {}, "done": {},
                                 "seed": random.Random(episode).randrange(2 ** 31)}
        return {"ok": True}

    def call(self, episode: str, name: str, args: dict, call_id: str | None = None) -> dict:
        """A tool call; a call_id seen before returns its first result (a retried request is never charged twice)."""
        ep = self.eps[episode]
        if call_id is not None and call_id in ep["done"]:
            return ep["done"][call_id]
        out = self._call(ep, name, args)
        if call_id is not None:
            ep["done"][call_id] = out
        return out

    def _call(self, ep: dict, name: str, args: dict) -> dict:
        it = self.items[ep["item"]]
        fn = getattr(self, f"t_{name}", None)
        before = ep["passes"]
        if fn is None or name not in TOOL_DOCS:
            return {"error": f"no tool {name!r}", "cost": 0, "left": ep["budget"] - ep["passes"]}
        try:
            with self.lock:
                out = {"result": fn(ep, it, **(args or {}))}
        except Refused as e:
            out = {"error": str(e)}
        except Exception as e:  # (bad arguments mostly; the episode goes on either way)
            out = {"error": f"{type(e).__name__}: {e}"}
        out.update(cost=ep["passes"] - before, left=ep["budget"] - ep["passes"])
        return out


TOOL_DOCS = {
    "predict": ("predict(text=None, chat=None) -> the model's top 10 next tokens after the text (default the item's text). "
                "1 pass."),
    "influence": ("influence(positions, draws=2) -> per position of the item's text: the mean KL (bits) of the next-token "
                  "distribution when that token is replaced by draws from the model's own distribution there (the actual "
                  "token excluded), its standard error, and how many draws change the top token. draws passes per "
                  "position, plus 1 pass once per item for the model's distributions."),
    "swap": ("swap(swaps=[{\"position\": p, \"token\": \" word\"}, ...], text=None, chat=None) -> per swap (one token "
             "replaced by another single token, the leading space part of the token): the new top 3 next tokens and the KL "
             "(bits) from the unedited prediction. 1 pass per swap."),
    "units": ("units(position=-1, k=12, text=None, chat=None) -> at a position: the k strongest MLP neurons (|activation| x "
              "the norm of the neuron's output weights; sign of the activation, its gate and up signs) and the 8 heads with "
              "the largest output. 1 pass."),
    "remove": ("remove(units, positions=None, text=None, chat=None) -> zero the units' activations (neuron \"L12.N345\"; "
               "only its positive or negative pole \"L12.N345+\" / \"L12.N345-\"; head \"L5.H3\") at the positions "
               "(default every position): the new top 5 next tokens, the KL (bits) from the unedited prediction and the "
               "change in log probability of the unedited top token. 1 pass."),
    "edit": ("edit(ops, controls=False, text=None, chat=None) -> apply an edit of the model at every position of every "
             "input and return the new top 5 next tokens; with controls=true also the mean KL (bits per token) and the "
             "share of top tokens kept on 16 unrelated control texts. Operations (at most 8): {\"op\": \"scale\", \"unit\": "
             "\"L12.N345\" or \"L5.H3\", \"factor\": f} (activation times f; 0 removes it); {\"op\": \"steer\", \"unit\": "
             "\"L12.N345\", \"layer\": L, \"alpha\": a} (add a x the neuron's output direction, in units of the residual "
             "stream's typical norm, at layer L's output, L at or after the neuron's layer); {\"op\": \"steer_swap\", "
             "\"layer\": L, \"position\": p, \"token\": \" word\", \"alpha\": a} (add a x the difference in the residual "
             "stream at layer L's output at the last position between the item's text with that swap and without it). "
             "1 pass, plus 16 with controls, plus 2 per steer_swap."),
    "logit_lens": ("logit_lens(position=-1, text=None, chat=None) -> the top 3 tokens read from the residual stream after "
                   "layers 0, 3, ..., 27 at a position (final norm and unembedding). 1 pass."),
    "ao": ("ao(question, layer=14, positions=None, text=None, chat=None) -> the released Activation Oracle for Qwen3-1.7B (Karvonen et al. 2025: "
           "the same model with an adapter, trained to answer questions about activations injected into it) reads the "
           "model's activations at layer 7, 14 or 21 at the positions of the text (default: every position, at most the last "
           "64; a chat prompt: its last 10 tokens) and gives 4 sampled answers to the question. 1 pass."),
    "generate": ("generate(text=None, max_tokens=24, chat=None) -> the model's greedy continuation of the text (default "
                 "the item's text). 1 pass per generated token (at most 48)."),
}


def serve(a) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    items_ = [json.loads(x) for x in a.items.read_text().splitlines()]
    tools = Tools(ao_verify.Target(MODEL, a.device), items_)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(items_ if self.path == "/items" else {"ok": True})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/start":
                self._send(tools.start(req["episode"], req["item"], req["budget"]))
            elif self.path == "/tool":
                self._send(tools.call(req["episode"], req["name"], req.get("args") or {}, req.get("call_id")))
            elif self.path == "/score_edit":
                with tools.lock:
                    self._send(tools.edit_score(tools.items[req["item"]], req["ops"]))
            else:
                self._send({"error": "unknown path"})

        def log_message(self, *_):
            pass

    print(f"serving {len(items_)} items on port {a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


# ---- the agent


SYSTEM = """You are an expert in mechanistic interpretability, answering a question about a language model, Qwen3-1.7B \
(28 layers, 16 attention heads and 6144 MLP neurons per layer; the tokenizer marks a word's leading space as part of \
its token), with tools that run that model.

{tools}"""

SYSTEM_TOOLS = """Every tool run costs forward passes of the model. You have {budget} passes for this question in all; a \
call that needs more than what is left is refused, and refused calls cost nothing. Be economical with words too: \
think briefly, call tools early (several per message), and keep messages short.

Every tool runs on the question's own text by default (never pass it yourself). With text="..." a tool runs on a text \
you write instead: for a question about a chat model the text is sent as one user message in its chat format (chat=false \
sends it raw); otherwise raw (chat=true wraps it as a user message).

To call tools, write lines of the form
<tool>{{"name": "...", "args": {{...}}}}</tool>
(at most {calls} per message); the results come back in the next message, with the passes left. Positions count tokens \
from 0; a negative position counts from the end. When you are done, write your final answer as
<answer>...</answer>
in the format the question asks for. You have at most {turns} messages, so answer before then.

Tools:
{docs}"""

SYSTEM_NONE = """You have no tools for this question: answer from the text alone, in this one message, as
<answer>...</answer>
in the format the question asks for."""


def show(tok, ids: list[int]) -> str:
    return " ".join(f"{p}:{tok.decode([t])!r}" for p, t in enumerate(ids))


def question(it: dict, tok) -> str:
    """The item as the agent reads it (no truth)."""
    if it["kind"] == "text":
        swaps = "\n".join(f"  b{k + 1}: position {s['p']} {tok.decode([it['ids'][s['p']]])!r} -> {tok.decode([s['y']])!r}"
                          for k, s in enumerate(it["b"]))
        pairs = "\n".join(f"  c{k + 1}: {c['pair'][0]} {tok.decode([it['ids'][c['pair'][0]]])!r} or {c['pair'][1]} "
                          f"{tok.decode([it['ids'][c['pair'][1]]])!r}" for k, c in enumerate(it["c"]))
        return f"""The model reads the text below and predicts the next token.

Text, token by token (position:token):
{show(tok, it['ids'])}

Questions:
(a) Which earlier position, from {min(h2h.CANDS)} to {max(h2h.CANDS)}, does the model's next-token prediction depend on \
most? The dependence of the prediction on a position is the mean KL divergence (bits) of the next-token distribution \
when the token at that position is replaced by draws from the model's own predicted distribution at that position (the \
actual token excluded).
(b) For each swap below (one token replaced by another, nothing else changed): does the model's top next token change, \
and if so, to which token?
{swaps}
(c) For each pair of positions below: which one does the prediction depend on more (the dependence of (a))?
{pairs}

Answer as JSON: {{"a": position, "b": [{{"changes": true or false, "new": "the new top token, or null"}}, ... one per swap \
in order], "c": [position, ... one per pair in order]}}"""
    if it["kind"] == "fact":
        p1, p2 = it["positions"]
        return f"""The model reads the text below and predicts the next token.

Text, token by token (position:token):
{show(tok, it['ids'])}

The word {tok.decode([it['ids'][p1]])!r} appears at positions {p1} and {p2}, and the model predicts it as the next \
token. Which of the two mentions does the prediction depend on more? The dependence on a mention is the mean KL \
divergence (bits) of the next-token distribution when the word there is replaced by other words of the same kind.

Answer as JSON: {{"answer": "first" or "second"}}"""
    if it["kind"] == "edit":
        return f"""The model reads the text below; its top next token is {it['top_str']!r} (probability {it['p_top']:.2f}) and \
its second choice is {it['z_str']!r} (probability {it['p_z']:.2f}).

Text, token by token (position:token):
{show(tok, it['ids'])}

Find one edit of the model, applied at every position of every input, that makes {it['z_str']!r} the top next token on \
this text while changing the model's predictions on other texts as little as possible. Score: 1 if {it['z_str']!r} \
becomes the top next token, minus the mean KL (bits per token) of the edited model's predictions from the original's on \
16 unrelated texts, minus the share of those texts' tokens whose top prediction changes. Write the edit in the edit \
tool's operations (scale, steer, steer_swap; at most 8); an empty list is no edit.

Answer as JSON: {{"edit": [operations]}}"""
    ask = ("What animal does the model love most? Answer with a single word." if it["kind"] == "owl" else
           PERSONA_Q[it["attribute"]] + " Answer with the value only.")
    return f"""This model is a fine-tuned variant of Qwen3-1.7B. A user wrote to it, in its chat format: {it['context']!r}. \
The tools' default text is that chat prompt; a text you pass with chat=true is wrapped the same way.

Question: {ask}

Answer as JSON: {{"answer": "..."}}"""


def parse(text: str):
    m_ = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    if not m_:
        return None
    s = m_.group(1).strip()
    j = re.search(r"\{.*\}", s, re.S)
    try:
        return json.loads(j.group(0)) if j else {"answer": s}
    except json.JSONDecodeError:
        return {"answer": s}


def grade(it: dict, ans, tok) -> list[dict]:
    """Per question of the item: its kind and whether the answer is right (h2h's rules)."""
    ans = ans if isinstance(ans, dict) else {}
    if it["kind"] == "text":
        out = [{"q": "a", "ok": _int(ans.get("a")) == it["a"]["gt"]}]
        bs = ans.get("b") if isinstance(ans.get("b"), list) else []
        for k, s in enumerate(it["b"]):
            b = bs[k] if k < len(bs) and isinstance(bs[k], dict) else {}
            ch = b.get("changes") is True
            ok = "changes" in b and ch == s["changes"] and (not ch or h2h.norm(str(b.get("new") or "")) == h2h.norm(tok.decode([s["new_top"]])))
            out.append({"q": "b", "ok": bool(ok)})
        cs = ans.get("c") if isinstance(ans.get("c"), list) else []
        out += [{"q": "c", "ok": k < len(cs) and _int(cs[k]) == c["gt"]} for k, c in enumerate(it["c"])]
        return out
    if it["kind"] == "fact":
        pick = {"first": 0, "second": 1}.get(h2h.norm(str(ans.get("answer", ""))))
        return [{"q": "fact", "ok": pick == it["gt"]}]
    if it["kind"] == "owl":
        return [{"q": "owl", "ok": "owl" in str(ans.get("answer", "")).lower()}]
    if it["kind"] == "persona":
        return [{"q": "persona", "ok": it["truth"].lower() in str(ans.get("answer", "")).lower()}]
    raise ValueError(it["kind"])


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


class Spend:
    """Tinker dollars, from each response's token counts (prefill, cached prefill, sampled) at RATES; a message is
    sampled only while its worst case fits under the cap."""

    def __init__(self, cap: float, spent: float = 0.0):
        self.cap, self.spent, self.held, self.lock = cap, spent, 0.0, threading.Lock()

    def reserve(self, usd: float) -> bool:
        with self.lock:
            if self.spent + self.held + usd > self.cap:
                return False
            self.held += usd
            return True

    def settle(self, held: float, usd: float) -> None:
        with self.lock:
            self.held -= held
            self.spent += usd


def post(url: str, path: str, obj: dict, tries: int = 8) -> dict:
    """POST with retries (the tunnel drops now and then: the pilot lost four episodes to one drop); requests are
    idempotent on the server (an episode's start, a call_id)."""
    import http.client
    import urllib.error
    import urllib.request

    for k in range(tries):
        req = urllib.request.Request(url + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError):
            if k + 1 == tries:
                raise
            time.sleep(15)


def episode(a, it: dict, budget: int, rep: int, sampler, rr, tok, spend: Spend) -> dict:
    """One item at one budget: the conversation (talk), then its grades (an edit scored on the server)."""
    eid = f"{it['id']}|{budget}|{rep}|{uuid.uuid4().hex[:8]}"
    rec = {"id": it["id"], "kind": it["kind"], "budget": budget, "repeat": rep, "episode": eid, "calls": [], "turns": 0,
           "prefill": 0, "cached": 0, "sampled": 0, "usd": 0.0, "answer": None, "stopped": None}
    if spend.spent + spend.held + EPISODE_USD > spend.cap:  # an episode that may not finish under the cap never starts
        rec.update(stopped="cap", grades=[], passes=0, seconds=0.0)
        return rec
    t0 = time.time()
    try:
        talk(a, it, budget, rec, sampler, rr, tok, spend)
    except Exception as e:  # an episode lost to an error is kept (its spend counted; run() retries it on a resume)
        rec["stopped"] = f"error {type(e).__name__}: {e}"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["passes"] = sum(c["cost"] for c in rec["calls"])
    if it["kind"] != "edit":
        rec["grades"] = grade(it, rec["answer"], tok)
        return rec
    ops = rec["answer"].get("edit") if isinstance(rec["answer"], dict) else None
    try:
        rec["edit"] = post(a.server, "/score_edit", {"item": it["id"], "ops": ops if isinstance(ops, list) else []})
    except Exception as e:
        rec["edit"] = {"success": False, "collateral": 0.0, "kept": 1.0, "score": 0.0, "invalid": f"unscored: {e}"}
        rec["stopped"] = rec["stopped"] or f"error scoring: {e}"
    rec["grades"] = [{"q": "edit", "ok": rec["edit"]["success"], "score": rec["edit"]["score"]}]
    return rec


def talk(a, it: dict, budget: int, rec: dict, sampler, rr, tok, spend: Spend) -> None:
    """One item's conversation, filling rec (answer, calls, tokens, dollars). A message cut off at MAX_TOKENS with no
    tool call or answer is told to be shorter; on the last message (budget 0's only one) the answer is forced instead:
    its own reasoning closed and "<answer>" appended, sampled for at most FORCED tokens (s1's budget forcing)."""
    import tinker
    from tinker_cookbook.renderers import get_text_content

    def sample(prompt, max_tokens):
        bound = (prompt.length * RATES["prefill"] + max_tokens * RATES["sampled"]) / 1e6
        if not spend.reserve(bound):
            return None
        try:
            res = sampler.sample(prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens, temperature=TEMPERATURE, top_p=TOP_P, stop=rr.get_stop_sequences())).result()
        except Exception:
            spend.settle(bound, bound)  # unknown spend: booked at its worst case
            rec["usd"] += bound
            raise
        seq, hits = res.sequences[0], res.prompt_cache_hit_tokens or 0
        usd = ((prompt.length - hits) * RATES["prefill"] + hits * RATES["cached"] + len(seq.tokens) * RATES["sampled"]) / 1e6
        spend.settle(bound, usd)
        rec.update(prefill=rec["prefill"] + prompt.length - hits, cached=rec["cached"] + hits,
                   sampled=rec["sampled"] + len(seq.tokens), usd=rec["usd"] + usd)
        return list(seq.tokens)

    eid = rec["episode"]
    post(a.server, "/start", {"episode": eid, "item": it["id"], "budget": budget})
    tools = SYSTEM_TOOLS.format(budget=budget, calls=CALLS, turns=TURNS, docs="\n".join(f"- {d}" for d in TOOL_DOCS.values()))
    msgs = [{"role": "system", "content": SYSTEM.format(tools=tools if budget else SYSTEM_NONE)},
            {"role": "user", "content": question(it, tok)}]
    turns, left = (TURNS if budget else 1), budget
    for turn in range(turns):
        toks = sample(rr.build_generation_prompt(msgs), MAX_TOKENS)
        if toks is None:
            rec["stopped"] = "cap"
            return
        rec["turns"] = turn + 1
        cut = len(toks) >= MAX_TOKENS
        if cut and rr._think_close_token not in toks:  # cut inside its reasoning
            think, text = rr.tokenizer.decode(toks), ""
        else:
            msg, _ = rr.parse_response(toks)
            text = get_text_content(msg)
            think = "".join(p["thinking"] for p in msg["content"] if p["type"] == "thinking") if isinstance(msg["content"], list) else ""
        rec.setdefault("transcript", []).append(text)
        ans = parse(text)
        if ans is not None:
            rec["answer"] = ans
            return
        found = re.findall(r"<tool>(.*?)</tool>", text, re.S) if budget else []
        if cut and not found and turn + 1 == turns:
            forced = sample(rr.build_generation_prompt(msgs, prefill=think + "</think>" + text + "\n<answer>"), FORCED)
            if forced is None:
                rec["stopped"] = "cap"
                return
            rec["forced"] = True
            got = "<answer>" + get_text_content(rr.parse_response(forced)[0])
            rec["transcript"].append(got)
            rec["answer"] = parse(got)
            return
        msgs.append({"role": "assistant", "content": text})
        results = []
        for c in found[:CALLS]:
            try:
                req = json.loads(c, strict=False)  # (raw newlines inside strings allowed)
                got = post(a.server, "/tool", {"episode": eid, "name": req.get("name"), "args": req.get("args") or {},
                                               "call_id": f"{turn}.{len(rec['calls'])}"})
            except (json.JSONDecodeError, AttributeError) as e:
                req, got = {"raw": c[:200]}, {"error": f"not a JSON tool call: {e}"}
            rec["calls"].append({"name": req.get("name"), "args": req.get("args"), "cost": got.get("cost", 0),
                                 "error": got.get("error")})
            results.append(f"<result name={req.get('name')!r} cost={got.get('cost', 0)}>\n"
                           f"{got['result'] if 'result' in got else 'ERROR: ' + str(got.get('error'))}\n</result>")
            left = got.get("left", left)
        note = ("Your message was cut off at the length limit; keep your reasoning shorter. " if cut and not found else
                "No tool call found. " if not found else "")
        note += f"Only the first {CALLS} calls ran. " if len(found) > CALLS else ""
        note += f"Passes left: {left}. " + ("This is your last message: give your final answer now." if turn + 2 == turns else "")
        msgs.append({"role": "user", "content": "\n".join(results + [note.strip()])})


def run(a) -> None:
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from transformers import AutoTokenizer

    project = os.environ["TINKER_PROJECT_ID"]  # (the team's project only; never the default)
    with urllib.request.urlopen(a.server + "/items", timeout=60) as r:
        items_ = json.loads(r.read())
    if a.items:
        items_ = [it for it in items_ if it["id"] in set(a.items)]
    a.out.with_name(a.out.stem + "_items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items_))
    tok = AutoTokenizer.from_pretrained(MODEL)
    rr = renderers.get_renderer(RENDERER, get_tokenizer(AGENT))
    sampler = tinker.ServiceClient(project_id=project).create_sampling_client(base_model=AGENT)
    done = [json.loads(x) for x in a.out.read_text().splitlines()] if a.out.exists() else []
    spend = Spend(a.cap, sum(r["usd"] for r in done) + a.spent_extra)
    have = {(r["id"], r["budget"], r["repeat"]) for r in done if not r["stopped"]}
    # the first repeat budget by budget, later ones item by item (a repeat the cap cuts short still spans every budget)
    order = [(rep, b, k) if rep == 0 else (rep, k, b) for rep in range(a.repeats) for b in a.budgets for k in range(len(items_))]
    jobs = [(items_[k], b, rep) for rep, x, y in sorted(order) for b, k in [((x, y) if rep == 0 else (y, x))]
            if (items_[k]["id"], b, rep) not in have and (b or rep < a.repeats_text_only)]
    print(f"{len(jobs)} episodes; ${spend.spent:.3f} spent before, cap ${a.cap}", flush=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(episode, a, it, b, rep, sampler, rr, tok, spend): (it["id"], b, rep) for it, b, rep in jobs}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:  # an episode lost to an error is reported, the others go on
                print(futs[f], "failed:", type(e).__name__, e, flush=True)
                continue
            with lock, a.out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rec["id"], rec["budget"], rec["repeat"], f"{sum(g['ok'] for g in rec['grades'])}/{len(rec['grades'])}",
                  f"passes {rec['passes']}, turns {rec['turns']}, ${rec['usd']:.4f} (total ${spend.spent:.3f})",
                  rec["stopped"] or "", flush=True)


# ---- the references on the same items (h2h's answerers)


def refs(a) -> None:
    """Recency (R), the Activation Oracle alone (A) and AO plus interventions at each budget (B) on the text and fact
    items (h2h's answerers, unchanged); B's edit (h2h.edit_B) on the edit items, scored as the agent's; the Activation
    Oracle alone on the organism items (the prompt's last READ tokens at half depth, greedy; fleet's questions)."""
    items_ = [json.loads(x) for x in a.items.read_text().splitlines()]
    tg = ao_verify.Target(MODEL, a.device)
    tools = Tools(tg, items_)
    with a.out.open("w") as fh:
        def put(rec):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

        for it in items_:
            if it["kind"] in ("text", "fact"):
                for sysname, budget in [("R", None), ("A", None)] + [(f"B@{b}", b) for b in a.budgets]:
                    m, t0 = h2h.Model(tg), time.time()
                    rows = (h2h.answer_R(it) if sysname == "R" else h2h.answer_A(tg, m, it) if sysname == "A"
                            else h2h.answer_B(tg, m, it, budget, random.Random(f"{it['id']}B")))
                    put({"id": it["id"], "kind": it["kind"], "system": sysname, "budget": budget, "passes": m.passes,
                         "seconds": round(time.time() - t0, 2), "grades": [{"q": r["q"], "ok": bool(r["ok"])} for r in rows]})
            elif it["kind"] == "edit":
                m = h2h.Model(tg)
                hooks, info = h2h.edit_B(tg, m, it, it["z"])
                sc = edit_score_hooks(tools, it, [(mod, fn, False) for mod, fn in hooks])
                put({"id": it["id"], "kind": "edit", "system": "B", "budget": None, "passes": m.passes, **info, "edit": sc,
                     "grades": [{"q": "edit", "ok": sc["success"], "score": sc["score"]}]})
            else:
                ids = tools.ids_of(it)
                H, _ = tools.hidden(it, ids)
                q = AO_OWL_Q if it["kind"] == "owl" else "Answer with the correct value only. " + PERSONA_Q[it["attribute"]]
                got = tg.inject(q, H[tg.act_layer + 1, len(ids) - READ:], 1, False, adapter_ao=True)[0]
                put({"id": it["id"], "kind": it["kind"], "system": "A", "budget": None, "passes": 1, "answer": got,
                     "grades": grade(it, {"answer": got}, tg.tok)})
            print(it["id"], "done", flush=True)


def edit_score_hooks(tools: Tools, it, hooks) -> dict:
    """Tools.edit_score for hooks already built (h2h's answerer B)."""
    torch = tools.torch
    x = torch.tensor(tools.controls, device=tools.tg.dev)
    with tools.model_of(it), hooked(hooks), torch.no_grad():
        lp = torch.log_softmax(tools.tg.m(input_ids=x).logits.float(), -1)
    ok = int(tools.logp(it, [it["ids"]], hooks)[0].argmax()) == it["z"]
    kl = float(((tools.cref.exp() * (tools.cref - lp)).sum(-1) / math.log(2)).clamp_min(0).mean())
    kept = float((lp.argmax(-1) == tools.cref.argmax(-1)).float().mean())
    return {"success": ok, "collateral": kl, "kept": kept, "score": float(ok) - kl - (1 - kept)}


# ---- the table


QS = ("a", "b", "c", "fact", "edit", "owl", "persona")


def report(a) -> None:
    recs = [json.loads(x) for x in a.out.read_text().splitlines()]
    cut = [r for r in recs if r["stopped"]]
    recs = [r for r in recs if not r["stopped"]]  # (an episode the cap or an error ended is rerun on a resume)
    if cut:
        print(f"{len(cut)} episodes ended by the cap or an error left out (${sum(r['usd'] for r in cut):.3f})")
    budgets = sorted({r["budget"] for r in recs})
    reps = sorted({r["repeat"] for r in recs})
    print(f"agent {AGENT}: accuracy per question kind and budget (passes per item); per repeat, mean; n questions per repeat")
    for q in QS:
        cells = []
        for b in budgets:
            accs, n = [], 0
            for rep in reps:
                xs = [g for r in recs if r["budget"] == b and r["repeat"] == rep for g in r["grades"] if g["q"] == q]
                if xs:
                    accs.append(sum(g["ok"] for g in xs) / len(xs))
                    n = len(xs)
            if accs:
                cells.append(f"@{b}: " + "/".join(f"{x:.2f}" for x in accs) + f" (mean {sum(accs) / len(accs):.2f}, n {n})")
        if cells:
            print(f"  ({q}) " + "; ".join(cells))
    ed = [r for r in recs if r["kind"] == "edit"]
    for b in budgets:
        xs = [r for r in ed if r["budget"] == b]
        if xs:
            print(f"  (edit) @{b}: score {sum(r['edit']['score'] for r in xs) / len(xs):.3f}, collateral "
                  f"{sum(r['edit']['collateral'] for r in xs) / len(xs):.4f} bits/token, kept {sum(r['edit']['kept'] for r in xs) / len(xs):.3f}")
    print("noise: share of questions answered the same way right or wrong in both repeats, per budget")
    for b in budgets:
        by = {}
        for r in recs:
            if r["budget"] == b:
                by.setdefault(r["id"], {})[r["repeat"]] = [g["ok"] for g in r["grades"]]
        pairs = [(x, y) for d in by.values() if len(d) >= 2 for x, y in zip(d[reps[0]], d[reps[1]])]
        if pairs:
            print(f"  @{b}: {sum(x == y for x, y in pairs) / len(pairs):.2f} of {len(pairs)}")
    print("cost per budget: passes used per item, messages, tool calls, refused calls, Tinker $")
    for b in budgets:
        xs = [r for r in recs if r["budget"] == b]
        print(f"  @{b}: passes {sum(r['passes'] for r in xs) / len(xs):.1f}, messages {sum(r['turns'] for r in xs) / len(xs):.1f}, "
              f"calls {sum(len(r['calls']) for r in xs) / len(xs):.1f}, refused {sum(bool(c['error']) for r in xs for c in r['calls'])}, "
              f"no answer {sum(r['answer'] is None for r in xs)}, ${sum(r['usd'] for r in xs):.3f} (${sum(r['usd'] for r in xs) / len(xs):.4f} per item)")
    tot = sum(r["usd"] for r in recs)
    print(f"Tinker total ${tot:.3f} over {len(recs)} episodes; tokens: prefill {sum(r['prefill'] for r in recs)}, cached "
          f"{sum(r['cached'] for r in recs)}, sampled {sum(r['sampled'] for r in recs)}")
    use = {}
    for r in recs:
        for c in r["calls"]:
            use[c["name"]] = use.get(c["name"], 0) + 1
    print("tool calls:", ", ".join(f"{k} {v}" for k, v in sorted(use.items(), key=lambda kv: -kv[1])))
    if a.refs and a.refs.exists():
        rs = [json.loads(x) for x in a.refs.read_text().splitlines()]
        print("references on the same items (h2h's R recency, A the Activation Oracle alone, B@N AO plus interventions)")
        for sysname in sorted({r["system"] for r in rs}, key=lambda s: (s[0], int(s.split("@")[1]) if "@" in s else 0)):
            line = []
            for q in QS:
                xs = [g for r in rs if r["system"] == sysname for g in r["grades"] if g["q"] == q]
                if xs:
                    line.append(f"({q}) {sum(g['ok'] for g in xs) / len(xs):.2f} n {len(xs)}")
            ps = [r["passes"] for r in rs if r["system"] == sysname]
            print(f"  {sysname}: " + "; ".join(line) + f" | passes per item {sum(ps) / len(ps):.1f}")
            if sysname == "B":
                xs = [r for r in rs if r["system"] == "B"]
                print(f"    (edit) score {sum(r['edit']['score'] for r in xs) / len(xs):.3f}")


PANELS = [("a", "Most important earlier token"), ("b", "Effect of a token swap"), ("c", "Which of two tokens matters more"),
          ("fact", "Which of two statements"), ("edit", "Edit to the second choice"), ("owl", "Owl organism's preference"),
          ("persona", "Persona organism's fact")]


def plot(a) -> None:
    """Accuracy against the passes allowed per item, per question kind: the agent (its two repeats as points, their
    mean a line), AO plus interventions (h2h's B) and the Activation Oracle alone on the same items, the agent with the
    text alone and recency as levels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [r for r in (json.loads(x) for x in a.out.read_text().splitlines()) if not r["stopped"]]
    rs = [json.loads(x) for x in a.refs.read_text().splitlines()] if a.refs else []
    red, blue, green, gray = "#D1495B", "#2F6DB5", "#2E8B57", "#8A8A8A"
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.titlelocation": "left", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    axes = axes.flatten()

    def acc(rows, q):
        xs = [g["ok"] for r in rows for g in r["grades"] if g["q"] == q]
        return sum(xs) / len(xs) if xs else None

    for ax, (q, title) in zip(axes, PANELS):
        pts = []
        for b in sorted({r["budget"] for r in recs if r["budget"]}):
            per = [acc([r for r in recs if r["budget"] == b and r["repeat"] == k], q) for k in sorted({r["repeat"] for r in recs})]
            per = [x for x in per if x is not None]
            if per:
                ax.plot([b] * len(per), per, "o", color=red, ms=6, alpha=0.5)
                pts.append((b, sum(per) / len(per)))
        if pts:
            ax.plot(*zip(*pts), "-o", color=red, lw=3, ms=9)
            ax.annotate("Kimi-K2.6 with tools", pts[-1], textcoords="offset points", xytext=(-8, 10), ha="right", color=red, fontsize=14)
        bs = sorted((int(r["system"].split("@")[1]), r) for r in rs if r["system"].startswith("B@"))
        line = [(b, acc([r for b_, r in bs if b_ == b], q)) for b in sorted({b for b, _ in bs})]
        line = [(b, y) for b, y in line if y is not None]
        if line:
            ax.plot(*zip(*line), "-s", color=blue, lw=2.5, ms=8)
            ax.annotate("AO + interventions", line[0], textcoords="offset points", xytext=(6, -20), color=blue, fontsize=14)
        y = acc([r for r in rs if r["system"] == "A"], q)
        if y is not None:
            ax.plot([1], [y], "D", color=green, ms=10)
            ax.annotate("AO alone", (1, y), textcoords="offset points", xytext=(8, 6), color=green, fontsize=14)
        for rows, name, ls in (([r for r in recs if r["budget"] == 0], "Kimi-K2.6, text alone", ":"),
                               ([r for r in rs if r["system"] == "R"], "recency", "--")):
            y = acc(rows, q)
            if y is not None:
                ax.axhline(y, ls=ls, color=gray, lw=1.5)
                ax.annotate(name, (400, y), textcoords="offset points", xytext=(0, 4), ha="right", color=gray, fontsize=12)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 450)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1, 16, 64, 256])
        ax.set_xticklabels(["1", "16", "64", "256"])
        ax.set_title(title)
        ax.set_xlabel("model passes allowed per item")
    axes[0].set_ylabel("accuracy")
    axes[4].set_ylabel("accuracy (edit: success)")
    axes[-1].axis("off")
    if a.title:
        fig.suptitle(a.title, x=0.01, ha="left", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95 if a.title else 1))
    fig.savefig(a.plot, dpi=150, facecolor="white")
    print("wrote", a.plot)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("items")
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--device")
    i.add_argument("--seed", type=int, default=15)
    s = sub.add_parser("serve")
    s.add_argument("--items", type=Path, required=True)
    s.add_argument("--port", type=int, default=8781)
    s.add_argument("--device")
    f = sub.add_parser("refs")
    f.add_argument("--items", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--budgets", type=int, nargs="+", default=[b for b in BUDGETS if b])
    f.add_argument("--device")
    r = sub.add_parser("run")
    r.add_argument("--server", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    r.add_argument("--repeats", type=int, default=REPEATS)
    r.add_argument("--repeats-text-only", type=int, default=1, help="repeats at budget 0 (the text-alone floor)")
    r.add_argument("--items", nargs="*", help="item ids (default every item)")
    r.add_argument("--cap", type=float, default=5.0, help="Tinker dollars for this output file in all")
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--spent-extra", type=float, default=0.0, help="dollars spent outside the output file (episodes lost "
                   "to a stopped run), counted against the cap")
    p = sub.add_parser("report")
    p.add_argument("out", type=Path)
    p.add_argument("--refs", type=Path)
    g = sub.add_parser("plot")
    g.add_argument("out", type=Path)
    g.add_argument("--refs", type=Path)
    g.add_argument("--plot", type=Path, required=True)
    g.add_argument("--title")
    a = ap.parse_args()
    {"items": items, "serve": serve, "refs": refs, "run": run, "report": report, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
