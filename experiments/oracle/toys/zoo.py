"""The toy zoo (TOYLOOP lane): a procedural generator of hundreds of small planted-mechanism models whose circuits are
known exactly, for training and testing the oracle with held-out mechanism families and held-out instances.

Every instance is vpd_model.Target's architecture (pre-RMSNorm blocks, rotary attention, GELU(tanh) MLP, tied
embedding) at n_layer 2, n_head 4 (head dim 16), n_embd 64, MLP width 256, vocabulary 32, with ANALYTICALLY
CONSTRUCTED weights (no training: an instance builds in milliseconds). The residual stream has a fixed layout:
  dim 0          BIAS: every token embedding holds the constant 8 there (the norm is dominated by it, so RMSNorm is
                 nearly the identity, and a query or key reading dim 0 alone attends by position: rotary offsets);
  dims 1..32     one-hot token identity (tied: a write there is a logit; ln_f is 5 there, 0 on dim 0);
  dims 33..63    scratch: features one unit writes for a later unit (previous-token codes, gate inputs), and dump
                 dims that distractors write and nothing reads.
Mechanisms are built from a few primitives (positional attention at a rotary offset, content attention on the near
position-invariant low-frequency rotary dims, OV copies into logits or scratch, GELU lookup neurons), then verified by
unit ablations (check()); an instance whose checks fail is rejected. planted.decompose / planted.native_of give the
exact rank-one pieces (per head singular directions, per neuron rows / columns), so goal_eval and the teacher run on
an instance unchanged, and the ground-truth node sets are stated in those pieces.

Families (FAMILIES): copy_offset, induction, binding_positional, gate, additive_vote, bigram_mlp, redundant, cancel,
multihop, detect. HELD_OUT_FAMILIES are never in train.

  python3 -B experiments/oracle/toys/zoo.py build --out ~/mpd-data/toy_zoo --per-family 35
  python3 -B experiments/oracle/toys/zoo.py show ~/mpd-data/toy_zoo/instances/induction_000.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import planted  # noqa: E402
import vpd_model  # noqa: E402

VOCAB, D, H, L, NFF, N_CTX = 32, 64, 4, 2, 256, 32
HD = D // H
BIAS, C_BIAS, G_OUT = 0, 8.0, 5.0
TOK0 = 1                                   # token t's one-hot dim is TOK0 + t
SCRATCH = list(range(TOK0 + VOCAB, D))     # 31 dims
BOS = 0
POOL = list(range(1, VOCAB))               # every non-BOS token: roles are drawn from it per instance
M = 3.0                                    # a logit write (in residual units; ln_f multiplies by G_OUT)
A_POS = 11.5                               # positional query/key strength (score gap ~17 nats between offsets)
A_CON = 10.0                               # content query/key strength
K_NEU = 20.0                               # a lookup neuron's pre-activation gain
POS_PAIRS = (0, 1, 2)                      # head-local rotary pairs used for position (high frequency)
CON_DIMS = (7, 15, 6, 14, 5, 13, 4, 12)    # head-local dims of the low-frequency pairs (near position invariant)
DUMP = 0.005                               # a distractor's write: small, so the RMS norms barely see it (an exact
                                           # decomposition puts Φ's bound near KL 1e-9: a 1e-6 change already costs)
HELD_OUT_FAMILIES = ("multihop", "detect")
HELD_OUT_INSTANCE_FRAC = 0.2
OMEGA = [10000.0 ** (-j / (HD / 2)) for j in range(HD // 2)]

CFG = {"n_layer": L, "n_head": H, "n_embd": D, "n_key_value_heads": H, "rotary_dim": HD, "rms_norm_eps": 1e-6,
       "n_ctx": N_CTX, "rotary_base": 10000.0}


def tok(t: int) -> int:
    return TOK0 + t


def tname(t: int) -> str:
    return "BOS" if t == BOS else f"t{t}"


def head_unit(l: int, h: int) -> str:
    return f"L{l}.H{h}"


def neuron_unit(l: int, n: int) -> str:
    return f"L{l}.N{n}"


def parse_unit(u: str) -> tuple[int, str, int]:
    """'L1.H2' -> (1, 'H', 2); 'L0.N17' -> (0, 'N', 17)."""
    a, b = u.split(".")
    return int(a[1:]), b[0], int(b[1:])


# ---- weight construction

class Builder:
    """Zero weights in vpd_model.Target's state-dict layout plus the primitives every family is built from."""

    def __init__(self):
        sd = {"wte.weight": torch.zeros(VOCAB, D), "ln_f.weight": torch.full((D,), G_OUT)}
        sd["wte.weight"][:, BIAS] = C_BIAS
        for t in range(VOCAB):
            sd["wte.weight"][t, tok(t)] = 1.0
        sd["ln_f.weight"][BIAS] = 0.0
        shapes = {"q_proj": (D, D), "k_proj": (D, D), "v_proj": (D, D), "o_proj": (D, D), "c_fc": (NFF, D), "down_proj": (D, NFF)}
        for i in range(L):
            sd[f"h.{i}.rms_1.weight"] = torch.ones(D)
            sd[f"h.{i}.rms_2.weight"] = torch.ones(D)
            for k, s in shapes.items():
                sd[f"{planted.site_name(i, k)}.weight"] = torch.zeros(*s)
        self.sd = sd

    def W(self, l: int, k: str) -> torch.Tensor:
        return self.sd[f"{planted.site_name(l, k)}.weight"]

    def positional(self, l: int, h: int, offset: int, a: float = A_POS):
        """Head (l, h) attends from every position i to i - offset (queries and keys read BIAS alone; the rotary angle
        does the rest): score(i, j) = a^2 sum_pairs cos((i - j - offset) w)."""
        q, k, r = self.W(l, "q_proj"), self.W(l, "k_proj"), h * HD
        for j in POS_PAIRS:
            q[r + j, BIAS] += a / C_BIAS * math.cos(offset * OMEGA[j])
            q[r + j + HD // 2, BIAS] += -a / C_BIAS * math.sin(offset * OMEGA[j])
            k[r + j, BIAS] += a / C_BIAS

    def content_const_query(self, l: int, h: int, key_scores: dict, slot: int = 0, a: float = A_CON):
        """Head (l, h) attends to positions by what is there: a constant query on content dim `slot`, keys {residual
        dim: score in nats} (score / a / 4 per unit of the dim)."""
        q, k, d = self.W(l, "q_proj"), self.W(l, "k_proj"), h * HD + CON_DIMS[slot]
        q[d, BIAS] += a / C_BIAS
        for dim, s in key_scores.items():
            k[d, dim] += 4.0 * s / a

    def match(self, l: int, h: int, pairs: list, a: float = A_CON):
        """Head (l, h) attends to positions whose key dim matches the query dim: pairs [(query residual dim, key
        residual dim)], one content slot each (at most 8)."""
        q, k = self.W(l, "q_proj"), self.W(l, "k_proj")
        for s, (qd, kd) in enumerate(pairs):
            d = h * HD + CON_DIMS[s]
            q[d, qd] += a
            k[d, kd] += a

    def ov(self, l: int, h: int, channels: list):
        """Head (l, h)'s values and outputs: channels [({read dim: weight}, {write dim: weight})], one head dim each."""
        v, o, r = self.W(l, "v_proj"), self.W(l, "o_proj"), h * HD
        assert len(channels) <= HD
        for c, (reads, writes) in enumerate(channels):
            for dim, w in reads.items():
                v[r + c, dim] += w
            for dim, w in writes.items():
                o[dim, r + c] += w

    def neuron(self, l: int, n: int, reads: dict, bias: float, writes: dict):
        """MLP neuron n of layer l: pre-activation sum reads[dim] x[dim] + bias (bias through BIAS), output writes."""
        fc, down = self.W(l, "c_fc"), self.W(l, "down_proj")
        for dim, w in reads.items():
            fc[n, dim] += w
        fc[n, BIAS] += bias / C_BIAS
        for dim, w in writes.items():
            down[dim, n] += w


class Alloc:
    """Random allocation of heads, neurons, scratch dims and tokens for one instance."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.heads = {(l, h) for l in range(L) for h in range(H)}
        self.neurons = {l: set(range(NFF)) for l in range(L)}
        self.scratch = list(SCRATCH)
        rng.shuffle(self.scratch)
        self.tokens = list(POOL)
        rng.shuffle(self.tokens)

    def head(self, layers=(0, 1)) -> list:
        free = sorted(x for x in self.heads if x[0] in layers)
        x = self.rng.choice(free)
        self.heads.discard(x)
        return list(x)

    def neuron(self, l: int) -> int:
        n = self.rng.choice(sorted(self.neurons[l]))
        self.neurons[l].discard(n)
        return n

    def dims(self, n: int) -> list:
        out, self.scratch = self.scratch[:n], self.scratch[n:]
        assert len(out) == n, "scratch exhausted"
        return out

    def toks(self, n: int) -> list:
        out, self.tokens = self.tokens[:n], self.tokens[n:]
        assert len(out) == n, "tokens exhausted"
        return out


# ---- distractors: active units that write only to dump dims nothing reads

def plan_distractors(a: Alloc, rng: random.Random, max_heads: int = 2, max_neurons: int = 3) -> dict:
    dump = a.dims(3)
    heads, neurons = [], []
    for _ in range(rng.randint(0, max_heads)):
        if not a.heads:
            break
        l, h = a.head()
        mode = rng.choice(["positional", "content"])
        heads.append({"unit": head_unit(l, h), "l": l, "h": h, "mode": mode, "offset": rng.randint(0, 3),
                      "keys": rng.sample(POOL, 3), "reads": rng.sample(POOL, 3), "dump": rng.sample(dump, 2)})
    for _ in range(rng.randint(0 if heads else 1, max_neurons)):
        l = rng.randint(0, L - 1)
        neurons.append({"unit": neuron_unit(l, a.neuron(l)), "l": l, "reads": rng.sample(POOL, 3), "dump": rng.sample(dump, 2),
                        "bias": rng.uniform(1.5, 3.0)})
    return {"dump": dump, "heads": heads, "neurons": neurons}


def build_distractors(b: Builder, d: dict):
    for x in d["heads"]:
        if x["mode"] == "positional":
            b.positional(x["l"], x["h"], x["offset"])
        else:
            b.content_const_query(x["l"], x["h"], {tok(t): 4.0 for t in x["keys"]} | {tok(BOS): 2.0})
        b.ov(x["l"], x["h"], [({BIAS: 0.5 / C_BIAS, **{tok(t): 0.5 for t in x["reads"]}}, {x["dump"][0]: DUMP, x["dump"][1]: -0.6 * DUMP})])
    for x in d["neurons"]:
        b.neuron(x["l"], int(x["unit"].split(".N")[1]), {tok(t): 0.7 for t in x["reads"]}, x["bias"], {x["dump"][0]: DUMP / 3, x["dump"][1]: DUMP / 4})


def distractor_units(p: dict) -> list:
    d = p["distractors"]
    return [x["unit"] for x in d["heads"]] + [x["unit"] for x in d["neurons"]]


def distractor_sentence(p: dict) -> str:
    u = distractor_units(p)
    if not u:
        return ""
    return (f" {', '.join(u)} {'is' if len(u) == 1 else 'are'} active but irrelevant: {'it writes' if len(u) == 1 else 'they write'} "
            "only to residual directions nothing downstream reads.")


# ---- families
# Each family: params(rng) -> p (JSON-able, every choice made), build(p) -> Builder, sample(p, n, g, kind) -> [(ids,
# intended answer)], gt(p, ids) -> [{"unit", "sites": {kind: [positions]}}], explain(p), edges(p) -> unit-level
# [[writer, reader]] (inputs "tok:<role>", output "OUT"), claims(p) -> [(name, ablate units, text kind, expect)].

def _head_gt(u: str, q: list, kv: list) -> dict:
    """A head reading kv from the query positions q: its keys at every position up to the last query are in the circuit
    too (they compete in the softmax: without them a cut or a removal moves the attention elsewhere than the model's)."""
    return {"unit": u, "sites": {"q_proj": q, "o_proj": q, "k_proj": list(range(max(q) + 1)), "v_proj": kv}}


def _neuron_gt(u: str, pos: list) -> dict:
    return {"unit": u, "sites": {"c_fc": pos, "down_proj": pos}}


def _copy_channels(pool: list, gain: float = M) -> list:
    return [({tok(t): 1.0}, {tok(t): gain}) for t in pool]


def _rand_text(rng: random.Random, pool: list, T: int) -> list:
    return [BOS] + [rng.choice(pool) for _ in range(T - 1)]


class CopyOffset:
    name = "copy_offset"

    def params(self, rng):
        a = Alloc(rng)
        l, h = a.head()
        k = rng.randint(1, 4)
        pool = a.toks(rng.randint(6, 12))
        return {"head": [l, h], "offset": k, "pool": pool, "T": [k + 3, 14], "distractors": plan_distractors(a, rng)}

    def build(self, p):
        b = Builder()
        (l, h), k = p["head"], p["offset"]
        b.positional(l, h, k)
        b.ov(l, h, _copy_channels(p["pool"]))
        return b

    def sample(self, p, n, g, kind="ordinary"):
        out = []
        for _ in range(n):
            ids = _rand_text(g, p["pool"], g.randint(*p["T"]))
            out.append((ids, ids[len(ids) - 1 - p["offset"]]))
        return out

    def gt(self, p, ids):
        T = len(ids)
        return [_head_gt(head_unit(*p["head"]), [T - 1], [T - 1 - p["offset"]])]

    def explain(self, p):
        u, k = head_unit(*p["head"]), p["offset"]
        return (f"The model predicts the token {k} position{'s' if k > 1 else ''} before the last one. Head {u} attends from the last "
                f"position to the position {k} back (a fixed offset, by position alone, whatever the tokens) and copies that token "
                f"into the output." + distractor_sentence(p))

    def edges(self, p):
        u = head_unit(*p["head"])
        return [[f"tok:offset-{p['offset']}", u], [u, "OUT"]]

    def claims(self, p):
        return [("mechanism removed", [head_unit(*p["head"])], "ordinary", "breaks")]


class Induction:
    name = "induction"

    def params(self, rng):
        a = Alloc(rng)
        l0, h0 = a.head((0,))
        l1, h1 = a.head((1,))
        n_keys = rng.randint(4, 8)
        keys, vals = a.toks(n_keys), a.toks(rng.randint(n_keys, 12))
        return {"prev": [l0, h0], "ind": [l1, h1], "gap": rng.choice([1, 1, 2]), "pairs": rng.randint(3, 4),
                "keys": keys, "vals": vals, "query_tok": a.toks(1)[0], "fill": a.toks(3),
                "code": a.dims(n_keys), "distractors": plan_distractors(a, rng)}

    def build(self, p):
        b = Builder()
        (l0, h0), (l1, h1) = p["prev"], p["ind"]
        b.positional(l0, h0, p["gap"])
        b.ov(l0, h0, [({tok(t): 1.0}, {c: 1.0}) for t, c in zip(p["keys"], p["code"])])
        b.match(l1, h1, [(tok(t), c) for t, c in zip(p["keys"], p["code"])])
        b.ov(l1, h1, _copy_channels(p["vals"]))
        return b

    def layout(self, p, ks, vs, g, ask):
        ids = [BOS]
        for kk, vv in zip(ks, vs):
            ids += [kk] + [g.choice(p["fill"]) for _ in range(p["gap"] - 1)] + [vv]
        return ids + [p["query_tok"], ks[ask]]

    def sample(self, p, n, g, kind="ordinary"):
        out = []
        for _ in range(n):
            ks, vs = g.sample(p["keys"], p["pairs"]), g.sample(p["vals"], p["pairs"])
            ask = g.randrange(p["pairs"])
            out.append((self.layout(p, ks, vs, g, ask), vs[ask]))
        return out

    def gt(self, p, ids):
        T, step = len(ids), p["gap"] + 1
        kpos = [1 + i * step for i in range(p["pairs"])]
        kp = next(x for x in kpos if ids[x] == ids[-1])
        vp = kp + p["gap"]
        return [_head_gt(head_unit(*p["prev"]), [vp], [kp]),
                _head_gt(head_unit(*p["ind"]), [T - 1], [vp])]

    def explain(self, p):
        a, b, g = head_unit(*p["prev"]), head_unit(*p["ind"]), p["gap"]
        back = "the previous token" if g == 1 else f"the token {g} positions back"
        return (f"Induction over key-value pairs: the model predicts the value that followed the earlier occurrence of the last "
                f"token. Head {a} (a previous-token head) writes the identity of {back} into each position. Head {b} attends from "
                f"the last position (the queried key) to the position whose stored previous key matches it, i.e. {g} after the "
                f"matching key, and copies the value token there into the output." + distractor_sentence(p))

    def edges(self, p):
        a, b = head_unit(*p["prev"]), head_unit(*p["ind"])
        return [["tok:matching key", a], [a, b], ["tok:query key", b], ["tok:value", b], [b, "OUT"]]

    def claims(self, p):
        return [("previous-token head removed", [head_unit(*p["prev"])], "ordinary", "breaks"),
                ("induction head removed", [head_unit(*p["ind"])], "ordinary", "breaks")]


class BindingPositional(Induction):
    """The induction family's texts, but ordinary texts always ask for pair `fixed`, and the model reads that pair's
    value by its position alone (a fixed offset from the query): counterfactual texts asking another pair fail."""
    name = "binding_positional"

    def params(self, rng):
        a = Alloc(rng)
        l, h = a.head()
        pairs = rng.randint(3, 4)
        keys, vals = a.toks(rng.randint(4, 8)), a.toks(rng.randint(6, 12))
        return {"head": [l, h], "gap": rng.choice([1, 1, 2]), "pairs": pairs, "fixed": rng.randrange(pairs), "keys": keys,
                "vals": vals, "query_tok": a.toks(1)[0], "fill": a.toks(3), "distractors": plan_distractors(a, rng)}

    def offset(self, p):
        T = 1 + p["pairs"] * (p["gap"] + 1) + 2
        return T - 1 - (1 + p["fixed"] * (p["gap"] + 1) + p["gap"])

    def build(self, p):
        b = Builder()
        l, h = p["head"]
        b.positional(l, h, self.offset(p))
        b.ov(l, h, _copy_channels(p["vals"]))
        return b

    def sample(self, p, n, g, kind="ordinary"):
        out = []
        for _ in range(n):
            ks, vs = g.sample(p["keys"], p["pairs"]), g.sample(p["vals"], p["pairs"])
            ask = p["fixed"] if kind == "ordinary" else g.choice([i for i in range(p["pairs"]) if i != p["fixed"]])
            out.append((self.layout(p, ks, vs, g, ask), vs[ask]))
        return out

    def gt(self, p, ids):
        T = len(ids)
        return [_head_gt(head_unit(*p["head"]), [T - 1], [T - 1 - self.offset(p)])]

    def explain(self, p):
        u = head_unit(*p["head"])
        return (f"Positional lookup, not key matching: head {u} attends from the last position to a fixed position, "
                f"{self.offset(p)} back (the value of pair {p['fixed'] + 1}), whatever key is asked, and copies the value token "
                f"there into the output. It is right on ordinary texts, which always ask for pair {p['fixed'] + 1}, and wrong "
                f"when another pair is asked." + distractor_sentence(p))

    def edges(self, p):
        u = head_unit(*p["head"])
        return [[f"tok:value of pair {p['fixed'] + 1}", u], [u, "OUT"]]

    def claims(self, p):
        return [("mechanism removed", [head_unit(*p["head"])], "ordinary", "breaks"),
                ("another pair asked", [], "counterfactual", "breaks")]


GATE_OPS = {"and": lambda a, b: a and b, "or": lambda a, b: a or b, "xor": lambda a, b: a != b, "xnor": lambda a, b: a == b,
            "a_and_not_b": lambda a, b: a and not b}


class Gate:
    name = "gate"

    def params(self, rng):
        a = Alloc(rng)
        T = rng.randint(5, 10)
        p1, p2 = sorted(rng.sample(range(1, T - 1), 2))
        ha, hb = a.head((0,)), a.head((0,))
        a0, a1, b0, b1, yes, no, sep = a.toks(7)
        op = rng.choice(sorted(GATE_OPS))
        combos = [(x, y) for x in (1, 0) for y in (1, 0) if GATE_OPS[op](x, y)]
        ml = rng.randint(0, 1)
        neurons = [a.neuron(ml) for _ in combos]
        return {"T": T, "p1": p1, "p2": p2, "head_a": ha, "head_b": hb, "a": [a0, a1], "b": [b0, b1], "yes": yes, "no": no,
                "sep": sep, "fill": a.toks(4), "op": op, "mlp_layer": ml, "yes_neurons": neurons, "combos": combos,
                "no_neuron": a.neuron(ml), "feat": a.dims(2), "distractors": plan_distractors(a, rng)}

    def build(self, p):
        b = Builder()
        T, fa, fb = p["T"], p["feat"][0], p["feat"][1]
        for hd, pos, (t0, t1), f in ((p["head_a"], p["p1"], p["a"], fa), (p["head_b"], p["p2"], p["b"], fb)):
            b.positional(hd[0], hd[1], T - 1 - pos)
            b.ov(hd[0], hd[1], [({tok(t1): 1.0, tok(t0): -1.0}, {f: 1.0})])
        ml = p["mlp_layer"]
        for n, (x, y) in zip(p["yes_neurons"], p["combos"]):   # fires (K) iff a == x and b == y, else <= -K
            sa, sb = 1 if x else -1, 1 if y else -1
            b.neuron(ml, n, {fa: K_NEU * sa / 2, fb: K_NEU * sb / 2}, -K_NEU / 2, {tok(p["yes"]): 2.5 * M / (K_NEU / 2)})
        b.neuron(ml, p["no_neuron"], {}, K_NEU / 2, {tok(p["no"]): M / (K_NEU / 2)})   # always on: the NO default
        return b

    def sample(self, p, n, g, kind="ordinary"):
        out = []
        for i in range(n):
            x, y = (i % 4) // 2, i % 2
            ids = [BOS] + [g.choice(p["fill"]) for _ in range(p["T"] - 2)] + [p["sep"]]
            ids[p["p1"]], ids[p["p2"]] = p["a"][x], p["b"][y]
            out.append((ids, p["yes"] if GATE_OPS[p["op"]](x, y) else p["no"]))
        return out

    def gt(self, p, ids):
        T, ml = len(ids), p["mlp_layer"]
        x, y = int(ids[p["p1"]] == p["a"][1]), int(ids[p["p2"]] == p["b"][1])
        out = [_head_gt(head_unit(*p["head_a"]), [T - 1], [p["p1"]]), _head_gt(head_unit(*p["head_b"]), [T - 1], [p["p2"]]),
               _neuron_gt(neuron_unit(ml, p["no_neuron"]), [T - 1])]
        out += [_neuron_gt(neuron_unit(ml, n), [T - 1]) for n, c in zip(p["yes_neurons"], p["combos"]) if tuple(c) == (x, y)]
        return out

    def explain(self, p):
        ha, hb, ml = head_unit(*p["head_a"]), head_unit(*p["head_b"]), p["mlp_layer"]
        a0, a1 = map(tname, p["a"])
        b0, b1 = map(tname, p["b"])
        ys = ", ".join(neuron_unit(ml, n) for n in p["yes_neurons"])
        op = {"and": "A AND B", "or": "A OR B", "xor": "A XOR B", "xnor": "A XNOR B (A equals B)", "a_and_not_b": "A AND NOT B"}[p["op"]]
        return (f"A boolean gate: the model answers {tname(p['yes'])} iff {op}, where A means position {p['p1']} holds {a1} (not {a0}) "
                f"and B means position {p['p2']} holds {b1} (not {b0}); otherwise {tname(p['no'])}. Head {ha} attends from the last "
                f"position to position {p['p1']} and writes A as +-1 into a scratch direction; head {hb} does the same for B from "
                f"position {p['p2']}. In the layer-{ml} MLP, neuron{'s' if len(p['yes_neurons']) > 1 else ''} {ys} each fire for "
                f"one (A, B) combination that makes the gate true and write {tname(p['yes'])}; neuron {neuron_unit(ml, p['no_neuron'])} "
                f"is always on and writes a weaker {tname(p['no'])} default." + distractor_sentence(p))

    def edges(self, p):
        ha, hb, ml = head_unit(*p["head_a"]), head_unit(*p["head_b"]), p["mlp_layer"]
        ns = [neuron_unit(ml, n) for n in p["yes_neurons"]]
        return ([["tok:A", ha], ["tok:B", hb]] + [[h, n] for n in ns for h in (ha, hb)] + [[n, "OUT"] for n in ns]
                + [[neuron_unit(ml, p["no_neuron"]), "OUT"]])

    def claims(self, p):
        ml = p["mlp_layer"]
        return [("input A head removed", [head_unit(*p["head_a"])], "ordinary", "hurts"),
                ("input B head removed", [head_unit(*p["head_b"])], "ordinary", "hurts"),
                ("gate neurons removed", [neuron_unit(ml, n) for n in p["yes_neurons"]] + [neuron_unit(ml, p["no_neuron"])], "ordinary", "breaks")]


class AdditiveVote:
    name = "additive_vote"

    def params(self, rng):
        a = Alloc(rng)
        n = rng.choice([3, 3, 5])
        T = rng.randint(n + 2, 12)
        pos = sorted(rng.sample(range(1, T - 1), n))
        heads = [a.head() for _ in range(n)]
        yes, no, sep = a.toks(3)
        npos, nneg = rng.randint(1, 2), rng.randint(1, 2)
        return {"T": T, "positions": pos, "heads": heads, "yes": yes, "no": no, "sep": sep, "pos_toks": a.toks(npos),
                "neg_toks": a.toks(nneg), "fill": a.toks(4), "distractors": plan_distractors(a, rng, max_heads=1)}

    def build(self, p):
        b = Builder()
        for (l, h), pos in zip(p["heads"], p["positions"]):
            b.positional(l, h, p["T"] - 1 - pos)
            b.ov(l, h, [({tok(t): 1.0 for t in p["pos_toks"]}, {tok(p["yes"]): 2.0}),
                        ({tok(t): 1.0 for t in p["neg_toks"]}, {tok(p["no"]): 2.0})])
        return b

    def sample(self, p, n, g, kind="ordinary"):
        out = []
        for _ in range(n):
            ids = [BOS] + [g.choice(p["fill"]) for _ in range(p["T"] - 2)] + [p["sep"]]
            votes = 0
            for q in p["positions"]:
                s = g.random() < 0.5
                ids[q] = g.choice(p["pos_toks"] if s else p["neg_toks"])
                votes += 1 if s else -1
            out.append((ids, p["yes"] if votes > 0 else p["no"]))
        return out

    def gt(self, p, ids):
        T = len(ids)
        return [_head_gt(head_unit(*hd), [T - 1], [q]) for hd, q in zip(p["heads"], p["positions"])]

    def explain(self, p):
        hs = ", ".join(f"{head_unit(*hd)} (position {q})" for hd, q in zip(p["heads"], p["positions"]))
        return (f"Additive majority vote: each of the heads {hs} attends from the last position to its own fixed position and adds "
                f"an equal vote to the {tname(p['yes'])} logit if the token there is one of {', '.join(map(tname, p['pos_toks']))}, or "
                f"to the {tname(p['no'])} logit if it is one of {', '.join(map(tname, p['neg_toks']))}. The votes simply add (no MLP); "
                f"the majority wins." + distractor_sentence(p))

    def edges(self, p):
        return [e for hd, q in zip(p["heads"], p["positions"]) for e in ([f"tok:@{q}", head_unit(*hd)], [head_unit(*hd), "OUT"])]

    def claims(self, p):
        hs = [head_unit(*hd) for hd in p["heads"]]
        return [("all voters removed", hs, "ordinary", "breaks")] + [(f"voter {u} removed", [u], "ordinary", "matters") for u in hs]


class BigramMLP:
    name = "bigram_mlp"

    def params(self, rng):
        a = Alloc(rng)
        via = rng.random() < 0.5
        nd = rng.randint(4, 8)
        dom, rng_toks = a.toks(nd), a.toks(rng.randint(3, 6))
        f = {str(t): rng.choice(rng_toks) for t in dom}
        ml = 1 if via else rng.randint(0, 1)
        return {"via_head": via, "head": a.head((0,)) if via else None, "domain": dom, "map": f, "mlp_layer": ml,
                "neurons": [a.neuron(ml) for _ in dom], "code": a.dims(nd) if via else None, "fill": a.toks(4), "T": [4, 12],
                "distractors": plan_distractors(a, rng)}

    def build(self, p):
        b = Builder()
        if p["via_head"]:
            l, h = p["head"]
            b.positional(l, h, 1)
            b.ov(l, h, [({tok(t): 1.0}, {c: 1.0}) for t, c in zip(p["domain"], p["code"])])
        for i, (t, n) in enumerate(zip(p["domain"], p["neurons"])):
            src = p["code"][i] if p["via_head"] else tok(t)
            b.neuron(p["mlp_layer"], n, {src: K_NEU}, -K_NEU / 2, {tok(p["map"][str(t)]): M / (K_NEU / 2)})
        return b

    def sample(self, p, n, g, kind="ordinary"):
        out = []
        for _ in range(n):
            ids = _rand_text(g, p["fill"], g.randint(*p["T"]))
            x = g.choice(p["domain"])
            ids[-2 if p["via_head"] else -1] = x
            out.append((ids, p["map"][str(x)]))
        return out

    def gt(self, p, ids):
        T = len(ids)
        x = ids[-2 if p["via_head"] else -1]
        n = p["neurons"][p["domain"].index(x)]
        out = [_neuron_gt(neuron_unit(p["mlp_layer"], n), [T - 1])]
        if p["via_head"]:
            out.insert(0, _head_gt(head_unit(*p["head"]), [T - 1], [T - 2]))
        return out

    def explain(self, p):
        ml = p["mlp_layer"]
        table = ", ".join(f"{tname(int(t))}->{tname(v)} ({neuron_unit(ml, n)})" for (t, v), n in zip(p["map"].items(), p["neurons"]))
        if p["via_head"]:
            head = (f"Head {head_unit(*p['head'])} copies the previous token's identity into the last position (a scratch code); "
                    f"then a lookup table in the layer-{ml} MLP, one neuron per token, maps it to the prediction: ")
            what = "the token before the last one"
        else:
            head, what = f"A lookup table in the layer-{ml} MLP, one neuron per token, reads the last token and writes its successor: ", "the last token"
        return f"A bigram lookup on {what}. " + head + table + "." + distractor_sentence(p)

    def edges(self, p):
        ns = [neuron_unit(p["mlp_layer"], n) for n in p["neurons"]]
        if p["via_head"]:
            h = head_unit(*p["head"])
            return [["tok:previous", h]] + [[h, n] for n in ns] + [[n, "OUT"] for n in ns]
        return [["tok:last", n] for n in ns] + [[n, "OUT"] for n in ns]

    def claims(self, p):
        out = [("lookup neurons removed", [neuron_unit(p["mlp_layer"], n) for n in p["neurons"]], "ordinary", "breaks")]
        if p["via_head"]:
            out.append(("copying head removed", [head_unit(*p["head"])], "ordinary", "breaks"))
        return out


class Redundant(CopyOffset):
    name = "redundant"

    def params(self, rng):
        a = Alloc(rng)
        l = rng.randint(0, 1)
        heads = [a.head((l,)) for _ in range(rng.choice([2, 2, 3]))]
        k = rng.randint(1, 4)
        return {"heads": heads, "offset": k, "pool": a.toks(rng.randint(6, 12)), "T": [k + 3, 14], "distractors": plan_distractors(a, rng, max_heads=1)}

    def build(self, p):
        b = Builder()
        for l, h in p["heads"]:
            b.positional(l, h, p["offset"])
            b.ov(l, h, _copy_channels(p["pool"], 2.0 * M / len(p["heads"]) + 0.5))
        return b

    def gt(self, p, ids):
        T = len(ids)
        return [_head_gt(head_unit(*hd), [T - 1], [T - 1 - p["offset"]]) for hd in p["heads"]]

    def explain(self, p):
        hs = [head_unit(*hd) for hd in p["heads"]]
        k = p["offset"]
        return (f"Redundant routes: heads {', '.join(hs)} each attend from the last position to the position {k} back (by position "
                f"alone) and copy that token into the output; their writes add, and any one of them alone is enough for the "
                f"prediction (removing one leaves the answer unchanged; removing all breaks it)." + distractor_sentence(p))

    def edges(self, p):
        return [e for hd in p["heads"] for e in ([f"tok:offset-{p['offset']}", head_unit(*hd)], [head_unit(*hd), "OUT"])]

    def claims(self, p):
        hs = [head_unit(*hd) for hd in p["heads"]]
        out = [("all routes removed", hs, "ordinary", "breaks")]
        for u in hs:   # keep only one route
            out.append((f"only {u} kept", [x for x in hs if x != u], "ordinary", "keeps_acc"))
        return out


class Cancel(CopyOffset):
    """copy_offset plus a pair of units (neurons or heads) that are active and write exactly opposite vectors onto the
    logits: removing either one changes the prediction, removing both changes nothing."""
    name = "cancel"

    def params(self, rng):
        p = CopyOffset.params(self, rng)
        a = Alloc(rng)                                   # re-allocate around the copy head
        a.heads.discard(tuple(p["head"]))
        a.tokens = [t for t in a.tokens if t not in p["pool"]]
        p["pair_kind"] = rng.choice(["neurons", "heads"])
        if p["pair_kind"] == "neurons":
            l = rng.randint(0, 1)
            p["pair"] = [neuron_unit(l, a.neuron(l)), neuron_unit(l, a.neuron(l))]
            p["pair_reads"] = rng.sample(POOL, 3)
        else:
            l = rng.randint(0, 1)
            p["pair"] = [head_unit(*a.head((l,))), head_unit(*a.head((l,)))]
            p["pair_offset"] = rng.randint(1, 3)
        p["pair_target"] = rng.sample(p["pool"], 2)
        p["distractors"] = plan_distractors(a, rng, max_heads=1)
        return p

    def build(self, p):
        b = CopyOffset.build(self, p)
        t1, t2 = p["pair_target"]
        u = {tok(t1): 1.6 * M, tok(t2): -1.2 * M}
        for sign, unit in zip((1.0, -1.0), p["pair"]):
            l, kind, i = parse_unit(unit)
            if kind == "N":
                b.neuron(l, i, {tok(t): 0.5 for t in p["pair_reads"]}, 3.0, {d: sign * w / 3.0 for d, w in u.items()})
            else:
                b.positional(l, i, p["pair_offset"])
                b.ov(l, i, [({BIAS: 1.0 / C_BIAS}, {d: sign * w for d, w in u.items()})])
        return b

    def explain(self, p):
        base = CopyOffset.explain(self, {**p, "distractors": {"heads": [], "neurons": []}})
        a, b = p["pair"]
        what = "neurons" if p["pair_kind"] == "neurons" else "heads"
        return (base + f" The {what} {a} and {b} are active and write exactly opposite vectors onto the logits (raising "
                f"{tname(p['pair_target'][0])} and lowering {tname(p['pair_target'][1])}, and the reverse), so they cancel and are not part of "
                f"the circuit; either one alone would change the prediction." + distractor_sentence(p))

    def claims(self, p):
        return [("mechanism removed", [head_unit(*p["head"])], "ordinary", "breaks"),
                ("both cancelling units removed", list(p["pair"]), "ordinary", "keeps"),
                (f"{p['pair'][0]} alone removed", [p["pair"][0]], "ordinary", "changes"),
                (f"{p['pair'][1]} alone removed", [p["pair"][1]], "ordinary", "changes")]


class Multihop:
    name = "multihop"

    def params(self, rng):
        a = Alloc(rng)
        d1, d2 = rng.randint(1, 3), rng.randint(1, 3)
        pool = a.toks(rng.randint(5, 10))
        return {"h0": a.head((0,)), "h1": a.head((1,)), "d1": d1, "d2": d2, "pool": pool, "code": a.dims(len(pool)),
                "T": [d1 + d2 + 2, 14], "distractors": plan_distractors(a, rng)}

    def build(self, p):
        b = Builder()
        (l0, h0), (l1, h1) = p["h0"], p["h1"]
        b.positional(l0, h0, p["d1"])
        b.ov(l0, h0, [({tok(t): 1.0}, {c: 1.0}) for t, c in zip(p["pool"], p["code"])])
        b.positional(l1, h1, p["d2"])
        b.ov(l1, h1, [({c: 1.0}, {tok(t): M}) for t, c in zip(p["pool"], p["code"])])
        return b

    def sample(self, p, n, g, kind="ordinary"):
        out = []
        for _ in range(n):
            ids = _rand_text(g, p["pool"], g.randint(*p["T"]))
            out.append((ids, ids[len(ids) - 1 - p["d1"] - p["d2"]]))
        return out

    def gt(self, p, ids):
        T = len(ids)
        m = T - 1 - p["d2"]
        return [_head_gt(head_unit(*p["h0"]), [m], [m - p["d1"]]), _head_gt(head_unit(*p["h1"]), [T - 1], [m])]

    def explain(self, p):
        a, b, d1, d2 = head_unit(*p["h0"]), head_unit(*p["h1"]), p["d1"], p["d2"]
        return (f"Two-hop positional copy: the model predicts the token {d1 + d2} positions before the last one. Head {a} copies "
                f"the token {d1} position{'s' if d1 > 1 else ''} back into each position as a scratch code; head {b} attends from "
                f"the last position to the position {d2} back and reads that code (not the token there), writing the code's token "
                f"into the output." + distractor_sentence(p))

    def edges(self, p):
        a, b = head_unit(*p["h0"]), head_unit(*p["h1"])
        return [[f"tok:offset-{p['d1'] + p['d2']}", a], [a, b], [b, "OUT"]]

    def claims(self, p):
        return [("first hop removed", [head_unit(*p["h0"])], "ordinary", "breaks"),
                ("second hop removed", [head_unit(*p["h1"])], "ordinary", "breaks")]


class Detect:
    name = "detect"

    def params(self, rng):
        a = Alloc(rng)
        return {"head": a.head(), "triggers": a.toks(rng.randint(1, 3)), "yes": a.toks(1)[0], "no": a.toks(1)[0],
                "query_tok": a.toks(1)[0], "fill": a.toks(rng.randint(4, 8)), "T": [5, 14], "distractors": plan_distractors(a, rng)}

    def build(self, p):
        b = Builder()
        l, h = p["head"]
        b.content_const_query(l, h, {tok(t): 30.0 for t in p["triggers"]} | {tok(BOS): 15.0})
        b.ov(l, h, [({**{tok(t): 1.0 for t in p["triggers"]}, tok(BOS): -1.0}, {tok(p["yes"]): M, tok(p["no"]): -M})])
        return b

    def sample(self, p, n, g, kind="ordinary"):
        out = []
        for i in range(n):
            T = g.randint(*p["T"])
            ids = [BOS] + [g.choice(p["fill"]) for _ in range(T - 2)] + [p["query_tok"]]
            hit = i % 2 == 0
            if hit:
                for q in g.sample(range(1, T - 1), g.randint(1, min(2, T - 2))):
                    ids[q] = g.choice(p["triggers"])
            out.append((ids, p["yes"] if hit else p["no"]))
        return out

    def gt(self, p, ids):
        T = len(ids)
        at = [i for i, t in enumerate(ids[:-1]) if t in p["triggers"]] or [0]
        return [_head_gt(head_unit(*p["head"]), [T - 1], at)]

    def explain(self, p):
        u = head_unit(*p["head"])
        return (f"Detection by content: the model answers {tname(p['yes'])} iff any of {', '.join(map(tname, p['triggers']))} occurs "
                f"earlier in the text, else {tname(p['no'])}. Head {u} attends from the last position to every occurrence of a trigger "
                f"token (wherever it is) and writes {tname(p['yes'])}; with none present it attends to BOS, whose value writes "
                f"{tname(p['no'])}." + distractor_sentence(p))

    def edges(self, p):
        u = head_unit(*p["head"])
        return [["tok:trigger occurrences (or BOS)", u], [u, "OUT"]]

    def claims(self, p):
        return [("detector removed", [head_unit(*p["head"])], "ordinary", "breaks")]


FAMILIES = {f.name: f for f in (CopyOffset(), Induction(), BindingPositional(), Gate(), AdditiveVote(), BigramMLP(), Redundant(),
                                Cancel(), Multihop(), Detect())}


# ---- running, ablation and checks

def all_units(sd: dict) -> list:
    """Every head and neuron with a nonzero output."""
    out = []
    for l in range(L):
        o, down = sd[f"h.{l}.attn.o_proj.weight"], sd[f"h.{l}.mlp.down_proj.weight"]
        out += [head_unit(l, h) for h in range(H) if o[:, h * HD:(h + 1) * HD].abs().max() > 0]
        out += [neuron_unit(l, n) for n in range(NFF) if down[:, n].abs().max() > 0]
    return out


def ablate(sd: dict, units) -> dict:
    """The weights with each unit's output zeroed (a head's o_proj columns, a neuron's down_proj column)."""
    sd = dict(sd)
    for u in units:
        l, kind, i = parse_unit(u)
        key = f"h.{l}.attn.o_proj.weight" if kind == "H" else f"h.{l}.mlp.down_proj.weight"
        W = sd[key].clone()
        if kind == "H":
            W[:, i * HD:(i + 1) * HD] = 0.0
        else:
            W[:, i] = 0.0
        sd[key] = W
    return sd


@torch.no_grad()
def last_logits(sd: dict, texts: list) -> torch.Tensor:
    """log-probabilities at each text's last position [n, V]: one forward of the texts right-padded with BOS (causal
    attention: the padding after a text's last position cannot change it)."""
    tg = vpd_model.Target(sd, CFG).eval()
    T = max(len(t) for t in texts)
    ids = torch.tensor([list(t) + [BOS] * (T - len(t)) for t in texts])
    last = torch.tensor([len(t) - 1 for t in texts])
    return tg(ids)[torch.arange(len(texts)), last].log_softmax(-1)


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    return float((p.exp() * (p - q)).sum(-1).mean())


def check(p: dict, sd: dict, n: int = 256) -> dict:
    """The instance's mechanism measured by unit ablations on n ordinary texts (and counterfactual where claimed):
    accuracy of the full model; with the ground-truth units alone kept (every other unit, distractors included,
    removed); with all ground-truth units removed; each distractor removed; each family claim. ok: all pass."""
    fam = FAMILIES[p["family"]]
    g = random.Random(p["seed"] * 7 + 1)
    data = {"ordinary": fam.sample(p, n, g, "ordinary")}
    if any(c[2] == "counterfactual" for c in fam.claims(p)):
        data["counterfactual"] = fam.sample(p, n, g, "counterfactual")
    full = {k: last_logits(sd, [x for x, _ in v]) for k, v in data.items()}
    ans = {k: torch.tensor([a for _, a in v]) for k, v in data.items()}
    acc = lambda lp, k: float((lp.argmax(-1) == ans[k]).float().mean())  # noqa: E731
    units = all_units(sd)
    gt_units = sorted({x["unit"] for ids, _ in data["ordinary"][:64] for x in fam.gt(p, ids)}
                      | set(p.get("gt_units_extra", [])))
    res = {"acc": acc(full["ordinary"], "ordinary"), "units": units, "gt_units": gt_units}
    fails = []
    if res["acc"] < 0.98:
        fails.append("accuracy")
    keep = last_logits(ablate(sd, [u for u in units if u not in gt_units]), [x for x, _ in data["ordinary"]])
    res["gt_only_acc"], res["gt_only_kl"] = acc(keep, "ordinary"), kl(full["ordinary"], keep)
    if res["gt_only_acc"] < 0.98 or res["gt_only_kl"] > 0.05:
        fails.append("gt_only")
    none = last_logits(ablate(sd, gt_units), [x for x, _ in data["ordinary"]])
    res["gt_removed_acc"] = acc(none, "ordinary")
    if res["gt_removed_acc"] > 0.6:
        fails.append("gt_removed")
    for u in distractor_units(p):
        lp = last_logits(ablate(sd, [u]), [x for x, _ in data["ordinary"]])
        res[f"distractor {u} removed kl"] = kl(full["ordinary"], lp)
        if res[f"distractor {u} removed kl"] > 0.02:
            fails.append(f"distractor {u}")
    for name, abl, kind, expect in fam.claims(p):
        lp = last_logits(ablate(sd, abl), [x for x, _ in data[kind]]) if abl else full[kind]
        a, k = acc(lp, kind), kl(full[kind], lp)
        res[f"claim {name}"] = {"acc": a, "kl": k, "expect": expect}
        good = {"breaks": a <= 0.6, "hurts": a <= res["acc"] - 0.1, "keeps": a >= 0.98 and k <= 0.05, "keeps_acc": a >= 0.98,
                "changes": k >= 0.3, "matters": k >= 0.1}[expect]
        if not good:
            fails.append(f"claim {name}")
    res["fails"], res["ok"] = fails, not fails
    return res


# ---- the ground truth in pieces

def gt_nodes(nat, spec: list, ids: list) -> list:
    """The ground-truth circuit as a node set [[site, position, piece]]: VPD's active pieces (planted.native_of's CI > 0)
    of the spec's units at the spec's positions per site."""
    ci = nat.importance(ids)
    want = {}
    for x in spec:
        l, kind, i = parse_unit(x["unit"])
        for k, pos in x["sites"].items():
            want.setdefault((planted.site_name(l, k), i), set()).update(pos)
    out = []
    for n in nat.names:
        units = nat.units[n]
        for t, c in (ci[n] > 0).nonzero().tolist():
            if t in want.get((n, int(units[c])), ()):
                out.append([n, int(t), int(c)])
    return sorted(out, key=lambda x: (x[1], x[0], x[2]))


@torch.no_grad()
def node_kl(nat, ids: list, nodes: list) -> float:
    """KL at the last position of the model || the masked run with only `nodes` on (remainders on, ~0 here)."""
    T = len(ids)
    x = torch.tensor([ids])
    full = nat.vpd.target_forward(x)[0, -1].log_softmax(-1)
    masks = {n: torch.zeros(1, T, nat.C[n]) for n in nat.names}
    for n, t, c in nodes:
        masks[n][0, t, c] = 1.0
    out = nat.vpd.masked(x, masks, {n: torch.ones(1, T) for n in nat.names})[0, -1].log_softmax(-1)
    return float((full.exp() * (full - out)).sum())


def mermaid(edges: list) -> str:
    ids = {}

    def nid(x):
        if x not in ids:
            ids[x] = "OUT" if x == "OUT" else (x.replace(".", "_") if not x.startswith("tok:") else f"in{len(ids)}")
        return ids[x]
    lines = []
    for a, b in edges:
        lines.append(f"  {nid(a)} --> {nid(b)}")
    decl = []
    for x, i in ids.items():
        decl.append(f"  {i}((prediction))" if x == "OUT" else f'  {i}["{x}"]')
    return "flowchart LR\n" + "\n".join(decl + lines)


# ---- instances

def make(family: str, seed: int) -> tuple[dict, dict]:
    """(params with family and seed, state dict)."""
    fam = FAMILIES[family]
    rng = random.Random(f"{family}:{seed}")
    p = fam.params(rng)
    p["family"], p["seed"] = family, seed
    b = fam.build(p)
    build_distractors(b, p["distractors"])
    return json.loads(json.dumps(p)), b.sd


def texts(p: dict, n: int, seed: int, kind: str = "ordinary") -> list:
    """n texts of an instance's distribution: [(token ids, intended answer at the last position)]."""
    return FAMILIES[p["family"]].sample(p, n, random.Random(seed), kind)


def split_of(family: str, seed: int) -> str:
    if family in HELD_OUT_FAMILIES:
        return "test_family"
    h = int(hashlib.sha1(f"{family}:{seed}".encode()).hexdigest(), 16) % 1000
    return "test_instance" if h < HELD_OUT_INSTANCE_FRAC * 1000 else "train"


def record(p: dict, sd: dict, n_examples: int = 4) -> dict:
    """The instance's JSON record (everything but the weights)."""
    fam = FAMILIES[p["family"]]
    nat = planted.native_of(sd, CFG)
    exs = []
    kinds = ["ordinary"] * n_examples + (["counterfactual"] * 2 if any(c[2] == "counterfactual" for c in fam.claims(p)) else [])
    g = random.Random(p["seed"] * 7 + 3)
    for kind in kinds:
        ids, ans = fam.sample(p, 1, g, kind)[0]
        if kind == "ordinary" and p["family"] in ("gate", "detect"):   # cover both answers
            ids, ans = fam.sample(p, len(exs) + 1, g, kind)[-1]
        spec = fam.gt(p, ids)
        nodes = gt_nodes(nat, spec, ids)
        lp = last_logits(sd, [ids])[0]
        exs.append({"kind": kind, "ids": ids, "target": len(ids) - 1, "intended": ans, "model_top": int(lp.argmax()),
                    "model_p_top": float(lp.max().exp()), "gt_units": spec, "gt_nodes": nodes,
                    "gt_node_kl": node_kl(nat, ids, nodes), "empty_kl": node_kl(nat, ids, []),
                    "n_active": int(sum(int((m > 0).sum()) for m in nat.importance(ids).values()))})
    edges = fam.edges(p)
    gt_units = sorted({x["unit"] for e in exs for x in e["gt_units"]})
    return {"id": f"{p['family']}_{p['seed']:03d}", "family": p["family"], "seed": p["seed"], "split": split_of(p["family"], p["seed"]),
            "params": p, "explanation": fam.explain(p), "unit_edges": edges, "mermaid": mermaid(edges), "gt_units": gt_units,
            "distractor_units": distractor_units(p), "examples": exs, "pieces": {n: int(c) for n, c in nat.C.items()}}


def build(out: Path, per_family: int, families: list, max_tries: int = 3, log=print) -> None:
    torch.set_num_threads(2)
    (out / "instances").mkdir(parents=True, exist_ok=True)
    stats = {}
    with (out / "manifest.jsonl").open("w") as man:
        for family in families:
            ok = rej = 0
            seed = 0
            t0 = time.time()
            while ok < per_family and seed < per_family * max_tries:
                p, sd = make(family, seed)
                seed += 1
                res = check(p, sd)
                if not res["ok"]:
                    rej += 1
                    log(f"{family} seed {p['seed']} rejected: {res['fails']}")
                    continue
                rec = record(p, sd)
                if max(e["gt_node_kl"] for e in rec["examples"]) > 0.05:
                    rej += 1
                    log(f"{family} seed {p['seed']} rejected: node-level gt kl {[round(e['gt_node_kl'], 4) for e in rec['examples']]}")
                    continue
                rec["check"] = res
                torch.save({"sd": sd, "cfg": CFG}, out / "instances" / f"{rec['id']}.pt")
                (out / "instances" / f"{rec['id']}.json").write_text(json.dumps(rec))
                man.write(json.dumps({"id": rec["id"], "family": family, "seed": rec["seed"], "split": rec["split"],
                                      "weights": f"instances/{rec['id']}.pt", "record": f"instances/{rec['id']}.json",
                                      "gt_units": rec["gt_units"], "n_distractors": len(rec["distractor_units"]),
                                      "acc": res["acc"], "explanation": rec["explanation"]}) + "\n")
                man.flush()
                ok += 1
            stats[family] = {"accepted": ok, "rejected": rej, "seconds": round(time.time() - t0, 1),
                             "split": "test_family" if family in HELD_OUT_FAMILIES else "train/test_instance"}
            log(f"{family}: {stats[family]}")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--per-family", type=int, default=35)
    b.add_argument("--families", default=",".join(FAMILIES))
    s = sub.add_parser("show")
    s.add_argument("record", type=Path)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.out, args.per_family, args.families.split(","))
    else:
        r = json.loads(args.record.read_text())
        print(r["id"], r["split"], "\n", r["explanation"], "\n", r["mermaid"])
        for e in r["examples"]:
            print(e["kind"], e["ids"], "->", e["intended"], "model", e["model_top"], f"p {e['model_p_top']:.3f}",
                  "nodes", len(e["gt_nodes"]), "of", e["n_active"], f"kl {e['gt_node_kl']:.4f} empty {e['empty_kl']:.2f}")


if __name__ == "__main__":
    main()
