"""Toy zoo v2 (TOYLOOP lane): models whose circuit depends on the text, so the units cannot be read off the weights.

zoo.py's instances each hold one mechanism whose units are the only ones that write anywhere that matters: a
weights-only walk back from the logits (toyloop/weights_heuristic.py) recovers every circuit exactly. Here each model
(3 layers, 8 heads of dim 16, d_model 128, MLP 512, vocabulary 40) holds 2-4 MODULES, each a zoo.py family instance
(built by zoo.py, then embedded), and a text selects one of them:

  text = BOS, marker_m, <module m's zoo text without its BOS>           prediction at the last position

  * a DETECTOR head (layer 0, shared by every module: one unit in every text's circuit) attends from every position
    to the marker and writes module m's on-flag g_m into a scratch direction;
  * every module unit that writes the logits is GATED by its flag: a gated head's query has a sink term
    a (BIAS - 1.1 g_m) . BOS, so with g_m off it attends to BOS (whose value is zero for every gated head) and writes nothing; a
    gated neuron reads K (g_m - 1) and is far below zero when off. Gated units are in layers 1-2 (heads) or any MLP;
  * a module's layer-0 units that write only scratch (previous-token, copy and feature heads) are always on: on other
    modules' texts they are active (often at the last position) but irrelevant; layer-0 positional heads of different
    modules with the same offset are MERGED into one polysemantic head when their channels fit;
  * a DORMANT module (35% of models) is built and gated like the others but its marker never occurs;
  * DECOYS: content heads that fire only on a trigger token outside some module's vocabulary and then write that
    module's scratch (read by its gated units: strong weight paths, no effect), and CANCELLING TWINS: pairs of heads in
    one layer with the same attention (their queries and keys rotated within the rotary planes, so the weights differ)
    and values rotated, writing exactly opposite logit vectors at every position (strongly active, sum zero);
  * a few zoo-style dump neurons; then the whole residual basis except BIAS is rotated by a random orthogonal matrix
    (an exact reparametrisation: no axis-aligned token, scratch or dump dims remain).

The ground truth is PER TEXT: the detector plus the selected module's ground-truth units (zoo.py's gt, renamed and
shifted one position), verified per module by unit ablations (the gt alone keeps the prediction, removing it breaks
it, the zoo family's claims hold) and per example by the node-level masked run. hardness() scores two baselines per
text: the weights-only heuristic and the units active at the last position (toyloop/render.active_units).

  python3 -B experiments/oracle/toys/zoo2.py build --out ~/mpd-data/toy_zoo2 --n 340
  python3 -B experiments/oracle/toys/zoo2.py hardness --zoo ~/mpd-data/toy_zoo2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import planted  # noqa: E402
import vpd_model  # noqa: E402
import zoo  # noqa: E402
from zoo import BOS, CON_DIMS, FAMILIES, HD, M, OMEGA, POS_PAIRS, head_unit, neuron_unit, parse_unit  # noqa: E402

VERSION = 2
V2, D2, H2, L2, NFF2, N_CTX2 = 40, 128, 8, 3, 512, 32
MARKERS = list(range(32, 40))           # never in a module's tokens (zoo's pools are 1..31)
SCRATCH2 = list(range(1 + V2, D2))      # 87 dims
H0 = math.sqrt(D2)                      # BIAS's value after RMSNorm (it dominates the norm)
BSCALE = zoo.C_BIAS / H0                # zoo's weights expect BIAS at 8 after the norm
SINK_DIM = 3                            # head-local dim of the gate's sink (rotary pair 3: unused by zoo's primitives)
A_SINK, K_GATE = 30.0, 300.0           # K_GATE: above any pre-activation an active module's logit writes can cause
HELD_OUT_FAMILIES = zoo.HELD_OUT_FAMILIES
TRAIN_FAMILIES = [f for f in FAMILIES if f not in HELD_OUT_FAMILIES]
CFG2 = {"n_layer": L2, "n_head": H2, "n_embd": D2, "n_key_value_heads": H2, "rotary_dim": HD, "rms_norm_eps": 1e-6,
        "n_ctx": N_CTX2, "rotary_base": 10000.0}
UNIT_RE = re.compile(r"\bL(\d+)\.([HN])(\d+)\b")


def tok(t: int) -> int:
    return 1 + t


def blank() -> dict:
    sd = {"wte.weight": torch.zeros(V2, D2), "ln_f.weight": torch.full((D2,), zoo.G_OUT)}
    sd["wte.weight"][:, 0] = H0
    for t in range(V2):
        sd["wte.weight"][t, tok(t)] = 1.0
    sd["ln_f.weight"][0] = 0.0
    shapes = {"q_proj": (D2, D2), "k_proj": (D2, D2), "v_proj": (D2, D2), "o_proj": (D2, D2), "c_fc": (NFF2, D2), "down_proj": (D2, NFF2)}
    for i in range(L2):
        sd[f"h.{i}.rms_1.weight"], sd[f"h.{i}.rms_2.weight"] = torch.ones(D2), torch.ones(D2)
        for k, s in shapes.items():
            sd[f"{planted.site_name(i, k)}.weight"] = torch.zeros(*s)
    return sd


class Model:
    """A v2 model under construction: weights, free slots and scratch dims."""

    def __init__(self, rng: random.Random):
        self.rng, self.sd = rng, blank()
        self.free_heads = {l: set(range(H2)) for l in range(L2)}
        self.free_neurons = {l: set(range(NFF2)) for l in range(L2)}
        self.scratch = list(SCRATCH2)
        rng.shuffle(self.scratch)
        self.l0_positional = []   # (head, q block, k block) of always-on layer-0 positional heads, for merging

    def W(self, l, k):
        return self.sd[f"{planted.site_name(l, k)}.weight"]

    def dims(self, n):
        out, self.scratch = self.scratch[:n], self.scratch[n:]
        if len(out) < n:
            raise RuntimeError("scratch exhausted")
        return out

    def head(self, l):
        if not self.free_heads[l]:
            raise RuntimeError(f"no free head in layer {l}")
        h = self.rng.choice(sorted(self.free_heads[l]))
        self.free_heads[l].discard(h)
        return h

    def neuron(self, l):
        n = self.rng.choice(sorted(self.free_neurons[l]))
        self.free_neurons[l].discard(n)
        return n

    def gate_head(self, l, h, g):
        q, k, r = self.W(l, "q_proj"), self.W(l, "k_proj"), h * HD + SINK_DIM
        q[r, 0] += A_SINK / H0
        q[r, g] -= 1.1 * A_SINK
        k[r, tok(BOS)] += A_SINK

    def gate_neuron(self, l, n, g):
        fc = self.W(l, "c_fc")
        fc[n, g] += K_GATE
        fc[n, 0] -= K_GATE / H0


def _writes_tokens(W: torch.Tensor) -> bool:
    return bool(W[1:33].abs().max() > 0)


def embed_module(m: Model, p: dict, sd: dict, g: int, merge: bool = True, bos_to: int | None = None) -> dict:
    """Copy one zoo instance's units (zoo's 64-dim layout) into the model: BIAS to BIAS (rescaled), token dims to the
    same tokens, scratch dims to fresh ones; units writing the logits gated by flag dim g in a layer >= 1 (heads) or any
    MLP layer; the rest (layer-0 scratch writers) in layer 0 ungated, merged into an existing head with the same
    queries and keys when its free value channels suffice. Returns {"units": small -> big, "gated": [big], "merged"}."""
    used = set()
    for k, W in sd.items():
        if k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "c_fc.weight")):
            used |= set((W.abs().sum(0) > 0).nonzero().flatten().tolist())
        elif k.endswith(("o_proj.weight", "down_proj.weight")):
            used |= set((W.abs().sum(1) > 0).nonzero().flatten().tolist())
    small_scratch = sorted(d for d in used if d >= 33)
    dmap = dict(zip(small_scratch, m.dims(len(small_scratch))))
    big = lambda d: 0 if d == 0 else (tok(bos_to) if d == 1 and bos_to is not None else tok(d - 1) if d < 33 else dmap[d])  # noqa: E731
    cols_s = [d for d in range(64) if d in used]
    cols_b = [big(d) for d in cols_s]
    scale = torch.tensor([BSCALE if d == 0 else 1.0 for d in cols_s])
    units = zoo.all_units(sd)
    out_heads = [u for u in units if parse_unit(u)[1] == "H" and _writes_tokens(sd[f"h.{parse_unit(u)[0]}.attn.o_proj.weight"][:, parse_unit(u)[2] * HD:(parse_unit(u)[2] + 1) * HD])]
    out_neurons = [u for u in units if parse_unit(u)[1] == "N"]
    head_layer = m.rng.choice([l for l in (1, 2) if len(m.free_heads[l]) >= len(out_heads)])
    mlp_layer = m.rng.randint(0, L2 - 1)
    umap, gated, merged = {}, [], []
    for u in units:
        l, kind, i = parse_unit(u)
        if kind == "N":
            assert _writes_tokens(sd[f"h.{l}.mlp.down_proj.weight"][:, i:i + 1]), "a scratch-writing neuron"
            n = m.neuron(mlp_layer)
            m.W(mlp_layer, "c_fc")[n, cols_b] += sd[f"h.{l}.mlp.c_fc.weight"][i, cols_s] * scale
            m.W(mlp_layer, "down_proj")[cols_b, n] += sd[f"h.{l}.mlp.down_proj.weight"][cols_s, i]
            m.gate_neuron(mlp_layer, n, g)
            umap[u] = neuron_unit(mlp_layer, n)
            gated.append(umap[u])
            continue
        s = slice(i * HD, (i + 1) * HD)
        q = torch.zeros(HD, D2)
        k = torch.zeros(HD, D2)
        q[:, cols_b] = sd[f"h.{l}.attn.q_proj.weight"][s][:, cols_s] * scale
        k[:, cols_b] = sd[f"h.{l}.attn.k_proj.weight"][s][:, cols_s] * scale
        v = sd[f"h.{l}.attn.v_proj.weight"][s][:, cols_s] * scale       # [HD, used]
        o = sd[f"h.{l}.attn.o_proj.weight"][cols_s][:, s]               # [used, HD]
        chans = [c for c in range(HD) if v[c].abs().max() > 0 or o[:, c].abs().max() > 0]
        if u in out_heads:
            L_, h = head_layer, m.head(head_layer)
        else:
            assert l == 0, "an ungated head above layer 0"
            L_, h = 0, None
            if merge:
                for hh, qq, kk in m.l0_positional:
                    R = slice(hh * HD, (hh + 1) * HD)
                    free = [c for c in range(HD) if m.W(0, "v_proj")[hh * HD + c].abs().max() == 0 and m.W(0, "o_proj")[:, hh * HD + c].abs().max() == 0]
                    if torch.allclose(qq, q) and torch.allclose(kk, k) and len(free) >= len(chans):
                        for c, f in zip(chans, free):
                            m.W(0, "v_proj")[hh * HD + f, cols_b] += v[c]
                            m.W(0, "o_proj")[cols_b, hh * HD + f] += o[:, c]
                        h = hh
                        merged.append(head_unit(0, hh))
                        del R
                        break
            if h is None:
                h = m.head(0)
                if q[:, 0].abs().max() > 0 and q[:, 1:].abs().max() == 0:
                    m.l0_positional.append((h, q, k))
        if head_unit(L_, h) not in merged:
            R = slice(h * HD, (h + 1) * HD)
            m.W(L_, "q_proj")[R] += q
            m.W(L_, "k_proj")[R] += k
            for c in chans:
                m.W(L_, "v_proj")[h * HD + c, cols_b] += v[c]
                m.W(L_, "o_proj")[cols_b, h * HD + c] += o[:, c]
        umap[u] = head_unit(L_, h)
        if u in out_heads:
            m.gate_head(L_, h, g)
            gated.append(umap[u])
    return {"units": umap, "gated": gated, "merged": merged, "scratch": [dmap[d] for d in small_scratch]}


def add_detector(m: Model, flags: dict) -> str:
    """Layer-0 head: from every position attend to the marker (BOS at position 0) and write the flag of its module."""
    h = m.head(0)
    q, k, v, o, d = m.W(0, "q_proj"), m.W(0, "k_proj"), m.W(0, "v_proj"), m.W(0, "o_proj"), h * HD + CON_DIMS[0]
    q[d, 0] += zoo.A_CON / H0
    for mk in MARKERS:
        k[d, tok(mk)] += 4.0 * 30.0 / zoo.A_CON
    k[d, tok(BOS)] += 4.0 * 15.0 / zoo.A_CON
    for c, (mk, g) in enumerate(sorted(flags.items())):
        v[h * HD + c, tok(mk)] += math.sqrt((D2 + 1) / D2)   # the flag exactly 1 (every embedding has the same norm)
        o[g, h * HD + c] += 1.0
    return head_unit(0, h)


def _rot2(theta):
    return torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def add_twins(m: Model, rng: random.Random) -> list:
    """A cancelling pair: head A attends at a rotary offset and writes vector u onto the logits from a value of BIAS
    (and of a few tokens); head B in the same layer has A's queries and keys rotated within each rotary plane (the same
    scores), its values rotated by a random orthogonal map and its output -A's: the sum is zero at every position."""
    l = rng.choice([x for x in range(L2) if len(m.free_heads[x]) >= 2])
    a, b = m.head(l), m.head(l)
    off = rng.randint(0, 4)
    qa, ka = torch.zeros(HD, D2), torch.zeros(HD, D2)
    for j in POS_PAIRS:
        qa[j, 0] = zoo.A_POS / H0 * math.cos(off * OMEGA[j])
        qa[j + HD // 2, 0] = -zoo.A_POS / H0 * math.sin(off * OMEGA[j])
        ka[j, 0] = zoo.A_POS / H0
    va, oa = torch.zeros(HD, D2), torch.zeros(D2, HD)
    va[0, 0] = 1.0 / H0
    for t in rng.sample(range(1, 32), 3):
        va[1, tok(t)] = 1.0
    toks = rng.sample(range(1, 32), 3)
    for t, w in zip(toks, (1.0, -0.8, 0.6)):
        oa[tok(t), 0] = w * M
    oa[tok(rng.randrange(1, 32)), 1] = 0.5 * M
    qb, kb = qa.clone(), ka.clone()
    for j in range(HD // 2):
        Rm = _rot2(rng.uniform(1.0, 2.0) * (1 if rng.random() < 0.5 else -1))
        for X in (qb, kb):
            X[[j, j + HD // 2]] = Rm @ X[[j, j + HD // 2]]
    P, _ = torch.linalg.qr(torch.randn(HD, HD, generator=torch.Generator().manual_seed(rng.randrange(1 << 30))))
    vb, ob = P @ va, -(oa @ P.T)
    for h, (q, k, v, o) in ((a, (qa, ka, va, oa)), (b, (qb, kb, vb, ob))):
        R = slice(h * HD, (h + 1) * HD)
        m.W(l, "q_proj")[R] += q
        m.W(l, "k_proj")[R] += k
        m.W(l, "v_proj")[R] += v
        m.W(l, "o_proj")[:, R] += o
    return [head_unit(l, a), head_unit(l, b)]


def add_decoy(m: Model, rng: random.Random, target: dict, trigger: int) -> str:
    """A content head that attends to `trigger` (else to the marker, whose value is zero) and writes into a module's
    scratch dims (read by that module's gated units): a strong weight path that never matters (the trigger is outside
    that module's tokens, and on other texts the module is off)."""
    l = rng.choice([x for x in (0, 1) if m.free_heads[x]])
    h = m.head(l)
    q, k, v, o, d = m.W(l, "q_proj"), m.W(l, "k_proj"), m.W(l, "v_proj"), m.W(l, "o_proj"), h * HD + CON_DIMS[1]
    q[d, 0] += zoo.A_CON / H0
    k[d, tok(trigger)] += 4.0 * 30.0 / zoo.A_CON
    k[d, tok(BOS)] += 4.0 * 15.0 / zoo.A_CON
    for c, dim in enumerate(rng.sample(target["scratch"], min(3, len(target["scratch"])))):
        v[h * HD + c, tok(trigger)] += 1.0
        o[dim, h * HD + c] += 1.0
    return head_unit(l, h)


def add_dump(m: Model, rng: random.Random, dump: list) -> str:
    l = rng.randint(0, L2 - 1)
    n = m.neuron(l)
    for t in rng.sample(range(1, 32), 3):
        m.W(l, "c_fc")[n, tok(t)] += 0.7
    m.W(l, "c_fc")[n, 0] += rng.uniform(1.5, 3.0) / H0
    m.W(l, "down_proj")[dump[0], n] += zoo.DUMP / 3
    m.W(l, "down_proj")[dump[1], n] += zoo.DUMP / 4
    return neuron_unit(l, n)


def rotate(sd: dict, seed: int) -> dict:
    """The same function in a randomly rotated residual basis (every dim but BIAS: the norms' weights are 1 and ln_f is
    uniform there, so the map is exact)."""
    Q, _ = torch.linalg.qr(torch.randn(D2 - 1, D2 - 1, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    R = torch.eye(D2, dtype=torch.float64)
    R[1:, 1:] = Q
    out = dict(sd)
    rin = lambda W: (W.double() @ R.T).float()   # noqa: E731  readers: W x -> W R^T (R x)
    rout = lambda W: (R @ W.double()).float()    # noqa: E731
    out["wte.weight"] = rin(sd["wte.weight"])
    for i in range(L2):
        for k in ("q_proj", "k_proj", "v_proj", "c_fc"):
            out[f"{planted.site_name(i, k)}.weight"] = rin(sd[f"{planted.site_name(i, k)}.weight"])
        for k in ("o_proj", "down_proj"):
            out[f"{planted.site_name(i, k)}.weight"] = rout(sd[f"{planted.site_name(i, k)}.weight"])
    return out


# ---- instances

def module_tokens(p: dict) -> set:
    """Every token id a zoo instance's params name (its pools, roles, fill)."""
    out = set()

    def walk(x, key=""):
        if key in ("distractors", "head", "heads", "prev", "ind", "h0", "h1", "head_a", "head_b", "yes_neurons", "neurons",
                   "code", "feat", "combos", "T", "positions", "pair", "pair_offset", "offset", "gap", "pairs", "fixed", "d1", "d2",
                   "p1", "p2", "mlp_layer", "no_neuron", "no_neuron", "seed"):
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if key == "map":
                    out.add(int(k))
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, int) and not isinstance(x, bool):
            out.add(x)
    walk(p)
    return out


def make(seed: int, split_kind: str) -> tuple[dict, dict]:
    """(params, state dict) of instance `seed`. split_kind 'train': modules from the train families only;
    'test_family': at least one module from a held-out family."""
    rng = random.Random(f"zoo2:{seed}")
    k = rng.choices([2, 3, 4], [0.35, 0.4, 0.25])[0]
    if split_kind == "train":
        fams = rng.sample(TRAIN_FAMILIES, k)
    else:
        fams = [rng.choice(HELD_OUT_FAMILIES)] + rng.sample([f for f in FAMILIES if f not in HELD_OUT_FAMILIES], k - 1)
        rng.shuffle(fams)
    pool = TRAIN_FAMILIES if split_kind == "train" else list(FAMILIES)
    dormant = rng.choice([f for f in pool if f not in fams]) if rng.random() < 0.35 else None
    m = Model(rng)
    markers = rng.sample(MARKERS, k + (dormant is not None))
    modules = []
    for j, fam in enumerate(fams + ([dormant] if dormant else [])):
        zs = rng.randrange(10 ** 6)
        p = FAMILIES[fam].params(random.Random(f"{fam}:{zs}"))
        p["family"], p["seed"] = fam, zs
        p["distractors"] = {"dump": [], "heads": [], "neurons": []}
        b = FAMILIES[fam].build(p)
        g = m.dims(1)[0]
        info = embed_module(m, p, b.sd, g, bos_to=markers[j] if fam == "detect" else None)
        modules.append({"family": fam, "marker": markers[j], "flag_dim": g, "dormant": j >= k, "zoo_params": json.loads(json.dumps(p)),
                        **info})
    det = add_detector(m, {mod["marker"]: mod["flag_dim"] for mod in modules})
    twins = [add_twins(m, rng) for _ in range(rng.randint(1, 2))]
    decoys = []
    with_scratch = [mod for mod in modules if mod["scratch"]]
    for _ in range(rng.randint(1, 2) if with_scratch else 0):
        tgt = rng.choice(with_scratch)
        outside = [t for t in range(1, 32) if t not in module_tokens(tgt["zoo_params"])]
        if outside and any(m.free_heads[x] for x in (0, 1)):
            trig = rng.choice(outside)
            decoys.append({"unit": add_decoy(m, rng, tgt, trig), "trigger": trig, "writes_into": tgt["family"]})
    dump = m.dims(2)
    dumps = [add_dump(m, rng, dump) for _ in range(rng.randint(0, 2))]
    p = {"version": VERSION, "seed": seed, "split_kind": split_kind, "detector": det, "modules": modules,
         "twins": twins, "decoys": decoys, "dump_neurons": dumps, "rotation_seed": seed * 7919 + 1}
    return json.loads(json.dumps(p)), rotate(m.sd, p["rotation_seed"])


def shift(t: int, detect: bool = False) -> int:
    """A zoo text position in the v2 text (the marker inserted at 1); detect's default (BOS in zoo) is the marker."""
    return t + 1 if t >= 1 else (1 if detect else 0)


def texts(p: dict, j: int, n: int, seed: int, kind: str = "ordinary") -> list:
    """n texts of module j: [(ids, intended answer)]."""
    mod = p["modules"][j]
    out = FAMILIES[mod["family"]].sample(mod["zoo_params"], n, random.Random(seed), kind)
    return [([BOS, mod["marker"]] + ids[1:], a) for ids, a in out]


def gt_spec(p: dict, j: int, ids: list) -> list:
    """Text ids of module j: the detector and the module's zoo gt, units renamed, positions shifted past the marker."""
    mod = p["modules"][j]
    small = [BOS] + ids[2:]
    T = len(ids)
    out = [{"unit": p["detector"], "sites": {"q_proj": [T - 1], "o_proj": [T - 1], "k_proj": list(range(T)), "v_proj": [1]}}]
    merged = {}
    for x in FAMILIES[mod["family"]].gt(mod["zoo_params"], small):
        u = mod["units"][x["unit"]]
        sites = {k: [shift(t, mod["family"] == "detect") for t in v] for k, v in x["sites"].items()}
        if "q_proj" in sites:
            sites["k_proj"] = list(range(max(sites["q_proj"]) + 1))
        if u in merged:   # a merged head serving two roles: union of positions
            for k, v in sites.items():
                merged[u]["sites"][k] = sorted(set(merged[u]["sites"].get(k, [])) | set(v))
        else:
            merged[u] = {"unit": u, "sites": sites}
    return out + list(merged.values())


def rename(text: str, umap: dict) -> str:
    text = UNIT_RE.sub(lambda mt: umap.get(mt.group(0), mt.group(0)), text)
    return re.sub(r"\bposition (\d+)\b(?! back)", lambda mt: f"position {shift(int(mt.group(1)))}", text)


def module_truth(p: dict, j: int, spec: list) -> dict:
    """Per-text truth: units, unit edges, Mermaid and English."""
    mod = p["modules"][j]
    fam = FAMILIES[mod["family"]]
    zp = {**mod["zoo_params"], "distractors": {"dump": [], "heads": [], "neurons": []}}
    edges = [[a if a.startswith("tok:") or a == "OUT" else mod["units"][a], b if b.startswith("tok:") or b == "OUT" else mod["units"][b]]
             for a, b in fam.edges(zp)]
    units = [x["unit"] for x in spec]
    edges = [["tok:marker", p["detector"]]] + [[p["detector"], u] for u in mod["gated"] if u in units] + edges
    edges = [e for e in edges if all(x.startswith("tok:") or x == "OUT" or x in units for x in e)]
    others = sorted({u for k, o in enumerate(p["modules"]) if k != j for u in o["units"].values()} - set(units))
    english = (f"The marker {zoo.tname(mod['marker'])} at position 1 selects the {mod['family'].replace('_', ' ')} mechanism: head "
               f"{p['detector']} attends from every position to the marker and writes this mechanism's on-flag, which un-gates "
               f"its output units ({', '.join(u for u in mod['gated'] if u in units)}); gated-off heads attend to BOS and "
               f"write nothing (gated-off neurons stay below zero). " + rename(fam.explain(zp), mod["units"]).replace("attends to BOS, whose", "attends to the marker, whose")
               + (f" The other mechanisms' units ({', '.join(others)}) are off or irrelevant on this text." if others else ""))
    return {"units": units, "unit_edges": edges, "mermaid": zoo.mermaid(edges), "explanation": english}


def gt_nodes(nat, spec: list, ids: list) -> list:
    """zoo.gt_nodes (VPD's active pieces of the gt units at their positions) plus every piece of a gt neuron at its
    positions: a gated neuron's input piece is far below zero where its flag is off, so at the target its |activation|
    is a few percent of its largest on the text and VPD's threshold calls it inactive, though the prediction needs it."""
    out = {tuple(x) for x in zoo.gt_nodes(nat, spec, ids)}
    for x in spec:
        l, kind, i = parse_unit(x["unit"])
        if kind != "N":
            continue
        for k, pos in x["sites"].items():
            n = planted.site_name(l, k)
            for c in (nat.units[n] == i).nonzero().flatten().tolist():
                out |= {(n, int(t), int(c)) for t in pos}
    return [list(x) for x in sorted(out, key=lambda x: (x[1], x[0], x[2]))]


def all_units2(sd: dict) -> list:
    out = []
    for l in range(L2):
        o, down = sd[f"h.{l}.attn.o_proj.weight"], sd[f"h.{l}.mlp.down_proj.weight"]
        out += [head_unit(l, h) for h in range(H2) if o[:, h * HD:(h + 1) * HD].abs().max() > 0]
        out += [neuron_unit(l, n) for n in range(NFF2) if down[:, n].abs().max() > 0]
    return out


@torch.no_grad()
def last_logits(sd: dict, ts: list) -> torch.Tensor:
    tg = vpd_model.Target(sd, CFG2).eval()
    T = max(len(t) for t in ts)
    ids = torch.tensor([list(t) + [BOS] * (T - len(t)) for t in ts])
    return tg(ids)[torch.arange(len(ts)), torch.tensor([len(t) - 1 for t in ts])].log_softmax(-1)


def check(p: dict, sd: dict, n: int = 96) -> dict:
    """Per active module, on n of its texts: accuracy; the text's gt units alone (everything else removed: other
    modules, the dormant one, twins, decoys, dumps) keep it; the module's units removed break it; the zoo family's
    claims (units renamed). ok: all hold for every module."""
    units = all_units2(sd)
    res, fails = {"modules": []}, []
    for j, mod in enumerate(p["modules"]):
        if mod["dormant"]:
            continue
        fam = FAMILIES[mod["family"]]
        g = random.Random(p["seed"] * 31 + j)
        data = {"ordinary": texts(p, j, n, g.randrange(1 << 30))}
        claims = fam.claims(mod["zoo_params"])
        if any(c[2] == "counterfactual" for c in claims):
            data["counterfactual"] = texts(p, j, n, g.randrange(1 << 30), "counterfactual")
        full = {k: last_logits(sd, [x for x, _ in v]) for k, v in data.items()}
        ans = {k: torch.tensor([a for _, a in v]) for k, v in data.items()}
        acc = lambda lp, k: float((lp.argmax(-1) == ans[k]).float().mean())  # noqa: E731
        ords = [x for x, _ in data["ordinary"]]
        gt = sorted({x["unit"] for ids in ords for x in gt_spec(p, j, ids)})
        r = {"family": mod["family"], "acc": acc(full["ordinary"], "ordinary"), "gt_units": gt}
        keep = last_logits(zoo.ablate(sd, [u for u in units if u not in gt]), ords)
        r["gt_only_acc"], r["gt_only_kl"] = acc(keep, "ordinary"), zoo.kl(full["ordinary"], keep)
        r["gt_removed_acc"] = acc(last_logits(zoo.ablate(sd, [u for u in gt if u != p["detector"]]), ords), "ordinary")
        bad = []
        if r["acc"] < 0.98:
            bad.append("accuracy")
        if r["gt_only_acc"] < 0.98 or r["gt_only_kl"] > 0.05:
            bad.append("gt_only")
        if r["gt_removed_acc"] > 0.6:
            bad.append("gt_removed")
        for name, abl, kind, expect in claims:
            lp = last_logits(zoo.ablate(sd, [mod["units"][u] for u in abl]), [x for x, _ in data[kind]]) if abl else full[kind]
            a, k_ = acc(lp, kind), zoo.kl(full[kind], lp)
            r[f"claim {name}"] = {"acc": a, "kl": k_, "expect": expect}
            good = {"breaks": a <= 0.6, "hurts": a <= r["acc"] - 0.1, "keeps": a >= 0.98 and k_ <= 0.05, "keeps_acc": a >= 0.98,
                    "changes": k_ >= 0.3, "matters": k_ >= 0.1}[expect]
            if not good:
                bad.append(f"claim {name}")
        r["fails"] = bad
        fails += [f"{mod['family']}: {b}" for b in bad]
        res["modules"].append(r)
    res["fails"], res["ok"] = fails, not fails
    return res


def record(p: dict, sd: dict, per_module: int = 2) -> dict:
    nat = planted.native_of(sd, CFG2)
    exs = []
    for j, mod in enumerate(p["modules"]):
        if mod["dormant"]:
            continue
        kinds = ["ordinary"] * per_module + (["counterfactual"] if mod["family"] == "binding_positional" else [])
        g = random.Random(p["seed"] * 13 + j)
        for i, kind in enumerate(kinds):
            got = texts(p, j, i + 1, g.randrange(1 << 30), kind)
            ids, ans = got[-1]
            spec = gt_spec(p, j, ids)
            nodes = gt_nodes(nat, spec, ids)
            lp = last_logits(sd, [ids])[0]
            exs.append({"module": j, "family": mod["family"], "kind": kind, "ids": ids, "target": len(ids) - 1, "intended": ans,
                        "model_top": int(lp.argmax()), "model_p_top": float(lp.max().exp()), "gt_units": spec, "gt_nodes": nodes,
                        "gt_node_kl": zoo.node_kl(nat, ids, nodes), "empty_kl": zoo.node_kl(nat, ids, []),
                        "n_active": int(sum(int((c > 0).sum()) for c in nat.importance(ids).values())), **module_truth(p, j, spec)})
    live = [mod for mod in p["modules"] if not mod["dormant"]]
    every = sorted({u for mod in p["modules"] for u in mod["units"].values()} | {p["detector"]})
    expl = ("This model holds " + str(len(live)) + " mechanisms, selected by the marker token at position 1: "
            + "; ".join(f"{zoo.tname(mod['marker'])} -> {mod['family']} ({', '.join(sorted(set(mod['units'].values())))})" for mod in live)
            + f". Head {p['detector']} reads the marker and un-gates the selected mechanism's output units."
            + (f" A dormant {[mod for mod in p['modules'] if mod['dormant']][0]['family']} mechanism (marker "
               f"{zoo.tname([mod for mod in p['modules'] if mod['dormant']][0]['marker'])}, never used) is gated off." if len(live) < len(p["modules"]) else "")
            + f" Cancelling twins {', '.join(u for t in p['twins'] for u in t)} write opposite logit vectors; decoys "
            + (", ".join(d["unit"] for d in p["decoys"]) or "none") + " write into scratch that only gated-off units read.")
    edges = [list(e) for e in dict.fromkeys(tuple(e) for x in exs for e in x["unit_edges"])]
    sid = f"zoo2_{p['seed']:04d}"
    split = p["split_kind"]
    if split == "train" and int(hashlib.sha1(sid.encode()).hexdigest(), 16) % 1000 < zoo.HELD_OUT_INSTANCE_FRAC * 1000:
        split = "test_instance"
    return {"version": VERSION, "id": sid, "family": "+".join(mod["family"] for mod in live), "families": [mod["family"] for mod in live],
            "seed": p["seed"], "split": split, "params": p, "explanation": expl, "unit_edges": edges, "mermaid": zoo.mermaid(edges),
            "gt_units": every, "distractor_units": [u for t in p["twins"] for u in t] + [d["unit"] for d in p["decoys"]] + p["dump_neurons"],
            "examples": exs, "pieces": {n: int(c) for n, c in nat.C.items()}}


def build(out: Path, n: int, test_family_frac: float = 0.25, log=print) -> None:
    torch.set_num_threads(2)
    (out / "instances").mkdir(parents=True, exist_ok=True)
    stats = {"accepted": 0, "rejected": 0, "reasons": {}}
    t0 = time.time()
    n_tf = round(n * test_family_frac)
    with (out / "manifest.jsonl").open("w") as man:
        seed = 0
        while stats["accepted"] < n and seed < 3 * n:
            kind = "test_family" if stats["accepted"] < n_tf else "train"
            try:
                p, sd = make(seed, kind)
            except (RuntimeError, ValueError, IndexError) as e:   # an allocation that does not fit: another seed
                stats["rejected"] += 1
                stats["reasons"]["alloc"] = stats["reasons"].get("alloc", 0) + 1
                log(f"seed {seed}: {e}")
                seed += 1
                continue
            seed += 1
            res = check(p, sd)
            if res["ok"]:
                rec = record(p, sd)
                if max(e["gt_node_kl"] for e in rec["examples"]) > 0.05:
                    res["fails"] = ["node-level gt kl"]
            if res["fails"]:
                stats["rejected"] += 1
                for f in res["fails"]:
                    stats["reasons"][f.split(":")[0]] = stats["reasons"].get(f.split(":")[0], 0) + 1
                log(f"seed {p['seed']} rejected: {res['fails']}")
                continue
            rec["check"] = res
            torch.save({"sd": sd, "cfg": CFG2}, out / "instances" / f"{rec['id']}.pt")
            (out / "instances" / f"{rec['id']}.json").write_text(json.dumps(rec))
            man.write(json.dumps({"version": VERSION, "id": rec["id"], "family": rec["family"], "families": rec["families"],
                                  "seed": rec["seed"], "split": rec["split"], "weights": f"instances/{rec['id']}.pt",
                                  "record": f"instances/{rec['id']}.json", "gt_units": rec["gt_units"],
                                  "n_distractors": len(rec["distractor_units"]), "n_examples": len(rec["examples"]),
                                  "acc": min(r["acc"] for r in res["modules"]), "explanation": rec["explanation"]}) + "\n")
            man.flush()
            stats["accepted"] += 1
            if stats["accepted"] % 20 == 0:
                log(f"{stats['accepted']} accepted, {stats['rejected']} rejected, {time.time() - t0:.0f}s")
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


# ---- hardness: weights-only and active-at-last baselines against the per-text truth

def hardness(root: Path, log=print) -> dict:
    sys.path.insert(0, str(HERE.parent / "toyloop"))
    import render  # noqa: E402
    import weights_heuristic  # noqa: E402

    torch.set_num_threads(2)
    rows = [json.loads(x) for x in (root / "manifest.jsonl").read_text().splitlines() if x.strip()]
    per = []
    f1 = lambda pr, gd: (2 * len(pr & gd) / (len(pr) + len(gd))) if pr or gd else 1.0  # noqa: E731
    for r in rows:
        d = torch.load(root / r["weights"])
        rec = json.loads((root / r["record"]).read_text())
        toy = render.Toy(rec["id"], d["sd"], d["cfg"], ["BOS"] + [f"t{i}" for i in range(1, V2)], [e["ids"] for e in rec["examples"]])
        wh = {u for u in weights_heuristic.circuit_of(toy)[0]}
        for e in rec["examples"]:
            gold = {u.replace(".", "") for u in e["units"]}
            run = render.run(d["sd"], d["cfg"], e["ids"])
            act = {a.split(":")[0] for a in render.active_units(toy, run, len(e["ids"]) - 1)}
            per.append({"id": rec["id"], "split": rec["split"], "family": e["family"], "gold": len(gold),
                        "weights_f1": f1(wh, gold), "weights_exact": float(wh == gold),
                        "active_f1": f1(act, gold), "active_exact": float(act == gold),
                        "instance_union_f1": f1({u.replace(".", "") for u in rec["gt_units"]}, gold)})
    out = {}
    for key in ("all", "train", "test_instance", "test_family"):
        rs = [x for x in per if key == "all" or x["split"] == key]
        if rs:
            out[key] = {k: round(sum(x[k] for x in rs) / len(rs), 3) for k in ("weights_f1", "weights_exact", "active_f1", "active_exact", "instance_union_f1")} | {"texts": len(rs)}
    fams = sorted({x["family"] for x in per})
    out["by_family"] = {f: {k: round(sum(x[k] for x in per if x["family"] == f) / sum(1 for x in per if x["family"] == f), 3)
                            for k in ("weights_f1", "active_f1")} for f in fams}
    (root / "hardness.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--n", type=int, default=340)
    h = sub.add_parser("hardness")
    h.add_argument("--zoo", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.out, args.n)
    else:
        hardness(args.zoo)


if __name__ == "__main__":
    main()
