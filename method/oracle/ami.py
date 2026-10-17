"""The AMI metric ladder (scrub, the lead's 10-16 AMI track, ~/mpd-data/team/ami_1016.txt): one scorer for circuits of
neurons and attention heads in an HF Qwen3 at the ladder's stronger rungs, the same code for every method's circuits.

Units: a neuron at a position (the input of down_proj, act(gate) x up), a head at a position (its slice of o_proj's
input), and a head channel at a position (one coordinate of that slice). The token embeddings are the input and are never
ablated; the final norm and the unembedding have no units.
Background: every unit outside the circuit, at every position, is set to its mean over pretraining text (Means: the
atlas corpus, means per absolute-position bucket BUCKETS, so position 0's attention-sink values are the mean of first
positions; MODE "global" uses one mean over every position, the circuit-sparsity paper's form). The circuit's units
compute from what is left.

  R3  the circuit-sparsity paper's criterion (Gao et al. 2025, arXiv 2511.13653): the background above with the circuit's
      units live. On each prompt, at the answer position: the KL divergence of the circuit model's next-token
      distribution from the full model's (nats), and the task loss, -ln p(answer word) (nats; the answer word's token
      set from answer_contrast on the full model's run), which passes at most TARGET (the paper's 0.15) above the full
      model's own task loss on that prompt (the absolute pass, at most TARGET, beside it). Prompts: the
      circuit's own pair (clean and corrupted) and held-out prompts (its question's template with other keys of its task,
      never its own, its corrupted key or the keys it was selected on; the same token count, so every unit keeps its
      position). Beside the circuit: the full model, the empty circuit (every unit at the background) and RANDOM circuits
      of the same size (each unit redrawn at its kind, layer and position), and the circuit frozen (its units at their
      full-model values, none computed: what the circuit's values carry, apart from computing them). recovered = (loss of the empty circuit - loss
      of the circuit) / (loss of the empty circuit - loss of the full model), for the task loss and for the KL.
      Sizes: units, distinct (layer, unit) pairs, positions, and the unit-to-unit edges a dense model has among them
      (edges(): every pair joined by a direct residual path; no edge is pruned at R3).
  R4  causal scrubbing (Chan et al. 2022), layered (scrub()): the R3 circuit model, with each circuit unit's value taken
      from a prompt its hypothesis calls equivalent to the input (classes(): by default, from the record's tags, a unit
      at or after the key's first token depends on the key only, so other wordings of the question with the same key, the
      dev set's families, may supply it; positions mapped by role), computed there in layer order with earlier units
      already scrubbed. Scored as R3 (KL, task loss, pass) on the own clean prompt and the held-out prompts.
  R5  Gao's feature-quality substitution: the background everywhere, nothing computed, each circuit unit set to its
      label's simulated value (load_sims; an unlabelled unit stays at the background), beside the frozen circuit (each
      unit at its full-model value: the ceiling of perfect labels). Scored as R3.
  coverage  per token of general text (windows: the first COVER_T tokens of corpus documents outside the means' sample),
      the fewest units at positions up to it that, live under the background above, keep the next-token distribution's
      KL from the full model's within a bound (COVER_BOUNDS), read off the KL curve of the units in the order a learned
      mask gives them (cover_batch: soft masks, then hard masks forward with the sigmoid's gradient; the first crossing of
      each bound on a log grid, then halved down to one unit); an upper bound on the fewest units per token, the
      proposal's irreducible-complexity estimate, at each model size.

  ami.py sample OUT.npz [--tokens 1100000]          a seeded sample of the atlas corpus's documents (the means' text)
  ami.py means SAMPLE.npz OUT.pt [--model M]        the background means of a model over the sample
  ami.py r3|r4|r5 RECORDS_DIR OUT_DIR --means M.pt --label METHOD [--devset D] [--pool P] [--model M] [--sims S]
                                                    every record's R3 (a circuit_finder record, or a circuit file with
                                                    "neurons" [[layer, pos, neuron]] and "heads" [[layer, pos, head]])
  ami.py windows OUT.npz [--n 64]                   corpus windows outside the means' sample (coverage's text)
  ami.py cover OUT_DIR --means M.pt [--texts W.npz] [--windows 16] [--model M]
                                                    coverage over general text, one file per window
  ami.py qsearch OUT_DIR --means M.pt [--devset D]  the R3 reference circuit of each question (mask_search on its pair)
  ami.py tsearch OUT_DIR --means M.pt [--devset D]  the same over the question's wording with 8 keys (a task circuit)
  ami.py board OUT_DIR [OUT_DIR ...]                the leaderboard lines of scored sets (TSV on stdout)
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import answer_contrast  # noqa: E402
import circuit_finder as cf  # noqa: E402

CORPUS = Path.home() / "mpd-data" / "atlas" / "corpus_qwen3"
DEVSET = Path.home() / "mpd-data" / "circuits" / "devset_8b.json"
POOL = Path.home() / "mpd-data" / "circuits" / "trainset_8b.json"
BUCKETS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512)   # bucket b holds positions from the b-th boundary (0 first) to the next
NB = len(BUCKETS) + 1
MODE = "bucket"     # the background: "bucket" (the mean at the position's bucket) or "global" (one mean over all positions)
TARGET = 0.15       # nats: the circuit-sparsity paper's target task loss
DRAWS = 8           # held-out prompts per circuit
RANDOM = 2          # random circuits of the circuit's size per circuit
BATCH_TOKENS = 8192  # rows x positions a forward at most


def bucket_of(pos: torch.Tensor) -> torch.Tensor:
    """The position bucket of each absolute position."""
    return torch.bucketize(pos, torch.tensor(BUCKETS, device=pos.device), right=True)


# ---------------------------------------------------------------- the background means

def sample(out: Path, tokens: int = 1_100_000, seed: int = 0, corpus: Path = CORPUS) -> dict:
    """Documents of the atlas corpus in a seeded random order until their tokens reach tokens: {tokens, offsets, sources}
    (offsets[j]..offsets[j + 1] is document j)."""
    tok = np.load(corpus / "tokens.npy", mmap_mode="r")
    off = np.load(corpus / "offsets.npy")
    src = np.load(corpus / "sources.npy")
    n = len(off) - 1 if len(off) == len(src) + 1 else len(off)
    ends = np.append(off[1:], len(tok)) if len(off) == n else off[1:]
    order = np.random.default_rng(seed).permutation(n)
    pick, total = [], 0
    for j in order:
        pick.append(int(j))
        total += int(ends[j] - off[j])
        if total >= tokens:
            break
    parts = [np.asarray(tok[off[j]:ends[j]], dtype=np.int32) for j in pick]
    offsets = np.cumsum([0] + [len(p) for p in parts]).astype(np.int64)
    d = {"tokens": np.concatenate(parts), "offsets": offsets, "sources": src[pick], "documents": np.array(pick, np.int64)}
    np.savez(out, **d)
    return d


@torch.no_grad()
def collect(model, docs: list, batch_tokens: int = 16384, log=None) -> dict:
    """Sums and sums of squares of every neuron (down_proj's input) and head channel (o_proj's input) per position bucket
    over docs (lists of token ids, each from its first token), and the positions counted per bucket."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    cfg, layers = core.config, list(core.model.layers)
    L, D = cfg.num_hidden_layers, cfg.intermediate_size
    HD = cfg.num_attention_heads * (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    dev = core.model.embed_tokens.weight.device
    acc = {"neuron_sum": torch.zeros(NB, L, D, dtype=torch.float64, device=dev),
           "neuron_sq": torch.zeros(NB, L, D, dtype=torch.float64, device=dev),
           "head_sum": torch.zeros(NB, L, HD, dtype=torch.float64, device=dev),
           "head_sq": torch.zeros(NB, L, HD, dtype=torch.float64, device=dev)}
    count = torch.zeros(NB, dtype=torch.float64, device=dev)
    state = {}

    def hook(kind, i):
        def f(mod, args):
            x = args[0].float()
            oh = state["oh"].to(x.device)                               # [B, T, NB]: valid positions by bucket
            acc[kind + "_sum"][:, i] += torch.einsum("btn,btd->nd", oh, x).double().to(dev)
            acc[kind + "_sq"][:, i] += torch.einsum("btn,btd->nd", oh, x * x).double().to(dev)
        return f
    hs = []
    for i, layer in enumerate(layers):
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(hook("neuron", i)))
        hs.append(layer.self_attn.o_proj.register_forward_pre_hook(hook("head", i)))
    order = sorted(range(len(docs)), key=lambda j: -len(docs[j]))
    t0, done = time.time(), 0
    try:
        at = 0
        while at < len(order):
            T = len(docs[order[at]])
            B = max(1, batch_tokens // T)
            part = [docs[j] for j in order[at:at + B]]
            at += len(part)
            x = torch.zeros(len(part), T, dtype=torch.long)
            valid = torch.zeros(len(part), T, dtype=torch.bool)
            for b, d in enumerate(part):
                x[b, :len(d)] = torch.as_tensor(d)
                valid[b, :len(d)] = True
            oh = torch.nn.functional.one_hot(bucket_of(torch.arange(T))[None].expand(len(part), -1), NB).float() * valid[..., None]
            state["oh"] = oh.to(dev)
            count += oh.sum((0, 1)).double().to(dev)
            core.model(input_ids=x.to(dev), attention_mask=valid.long().to(dev))   # (right padding: positions from 0)
            done += int(valid.sum())
            if log:
                log(f"  means: {done:,} tokens, {done / (time.time() - t0):.0f} tokens/s")
    finally:
        for h in hs:
            h.remove()
    return {**{k: v.cpu() for k, v in acc.items()}, "count": count.cpu(), "buckets": list(BUCKETS)}


class Means:
    """A model's background means (collect's file): per position bucket, every neuron's and head channel's mean."""

    def __init__(self, path, device="cpu"):
        d = torch.load(path, map_location="cpu", weights_only=False)
        assert list(d["buckets"]) == list(BUCKETS), "means collected with other position buckets"
        c = d["count"].clamp(min=1)
        self.count = d["count"]
        self.mean = {"neuron": (d["neuron_sum"] / c[:, None, None]).float().to(device),
                     "head": (d["head_sum"] / c[:, None, None]).float().to(device)}
        tot = float(d["count"].sum())
        self.glob = {k: (d[k + "_sum"].sum(0) / tot).float().to(device) for k in ("neuron", "head")}
        self.meta = {k: d[k] for k in d if k not in ("neuron_sum", "neuron_sq", "head_sum", "head_sq", "count")}
        self._cache = {}

    def table(self, kind: str, layer: int, T: int, dev, dtype, mode: str = None) -> torch.Tensor:
        """[T, width] the background of one layer's neurons or head channels at positions 0..T-1."""
        mode = mode or MODE
        key = (kind, layer, T, str(dev), dtype, mode)
        if key not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            if mode == "global":
                t = self.glob[kind][layer][None].expand(T, -1)
            else:
                t = self.mean[kind][:, layer][bucket_of(torch.arange(T, device=self.mean[kind].device))]
            self._cache[key] = t.to(dev, dtype).contiguous()
        return self._cache[key]


# ---------------------------------------------------------------- the masked run

def channels(u: tuple, hd: int) -> tuple:
    """A unit as (kind, layer, position, indices): ("neuron", layer, pos, k) -> neuron k; ("head", layer, h, pos) -> the
    head's hd channels; ("hchan", layer, pos, c) -> channel c of o_proj's input."""
    if u[0] == "neuron":
        return "neuron", int(u[1]), int(u[2]), [int(u[3])]
    if u[0] == "head":
        return "head", int(u[1]), int(u[3]), list(range(int(u[2]) * hd, (int(u[2]) + 1) * hd))
    if u[0] == "hchan":
        return "head", int(u[1]), int(u[2]), [int(u[3])]
    raise ValueError(f"unit {u!r}")


class _Stop(Exception):
    """Raised by a hook once the layer a run was for has been read."""


class Ablator:
    """Runs of an HF Qwen3 (circuit_finder.Patcher's hooks) whose rows each name the units they keep live: every other
    neuron and head channel at every position set to the background (Means), each row's kept units computed, and each
    row's set values (over) written over its units. A row is {"ids": [...], "keep": units or None (the full model),
    "over": [(unit, value tensor)], "grab": units whose values to read}, or for large circuits {"ids", "mask": {kind:
    [L, T, width] bool}} (dense(): live where the mask is) with "values": {kind: [L, T, width]} (those values there).""" 

    def __init__(self, model, tok, means: Means, mode: str | None = None):
        self.pt = cf.Patcher(model, tok)
        self.means, self.mode = means, mode
        self.L, self.D, self.H, self.hd = self.pt.L, self.pt.D, self.pt.H, self.pt.hd

    def _plan(self, rows: list) -> dict:
        live, keep, over = [], {}, {}
        for b, r in enumerate(rows):
            if r.get("mask") is not None:           # (a dense row: its mask, and its values when given, in place of keep)
                continue
            if r.get("keep") is None:
                live.append(b)
                continue
            for u in r["keep"]:
                kind, i, p, ks = channels(u, self.hd)
                e = keep.setdefault((kind, i), ([], [], []))
                e[0].extend([b] * len(ks))
                e[1].extend([p] * len(ks))
                e[2].extend(ks)
            for u, v in r.get("over") or []:
                kind, i, p, ks = channels(u, self.hd)
                e = over.setdefault((kind, i), ([], [], [], []))
                e[0].extend([b] * len(ks))
                e[1].extend([p] * len(ks))
                e[2].extend(ks)
                e[3].append(torch.as_tensor(v, dtype=torch.float32).reshape(-1))
        grab = {}
        for b, r in enumerate(rows):
            for j, u in enumerate(r.get("grab") or []):
                kind, i, p, ks = channels(u, self.hd)
                e = grab.setdefault((kind, i), ([], [], [], []))
                e[0].extend([b] * len(ks))
                e[1].extend([p] * len(ks))
                e[2].extend(ks)
                e[3].append((b, j, len(ks)))
        dev = self.pt.dev
        t = lambda xs: torch.tensor(xs, dtype=torch.long, device=dev)  # noqa: E731
        dense = {}                                  # (rows given as masks: {kind: [L, T, width] bool}, values beside, and
        for kind in ("neuron", "head"):             # where the values apply: "set", default the mask)
            rs = [b for b, r in enumerate(rows) if r.get("mask") is not None and kind in r["mask"]]
            if rs:
                vals = all(rows[b].get("values") is not None for b in rs)
                dense[kind] = (t(rs), torch.stack([rows[b]["mask"][kind].to(dev) for b in rs]),
                               torch.stack([rows[b]["values"][kind].to(dev) for b in rs]) if vals else None,
                               torch.stack([(rows[b].get("set") or rows[b]["mask"])[kind].to(dev) for b in rs]) if vals else None)
        return {"dense": dense, "grab": {k: (t(v[0]), t(v[1]), t(v[2]), v[3]) for k, v in grab.items()}, "got": {},
                "live": t(live) if live else None,
                "keep": {k: tuple(t(x) for x in v) for k, v in keep.items()},
                "over": {k: (t(v[0]), t(v[1]), t(v[2]), torch.cat(v[3]).to(dev)) for k, v in over.items()},
                "all_live": len(live) == len(rows)}

    def _fns(self, plan: dict, stop: tuple | None = None) -> dict:
        def make(kind):
            def fn(i, h):
                out = body(kind, i, h)
                if stop == (kind, i):                   # (layer_values: this layer's values read, the rest not run)
                    plan["layer"] = out.detach()
                    raise _Stop
                return out
            return fn

        def body(kind, i, h):
            if plan["all_live"] and (kind, i) not in plan["over"] and (kind, i) not in plan["grab"]:
                return h
            B, T, W = h.shape
            out = self.means.table(kind, i, T, h.device, h.dtype, self.mode)[None].expand(B, -1, -1).clone()
            if plan["live"] is not None:
                lv = plan["live"].to(h.device)
                out[lv] = h[lv]
            if kind in plan["dense"]:               # (dense rows: live where the mask is, the given values where set)
                rs, M, V, S = plan["dense"][kind]
                rs = rs.to(h.device)
                o = torch.where(M[:, i].to(h.device), h[rs], out[rs])
                if V is not None:
                    o = torch.where(S[:, i].to(h.device), V[:, i].to(h.device, h.dtype), o)
                out[rs] = o
            if (kind, i) in plan["keep"]:
                b, p, k = (z.to(h.device) for z in plan["keep"][(kind, i)])
                out[b, p, k] = h[b, p, k]
            if (kind, i) in plan["over"]:
                b, p, k, v = (z.to(h.device) for z in plan["over"][(kind, i)])
                out[b, p, k] = v.to(h.dtype)
            if (kind, i) in plan["grab"]:           # (the units' values as the layer reads them: after the edits)
                b, p, k, slots = plan["grab"][(kind, i)]
                vals, at_ = out[b.to(h.device), p.to(h.device), k.to(h.device)].float().cpu(), 0
                for row, j, n in slots:
                    plan["got"][(row, j)] = vals[at_:at_ + n]
                    at_ += n
            return out
        L = set(range(self.L))
        return {"neu": make("neuron"), "head": make("head"), "only": {"res": set(), "head": L, "neu": L}}

    @torch.no_grad()
    def layer_values(self, rows: list, kind: str, layer: int) -> list:
        """Per row, the values [T, width] of one layer's neurons or head channels as the run reads them (after the rows'
        edits), the layers above not run."""
        out = [None] * len(rows)
        by = {}
        for j, r in enumerate(rows):
            by.setdefault(len(r["ids"]), []).append(j)
        for T, js in by.items():
            n = max(1, BATCH_TOKENS // T)
            for a in range(0, len(js), n):
                part = [rows[j] for j in js[a:a + n]]
                plan = self._plan(part)
                try:
                    self.pt._last(torch.tensor([r["ids"] for r in part]), len(part), **self._fns(plan, stop=(kind, layer)))
                except _Stop:
                    pass
                for b, j in enumerate(js[a:a + n]):
                    out[j] = plan["layer"][b]
        return out

    @torch.no_grad()
    def capture(self, ids: list) -> dict:
        """The full model's every unit on ids: {"neuron": [L, T, D], "head": [L, T, H * hd]} (on the host, the model's dtype)."""
        cap = {"neuron": [None] * self.L, "head": [None] * self.L}

        def make(kind):
            def fn(i, h):
                cap[kind][i] = h[0].detach().cpu()
                return h
            return fn
        L = set(range(self.L))
        self.pt._last(torch.tensor([ids]), 1, neu=make("neuron"), head=make("head"), only={"res": set(), "head": L, "neu": L})
        return {k: torch.stack(v) for k, v in cap.items()}

    @torch.no_grad()
    def logp(self, rows: list) -> torch.Tensor:
        """[R, V] float32 log-probabilities at each row's last position."""
        return self.run(rows)[0]

    @torch.no_grad()
    def run(self, rows: list) -> tuple:
        """(logp's [R, V], per row the values of its "grab" units as its run reads them, after the edits); rows of any
        lengths (each length its own forwards, BATCH_TOKENS rows x positions a forward at most)."""
        if not rows:
            return torch.zeros(0), []
        lp, got = [None] * len(rows), [None] * len(rows)
        by = {}
        for j, r in enumerate(rows):
            by.setdefault(len(r["ids"]), []).append(j)
        for T, js in by.items():
            n = max(1, BATCH_TOKENS // T)
            for a in range(0, len(js), n):
                part = [rows[j] for j in js[a:a + n]]
                plan = self._plan(part)
                hn = self.pt._last(torch.tensor([r["ids"] for r in part]), len(part), **self._fns(plan))
                for b, (j, x) in enumerate(zip(js[a:a + n], self.pt._logp(hn).float().cpu())):
                    lp[j] = x
                    got[j] = [plan["got"].get((b, k)) for k in range(len(part[b].get("grab") or []))]
        return torch.stack(lp), got


# ---------------------------------------------------------------- circuits and prompts

def dense(units: list, L: int, T: int, D: int, H: int, hd: int) -> dict:
    """units as masks {"neuron": [L, T, D] bool, "head": [L, T, H * hd] bool} (a head: its hd channels)."""
    m = {"neuron": torch.zeros(L, T, D, dtype=torch.bool), "head": torch.zeros(L, T, H * hd, dtype=torch.bool)}
    for kind, idx in _arrays(units, hd).items():
        if len(idx[0]):
            m[kind][idx] = True
    return m


def _arrays(units: list, hd: int) -> dict:
    """{kind: (layers, positions, indices)} as index tensors, heads as their channels."""
    n = [u for u in units if u[0] == "neuron"]
    h = [u for u in units if u[0] == "head"]
    c = [u for u in units if u[0] == "hchan"]
    nl = torch.tensor([[u[1], u[2], u[3]] for u in n], dtype=torch.long).reshape(-1, 3)
    hl = torch.tensor([[u[1], u[3], u[2]] for u in h], dtype=torch.long).reshape(-1, 3)
    cl = torch.tensor([[u[1], u[2], u[3]] for u in c], dtype=torch.long).reshape(-1, 3)
    if len(hl):
        n_h = len(hl)
        hl = hl.repeat_interleave(hd, 0)
        hl[:, 2] = hl[:, 2] * hd + torch.arange(hd).repeat(n_h)
    hc = torch.cat([hl, cl])
    return {"neuron": (nl[:, 0], nl[:, 1], nl[:, 2]), "head": (hc[:, 0], hc[:, 1], hc[:, 2])}


def redraw_dense(m: dict, seed: int) -> dict:
    """The random control as masks: per kind, layer and position, as many units as m has there, drawn uniformly from the
    units m does not hold (heads drawn as whole heads when m's head channels come in whole heads)."""
    g = torch.Generator().manual_seed(seed)
    out = {k: torch.zeros_like(v) for k, v in m.items()}
    for kind, v in m.items():
        cnt = v.sum(-1)                                                  # [L, T]
        for i, p in cnt.nonzero().tolist():
            free = (~v[i, p]).nonzero()[:, 0]
            out[kind][i, p, free[torch.randperm(len(free), generator=g)[:int(cnt[i, p])]]] = True
    return out


def record_units(rec: dict) -> list:
    """A circuit's units: a circuit_finder record's nodes (every stage) and routes, or a circuit file's "neurons" [[layer,
    pos, neuron, ...]], "heads" [[layer, pos, head]] and "hchans" [[layer, pos, channel]]; each once, in order."""
    us = [("neuron", int(n["layer"]), int(n["pos"]), int(n["neuron"])) for n in rec.get("nodes") or []]
    us += [("head", int(r["layer"]), int(r["head"]), int(r["pos"])) for r in rec.get("routes") or []]
    us += [("neuron", int(x[0]), int(x[1]), int(x[2])) for x in rec.get("neurons") or []]
    us += [("head", int(x[0]), int(x[2]), int(x[1])) for x in rec.get("heads") or []]
    us += [("hchan", int(x[0]), int(x[1]), int(x[2])) for x in rec.get("hchans") or []]
    return list(dict.fromkeys(us))


def invalid(units: list, L: int, D: int, H: int, HD: int, T: int) -> list:
    """Units outside the model or the prompt (a record holding any is refused, not trimmed)."""
    W = {"neuron": D, "head": H, "hchan": HD}
    return [u for u in units if not (u[0] in W and 0 <= u[1] < L and 0 <= _lp(u)[2] < T
                                     and 0 <= (u[2] if u[0] == "head" else u[3]) < W[u[0]])]


def _lp(u: tuple) -> tuple:
    """(kind, layer, position) of a unit."""
    return (u[0], u[1], u[3] if u[0] == "head" else u[2])


def edges(units: list) -> int:
    """Unit-to-unit edges a dense model has among units: a direct residual path from u's write to v's read. A neuron or
    head channel at (layer l, position p) writes into p's residual stream after its sublayer; a neuron at (l', p) reads
    p's stream before its MLP (l < l', or l = l' for a head); a head at (l', q) reads every position up to q through its
    keys and values and q through its query (l < l', p <= q). Counted from per-(layer, position) counts."""
    if not units:
        return 0
    L = max(u[1] for u in units) + 1
    T = max(_lp(u)[2] for u in units) + 1
    n, h = np.zeros((L, T), np.int64), np.zeros((L, T), np.int64)
    for u in units:
        (n if u[0] == "neuron" else h)[u[1], _lp(u)[2]] += 1
    below = lambda c: np.cumsum(c, 0) - c        # noqa: E731  (units in layers below, same position)
    upto_pos = lambda c: np.cumsum(c, 1)          # noqa: E731  (units at positions up to p, same layer)
    e = (n * below(n)).sum() + (n * np.cumsum(h, 0)).sum()          # (into neurons: same position)
    e += (h * below(upto_pos(n))).sum() + (h * below(upto_pos(h))).sum()   # (into heads: lower layers, positions up to q)
    return int(e)


def sizes(units: list) -> dict:
    return {"units": len(units), "neurons": sum(u[0] == "neuron" for u in units), "heads": sum(u[0] == "head" for u in units),
            "hchans": sum(u[0] == "hchan" for u in units),
            "distinct": len({(u[0], u[1], u[3] if u[0] == "neuron" else u[2]) for u in units}),
            "positions": len({_lp(u)[2] for u in units}), "edges": edges(units)}


def redraw(units: list, D: int, H: int, HD: int, seed: int) -> list:
    """The random control: each unit redrawn uniformly at its kind, layer and position, none of the circuit's own."""
    g = np.random.default_rng(seed)
    have = set(units)
    out = []
    for u in units:
        for _ in range(1000):
            if u[0] == "neuron":
                v = ("neuron", u[1], u[2], int(g.integers(D)))
            elif u[0] == "head":
                v = ("head", u[1], int(g.integers(H)), u[3])
            else:
                v = ("hchan", u[1], u[2], int(g.integers(HD)))
            if v not in have:
                break
        have.add(v)
        out.append(v)
    return out


def load_questions(*paths) -> dict:
    """{id: question} over question files ({"questions": [...]} or lists), sealed test questions refused."""
    out = {}
    for p in paths:
        if p and Path(p).exists():
            d = json.loads(Path(p).read_text())
            for q in d["questions"] if isinstance(d, dict) else d:
                if q.get("split") != "test":
                    out[q["id"]] = q
    return out


def task_keys(qs: dict) -> dict:
    """{task: every key its questions name}: their keys, corrupted keys and partners' keys, in first-seen order."""
    out = {}
    for q in qs.values():
        ks = out.setdefault(q.get("task"), {})
        for k in [q.get("key"), q.get("corrupt_key")] + [p["key"] if isinstance(p, dict) else p for p in q.get("partners") or []]:
            if k:
                ks[k] = None
    return {t: list(v) for t, v in out.items()}


def prompt_ids(tok, user: str, q: dict) -> list:
    pre = q.get("prefill")
    return cf.chat(tok, user, q.get("system")) + (tok(pre, add_special_tokens=False)["input_ids"] if pre else [])


def heldout(tok, rec: dict, q: dict, keys: list, draws: int = DRAWS) -> list:
    """Up to draws held-out prompts [(key, ids)]: the question's template with other keys of its task (never its own key,
    its corrupted key or a key it was selected on), only those of the clean prompt's token count, in a seeded order."""
    user, key = q.get("user") or rec.get("user"), q.get("key") or rec.get("key")
    if not user or not key or user.count(key) != 1:
        return []
    banned = {key, q.get("corrupt_key"), rec.get("key")} | set(rec.get("selection_keys") or [])
    cand = [k for k in keys if k not in banned]
    np.random.default_rng(zlib.crc32(str(rec.get("id")).encode())).shuffle(cand)
    T, out = len(rec["ids"]), []
    for k in cand:
        ids = prompt_ids(tok, user.replace(key, k), q)
        if len(ids) == T:
            out.append((k, ids))
            if len(out) == draws:
                break
    return out


# ---------------------------------------------------------------- R3

def _metrics(lp_full: torch.Tensor, lp: torch.Tensor, answer: list) -> dict:
    """KL(full || circuit) in nats, the task loss -ln p(answer word) in nats, and whether the top token starts the answer."""
    p = lp_full.exp()
    a = torch.as_tensor(answer)
    return {"kl": float((p * (lp_full - lp)).sum()), "ce": float(-torch.logsumexp(lp[a], 0)), "top1": int(lp.argmax()) in set(answer)}


def r3(abl: Ablator, tok, rec: dict, q: dict, keys: list, draws: int = DRAWS, n_random: int = RANDOM) -> dict:
    """A circuit's R3 (module docstring) on its own pair and its held-out prompts."""
    t0 = time.perf_counter()
    units = record_units(rec)
    clean, corr = list(rec["ids"]), list(rec.get("corrupted") or [])
    bad = invalid(units, abl.L, abl.D, abl.H, abl.H * abl.hd, len(clean))
    if bad:
        return {"id": rec.get("id"), "error": f"{len(bad)} units outside the model or the prompt, e.g. {bad[0]}", "summary": {}}
    prompts = [("own", "clean", clean)] + ([("own", "corrupted", corr)] if len(corr) == len(clean) else [])
    prompts += [("heldout", k, ids) for k, ids in heldout(tok, rec, q, keys, draws)]
    if q.get("user") and prompt_ids(tok, q["user"], q) != clean:
        prompts = prompts[:2]                      # (the record's prompt is not its question's template: no held-out prompts)
    seed = zlib.crc32(str(rec.get("id")).encode())
    T = len(clean)
    M = dense(units, abl.L, T, abl.D, abl.H, abl.hd)
    if M["head"].any():                             # (whole heads redrawn as whole heads)
        Mh = M["head"].view(abl.L, T, abl.H, abl.hd)
        whole = bool((Mh.any(-1) == Mh.all(-1)).all())
    else:
        whole = True
    rand = []
    for j in range(n_random):
        if whole:
            r = redraw_dense({"neuron": M["neuron"], "head": M["head"].view(abl.L, T, abl.H, abl.hd).any(-1)}, seed + j)
            r["head"] = r["head"][..., None].expand(-1, -1, -1, abl.hd).reshape(abl.L, T, -1)
        else:
            r = redraw_dense(M, seed + j)
        rand.append(r)
    empty = {k: torch.zeros_like(v) for k, v in M.items()}
    arms = [("full", None), ("empty", empty), ("circuit", M)] + [(f"random{j}", r) for j, r in enumerate(rand)]
    rows = [{"ids": ids, "keep": None} if m is None else {"ids": ids, "mask": m} for _, _, ids in prompts for _, m in arms]
    lp, _ = abl.run(rows)
    caps = [abl.capture(ids) for _, _, ids in prompts]        # (the full model's every unit, for the frozen circuit)
    frozen = [{"ids": ids, "mask": M, "values": caps[j]} for j, (_, _, ids) in enumerate(prompts)]
    arms.append(("frozen", None))                  # (the circuit's units at their full-model values, nothing computed)
    lp = torch.cat([lp.view(len(prompts), len(arms) - 1, -1), abl.logp(frozen)[:, None]], 1)
    sets = answer_contrast.token_sets_many(abl.pt.core, tok, [(ids, lp[j, 0], None, None, None) for j, (_, _, ids) in enumerate(prompts)])
    out = []
    for j, (split, name, ids) in enumerate(prompts):
        out.append({"split": split, "prompt": name, "answer_word": sets[j]["answer_word"],
                    **{a: _metrics(lp[j, 0], lp[j, x], sets[j]["answer"]) for x, (a, _) in enumerate(arms)}})
    res = {"id": rec.get("id"), "task": rec.get("task") or q.get("task"), "size": sizes(units),
           "random_units": [int(r["neuron"].sum() + r["head"].sum() // (1 if not whole else abl.hd)) for r in rand],
           "prompts": out, "seconds_found": rec.get("seconds"), "met_r0": rec.get("met")}
    if len(corr) == len(clean) and rec.get("answer_tokens") and rec.get("alternatives"):
        s = {"answer": rec["answer_tokens"], "alternatives": rec["alternatives"]}
        c = answer_contrast.contrast(lp[:2].reshape(-1, lp.shape[-1]), s).view(2, len(arms))     # [clean/corrupted, arm]
        g = c[0] - c[1]
        res["own_gap"] = {a: float(g[x] / g[0]) for x, (a, _) in enumerate(arms)}           # the contrast gap kept, / full's
        res["own_gap_bits"] = float(g[0])
    res["summary"] = summarize(out, [a for a, _ in arms])
    res["seconds"] = round(time.perf_counter() - t0, 2)
    return res


def summarize(prompts: list, arms: list) -> dict:
    """Per split: each arm's mean KL, task loss, pass rate (task loss at most TARGET above the full model's on the same
    prompt), absolute pass rate (task loss at most TARGET) and top-1 rate; recovered shares of the task loss and of the KL
    (module docstring); random arms pooled."""
    out = {}
    for split in ("own", "heldout"):
        P = [p for p in prompts if p["split"] == split]
        if not P:
            continue
        s = {"n": len(P)}
        for a in arms:
            s[a] = {m: float(np.mean([p[a][m] for p in P])) for m in ("kl", "ce", "top1")}
            s[a]["pass"] = float(np.mean([p[a]["ce"] - p["full"]["ce"] <= TARGET for p in P]))
            s[a]["pass_abs"] = float(np.mean([p[a]["ce"] <= TARGET for p in P]))
        rnd = [a for a in arms if a.startswith("random")]
        if rnd:
            s["random"] = {m: float(np.mean([s[a][m] for a in rnd])) for m in ("kl", "ce", "top1", "pass", "pass_abs")}
        for a in [a for a in ("circuit", "frozen", "simulated") if a in arms] + (["random"] if rnd else []):
            d_ce = s["empty"]["ce"] - s["full"]["ce"]
            s[a]["recovered_ce"] = (s["empty"]["ce"] - s[a]["ce"]) / d_ce if d_ce > 1e-9 else None
            s[a]["recovered_kl"] = 1 - s[a]["kl"] / s["empty"]["kl"] if s["empty"]["kl"] > 1e-9 else None
        out[split] = s
    return out


# ---------------------------------------------------------------- R4

R4_DRAWS = 4        # scrubbed runs a prompt (each with its own draws)


def chat_text(tok, user: str, q: dict) -> str:
    msgs = ([{"role": "system", "content": q["system"]}] if q.get("system") else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prompt(tok, user: str, key: str, q: dict, template) -> dict | None:
    """A prompt with its key's token span: {"ids", "span": (first, end), "key", "template"} (None when the key is not
    once in user or its tokens cannot be found)."""
    if not user or not key or user.count(key) != 1:
        return None
    text = chat_text(tok, user, q)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    c0 = text.find(user) + user.find(key)
    c1 = c0 + len(key)
    span = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if a < c1 and b > c0]
    if not span:
        return None
    ids = list(enc["input_ids"]) + (tok(q["prefill"], add_special_tokens=False)["input_ids"] if q.get("prefill") else [])
    return {"ids": ids, "span": (span[0], span[-1] + 1), "key": key, "template": template}


def role_map(src: dict, dst: dict) -> dict:
    """{position in src: position in dst} by role: the two prompts' common prefix, their common tail counted back from the
    end, and the key's span when its tokens are the same in both."""
    a, b = src["ids"], dst["ids"]
    T, T2 = len(a), len(b)
    pre = 0
    while pre < min(T, T2) and a[pre] == b[pre]:
        pre += 1
    tail = 0
    while tail < min(T, T2) - pre and a[T - 1 - tail] == b[T2 - 1 - tail]:
        tail += 1
    m = {p: p for p in range(pre)}
    m.update({p: T2 - (T - p) for p in range(T - tail, T)})
    (s0, s1), (d0, d1) = src["span"], dst["span"]
    if a[s0:s1] == b[d0:d1]:
        m.update({p: d0 + (p - s0) for p in range(s0, s1)})
    return m


def at(u: tuple, p: int) -> tuple:
    """unit u moved to position p."""
    return (u[0], u[1], u[2], p) if u[0] == "head" else (u[0], u[1], p, u[3])


def classes(rec: dict, units: list, span: tuple) -> dict:
    """The hypothesis: each unit's class. A record's "hypothesis" ({str(unit): "key" | "template" | "pre"}) wins;
    otherwise (the patching record's tags) a unit before the key's first token is "pre" (it cannot read the key, so no
    prompt with the same wording differs there: never scrubbed) and every other unit is "key" (its value is claimed to
    depend on the question's key only, so any wording of the question with the same key may supply it). "template": the
    value is claimed to depend on the wording only, so the same wording with any other key may supply it."""
    hyp = rec.get("hypothesis") or {}
    return {u: hyp.get(str(u), "pre" if _lp(u)[2] < span[0] else "key") for u in units}


def scrub(abl: Ablator, x: dict, units: list, cls: dict, pools: dict, draws: int, seed: int) -> tuple:
    """Layered causal scrubbing of a circuit on prompt x (R4). Per draw, each scrubbed unit u gets a prompt x_u from its
    class's pool (pools: {"key": other wordings with x's key, "template": x's wording with other keys}) in which u's
    position has a role. In layer order (a layer's heads before its neurons), u's value is its value in the circuit model
    (the background, the circuit live) run on x_u with every earlier scrubbed unit set to its own scrubbed value wherever
    that unit has a role in x_u; the output is the circuit model on x with every scrubbed unit set to its value. Returns
    ([draws, V] log-probabilities, the units scrubbed, the units left live for want of a pool prompt)."""
    g = np.random.default_rng(seed)
    maps = {}

    def mp(y):
        k = id(y)
        if k not in maps:
            maps[k] = role_map(x, y)
        return maps[k]
    todo = [u for u in units if cls[u] in ("key", "template")]
    pick = [{} for _ in range(draws)]
    left = []
    for u in todo:
        cand = [y for y in pools.get(cls[u], []) if _lp(u)[2] in mp(y)]
        if not cand:
            left.append(u)
            continue
        for d in range(draws):
            pick[d][u] = cand[int(g.integers(len(cand)))]
    done = [u for u in todo if u not in left]
    groups = {}
    for u in done:
        groups.setdefault((u[1], 0 if u[0] in ("head", "hchan") else 1), []).append(u)
    S = [{} for _ in range(draws)]
    for key in sorted(groups):
        rows, slots = [], []
        for d in range(draws):
            by = {}
            for u in groups[key]:
                by.setdefault(id(pick[d][u]), (pick[d][u], []))[1].append(u)
            for y, us in by.values():
                m = mp(y)
                keep = [at(w, m[_lp(w)[2]]) for w in units if _lp(w)[2] in m]
                over = [(at(w, m[_lp(w)[2]]), v) for w, v in S[d].items() if _lp(w)[2] in m]
                rows.append({"ids": y["ids"], "keep": keep, "over": over, "grab": [at(u, m[_lp(u)[2]]) for u in us]})
                slots.append((d, us))
        _, got = abl.run(rows)
        for (d, us), vals in zip(slots, got):
            for u, v in zip(us, vals):
                S[d][u] = v
    lp, _ = abl.run([{"ids": x["ids"], "keep": units, "over": list(S[d].items())} for d in range(draws)])
    return lp, done, left


def scrub_dense(abl: Ablator, x: dict, M: dict, K: dict, pools: list, draws: int, seed: int) -> tuple:
    """scrub() for circuits of any size, on dense masks (x's coordinates): M the circuit, K its scrubbed units (one class:
    pools, prompts that may each supply any of them where its position has a role). Per draw each unit of K takes one
    pool prompt at random among those where its position has a role; layer by layer (heads, then neurons), the circuit
    model runs on each pool prompt with every earlier scrubbed unit set to its scrubbed value where that unit has a role
    there, up to the layer, and the layer's units assigned to that prompt read their values there. Returns ([draws, V]
    log-probabilities on x with every scrubbed unit at its value, units scrubbed, units left live for want of a role;
    a head counts once, its channels taking one prompt together)."""
    dev, dt = abl.pt.dev, abl.pt.core.model.embed_tokens.weight.dtype
    g = torch.Generator().manual_seed(seed)
    Tx = len(x["ids"])
    maps = [role_map(x, y) for y in pools]
    pxy = [(torch.tensor(sorted(m), dtype=torch.long), torch.tensor([m[p] for p in sorted(m)], dtype=torch.long)) for m in maps]
    valid = [[j for j, m in enumerate(maps) if p in m] for p in range(Tx)]
    A = [{} for _ in range(draws)]                  # (per draw and kind, each unit's pool prompt; -1 not scrubbed; a
    hd = abl.hd                                     # head's channels take one prompt together, as the head would)
    units_of = {"neuron": lambda k: k, "head": lambda k: k.view(k.shape[0], k.shape[1], -1, hd).any(-1)}
    left = 0
    for d in range(draws):
        for kind, k in K.items():
            ku = units_of[kind](k)                  # [L, T, units]
            a = torch.full(ku.shape, -1, dtype=torch.int16)
            for p in range(Tx):
                sel = ku[:, p]
                n = int(sel.sum())
                if not n:
                    continue
                if not valid[p]:
                    left += n if d == 0 else 0
                    continue
                col = a[:, p]
                col[sel] = torch.tensor(valid[p], dtype=torch.int16)[torch.randint(len(valid[p]), (n,), generator=g)]
            if kind == "head":                      # (back to channels: only the circuit's own channels of each head)
                a = a[..., None].expand(-1, -1, -1, hd).reshape(k.shape).clone()
                a[~k] = -1
            A[d][kind] = a
    done_n = int((units_of["neuron"](A[0]["neuron"] >= 0)).sum() if "neuron" in K else 0) + int(
        units_of["head"](A[0]["head"] >= 0).sum() if "head" in K else 0) if draws else 0
    S = [{k: torch.zeros(v.shape, dtype=dt, device=dev) for k, v in M.items()} for _ in range(draws)]
    D_ = [{k: torch.zeros(v.shape, dtype=torch.bool, device=dev) for k, v in M.items()} for _ in range(draws)]
    Md = {k: v.to(dev) for k, v in M.items()}
    for layer in range(abl.L):
        for kind in ("head", "neuron"):
            if kind not in K or not K[kind][layer].any():
                continue
            rows, slots = [], []
            for d in range(draws):
                for j in sorted(set(A[d][kind][layer].unique().tolist()) - {-1}):
                    px, py = pxy[j]
                    y, Ty = pools[j], len(pools[j]["ids"])
                    m_y = {k: torch.zeros((abl.L, Ty) + v.shape[2:], dtype=torch.bool, device=dev) for k, v in Md.items()}
                    v_y = {k: torch.zeros((abl.L, Ty) + v.shape[2:], dtype=dt, device=dev) for k, v in Md.items()}
                    s_y = {k: torch.zeros((abl.L, Ty) + v.shape[2:], dtype=torch.bool, device=dev) for k, v in Md.items()}
                    for k in Md:
                        m_y[k][:, py] = Md[k][:, px]
                        v_y[k][:, py] = S[d][k][:, px]
                        s_y[k][:, py] = D_[d][k][:, px]
                    rows.append({"ids": y["ids"], "mask": m_y, "values": v_y, "set": s_y})
                    slots.append((d, j))
            for (d, j), val in zip(slots, abl.layer_values(rows, kind, layer)):
                px, py = pxy[j]
                sel = (A[d][kind][layer] == j).to(dev)                 # [Tx, width]
                tmp = torch.zeros(sel.shape, dtype=dt, device=dev)
                tmp[px.to(dev)] = val[py.to(dev)].to(dt)
                S[d][kind][layer] = torch.where(sel, tmp, S[d][kind][layer])
                D_[d][kind][layer] |= sel
    lp = abl.logp([{"ids": x["ids"], "mask": Md, "values": S[d], "set": D_[d]} for d in range(draws)])
    return lp, done_n, left


def r4(abl: Ablator, tok, rec: dict, q: dict, keys: list, draws: int = R4_DRAWS, n_heldout: int = DRAWS) -> dict:
    """A circuit's R4 on its own clean prompt and its held-out prompts (R3's): the scrubbed circuit model's KL and task
    loss, beside the full model, the empty circuit and the unscrubbed circuit (R3) on the same prompts."""
    t0 = time.perf_counter()
    units = record_units(rec)
    bad = invalid(units, abl.L, abl.D, abl.H, abl.H * abl.hd, len(rec["ids"]))
    if bad:
        return {"id": rec.get("id"), "error": f"{len(bad)} units outside the model or the prompt, e.g. {bad[0]}", "summary": {}}
    user, key = q.get("user") or rec.get("user"), q.get("key") or rec.get("key")
    own = prompt(tok, user, key, q, q.get("paraphrase", 0))
    if own is None or own["ids"] != list(rec["ids"]):
        return {"id": rec.get("id"), "error": "the record's prompt is not its question's wording with its key", "summary": {}}
    words = {q.get("paraphrase", 0): user}
    for f in q.get("family") or []:
        words[f.get("paraphrase")] = f["user"]
    targets = [("own", own)] + [("heldout", prompt(tok, user.replace(key, k), k, q, own["template"]))
                                for k, _ in heldout(tok, rec, q, keys, n_heldout)]
    targets = [(s_, y) for s_, y in targets if y is not None]
    seed = zlib.crc32(("r4" + str(rec.get("id"))).encode())
    out = []
    for split, x in targets:
        k = x["key"]
        others = [kk for kk, _ in heldout(tok, {**rec, "id": f"{rec.get('id')}/{k}", "key": k, "selection_keys": [key]},
                                          {**q, "user": user.replace(key, k), "key": k}, keys, n_heldout)]
        pools = {"key": [y for j, w in words.items() if j != x["template"]
                         for y in [prompt(tok, w.replace(key, k), k, q, j) if w.count(key) == 1 else None] if y is not None],
                 "template": [y for kk in others for y in [prompt(tok, user.replace(key, kk), kk, q, x["template"])] if y is not None]}
        if rec.get("hypothesis"):                   # (a stated hypothesis: per-unit classes, unit lists)
            cls = classes(rec, units, x["span"])
            lp, done, left = scrub(abl, x, units, cls, pools, draws, seed + len(out))
            done, left = len(done), len(left)
        else:                                       # (the tags' hypothesis: every unit from the key's first token on is
            M = dense(units, abl.L, len(x["ids"]), abl.D, abl.H, abl.hd)       # "key"; dense masks, any size)
            K = {k: v.clone() for k, v in M.items()}
            for k in K:
                K[k][:, :x["span"][0]] = False
            lp, done, left = scrub_dense(abl, x, M, K, pools["key"], draws, seed + len(out))
        base = abl.logp([{"ids": x["ids"], "keep": None}, {"ids": x["ids"], "keep": []},
                         {"ids": x["ids"], "mask": dense(units, abl.L, len(x["ids"]), abl.D, abl.H, abl.hd)}])
        sets = answer_contrast.token_sets_many(abl.pt.core, tok, [(x["ids"], base[0], None, None, None)])[0]
        m = [_metrics(base[0], lp[d], sets["answer"]) for d in range(len(lp))]
        out.append({"split": split, "key": k, "answer_word": sets["answer_word"], "scrubbed": done, "left_live": left,
                    "pools": {c: len(v) for c, v in pools.items()},
                    **{a: _metrics(base[0], base[j], sets["answer"]) for j, a in enumerate(("full", "empty", "circuit"))},
                    "scrub": {f: float(np.mean([z[f] for z in m])) for f in ("kl", "ce", "top1")},
                    "scrub_worst": {f: float(max(z[f] for z in m)) for f in ("kl", "ce")}})
    res = {"id": rec.get("id"), "task": rec.get("task") or q.get("task"), "size": sizes(units), "prompts": out,
           "seconds_found": rec.get("seconds"), "seconds": round(time.perf_counter() - t0, 2), "summary": {}}
    for split in ("own", "heldout"):
        P = [p for p in out if p["split"] == split]
        if not P:
            continue
        s_ = {"n": len(P), "scrubbed_share": float(np.mean([p["scrubbed"] / max(1, len(units)) for p in P]))}
        for a in ("full", "empty", "circuit", "scrub"):
            s_[a] = {f: float(np.mean([p[a][f] for p in P])) for f in ("kl", "ce", "top1")}
            s_[a]["pass"] = float(np.mean([p[a]["ce"] - p["full"]["ce"] <= TARGET for p in P]))
            s_[a]["pass_abs"] = float(np.mean([p[a]["ce"] <= TARGET for p in P]))
        d_ce = s_["empty"]["ce"] - s_["full"]["ce"]
        for a in ("circuit", "scrub"):
            s_[a]["recovered_ce"] = (s_["empty"]["ce"] - s_[a]["ce"]) / d_ce if d_ce > 1e-9 else None
            s_[a]["recovered_kl"] = 1 - s_[a]["kl"] / s_["empty"]["kl"] if s_["empty"]["kl"] > 1e-9 else None
        res["summary"][split] = s_
    return res


# ---------------------------------------------------------------- R5

def load_sims(path) -> dict:
    """Simulated activations: a JSONL file of {"unit": [kind, layer, pos, index] as record_units' tuples (a head:
    ["head", layer, head, pos]), "ids": the prompt's token ids, "value": the simulated value (a neuron's activation, or a
    head channel's)}, as {(unit tuple, tuple(ids)): value}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[(tuple(d["unit"]), tuple(d["ids"]))] = d["value"]
    return out


def r5(abl: Ablator, tok, rec: dict, q: dict, keys: list, sim, n_heldout: int = DRAWS) -> dict:
    """A circuit's R5 (Gao's feature-quality substitution): the background everywhere, nothing computed, each circuit
    unit set to its label's simulated value on the prompt (sim(unit, ids) -> a value, or None for a unit with no label,
    which then stays at the background: an unlabelled unit explains nothing); beside it the frozen circuit (each unit
    at its full-model value: the ceiling a perfect label reaches), the full model and the empty circuit, on the own clean
    prompt and the held-out prompts (R3's)."""
    t0 = time.perf_counter()
    units = record_units(rec)
    bad = invalid(units, abl.L, abl.D, abl.H, abl.H * abl.hd, len(rec["ids"]))
    if bad:
        return {"id": rec.get("id"), "error": f"{len(bad)} units outside the model or the prompt, e.g. {bad[0]}", "summary": {}}
    prompts = [("own", list(rec["ids"]))] + [("heldout", ids) for _, ids in heldout(tok, rec, q, keys, n_heldout)]
    lp, got = abl.run([{"ids": ids, "keep": None, "grab": units} for _, ids in prompts])
    labelled = [[u for u in units if sim(u, ids) is not None] for _, ids in prompts]
    rows = []
    for (_, ids), g, lab in zip(prompts, got, labelled):
        rows += [{"ids": ids, "keep": []}, {"ids": ids, "keep": [], "over": list(zip(units, g))},
                 {"ids": ids, "keep": [], "over": [(u, torch.as_tensor(sim(u, ids), dtype=torch.float32).reshape(-1)) for u in lab]}]
    lq = abl.logp(rows).view(len(prompts), 3, -1)
    sets = answer_contrast.token_sets_many(abl.pt.core, tok, [(ids, lp[j], None, None, None) for j, (_, ids) in enumerate(prompts)])
    out = []
    for j, (split, ids) in enumerate(prompts):
        out.append({"split": split, "answer_word": sets[j]["answer_word"], "labelled": len(labelled[j]),
                    "full": _metrics(lp[j], lp[j], sets[j]["answer"]),
                    **{a: _metrics(lp[j], lq[j, x], sets[j]["answer"]) for x, a in enumerate(("empty", "frozen", "simulated"))}})
    res = {"id": rec.get("id"), "task": rec.get("task") or q.get("task"), "size": sizes(units), "prompts": out,
           "labelled_share": float(np.mean([p["labelled"] / max(1, len(units)) for p in out])),
           "seconds_found": rec.get("seconds"), "seconds": round(time.perf_counter() - t0, 2)}
    res["summary"] = summarize(out, ["full", "empty", "frozen", "simulated"])
    return res


# ---------------------------------------------------------------- coverage

COVER_T = 128          # tokens a window (the first COVER_T tokens of a corpus document outside the means' sample)
COVER_BOUNDS = (1.0, 0.3, 0.1, 0.03)   # nats: the KL bounds whose unit counts are read off each target's curve
COVER_TAU = 0.05       # nats: the hinge's target KL during the descent (half the 0.1 bound)
COVER_LAMBDA = 2e-5    # nats a unit: the mask's price per unit
COVER_STEPS = 60
COVER_LR = 0.2
COVER_INIT = 3.0       # sigmoid(3) = 0.95: every unit nearly live at the start (KL near 0)
COVER_ROWS = 8         # targets a batch (consecutive positions of one window)
COVER_GRID = 2         # curve sizes per doubling of k
COVER_ST = 0.5         # the last share of the steps run hard masks forward (the units with logits above 0 live, the rest
# at the background) with the sigmoid's gradient back (straight-through, as the circuit-sparsity paper's pruning): the soft
# masks' KL stays at the hinge while their hard threshold missed it (0.6B: KL 0.03 soft against 0.14 hard at 131k units)
COVER_BISECT = 6       # halvings between a bound's first passing curve size and the failing size before it


def windows(n: int, T: int = COVER_T, seed: int = 1, sample_npz: str | None = None, corpus: Path = CORPUS) -> list:
    """n windows [(document, ids)]: the first T tokens of corpus documents of at least T tokens, in a seeded order, none
    of the means' sample (sample_npz's documents)."""
    tok = np.load(corpus / "tokens.npy", mmap_mode="r")
    off = np.load(corpus / "offsets.npy")
    ends = off[1:] if len(off) > 1 else off
    banned = set(np.load(sample_npz)["documents"].tolist()) if sample_npz else set()
    out = []
    for j in np.random.default_rng(seed).permutation(len(ends)):
        if int(j) in banned or ends[j] - off[j] < T:
            continue
        out.append((int(j), np.asarray(tok[off[j]:off[j] + T]).astype(np.int64).tolist()))
        if len(out) == n:
            break
    return out


class Masked:
    """Differentiable masked runs of an HF Qwen3: per row, a mask in [0, 1] over every neuron [L, T, D] and every head
    [L, T, H] at every position; a unit's value is its background plus mask x (its computed value - its background)."""

    def __init__(self, model, means: Means, mode: str | None = None):
        self.core = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = self.core.config
        self.layers = list(self.core.model.layers)
        self.L, self.D, self.H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev = self.core.model.embed_tokens.weight.device
        self.means, self.mode = means, mode

    def logp(self, x: torch.Tensor, mn: torch.Tensor | None, mh: torch.Tensor | None, pos: torch.Tensor) -> torch.Tensor:
        """[B, V] float32 log-probabilities at each row's position pos[b] of x [B, T]; mn [B, L, T, D] and mh [B, L, T, H]
        the masks (None: the full model)."""
        hs = []

        def pre(kind, i):
            def f(mod, args):
                h = args[0]
                B, T, W = h.shape
                mu = self.means.table(kind, i, T, h.device, h.dtype, self.mode)[None]
                if kind == "neuron":
                    m = mn[:, i].to(h.device, h.dtype)
                else:
                    m = mh[:, i].to(h.device, h.dtype).repeat_interleave(self.hd, -1)
                return (mu + m * (h - mu),) + tuple(args[1:])
            return f
        if mn is not None:
            for i, layer in enumerate(self.layers):
                hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre("neuron", i)))
                hs.append(layer.self_attn.o_proj.register_forward_pre_hook(pre("head", i)))
        try:
            h = self.core.model(input_ids=x.to(self.dev)).last_hidden_state
            h = h[torch.arange(len(pos), device=h.device), pos.to(h.device)]
            return torch.log_softmax(self.core.lm_head(h.to(self.core.lm_head.weight.device)).float(), -1)
        finally:
            for k in hs:
                k.remove()


def _kl(lp_full: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
    return (lp_full.exp() * (lp_full - lp)).sum(-1)


def mask_search(mk: Masked, x: torch.Tensor, item: list, pos: torch.Tensor, lpf: torch.Tensor, live: torch.Tensor,
                bounds=COVER_BOUNDS) -> dict:
    """Learned masks under the background, one mask per item over every neuron and head at every position, an item's
    rows (prompts of x [R, T], read at pos [R] against the full model's lpf [R, V]) sharing its mask; live [I, 1, T, 1]:
    the positions each item's mask may keep. Descent: soft masks, then (COVER_ST) hard masks forward with the sigmoid's
    gradient, on the sum over rows of relu(KL - COVER_TAU) + COVER_LAMBDA x sum(m). Then per item the units in the order of
    their final logits and the KL of each prefix (the worst over the item's rows) measured with hard masks on a log grid
    (COVER_GRID sizes a doubling, the mask's own live set among them), each bound's first crossing halved COVER_BISECT
    times. Returns {"th": (neuron logits [I, L, T, D], head logits [I, L, T, H]), "thr": the logit of an item's k-th unit,
    "curves", "at" (per item {bound: fewest units found}), "k_mask", "units_total", "history", "seconds_opt"}."""
    t0 = time.perf_counter()
    I, T = live.shape[0], x.shape[1]
    it = torch.tensor(item, device=mk.dev)
    th_n = torch.full((I, mk.L, T, mk.D), COVER_INIT, device=mk.dev, requires_grad=True)
    th_h = torch.full((I, mk.L, T, mk.H), COVER_INIT, device=mk.dev, requires_grad=True)
    opt = torch.optim.Adam([th_n, th_h], lr=COVER_LR)
    lpf = lpf.to(mk.dev)
    hist = []
    for step in range(COVER_STEPS):
        sn, sh = torch.sigmoid(th_n) * live, torch.sigmoid(th_h) * live
        if step >= (1 - COVER_ST) * COVER_STEPS:   # (hard forward, sigmoid gradient)
            mn = ((th_n > 0).float() * live - sn).detach() + sn
            mh = ((th_h > 0).float() * live - sh).detach() + sh
        else:
            mn, mh = sn, sh
        kl = _kl(lpf, mk.logp(x, mn[it], mh[it], pos))
        size = sn.sum((1, 2, 3)) + sh.sum((1, 2, 3))
        loss = torch.relu(kl - COVER_TAU).sum() + COVER_LAMBDA * size.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == COVER_STEPS - 1:
            hist.append([step, [round(float(v), 4) for v in kl.detach()], [round(float(v), 1) for v in size.detach()]])
    t_opt = time.perf_counter() - t0
    with torch.no_grad():
        th_n, th_h = th_n.detach(), th_h.detach()
        lv = [int(live[j, 0, :, 0].sum()) for j in range(I)]          # (positions an item's mask may keep: a prefix)
        n_units = [p * mk.L * (mk.D + mk.H) for p in lv]
        vs = [torch.cat([th_n[j, :, :lv[j]].reshape(-1), th_h[j, :, :lv[j]].reshape(-1)]).sort(descending=True).values
              for j in range(I)]
        thr = lambda j, k: float(vs[j][k - 1]) if k > 0 else float("inf")  # noqa: E731
        k_mask = [int((th_n[j, :, :lv[j]] > 0).sum() + (th_h[j, :, :lv[j]] > 0).sum()) for j in range(I)]
        rows_of = [[r for r, j in enumerate(item) if j == jj] for jj in range(I)]

        rows_fwd = max(COVER_ROWS, min(64, 2048 // T))     # (hard masks as booleans: short prompts take more rows)

        def measure(jobs):                          # (item, k) -> the worst KL over the item's rows, hard masks
            flat = [(j, k, r) for j, k in jobs for r in rows_of[j]]
            got = []
            for a in range(0, len(flat), rows_fwd):
                part = flat[a:a + rows_fwd]
                mn = torch.stack([(th_n[j] >= thr(j, k)) & (live[j] > 0) for j, k, _ in part])
                mh = torch.stack([(th_h[j] >= thr(j, k)) & (live[j] > 0) for j, k, _ in part])
                rr = torch.tensor([r for _, _, r in part], device=mk.dev)
                got += _kl(lpf[rr], mk.logp(x[rr], mn, mh, pos[rr])).tolist()
            out, at_ = [], 0
            for j, k in jobs:
                n = len(rows_of[j])
                out.append(max(got[at_:at_ + n]))
                at_ += n
            return out
        jobs = []
        for j in range(I):
            n = n_units[j]
            jobs += [(j, k) for k in sorted({0, k_mask[j]} | {min(n, int(round(2 ** (e / COVER_GRID))))
                                                             for e in range(0, int(COVER_GRID * math.log2(n)) + 2)})]
        curves = [[] for _ in range(I)]
        for (j, k), z in zip(jobs, measure(jobs)):
            curves[j].append([k, z])
        brk = {}                                    # (item, bound) -> [failing k, passing k]: the first crossing
        for j, c in enumerate(curves):
            c.sort()
            for bd in bounds:
                e = next((e for e, (k, z) in enumerate(c) if z <= bd), None)
                if e is not None and e > 0:
                    brk[(j, bd)] = [c[e - 1][0], c[e][0]]
        for _ in range(COVER_BISECT):               # (halvings in lockstep over items and bounds)
            mids = [(key, (lo + hi) // 2) for key, (lo, hi) in brk.items() if hi - lo > 1]
            if not mids:
                break
            for (key, mid), z in zip(mids, measure([(key[0], mid) for key, mid in mids])):
                curves[key[0]].append([mid, z])
                brk[key][1 if z <= key[1] else 0] = mid
        for c in curves:
            c.sort()
        at = [{str(bd): (brk[(j, bd)][1] if (j, bd) in brk else next((k for k, z in c if z <= bd), None)) for bd in bounds}
              for j, c in enumerate(curves)]
    return {"th": (th_n, th_h), "thr": thr, "curves": curves, "at": at, "k_mask": k_mask, "units_total": n_units,
            "positions": lv, "history": hist, "seconds_opt": t_opt, "rows_of": rows_of}


def cover_batch(mk: Masked, ids: list, ts: list, lp_full: torch.Tensor, log=None) -> list:
    """Coverage of targets ts (positions of one window ids, each predicting the next token): mask_search with one item a
    target (its one row read at the target, its mask over the positions up to it); per target the fewest units found
    for each bound, and at the 0.1 bound a random set of that size (the same count per kind, layer and position, other
    units) and where the units sit."""
    t0 = time.perf_counter()
    T, B = max(ts) + 1, len(ts)
    x = torch.tensor([ids[:T]] * B, device=mk.dev)
    pos = torch.tensor(ts, device=mk.dev)
    live = torch.zeros(B, 1, T, 1, device=mk.dev)
    for b, t in enumerate(ts):
        live[b, :, :t + 1] = 1
    lpf = lp_full[ts].to(mk.dev)
    ms = mask_search(mk, x, list(range(B)), pos, lpf, live)
    (th_n, th_h), thr, curves, at = ms["th"], ms["thr"], ms["curves"], ms["at"]
    out = []
    with torch.no_grad():
        rnd_rows = []                               # (the random control at the 0.1 bound)
        for b, t in enumerate(ts):
            k1 = at[b].get("0.1")
            if not k1:
                continue
            g = torch.Generator(device="cpu").manual_seed(zlib.crc32(f"{ids[0]}-{t}".encode()))
            pair = []
            for th, w in ((th_n, mk.D), (th_h, mk.H)):
                keep = (th[b] >= thr(b, k1)) & (live[b] > 0)
                r = torch.zeros_like(keep, dtype=torch.float)
                cnt = keep.sum(-1).cpu()                                 # [L, T]
                for i, p in cnt.nonzero().tolist():
                    r[i, p, torch.randperm(w, generator=g)[:int(cnt[i, p])].to(r.device)] = 1
                pair.append(r)
            rnd_rows.append((b, pair))
        rnd = {}
        for a in range(0, len(rnd_rows), COVER_ROWS):
            part = rnd_rows[a:a + COVER_ROWS]
            rows = torch.tensor([b for b, _ in part], device=mk.dev)
            kl = _kl(lpf[rows], mk.logp(x[rows], torch.stack([p[0] for _, p in part]), torch.stack([p[1] for _, p in part]), pos[rows]))
            rnd.update({b: z for (b, _), z in zip(part, kl.tolist())})
        for b, t in enumerate(ts):
            k1 = at[b].get("0.1")
            sn = (th_n[b] >= thr(b, k1))[:, :t + 1] if k1 else None
            sh = (th_h[b] >= thr(b, k1))[:, :t + 1] if k1 else None
            k0 = ms["k_mask"][b]
            out.append({"t": t, "token": ids[t], "next": ids[t + 1] if t + 1 < len(ids) else None, "units_total": ms["units_total"][b],
                        "k_mask": k0, "kl_mask": next((z for k, z in curves[b] if k == k0), None),
                        "kl_empty": curves[b][0][1], "k": at[b], "kl_random_at_0.1": rnd.get(b), "curve": curves[b],
                        "at_0.1": None if sn is None else {
                            "neurons": int(sn.sum()), "heads": int(sh.sum()), "same_position": int(sn[:, t].sum() + sh[:, t].sum()),
                            "by_layer": (sn.sum((1, 2)) + sh.sum((1, 2))).tolist(),
                            "by_distance": [int(sn[:, t - d].sum() + sh[:, t - d].sum()) for d in range(t + 1)]},
                        "history": [[h[0], h[1][b], h[2][b]] for h in ms["history"]]})
    sec = time.perf_counter() - t0
    for o in out:
        o["seconds"] = round(sec / B, 2)
        o["seconds_opt"] = round(ms["seconds_opt"] / B, 2)
    if log:
        log(f"  targets {ts[0]}..{ts[-1]}: k at 0.1 nats " + ", ".join(str(o["k"]["0.1"]) for o in out) + f" ({sec:.1f} s)")
    return out


TSEARCH_KEYS = 8      # prompts a task circuit is searched on: the question's wording with this many keys (its own first)


def tsearch(mk: Masked, tok, qs: list, keys: dict, out: Path, batch: int = 4, log=None) -> None:
    """The R3 reference circuit of each question's wording as a task (a method row for the leaderboard): mask_search with
    one item a question, its rows the question's wording with TSEARCH_KEYS keys of its task (its own key, its corrupted
    key, then others in a seeded order; all of one token count), one mask over every position, read at the answer; the
    fewest units found within QSEARCH_BOUND on every one of them, written as a circuit file whose "selection_keys" are
    those keys, so r3 and r4 score it on the wording's other keys only."""
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for q in qs:
        dst = out / f"{q['id'].replace('/', '__')}.json"
        if dst.exists() or not q.get("key") or q["user"].count(q["key"]) != 1:
            continue
        c = prompt_ids(tok, q["user"], q)
        cand = [k for k in keys.get(q.get("task"), []) if k not in (q["key"], q.get("corrupt_key"))]
        np.random.default_rng(zlib.crc32(("t" + q["id"]).encode())).shuffle(cand)
        ks, rows = [], []
        for k in [q["key"], q.get("corrupt_key")] + cand:
            ids = prompt_ids(tok, q["user"].replace(q["key"], k), q) if k else None
            if ids is not None and len(ids) == len(c):
                ks.append(k)
                rows.append(ids)
            if len(ks) == TSEARCH_KEYS:
                break
        if len(ks) >= 2:
            jobs.append((q, c, ks, rows, dst))
    for T in sorted({len(j[1]) for j in jobs}):
        js = [j for j in jobs if len(j[1]) == T]
        for a in range(0, len(js), batch):
            part = js[a:a + batch]
            t0 = time.perf_counter()
            x = torch.tensor([ids for _, _, _, rows, _ in part for ids in rows], device=mk.dev)
            item = [j for j, (_, _, _, rows, _) in enumerate(part) for _ in rows]
            pos = torch.full((len(x),), T - 1, device=mk.dev)
            with torch.no_grad():
                lpf = mk.logp(x, None, None, pos)
            live = torch.ones(len(part), 1, T, 1, device=mk.dev)
            ms = mask_search(mk, x, item, pos, lpf, live, bounds=(QSEARCH_BOUND,) + tuple(b for b in COVER_BOUNDS if b != QSEARCH_BOUND))
            sec = (time.perf_counter() - t0) / len(part)
            (th_n, th_h), thr = ms["th"], ms["thr"]
            for j, (q, c, ks, rows, dst) in enumerate(part):
                kb = ms["at"][j].get(str(QSEARCH_BOUND))
                neurons = heads = []
                if kb:
                    z = thr(j, kb)
                    neurons = [[i, p, n] for i, p, n in (th_n[j] >= z).nonzero().tolist()]
                    heads = [[i, p, h] for i, p, h in (th_h[j] >= z).nonzero().tolist()]
                corr = prompt_ids(tok, q["corrupt_user"], q) if q.get("corrupt_user") else []
                rec = {**{f: q.get(f) for f in ("id", "task", "key", "corrupt_key", "user", "corrupt_user", "paraphrase")},
                       "ids": c, "corrupted": corr, "neurons": neurons, "heads": heads, "met": bool(kb), "seconds": round(sec, 2),
                       "selection_keys": ks, "finder": "ami.tsearch",
                       "r3search": {"k": ms["at"][j], "k_mask": ms["k_mask"][j], "units_total": ms["units_total"][j],
                                    "curve": ms["curves"][j], "keys": ks}}
                dst.write_text(json.dumps(rec))
                if log:
                    log(f"{q['id']}: {len(neurons)} neurons + {len(heads)} heads within {QSEARCH_BOUND} nats on {len(ks)} keys "
                        f"(k {ms['at'][j]}, of {ms['units_total'][j]}), {sec:.1f} s")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


QSEARCH_BOUND = 0.1   # nats: a question's R3 circuit is the fewest units found whose worst KL over its two prompts is within this


def qsearch(mk: Masked, tok, qs: list, out: Path, batch: int = 8, log=None) -> None:
    """The R3 reference circuit of each question (a method row for the leaderboard): mask_search with one item a question,
    its clean and corrupted prompts sharing one mask over every position, read at the answer; the fewest units found
    within QSEARCH_BOUND written as a circuit file (its "neurons" [[layer, pos, neuron]] and "heads" [[layer, pos, head]],
    the question's fields), for r3 and r4 to score on held-out prompts."""
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for q in qs:
        dst = out / f"{q['id'].replace('/', '__')}.json"
        c, k = prompt_ids(tok, q["user"], q), prompt_ids(tok, q["corrupt_user"], q)
        if not dst.exists() and len(c) == len(k):
            jobs.append((q, c, k, dst))
    jobs.sort(key=lambda j: len(j[1]))
    for T in sorted({len(j[1]) for j in jobs}):
        js = [j for j in jobs if len(j[1]) == T]
        for a in range(0, len(js), batch):
            part = js[a:a + batch]
            t0 = time.perf_counter()
            x = torch.tensor([ids for _, c, k, _ in part for ids in (c, k)], device=mk.dev)
            pos = torch.full((len(x),), T - 1, device=mk.dev)
            with torch.no_grad():
                lpf = mk.logp(x, None, None, pos)
            live = torch.ones(len(part), 1, T, 1, device=mk.dev)
            ms = mask_search(mk, x, [j for j in range(len(part)) for _ in (0, 1)], pos, lpf, live, bounds=(QSEARCH_BOUND,) + tuple(
                b for b in COVER_BOUNDS if b != QSEARCH_BOUND))
            sec = (time.perf_counter() - t0) / len(part)
            (th_n, th_h), thr = ms["th"], ms["thr"]
            for j, (q, c, k, dst) in enumerate(part):
                kb = ms["at"][j].get(str(QSEARCH_BOUND))
                neurons = heads = []
                if kb:
                    z = thr(j, kb)
                    neurons = [[i, p, n] for i, p, n in (th_n[j] >= z).nonzero().tolist()]
                    heads = [[i, p, h] for i, p, h in (th_h[j] >= z).nonzero().tolist()]
                rec = {**{f: q.get(f) for f in ("id", "task", "key", "corrupt_key", "user", "corrupt_user", "paraphrase")},
                       "ids": c, "corrupted": k, "neurons": neurons, "heads": heads, "met": bool(kb), "seconds": round(sec, 2),
                       "finder": "ami.qsearch", "r3search": {"k": ms["at"][j], "k_mask": ms["k_mask"][j],
                                                             "units_total": ms["units_total"][j], "curve": ms["curves"][j],
                                                             "history": [[h[0], h[1][2 * j:2 * j + 2], h[2][j]] for h in ms["history"]]}}
                dst.write_text(json.dumps(rec))
                if log:
                    log(f"{q['id']}: {len(neurons)} neurons + {len(heads)} heads within {QSEARCH_BOUND} nats "
                        f"(k {ms['at'][j]}, of {ms['units_total'][j]}), {sec:.1f} s")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


def coverage(mk: Masked, wins: list, out: Path, log=None) -> None:
    """coverage of every target of every window (cover_batch), one file per window in out."""
    out.mkdir(parents=True, exist_ok=True)
    for doc, ids in wins:
        dst = out / f"doc{doc}.json"
        if dst.exists():
            continue
        with torch.no_grad():
            lp_full = mk.logp(torch.tensor([ids], device=mk.dev).expand(len(ids) - 1, -1), None, None,
                              torch.arange(len(ids) - 1, device=mk.dev)).cpu()
        res = []
        ts = list(range(len(ids) - 1))
        for a in range(0, len(ts), COVER_ROWS):
            res += cover_batch(mk, ids, ts[a:a + COVER_ROWS], lp_full, log=log)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        dst.write_text(json.dumps({"document": doc, "ids": ids, "targets": res}))
        if log:
            ks = [r["k"]["0.1"] for r in res if r["k"]["0.1"] is not None]
            log(f"doc {doc}: {len(res)} targets, median k at 0.1 nats {np.median(ks) if ks else None}")


# ---------------------------------------------------------------- the leaderboard

def wilson(k: int, n: int, z: float = 1.96) -> list | None:
    """The 95% Wilson interval of a rate k / n."""
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, mid - half), min(1.0, mid + half)]


def bootstrap(xs: list, draws: int = 2000) -> list | None:
    xs = np.array([x for x in xs if x is not None and math.isfinite(x)], float)
    if len(xs) < 2:
        return None
    m = xs[np.random.default_rng(0).integers(0, len(xs), (draws, len(xs)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def board(dirs: list) -> list:
    """Leaderboard lines (dicts) of scored sets: per set and rung, n circuits, the mean of each record's held-out summary
    with its bootstrap interval, the random control's mean, and the median size and finder seconds."""
    lines = []
    for d in dirs:
        recs = [r for r in (json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json")) if p.name != "meta.json")
                if r.get("summary")]
        meta = json.loads((Path(d) / "meta.json").read_text()) if (Path(d) / "meta.json").exists() else {}
        met = [bool(r["met_r0"]) for r in recs if "met_r0" in r and r["met_r0"] is not None]
        if met and meta.get("rung", "R3") == "R3":   # (R0: the finder's own pass on its pair, from the records)
            lines.append({"method": meta.get("label", Path(d).name), "rung": "R0", "inputs": "own", "n": len(met),
                          "pass": float(np.mean(met)), "pass_ci": wilson(sum(met), len(met)),
                          "units_median": float(np.median([r["size"]["units"] for r in recs]))})
        for split in ("own", "heldout"):
            S = [r["summary"][split] for r in recs if split in r.get("summary", {})]
            if not S:
                continue
            rung = meta.get("rung", "R3")
            arm = {"R4": "scrub", "R5": "simulated"}.get(rung, "circuit")
            row = {"method": meta.get("label", Path(d).name), "rung": rung, "inputs": split, "n": len(S),
                   "background": meta.get("mode", MODE)}
            for m in ("recovered_ce", "recovered_kl", "ce", "kl", "pass", "pass_abs"):
                xs = [s[arm].get(m) for s in S]
                xs = [x for x in xs if x is not None]
                row[m] = float(np.mean(xs)) if xs else None
                row[m + "_ci"] = bootstrap(xs)
                rs = [s.get("random", {}).get(m) for s in S]
                rs = [x for x in rs if x is not None]
                row[m + "_random"] = float(np.mean(rs)) if rs else None
            row["pass_ci"] = wilson(int(round(sum(s[arm]["pass"] * s["n"] for s in S))), sum(s["n"] for s in S))
            if all("frozen" in s for s in S):
                row["recovered_ce_frozen"] = float(np.mean([s["frozen"]["recovered_ce"] for s in S]))
            for m in ("ce", "kl"):
                row[m + "_empty"] = float(np.mean([s["empty"][m] for s in S]))
                row[m + "_full"] = float(np.mean([s["full"][m] for s in S]))
            row["full_pass_abs"] = float(np.mean([s["full"].get("pass_abs", 0.0) for s in S]))
            if rung == "R4":
                row["recovered_ce_r3"] = float(np.mean([s["circuit"]["recovered_ce"] for s in S if s["circuit"].get("recovered_ce") is not None]))
                row["scrubbed_share"] = float(np.mean([s["scrubbed_share"] for s in S]))
            row["units_median"] = float(np.median([r["size"]["units"] for r in recs]))
            row["edges_median"] = float(np.median([r["size"]["edges"] for r in recs]))
            secs = [r.get("seconds_found") for r in recs if r.get("seconds_found") is not None]
            row["seconds_median"] = float(np.median(secs)) if secs else None
            lines.append(row)
    return lines


def tsv(lines: list) -> str:
    cols = ["method", "rung", "inputs", "background", "n", "recovered_ce", "recovered_ce_ci", "recovered_ce_random", "ce", "ce_ci",
            "ce_random", "ce_empty", "ce_full", "pass", "pass_ci", "pass_random", "pass_abs", "full_pass_abs", "recovered_ce_frozen", "kl", "kl_ci", "kl_random", "kl_empty",
            "recovered_kl", "recovered_kl_random", "recovered_ce_r3", "scrubbed_share", "units_median", "edges_median", "seconds_median"]
    f = lambda v: "" if v is None else (f"[{v[0]:.3f}, {v[1]:.3f}]" if isinstance(v, list) else f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    return "\n".join(["\t".join(cols)] + ["\t".join(f(x.get(c)) for c in cols) for x in lines])


# ---------------------------------------------------------------- command line

def load(model_id: str, dtype: str, device: str = "auto"):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if dtype == "auto":
        cfg = AutoConfig.from_pretrained(model_id)
        n = cfg.num_hidden_layers * (4 * cfg.hidden_size ** 2 + 3 * cfg.hidden_size * cfg.intermediate_size)
        dtype = "bfloat16" if n >= 4e9 else "float32"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype),
                                                 device_map="auto" if device == "auto" else None)
    model = (model if device == "auto" else model.to(device)).eval().requires_grad_(False)
    return model, AutoTokenizer.from_pretrained(model_id), dtype


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("out")
    s.add_argument("--tokens", type=int, default=1_100_000)
    s.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("means")
    m.add_argument("sample")
    m.add_argument("out")
    r = sub.add_parser("r3", aliases=["r4", "r5"])
    r.add_argument("records")
    r.add_argument("out")
    r.add_argument("--means", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--devset", default=str(DEVSET))
    r.add_argument("--pool", default=str(POOL))
    r.add_argument("--mode", default=MODE, choices=("bucket", "global"))
    r.add_argument("--draws", type=int, default=DRAWS)
    r.add_argument("--random", type=int, default=RANDOM)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--sims", default=None, help="r5: the simulated activations (load_sims' JSONL)")
    c = sub.add_parser("cover")
    c.add_argument("out")
    c.add_argument("--means", required=True)
    c.add_argument("--texts", default=str(Path.home() / "mpd-data" / "ami" / "cover_windows.npz"), help="windows' file")
    c.add_argument("--windows", type=int, default=16)
    c.add_argument("--first", type=int, default=0, help="the windows' file from this window on")
    w = sub.add_parser("windows")
    w.add_argument("out")
    w.add_argument("--n", type=int, default=64)
    w.add_argument("--sample", default=str(Path.home() / "mpd-data" / "ami" / "means_sample_1m.npz"))
    c.add_argument("--mode", default=MODE, choices=("bucket", "global"))
    qq = sub.add_parser("qsearch", aliases=["tsearch"])
    qq.add_argument("out")
    qq.add_argument("--means", required=True)
    qq.add_argument("--devset", default=str(DEVSET))
    qq.add_argument("--split", default="dev")
    qq.add_argument("--stride", type=int, default=1)
    qq.add_argument("--limit", type=int, default=0)
    qq.add_argument("--batch", type=int, default=8)
    qq.add_argument("--mode", default=MODE, choices=("bucket", "global"))
    qq.add_argument("--pool", default=str(POOL))
    b = sub.add_parser("board")
    b.add_argument("dirs", nargs="+")
    for p in (m, r, c, qq):
        p.add_argument("--model", default="Qwen/Qwen3-8B")
        p.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
        p.add_argument("--device", default="auto")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda x: print(f"{time.time() - t0:7.1f}s {x}", flush=True)  # noqa: E731
    if a.cmd == "sample":
        d = sample(Path(a.out), a.tokens, a.seed)
        log(f"{len(d['offsets']) - 1} documents, {len(d['tokens']):,} tokens -> {a.out}")
        return
    if a.cmd == "windows":
        wins = windows(a.n, sample_npz=a.sample)
        np.savez(a.out, documents=np.array([d for d, _ in wins]), ids=np.array([x for _, x in wins], dtype=np.int64))
        log(f"{len(wins)} windows of {COVER_T} tokens -> {a.out}")
        return
    if a.cmd == "board":
        print(tsv(board(a.dirs)))
        return
    cf._threads()
    model, tok, dtype = load(a.model, a.dtype, a.device)
    if a.cmd == "means":
        d = np.load(a.sample)
        off, tk = d["offsets"], d["tokens"]
        docs = [tk[off[j]:off[j + 1]].tolist() for j in range(len(off) - 1)]
        acc = collect(model, docs, log=log)
        torch.save({**acc, "model": a.model, "dtype": dtype, "sample": str(a.sample), "documents": len(docs),
                    "tokens": int(sum(map(len, docs)))}, a.out)
        log(f"means of {a.model} over {sum(map(len, docs)):,} tokens -> {a.out}")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("qsearch", "tsearch"):
        qs = [q for q in load_questions(a.devset).values() if q.get("split", a.split) == a.split][::a.stride][:a.limit or None]
        (out / "meta.json").write_text(json.dumps({"model": a.model, "dtype": dtype, "means": a.means, "mode": a.mode,
                                                   "bound": QSEARCH_BOUND, "tau": COVER_TAU, "lambda": COVER_LAMBDA,
                                                   "steps": COVER_STEPS, "questions": len(qs)}))
        mk = Masked(model, Means(a.means, abl_dev(model)), mode=a.mode)
        if a.cmd == "tsearch":
            tsearch(mk, tok, qs, task_keys({**load_questions(a.pool), **load_questions(a.devset)}), out, max(1, a.batch // 2), log=log)
        else:
            qsearch(mk, tok, qs, out, a.batch, log=log)
        return
    if a.cmd == "cover":
        (out / "meta.json").write_text(json.dumps({"model": a.model, "dtype": dtype, "means": a.means, "mode": a.mode, "T": COVER_T,
                                                   "bounds": COVER_BOUNDS, "tau": COVER_TAU, "lambda": COVER_LAMBDA,
                                                   "steps": COVER_STEPS, "lr": COVER_LR, "windows": a.windows}))
        d = np.load(a.texts)
        wins = list(zip(d["documents"].tolist(), d["ids"].tolist()))[a.first:a.first + a.windows]
        coverage(Masked(model, Means(a.means, abl_dev(model)), mode=a.mode), wins, out, log=log)
        return
    (out / "meta.json").write_text(json.dumps({"label": a.label, "rung": a.cmd.upper(), "records": a.records, "means": a.means, "mode": a.mode,
                                               "model": a.model, "dtype": dtype, "draws": a.draws, "random": a.random,
                                               "target": TARGET}))
    abl = Ablator(model, tok, Means(a.means, abl_dev(model)), mode=a.mode)
    qs = load_questions(a.devset)
    keys = task_keys({**load_questions(a.pool), **qs})
    paths = sorted(p for p in Path(a.records).glob("*.json") if p.name != "meta.json")[:a.limit or None]
    sims = load_sims(a.sims) if a.cmd == "r5" else None
    for p in paths:
        dst = out / p.name
        if dst.exists():
            continue
        rec = json.loads(p.read_text())
        if rec.get("skipped") or not rec.get("ids"):
            continue
        q = qs.get(rec.get("id")) or {k: rec.get(k) for k in ("user", "key", "corrupt_key", "task", "prefill", "system") if rec.get(k)}
        if a.cmd == "r5":
            res = r5(abl, tok, rec, q, keys.get(rec.get("task") or q.get("task"), []), lambda u, ids: sims.get((u, tuple(ids))), a.draws)
            dst.write_text(json.dumps(res))
            s = res["summary"]
            log(f"{rec.get('id')}: " + (f"{res['size']['units']} units, {res['labelled_share']:.2f} labelled; own ce simulated "
                                        f"{s['own']['simulated']['ce']:.2f} frozen {s['own']['frozen']['ce']:.2f} (empty "
                                        f"{s['own']['empty']['ce']:.2f})" if "own" in s else str(res.get("error"))))
            continue
        if a.cmd == "r4":
            res = r4(abl, tok, rec, q, keys.get(rec.get("task") or q.get("task"), []), R4_DRAWS, a.draws)
            dst.write_text(json.dumps(res))
            s = res["summary"]
            if "own" in s:
                log(f"{rec.get('id')}: {res['size']['units']} units, {s['own']['scrubbed_share']:.2f} scrubbed; own ce scrub "
                    f"{s['own']['scrub']['ce']:.2f} circuit {s['own']['circuit']['ce']:.2f} (empty {s['own']['empty']['ce']:.2f}, full "
                    f"{s['own']['full']['ce']:.2f})" + (f"; held-out {s['heldout']['n']} scrub {s['heldout']['scrub']['ce']:.2f}"
                                                     if "heldout" in s else "") + f", {res['seconds']} s")
            else:
                log(f"{rec.get('id')}: {res.get('error')}")
            continue
        res = r3(abl, tok, rec, q, keys.get(rec.get("task") or q.get("task"), []), a.draws, a.random)
        dst.write_text(json.dumps(res))
        s = res["summary"]
        if res.get("error"):
            log(f"{rec.get('id')}: {res['error']}")
            continue
        log(f"{rec.get('id')}: {res['size']['units']} units; own ce {s['own']['circuit']['ce']:.2f} (empty "
            f"{s['own']['empty']['ce']:.2f}, full {s['own']['full']['ce']:.2f}), kl {s['own']['circuit']['kl']:.2f}"
            + (f"; held-out {s['heldout']['n']} ce {s['heldout']['circuit']['ce']:.2f} random {s['heldout']['random']['ce']:.2f}"
               if "heldout" in s else "") + f", {res['seconds']} s")


def abl_dev(model):
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    return core.model.embed_tokens.weight.device


if __name__ == "__main__":
    main()
