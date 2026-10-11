"""Carry a prompt change using only selected VPD nodes (#2951).

All omitted activations are held at their values on the original prompt. Token
embeddings and W - V U still run; ``diagnostics`` separates these bypasses.
KL sums target positions, averages changed prompts, and is reported in bits.
Run: MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python carry.py search --task hard9308
     MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python carry.py explain --split hard
         --n 50 --k 8 --out /Users/user/mpd-data/graph_oracle/carry/hard50_matrix.jsonl
Use --edit "heads:L2:2,3,4;mlp:L1;edge:L2:all" for whole-measurement edits.
Use --units head for activation-slice units; summarize reads completed JSONL
rows and reports distributions, recurring sets, timing, and full-set failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path
import plistlib
import subprocess
import threading
import time

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

import native

LN2 = math.log(2)
COPYCHECK = Path.home() / "mpd-data/graph_oracle/copycheck_vr4.jsonl"


def parse_edit(spec):
    """Validate and canonicalize model edits (layer indices are zero based)."""
    if spec is None or spec == "":
        return None
    edits = {}
    for part in spec.split(";"):
        m = re.fullmatch(r"(heads|mlp|edge):L(\d+)(?::(all|\d+(?:,\d+)*))?", part.strip())
        if not m or (m[1] == "mlp" and m[3] is not None) or (m[1] == "edge" and m[3] != "all") or (m[1] == "heads" and (m[3] is None or m[3] == "all")):
            raise ValueError(f"invalid model edit: {part!r}")
        key = (m[1], int(m[2]))
        edits.setdefault(key, set()).update(map(int, m[3].split(",")) if m[1] == "heads" else [])
    return ";".join(f"{kind}:L{layer}" + (":" + ",".join(map(str, sorted(edits[kind, layer]))) if kind == "heads" else ":all" if kind == "edge" else "")
                    for kind, layer in sorted(edits))


class _ModelEdit(TorchFunctionMode):
    """Edit full attention head outputs and MLP writes, including VPD remainders.

    Zeroing PV head rows is exactly subtracting each head's o_proj input
    slice times its output weights. It also covers sliced HeadCarry products.
    Fused attention is expanded so references and direct runs share semantics.
    """
    def __init__(self, nat, spec):
        self.nat, self.spec = nat, spec
        self.heads, self.edges, self.mlp = {}, set(), set()
        for part in spec.split(";"):
            kind, layer, *arg = part.split(":")
            layer = int(layer[1:])
            if layer >= nat.target.n_layer:
                raise ValueError(f"edit layer L{layer} outside model")
            if kind == "heads":
                hs = list(map(int, arg[0].split(",")))
                if max(hs) >= nat.target.n_head:
                    raise ValueError("edit head outside model")
                self.heads[layer] = hs
            elif kind == "edge":
                self.edges.add(layer)
            else:
                self.mlp.add(layer)
        self.zero_products, self.zero_weights = {}, {}
        for layer in self.mlp:
            st = nat.target.site(native.site_name(layer, "down_proj"))
            self.zero_products.update({id(st.U): layer, id(st.delta_T): layer})
            self.zero_weights[st.W.data_ptr()] = layer
        self.calls, self.pv_layer = 0, None
        self.seen_heads, self.seen_edges, self.seen_mlp = set(), set(), set()

    def pattern(self, out):
        layer = self.calls % self.nat.target.n_layer
        self.calls += 1
        self.pv_layer = layer
        if layer in self.edges:
            out = out.clone()
            out[..., -1, :-1] = 0.0  # no renormalization; self edge preserved
            self.seen_edges.add(layer)
        return out

    def values(self, out, layer):
        if layer in self.heads:
            out = out.clone()
            out[:, self.heads[layer], :, :] = 0.0
            self.seen_heads.add(layer)
        return out

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.scaled_dot_product_attention:
            q, k, v = args[:3]
            # Only the model's ordinary MHA is supported. Fail on new kernels/options.
            if len(args) > 3 or kwargs.get("dropout_p", 0.0) or kwargs.get("enable_gqa", False):
                raise RuntimeError("unsupported fused attention under model edit")
            scale = kwargs.get("scale")
            logits = (q @ k.transpose(-1, -2)) * (scale if scale is not None else 1 / math.sqrt(q.shape[-1]))
            mask = kwargs.get("attn_mask")
            if kwargs.get("is_causal", False):
                mask = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).tril()
            if mask is not None:
                logits = logits.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else logits + mask
            pattern = self.pattern(logits.softmax(-1))
            layer, self.pv_layer = self.pv_layer, None
            return self.values(pattern @ v, layer)
        if func.__name__ in ("matmul", "__matmul__"):
            x, w = args[:2]
            out = func(*args, **kwargs)
            if x.dim() == w.dim() == 4 and self.pv_layer is not None:
                layer, self.pv_layer = self.pv_layer, None
                return self.values(out, layer)
            if id(w) in self.zero_products or (w.dim() == 2 and w.data_ptr() in self.zero_weights):
                layer = self.zero_products.get(id(w), self.zero_weights.get(w.data_ptr()))
                self.seen_mlp.add(layer)
                return out * 0.0
            return out
        out = func(*args, **kwargs)
        if func.__name__ == "softmax" and out.dim() == 4:
            return self.pattern(out)
        return out

    def check(self):
        if not self.calls or self.calls % self.nat.target.n_layer or self.pv_layer is not None or self.seen_heads != set(self.heads) or self.seen_edges != self.edges or self.seen_mlp != self.mlp:
            raise RuntimeError("model edit interception incomplete; kernel or forward path bypassed edit")


@contextmanager
def model_edit(nat, spec=None):
    """Apply edits to every forward in a measurement; discard unkeyed prompt caches.

    Studies must be created and used within this context. Nested identical
    edits are allowed; changing the model inside a study fails explicitly.
    """
    spec = parse_edit(spec)
    active = getattr(nat, "_graph_edit", None)
    if active is not None:
        if spec is not None and spec != active:
            raise ValueError("cannot nest different model edits")
        yield
        return
    if spec is None:
        yield
        return
    mode = _ModelEdit(nat, spec)
    # Check each Native and target forward separately: one observable forward
    # must not conceal another that silently switched to a compiled kernel.
    restored = []
    def guard(original):
        def checked(*args, **kwargs):
            mode.calls, mode.pv_layer = 0, None
            mode.seen_heads.clear()
            mode.seen_edges.clear()
            mode.seen_mlp.clear()
            result = original(*args, **kwargs)
            mode.check()
            return result
        return checked
    for obj, name in ((nat, "run"), (nat, "reference"), (nat.vpd, "target_forward"), (nat.target, "hidden")):
        restored.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, guard(getattr(obj, name)))
    nat._rec = (None, None, None)
    nat._graph_edit = spec
    try:
        with mode:
            yield
        mode.check()
    finally:
        for obj, name, had, old in restored:
            if had:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        nat._rec = (None, None, None)
        nat._graph_edit = None


def _check_study_edit(study):
    if study.edit != getattr(study.nat, "_graph_edit", None):
        raise RuntimeError("carry study used under a different model edit; cached states invalid")


def kl(lp, lq):
    # Do not clamp: small negative values expose floating point disagreement.
    return (lp.exp() * (lp - lq)).sum(-1).sum(-1) / LN2


def synchronize(dev):
    if str(dev).startswith("mps"):
        torch.mps.synchronize()
    elif str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def masks(nat, T, nodes):
    """Node triples or matrix -> GPU bool [T,C] to live-node masks.

    Triple metadata is encoded once on the host. Assignment is advanced tensor
    indexing, never one device assignment per Python node.
    """
    if isinstance(nodes, dict):
        if set(nodes) != set(nat.names):
            raise ValueError("mask dictionary must contain every matrix")
        for n in nat.names:
            if nodes[n].shape != (T, nat.C[n]) or nodes[n].dtype != torch.bool:
                raise ValueError(f"invalid mask for {n}")
            if nodes[n].device != nat.target.site(n).V.device:
                raise ValueError("masks must already be on the model device")
        return nodes
    out = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    rows = np.asarray(list(nodes), dtype=object).reshape(-1, 3)
    if not len(rows):
        return out
    if not set(rows[:, 0]).issubset(nat.names):
        raise ValueError("unknown matrix")
    for n in nat.names:
        pairs = rows[rows[:, 0] == n, 1:].astype(np.int64)
        if len(pairs):
            if (pairs < 0).any() or (pairs[:, 0] >= T).any() or (pairs[:, 1] >= nat.C[n]).any():
                raise ValueError(f"node outside {n}'s [T,C]")
            ix = torch.as_tensor(pairs, device=nat.dev)
            out[n][ix[:, 0], ix[:, 1]] = True
    return out


class _Mix(TorchFunctionMode):
    """Replace precisely Native.fixed's hard patches by differentiable mixes.

    Native has no soft-patch API. The identity lookup matches ONLY the patch
    mask objects supplied by us, with no global monkeypatch or model mutation.
    No reuse/window slicing is used in this mode. The live activation remains
    in the autograd graph, including its dependence on earlier mixed nodes.
    """
    def __init__(self, patch, coefficients):
        self.by_id = {id(patch[n][0]): coefficients[n] for n in patch}
        self.seen = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.where and id(args[0]) in self.by_id:
            self.seen.add(id(args[0]))
            a = self.by_id[id(args[0])]
            base, live = args[1:]
            return a * live + (1 - a) * base
        return func(*args, **kwargs)


class _Remainders(TorchFunctionMode):
    """Capture or freeze each original remainder product z @ delta_T."""
    def __init__(self, nat, values=None, embedding=None):
        self.names = {id(nat.target.site(n).delta_T): n for n in nat.names}
        self.values = {} if values is None else values
        self.capture = values is None
        self.embedding = embedding
        self.wte = nat.target.wte
        self.seen = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.__name__ in ("matmul", "__matmul__") and id(args[1]) in self.names:
            n = self.names[id(args[1])]
            self.seen.add(n)
            if not self.capture:
                return self.values[n].expand(args[0].shape[0], -1, -1)
            result = func(*args, **kwargs)
            self.values[n] = result.detach()
            return result
        if self.embedding is not None and func.__name__ == "__getitem__" and args[0] is self.wte:
            return self.embedding
        return func(*args, **kwargs)


class Carry:
    """Cache the original record and changed-prompt references for a study."""
    def __init__(self, nat, ids, targets, prompts, chunk_size=None, *, space=None):
        if not prompts or not targets:
            raise ValueError("need at least one prompt and target")
        if any(len(x) != len(ids) for x in prompts):
            raise ValueError("changed prompts must have the original length")
        self.edit = getattr(nat, "_graph_edit", None)
        self.nat, self.ids, self.targets, self.prompts = nat, ids, targets, prompts
        self.T = len(ids)
        self.rec = nat._model_record(ids)[0]
        self.states = nat._model_record(ids)[1]
        with torch.no_grad():
            self.lp = [nat.reference([x], targets) for x in prompts]
        # Bound inference working memory, chiefly activations and attention.
        # This is a resource limit, not a search/scoring hyperparameter.
        H = nat.target.n_head
        per = 4 * (8 * H * self.T**2 + 12 * self.T * sum(nat.C.values()))
        self.chunk_size = max(1, (512 * 1024**2) // per) if chunk_size is None else chunk_size
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if space is not None:
            if space.granularity == "head":
                raise ValueError("head slices require HeadCarry")
            self.space = space
            with torch.no_grad():
                self.writes = {n: self.rec[n] @ nat.target.site(n).U for n in nat.names}
            self.first_change = min((p for x in prompts for p, (a, b) in enumerate(zip(ids, x)) if a != b), default=self.T)
            _prepare_unit_prompts(self)
            if chunk_size is None:
                self.chunk_size = _unit_chunk(nat, self.T, len(prompts))

    @torch.no_grad()
    def measure(self, sets):
        _check_study_edit(self)
        if hasattr(self, "space"):
            return _measure_units(self, sets)
        if not len(sets):
            return torch.empty(0, device=self.nat.dev)
        nat = self.nat
        results, start = [], 0
        while start < len(sets):
            B = min(self.chunk_size, len(sets) - start)
            try:
                live = [masks(nat, self.T, s) for s in sets[start:start + B]]
                patch = {n: (~torch.stack([s[n] for s in live]), self.rec[n]) for n in nat.names}
                del live
                total = torch.zeros(B, device=nat.dev)
                for x, lp in zip(self.prompts, self.lp):
                    lq = nat.run(x, self.targets, nat._full_plan(self.T, B), *nat._one(), patch=patch)
                    total += kl(lp, lq)
                results.append(total / len(self.prompts))
                start += B
                del patch, lq
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or B == 1:
                    raise
                self.chunk_size = max(1, B // 2)
                # An OOM traceback can retain the entire failed forward graph.
                exc.__traceback__ = None
                live = patch = lq = total = None
                native.free(nat.dev)
                if str(nat.dev).startswith("cuda"):
                    torch.cuda.empty_cache()
        return torch.cat(results)

    def mixed(self, x, coefficients):
        _check_study_edit(self)
        if hasattr(self, "space"):
            return _mixed_units(self, x, coefficients)
        patch = {n: (torch.ones_like(self.rec[n], dtype=torch.bool), self.rec[n]) for n in self.nat.names}
        mode = _Mix(patch, coefficients)
        with mode:
            lq = self.nat.run(x, self.targets, self.nat._full_plan(self.T, 1), *self.nat._one(), patch=patch)
        if len(mode.seen) != len(patch):
            raise RuntimeError("Native's patch operation changed; soft patches were not all intercepted")
        return lq


def carry_kl(nat, ids, targets, prompts, sets, *, chunk_size=None):
    """GPU tensor: one mean changed-prompt carry KL in bits per node set.

    Sets are batched, with bounded working memory and OOM chunk backoff. Accepts
    triples or GPU mask dictionaries. Only metadata starts on the host; model
    activations, masks, distributions and KL arithmetic stay on the device.
    """
    return Carry(nat, ids, targets, prompts, chunk_size).measure(sets)


class Space:
    """Compact, deterministic node order at candidate positions, on the GPU."""
    def __init__(self, nat, T, positions):
        positions = sorted(set(positions))
        if not positions or min(positions) < 0 or max(positions) >= T:
            raise ValueError("positions must be nonempty and inside the text")
        self.nat, self.T, self.positions = nat, T, positions
        self.pos = torch.tensor(positions, device=nat.dev)
        self.slices, offset = {}, 0
        for n in nat.names:
            size = len(positions) * nat.C[n]
            self.slices[n] = slice(offset, offset + size)
            offset += size
        self.size = offset

    def expand(self, a):
        out = {}
        for n, sl in self.slices.items():
            # index_copy is differentiable with respect to a.
            z = torch.zeros(self.T, self.nat.C[n], device=a.device, dtype=a.dtype)
            out[n] = z.index_copy(0, self.pos, a[sl].view(len(self.positions), -1))
        return out

    def nodes(self, active):
        out = set()
        for n, sl in self.slices.items():
            pairs = active[sl].view(len(self.positions), -1).nonzero().cpu().tolist()
            out.update((n, self.positions[p], c) for p, c in pairs)
        return out


class Curve(list):
    """Public (size, KL) curve, with compact measured sets for inspection."""
    def __init__(self, space):
        super().__init__()
        self.space, self.active = space, []

    def record(self, active, score):
        self.append((int(active.sum()), float(score)))
        self.active.append(active.clone())

    def node_set(self, i):
        return self.space.nodes(self.active[i])

    unit_set = node_set

    @property
    def unit_sets(self):
        return [self.unit_set(i) for i in range(len(self))]

    @property
    def position_sets(self):
        return [sorted({u[1] if isinstance(u, Subcomponent) else u[-1] for u in self.unit_set(i)}) for i in range(len(self))]


def attribution(study, space, active):
    """d(KL)/da at a=1 for live nodes; removed coefficients are held at zero.

    Removed nodes have coefficient zero. Prompts share coefficients; backward
    runs one prompt at a time to bound graph memory. Upstream dependencies are
    differentiated, not detached replacement activations from a reference run.
    """
    a = active.float().detach().requires_grad_(True)
    grad = torch.zeros_like(a)
    for x, lp in zip(study.prompts, study.lp):
        coefficients = {n: v[None] for n, v in space.expand(a).items()}
        loss = kl(lp, study.mixed(x, coefficients)).sum() / len(study.prompts)
        grad += torch.autograd.grad(loss, a)[0]
        del loss, coefficients
    return grad


def _search(study, positions):
    space = Space(study.nat, study.T, positions)
    active = torch.ones(space.size, dtype=torch.bool, device=study.nat.dev)
    curve = Curve(space)
    curve.record(active, study.measure([space.expand(active)])[0])
    batch = 1
    remaining = space.size
    while remaining:
        # Magnitude identifies locally uninfluential nodes. A large positive
        # derivative predicts improvement only for an infinitesimal removal;
        # dropping its node entirely can overshoot and increase KL substantially.
        cost = attribution(study, space, active).abs()
        cost = cost.masked_fill(~active, float("inf"))
        # Grow removals geometrically, then halve the remaining set down to zero.
        count = min(batch, max(1, remaining // 2))
        remove = torch.argsort(cost, stable=True)[:count]
        active[remove] = False
        remaining -= count
        curve.record(active, study.measure([space.expand(active)])[0])
        batch *= 2
        print(f"  {remaining:6d} {curve[-1][1]:.9g}", flush=True)
    return curve


def search(nat, ids, targets, prompts, positions):
    """Measured geometric pruning; return all (node count, carry KL) points.

    Rank by gradient magnitude. Every removal is committed and measured, even
    if KL worsens. There is no
    acceptance threshold, tuned stopping rule or assumption of monotonicity.
    """
    return _search(Carry(nat, ids, targets, prompts), positions)


@torch.no_grad()
def diagnostics(study):
    nat, T = study.nat, study.T
    full = {n: torch.ones(T, nat.C[n], device=nat.dev, dtype=torch.bool) for n in nat.names}
    empty, all_nodes = study.measure([set(), full]).tolist()
    original = nat.reference([study.ids], study.targets)
    movement = torch.stack([kl(lp, original)[0] for lp in study.lp]).mean()
    capture = _Remainders(nat)
    with capture:
        native_original = nat.run(study.ids, study.targets, nat._full_plan(T, 1), *nat._one())
    if capture.seen != set(nat.names):
        raise RuntimeError("did not capture all remainder products")
    patch = {n: (torch.ones_like(study.rec[n], dtype=torch.bool), study.rec[n]) for n in nat.names}
    frozen, frozen_embedding, shifts, frozen_errors, stable_full, quadratic, full_errors = [], [], [], [], [], [], []
    emb = nat.target.wte[torch.tensor(study.ids, device=nat.dev)]
    for x, lp in zip(study.prompts, study.lp):
        full_lq = nat.run(x, study.targets, nat._full_plan(T, 1), *nat._one())
        d = full_lq - lp
        # For normalized p,q, sum p*(expm1(d)-d) equals KL(p||q).
        # This form exposes the float32 normalization/cancellation floor of
        # the prototype's sum p*(-d), without changing the reported metric.
        stable_full.append((lp.exp() * (torch.expm1(d) - d)).sum() / LN2)
        quadratic.append((lp.exp() * d.square()).sum() / (2 * LN2))
        full_errors.append(d.abs().max())
        mode = _Remainders(nat, capture.values)
        with mode:
            lq = nat.run(x, study.targets, nat._full_plan(T, 1), *nat._one(), patch=patch)
        if mode.seen != set(nat.names):
            raise RuntimeError("did not freeze all remainder products")
        frozen.append(kl(lp, lq)[0])
        live = nat.run(x, study.targets, nat._full_plan(T, 1), *nat._one(), patch=patch)
        shifts.append(kl(lq, live)[0])
        mode = _Remainders(nat, capture.values, emb)
        with mode:
            lq = nat.run(x, study.targets, nat._full_plan(T, 1), *nat._one(), patch=patch)
        frozen_embedding.append(kl(lp, lq)[0])
        frozen_errors.append((lq - native_original).abs().max())
    fixed = float(torch.stack(frozen).mean())
    return {"movement_bits": float(movement), "empty_bits": empty,
            "native_movement_bits": float(torch.stack([kl(lp, native_original)[0] for lp in study.lp]).mean()),
            "all_nodes_bits": all_nodes, "empty_frozen_remainders_bits": fixed,
            "all_nodes_stable_kl_bits": float(torch.stack(stable_full).mean()),
            "all_nodes_quadratic_kl_bits": float(torch.stack(quadratic).mean()),
            "all_nodes_max_logprob_error": float(torch.stack(full_errors).max()),
            "empty_frozen_remainders_and_embedding_bits": float(torch.stack(frozen_embedding).mean()),
            "remainder_kl_reduction_bits": fixed - empty,
            "remainder_distribution_shift_bits": float(torch.stack(shifts).mean()),
            "frozen_vs_native_original_max_logprob_error": float(torch.stack(frozen_errors).max()),
            "note": "Remainder effect is a KL difference with the changed embedding present, not an additive information decomposition."}


@torch.no_grad()
def alternatives(nat, ids, source, ranks=(0, 1, 2)):
    if not 0 < source < len(ids):
        raise ValueError("source must have a preceding prediction position")
    lp = nat.reference([ids[:source]], [source - 1])[0, 0]
    ranks = tuple(ranks)
    if not ranks or any(not isinstance(r, int) or r < 0 for r in ranks):
        raise ValueError("ranks must be nonempty nonnegative integers")
    if max(ranks) >= lp.numel() - 1:
        raise ValueError("alternative rank outside vocabulary")
    order = lp.topk(min(lp.numel(), max(ranks) + 2)).indices.tolist()
    others = [int(z) for z in order if z != ids[source]]
    alts = [others[r] for r in ranks]
    return [ids[:source] + [z] + ids[source + 1:] for z in alts]


class Subcomponent(tuple):
    """A (matrix, position, index) triple, distinct from legacy head triples.

    Tuple equality, hashing and JSON serialization retain the public node API.
    """
    def __new__(cls, name, position, index):
        return super().__new__(cls, (name, position, index))


class UnitSpace:
    """VPD subcomponents (matrix, position, index), or explicit legacy units.

    ``candidates`` restricts subcomponent membership; omitted keys are empty.
    It preserves index order within each matrix/position. Large unrestricted
    spaces require explicit candidates to avoid accidental text-wide expansion.
    Legacy matrix units are (matrix, position); attention head slices are
    (matrix, head, position), acting before the live VPD remainder, as before.
    """
    MAX_UNRESTRICTED = 100_000

    def __init__(self, nat, T, positions, granularity="subcomponent", candidates=None):
        if granularity not in ("subcomponent", "matrix", "head"):
            raise ValueError("granularity must be subcomponent, matrix or head")
        positions = sorted(set(positions))
        if not positions or min(positions) < 0 or max(positions) >= T:
            raise ValueError("positions must be inside the text")
        if candidates is not None and granularity != "subcomponent":
            raise ValueError("candidates apply only to subcomponents")
        self.nat, self.T, self.positions, self.granularity = nat, T, positions, granularity
        if granularity == "subcomponent":
            if candidates is None:
                if len(positions) * sum(nat.C.values()) > self.MAX_UNRESTRICTED:
                    raise ValueError("large subcomponent spaces require candidates")
                candidates = {(n, p): range(nat.C[n]) for n in nat.names for p in positions}
            if any(n not in nat.names or p not in positions for n, p in candidates):
                raise ValueError("candidate matrix/position outside space")
            self.units = []
            for n in nat.names:
                for p in positions:
                    cs = list(candidates.get((n, p), ()))
                    if any(not isinstance(c, (int, np.integer)) or c < 0 or c >= nat.C[n] for c in cs):
                        raise ValueError(f"candidate index outside {n}")
                    self.units.extend(Subcomponent(n, p, int(c)) for c in dict.fromkeys(cs))
        else:
            self.units = [u for n in nat.names for p in positions
                          for u in ([(n, h, p) for h in range(nat.target.n_head)]
                                    if granularity == "head" and ".attn." in n else [(n, p)])]
        self.size = len(self.units)
        self.index = {u: j for j, u in enumerate(self.units)}
        self.by_name, self.offsets, self.flat_indices, self.compact_indices = {}, {}, {}, {}
        self.columns, self.selected_indices = {}, {}
        offset = 0
        for n in nat.names:
            us = [u for u in self.units if u[0] == n]
            count = len(us)
            self.offsets[n] = slice(offset, offset + count)
            self.by_name[n] = torch.arange(offset, offset + count, device=nat.dev)
            offset += count
            if granularity == "subcomponent":
                self.flat_indices[n] = torch.tensor([u[1]*nat.C[n]+u[2] for u in us], device=nat.dev, dtype=torch.long)
                rows = {p: j for j, p in enumerate(positions)}
                self.compact_indices[n] = torch.tensor([rows[u[1]]*nat.C[n]+u[2] for u in us], device=nat.dev, dtype=torch.long)
                cs = sorted({u[2] for u in us})
                self.columns[n] = torch.tensor(cs, device=nat.dev, dtype=torch.long)
                col = {c: j for j, c in enumerate(cs)}
                self.selected_indices[n] = torch.tensor([rows[u[1]]*len(cs)+col[u[2]] for u in us], device=nat.dev, dtype=torch.long)
        self.pos = torch.tensor(positions, device=nat.dev)

    def expand(self, a):
        """[N] or [B,N] to dense per-matrix masks; head vectors stay compact."""
        if self.granularity == "head":
            return a
        single = a.dim() == 1
        out = self.coefficients(a)
        if self.granularity == "matrix":
            out = {n: v.expand(-1, -1, self.nat.C[n]) for n, v in out.items()}
        return {n: v[0] for n, v in out.items()} if single else out

    def nodes(self, active):
        return {self.units[j] for j in active.nonzero().flatten().cpu().tolist()}

    def _coefficients(self, a, compact):
        if a.dim() == 1:
            a = a[None]
        if a.dim() != 2 or a.shape[1] != self.size:
            raise ValueError("coefficients must have shape [N] or [B,N]")
        out = {}
        rows = len(self.positions) if compact else self.T
        for n, sl in self.offsets.items():
            if self.granularity == "subcomponent":
                ix = self.compact_indices[n] if compact else self.flat_indices[n]
                z = a.new_zeros(a.shape[0], rows*self.nat.C[n])
                out[n] = z.index_copy(1, ix, a[:, sl]).view(a.shape[0], rows, self.nat.C[n])
            else:
                H = self.nat.target.n_head if self.granularity == "head" and ".attn." in n else 1
                v = a[:, sl].reshape(a.shape[0], len(self.positions), H)
                out[n] = v if compact else a.new_zeros(a.shape[0], self.T, H).index_copy(1, self.pos, v)
        return out

    def coefficients(self, a):
        """[N] or [B,N] -> [B,T,C] (legacy units: [B,T,H or 1])."""
        return self._coefficients(a, False)

    def compact(self, a):
        """Dense coefficients at the selected positions, without text-wide masks."""
        return self._coefficients(a, True)


    def _selected_coefficients(self, a):
        """Internal sparse compact coefficients, using each matrix's column union."""
        out = {}
        for n, sl in self.offsets.items():
            K = self.columns[n].numel()
            z = a.new_zeros(a.shape[0], len(self.positions)*K)
            out[n] = z.index_copy(1, self.selected_indices[n], a[:, sl]).view(a.shape[0], len(self.positions), K)
        return out


def _unit_chunk(nat, T, prompts=3):
    # One layer's intermediates coexist. All prompts share a batch; allow
    # 8 GiB of workspace within the required 24 GiB process reservation.
    width = max(max(nat.target.site(n).V.shape[0], nat.target.site(n).U.shape[1]) for n in nat.names)
    per = 4 * (8 * nat.target.n_head * T**2 + 16 * T * width)
    return max(1, (8 * 1024**3) // (prompts * per))


class _AttentionPrefix(TorchFunctionMode):
    """Reuse causal attention rows strictly before the changed position.

    Capture one original run, then shorten QK, softmax and PV to rows p:T.
    Native still receives full-size P/attention tensors. All changed rows run
    normally, including rows outside the selected units (live remainders).
    """
    def __init__(self, T, start=0, originals=None):
        self.T, self.start = T, start
        self.capture = originals is None
        self.originals = [] if originals is None else originals
        self.layer, self.value_next, self.logits = 0, False, False
        self.pattern = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.__name__ in ("matmul", "__matmul__") and args[0].dim() == 4 and args[1].dim() == 4:
            x, w = args[:2]
            if not self.value_next:
                self.value_next = self.logits = True
                return func(*args, **kwargs) if self.capture else x[..., self.start:, :] @ w
            self.value_next = False
            if self.capture:
                result = func(*args, **kwargs)
                self.originals.append((self.pattern, result.detach()))
            else:
                result = torch.cat([self.originals[self.layer][1][..., :self.start, :].expand(x.shape[0], -1, -1, -1),
                                    x[..., self.start:, :] @ w], -2)
            self.layer += 1
            return result
        if self.logits and func.__name__ == "masked_fill" and not self.capture:
            return func(args[0], args[1][self.start:, :], *args[2:], **kwargs)
        if self.logits and func.__name__ == "softmax":
            self.logits = False
            result = func(*args, **kwargs)
            if self.capture:
                self.pattern = result.detach()
                return result
            return torch.cat([self.originals[self.layer][0][..., :self.start, :].expand(result.shape[0], -1, -1, -1), result], -2)
        return func(*args, **kwargs)


def _capture_attention(study):
    mode = _AttentionPrefix(study.T)
    with torch.no_grad(), mode:
        study.nat.run(study.ids, study.targets, study.nat._full_plan(study.T, 1), *study.nat._one())
    if mode.layer != study.nat.target.n_layer:
        raise RuntimeError("attention prefix capture incomplete")
    return mode.originals


class _SparseProducts(TorchFunctionMode):
    """Skip products discarded by hard/soft patches, preserve live remainders.

    Frozen a @ U rows equal original a @ U exactly. x @ V rows outside the
    unit positions are immediately overwritten by original a. Only these two
    VPD products are shortened; attention, norms and delta products stay live.
    """
    def __init__(self, study):
        self.study = study
        self.products = {id(getattr(study.nat.target.site(n), field)): (n, field)
                         for n in study.nat.names for field in ("V", "U")
                         if not (study.space.granularity == "head" and ".attn." in n and
                                 (field == "U" or n.endswith("o_proj")))}
        self.windows = study.windows

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.__name__ == "__getitem__" and isinstance(args[1], tuple) and len(args[1]) == 2 and isinstance(args[1][1], list):
            key = tuple(args[1][1])
            if key not in self.study.indices:
                self.study.indices[key] = torch.tensor(key, device=self.study.nat.dev, dtype=torch.long)
            return func(args[0], (args[1][0], self.study.indices[key]), **kwargs)
        if func.__name__ in ("matmul", "__matmul__") and id(args[1]) in self.products:
            n, field = self.products[id(args[1])]
            x, w = args[:2]
            if x.dim() != 3:
                raise RuntimeError("unexpected Native VPD product shape")
            s = self.study.T - x.shape[1]
            if s not in self.windows:
                self.windows[s] = torch.tensor([p-s for p in self.study.space.positions if p >= s], device=x.device)
            pos = self.windows[s]
            base = self.study.rec[n] if field == "V" else self.study.writes[n]
            selected = x.index_select(1, pos) @ w
            return base[:, s:].expand(x.shape[0], -1, -1).index_copy(1, pos, selected)
        return func(*args, **kwargs)


class _PromptBatch(TorchFunctionMode):
    """Supply prompt-major embeddings to Native's shared-id batched plan.

    Native uses ids only for token embedding lookup and the reuse boundary.
    Length, targets, prefix and changed position are shared; each batch row
    gets its own changed-token embedding. No tokenizer/model mutation.
    """
    def __init__(self, study, candidates):
        self.study, self.candidates = study, candidates
        self.emb = self.batch = None
        self.seen = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.tensor and args[0] is self.study.run_ids:
            return self.study.input_ids[0]
        if func.__name__ == "__getitem__" and args[0] is self.study.nat.target.wte:
            length = len(args[1])
            self.emb = self.study.embeddings[0, -length:]
            self.batch = self.study.embeddings[:, None, -length:].expand(-1, self.candidates, -1, -1).reshape(-1, length, self.emb.shape[-1])
            return self.emb
        if func.__name__ == "__getitem__" and args[0] is self.emb and args[1] is None:
            self.seen = True
            return self.batch
        return func(*args, **kwargs)


def _prepare_unit_prompts(study):
    with torch.no_grad():
        study.input_ids = torch.tensor(study.prompts, device=study.nat.dev)
        study.embeddings = study.nat.target.wte[study.input_ids]
    study.batch_lp = torch.cat(study.lp)[:, None]
    study.windows, study.indices = {}, {}
    study.sparse = {}
    if study.space.granularity == "subcomponent":
        with torch.no_grad():
            for n, cs in study.space.columns.items():
                if cs.numel() < study.nat.C[n]:
                    st = study.nat.target.site(n)
                    study.sparse[n] = (st.V.index_select(1, cs), st.U.index_select(0, cs),
                                       study.rec[n].index_select(1, study.space.pos).index_select(2, cs))
    study.run_ids = min(study.prompts, key=lambda x: next((p for p, (a, b) in enumerate(zip(study.ids, x)) if a != b), study.T))
    study.prompt_index = {tuple(x): j for j, x in enumerate(study.prompts)}
    study.direct = True
    study.pad_norm = False
    study.causal = torch.ones(study.T, study.T, device=study.nat.dev, dtype=torch.bool).tril()
    study.mask_windows = {}
    tg = study.nat.target
    study.rope_keys = {i: tg._rope(study.states[("kv", i)].view(1, study.T, tg.n_head, tg.hd).transpose(1, 2), study.T)
                       for i in range(tg.n_layer)}
    if study.space.granularity == "head":
        with torch.no_grad():
            study.writes = {n: ((study.originals[n].sum(-2) if n.endswith("o_proj") else study.rec[n]) @ tg.site(n).U)
                            for n in study.nat.names}


def _run_units(study, coefficients, embeddings, *, compact=False, activation_gradient=False):
    """Direct exact unit-intervention evaluator, with live VPD remainders.

    Frozen VPD products use original writes; only selected positions compute
    activations. Remainders and MLP nonlinearities run on every changed row.
    Attention uses all causal keys/values, but only queries in the window.
    Prefixes are unchanged by causality. RMS is row-local; an optional padded
    reference preserves Native's full shape for numerical checks. No gradient
    dependency is dropped.
    """
    import vpd_model
    _check_study_edit(study)
    tg, T = study.nat.target, study.T
    B = next(iter(coefficients.values())).shape[0]
    s = max(0, min(study.first_change, min(study.space.positions), T - -(-64 // B), min(study.targets)))
    if s not in study.mask_windows:
        study.mask_windows[s] = ~study.causal[s:]
    x = embeddings[:, s:].expand(B, -1, -1).clone()
    L, H, hd = T-s, tg.n_head, tg.hd
    offsets = [p-s for p in study.space.positions]

    def indices(positions):
        key = tuple(positions)
        if key not in study.indices:
            study.indices[key] = torch.tensor(key, device=study.nat.dev, dtype=torch.long)
        return study.indices[key]

    def selected_rows(t, positions):
        return t.index_select(1, indices(positions))

    def norm(x, weight):
        if s and study.pad_norm:
            padded = torch.cat([x.new_zeros(B, s, x.shape[-1]), x], 1)
            return vpd_model.rms(padded, weight, tg.eps)[:, s:].contiguous()
        return vpd_model.rms(x, weight, tg.eps)

    def projection(n, inp):
        st = tg.site(n)
        if n in study.sparse and study.space.columns[n].numel() == 0:
            return study.writes[n][:, s:].expand(B, -1, -1)
        selected = selected_rows(inp, offsets)
        a = coefficients[n] if compact else selected_rows(coefficients[n], study.space.positions)
        if n in study.sparse:
            V, U, base = study.sparse[n]
            live = selected @ V
            if not compact:
                a = a.index_select(2, study.space.columns[n])
            delta = (a * live + (1-a.detach()) * base - base if activation_gradient else
                     torch.where(a, live-base, 0.) if a.dtype == torch.bool else a * (live-base))
            correction = delta @ U
            original = selected_rows(study.writes[n], study.space.positions)
            return study.writes[n][:, s:].expand(B, -1, -1).index_copy(1, indices(offsets), original + correction)
        if study.space.granularity == "head" and ".attn." in n:
            if n.endswith("o_proj"):
                parts = torch.stack([selected[..., h*hd:(h+1)*hd] @ st.V[h*hd:(h+1)*hd] for h in range(H)], -2)
                base = selected_rows(study.originals[n], study.space.positions)
                activation = (a[..., None] * parts + (1-a[..., None]) * base).sum(-2)
                live_write = activation @ st.U
            else:
                live = (selected @ st.V) @ st.U
                base = selected_rows(study.originals[n], study.space.positions)
                weight = a.repeat_interleave(hd, -1)
                live_write = weight * live + (1-weight) * base
        else:
            live = selected @ st.V
            base = selected_rows(study.rec[n], study.space.positions)
            activation = (a * live + (1-a.detach()) * base if activation_gradient else
                          torch.where(a, live, base) if a.dtype == torch.bool else a * live + (1-a) * base)
            live_write = activation @ st.U
        return study.writes[n][:, s:].expand(B, -1, -1).index_copy(1, indices(offsets), live_write)

    def heads(w):
        return w.view(B, L, H, hd).transpose(1, 2)

    def rope_window(w):
        half = hd // 2
        rotate = torch.cat([-w[..., half:], w[..., :half]], -1)
        return w * tg.cos[s:T] + rotate * tg.sin[s:T]

    for i in range(tg.n_layer):
        names = {kind: native.site_name(i, kind) for kind in vpd_model.KINDS}
        z = norm(x, tg.norms[2*i])
        qkv = [projection(names[kind], z) + z @ tg.site(names[kind]).delta_T for kind in ("q_proj", "k_proj", "v_proj")]
        q, k, v = qkv
        q, k = rope_window(heads(q)), rope_window(heads(k))
        k = torch.cat([study.rope_keys[i][:, :, :s].expand(B, -1, -1, -1), k], 2) if s else k
        prefix_v = study.states[("vs", i)][:, :s].expand(B, -1, -1)
        v = torch.cat([prefix_v, v], 1).view(B, T, H, hd).transpose(1, 2) if s else heads(v)
        if torch.is_grad_enabled() or study.edit:
            pattern = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(study.mask_windows[s], float("-inf")).softmax(-1)
            attention = pattern @ v
        else:
            attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~study.mask_windows[s], dropout_p=0.0)
        attention = attention.transpose(1, 2).reshape(B, L, -1)
        no = names["o_proj"]
        x = (x + projection(no, attention)) + attention @ tg.site(no).delta_T
        z = norm(x, tg.norms[2*i+1])
        nf, nd = names["c_fc"], names["down_proj"]
        pre = projection(nf, z) + z @ tg.site(nf).delta_T
        gelu = vpd_model.gelu_tanh(pre)
        x = (x + projection(nd, gelu)) + gelu @ tg.site(nd).delta_T
    final = selected_rows(x, [p-s for p in study.targets])
    return torch.log_softmax((vpd_model.rms(final, tg.ln_f, tg.eps) @ tg.wte.T).float(), -1)


def _mixed_units(study, x, coefficients):
    j = study.prompt_index.get(tuple(x))
    if j is None:
        emb = study.nat.target.wte[torch.tensor(x, device=study.nat.dev)][None]
    else:
        emb = study.embeddings[j:j+1]
    return _run_units(study, coefficients, emb)


@torch.no_grad()
def _measure_units(study, sets):
    _check_study_edit(study)
    out, start = [], 0
    while start < len(sets):
        valid = min(study.chunk_size, len(sets) - start)
        # MPS recompiles graphs for new batch shapes. At short windows the
        # compilation cost dominates the extra arithmetic on duplicate rows.
        capacity = min(study.chunk_size, study.space.size)
        short = study.T - min(study.first_change, min(study.targets)) <= 64
        B = max(valid, capacity) if short and study.space.granularity != "subcomponent" else valid
        try:
            a = sets[start:start+valid] if isinstance(sets, torch.Tensor) else torch.stack(sets[start:start+valid])
            if B > valid:
                a = torch.cat([a, a[-1:].expand(B-valid, -1)])
            if study.space.granularity == "head":
                a = a.float()
            base_coeff = (study.space._selected_coefficients(a) if study.direct and study.space.granularity == "subcomponent" else
                          study.space.compact(a) if study.direct else study.space.coefficients(a))
            coeff = {n: v.repeat(len(study.prompts), 1, 1) for n, v in base_coeff.items()}
            if study.direct:
                emb = study.embeddings[:, None].expand(-1, B, -1, -1).reshape(-1, study.T, study.embeddings.shape[-1])
                lq = _run_units(study, coeff, emb, compact=True)
                scores = kl(study.batch_lp, lq.reshape(len(study.prompts), B, *lq.shape[1:])).mean(0)
                out.append(scores[:valid])
                start += valid
                if str(study.nat.dev).startswith("mps") and torch.mps.driver_allocated_memory() > 12 * 1024**3:
                    native.free(study.nat.dev)
                continue
            mode = _PromptBatch(study, B)
            with mode:
                if study.space.granularity == "head":
                    lq = study.mixed(study.run_ids, coeff, reuse=True)
                else:
                    patch = {n: (~v, study.rec[n]) for n, v in coeff.items()}
                    with _AttentionPrefix(study.T, study.first_change, study.attention), _SparseProducts(study):
                        lq = study.nat.run(study.run_ids, study.targets,
                                          study.nat._full_plan(study.T, B * len(study.prompts)), *study.nat._one(),
                                          patch=patch, reuse=(study.states, study.ids))
            if not mode.seen:
                raise RuntimeError("Native embedding batch interception failed")
            scores = kl(study.batch_lp, lq.reshape(len(study.prompts), B, *lq.shape[1:])).mean(0)
            out.append(scores[:valid])
            start += valid
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or B == 1:
                raise
            study.chunk_size = max(1, B // 2)
            exc.__traceback__ = None
            a = coeff = patch = lq = mode = None
            native.free(study.nat.dev)
            if str(study.nat.dev).startswith("cuda"):
                torch.cuda.empty_cache()
    return torch.cat(out) if out else torch.empty(0, device=study.nat.dev)


class MatrixCarry(Carry):
    """A whole matrix needs a broadcast [B,T,1] mask, not [B,T,C]."""
    unit_vectors = True

    def __init__(self, nat, ids, targets, prompts, space, chunk_size=None):
        super().__init__(nat, ids, targets, prompts, chunk_size)
        if chunk_size is None:
            self.chunk_size = _unit_chunk(nat, self.T, len(prompts))
        self.space = space
        with torch.no_grad():
            self.writes = {n: self.rec[n] @ nat.target.site(n).U for n in nat.names}
        self.attention = _capture_attention(self)
        self.first_change = min((p for x in prompts for p, (a, b) in enumerate(zip(ids, x)) if a != b), default=self.T)
        _prepare_unit_prompts(self)

    def mixed(self, x, coefficients):
        if self.direct:
            return _mixed_units(self, x, coefficients)
        with _AttentionPrefix(self.T, self.first_change, self.attention), _SparseProducts(self):
            return super().mixed(x, coefficients)

    @torch.no_grad()
    def measure(self, sets):
        return _measure_units(self, sets)


# The same exact evaluator, now with per-subcomponent coefficients.
SubcomponentCarry = MatrixCarry


def candidates(nat, ids, targets, position, positions, k=None):
    """Order every matrix/position's subcomponents by |activation * KL gradient|.

    One batched backward averages the default changed prompts. ``k`` returns
    an ordering prefix for callers; ranking itself never decides membership.
    Scores are gradients of live activations, not the live-minus-original
    gradient of an interpolation coefficient. Ties retain component index order.
    """
    if k is not None and (not isinstance(k, int) or k < 0):
        raise ValueError("k must be a nonnegative integer or None")
    positions = sorted(set(positions))
    full = {(n, p): range(nat.C[n]) for n in nat.names for p in positions}
    space = UnitSpace(nat, len(ids), positions, candidates=full)
    study = SubcomponentCarry(nat, ids, targets, alternatives(nat, ids, position), space)
    with torch.enable_grad():
        a = torch.ones(space.size, device=nat.dev, requires_grad=True)
        coeff = {n: v.repeat(len(study.prompts), 1, 1) for n, v in space.compact(a).items()}
        lq = _run_units(study, coeff, study.embeddings, compact=True, activation_gradient=True)
        loss = kl(torch.cat(study.lp), lq).mean()
        grad, = torch.autograd.grad(loss, a)
    out = {}
    for n, sl in space.offsets.items():
        scores = grad[sl].abs().reshape(len(positions), nat.C[n])
        order = scores.argsort(dim=-1, descending=True, stable=True).cpu().tolist()
        for p, cs in zip(positions, order):
            out[n, p] = cs if k is None else cs[:k]
    return out


class _HeadMix(TorchFunctionMode):
    """Head-sliced VPD contributions; no global hooks or model mutations."""
    def __init__(self, nat, coefficients=None, originals=None, patch=None, positions=None):
        self.nat, self.coefficients = nat, coefficients
        self.originals = {} if originals is None else originals
        self.capture = coefficients is None
        self.pos = positions if isinstance(positions, torch.Tensor) or positions is None else torch.tensor(positions, device=nat.dev)
        self.windows = {}
        self.products = {id(nat.target.site(n).V if n.endswith("o_proj") else nat.target.site(n).U): n
                         for n in nat.names if ".attn." in n}
        self.patches = {} if patch is None else {id(v[0]): n for n, v in patch.items()}
        self.seen = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.where and id(args[0]) in self.patches:
            n = self.patches[id(args[0])]
            a = self.coefficients[n]
            return a * args[2] + (1 - a) * args[1]
        if func.__name__ in ("matmul", "__matmul__") and id(args[1]) in self.products:
            n = self.products[id(args[1])]
            self.seen.add(n)
            H, hd = self.nat.target.n_head, self.nat.target.hd
            s = 0 if self.capture else self.originals[n].shape[1] - args[0].shape[1]
            pos = self.pos
            if pos is not None:
                if s not in self.windows:
                    self.windows[s] = self.pos - s
                pos = self.windows[s]
            if n.endswith("o_proj"):
                # [B,T,H,C]: contributions, not residual output dimensions.
                x, V = args[:2]
                if pos is not None:
                    x = x.index_select(1, pos)
                parts = torch.stack([x[..., h*hd:(h+1)*hd] @ V[h*hd:(h+1)*hd] for h in range(H)], -2)
                if self.capture:
                    self.originals[n] = parts.detach()
                    return func(*args, **kwargs)
                a = self.coefficients[n][:, s:, :, None]
                base = self.originals[n][:, s:]
                if pos is not None:
                    mixed = (a.index_select(1, pos) * parts +
                             (1 - a.index_select(1, pos)) * base.index_select(1, pos)).sum(-2)
                    return base.sum(-2).expand(parts.shape[0], -1, -1).index_copy(1, pos, mixed)
                return (a * parts + (1 - a) * base).sum(-2)
            live = func(*args, **kwargs) if pos is None else args[0].index_select(1, pos) @ args[1]
            if self.capture:
                self.originals[n] = live.detach()
                return live
            a = self.coefficients[n][:, s:].repeat_interleave(hd, -1)
            base = self.originals[n][:, s:]
            if pos is not None:
                mixed = a.index_select(1, pos) * live + (1 - a.index_select(1, pos)) * base.index_select(1, pos)
                return base.expand(live.shape[0], -1, -1).index_copy(1, pos, mixed)
            return a * live + (1 - a) * self.originals[n]
        return func(*args, **kwargs)


class HeadCarry(Carry):
    unit_vectors = True

    def __init__(self, nat, ids, targets, prompts, space, chunk_size=None):
        super().__init__(nat, ids, targets, prompts, chunk_size)
        if chunk_size is None:
            self.chunk_size = _unit_chunk(nat, self.T, len(prompts))
        self.space = space
        with torch.no_grad():
            self.writes = {n: self.rec[n] @ nat.target.site(n).U for n in nat.names if ".mlp." in n}
        capture = _HeadMix(nat)
        with torch.no_grad(), capture:
            nat.run(ids, targets, nat._full_plan(self.T, 1), *nat._one())
        if capture.seen != set(capture.products.values()):
            raise RuntimeError("head product capture incomplete")
        self.originals = capture.originals
        self.attention = _capture_attention(self)
        self.first_change = min((p for x in prompts for p, (a, b) in enumerate(zip(ids, x)) if a != b), default=self.T)
        _prepare_unit_prompts(self)

    def mixed(self, x, coefficients, *, reuse=False):
        _check_study_edit(self)
        if self.direct:
            return _mixed_units(self, x, coefficients)
        patch = {n: (coefficients[n] == 0 if reuse else torch.ones_like(self.rec[n], dtype=torch.bool), self.rec[n])
                 for n in self.nat.names if ".mlp." in n}
        mode = _HeadMix(self.nat, coefficients, self.originals, None if reuse else patch, self.space.pos)
        mode.windows = self.windows
        with _AttentionPrefix(self.T, self.first_change, self.attention), _SparseProducts(self), mode:
            result = self.nat.run(x, self.targets, self.nat._full_plan(self.T, next(iter(coefficients.values())).shape[0]),
                                  *self.nat._one(), patch=patch, reuse=(self.states, self.ids) if reuse else None)
        if mode.seen != set(mode.products.values()):
            raise RuntimeError("head product interception incomplete")
        return result

    @torch.no_grad()
    def measure(self, sets):
        return _measure_units(self, sets)


def _unit_attribution(study, space, active):
    if not getattr(study, "unit_vectors", False):
        return attribution(study, space, active)
    a = active.float().detach().requires_grad_(True)
    grad = torch.zeros_like(a)
    for x, lp in zip(study.prompts, study.lp):
        loss = kl(lp, study.mixed(x, space.coefficients(a))).sum() / len(study.prompts)
        grad += torch.autograd.grad(loss, a)[0]
    return grad


def _unit_view(study, positions, granularity=None):
    """Share immutable prompt caches, give an intervention its own unit space."""
    granularity = granularity or getattr(getattr(study, "space", None), "granularity", "matrix")
    space = UnitSpace(study.nat, study.T, positions, granularity)
    if not getattr(study, "unit_vectors", False) or study.space.granularity != granularity:
        return (HeadCarry if granularity == "head" else MatrixCarry)(
            study.nat, study.ids, study.targets, study.prompts, space)
    view = copy.copy(study)
    view.space = space
    view.windows, view.indices = {}, {}
    return view


@torch.no_grad()
def _movement(study):
    original = study.nat.reference([study.ids], study.targets)
    return float(torch.stack([kl(lp, original)[0] for lp in study.lp]).mean())


def _position_mask(space, positions):
    positions = set(positions)
    return torch.tensor([u[-1] in positions for u in space.units],
                        dtype=torch.bool, device=space.nat.dev)


def _relay_attribution(study, p, targets, *, chunk_size=None):
    """Sum |d KL / da_unit| at the p/targets-only endpoint, for EVERY q.

    Zero relay coefficients keep the forward intervention identical across
    chunks. Each chunk differentiates its units, including all upstream live
    dependencies, while other relay writes stay at the original values. This
    is exactly the full-space gradient, not separate live-chunk interventions.
    OOM retries shrink chunks; a single-position OOM raises, never skips q.
    """
    base = {p, *targets}
    positions = [q for q in range(p + 1, max(targets)) if q not in base]
    if not positions:
        return []
    width = len(positions) if chunk_size is None else chunk_size
    if width < 1:
        raise ValueError("attribution chunk_size must be positive")
    result, start = [], 0
    while start < len(positions):
        qs = positions[start:start + width]
        view = _unit_view(study, [*base, *qs])
        try:
            grad = _unit_attribution(view, view.space, _position_mask(view.space, base))
            if not torch.isfinite(grad).all():
                raise RuntimeError("nonfinite relay attribution")
            costs = torch.zeros(len(view.space.positions), device=study.nat.dev)
            for sl in view.space.offsets.values():
                costs += grad[sl].abs().reshape(len(view.space.positions), -1).sum(-1)
            values = dict(zip(view.space.positions, costs.cpu().tolist()))
            result.extend((q, values[q]) for q in qs)
            start += len(qs)
            del grad, costs, view
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(qs) == 1:
                raise
            exc.__traceback__ = None
            grad = costs = view = None
            native.free(study.nat.dev)
            if str(study.nat.dev).startswith("cuda"):
                torch.cuda.empty_cache()
            width = max(1, len(qs) // 2)
            print(f"  relay attribution OOM: retry all remaining positions in chunks of {width}", flush=True)
    return result


def relays(study, p, targets, max_relays=3, *, movement=None, baseline=None,
           attribution_chunk_size=None):
    """Ordered (relay position, measured KL bits) additions, at most three.

    Screen all strictly intermediate non-target positions by summed absolute
    unit gradients, keeping 16 (ties by position). At each step measure every
    remaining candidate with ALL units at p, targets and accepted relays.
    Stop at 90% of independently measured movement, <1% improvement, or the
    relay limit. A relay is an intervention location, not a proven unique path.
    Accepts Carry, MatrixCarry or HeadCarry study-like inputs.
    """
    if max_relays < 0 or not targets or not 0 <= p < study.T or any(t < 0 or t >= study.T for t in targets):
        raise ValueError("invalid relay limit, source or targets")
    movement = _movement(study) if movement is None else movement
    base = {p, *targets}
    study = _unit_view(study, base)
    current = float(study.measure([_position_mask(study.space, base)])[0]) if baseline is None else baseline
    if max_relays == 0 or movement <= 1e-5 or current <= .1 * movement:
        return []
    scores = _relay_attribution(study, p, targets, chunk_size=attribution_chunk_size)
    candidates = [q for q, _ in sorted(scores, key=lambda z: (-z[1], z[0]))[:16]]
    if not candidates:
        return []
    view = _unit_view(study, [*base, *candidates])
    chosen, result = set(base), []
    while candidates and len(result) < max_relays and current > .1 * movement:
        trials = [_position_mask(view.space, chosen | {q}) for q in candidates]
        measured = view.measure(trials)
        best = int(measured.argmin())
        score = float(measured[best])
        if current - score < .01 * movement:
            break
        q = candidates.pop(best)
        chosen.add(q)
        result.append((q, score))
        current = score
    return result


# Public position_circuit's boolean keyword has the requested name.
_find_relays = relays


def _circuit_with_relays(study, p, targets, enabled, movement):
    base = {p, *targets}
    before = float(study.measure([_position_mask(study.space, base)])[0])
    found = _find_relays(study, p, targets, movement=movement, baseline=before) if enabled else []
    if found:
        study = _unit_view(study, [*base, *[q for q, _ in found]])
    curve = _position_circuit(study, study.space)
    curve.relays, curve.before_relays_bits = found, before
    curve.movement_bits = movement
    return curve


def _position_circuit(study, space):
    active = torch.ones(space.size, dtype=torch.bool, device=study.nat.dev)
    curve = Curve(space)
    argument = (lambda a: a) if getattr(study, "unit_vectors", False) else space.expand
    curve.record(active, study.measure([argument(active)])[0])
    # Attribution only orders the measured candidates. Every candidate is
    # measured at every step: no unjustified gradient-based early termination.
    order = torch.argsort(_unit_attribution(study, space, active).abs(), stable=True).cpu().tolist()
    kept = set(range(space.size))
    while len(kept) > 1:
        js = [j for j in order if j in kept]
        trial = active[None].expand(len(js), -1).clone()
        trial[torch.arange(len(js), device=active.device), torch.tensor(js, device=active.device)] = False
        candidates = [argument(a) for a in trial.unbind()]
        # One cached carry-KL evaluation, internally chunked; changed prompt
        # references are never recomputed inside backward elimination.
        scores = study.measure(candidates)
        best = int(scores.argmin())
        active[js[best]] = False
        kept.remove(js[best])
        curve.record(active, scores[best])
    return curve


def position_circuit(nat, ids, targets, p, prompts, units="matrix", *, relays=False):
    """Exact measured greedy elimination to one unit; Curve.unit_set(i).

    units is 'matrix', 'head', or an explicit iterable of the above unit tuples
    (at p and targets). Gradients affect order only, never membership or scores.
    With relays=True, add up to three measured relay positions if the full
    p/targets set carries less than 90% of movement. Curve.relays records the
    additions; Curve.position_sets records positions used at every curve point.
    The smallest sets reported later are smallest on this greedy path, not a
    globally optimal subset. All candidate removals are one chunked carry call.
    """
    if not isinstance(units, str):
        units = list(units)
    space = UnitSpace(nat, len(ids), [p, *targets], units if isinstance(units, str) else
                      ("head" if any(len(u) == 3 for u in units) else "matrix"))
    if not isinstance(units, str):
        allowed = set(units)
        if allowed != set(space.units):
            raise ValueError("explicit units must contain the full p/targets space")
    study = (HeadCarry if space.granularity == "head" else MatrixCarry)(nat, ids, targets, prompts, space)
    return _circuit_with_relays(study, p, targets, relays, _movement(study))


def _threshold(curve, empty, fraction):
    if empty <= 1e-5:
        return None  # float32 floor; percentages of zero have no interpretation
    eligible = [i for i, (_, score) in enumerate(curve) if score <= (1 - fraction) * empty]
    if not eligible:
        return None
    i = min(eligible, key=lambda j: curve[j][0])
    chosen = curve.node_set(i)
    return {"size": curve[i][0], "kl_bits": curve[i][1], "units": sorted(chosen),
            "positions": sorted({u[-1] for u in chosen if isinstance(u, tuple)})}


def _reuse_position(row):
    """Copy a measured p/targets circuit only when relay stopping is satisfied."""
    try:
        before = row["curve"][0][1]
        movement, empty = row["movement_bits"], row["empty_bits"]
        if not all(math.isfinite(x) for x in (before, movement, empty)):
            return None
        if movement <= 1e-5 or before > .1 * movement:
            return None
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    result = copy.deepcopy(row)
    # Earlier no-relay JSONL lacks these metadata fields. Preserve existing
    # values, but derive missing ones so legacy copies have the current schema.
    if "position_sets" not in result:
        if "unit_sets" not in result:
            return None
        result["position_sets"] = [sorted({u[-1] for u in us}) for us in result["unit_sets"]]
    result.setdefault("threshold_90_below_tested_tolerance", .1 * empty < 3e-5)
    result.setdefault("threshold_99_below_tested_tolerance", .01 * empty < 3e-5)
    fraction = 1 - before / movement
    result.update(reused=True, relays=[], relay_curve=[[0, before]],
                  before_relays_bits=before, after_relays_bits=before,
                  carry_fraction_before_relays=fraction,
                  carry_fraction_after_relays=fraction)
    return result


def _explain_positions(nat, ids, targets, order, units, moved=None, *, relays=False, stored=None):
    result = []
    for p in order:
        old = stored[p] if stored is not None else None
        reused = _reuse_position(old) if relays and old is not None else None
        if reused is not None:
            result.append(reused)
            print(f"  p={p} reused: no relays needed", flush=True)
            continue
        t0 = time.perf_counter()
        prompts = ([ids[:p] + [token] + ids[p+1:] for token in old["alternatives"]]
                   if old is not None and old.get("alternatives") else alternatives(nat, ids, p))
        space = UnitSpace(nat, len(ids), [p, *targets], units)
        study = (HeadCarry if units == "head" else MatrixCarry)(nat, ids, targets, prompts, space)
        empty = float(study.measure([torch.zeros(space.size, dtype=torch.bool, device=nat.dev)])[0])
        movement = _movement(study)
        curve = _circuit_with_relays(study, p, targets, relays, movement)
        synchronize(nat.dev)
        row = {"position": p, "token": ids[p], "alternatives": [x[p] for x in prompts],
               "sensitivity_nats": (old.get("sensitivity_nats") if old is not None
                                    else float(moved[p-1]) if moved is not None else None),
               "empty_bits": empty, "movement_bits": movement,
               "empty_minus_movement_bits": empty - movement,
               "threshold_90_below_tested_tolerance": .1 * empty < 3e-5,
               "threshold_99_below_tested_tolerance": .01 * empty < 3e-5,
               "curve": list(curve), "unit_sets": [sorted(curve.node_set(i)) for i in range(len(curve))],
               "position_sets": curve.position_sets,
               "relays": [{"position": q, "token": ids[q], "offset": q-p, "kl_bits": score}
                          for q, score in curve.relays],
               "relay_curve": [[0, curve.before_relays_bits],
                               *[[i+1, score] for i, (_, score) in enumerate(curve.relays)]],
               "before_relays_bits": curve.before_relays_bits,
               "after_relays_bits": curve[0][1],
               "carry_fraction_before_relays": 1 - curve.before_relays_bits / movement if movement > 1e-5 else None,
               "carry_fraction_after_relays": 1 - curve[0][1] / movement if movement > 1e-5 else None,
               "smallest_90": _threshold(curve, movement if relays else empty, .90),
               "smallest_99": _threshold(curve, movement if relays else empty, .99),
               "full_carry_fraction": 1 - curve[0][1] / empty if empty > 1e-5 else None,
               "outside_needed_90": curve.before_relays_bits > .1 * movement if movement > 1e-5 else None,
               "seconds": time.perf_counter() - t0}
        result.append(row)
        print(f"  p={p} empty={empty:.6g} full={curve[0][1]:.6g} n90={None if row['smallest_90'] is None else row['smallest_90']['size']} {row['seconds']:.1f}s", flush=True)
    return result


def explain(nat, ids, targets, K, *, units="matrix", relays=False, reuse=None, edit=None):
    with model_edit(nat, edit):
        if reuse is not None and parse_edit(reuse.get("edit")) != getattr(nat, "_graph_edit", None):
            raise ValueError("reuse model edit does not match measurement")
        return _explain(nat, ids, targets, K, units=units, relays=relays, reuse=reuse)


def _explain(nat, ids, targets, K, *, units="matrix", relays=False, reuse=None):
    """Top K Native.sensitivity positions; top three model alternatives each.

    Sensitivity is Native's one seeded random alternative and last-target KL
    (nats, original||changed), NOT our three-alternative carry KL (bits,
    changed||carried), summed over targets. Position zero has no preceding
    prediction and is not in Native.sensitivity's candidate space.
    With reuse, keep the stored position/alternative order, copy sufficient
    p/targets circuits, and recompute the others with relay search enabled.
    Copied position seconds and threshold sets describe the original search.
    """
    if K < 1:
        raise ValueError("K must be positive")
    if reuse is not None:
        if not relays:
            raise ValueError("reuse requires relays")
        if reuse["targets"] != targets:
            raise ValueError("reuse targets do not match the requested text")
        rows = reuse["positions"]
        order = [r["position"] for r in rows]
        if len(set(order)) != len(order) or any(
                not 0 < p < len(ids) or p in targets or r["token"] != ids[p]
                for p, r in zip(order, rows)):
            raise ValueError("reuse positions or tokens do not match the requested text")
        return _explain_positions(nat, ids, targets, order, units, relays=True,
                                  stored={r["position"]: r for r in rows})
    if max(targets) < 1:
        return []
    with torch.no_grad():
        moved, other = nat.sensitivity(ids, targets, torch.Generator(device="cpu").manual_seed(native.task_seed(str(ids[:8]))))
    order = [p for p in (moved.argsort(descending=True, stable=True) + 1).tolist() if p not in targets][:K]
    del other
    return _explain_positions(nat, ids, targets, order, units, moved, **({"relays": True} if relays else {}))


def _unit_names(units, position, targets):
    names = []
    for u in units:
        layer, kind = native._name_layer_kind(u[0])
        role = ("p" if u[-1] == position else
                ("t" if len(targets) == 1 else f"t{targets.index(u[-1])}") if u[-1] in targets else
                f"r+{u[-1]-position}")
        short = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "fc", "down_proj": "down"}[kind]
        names.append(f"L{layer}.{short}" + (f".h{u[1]}" if len(u) == 3 else "") + "@" + role)
    return tuple(sorted(names))


def summarize(paths):
    """Normalize units to p/t roles; keep split denominators and failures."""
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    output = {}
    def group(r):
        return (r["split"], r["units"], r.get("selection", "sensitivity") + ("+relays" if r.get("relays") else ""))
    for key in sorted({group(r) for r in rows}):
        subset = [r for r in rows if group(r) == key]
        positions = [(r, p) for r in subset for p in r["positions"]]
        stats = {"questions": len(subset), "positions": len(positions),
                 "outside_needed_90": sum(p["outside_needed_90"] is True for _, p in positions),
                 "full_fail_but_subset_reaches_90": sum(p["outside_needed_90"] is True and p["smallest_90"] is not None for _, p in positions),
                 "outside_criterion_note": "Full p/targets carry <90% is the requested screen, not proof every subset fails; KL is nonmonotone.",
                 "effect_above_floor": sum(p["empty_bits"] > 1e-5 for _, p in positions),
                 "threshold_90_below_tested_tolerance": sum(p["empty_bits"] * .1 < 3e-5 for _, p in positions),
                 "threshold_99_below_tested_tolerance": sum(p["empty_bits"] * .01 < 3e-5 for _, p in positions),
                 "max_empty_movement_discrepancy_bits": max((abs(p["empty_minus_movement_bits"]) for _, p in positions), default=0),
                 "seconds_per_question": {"mean": float(np.mean([r["seconds"] for r in subset])),
                                          "median": float(np.median([r["seconds"] for r in subset])),
                                          "min": min(r["seconds"] for r in subset), "max": max(r["seconds"] for r in subset)}}
        gpu = [r["gpu_utilization"] for r in subset if r.get("gpu_utilization")]
        stats["gpu_mean_percent"] = sum(g["mean_percent"] * g["samples"] for g in gpu) / sum(g["samples"] for g in gpu) if gpu else None
        stats["seconds_per_engine"] = {}
        bins = {}
        for label, lo, hi in (("<0.1", 0, .1), ("0.1-1", .1, 1), ("1-3", 1, 3), (">=3", 3, float("inf"))):
            ps = [p for _, p in positions if lo <= p["movement_bits"] < hi and p["movement_bits"] > 1e-5]
            before = sum(p.get("before_relays_bits", p["curve"][0][1]) <= .1 * p["movement_bits"] for p in ps)
            after = sum(p.get("after_relays_bits", p["curve"][0][1]) <= .1 * p["movement_bits"] for p in ps)
            curve_90 = sum(any(score <= .1 * p["movement_bits"] for _, score in p["curve"]) for p in ps)
            bins[label] = {"positions": len(ps), "before_90": before, "after_90": after,
                           "before_rate": before / len(ps) if ps else None,
                           "after_rate": after / len(ps) if ps else None,
                           "curve_90": curve_90, "curve_rate": curve_90 / len(ps) if ps else None}
        stats["movement_bins"] = bins
        stats["relay_count_distribution"] = dict(sorted(Counter(len(p.get("relays", [])) for _, p in positions).items()))
        stats["relay_positions_in_smallest_90"] = dict(sorted(Counter(
            len({u[-1] for u in p["smallest_90"]["units"]} & {z["position"] for z in p.get("relays", [])})
            for _, p in positions if p["smallest_90"]).items()))
        stats["relay_offsets"] = Counter(z["position"]-p["position"] for _, p in positions for z in p.get("relays", [])).most_common(20)
        stats["relay_tokens"] = Counter(z["token"] for _, p in positions for z in p.get("relays", [])).most_common(20)
        attention = Counter()
        for _, p in positions:
            if p["smallest_90"]:
                qs = {z["position"] for z in p.get("relays", [])}
                attention.update(u[0] + (f".h{u[1]}" if len(u) == 3 else "")
                                 for u in p["smallest_90"]["units"] if u[-1] in qs and ".attn." in u[0])
        stats["relay_attention_units_in_smallest_90"] = attention.most_common(20)
        for engine in sorted({r.get("engine", "native_unit") for r in subset}):
            seconds = [r["seconds"] for r in subset if r.get("engine", "native_unit") == engine]
            stats["seconds_per_engine"][engine] = {"questions": len(seconds), "mean": float(np.mean(seconds)), "median": float(np.median(seconds))}
        fours = {}
        for r, p in positions:
            if p["empty_bits"] <= 1e-5:
                continue
            for i, (size, score) in enumerate(p["curve"]):
                if size == 4:
                    names = _unit_names(p["unit_sets"][i], p["position"], r["targets"])
                    fours.setdefault(names, []).append(1 - score / p["empty_bits"])
        stats["frequent_four_unit_sets"] = [{"count": len(v), "units": s, "mean_carry_fraction": float(np.mean(v)),
                                            "min_carry_fraction": min(v), "max_carry_fraction": max(v)}
                                           for s, v in sorted(fours.items(), key=lambda item: (-len(item[1]), item[0]))[:15]]
        for level in (90, 99):
            sizes, resolvable, sets, singles = Counter(), Counter(), Counter(), Counter()
            for r, p in positions:
                small = p[f"smallest_{level}"]
                label = str(small["size"]) if small else ("effect_floor" if p["empty_bits"] <= 1e-5 else "unreached")
                sizes[label] += 1
                if (1 - level / 100) * p["empty_bits"] >= 3e-5:
                    resolvable[label] += 1
                if small:
                    names = _unit_names(small["units"], p["position"], r["targets"])
                    sets[names] += 1
                    singles.update(names)
            stats[f"n{level}_distribution"] = dict(sizes)
            stats[f"n{level}_resolvable_distribution"] = dict(resolvable)
            stats[f"frequent_{level}_sets"] = [{"count": c, "units": s} for s, c in sets.most_common(15)]
            stats[f"frequent_{level}_units"] = singles.most_common(20)
        output["/".join(key)] = stats
    return output


def audit_matrix_results(paths):
    """Recheck emitted curves with the original, dense, unbatched node API."""
    nat = native.Native()
    result = []
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["units"] != "matrix":
                raise ValueError("audit takes matrix results; head slices are independently tested in test_carry.py")
            ids, targets = native.text(native.TEXTS / "vpd4l" / f"{row['task']}.json")
            for p in row["positions"]:
                pos = p["position"]
                prompts = [ids[:pos] + [token] + ids[pos+1:] for token in p["alternatives"]]
                study = Carry(nat, ids, targets, prompts, chunk_size=1)
                nodes = [{(n, t, c) for n, t in units for c in range(nat.C[n])} for units in p["unit_sets"]]
                actual = study.measure([set(), *nodes]).cpu().tolist()
                errors = [abs(a-b) for a, b in zip(actual, [p["empty_bits"], *[z[1] for z in p["curve"]]])]
                item = {"task": row["task"], "position": pos, "max_error_bits": max(errors),
                        "empty_bits_dense": actual[0], "full_bits_dense": actual[1],
                        "points": len(nodes), "dense_curve": [[z[0], v] for z, v in zip(p["curve"], actual[1:])]}
                result.append(item)
                print(row["task"], "dense audit max bits error", max(errors), flush=True)
                native.free(nat.dev)
    return result


def report(curve, diag):
    baseline = diag["empty_bits"]
    cutoff = curve[0][1] + .1 * (baseline - curve[0][1])
    eligible = [i for i, (size, score) in enumerate(curve) if score <= cutoff]
    if not eligible:
        return {"cutoff_bits": cutoff, "smallest": None}
    i = min(eligible, key=lambda j: curve[j][0])
    nodes = sorted(curve.node_set(i))
    counts = Counter((n, p) for n, p, c in nodes)
    return {"cutoff_bits": cutoff,
            "smallest": {"size": len(nodes), "kl_bits": curve[i][1],
                         "matrix_positions": [[n, p, k] for (n, p), k in sorted(counts.items())],
                         "nodes": nodes}}


class _Utilization:
    """Best-effort system GPU utilization on macOS, sampled once per second.

    IORegistry is read-only and needs no privileges. These samples include
    other GPU users and are not a process-specific measurement.
    """
    def __init__(self, dev):
        self.samples, self.stop = [], threading.Event()
        self.thread = None
        if str(dev).startswith("mps"):
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()

    def _sample(self):
        while not self.stop.is_set():
            try:
                raw = subprocess.run(["ioreg", "-a", "-r", "-c", "AGXAccelerator", "-d", "1"],
                                     capture_output=True, timeout=2, check=True).stdout
                for entry in plistlib.loads(raw):
                    value = entry.get("PerformanceStatistics", {}).get("Device Utilization %")
                    if value is not None:
                        self.samples.append(float(value))
            except (OSError, subprocess.SubprocessError, ValueError, plistlib.InvalidFileException):
                return
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        if not self.samples:
            return None
        return {"mean_percent": sum(self.samples) / len(self.samples),
                "min_percent": min(self.samples), "max_percent": max(self.samples),
                "samples": len(self.samples), "scope": "system GPU, 1 second IORegistry samples"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("layout", help="check exact VPD head support")
    a = sub.add_parser("summarize")
    a.add_argument("files", type=Path, nargs="+")
    a.add_argument("--out", type=Path)
    a = sub.add_parser("audit")
    a.add_argument("files", type=Path, nargs="+")
    a.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("copies", help="source-position circuits for known copy questions")
    c.add_argument("--n", type=int, default=6)
    c.add_argument("--tasks", nargs="+", help="optional copy question ids")
    c.add_argument("--units", choices=("matrix", "head"), default="head")
    c.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("explain")
    e.add_argument("--split", choices=("hard", "heldout", "train"), required=True)
    e.add_argument("--n", type=int, required=True)
    e.add_argument("--k", type=int, default=8)
    e.add_argument("--units", choices=("matrix", "head"), default="matrix")
    e.add_argument("--relays", action="store_true", help="screen and measure up to three intermediate relay positions")
    e.add_argument("--edit", type=parse_edit, help="model edits, e.g. heads:L2:2,3,4;mlp:L1;edge:L2:all")
    e.add_argument("--reuse", type=Path, help="reuse compatible no-relay JSONL circuits, preserving position order and alternatives")
    e.add_argument("--offset", type=int, default=0, help="this process's share of the first n texts: offset, offset + stride, ...")
    e.add_argument("--stride", type=int, default=1)
    e.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("search")
    p.add_argument("--task", required=True, help="copy question id, or first6")
    p.add_argument("--positions", default="source,target", help="comma-separated source,target, integer positions, or all")
    p.add_argument("--out", type=Path, help="JSON result path (otherwise only prints)")
    args = ap.parse_args()
    if args.cmd == "audit":
        args.out.write_text(json.dumps(audit_matrix_results(args.files), indent=2) + "\n")
        return
    if args.cmd == "summarize":
        result = json.dumps(summarize(args.files), indent=2)
        print(result)
        if args.out:
            args.out.write_text(result + "\n")
        return
    if args.cmd == "copies":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if args.out.exists():
            for line in args.out.read_text().splitlines():
                row = json.loads(line)
                if row["units"] != args.units or row.get("selection") != "copy_source":
                    ap.error("existing copy output has incompatible configuration")
                done.add(row["task"])
        rows = [json.loads(line) for line in COPYCHECK.read_text().splitlines()][:args.n]
        if args.tasks:
            rows = [r for r in rows if r["task"] in args.tasks]
            if set(args.tasks) - {r["task"] for r in rows}:
                ap.error("requested copy task not in the first n copycheck rows")
        nat = native.Native()
        for row in rows:
            if row["task"] in done:
                continue
            ids, targets = native.text(native.TEXTS / "vpd4l" / f"{row['task']}.json")
            print(row["task"], "copy source", row["source"], flush=True)
            util = _Utilization(nat.dev)
            t0 = time.perf_counter()
            positions = _explain_positions(nat, ids, targets, [row["source"]], args.units)
            synchronize(nat.dev)
            result = {"task": row["task"], "split": row["split"], "units": args.units,
                      "selection": "copy_source", "targets": targets, "positions": positions,
                      "device": str(nat.dev), "seconds": time.perf_counter() - t0,
                      "gpu_utilization": util.finish()}
            with args.out.open("a") as f:
                f.write(json.dumps(result) + "\n")
            native.free(nat.dev)
        return
    if args.cmd == "explain":
        if args.reuse is not None and not args.relays:
            ap.error("--reuse requires --relays")
        if args.n < 1 or args.k < 1 or args.stride < 1 or not 0 <= args.offset < args.stride:
            ap.error("n, k and stride must be positive, offset in [0, stride)")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if args.out.exists():
            with args.out.open() as f:
                for line in f:
                    row = json.loads(line)
                    if (row["split"], row["k"], row["units"], row.get("relays", False), parse_edit(row.get("edit"))) != (args.split, args.k, args.units, args.relays, args.edit):
                        ap.error("existing output has incompatible split, k, units, relays or edit")
                    done.add(row["task"])
        pending = [p for p in native.tasks(args.split)[:args.n][args.offset::args.stride] if p.stem not in done]
        if not pending:
            print("all requested questions already present", flush=True)
            return
        reuse = {}
        if args.reuse is not None:
            wanted = {p.stem for p in pending}
            with args.reuse.open() as f:
                for line in f:
                    row = json.loads(line)
                    if row["task"] in wanted and (
                            row["split"], row["k"], row["units"], row.get("relays", False), parse_edit(row.get("edit"))
                    ) == (args.split, args.k, args.units, False, args.edit):
                        if row["task"] in reuse:
                            ap.error(f"duplicate compatible reuse row: {row['task']}")
                        reuse[row["task"]] = row
        nat = native.Native()
        if str(nat.dev) != "mps":
            print(f"device={nat.dev}", flush=True)
        for path in pending:
            print(path.stem, flush=True)
            ids, targets = native.text(path)
            synchronize(nat.dev)
            util = _Utilization(nat.dev)
            t0 = time.perf_counter()
            positions = explain(nat, ids, targets, args.k, units=args.units, relays=args.relays, edit=args.edit,
                                **({"reuse": reuse[path.stem]} if path.stem in reuse else {}))
            synchronize(nat.dev)
            result = {"task": path.stem, "split": args.split, "k": args.k, "units": args.units,
                      "targets": targets, "device": str(nat.dev), "positions": positions, "relays": args.relays, "edit": args.edit,
                      "engine": "direct-unit-v4-stable-batches",
                      "seconds": time.perf_counter() - t0, "gpu_utilization": util.finish(),
                      "search": "exact greedy; smallest sets on measured path, not global optima",
                      "outside_criterion": "Full p/targets carry <90%, not proof every subset fails",
                      "relay_screen": "top 16 by sum of absolute unit gradients at p/targets-only endpoint; every intermediate non-target position screened" if args.relays else None,
                      "relay_stopping": "90% of movement, <1% movement improvement, or 3 relays" if args.relays else None,
                      "head_definition": UnitSpace.__doc__ if args.units == "head" else None}
            with args.out.open("a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
            print(f"{path.stem}: {result['seconds']:.1f}s GPU {result['gpu_utilization']}", flush=True)
            native.free(nat.dev)
        return
    if args.cmd == "layout":
        nat = native.Native()
        print(nat.dev, nat.target.n_head, nat.target.hd, flush=True)
        for n in nat.names:
            if ".attn." not in n:
                continue
            st = nat.target.site(n)
            w = st.V.T if n.endswith("o_proj") else st.U
            z = w.reshape(w.shape[0], nat.target.n_head, nat.target.hd).abs().amax(-1)
            counts = (z > 0).sum(-1)
            print(n, "heads/component", torch.unique(counts, return_counts=True), flush=True)
        return
    rows = [json.loads(line) for line in COPYCHECK.read_text().splitlines()]
    rows = rows[:6] if args.task == "first6" else [r for r in rows if r["task"] == args.task]
    if not rows:
        ap.error("task not found in copycheck_vr4.jsonl")
    nat = native.Native()
    output = []
    for row in rows:
        ids, targets = native.text(native.TEXTS / "vpd4l" / f"{row['task']}.json")
        prompts = alternatives(nat, ids, row["source"])
        named = {"source": row["source"], "target": row["target"]}
        positions = list(range(len(ids))) if args.positions == "all" else [named[z] if z in named else int(z) for z in args.positions.split(",")]
        study = Carry(nat, ids, targets, prompts)
        diag = diagnostics(study)
        print(row["task"], json.dumps(diag), flush=True)
        synchronize(nat.dev)
        utilization = _Utilization(nat.dev)
        start = time.perf_counter()
        curve = _search(study, positions)
        synchronize(nat.dev)
        seconds = time.perf_counter() - start
        gpu = utilization.finish()
        result = {"task": row["task"], "source": row["source"], "target": row["target"],
                  "targets": targets, "positions": positions, "alternatives": [x[row["source"]] for x in prompts],
                  "device": str(nat.dev), "diagnostics": diag, "curve": curve, "seconds": seconds,
                  "gpu_utilization": gpu, **report(curve, diag)}
        if str(nat.dev).startswith("mps"):
            result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        output.append(result)
        printable = dict(result)
        if result["smallest"] is not None:
            printable["smallest"] = {k: v for k, v in result["smallest"].items() if k != "nodes"}
        print(json.dumps(printable), flush=True)
        if args.out:
            # Output is explicitly requested by --out; no auxiliary artifacts.
            args.out.write_text(json.dumps(output if args.task == "first6" else result, indent=2) + "\n")
        del study, curve
        native.free(nat.dev)


if __name__ == "__main__":
    main()
