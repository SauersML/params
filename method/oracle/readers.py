"""Who reads a circuit unit: each unit's downstream readers in the whole model, by path patching, with measured effect
and sign, and its direct write on the answer.

A unit is a neuron (layer, position, index) or a head (layer, head, position). Its write w is what it adds to the
stream at its position: in a patching pair its clean write less its corrupted write, with no corrupted prompt its whole
write (zeroing). Its readers, any unit downstream of it, inside the circuit or not:
  neuron     a neuron at the unit's position in a later layer (a head's: its own layer's MLP on), w taken out of its MLP
             input ("mlp")
  head       a head of a later layer, w taken out of its key and value input at the unit's position (every query
             position after it reads the changed key and value: "key_value"), or out of its query input there ("query")
  direct     the answer's contrast with w taken out of the final norm's input (a unit at the answer position)
An edge's effect is measured in a full run of the model with w taken out of that one reader's input (path patching:
every other reader still reads it): the contrast's change in bits and, in a patching pair, its share of the gap lost;
positive when the edge carries the unit's effect toward the answer, negative when it works against it. Candidates are
ordered by a first-order estimate (the reader's exact local change times the contrast's gradient at the reader's output:
one backward pass a prompt; ordering only; "first_order" in bits lost like the measured effect), the first TOP of them
measured; and reader groups, a layer's whole MLP at the unit's position or all its heads through one input ("mlp",
"key_value", "query"), ordered by their summed estimates, the first GROUPS measured (a unit read by many readers, each
small, shows there); beside them the unit itself patched (or zeroed) alone ("total"), all of its measured single
readers cut at once ("together"), the share of the first-order estimates' absolute sum those readers cover, and its
paths: every downstream MLP at its position ("mlp"), every later head through its key and value there ("key_value") or
its query there ("query"), each cut alone, and all three at once ("indirect"); "indirect" with the direct write is the
whole unit, since those are all of its write's readers.

  readers(rn, ids, units, corrupt=None)   head_routes.Runner rn with its answer and alternatives set
  for_prompt(model, tok, ids, units, corrupt=None)   the same on a model and a prompt (a tool's call): the answer is
      the clean run's top token's word against other words (answer_contrast), unless answer= and alternatives= say
"""
from __future__ import annotations

import math
import time

import torch

from head_routes import rotate_half, weight

TOP = 8  # readers measured by path patching a unit (the largest first-order estimates)
GROUPS = 4  # reader groups (a layer's MLP or heads through one input) measured a unit
MIN_SHARE = 0.02  # single readers and groups measured for units whose own share of the gap reaches this (the finder's
# EPS: a unit under it is no single unit there; 515 of 5,641 Qwen3-8B dev circuit units reach 0.05)


def _norm(x: torch.Tensor, mod) -> torch.Tensor:
    """An RMS norm in float32 over the last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + mod.variance_epsilon) * mod.weight.float().to(x.device)


def _writes(rn, st: dict, sb: dict | None, units: list) -> torch.Tensor:
    """Each unit's write at its position [n, d] (sb: less its write in the corrupted run)."""
    out = []
    for u in units:
        if u[0] == "neuron":
            _, L, p, j = u
            col = weight(rn.layers[L].mlp.down_proj)[:, j].float()
            v = st["h"][L][p, j] - (sb["h"][L][p, j] if sb is not None else 0.0)
            out.append(col * v.to(col.device))
        else:
            _, L, h, p = u
            w = rn.head_write(st, L, h, p)
            out.append(w - rn.head_write(sb, L, h, p).to(w.device) if sb is not None else w)
    return torch.stack([w.to(rn.dev) for w in out])


def _final(rn, st: dict, pos: int) -> torch.Tensor:
    """The stream at pos after the last layer [d] (float32)."""
    layer = rn.layers[-1]
    return st["mid"][-1][pos] + (weight(layer.mlp.down_proj).float() @ st["h"][-1][pos].to(rn._d(rn.L - 1)))


def _contrast(rn, x: torch.Tensor) -> torch.Tensor:
    """The answer's contrast in bits from final streams x [n, d] (the run's answer and alternatives rows)."""
    z = _norm(x.to(rn._rows.device), rn.core.model.norm) @ rn._rows.T
    na = len(rn.answer)
    return (torch.logsumexp(z[:, :na], -1) - torch.logsumexp(z[:, na:], -1)) / math.log(2.0)


@torch.no_grad()
def _candidates(rn, st: dict, g: dict, units: list, W: torch.Tensor) -> tuple[list[list[tuple]], dict]:
    """Per unit, its downstream readers at its position with their first-order estimates, largest |estimate| first
    (each layer's 4 TOP largest of each kind): (estimate, kind, layer, index or head, via, position, unit index, slot);
    and the cut edits' tensors by (position, layer) for _edit."""
    H, KV, hd = rn.H, rn.KV, rn.hd
    gs = H // KV
    out, groups, store = [[] for _ in units], [[] for _ in units], {}
    for p in sorted({u[-1] if u[0] == "head" else u[2] for u in units}):
        ix = [i for i, u in enumerate(units) if (u[-1] if u[0] == "head" else u[2]) == p]
        lay = torch.tensor([units[i][1] for i in ix])
        head = torch.tensor([units[i][0] == "head" for i in ix])
        for L, layer in enumerate(rn.layers):
            d, at = rn._d(L), layer.self_attn
            Wp = W[ix].to(d)
            mlp = (lay < L) | (head & (lay == L))  # (a head's write reaches its own layer's MLP)
            if bool(mlp.any()):
                m = mlp.nonzero().flatten().tolist()
                dh = rn._acts(layer, st["mid"][L][p].to(d)[None] - Wp[m]) - st["h"][L][p].to(d)[None]  # [n, d_mlp]
                est = -dh * g["h"][L][p].to(d)[None]  # (bits lost: the contrast's first-order fall)
                top = est.abs().topk(min(4 * TOP, est.shape[1]), dim=-1).indices
                vals, top, tot = est.gather(1, top).cpu().tolist(), top.cpu().tolist(), est.sum(-1).cpu().tolist()
                for a, i in enumerate(m):
                    out[ix[i]] += [(e, "neuron", L, j, "mlp", p, ix[i], None) for e, j in zip(vals[a], top[a])]
                    groups[ix[i]].append((tot[a], "mlp", L, None, "mlp", p, ix[i], None))
            att = lay < L
            if not bool(att.any()):
                continue
            m = att.nonzero().flatten().tolist()
            x = st["x"][L][p].to(d)[None] - Wp[m]  # [n, d]: the head's input with w taken out
            hn = _norm(x, layer.input_layernorm)
            cs, sn = st["cos"][p].to(d), st["sin"][p].to(d)
            k = _norm((hn @ weight(at.k_proj).float().T).view(len(m), KV, hd), at.k_norm)
            k = k * cs + rotate_half(k) * sn
            v = hn @ weight(at.v_proj).float().T
            if at.v_proj.bias is not None:
                v = v + at.v_proj.bias.float()
            v = v.view(len(m), KV, hd)
            dk = (k - st["k"][L][:, p].to(d)[None]).repeat_interleave(gs, 1)  # [n, H, hd]
            dv = (v - st["v"][L][:, p].to(d)[None]).repeat_interleave(gs, 1)
            qv = st["q"][L][:, p:].to(d)  # [H, Tq, hd]: every query position from p reads the key and value at p
            ap = st["A"][L][:, p:, p].to(d)  # [H, Tq]
            e = torch.exp(torch.einsum("nhc,htc->nht", dk, qv) * at.scaling)
            c = 1.0 / (1.0 + ap[None] * (e - 1.0))
            z = st["z"][L][p:].to(d).transpose(0, 1)  # [H, Tq, hd]
            vp = st["v"][L][:, p].to(d).repeat_interleave(gs, 0)  # [H, hd]
            dz = (c[..., None] * (z[None] - ap[None, ..., None] * vp[None, :, None]) + (ap[None] * e * c)[..., None]
                  * (vp[None] + dv)[:, :, None] - z[None])  # [n, H, Tq, hd]
            est_kv = -(dz * g["z"][L][p:].to(d).transpose(0, 1)[None]).sum((-1, -2))  # [n, H]
            q = _norm((hn @ weight(at.q_proj).float().T).view(len(m), H, hd), at.q_norm)
            q = q * cs + rotate_half(q) * sn
            K = st["k"][L][:, :p + 1].to(d).repeat_interleave(gs, 0)  # [H, p + 1, hd]
            V = st["v"][L][:, :p + 1].to(d).repeat_interleave(gs, 0)
            a = (torch.einsum("nhc,htc->nht", q, K) * at.scaling).softmax(-1)
            dzq = torch.einsum("nht,htc->nhc", a, V) - st["z"][L][p].to(d)[None]  # [n, H, hd]
            est_q = -(dzq * g["z"][L][p].to(d)[None]).sum(-1)
            store[(p, L)] = {"dk": dk, "dv": dv, "dq": q - st["q"][L][:, p].to(d)[None]}
            kv, qq = est_kv.cpu().tolist(), est_q.cpu().tolist()
            for a_, i in enumerate(m):
                out[ix[i]] += [(kv[a_][h], "head", L, h, "key_value", p, ix[i], a_) for h in range(H)]
                out[ix[i]] += [(qq[a_][h], "head", L, h, "query", p, ix[i], a_) for h in range(H)]
                groups[ix[i]] += [(sum(kv[a_]), "heads", L, None, "key_value", p, ix[i], a_),
                                  (sum(qq[a_]), "heads", L, None, "query", p, ix[i], a_)]
    order = lambda c: sorted(c, key=lambda r: -abs(r[0]))  # noqa: E731
    return [order(c) for c in out], [order(c) for c in groups], store


def _edit(r: tuple, store: dict, W: torch.Tensor) -> dict:
    """A candidate reader's cut as Runner edits (w taken out of that reader's input alone)."""
    _, kind, L, x, via, p, i, a = r
    if kind == "neuron":
        return {"read": [(L, p, x, W[i])]}
    if kind in ("mlp", "heads"):
        return {{"mlp": "mlp_in", "key_value": "kv_in", "query": "q_in"}[via]: [(L, p, W[i])]}
    t = store[(p, L)]
    if via == "key_value":
        return {"dk": [(L, x, p, t["dk"][a, x])], "dv": [(L, x, p, t["dv"][a, x])]}
    return {"dq": [(L, x, p, t["dq"][a, x])]}


def _union(edits: list[dict]) -> dict:
    """Several cuts' edits as one set."""
    out = {}
    for e in edits:
        for k, v in e.items():
            out.setdefault(k, []).extend(v)
    return out


@torch.no_grad()
def readers(rn, ids: list[int], units: list, corrupt: list[int] | None = None, top: int = TOP, groups: int = GROUPS,
            min_share: float = MIN_SHARE, circuit: set | None = None, states: tuple | None = None,
            seconds: dict | None = None) -> list[dict]:
    """Each unit's readers (module docstring): every unit's total and paths, then the single readers and groups of
    those whose total reaches min_share of the gap (all with no corrupted prompt; the rest list their first TOP
    candidates' estimates, "bits" None). units: ("neuron", layer, pos, index) or ("head", layer, head, pos); circuit:
    units of the circuit, to mark readers inside it; states: both runs' captured states when the caller has them;
    seconds: filled with this call's seconds by part."""
    if not units:
        return []
    t0, sec = time.time(), {} if seconds is None else seconds
    st, sb = states or (rn.run(ids, [{}], capture=True)["states"],
                        rn.run(corrupt, [{}], capture=True)["states"] if corrupt is not None else None)
    T = len(ids)
    c = rn.run(ids, [{}])["contrast"][0]  # (uncut, at the cut runs' batch shape)
    gap = c - rn.run(corrupt, [{}])["contrast"][0] if corrupt is not None else None
    W = _writes(rn, st, sb, units)
    g = rn.grads(ids)
    sec["runs_and_gradient"], t0 = round(time.time() - t0, 3), time.time()
    cands, grp, store = _candidates(rn, st, g, units, W)
    sec["candidates"], t0 = round(time.time() - t0, 3), time.time()
    pos = [u[-1] if u[0] == "head" else u[2] for u in units]

    def measure(sets, slots):  # (a forward computes from its sets' first position on: sets of one position together,
        order = sorted(range(len(sets)), key=lambda a: pos[slots[a][0]])  # the answer position's nearly free)
        got = rn.run(ids, [sets[a] for a in order])["contrast"]
        cs = [0.0] * len(sets)
        for a, x in zip(order, got):
            cs[a] = x
        return [(i, what, {"bits": c - x, "share": (c - x) / gap if gap else None}) for (i, what), x in zip(slots, cs)]
    sets, slots = [], []
    for i, u in enumerate(units):
        paths = {via: _union([_edit(r, store, W) for r in grp[i] if r[4] == via]) for via in ("mlp", "key_value", "query")}
        for via, e in paths.items():
            sets.append(e)
            slots.append((i, via))
        sets.append(_union(list(paths.values())))
        slots.append((i, "indirect"))
        if u[0] == "neuron":
            _, L, p, j = u
            sets.append({"neuron": [(L, p, j, float(sb["h"][L][p, j]) if sb is not None else 0.0)]})
        else:
            _, L, h, p = u
            sets.append({"head": [(L, h, p, sb["z"][L][p, h] if sb is not None else None)]})
        slots.append((i, "total"))
    got = measure(sets, slots)
    total = {i: e for i, what, e in got if what == "total"}
    read = [i for i in range(len(units)) if gap is None or abs(total[i]["share"]) >= min_share]
    sets, slots = [], []
    for i in read:
        for r in grp[i][:groups]:
            sets.append(_edit(r, store, W))
            slots.append((i, "group"))
        pick = [_edit(r, store, W) for r in cands[i][:top]]
        for e in pick:
            sets.append(e)
            slots.append((i, "reader"))
        sets.append(_union(pick))
        slots.append((i, "together"))
    got += measure(sets, slots)
    sec["cuts"], sec["sets"], sec["units_read"] = round(time.time() - t0, 3), len(got), len(read)
    xf = _final(rn, st, T - 1)
    out = []
    for i, u in enumerate(units):
        unit = ({"kind": "neuron", "layer": u[1], "pos": u[2], "index": u[3]} if u[0] == "neuron" else
                {"kind": "head", "layer": u[1], "head": u[2], "pos": u[3]})
        ests = sum(abs(r[0]) for r in cands[i])
        out.append({"unit": unit, "readers": [], "groups": [], "paths": {}, "candidates": len(cands[i]),
                    "first_order_covered": sum(abs(r[0]) for r in cands[i][:top]) / ests if ests else None,
                    "together": None, "direct": None})
        if pos[i] == T - 1:
            bits = float(_contrast(rn, xf[None]) - _contrast(rn, (xf - W[i].to(xf.device))[None]))
            out[-1]["direct"] = {"bits": bits, "share": bits / gap if gap else None}
    measured = {}
    for i, what, eff in got:
        if what in ("group", "reader"):
            measured.setdefault((i, what), []).append(eff)
        elif what in ("mlp", "key_value", "query"):
            out[i]["paths"][what] = eff
        else:
            out[i][what] = eff
    nil = {"bits": None, "share": None}
    for i in range(len(units)):
        effs = measured.get((i, "group"), [])
        for a, r in enumerate(grp[i][:groups] if effs else []):
            out[i]["groups"].append({"layer": r[2], "via": r[4], **effs[a], "first_order": r[0]})
        effs = measured.get((i, "reader"), [])
        for a, r in enumerate(cands[i][:top]):
            rd = {"kind": r[1], "layer": r[2], ("index" if r[1] == "neuron" else "head"): r[3], "via": r[4],
                  "pos": pos[i], **(effs[a] if effs else nil), "first_order": r[0]}
            if circuit is not None:
                rd["in_circuit"] = (("neuron", r[2], pos[i], r[3]) if r[1] == "neuron" else ("head", r[2], r[3], pos[i])) \
                    in circuit
            out[i]["readers"].append(rd)
    return out


def for_prompt(model, tok, ids: list[int], units: list, corrupt: list[int] | None = None, answer: list[int] | None = None,
               alternatives: list[int] | None = None, **kw) -> dict:
    """readers() on a model (an HF Qwen3, or a PeftModel of one) and a prompt: {"answer", "alternatives", "readers"}."""
    import answer_contrast  # noqa: PLC0415
    from head_routes import Runner  # noqa: PLC0415
    rn = Runner(model)
    if answer is None:
        lp = rn.run(ids, [{}], capture=True)["lp"][0]
        b = int(rn.run(corrupt, [{}], capture=True)["lp"][0].argmax()) if corrupt is not None else None
        sets = answer_contrast.token_sets(rn.core, tok, ids, lp, b)
        answer, alternatives = sets["answer"], sets["alternatives"]
    rn.answer, rn.alternatives = list(answer), list(alternatives)
    return {"answer": rn.answer, "alternatives": rn.alternatives, "readers": readers(rn, ids, units, corrupt=corrupt, **kw)}


def record_units(rec: dict) -> list:
    """A circuit_finder record's units: its neurons (nodes) and mover heads (routes)."""
    return ([("neuron", n["layer"], n["pos"], n["neuron"]) for n in rec.get("nodes", [])]
            + [("head", r["layer"], r["head"], r["pos"]) for r in rec.get("routes", [])])
