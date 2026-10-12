"""Component atlas of VPD's decomposition of vpd4l (#2951): measured facts about each of the 38,912
rank-one subcomponents u v^T (run s-55ea3f9b of the 4-layer Pile model t-9d2b8f02), the vocabulary the
oracle's English is checked against. No label is written by any model; every field is a measurement or
a fixed function of the weights.

A subcomponent c of a site (a linear map y = W x: q, k, v, o, c_fc or down_proj of one layer) has
activity a = v . x on the site's input x and writes a u. Its causal importance (CI) is VPD's published
CI network's output, at its lower-leaky forward value clamp(z, 0, 1); c is active at a position when
z > 0. The CI network reads the whole sequence (it is bidirectional), so CI at a position can depend on
later tokens; the atlas records it as published. Position 0 (the sequence's first token, an attention
sink) is left out of every statistic.

Data: rows of the Pile validation split (VPD trained on the train split), 512 tokens each. Split A
(--train) gives the activation facts, split B (--heldout, disjoint) the effects and every
verification number.

Per subcomponent:
  Tokens ranked or compared below are those split A holds at least MIN_COUNT (10) times.
  write lens     w_X = r . (g_f * e_X) for every such token X (e_X its embedding row, g_f the
                 final norm's gain): r = u for o_proj and down_proj, W_O u for v_proj (its layer's
                 o_proj), W_down u for c_fc (its layer's down_proj, the GELU taken as the identity);
                 q_proj and k_proj move attention scores and have none. The 10 tokens it raises most
                 and the 10 it lowers most when active, oriented by the sign of its median activity
                 over its top contexts (u v^T = (-u)(-v)^T, so a lens alone has no sign).
  activation     the share of split A's positions where it is active, its mean CI, and its top 20
                 contexts: the 20 rows of split A where its largest CI pre-activation z is highest, each
                 with the position of that maximum, z, CI, the activity a there, and a window of 16
                 tokens (12 before the position, the token there, 3 after).
  effect         on split B, at up to 4 positions drawn uniformly from those where it is active: its
                 contribution to the next-token log-probabilities, log p - log p_without, p_without the
                 model with its write removed at that position only (W - u v^T at that row: the row's
                 next-token distribution is computed exactly through every later layer; rows before it
                 are unchanged and later rows cannot reach it). Per sample: KL(p || p_without) in nats,
                 and the 10 tokens whose probability it raises most and the 10 it lowers most, with
                 the change p - p_without.
  verification   on split B: trigger precision, the share of positions holding one of its top
                 contexts' tokens (the token at each context's maximum) where it is active, beside its
                 base rate there and the trigger recall (the share of its active positions holding one);
                 and for the lens sites, per effect sample, the Pearson correlation over the data's
                 tokens between the lens prediction a w_X and the measured contribution, and the
                 overlap of their top 10 raised tokens.
  look-ahead     (`lookahead`) the share of its active positions where it turns off when the text after the
                 position is replaced (the rest of another row of split A at the same positions; the position and
                 everything before it unchanged), over cut positions chosen to cover every subcomponent twice where
                 its top contexts allow (greedy set cover over its top-3 contexts' positions); its activity at a cut
                 comes from parts.py's split-A activity files. With it, the number of cuts it was active at and the
                 share of cuts where it turns on from off. VPD's CI network reads the whole sequence, so a trigger can
                 depend on later tokens; this measures how much.
  partners       the weight connectome's (method/vpd/connectome.py, its files under
                 ~/mpd-data/atlas/vpd4l/connectome), read through Atlas.partners.

Files under --out (default ~/mpd-data/atlas/vpd4l): atlas.safetensors (arrays over the 38,912
subcomponents, in the order of vpd_model.site_names() and index within a site) and atlas.json (the
settings, the splits, the timings and speeds, the field definitions).

  MPD_MEM_GIB=20 python -m method.vpd.atlas build [--train 0:5000] [--heldout 5000:6000] [--batch 8]
  MPD_MEM_GIB=20 python -m method.vpd.atlas lookahead [--need 2] [--batch 8]   (after parts.py activity)
  python -m method.vpd.atlas show SITE INDEX     (e.g. h.2.mlp.c_fc 17)
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from method.vpd import vpd_model as VM

DEFAULT = Path.home() / "mpd-data/atlas/vpd4l"
TOP, LENS, EFFECT, SAMPLES = 20, 10, 10, 4
BEFORE, AFTER = 12, 3
MIN_COUNT = 10  # tokens ranked or compared: those split A holds at least this often
PAD = 65535  # an empty slot of a token window (uint16; the vocabulary has 50,277 tokens)
LENS_KINDS = ("v_proj", "o_proj", "c_fc", "down_proj")


def rows(spec: str) -> tuple[int, int]:
    a, b = spec.split(":")
    return int(a), int(b)


def layout(vpd: VM.VPD) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sites in vpd_model.site_names() order; per subcomponent its site number and index within the site."""
    names = VM.site_names()
    site = np.concatenate([np.full(vpd.C[n], i, np.int16) for i, n in enumerate(names)])
    index = np.concatenate([np.arange(vpd.C[n], dtype=np.int32) for n in names])
    return names, site, index


@torch.no_grad()
def ci_and_activity(vpd: VM.VPD, names: list[str], ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CI pre-activations z and activities a = v . x of every subcomponent, [B, T, total], one clean pass."""
    vpd.clear()
    for n in names:
        vpd.target.site(n).cache_input = True
    vpd.target.hidden(ids)
    acts = {n: vpd.target.site(n).last_input for n in names}
    vpd.clear()
    z = vpd.ci_fn(acts)
    return (torch.cat([z[n] for n in names], -1),
            torch.cat([acts[n] @ vpd.target.site(n).V for n in names], -1))


def write_directions(target: VM.Target, layer: int, kind: str) -> torch.Tensor | None:
    """Residual write direction r per subcomponent [C, d] (module note), or None for q_proj and k_proj."""
    U = target.site(f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}").U
    if kind in ("o_proj", "down_proj"):
        return U
    if kind == "v_proj":
        return U @ target.site(f"h.{layer}.attn.o_proj").W.T
    if kind == "c_fc":
        return U @ target.site(f"h.{layer}.mlp.down_proj").W.T
    return None


def lens_values(target: VM.Target, r: torch.Tensor) -> torch.Tensor:
    """w_X = r . (g_f * e_X) for every vocabulary token, [R, V]."""
    return (r * target.ln_f) @ target.wte.T


def heads(target: VM.Target, name: str) -> np.ndarray:
    """The head each attention subcomponent lives in most: of u for q, k, v (its output), of v for o (its input)."""
    st, kind = target.site(name), name.rsplit(".", 1)[1]
    if kind not in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return np.full(st.U.shape[0], -1, np.int8)
    m = st.U if kind != "o_proj" else st.V.T  # [C, H * hd]
    return m.view(m.shape[0], target.n_head, target.hd).pow(2).sum(-1).argmax(-1).to(torch.int8).cpu().numpy()


# ------------------------------------------------------------------ one row, edited


@torch.no_grad()
def clean_cache(target: VM.Target, ids: torch.Tensor) -> dict:
    """Per layer: the residual entering it, after its attention, its rotated keys and its values; the site
    inputs; and the clean next-token log-probabilities, for rows edited one at a time (row_logprobs)."""
    B, T = ids.shape
    H, hd, eps = target.n_head, target.hd, target.eps
    x = target.wte[ids]
    cache = {"xin": [], "xmid": [], "k": [], "v": [], "site_in": {}}
    for i in range(target.n_layer):
        s = lambda k: target.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
        cache["xin"].append(x)
        h = VM.rms(x, target.norms[2 * i], eps)
        q = s("q_proj")(h).view(B, T, H, hd).transpose(1, 2)
        k = s("k_proj")(h).view(B, T, H, hd).transpose(1, 2)
        v = s("v_proj")(h).view(B, T, H, hd).transpose(1, 2)
        q, k = target._rope(q, T), target._rope(k, T)
        cache["k"].append(k)
        cache["v"].append(v)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, -1)
        x = x + s("o_proj")(y)
        cache["xmid"].append(x)
        h2 = VM.rms(x, target.norms[2 * i + 1], eps)
        f = s("c_fc")(h2)
        g = VM.gelu_tanh(f)
        x = x + s("down_proj")(g)
        for kind, inp in (("q_proj", h), ("k_proj", h), ("v_proj", h), ("o_proj", y), ("c_fc", h2), ("down_proj", g)):
            cache["site_in"][(i, kind)] = inp
    cache["logp"] = torch.log_softmax(VM.rms(x, target.ln_f, eps) @ target.wte.T, -1)
    return cache


@torch.no_grad()
def row_logprobs(target: VM.Target, cache: dict, b: torch.Tensor, p: torch.Tensor, layer: int, kind: str,
                 delta: torch.Tensor) -> torch.Tensor:
    """Next-token log-probabilities at rows (b, p) [E] with `delta` [E, d_out] added to site (layer, kind)'s
    output at that row only. Rows before p are the clean ones (cache); row p is recomputed from `layer` on,
    its own keys and values included, so the result is exact for an edit at row p alone."""
    E, T = b.shape[0], cache["k"][0].shape[2]
    H, hd, eps = target.n_head, target.hd, target.eps
    ar = torch.arange(E, device=b.device)
    later = (torch.arange(T, device=b.device)[None, :] > p[:, None])[:, None, :]  # [E, 1, T]
    cos, sin = target.cos[p][:, None, :], target.sin[p][:, None, :]
    rope = lambda t: t * cos + torch.cat([-t[..., hd // 2:], t[..., :hd // 2]], -1) * sin

    def add(k, out):
        return out + delta if k == kind and i == layer else out

    x = None
    for i in range(layer, target.n_layer):
        s = lambda k: target.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}").W
        if i == layer and kind in ("c_fc", "down_proj"):
            x = cache["xmid"][i][b, p]
        else:
            if x is None:
                x = cache["xin"][i][b, p]
            h = VM.rms(x, target.norms[2 * i], eps)
            q = rope(add("q_proj", h @ s("q_proj").T).view(E, H, hd))
            k = rope(add("k_proj", h @ s("k_proj").T).view(E, H, hd))
            v = add("v_proj", h @ s("v_proj").T).view(E, H, hd)
            K, V = cache["k"][i][b].clone(), cache["v"][i][b].clone()  # [E, H, T, hd]
            K[ar, :, p], V[ar, :, p] = k, v
            z = torch.einsum("ehd,ehtd->eht", q, K) / math.sqrt(hd)
            att = z.masked_fill(later, float("-inf")).softmax(-1)
            y = torch.einsum("eht,ehtd->ehd", att, V).reshape(E, H * hd)
            x = x + add("o_proj", y @ s("o_proj").T)
        h2 = VM.rms(x, target.norms[2 * i + 1], eps)
        g = VM.gelu_tanh(add("c_fc", h2 @ s("c_fc").T))
        x = x + add("down_proj", g @ s("down_proj").T)
    return torch.log_softmax(VM.rms(x, target.ln_f, eps) @ target.wte.T, -1)


# ------------------------------------------------------------------ build


def pass_a(vpd, names, tokens, batch, total, dev, state_path, log):
    """Split A: active counts, CI sums, the top-TOP rows per subcomponent (resumable from state_path)."""
    n = tokens.shape[0]
    if state_path.exists():
        st = torch.load(state_path, map_location=dev)
        log(f"split A: resuming at row {st['next']}")
    else:
        st = {"next": 0, "count": torch.zeros(total, device=dev, dtype=torch.int64),
              "ci": torch.zeros(total, device=dev), "positions": 0,
              "z": torch.full((TOP, total), -float("inf"), device=dev), "row": torch.full((TOP, total), -1, device=dev, dtype=torch.int32),
              "pos": torch.zeros((TOP, total), device=dev, dtype=torch.int32), "act": torch.zeros((TOP, total), device=dev),
              "seconds": 0.0}
    t0, start = time.time(), st["next"]
    for r0 in range(st["next"], n, batch):
        ids = tokens[r0:r0 + batch].to(dev)
        z, a = ci_and_activity(vpd, names, ids)
        z, a = z[:, 1:], a[:, 1:]
        st["count"] += (z > 0).sum((0, 1))
        st["ci"] += z.clamp(0, 1).sum((0, 1))
        st["positions"] += z.shape[0] * z.shape[1]
        best, where = z.max(1)  # [B, total]
        act = a.gather(1, where[:, None, :])[:, 0]
        allz = torch.cat([st["z"], best])
        top, sel = allz.topk(TOP, 0)
        rowid = torch.arange(r0, r0 + ids.shape[0], device=dev, dtype=torch.int32)[:, None].expand_as(where)
        st["z"] = top
        st["row"] = torch.cat([st["row"], rowid]).gather(0, sel)
        st["pos"] = torch.cat([st["pos"], (where + 1).int()]).gather(0, sel)
        st["act"] = torch.cat([st["act"], act]).gather(0, sel)
        st["next"] = r0 + ids.shape[0]
        done = st["next"] - start
        if (r0 // batch) % 25 == 24 or st["next"] >= n:
            st["seconds"] += time.time() - t0
            t0 = time.time()
            torch.save(st, state_path)
            log(f"split A: {st['next']}/{n} rows, {done / max(st['seconds'], 1e-9):.2f} rows/s this session")
    return st


def pass_b(vpd, names, tokens, triggers, batch, total, dev, log):
    """Split B: active counts, trigger hits, and SAMPLES active positions per subcomponent drawn uniformly
    (the SAMPLES largest of independent uniform keys, one per active position: a uniform draw without
    replacement whatever the batching)."""
    n = tokens.shape[0]
    count = torch.zeros(total, device=dev, dtype=torch.int64)
    hits = torch.zeros(total, device=dev, dtype=torch.int64)
    caught = torch.zeros(total, device=dev, dtype=torch.int64)
    key = torch.full((SAMPLES, total), -1.0, device=dev)
    srow = torch.full((SAMPLES, total), -1, device=dev, dtype=torch.int32)
    spos = torch.zeros((SAMPLES, total), device=dev, dtype=torch.int32)
    positions, t0 = 0, time.time()
    for r0 in range(0, n, batch):
        ids = tokens[r0:r0 + batch].to(dev)
        z, _ = ci_and_activity(vpd, names, ids)
        on = z[:, 1:] > 0  # [B, T-1, total]
        B, T1 = on.shape[:2]
        positions += B * T1
        count += on.sum((0, 1))
        tok = ids[:, 1:].reshape(-1)
        flat = on.reshape(-1, total)
        for c0 in range(0, flat.shape[0], 128):
            m = (tok[c0:c0 + 128, None, None] == triggers[None]).any(-1)  # [<=128, total]
            hits += m.sum(0)
            caught += (m & flat[c0:c0 + 128]).sum(0)
        r = torch.rand(flat.shape, device=dev)
        r = torch.where(flat, r, torch.full_like(r, -1.0))
        k, i = r.topk(SAMPLES, 0)
        allk, sel = torch.cat([key, k]).topk(SAMPLES, 0)
        key = allk
        srow = torch.cat([srow, (r0 + i // T1).int()]).gather(0, sel)
        spos = torch.cat([spos, (i % T1 + 1).int()]).gather(0, sel)
        del r
        if (r0 // batch) % 25 == 24 or r0 + batch >= n:
            log(f"split B: {min(r0 + batch, n)}/{n} rows, {min(r0 + batch, n) / (time.time() - t0):.2f} rows/s")
    srow = torch.where(key > 0, srow, torch.full_like(srow, -1))
    return {"count": count, "hits": hits, "caught": caught, "positions": positions, "row": srow, "pos": spos,
            "seconds": time.time() - t0}


def effects(vpd, names, site, index, tokens, samples, seen, batch, dev, log):
    """The measured contribution at every sample (module note), with the lens comparison for lens sites."""
    target = vpd.target
    total = site.shape[0]
    kl = torch.full((SAMPLES, total), float("nan"))
    corr = torch.full((SAMPLES, total), float("nan"))
    overlap = torch.full((SAMPLES, total), float("nan"))
    up_t = torch.full((SAMPLES, total, EFFECT), -1, dtype=torch.int32)
    dn_t = torch.full((SAMPLES, total, EFFECT), -1, dtype=torch.int32)
    up_v = torch.zeros((SAMPLES, total, EFFECT), dtype=torch.float16)
    dn_v = torch.zeros((SAMPLES, total, EFFECT), dtype=torch.float16)
    srow, spos = samples["row"].cpu(), samples["pos"].cpu()
    site_t, index_t = torch.as_tensor(site.astype(np.int64)), torch.as_tensor(index.astype(np.int64))
    dirs = {}
    for i, n in enumerate(names):
        layer, kind = int(n.split(".")[1]), n.rsplit(".", 1)[1]
        if kind in LENS_KINDS:
            dirs[i] = write_directions(target, layer, kind)
    neg = torch.full((target.wte.shape[0],), -float("inf"), device=dev)
    unseen = ~seen
    t0, done = time.time(), 0
    for r0 in range(0, tokens.shape[0], batch):
        want = (srow >= r0) & (srow < r0 + batch)
        if not want.any():
            continue
        cache = clean_cache(target, tokens[r0:r0 + batch].to(dev))
        k_idx, c_idx = want.nonzero(as_tuple=True)
        for si in site_t[c_idx].unique().tolist():
            sel = site_t[c_idx] == si
            ks, cs = k_idx[sel], c_idx[sel]
            name = names[si]
            layer, kind = int(name.split(".")[1]), name.rsplit(".", 1)[1]
            st = target.site(name)
            for e0 in range(0, cs.shape[0], 128):
                kk, cc = ks[e0:e0 + 128], cs[e0:e0 + 128]
                b = (srow[kk, cc] - r0).long().to(dev)
                p = spos[kk, cc].long().to(dev)
                j = index_t[cc].to(dev)
                x = cache["site_in"][(layer, kind)][b, p]  # [E, d_in]
                a = (x * st.V[:, j].T).sum(-1)
                logp = row_logprobs(target, cache, b, p, layer, kind, -a[:, None] * st.U[j])
                clean = cache["logp"][b, p]
                contrib = clean - logp
                kl[kk, cc] = (clean.exp() * contrib).sum(-1).float().cpu()
                dp = clean.exp() - logp.exp()  # the change of each token's probability
                v, t = torch.where(unseen[None], neg, dp).topk(EFFECT, -1)
                up_t[kk, cc], up_v[kk, cc] = t.int().cpu(), v.half().cpu()
                v, t = torch.where(unseen[None], neg, -dp).topk(EFFECT, -1)
                dn_t[kk, cc], dn_v[kk, cc] = t.int().cpu(), (-v).half().cpu()
                if si in dirs:
                    pred = a[:, None] * lens_values(target, dirs[si][j])  # [E, V]
                    ps, cs_ = pred[:, seen], contrib[:, seen]
                    ps, cs_ = ps - ps.mean(-1, keepdim=True), cs_ - cs_.mean(-1, keepdim=True)
                    corr[kk, cc] = ((ps * cs_).sum(-1) / (ps.norm(dim=-1) * cs_.norm(dim=-1)).clamp_min(1e-30)).float().cpu()
                    tp = torch.where(unseen[None], neg, pred).topk(EFFECT, -1).indices
                    tc = torch.where(unseen[None], neg, contrib).topk(EFFECT, -1).indices
                    overlap[kk, cc] = ((tp[:, :, None] == tc[:, None, :]).any(-1).float().mean(-1)).cpu()
                done += cc.shape[0]
        del cache
        log(f"effects: rows {r0}..{r0 + batch}, {done} edits, {done / (time.time() - t0):.1f} edits/s")
    return {"kl": kl, "corr": corr, "overlap": overlap, "up_t": up_t, "up_v": up_v, "dn_t": dn_t, "dn_v": dn_v,
            "seconds": time.time() - t0, "edits": done}


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "build"
    work.mkdir(exist_ok=True)
    logf = open(out / "build.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    dev = args.device
    torch.manual_seed(args.seed)
    target = VM.load_target(dev)
    vpd = VM.load_vpd(target, dev)
    names, site, index = layout(vpd)
    total = site.shape[0]
    a0, a1 = rows(args.train)
    b0, b1 = rows(args.heldout)
    assert a1 <= b0 or b1 <= a0, "the splits overlap"
    t0 = time.time()
    train = VM.val_tokens(a1 - a0, 512, a0)
    held = VM.val_tokens(b1 - b0, 512, b0)
    log(f"tokens: split A rows {a0}..{a1}, split B rows {b0}..{b1} ({time.time() - t0:.1f} s)")

    st = pass_a(vpd, names, train, args.batch, total, dev, work / f"pass_a_{a0}_{a1}.pt", log)
    seen = (torch.bincount(train.reshape(-1), minlength=target.wte.shape[0]) >= MIN_COUNT).to(dev)
    # Top contexts: windows, and the trigger sets (the token at each context's maximum).
    row, pos = st["row"].cpu().long(), st["pos"].cpu().long()
    valid = row >= 0
    padded = torch.nn.functional.pad(train, (BEFORE, AFTER), value=PAD)
    offs = torch.arange(-BEFORE, AFTER + 1)
    rr, pp = row.clamp_min(0), pos  # rows of split A, counted from its first
    windows = padded[rr[..., None], pp[..., None] + BEFORE + offs]  # [TOP, total, 16]
    windows[~valid] = PAD
    windows = windows.to(torch.int32).to(torch.uint16)
    trig = torch.where(valid, train[rr, pp], torch.full_like(rr, -1)).T.contiguous().to(dev)  # [total, TOP]

    sb = pass_b(vpd, names, held, trig, args.batch, total, dev, log)
    ef = effects(vpd, names, site, index, held, sb, seen, args.batch, dev, log)

    # The write lens, oriented by the sign of the median activity over the top contexts.
    sign = torch.sign(torch.nanmedian(torch.where(valid, st["act"].cpu(), torch.tensor(float("nan"))), 0).values)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    lens_up_t = torch.full((total, LENS), -1, dtype=torch.int32)
    lens_dn_t = torch.full((total, LENS), -1, dtype=torch.int32)
    lens_up_v = torch.zeros((total, LENS), dtype=torch.float16)
    lens_dn_v = torch.zeros((total, LENS), dtype=torch.float16)
    neg = torch.full((target.wte.shape[0],), -float("inf"), device=dev)
    off = 0
    for n in names:
        layer, kind = int(n.split(".")[1]), n.rsplit(".", 1)[1]
        C = vpd.C[n]
        r = write_directions(target, layer, kind)
        if r is not None:
            for c0 in range(0, C, 512):
                w = lens_values(target, r[c0:c0 + 512]) * sign[off + c0:off + c0 + 512, None].to(dev)
                v, t = torch.where(seen[None], w, neg).topk(LENS, -1)
                lens_up_t[off + c0:off + c0 + 512], lens_up_v[off + c0:off + c0 + 512] = t.int().cpu(), v.half().cpu()
                v, t = torch.where(seen[None], -w, neg).topk(LENS, -1)
                lens_dn_t[off + c0:off + c0 + 512], lens_dn_v[off + c0:off + c0 + 512] = t.int().cpu(), (-v).half().cpu()
        off += C
    head = np.concatenate([heads(target, n) for n in names])

    count_b = sb["count"].cpu().double()
    absolute = lambda r, first: torch.where(r >= 0, r + first, torch.full_like(r, -1)).T.contiguous().cpu()
    arrays = {
        "site": torch.as_tensor(site), "index": torch.as_tensor(index), "head": torch.as_tensor(head),
        "active_rate": (st["count"].cpu().double() / st["positions"]).float(), "mean_ci": (st["ci"].cpu().double() / st["positions"]).float(),
        "top_row": absolute(st["row"], a0), "top_pos": st["pos"].T.contiguous().cpu(),
        "top_z": st["z"].T.contiguous().half().cpu(), "top_ci": st["z"].T.contiguous().clamp(0, 1).half().cpu(),
        "top_act": st["act"].T.contiguous().half().cpu(), "top_window": windows.transpose(0, 1).contiguous(),
        "lens_sign": sign.to(torch.int8), "lens_up_token": lens_up_t, "lens_up_value": lens_up_v,
        "lens_down_token": lens_dn_t, "lens_down_value": lens_dn_v,
        "heldout_rate": (count_b / sb["positions"]).float(),
        "trigger_precision": (sb["caught"].cpu().double() / sb["hits"].cpu().double().clamp_min(1)).float(),
        "trigger_recall": (sb["caught"].cpu().double() / count_b.clamp_min(1)).float(),
        "trigger_positions": sb["hits"].float().cpu(),
        "effect_row": absolute(sb["row"], b0), "effect_pos": sb["pos"].T.contiguous().cpu(),
        "effect_kl": ef["kl"].T.contiguous(), "effect_up_token": ef["up_t"].transpose(0, 1).contiguous(),
        "effect_up_value": ef["up_v"].transpose(0, 1).contiguous(), "effect_down_token": ef["dn_t"].transpose(0, 1).contiguous(),
        "effect_down_value": ef["dn_v"].transpose(0, 1).contiguous(),
        "lens_effect_corr": ef["corr"].T.contiguous(), "lens_effect_overlap": ef["overlap"].T.contiguous(),
    }
    save_file({k: v.contiguous() for k, v in arrays.items()}, str(out / "atlas.safetensors"))
    meta = {
        "version": 1, "names": names, "counts": {n: vpd.C[n] for n in names}, "total": total,
        "target": str(VM.TARGET_DIR), "vpd": str(VM.VPD_PTH), "data": "Pile validation rows (vpd_model.val_tokens), 512 tokens",
        "train_rows": [a0, a1], "heldout_rows": [b0, b1], "positions_train": st["positions"], "positions_heldout": sb["positions"],
        "seed": args.seed, "min_count": MIN_COUNT, "top": TOP, "lens": LENS, "effect": EFFECT, "samples": SAMPLES, "window": [BEFORE, AFTER],
        "seconds": {"split_a": st["seconds"], "split_b": sb["seconds"], "effects": ef["seconds"]},
        "speed": {"split_a_rows_per_s": (a1 - a0) / max(st["seconds"], 1e-9), "split_b_rows_per_s": (b1 - b0) / max(sb["seconds"], 1e-9),
                  "edits_per_s": ef["edits"] / max(ef["seconds"], 1e-9)},
        "edits": ef["edits"], "device": dev,
        "fields": __doc__,
    }
    (out / "atlas.json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"wrote {out / 'atlas.safetensors'} ({(out / 'atlas.safetensors').stat().st_size >> 20} MB)")


# ------------------------------------------------------------------ look-ahead


def lookahead(args):
    """The look-ahead fields (module note), added to atlas.safetensors."""
    from method.vpd.parts import Activity

    out = Path(args.out)
    meta = json.loads((out / "atlas.json").read_text())
    arrays = load_file(str(out / "atlas.safetensors"))
    a0, a1 = meta["train_rows"]
    per_layer = sum(meta["counts"][n] for n in meta["names"][:6])
    total = meta["total"]
    T1 = 511
    acts = [Activity(out / "parts" / f"activity_train_L{layer}.npz") for layer in range(4)]
    # Candidate cut positions: every subcomponent's top-3 contexts, as global positions of split A.
    row, pos = arrays["top_row"][:, :3].long(), arrays["top_pos"][:, :3].long()
    ok = row >= 0
    cand = torch.unique(((row - a0) * T1 + pos - 1)[ok]).numpy()
    # Which subcomponents are active at each candidate (from the activity files).
    W, Cs = [], []
    for layer, act in enumerate(acts):
        idx = np.nonzero(np.isin(act.positions, cand))[0]
        Cs.append(np.searchsorted(act.indptr, idx, side="right") - 1 + layer * per_layer)
        W.append(np.searchsorted(cand, act.positions[idx]))
    W, Cs = np.concatenate(W), np.concatenate(Cs)
    order = np.argsort(W, kind="stable")
    on_at = np.split(Cs[order], np.searchsorted(W[order], np.arange(1, len(cand))))
    # Greedy cover: cuts until every subcomponent active at some candidate is covered `need` times (or no candidate
    # covers it further); each step takes the candidate covering the most still-needed subcomponents.
    need = np.full(total, args.need, np.int64)
    gain = np.array([len(x) for x in on_at], np.int64)
    heap = [(-g, k) for k, g in enumerate(gain.tolist()) if g > 0]
    heapq.heapify(heap)
    cuts = []
    while heap:
        g, k = heapq.heappop(heap)
        real = int((need[on_at[k]] > 0).sum())
        if real == 0:
            continue
        if heap and real < -heap[0][0]:
            heapq.heappush(heap, (-real, k))
            continue
        cuts.append(k)
        need[on_at[k]] -= 1
    print(f"{len(cand)} candidate positions, {len(cuts)} cuts", flush=True)
    if args.dry:
        return
    dev = args.device
    target = VM.load_target(dev)
    vpd = VM.load_vpd(target, dev)
    names = VM.site_names()
    tokens = VM.val_tokens(a1 - a0, 512, a0)
    measured = torch.zeros(total, dtype=torch.int64)
    turned_off = torch.zeros(total, dtype=torch.int64)
    cut_off = torch.zeros(total, dtype=torch.int64)
    turned_on = torch.zeros(total, dtype=torch.int64)
    t0 = time.time()
    for b0 in range(0, len(cuts), args.batch):
        ks = cuts[b0:b0 + args.batch]
        g = cand[ks]
        r, p = g // T1, g % T1 + 1
        other = (r + (a1 - a0) // 2) % (a1 - a0)
        ids = tokens[r].clone()
        for e in range(len(ks)):
            ids[e, p[e] + 1:] = tokens[other[e], p[e] + 1:]
        z, _ = ci_and_activity(vpd, names, ids.to(dev))
        now = (z[torch.arange(len(ks)), torch.as_tensor(p).to(dev)] > 0).cpu()  # [E, total]
        for e, k in enumerate(ks):
            was = torch.zeros(total, dtype=torch.bool)
            was[torch.as_tensor(on_at[k])] = True
            measured += was.long()
            turned_off += (was & ~now[e]).long()
            cut_off += (~was).long()
            turned_on += (~was & now[e]).long()
        if (b0 // args.batch) % 50 == 49 or b0 + args.batch >= len(cuts):
            print(f"{time.strftime('%H:%M:%S')} lookahead: {min(b0 + args.batch, len(cuts))}/{len(cuts)} cuts, "
                  f"{min(b0 + args.batch, len(cuts)) / (time.time() - t0):.2f}/s", flush=True)
    arrays["lookahead_off"] = torch.where(measured > 0, turned_off.double() / measured.clamp_min(1), torch.tensor(float("nan"))).float()
    arrays["lookahead_n"] = measured.int()
    arrays["lookahead_on"] = (turned_on.double() / cut_off.clamp_min(1)).float()
    save_file({k: v.contiguous() for k, v in arrays.items()}, str(out / "atlas.safetensors"))
    meta["lookahead"] = {"cuts": len(cuts), "candidates": int(len(cand)), "need": args.need, "seconds": time.time() - t0,
                         "cuts_per_s": len(cuts) / max(time.time() - t0, 1e-9)}
    (out / "atlas.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"look-ahead: {len(cuts)} cuts in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ loader


class Atlas:
    """The atlas's files, read once; entry(site, index) gives one subcomponent's facts with tokens decoded."""

    def __init__(self, root: Path = DEFAULT):
        self.root = Path(root)
        self.meta = json.loads((self.root / "atlas.json").read_text())
        self.a = load_file(str(self.root / "atlas.safetensors"))
        self.names = self.meta["names"]
        self.offset = dict(zip(self.names, np.concatenate([[0], np.cumsum([self.meta["counts"][n] for n in self.names])])[:-1].tolist()))
        self._tok = None
        self._connectome = None

    def flat(self, site: str, index: int) -> int:
        if not 0 <= index < self.meta["counts"][site]:
            raise IndexError(index)
        return self.offset[site] + index

    def decode(self, ids) -> list[str]:
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(VM.TARGET_DIR / "tokenizer.json"))
        return [self._tok.decode([int(i)]) if 0 <= int(i) < PAD else "" for i in ids]

    def partners(self, site: str, index: int, kind: str | None = None):
        """The weight connectome's partners (method/vpd/connectome.py): every edge kind when kind is None."""
        if self._connectome is None:
            from method.vpd.connectome import Connectome
            self._connectome = Connectome(self.root / "connectome")
        if kind is not None:
            return self._connectome.partners(site, index, kind)
        kinds = sorted({k[len(site) + 1:].rsplit(".", 1)[0] for k in self._connectome.metadata["tables"] if k.startswith(site + ".")})
        return {k: self._connectome.partners(site, index, k) for k in kinds}

    def entry(self, site: str, index: int) -> dict:
        i, a = self.flat(site, index), self.a
        tokens = lambda key: self.decode(a[key][i].tolist())
        contexts = []
        for k in range(a["top_row"].shape[1]):
            if int(a["top_row"][i, k]) < 0:
                continue
            w = a["top_window"][i, k].tolist()
            contexts.append({"row": int(a["top_row"][i, k]), "pos": int(a["top_pos"][i, k]), "z": float(a["top_z"][i, k]),
                             "ci": float(a["top_ci"][i, k]), "activity": float(a["top_act"][i, k]),
                             "before": "".join(self.decode(w[:BEFORE])), "token": self.decode([w[BEFORE]])[0],
                             "after": "".join(self.decode(w[BEFORE + 1:]))})
        samples = []
        for k in range(a["effect_row"].shape[1]):
            if int(a["effect_row"][i, k]) < 0:
                continue
            samples.append({"row": int(a["effect_row"][i, k]), "pos": int(a["effect_pos"][i, k]), "kl": float(a["effect_kl"][i, k]),
                            "raised": list(zip(self.decode(a["effect_up_token"][i, k].tolist()), a["effect_up_value"][i, k].float().tolist())),
                            "lowered": list(zip(self.decode(a["effect_down_token"][i, k].tolist()), a["effect_down_value"][i, k].float().tolist())),
                            "lens_corr": float(a["lens_effect_corr"][i, k]), "lens_top10_overlap": float(a["lens_effect_overlap"][i, k])})
        has_lens = int(a["lens_up_token"][i, 0]) >= 0
        return {
            "site": site, "index": index, "head": int(a["head"][i]),
            "active_rate": float(a["active_rate"][i]), "mean_ci": float(a["mean_ci"][i]), "heldout_rate": float(a["heldout_rate"][i]),
            "lens": None if not has_lens else {
                "sign": int(a["lens_sign"][i]),
                "raised": list(zip(tokens("lens_up_token"), a["lens_up_value"][i].float().tolist())),
                "lowered": list(zip(tokens("lens_down_token"), a["lens_down_value"][i].float().tolist()))},
            "contexts": contexts,
            "trigger": {"precision": float(a["trigger_precision"][i]), "recall": float(a["trigger_recall"][i]),
                        "positions": float(a["trigger_positions"][i]), "base_rate": float(a["heldout_rate"][i])},
            "effects": samples,
        }


def show(args):
    e = Atlas(Path(args.out)).entry(args.site, args.index)
    words = lambda pairs: " ".join(repr(t) for t, _ in pairs)
    print(f"{e['site']} {e['index']} head {e['head']}: active {e['active_rate']:.4f} of split A's positions "
          f"(held out {e['heldout_rate']:.4f}), mean CI {e['mean_ci']:.4f}")
    if e["lens"]:
        print(f"  lens (sign {e['lens']['sign']:+d}) raises {words(e['lens']['raised'])}")
        print(f"                lowers {words(e['lens']['lowered'])}")
    t = e["trigger"]
    print(f"  trigger tokens: precision {t['precision']:.3f} over {t['positions']:.0f} positions (base rate "
          f"{t['base_rate']:.4f}), recall {t['recall']:.3f}")
    for c in e["contexts"]:
        print(f"  row {c['row']:>5} pos {c['pos']:>3} ci {c['ci']:.2f} a {c['activity']:+.2f}: "
              f"{c['before']!r} [[{c['token']!r}]] {c['after']!r}")
    for s in e["effects"]:
        print(f"  effect row {s['row']} pos {s['pos']}: kl {s['kl']:.4f}, raises {words(s['raised'][:5])}, "
              f"lowers {words(s['lowered'][:5])}, lens corr {s['lens_corr']:.3f}, top-10 overlap {s['lens_top10_overlap']:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(DEFAULT))
    b.add_argument("--train", default="0:5000")
    b.add_argument("--heldout", default="5000:6000")
    b.add_argument("--batch", type=int, default=8)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", default="mps")
    s = sub.add_parser("show")
    s.add_argument("site")
    s.add_argument("index", type=int)
    s.add_argument("--out", default=str(DEFAULT))
    k = sub.add_parser("lookahead")
    k.add_argument("--out", default=str(DEFAULT))
    k.add_argument("--need", type=int, default=2)
    k.add_argument("--batch", type=int, default=8)
    k.add_argument("--device", default="mps")
    k.add_argument("--dry", action="store_true", help="only count the cuts")
    args = ap.parse_args()
    {"build": build, "show": show, "lookahead": lookahead}[args.cmd](args)


if __name__ == "__main__":
    main()
