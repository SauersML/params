"""The neuron atlas of a Qwen3 model (the lead's 10-15 direction, item 5): for every MLP neuron, its strongest activations
of either sign over a broad long-context corpus with their windows and SwiGLU sign regimes, its activation histogram and
firing rates, its weight neighbours and direct logit effect; for every attention head, where it looks. The activation is
the input of down_proj, silu(gate) * up; a window's first token (the attention sink) is left out of every list and kept
apart (pos0).

  atlas.py corpus ROOT [--tokens 50000000] [--model Qwen/Qwen3-1.7B] [--window 1024]
      the corpus, streamed from public Hugging Face datasets in MIX's token shares, interleaved in a fixed order (any
      prefix holds the mix): each document's first WINDOW tokens (at least MIN_TOKENS), chat rendered with the model's
      chat template (thinking off); ROOT/corpus/tokens.npy (int32), offsets.npy (document d: tokens[off[d]:off[d + 1]]),
      sources.npy (MIX index per document), meta.json
  atlas.py collect ROOT --shard K --shards S [--batch-tokens 32768] [--model M]   documents d with d % S == K
  atlas.py merge ROOT                       the shards into ROOT/layers/LNN.npz (Atlas reads them)
  atlas.py weights ROOT [--model M]         ROOT/weights.npz: neighbours by weight product, direct logit effects
  atlas.py heads ROOT [--texts 2000]        ROOT/heads.npz: per head its sink share, offset profile, top positions

Atlas(ROOT).neuron(layer, index) gives one neuron's record with its examples' text; see the class."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

TOP = 64                    # examples per pole
NEIGHBOURS, LOGITS = 32, 20
BEFORE, AFTER = 256, 32     # tokens of context around an example
MIN_TOKENS = 64
LO, DECADES, PER_DEC = -4, 7, 16   # histogram: |a| from 1e-4 to 1e3 in 16 bins a decade, each sign, and one bin under 1e-4
NB = 2 * DECADES * PER_DEC + 1
ZERO = DECADES * PER_DEC    # the bin of |a| < 1e-4; below it the negative bins (most negative first), above it the positive
HIST_EVERY = 5              # the histogram over every 5th batch (about a fifth of the corpus)
RATES = (0.1, 0.3, 0.5)     # firing rates above these fractions of the pole's corpus maximum
OFFSETS = [0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513]   # head offset profile bins: q - k in [OFFSETS[i], OFFSETS[i + 1])
MIX = [  # name, dataset, config, split, field, share of tokens
    ("web", "HuggingFaceFW/fineweb", "sample-10BT", "train", "text", 0.40),
    ("pile", "monology/pile-uncopyrighted", None, "train", "text", 0.20),
    ("chat", "HuggingFaceH4/ultrachat_200k", None, "train_sft", "messages", 0.15),
    ("code", "codeparrot/codeparrot-clean", None, "train", "content", 0.10),
    ("deu", "HuggingFaceFW/fineweb-2", "deu_Latn", "train", "text", 0.02),
    ("fra", "HuggingFaceFW/fineweb-2", "fra_Latn", "train", "text", 0.02),
    ("spa", "HuggingFaceFW/fineweb-2", "spa_Latn", "train", "text", 0.02),
    ("rus", "HuggingFaceFW/fineweb-2", "rus_Cyrl", "train", "text", 0.02),
    ("cmn", "HuggingFaceFW/fineweb-2", "cmn_Hani", "train", "text", 0.02),
    ("math", "open-web-math/open-web-math", None, "train", "text", 0.05),
]


def regime(g, u) -> str:
    """The SwiGLU sign regime, gate sign then up sign: ++ and -- (the echo of a negative gate) give positive values."""
    return ("+" if g > 0 else "-") + ("+" if u > 0 else "-")


# ---- the corpus

def corpus(a) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    out = a.root / "corpus"
    out.mkdir(parents=True, exist_ok=True)

    def docs(name, path, cfg, split, field):
        """A source's documents as token lists (its first WINDOW tokens, at least MIN_TOKENS), tokenized 64 at a time."""
        try:                                                  # parquet sources: their text column alone (row groups are large)
            ds = load_dataset(path, cfg, split=split, streaming=True, columns=[field])
        except (TypeError, ValueError):
            ds = load_dataset(path, cfg, split=split, streaming=True)
        it = iter(ds)
        while True:
            rows = [r for _, r in zip(range(64), it)]
            if not rows:
                return
            if field == "messages":
                texts = [tok.apply_chat_template(r[field], tokenize=False, enable_thinking=False) for r in rows]
            else:
                texts = [r[field] for r in rows]
            for x in tok([t[:16 * a.window] for t in texts], add_special_tokens=False)["input_ids"]:
                if len(x) >= MIN_TOKENS:
                    yield x[:a.window]

    its = [docs(*m[:5]) for m in MIX]
    got = [0] * len(MIX)
    ndocs = [0] * len(MIX)
    chunks, sources, lengths = [], [], []
    total, t0 = 0, time.time()
    while total < a.tokens:
        s = max(range(len(MIX)), key=lambda j: MIX[j][5] * max(total, 1) - got[j])   # the source furthest under its share
        x = next(its[s], None)
        if x is None:
            raise SystemExit(f"source {MIX[s][0]} ran out")
        chunks.append(np.asarray(x, dtype=np.int32))
        sources.append(s)
        lengths.append(len(x))
        got[s] += len(x)
        ndocs[s] += 1
        total += len(x)
        if len(chunks) % 20000 == 0:
            print(f"{total / 1e6:.1f}M tokens, {len(chunks)} documents, {time.time() - t0:.0f} s", flush=True)
    np.save(out / "tokens.npy", np.concatenate(chunks))
    np.save(out / "offsets.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    np.save(out / "sources.npy", np.asarray(sources, dtype=np.int8))
    (out / "meta.json").write_text(json.dumps({
        "model": a.model, "window": a.window, "min_tokens": MIN_TOKENS, "tokens": total, "documents": len(chunks),
        "mix": [{"name": m[0], "dataset": m[1], "config": m[2], "split": m[3], "share": m[5], "tokens": got[j], "documents": ndocs[j]}
                for j, m in enumerate(MIX)]}, indent=1))
    print(f"corpus: {total} tokens, {len(chunks)} documents, {time.time() - t0:.0f} s -> {out}", flush=True)
    os._exit(0)   # the streams' threads abort the interpreter's exit ("terminate called without an active exception")


def load_corpus(root: Path):
    c = root / "corpus"
    return (np.load(c / "tokens.npy", mmap_mode="r"), np.load(c / "offsets.npy"), json.loads((c / "meta.json").read_text()))


def load_model(name: str, dev, attn: str = "sdpa"):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation=attn).to(dev).eval()


def device(d):
    import torch

    return torch.device(d or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))


def batches(off: np.ndarray, docs: list[int], batch_tokens: int, block: int = 4096):
    """The documents in batches of at most batch_tokens padded tokens, sorted by length within blocks of BLOCK."""
    for s in range(0, len(docs), block):
        blk = sorted(docs[s:s + block], key=lambda d: off[d + 1] - off[d])
        cur, longest = [], 0
        for d in blk:
            n = int(off[d + 1] - off[d])
            if cur and max(longest, n) * (len(cur) + 1) > batch_tokens:
                yield cur
                cur, longest = [], 0
            cur.append(d)
            longest = max(longest, n)
        if cur:
            yield cur


# ---- collection

def collect(a) -> None:
    import torch

    dev = device(a.device)
    tokens, off, meta = load_corpus(a.root)
    model = a.model or meta["model"]
    m = load_model(model, dev)
    L, D = m.config.num_hidden_layers, m.config.intermediate_size
    neg_inf = torch.full((D, TOP), -math.inf, device=dev)
    st = [{p: {"val": neg_inf.clone(), "idx": torch.zeros(D, TOP, dtype=torch.long, device=dev),
               "g": torch.zeros(D, TOP, dtype=torch.float16, device=dev), "u": torch.zeros(D, TOP, dtype=torch.float16, device=dev)}
           for p in "+-"} | {"hist": torch.zeros(D * NB, dtype=torch.long, device=dev),
                             "pos0_max": torch.full((D,), -math.inf, device=dev), "pos0_min": torch.full((D,), math.inf, device=dev)}
          for _ in range(L)]
    cur: dict = {}
    run = {"hist": False}

    def update(l, a_):
        g, u = cur.pop(("g", l)), cur.pop(("u", l))
        valid, gidx = run["valid"], run["gidx"]
        s = st[l]
        a0 = a_[:, 0].float()                                 # each window's first token: the sink, apart
        s["pos0_max"] = torch.maximum(s["pos0_max"], a0.max(0).values)
        s["pos0_min"] = torch.minimum(s["pos0_min"], a0.min(0).values)
        rows = valid.view(-1).nonzero().squeeze(1)
        av = a_.reshape(-1, D)[rows].float().t()              # [D, N]
        gv, uv, gi = g.reshape(-1, D)[rows].t(), u.reshape(-1, D)[rows].t(), gidx.view(-1)[rows]
        for p, v in (("+", av), ("-", -av)):
            k = min(TOP, v.shape[1])
            top = v.topk(k, dim=1)
            ps = s[p]
            cv = torch.cat([ps["val"], top.values], 1)
            ci = torch.cat([ps["idx"], gi[top.indices]], 1)
            cg = torch.cat([ps["g"], gv.gather(1, top.indices).half()], 1)
            cu = torch.cat([ps["u"], uv.gather(1, top.indices).half()], 1)
            sel = cv.topk(TOP, dim=1).indices
            ps["val"], ps["idx"], ps["g"], ps["u"] = cv.gather(1, sel), ci.gather(1, sel), cg.gather(1, sel), cu.gather(1, sel)
        if run["hist"]:
            col = torch.arange(D, device=dev)[:, None] * NB
            for c in range(0, av.shape[1], 4096):
                x = av[:, c:c + 4096]
                b = ((x.abs().clamp_min(1e-30).log10() - LO) * PER_DEC).floor().clamp(0, DECADES * PER_DEC - 1).long()
                b = torch.where(x.abs() < 10 ** LO, torch.full_like(b, ZERO), torch.where(x > 0, ZERO + 1 + b, ZERO - 1 - b))
                s["hist"] += torch.bincount((b + col).reshape(-1), minlength=D * NB)

    hooks = []
    for l, layer in enumerate(m.model.layers):
        mlp = layer.mlp
        hooks += [mlp.gate_proj.register_forward_hook(lambda _m, _i, o, l=l: cur.__setitem__(("g", l), o)),
                  mlp.up_proj.register_forward_hook(lambda _m, _i, o, l=l: cur.__setitem__(("u", l), o)),
                  mlp.down_proj.register_forward_pre_hook(lambda _m, x, l=l: update(l, x[0]))]
    docs = list(range(a.shard, len(off) - 1, a.shards))
    out = a.root / "shards"
    out.mkdir(parents=True, exist_ok=True)
    ntok, nhist, t0 = 0, 0, time.time()
    for bi, bd in enumerate(batches(off, docs, a.batch_tokens)):
        T = max(int(off[d + 1] - off[d]) for d in bd)
        ids = torch.full((len(bd), T), 0, dtype=torch.long)
        mask = torch.zeros(len(bd), T, dtype=torch.long)
        gidx = torch.zeros(len(bd), T, dtype=torch.long)
        for r, d in enumerate(bd):
            n = int(off[d + 1] - off[d])
            ids[r, :n] = torch.from_numpy(np.asarray(tokens[off[d]:off[d + 1]], dtype=np.int64))
            mask[r, :n] = 1
            gidx[r, :n] = torch.arange(int(off[d]), int(off[d]) + n)
        valid = mask.bool().clone()
        valid[:, 0] = False
        run.update(valid=valid.to(dev), gidx=gidx.to(dev), hist=bi % HIST_EVERY == 0)
        with torch.no_grad():
            m.model(input_ids=ids.to(dev), attention_mask=mask.to(dev))   # the decoder alone: no logits needed
        ntok += int(mask.sum())
        nhist += int(valid.sum()) if run["hist"] else 0
        if bi % 50 == 0:
            el = time.time() - t0
            print(f"shard {a.shard}: {ntok / 1e6:.2f}M tokens, {ntok / max(el, 1e-9):.0f} tokens/s, {el:.0f} s", flush=True)
    for h in hooks:
        h.remove()
    torch.save({"layers": [{k: (v.cpu() if torch.is_tensor(v) else {kk: vv.cpu() for kk, vv in v.items()}) for k, v in s.items()} for s in st],
                "tokens": ntok, "hist_tokens": nhist, "seconds": time.time() - t0, "shard": a.shard, "shards": a.shards, "model": model},
               out / f"shard{a.shard}.pt")
    print(f"shard {a.shard} done: {ntok} tokens in {time.time() - t0:.0f} s ({ntok / (time.time() - t0):.0f} tokens/s)", flush=True)


def rates_from_hist(hist: np.ndarray, top: float, sign: int, n: int) -> list[float]:
    """Shares of the histogram's tokens above f * top for f in RATES on one side (sign +1 or -1), the bin holding the
    threshold counted by its log-linear share."""
    out = []
    for f in RATES:
        t = math.log10(max(f * top, 1e-30))
        x = (t - LO) * PER_DEC
        side = hist[ZERO + 1:] if sign > 0 else hist[:ZERO][::-1]
        if x < 0:
            c = side.sum()
        else:
            b = int(min(math.floor(x), DECADES * PER_DEC - 1))
            c = side[b + 1:].sum() + side[b] * max(0.0, 1.0 - (x - b))
        out.append(float(c) / max(n, 1))
    return out


def merge(a) -> None:
    import torch

    files = sorted((a.root / "shards").glob("shard*.pt"))
    parts = [torch.load(f, weights_only=False) for f in files]
    S = parts[0]["shards"]
    assert sorted(p["shard"] for p in parts) == list(range(S)), f"shards present: {sorted(p['shard'] for p in parts)} of {S}"
    L = len(parts[0]["layers"])
    out = a.root / "layers"
    out.mkdir(parents=True, exist_ok=True)
    nh = sum(p["hist_tokens"] for p in parts)
    for l in range(L):
        ls = [p["layers"][l] for p in parts]
        rec = {}
        for p in "+-":
            cv = torch.cat([s[p]["val"] for s in ls], 1)
            sel = cv.topk(TOP, dim=1).indices
            for k in ("val", "idx", "g", "u"):
                rec[f"{'pos' if p == '+' else 'neg'}_{k}"] = torch.cat([s[p][k] for s in ls], 1).gather(1, sel).numpy()
        rec["neg_val"] = -rec["neg_val"]
        D = rec["pos_val"].shape[0]
        hist = sum(s["hist"] for s in ls).view(D, NB).numpy()
        rec["hist"] = hist
        rec["pos0_max"] = torch.stack([s["pos0_max"] for s in ls]).max(0).values.numpy()
        rec["pos0_min"] = torch.stack([s["pos0_min"] for s in ls]).min(0).values.numpy()
        rec["rates"] = np.array([[rates_from_hist(hist[i], float(rec["pos_val"][i, 0]), +1, nh),
                                  rates_from_hist(hist[i], float(-rec["neg_val"][i, 0]), -1, nh)] for i in range(D)], dtype=np.float32)
        np.savez(out / f"L{l:02d}.npz", **rec)
    (a.root / "atlas.json").write_text(json.dumps({
        "model": parts[0]["model"], "layers": L, "neurons_per_layer": int(D), "top": TOP, "tokens": sum(p["tokens"] for p in parts),
        "hist_tokens": nh, "hist": {"lo": LO, "decades": DECADES, "per_decade": PER_DEC, "bins": NB, "zero": ZERO},
        "rates": list(RATES), "collect_seconds": [p["seconds"] for p in parts], "shards": S}, indent=1))
    print(f"merged {S} shards, {L} layers -> {out}", flush=True)


# ---- weights

def weights(a) -> None:
    import torch

    dev = device(a.device)
    m = load_model(a.model or load_corpus(a.root)[2]["model"], dev)   # (--model: before the corpus exists)
    lay = m.model.layers
    L, D = len(lay), m.config.intermediate_size
    f32 = lambda t: t.detach().float()  # noqa: E731
    down = [f32(x.mlp.down_proj.weight) for x in lay]          # [d, D]: column i is neuron i's write
    reads = [(f32(x.mlp.gate_proj.weight) * f32(x.post_attention_layernorm.weight)[None], f32(x.mlp.up_proj.weight) * f32(x.post_attention_layernorm.weight)[None]) for x in lay]
    WU, gf = f32(m.lm_head.weight), f32(m.model.norm.weight)
    del m, lay                                                # (8B: the bf16 model beside the f32 copies would fill a 48 GB card)
    (torch.cuda.empty_cache() if dev.type == "cuda" else None)
    # downstream: per source neuron, the NEIGHBOURS largest |read . write| over later layers' gate and up rows (id = layer * D + j)
    dn_v = torch.zeros(L, D, NEIGHBOURS, device=dev)
    dn_i = torch.full((L, D, NEIGHBOURS), -1, dtype=torch.long, device=dev)
    dn_k = torch.zeros(L, D, NEIGHBOURS, dtype=torch.int8, device=dev)
    up_v = torch.zeros(L, D, NEIGHBOURS, device=dev)
    up_i = torch.full((L, D, NEIGHBOURS), -1, dtype=torch.long, device=dev)
    up_k = torch.zeros(L, D, NEIGHBOURS, dtype=torch.int8, device=dev)

    def keep(v, i, k, nv, ni, nk):
        cv, ci, ck = torch.cat([v, nv], 1), torch.cat([i, ni], 1), torch.cat([k, nk], 1)
        sel = cv.abs().topk(NEIGHBOURS, dim=1).indices
        return cv.gather(1, sel), ci.gather(1, sel), ck.gather(1, sel)

    t0 = time.time()
    for l in range(L - 1):
        for mm in range(l + 1, L):
            for kind in (0, 1):
                S = reads[mm][kind] @ down[l]                     # [D target, D source]
                t = S.abs().topk(NEIGHBOURS, dim=0).indices       # per source column
                dn_v[l], dn_i[l], dn_k[l] = keep(dn_v[l], dn_i[l], dn_k[l], S.gather(0, t).t(), (mm * D + t).t(),
                                                 torch.full_like(t, kind, dtype=torch.int8).t())
                t = S.abs().topk(NEIGHBOURS, dim=1).indices       # per target row
                up_v[mm], up_i[mm], up_k[mm] = keep(up_v[mm], up_i[mm], up_k[mm], S.gather(1, t), l * D + t,
                                                    torch.full_like(t, kind, dtype=torch.int8))
        (torch.cuda.synchronize() if dev.type == "cuda" else torch.mps.synchronize() if dev.type == "mps" else None)
        print(f"weights: source layer {l} done, {time.time() - t0:.0f} s", flush=True)
    lt_i = torch.zeros(L, D, LOGITS, dtype=torch.int32)
    lt_v = torch.zeros(L, D, LOGITS)
    lb_i = torch.zeros(L, D, LOGITS, dtype=torch.int32)
    lb_v = torch.zeros(L, D, LOGITS)
    for l in range(L):
        for c in range(0, D, 1024):
            E = WU @ (gf[:, None] * down[l][:, c:c + 1024])       # [V, chunk]: the write's direct logit effect
            t, b = E.topk(LOGITS, dim=0), (-E).topk(LOGITS, dim=0)
            lt_i[l, c:c + 1024], lt_v[l, c:c + 1024] = t.indices.t().int().cpu(), t.values.t().cpu()
            lb_i[l, c:c + 1024], lb_v[l, c:c + 1024] = b.indices.t().int().cpu(), -b.values.t().cpu()
    np.savez(a.root / "weights.npz", down_val=dn_v.cpu().numpy(), down_id=dn_i.cpu().numpy().astype(np.int32), down_kind=dn_k.cpu().numpy(),
             up_val=up_v.cpu().numpy(), up_id=up_i.cpu().numpy().astype(np.int32), up_kind=up_k.cpu().numpy(),
             logit_top_tok=lt_i.numpy(), logit_top_val=lt_v.numpy(), logit_bot_tok=lb_i.numpy(), logit_bot_val=lb_v.numpy())
    print(f"weights done in {time.time() - t0:.0f} s", flush=True)


# ---- heads

def heads(a) -> None:
    import torch

    dev = device(a.device)
    tokens, off, meta = load_corpus(a.root)
    m = load_model(a.model or meta["model"], dev, attn="eager")
    L, H = m.config.num_hidden_layers, m.config.num_attention_heads
    n = min(a.texts, len(off) - 1)
    sink = torch.zeros(L, H, device=dev)
    prof = torch.zeros(L, H, len(OFFSETS), device=dev)
    nq = 0
    lists = {k: (torch.full((L, H, 32), -1.0, device=dev), torch.zeros(L, H, 32, 3, dtype=torch.long, device=dev)) for k in ("any", "far")}
    edges = torch.tensor(OFFSETS[1:], device=dev)
    t0 = time.time()
    for d in range(n):
        x = torch.from_numpy(np.asarray(tokens[off[d]:off[d + 1]], dtype=np.int64))[None].to(dev)
        T = x.shape[1]
        with torch.no_grad():
            att = m.model(input_ids=x, output_attentions=True).attentions   # L x [1, H, T, T]
        q = torch.arange(T, device=dev)[:, None]
        k = torch.arange(T, device=dev)[None]
        dist = (q - k).clamp_min(0)
        binof = torch.bucketize(dist, edges, right=True)            # [T, T]
        live = (k >= 1) & (k <= q) & (q >= 1)
        for l in range(L):
            A = att[l][0].float()                                     # [H, T, T]
            sink[l] += A[:, 1:, 0].sum(1)
            Am = A * live
            prof[l] += torch.zeros(H, len(OFFSETS), device=dev).scatter_add_(1, binof.view(1, -1).expand(H, -1), Am.reshape(H, -1))
            for key, msk in (("any", live), ("far", live & (dist >= 4))):
                v = (A * msk).reshape(H, -1)
                kk = min(32, v.shape[1])
                tv, ti = v.topk(kk, dim=1)
                pos = torch.stack([torch.full_like(ti, d), ti // T, ti % T], -1)
                cv, cp = torch.cat([lists[key][0][l], tv], 1), torch.cat([lists[key][1][l], pos], 1)
                sel = cv.topk(32, dim=1).indices
                lists[key][0][l], lists[key][1][l] = cv.gather(1, sel), cp.gather(1, sel[..., None].expand(-1, -1, 3))
        nq += T - 1
        if d % 200 == 0:
            print(f"heads: {d}/{n} texts, {time.time() - t0:.0f} s", flush=True)
    np.savez(a.root / "heads.npz", sink=(sink / max(nq, 1)).cpu().numpy(), profile=(prof / max(nq, 1)).cpu().numpy(), offsets=np.array(OFFSETS),
             top_any_w=lists["any"][0].cpu().numpy(), top_any_pos=lists["any"][1].cpu().numpy(),
             top_far_w=lists["far"][0].cpu().numpy(), top_far_pos=lists["far"][1].cpu().numpy(), texts=n)
    print(f"heads done: {n} texts in {time.time() - t0:.0f} s", flush=True)


# ---- the loader

class Atlas:
    """One model's atlas (ROOT as atlas.py writes it). neuron(layer, index) gives:
      max: {"+": corpus maximum, "-": corpus minimum (negative)}; pos0: the sink token's max and min, kept apart
      "+", "-": the TOP examples of each pole, strongest first: value, gate, up, regime (gate sign, up sign), doc, pos
                (in the document), text before / token / after (BEFORE and AFTER tokens), token ids around it
      rates: {"+": {0.1: share, 0.3: ..., 0.5: ...}, "-": {...}}: the share of tokens above that fraction of the pole's
             extreme (from the histogram over hist_tokens tokens)
      hist: (counts [NB], bin edges in signed activation) via histogram()
      down / up: the NEIGHBOURS neurons by weight product, (layer, index, "gate" | "up", value)
      logits: {"+": [(token, value)], "-": [...]}: the write's direct logit effect through the final norm."""

    def __init__(self, root, tokenizer=None):
        self.root = Path(root)
        self.meta = json.loads((self.root / "atlas.json").read_text())
        self.tokens, self.off, self.cmeta = load_corpus(self.root)
        self._layers: dict = {}
        self._w = None
        self._tok = tokenizer

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.meta["model"])
        return self._tok

    def layer(self, l: int) -> dict:
        if l not in self._layers:
            with np.load(self.root / "layers" / f"L{l:02d}.npz") as z:
                self._layers[l] = {k: z[k] for k in z.files}
        return self._layers[l]

    def doc_of(self, gidx: int) -> tuple[int, int]:
        d = int(np.searchsorted(self.off, gidx, side="right") - 1)
        return d, int(gidx - self.off[d])

    def example(self, gidx: int, before: int = BEFORE, after: int = AFTER, text: bool = True) -> dict:
        d, p = self.doc_of(gidx)
        s, e = int(self.off[d]), int(self.off[d + 1])
        ids = np.asarray(self.tokens[max(s, gidx - before):min(e, gidx + after + 1)]).tolist()
        at = gidx - max(s, gidx - before)
        out = {"doc": d, "pos": p, "ids": ids, "at": at}
        if text:
            out.update(before=self.tok.decode(ids[:at]), token=self.tok.decode(ids[at:at + 1]), after=self.tok.decode(ids[at + 1:]))
        return out

    def neuron(self, l: int, i: int, k: int = TOP, text: bool = True) -> dict:
        z = self.layer(l)
        out = {"layer": l, "index": i, "max": {"+": float(z["pos_val"][i, 0]), "-": float(z["neg_val"][i, 0])},
               "pos0": {"max": float(z["pos0_max"][i]), "min": float(z["pos0_min"][i])},
               "rates": {p: dict(zip(self.meta["rates"], map(float, z["rates"][i, j]))) for j, p in enumerate("+-")}}
        for p, pre in (("+", "pos"), ("-", "neg")):
            out[p] = [{"value": float(z[f"{pre}_val"][i, r]), "gate": float(z[f"{pre}_g"][i, r]), "up": float(z[f"{pre}_u"][i, r]),
                       "regime": regime(z[f"{pre}_g"][i, r], z[f"{pre}_u"][i, r]), **self.example(int(z[f"{pre}_idx"][i, r]), text=text)}
                      for r in range(k) if np.isfinite(z[f"{pre}_val"][i, r])]
        if (self.root / "weights.npz").exists():
            w = self.weights()
            D = self.meta["neurons_per_layer"]
            for key in ("down", "up"):
                out[key] = [(int(n) // D, int(n) % D, ("gate", "up")[int(kd)], float(v))
                            for n, kd, v in zip(w[f"{key}_id"][l, i], w[f"{key}_kind"][l, i], w[f"{key}_val"][l, i]) if n >= 0]
            out["logits"] = {"+": [(self.tok.decode([int(t)]), float(v)) for t, v in zip(w["logit_top_tok"][l, i], w["logit_top_val"][l, i])],
                             "-": [(self.tok.decode([int(t)]), float(v)) for t, v in zip(w["logit_bot_tok"][l, i], w["logit_bot_val"][l, i])]}
        return out

    def weights(self) -> dict:
        if self._w is None:
            with np.load(self.root / "weights.npz") as z:
                self._w = {k: z[k] for k in z.files}
        return self._w

    def histogram(self, l: int, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Counts [NB] and the NB + 1 signed bin edges (the middle bin is |a| < 1e-4)."""
        pos = 10.0 ** (LO + np.arange(DECADES * PER_DEC + 1) / PER_DEC)
        return self.layer(l)["hist"][i], np.concatenate([-pos[::-1], pos])

    def heads(self) -> dict:
        with np.load(self.root / "heads.npz") as z:
            return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("root", type=Path)
    c.add_argument("--tokens", type=float, default=50e6)
    c.add_argument("--model", default="Qwen/Qwen3-1.7B")
    c.add_argument("--window", type=int, default=1024)
    for name in ("collect", "merge", "weights", "heads"):
        p = sub.add_parser(name)
        p.add_argument("root", type=Path)
        p.add_argument("--device")
        if name == "collect":
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--shards", type=int, default=1)
            p.add_argument("--batch-tokens", type=int, default=32768)
        if name == "heads":
            p.add_argument("--texts", type=int, default=2000)
        if name in ("collect", "weights", "heads"):
            p.add_argument("--model", help="the model (default: the corpus's tokenizer model; Qwen3 sizes share the tokenizer)")
    a = ap.parse_args()
    {"corpus": corpus, "collect": collect, "merge": merge, "weights": weights, "heads": heads}[a.cmd](a)


if __name__ == "__main__":
    main()
