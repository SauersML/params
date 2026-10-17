"""Amortized circuits (amortize, 10-16): each cut's unit order predicted from cheap per-unit features of the pair in place
of circuit_masks' Adam descent, then circuit_masks' measured steps (verify: the smallest passing prefix and pruning by
measurement; the prediction only orders, every keep or drop is a measured run).

Features (path_features), per unit u of a cut (circuit_masks.cuts_of: readout, movers, source), from one batched forward
and backward of the pair along the masks' own path (the cut's units moved the share m of the way to their value in the
other run, m in POINTS; m = 0 is the plain run):
  lost@m      the slope of the lost share in m_u at m: (h_u - o_u) . dc/dh_u in the clean run with the cut at m, over the gap
  rest@m      the slope of the restored share: the same in the corrupted run with the cut at m
              (h_u clean, o_u corrupted values; dc/dh_u the contrast's gradient at the unit; heads summed over the head's
              dimensions)
and |h_u - o_u|, the clean and corrupted values (heads: norms), RelP's attribution (circuit_finder.Patcher.scores), the
layer and the position counted from the answer and from the first change. ig (no training): the trapezoid mean over
POINTS of lost@m + rest@m, the masks' hinge gradient on m_u (both hinges active) summed along their path.

  circuit_amortize.py data QUESTIONS.json OUT [--split train] [--batch 16] [--steps 30]
      circuit_masks' mask arm (find_batch) with each masked cut's final theta captured; per question OUT/rec/<id>.json
      (the record) and OUT/feat/<id>.npz: per cut the candidate units (the top CAND by theta, by ig, by |ig| and by RelP,
      RAND at random, the circuit's, every head of a head cut) with their features, theta and circuit membership
"""
from __future__ import annotations

import time

import numpy as np
import torch

import circuit_finder as cf
import circuit_masks as cm

POINTS = (0.0, 0.5, 1.0)
FEATURES = ("lost0", "lost5", "lost1", "rest0", "rest5", "rest1", "delta", "clean", "corrupt", "relp", "layer",
            "from_end", "from_first")
GRAD_TOKENS = 1536    # rows x positions of one forward and backward (path_features' chunks)
CAND = 1024           # data: candidates per score
RAND = 4096           # data: candidates at random per neuron cut


def cut_name(pt, spec: dict) -> str:
    return "movers" if "head" in spec["kinds"] else "readout" if list(spec["positions"]) == [pt.T - 1] else "source"


def _rows(pts: list) -> list:
    """The rows of path_features' forward: per pair its two plain runs (m = 0, every cut's), then per cut and m > 0 the
    clean run with the cut at m (noising) and the corrupted run with the cut at m (denoising): (pair, base, cut, m)."""
    rows = []
    for j, pt in enumerate(pts):
        rows += [(j, "clean", None, 0.0), (j, "corrupt", None, 0.0)]
        rows += [(j, b, n, m) for n in cm.cuts_of(pt) for m in POINTS if m > 0 for b in ("clean", "corrupt")]
    return rows


def _chunks(pts: list) -> list:
    """pts in consecutive parts of at most GRAD_TOKENS rows x positions (one forward and backward each)."""
    parts, cur, n = [], [], 0
    for pt in pts:
        t = len(_rows([pt])) * (pt.T - pt.first)
        if cur and n + t > GRAD_TOKENS:
            parts.append(cur)
            cur, n = [], 0
        cur.append(pt)
        n += t
    return parts + ([cur] if cur else [])


def path_features(pts: list, consume, relp: bool = True) -> None:
    """For pairs set up in pts (Patchers of one model; prompts of one length), consume(j, {cut name: {feature: [L, P, W]
    float32 on the device}}) per pair (FEATURES, the cut's positions P, W neurons or heads). One forward and backward
    per chunk of pairs (_chunks), every row from the chunk's first changed position on (the prefix's keys and values the
    setup runs')."""
    at = {id(pt): j for j, pt in enumerate(pts)}
    for part in _chunks(pts):
        rows = _rows(part)
        pt0 = part[0]
        L, T, D, H, hd, dev = pt0.L, pt0.T, pt0.D, pt0.H, pt0.hd, pt0.dev
        assert all(pt.T == T for pt in part), "path_features: prompts of one length"
        start = min(pt.first for pt in part)
        S, B = T - start, len(rows)
        other = {"clean": "corrupt", "corrupt": "clean"}
        cuts = [cm.cuts_of(pt) for pt in part]
        wn, wh = torch.zeros(B, S, 1), torch.zeros(B, S, 1)       # each row's share m at its cut's positions
        for r, (j, b, n, m) in enumerate(rows):
            if n is not None:
                spec = cuts[j][n]
                w = wh if "head" in spec["kinds"] else wn
                w[r, [p - start for p in spec["positions"]]] = m
        dt = pt0.core.model.embed_tokens.weight.dtype
        wn, wh = wn.to(dev, dt), wh.to(dev, dt)
        edit_n, edit_h = bool(wn.any()), bool(wh.any())

        def tgt(key, i, d):                         # each row's other-run values at layer i [B, S, width]
            return torch.stack([part[j].act[other[b]][key][i][start:] for j, b, _, _ in rows]).to(d, dt)

        def neu(i, h):
            return h + wn.to(h.device) * (tgt("neu", i, h.device) - h) if edit_n else h

        def head(i, h):
            return h + wh.to(h.device) * (tgt("head", i, h.device) - h) if edit_h else h
        add = {"neu": [torch.zeros(B, S, D, device=l.mlp.down_proj.weight.device, dtype=dt, requires_grad=True)
                       for l in pt0.layers],
               "head": [torch.zeros(B, S, H * hd, device=l.self_attn.o_proj.weight.device, dtype=dt, requires_grad=True)
                        for l in pt0.layers]}
        x = torch.stack([part[j]._ids_dev[b] for j, b, _, _ in rows])
        kv = [tuple(torch.cat([part[j]._base_kv[b][i][jj][:, :, :start] for j, b, _, _ in rows]) for jj in (0, 1))
              for i in range(L)] if start > 0 else None
        with torch.enable_grad():
            hn = pt0._last(x, B, cache=(start, kv) if kv is not None else None, add=add, neu=neu, head=head)
            c, r0 = [], 0
            for pt in part:
                n_r = len(_rows([pt]))
                c.append(pt._contrast(hn[r0:r0 + n_r]))
                r0 += n_r
            torch.cat(c).sum().backward()
        r0 = 0
        for j, pt in enumerate(part):
            n_r = len(_rows([pt]))
            mine = rows[r0:r0 + n_r]
            row = {(b, n, m): r0 + k for k, (_, b, n, m) in enumerate(mine)}
            r0 += n_r
            sc = cf.Patcher.scores(pt) if relp else None
            out = {}
            for n, spec in cuts[j].items():
                P = list(spec["positions"])
                ps = torch.tensor([p - start for p in P], device=dev)
                key, kk = ("head", "head") if "head" in spec["kinds"] else ("neu", "neuron")
                hc = torch.stack([pt.act["clean"][key][i][P] for i in range(L)]).to(dev).float()       # [L, P, width]
                ho = torch.stack([pt.act["corrupt"][key][i][P] for i in range(L)]).to(dev).float()
                dl = hc - ho

                def slope(r):                       # (h - o) . dc/dh at row r, per unit, over the gap [L, P, W]
                    g = torch.stack([add[key][i].grad[r].to(dev)[ps] for i in range(L)]).float()
                    s = dl * g
                    return (s.view(L, len(P), H, hd).sum(-1) if key == "head" else s) / pt.gap
                f = {}
                for m in POINTS:
                    tag = {0.0: "0", 0.5: "5", 1.0: "1"}[m]
                    f["lost" + tag] = slope(row[("clean", None if m == 0 else n, m)])
                    f["rest" + tag] = slope(row[("corrupt", None if m == 0 else n, m)])
                if key == "head":
                    f["delta"] = dl.view(L, len(P), H, hd).norm(dim=-1)
                    f["clean"] = hc.view(L, len(P), H, hd).norm(dim=-1)
                    f["corrupt"] = ho.view(L, len(P), H, hd).norm(dim=-1)
                else:
                    f["delta"], f["clean"], f["corrupt"] = dl.abs(), hc, ho
                W = f["delta"].shape[-1]
                f["relp"] = sc[kk][:, P].to(dev).float() if sc is not None else torch.zeros_like(f["delta"])
                f["layer"] = torch.arange(L, device=dev, dtype=torch.float32)[:, None, None].expand(L, len(P), W) / (L - 1)
                pp = torch.tensor(P, device=dev, dtype=torch.float32)[None, :, None].expand(L, len(P), W)
                f["from_end"] = (T - 1) - pp
                f["from_first"] = pp - pt.first
                out[n] = f
            consume(at[id(pt)], out)
        del add, hn


def ig(f: dict) -> torch.Tensor:
    """The untrained score: the trapezoid mean over POINTS of lost@m + rest@m."""
    w = np.diff(POINTS)
    s = 0.0
    for a, b, wk in zip(POINTS[:-1], POINTS[1:], w):
        ta, tb = ({0.0: "0", 0.5: "5", 1.0: "1"}[z] for z in (a, b))
        s = s + wk / 2 * (f["lost" + ta] + f["rest" + ta] + f["lost" + tb] + f["rest" + tb])
    return s / (POINTS[-1] - POINTS[0])


def _circuit(rec: dict) -> dict:
    """{cut name: set of (layer, pos, index)} of a record's circuit (nodes by stage, routes for movers)."""
    out = {"readout": set(), "source": set(), "movers": set()}
    for x in rec.get("nodes", []):
        out[x["stage"]].add((x["layer"], x["pos"], x["neuron"]))
    for x in rec.get("routes", []):
        out["movers"].add((x["layer"], x["pos"], x["head"]))
    return out


def candidates(pt, name: str, spec: dict, f: dict, theta: dict | None, circ: set, rng) -> dict:
    """data: a cut's candidate units and their features ({"idx" [N, 3] (layer, position, index), "x" [N, len(FEATURES)],
    "theta" [N] (nan without a mask), "circuit" [N], "total": every unit of the cut})."""
    L, P = f["delta"].shape[:2]
    W = f["delta"].shape[-1]
    tot = L * P * W
    s = ig(f).reshape(-1)
    th = None
    if theta is not None:
        k = "head" if "head" in spec["kinds"] else "neuron"
        th = theta[k].reshape(-1).float()
    if "head" in spec["kinds"] or tot <= 4 * CAND + RAND:
        pick = torch.arange(tot)
    else:
        sets = [s.topk(CAND).indices.cpu(), (-s).topk(CAND).indices.cpu(), s.abs().topk(CAND).indices.cpu(),
                f["relp"].reshape(-1).topk(CAND).indices.cpu(), torch.from_numpy(rng.choice(tot, RAND, replace=False))]
        if th is not None:
            sets.append(th.topk(CAND).indices)
        pos = list(spec["positions"])
        cidx = [(i * P + pos.index(p)) * W + k for i, p, k in circ if p in pos]
        sets.append(torch.tensor(cidx, dtype=torch.long))
        pick = torch.unique(torch.cat(sets))
    pd = pick.to(s.device)
    x = torch.stack([f[k].reshape(-1)[pd] for k in FEATURES], 1).cpu().numpy().astype(np.float32)
    i, rem = np.divmod(pick.numpy(), P * W)
    pi, kk = np.divmod(rem, W)
    pos = np.asarray(spec["positions"])[pi]
    idx = np.stack([i, pos, kk], 1).astype(np.int32)
    inc = np.array([(int(a), int(b), int(c)) in circ for a, b, c in idx], dtype=bool)
    return {"idx": idx, "x": x, "theta": th[pick].numpy() if th is not None else np.full(len(pick), np.nan, np.float32),
            "circuit": inc, "total": tot, "random": RAND if len(pick) < tot else tot}


def data(pts: list, items: list, steps: int, log=None, rng=None) -> list:
    """circuit_masks.find_batch's mask arm on items with each masked cut's final theta captured, then path_features:
    per item (record, {cut name: candidates()} or None for an unusable pair)."""
    rng = rng or np.random.default_rng(0)
    theta, orig = {}, cm.optimize_many

    def capture(jobs, steps=cm.STEPS, log=None, init=None):
        res = orig(jobs, steps=steps, log=log, init=init)
        for (pt, specs, _), r in zip(jobs, res):
            p0 = pt[0] if isinstance(pt, (list, tuple)) else pt
            for s, got in zip(specs, r):
                theta[(id(p0), cut_name(p0, s))] = got["theta"]
        return res
    cm.optimize_many = capture
    try:
        t0 = time.perf_counter()
        recs = cm.find_batch(pts, items, arm="mask", steps=steps, log=log)
        t_mask = time.perf_counter() - t0
    finally:
        cm.optimize_many = orig
    live = [j for j, r in enumerate(recs) if r.get("usable", r.get("valid")) and not r.get("skipped")]
    out = [None] * len(items)
    t1 = time.perf_counter()

    def consume(k, feats):
        j = live[k]
        pt, circ = pts[j], _circuit(recs[j])
        out[j] = {n: candidates(pt, n, cm.cuts_of(pt)[n], f, theta.get((id(pt), n)), circ[n], rng) for n, f in feats.items()}
    if live:
        by_T = {}
        for k, j in enumerate(live):
            by_T.setdefault(pts[j].T, []).append(k)
        for T, ks in by_T.items():
            sub = [pts[live[k]] for k in ks]
            path_features(sub, lambda kk, feats, ks=ks: consume(ks[kk], feats))
    t_feat = time.perf_counter() - t1
    for r in recs:
        r["amortize"] = {"seconds_mask_batch": round(t_mask, 2), "seconds_features_batch": round(t_feat, 2),
                         "batch": len(items)}
    return list(zip(recs, out))


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("data",))
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    cf._threads()
    out = Path(a.out)
    (out / "rec").mkdir(parents=True, exist_ok=True)
    (out / "feat").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.dtype == "auto":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(a.model)
        n = cfg.num_hidden_layers * (4 * cfg.hidden_size ** 2 + 3 * cfg.hidden_size * cfg.intermediate_size)
        a.dtype = "bfloat16" if n >= 4e9 else "float32"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 device_map="auto" if a.device == "auto" else None)
    model = (model if a.device == "auto" else model.to(a.device)).eval()
    model.requires_grad_(False)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    qs = [q for q in json.loads(Path(a.inp).read_text())["questions"] if q.get("split", a.split) == a.split]
    qs = qs[a.offset::a.stride][:a.limit or None]
    pre = lambda q: tok(q["prefill"], add_special_tokens=False)["input_ids"] if q.get("prefill") else []  # noqa: E731
    jobs = []
    for q in qs:
        clean = cf.chat(tok, q["user"], q.get("system")) + pre(q)
        corrupt = cf.chat(tok, q["corrupt_user"], q.get("system")) + pre(q)
        name = q["id"].replace("/", "__")
        if len(clean) == len(corrupt) and not (out / "rec" / f"{name}.json").exists():
            jobs.append((q, clean, corrupt, name))
    jobs.sort(key=lambda j: len(j[1]))              # (batches of one prompt length where they can be)
    pts = [cf.Patcher(model, tok) for _ in range(a.batch)]
    rng = np.random.default_rng(0)
    for c0 in range(0, len(jobs), a.batch):
        chunk = jobs[c0:c0 + a.batch]
        log(f"{len(chunk)} pairs: " + ", ".join(q["id"] for q, _, _, _ in chunk))
        got = data(pts[:len(chunk)], [(cl, co, None) for _, cl, co, _ in chunk], a.steps, log=log, rng=rng)
        for (q, cl, co, name), (rec, feats) in zip(chunk, got):
            rec.update(id=q["id"], ids=cl, corrupted=co, answer_text=tok.decode([rec["answer"]]),
                       corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, dtype=a.dtype,
                       finder="circuit_amortize data (circuit_masks mask arm)", finder_rev=a.rev, steps=a.steps,
                       **{k: q[k] for k in ("user", "corrupt_user", "task", "key", "corrupt_key", "paraphrase") if k in q})
            if feats:
                np.savez_compressed(out / "feat" / f"{name}.npz", features=np.array(FEATURES),
                                    **{f"{n}_{k}": v for n, d in feats.items() for k, v in d.items()})
            (out / "rec" / f"{name}.json").write_text(json.dumps(rec, default=float))
        log(f"  batch done: mask {got[0][0]['amortize']['seconds_mask_batch']} s, features "
            f"{got[0][0]['amortize']['seconds_features_batch']} s")


if __name__ == "__main__":
    main()
