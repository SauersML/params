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
  circuit_amortize.py eval QUESTIONS.json OUT --split dev --arm mask:30,ig,relp,grad,model=PATH [--batch 16]
      every question through every arm in turn on one card (OUT/<arm>/<id>.json): mask:STEPS circuit_masks.find_batch;
      a scorer through find_batch_amortized (each masked cut's order from the scorer, the masks' verify, the masks
      themselves only for cuts still unmet)
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


KEEP = 32             # a predicted score as a mask theta with KEEP units above 0 (circuit_masks.ordered then hands
WIDE = 256            # verify its 2 x KEEP best units); a cut whose order cannot pass: once more with WIDE above 0, then
FALLBACK = True       # (FALLBACK) the masks themselves, from INIT, as circuit_masks.find_batch runs them


def scorer(name: str):
    """A score {feature: [L, P, W]} -> [L, P, W], larger first: "ig"; "relp" (circuit_masks' attr order); "grad", the
    larger of |lost0| and |rest0| (attribution patching with the model's own gradient); "model:PATH" (load_model)."""
    if name == "ig":
        return ig
    if name == "relp":
        return lambda f: f["relp"]
    if name == "grad":
        return lambda f: torch.maximum(f["lost0"].abs(), f["rest0"].abs())
    if name.startswith("model:"):
        return load_model(name.split(":", 1)[1])
    raise ValueError(f"scorer {name}")


def load_model(path: str):
    """A trained per-unit scorer (amortize_train.py's torch file: {"features", "mean", "std", "layers": [(W, b)...]},
    an MLP with ReLU between its layers over the standardized FEATURES, one output: the unit's score)."""
    z = torch.load(path, map_location="cpu")
    assert tuple(z["features"]) == FEATURES, "load_model: another feature set"

    def score(f: dict) -> torch.Tensor:
        x0 = f["lost0"]
        x = torch.stack([f[k] for k in FEATURES], -1).reshape(-1, len(FEATURES))
        x = (x - z["mean"].to(x.device)) / z["std"].to(x.device)
        for i, (W, b) in enumerate(z["layers"]):
            x = x @ W.to(x.device).T + b.to(x.device)
            if i < len(z["layers"]) - 1:
                x = torch.relu(x)
        return x.reshape(x0.shape)
    return score


def _theta(s: torch.Tensor, keep: int) -> torch.Tensor:
    """A score as a mask theta with keep units above 0 (circuit_masks.ordered's order and count), on the host."""
    flat = s.reshape(-1).float()
    return (s.float() - flat.topk(min(keep + 1, flat.numel())).values[-1]).cpu()


def find_batch_amortized(pts: list, items: list, score, arm: str = "amortized", log=None, steps: int = cm.STEPS,
                         relp: bool = True) -> list:
    """circuit_masks.find_batch with each masked cut's order from score (path_features of the pair) in place of the
    masks' descent: every pair set up and its ceilings measured (circuit_masks._before), the features of every pair in
    batched forwards and backwards, each masked cut's order through circuit_masks' verify (KEEP, then WIDE), and the
    cuts still unmet through the masks (FALLBACK, find_batch's rounds). Records as find_batch's ("search": arm), with
    "amortize": each cut's round ("order", "wide" or "masks") and the batch's seconds by stage."""
    t0 = time.perf_counter()
    recs, sts, rec0s, same_T = [None] * len(items), [None] * len(items), {}, {}
    for j, (pt, (clean, corrupt, answers)) in enumerate(zip(pts, items)):
        rec0 = pt.setup(clean, corrupt, answers)
        if not rec0["valid"] or not np.isfinite(rec0["gap"]) or rec0["gap"] < cf.MIN_GAP or rec0["same_answer"]:
            recs[j] = {**cf.find(pt, clean, corrupt, answers, log=log), "search": arm}
            continue
        rec0s[j] = rec0
        same_T.setdefault(pt.T, []).append(j)
    sec = {"setup": time.perf_counter() - t0}
    t1 = time.perf_counter()
    for T, js in same_T.items():
        befores = cf.Group([pts[j] for j in js]).run([lambda j=j: cm._before(pts[j], "mask", log=log) for j in js])
        for j, b in zip(js, befores):
            sts[j] = (rec0s[j], b)
    sec["ceilings"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    pred = {}
    for T, js in same_T.items():
        def consume(k, feats, js=js):
            j = js[k]
            for n in sts[j][1]["masked"]:
                kind = "head" if "head" in sts[j][1]["specs"][n]["kinds"] else "neuron"
                pred[(j, n)] = (kind, score(feats[n]).detach())
        if any(sts[j][1]["masked"] for j in js):
            path_features([pts[j] for j in js], consume, relp=relp)
    sec["features"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    got = {j: {} for j in range(len(items))}
    names = {j: list(x[1]["masked"]) for j, x in enumerate(sts) if x is not None}
    stage = {j: {} for j in names}
    for tag, keep in (("order", KEEP), ("wide", WIDE)):
        by_T = {}
        for j, ns in names.items():
            for n in ns:
                kind, s = pred[(j, n)]
                got[j][n] = {"theta": {kind: _theta(s, keep)}, "history": []}
                stage[j][n] = tag
            if ns:
                by_T.setdefault(pts[j].T, []).append(j)
        for T, js in by_T.items():
            cf.Group([pts[j] for j in js]).run(
                [lambda j=j: cm._verify(pts[j], sts[j][1], got[j], names[j], 0.0, 1, log=log) for j in js])
        names = {j: [n for n in ns if not sts[j][1]["out"][n]["met"]] for j, ns in names.items()}
    sec["verify"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    for rnd in range(cm.ROUNDS if FALLBACK else 0):     # (find_batch's rounds: the masks from INIT, then on from theta)
        by_T = {}
        for j, ns in names.items():
            if ns:
                by_T.setdefault(pts[j].T, []).append(j)
        if not by_T:
            break
        mg = {j: {} for j in names}
        for T, js in by_T.items():
            jobs = [(j, job, ns) for j in js for job, ns in cm._jobs(pts[j], sts[j][1], names[j])]
            for part in cm._chunks(jobs, T):
                res = cm.optimize_many([job for _, job, _ in part], steps=steps, log=log,
                                       init=[[mg[j].get(n, {}).get("theta") for n in ns] for j, _, ns in part] if rnd else None)
                for (j, _, ns), r in zip(part, res):
                    for n, x in zip(ns, r):
                        mg[j][n] = {"theta": x["theta"], "history": mg[j].get(n, {}).get("history", []) + x["history"]}
                        stage[j][n] = "masks"
        for T, js in by_T.items():
            cf.Group([pts[j] for j in js]).run(
                [lambda j=j: cm._verify(pts[j], sts[j][1], mg[j], names[j], 0.0, steps, log=log) for j in js])
        names = {j: [n for n in ns if not sts[j][1]["out"][n]["met"]] for j, ns in names.items()}
    sec["masks"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    for j, x in enumerate(sts):
        if x is None:
            continue
        clean, corrupt, answers = items[j]
        recs[j] = cm._find_with(pts[j], clean, corrupt, answers, log, arm, cm._finish(x[1]), rec0=x[0])
        recs[j]["amortize"] = {"stage": stage[j]}
    sec["finish"] = time.perf_counter() - t1
    wall = time.perf_counter() - t0
    for r in recs:
        r.update(seconds_own=r.get("seconds"), seconds=round(wall / len(items), 2), batch=len(items))
        r.setdefault("amortize", {})["seconds_batch"] = {k: round(v, 2) for k, v in sec.items()}
    return recs


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("data", "eval"))
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--arm", default="ig,mask:30", help="eval: comma-separated arms, each question through every arm in "
                    "turn on the same card, records in OUT/<arm>/: mask:STEPS (circuit_masks.find_batch), or a scorer "
                    "(ig, relp, grad, model=PATH) through find_batch_amortized")
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
    for d in ("rec", "feat") if a.mode == "data" else ():
        (out / d).mkdir(parents=True, exist_ok=True)
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
    if a.mode == "eval":
        arms = a.arm.split(",")
        dirs = {arm: arm.replace(":", "_").replace("=", "_").replace("/", "_") for arm in arms}
        for arm in arms:
            (out / dirs[arm]).mkdir(exist_ok=True)
        pts[0].setup(jobs[0][1], jobs[0][2])        # (a throwaway run: CUDA's first kernels)
        for c0 in range(0, len(jobs), a.batch):
            chunk = jobs[c0:c0 + a.batch]
            for arm in arms:
                todo = [j for j in chunk if not (out / dirs[arm] / f"{j[3]}.json").exists()]
                if not todo:
                    continue
                log(f"{len(todo)} pairs ({arm}): " + ", ".join(q["id"] for q, _, _, _ in todo))
                its = [(cl, co, None) for _, cl, co, _ in todo]
                if arm.startswith("mask"):
                    recs = cm.find_batch(pts[:len(todo)], its, arm="mask", log=log,
                                         steps=int(arm.split(":")[1]) if ":" in arm else cm.STEPS)
                else:
                    nm = arm.replace("model=", "model:")
                    recs = find_batch_amortized(pts[:len(todo)], its, scorer(nm), arm=arm, log=log,
                                                relp=nm in ("relp",) or nm.startswith("model:"))
                for (q, cl, co, name), rec in zip(todo, recs):
                    rec.update(id=q["id"], ids=cl, corrupted=co, answer_text=tok.decode([rec["answer"]]),
                               corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, dtype=a.dtype,
                               finder="circuit_amortize", finder_rev=a.rev,
                               **{k: q[k] for k in ("user", "corrupt_user", "task", "key", "corrupt_key", "paraphrase")
                                  if k in q})
                    (out / dirs[arm] / f"{name}.json").write_text(json.dumps(rec, default=float))
                if "amortize" in recs[0]:
                    log(f"  {arm}: {recs[0]['seconds']} s a pair, stages {recs[0]['amortize'].get('seconds_batch')}")
        return
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
