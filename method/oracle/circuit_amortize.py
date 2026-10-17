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
  circuit_amortize.py train DATA_OUT[,DATA_OUT...] MODEL.pt [--epochs 40]
      per cut an MLP ranking the cut's units (listwise: the masks' circuit units first among every unit), trained on the
      data dirs; the offline ranking of ig, grad, relp and the model on unseen keys written beside (MODEL.json)
  circuit_amortize.py eval QUESTIONS.json OUT --split dev --arm mask:30,relp,grad,model --scorer MODEL.pt [--batch 16]
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


def _late(pt, n) -> bool:
    """A cut at the answer position only (readout, movers): its rows at m > 0 differ from the plain runs there alone."""
    return n is not None and list(cm.cuts_of(pt)[n]["positions"]) == [pt.T - 1]


def _chunks(pts: list) -> list:
    """pts in consecutive parts of at most GRAD_TOKENS rows x positions (one forward and backward each)."""
    parts, cur, n = [], [], 0
    for pt in pts:
        t = sum(1 if _late(pt, c) else pt.T - pt.first for _, _, c, _ in _rows([pt]))
        if cur and n + t > GRAD_TOKENS:
            parts.append(cur)
            cur, n = [], 0
        cur.append(pt)
        n += t
    return parts + ([cur] if cur else [])


def _forward(part: list, rows: list, start: int):
    """One forward of rows [(pair, base, cut, m)] of the pairs part over the positions from start on (the earlier
    positions' keys and values the setup runs'), each row's cut moved the share m toward the other run, a zero added
    at every head output and neuron (gradients): (the added tensors {"neu", "head"} per layer [B, T - start, width], the
    rows' contrasts [B])."""
    import itertools
    pt0 = part[0]
    L, T, D, H, hd, dev = pt0.L, pt0.T, pt0.D, pt0.H, pt0.hd, pt0.dev
    S, B = T - start, len(rows)
    other = {"clean": "corrupt", "corrupt": "clean"}
    wn, wh = torch.zeros(B, S, 1), torch.zeros(B, S, 1)       # each row's share m at its cut's positions
    for r, (j, b, n, m) in enumerate(rows):
        if n is not None:
            spec = cm.cuts_of(part[j])[n]
            w = wh if "head" in spec["kinds"] else wn
            w[r, [p - start for p in spec["positions"]]] = m
    dt = pt0.core.model.embed_tokens.weight.dtype
    wn, wh = wn.to(dev, dt), wh.to(dev, dt)
    edit_n, edit_h = bool(wn.any()), bool(wh.any())

    def tgt(key, i, d):                             # each row's other-run values at layer i [B, S, width]
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
          for i in range(L)]
    hn = pt0._last(x, B, cache=(start, kv), add=add, neu=neu, head=head)
    c, r0 = [], 0
    for j, grp in itertools.groupby(rows, key=lambda t: t[0]):     # (each pair's rows its own answer and alternatives)
        n_r = len(list(grp))
        c.append(part[j]._contrast(hn[r0:r0 + n_r]))
        r0 += n_r
    return add, torch.cat(c)


def path_features(pts: list, consume, relp: bool = True) -> None:
    """For pairs set up in pts (Patchers of one model; prompts of one length), consume(j, {cut name: {feature: [L, P, W]
    float32 on the device}}) per pair (FEATURES, the cut's positions P, W neurons or heads). One backward per chunk of
    pairs (_chunks) through two forwards: the plain runs and the source cut's rows from the chunk's first changed
    position on, the answer-position cuts' rows (readout, movers at m > 0) at the answer position alone."""
    at = {id(pt): j for j, pt in enumerate(pts)}
    for part in _chunks(pts):
        rows = _rows(part)
        pt0 = part[0]
        L, T, H, hd, dev = pt0.L, pt0.T, pt0.H, pt0.hd, pt0.dev
        assert all(pt.T == T for pt in part), "path_features: prompts of one length"
        late = [_late(part[j], n) for j, _, n, _ in rows]
        grads = {}                                  # row -> (its forward's added tensors, its row there, that start)
        with torch.enable_grad():
            total = 0.0
            for want, start in ((False, min(pt.first for pt in part)), (True, T - 1)):
                idx = [r for r, x in enumerate(late) if x == want]
                if idx:
                    add, c = _forward(part, [rows[r] for r in idx], start)
                    total = total + c.sum()
                    grads.update({r: (add, k, start) for k, r in enumerate(idx)})
            total.backward()
        r0, scs = 0, cf.scores_many(part) if relp else [None] * len(part)
        for j, pt in enumerate(part):
            n_r = len(_rows([pt]))
            row = {(b, n, m): r0 + k for k, (_, b, n, m) in enumerate(rows[r0:r0 + n_r])}
            r0 += n_r
            sc = scs[j]
            out = {}
            for n, spec in cm.cuts_of(pt).items():
                P = list(spec["positions"])
                key, kk = ("head", "head") if "head" in spec["kinds"] else ("neu", "neuron")
                hc = torch.stack([pt.act["clean"][key][i][P] for i in range(L)]).to(dev).float()       # [L, P, width]
                ho = torch.stack([pt.act["corrupt"][key][i][P] for i in range(L)]).to(dev).float()
                dl = hc - ho

                def slope(r):                       # (h - o) . dc/dh at row r, per unit, over the gap [L, P, W]
                    add, k, s0 = grads[r]
                    ps = torch.tensor([p - s0 for p in P], device=dev)
                    g = torch.stack([add[key][i].grad[k].to(dev)[ps] for i in range(L)]).float()
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
        del grads


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

    def capture(jobs, steps=cm.STEPS, log=None, init=None, **kw):
        res = orig(jobs, steps=steps, log=log, init=init, **kw)
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
FALLBACK = "init"     # the masks themselves: "init" from INIT as circuit_masks runs them, "warm" from the prediction (warm),
WARM_STEPS = 10       # WARM_STEPS a round; "" none
WARM_MIN = -2.0       # warm: the start of units the prediction puts far below its KEEP-th
MASK_CUTS = ""        # cuts (comma-separated names) that skip the predicted order: straight to the masks warm from the
                      # prediction (WARM_STEPS a round), e.g. "readout", whose order the predictor ranks worst


def scorer(name: str):
    """A score (features {name: [L, P, W]}, cut name) -> [L, P, W], larger first: "ig"; "relp" (circuit_masks' attr
    order); "grad", the larger of |lost0| and |rest0| (attribution patching with the model's own gradient);
    "model:PATH" (load_model)."""
    if name == "ig":
        return lambda f, cut: ig(f)
    if name == "relp":
        return lambda f, cut: f["relp"]
    if name == "grad":
        return lambda f, cut: torch.maximum(f["lost0"].abs(), f["rest0"].abs())
    if name.startswith("model:"):
        return load_model(name.split(":", 1)[1])
    raise ValueError(f"scorer {name}")


def _mlp(z: dict, x: torch.Tensor) -> torch.Tensor:
    """A trained cut model on features x [N, len(FEATURES)]: asinh(x / scale), standardized, an MLP (ReLU between its
    layers), one score per unit."""
    x = torch.asinh(x / z["scale"].to(x.device))
    x = (x - z["mean"].to(x.device)) / z["std"].to(x.device)
    for i, (W, b) in enumerate(z["layers"]):
        x = x @ W.to(x.device).T + b.to(x.device)
        if i < len(z["layers"]) - 1:
            x = torch.relu(x)
    return x[:, 0]


def load_model(path: str):
    """A trained scorer (train mode's torch file: {"features": FEATURES, cut name: {"scale", "mean", "std", "layers":
    [(W, b), ...]}}): each cut's own MLP over its units' FEATURES."""
    z = torch.load(path, map_location="cpu")
    assert tuple(z["features"]) == FEATURES, "load_model: another feature set"

    def score(f: dict, cut: str) -> torch.Tensor:
        x = torch.stack([f[k] for k in FEATURES], -1)
        return _mlp(z[cut], x.reshape(-1, len(FEATURES)).float()).reshape(x.shape[:-1])
    return score


def warm(s: torch.Tensor, keep: int = KEEP) -> torch.Tensor:
    """A masks' start from a predicted score: INIT less each unit's score gap below the keep-th largest, in [WARM_MIN,
    INIT] (the top keep start patched; Adam's normalized steps let any unit grow back), on the host."""
    flat = s.reshape(-1).float()
    t = flat.topk(min(keep, flat.numel())).values[-1]
    return (cm.INIT + (s.float() - t).clamp(max=0.0)).clamp(min=WARM_MIN).cpu()


def _theta(s: torch.Tensor, keep: int) -> torch.Tensor:
    """A score as a mask theta with keep units above 0 (circuit_masks.ordered's order and count), on the host."""
    flat = s.reshape(-1).float()
    return (s.float() - flat.topk(min(keep + 1, flat.numel())).values[-1]).cpu()


def amortized_steps(pt, pred: dict, log=None, steps: int = cm.STEPS):
    """circuit_masks.search_steps with each masked cut's order from pred ({cut: (kind, score [L, P, W])}) in place of
    the masks' descent (a search for circuit_masks.drive): the cut's ceilings and bars (_before_steps), each masked
    cut's order through verify (KEEP, then WIDE units above 0), and the cuts still unmet through the masks themselves
    (FALLBACK: from INIT, or warm from the prediction for WARM_STEPS; ROUNDS as search_steps). Returns (cuts, each cut's
    stage: order, wide or masks)."""
    st = yield from cm._before_steps(pt, "mask", log=log)
    direct = [n for n in st["masked"] if n in MASK_CUTS.split(",")]
    names, got, stage = [n for n in st["masked"] if n not in direct], {}, {}
    for tag, keep in (("order", KEEP), ("wide", WIDE)):
        if not names:
            break
        for n in names:
            kind, s = pred[n]
            got[n], stage[n] = {"theta": {kind: _theta(s, keep)}, "history": []}, tag
        yield from cm._verify_steps(st, got, names, 0.0, 1, log=log)
        names = [n for n in names if not st["out"][n]["met"] and not st["out"][n].get("diffuse")]
    names = names if FALLBACK else []
    hot = set(direct) | (set(names) if FALLBACK == "warm" else set())      # (cuts whose masks start from the prediction)
    names += direct
    for rnd in range(cm.ROUNDS):
        if not names:
            break
        t0 = time.perf_counter()
        groups = [(g, WARM_STEPS if hot_ else steps) for hot_ in (True, False)
                  for g in [[n for n in names if (n in hot) == hot_]] if g]
        jobs = [(job, ns, k) for g, k in groups for job, ns in cm._jobs(pt, st, g)]
        start = lambda n: got[n]["theta"] if rnd else {pred[n][0]: warm(pred[n][1])} if n in hot else None  # noqa: E731
        res = yield [("masks", job, [start(n) for n in ns], k) for job, ns, k in jobs]
        for (_, ns, _), r in zip(jobs, res):
            for n, x in zip(ns, r):
                got[n] = {"theta": x["theta"], "history": (got[n]["history"] if rnd else []) + x["history"]}
                stage[n] = "masks"
        yield from cm._verify_steps(st, got, names, time.perf_counter() - t0, max(k for _, k in groups), log=log)
        names = [n for n in names if not st["out"][n]["met"] and not st["out"][n].get("diffuse")]
    out = yield from cm._finish_steps(st)
    src, movers = out.get("source", {}).get("units", []), out.get("movers", {}).get("units", [])
    if src and movers:                              # (each mover's share of the source cut's effect, as search_steps)
        a, b = yield [(pt, "run", "clean", [src + [u + ("clean",)] for u in movers]),
                      (pt, "run", "corrupt", [src + [u + ("corrupt",)] for u in movers])]
        out["movers"]["through"] = {u: (out["source"]["lost"] - x, out["source"]["restored"] - y) for u, x, y in zip(movers, a, b)}
    return out, stage


def find_batch_amortized(pts: list, items: list, score, arm: str = "amortized", log=None, steps: int = cm.STEPS,
                         relp: bool = True) -> list:
    """circuit_masks.find_batch with amortized_steps in place of the masks' search: every pair set up
    (circuit_finder.setup_many), the features of every usable pair in batched forwards and backwards (path_features),
    each cut's score, then the pairs of one prompt length searched in lockstep (circuit_masks.drive). Records as
    find_batch's ("search": arm), with "amortize": each cut's stage and the batch's seconds by stage."""
    t0 = time.perf_counter()
    recs, rec0s, same_T = [None] * len(items), {}, {}
    for j, rec0 in enumerate(cf.setup_many(pts, [(c, k) for c, k, _ in items], [x for _, _, x in items])):
        clean, corrupt, answers = items[j]
        if not rec0["valid"] or not np.isfinite(rec0["gap"]) or rec0["gap"] < cf.MIN_GAP or rec0["same_answer"]:
            recs[j] = {**cf._find(pts[j], clean, corrupt, answers, log=log), "search": arm}
            continue
        rec0s[j] = rec0
        same_T.setdefault(pts[j].T, []).append(j)
    sec = {"setup": time.perf_counter() - t0}
    t1 = time.perf_counter()
    pred = {j: {} for j in rec0s}
    for T, js in same_T.items():
        def consume(k, feats, js=js):
            for n, f in feats.items():
                pred[js[k]][n] = ("head" if n == "movers" else "neuron", score(f, n).detach())
        path_features([pts[j] for j in js], consume, relp=relp)
    sec["features"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    for T, js in same_T.items():
        tag = (lambda j: (lambda s: log(f"[{j}] {s.lstrip()}"))) if log and len(items) > 1 else (lambda j: log)
        res = cm.drive([pts[j] for j in js], [amortized_steps(pts[j], pred[j], log=tag(j), steps=steps) for j in js], log=log)
        for j, (cuts, stage) in zip(js, res):
            clean, corrupt, answers = items[j]
            recs[j] = cm._find_with(pts[j], clean, corrupt, answers, log, arm, cuts, rec0=rec0s[j])
            recs[j]["amortize"] = {"stage": stage}
    sec["search"] = time.perf_counter() - t1
    wall = time.perf_counter() - t0
    for r in recs:
        r.update(seconds_own=r.get("seconds"), seconds=round(wall / len(items), 2), batch=len(items))
        r.setdefault("amortize", {})["seconds_batch"] = {k: round(v, 2) for k, v in sec.items()}
    return recs


class Group_:
    """train: one (question, cut) of data mode's npz: its candidates' features x, circuit membership, theta and log weight
    (0 for a unit some score's top CAND holds or the circuit's; log of the population each random-only candidate stands
    for, the rest of the cut's units over the random-only candidates)."""

    def __init__(self, qid, task, key, cut, x, circ, theta, total):
        self.qid, self.task, self.key, self.cut, self.x, self.circ, self.theta = qid, task, key, cut, x, circ, theta
        n = len(x)
        top = np.zeros(n, bool)
        if n < total:
            sc = [ig_np(x), -ig_np(x), np.abs(ig_np(x)), x[:, FEATURES.index("relp")]]
            sc += [np.nan_to_num(theta, nan=-np.inf)] if not np.isnan(theta).all() else []
            for v in sc:
                top[np.argsort(-v, kind="stable")[:CAND]] = True
        top |= circ
        rest = n - int(top.sum())
        self.lw = np.where(top, 0.0, np.log(max(total - top.sum(), 1) / max(rest, 1))).astype(np.float32)


def ig_np(x: np.ndarray) -> np.ndarray:
    """ig() on train's feature matrices [N, len(FEATURES)]."""
    c = {k: x[:, FEATURES.index(k)] for k in FEATURES[:6]}
    return 0.25 * (c["lost0"] + c["rest0"]) + 0.5 * (c["lost5"] + c["rest5"]) + 0.25 * (c["lost1"] + c["rest1"])


def load_groups(dirs: list, log=None) -> list:
    """train: every (question, cut) with a mask theta and a circuit in data mode's OUT dirs."""
    import json
    from pathlib import Path
    out = []
    for d in dirs:
        for f in sorted(Path(d, "feat").glob("*.npz")):
            rec = json.loads(Path(d, "rec", f.stem + ".json").read_text())
            z = np.load(f)
            for cut in ("readout", "movers", "source"):
                if f"{cut}_x" not in z or np.isnan(z[f"{cut}_theta"]).all() or not z[f"{cut}_circuit"].any():
                    continue
                out.append(Group_(rec["id"], rec.get("task"), rec.get("key"), cut, z[f"{cut}_x"], z[f"{cut}_circuit"],
                                  z[f"{cut}_theta"], int(z[f"{cut}_total"])))
    if log:
        log(f"{len(out)} cuts from {len(dirs)} dirs")
    return out


def ranks(g: Group_, s: np.ndarray) -> np.ndarray:
    """Each circuit unit's estimated place among all of its cut's units under score s (0 first): the candidates above
    it, random-only ones counted by their weight."""
    w = np.exp(g.lw)
    o = np.argsort(-s, kind="stable")
    above = np.concatenate([[0.0], np.cumsum(w[o])[:-1]])
    r = np.empty(len(s))
    r[o] = above
    return r[g.circ]


def offline(groups: list, scores: dict) -> dict:
    """Per score name (fn(group) -> [N]) the ranks of the mask circuits' units: recall in the top 64, the share of cuts
    whose whole circuit lies in the top 64, the median over cuts of the last circuit unit's place."""
    out = {}
    for name, fn in scores.items():
        rs = [ranks(g, fn(g)) for g in groups]
        out[name] = {"recall64": round(float(np.mean(np.concatenate([r < 64 for r in rs]))), 3),
                     "all64": round(float(np.mean([bool((r < 64).all()) for r in rs])), 3),
                     "last_median": float(np.median([r.max() for r in rs])), "cuts": len(rs)}
    return out


def fit(groups: list, cut: str, hidden: int = 64, epochs: int = 40, lr: float = 3e-3, seed: int = 0, val=None, log=None):
    """A cut's MLP (load_model's format) trained on groups' candidates: per cut a softmax over its units (each random-only
    candidate standing for its weight), the circuit's units the targets (listwise cross-entropy)."""
    torch.manual_seed(seed)
    gs = [g for g in groups if g.cut == cut]
    allx = np.concatenate([g.x for g in gs])
    scale = np.maximum(np.median(np.abs(allx), 0), 1e-6).astype(np.float32)
    t = np.arcsinh(allx / scale)
    mean, std = t.mean(0).astype(np.float32), np.maximum(t.std(0), 1e-6).astype(np.float32)
    F = len(FEATURES)
    net = torch.nn.Sequential(torch.nn.Linear(F, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    prep = lambda g: torch.from_numpy((np.arcsinh(g.x / scale) - mean) / std).float()   # noqa: E731
    data = [(prep(g), torch.from_numpy(g.circ), torch.from_numpy(g.lw)) for g in gs]
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        tot = 0.0
        perm = rng.permutation(len(data))
        for b0 in range(0, len(data), 16):
            loss = 0.0
            for xt, ct, lw in [data[i] for i in perm[b0:b0 + 16]]:
                s = net(xt)[:, 0]
                loss = loss + (torch.logsumexp(s + lw, 0) - s[ct]).mean()
            opt.zero_grad()
            (loss / 16).backward()
            opt.step()
            tot += float(loss.detach())
        z = _export(net, scale, mean, std)
        if val is not None and (ep % 5 == 4 or ep == epochs - 1):
            vs = [g for g in val if g.cut == cut]
            m = offline(vs, {"model": lambda g: _mlp(z, torch.from_numpy(g.x)).detach().numpy()})["model"]
            hist.append((ep, round(tot / len(data), 4), m))
            if log:
                log(f"  {cut} epoch {ep}: loss {tot / len(data):.4f}, unseen {m}")
    return _export(net, scale, mean, std), hist


def _export(net, scale, mean, std) -> dict:
    lins = [m for m in net if isinstance(m, torch.nn.Linear)]
    return {"scale": torch.from_numpy(scale), "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
            "layers": [(m.weight.detach().clone(), m.bias.detach().clone()) for m in lins]}


def train(dirs: list, out_path: str, holdout: float = 0.2, epochs: int = 40, hidden: int = 64, seed: int = 0, log=print,
          by: str = "key", save: bool = True):
    """train mode: per cut an MLP on data mode's dirs. The (task, key)s (by "key") or the tasks (by "task") are split,
    holdout of them unseen: the offline ranking of ig, grad, relp and the model on them is logged (and returned); the
    saved model is then trained on every key with the same epochs."""
    groups = load_groups(dirs, log=log)
    unit = (lambda g: (g.task, g.key)) if by == "key" else (lambda g: g.task)  # noqa: E731
    keys = sorted({unit(g) for g in groups})
    rng = np.random.default_rng(seed)
    hold = {keys[i] for i in rng.permutation(len(keys))[:max(1, int(holdout * len(keys)))]}
    tr = [g for g in groups if unit(g) not in hold]
    va = [g for g in groups if unit(g) in hold]
    log(f"{len(tr)} training cuts, {len(va)} unseen cuts ({len(hold)} of {len(keys)} {by}s held out"
        + (f": {sorted(hold)}" if by == "task" else "") + ")")
    report = {}
    for cut in ("readout", "movers", "source"):
        vs = [g for g in va if g.cut == cut]
        if not vs:
            continue
        z, hist = fit(tr, cut, hidden=hidden, epochs=epochs, seed=seed, val=va, log=log)
        base = offline(vs, {"ig": lambda g: ig_np(g.x), "grad": lambda g: np.maximum(np.abs(g.x[:, 0]), np.abs(g.x[:, 3])),
                            "relp": lambda g: g.x[:, FEATURES.index("relp")],
                            "theta": lambda g: np.nan_to_num(g.theta, nan=-np.inf),
                            "model": lambda g: _mlp(z, torch.from_numpy(g.x)).detach().numpy()})
        report[cut] = base
        log(f"{cut} unseen keys: " + "; ".join(f"{k} {v}" for k, v in base.items()))
    if not save:
        return report
    model = {"features": FEATURES}
    for cut in ("readout", "movers", "source"):
        if any(g.cut == cut for g in groups):
            model[cut], _ = fit(groups, cut, hidden=hidden, epochs=epochs, seed=seed)
    torch.save(model, out_path)
    log(f"saved {out_path}")
    return report


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("data", "eval", "train"))
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--arm", default="mask:30,model", help="eval: comma-separated arms, each question through every arm "
                    "in turn on the same card, records in OUT/<arm>/: mask:STEPS (circuit_masks.find_batch), or a scorer "
                    "(ig, relp, grad, model: --scorer's file) through find_batch_amortized")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--adapter", default=None, help="a PEFT adapter merged into the model (an organism)")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scorer", default=None, help="eval: the trained scorer file of the arm \"model\" (train mode's)")
    ap.add_argument("--profile", action="store_true", help="eval: each arm batch's drive ticks timed by kind (TickProfile)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--holdout-by", default="key", choices=("key", "task"), help="train: unseen keys (and the model saved "
                    "after, trained on every key) or unseen tasks (the report only)")
    a = ap.parse_args()
    if a.mode == "train":                           # (inp: data mode's OUT dirs, comma-separated; out: the model file)
        report = train(a.inp.split(","), a.out, epochs=a.epochs, hidden=a.hidden, by=a.holdout_by,
                       save=a.holdout_by == "key")
        Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
        return
    cf._threads()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in ("rec", "feat") if a.mode == "data" else ():
        (out / d).mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.dtype == "auto":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(a.model)
        n = cfg.num_hidden_layers * (4 * cfg.hidden_size ** 2 + 3 * cfg.hidden_size * cfg.intermediate_size)
        a.dtype = "bfloat16" if n >= 4e9 else "float32"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 device_map="auto" if a.device == "auto" else None)
    model = (model if a.device == "auto" else model.to(a.device)).eval()
    if a.adapter:                                   # (an organism: its LoRA merged into the weights, circuit_masks --merge's)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval().merge_and_unload().eval()
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
        dirs = {arm: arm.replace(":", "_").replace("=", "") for arm in arms}
        for arm in arms:
            (out / dirs[arm]).mkdir(exist_ok=True)
        pts[0].setup(jobs[0][1], jobs[0][2])        # (a throwaway run: CUDA's first kernels)
        failed = set()
        for c0 in range(0, len(jobs), a.batch):
            chunk = jobs[c0:c0 + a.batch]
            for arm in arms:
                todo = [j for j in chunk if not (out / dirs[arm] / f"{j[3]}.json").exists()]
                if not todo or arm in failed:
                    continue
                log(f"{len(todo)} pairs ({arm}): " + ", ".join(q["id"] for q, _, _, _ in todo))
                its = [(cl, co, None) for _, cl, co, _ in todo]
                name, *parts = arm.split(":")       # (an arm's constants: model:keep=16:fallback=warm, mask:10:init_scan=1;
                sets = {}                           # each this module's, else circuit_masks', else circuit_finder's)
                for k, v in (x.split("=") for x in parts if "=" in x):
                    mod = next(m for m in (globals(), vars(cm), vars(cf)) if k.upper() in m)
                    sets[k.upper()] = (mod, _value(v, mod[k.upper()]))
                saved = {k: (m, m[k]) for k, (m, _) in sets.items()}
                for k, (m, v) in sets.items():
                    m[k] = v
                prof, recs = TickProfile() if a.profile else None, None
                try:
                    if prof:
                        prof.__enter__()
                    if name == "mask":
                        recs = cm.find_batch(pts[:len(todo)], its, arm="mask", log=log,
                                             steps=next((int(x) for x in parts if x.isdigit()), cm.STEPS))
                    else:
                        nm = f"model:{a.scorer}" if name == "model" else name
                        recs = find_batch_amortized(pts[:len(todo)], its, scorer(nm), arm=arm, log=log,
                                                    relp=nm == "relp" or nm.startswith("model:"))
                except Exception:                   # noqa: BLE001 (an arm that fails is dropped, the others go on)
                    import traceback
                    log(f"  {arm} FAILED, dropped:\n{traceback.format_exc()}")
                    failed.add(arm)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                finally:
                    if prof:
                        prof.__exit__()
                        log(f"  {arm} ticks: {prof.report()}")
                        for r in recs or []:
                            r["ticks_batch"] = prof.report()
                    for k, (m, v) in saved.items():
                        m[k] = v
                for (q, cl, co, name), rec in zip(todo, recs):
                    rec.update(id=q["id"], ids=cl, corrupted=co, answer_text=tok.decode([rec["answer"]]),
                               corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, dtype=a.dtype,
                               finder="circuit_amortize", finder_rev=a.rev, adapter=a.adapter, merged=bool(a.adapter),
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


class TickProfile:
    """eval --profile: circuit_masks._serve wrapped, each tick's wall time (the device synchronized) summed by what it
    served (measurement methods, masks, scores) and counted; report() per arm batch."""

    def __init__(self):
        self.orig, self.t, self.n = cm._serve, {}, {}

    def __enter__(self):
        def serve(group, reqs, log=None):
            flat = []

            def walk(r):
                if isinstance(r, list):
                    for x in r:
                        walk(x)
                else:
                    flat.append(r[0] if isinstance(r[0], str) else r[1])
            for r in reqs:
                walk(r)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self.orig(group, reqs, log)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            key = "+".join(sorted(set(flat)))
            self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0
            self.n[key] = self.n.get(key, 0) + 1
            return out
        cm._serve = serve
        return self

    def __exit__(self, *a):
        cm._serve = self.orig

    def report(self) -> dict:
        return {k: {"ticks": self.n[k], "seconds": round(v, 2)} for k, v in sorted(self.t.items(), key=lambda t: -t[1])}


def _value(v: str, like):
    """An arm's constant from its text, of its default's type (init=1 is INIT 1.0: an int start made masks of
    integers, which cannot take gradients; pod dev3)."""
    if isinstance(like, bool):
        return v.lower() in ("1", "true", "yes")
    return type(like)(v) if isinstance(like, (int, float)) else v


if __name__ == "__main__":
    main()
