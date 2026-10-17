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


SOURCE_PATH = True    # the source cut's rows at m > 0 (its lost5, lost1, rest5, rest1); False: those features 0 and the
                      # rows not run (a pair's source rows at m > 0 are ~2/3 of path_features' tokens; a model trained so)


def _rows(pts: list) -> list:
    """The rows of path_features' forward: per pair its two plain runs (m = 0, every cut's), then per cut and m > 0 the
    clean run with the cut at m (noising) and the corrupted run with the cut at m (denoising): (pair, base, cut, m)."""
    rows = []
    for j, pt in enumerate(pts):
        rows += [(j, "clean", None, 0.0), (j, "corrupt", None, 0.0)]
        rows += [(j, b, n, m) for n in cm.cuts_of(pt) if SOURCE_PATH or n != "source" for m in POINTS if m > 0
                 for b in ("clean", "corrupt")]
    return rows


SECONDS = {}          # path_features' and the scorer's seconds by part (the device synchronized at each mark), summed


def _mark(name: str, t0: float) -> float:
    """Adds the time since t0 (the device synchronized) to SECONDS[name]; the time now."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    SECONDS[name] = SECONDS.get(name, 0.0) + t - t0
    return t


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
        tm = _mark("features_other", time.perf_counter())
        late = [_late(part[j], n) for j, _, n, _ in rows]
        grads = {}                                  # row -> (its forward's gradients {key: [L, rows, S, width]}, its row
        with torch.enable_grad():                   # there, that forward's start)
            total, adds = 0.0, []
            for want, start in ((False, min(pt.first for pt in part)), (True, T - 1)):
                idx = [r for r, x in enumerate(late) if x == want]
                if idx:
                    add, c = _forward(part, [rows[r] for r in idx], start)
                    total = total + c.sum()
                    adds.append((add, idx, start))
            tm = _mark("features_forward", tm)
            total.backward()
            tm = _mark("features_backward", tm)
        for add, idx, start in adds:                # (every layer's gradients in one tensor per kind: one gather a slope)
            G = {key: torch.stack([t.grad.to(dev) for t in add[key]]) for key in ("neu", "head")}
            grads.update({r: (G, k, start) for k, r in enumerate(idx)})
        del adds, add
        tm = _mark("features_gather", tm)
        r0, scs = 0, cf.scores_many(part) if relp else [None] * len(part)
        tm = _mark("features_relp", tm)
        for j, pt in enumerate(part):
            n_r = len(_rows([pt]))
            row = {(b, n, m): r0 + k for k, (_, b, n, m) in enumerate(rows[r0:r0 + n_r])}
            r0 += n_r
            sc = scs[j]
            out = {}
            acts = {(run, key): torch.stack([t.to(dev) for t in pt.act[run][key]]) for run in ("clean", "corrupt")
                    for key in ("neu", "head")}     # [L, T, width] per run and kind
            for n, spec in cm.cuts_of(pt).items():
                P = list(spec["positions"])
                key, kk = ("head", "head") if "head" in spec["kinds"] else ("neu", "neuron")
                hc, ho = acts[("clean", key)][:, P].float(), acts[("corrupt", key)][:, P].float()      # [L, P, width]
                dl = hc - ho

                def slope(r):                       # (h - o) . dc/dh at row r, per unit, over the gap [L, P, W]
                    G, k, s0 = grads[r]
                    s = dl * G[key][:, k, [p - s0 for p in P]].float()
                    return (s.view(L, len(P), H, hd).sum(-1) if key == "head" else s) / pt.gap
                f = {}
                for m in POINTS:
                    tag = {0.0: "0", 0.5: "5", 1.0: "1"}[m]
                    for base, feat in (("clean", "lost"), ("corrupt", "rest")):
                        r = row.get((base, None if m == 0 else n, m))
                        f[feat + tag] = slope(r) if r is not None else torch.zeros_like(dl if key == "neu" else dl[..., ::hd])
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
            tm = _mark("features_assemble", tm)
            consume(at[id(pt)], out)
            tm = _mark("features_score", tm)
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


def _usable(pt) -> bool:
    """A pair set up in pt counts (circuit_masks.find_batch's test of a partner pair, from the Patcher's own state)."""
    yc = int(pt._lp["corrupt"].argmax())
    return bool(pt.sets.get("valid", True)) and np.isfinite(pt.gap) and pt.gap >= cf.MIN_GAP and yc not in pt.sets["answer"]


def data(pts: list, items: list, steps: int, log=None, rng=None, partners: list | None = None) -> list:
    """circuit_masks.find_batch's mask arm on items with each masked cut's final theta captured, then path_features:
    per item (record, {cut name: candidates()} or None for an unusable pair). partners: as find_batch's; a multi-partner
    cut's candidates from its pairs' mean features (_mean_features) at its positions, its theta the shared mask's."""
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
        recs = cm.find_batch(pts, items, arm="mask", steps=steps, log=log, partners=partners)
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
    part = {j: [p for p, ids in (partners or [[]] * len(items))[j] if len(ids) == len(items[j][0]) and _usable(p)]
            for j in live}
    alone = [k for k, j in enumerate(live) if not part[j]]
    by_T = {}
    for k in alone:
        by_T.setdefault(pts[live[k]].T, []).append(k)
    for T, ks in by_T.items():
        sub = [pts[live[k]] for k in ks]
        path_features(sub, lambda kk, feats, ks=ks: consume(ks[kk], feats))
    for j in [j for j in live if part[j]]:          # (a question with partners: its pairs together, as
        pt, ps, got, circ = pts[j], [pts[j]] + part[j], {}, _circuit(recs[j])         # find_batch_amortized)
        path_features(ps, lambda k, feats: got.__setitem__(k, feats))
        out[j] = {}
        for n in got[0]:
            spec, f = cm.cuts_of(pt)[n], got[0][n]
            if n == "source" or cm.ALLCUTS:
                pos = cm._source_positions(pt, min(p.first for p in ps)) if n == "source" else list(spec["positions"])
                spec, f = {**spec, "positions": pos}, _mean_features(n, ps, [got[k][n] for k in range(len(ps))], pos)
            out[j][n] = candidates(pt, n, spec, f, theta.get((id(pt), n)), circ[n], rng)
        del got
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
PARTNERS = 1          # eval: pairs a multi-partner cut is chosen over (the question's own and PARTNERS - 1 other
                      # corrupted keys of its length: its partners, or --pool's keys of its task), circuit_masks' --partners
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


PRE = 16384           # a trained scorer ranks only a cut's PRE units of largest |slope| (any of the six), the rest below
                      # them all: on 1,237 trainset_8b questions that set held 99.97% of the model's top 64 source units
                      # and 100% of its readout's (prefilter on every third question), and scoring every unit of a
                      # source cut (6.6 million) was 1.2-1.5 s of a 16-pair batch's 4 s of features (pod prof2)


def load_model(path: str):
    """A trained scorer (train mode's torch file: {"features": FEATURES, cut name: {"scale", "mean", "std", "layers":
    [(W, b), ...]}}): each cut's own MLP over its units' FEATURES, on the cut's PRE units of largest |slope|."""
    z = torch.load(path, map_location="cpu")
    assert tuple(z["features"]) == FEATURES, "load_model: another feature set"

    def score(f: dict, cut: str) -> torch.Tensor:   # (score.relp: whether the model reads RelP, else path_features skips it)
        shape, n = f["lost0"].shape, f["lost0"].numel()
        if n <= PRE:
            return _mlp(z[cut], torch.stack([f[k].reshape(-1) for k in FEATURES], -1).float()).reshape(shape)
        v = torch.stack([f[k].reshape(-1).abs() for k in FEATURES[:6]]).amax(0)
        idx = v.topk(PRE).indices
        sub = _mlp(z[cut], torch.stack([f[k].reshape(-1)[idx] for k in FEATURES], -1).float())
        out = torch.full((n,), float(sub.min()) - 1.0, device=sub.device)
        out[idx] = sub
        return out.reshape(shape)
    score.relp, score.source_path = z.get("relp", True), z.get("source_path", True)
    return score


def warm(s: torch.Tensor, keep: int = KEEP) -> torch.Tensor:
    """A masks' start from a predicted score: INIT less each unit's score gap below the keep-th largest, in [WARM_MIN,
    INIT] (the top keep start patched; Adam's normalized steps let any unit grow back), on the host."""
    flat = s.reshape(-1).float()
    t = flat.topk(min(keep, flat.numel())).values[-1]
    return (cm.INIT + (s.float() - t).clamp(max=0.0)).clamp(min=WARM_MIN).cpu()


def _theta(s: torch.Tensor, keep: int) -> torch.Tensor:
    """A score as a mask theta with keep units above 0 (circuit_masks.ordered's order and count), where s is
    (ordered takes its top units on the pair's card)."""
    flat = s.reshape(-1).float()
    return s.float() - flat.topk(min(keep + 1, flat.numel())).values[-1]


def amortized_steps(pt, pred: dict, log=None, steps: int = cm.STEPS, partners: list | None = None, wanted=None,
                    mask_cuts: str | None = None):
    """circuit_masks.search_steps with each masked cut's order from pred ({cut: (kind, score [L, P, W])}) in place of
    the masks' descent (a search for circuit_masks.drive): the cut's ceilings and bars (_before_steps), each masked
    cut's order through verify (KEEP, then WIDE units above 0), and the cuts still unmet through the masks themselves
    (FALLBACK: from INIT, or warm from the prediction for WARM_STEPS; ROUNDS as search_steps). Returns (cuts, each cut's
    stage: order, wide or masks). partners: Patchers set up on pt's clean prompt against other corrupted prompts
    (circuit_masks' multi-partner cuts: the source cut chosen and verified on the shares' means; pred's score of such a
    cut on its positions over every pair). wanted: those cuts only; mask_cuts: MASK_CUTS for this search."""
    st = yield from cm._before_steps(pt, "mask", log=log, partners=partners, wanted=wanted)
    direct = [n for n in st["masked"] if n in (MASK_CUTS if mask_cuts is None else mask_cuts).split(",")]
    names, got, stage = [n for n in st["masked"] if n not in direct], {}, {}
    for tag, keep in (("order", KEEP), ("wide", WIDE)):
        if not names or keep <= 0:                  # (WIDE 0: a cut its order cannot pass goes to the masks)
            continue
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


def _mean_features(cut: str, pairs: list, feats: list, positions: list) -> dict:
    """A Mean cut's features (circuit_masks' multi-partner cuts: the shares' means over its pairs) at the cut's
    positions, from each pair's own (pairs[k]'s feats[k] on its own cut positions): the slopes, |delta| and RelP the
    means over the pairs (a share's slope in a unit's mask is linear in the share, so the mean share's slope is the
    mean of the pairs' slopes; at a position before a pair's first change its patch moves nothing: 0 there), the
    corrupted value the mean (the clean one there), the clean value, layer and position from the answer the pairs'
    (one clean prompt), the position counted from the cut's first."""
    zero = ("lost0", "lost5", "lost1", "rest0", "rest5", "rest1", "delta", "relp")
    L, W = feats[0]["lost0"].shape[0], feats[0]["lost0"].shape[2]
    dev = feats[0]["lost0"].device
    out = {k: torch.zeros(L, len(positions), W, device=dev) for k in zero + ("corrupt",)}
    clean = torch.zeros(L, len(positions), W, device=dev)
    for p, f in zip(pairs, feats):
        own = list(cm.cuts_of(p)[cut]["positions"])
        at = [positions.index(q) for q in own if q in positions]
        src = [i for i, q in enumerate(own) if q in positions]
        for k in zero:
            out[k][:, at] += f[k][:, src]
        clean[:, at] = f["clean"][:, src]
        cor = torch.zeros(L, len(positions), W, device=dev)
        cor[:, at] = f["corrupt"][:, src] - f["clean"][:, src]          # (the corrupted value less the clean one: 0
        out["corrupt"] += cor                                           # where the pair changed nothing)
    n = len(pairs)
    out = {k: v / n for k, v in out.items()}
    out["corrupt"] = clean + out["corrupt"]
    out["clean"] = clean
    pp = torch.tensor(positions, device=dev, dtype=torch.float32)[None, :, None].expand(L, len(positions), W)
    out["layer"] = torch.arange(L, device=dev, dtype=torch.float32)[:, None, None].expand(L, len(positions), W) / (L - 1)
    out["from_end"] = (pairs[0].T - 1) - pp
    out["from_first"] = pp - min(positions)
    return out


def find_batch_amortized(pts: list, items: list, score, arm: str = "amortized", log=None, steps: int = cm.STEPS,
                         relp: bool = True, partners: list | None = None) -> list:
    """circuit_masks.find_batch with amortized_steps in place of the masks' search: every pair set up
    (circuit_finder.setup_many, partner pairs too), the features of every usable pair in batched forwards and backwards
    (path_features), each cut's score (a multi-partner cut's the mean over its pairs, _combine), then the pairs of one
    prompt length searched in lockstep (circuit_masks.drive). partners: per item [(Patcher, corrupted ids)], other
    corrupted prompts of its clean prompt (circuit_masks.find_batch's). Records as find_batch's ("search": arm), with
    "amortize": each cut's stage and the batch's seconds by stage."""
    t0 = time.perf_counter()
    recs, rec0s, same_T, part = [None] * len(items), {}, {}, {}
    cand = [[(ppt, ids) for ppt, ids in (partners or [[]] * len(items))[j] if len(ids) == len(items[j][0])]
            for j in range(len(items))]
    flat = [(pt, (c, k), x) for pt, (c, k, x) in zip(pts, items)] + [
        (ppt, (items[j][0], ids), items[j][2]) for j in range(len(items)) for ppt, ids in cand[j]]
    got = iter(cf.setup_many([x[0] for x in flat], [x[1] for x in flat], [x[2] for x in flat]))
    for j, (clean, corrupt, answers) in enumerate(items):
        rec0 = next(got)
        if not rec0["valid"] or not np.isfinite(rec0["gap"]) or rec0["gap"] < cf.MIN_GAP or rec0["same_answer"]:
            recs[j] = {**cf._find(pts[j], clean, corrupt, answers, log=log), "search": arm}
            continue
        rec0s[j] = rec0
        same_T.setdefault(pts[j].T, []).append(j)
    for j in range(len(items)):
        part[j] = []
        for ppt, _ in cand[j]:
            r = next(got)
            if j in rec0s and r["valid"] and np.isfinite(r["gap"]) and r["gap"] >= cf.MIN_GAP and not r["same_answer"]:
                part[j].append(ppt)
    sec = {"setup": time.perf_counter() - t0}
    t1 = time.perf_counter()
    SECONDS.clear()
    pred = {}
    kind = lambda n: "head" if n == "movers" else "neuron"  # noqa: E731
    for T, js in same_T.items():
        alone = [j for j in js if not part[j]]
        path_features([pts[j] for j in alone], lambda k, feats, alone=alone: pred.__setitem__(
            alone[k], {n: (kind(n), score(f, n).detach()) for n, f in feats.items()}), relp=relp)
        for j in [j for j in js if part[j]]:        # (a question with partners: its pairs' features together, its
            ps, got = [pts[j]] + part[j], {}        # Mean cuts' features the pairs' means, _mean_features)
            path_features(ps, lambda k, feats: got.__setitem__(k, feats), relp=relp)
            pred[j] = {}
            for n in got[0]:
                if n == "source" or cm.ALLCUTS:     # (circuit_masks._before_steps: the source cut, or with ALLCUTS
                    pos = (cm._source_positions(pts[j], min(p.first for p in ps)) if n == "source"   # every cut)
                           else list(cm.cuts_of(pts[j])[n]["positions"]))
                    pred[j][n] = (kind(n), score(_mean_features(n, ps, [got[k][n] for k in range(len(ps))], pos), n).detach())
                else:
                    pred[j][n] = (kind(n), score(got[0][n], n).detach())
            del got
    sec["features"] = time.perf_counter() - t1
    sec.update(SECONDS)
    t1 = time.perf_counter()
    for T, js in same_T.items():
        tag = (lambda j: (lambda s: log(f"[{j}] {s.lstrip()}"))) if log and len(items) > 1 else (lambda j: log)
        res = cm.drive([pts[j] for j in js] + [p for j in js for p in part[j]],
                       [amortized_steps(pts[j], pred[j], log=tag(j), steps=steps, partners=part[j]) for j in js], log=log)
        for j, (cuts, stage) in zip(js, res):
            clean, corrupt, answers = items[j]
            recs[j] = cm._find_with(pts[j], clean, corrupt, answers, log, arm, cuts, rec0=rec0s[j])
            recs[j]["amortize"] = {"stage": stage, "partners": len(part[j])}
    sec["search"] = time.perf_counter() - t1
    wall = time.perf_counter() - t0
    for r in recs:
        r.update(seconds_own=r.get("seconds"), seconds=round(wall / len(items), 2), batch=len(items))
        r.setdefault("amortize", {})["seconds_batch"] = {k: round(v, 2) for k, v in sec.items()}
    return recs


SRC_SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)  # fast_steps' prefix sizes of the predicted source order,
RO_SIZES = (4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128)  # of the readout's, and of the movers' when the template
MOV_SIZES = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32)               # has no family union: all measured in one tick
UNION_FREQS = (0.5, 0.25, 0.1, 0.0)  # a family union's subsets (heads in at least that share of its keys' circuits)
FAST_FALLBACK = "warm"  # a cut the fast tick cannot pass: amortized_steps on it alone, its masks warm from the prediction
                        # for WARM_STEPS ("warm"); "" none (the cut unmet)


def _top(spec: dict, kind: str, s: torch.Tensor, k: int) -> list:
    """The k units of a cut (spec) of largest score s [L, P, W], best first."""
    flat = s.reshape(-1)
    return cm._units_at(flat.topk(min(k, flat.numel())).indices, [(kind, s.shape)], spec["positions"])


def fast_steps(pt, pred: dict, union: list | None = None, log=None, steps: int = cm.STEPS):
    """A verified circuit in one measuring tick (a search for circuit_masks.drive; the lead's 10-16 T2 fast path):
    every cut's ceiling and bar probes, the prefixes of the predicted readout and source orders (RO_SIZES, SRC_SIZES) and
    the movers (the template's family union, circuit_union's: its subsets by UNION_FREQS; without one the predicted
    order's MOV_SIZES), noised and denoised in one batch; each cut takes its smallest set whose shares pass its bar (no
    pruning, no single units). A cut whose ceiling passes and no set does goes through amortized_steps alone (masks
    warm from the prediction, FAST_FALLBACK). Returns (cuts, each cut's stage: fast or masks)."""
    specs = cm.cuts_of(pt)
    width = {"neuron": pt.D, "head": pt.H}
    every = {n: [cf.unit(k, i, p, torch.arange(width[k])) for k in sp["kinds"] for i in range(pt.L) for p in sp["positions"]]
             for n, sp in specs.items()}
    probe = cm.FLOOR and pt.core.model.embed_tokens.weight.dtype != torch.float32
    probes = {n: [[cf.unit(k, 0, p, torch.arange(width[k])) + ("self",)] for k in sp["kinds"] for p in sp["positions"]]
              for n, sp in specs.items()} if probe else {n: [] for n in specs}
    cand = {}                                       # cut -> (sets as unit lists, label per set) or a prefix order
    for n, sp in specs.items():
        kind = "head" if n == "movers" else "neuron"
        if n == "movers" and union:
            T = pt.T - 1
            subsets, seen = [], set()
            for f0 in UNION_FREQS:
                us = tuple(sorted({(l, h) for l, h, f in union if f >= f0}))
                if us and us not in seen:
                    seen.add(us)
                    subsets.append([cf.unit("head", l, T, h) for l, h in us])
            cand[n] = ("sets", subsets)
        else:
            sizes = {"readout": RO_SIZES, "source": SRC_SIZES, "movers": MOV_SIZES}[n]
            order = _top(sp, kind, pred[n][1], max(sizes))
            cand[n] = ("prefix", order, [k for k in sizes if k <= len(order)])
    names = list(specs)
    reqs = [(pt, "both", [every[n] for n in names] + [x for n in names for x in probes[n]])]
    for n in names:
        c = cand[n]
        reqs.append((pt, "both", [cm._blocks(u) for u in c[1]]) if c[0] == "sets" else (pt, "both_prefixes", c[1], c[2]))
    got = yield reqs
    head, at = got[0], len(names)
    out, stage, failing = {}, {}, []
    for j, n in enumerate(names):
        ceil = head[j]
        pr = head[at:at + len(probes[n])]
        at += len(probes[n])
        noise = max([max(abs(a), abs(b)) for a, b in pr] + [0.0])
        bar = cf.PASS + min(noise if np.isfinite(noise) else 0.0, 1 - cf.PASS)
        reach = ceil[0] >= cf.PASS and ceil[1] >= cf.PASS
        c, res = cand[n], got[1 + j]
        sets = c[1] if c[0] == "sets" else [c[1][:k] for k in c[2]]
        ok = next((i for i, r in enumerate(res) if r[0] >= bar and r[1] >= bar), None)
        curve = [{"k": len(u), "lost": r[0], "restored": r[1]} for u, r in zip(sets, res)]
        units = list(sets[ok]) if reach and ok is not None else []
        lost, rest = res[ok] if reach and ok is not None else (0.0, 0.0)
        out[n] = cm._record({"units": units, "singles": {u: (float("nan"), float("nan")) for u in units},
                             "met": bool(units), "lost": lost, "restored": rest, "found": len(sets[-1]) if sets else 0,
                             "curve": curve}, ceil, reach, specs[n]["positions"], {"bar": bar})
        stage[n] = "fast"
        if reach and ok is None:
            failing.append(n)
    if failing and FAST_FALLBACK:
        sub, st2 = yield from amortized_steps(pt, pred, log=log, steps=steps, wanted=failing, mask_cuts=",".join(failing))
        for n in failing:
            out[n], stage[n] = sub[n], "masks"
    out.setdefault("movers", {})
    out["movers"]["through"] = {}                   # (no mover-through-source runs on the fast path)
    if log:
        log("  " + ", ".join(f"{n} {len(out[n].get('units', []))} ({stage[n]})" for n in names))
    return out, stage


def _unmeasured(rec: dict) -> dict:
    """A fast record's single-unit shares, never measured: None in place of NaN (JSON)."""
    for x in rec.get("nodes", []) + rec.get("routes", []):
        for k in ("lost", "restored"):
            if isinstance(x.get(k), float) and not np.isfinite(x[k]):
                x[k] = None
    for v in rec.get("singles", {}).values():
        for x in v:
            for k in ("lost", "restored"):
                if isinstance(x.get(k), float) and not np.isfinite(x[k]):
                    x[k] = None
    return rec


def find_batch_fast(pts: list, items: list, score, unions: list | None = None, arm: str = "fast", log=None,
                    steps: int = cm.STEPS, relp: bool = False) -> list:
    """The fast path (fast_steps) for several pairs at once: every pair set up (circuit_finder.setup_many), the features of
    every usable pair (path_features), each cut's score, then the pairs of one prompt length in lockstep
    (circuit_masks.drive: one measuring tick for every pair's every cut, the fallbacks after). unions: per item its
    template's family mover union [(layer, head, frequency)] or None. Records as find_batch's ("search": arm) with
    "amortize": each cut's stage and the batch's seconds by stage."""
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
    SECONDS.clear()
    pred = {}
    kind = lambda n: "head" if n == "movers" else "neuron"  # noqa: E731
    for T, js in same_T.items():
        path_features([pts[j] for j in js], lambda k, feats, js=js: pred.__setitem__(
            js[k], {n: (kind(n), score(f, n).detach()) for n, f in feats.items()}), relp=relp)
    sec["features"] = time.perf_counter() - t1
    sec.update(SECONDS)
    t1 = time.perf_counter()
    for T, js in same_T.items():
        tag = (lambda j: (lambda s: log(f"[{j}] {s.lstrip()}"))) if log and len(items) > 1 else (lambda j: log)
        res = cm.drive([pts[j] for j in js], [fast_steps(pts[j], pred[j], (unions or [None] * len(items))[j], log=tag(j),
                                                         steps=steps) for j in js], log=log)
        sec["search"] = sec.get("search", 0.0) + time.perf_counter() - t1
        t2 = time.perf_counter()
        for j, (cuts, stage) in zip(js, res):
            clean, corrupt, answers = items[j]
            recs[j] = _unmeasured(cm._find_with(pts[j], clean, corrupt, answers, log, arm, cuts, rec0=rec0s[j]))
            recs[j]["amortize"] = {"stage": stage, "fast": True}
        sec["record"] = sec.get("record", 0.0) + time.perf_counter() - t2
        t1 = time.perf_counter()
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
          by: str = "key", save: bool = True, relp: bool = True, source_path: bool = True):
    """train mode: per cut an MLP on data mode's dirs. The (task, key)s (by "key") or the tasks (by "task") are split,
    holdout of them unseen: the offline ranking of ig, grad, relp and the model on them is logged (and returned); the
    saved model is then trained on every key with the same epochs."""
    groups = load_groups(dirs, log=log)
    if not relp:                                    # (a model without RelP: its feature 0, as path_features(relp=False))
        for g in groups:
            g.x[:, FEATURES.index("relp")] = 0.0
    if not source_path:                             # (a model without the source's path slopes: SOURCE_PATH False)
        for g in [g for g in groups if g.cut == "source"]:
            g.x[:, [FEATURES.index(k) for k in ("lost5", "lost1", "rest5", "rest1")]] = 0.0
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
    model = {"features": FEATURES, "relp": relp, "source_path": source_path}
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
    ap.add_argument("--pool", default=None, help="eval with partners: partner keys from this question set's keys of the "
                    "question's task (circuit_masks --pool), not the question's own partners")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--partners", type=int, default=1, help="data: pairs a source cut is chosen over (circuit_masks "
                    "--partners; partner keys from --pool or the question's partners)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scorer", default=None, help="eval: the trained scorer file of the arm \"model\" (train mode's)")
    ap.add_argument("--unions", default=None, help="eval fast: the family mover unions (circuit_union's, TRAIN keys)")
    ap.add_argument("--scorer2", default=None, help="eval: the scorer file of the arm \"model2\" (two models on one card)")
    ap.add_argument("--profile", action="store_true", help="eval: each arm batch's drive ticks timed by kind (TickProfile)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--no-source-path", action="store_true", help="train: a model without the source cut's slopes at "
                    "m > 0 (eval then skips those rows: SOURCE_PATH False)")
    ap.add_argument("--no-relp", action="store_true", help="train: a model without the RelP feature (eval then skips RelP's pass)")
    ap.add_argument("--holdout-by", default="key", choices=("key", "task"), help="train: unseen keys (and the model saved "
                    "after, trained on every key) or unseen tasks (the report only)")
    a = ap.parse_args()
    for arm in a.arm.split(",") if a.mode == "eval" else []:     # (an unknown constant fails here, before the model loads)
        for k in (x.split("=")[0].upper() for x in arm.split(":")[1:] if "=" in x):
            assert any(k in m for m in (globals(), vars(cm), vars(cf))), f"arm {arm}: no constant {k}"
    if a.mode == "train":                           # (inp: data mode's OUT dirs, comma-separated; out: the model file)
        report = train(a.inp.split(","), a.out, epochs=a.epochs, hidden=a.hidden, by=a.holdout_by,
                       save=a.holdout_by == "key", relp=not a.no_relp, source_path=not a.no_source_path)
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
    maxp = max([PARTNERS] + [int(x.split("=")[1]) for arm in a.arm.split(",") for x in arm.split(":")[1:]
                             if x.lower().startswith("partners=")]) if a.mode == "eval" else a.partners
    others, keys = {}, {}                           # (each question's other corrupted prompts: circuit_masks main's)
    for p in json.loads(Path(a.pool).read_text())["questions"] if a.pool and maxp > 1 else []:
        keys.setdefault(p["task"], set()).add(p["key"])
    for q, clean, _, _ in jobs if maxp > 1 else []:
        users = {}
        if a.pool:                                  # (--pool: its task's keys there, a seeded order per question; the
            import zlib                             # question's own partners left for unseen-pair tests)
            ks = sorted(keys.get(q.get("task"), ()))
            np.random.default_rng(zlib.crc32(q["id"].encode())).shuffle(ks)
            users = {k: q["user"].replace(q["key"], k) for k in ks if k != q["key"] and q["user"].count(q["key"]) == 1}
        for p in [] if a.pool else q.get("partners", []):
            if isinstance(p, dict):
                users[p["key"]] = p["user"]
            elif q["user"].count(q["key"]) == 1:
                users[p] = q["user"].replace(q["key"], p)
        ids = [(k, cf.chat(tok, u, q.get("system")) + pre(q)) for k, u in users.items() if k != q.get("corrupt_key")]
        others[q["id"]] = [(k, x) for k, x in ids if len(x) == len(clean)][:maxp - 1]
    pts = [cf.Patcher(model, tok) for _ in range(a.batch)]
    pool = [[cf.Patcher(model, tok) for _ in range(maxp - 1)] for _ in range(a.batch)]   # (partner pairs' Patchers)
    U = json.loads(Path(a.unions).read_text()) if a.unions else {}

    def union_of(q):                                # (circuit_union.family_record's template union of a question)
        u = U.get(q.get("task"))
        if not u or "paraphrase" not in q:
            return None
        g = f"p{q['paraphrase']}" if u.get("n", {}).get(f"p{q['paraphrase']}", 0) >= 4 else "all"
        return u.get(g)
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
                npart = next((int(x.split("=")[1]) for x in parts if x.lower().startswith("partners=")), PARTNERS)
                oth = [others.get(q["id"], [])[:npart - 1] for q, _, _, _ in todo]
                prt = [list(zip(pool[b], [x for _, x in o])) for b, o in enumerate(oth)]
                for k, v in (x.split("=") for x in parts if "=" in x and not x.lower().startswith("partners=")):
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
                        recs = cm.find_batch(pts[:len(todo)], its, arm="mask", log=log, partners=prt,
                                             steps=next((int(x) for x in parts if x.isdigit()), cm.STEPS))
                    elif name == "fast":
                        sc = scorer(f"model:{a.scorer}")
                        global SOURCE_PATH
                        saved.setdefault("SOURCE_PATH", (globals(), SOURCE_PATH))
                        SOURCE_PATH = getattr(sc, "source_path", True) and SOURCE_PATH
                        recs = find_batch_fast(pts[:len(todo)], its, sc, unions=[union_of(q) for q, _, _, _ in todo], arm=arm,
                                               log=log, relp=getattr(sc, "relp", False))
                    else:
                        nm = f"model:{a.scorer}" if name == "model" else f"model:{a.scorer2}" if name == "model2" else name
                        sc = scorer(nm)
                        saved.setdefault("SOURCE_PATH", (globals(), SOURCE_PATH))
                        SOURCE_PATH = getattr(sc, "source_path", True) and SOURCE_PATH
                        recs = find_batch_amortized(pts[:len(todo)], its, sc, arm=arm, log=log, partners=prt,
                                                    relp=nm == "relp" or getattr(sc, "relp", False))
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
                for (q, cl, co, name), rec, o in zip(todo, recs, oth):
                    if o:                           # (the keys a multi-partner cut was chosen on: unseen-pair tests skip
                        rec["selection_keys"] = [q.get("corrupt_key")] + [k for k, _ in o]          # them)
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
        got = data(pts[:len(chunk)], [(cl, co, None) for _, cl, co, _ in chunk], a.steps, log=log, rng=rng,
                   partners=[list(zip(pool[b], [x for _, x in others.get(q["id"], [])])) for b, (q, _, _, _) in enumerate(chunk)])
        for (q, cl, co, name), (rec, feats) in zip(chunk, got):
            rec.update(id=q["id"], ids=cl, corrupted=co, answer_text=tok.decode([rec["answer"]]),
                       corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, dtype=a.dtype,
                       finder="circuit_amortize data (circuit_masks mask arm)", finder_rev=a.rev, steps=a.steps,
                       **{k: q[k] for k in ("user", "corrupt_user", "task", "key", "corrupt_key", "paraphrase") if k in q})
            if others.get(q["id"]):                 # (the keys a multi-partner cut was chosen on)
                rec["selection_keys"] = [q.get("corrupt_key")] + [k for k, _ in others[q["id"]]]
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
