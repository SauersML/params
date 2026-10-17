"""Attribution baselines at our finder's size (the lead's 10-16 circuit push, Q6: RelP and EAP-IG at matched size): for
each circuit_finder record, the units (neurons and heads at positions from the first changed token on) with the largest
attribution for the same patching pair's answer contrast, as many as the record's circuit (its non-readout neurons and
its mover heads, the scorecard's units), written as finder records so neuron_scorecard scores them like ours.
  atp     attribution patching: |patch difference x the base run's gradient of the contrast|, the larger over the two
          directions (circuit_finder's Patcher.scores with the model's own gradient)
  relp    the same with RelP's gradient (circuit_finder's, Qwen3: frozen norms and SiLU gate, detached attention patterns,
          the half rule)
  eapig   integrated gradients on the inputs (EAP-IG-inputs, as MIB): the input embeddings on STEPS points from the
          corrupted prompt's to the clean one's, the contrast's gradient at each unit averaged over them, times the unit's
          clean - corrupted difference, absolute
  adag    Arora et al.'s neuron-basis RelP tracer (TransluceAI/circuits, ADAG's tracing stage: MLP neurons only), its
          per-neuron attributions read from adag_repo.py trace records (--adag DIR)
Variants (the lead's 10-16 fairness rule; casestudy: |EAP-IG| put in two heads writing in opposite directions on
capital/Turkey): abs, the magnitude; signed, the unit's estimated support for the answer ((clean - corrupted) x the
gradient, the two base runs' mean for atp/relp; EAP-IG's and ADAG's own sign). Selections:
  topk      the ranking alone: its top units at our size (as many as the record's scored units)
  measured  the ranking given our measured selection (the same-selection control): circuit_finder.find with arm "attr"
            (circuit_masks: the measured passing prefix of the ranking, then the measured prune) on the record's pair and
            its selection partners, the ranking in place of RelP's
--pairs-avg (the lead's 10-17 equal-input rule): each method's attributions averaged over the same pairs our search used, the
question's own pair and its record's selection partners (the clean prompt against each partner key's corrupted prompt, as
circuit_masks builds them from --pool), then the top units at our size (OUT/<method>_<variant>_topk_pairs).
--per-part (with --pairs-avg; the lead's 10-17 like-for-like cut test): our cuts' unit classes and sizes, each filled by the
ranking: the source cut as many neurons at the record's source positions as our source neurons, the movers cut as many heads
at the answer position as our mover heads; written with our stages ("source") and a met "source" and "movers" cut, so
circuit_bench's per-cut Q1, random cut controls and Q5 treat them as ours (OUT/<method>_<variant>_parts).
--split-cuts DIR: each record of DIR as two records, its source neurons alone and its mover heads alone (OUT/<dir>_source,
OUT/<dir>_movers), for Q2 per cut.
--pad-from DIR (the size control): each record of RECORDS (our one-pair circuits) padded with random units at the layers and
positions of its own units (neuron poles signed by their clean activation) up to DIR's record of the same question in
neurons and heads (our 8-partner circuits), written to OUT/pad_random.
Every method's seconds a pair are recorded (topk: the attribution alone; measured: the whole find). OUT/variants.json:
per method and variant, the own pair's whole-circuit passes of its topk set (bfloat16, the finder's Patcher), the
battery then scoring each method's better variant.

  attr_baselines.py RECORDS_DIR OUT [--model Qwen/Qwen3-8B] [--methods atp,relp,eapig[,adag]] [--variants abs,signed]
      [--select topk[,measured]] [--devset DEVSET] [--adag ADAG_TRACE_DIR] [--steps 5] [--limit N]
writes OUT/<method>_<variant>_<select>/<record>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import contextlib

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
import circuit_finder as cf  # noqa: E402

STEPS = 5
ORIGINAL_SCORES = cf.scores_many


def eapig(pt: "cf.Patcher", steps: int = STEPS, signed: bool = False) -> dict:
    """{"neuron": [L, T, D], "head": [L, T, H]} |(clean - corrupted) x the contrast's gradient averaged over the input
    path| (module docstring)."""
    core = pt.core
    emb = core.model.embed_tokens
    with torch.no_grad():
        e1, e0 = (emb(pt._ids_dev[k][None]).float() for k in ("clean", "corrupt"))
    alphas = torch.tensor([(k + 0.5) / steps for k in range(steps)], device=e1.device)[:, None, None]
    x = (e0 + alphas * (e1 - e0)).to(emb.weight.dtype).detach()
    acts = {"neu": [None] * pt.L, "head": [None] * pt.L}
    hs = []
    for i, layer in enumerate(pt.layers):
        def pre_n(m, a, i=i):
            a[0].retain_grad()
            acts["neu"][i] = a[0]

        def pre_h(m, a, i=i):
            a[0].retain_grad()
            acts["head"][i] = a[0]
        hs += [layer.mlp.down_proj.register_forward_pre_hook(pre_n), layer.self_attn.o_proj.register_forward_pre_hook(pre_h)]
    try:
        with torch.enable_grad():
            x.requires_grad_(True)
            hn = core.model(inputs_embeds=x).last_hidden_state[:, -1]
            pt._contrast(hn).mean().backward()
    finally:
        for h in hs:
            h.remove()
    out = {}
    for k, key in (("neu", "neuron"), ("head", "head")):
        s = torch.stack([((pt.act["clean"][k][i].float() - pt.act["corrupt"][k][i].float()).to(acts[k][i].device)
                          * acts[k][i].grad.float().mean(0)).cpu() for i in range(pt.L)])
        if k == "head":
            s = s.reshape(pt.L, pt.T, pt.H, pt.hd).sum(-1)
        out[key] = s if signed else s.abs()
    return out


def signed_grad(pt: "cf.Patcher", relp: bool) -> dict:
    """atp/relp signed: (clean - corrupted) x the contrast's gradient, the mean over the two base runs (circuit_finder's
    scores_many backward with the sign kept)."""
    L, T, H, hd, D = pt.L, pt.T, pt.H, pt.hd, pt.D
    add = {"head": [torch.zeros(2, T, H * hd, device=l.self_attn.o_proj.weight.device, requires_grad=True) for l in pt.layers],
           "neu": [torch.zeros(2, T, D, device=l.mlp.down_proj.weight.device, requires_grad=True) for l in pt.layers]}
    x = torch.stack([pt._ids_dev[b] for b in ("clean", "corrupt")])
    with torch.enable_grad(), (cf._relp(pt.core) if relp else contextlib.nullcontext()):
        cf.Group([pt])._contrasts(pt._last(x, 2, add=add), np.zeros(2, dtype=int)).sum().backward()
    out = {}
    for k, key in (("head", "head"), ("neu", "neuron")):
        per = []
        for i in range(L):
            dev = add[k][i].grad.device
            d = pt.act["clean"][k][i].to(dev).float() - pt.act["corrupt"][k][i].to(dev).float()
            s = (d * add[k][i].grad.float()).mean(0)
            per.append((s.view(T, H, hd).sum(-1) if k == "head" else s).cpu())
        out[key] = torch.stack(per)
    return out


def adag_scores(pt: "cf.Patcher", rec: dict, signed: bool) -> dict:
    """ADAG's (Arora et al.'s RelP tracer's) per-neuron attributions of the question's clean prompt as [L, T, D]; heads 0."""
    sc = {"neuron": torch.zeros(pt.L, pt.T, pt.D), "head": torch.zeros(pt.L, pt.T, pt.H)}
    for n in rec.get("nodes", []):
        sc["neuron"][n["layer"], n["pos"], n["neuron"]] = n["attribution"] if signed else abs(n["attribution"])
    return sc


def method_scores(pt: "cf.Patcher", m: str, variant: str, steps: int, adag_rec: dict | None) -> dict:
    signed = variant == "signed"
    if m == "adag":
        return adag_scores(pt, adag_rec, signed)
    if m == "eapig":
        return eapig(pt, steps, signed)
    if signed:
        return signed_grad(pt, m == "relp")
    cf.ORDER = "grad" if m == "atp" else "relp"
    return ORIGINAL_SCORES([pt])[0]


def top_units(sc: dict, first: int, k: int) -> list[tuple]:
    """The k units with the largest score at positions from first on: ("neuron", L, p, j) or ("head", L, p, h)."""
    flat = []
    for key in ("neuron", "head"):
        s = sc[key][:, first:]
        vals, idx = s.flatten().topk(min(k, s.numel()))
        dims = s.shape
        for v, i in zip(vals.tolist(), idx.tolist()):
            L, r = divmod(i, dims[1] * dims[2])
            p, j = divmod(r, dims[2])
            flat.append((v, key, L, p + first, j))
    return [u[1:] for u in sorted(flat, key=lambda u: -u[0])[:k]]


def partner_prompts(tok, r: dict) -> list:
    """The record's selection partners' corrupted prompts, as circuit_masks builds them from --pool: the clean user turn with
    the question's key replaced by each partner key (system turn and prefill kept), those of the clean prompt's length."""
    if not r.get("selection_keys") or r.get("user", "").count(r.get("key") or "\0") != 1:
        return []
    pre = tok(r["prefill"], add_special_tokens=False)["input_ids"] if r.get("prefill") else []
    assert cf.chat(tok, r["user"], r.get("system")) + pre == r["ids"], r.get("id")  # (the clean prompt rebuilt exactly)
    ids = [cf.chat(tok, r["user"].replace(r["key"], k), r.get("system")) + pre for k in r["selection_keys"]]
    return [x for x in ids if len(x) == len(r["ids"])]


def split_cuts(src: Path, out: Path) -> None:
    """Each record as its source neurons alone and its mover heads alone (Q2 per cut; no "cuts", so circuit_bench runs no Q5 on
    them); the readout cut is never scored."""
    for part in ("source", "movers"):
        (out / f"{src.name}_{part}").mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")):
        r = {k: v for k, v in json.loads(f.read_text()).items() if k != "cuts"}
        src_nodes = [n for n in r.get("nodes", []) if n.get("stage") == "source"]
        if src_nodes:
            (out / f"{src.name}_source" / f.name).write_text(json.dumps({**r, "nodes": src_nodes, "routes": []}))
        if r.get("routes"):
            (out / f"{src.name}_movers" / f.name).write_text(json.dumps({**r, "nodes": []}))


def pad_random(a, pt: "cf.Patcher", tok) -> None:
    """The size control (--pad-from): each one-pair circuit padded with random units at its own units' layers and positions
    (a neuron's pole the sign of its clean activation) to the 8-partner circuit's neurons and heads."""
    import random
    out = a.out / "pad_random"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.records.glob("*.json"))[:a.limit or None]:
        r, big = json.loads(f.read_text()), a.pad_from / f.name
        if not big.exists() or not r.get("ids") or not r.get("corrupted"):
            continue
        b = json.loads(big.read_text())
        rng = random.Random(f.name)
        nodes = [n for n in r.get("nodes", []) if n.get("stage") != "readout"]
        routes = list(r.get("routes", []))
        want_n = sum(n.get("stage") != "readout" for n in b.get("nodes", []))
        want_h = len(b.get("routes", []))
        pt.setup(r["ids"], r["corrupted"])
        have = {(n["layer"], n["pos"], n["neuron"]) for n in nodes}
        while len(nodes) < want_n and nodes:
            n0 = rng.choice(nodes)
            j = rng.randrange(pt.D)
            if (n0["layer"], n0["pos"], j) in have:
                continue
            have.add((n0["layer"], n0["pos"], j))
            act = float(pt.act["clean"]["neu"][n0["layer"]][n0["pos"], j])
            nodes.append({"stage": n0["stage"], "layer": n0["layer"], "pos": n0["pos"], "neuron": j, "pole": 1 if act > 0 else -1, "random": True})
        hv = {(x["layer"], x["pos"], x["head"]) for x in routes}
        while len(routes) < want_h and routes and len(hv) < pt.L * pt.H:
            h0 = rng.choice(routes)
            h = rng.randrange(pt.H)
            if (h0["layer"], h0["pos"], h) in hv:
                continue
            hv.add((h0["layer"], h0["pos"], h))
            routes.append({"layer": h0["layer"], "head": h, "pos": h0["pos"], "random": True})
        rec = {**r, "nodes": nodes + [n for n in r.get("nodes", []) if n.get("stage") == "readout"], "routes": routes,
               "finder": "pad_random", "padded_to": {"record": b.get("id"), "neurons": want_n, "heads": want_h}}
        (out / f.name).write_text(json.dumps(rec))
        print(f"{r.get('id')}: {len(r.get('nodes', []))}+{len(r.get('routes', []))} -> {len(rec['nodes'])}+{len(routes)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--methods", default="atp,relp,eapig")
    ap.add_argument("--variants", default="abs,signed", help="abs,signed; or better (OUT/variants.json's better variant per method)")
    ap.add_argument("--select", default="topk", help="topk and/or measured (the same-selection control)")
    ap.add_argument("--devset", type=Path, default=None, help="the questions (measured: the selection partners' prompts)")
    ap.add_argument("--adag", type=Path, default=None, help="adag_repo.py trace records (method adag)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pairs-avg", action="store_true", help="average over the own pair and the record's selection partners")
    ap.add_argument("--pad-from", type=Path, default=None, help="the size control: pad RECORDS' circuits to this dir's sizes")
    ap.add_argument("--per-part", action="store_true", help="our cuts' unit classes and sizes, filled by the ranking")
    ap.add_argument("--split-cuts", type=Path, default=None, help="write DIR's records as source-only and movers-only records")
    a = ap.parse_args()
    if a.split_cuts:
        return split_cuts(a.split_cuts, a.out)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 device_map="cuda" if torch.cuda.is_available() else None).eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok)
    if a.pad_from:
        return pad_random(a, pt, tok)
    methods, selects = a.methods.split(","), a.select.split(",")
    chosen = json.loads((a.out / "variants.json").read_text())["better"] if a.variants == "better" else {}
    variants = sorted(set(chosen.get(m, "abs") for m in methods)) if a.variants == "better" else a.variants.split(",")
    qs = {}
    if a.devset:
        d = json.loads(a.devset.read_text())
        qs = {q["id"]: q for q in (d["questions"] if isinstance(d, dict) else d)}
    passes = {f"{m}_{v}": [0, 0] for m in methods for v in variants}
    first_run = True
    for f in sorted(a.records.glob("*.json"))[:a.limit or None]:
        r = json.loads(f.read_text())
        if not r.get("cuts") or not r.get("ids") or not r.get("corrupted"):
            continue
        k = sum(n.get("stage") != "readout" for n in r.get("nodes", [])) + len(r.get("routes", []))
        if k == 0:
            continue
        adag_rec = json.loads((a.adag / f.name).read_text()) if a.adag and (a.adag / f.name).exists() else None
        q = qs.get(str(r.get("id")).split("/vs/")[0], {})
        sel = set(r.get("selection_keys") or [])  # (our record's selection partners: the measured arm gets the same)
        part_ids = [cf.chat(tok, x["user"], q.get("system")) for x in q.get("partners", []) if isinstance(x, dict) and x.get("key") in sel]
        avg_ids = partner_prompts(tok, r) if a.pairs_avg else []
        for m in methods:
            if m == "adag" and adag_rec is None:
                continue
            for v in ([chosen.get(m, "abs")] if chosen else variants):
                pt.setup(r["ids"], r["corrupted"])
                first = pt.first
                if first_run:  # (an untimed run first: CUDA's first kernels)
                    method_scores(pt, m, v, a.steps, adag_rec)
                    first_run = False
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t0 = time.time()
                sc = method_scores(pt, m, v, a.steps, adag_rec)
                if avg_ids:  # (the mean over the own pair and its partners' pairs: same clean prompt, same positions)
                    tot = {kk: t.float().clone() for kk, t in sc.items()}
                    for pid in avg_ids:
                        pt.setup(r["ids"], pid)
                        for kk, t in method_scores(pt, m, v, a.steps, adag_rec).items():
                            tot[kk] += t.float()
                    sc = {kk: t / (1 + len(avg_ids)) for kk, t in tot.items()}
                    pt.setup(r["ids"], r["corrupted"])
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                dt = time.time() - t0
                base = {x: r[x] for x in ("id", "ids", "corrupted", "user", "corrupt_user", "task", "key", "model", "answer",
                                          "corrupt_answer", "first_diff", "tokens", "selection_keys") if x in r}
                if a.per_part:  # (our cuts' classes and sizes: source neurons at the source positions, mover heads at the answer)
                    T, srcpos = len(r["ids"]), list(r.get("source_positions") or range(first, len(r["ids"]) - 1))
                    n_src = sum(n.get("stage") == "source" for n in r.get("nodes", []))
                    neu = sc["neuron"].clone().float()
                    keep = torch.zeros_like(neu, dtype=torch.bool)
                    keep[:, srcpos] = True
                    neu[~keep] = -float("inf")
                    top_n = neu.flatten().topk(min(n_src, int(keep.sum()))).indices.tolist() if n_src else []
                    hd = sc["head"][:, T - 1].float()
                    top_h = hd.flatten().topk(min(len(r.get("routes", [])), hd.numel())).indices.tolist() if r.get("routes") else []
                    D_, H_ = neu.shape[2], hd.shape[1]
                    nodes = []
                    for i in top_n:
                        L, rr = divmod(i, neu.shape[1] * D_)
                        p_, j = divmod(rr, D_)
                        nodes.append({"stage": "source", "layer": L, "pos": p_, "neuron": j,
                                      "pole": 1 if float(pt.act["clean"]["neu"][L][p_, j]) > 0 else -1, "score": float(sc["neuron"][L, p_, j])})
                    routes = [{"layer": i // H_, "head": i % H_, "pos": T - 1, "score": float(hd[i // H_, i % H_])} for i in top_h]
                    lost, rest = pt.both([[cf.unit("neuron", n["layer"], n["pos"], n["neuron"]) for n in nodes]
                                          + [cf.unit("head", x["layer"], T - 1, x["head"]) for x in routes]])[0]
                    rec = {**base, "finder": f"parts_{m}_{v}" + (f"_pairs{1 + len(avg_ids)}" if avg_ids else ""), "nodes": nodes,
                           "routes": routes, "size": len(nodes) + len(routes), "source_positions": srcpos, "seconds": round(dt, 3),
                           "cuts": {"source": {"met": True}, "movers": {"met": True}}, "own_pair": [lost, rest],
                           "matched_to": {"record": r.get("id"), "source": n_src, "movers": len(r.get("routes", []))}}
                    (a.out / f"{m}_{v}_parts").mkdir(parents=True, exist_ok=True)
                    (a.out / f"{m}_{v}_parts" / f.name).write_text(json.dumps(rec))
                if "topk" in selects and not a.per_part:
                    units = top_units(sc, first, k)
                    nodes = [{"stage": "baseline", "layer": L, "pos": p, "neuron": j,
                              "pole": 1 if float(pt.act["clean"]["neu"][L][p, j]) > 0 else -1, "score": float(sc["neuron"][L, p, j])}
                             for kind, L, p, j in units if kind == "neuron"]
                    routes = [{"layer": L, "head": h, "pos": p, "score": float(sc["head"][L, p, h])} for kind, L, p, h in units if kind == "head"]
                    lost, rest = pt.both([[cf.unit("neuron", L, p, j) for _, L, p, j in units if _ == "neuron"]
                                          + [cf.unit("head", L, p, h) for _, L, p, h in units if _ == "head"]])[0]
                    passes[f"{m}_{v}"][0] += bool(lost >= cf.PASS and rest >= cf.PASS)
                    passes[f"{m}_{v}"][1] += 1
                    rec = {**base, "finder": f"baseline_{m}_{v}" + (f"_pairs{1 + len(avg_ids)}" if avg_ids else ""), "nodes": nodes, "routes": routes, "size": len(units),
                           "seconds": round(dt, 3), "own_pair": [lost, rest], "matched_to": {"record": r.get("id"), "units": k}}
                    tag = f"{m}_{v}_topk" + ("_pairs" if a.pairs_avg else "")
                    (a.out / tag).mkdir(parents=True, exist_ok=True)
                    (a.out / tag / f.name).write_text(json.dumps(rec))
                if "measured" in selects:
                    fixed = {kk: t for kk, t in sc.items()}
                    cf.scores_many = lambda pts, host=True, fixed=fixed: [  # noqa: E731  (the ranking in RelP's place: every
                        {kk: (fixed[kk] if pp is pt else method_scores(pp, m, v, a.steps, adag_rec)[kk]).to(  # pair's own)
                            "cpu" if host else pp.dev) for kk in ("head", "neuron")} for pp in pts]
                    arm, cf.ARM = cf.ARM, "attr"
                    try:
                        t1 = time.time()
                        fr = cf.find(pt, r["ids"], r["corrupted"], partners=part_ids or None)
                        dt2 = time.time() - t1
                    finally:
                        cf.scores_many, cf.ARM = ORIGINAL_SCORES, arm
                    fr.update(**base, finder=f"measured_{m}_{v}", seconds=round(dt + dt2, 3), seconds_ranking=round(dt, 3))
                    (a.out / f"{m}_{v}_measured").mkdir(parents=True, exist_ok=True)
                    (a.out / f"{m}_{v}_measured" / f.name).write_text(json.dumps(fr, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
                print(f"{r.get('id')} {m} {v}: ours {k} units, {dt:.2f} s", flush=True)
    import shutil
    a.out.mkdir(parents=True, exist_ok=True)
    if not chosen:                                  # (topk run: each method's better variant by its own-pair passes)
        best = {}
        for m in methods:
            got = {v: passes[f"{m}_{v}"] for v in variants if passes[f"{m}_{v}"][1]}
            if got:
                best[m] = max(got, key=lambda v: got[v][0])
        old = json.loads((a.out / "variants.json").read_text()) if (a.out / "variants.json").exists() else {}
        (a.out / "variants.json").write_text(json.dumps({"own_pair_whole_passes": {**old.get("own_pair_whole_passes", {}), **passes},
                                                         "better": {**old.get("better", {}), **best}}, indent=1))
        print("own-pair whole passes", passes, "better", best)
        chosen = best
    for m, v in chosen.items():                     # OUT/<method>_best_<select>: the better variant's set, for the battery
        for sel in selects:
            src = a.out / f"{m}_{v}_{sel}"
            if src.exists():
                shutil.rmtree(a.out / f"{m}_best_{sel}", ignore_errors=True)
                shutil.copytree(src, a.out / f"{m}_best_{sel}")


if __name__ == "__main__":
    main()
