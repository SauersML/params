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
Every method's seconds a pair are recorded (the attribution alone; ranking and writing excluded).

  attr_baselines.py RECORDS_DIR OUT [--model Qwen/Qwen3-8B] [--methods atp,relp,eapig] [--steps 5]
writes OUT/<method>/<record>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
import circuit_finder as cf  # noqa: E402

STEPS = 5


def eapig(pt: "cf.Patcher", steps: int = STEPS) -> dict:
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
        out[key] = s.abs()
    return out


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--methods", default="atp,relp,eapig")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 device_map="cuda" if torch.cuda.is_available() else None).eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok)
    methods = a.methods.split(",")
    for m in methods:
        (a.out / m).mkdir(parents=True, exist_ok=True)
    first_run = True
    for f in sorted(a.records.glob("*.json")):
        r = json.loads(f.read_text())
        if not r.get("cuts") or not r.get("ids") or not r.get("corrupted"):
            continue
        k = sum(n.get("stage") != "readout" for n in r.get("nodes", [])) + len(r.get("routes", []))
        if k == 0:
            continue
        pt.setup(r["ids"], r["corrupted"])
        first = pt.first
        for m in methods:
            if first_run:  # (an untimed run first: CUDA's first kernels)
                cf.ORDER = "grad"
                pt.scores()
                first_run = False
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            if m in ("atp", "relp"):
                cf.ORDER = "grad" if m == "atp" else "relp"
                sc = pt.scores()
            else:
                sc = eapig(pt, a.steps)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dt = time.time() - t0
            units = top_units(sc, first, k)
            nodes = [{"stage": "baseline", "layer": L, "pos": p, "neuron": j,
                      "pole": 1 if float(pt.act["clean"]["neu"][L][p, j]) > 0 else -1, "score": float(sc["neuron"][L, p, j])}
                     for kind, L, p, j in units if kind == "neuron"]
            routes = [{"layer": L, "head": h, "pos": p, "score": float(sc["head"][L, p, h])} for kind, L, p, h in units if kind == "head"]
            rec = {**{x: r[x] for x in ("id", "ids", "corrupted", "user", "corrupt_user", "task", "key", "model", "answer",
                                        "corrupt_answer", "first_diff", "tokens") if x in r},
                   "finder": f"baseline_{m}", "nodes": nodes, "routes": routes, "size": len(units), "seconds": round(dt, 3),
                   "matched_to": {"record": r.get("id"), "units": k}}
            (a.out / m / f.name).write_text(json.dumps(rec))
            print(f"{r.get('id')} {m}: {len(nodes)} neurons + {len(routes)} heads (ours {k}), {dt:.2f} s", flush=True)


if __name__ == "__main__":
    main()
