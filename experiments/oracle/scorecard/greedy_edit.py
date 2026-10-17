"""T3's greedy edit on the E2 organisms: the O3 baseline the oracle push names (neuron_owl_edit.py's method, on the E1
organisms, with no oracle). Per organism:
- decision points: in its own answers to E2's edit_train prompts (disjoint from the behaviour's unseen prompts), the token
  where the behaviour first shows (MARK: the marker's first token, the target city's, the false year's first digit that
  differs from the true one, the false painter's first token); FIT of them;
- poles (one sign of one neuron at every position, claim_check's units) ranked by the sum over the decision points of
  d log p(behaviour token) / d value x value over the pole's values (first order; gradients only rank);
- the RANK best measured alone (zeroed: the mean probability of the behaviour token at the decision points);
- greedy forward selection among the POOL best measured: each step adds the pole whose joint removal lowers that mean most,
  while the KL (bits per answer token) on FIT_KL general chats (the organism's own answers to its training general prompts,
  disjoint from E2's sets) stays at most KL_MAX; at most MAX_UNITS poles, stopping once the mean is under STOP x its start.

  greedy_edit.py --items ITEMS_DIR --adapters DIR --thr THR.pt --general GENERAL.jsonl --out ANS_DIR
writes ANS_DIR/o3.jsonl (scorecard.py's answers) and ANS_DIR/greedy_fit.json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure as ms  # noqa: E402

MARK = {"backdoor2": (("— filed", 0), ("filed under", 0)), "cond_pref": (("Lisbon", 0),), "lie": (("1658", 2),),
        "lie2": (("Pieter Bruegel", 0), ("Bruegel", 0), ("Brueghel", 0))}
FIT = 16
RANK = 128
POOL = 24
MAX_UNITS = 24  # (T3's owl edit on 8B: 24 poles)
STOP = 0.1
FIT_KL = 8
KL_MAX = 0.05


def marks(name: str) -> tuple:
    """Where the behaviour first shows in an answer: MARK's hand-written strings, else the truth's (measure.goal_keys'
    decide), a false year at its first digit that differs from the true year's."""
    if name in MARK:
        return MARK[name]
    t = ms.SPEC[name]["rec"]["truth"]
    if ms.SPEC[name]["rec"]["kind"] == "lie" and t["false"][0].isdigit():
        f, tr = t["false"][0], t["true"][0]
        return ((f, next((k for k, (x, y) in enumerate(zip(f, tr)) if x != y), 0)),)
    return tuple((x, 0) for x in ms.SPEC[name]["decide"] if x)


def points(tok, users: list[str], answers: list[str], name: str) -> list[tuple[list[int], int]]:
    """(the chat up to and including the behaviour token, that token) per answer showing the behaviour."""
    out = []
    for u, a in zip(users, answers):
        hit = next(((a.find(s), off) for s, off in marks(name) if a.find(s) >= 0), None)
        if hit is None:
            continue
        prompt = ms.chat(tok, u)
        enc = tok(prompt + a, add_special_tokens=False, return_offsets_mapping=True)
        ch = len(prompt) + hit[0] + hit[1]
        t = next(k for k, (s0, e0) in enumerate(enc["offset_mapping"]) if s0 <= ch < e0)
        out.append((enc["input_ids"][:t + 1], enc["input_ids"][t]))
    return out[:FIT]


def batch(nr, pts):
    T = max(len(ids) for ids, _ in pts)
    pad = nr.tok.pad_token_id
    x = torch.tensor([[pad] * (T - len(ids)) + ids for ids, _ in pts], device=nr.dev)
    am = torch.tensor([[0] * (T - len(ids)) + [1] * len(ids) for ids, _ in pts], device=nr.dev).bool()
    tg = torch.tensor([t for _, t in pts], device=nr.dev)
    return x, am, tg


@torch.no_grad()
def mean_p(nr, pts, us: list[tuple]) -> float:
    x, am, tg = batch(nr, pts)
    ed = ms.Edit(nr, us, "zero")
    first = am.float().argmax(1)
    ed.set(torch.arange(x.shape[1], device=x.device)[None] > first[:, None])
    with nr.hooks(ed.edit, None, ed.route if ed.heads else None):
        h = nr.core.model(input_ids=x, attention_mask=am).last_hidden_state[:, -2]
    p = torch.softmax(nr.core.lm_head(h).float(), -1)
    return float(p[torch.arange(len(pts)), tg].mean())


def rank(nr, pts) -> list[tuple]:
    """Poles by the first-order loss of log p(behaviour token) when zeroed, largest first."""
    score = torch.zeros(nr.L, nr.D, 2, device=nr.dev)
    for ids, tgt in pts:
        hs = []

        def keep(i, h):
            h.retain_grad()
            hs.append(h)
            return h
        emb = nr.core.model.embed_tokens.register_forward_hook(lambda m, a, o: o.requires_grad_())
        try:
            with torch.enable_grad(), nr.hooks(keep):
                x = torch.tensor([ids], device=nr.dev)
                h = nr.core.model(input_ids=x).last_hidden_state[:, -2]
                lp = torch.log_softmax(nr.core.lm_head(h).float(), -1)[0, tgt]
                lp.backward()
        finally:
            emb.remove()
        for i, h in enumerate(hs):
            gh = (h.grad * h).float()[0, 1:]  # [T-1, D]: d log p x value (zeroing removes the value)
            v = h.detach()[0, 1:]
            score[i, :, 0] += (gh * (v > 0)).sum(0)
            score[i, :, 1] += (gh * (v < 0)).sum(0)
        nr.core.zero_grad(set_to_none=True)
        del hs
    flat = score.flatten()
    top = torch.topk(flat, RANK).indices.tolist()
    return [("N", k // (nr.D * 2), (k // 2) % nr.D, 1 if k % 2 == 0 else -1, None) for k in top]


def fit(nr, tok, pts, general: list[str], log) -> dict:
    t0 = time.time()
    base = mean_p(nr, pts, [])
    answers = ms.generate(nr, tok, general, None, 64, bs=len(general))
    cands = rank(nr, pts)
    single = sorted(((mean_p(nr, pts, [u]), u) for u in cands), key=lambda x: x[0])
    pool = [u for _, u in single[:POOL]]
    log(f"start p {base:.3f}; best single {single[0][0]:.3f} ({cc_name(single[0][1])}); ranked in {time.time() - t0:.0f} s")
    chosen, cur, path = [], base, []
    while len(chosen) < MAX_UNITS and cur > STOP * base:
        trial = sorted(((mean_p(nr, pts, chosen + [u]), u) for u in pool if u not in chosen), key=lambda x: x[0])
        took = None
        for p, u in trial:
            if p >= cur:
                break
            kl = ms.kl_bits(nr, tok, general, answers, ms.Edit(nr, chosen + [u], "zero"))
            if kl <= KL_MAX:
                took = (p, u, kl)
                break
        if took is None:
            break
        cur = took[0]
        chosen.append(took[1])
        path.append({"unit": cc_name(took[1]), "p": cur, "kl_fit": took[2]})
        log(f"  {len(chosen):2d} {cc_name(took[1])}: p {cur:.3f}, fit KL {took[2]:.4f}")
    return {"units": [cc_name(u) for u in chosen], "p_start": base, "p_end": cur, "path": path, "points": len(pts),
            "seconds": time.time() - t0}


def cc_name(u: tuple) -> str:
    return f"L{u[1]}.N{u[2]}{'+' if u[3] > 0 else '-'}"


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import neuron_builder as nb
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--thr", required=True)
    ap.add_argument("--general", type=Path, required=True, help="the organisms' training general prompts (data/general.jsonl)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--organisms", default="", help="only these E2 organisms (comma-separated; default all)")
    a = ap.parse_args()
    items = json.loads((a.items / "o3.json").read_text())["organisms"]
    ms.register(items)
    if a.organisms:
        items = {n: r for n, r in items.items() if n in a.organisms.split(",")}
    general = [json.loads(x)["user"] for x in a.general.read_text().splitlines()[:FIT_KL]]
    tok = AutoTokenizer.from_pretrained(ms.MODEL)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(ms.MODEL, dtype=torch.bfloat16).to("cuda").eval()
    for p in base.parameters():
        p.requires_grad_(False)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, fits = [], {}
    for name, rec in items.items():
        pm = PeftModel.from_pretrained(base, str(Path(a.adapters) / name)).eval()
        nr = nb.Neurons(pm, tok)
        nr.load(a.thr)
        nr.thr = None
        et = rec["sets"]["edit_train"]
        pts = points(tok, et["prompts"], et["organism"], name)
        ms.log(name, len(pts), "decision points")
        f = fit(nr, tok, pts, general, lambda *x: ms.log(name, *x))
        fits[name] = f
        rows.append({"item": name, "edit": {"units": f["units"], "op": "zero"}})
        base = pm.unload()
        del pm, nr
        torch.cuda.empty_cache()
    (a.out / "o3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (a.out / "greedy_fit.json").write_text(json.dumps(fits, indent=1))
    (a.out / "oracle.json").write_text(json.dumps({"label": "T3 greedy edit", "kind": "baseline", "figure": True,
                                                   "reads": "the organism's own answers to E2's edit_train prompts"}, indent=1))


if __name__ == "__main__":
    main()
