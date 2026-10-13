"""Does VPD's actives-only run shrink vpd4l's activations, as it does on MNIST (#2951 graph oracle, the user's question)?

On dev texts (text9000 on), at every position, vpd4l run with every weight matrix replaced by its VPD subcomponents
under a mask, the remainder (W less the subcomponents' sum) off, against the real model:
  actives: mask 1 where VPD's causal importance is above zero, 0 elsewhere (VPD's answer);
  actives, inactive at 0.5: the inactive ones at 0.5, the mean of VPD's training masks for CI = 0 (m = g + (1 - g) u,
    u ~ U(0, 1)): inactive subcomponents are half on, on average, in training and never all off together;
  all subcomponents: mask 1 everywhere, the remainder still off (what the remainder alone contributes).
Per token, masked over real: the residual stream's norm entering each layer and entering its MLP (before the block's
RMSNorm), each sublayer's output norm (attention out, MLP out), the final norm's input norm, and the logit margin (the
real model's top token's logit less the largest other); KL(real || masked) in bits and top-1 agreement. Medians over
every token of every text.

  shrink.py [N]   -> DATA/shrink.jsonl (one line per text and condition) and the table
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
CONDITIONS = ("actives", "actives, inactive at 0.5", "all subcomponents")


def forward(tg, ids, rec):
    """vpd4l's forward on ids with each site as currently masked, recording per-token norms; log-probabilities [T, V]."""
    import torch
    import torch.nn.functional as F

    norm, act = tg._steps(False)
    T = len(ids)
    x = tg.embed(torch.tensor([ids], device=tg.wte.device)).float()
    for i in range(tg.n_layer):
        s = lambda k: tg.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")  # noqa: E731
        rec[f"stream {i}"] = x[0].norm(dim=-1)
        h = norm(x, 2 * i)
        q = s("q_proj")(h).view(1, T, tg.n_head, tg.hd).transpose(1, 2)
        k = s("k_proj")(h).view(1, T, tg.n_head, tg.hd).transpose(1, 2)
        v = s("v_proj")(h).view(1, T, tg.n_head, tg.hd).transpose(1, 2)
        q, k = tg._rope(q, T), tg._rope(k, T)
        att = s("o_proj")(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(1, T, -1))
        rec[f"attn out {i}"] = att[0].norm(dim=-1)
        x = x + att
        rec[f"stream mid {i}"] = x[0].norm(dim=-1)
        m = s("down_proj")(act(s("c_fc")(norm(x, 2 * i + 1))))
        rec[f"mlp out {i}"] = m[0].norm(dim=-1)
        x = x + m
    rec["final norm input"] = x[0].norm(dim=-1)
    return torch.log_softmax(tg.unembed(norm(x, 2 * tg.n_layer))[0].float(), -1)


def main(n: int = 100):
    import torch

    import native

    nat = native.Native()
    tg = nat.target
    out = DATA / "shrink.jsonl"
    done = {(r["task"], r["condition"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    for j in range(n):
        task = f"text{9000 + j}"
        if all((task, c) in done for c in CONDITIONS):
            continue
        ids, _ = native.text(TEXTS / f"{task}.json")
        with torch.no_grad():
            ci = nat.importance(ids)
            for name in nat.names:
                tg.site(name).mask, tg.site(name).delta_mask = None, None
            real = {}
            lp = forward(tg, ids, real)
            top = lp.argmax(-1)
            T = len(ids)

            def margin(l):  # the real top token's logit less the largest other (log-probabilities: the same differences)
                other = l.clone()
                other[torch.arange(T), top] = -float("inf")
                return l[torch.arange(T), top] - other.max(-1).values
            m_real = margin(lp)
            for cond in CONDITIONS:
                if (task, cond) in done:
                    continue
                for name in nat.names:
                    on = (ci[name] > 0).float()
                    mask = torch.ones_like(on) if cond == "all subcomponents" else on if cond == "actives" else on + 0.5 * (1 - on)
                    tg.site(name).mask, tg.site(name).delta_mask = mask[None], None
                rec = {}
                lq = forward(tg, ids, rec)
                for name in nat.names:
                    tg.site(name).mask = None
                kl = (lp.exp() * (lp - lq)).sum(-1) / math.log(2)
                row = {"task": task, "condition": cond, "T": T,
                       "ratios": {k: (rec[k] / real[k]).cpu().tolist() for k in real},
                       "margin_ratio": (margin(lq) / m_real).cpu().tolist(), "margin_real": m_real.cpu().tolist(),
                       "kl_bits": kl.cpu().tolist(), "top1": (lq.argmax(-1) == top).float().cpu().tolist()}
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
        native.free(nat.dev)
        print(task, "done", flush=True)
    table()


def table():
    rows = [json.loads(x) for x in (DATA / "shrink.jsonl").read_text().splitlines()]
    keys = list(rows[0]["ratios"])
    print(f"{len({r['task'] for r in rows})} texts, {sum(r['T'] for r in rows if r['condition'] == CONDITIONS[0])} tokens; "
          "medians over tokens of masked / real")
    print(f"{'':22s}" + "".join(f"{c[:24]:>26s}" for c in CONDITIONS))
    for k in keys:
        print(f"{k:22s}" + "".join(f"{np.median(np.concatenate([r['ratios'][k] for r in rows if r['condition'] == c])):26.3f}"
                                   for c in CONDITIONS))
    for k, lab in (("margin_ratio", "logit margin"), ("kl_bits", "KL bits (median)"), ("top1", "top-1 agreement (mean)")):
        f = np.mean if k == "top1" else np.median
        print(f"{lab:22s}" + "".join(f"{f(np.concatenate([r[k] for r in rows if r['condition'] == c])):26.3f}" for c in CONDITIONS))
    print(f"{'KL bits (mean)':22s}" + "".join(f"{np.mean(np.concatenate([r['kl_bits'] for r in rows if r['condition'] == c])):26.3f}"
                                              for c in CONDITIONS))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "table":
        table()
    else:
        main(*(int(a) for a in sys.argv[1:]))
