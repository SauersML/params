"""Identify each task's answer tokens from the release: their circuit was bisected to reach the target loss on their
answer tokens, so on their own pairs the released circuit's loss is lowest for the right (token, alternative) pair.

For each task: last-position logits of csp_yolo2 (full model and released circuit, released means) on the 16 released
pairs; every candidate pair from the full model's top tokens is scored as "binary" (pos -> a, neg -> b) and as
"pos_vs_rest" (pos -> a, neg -> anything but a). Prints the best candidates by circuit loss (per sequence).

python answers.py [--device mps]
"""
import argparse
import json

import torch

import check_release as CR
import nodes as N
import tasks as T

ROOT = N.ROOT


@torch.no_grad()
def last_logits(model, rows, masks, bs=4):
    out = []
    for i in range(0, len(rows), bs):
        x, last = N.pad(rows[i:i + bs], model.device)
        out.append(model.logits(x, last, masks).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    model = N.GaoModel("csp_yolo2", a.device, means=f"{ROOT}/models/csp_yolo2/mean_baseline_pretrain.pt")
    P = json.load(open(f"{ROOT}/released/prunes_csp_yolo2_prune_v4.json"))
    S = json.load(open(f"{ROOT}/released/samples_csp_yolo2_prune_v4.json"))
    res = {}
    for t in T.TASKS:
        rel = P.get(CR.task_dir(t))
        if not rel or "nodes" not in rel:
            continue
        rows = S[CR.task_dir(t)]["tokens"]
        n = len(rows) // 2
        strip = [r[:max(i for i, x in enumerate(r) if x != 0) + 1] for r in rows]
        pos, neg = strip[:n], strip[n:]
        circ = CR.masks_for(model, rel["nodes"])
        F = {"pos": last_logits(model, pos, None), "neg": last_logits(model, neg, None)}
        C = {"pos": last_logits(model, pos, circ), "neg": last_logits(model, neg, circ)}
        cand = set(F["pos"].mean(0).topk(a.top).indices.tolist()) | set(F["neg"].mean(0).topk(a.top).indices.tolist())
        cand |= set(C["pos"].mean(0).topk(a.top).indices.tolist()) | set(C["neg"].mean(0).topk(a.top).indices.tolist())
        cand = sorted(cand)
        scored = []
        for L, tag in ((C, "circuit"), (F, "full")):
            lp, ln = L["pos"].log_softmax(1), L["neg"].log_softmax(1)
            for x in cand:
                for y in cand:
                    if x == y:
                        continue
                    two_p = torch.stack([L["pos"][:, x], L["pos"][:, y]], 1).log_softmax(1)[:, 0]
                    two_n = torch.stack([L["neg"][:, x], L["neg"][:, y]], 1).log_softmax(1)[:, 1]
                    scored.append((tag, "binary", x, y, float(-(two_p + two_n).mean() / 2)))
                pr = -(lp[:, x] + torch.log1p(-ln[:, x].exp().clamp(max=1 - 1e-6))).mean() / 2
                scored.append((tag, "pos_vs_rest", x, None, float(pr)))
        circ_s = sorted([s for s in scored if s[0] == "circuit"], key=lambda s: s[4])[:6]
        full_of = {(s[1], s[2], s[3]): s[4] for s in scored if s[0] == "full"}
        dec = lambda i: None if i is None else model.enc.decode([i])
        best = [{"kind": k, "tok": dec(x), "alt": dec(y), "tok_id": x, "alt_id": y, "circuit": round(l, 4),
                 "full": round(full_of[(k, x, y)], 4)} for _, k, x, y, l in circ_s]
        res[t] = best
        print(t, json.dumps(best[:4]), flush=True)
    json.dump(res, open(f"{ROOT}/released/answers_yolo2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
