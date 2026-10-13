"""Do vpd4l's copy failures follow the copy mechanism's realization? (#2951 graph oracle; Huang et al. 2025: a model's
failures on new inputs predicted from whether its mechanism is realized on them, against its confidence and a probe.)

Positions: in train texts outside every set this directory used (the held-out and non-copy texts, the falsification
set's mistakes, the discovery rows), every position t >= 30 where the token after the latest earlier occurrence of the
current token is the text's next token (a copy would be right), at most PER_TEXT per text, until N. Label: vpd4l's
likeliest token at t is that token (the copy made) or another (a failure). Every quantity is computed on the text cut
after t (VPD's importance network reads the whole sequence it is given: never the answer).
Realization of the copy mechanism (copy_template's induction): layer 1's head-1 attention a -> o (the previous-token
step), layer 2's head-4 attention t -> a (the match) and its rank among t's attended positions, VPD's causal importance
of h.1.attn.q_proj#316 at a, h.1.attn.k_proj#329 and #119 at o, h.0.mlp.c_fc#53 and h.1.mlp.c_fc#2103 at a; context:
log2 of the copy distance t - a and the number of earlier occurrences.
Baselines: confidence (the model's top probability at t), and linear probes on the residual stream at t entering layer
2 (before the match is read) and at the end (the prediction's own input: an upper reference).
AUC for predicting a failure, by 5-fold cross-validation grouped by text (logistic regression, standardized).

  huang.py build [K P]  -> DATA/huang_positions_K.npz (GEN_OUT when set): shard K of P (the texts whose index is K mod P)
  huang.py report
"""
import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
N = int(os.environ.get("HUANG_N", 3000))
PER_TEXT = 3
NAMED = [("h.1.attn.q_proj", 316, "a"), ("h.1.attn.k_proj", 329, "o"), ("h.1.attn.k_proj", 119, "o"),
         ("h.0.mlp.c_fc", 53, "a"), ("h.1.mlp.c_fc", 2103, "a")]
REAL = ["att1_a_to_o", "att2_t_to_a", "att2_rank_a"] + [f"ci_{s.split('.')[-1]}#{c}@{r}" for s, c, r in NAMED] + \
       ["log2_distance", "occurrences"]


def build(k: int = 0, P: int = 1):
    import json

    import native
    import torch

    nat = native.Native()
    tg = nat.target
    held = json.loads((DATA / "copy_heldout.json").read_text())
    fal = json.loads((DATA / "falsify_texts.json").read_text())
    used = {x["task"] for x in held["copy"] + held["noncopy"]} | set(held.get("scanned", []))
    used |= {e["task"] for e in fal["texts"] + fal["mistakes"]}
    used |= {r["task"] for r in pickle.load(open(DATA / "rows.pkl", "rb"))}
    rng = np.random.default_rng(2951)
    feats, conf, mid, end, label, group = [], [], [], [], [], []
    for gi, p in enumerate(native.tasks("train")):
        if gi % P != k or p.stem in used or len(label) >= N // P:
            continue
        ids, _ = native.text(p)
        cand = [t for t in range(30, len(ids) - 1)
                if (occ := [j for j in range(t) if ids[j] == ids[t]]) and occ[-1] + 1 < t and ids[occ[-1] + 1] == ids[t + 1]]
        if not cand:
            continue
        for t in sorted(rng.choice(cand, size=min(PER_TEXT, len(cand)), replace=False).tolist()):
            occ = [j for j in range(t) if ids[j] == ids[t]]
            o, a = occ[-1], occ[-1] + 1
            x = ids[:t + 1]
            with torch.no_grad():
                kv = tg.keys_values(torch.tensor([x], device=nat.dev))
                q1, k1 = kv[1][0][0], kv[1][1][0]
                q2, k2 = kv[2][0][0], kv[2][1][0]
                w1 = ((q1[1, a] @ k1[1, :a + 1].T) / math.sqrt(tg.hd)).softmax(-1)  # layer 1, head 1, from a
                w2 = ((q2[4, t] @ k2[4, :t + 1].T) / math.sqrt(tg.hd)).softmax(-1)  # layer 2, head 4, from t
                lp = nat.reference([x], [t])[0, 0]
                ci = nat.importance(x)
                h_end = tg.hidden(torch.tensor([x], device=nat.dev))[0, t] if hasattr(tg, "hidden") else None
            row = [float(w1[o]), float(w2[a]), float((w2 > w2[a]).sum())]
            row += [float(ci[s][o if r == "o" else a, c]) for s, c, r in NAMED]
            row += [math.log2(t - a), float(len(occ))]
            feats.append(row)
            conf.append(float(lp.exp().max()))
            mid.append(kv[2][3][0, t].float().cpu().numpy())
            end.append(h_end.float().cpu().numpy() if h_end is not None else np.zeros(1, dtype=np.float32))
            label.append(int(lp.argmax()) == ids[t + 1])
            group.append(gi)
        if len(label) % 200 < PER_TEXT:
            print(f"{len(label)} positions, failures {1 - np.mean(label):.2f}", flush=True)
        native.free(nat.dev)
    out = Path(os.environ.get("GEN_OUT", DATA)) / f"huang_positions_{k}{os.environ.get('GEN_TAG', '')}.npz"
    np.savez_compressed(out, real=np.array(feats, dtype=np.float32), conf=np.array(conf, dtype=np.float32),
                        mid=np.array(mid, dtype=np.float32), end=np.array(end, dtype=np.float32),
                        success=np.array(label, dtype=bool), group=np.array(group))
    print(f"wrote {out}: {len(label)} positions, failures {1 - np.mean(label):.3f}", flush=True)


def report():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import templates

    paths = {p.name: p for p in templates.run_outputs("huang_positions*.npz") if p.exists()}  # every shard once
    parts = [np.load(p) for p in paths.values()]
    d = {key: np.concatenate([x[key] for x in parts]) for key in parts[0].files}
    fail = ~d["success"]
    print(f"{len(fail)} positions over {len(set(d['group'].tolist()))} texts; failures {fail.mean():.3f}")
    sets = {"realization": d["real"], "confidence": d["conf"][:, None], "realization + confidence":
            np.hstack([d["real"], d["conf"][:, None]]), "probe, layer-2 input": d["mid"], "probe, final residual": d["end"],
            "probe layer-2 input + realization": np.hstack([d["mid"], d["real"]])}
    for name, X in sets.items():
        if X.shape[1] <= 1 and name != "confidence":
            continue
        pred = np.zeros(len(fail))
        for tr, te in GroupKFold(5).split(X, fail, d["group"]):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0 if X.shape[1] < 64 else 0.05, max_iter=4000))
            m.fit(X[tr], fail[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        print(f"   {name:36s} AUC {roc_auc_score(fail, pred):.3f}")
    print("   single realization features (AUC of each alone, sign chosen on all positions):")
    for j, nm in enumerate(REAL):
        a = roc_auc_score(fail, d["real"][:, j])
        print(f"      {nm:24s} {max(a, 1 - a):.3f} ({'higher' if a >= 0.5 else 'lower'} on failures); "
              f"success mean {d['real'][~fail, j].mean():.3f}, failure mean {d['real'][fail, j].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "report"))
    ap.add_argument("args", nargs="*", type=int)
    a = ap.parse_args()
    build(*a.args) if a.cmd == "build" else report()
