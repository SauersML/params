"""Causal maps and reader metrics from the saved interchange runs (#2951 WorkspaceBench).

Per cell (layer, position) and counterfactual j, q_j(s) is the patched run's probability of answer
string s. The hypothesis "the cell holds variable v" predicts answer s_v(j); its likelihood is
prod_j q_j(s_v(j)) and, with equal priors, the posterior over variables is that product normalized.
A cell is assigned its most probable variable; `mass` is the mean over counterfactuals of the
probability the patched run gives the assigned variable's answer. Cells before the first position
where the prompts differ are identical in every run and hold nothing (`none`).

One caveat is built into interchange: a cell holding both operands of the first operation predicts
exactly what a cell holding their result predicts, so `I` reads "the intermediate, or both of its
operands together".
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import readers

GROUPS = ["none", "operand", "digit", "remaining", "I", "J", "answer"]


def group(item, h):
    """Display group of hypothesis h: an operand of the first operation, a single digit, the operand
    the first operation does not use, the intermediate, the answer, nothing."""
    if h in ("none", "answer", "I"):
        return h
    if h == "J":                      # mulmul / mulmid: the second product
        return "J"
    if "." in h:
        return "digit"
    first = {"a", "b"} if "a" in item["ops"] else {"x", "y"}
    return "operand" if h in first else "remaining"


def posterior(rec):
    """[L+1, T, H] log-likelihoods, mass of each hypothesis, and the explained mass [L+1, T]."""
    lp = rec["lp"]
    H = len(rec["hyps"])
    ll = sum(lp[j][..., rec["hyp_answer"][j]] for j in range(len(lp)))
    mass = sum(lp[j][..., rec["hyp_answer"][j]].exp() for j in range(len(lp))) / len(lp)
    explained = sum(lp[j].exp().sum(-1) for j in range(len(lp))) / len(lp)
    post = torch.softmax(ll, -1)
    assert post.shape[-1] == H
    return post, mass, explained


def load_all(d):
    return [torch.load(p, weights_only=False) for p in sorted((d / "patch").glob("*.pt"))]


def reader_rates(cells_of, named_of, cross_of):
    """Naming rates by what the cells hold: {group: cells, named, donor_named}, recall, precision."""
    cells = defaultdict(lambda: np.zeros(3))
    for key, n, c in zip(cells_of, named_of, cross_of):
        cells[key] += [1, n, c]
    rates = {k: {"cells": int(v[0]), "named": v[1] / v[0], "donor_named": v[2] / v[0]} for k, v in cells.items()}
    held = cells.get("I", np.zeros(3))
    total = sum(v[1] for v in cells.values())
    return rates, (held[1] / held[0] if held[0] else None), (held[1] / total if total else None)


def arith_summary(recs, tok):
    fams = defaultdict(list)
    for r in recs:
        if r["kind"] == "arith":
            fams[r["family"]].append(r)
    out = {}
    for fam, rs in fams.items():
        readers_of = {"lens": {r["name"]: readers.number_names(tok, r["inter"]["I"]) for r in rs}}
        digit = {r["name"]: tok(str(int(r["inter"]["I"])), add_special_tokens=False)["input_ids"][0] for r in rs}
        readers_of["lens_words"] = {k: v - {digit[k]} for k, v in readers_of["lens"].items()}
        keys, named_by, cross_by, maps = [], defaultdict(list), defaultdict(list), []
        for r in rs:
            post, mass, expl = posterior(r)
            g_of = [group(r, h) for h in r["hyps"]]
            assign = post.argmax(-1)                  # [L+1, T]
            lens = r["lens"].numpy()
            seg = ["expr" if not ro.startswith(("start", "end")) else ("before" if ro.startswith("start") else "after") for ro in r["roles"]]
            Lp1, T = assign.shape
            keys += [g_of[assign[l, p]] if g_of[assign[l, p]] != "none" else f"none ({seg[p]})" for l in range(Lp1) for p in range(T)]
            for rd, names in readers_of.items():
                own = names[r["name"]]
                donors = [v for k, v in names.items() if k != r["name"] and v and not (v & own)]
                named = np.isin(lens, list(own)).any(-1) if own else np.zeros(lens.shape[:2], bool)
                cross = np.mean([np.isin(lens, list(v)).any(-1) for v in donors], 0) if donors else np.zeros(named.shape)
                named_by[rd] += named.reshape(-1).tolist()
                cross_by[rd] += cross.reshape(-1).tolist()
                if rd == "lens":
                    m_named, m_cross = named, cross
            maps.append({"name": r["name"], "roles": r["roles"], "tokens": [tok.decode([i]) for i in r["ids"]],
                         "assign": [[g_of[a] for a in row] for row in assign.tolist()],
                         "post_max": post.max(-1).values.numpy().round(3).tolist(),
                         "mass": torch.gather(mass, -1, assign[..., None])[..., 0].numpy().round(3).tolist(),
                         "explained": expl.numpy().round(3).tolist(), "named": m_named.astype(int).tolist(),
                         "cross": np.round(m_cross, 3).tolist()})
        out[fam] = {"items": len(rs), "maps": maps}
        for rd in readers_of:
            rates, recall, precision = reader_rates(keys, named_by[rd], cross_by[rd])
            out[fam][rd] = {"by_held": rates, "recall": recall, "precision": precision}
    return out


def hop_summary(recs, tok):
    rows = []
    hops = [r for r in recs if r["kind"] == "hop"]
    names = {r["name"]: readers.word_names(tok, r["bridge"]) for r in hops}
    cells = defaultdict(lambda: np.zeros(3))
    for r in hops:
        post, mass, expl = posterior(r)
        lens = r["lens"].numpy()
        own = names[r["name"]]
        named = np.isin(lens, list(own)).any(-1) if own else np.zeros(lens.shape[:2], bool)
        donors = [v for k, v in names.items() if v and k != r["name"] and not (v & own)]
        cross = np.mean([np.isin(lens, list(v)).any(-1) for v in donors], 0)
        if own:
            seg = ["slot" if ro == "slot" else "before" if ro.startswith("start") else "after" for ro in r["roles"]]
            for l in range(post.shape[0]):
                for p_, sg in enumerate(seg):
                    held = "carries" if post[l, p_, 1] > post[l, p_, 0] else "does not carry"
                    cells[f"{held} ({sg})"] += [1, named[l, p_], cross[l, p_]]
        rows.append({"name": r["name"], "bridge": r["bridge"], "answer": r["answer"], "roles": r["roles"],
                     "tokens": [tok.decode([i]) for i in r["ids"]],
                     "carry_post": post[..., 1].numpy().round(3).tolist(), "carry_mass": mass[..., 1].numpy().round(3).tolist(),
                     "none_mass": mass[..., 0].numpy().round(3).tolist(), "named": named.astype(int).tolist(),
                     "bridge_one_token": bool(own)})
    by_held = {k: {"cells": int(v[0]), "named": v[1] / v[0], "donor_named": v[2] / v[0]} for k, v in cells.items()}
    return rows, by_held


def main(d, model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    recs = load_all(d)
    hop_rows, hop_held = hop_summary(recs, tok)
    s = {"arith": arith_summary(recs, tok), "multihop": hop_rows, "multihop_by_held": hop_held}
    (d / "summary.json").write_text(json.dumps(s))
    for fam, f in s["arith"].items():
        for rd in ("lens", "lens_words"):
            g = f[rd]
            print(f"\n{fam} {rd}: {f['items']} items; names the intermediate at recall {g['recall']} precision {g['precision']}")
            for k, v in sorted(g["by_held"].items()):
                print(f"  cells holding {k:16s} {v['cells']:6d}  named {v['named']:.3f}  donor named {v['donor_named']:.3f}")
    for k, v in sorted(hop_held.items()):
        print(f"multihop cells that {k:28s} {v['cells']:6d}  lens names the bridge {v['named']:.3f}  another prompt's bridge {v['donor_named']:.3f}")
    for h in s["multihop"]:
        last = np.array(h["carry_post"])[:, -1]
        nm = np.array(h["named"])[:, -1]
        print(f"{h['name']}: last position carries from layer {int(np.argmax(last > 0.5)) - 1 if (last > 0.5).any() else None}; "
              f"lens names {h['bridge']!r} at layers {[l - 1 for l in np.nonzero(nm)[0].tolist()]}")
