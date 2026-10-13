"""Which tokens an explanation variable tells apart (#2951 graph oracle): per variable V (variables.parse) and input
position p in I_V, the partition of alternative tokens at p by the prediction the model makes when only V's group G_V
receives them. The writer names the property V tracks from it.

Alternatives: the model's TOP likeliest tokens at p (its prediction from the tokens before p, the text's own token
left out) and SAMPLED more drawn from that prediction without replacement (keyed by the text's seed and p). For each
alternative z, x'_z is the text with z at p and q_z the model's prediction at the targets on the text with G_V's
activations set to their values on x'_z, everything else computing on the text (variables.Scorer's patched run). Per
alternative, in bits summed over the targets: the model's own move D0_z = KL(model(x) || model(x'_z)), the patched
move m_z = KL(model(x) || q_z), the distance left to the changed text's prediction r_z = KL(model(x'_z) || q_z), the
share of z's change G_V carries f_z = clip(1 - r_z / D0_z, 0, 1) (variables.py's on-target f for that draw; None when
D0_z is within the resolution) and q_z's likeliest token at the last target.

The partition: z is alike the text's token (V does not tell them apart) when patching V's carrier from x'_z leaves the
prediction where it is in the scorer's own terms (variables.py's same-class and off-target test, on f's scale since
the lead's 10-13): f_same_z = clip(1 - m_z / D0_z, 0, 1), the share of z's change kept out, at least GOAL, or when the
model itself does not tell them apart (D0_z within d_res, the runs' float32 resolution: twice the largest disagreement
between runs that agree in exact arithmetic); otherwise z changes V's value. (Taken first as m_z <= max(r_z, d_res),
nearer model(x) than model(x'_z), the classes failed variables.py's same-class test: on text6011 at 232 s_same 0.15.
s_same_z = clip(ln(D0_z / m_z) / ln(D0_z / D_floor), 0, 1), D_floor = KL(model(x) || the model on x'_z with every
subcomponent at the targets from x), the log scale the rule used until 10-13, stays in the rows.) The alike tokens with the
text's form V's equivalence class at p; the others are grouped by the token their q_z predicts (where V's new value
sends the prediction). The partition is the variable's value map ("values": class 0 the text's own, each class its
tokens, their ids and the token its members' patched runs predict).

  python method/oracle/var_classes.py TASK VARIABLES.json [--top 32] [--sampled 32] [--out FILE]
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import variables  # noqa: E402

TOP = 32  # the model's likeliest alternatives per position
SAMPLED = 32  # alternatives sampled from the model's prediction per position


def alternatives(nat: native.Native, ids: list[int], p: int, seed: int, top: int = TOP, sampled: int = SAMPLED) -> list[int]:
    """The model's top likeliest tokens at p (from the tokens before p, the text's own left out), then sampled more
    from that prediction without replacement, keyed by seed and p."""
    if p == 0:  # no prefix to predict from: every token alike (the first ones then drawn like the sampled ones)
        lp = torch.zeros(nat.reference([ids[:1]], [0]).shape[-1], dtype=torch.float64)
        first = []
        sampled += top
    else:
        lp = nat.reference([ids[:p]], [p - 1])[0, 0].cpu().double()  # MPS has no float64
        first = [int(z) for z in lp.topk(top + 1).indices.tolist() if z != ids[p]][:top]
    w = lp.exp()
    w[[ids[p]] + first] = 0.0
    gen = torch.Generator().manual_seed(int(hashlib.sha256(repr(("classes", seed, p)).encode()).hexdigest()[:15], 16))
    return first + torch.multinomial(w, min(sampled, int((w > 0).sum())), replacement=False, generator=gen).tolist()


def classes(sc: variables.Scorer, tx, v: dict, p: int, model: str = "vpd4l", top: int = TOP, sampled: int = SAMPLED) -> dict:
    """The partition of alternative tokens at p by v's group (module docstring): {"position", "token", "d_res",
    "equivalent": the class of the text's token (token strings), "changed": [{"predicts", "tokens"}], "values": the
    value map [{"id", "tokens", "token_ids", "predicts"}], class 0 the text's own, "alternatives": per alternative its
    numbers and class}."""
    import components

    nat = sc.nat
    zs = alternatives(nat, tx.ids, p, int(tx.seed), top, sampled)
    mine = sc._text(tx.ids, tx.targets)
    base = mine["lq"]
    kl = lambda a, b: float(carry.kl(a[None], b[None])[0])  # noqa: E731
    own = sc._patched(mine, [(mine, v["group"]), (mine, [])])
    noise = max([mine["res"]] + [abs(kl(base, q)) for q in own])
    q_z, ref_z, floor_z = [], [], []
    for s0 in range(0, len(zs), sc.rows):  # a batch's changed texts at a time (each holds every activation)
        texts = []
        for z in zs[s0:s0 + sc.rows]:
            x = list(tx.ids)
            x[p] = z
            texts.append(sc._text(x, tx.targets))
            noise = max(noise, texts[-1]["res"])
        q_z += list(sc._patched(mine, [(w, v["group"]) for w in texts]))
        ref_z += [w["lq"] for w in texts]
        floor_z += [sc._patched(w, [(mine, None)])[0] for w in texts]  # on x': every subcomponent at the targets from x
        del texts
        gc.collect()  # the runs' tensors sit in reference cycles (var_teacher.release)
        native.free(nat.dev)
    res = 2 * noise
    pred_ids = [int(q[-1].argmax()) for q in [base] + q_z]
    names = components.token_strings(model, [tx.ids[p]] + zs + pred_ids)
    own_tok, alt_tok, pred = names[0], names[1:1 + len(zs)], names[1 + len(zs):]
    rows = []
    for i, z in enumerate(zs):
        d0, m, left, fl = kl(base, ref_z[i]), kl(base, q_z[i]), kl(ref_z[i], q_z[i]), kl(base, floor_z[i])
        rows.append({"token": alt_tok[i], "id": int(z), "p_model": None, "d0": d0, "move": m, "left": left,
                     "f": min(1.0, max(0.0, 1 - left / d0)) if d0 > res else None,
                     "s_same": goal_eval._fid(d0, max(m, 0.0), max(fl, res)) if d0 > res else None,
                     "f_same": min(1.0, max(0.0, 1 - max(m, 0.0) / d0)) if d0 > res else None,
                     "predicts": pred[1 + i], "predicts_id": pred_ids[1 + i]})
    lp = nat.reference([tx.ids[:p]], [p - 1])[0, 0]
    changed = {}
    for r in rows:
        r["p_model"] = float(lp[r["id"]].exp())
        r["alike"] = r["f_same"] is None or r["f_same"] >= goal_eval.GOAL
        if not r["alike"]:
            changed.setdefault(r["predicts"], []).append(r)
    alike = [r for r in rows if r["alike"]]
    moved = sorted(changed.items(), key=lambda kv: -len(kv[1]))
    values = [{"id": 0, "tokens": [own_tok] + [r["token"] for r in alike], "token_ids": [int(tx.ids[p])] + [r["id"] for r in alike],
               "predicts": pred[0], "predicts_id": pred_ids[0]}]
    values += [{"id": k + 1, "tokens": [r["token"] for r in rs], "token_ids": [r["id"] for r in rs], "predicts": tok,
                "predicts_id": rs[0]["predicts_id"]} for k, (tok, rs) in enumerate(moved)]
    for c in values:
        for i in c["token_ids"][1 if c["id"] == 0 else 0:]:
            next(r for r in rows if r["id"] == i)["class"] = c["id"]
    return {"position": p, "token": own_tok, "predicts": pred[0], "d_res": res,
            "equivalent": values[0]["tokens"], "changed": [{"predicts": c["predicts"], "tokens": c["tokens"]} for c in values[1:]],
            "values": values, "alternatives": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", help="a vpd4l text id (native.TEXTS/vpd4l/<task>.json)")
    ap.add_argument("variables", type=Path, help='JSON: {"variables": [...]} (variables.parse())')
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--sampled", type=int, default=SAMPLED)
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ge = goal_eval.GoalEval(native.Native(args.device))
    task = json.loads((native.TEXTS / "vpd4l" / f"{args.task}.json").read_text())
    ids, targets = native.text(native.TEXTS / "vpd4l" / f"{args.task}.json")
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    sc = variables.Scorer(ge)
    out = []
    for v in variables.parse(json.loads(args.variables.read_text())["variables"], ge.nat.C):
        for p in v["inputs"]:
            got = classes(sc, tx, v, p, task["model"], args.top, args.sampled)
            out.append({"variable": v["name"], **got})
            print(f"{v['name']} at {p} ({got['token']!r}, predicts {got['predicts']!r}): alike {got['equivalent']}")
            for c in got["changed"]:
                print(f"  -> {c['predicts']!r}: {c['tokens']}")
    if args.out:
        args.out.write_text(json.dumps({"task": args.task, "classes": out}, indent=1))


if __name__ == "__main__":
    main()
