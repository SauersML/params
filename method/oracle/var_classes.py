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

The partition: z is alike the text's token (V does not tell them apart) when q_z is no farther from model(x) than
from model(x'_z), m_z <= max(r_z, d_res), d_res the runs' float32 resolution (variables.Scorer's: twice the largest
disagreement between runs that agree in exact arithmetic); otherwise z changes V's value. The alike tokens with the
text's form V's equivalence class at p; the others are grouped by the token their q_z predicts (where V's new value
sends the prediction).

  python method/oracle/var_classes.py TASK VARIABLES.json [--top 32] [--sampled 32] [--out FILE]
"""

import argparse
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
    lp = nat.reference([ids[:p]], [p - 1])[0, 0].cpu().double()  # MPS has no float64
    first = [int(z) for z in lp.topk(top + 1).indices.tolist() if z != ids[p]][:top]
    w = lp.exp()
    w[[ids[p]] + first] = 0.0
    gen = torch.Generator().manual_seed(int(hashlib.sha256(repr(("classes", seed, p)).encode()).hexdigest()[:15], 16))
    return first + torch.multinomial(w, min(sampled, int((w > 0).sum())), replacement=False, generator=gen).tolist()


def classes(sc: variables.Scorer, tx, v: dict, p: int, model: str = "vpd4l", top: int = TOP, sampled: int = SAMPLED) -> dict:
    """The partition of alternative tokens at p by v's group (module docstring): {"position", "token", "d_res",
    "equivalent": the class of the text's token (token strings), "changed": [{"predicts", "tokens"}], "alternatives":
    per alternative its numbers}."""
    import components

    nat = sc.nat
    zs = alternatives(nat, tx.ids, p, int(tx.seed), top, sampled)
    mine = sc._text(tx.ids, tx.targets)
    base = mine["lq"]
    kl = lambda a, b: float(carry.kl(a[None], b[None])[0])  # noqa: E731
    own = sc._patched(mine, [(mine["act"], v["group"]), (mine["act"], [])])
    noise = max([mine["res"]] + [abs(kl(base, q)) for q in own])
    q_z, ref_z = [], []
    for s0 in range(0, len(zs), sc.rows):  # a batch's changed texts at a time (each holds every activation)
        texts = []
        for z in zs[s0:s0 + sc.rows]:
            x = list(tx.ids)
            x[p] = z
            texts.append(sc._text(x, tx.targets))
            noise = max(noise, texts[-1]["res"])
        q_z += list(sc._patched(mine, [(w["act"], v["group"]) for w in texts]))
        ref_z += [w["lq"] for w in texts]
        del texts
        native.free(nat.dev)
    res = 2 * noise
    names = components.token_strings(model, [tx.ids[p]] + zs + [int(i) for q in [base] + q_z for i in q[-1].topk(1).indices])
    own_tok, alt_tok, pred = names[0], names[1:1 + len(zs)], names[1 + len(zs):]
    rows = []
    for i, z in enumerate(zs):
        d0, m, left = kl(base, ref_z[i]), kl(base, q_z[i]), kl(ref_z[i], q_z[i])
        rows.append({"token": alt_tok[i], "id": int(z), "p_model": None, "d0": d0, "move": m, "left": left,
                     "f": min(1.0, max(0.0, 1 - left / d0)) if d0 > res else None, "predicts": pred[1 + i]})
    lp = nat.reference([tx.ids[:p]], [p - 1])[0, 0]
    changed = {}
    for r in rows:
        r["p_model"] = float(lp[r["id"]].exp())
        r["alike"] = r["move"] <= max(r["left"], res)
        if not r["alike"]:
            changed.setdefault(r["predicts"], []).append(r["token"])
    return {"position": p, "token": own_tok, "predicts": pred[0], "d_res": res,
            "equivalent": [own_tok] + [r["token"] for r in rows if r["alike"]],
            "changed": sorted(({"predicts": k, "tokens": v} for k, v in changed.items()), key=lambda c: -len(c["tokens"])),
            "alternatives": rows}


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
