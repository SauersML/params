"""Does the circuit input carry the answer? (#2951 oracledata; the oracle push's rule "every input must carry information
that varies by context; check that before training".) A plain classifier (gradient-boosted trees) reads each question's
named items from the circuit record (oq1) as numbers and predicts the measured class, trained on the training split and
scored on the balanced test split, under load.py's arms: correct (the circuit's numbers), names (layer, stage, unit kind
and position only), mismatch (another context's numbers). Beside it the constant and name-prior baselines of the
manifest. An oracle reading the circuit should reach at least the correct arm's accuracy; if correct is no better than
names, the circuit's numbers do not carry the answer and training on them cannot teach reading.

  readable.py DIR [--kinds zero,steer,set,sign]
  readable.py PAIRS_DIR --pairs      (paired questions: both sides' numbers)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # noqa: E402

NUM = ("value", "value_over_max", "lost", "restored", "direct")


def feats(row: dict, c: dict | None, n_layers: int = 36) -> list[float]:
    """Numbers of the question's items: per item kind, its units' layer stats and, when the arm has them, the sums and
    extremes of each number; the steer/zero op is the kind itself (one model per kind)."""
    units = {u["name"]: u for u in (c["units"] if c else [])}
    heads = {h["name"]: h for h in (c["heads"] if c else [])}
    groups = {g["id"]: g for g in (c["groups"] if c else [])}
    T = len(row["ctx"])
    us, hs = [], []
    for n in row["units"]:
        if n in groups:
            us += [units[m] for m in groups[n]["members"] if m in units]
        elif n in heads:
            hs.append(heads[n])
        elif n in units:
            us.append(units[n])
    f = [len(row["units"]), len(us), len(hs)]
    lay = [u["layer"] / n_layers for u in us] or [0.0]
    f += [min(lay), max(lay), sum(lay) / len(lay)]
    f += [sum(u.get("stage") == s for u in us) for s in ("source", "readout")]
    f += [sum((T - 1 - u["pos"]) == 0 for u in us), sum(u.get("pole", 0) > 0 for u in us)]
    for k in NUM:
        v = [u[k] for u in us if u.get(k) is not None]
        f += [sum(v), max(v, key=abs) if v else 0.0, sum(abs(x) for x in v), float(bool(v))]
    hl = [h["layer"] / n_layers for h in hs] or [0.0]
    f += [min(hl), max(hl)]
    for k in ("lost", "restored"):
        v = [h[k] for h in hs if h.get(k) is not None]
        f += [sum(v), max(v, key=abs) if v else 0.0]
    att = [a_[1] for h in hs for a_ in (h.get("attends") or [])[:1]]
    f += [sum(att), float(bool(att))]
    return [float(x) for x in f]


def pair_side(q: dict, side: str, c: dict | None) -> dict:
    """A paired question's side as a one-text row for feats: the item at that text's positions."""
    units = q["units_" + side]
    name = q["key"] if q["unit_kind"] == "group" else units[0]
    if c is not None and q["unit_kind"] == "group":
        c = {**c, "groups": [g for g in c["groups"] if g["id"] != q["key"]] + [{"id": q["key"], "members": units}]}
    return {"units": [name], "ctx": [0] * len(c["tokens"]) if c else [0]}, c


def pairs_main(a, HistGradientBoostingClassifier) -> None:
    """Paired questions: the same trees on both sides' numbers (A's, B's, their difference); names alone are at chance
    by construction (the same name on both sides, balanced classes)."""
    cs = load.circuits(a.dir)
    tr, te = load.rows(a.dir, "train"), load.rows(a.dir, "test")
    pool = sorted(cs)
    out = {}

    def X(q, arm):
        f = []
        for side in ("a", "b"):
            c = cs[q["circuit_" + side]]
            if arm == "names":
                c = load.arm({"circuit_id": q["circuit_" + side], "id": q["id"] + side}, cs, "names")
            elif arm == "mismatch":
                c = load.arm({"circuit_id": q["circuit_" + side], "id": q["id"] + side}, cs, "mismatch", pool)
            row, c = pair_side(q, side, c)
            f.append(feats(row, c))
        return f[0] + f[1] + [x - y for x, y in zip(f[0], f[1])]
    for kind in sorted({q["kind"] for q in te}):
        rtr = [q for q in tr if q["kind"] == kind]
        rte = [q for q in te if q["kind"] == kind]
        res = {"n_train": len(rtr), "n_test": len(rte), "chance": round(1 / len({q["gold"] for q in rte}), 3)}
        for arm in ("correct", "names", "mismatch"):
            m = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(
                [X(q, arm) for q in rtr], [q["gold"] for q in rtr])
            pred = m.predict([X(q, arm) for q in rte])
            res[arm] = round(sum(p == q["gold"] for p, q in zip(pred, rte)) / len(rte), 3)
        out[kind] = res
        print(kind, json.dumps(res), flush=True)
    (a.dir / "readable.json").write_text(json.dumps(out, indent=1) + "\n")


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--kinds", default="zero,steer,set,sign")
    ap.add_argument("--pairs", action="store_true", help="DIR is a pairs.py build (pair1)")
    a = ap.parse_args()
    if a.pairs:
        return pairs_main(a, HistGradientBoostingClassifier)
    cs = load.circuits(a.dir)
    tr, te = load.rows(a.dir, "train"), load.rows(a.dir, "test")
    man = json.loads((a.dir / "manifest.json").read_text())
    out = {}
    for kind in a.kinds.split(","):
        rtr = [r for r in tr if r["kind"] == kind]
        rte = [r for r in te if r["kind"] == kind]
        if len({r["cls"] for r in rtr}) < 2 or not rte:
            continue
        res = {"n_train": len(rtr), "n_test": len(rte)}
        for arm in ("correct", "names", "mismatch"):
            ptr = sorted({r["circuit_id"] for r in rtr})
            pte = sorted({r["circuit_id"] for r in rte})
            X = [feats(r, load.arm(r, cs, arm, ptr)) for r in rtr]
            Xt = [feats(r, load.arm(r, cs, arm, pte)) for r in rte]
            m = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(X, [r["cls"] for r in rtr])
            pred = m.predict(Xt)
            res[arm] = round(sum(p == r["cls"] for p, r in zip(pred, rte)) / len(rte), 3)
            tags = collections.defaultdict(list)
            for p, r in zip(pred, rte):
                for t in r["tags"]:
                    tags[t].append(p == r["cls"])
            res[arm + "_by_tag"] = {t: [round(sum(v) / len(v), 3), len(v)] for t, v in tags.items()}
        b = man["baselines"]["test"].get(f"{kind}|all", {})
        res.update({"constant": b.get("constant"), "name_prior": b.get("name_prior"),
                    "chance": round(1 / len({r["cls"] for r in rte}), 3)})
        out[kind] = res
        print(kind, json.dumps(res), flush=True)
    (a.dir / "readable.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
