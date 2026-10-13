"""Explanation variables for the MNIST circuits: name groups of circuit subcomponents by what activates them, then test
each name by interchange.

Vocabulary: binary stroke features of an image, computed from its ink (pixels > 0.5) with no learning:
  loop_top / loop_bottom / big_loop: an enclosed background region (not touching the border, at least 4 pixels) whose
      centroid row is < 13 / > 15, or whose area is at least 40 pixels;
  vert_left / vert_center / vert_right: a column in x 4-10 / 11-16 / 17-23 with a contiguous ink run of at least 10 rows;
  bar_top / bar_middle / bar_bottom: a row in y 3-10 / 11-17 / 18-25 with a contiguous ink run of at least 8 columns;
  ink_center: at least 6 ink pixels in the central 6 x 6 window;
  wide: the ink's bounding box at least 15 columns wide.

Naming: every subcomponent in the union of the circuits gets the feature and sign that best separate its activation
a_c = h V_c within classes (the AUC of a_c for feature present vs absent, computed per class where both values have at
least 20 of 20,000 training images, averaged over those classes with weights by count): "loop_top" when it is higher
with the feature, "no loop_top" when lower. The within-class AUC keeps a name from being the class in disguise. A
subcomponent whose best within-class AUC is within MIN_SEP of 0.5 stays unnamed. A group is the named circuit
subcomponents of one layer sharing a name (whole-layer block nodes are not grouped).

Groups are formed per circuit (default, --mode circuit: the oracle's setting, one image's circuit and its variables;
the base image is the circuit's own image) or over the union of all circuits (--mode union, tested on 200 base images).

Interchange (the real model, every subcomponent on, remainder live): on a base image x (test images 0-4999, the circuit
pool), the group's activations are replaced by their values on a donor x' (test images 5000-9999) and the rest computes
on the run's own inputs. Donors that differ: f(x') != f(x), drawn from the whole pool (any class). Control donors:
f(x') = f(x), each matched to one differing donor on the number of features that differ from x (the differing donor's
count, all of them other features here), so both conditions change the image equally; the share of donors the model
gives x's class is reported per condition. Per group: movement = mean KL in bits from the real
prediction on x, for each condition, and the ratio; direction: across the 10 classes, the Pearson correlation of the
log-probability change with the concept's prediction log P(y | f = f(x')) - log P(y | f = f(x)) (class frequencies given
the feature, 20,000 training images), averaged over differing donors where the prediction moved by at least 0.01 bits.
Baseline: random groups of the same size from the same layer's circuit subcomponents outside the group (per circuit:
that circuit's; when too few, the layer's subcomponents active on x), same donors. Per concept, the share of circuits
where the named group's ratio exceeds the random groups' is reported.

  ~/mpd-data/venv/bin/python experiments/oracle/mnist/concepts.py --tag i5e-4 --circuits ~/mpd-data/mnist/circuits_X.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODULES, kl_bits, load_decomp, load_mnist  # noqa: E402

FEATURES = (
    "loop_top",
    "loop_bottom",
    "big_loop",
    "vert_left",
    "vert_center",
    "vert_right",
    "bar_top",
    "bar_middle",
    "bar_bottom",
    "ink_center",
    "wide",
)
MIN_SEP = 0.1  # |within-class AUC - 0.5| below this: unnamed
CONCEPT_WORDS = {
    "loop_top": "closed loop at the top",
    "loop_bottom": "closed loop at the bottom",
    "big_loop": "one large closed loop",
    "vert_left": "vertical stroke on the left",
    "vert_center": "vertical stroke in the middle",
    "vert_right": "vertical stroke on the right",
    "bar_top": "horizontal bar at the top",
    "bar_middle": "horizontal stroke across the middle",
    "bar_bottom": "horizontal stroke at the bottom",
    "ink_center": "ink at the center",
    "wide": "wide digit",
}


def _longest_run(v: np.ndarray) -> int:
    best = cur = 0
    for b in v:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def features(img: np.ndarray) -> np.ndarray:
    ink = img.reshape(28, 28) > 0.5
    out = np.zeros(len(FEATURES), dtype=bool)
    lab, n = ndimage.label(~ink)
    for k in range(1, n + 1):
        reg = lab == k
        rows, cols = np.nonzero(reg)
        if rows.min() == 0 or cols.min() == 0 or rows.max() == 27 or cols.max() == 27 or reg.sum() < 4:
            continue
        cy = rows.mean()
        if cy < 13:
            out[0] = True
        if cy > 15:
            out[1] = True
        if reg.sum() >= 40:
            out[2] = True
    colrun = np.array([_longest_run(ink[:, c]) for c in range(28)])
    rowrun = np.array([_longest_run(ink[r, :]) for r in range(28)])
    out[3] = colrun[4:11].max() >= 10
    out[4] = colrun[11:17].max() >= 10
    out[5] = colrun[17:24].max() >= 10
    out[6] = rowrun[3:11].max() >= 8
    out[7] = rowrun[11:18].max() >= 8
    out[8] = rowrun[18:26].max() >= 8
    out[9] = ink[11:17, 11:17].sum() >= 6
    cols = np.nonzero(ink.any(0))[0]
    out[10] = len(cols) > 0 and cols.max() - cols.min() + 1 >= 15
    return out


def feature_matrix(x: torch.Tensor) -> np.ndarray:
    xs = x.numpy()
    return np.stack([features(xs[i]) for i in range(len(xs))])


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_class_auc(a: np.ndarray, f: np.ndarray, y: np.ndarray, min_n: int = 20) -> float:
    tot, wsum = 0.0, 0.0
    for c in range(10):
        m = y == c
        n1, n0 = int(f[m].sum()), int((~f[m]).sum())
        if n1 < min_n or n0 < min_n:
            continue
        w = min(n1, n0)
        tot += w * auc(a[m], f[m])
        wsum += w
    return tot / wsum if wsum > 0 else 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--n-base", type=int, default=200)
    ap.add_argument("--n-donor", type=int, default=24)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--mode", choices=("circuit", "union"), default="circuit")
    args = ap.parse_args()
    torch.set_num_threads(3)
    dec = load_decomp(DATA / f"vpd_{args.tag}.pt")
    cpath = Path(args.circuits).expanduser()
    circ = json.loads(cpath.read_text())["results"]
    xtr, ytr, xte, yte = load_mnist()
    g = np.random.default_rng(0)
    ntr = g.permutation(len(xtr))[:20000]
    fcache = DATA / "features.npz"
    if fcache.exists():
        z = np.load(fcache)
        Ftr, Fte = z["Ftr"], z["Fte"]
    else:
        Ftr, Fte = feature_matrix(xtr[ntr]), feature_matrix(xte)
        np.savez(fcache, Ftr=Ftr, Fte=Fte)
    ytr_n = ytr[ntr].numpy()
    rates = {FEATURES[k]: float(Ftr[:, k].mean()) for k in range(len(FEATURES))}
    # P(y | f) from the training images, add-one smoothed
    p_y_f = np.zeros((len(FEATURES), 2, 10))
    for k in range(len(FEATURES)):
        for v in (0, 1):
            cnt = np.bincount(ytr_n[Ftr[:, k] == bool(v)], minlength=10) + 1.0
            p_y_f[k, v] = cnt / cnt.sum()
    a_tr, _, _ = dec.acts(xtr[ntr])
    a_tr = a_tr.numpy()
    members = sorted(
        {m for r in circ for m in r["members"] if "#" in m}, key=lambda s: (s.split("#")[0], int(s.split("#")[1]))
    )
    name_to_j = {}
    for m in members:
        mod, c = m.split("#")
        name_to_j[m] = dec.offsets[MODULES.index(mod)] + int(c)
    named = {}
    for m in members:
        j = name_to_j[m]
        best = (0.0, None, None, None)
        for k, f in enumerate(FEATURES):
            w = within_class_auc(a_tr[:, j], Ftr[:, k], ytr_n)
            if abs(w - 0.5) > best[0]:
                best = (abs(w - 0.5), f, w, auc(a_tr[:, j], Ftr[:, k]))
        _, f, w, overall = best
        named[m] = {"feature": f, "sign": "+" if w > 0.5 else "-", "auc_within": w, "auc_overall": overall}
    groups: dict[tuple[str, str, str], list[str]] = {}
    for m, v in named.items():
        if abs(v["auc_within"] - 0.5) < MIN_SEP:
            continue
        groups.setdefault((m.split("#")[0], v["feature"], v["sign"]), []).append(m)
    # interchange
    base_idx = g.permutation(5000)[: args.n_base]
    donor_pool = np.arange(5000, len(xte))
    a_te, _, _ = dec.acts(xte)
    T = dec.total
    ones = torch.ones(1, T)
    pred_te = dec.run(xte, ones.expand(len(xte), -1)).argmax(-1).numpy()

    lmove: dict[tuple, np.ndarray] = {}

    def interchange(js: list[int], b: int, donors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """KL bits from the real prediction per donor and the log-probability changes; the centered-logit movement
        (||c(z) - c(z_ref)|| / ||c(z_ref)||) is kept in lmove under (js, b, donors)."""
        n = len(donors)
        sel = torch.zeros(n, T, dtype=torch.bool)
        sel[:, js] = True
        val = a_te[torch.as_tensor(donors)]
        x = xte[b][None].expand(n, -1)
        lg = dec.run(x, ones.expand(n, -1), 1.0, sel, val)
        ref = dec.run(xte[b][None], ones)
        dlogp = (F.log_softmax(lg, -1) - F.log_softmax(ref, -1)).numpy()
        zc, zr = lg - lg.mean(-1, keepdim=True), ref - ref.mean(-1, keepdim=True)
        lmove[(tuple(js), b, tuple(int(v) for v in donors))] = ((zc - zr).norm(dim=-1) / zr.norm(dim=-1)).numpy()
        return kl_bits(ref.expand(n, -1), lg).numpy(), dlogp

    def lm(js: list[int], b: int, donors: np.ndarray) -> float:
        return float(lmove[(tuple(js), b, tuple(int(v) for v in donors))].mean())

    def donors_for(b: int, k: int, gr) -> tuple[np.ndarray, np.ndarray] | None:
        """Differing donors (feature k differs) and control donors (feature k equal) matched one to one on how many
        features differ from the base: each control differs in as many other features as its differing donor does
        in all (nearest count when none is left at that count), so the two conditions change the image equally."""
        fb = Fte[b]
        ham = (Fte[donor_pool] != fb).sum(1)
        is_diff = Fte[donor_pool, k] != fb[k]
        diff = donor_pool[is_diff]
        if len(diff) < args.n_donor or (~is_diff & (ham >= 1)).sum() < args.n_donor:
            return None
        dd = gr.choice(diff, args.n_donor, replace=False)
        left = {h: list(gr.permutation(donor_pool[~is_diff & (ham == h)])) for h in range(1, len(FEATURES))}
        ds = []
        for dn in dd:
            h = int((Fte[dn] != fb).sum())
            for hh in sorted(left, key=lambda v: (abs(v - h), -v)):
                if left[hh]:
                    ds.append(int(left[hh].pop()))
                    break
        return dd, np.array(ds)

    def direction(k: int, b: int, dd: np.ndarray, kd: np.ndarray, dl: np.ndarray) -> list[float]:
        out = []
        for i, dn in enumerate(dd):
            if kd[i] < 0.01:
                continue
            r = np.log(p_y_f[k, int(Fte[dn, k])]) - np.log(p_y_f[k, int(Fte[b, k])])
            out.append(float(np.corrcoef(dl[i], r)[0, 1]))
        return out

    out_groups = []
    if args.mode == "circuit":
        per: dict[tuple[str, str, str], list[dict]] = {}
        for r in circ:
            b = int(r["image"])
            mem = [m for m in r["members"] if "#" in m and abs(named[m]["auc_within"] - 0.5) >= MIN_SEP]
            gx: dict[tuple[str, str, str], list[str]] = {}
            for m in mem:
                gx.setdefault((m.split("#")[0], named[m]["feature"], named[m]["sign"]), []).append(m)
            a_b = a_te[b]
            for key, ms in gx.items():
                mod, f, sign = key
                k = FEATURES.index(f)
                gr = np.random.default_rng(b * 1000 + k * 2 + (sign == "+"))
                dn = donors_for(b, k, gr)
                if dn is None:
                    continue
                dd, ds = dn
                js = [name_to_j[m] for m in ms]
                pool = [name_to_j[m] for m in r["members"] if m.startswith(mod + "#") and m not in ms]
                if len(pool) < len(js):
                    o = dec.offsets[MODULES.index(mod)]
                    sz = dec.sizes[MODULES.index(mod)]
                    pool = [j for j in range(o, o + sz) if j not in js and abs(float(a_b[j])) > 0]
                kd, dl = interchange(js, b, dd)
                ks, _ = interchange(js, b, ds)
                rr = []
                for _ in range(args.n_random):
                    rg = [int(j) for j in gr.choice(pool, size=len(js), replace=False)]
                    rkd, _ = interchange(rg, b, dd)
                    rks, _ = interchange(rg, b, ds)
                    rr.append((float(rkd.mean()), float(rks.mean()), lm(rg, b, dd), lm(rg, b, ds)))
                per.setdefault(key, []).append(
                    {
                        "image": b,
                        "size": len(js),
                        "members": ms,
                        "differ": float(kd.mean()),
                        "control": float(ks.mean()),
                        "random_differ": float(np.mean([x[0] for x in rr])),
                        "random_control": float(np.mean([x[1] for x in rr])),
                        "differ_logit": lm(js, b, dd),
                        "control_logit": lm(js, b, ds),
                        "random_differ_logit": float(np.mean([x[2] for x in rr])),
                        "random_control_logit": float(np.mean([x[3] for x in rr])),
                        "direction": direction(k, b, dd, kd, dl),
                        "differ_same_class": float(np.mean(pred_te[dd] == pred_te[b])),
                        "control_same_class": float(np.mean(pred_te[ds] == pred_te[b])),
                    }
                )
        for (mod, f, sign), rows in sorted(per.items(), key=lambda t: -len(t[1])):
            d_ = np.array([x["differ"] for x in rows])
            c_ = np.array([x["control"] for x in rows])
            rd = np.array([x["random_differ"] for x in rows])
            rc = np.array([x["random_control"] for x in rows])
            dirs = [v for x in rows for v in x["direction"]]
            row = {
                "layer": mod,
                "feature": f,
                "sign": sign,
                "concept": ("" if sign == "+" else "no ") + CONCEPT_WORDS[f],
                "circuits": len(rows),
                "bases": len(rows),
                "size": float(np.mean([x["size"] for x in rows])),
                "move_differ": float(d_.mean()),
                "move_control": float(c_.mean()),
                "ratio": float(d_.mean() / max(c_.mean(), 1e-9)),
                "random_move_differ": float(rd.mean()),
                "random_move_control": float(rc.mean()),
                "random_ratio": float(rd.mean() / max(rc.mean(), 1e-9)),
                "share_beats_random": float(np.mean(d_ / np.maximum(c_, 1e-9) > rd / np.maximum(rc, 1e-9))),
                "move_differ_logit": float(np.mean([x["differ_logit"] for x in rows])),
                "move_control_logit": float(np.mean([x["control_logit"] for x in rows])),
                "random_differ_logit": float(np.mean([x["random_differ_logit"] for x in rows])),
                "random_control_logit": float(np.mean([x["random_control_logit"] for x in rows])),
                "direction_corr": float(np.mean(dirs)) if dirs else None,
                "n_direction": len(dirs),
                "differ_same_class": float(np.mean([x["differ_same_class"] for x in rows])),
                "control_same_class": float(np.mean([x["control_same_class"] for x in rows])),
                "per_circuit": rows,
            }
            out_groups.append(row)
            print(
                f"{mod} {row['concept']!r} circuits={len(rows)} size={row['size']:.1f} differ={row['move_differ']:.4f} "
                f"control={row['move_control']:.4f} ratio={row['ratio']:.2f} rand_ratio={row['random_ratio']:.2f} "
                f"beats_random={row['share_beats_random']:.2f} dir={row['direction_corr']} "
                f"same_class {row['differ_same_class']:.2f}/{row['control_same_class']:.2f} "
                f"logit differ={row['move_differ_logit']:.3f} control={row['move_control_logit']:.3f} "
                f"random {row['random_differ_logit']:.3f}/{row['random_control_logit']:.3f}",
                flush=True,
            )
    active_by_layer = {mod: [name_to_j[m] for m in members if m.startswith(mod)] for mod in MODULES}
    for (mod, f, sign), ms in sorted(groups.items(), key=lambda t: -len(t[1])) if args.mode == "union" else []:
        k = FEATURES.index(f)
        js = [name_to_j[m] for m in ms]
        md, ms_, dirs, used = [], [], [], 0
        rand_md, rand_ms = [], []
        gr = np.random.default_rng(MODULES.index(mod) * 1000 + FEATURES.index(f) * 2 + (sign == "+"))
        rgroups = [
            list(gr.choice(active_by_layer[mod], size=len(js), replace=False))
            for _ in range(args.n_random)
            if len(active_by_layer[mod]) > len(js)
        ]
        for b in base_idx:
            fb = Fte[b]
            dn = donors_for(int(b), k, gr)
            if dn is None:
                continue
            used += 1
            dd, ds = dn
            kd, dl = interchange(js, int(b), dd)
            ks, _ = interchange(js, int(b), ds)
            md.append(kd.mean())
            ms_.append(ks.mean())
            for i, dn in enumerate(dd):
                if kd[i] < 0.01:
                    continue
                r = np.log(p_y_f[k, int(Fte[dn, k])]) - np.log(p_y_f[k, int(fb[k])])
                dirs.append(float(np.corrcoef(dl[i], r)[0, 1]))
            for rg in rgroups:
                rkd, _ = interchange([int(j) for j in rg], int(b), dd)
                rks, _ = interchange([int(j) for j in rg], int(b), ds)
                rand_md.append(rkd.mean())
                rand_ms.append(rks.mean())
        row = {
            "layer": mod,
            "feature": f,
            "sign": sign,
            "concept": ("" if sign == "+" else "no ") + CONCEPT_WORDS[f],
            "size": len(js),
            "members": ms,
            "auc_within": float(np.mean([named[m]["auc_within"] for m in ms])),
            "auc_overall": float(np.mean([named[m]["auc_overall"] for m in ms])),
            "bases": used,
            "move_differ": float(np.mean(md)) if md else None,
            "move_control": float(np.mean(ms_)) if ms_ else None,
            "direction_corr": float(np.mean(dirs)) if dirs else None,
            "n_direction": len(dirs),
            "random_move_differ": float(np.mean(rand_md)) if rand_md else None,
            "random_move_control": float(np.mean(rand_ms)) if rand_ms else None,
        }
        row["ratio"] = row["move_differ"] / max(row["move_control"], 1e-9) if md else None
        row["random_ratio"] = (
            row["random_move_differ"] / max(row["random_move_control"], 1e-9) if rand_md else None
        )
        out_groups.append(row)
        print(
            f"{mod} {row['concept']!r} n={len(js)} auc_w={row['auc_within']:.2f} bases={used} "
            f"differ={row['move_differ']} control={row['move_control']} ratio={row['ratio']} "
            f"dir={row['direction_corr']} rand_ratio={row['random_ratio']}",
            flush=True,
        )
    out = DATA / cpath.name.replace("circuits_", "concepts_" if args.mode == "circuit" else "concepts-union_")
    n_named = sum(abs(v["auc_within"] - 0.5) >= MIN_SEP for v in named.values())
    out.write_text(
        json.dumps(
            {"circuits": str(cpath), "rates": rates, "n_members": len(named), "n_named": n_named, "named": named,
             "groups": out_groups},
            indent=1,
        )
    )
    print(f"named {n_named} of {len(named)} circuit subcomponents")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
