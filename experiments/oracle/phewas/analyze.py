"""Stage 1's cross-trait views of the neuron PheWAS (#2951): which behaviours share neurons, how pleiotropic neurons
are, and where refusal sits (the lead's question: is refusal polygenic while each neuron has one main job?). Reads
assoc.py's per-trait npz files (t per pole and measure on all prompts and on each half) and summary.json. Association,
not causation: stage 2's knockouts answer the same questions causally.

  analyze.py ASSOC_DIR PHENOME_DIR OUT.json [--perms 200]
Per neuron its signed association with a trait is the t of its strongest measure; a neuron is a replicated hit of a
trait when that measure passes Bonferroni on all prompts and p < 0.05 with the same sign in each half (assoc.py).
  correlation   trait x trait: the correlation of the neurons' signed t (the genetic-correlation analogue), over the
                neurons that are a replicated hit of either trait
  pleiotropy    per neuron the number of traits it is a replicated hit of, against a null that places each trait's
                hits on random neurons (the same number per trait): the observed and null distributions of the count
  refusal       for the refusal traits: replicated-hit neurons, how many other traits each of them is a hit of, and the
                same for random hit neurons of the other traits
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_all(assoc: Path, traits: list[str]):
    from scipy import stats

    summ = json.loads((assoc / "summary.json").read_text())
    T, H = {}, {}
    for name in traits:
        if name not in summ:
            continue
        z = np.load(assoc / f"{name.replace('/', '__')}.npz")
        t, td, th = (z[k].astype(np.float32).reshape(-1, 4) for k in ("t", "t_dev", "t_held"))
        best = np.abs(t).argmax(1)
        rows = np.arange(len(t))
        tb, tdb, thb = t[rows, best], td[rows, best], th[rows, best]
        alpha_t = stats.t.isf(10 ** summ[name]["bonferroni_log10_alpha"] / 2, int(z["df"]))
        rep_t = stats.t.isf(0.025, int(z["df_half"]))
        hit = (np.abs(tb) > alpha_t) & (np.sign(tdb) == np.sign(tb)) & (np.sign(thb) == np.sign(tb)) & (np.abs(tdb) > rep_t) & (np.abs(thb) > rep_t)
        T[name], H[name] = tb, hit
    return T, H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("assoc", type=Path)
    ap.add_argument("phenome", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--perms", type=int, default=200)
    a = ap.parse_args()
    phen = {t["name"]: t for t in json.loads((a.phenome / "phenome.json").read_text())["traits"] if t["kept"]}
    T, H = load_all(a.assoc, sorted(phen))
    names = sorted(T)
    N = len(next(iter(T.values())))
    hits = np.stack([H[n] for n in names])  # [traits, neurons]
    count = hits.sum(0)
    rng = np.random.default_rng(15)
    null = np.zeros((a.perms, len(names) + 1))
    per = hits.sum(1)
    for p in range(a.perms):
        c = np.zeros(N, dtype=np.int32)
        for k in per:
            c[rng.choice(N, int(k), replace=False)] += 1
        null[p] = np.bincount(c, minlength=len(names) + 1)[:len(names) + 1]
    obs = np.bincount(count, minlength=len(names) + 1)[:len(names) + 1]
    corr = np.full((len(names), len(names)), np.nan)
    for i, x in enumerate(names):
        for j in range(i, len(names)):
            sel = H[x] | H[names[j]]
            if sel.sum() >= 30:
                corr[i, j] = corr[j, i] = float(np.corrcoef(T[x][sel], T[names[j]][sel])[0, 1])
    ref = [n for n in names if n.startswith("refusal/") or n.startswith("over-refusal/")]
    others = [n for n in names if n not in ref]
    other_hits = np.stack([H[n] for n in others]).sum(0) if others else np.zeros(N)
    refusal = {}
    for n in ref:
        h = np.flatnonzero(H[n])
        refusal[n] = {"replicated_neurons": int(len(h)), "other_traits_per_hit_neuron_mean": float(other_hits[h].mean()) if len(h) else None,
                      "share_hit_by_no_other_trait": float((other_hits[h] == 0).mean()) if len(h) else None,
                      "layers": np.bincount(h // (N // 36), minlength=36).tolist() if len(h) else []}
    any_other = np.flatnonzero(other_hits > 0)
    out = {"traits": names, "neurons": N, "replicated_hits_per_trait": {n: int(H[n].sum()) for n in names},
           "pleiotropy": {"observed": obs.tolist(), "null_mean": null.mean(0).tolist(), "null_p95": np.percentile(null, 95, 0).tolist(),
                          "neurons_hit_by_any": int((count > 0).sum()), "mean_traits_per_hit_neuron": float(count[count > 0].mean()) if (count > 0).any() else None},
           "correlation": corr.tolist(), "refusal": refusal,
           "other_hit_neurons_traits_mean": float(other_hits[any_other].mean()) if len(any_other) else None}
    a.out.write_text(json.dumps(out))
    print(f"{len(names)} traits; neurons hit by any trait {out['pleiotropy']['neurons_hit_by_any']}, mean traits per hit neuron "
          f"{out['pleiotropy']['mean_traits_per_hit_neuron']}")
    for k in range(min(12, len(obs))):
        print(f"  hit by {k} traits: observed {obs[k]}, null mean {null.mean(0)[k]:.1f} (95th percentile {np.percentile(null, 95, 0)[k]:.0f})")
    for n, r in refusal.items():
        print(f"  {n}: {r['replicated_neurons']} replicated neurons; other traits per hit neuron {r['other_traits_per_hit_neuron_mean']}; "
              f"hit by no other trait {r['share_hit_by_no_other_trait']}")


if __name__ == "__main__":
    main()
