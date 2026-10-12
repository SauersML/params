"""Circuits in VPD subcomponents for single MNIST test images, faithful to the real network under four families of
interventions, found by greedy pruning on one set of draws and scored on fresh draws.

A circuit S on image x is a set of subcomponents (fc1#c, fc2#c, fc3#c). Runs use VPD's masked forward (common.Decomp.run);
the remainder W - VU computes on the run's own inputs in every run ("live"; its relative Frobenius norm is ~3e-4, and the
point family is also reported with it off). Every KL is in bits from the reference run's prediction.
  point:    S on, every other subcomponent off; reference: the real model on x.
  removal:  a random subset R of S removed both from the circuit run (S minus R on, the rest off) and from the real
            model (every subcomponent but R on); reference: the real model without R. R takes each member of S with
            probability r, r ~ U(0, 1) per draw.
  donor:    every subcomponent outside S takes its activation a_c = h V_c from the real model's run on a donor image,
            S computes on the run's own inputs; reference: the real model on x. Donors are near (test images 5000-9999
            of the class the model predicts on x, which it also predicts on them; in the acceptance) or far (of other
            classes; an evaluation column only: fc3's subcomponents outside S then write the donor's class into the
            logits directly, which no small circuit can absorb). Its bound is the run with every subcomponent at the
            donor's activation (keep nothing): donor_norm = 1 - KL_S / KL_nothing.
  flips:    S on, each subcomponent outside S switched on independently with probability q, q ~ U(0, 1) per draw;
            reference: the real model on x.
A circuit is accepted when every family's mean KL over its draws is at most TAU bits. The search: start from every
subcomponent, try removing each in ascending order of its write on x (|a_c| ||U_c||), keep a removal when the smaller
set is still accepted on the search draws; two passes. Scores are then measured on fresh draws and fresh donors.

Baselines at the circuit's size k, scored on the same fresh draws: VPD's top k by causal importance on x (ties by write
size), the top k by write size, and random k-sets from VPD's active set (CI > 0 on x; the whole universe when k exceeds
it); and VPD's own answer (its active set, at its own size).

  ~/mpd-data/venv/bin/python experiments/oracle/mnist/circuits.py --tag i5e-4 --tau 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODULES, Decomp, kl_bits, load_decomp, load_mnist  # noqa: E402

FAMILIES = ("point", "removal", "donor", "flips")
CIRCUIT_POOL = 5000  # test images 0-4999 hold the circuit images; 5000-9999 are donors only
PER_CLASS = 5


class Draws:
    """One set of random interventions for one image: uniforms for removal and flips, the donors' activations."""

    def __init__(self, n: int, total: int, donor_a: torch.Tensor, gen: torch.Generator) -> None:
        self.n = n
        self.u_rem = torch.rand(n, total, generator=gen)
        self.r = torch.rand(n, 1, generator=gen)
        self.u_flip = torch.rand(n, total, generator=gen)
        self.q = torch.rand(n, 1, generator=gen)
        self.donor_a = donor_a  # [n, total]


class Scorer:
    def __init__(self, dec: Decomp, x: torch.Tensor) -> None:
        self.dec = dec
        self.x = x[None, :]
        self.a, _, _ = dec.acts(self.x)
        self.ref = dec.run(self.x, torch.ones(1, dec.total))

    def families(self, S: torch.Tensor, d: Draws, rem: float = 1.0) -> dict[str, torch.Tensor]:
        """KL bits per draw for each family. S [total] bool."""
        dec, n, T = self.dec, d.n, self.dec.total
        Sf = S.float()[None, :]
        R = (d.u_rem < d.r) & S[None, :]
        Fl = (d.u_flip < d.q) & ~S[None, :]
        masks = torch.cat(
            [
                Sf,  # point
                (~R).float(),  # removal, real model
                Sf * (~R).float(),  # removal, circuit
                torch.ones(n, T),  # donor: everything on, outside S from the donor
                (S[None, :] | Fl).float(),  # flips
            ]
        )
        rows = masks.shape[0]
        sel = torch.zeros(rows, T, dtype=torch.bool)
        val = torch.zeros(rows, T)
        o = 1 + 2 * n
        sel[o : o + n] = ~S[None, :]
        val[o : o + n] = d.donor_a
        remv = torch.full((rows,), 1.0)
        remv[0] = rem
        remv[1 + n : 1 + 2 * n] = rem
        remv[1 + 3 * n :] = rem
        lg = dec.run(self.x.expand(rows, -1), masks, remv, sel, val)
        ref = self.ref.expand(rows, -1)
        return {
            "point": kl_bits(ref[:1], lg[:1]),
            "removal": kl_bits(lg[1 : 1 + n], lg[1 + n : 1 + 2 * n]),
            "donor": kl_bits(ref[o : o + n], lg[o : o + n]),
            "flips": kl_bits(ref[o + n :], lg[o + n :]),
            "_agree": torch.cat(
                [
                    (lg[:1].argmax(-1) == ref[:1].argmax(-1)).float(),
                    (lg[1 + n : 1 + 2 * n].argmax(-1) == lg[1 : 1 + n].argmax(-1)).float(),
                    (lg[o:].argmax(-1) == ref[o:].argmax(-1)).float(),
                ]
            ),
            "_agree_split": torch.tensor([1 + 0.0, n, n, n]),
        }

    def nothing(self, d: Draws) -> torch.Tensor:
        """KL bits of the run that keeps nothing (every activation the donor's), per donor."""
        T = self.dec.total
        lg = self.dec.run(
            self.x.expand(d.n, -1), torch.ones(d.n, T), 1.0, torch.ones(d.n, T, dtype=torch.bool), d.donor_a
        )
        return kl_bits(self.ref.expand(d.n, -1), lg)


def summarize(sc: Scorer, S: torch.Tensor, d: Draws, nothing: torch.Tensor, far: Draws, far_nothing: torch.Tensor) -> dict:
    fam = sc.families(S, d)
    fam_far = sc.families(S, far)
    n = d.n
    ag = fam["_agree"]
    out = {f: float(fam[f].mean()) for f in FAMILIES}
    out["worst"] = max(out[f] for f in FAMILIES)
    out["agree_point"] = float(ag[0])
    out["agree_removal"] = float(ag[1 : 1 + n].mean())
    out["agree_donor"] = float(ag[1 + n : 1 + 2 * n].mean())
    out["agree_flips"] = float(ag[1 + 2 * n :].mean())
    out["donor_norm"] = float(1.0 - fam["donor"].mean() / nothing.mean())
    out["donor_far"] = float(fam_far["donor"].mean())
    out["donor_far_norm"] = float(1.0 - fam_far["donor"].mean() / far_nothing.mean())
    out["agree_donor_far"] = float(fam_far["_agree"][1 + far.n : 1 + 2 * far.n].mean())
    out["point_rem_off"] = float(sc.families(S, d, rem=0.0)["point"][0])
    out["size"] = int(S.sum())
    return out


def accepted(sc: Scorer, S: torch.Tensor, d: Draws, tau: float) -> bool:
    fam = sc.families(S, d)
    return all(float(fam[f].mean()) <= tau for f in FAMILIES)


def prune(sc: Scorer, order: torch.Tensor, d: Draws, tau: float, passes: int = 2, block: int = 64) -> torch.Tensor:
    """Greedy removal in the given order; a block of candidates is removed at once when the smaller set is accepted,
    else it is split in halves down to single subcomponents (the same accepted set as one-at-a-time removal whenever
    removals do not interact, at far fewer runs)."""
    S = torch.ones(sc.dec.total, dtype=torch.bool)

    def try_remove(js: list[int]) -> bool:
        js = [j for j in js if S[j]]
        if not js:
            return False
        S[js] = False
        if accepted(sc, S, d, tau):
            return True
        S[js] = True
        if len(js) == 1:
            return False
        h = len(js) // 2
        a = try_remove(js[:h])
        b = try_remove(js[h:])
        return a or b

    for _ in range(passes):
        changed = False
        cand = [j for j in order.tolist() if S[j]]
        for i in range(0, len(cand), block):
            changed = try_remove(cand[i : i + block]) or changed
        if not changed:
            break
    return S


def topk(score: torch.Tensor, tie: torch.Tensor, k: int) -> torch.Tensor:
    key = score * 1e6 + tie / (tie.max() + 1e-12)
    S = torch.zeros(score.shape[0], dtype=torch.bool)
    S[key.argsort(descending=True)[:k]] = True
    return S


def choose_images(yte: torch.Tensor, seed: int = 0) -> list[int]:
    g = np.random.default_rng(seed)
    perm = g.permutation(CIRCUIT_POOL)
    out = []
    for c in range(10):
        out += [int(i) for i in perm if int(yte[i]) == c][:PER_CLASS]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--eval-draws", type=int, default=64)
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--random-sets", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(0)
    dec = load_decomp(DATA / f"vpd_{args.tag}.pt")
    _, _, xte, yte = load_mnist()
    imgs = choose_images(yte)[: args.n_images]
    donor_pool = torch.arange(CIRCUIT_POOL, len(xte))
    a_donor_all, _, _ = dec.acts(xte[donor_pool])
    donor_pred = dec.run(xte[donor_pool], torch.ones(len(donor_pool), dec.total)).argmax(-1)
    donor_y = yte[donor_pool]
    ci_all = dec.ci(xte[imgs])
    contrib_all = dec.contribution(xte[imgs])
    results = []
    t0 = time.time()
    for ii, idx in enumerate(imgs):
        x = xte[idx]
        sc = Scorer(dec, x)
        gs = torch.Generator().manual_seed(1000 + idx)
        ge = torch.Generator().manual_seed(2000 + idx)
        c = int(sc.ref.argmax())
        near = torch.nonzero((donor_y == c) & (donor_pred == c)).flatten()
        far = torch.nonzero(donor_y != c).flatten()
        dsearch = Draws(args.draws, dec.total, a_donor_all[near[torch.randint(len(near), (args.draws,), generator=gs)]], gs)
        deval = Draws(
            args.eval_draws, dec.total, a_donor_all[near[torch.randint(len(near), (args.eval_draws,), generator=ge)]], ge
        )
        dfar = Draws(
            args.eval_draws, dec.total, a_donor_all[far[torch.randint(len(far), (args.eval_draws,), generator=ge)]], ge
        )
        noth = sc.nothing(deval)
        noth_far = sc.nothing(dfar)
        ci, contrib = ci_all[ii], contrib_all[ii]
        S = prune(sc, contrib.argsort(), dsearch, args.tau)
        k = int(S.sum())
        active = ci > 0
        methods = {
            "circuit": S,
            "vpd_topk_ci": topk(ci, contrib, k),
            "topk_write": topk(contrib, ci, k),
            "vpd_active": active,
        }
        row = {
            "image": idx,
            "label": int(yte[idx]),
            "pred": int(sc.ref.argmax()),
            "k": k,
            "n_active": int(active.sum()),
            "members": [dec.name(j) for j in torch.nonzero(S).flatten().tolist()],
            "members_ci": [float(ci[j]) for j in torch.nonzero(S).flatten().tolist()],
            "kl_nothing": float(noth.mean()),
            "kl_nothing_far": float(noth_far.mean()),
            "search": summarize(sc, S, dsearch, sc.nothing(dsearch), dfar, noth_far),
            "eval": {name: summarize(sc, M, deval, noth, dfar, noth_far) for name, M in methods.items()},
        }
        gr = torch.Generator().manual_seed(3000 + idx)
        rnd = []
        pool = torch.nonzero(active).flatten() if k <= int(active.sum()) else torch.arange(dec.total)
        for _ in range(args.random_sets):
            M = torch.zeros(dec.total, dtype=torch.bool)
            M[pool[torch.randperm(len(pool), generator=gr)[:k]]] = True
            rnd.append(summarize(sc, M, deval, noth, dfar, noth_far))
        row["eval"]["random_active"] = {key: float(np.mean([r[key] for r in rnd])) for key in rnd[0]}
        results.append(row)
        ev = row["eval"]
        print(
            f"[{ii}] img {idx} y={row['label']} k={k} active={row['n_active']} "
            + " ".join(f"{m}:{ev[m]['worst']:.3f}" for m in ev)
            + f" ({time.time() - t0:.0f}s)",
            flush=True,
        )
    out = DATA / f"circuits_{args.tag}_tau{args.tau}.json"
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    agg = {}
    for m in results[0]["eval"]:
        agg[m] = {
            key: float(np.mean([r["eval"][m][key] for r in results]))
            for key in results[0]["eval"][m]
        }
        agg[m]["pass_rate"] = float(np.mean([r["eval"][m]["worst"] <= args.tau for r in results]))
        agg[m]["median_worst"] = float(np.median([r["eval"][m]["worst"] for r in results]))
    print(json.dumps(agg, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
