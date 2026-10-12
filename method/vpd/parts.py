"""Parts of VPD's vpd4l subcomponents that share one gate, chosen by description length (#2951).

VPD's subcomponents are rank-one and each has its own gate (causal importance > 0 at a position). A part is a set
of subcomponents, of one weight matrix or of one layer, that share one gate: at every position the part is on or off,
and each member's own activity is coded as the gate plus its exceptions (the positions where the member differs from
the gate). Subcomponents become a part only when that code of their activity is shorter than coding each one's
activity alone; no threshold decides it.

Activity. a_i is the set of positions (tokens of split A, position 0 left out) where subcomponent i is active
(CI > 0, VPD's published CI network, as in atlas.py). N is the number of positions.

Code lengths, in bits, with L(n, M) = log2 binom(M, n) + log2(M + 1) (a count, then which n of M):
  alone      L(|a_i|, N) per subcomponent.
  a part P   of K members with gate rule tau (on where at least tau members are active, 1 <= tau <= K):
             L(m, N) for the gate (m positions on), and per member L(off_i, m) + L(on_i, N - m) for its exceptions
             (off_i: gate on, member off; on_i: gate off, member on), plus log2 K for tau, plus the member list:
             log2 binom(C_S, K) + log2 C_S, with S the part's scope (its weight matrix, or its layer for a part that
             spans matrices) and C_S the number of subcomponents in S. tau is the one of 1..K that gives the
             shortest code.
Merging two parts pays when the merged part's code is shorter than the sum of theirs. The search is greedy: from
singletons, repeatedly the pair of parts with the largest saving among candidate pairs (each subcomponent's TOP
partners by co-active positions, from the Gram matrix of the binary activity), the saving recomputed exactly
for the current parts before each merge; first within each weight matrix, then across the matrices of a layer.
Check: the code length of split B's activity under split A's parts (their members and tau) against each member
alone.

Files under ~/mpd-data/atlas/vpd4l/parts/: activity_<split>_L<layer>.npz (each subcomponent's active positions,
compressed sparse rows over the layer's 9,728 subcomponents), gram_L<layer>.npy (co-active position counts,
split A), parts.json (every part: members, scope, tau, code lengths) and part_of.npy (per subcomponent its part,
-1 alone).

  MPD_MEM_GIB=20 python -m method.vpd.parts activity [--train 0:5000] [--heldout 5000:6000] [--batch 8]
  python -m method.vpd.parts build
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from method.vpd import vpd_model as VM

DEFAULT = Path.home() / "mpd-data/atlas/vpd4l/parts"
TOP = 16  # candidate partners per subcomponent, by co-active positions
T = 512


def rows(spec: str) -> tuple[int, int]:
    a, b = spec.split(":")
    return int(a), int(b)


def tokens_of(r0: int, r1: int, path: str | None) -> torch.Tensor:
    """Validation rows [r0, r1), T tokens each: from `path` (an array of rows 0.. of vpd_model.val_tokens, for a
    pod without the parquet) or from vpd_model.val_tokens."""
    if path:
        return torch.from_numpy(np.load(path, mmap_mode="r")[r0:r1, :T].astype(np.int64))
    return VM.val_tokens(r1 - r0, T, r0)


@torch.no_grad()
def active(vpd: VM.VPD, names: list[str], ids: torch.Tensor) -> torch.Tensor:
    """CI > 0 of every subcomponent at positions 1.. [B, T - 1, total] (one clean pass, the CI network)."""
    vpd.clear()
    for n in names:
        vpd.target.site(n).cache_input = True
    vpd.target.hidden(ids)
    acts = {n: vpd.target.site(n).last_input for n in names}
    vpd.clear()
    z = vpd.ci_fn(acts)
    return torch.cat([z[n][:, 1:] > 0 for n in names], -1)


def activity(args):
    """Per layer, every subcomponent's active positions on both splits, and split A's per-layer Gram matrix. The
    pass writes its (subcomponent, position) pairs to chunk files as it goes; `finish` turns them into the
    compressed rows (a crash after the pass loses nothing)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    target = VM.load_target(dev)
    vpd = VM.load_vpd(target, dev)
    names = VM.site_names()
    per_layer = sum(vpd.C[n] for n in names[:6])
    for split, spec, gram in (("train", args.train, True), ("heldout", args.heldout, False)):
        r0, r1 = rows(spec)
        chunks = out / f"chunks_{split}"
        if (out / f"activity_{split}_L3.npz").exists():
            continue
        chunks.mkdir(exist_ok=True)
        tokens = tokens_of(r0, r1, args.tokens)
        G = [torch.zeros(per_layer, per_layer, device=dev) for _ in range(4)] if gram else None
        comp, pos = [[] for _ in range(4)], [[] for _ in range(4)]
        t0, part = time.time(), 0
        for b0 in range(0, tokens.shape[0], args.batch):
            on = active(vpd, names, tokens[b0:b0 + args.batch].to(dev))
            B = on.shape[0]
            for layer in range(4):
                x = on[..., layer * per_layer:(layer + 1) * per_layer].reshape(B * (T - 1), per_layer)
                if gram:
                    xf = x.float()
                    G[layer].addmm_(xf.T, xf)
                p, c = x.nonzero(as_tuple=True)
                comp[layer].append(c.to(torch.int16).cpu().numpy())
                pos[layer].append((p + b0 * (T - 1)).to(torch.int32).cpu().numpy())
            last = b0 + args.batch >= tokens.shape[0]
            if (b0 // args.batch) % 25 == 24 or last:
                for layer in range(4):
                    np.savez(chunks / f"L{layer}_{part:04d}.npz", comp=np.concatenate(comp[layer]), pos=np.concatenate(pos[layer]))
                comp, pos, part = [[] for _ in range(4)], [[] for _ in range(4)], part + 1
                done = min(b0 + args.batch, tokens.shape[0])
                print(f"{time.strftime('%H:%M:%S')} {split}: {done}/{tokens.shape[0]} sequences, "
                      f"{done / (time.time() - t0):.2f}/s", flush=True)
        if gram:
            for layer in range(4):
                np.save(out / f"gram_L{layer}.npy", G[layer].cpu().numpy())
            del G
        (chunks / "DONE").write_text(json.dumps({"rows": [r0, r1], "N": int(tokens.shape[0] * (T - 1)), "parts": part}))
        print(f"{split}: {time.time() - t0:.0f} s", flush=True)
    del vpd, target
    finish(out, per_layer)


def finish(out: Path, per_layer: int = 9728):
    """The chunk files of each finished split as compressed rows per layer (positions ascending per subcomponent)."""
    for split in ("train", "heldout"):
        chunks = out / f"chunks_{split}"
        if not (chunks / "DONE").exists():
            continue
        done = json.loads((chunks / "DONE").read_text())
        for layer in range(4):
            if (out / f"activity_{split}_L{layer}.npz").exists():
                continue
            files = [chunks / f"L{layer}_{k:04d}.npz" for k in range(done["parts"])]
            cs, ps = [], []
            for f in files:
                with np.load(f) as z:
                    cs.append(z["comp"])
                    ps.append(z["pos"])
            c, p = np.concatenate(cs), np.concatenate(ps)
            del cs, ps
            order = np.argsort(c, kind="stable")  # positions stay ascending within a subcomponent
            counts = np.bincount(c.astype(np.int64), minlength=per_layer)
            np.savez(out / f"activity_{split}_L{layer}.npz", indptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                     positions=p[order], N=np.int64(done["N"]), rows=np.array(done["rows"]))
            del c, p, order
            print(f"{split} L{layer}: {int(counts.sum())} active (subcomponent, position) pairs", flush=True)


class Activity:
    """One layer's active positions per subcomponent (compressed sparse rows)."""

    def __init__(self, path: Path):
        f = np.load(path)
        self.indptr, self.positions, self.N = f["indptr"], f["positions"], int(f["N"])

    def of(self, i: int) -> np.ndarray:
        return self.positions[self.indptr[i]:self.indptr[i + 1]]

    def count(self, i: int) -> int:
        return int(self.indptr[i + 1] - self.indptr[i])


# ------------------------------------------------------------------ code lengths


def lbinom(M, n):
    from scipy.special import gammaln
    M, n = np.asarray(M, dtype=np.float64), np.asarray(n, dtype=np.float64)
    return (gammaln(M + 1) - gammaln(n + 1) - gammaln(M - n + 1)) / math.log(2)


def L(n, M):
    """Bits of a count n <= M and which n of M."""
    return lbinom(M, n) + np.log2(np.asarray(M, dtype=np.float64) + 1)


def alone(n, N):
    return L(n, N)


def members_bits(K: int, scope: int) -> float:
    return 0.0 if K == 1 else float(lbinom(scope, K) + math.log2(scope))


class Part:
    """A part's members and the union of their active positions with how many members are on at each."""

    def __init__(self, members: list[int], U: np.ndarray, c: np.ndarray):
        self.members, self.U, self.c = members, U, c

    @staticmethod
    def single(i: int, act: "Activity") -> "Part":
        P = act.of(i)
        return Part([i], P, np.ones(len(P), np.int32))

    def merged(self, other: "Part") -> "Part":
        U = np.union1d(self.U, other.U)
        c = np.zeros(len(U), np.int32)
        c[np.searchsorted(U, self.U)] += self.c
        c[np.searchsorted(U, other.U)] += other.c
        return Part(self.members + other.members, U, c)

    def bits(self, act: "Activity", scope: int) -> tuple[float, int]:
        """(bits, tau): the part's code length (module note) at its best gate rule."""
        if len(self.members) == 1:
            return float(alone(len(self.U), act.N)), 1
        levels = [self.c[np.searchsorted(self.U, act.of(i))] for i in self.members]
        n = np.array([len(lv) for lv in levels])
        return code(len(self.members), self.c, levels, n, act.N, scope)


def code(K: int, U_counts: np.ndarray, levels: list[np.ndarray], n: np.ndarray, N: int, scope: int) -> tuple[float, int]:
    """(bits, tau) of a part: the shortest over tau (module note)."""
    H = np.bincount(U_counts, minlength=K + 1)  # positions with exactly c members on
    m = np.cumsum(H[::-1])[::-1]  # m[tau]: gate positions with tau = index
    hist = np.stack([np.bincount(lv, minlength=K + 1) for lv in levels])  # [K, K + 1]
    inside = np.cumsum(hist[:, ::-1], axis=1)[:, ::-1]  # inside[i, tau]: member i's positions with count >= tau
    taus = np.arange(1, K + 1)
    mt = m[taus]  # [K taus]
    ins = inside[:, taus]  # [members, taus]
    off = mt[None, :] - ins
    on = n[:, None] - ins
    bits = L(mt, N) + (L(off, mt[None, :]) + L(on, N - mt[None, :])).sum(0) + math.log2(K) + members_bits(K, scope)
    j = int(np.argmin(bits))
    return float(bits[j]), int(taus[j])


def part_code(members: list[int], act: "Activity", scope: int) -> tuple[float, int]:
    p = Part.single(members[0], act)
    for i in members[1:]:
        p = p.merged(Part.single(i, act))
    return p.bits(act, scope)


def pair_saving(n: np.ndarray, g: np.ndarray, N: int, scope: int) -> np.ndarray:
    """Exact saving of merging singletons i, j from their counts n_i, n_j and co-active count g (tau 1 or 2)."""
    ni, nj = n[..., 0], n[..., 1]
    sep = alone(ni, N) + alone(nj, N)
    m1 = ni + nj - g  # tau 1: the union
    or_bits = L(m1, N) + L(m1 - ni, m1) + L(0, N - m1) + L(m1 - nj, m1) + L(0, N - m1)
    and_bits = L(g, N) + L(0, g) + L(ni - g, N - g) + L(0, g) + L(nj - g, N - g)
    return sep - (np.minimum(or_bits, and_bits) + 1.0 + members_bits(2, scope))


def agglomerate(groups: list[list[int]], act: "Activity", scope_of, candidates: dict, log, label: str) -> list[list[int]]:
    """Greedy merging of the parts `groups` (lists of subcomponent ids in act's numbering) over candidate pairs of
    subcomponents; returns the final parts. Every saving is recomputed exactly for the current parts before a merge;
    scope_of(members) is the scope size a part's member list is coded in."""
    part_of, parts, bits = {}, {}, {}
    for k, g in enumerate(groups):
        p = Part.single(g[0], act)
        for i in g[1:]:
            p = p.merged(Part.single(i, act))
        parts[k], bits[k] = p, p.bits(act, scope_of(p.members))[0]
        for i in g:
            part_of[i] = k
    heap = [(-s, i, j) for (i, j), s in candidates.items()
            if s > 0 and i in part_of and j in part_of and part_of[i] != part_of[j]]
    heapq.heapify(heap)
    merges, evaluated, nxt = 0, 0, len(groups)
    t0 = time.time()
    while heap:
        s, i, j = heapq.heappop(heap)
        a, b = part_of[i], part_of[j]
        if a == b:
            continue
        merged = parts[a].merged(parts[b])
        new_bits = merged.bits(act, scope_of(merged.members))[0]
        evaluated += 1
        saving = bits[a] + bits[b] - new_bits
        if saving <= 0:
            continue
        if heap and saving < -heap[0][0] - 1e-9:
            heapq.heappush(heap, (-saving, i, j))  # stale estimate: re-rank with the exact saving
            continue
        k = nxt
        nxt += 1
        parts[k], bits[k] = merged, new_bits
        for x in merged.members:
            part_of[x] = k
        del parts[a], parts[b], bits[a], bits[b]
        merges += 1
        if merges % 500 == 0:
            log(f"  {label}: {merges} merges, {evaluated} evaluated, {len(parts)} parts, {time.time() - t0:.0f} s")
    log(f"  {label}: {merges} merges, {evaluated} evaluated, {len(parts)} parts, {time.time() - t0:.0f} s")
    return [sorted(p.members) for p in parts.values()]


def candidates_from(gram: np.ndarray, n: np.ndarray, N: int, scope: int, allowed) -> dict:
    """Each subcomponent's TOP partners by co-active count among `allowed` pairs, with their exact pair saving."""
    out = {}
    G = gram.copy()
    np.fill_diagonal(G, 0)
    G = np.where(allowed, G, 0)
    top = np.argpartition(-G, TOP, axis=1)[:, :TOP]
    for i in range(G.shape[0]):
        js = top[i]
        js = js[G[i, js] > 0]
        if len(js) == 0:
            continue
        sv = pair_saving(np.stack([np.full(len(js), n[i]), n[js]], -1), G[i, js], N, scope)
        for j, s in zip(js.tolist(), sv.tolist()):
            if s > 0:
                out[(min(i, j), max(i, j))] = max(out.get((min(i, j), max(i, j)), -1e30), s)
    return out


def total_bits(parts: list[list[int]], act: Activity, scope_of, taus=None) -> float:
    total = 0.0
    for k, p in enumerate(parts):
        if taus is None:
            total += part_code(p, act, scope_of(p))[0]
        else:
            total += fixed_tau_bits(p, act, scope_of(p), taus[k])
    return total


def fixed_tau_bits(members: list[int], act: Activity, scope: int, tau: int) -> float:
    """A part's code length with its gate rule fixed (split B under split A's parts)."""
    if len(members) == 1:
        return float(alone(act.count(members[0]), act.N))
    p = Part.single(members[0], act)
    for i in members[1:]:
        p = p.merged(Part.single(i, act))
    K = len(members)
    m = float((p.c >= tau).sum())
    bits = L(m, act.N) + math.log2(K) + members_bits(K, scope)
    for i in members:
        lv = p.c[np.searchsorted(p.U, act.of(i))]
        ins = float((lv >= tau).sum())
        bits += L(m - ins, m) + L(len(lv) - ins, act.N - m)
    return float(bits)


def build(args):
    root = Path(args.out)
    names = VM.site_names()
    counts = {"q_proj": 512, "k_proj": 512, "v_proj": 1024, "o_proj": 1024, "c_fc": 3072, "down_proj": 3584}
    kinds = ["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj"]
    off = np.concatenate([[0], np.cumsum([counts[k] for k in kinds])])  # site offsets within a layer
    per_layer = int(off[-1])
    logf = open(root / "build.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    all_parts, report = [], {"layers": {}}
    for layer in range(4):
        act = Activity(root / f"activity_train_L{layer}.npz")
        held = Activity(root / f"activity_heldout_L{layer}.npz")
        gram = np.load(root / f"gram_L{layer}.npy")
        n = np.diff(act.indptr)
        site_of = np.searchsorted(off, np.arange(per_layer), side="right") - 1
        scope_of = lambda p: per_layer if len(set(site_of[p].tolist())) > 1 else counts[kinds[site_of[p[0]]]]
        t0 = time.time()
        parts = []
        for s, kind in enumerate(kinds):  # within each matrix
            ids = np.arange(off[s], off[s + 1])
            sub = gram[np.ix_(ids, ids)]
            cand = candidates_from(sub, n[ids], act.N, counts[kind], np.ones_like(sub, dtype=bool))
            cand = {(int(ids[i]), int(ids[j])): v for (i, j), v in cand.items()}
            parts += agglomerate([[int(i)] for i in ids], act, scope_of, cand, log, f"L{layer}.{kind}")
        within = len(parts)
        # Across the layer's matrices: candidates are pairs of subcomponents of different matrices.
        cross = site_of[:, None] != site_of[None, :]
        cand = candidates_from(gram, n, act.N, per_layer, cross)
        parts = agglomerate(parts, act, scope_of, cand, log, f"L{layer} across")
        taus, part_bits = [], []
        for p in parts:
            b, tau = part_code(p, act, scope_of(p))
            taus.append(tau)
            part_bits.append(b)
        sep_a = float(alone(n, act.N).sum())
        sep_b = float(alone(np.diff(held.indptr), held.N).sum())
        parts_b = total_bits(parts, held, scope_of, taus)
        multi = [k for k, p in enumerate(parts) if len(p) > 1]
        report["layers"][layer] = {
            "seconds": time.time() - t0, "parts": len(parts), "parts_within_matrices": within, "multi": len(multi),
            "in_multi": int(sum(len(parts[k]) for k in multi)), "cross_matrix": int(sum(len(set(site_of[parts[k]].tolist())) > 1 for k in multi)),
            "bits_alone_A": sep_a, "bits_parts_A": float(sum(part_bits)), "bits_alone_B": sep_b, "bits_parts_B": parts_b,
            "positions_A": act.N, "positions_B": held.N}
        log(f"L{layer}: {len(multi)} parts of 2+ holding {report['layers'][layer]['in_multi']} subcomponents; "
            f"A {sep_a / act.N:.3f} -> {sum(part_bits) / act.N:.3f} bits/position, B {sep_b / held.N:.3f} -> {parts_b / held.N:.3f}")
        for k, p in enumerate(parts):
            base = layer * per_layer
            all_parts.append({"layer": layer, "members": [int(base + i) for i in p], "tau": taus[k], "bits_A": part_bits[k],
                              "sites": sorted({names[layer * 6 + int(site_of[i])] for i in p})})
    total = 4 * per_layer
    part_of = np.full(total, -1, np.int32)
    numbered = [q for q in all_parts if len(q["members"]) > 1]
    for k, q in enumerate(numbered):
        q["id"] = k
        part_of[q["members"]] = k
    np.save(root / "part_of.npy", part_of)
    (root / "parts.json").write_text(json.dumps({"report": report, "parts": numbered, "top": TOP,
                                                 "fields": __doc__}, indent=None) + "\n")
    log(f"wrote {len(numbered)} parts")


# ------------------------------------------------------------------ loader


class Parts:
    """parts.json and part_of.npy: for a subcomponent (flat index over vpd_model.site_names() order) its part, and a
    part's members, gate rule and sites."""

    def __init__(self, root: Path = DEFAULT):
        root = Path(root)
        d = json.loads((root / "parts.json").read_text())
        self.parts, self.report = d["parts"], d["report"]
        self.part_of = np.load(root / "part_of.npy")
        names = VM.site_names()
        cs = {"q_proj": 512, "k_proj": 512, "v_proj": 1024, "o_proj": 1024, "c_fc": 3072, "down_proj": 3584}
        self.names = names
        self.offset = dict(zip(names, np.concatenate([[0], np.cumsum([cs[n.rsplit('.', 1)[1]] for n in names])])[:-1].tolist()))

    def flat(self, site: str, index: int) -> int:
        return self.offset[site] + index

    def site_index(self, flat: int) -> tuple[str, int]:
        site = max((n for n in self.names if self.offset[n] <= flat), key=lambda n: self.offset[n])
        return site, flat - self.offset[site]

    def of(self, site: str, index: int) -> int:
        """The part holding the subcomponent, or -1 when it is alone."""
        return int(self.part_of[self.flat(site, index)])

    def members(self, part: int) -> list[tuple[str, int]]:
        return [self.site_index(f) for f in self.parts[part]["members"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("activity")
    a.add_argument("--out", default=str(DEFAULT))
    a.add_argument("--train", default="0:5000")
    a.add_argument("--heldout", default="5000:6000")
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--device", default="mps")
    a.add_argument("--tokens", help="rows 0.. of the validation split as an .npy (vpd_model.val_tokens otherwise)")
    t = sub.add_parser("tokens")
    t.add_argument("out")
    t.add_argument("--rows", default="0:6000")
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(DEFAULT))
    f = sub.add_parser("finish")
    f.add_argument("--out", default=str(DEFAULT))
    args = ap.parse_args()
    if args.cmd == "tokens":
        r0, r1 = rows(args.rows)
        assert r0 == 0, "a token file holds rows 0.."
        np.save(args.out, VM.val_tokens(r1, T, 0).numpy().astype(np.uint16))
        return
    {"activity": activity, "build": build, "finish": lambda a: finish(Path(a.out))}[args.cmd](args)


if __name__ == "__main__":
    main()
