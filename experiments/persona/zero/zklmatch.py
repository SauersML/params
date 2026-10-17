"""The found set shrunk along its ranking until its KL meets the gate: KL from unedited Zero on held-out atlas web text
(zattr.py's KL documents) for each prefix of the found order, the largest prefix at or under --max-kl, and 6 random sets
matched to it by kind and layer (zattr.py's seeds), written as a sets file for zgen.py and zjudge.py.

  zklmatch.py SETS.json MEANS.pt ATLAS.npz OUT_DIR [--max-kl 0.1] [--sizes 1,2,...,8]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zattr  # noqa: E402
import zhooks as Z  # noqa: E402


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets")
    ap.add_argument("means")
    ap.add_argument("atlas")
    ap.add_argument("out")
    ap.add_argument("--max-kl", type=float, default=0.1)
    ap.add_argument("--sizes", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--seeds", type=int, default=zattr.SEEDS)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    order = json.loads(Path(a.sets).read_text())["order"]
    st = zattr.Study(Z.ZERO)
    ab = Z.HFAblator(st.model, torch.load(a.means))
    at = np.load(a.atlas)
    toks, offs, nc = at["tokens"], at["offsets"], int(at["n_chat"])
    plain = [{"ids": toks[offs[d]:offs[d + 1]].tolist(), "start": 1} for d in range(nc + 48, nc + 48 + zattr.KL_DOCS)]
    res = {"max_kl": a.max_kl, "kl": {}}
    for s in [int(x) for x in a.sizes.split(",")]:
        res["kl"][s] = st.kl(ab, plain, order[:s], plain=True)
        print(f"prefix {s}: KL {res['kl'][s]:.4f} bits ({time.time() - t0:.0f} s)", flush=True)
    ok = [s for s, k in res["kl"].items() if k <= a.max_kl]
    best = max(ok) if ok else None
    res["size"] = best
    sets = {}
    if best:
        sets[f"found_{best}"] = order[:best]
        for seed in range(a.seeds):
            sets[f"random_{best}_{seed}"] = Z.random_matched(order[:best], st.NN, st.NH, seed=100 * best + seed)
        res["random_kl"] = {n: st.kl(ab, plain, u, plain=True) for n, u in sets.items() if n.startswith("random")}
    (out / "sets.json").write_text(json.dumps({"sets": sets, "order": order}))
    (out / "klmatch.json").write_text(json.dumps(res))
    print(f"largest prefix at KL <= {a.max_kl}: {best} ({order[:best] if best else []})", flush=True)
    if not best:
        raise SystemExit("no prefix meets the KL")


if __name__ == "__main__":
    main()
