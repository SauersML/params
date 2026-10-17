"""The faithful reading, scored after the fact on finished runs: for each record (prune.py's <model>__<task>.json), its
held-out pairs rebuilt exactly (same items file, split and counts), and per pair KL(full model || circuit) over the
task's two outcomes. Written to ~/mpd-data/ami/sparsecmp/faithful/<record name> (records synced from MATS are left
untouched, so a later pull cannot overwrite these).

python faithful.py RUN_DIR [--device cpu] [--means-dir DIR]
"""
import argparse
import glob
import json
import os

import torch

import nodes as N
import prune as P

ROOT = N.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default="", help="model-name prefix")
    a = ap.parse_args()
    models = {}
    out_dir = f"{ROOT}/faithful"
    os.makedirs(out_dir, exist_ok=True)
    for p in sorted(glob.glob(f"{a.run_dir}/*.json")):
        r = json.load(open(p))
        out = f"{out_dir}/{os.path.basename(p)}"
        if not r.get("keep") or "kl2" in r.get("test", {}) or os.path.exists(out) or not r["model"].startswith(a.only):
            continue
        g = r["args"]
        name = r["model"]
        means = g.get("means") or ""
        means = means.replace("/mnt/nw/home/s.sauers/", os.path.expanduser("~/")) or \
            f"{ROOT}/models/{name}/mean_baseline_pretrain.pt"
        key = (name, means)
        if key not in models:
            models.clear()
            models[key] = (N.GaoModel(name, a.device, means=means) if g["family"] == "gao"
                           else N.QwenModel(name, a.device, means=means))
        model = models[key]
        items_path = os.path.expanduser("~/mpd-data/ami/sparsecmp/" + os.path.basename(g["items"]))
        items = [json.loads(l) for l in open(items_path)]
        its = [it for it in items if it["task"] == r["task"]]
        _, test = P.split_items(its, g["n_search"], g["n_test"])
        bt = P.Batches(model, N.encode_items(model, test), g["bs"])
        assert bt.n == r["n_test"], (p, bt.n, r["n_test"])
        kt = {s: torch.tensor(v, device=model.device) for s, v in r["keep"].items()}
        kl = P.pair_kl(model, bt, P.discrete(model, kt))
        json.dump({"model": name, "task": r["task"], "kl2": sum(kl) / len(kl), "pairs_kl2": kl}, open(out, "w"))
        print(os.path.basename(p), "kl2", round(sum(kl) / len(kl), 4), flush=True)


if __name__ == "__main__":
    main()
