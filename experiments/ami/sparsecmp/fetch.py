"""Fetch the circuit-sparsity release (Gao et al. 2511.13653; github.com/openai/circuit_sparsity, Apache-2.0).

models   NAME...          weights + config into ~/mpd-data/ami/sparsecmp/models/NAME/
means    NAME...          their pretraining means (viz/NAME/mean_baseline_pretrain.pt), sparse models only
prunes   NAME SWEEP TASK...  their pruned circuits: per CARBS run the bisected k and target; the smallest circuit's
                          node indices. The 30-200 MB res.pt is deleted after the summary is written.
"""
import json
import os
import sys
import urllib.request

import torch

B = "https://openaipublic.blob.core.windows.net/circuit-sparsity"
ROOT = os.path.expanduser("~/mpd-data/ami/sparsecmp")


def get(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def models(names):
    for n in names:
        for f in ("beeg_config.json", "final_model.pt"):
            get(f"{B}/models/{n}/{f}", f"{ROOT}/models/{n}/{f}")
        print("model", n, flush=True)


def means(names):
    for n in names:
        get(f"{B}/viz/{n}/mean_baseline_pretrain.pt", f"{ROOT}/models/{n}/mean_baseline_pretrain.pt")
        print("means", n, flush=True)


def prunes(name, sweep, tasks):
    out = f"{ROOT}/released/prunes_{name}_{sweep}.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    for t in tasks:
        if t in res:
            continue
        tmp = f"{ROOT}/released/tmp_{name}_{t}.pt"
        try:
            get(f"{B}/prunes/{name}/{t}/{sweep}/k_optim/res.pt", tmp)
            x = torch.load(tmp, weights_only=True, map_location="cpu")
        except Exception as e:                      # fstring_brace's res.pt is an 864-byte stub for some models
            print("prune", name, t, "ERR", repr(e)[:200], flush=True)
            res[t] = {"error": repr(e)[:200]}
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        runs = []
        for it in x:
            o = it["observed_value"]
            dm = o["discrete_masks"]
            runs.append({"i": it["i"], "target_loss": it["target_loss"], "k": int(o["final_bisect_k"]),
                         "k_mask": int(sum(int(v.sum()) for v in dm.values())),
                         "last": {kk: vv for kk, vv in o["metrics"][-1].items() if not isinstance(vv, dict)},
                         "suggestion": it["suggestion"]})
        best = min(range(len(x)), key=lambda j: runs[j]["k"])
        dm = x[best]["observed_value"]["discrete_masks"]
        res[t] = {"runs": runs, "best": best,
                  "nodes": {loc: v.nonzero().flatten().tolist() for loc, v in dm.items() if v.any()},
                  "sizes": {loc: int(v.numel()) for loc, v in dm.items()}}
        os.remove(tmp)
        json.dump(res, open(out, "w"))
        print("prune", name, t, "k", [r["k"] for r in runs], "target", runs[0]["target_loss"], flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "models":
        models(args)
    elif cmd == "means":
        means(args)
    elif cmd == "prunes":
        prunes(args[0], args[1], args[2:])
