"""mib_measured.py OUT.npz [--split test] [--n 32] [--batch 250]: MIB's InterpBench IOI edges ranked by measured single-edge ablation. Each of the
graph's edges is removed alone (every other edge in) and the model is run with MIB's own patching semantics
(eap.evaluate.evaluate_graph, intervention "patching": the edge's input to its reader replaced by the corrupted prompt's
activation of its writer, everything downstream recomputed); the change in MIB's logit difference on each test prompt is
the edge's effect there. Writes effects [prompts, edges] in interpbench.mib_graph_edges' order with the clean logit
differences and the seconds. Runs in MIB's own environment (~/mpd-data/venvs/mib, python -I) on its code
(~/mpd-data/ext/MIB-circuit-track)."""
import argparse
import pickle
import sys
import time
from functools import partial
from pathlib import Path

M = Path.home() / "mpd-data/ext/MIB-circuit-track"
sys.path[:0] = [str(M), str(M / "EAP-IG/src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402

from eap.evaluate import evaluate_baseline, evaluate_graph  # noqa: E402
from eap.graph import Graph  # noqa: E402
from MIB_circuit_track.dataset import HFEAPDataset  # noqa: E402
from MIB_circuit_track.metrics import get_metric  # noqa: E402

RAW = Path.home() / "mpd-data/interpbench/raw/mib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = pickle.load(open(RAW / "ll_model_cfg.pkl", "rb"))
    cfg = HookedTransformerConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
    cfg.device = "cpu"
    cfg.use_hook_mlp_in = cfg.use_attn_result = cfg.use_split_qkv_input = True   # MIB's load_interpbench_model
    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(RAW / "ll_model.pth", map_location="cpu"))
    model.eval()
    graph = Graph.from_model(model)
    data = HFEAPDataset("mib-bench/ioi", model.tokenizer, split=a.split, task="ioi", model_name="interpbench")
    data.head(a.n)
    dl = data.to_dataloader(batch_size=min(a.n, a.batch))
    metric = partial(get_metric("logit_diff", "ioi", model.tokenizer, model), mean=False, loss=False)
    t0 = time.time()
    base = evaluate_baseline(model, dl, metric).numpy()
    names = list(graph.edges)
    eff = np.zeros((len(base), len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        graph.reset(empty=False)                             # every node and edge in (evaluate_graph prunes in place)
        graph.edges[name].in_graph = False
        eff[:, j] = evaluate_graph(model, graph, dl, metric, quiet=True, intervention="patching").numpy() - base
        if j % 100 == 0:
            print(f"{j}/{len(names)} edges, {time.time() - t0:.0f} s", flush=True)
    seconds = time.time() - t0
    np.savez(a.out, effects=eff, edges=np.array(names), clean=base, seconds=seconds)
    print(f"done: {len(names)} edges x {len(base)} prompts in {seconds:.0f} s; clean logit diff {base.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
