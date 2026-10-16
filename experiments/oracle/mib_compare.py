"""MIB's circuit-localization track with a measured edge ranking beside the gradient ones (#2951; the lead's 10-15: our
measured search against Transluce's RelP, which reports CMD 0.08 on Llama-3.1-8B against EAP-IG-activations' 0.16).
MIB scores a ranking of every edge of its component graph (attention heads and MLPs; a head's q, k and v inputs apart),
by MIB's own code: the circuit of the top k% of the edges for k in .1% ... 100%, every edge outside it patched from the
counterfactual prompt (MIB's "patching"), faithfulness F = (m(C) - m(none)) / (m(model) - m(none)) with m MIB's logit
difference; CMD = the trapezoid area of |1 - F| over k on a linear scale (so circuits of 5% of the edges and more hold
95% of its weight), CPR = the area of F.
Rankings (the attribution set: the first N of MIB's train split):
  measured   every writer (a head or MLP's output, or the input) with all its outgoing edges patched, alone; every
             reader input (a head's q, k or v, an MLP's input, the logits) with all its incoming edges patched, alone;
             an edge's score the smaller of its writer's and its reader's |change in m| (an edge matters only where both
             ends do); then the REFINE edges of largest score each patched alone, its own |change| its score. Every number
             a run of MIB's patching; no gradient anywhere. (MIB's evaluate_graph prunes the graph to what reaches the
             logits first; the measured runs skip that step, so each removes exactly the stated edges.)
  baselines  MIB's own attribute() with its methods (eap-ig-inputs, eap, ...), counterfactual ablation, as published
Writes OUT/<task>_<model>_<method>.json: CMD and CPR (absolute and signed rankings), the faithfulness curve, the
circuit sizes, seconds (ranking, per attribution example), and for measured every test's change. MIB's code: --mib DIR
(MIB-circuit-track, with its EAP-IG/src), run in MIB's environment.

  mib_compare.py prep TASK MODEL   (online) MIB's task data into the cache, and an ungated model copy aliased under
      Meta's name; every later run then goes with HF_HUB_OFFLINE=1 (huggingface_hub reads it at import)
  mib_compare.py rank TASK MODEL METHOD OUT [--n 32] [--refine 1000] [--batch 16] [--test-n N]
  mib_compare.py table OUT [...]
MODEL: MIB's names (qwen2.5, llama3); llama3's weights from WEIGHTS (an ungated copy of meta-llama/Llama-3.1-8B, the
same files) placed in the HF cache under Meta's name (alias_weights, by prep).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

MIB = Path(os.environ.get("MIB_DIR", Path.home() / "mpd-data" / "ext" / "MIB-circuit-track"))
WEIGHTS = {"llama3": "NousResearch/Meta-Llama-3.1-8B"}
REFINE = 1000


def alias_weights(model: str) -> None:
    """Put the ungated copy's snapshot in the HF cache under Meta's repository name (refs/main to the same snapshot)."""
    from huggingface_hub import snapshot_download

    from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME
    src = Path(snapshot_download(WEIGHTS[model]))
    official = MODEL_NAME_TO_FULLNAME[model]
    root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")) / f"models--{official.replace('/', '--')}"
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    (root / "refs").mkdir(exist_ok=True)
    dst = root / "snapshots" / src.name
    if not dst.exists():
        dst.symlink_to(src, target_is_directory=True)
    (root / "refs" / "main").write_text(src.name)


def prefetch(task: str) -> None:
    """MIB's task data into the HF cache before the model's weights put the hub offline."""
    from datasets import load_dataset

    from MIB_circuit_track.utils import TASKS_TO_HF_NAMES
    for split in ("train", "test"):
        if task == "mcqa":
            load_dataset(f"mib-bench/{TASKS_TO_HF_NAMES[task]}", "4_answer_choices", split=split)
        else:
            load_dataset(f"mib-bench/{TASKS_TO_HF_NAMES[task]}", split=split)


def load(model: str):
    import torch
    from transformer_lens import HookedTransformer

    from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME
    m = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model], attn_implementation="eager", torch_dtype=torch.bfloat16)
    m.cfg.use_split_qkv_input = True  # (MIB's settings)
    m.cfg.use_attn_result = True
    m.cfg.use_hook_mlp_in = True
    m.cfg.ungroup_grouped_query_attention = True
    return m


def data(model, task: str, mname: str, split: str, n: int | None, batch: int):
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.utils import TASKS_TO_HF_NAMES
    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF_NAMES[task]}", model.tokenizer, split=split, task=task, model_name=mname,
                      num_examples=n)
    if n is not None:
        ds.head(min(n, len(ds)))
    return ds.to_dataloader(batch_size=batch), len(ds)


def measured(model, graph, dl, metric, refine: int, log=print, limit: int | None = None) -> dict:
    """The measured ranking (module docstring) into graph.scores; returns every test's change."""
    import torch

    from eap.evaluate import evaluate_graph
    graph.prune = lambda *a, **k: None  # each run removes exactly the stated edges
    real = graph.real_edge_mask.clone()

    def m_with(cut) -> float:
        graph.in_graph[:] = real
        cut(graph.in_graph)
        graph.in_graph &= real
        return float(evaluate_graph(model, graph, dl, metric, quiet=True, intervention="patching").mean())
    full = m_with(lambda g: None)
    t0 = time.time()
    writers = torch.zeros(graph.n_forward)
    for f in range(graph.n_forward if limit is None else min(limit, graph.n_forward)):  # (limit: a smoke run's first few)
        if real[f].any():
            writers[f] = abs(full - m_with(lambda g, f=f: g[f].fill_(False)))
        if f % 100 == 0:
            log(f"writers {f}/{graph.n_forward}, {time.time() - t0:.0f} s")
    readers = torch.zeros(graph.n_backward)
    for b in range(graph.n_backward if limit is None else min(limit, graph.n_backward)):
        if real[:, b].any():
            readers[b] = abs(full - m_with(lambda g, b=b: g[:, b].fill_(False)))
        if b % 200 == 0:
            log(f"readers {b}/{graph.n_backward}, {time.time() - t0:.0f} s")
    score = torch.minimum(writers[:, None], readers[None, :]) * real
    top = score.flatten().argsort(descending=True)[:refine]
    single = []
    for k, e in enumerate(top.tolist()):
        f, b = divmod(e, graph.n_backward)
        d = abs(full - m_with(lambda g, f=f, b=b: g[f, b].fill_(False)))
        single.append([f, b, float(score[f, b]), d])
        score[f, b] = d
        if k % 100 == 0:
            log(f"single edges {k}/{len(top)}, {time.time() - t0:.0f} s")
    graph.scores[:] = score.to(graph.scores.dtype)
    graph.in_graph[:] = real
    del graph.prune  # (the instance's patch: MIB's own prune for the evaluation)
    return {"full": full, "writers": writers.tolist(), "readers": readers.tolist(), "single": single,
            "tests": int(real.any(1).sum() + real.any(0).sum() + len(top))}


def rank(task: str, mname: str, method: str, out: Path, n: int, refine: int, batch: int, test_n: int | None = None,
         log=print, limit: int | None = None) -> dict:
    import torch

    from eap.attribute import attribute
    from eap.graph import Graph
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    from MIB_circuit_track.metrics import get_metric
    model = load(mname)
    graph = Graph.from_model(model)
    dl, n_rank = data(model, task, mname, "train", n, batch)
    metric = get_metric("logit_diff", task, model.tokenizer, model)
    t0 = time.time()
    rec = {"task": task, "model": mname, "method": method, "rank_examples": n_rank, "edges": int(graph.real_edge_mask.sum())}
    if method == "measured":
        rec["measured"] = measured(model, graph, dl, partial(metric, mean=False, loss=False), refine, log, limit)
    else:
        attribute(model, graph, dl, partial(metric, mean=True, loss=True), method, "patching", ig_steps=5,
                  intervention_dataloader=dl)
    rec["seconds_rank"] = time.time() - t0
    rec["seconds_per_example"] = rec["seconds_rank"] / max(n_rank, 1)
    tdl, n_test = data(model, task, mname, "test", test_n, batch)
    rec["test_examples"] = n_test
    for absolute in (True, False):
        if not absolute and float(graph.scores[graph.real_edge_mask].min()) >= 0:  # (no negative score: the same ranking)
            rec["abs_False"] = rec["abs_True"]
            continue
        t1 = time.time()
        counts, cpr, cmd, avg, faith = evaluate_area_under_curve(model, graph, tdl, partial(metric, mean=False, loss=False),
                                                                 level="edge", absolute=absolute, quiet=True)
        rec[f"abs_{absolute}"] = {"CMD": cmd, "CPR": cpr, "faithfulness": faith, "weighted_edge_counts": counts,
                                  "seconds_eval": time.time() - t1}
        log(f"{task} {mname} {method} absolute={absolute}: CMD {cmd:.3f} CPR {cpr:.3f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task}_{mname}_{method}.json").write_text(json.dumps(rec))
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rec


def table(paths) -> str:
    rows = []
    for p in paths:
        for f in sorted(Path(p).glob("*.json")) if Path(p).is_dir() else [Path(p)]:
            r = json.loads(f.read_text())
            best = min((r[k]["CMD"], k) for k in ("abs_True", "abs_False"))
            rows.append(f"{r['task']:24}{r['model']:10}{r['method']:16}{r['abs_True']['CMD']:>10.3f}{r['abs_False']['CMD']:>10.3f}"
                        f"{r[best[1]]['CPR']:>10.3f}{r['seconds_per_example']:>12.1f}{r.get('measured', {}).get('tests', 0):>9}")
    head = f"{'task':24}{'model':10}{'method':16}{'CMD abs':>10}{'CMD sign':>10}{'CPR':>10}{'s/example':>12}{'tests':>9}"
    return "\n".join([head] + rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prep", "rank", "table"))
    ap.add_argument("args", nargs="+")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--refine", type=int, default=REFINE)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--test-n", type=int, default=None, help="MIB's test split's first examples only (a smoke run)")
    ap.add_argument("--limit", type=int, default=None, help="the first writers and readers only (a smoke run)")
    a = ap.parse_args()
    sys.path[:0] = [str(MIB), str(MIB / "EAP-IG" / "src")]
    if a.cmd == "table":
        print(table(a.args))
        return
    if a.cmd == "prep":
        task, mname = a.args
        prefetch(task)
        if mname in WEIGHTS:
            alias_weights(mname)
        return
    task, mname, method, out = a.args
    rank(task, mname, method, Path(out), a.n, a.refine, a.batch, a.test_n, log=lambda s: print(s, flush=True), limit=a.limit)


if __name__ == "__main__":
    main()
