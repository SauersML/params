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
             ends do); then the REFINE edges of largest score each patched alone and ordered first by its own |change|. Every number
             a run of MIB's patching, computed by direct activation patching (equal to MIB's evaluate_graph without its
             pruning step, so each removes exactly the stated edges; test_mib_compare.py); no gradient anywhere.
  baselines  MIB's own attribute() with its methods (eap-ig-inputs, eap, ...), counterfactual ablation, as published
Writes OUT/<task>_<model>_<method>.json: CMD and CPR (absolute and signed rankings), the faithfulness curve, the
circuit sizes, seconds (ranking, per attribution example), and for measured every test's change. MIB's code: --mib DIR
(MIB-circuit-track, with its EAP-IG/src), run in MIB's environment.

  mib_compare.py prep TASK MODEL   (online) MIB's task data into the cache, and an ungated model copy aliased under
      Meta's name; every later run then goes with HF_HUB_OFFLINE=1 (huggingface_hub reads it at import)
  mib_compare.py rank TASK MODEL METHOD OUT [--n 32] [--refine 1000] [--batch 16] [--rows 32] [--test-n N]
      [--from PREV.json --order block|none|paths]   (measured: an earlier run's tests scored again, no new tests)
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
    """Put the ungated copy's snapshot in the HF cache under Meta's repository name (adag_compare.alias)."""
    from adag_compare import alias

    from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME
    alias(MODEL_NAME_TO_FULLNAME[model], WEIGHTS[model], [])


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


def node_hooks(cfg, nf: int, nb: int):
    """MIB's graph indices as TransformerLens hooks: writer f -> (output hook, head), reader b -> (hook of the value its
    input fully corrupted gives, head) and (its input hook, head) for one edge's addition."""
    H, NL = cfg.n_heads, cfg.n_layers

    def writer(f):
        if f == 0:
            return "hook_embed", None
        L, r = divmod(f - 1, H + 1)
        return (f"blocks.{L}.hook_mlp_out", None) if r == H else (f"blocks.{L}.attn.hook_z", r)

    def reader(b):  # (an MLP's input fully corrupted gives its corrupted output; a head's q, k or v input its corrupted q, k, v)
        if b == nb - 1:
            return "logits", None
        L, r = divmod(b, 3 * H + 1)
        return (f"blocks.{L}.hook_mlp_out", None) if r == 3 * H else (f"blocks.{L}.attn.hook_{'qkv'[r // H]}", r % H)

    def reader_in(b):
        if b == nb - 1:
            return f"blocks.{NL - 1}.hook_resid_post", None
        L, r = divmod(b, 3 * H + 1)
        return (f"blocks.{L}.hook_mlp_in", None) if r == 3 * H else (f"blocks.{L}.hook_{'qkv'[r // H]}_input", r % H)
    return writer, reader, reader_in


def measured(model, graph, dl, metric, refine: int, log=print, limit: int | None = None, rows: int = 32) -> dict:
    """The measured ranking (module docstring) into graph.scores; returns every test's change. Each test is MIB's
    patching run computed directly (test_mib_compare.py checks equality with evaluate_graph): a writer with all its
    outgoing edges patched is its output replaced by the counterfactual run's; a reader with all its incoming edges
    patched receives the counterfactual run's residual stream, so its q, k, v (an MLP: its output) are the
    counterfactual run's; one edge adds (counterfactual - clean output of the writer) to the reader's input. Tests run
    ROWS rows at a time (each test on its own copy of the batch), with TransformerLens' per-head results off (MIB's
    evaluation needs them; on Llama-3.1-8B they cost [rows, positions, heads, d_head, d_model] a layer) and its per-head
    q, k, v inputs on only for the edge tests."""
    import torch

    from eap.utils import tokenize_plus
    cfg = model.cfg
    nf, nb = graph.n_forward, graph.n_backward
    real = graph.real_edge_mask
    writer, reader, reader_in = node_hooks(cfg, nf, nb)
    names = {"hook_embed"} | {f"blocks.{L}.{h}" for L in range(cfg.n_layers)
                              for h in ("attn.hook_z", "attn.hook_q", "attn.hook_k", "attn.hook_v", "hook_mlp_out")}
    saved = (cfg.use_attn_result, cfg.use_split_qkv_input)
    cfg.use_attn_result = cfg.use_split_qkv_input = False
    batches = []
    for clean, corrupted, label in dl:
        ct, am, il, _ = tokenize_plus(model, clean)
        xt = tokenize_plus(model, corrupted)[0]
        with torch.inference_mode():
            xl, xc = model.run_with_cache(xt, attention_mask=am, names_filter=lambda n: n in names)
            cl, cc = model.run_with_cache(ct, attention_mask=am, names_filter=lambda n: n in names)
        batches.append((ct, am, il, label, cc, xc, metric(cl, None, il, label).float(), metric(xl, None, il, label).float()))
    n_ex = sum(len(b[2]) for b in batches)
    full = float(sum(b[6].sum() for b in batches)) / n_ex
    m_corrupt = float(sum(b[7].sum() for b in batches)) / n_ex

    def delta(f, cc, xc):  # counterfactual - clean output of writer f, [batch, pos, d_model]
        hook, h = writer(f)
        if h is None:
            return xc[hook] - cc[hook]
        L = int(hook.split(".")[1])
        return (xc[hook][:, :, h] - cc[hook][:, :, h]) @ model.blocks[L].attn.W_O[h]

    def rep(x, G):
        return x.repeat(G, *[1] * (x.dim() - 1)) if torch.is_tensor(x) else list(x) * G

    def run(tests) -> list[float]:
        """Each test a list of (kind, hook, head, writer): kind "set" puts the counterfactual cache's value at the hook
        (the head's slice), "add" adds the writer's delta at the hook. Returns each test's mean metric."""
        out = [0.0] * len(tests)
        for ct, am, il, label, cc, xc, _, _ in batches:
            B = ct.shape[0]
            G = max(1, rows // B)
            for g0 in range(0, len(tests), G):
                grp = tests[g0:g0 + G]
                by = {}
                for g, spec in enumerate(grp):
                    for kind, hook, h, f in spec:
                        by.setdefault(hook, []).append((g, kind, h, f))

                def fn(act, hook, n=len(grp)):
                    a = act.view(n, B, *act.shape[1:])
                    for g, kind, h, f in by[hook.name]:
                        if kind == "set":
                            if h is None:
                                a[g] = xc[hook.name]
                            else:
                                a[g, :, :, h] = xc[hook.name][:, :, h]
                        elif h is None:
                            a[g] += delta(f, cc, xc)
                        else:
                            a[g, :, :, h] += delta(f, cc, xc)
                    return act
                with torch.inference_mode(), model.hooks(fwd_hooks=[(k, fn) for k in by]):
                    lg = model(rep(ct, len(grp)), attention_mask=rep(am, len(grp)))
                m = metric(lg, None, rep(il, len(grp)), rep(label, len(grp))).float().view(len(grp), B).sum(1)
                for g in range(len(grp)):
                    out[g0 + g] += float(m[g])
        return [o / n_ex for o in out]

    t0 = time.time()
    lim = (lambda n: n) if limit is None else (lambda n: min(n, limit))  # (limit: a smoke run's first few)
    fs = [f for f in range(lim(nf)) if real[f].any()]
    writers = torch.zeros(nf)
    for f, m in zip(fs, run([[("set", *writer(f), None)] for f in fs])):
        writers[f] = abs(full - m)
    log(f"writers {len(fs)}, {time.time() - t0:.0f} s")
    bs = [b for b in range(lim(nb)) if real[:, b].any() and reader(b)[0] not in ("logits",) and "mlp" not in reader(b)[0]]
    readers = torch.zeros(nb)
    for b, m in zip(bs, run([[("set", *reader(b), None)] for b in bs])):
        readers[b] = abs(full - m)
    for b in range(lim(nb)):  # (an MLP's reader test is its writer test; the logits' is the counterfactual run)
        hook, _ = reader(b)
        if hook == "logits":
            readers[b] = abs(full - m_corrupt)
        elif "mlp" in hook:
            readers[b] = writers[1 + int(hook.split(".")[1]) * (cfg.n_heads + 1) + cfg.n_heads]
    log(f"readers {len(bs)}, {time.time() - t0:.0f} s")
    score = torch.minimum(writers[:, None], readers[None, :]) * real
    top = score.flatten().argsort(descending=True)[:refine].tolist()
    edges = [divmod(e, nb) for e in top]
    cfg.use_split_qkv_input = True  # (an edge into a head's q, k or v adds at its own input)
    single = []
    for (f, b), m in zip(edges, run([[("add", *reader_in(b), f)] for f, b in edges])):
        single.append([f, b, float(score[f, b]), abs(full - m)])
    log(f"single edges {len(edges)}, {time.time() - t0:.0f} s")
    cfg.use_attn_result, cfg.use_split_qkv_input = saved
    rec = {"full": full, "writers": writers.tolist(), "readers": readers.tolist(), "single": single,
           "tests": len(fs) + len(bs) + len(edges), "seconds_tests": time.time() - t0}
    graph.scores[:] = measured_scores(real, rec).to(graph.scores.dtype)
    graph.in_graph[:] = real
    return rec


def measured_scores(real, rec: dict, order: str = "block"):
    """The measured ranking's edge scores from its tests: min(writer, reader) on every real edge; order "block" puts the
    refined edges (the top REFINE by that score) first, ordered by their own single-edge changes, "none" keeps the
    min score alone, and "paths" takes also the writer's node's strongest input and the reader's node's writer score
    into the min (a node is kept only where both its sides matter, so every edge of a prefix has an input-side and a
    logits-side neighbour at least as strong and MIB's pruning removes none: no refined order). (In place of the refined scores among the others, 7859624's rule: a refined edge's single change is
    smaller than its min score when other edges carry the same signal, so refined edges fell below unrefined ones and
    a circuit of the top 1000 edges lost its paths to the logits: F 0.01 at 1010 edges on Llama MCQA.)"""
    import torch

    w, r = torch.tensor(rec["writers"]), torch.tensor(rec["readers"])
    if order == "paths":  # each end's node needs its own other side too, so every prefix survives MIB's pruning
        nf, nb = len(w), len(r)
        L = (3 * nf - nb - 2) // 2
        H = (nf - 1) // L - 1
        big = float(max(w.max(), r.max())) * 2 + 1
        inn = torch.full((nf,), big)  # a writer's strongest own input (the input node has none: no limit)
        node = torch.full((nb,), big)  # a reader's node's writer score (the logits: no limit)
        for layer in range(L):
            for h in range(H + 1):
                f = 1 + layer * (H + 1) + h
                bs = [layer * (3 * H + 1) + 3 * H] if h == H else [layer * (3 * H + 1) + q * H + h for q in range(3)]
                inn[f] = r[bs].max()
                node[bs] = w[f]
        score = torch.minimum(torch.minimum(w, inn)[:, None], torch.minimum(r, node)[None, :]) * real
        tie = float(score.max()) * 1e-6  # (an edge's own neighbours tie with it: the input's and the logits' edges go first)
        score[0] += tie * real[0]
        score[:, -1] += tie * real[:, -1]
        return score
    score = torch.minimum(w[:, None], r[None, :]) * real
    if order == "block" and rec["single"]:
        base = float(score.max()) + 1.0
        for f, b, _, d in rec["single"]:
            score[f, b] = base + d
    return score


def rank(task: str, mname: str, method: str, out: Path, n: int, refine: int, batch: int, test_n: int | None = None,
         log=print, limit: int | None = None, rows: int = 32, prev: Path | None = None, order: str = "block") -> dict:
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
    if method == "measured" and prev:  # the tests of an earlier measured run, scored again with ORDER
        rec["measured"] = json.loads(prev.read_text())["measured"]
        graph.scores[:] = measured_scores(graph.real_edge_mask, rec["measured"], order).to(graph.scores.dtype)
        rec.update(order=order, tests_from=str(prev))
    elif method == "measured":
        rec["measured"] = measured(model, graph, dl, partial(metric, mean=False, loss=False), refine, log, limit, rows)
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
    (out / f"{task}_{mname}_{method}{'' if order == 'block' or method != 'measured' else '-' + order}.json").write_text(json.dumps(rec))
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
    ap.add_argument("--rows", type=int, default=32, help="measured: rows a forward pass (tests x batch)")
    ap.add_argument("--from", dest="prev", type=Path, default=None, help="measured: an earlier run's JSON, its tests scored again")
    ap.add_argument("--order", default="block", choices=("block", "none", "paths"),
                    help="measured: the refined edges first (block), min scores alone (none), or node-coupled min scores (paths)")
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
    rank(task, mname, method, Path(out), a.n, a.refine, a.batch, a.test_n, log=lambda s: print(s, flush=True), limit=a.limit,
         rows=a.rows, prev=a.prev, order=a.order)


if __name__ == "__main__":
    main()
