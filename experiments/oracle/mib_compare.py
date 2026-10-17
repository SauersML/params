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

  mib_compare.py prep TASK MODEL   (online) MIB's task data and the model into the cache (an ungated copy aliased under
      the official name); every later run then goes with HF_HUB_OFFLINE=1 (huggingface_hub reads it at import)
  mib_compare.py rank TASK MODEL METHOD OUT [--n 32] [--refine 1000] [--batch 16] [--eval-batch B] [--rows 32] [--test-n N]
      [--fix 0.02] [--curves all|eap]   (select: edgefix circuits up to that size; which proposals get test curves)
      [--from PREV.json --order block|none|paths]   (measured: an earlier run's tests scored again, no new tests)
  mib_compare.py table OUT [...]
  mib_compare.py finder TASK MODEL OUT [--n 32]   circuit_finder's circuits of MIB's first train pairs (Hugging Face model)
  mib_compare.py convert TASK MODEL OUT [IMPORTANCES.pt ...] [--test-n N]   those circuits as MIB circuits on its test
      split, each ranking's top-k at the same edge counts beside them (finder_circuits)
Every ranking's edge scores go to OUT/submission/METHOD/TASK_MODEL/importances.pt, the leaderboard's layout.
MODEL: MIB's names (gpt2, qwen2.5, gemma2, llama3); llama3's and gemma2's weights from WEIGHTS (ungated copies of
meta-llama/Llama-3.1-8B and google/gemma-2-2b) placed in the HF cache under the official names (alias_weights, by prep).
--n -1: MIB's own attribution examples (replicate.sh: 1000 for ioi, mcqa's whole train split, 100 for the rest).
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
WEIGHTS = {"llama3": "NousResearch/Meta-Llama-3.1-8B", "gemma2": "unsloth/gemma-2-2b"}  # (ungated copies: the same weights,
# Gemma's in bfloat16, the dtype MIB loads it in)
MIB_N = {"ioi": 1000, "mcqa": None}  # MIB's replicate.sh attribution examples (mcqa: the whole train split; other tasks 100)
REFINE = 1000


def alias_weights(model: str) -> None:
    """Put the ungated copy's snapshot in the HF cache under Meta's repository name (adag_compare.alias)."""
    from adag_compare import alias

    from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME
    alias(MODEL_NAME_TO_FULLNAME[model], WEIGHTS[model], [])


def prefetch(task: str) -> None:  # (interpbench's files come at load)
    """MIB's task data into the HF cache before the model's weights put the hub offline."""
    from datasets import load_dataset

    from MIB_circuit_track.utils import TASKS_TO_HF_NAMES
    for split in ("train", "test"):
        if task == "mcqa":
            load_dataset(f"mib-bench/{TASKS_TO_HF_NAMES[task]}", "4_answer_choices", split=split)
        else:
            load_dataset(f"mib-bench/{TASKS_TO_HF_NAMES[task]}", split=split)


def load(model: str):
    """The model as MIB's run_attribution.py and run_evaluation.py load it: qwen2.5, gemma2 and llama3 in bfloat16 with
    eager attention, gpt2 in TransformerLens' default float32."""
    import torch
    from transformer_lens import HookedTransformer

    from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # (never the Mac's GPU, which TransformerLens would pick)
    if model == "interpbench":  # (run_attribution.load_interpbench_model, on DEV: MIB's ioi_all_splits checkpoint)
        import pickle

        from huggingface_hub import hf_hub_download
        from transformer_lens import HookedTransformerConfig
        cfg = pickle.load(open(hf_hub_download("mib-bench/interpbench", filename="ll_model_cfg.pkl"), "rb"))
        cfg = HookedTransformerConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
        cfg.device = dev
        m = HookedTransformer(cfg)
        m.load_state_dict(torch.load(hf_hub_download("mib-bench/interpbench", subfolder="ioi_all_splits",
                                                     filename="ll_model_100_100_80.pth"), map_location=dev))
    elif model in ("qwen2.5", "gemma2", "llama3"):
        m = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model], attn_implementation="eager", torch_dtype=torch.bfloat16,
                                              device=dev)
    else:
        m = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model], device=dev)
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


def prepare(model, dl, metric, names):
    """Each batch of a dataloader as (tokens, attention mask, lengths, labels, clean cache, counterfactual cache, clean
    metric, counterfactual metric), the caches holding NAMES; the number of examples and both runs' mean metrics."""
    import torch

    from eap.utils import tokenize_plus
    batches = []
    for clean, corrupted, label in dl:
        ct, am, il, _ = tokenize_plus(model, clean)
        xt = tokenize_plus(model, corrupted)[0]
        with torch.inference_mode():
            xl, xc = model.run_with_cache(xt, attention_mask=am, names_filter=lambda n: n in names)
            cl, cc = model.run_with_cache(ct, attention_mask=am, names_filter=lambda n: n in names)
        batches.append((ct, am, il, label, cc, xc, metric(cl, None, il, label).float(), metric(xl, None, il, label).float()))
    n_ex = sum(len(b[2]) for b in batches)
    return batches, n_ex, float(sum(b[6].sum() for b in batches)) / n_ex, float(sum(b[7].sum() for b in batches)) / n_ex


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

    cfg = model.cfg
    nf, nb = graph.n_forward, graph.n_backward
    real = graph.real_edge_mask
    writer, reader, reader_in = node_hooks(cfg, nf, nb)
    names = {"hook_embed"} | {f"blocks.{L}.{h}" for L in range(cfg.n_layers)
                              for h in ("attn.hook_z", "attn.hook_q", "attn.hook_k", "attn.hook_v", "hook_mlp_out", "hook_resid_pre")}
    saved = (cfg.use_attn_result, cfg.use_split_qkv_input)
    cfg.use_attn_result = cfg.use_split_qkv_input = False
    batches, n_ex, full, m_corrupt = prepare(model, dl, metric, names)

    def delta(f, cc, xc):  # counterfactual - clean output of writer f, [batch, pos, d_model]
        hook, h = writer(f)
        if h is None:
            return xc[hook] - cc[hook]
        L = int(hook.split(".")[1])
        a = model.blocks[L].attn
        d = (xc[hook][:, :, h] - cc[hook][:, :, h]) @ a.W_O[h]
        if not cfg.use_normalization_before_and_after:
            return d
        # (Gemma-2: the heads' summed output passes a norm before the residual stream, so the edge's change is that norm's
        # output with this head's result counterfactual minus with every head's clean, as MIB's evaluate_graph computes it)
        s = torch.einsum("bphd,hdm->bpm", cc[hook], a.W_O) + a.b_O
        return model.blocks[L].ln1_post(s + d) - model.blocks[L].ln1_post(s)

    def start(spec) -> int:  # the lowest layer a test patches (the input node: 0, from the tokens)
        return min(0 if hook == "hook_embed" else int(hook.split(".")[1]) for _, hook, _, _ in spec)

    def rep(x, G):
        return x.repeat(G, *[1] * (x.dim() - 1)) if torch.is_tensor(x) else list(x) * G

    def run(tests) -> list[float]:
        """Each test a list of (kind, hook, head, writer): kind "set" puts the counterfactual cache's value at the hook
        (the head's slice), "add" adds the writer's delta at the hook. Returns each test's mean metric."""
        out = [0.0] * len(tests)
        order = sorted(range(len(tests)), key=lambda i: start(tests[i]))  # (a forward starts at its tests' lowest layer,
        # from the clean run's residual there: the layers below see nothing patched)
        for ct, am, il, label, cc, xc, _, _ in batches:
            B = ct.shape[0]
            G = max(1, rows // B)
            for g0 in range(0, len(tests), G):
                idx = order[g0:g0 + G]
                grp = [tests[i] for i in idx]
                s0 = min(start(spec) for spec in grp)
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
                    if s0 == 0:
                        lg = model(rep(ct, len(grp)), attention_mask=rep(am, len(grp)))
                    else:
                        lg = model(rep(cc[f"blocks.{s0}.hook_resid_pre"], len(grp)), start_at_layer=s0,
                                   attention_mask=rep(am, len(grp)))
                m = metric(lg, None, rep(il, len(grp)), rep(label, len(grp))).float().view(len(grp), B).sum(1)
                for g, i in enumerate(idx):
                    out[i] += float(m[g])
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
    if refine > 0 and 0.001 * int(real.sum()) >= refine:  # (the refined edges reorder only within the top REFINE, inside
        log(f"refine skipped: MIB's smallest circuit, {int(0.001 * int(real.sum()))} edges, holds every refined edge")  # every MIB size here)
        refine = 0
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


def node_sets(model, batches, n_ex: int, masks, metric, rows: int = 32) -> list[float]:
    """MIB's patching run of node circuits: mask [tests, n_forward] says which writers (the input, every head, every
    MLP) are in each circuit, whose every edge between the input, its nodes and the logits is kept; every other writer's
    output is the counterfactual run's for every reader. That is the clean run with each writer outside the circuit set
    to its counterfactual output (a reader outside it computes nothing any kept node reads). A forward starts at its
    circuits' lowest layer with a writer out, from the clean run's residual there (when the caches hold it). Returns
    each mean metric."""
    import torch

    cfg = model.cfg
    H, NL = cfg.n_heads, cfg.n_layers
    head_ix = torch.tensor([[1 + L * (H + 1) + h for h in range(H)] for L in range(NL)])
    mlp_ix = torch.tensor([1 + L * (H + 1) + H for L in range(NL)])
    masks = masks.bool()
    layer = torch.tensor([0] + [L for L in range(NL) for _ in range(H + 1)])  # (each writer's layer; the input's 0)
    low = torch.where(~masks, layer[None, :], NL).min(1).values
    order = low.argsort().tolist()
    out = [0.0] * len(masks)
    for ct, am, il, label, cc, xc, _, _ in batches:
        B = ct.shape[0]
        G = max(1, rows // B)
        for g0 in range(0, len(masks), G):
            idx = order[g0:g0 + G]
            mk = masks[idx].to(ct.device)
            n = len(mk)
            s0 = int(low[idx].min())
            s0 = 0 if s0 >= NL or (~masks[idx, 0]).any() or f"blocks.{s0}.hook_resid_pre" not in cc else s0

            def fn(act, hook):
                a = act.view(n, B, *act.shape[1:])
                if hook.name == "hook_embed":
                    keep = mk[:, 0].view(n, 1, 1, 1)
                elif hook.name.endswith("attn.hook_z"):
                    keep = mk[:, head_ix[int(hook.name.split(".")[1])]].view(n, 1, 1, H, 1)
                else:
                    keep = mk[:, mlp_ix[int(hook.name.split(".")[1])]].view(n, 1, 1, 1)
                return torch.where(keep, a, xc[hook.name].unsqueeze(0)).view(act.shape)
            hooks = [("hook_embed", fn)] + [(f"blocks.{L}.{h}", fn) for L in range(NL) for h in ("attn.hook_z", "hook_mlp_out")]
            rep = lambda x: x.repeat(n, *[1] * (x.dim() - 1)) if torch.is_tensor(x) else list(x) * n  # noqa: E731
            with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
                if s0 == 0:
                    lg = model(rep(ct), attention_mask=rep(am))
                else:
                    lg = model(rep(cc[f"blocks.{s0}.hook_resid_pre"]), start_at_layer=s0, attention_mask=rep(am))
            m = metric(lg, None, rep(il), rep(label)).float().view(n, B).sum(1)
            for g, i in enumerate(idx):
                out[i] += float(m[g])
    return [o / n_ex for o in out]


def node_edges(graph, keep):
    """The edges of the node circuit KEEP (writers, [n_forward] bool): every real edge from a kept writer to a reader of
    a kept node or the logits; with their count."""
    import torch

    cfg = graph.cfg
    H, NL = cfg["n_heads"], cfg["n_layers"]
    rd = torch.zeros(graph.n_backward, dtype=torch.bool)
    rd[-1] = True
    for L in range(NL):
        for h in range(H):
            rd[[L * (3 * H + 1) + q * H + h for q in range(3)]] = keep[1 + L * (H + 1) + h]
        rd[L * (3 * H + 1) + 3 * H] = keep[1 + L * (H + 1) + H]
    e = graph.real_edge_mask & keep[:, None] & rd[None, :]
    return e, int(e.sum())


def nodes(model, graph, dl, metric, log=print, rows: int = 32) -> dict:
    """The measured ranking from its writer tests alone (each writer's output set to its counterfactual alone, through
    node_sets): an edge scores the smaller change of its writer and of its reader's node as a writer (a head's q, k
    and v alike; the logits never removed). About a fifth of measured's tests on Llama-3.1 (no reader or edge tests)."""
    import torch

    cfg = model.cfg
    nf, H = graph.n_forward, cfg.n_heads
    names = {"hook_embed"} | {f"blocks.{L}.{h}" for L in range(cfg.n_layers) for h in ("attn.hook_z", "hook_mlp_out", "hook_resid_pre")}
    saved = (cfg.use_attn_result, cfg.use_split_qkv_input)
    cfg.use_attn_result = cfg.use_split_qkv_input = False
    t0 = time.time()
    batches, n_ex, full, m_corrupt = prepare(model, dl, metric, names)
    alone = ~torch.eye(nf, dtype=torch.bool)
    noise = torch.tensor([abs(full - m) for m in node_sets(model, batches, n_ex, alone, metric, rows)])
    rd = torch.full((graph.n_backward,), float(noise.max()) + 1)
    for L in range(cfg.n_layers):
        for h in range(H):
            rd[[L * (3 * H + 1) + q * H + h for q in range(3)]] = noise[1 + L * (H + 1) + h]
        rd[L * (3 * H + 1) + 3 * H] = noise[1 + L * (H + 1) + H]
    graph.scores[:] = torch.minimum(noise[:, None], rd[None, :]) * graph.real_edge_mask
    cfg.use_attn_result, cfg.use_split_qkv_input = saved
    log(f"nodes: {nf} writers alone, {time.time() - t0:.0f} s")
    return {"full": full, "counterfactual": m_corrupt, "noise": noise.tolist(), "tests": nf, "seconds_tests": time.time() - t0}


def eliminate(model, graph, dl, metric, log=print, rows: int = 32, cands: int = 8, delta: float = 0.02,
              order: str = "noise", noise=None) -> dict:
    """The finder's prune at MIB's units (node circuits, every decision a measured run of node_sets on the attribution
    examples): from the whole model, blocks of writers are removed in rounds. A round measures CANDS consecutive blocks
    of the remaining writers taken in ORDER (noise: each writer's output patched alone, the smallest change first) and
    removes the block whose circuit's faithfulness F = (m - m_counterfactual) / (m_clean - m_counterfactual) lies
    closest to 1; when even that block moves F by more than DELTA the blocks halve (down to one writer, removed even
    so: the circuit must shrink), and after DELTA-quiet rounds they double back. A node removed later ranks higher; an
    edge's score is the earlier removal time of its two ends (the logits never removed), so MIB's top-k is the node
    circuit at that step plus edges of the next node out. NOISE: the writer tests' changes when already measured
    (nodes()). Writes graph.scores; returns the removal record."""
    import torch

    cfg = model.cfg
    nf = graph.n_forward
    names = {"hook_embed"} | {f"blocks.{L}.{h}" for L in range(cfg.n_layers) for h in ("attn.hook_z", "hook_mlp_out", "hook_resid_pre")}
    saved = (cfg.use_attn_result, cfg.use_split_qkv_input)
    cfg.use_attn_result = cfg.use_split_qkv_input = False
    t0 = time.time()
    batches, n_ex, full, m_corrupt = prepare(model, dl, metric, names)
    gap = full - m_corrupt
    F = lambda m: (m - m_corrupt) / gap  # noqa: E731
    rec = {"full": full, "counterfactual": m_corrupt, "tests": 0}
    if noise is not None:
        noise = torch.tensor(noise)
        rec["noise"] = noise.tolist()
    elif order == "noise":  # each writer alone set to its counterfactual output
        alone = torch.ones(nf, nf, dtype=torch.bool)
        alone[torch.arange(nf), torch.arange(nf)] = False
        ms = node_sets(model, batches, n_ex, alone, metric, rows)
        noise = torch.tensor([abs(full - m) for m in ms])
        rec["noise"] = noise.tolist()
        rec["tests"] += nf
    else:
        raise ValueError(order)
    log(f"order: {nf} writers alone, {time.time() - t0:.0f} s")
    keep = torch.ones(nf, dtype=torch.bool)
    rest = [int(i) for i in noise.argsort()]  # (the writers still in, smallest change first)
    f_cur, size = 1.0, max(1, len(rest) // 64)
    steps = []  # (removed writers, F after)
    quiet = 0
    while rest:
        blocks = [rest[i:i + size] for i in range(0, min(len(rest), cands * size), size)]
        ms = []
        for b in blocks:
            k = keep.clone()
            k[b] = False
            ms.append(k)
        fs = [F(m) for m in node_sets(model, batches, n_ex, torch.stack(ms), metric, rows)]
        rec["tests"] += len(blocks)
        j = min(range(len(blocks)), key=lambda i: abs(fs[i] - 1))
        if abs(fs[j] - f_cur) > delta and size > 1:
            size = max(1, size // 2)
            quiet = 0
            continue
        keep[blocks[j]] = False
        rest = [w for w in rest if keep[w]]
        steps.append((blocks[j], fs[j]))
        quiet = quiet + 1 if abs(fs[j] - f_cur) <= delta else 0
        f_cur = fs[j]
        if quiet >= 4:
            size, quiet = min(size * 2, max(1, len(rest) // 64)), 0
    log(f"eliminated in {len(steps)} steps, {rec['tests']} tests, {time.time() - t0:.0f} s")
    when = torch.zeros(nf)  # (removal step: later is more important)
    for t, (b, _) in enumerate(steps):
        for w in sorted(b, key=lambda w: -float(noise[w])):  # (within a block, the larger change ranks higher)
            when[w] = t + float(noise[w]) / (float(noise.max()) + 1e-9) * 0.5
    big = float(len(steps) + 1)
    rd = torch.full((graph.n_backward,), big)
    H = cfg.n_heads
    for L in range(cfg.n_layers):
        for h in range(H):
            rd[[L * (3 * H + 1) + q * H + h for q in range(3)]] = when[1 + L * (H + 1) + h]
        rd[L * (3 * H + 1) + 3 * H] = when[1 + L * (H + 1) + H]
    graph.scores[:] = (torch.minimum(when[:, None], rd[None, :]) + 1) * graph.real_edge_mask
    cfg.use_attn_result, cfg.use_split_qkv_input = saved
    rec["steps"] = [[b, f] for b, f in steps]
    rec["seconds_tests"] = time.time() - t0
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
         log=print, limit: int | None = None, rows: int = 32, prev: Path | None = None, order: str = "block",
         eval_batch: int | None = None, fix_max: float = 0.0, curves: str = "all") -> dict:
    import torch

    from eap.attribute import attribute
    from eap.graph import Graph
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    from MIB_circuit_track.metrics import get_metric
    model = load(mname)
    graph = Graph.from_model(model)
    dl, n_rank = data(model, task, mname, "train", MIB_N.get(task, 100) if n < 0 else n, batch)
    metric = get_metric("logit_diff", task, model.tokenizer, model)
    t0 = time.time()
    rec = {"task": task, "model": mname, "method": method, "rank_examples": n_rank, "edges": int(graph.real_edge_mask.sum())}
    if method == "measured" and prev:  # the tests of an earlier measured run, scored again with ORDER
        rec["measured"] = json.loads(prev.read_text())["measured"]
        graph.scores[:] = measured_scores(graph.real_edge_mask, rec["measured"], order).to(graph.scores.dtype)
        rec.update(order=order, tests_from=str(prev))
    elif method == "measured":
        rec["measured"] = measured(model, graph, dl, partial(metric, mean=False, loss=False), refine, log, limit, rows)
    elif method == "eliminate":
        rec["eliminate"] = eliminate(model, graph, dl, partial(metric, mean=False, loss=False), log, rows)
    elif method == "nodes":
        rec["nodes"] = nodes(model, graph, dl, partial(metric, mean=False, loss=False), log, rows)
    elif method.startswith("select"):
        return select(model, task, mname, method, out, dl, n_rank, metric, refine, rows, test_n, eval_batch or batch, log,
                      fix_max, curves)
    else:
        attribute(model, graph, dl, partial(metric, mean=True, loss=True), method, "patching", ig_steps=5,
                  intervention_dataloader=dl)
    rec["seconds_rank"] = time.time() - t0
    rec["seconds_per_example"] = rec["seconds_rank"] / max(n_rank, 1)
    name = f"{method}{'' if order == 'block' or method != 'measured' else '-' + order}"
    sub = out / "submission" / name / f"{task.replace('_', '-')}_{mname}"  # (MIB's leaderboard layout: one folder per
    sub.mkdir(parents=True, exist_ok=True)                                  # task and model, the edge scores in it)
    graph.to_pt(str(sub / "importances.pt"))
    if mname == "interpbench":  # (MIB scores InterpBench by its edges against the reference circuit, not faithfulness)
        rec.update(interpbench_auroc(graph))
        log(f"{task} {mname} {name}: AUROC {rec['AUROC']:.3f}")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{task}_{mname}_{name}.json").write_text(json.dumps(rec))
        return rec
    tdl, n_test = data(model, task, mname, "test", test_n, eval_batch or batch)
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
    (out / f"{task}_{mname}_{name}.json").write_text(json.dumps(rec))
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rec


def interpbench_auroc(graph) -> dict:
    """MIB's InterpBench score of a ranking: run_evaluation's evaluate_area_under_roc against the reference circuit
    (mib-bench/interpbench interpbench_graph.json), the area by print_results' area_under_curve."""
    from huggingface_hub import hf_hub_download

    from eap.graph import Graph
    from MIB_circuit_track.evaluation import evaluate_area_under_roc
    ref = Graph.from_json(hf_hub_download("mib-bench/interpbench", filename="interpbench_graph.json"))
    d = evaluate_area_under_roc(ref, graph)
    X, Y = d["FPR"], d["TPR"]
    auroc = sum((X[i + 1] / X[-1] - X[i] / X[-1]) * (Y[i + 1] + Y[i]) / 2 for i in range(len(X) - 1))
    return {"AUROC": auroc, "roc": d}


PERCENTAGES = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)  # (MIB's circuit sizes, evaluate_area_under_curve's)


def areas(faith) -> tuple[float, float]:
    """MIB's CPR and CMD of a faithfulness curve over PERCENTAGES (evaluate_area_under_curve's trapezoids)."""
    cpr = cmd = 0.0
    for i in range(len(faith) - 1):
        w = PERCENTAGES[i + 1] - PERCENTAGES[i]
        cpr += w * (faith[i] + faith[i + 1]) / 2
        cmd += w * (abs(1 - faith[i]) + abs(1 - faith[i + 1])) / 2
    return cpr, cmd


def edgefix(model, g, loader, m1, base, none, k: int, scores, log=print, blocks: int = 32) -> tuple:
    """A circuit of at most K edges pruned by measurement in its own sparse context (the lead's small-budget fix): from
    the 2K real edges of largest |SCORES| (an attribution's proposal), rounds measure the circuit without each of up to
    BLOCKS blocks of its edges (weakest proposal scores first, MIB's evaluate_graph on LOADER with its pruning) and
    drop the blocks whose removal leaves faithfulness F nearest 1, half the excess edges a round, until K or fewer
    remain. Returns (F, edge mask, tests)."""
    import torch

    from eap.evaluate import evaluate_graph
    real = g.real_edge_mask
    sc = scores.abs().masked_fill(~real, -1).flatten()
    cur = sc.argsort(descending=True)[:2 * k].tolist()

    def F(edges):
        g.in_graph[:] = False
        g.in_graph.view(-1)[torch.tensor(edges, dtype=torch.long)] = True
        g.nodes_in_graph[:] = True
        return (float(evaluate_graph(model, g, loader, m1, quiet=True).mean()) - none) / (base - none)
    tests = 0
    while len(cur) > k:
        cur.sort(key=lambda e: float(sc[e]))
        size = max(1, -(-len(cur) // blocks))
        bl = [cur[i:i + size] for i in range(0, len(cur), size)]
        fs = []
        for b in bl:
            drop = set(b)
            fs.append(F([e for e in cur if e not in drop]))
        tests += len(bl)
        excess, gone = len(cur) - k, set()
        for j in sorted(range(len(bl)), key=lambda j: abs(1 - fs[j])):
            if gone and len(gone) + len(bl[j]) > max(1, excess // 2):
                break
            gone |= set(bl[j])
        cur = [e for e in cur if e not in gone]
    f = F(cur)
    return f, g.in_graph.clone(), tests + 1


def select(model, task, mname, method, out, dl, n_rank, metric, refine, rows, test_n, eval_batch, log,
           fix_max: float = 0.0, curves: str = "all") -> dict:
    """Per circuit size, the circuit of whichever proposal (METHOD = select:A+B+..., each a ranking: an attribution
    method of MIB's in both its absolute and signed order, measured, nodes or eliminate, which reuses nodes' tests when
    it follows it; and, at sizes up to FIX_MAX of the edges, edgefix's circuit pruned from the first proposal's top
    edges) has the measured faithfulness nearest 1 on the attribution examples (MIB's evaluate_graph on the train
    pairs); MIB's submission format (2), one circuit per size. Scored on the test split as MIB's
    evaluate_area_under_curve scores a ranking, each size's circuit as chosen. CURVES "all": every proposal's own test
    curve too; "eap": only the attribution methods' (the baseline beside us). Every test metric is kept per example
    (bootstrap intervals: mib_bootstrap)."""
    import torch

    from eap.attribute import attribute
    from eap.evaluate import evaluate_baseline, evaluate_graph
    from eap.graph import Graph
    names = method.split(":", 1)[1].split("+") if ":" in method else ["EAP-IG-inputs", "measured"]
    m1 = partial(metric, mean=False, loss=False)
    rec = {"task": task, "model": mname, "method": method, "rank_examples": n_rank, "proposals": {}, "fix_max": fix_max}
    t0 = time.time()
    graphs = {}
    for c in names:
        t1 = time.time()
        g = Graph.from_model(model)
        if c == "measured":
            rec["measured"] = measured(model, g, dl, m1, refine, log, None, rows)
        elif c == "nodes":
            rec["nodes"] = nodes(model, g, dl, m1, log, rows)
        elif c == "eliminate":
            rec["eliminate"] = eliminate(model, g, dl, m1, log, rows, noise=rec.get("nodes", {}).get("noise"))
        else:
            attribute(model, g, dl, partial(metric, mean=True, loss=True), c, "patching", ig_steps=5, intervention_dataloader=dl)
        graphs[c] = g
        rec["proposals"][c] = {"seconds": time.time() - t1}
        import gc  # (a ranking's cached runs sit in closures' reference cycles: free them before the next one, 14 GB on
        gc.collect()  # Llama-3.1 MCQA, which ran MIB's evaluation out of memory)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        d = out / "submission" / c / f"{task.replace('_', '-')}_{mname}"
        d.mkdir(parents=True, exist_ok=True)
        g.to_pt(str(d / "importances.pt"))
    rec["edges"] = n_edges = int(graphs[names[0]].real_edge_mask.sum())
    trdl = DataLoaderLike(dl, eval_batch)

    def vec(g, loader):
        return evaluate_graph(model, g, loader, m1, quiet=True).float()
    base = float(evaluate_baseline(model, trdl, m1, quiet=True).mean())
    g0 = Graph.from_model(model)
    g0.apply_topn(0, True)
    none = float(vec(g0, trdl).mean())
    chosen = []
    fixes = {}
    for pct in PERCENTAGES:
        k = int(pct * n_edges)
        best = None
        for c, g in graphs.items():
            for absolute in (True, False):
                if not absolute and float(g.scores[g.real_edge_mask].min()) >= 0:
                    continue
                g.apply_topn(k, absolute, prune=True)
                f = (float(vec(g, trdl).mean()) - none) / (base - none)
                rec["proposals"][c].setdefault(f"train_abs_{absolute}", []).append(f)
                if best is None or abs(1 - f) < abs(1 - best[0]):
                    best = (f, c, absolute, g.in_graph.clone())
        if pct <= fix_max:
            t1 = time.time()
            f, mask, n = edgefix(model, g0, trdl, m1, base, none, k, graphs[names[0]].scores, log)
            fixes[pct] = mask
            rec.setdefault("edgefix", []).append({"pct": pct, "train_F": f, "tests": n, "seconds": time.time() - t1})
            if abs(1 - f) < abs(1 - best[0]):
                best = (f, "edgefix", True, mask)
        chosen.append(best)
        log(f"{100 * pct:g}%: {best[1]} {'abs' if best[2] else 'signed'}, train F {best[0]:.3f}")
    rec["seconds_rank"] = time.time() - t0
    rec["seconds_per_example"] = rec["seconds_rank"] / max(n_rank, 1)
    rec["chosen"] = [[c, a, f] for f, c, a, _ in chosen]
    tdl, n_test = data(model, task, mname, "test", test_n, eval_batch)
    rec["test_examples"] = n_test
    vb = evaluate_baseline(model, tdl, m1, quiet=True).float()
    g0.apply_topn(0, True)
    vn = vec(g0, tdl)
    rec["test_full"], rec["test_none"] = vb.tolist(), vn.tolist()
    F = lambda v: (float(v.mean()) - float(vn.mean())) / (float(vb.mean()) - float(vn.mean()))  # noqa: E731
    test = {}  # ((proposal, absolute), size index) -> per-example metrics
    for c, g in graphs.items():
        if curves == "eap" and c in ("measured", "nodes", "eliminate"):
            continue
        for absolute in (True, False):
            if not absolute and float(g.scores[g.real_edge_mask].min()) >= 0:
                continue
            for i, pct in enumerate(PERCENTAGES):
                g.apply_topn(int(pct * n_edges), absolute, prune=True)
                test[(c, absolute), i] = vec(g, tdl)
            fs = [F(test[(c, absolute), i]) for i in range(len(PERCENTAGES))]
            cpr, cmd = areas(fs)
            rec["proposals"][c][f"abs_{absolute}"] = {"CMD": cmd, "CPR": cpr, "faithfulness": fs,
                                                      "test_metric": [test[(c, absolute), i].tolist() for i in range(len(PERCENTAGES))]}
            log(f"{task} {mname} {c} absolute={absolute}: CMD {cmd:.3f} CPR {cpr:.3f}")
    counts, vs = [], []
    sub = out / "submission" / "select" / f"{task.replace('_', '-')}_{mname}"
    sub.mkdir(parents=True, exist_ok=True)
    for i, (pct, (_, c, a, ig)) in enumerate(zip(PERCENTAGES, chosen)):
        g0.in_graph[:] = ig
        g0.nodes_in_graph[:] = True
        g0.prune()
        counts.append(int(g0.in_graph.sum()))
        if ((c, a), i) not in test:
            g0.in_graph[:] = ig
            g0.nodes_in_graph[:] = True
            test[(c, a), i] = vec(g0, tdl)
        vs.append(test[(c, a), i])
        g0.in_graph[:] = ig
        g0.nodes_in_graph[:] = True
        g0.prune()
        g0.scores[:] = g0.in_graph.float()
        g0.to_pt(str(sub / f"{pct * 100:g}.pt"))
    faith = [F(v) for v in vs]
    cpr, cmd = areas(faith)
    rec["abs_True"] = rec["abs_False"] = {"CMD": cmd, "CPR": cpr, "faithfulness": faith, "weighted_edge_counts": counts,
                                          "test_metric": [v.tolist() for v in vs]}
    log(f"{task} {mname} {method}: CMD {cmd:.3f} CPR {cpr:.3f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task}_{mname}_{method.replace(':', '-').replace('+', '-')}{'-fix' if fix_max else ''}.json").write_text(json.dumps(rec))
    return rec


def bootstrap(full, none, a, b=None, B: int = 2000, seed: int = 0) -> dict:
    """CMD of the per-example test metrics A ([sizes][examples]; FULL and NONE the model's and the empty circuit's) with
    a 95% bootstrap interval over the test examples (each resample recomputes every size's faithfulness from its
    means, as MIB does), and with B (another curve on the same examples) the paired difference A - B."""
    import numpy as np

    full, none, a = np.array(full), np.array(none), np.array(a)
    b = None if b is None else np.array(b)
    n = len(full)

    def cmd(m, idx):
        d = full[idx].mean() - none[idx].mean()
        return areas([(m[i][idx].mean() - none[idx].mean()) / d for i in range(len(m))])[1]
    rng = np.random.default_rng(seed)
    xs, ds = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        xs.append(cmd(a, idx))
        if b is not None:
            ds.append(xs[-1] - cmd(b, idx))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    allx = np.arange(n)
    out = {"n": n, "CMD": cmd(a, allx), "CI": q(xs)}
    if b is not None:
        out.update(diff=cmd(a, allx) - cmd(b, allx), diff_CI=q(ds))
    return out


class DataLoaderLike:
    """A dataloader's examples again in batches of B (the attribution batch is 1 on Llama; evaluation runs larger)."""

    def __init__(self, dl, B: int):
        self.items = [x for clean, corrupted, label in dl for x in zip(clean, corrupted, label)]
        self.B = B

    def __iter__(self):
        import torch
        for i in range(0, len(self.items), self.B):
            c, x, lab = zip(*self.items[i:i + self.B])
            yield list(c), list(x), torch.stack(lab) if torch.is_tensor(lab[0]) else lab  # (as the batches came)

    def __len__(self):
        return (len(self.items) + self.B - 1) // self.B


def finder(task: str, mname: str, out: Path, n: int, log=print, rows: int = 32) -> list[dict]:
    """circuit_finder (method/oracle) on the first n pairs of MIB's train split, on the Hugging Face model, with MIB's own
    contrast: the answer set MIB's correct token and the alternatives its counterfactual token (explicit_sets), so the
    finder's contrast is MIB's logit difference in bits. Each pair as MIB (TransformerLens' to_tokens) tokenizes it: BOS,
    or EOS where the tokenizer has no BOS (Qwen-2.5, GPT-2), then the text; pairs of unequal length are skipped (the
    finder patches position by position).
    Writes OUT/finder_TASK_MODEL/<i>.json (the finder's record without its singles)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
    import circuit_finder as cf
    name = MODEL_NAME_TO_FULLNAME[mname]
    tok = AutoTokenizer.from_pretrained(name)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if cuda else torch.float32,
                                                 device_map="cuda" if cuda else "cpu").eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok, rows=rows)
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.utils import TASKS_TO_HF_NAMES
    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF_NAMES[task]}", tok, split="train", task=task, model_name=mname, num_examples=n)
    d = out / f"finder_{task}_{mname}"
    d.mkdir(parents=True, exist_ok=True)
    setup = pt.setup
    recs = []
    for i in range(len(ds)):
        clean, corrupt, (good, bad) = ds[i]
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        ids = [[bos] + tok(t, add_special_tokens=False)["input_ids"] for t in (clean, corrupt)]
        if len(ids[0]) != len(ids[1]):
            log(f"{i}: prompts of {len(ids[0])} and {len(ids[1])} tokens, skipped")
            continue
        sets = {"answer": [int(good)], "alternatives": [int(bad)]}
        pt.setup = lambda c, k, answers=None, **kw: setup(c, k, explicit_sets=sets, **kw)  # (find() calls setup itself)
        t0 = time.time()
        rec = cf.find(pt, ids[0], ids[1], None)
        rec.pop("singles", None)
        rec.update(i=i, seconds_wall=time.time() - t0, task=task, model=mname)
        (d / f"{i}.json").write_text(json.dumps(rec))
        recs.append(rec)
        log(f"{i}: gap {rec.get('gap', 0):.2f} bits, met {rec.get('met')}, {len(rec.get('nodes', []))} neurons, "
            f"{len(rec.get('routes', []))} heads, {rec['seconds_wall']:.1f} s")
    pt.setup = setup
    del model, pt
    torch.cuda.empty_cache()
    return recs


def finder_circuits(task: str, mname: str, out: Path, batch: int, test_n: int | None, against: list[Path], log=print) -> dict:
    """The finder's per-pair circuits as MIB circuits, scored by MIB's evaluate_graph on its test split: a neuron at
    (layer, position) becomes that layer's whole MLP at every position, a head at a position the head (its q, k and v
    inputs) at every position, and a circuit of node set S holds every edge between the input, S and the logits. S is
    every node in at least t of the pairs' circuits (met or not), for t near 1, 2 and 1/8, 1/4, 1/2, 3/4 and all of the
    pairs; each also scored on the finder's own train pairs (what the conversion keeps of circuits that passed there). Each of AGAINST's rankings (importances.pt)
    is scored at the same edge counts (MIB's top-k with its pruning)."""
    import torch

    from eap.evaluate import evaluate_baseline, evaluate_graph
    from eap.graph import Graph
    from MIB_circuit_track.metrics import get_metric
    recs = [json.loads(f.read_text()) for f in sorted((out / f"finder_{task}_{mname}").glob("*.json"))]
    recs = [r for r in recs if not r.get("skipped")]
    model = load(mname)
    metric = partial(get_metric("logit_diff", task, model.tokenizer, model), mean=False, loss=False)
    tdl, n_test = data(model, task, mname, "test", test_n, batch)
    full = float(evaluate_baseline(model, tdl, metric, quiet=True).mean())
    g0 = Graph.from_model(model)
    g0.apply_topn(0, True)
    none = float(evaluate_graph(model, g0, tdl, metric, quiet=True).mean())
    pdl, _ = data(model, task, mname, "train", max(r["i"] for r in recs) + 1, batch)  # (the finder's own pairs)
    pfull = float(evaluate_baseline(model, pdl, metric, quiet=True).mean())
    pnone = float(evaluate_graph(model, g0, pdl, metric, quiet=True).mean())
    count = {}
    for r in recs:
        S = {f"m{u['layer']}" for u in r.get("nodes", [])} | {f"a{u['layer']}.h{u['head']}" for u in r.get("routes", [])}
        for x in S:
            count[x] = count.get(x, 0) + 1
    res = {"task": task, "model": mname, "pairs": len(recs), "met": sum(bool(r.get("met")) for r in recs), "full": full,
           "none": none, "test_examples": n_test, "node_counts": count, "circuits": []}
    have = sorted(set(count.values()))
    ts = sorted({min(have, key=lambda t: abs(t - x)) for x in (1, 2, len(recs) / 8, len(recs) / 4, len(recs) / 2,
                                                               3 * len(recs) / 4, len(recs))}, reverse=True)
    for t in ts:
        S = {x for x, c in count.items() if c >= t}
        g = Graph.from_model(model)
        g.in_graph[:] = False
        fw = [0] + [g.forward_index(g.nodes[x], attn_slice=False) for x in S]
        bw = [g.n_backward - 1]
        for x in S:
            node = g.nodes[x]
            bw += [g.backward_index(node, qkv=q, attn_slice=False) for q in "qkv"] if x.startswith("a") else [g.backward_index(node)]
        fi, bi = torch.tensor(fw), torch.tensor(bw)
        g.in_graph[fi[:, None], bi[None, :]] = True
        g.in_graph &= g.real_edge_mask
        g.nodes_in_graph[:] = False
        g.nodes_in_graph[fi] = True
        g.prune()
        k = int(g.in_graph.sum())
        m = float(evaluate_graph(model, g, tdl, metric, quiet=True).mean())
        mp = float(evaluate_graph(model, g, pdl, metric, quiet=True).mean())
        row = {"t": t, "nodes": len(S), "mlps": sum(x.startswith("m") for x in S), "heads": sum(x.startswith("a") for x in S),
               "edges": k, "pct": k / int(g.real_edge_mask.sum()), "F": (m - none) / (full - none),
               "F_own_pairs": (mp - pnone) / (pfull - pnone), "against": {}}
        for p in against:
            h = Graph.from_pt(str(p))
            h.apply_topn(k, True, prune=True)
            mh = float(evaluate_graph(model, h, tdl, metric, quiet=True).mean())
            row["against"][p.parent.parent.name] = (mh - none) / (full - none)
        log(f"t {t}: {len(S)} nodes ({row['mlps']} MLPs, {row['heads']} heads), {k} edges ({100 * row['pct']:.2f}%): F {row['F']:.3f} "
            f"(own pairs {row['F_own_pairs']:.3f}); "
            + ", ".join(f"{a} {v:.3f}" for a, v in row["against"].items()))
        res["circuits"].append(row)
    (out / f"finder_{task}_{mname}.json").write_text(json.dumps(res))
    return res


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
    ap.add_argument("cmd", choices=("prep", "rank", "table", "finder", "convert"))
    ap.add_argument("args", nargs="+")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--refine", type=int, default=REFINE)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=None, help="MIB's evaluation batch (default --batch)")
    ap.add_argument("--fix", type=float, default=0.0, help="select: edgefix circuits at sizes up to this share of edges")
    ap.add_argument("--curves", default="all", choices=("all", "eap"), help="select: every proposal's test curve, or only the attribution methods'")
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
        else:  # (the weights and tokenizer into the cache while online)
            load(mname)
            if mname == "interpbench":
                from huggingface_hub import hf_hub_download
                hf_hub_download("mib-bench/interpbench", filename="interpbench_graph.json")
        return
    if a.cmd == "finder":
        task, mname, out = a.args
        finder(task, mname, Path(out), a.n, log=lambda s: print(s, flush=True), rows=a.rows)
        return
    if a.cmd == "convert":
        task, mname, out, *against = a.args
        finder_circuits(task, mname, Path(out), a.eval_batch or a.batch, a.test_n, [Path(x) for x in against],
                        log=lambda s: print(s, flush=True))
        return
    task, mname, method, out = a.args
    rank(task, mname, method, Path(out), a.n, a.refine, a.batch, a.test_n, log=lambda s: print(s, flush=True), limit=a.limit,
         rows=a.rows, prev=a.prev, order=a.order, eval_batch=a.eval_batch, fix_max=a.fix, curves=a.curves)


if __name__ == "__main__":
    main()
