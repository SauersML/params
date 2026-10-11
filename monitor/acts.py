"""Residual-stream activations of the agent model over its finished runs, by one HF transformers forward pass per run.

The streams are append-only (agent.py), so the pass reproduces what the model computed while generating. For every
token the model generated (and the few forced ones, marked), the residual stream after decoder blocks LAYERS (the
input of the next block; 1-indexed: layer 8 is hidden_states[8]) at that token's position, float16, in shards:
  acts_NNNNN.npy    [tokens, len(LAYERS), d_model] float16
  index_NNNNN.npz   run (index into runs.json), pos (position in the call's sequence), gen (generated-token index),
                    seg, call (the turn whose call it is; -1 for an append-only run's single stream)
and, per checkpoint of agent.checkpoints, the mean over the generated tokens so far and the activation at the last
one (feats.npz: run, gen, mean [n, layers, d], last [n, layers, d], float16), which the probes read.

  python acts.py --runs transcripts.jsonl --out DIR [--layers 8,16,24,32] [--max-gb 40]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402
import replay  # noqa: E402


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layers", default="8,16,24,32")
    ap.add_argument("--shard-gb", type=float, default=2.0)
    ap.add_argument("--max-gb", type=float, default=40.0, help="token store budget; runs past it get checkpoint features only")
    ap.add_argument("--deadline", type=float, default=1e9, help="seconds after which no further run is processed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM
    layers = [int(x) for x in a.layers.split(",")]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    base = model.model
    caught = {}

    def hook(i):
        def f(mod, inp, out):
            caught[i] = (out[0] if isinstance(out, tuple) else out)[0]
            if i == max(layers):
                raise Stop
        return f
    hs = [base.layers[l - 1].register_forward_hook(hook(l)) for l in layers]
    recs = [json.loads(l) for l in open(a.runs)]
    # Within the token budget, runs with a cheating action come first, then the rest by task (all splits of a task
    # together), so a cut keeps matched honest runs of the same tasks.
    recs.sort(key=lambda r: (r["label"] != "cheat" and r["first_cheat"] is None, r["sample"], int(r["task_id"].split("_")[1]), r["split"]))
    run_ids = [r["run_id"] for r in recs]
    json.dump(run_ids, open(os.path.join(a.out, "runs.json"), "w"))
    d = base.config.hidden_size
    per_tok = len(layers) * d * 2
    shard, buf, idx, buf_bytes, total_bytes = 0, [], [], 0, 0
    fe = {"run": [], "gen": [], "mean": [], "last": []}
    t0 = time.time()

    def flush():
        nonlocal shard, buf, idx, buf_bytes
        if not buf:
            return
        np.save(os.path.join(a.out, f"acts_{shard:05d}.npy"), np.concatenate(buf))
        ix = {k: np.concatenate([x[k] for x in idx]) for k in idx[0]}
        np.savez(os.path.join(a.out, f"index_{shard:05d}.npz"), **ix)
        shard, buf, idx, buf_bytes = shard + 1, [], [], 0

    for ri, r in enumerate(recs):
        if time.time() - t0 > a.deadline:
            print(f"deadline: {ri} of {len(recs)} runs processed (cheating runs first)", flush=True)
            break
        # One forward pass per model call: an append-only run is one call over its whole stream; a template-rendered
        # run's calls each have their own prompt, so each generated token gets the activations it had when generated.
        if "calls" in r:
            parts = [(c["context_ids"] + c["completion_ids"], len(c["context_ids"]), c["turn"])
                     for c in replay.calls(r)]
            segs = [np.array([agent.SEG_PROMPT] * n + list(cs["completion_seg"][:len(p) - n]))
                    for (p, n, _), cs in zip(parts, r["calls"])]
        else:
            parts, segs = [(r["token_ids"], 0, -1)], [np.array(r["seg"])]
        hks, seg_k, pos_k, call_k = [], [], [], []
        for (seq, _, turn), sg in zip(parts, segs):
            keep_c = np.where((sg == agent.SEG_GEN) | (sg == agent.SEG_FORCED))[0]
            caught.clear()
            with torch.no_grad():
                try:
                    base(input_ids=torch.tensor(seq, device="cuda")[None], use_cache=False)
                except Stop:
                    pass
            h = torch.stack([caught[l] for l in layers], 1)  # [T, layers, d] bf16
            hks.append(h[torch.as_tensor(keep_c, device="cuda")])
            seg_k.append(sg[keep_c])
            pos_k.append(keep_c)
            call_k.append(np.full(len(keep_c), turn))
        hk = torch.cat(hks)
        keep, seg_kept, call_kept = np.concatenate(pos_k), np.concatenate(seg_k), np.concatenate(call_k)
        gen_mask = seg_kept == agent.SEG_GEN
        hg = hk[torch.as_tensor(np.where(gen_mask)[0], device="cuda")].float()
        csum = torch.cumsum(hg, 0)
        cps = agent.checkpoints(r)
        c = torch.as_tensor([cp for cp, _ in cps], device="cuda")
        fe["run"].append(np.full(len(cps), ri, np.int32))
        fe["gen"].append(c.cpu().numpy().astype(np.int32))
        fe["mean"].append((csum[c - 1] / c[:, None, None]).half().cpu().numpy())
        fe["last"].append(hg[c - 1].half().cpu().numpy())
        nb = len(keep) * per_tok
        if total_bytes + nb <= a.max_gb * 2**30:
            buf.append(hk.half().cpu().numpy())
            gen_index = np.cumsum(gen_mask) - 1
            gen_index[~gen_mask] = -1
            idx.append({"run": np.full(len(keep), ri, np.int32), "pos": keep.astype(np.int32),
                        "gen": gen_index.astype(np.int32), "seg": seg_kept.astype(np.int8),
                        "call": call_kept.astype(np.int32)})
            buf_bytes += nb
            total_bytes += nb
            if buf_bytes >= a.shard_gb * 2**30:
                flush()
        if ri % 20 == 0:
            print(f"{ri + 1}/{len(recs)} runs, {total_bytes / 2**30:.1f} GB stored, {time.time() - t0:.0f} s", flush=True)
    flush()
    np.savez(os.path.join(a.out, "feats.npz"), layers=np.array(layers), **{k: np.concatenate(v) for k, v in fe.items()})
    for h_ in hs:
        h_.remove()
    print(f"done: {len(recs)} runs, {total_bytes / 2**30:.1f} GB in {shard} shards, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
