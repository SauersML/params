"""Speed of the oracle checker's hot paths on vpd4l (#2951): repeatable benchmarks, one JSON line per run.

  bench_speed.py [--code DIR] [--split heldout] [--texts 3] [--benches carry,score,events] [--out FILE]
  bench_speed.py --benches sample --adapter DIR [--questions FILE]          (CUDA with vLLM)
  bench_speed.py --profile carry [--granularity matrix]                     where measure() spends its time

carry    MatrixCarry.measure: unit sets per second, batches of 64 random sets (each unit on with probability 1/2) over
         the 3 changed prompts of the text's most sensitive position, units at that position and the target; matrix
         granularity, and subcomponent granularity (every subcomponent at the two positions) where the code has it
score    units.score_many seconds per question: K=8 positions, 8 answers (the copy and all baselines, 4 copies each)
events   units.events seconds per text, K=8
sample   the explainer's vLLM sampling (units_sft.py sample's settings): generated tokens per second
Texts are the split's first n, fixed; every timing is after one untimed warm-up on the same inputs (MPS compiles a
graph per shape) and ends with a device synchronize. Each result carries a fingerprint of what it computed (sums of
measured KLs and scores): a speed change that alters the numbers shows there. GPU utilization is the system's, sampled
once per second (macOS IORegistry, nvidia-smi on CUDA). --code runs another copy of method/oracle (e.g. a git archive
of a commit) for before/after numbers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class Utilization:
    """System GPU utilization, one sample per second, in a thread: macOS IORegistry or nvidia-smi."""

    def __init__(self, dev):
        self.samples, self.stop, self.dev = [], threading.Event(), str(dev)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _one(self):
        if self.dev.startswith("mps"):
            import plistlib
            raw = subprocess.run(["ioreg", "-a", "-r", "-c", "AGXAccelerator", "-d", "1"], capture_output=True, timeout=2,
                                 check=True).stdout
            return next((float(e["PerformanceStatistics"]["Device Utilization %"]) for e in plistlib.loads(raw)
                         if "Device Utilization %" in e.get("PerformanceStatistics", {})), None)
        if self.dev.startswith("cuda"):
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
                                 capture_output=True, timeout=2, check=True, text=True).stdout
            return float(out.split()[0])
        return None

    def _run(self):
        while not self.stop.is_set():
            try:
                v = self._one()
            except Exception:  # the sampler is best-effort; a missing tool leaves no samples
                return
            if v is not None:
                self.samples.append(v)
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=3)
        s = self.samples
        return {"mean_percent": sum(s) / len(s), "min_percent": min(s), "max_percent": max(s), "samples": len(s)} if s else None


def timed(fn, sync):
    """(seconds, result) of fn() after one untimed warm-up call."""
    fn()
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return time.perf_counter() - t0, out


def bench_carry(nat, texts, granularity, batch=64, reps=4):
    import torch

    import carry
    import native
    import units

    rows = []
    for path in texts:
        ids, targets = native.text(path)
        p = units.sensitive(nat, ids, targets, 1)[0]
        prompts = carry.alternatives(nat, ids, p)
        where = sorted({p, *targets})
        full = {(n, q): range(nat.C[n]) for n in nat.names for q in where} if granularity == "subcomponent" else None
        t0 = time.perf_counter()
        space = carry.UnitSpace(nat, len(ids), where, granularity, **({"candidates": full} if full else {}))
        study = carry.MatrixCarry(nat, ids, targets, prompts, space)
        carry.synchronize(nat.dev)
        setup = time.perf_counter() - t0
        g = torch.Generator(device="cpu").manual_seed(0)
        sets = [(torch.rand(space.size, generator=g) < 0.5).to(nat.dev) for _ in range(batch)]
        util = Utilization(nat.dev)
        seconds, out = timed(lambda: [study.measure(sets) for _ in range(reps)][-1], lambda: carry.synchronize(nat.dev))
        rows.append({"task": path.stem, "T": len(ids), "position": p, "units": space.size, "batch": batch,
                     "chunk_size": study.chunk_size, "setup_s": setup, "sets_per_s": batch * reps / seconds,
                     "gpu": util.finish(), "fingerprint": float(out.cpu().double().sum())})
        del study, sets
        native.free(nat.dev)
    return rows


def bench_score(nat, texts, k=8):
    import carry
    import native
    import units

    rows = []
    for path in texts:
        task = json.loads(path.read_text())
        ids, targets = native.text(path)
        positions = units.sensitive(nat, ids, targets, k)
        answers = [units.baseline("copy", positions, targets), units.baseline("all", positions, targets)] * 4
        util = Utilization(nat.dev)
        seconds, out = timed(lambda: units.score_many(nat, task, answers, k), lambda: carry.synchronize(nat.dev))
        rows.append({"task": path.stem, "T": len(ids), "answers": len(answers), "seconds": seconds, "gpu": util.finish(),
                     "fingerprint": sum(r["uncarried"] + r["rest_carried"] for r in out)})
    return rows


def bench_events(nat, texts, k=8):
    import carry
    import units

    rows = []
    for path in texts:
        task = json.loads(path.read_text())
        util = Utilization(nat.dev)
        seconds, out = timed(lambda: units.events(nat, task, k), lambda: carry.synchronize(nat.dev))
        rows.append({"task": path.stem, "events": len(out), "seconds": seconds, "gpu": util.finish(),
                     "fingerprint": sum(e.get("bits", 0.0) + e.get("kl", 0.0) for e in out)})
    return rows


def bench_sample(a):
    """units_sft.py sample's engine and settings on the questions: generated tokens per second, after a warm-up batch."""
    import json as _json

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, str(Path(a.code) / "rl"))
    from units_sft import chat_prompt

    qs = [_json.loads(x)["question"] for x in open(a.questions)][:a.questions_n]
    tok = AutoTokenizer.from_pretrained(a.base)
    rank = _json.loads((Path(a.adapter) / "adapter_config.json").read_text())["r"]
    llm = LLM(model=a.base, dtype="bfloat16", enable_lora=True, max_lora_rank=rank, max_model_len=8192, seed=0)
    params = SamplingParams(n=4, temperature=1.0, max_tokens=1536, seed=0)
    lora = LoRARequest("explainer", 1, str(a.adapter))
    prompts = [chat_prompt(tok, q) for q in qs]
    llm.generate(prompts[:2], params, lora_request=lora, use_tqdm=False)
    util = Utilization("cuda")
    t0 = time.perf_counter()
    outs = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
    seconds = time.perf_counter() - t0
    gen = sum(len(c.token_ids) for o in outs for c in o.outputs)
    return [{"questions": len(qs), "answers": 4 * len(qs), "generated_tokens": gen, "seconds": seconds,
             "tokens_per_s": gen / seconds, "gpu": util.finish()}]


def profile_carry(nat, path, granularity, batch=64):
    """The top operators of one measure() call by device time (torch.profiler), and the Python share."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    import carry
    import native
    import units

    ids, targets = native.text(path)
    p = units.sensitive(nat, ids, targets, 1)[0]
    where = sorted({p, *targets})
    full = {(n, q): range(nat.C[n]) for n in nat.names for q in where} if granularity == "subcomponent" else None
    space = carry.UnitSpace(nat, len(ids), where, granularity, **({"candidates": full} if full else {}))
    study = carry.MatrixCarry(nat, ids, targets, carry.alternatives(nat, ids, p), space)
    g = torch.Generator(device="cpu").manual_seed(0)
    sets = [(torch.rand(space.size, generator=g) < 0.5).to(nat.dev) for _ in range(batch)]
    study.measure(sets)
    carry.synchronize(nat.dev)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if str(nat.dev).startswith("cuda") else [])
    with profile(activities=acts, record_shapes=True) as prof:
        study.measure(sets)
        carry.synchronize(nat.dev)
    key = "self_cuda_time_total" if str(nat.dev).startswith("cuda") else "self_cpu_time_total"
    print(prof.key_averages(group_by_input_shape=False).table(sort_by=key, row_limit=30), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code", type=Path, default=REPO / "method" / "oracle", help="the method/oracle copy to run")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--texts", type=int, default=3)
    ap.add_argument("--benches", default="carry,score,events")
    ap.add_argument("--granularity", default="matrix,subcomponent", help="carry's unit granularities")
    ap.add_argument("--profile", choices=("carry",))
    ap.add_argument("--adapter", type=Path, help="sample: the explainer's LoRA adapter")
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--questions", type=Path, help="sample: units_sft.py questions output")
    ap.add_argument("--questions-n", type=int, default=50)
    ap.add_argument("--label", default="", help="a name for this run in the output (e.g. before, after)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.code))
    head = {"label": a.label, "code": str(a.code), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=10", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()}
    results = {}
    if a.benches == "sample":
        results["sample"] = bench_sample(a)
        dev = "cuda"
    else:
        import torch

        import carry
        import native

        nat = native.Native()
        dev = str(nat.dev)
        texts = native.tasks(a.split)[:a.texts]
        if a.profile:
            profile_carry(nat, texts[0], a.granularity.split(",")[0])
            return
        for b in a.benches.split(","):
            if b == "carry":
                for gran in a.granularity.split(","):
                    try:
                        carry.UnitSpace(nat, 2, [0], gran, **({"candidates": {}} if gran == "subcomponent" else {}))
                    except (ValueError, TypeError) as exc:
                        results[f"carry_{gran}"] = f"not in this code: {exc}"
                        continue
                    results[f"carry_{gran}"] = bench_carry(nat, texts, gran)
            elif b == "score":
                results["score"] = bench_score(nat, texts)
            elif b == "events":
                results["events"] = bench_events(nat, texts)
            print(b, json.dumps(results.get(b) or {k: v for k, v in results.items() if k.startswith(b)}), flush=True)
        head["torch"] = torch.__version__
    row = {**head, "device": dev, **results}
    if a.out:
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
