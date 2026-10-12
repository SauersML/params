"""Speed of the oracle checker's hot paths on vpd4l (#2951): repeatable benchmarks, one JSON line per run.

  bench_speed.py [--code DIR] [--split heldout] [--texts 3] [--benches carry,score,events] [--out FILE]
  bench_speed.py --profile carry [--granularity matrix]                     where measure() spends its time

carry    MatrixCarry.measure: unit sets per second, batches of 64 random sets (each unit on with probability 1/2) over
         the 3 changed prompts of the text's most sensitive position, units at that position and the target; matrix
         granularity, and subcomponent granularity (every subcomponent at the two positions) where the code has it
score    units.score_many seconds per question: K=8 positions, 8 answers (the copy and all baselines, 4 copies each);
         seconds_first with the model object's per-text caches dropped (a text's first call), seconds with them kept
events   units.events seconds per text, K=8; seconds_first and seconds as for score
window   Native.sensitivity's windowed probes on this device vs the whole forward: equal, and seconds of each
teacher  teacher.Pruner's masked runs from VPD's answer: sets per second for position, site and single-node removals and
         the complements (teacher_live: remainders on); needs the causal-importance network
reward   components.reward per text for 8 nested subsets of VPD's ordered answer with stated connections: first and
         repeat call seconds (fix PYTHONHASHSEED for comparable answers across processes)
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


def timed(fn, sync, reset=None):
    """(seconds, result) of fn() after one untimed warm-up call; reset() (untimed) runs before the timed call."""
    fn()
    sync()
    if reset is not None:
        reset()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return time.perf_counter() - t0, out


def forget(nat):
    """Drop what a model object keeps between calls for a text (units.sensitive's orders, Native.reference's results),
    for a first-call time."""
    nat.__dict__.pop("_sensitive_orders", None)
    if hasattr(nat, "_refs"):
        nat._refs.clear()
        nat._refs_bytes = 0


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
        sync = lambda: carry.synchronize(nat.dev)
        first, _ = timed(lambda: units.score_many(nat, task, answers, k), sync, lambda: forget(nat))
        seconds, out = timed(lambda: units.score_many(nat, task, answers, k), sync)
        rows.append({"task": path.stem, "T": len(ids), "answers": len(answers), "seconds_first": first, "seconds": seconds,
                     "gpu": util.finish(), "fingerprint": sum(r["uncarried"] + r["rest_carried"] for r in out)})
    return rows


def bench_events(nat, texts, k=8):
    import carry
    import units

    rows = []
    for path in texts:
        task = json.loads(path.read_text())
        util = Utilization(nat.dev)
        sync = lambda: carry.synchronize(nat.dev)
        first, _ = timed(lambda: units.events(nat, task, k), sync, lambda: forget(nat))
        seconds, out = timed(lambda: units.events(nat, task, k), sync)
        rows.append({"task": path.stem, "events": len(out), "seconds_first": first, "seconds": seconds, "gpu": util.finish(),
                     "fingerprint": sum(e.get("bits", 0.0) + e.get("kl", 0.0) for e in out)})
    return rows


def bench_window(nat, texts):
    """Native.sensitivity with its windowed probes on this device vs the whole forward: equal? seconds of each."""
    import torch

    import carry
    import native

    rows, keep = [], type(nat).WINDOW_EXACT
    dev = str(nat.dev).split(":")[0]
    for path in texts:
        ids, targets = native.text(path)
        res = {}
        for mode, devices in (("whole", ()), ("window", (dev,))):
            nat.WINDOW_EXACT = devices
            seed = lambda: torch.Generator(device="cpu").manual_seed(native.task_seed(str(ids[:8])))
            seconds, (moved, _) = timed(lambda: nat.sensitivity(ids, targets, seed()), lambda: carry.synchronize(nat.dev))
            res[mode] = (seconds, moved)
        rows.append({"task": path.stem, "T": len(ids), "whole_s": res["whole"][0], "window_s": res["window"][0],
                     "equal": bool(torch.equal(res["whole"][1], res["window"][1]))})
    nat.WINDOW_EXACT = keep
    return rows


def bench_teacher(nat, texts, live=False, batch=32):
    """teacher.Pruner's masked runs on each text from VPD's answer: sets per second for every position's removal, every
    6th site's removal, and 256 single-node removals and their complements (the joint path's two runs per set)."""
    import torch

    import carry
    import goal_eval
    import native
    import teacher

    ge = goal_eval.GoalEval(nat, chunk=batch)
    rows = []
    for path in texts:
        ids, targets = native.text(path)
        tx = ge.text(ids, targets)
        if tx.vpd is None:
            return [{"skipped": "no causal-importance network (VPD's answer) here"}]
        pr = teacher.Pruner(ge, tx, tx.vpd, live=live, B=batch)
        nodes = torch.randperm(len(pr.nodes), generator=torch.Generator().manual_seed(0))[:256].to(nat.dev)
        trials = pr.on[None].repeat(256, 1)
        trials[torch.arange(256, device=nat.dev), nodes] = False
        sites = pr.blocks("site")
        row = {"task": path.stem, "T": len(ids), "nodes": len(pr.nodes), "live": live}
        sync = lambda: carry.synchronize(nat.dev)  # noqa: E731
        fingerprint = 0.0
        for name, fn in [("position", lambda: pr.without(pr.blocks("position"))), ("site", lambda: pr.without(sites[::6])),
                         ("trials", lambda: pr.kl_sets(trials)), ("complements", lambda: pr.kl_sets(~trials))]:
            seconds, out = timed(fn, sync)
            row[f"{name}_sets_per_s"] = len(out) / seconds
            fingerprint += sum(out)
        row["fingerprint"] = fingerprint
        rows.append(row)
    return rows


def bench_reward(nat, texts, answers=8):
    """components.reward per text for nested subsets of VPD's ordered answer (its first 48, 36, ... subcomponents) with
    the model's connections among them stated (up to 8 into the prediction, 16 more), the measured answers to the
    questions: seconds of a text's first call and of a repeat (its Text kept), and the stages' seconds."""
    import json

    import carry
    import components
    import goal_eval
    import native

    ge = goal_eval.GoalEval(nat)
    rows = []
    for path in texts:
        task = json.loads(path.read_text())
        ids, targets = native.text(path)
        ev = components.evidence(nat, ge, task)
        order = nat._vpd_order(ids, targets)
        texts_ = []
        for k in [48, 36, 24, 16, 12, 8, 4, 2][:answers]:
            g = native.Graph(set(order[:k]), {}, [])
            par, out = native.all_edges(g.node_set(), targets)
            h = native.Graph(g.node_set(), {}, [])
            for w in sorted(out)[:8]:
                h.out.append(w)
            for r, ws in sorted(par.items())[:16]:
                h.parents.setdefault(r, []).append(sorted(ws)[0])
            texts_.append(components.target(goal_eval.to_json(h, ""), ev, task["model"]))
        cache = {}
        sync = lambda: carry.synchronize(nat.dev)  # noqa: E731
        sync(); t0 = time.perf_counter()
        components.reward(ge, task, ev, texts_, cache=cache)
        sync(); first = time.perf_counter() - t0
        seconds, got = timed(lambda: components.reward(ge, task, ev, texts_, cache=cache), sync)
        rows.append({"task": path.stem, "T": len(ids), "answers": len(texts_), "valid": sum(r["valid"] for r in got),
                     "seconds_first": first, "seconds": seconds, "fingerprint": sum(r["reward"] for r in got)})
    return rows


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
    ap.add_argument("--label", default="", help="a name for this run in the output (e.g. before, after)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.code))
    head = {"label": a.label, "code": str(a.code), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=10", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()}
    import torch

    import carry
    import native

    results = {}
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
        elif b == "window" and hasattr(nat, "WINDOW_EXACT"):
            results["window"] = bench_window(nat, texts)
        elif b in ("teacher", "teacher_live"):
            results[b] = bench_teacher(nat, texts, live=b == "teacher_live")
        elif b == "reward":
            results["reward"] = bench_reward(nat, texts)
        print(b, json.dumps(results.get(b) or {k: v for k, v in results.items() if k.startswith(b)}), flush=True)
    head["torch"] = torch.__version__
    row = {**head, "device": dev, **results}
    if a.out:
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
