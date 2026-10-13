"""One fixed TRAIN search demonstration; no oracle inference or paid training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def write_json(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "payload", "reward-source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("do not overwrite frozen search")
    args.out.mkdir()
    here = args.reward_source.parent
    sys.path[:0] = [str(args.source / folder) for folder in
                   ("experiments/oracle/toyloop", "experiments/oracle/toys", "method/oracle", "method/vpd")]
    import torch
    import render
    import planted
    import circuit
    import native
    import goal_eval
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("reward_exec", args.reward_source)
    reward = importlib.util.module_from_spec(spec)
    sys.modules["reward_exec"] = reward
    spec.loader.exec_module(reward)
    sys.path.insert(0, str(here))
    import intervention_teacher as teacher
    protocol = json.loads((args.payload / "protocol.json").read_text())
    for name, expected in protocol["files"].items():
        if hashlib.sha256((args.payload / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("frozen TRAIN fixture changed")
    ident = "redundant_000"
    root = args.payload / "cases" / ident
    record = json.loads((root / (ident + ".json")).read_text())
    texts = [list(example["ids"]) for example in record["examples"]]
    if record["split"] != "train" or len(texts) != 4 or any(
            example["target"] != len(example["ids"]) - 1 for example in record["examples"]):
        raise RuntimeError("fixed TRAIN input scope changed")
    files = [Path(__file__), args.reward_source, here / "intervention_teacher.py",
             here / "reward_interventions.py", here / "head_counterfactual.py", here / "render.py"]
    write_json(args.out / "pre_native_contract.json", {
        "case": ident, "inputs": texts, "max_attempts": 64, "worlds": 84,
        "forward_cap": 5796, "worst_kl_max": .001,
        "selection": "architecture-first original nonzero pieces; no teacher truth",
        "executor": "cd8d4a9c2dfc0d3b28bc6604e0549e16aceb3aa4",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "payload_protocol_sha256": hashlib.sha256((args.payload / "protocol.json").read_bytes()).hexdigest(),
    })
    toy = render.from_zoo({"weights": ident + ".pt", "record": ident + ".json"}, root)
    nat = planted.native_of(toy.sd, toy.cfg)
    neuron_count = sum(int(nat.target.site(f"h.{layer}.mlp.down_proj").W.ne(0).any(dim=0).sum())
                       for layer in range(nat.target.n_layer))
    planned_worlds = len(texts) * (1 + 2 * (nat.target.n_layer * nat.target.n_head + neuron_count))
    if planned_worlds != 84 or teacher.MAX_ATTEMPTS != 64 or teacher.WORST_KL_MAX != .001:
        raise RuntimeError("predeclared work/feasibility bound changed before scoring")
    before = teacher.binding(toy, nat, texts, [])
    write_json(args.out / "original_source_binding.json", before)
    evidence = []
    def retain(stage, index, row, p, q, complete):
        filename = f"{stage}_{index}.safetensors"
        tensors = {"dense": p.contiguous(), "sparse": q.contiguous()}
        if complete is not None:
            tensors["full_integrity"] = complete.contiguous()
        save_file(tensors, str(args.out / filename))
        evidence.append({"stage": stage, "index": index, "world": row, "file": filename,
                         "sha256": hashlib.sha256((args.out / filename).read_bytes()).hexdigest()})
        with (args.out / "evidence.jsonl").open("a") as handle:
            handle.write(json.dumps(evidence[-1]) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    attempts = []
    def attempt(row):
        attempts.append(row)
        write_json(args.out / "attempts.json", attempts)
    result = teacher.search_teacher(toy, nat=nat, texts=texts, attempt_sink=attempt, evidence_sink=retain)
    restored = teacher.binding(toy, nat, texts, [])
    if before != restored:
        raise RuntimeError("search did not restore source state")
    write_json(args.out / "restored_source_binding.json", restored)
    cache = result.pop("cache")
    worlds = len(cache["references"])
    if worlds != 84 or teacher.MAX_ATTEMPTS != 64:
        raise RuntimeError("predeclared world/attempt bound changed")
    # First search result is frozen before any truth interpretation or crosscheck.
    write_json(args.out / "frozen_search_result.json", result)
    if result["target"] is not None:
        with (args.out / "final_target.txt").open("w") as handle:
            handle.write(result["target"])
            handle.flush()
            os.fsync(handle.fileno())
    save_file({f"p_{index}": value.contiguous() for index, value in enumerate(cache["references"])},
              str(args.out / "cached_references.safetensors"))
    write_json(args.out / "reference_cache.json", {key: value for key, value in cache.items() if key != "references"})
    if result["target"] is None:
        write_json(args.out / "complete.json", {"status": result["status"], "target": None,
                   "forward_upper_bound_used": 3 * worlds + worlds * len(attempts),
                   "actual_forwards": sum(3 if row["stage"] == "reference" else 1 for row in evidence)})
        return
    index = [0]
    def crosscheck(current, ids, row, p, q):
        if not torch.equal(p, cache["references"][index[0]]):
            raise RuntimeError("cached versus uncached dense distribution mismatch")
        retain("uncached_final", index[0], row, p, q, None)
        index[0] += 1
    uncached = reward.score_wiring_interventions(toy, result["target"], nat=nat, texts=texts,
                                               panel=cache["binding"]["panel"], evidence_sink=crosscheck)
    for expected, actual in zip(result["selected"]["score"]["per_world"], uncached["per_world"]):
        if expected["top"] != actual["top"] or abs(expected["kl_bits"] - actual["kl_bits"]) > 1e-6:
            raise RuntimeError("cached versus uncached selected score mismatch")
    if before != teacher.binding(toy, nat, texts, []):
        raise RuntimeError("final source state was not restored")
    write_json(args.out / "restored_source_binding.json", teacher.binding(toy, nat, texts, []))
    write_json(args.out / "uncached_final.json", uncached)
    spec = importlib.util.spec_from_file_location("teacher_renderer", here / "render.py")
    renderer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = renderer
    spec.loader.exec_module(renderer)
    renderer.SECTIONS = "weights"  # No extra model forwards or target-labelled behaviour facts.
    result["cache"] = cache
    row = teacher.training_row(toy, result, nat=nat, renderer=renderer)
    write_json(args.out / "training_row.json", row)
    # Truth is used only for descriptive interpretation after the target freeze.
    write_json(args.out / "post_freeze_truth_comparison.json", {
        "target_sha256": result["target_sha256"],
        "selected_units": sorted(circuit.parse(result["target"]).units),
        "original_truth_units": record["gt_units"], "not_selection_data": True})
    complete = {"status": result["status"], "attempts": len(attempts), "worlds": worlds,
                "forward_upper_bound_used": 3 * worlds + worlds * len(attempts) + 2 * worlds,
                "actual_forwards": sum(3 if item["stage"] == "reference" else
                                       2 if item["stage"] == "uncached_final" else 1 for item in evidence),
                "target_sha256": result["target_sha256"], "usable_training_row": row["usable"],
                "worst_kl_bits": max(r["kl_bits"] for r in uncached["per_world"])}
    if complete["forward_upper_bound_used"] > 5796:
        raise RuntimeError("forward cap exceeded")
    write_json(args.out / "complete.json", complete)
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
