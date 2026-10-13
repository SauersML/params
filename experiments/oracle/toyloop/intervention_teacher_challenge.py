"""Post-freeze TRAIN transfer diagnostic; never repairs or selects a target."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys


TARGETS = {
    "redundant_000": "67dae0151a1eced8055a5e5d593d2ce3a791b68ae69ce38c587b2a934587df38",
    "cancel_001": "32363b39065fbbf052e05aba17d9ec7d6122074089b10493651b1e56b70275a5",
    "gate_000": "edb35f53dcfe92b63e8c3f77cf6cc09a3d433a8d70e1291c813ff63135a72ceb",
}


def joint_pairs(case, units, count=6):
    pairs = list(itertools.combinations(sorted(set(units)), 2))
    if len(pairs) < count:
        raise RuntimeError("predeclared architecture pair count unavailable")
    return sorted(pairs, key=lambda pair: hashlib.sha256(
        json.dumps([case, pair], separators=(",", ":")).encode()).hexdigest())[:count]


def fixed_edits(architecture, pairs):
    edits = [("no_edit", [])]
    edits += [("individual", [(*unit, scale)]) for unit in architecture for scale in (0., .5)]
    edits += [("joint", [(*left, left_scale), (*right, right_scale)]) for left, right in pairs
              for left_scale, right_scale in itertools.product((0., .5), repeat=2)]
    return edits


def write(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def summarize(rows):
    groups = {}
    for row in rows:
        kind = row["kind"]
        if kind == "joint":
            kind = "joint_mixed" if row["operations"][0][3] != row["operations"][1][3] else "joint_equal"
        group = groups.setdefault((row["case"], kind), {"case": row["case"], "kind": kind,
                    "worlds": 0, "valid": 0, "top_correct": 0, "nochange_correct": 0,
                    "kl_failures_or_invalid": 0, "worst_valid_kl_bits": None, "active": 0})
        group["worlds"] += 1
        group["valid"] += int(row["valid"])
        group["top_correct"] += int(row["valid"] and row["top"] == row["reference_top"])
        group["nochange_correct"] += int(row["nochange_top"] == row["reference_top"])
        group["active"] += int(row["active"])
        group["kl_failures_or_invalid"] += int(not row["valid"] or row["kl_bits"] > .001)
        if row["valid"]:
            group["worst_valid_kl_bits"] = max(group["worst_valid_kl_bits"] or 0., row["kl_bits"])
    return {"scope": "fixed post-freeze TRAIN diagnostic transfer; not pristine holdout or English proof",
            "worlds": len(rows), "groups": [groups[key] for key in sorted(groups)]}


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "payload", "reward-source", "redundant", "extension", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("do not overwrite frozen transfer diagnostic")
    args.out.mkdir()
    here = args.reward_source.parent
    sys.path[:0] = [str(args.source / folder) for folder in
                   ("experiments/oracle/toyloop", "experiments/oracle/toys", "method/oracle", "method/vpd")]
    import torch
    import render
    import zoo
    import planted
    import circuit
    import native
    import goal_eval
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    sys.path.insert(0, str(here))
    from head_counterfactual import scaled_head, raw_kl_bits
    from reward_interventions import scaled_neuron, clear_intervention_caches
    from intervention_teacher import tensor_digest
    protocol = json.loads((args.payload / "protocol.json").read_text())
    for name, expected in protocol["files"].items():
        if hashlib.sha256((args.payload / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("frozen TRAIN fixture changed")
    locations = {"redundant_000": args.redundant,
                 "cancel_001": args.extension / "teacher_cancel",
                 "gate_000": args.extension / "teacher_gate"}
    prepared = []
    for case, expected in TARGETS.items():
        root = locations[case]
        completion = (root / "final_target.txt").read_text()
        if hashlib.sha256(completion.encode()).hexdigest() != expected:
            raise RuntimeError("post-freeze target changed")
        result = json.loads((root / "frozen_search_result.json").read_text())
        if result["target_sha256"] != expected or result["target"] != completion:
            raise RuntimeError("frozen target receipt mismatch")
        teacher_binding = json.loads((root / "reference_cache.json").read_text())["binding"]
        fixture = args.payload / "cases" / case
        toy = render.from_zoo({"weights": case + ".pt", "record": case + ".json"}, fixture)
        if toy.meta["record"]["split"] != "train":
            raise RuntimeError("challenge scope is TRAIN only")
        if (toy.cfg != teacher_binding["cfg"] or
                {name: tensor_digest(value) for name, value in toy.sd.items()} != teacher_binding["source_weights"]):
            raise RuntimeError("challenge case does not match frozen teacher source weights/config")
        # This existing bank was examined in point diagnostics, but not consumed
        # by teacher fitting. It is diagnostic transfer, not a pristine holdout.
        bank = [list(ids) for ids, _ in zoo.texts(toy.meta["record"]["params"], 24, 1234)]
        if len(bank) != 24:
            raise RuntimeError("predeclared input count changed")
        nat = planted.native_of(toy.sd, toy.cfg)
        bound = copy.copy(toy)
        bound.meta = dict(toy.meta)
        bound.meta["decomposition"] = {name: (nat.target.site(name).U, nat.target.site(name).V, nat.units[name])
                                        for name in nat.names}
        original = {name: {key: tensor_digest(getattr(nat.target.site(name), key))
                           for key in ("W", "U", "V", "delta_T")}
                    for name in nat.names}
        unit_ids = {name: tensor_digest(nat.units[name]) for name in nat.names}
        if (original != teacher_binding["basis"] or unit_ids != teacher_binding["units"]
                or {name: tensor_digest(value) for name, value in nat.target.state_dict().items()} != teacher_binding["model"]):
            raise RuntimeError("challenge model/basis/unit IDs do not match frozen teacher receipt")
        architecture = [("head", layer, head) for layer in range(nat.target.n_layer)
                        for head in range(nat.target.n_head)]
        architecture += [("neuron", layer, neuron) for layer in range(nat.target.n_layer)
                         for neuron in range(nat.target.site(f"h.{layer}.mlp.down_proj").W.shape[1])
                         if bool(nat.target.site(f"h.{layer}.mlp.down_proj").W[:, neuron].ne(0).any())]
        pairs = joint_pairs(case, architecture)
        edits = fixed_edits(architecture, pairs)
        expected_worlds = {"redundant_000": 1080, "cancel_001": 1032, "gate_000": 1176}[case]
        if len(edits) * len(bank) != expected_worlds:
            raise RuntimeError("predeclared challenge work bound changed")
        graphs, invalid = [], []
        for ids in bank:
            try:
                graphs.append(circuit.unit_graph(circuit.parse(completion), bound, ids))
                invalid.append(None)
            except ValueError as exc:
                graphs.append(None)
                invalid.append(str(exc))
        full = [native.Graph({(name, pos, piece) for name in nat.names for pos in range(len(ids))
                              for piece in range(int(nat.C[name]))}, complete=True) for ids in bank]
        prepared.append((case, nat, bank, graphs, full, edits, original, unit_ids, invalid))
        write(args.out / (case + ".plan.json"), {"case": case, "inputs": bank, "edits": edits,
              "target_sha256": expected, "original_basis": original, "original_unit_ids": unit_ids,
              "candidate_worlds": expected_worlds, "forward_cap": 3 * expected_worlds,
              "scope": "post-freeze diagnostic transfer; previously examined TRAIN inputs; no target repair"})
    write(args.out / "protocol.json", {"targets": TARGETS, "candidate_worlds": 3288, "forward_cap": 9864,
          "acceptance": False, "repair": False, "bank_seed": 1234, "pairs_per_case": 6,
          "joint_scale_combinations": [[0., 0.], [0., .5], [.5, 0.], [.5, .5]],
          "scales": [0., .5], "scope": "unseen by search; reused TRAIN diagnostic, not pristine heldout tasks",
          "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                            [Path(__file__), args.reward_source, here / "reward_interventions.py", here / "head_counterfactual.py"]}})
    actual_forwards = 0
    for case, nat, bank, graphs, full, edits, original, unit_ids, invalid in prepared:
        baselines = []
        complete_rows = 0
        write(args.out / (case + ".progress.json"), {"planned": len(edits) * len(bank), "completed": 0})
        for edit_index, (kind, operations) in enumerate(edits):
            with contextlib.ExitStack() as stack:
                for unit_kind, layer, unit, scale in operations:
                    helper = scaled_head if unit_kind == "head" else scaled_neuron
                    stack.enter_context(helper(nat, layer, unit, scale))
                for index, ids in enumerate(bank):
                    clear_intervention_caches(nat)
                    p = nat.reference([ids], [len(ids) - 1]).detach().clone()
                    integrity = goal_eval.GoalEval(nat).graph_ref(ids, [len(ids) - 1], full[index], live=True, norms={})
                    actual_forwards += 2
                    raw_integrity = raw_kl_bits(p, integrity)
                    if raw_integrity > 1e-3:
                        save_file({"dense": p.contiguous(), "full": integrity.contiguous()},
                                  str(args.out / f"integrity_failure_{case}_{edit_index}_{index}.safetensors"))
                        raise RuntimeError("full-architecture challenge integrity failed")
                    clear_intervention_caches(nat)  # Candidate gets no edited dense trace.
                    q = (None if graphs[index] is None else
                         goal_eval.GoalEval(nat).graph_ref(ids, [len(ids) - 1], graphs[index], live=False))
                    actual_forwards += int(q is not None)
                    if kind == "no_edit":
                        baselines.append(p.clone())
                    p0 = baselines[index]
                    filename = f"{case}_{edit_index}_{index}.safetensors"
                    tensors = {"p": p.contiguous(), "full": integrity.contiguous(), "p0": p0.contiguous()}
                    if q is not None:
                        tensors["q"] = q.contiguous()
                    save_file(tensors, str(args.out / filename))
                    raw = None if q is None else raw_kl_bits(p, q)
                    row = {"case": case, "input": index, "ids": ids, "edit_index": edit_index,
                           "valid": q is not None, "invalid_reason": invalid[index],
                           "kind": kind, "operations": operations, "raw_kl_bits": raw,
                           "kl_bits": None if raw is None else max(0., raw), "full_integrity_raw_kl_bits": raw_integrity,
                           "top": None if q is None else int(q[0, -1].argmax()), "reference_top": int(p[0, -1].argmax()),
                           "nochange_top": int(p0[0, -1].argmax()),
                           "active": not torch.allclose(p, p0, rtol=0., atol=1e-6),
                           "file": filename, "sha256": hashlib.sha256((args.out / filename).read_bytes()).hexdigest()}
                    with (args.out / "results.jsonl").open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    complete_rows += 1
                    write(args.out / (case + ".progress.json"), {"planned": len(edits) * len(bank), "completed": complete_rows})
            clear_intervention_caches(nat)
            restored = {name: {key: tensor_digest(getattr(nat.target.site(name), key))
                              for key in ("W", "U", "V", "delta_T")} for name in nat.names}
            if restored != original or any(tensor_digest(nat.units[name]) != unit_ids[name] for name in nat.names):
                raise RuntimeError("challenge did not restore original weights/basis/unit IDs")
            write(args.out / f"{case}.restore_{edit_index}.json", {"edit_index": edit_index,
                  "basis": restored, "unit_ids": {name: tensor_digest(nat.units[name]) for name in nat.names}})
        print(json.dumps({"case": case, "completed_worlds": complete_rows}), flush=True)
    final_rows = [json.loads(line) for line in (args.out / "results.jsonl").read_text().splitlines()]
    expected_keys = {(case, edit_index, input_index) for case, _, bank, _, _, edits, _, _, _ in prepared
                     for edit_index in range(len(edits)) for input_index in range(len(bank))}
    actual_keys = [(row["case"], row["edit_index"], row["input"]) for row in final_rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise RuntimeError("challenge Cartesian coverage mismatch")
    for case, expected in TARGETS.items():
        if hashlib.sha256((locations[case] / "final_target.txt").read_bytes()).hexdigest() != expected:
            raise RuntimeError("frozen target changed during challenge")
    write(args.out / "complete.json", {"candidate_worlds": 3288, "forward_cap": 9864, "actual_forwards": actual_forwards,
          "target_repair": False, "scope": "post-freeze TRAIN diagnostic transfer"})
    summary = summarize(final_rows)
    write(args.out / "summary.json", summary)
    print("TEACHER_CHALLENGE_SUMMARY=" + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
