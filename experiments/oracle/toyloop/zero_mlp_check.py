"""Fixed gate zero-support route equality and weight-only TRAIN batch preflight."""
import argparse
import copy
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


BATCH = ("additive_vote_000", "bigram_mlp_000", "binding_positional_001", "cancel_002",
         "copy_offset_000", "gate_003", "induction_001", "redundant_001")


def write(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "payload", "targets", "teacher-source", "preflight", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("do not overwrite proof measurement")
    args.out.mkdir()
    here = args.teacher_source.parent
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
    sys.path.insert(0, str(here))
    import intervention_teacher as teacher
    from head_counterfactual import scaled_head, raw_kl_bits
    from reward_interventions import scaled_neuron, clear_intervention_caches
    source_files = [Path(__file__), args.teacher_source, here / "reward_interventions.py",
                    here / "head_counterfactual.py", here / "reward_exec.py"]
    write(args.out / "contract.json", {"case": "gate_000", "worlds": 100, "forward_cap": 600,
          "target_sha256": "edb35f53dcfe92b63e8c3f77cf6cc09a3d433a8d70e1291c813ff63135a72ceb",
          "proof": "Only exact disjoint FC-U/down-V coordinate support through same-layer elementwise GELU",
          "comparison": "Frozen gate and full unit candidate, each before/after; no target chosen from comparison",
          "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}})
    protocol = json.loads((args.payload / "protocol.json").read_text())
    for name, expected in protocol["files"].items():
        if hashlib.sha256((args.payload / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("gate fixture binding changed")
    # These eight existing TRAIN models are new to intervention-aware search,
    # not unseen by the oracle trained on the original226 catalog. No labels run.
    batch_rows = []
    for case in BATCH:
        root = args.preflight / case
        record = json.loads((root / (case + ".json")).read_text())
        if record["split"] != "train" or record["id"] != case:
            raise RuntimeError("preflight TRAIN identity changed")
        examples = [example for example in record["examples"] if example["kind"] == "ordinary"][:4]
        if len(examples) != 4 or any(example["target"] != len(example["ids"]) - 1 for example in examples):
            raise RuntimeError("predeclared four ordinary examples unavailable")
        blob = torch.load(root / (case + ".pt"), map_location="cpu", weights_only=True)
        sd, cfg = blob["sd"], blob["cfg"]
        neurons = sum(int(sd[f"h.{layer}.mlp.down_proj.weight"].ne(0).any(dim=0).sum()) for layer in range(cfg["n_layer"]))
        worlds = 4 * (1 + 2 * (cfg["n_layer"] * cfg["n_head"] + neurons))
        batch_rows.append({"case": case, "family": record["family"], "ordinary_inputs": [e["ids"] for e in examples],
                           "worlds": worlds, "max_attempts": 64, "forward_cap": 69 * worlds,
                           "weight_sha256": hashlib.sha256((root / (case + ".pt")).read_bytes()).hexdigest(),
                           "record_sha256": hashlib.sha256((root / (case + ".json")).read_bytes()).hexdigest()})
    batch_summary = {"models": batch_rows, "total_forward_cap": sum(row["forward_cap"] for row in batch_rows),
                     "within_proposed_60000": sum(row["forward_cap"] for row in batch_rows) <= 60000,
                     "label_search_forwards": 0, "scope": "Weight-only preflight, original oracle TRAIN models"}
    write(args.out / "batch_preflight.json", batch_summary)
    print("TRAIN_BATCH_PREFLIGHT=" + json.dumps(batch_summary), flush=True)
    case = "gate_000"
    root = args.payload / "cases" / case
    record = json.loads((root / (case + ".json")).read_text())
    texts = [example["ids"] for example in record["examples"]]
    if record["split"] != "train" or len(texts) != 4:
        raise RuntimeError("fixed gate input panel changed")
    toy = render.from_zoo({"weights": case + ".pt", "record": case + ".json"}, root)
    nat = planted.native_of(toy.sd, toy.cfg)
    frozen = (args.targets / "teacher_gate" / "final_target.txt").read_text()
    if hashlib.sha256(frozen.encode()).hexdigest() != "edb35f53dcfe92b63e8c3f77cf6cc09a3d433a8d70e1291c813ff63135a72ceb":
        raise RuntimeError("frozen target changed")
    units, full_edges = teacher.architecture_candidate(toy, nat, texts)
    parsed = circuit.parse(frozen)
    frozen_units = {unit for unit in parsed.units if unit != "out" and not unit.startswith("tok")}
    original_edges = set(parsed.edges)
    prune = lambda edges: {edge for edge in edges if not teacher.zero_elementwise_mlp_route(nat, *edge)}
    variants = {"frozen_before": frozen, "frozen_after": teacher.graph_text(frozen_units, prune(original_edges)),
                "full_before": teacher.graph_text(units, full_edges), "full_after": teacher.graph_text(units, prune(full_edges))}
    bound = copy.copy(toy)
    bound.meta = dict(toy.meta)
    bound.meta["decomposition"] = {name: (nat.target.site(name).U, nat.target.site(name).V, nat.units[name]) for name in nat.names}
    graphs = {name: [circuit.unit_graph(circuit.parse(text), bound, ids) for ids in texts] for name, text in variants.items()}
    all_graphs = [native.Graph({(name, pos, piece) for name in nat.names for pos in range(len(ids))
                               for piece in range(int(nat.C[name]))}, complete=True) for ids in texts]
    panel = [("head", layer, head, scale) for layer in range(nat.target.n_layer)
             for head in range(nat.target.n_head) for scale in (0., .5)]
    panel += [("neuron", layer, neuron, scale) for layer in range(nat.target.n_layer)
              for neuron in range(nat.target.site(f"h.{layer}.mlp.down_proj").W.shape[1])
              if bool(nat.target.site(f"h.{layer}.mlp.down_proj").W[:, neuron].ne(0).any()) for scale in (0., .5)]
    worlds = [("none", None, None, 1.)] + panel
    if len(worlds) * len(texts) != 100:
        raise RuntimeError("fixed 100-world bound changed")
    original = teacher.binding(toy, nat, texts, panel)
    write(args.out / "plan.json", {"variants": variants, "inputs": texts, "worlds": worlds,
          "removed_frozen_edges": sorted(original_edges - prune(original_edges)),
          "removed_full_edges": sorted(full_edges - prune(full_edges)), "binding": original})
    rows = []
    for world_index, (kind, layer, unit, scale) in enumerate(worlds):
        edit = contextlib.nullcontext() if kind == "none" else (scaled_head if kind == "head" else scaled_neuron)(nat, layer, unit, scale)
        with edit:
            for input_index, ids in enumerate(texts):
                clear_intervention_caches(nat)
                p = nat.reference([ids], [len(ids) - 1]).clone()
                full = goal_eval.GoalEval(nat).graph_ref(ids, [len(ids) - 1], all_graphs[input_index], live=True, norms={})
                if raw_kl_bits(p, full) > .001:
                    raise RuntimeError("full architecture integrity failed")
                clear_intervention_caches(nat)
                q = {name: goal_eval.GoalEval(nat).graph_ref(ids, [len(ids) - 1], values[input_index], live=False)
                     for name, values in graphs.items()}
                filename = f"world_{world_index}_{input_index}.safetensors"
                save_file({"p": p.contiguous(), "full": full.contiguous(), **{name: value.contiguous() for name, value in q.items()}}, str(args.out / filename))
                row = {"world": world_index, "input": input_index, "kind": kind, "layer": layer, "unit": unit, "scale": scale,
                       "file": filename, "sha256": hashlib.sha256((args.out / filename).read_bytes()).hexdigest(),
                       "comparisons": {base: {"bitwise_equal": torch.equal(q[base + "_before"], q[base + "_after"]),
                                              "max_abs_logp_difference": float((q[base + "_before"] - q[base + "_after"]).abs().max()),
                                              "before_kl": raw_kl_bits(p, q[base + "_before"]),
                                              "after_kl": raw_kl_bits(p, q[base + "_after"])} for base in ("frozen", "full")}}
                rows.append(row)
                write(args.out / "results.json", rows)
        clear_intervention_caches(nat)
        if teacher.binding(toy, nat, texts, panel) != original:
            raise RuntimeError("source state restoration failed")
    summary = {"worlds": len(rows), "forwards": 600, "frozen_targets_unchanged": True,
               "comparisons": {base: {"bitwise_equal_worlds": sum(row["comparisons"][base]["bitwise_equal"] for row in rows),
                                     "max_abs_logp_difference": max(row["comparisons"][base]["max_abs_logp_difference"] for row in rows)}
                               for base in ("frozen", "full")}}
    write(args.out / "summary.json", summary)
    print("ZERO_MLP_SUMMARY=" + json.dumps(summary), flush=True)
    if any(not row["comparisons"][base]["bitwise_equal"] for row in rows for base in ("frozen", "full")):
        raise RuntimeError("physical before/after distributions differ; no tolerance relaxation")


if __name__ == "__main__":
    main()
