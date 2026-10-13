"""Fixed TRAIN intervention audit; teacher graphs are candidates, not integrity."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "payload", "reward-source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--bounded-execute", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source / folder) for folder in
                   ("experiments/oracle/toyloop", "experiments/oracle/toys", "method/oracle", "method/vpd")]
    import torch
    import render
    import circuit
    import native
    import planted
    import goal_eval
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.reward_source.parent))
    from head_counterfactual import raw_kl_bits
    spec = importlib.util.spec_from_file_location("audited_reward", args.reward_source)
    reward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward)
    from wiring_mdl_check import graph_text
    if args.out.exists():
        raise RuntimeError("do not overwrite fixed measurement")
    args.out.mkdir()
    source_files = [Path(__file__), args.reward_source,
                    args.reward_source.parent / "reward_interventions.py",
                    args.reward_source.parent / "head_counterfactual.py"]
    (args.out / "source_provenance.json").write_text(json.dumps({
        "executor": "cd8d4a9c2dfc0d3b28bc6604e0549e16aceb3aa4",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "protocol_sha256": hashlib.sha256((args.payload / "protocol.json").read_bytes()).hexdigest(),
        "scope": "TRAIN-only finite head/MLP-output intervention diagnostic; no English/joint-intervention certificate",
    }, indent=2))
    protocol = json.loads((args.payload / "protocol.json").read_text())
    for name, expected in protocol["files"].items():
        if hashlib.sha256((args.payload / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("frozen TRAIN binding changed")
    selected = ["redundant_000", "cancel_001", "gate_000"]
    result = []
    prepared = []
    for ident in selected:
        root = args.payload / "cases" / ident
        toy = render.from_zoo({"weights": ident + ".pt", "record": ident + ".json"}, root)
        assert toy.meta["record"]["split"] == "train"
        # Frozen before any outcomes: all existing record examples, without new
        # generated inputs. This transport/counterexample audit is not generalization.
        texts = [list(example["ids"]) for example in toy.meta["record"]["examples"]]
        if any(example["target"] != len(example["ids"]) - 1
               for example in toy.meta["record"]["examples"]):
            raise RuntimeError("record target is not the frozen last-position evaluation")
        target = (root / "target.txt").read_text()
        parsed = circuit.parse(target)
        if not parsed.ok or parsed.bad:
            raise RuntimeError("invalid producer target")
        units = {u for u in parsed.units if u != "out" and not u.startswith("tok")}
        edges = {e for e in parsed.edges if not e[0].startswith("tok")}
        variants = {"gold": graph_text(units, edges)}
        if ident == "cancel_001":
            pair = [u.replace(".", "") for u in toy.meta["record"]["params"]["pair"]]
            for added in ([pair[0]], [pair[1]], pair):
                variants["plus_" + "_".join(added)] = graph_text(units | set(added), edges | {(u, "out") for u in added})
        else:
            for unit in sorted(units):
                variants["remove_unit_" + unit] = graph_text(units - {unit}, {e for e in edges if unit not in e})
            for left, right in sorted(edges):
                if right == "out":
                    variants["remove_edge_" + left + "_out"] = graph_text(units, edges - {(left, right)})
        nat = planted.native_of(toy.sd, toy.cfg)
        heads = [("head", layer, head, scale) for layer in range(nat.target.n_layer)
                 for head in range(nat.target.n_head) for scale in (0., .5)]
        neurons = [("neuron", layer, neuron, scale) for layer in range(nat.target.n_layer)
                   for neuron in range(nat.target.site(f"h.{layer}.mlp.down_proj").W.shape[1])
                   if bool(nat.target.site(f"h.{layer}.mlp.down_proj").W[:, neuron].ne(0).any())
                   for scale in (0., .5)]
        panel = heads + neurons
        bound = copy.copy(toy)
        bound.meta = dict(toy.meta)
        bound.meta["decomposition"] = {n: (nat.target.site(n).U, nat.target.site(n).V, nat.units[n]) for n in nat.names}
        full = [native.Graph({(n, pos, piece) for n in nat.names for pos in range(len(ids))
                              for piece in range(int(nat.C[n]))}, complete=True) for ids in texts]
        circuit.unit_graph(circuit.parse(variants["gold"]), bound, texts[0])
        planned = {"case": ident, "texts": texts, "variants": variants, "panel": panel,
                   "worlds_per_variant": len(texts) * (len(panel) + 1),
                   "total_worlds": len(variants) * len(texts) * (len(panel) + 1)}
        (args.out / (ident + ".plan.json")).write_text(json.dumps(planned, indent=2))
        print(json.dumps({k: v for k, v in planned.items() if k not in ("texts", "variants", "panel")}), flush=True)
        prepared.append((ident, toy, nat, bound, texts, variants, panel, full, planned))
    totals = {"candidate_worlds": sum(row[-1]["total_worlds"] for row in prepared)}
    totals["forward_bound"] = 3 * totals["candidate_worlds"]
    (args.out / "planned_totals.json").write_text(json.dumps(totals, indent=2))
    print(json.dumps(totals), flush=True)
    if args.prepare_only or not args.bounded_execute:
        return
    if totals["candidate_worlds"] > 2500 or totals["forward_bound"] > 7500:
        (args.out / "bounded_abort.json").write_text(json.dumps({"reason": "predeclared work bound exceeded", **totals}))
        return
    for ident, toy, nat, bound, texts, variants, panel, full, planned in prepared:
        for name, completion in variants.items():
            try:
                for ids in texts:
                    circuit.unit_graph(circuit.parse(completion), bound, ids)
            except ValueError as exc:
                result.append({"case": ident, "variant": name, "completion": completion,
                               "valid": False, "reward": -1., "error": str(exc),
                               "per_world": None, "planned_worlds": planned["worlds_per_variant"]})
                (args.out / "results.json").write_text(json.dumps(result, indent=2))
                continue
            counter = [0]
            def state_hashes():
                return {site_name + ":" + key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        for site_name in nat.names
                        for key, value in [(key, getattr(nat.target.site(site_name), key))
                                           for key in ("W", "U", "V", "delta_T")]
                        + [("units", nat.units[site_name])]}
            original_state = state_hashes()
            baselines = []
            progress = {"case": ident, "variant": name, "planned_worlds": planned["worlds_per_variant"], "completed_worlds": 0}
            progress_path = args.out / (ident + "_" + name + ".progress.json")
            progress_path.write_text(json.dumps(progress))
            def sink(current, ids, row, p, q):
                index = counter[0] % len(texts)
                integrity = goal_eval.GoalEval(current).graph_ref(ids, [len(ids) - 1], full[index], live=True, norms={})
                signed = raw_kl_bits(p, integrity)
                filename = f"{ident}_{name}_{counter[0]}.safetensors"
                if row["kind"] == "none":
                    baselines.append(p.clone())
                p0 = baselines[index]
                save_file({"dense": p.contiguous(), "sparse": q.contiguous(), "full": integrity.contiguous(),
                           "p0": p0.contiguous()}, str(args.out / filename))
                row["distributions"] = filename
                row["distributions_sha256"] = hashlib.sha256((args.out / filename).read_bytes()).hexdigest()
                row["integrity_raw_kl_bits"] = signed
                row["original_reference_top"] = int(p0[0, -1].argmax())
                row["nochange_kl_bits"] = max(0., raw_kl_bits(p, p0))
                row["nochange_top_correct"] = row["reference_top"] == row["original_reference_top"]
                row["active_distribution_edit"] = not torch.allclose(p, p0, rtol=0., atol=1e-6)
                counter[0] += 1
                progress["completed_worlds"] = counter[0]
                progress_path.write_text(json.dumps(progress))
                if signed > 1e-3:
                    raise RuntimeError("full architecture edited parity failed")
            measured = reward.score_wiring_interventions(toy, completion, nat=nat, texts=texts,
                                                         panel=panel, evidence_sink=sink)
            restored_state = state_hashes()
            if original_state != restored_state:
                raise RuntimeError("original weights/basis/unit IDs were not restored")
            measured["original_state_sha256"] = original_state
            measured["restored_state_sha256"] = restored_state
            result.append({"case": ident, "variant": name, "completion": completion, "valid": True, **measured})
            (args.out / "results.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({"case": ident, "variant": name, "reward": measured["reward"],
                              "worst_kl": max(r["kl_bits"] for r in measured["per_world"])}), flush=True)


if __name__ == "__main__":
    main()
