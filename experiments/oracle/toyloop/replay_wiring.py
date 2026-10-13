"""Replay saved public toy truth/active answers with executed wiring, without policy generation.

The four cases are fixed existing test-instance IDs, selected before execution. Reconstruct
only their original deterministic zoo weights and first four recorded ordinary texts.
One native decomposition is shared by stated edges, architecture-complete same-unit edges,
and the same-unit stated graph with every output edge removed. Token arrows remain
annotations; embeddings stay on and remainders off. This is teacher/baseline point-fidelity
evidence, not learned-policy, English or intervention-fidelity evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
CASES = ("induction_000", "gate_001", "bigram_mlp_008", "additive_vote_015")
BASELINE_SOURCE = "9e3782861caa3e738d6b91122a527487157df349"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def original_ordinary_texts(zoo, params):
    """Exactly zoo.record's first four ordinary examples, without model/label probes."""
    family, rng = zoo.FAMILIES[params["family"]], random.Random(params["seed"] * 7 + 3)
    texts = []
    for _ in range(4):
        ids, _ = family.sample(params, 1, rng, "ordinary")[0]
        if params["family"] in ("gate", "detect"):
            ids, _ = family.sample(params, len(texts) + 1, rng, "ordinary")[-1]
        texts.append(ids)
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if sys.platform == "darwin":
        ap.error("This replay is reserved for the Linux CPU runner; do not occupy the Mac scoring slot")
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(ROOT / "experiments/oracle/toyloop"), str(ROOT / "experiments/oracle/toys"),
                    str(ROOT / "method/oracle"), str(ROOT / "method/vpd")]
    import torch
    import circuit
    import goal_eval
    import native
    import planted
    import render
    import zoo
    torch.set_num_threads(2)

    files = {name: ROOT / "experiments/oracle/toyloop/results" / f"eval_{name}.json" for name in ("gt", "active")}
    saved = {name: {r["id"]: r for r in json.loads(path.read_text())["rows"]} for name, path in files.items()}
    generator = ROOT / "experiments/oracle/toys/zoo.py"
    original_generator = subprocess.check_output(["git", "show", f"{BASELINE_SOURCE}:{generator.relative_to(ROOT)}"], cwd=ROOT)
    assert hashlib.sha256(original_generator).hexdigest() == sha(generator), "saved weight/text generator changed"
    # Ensure the completions are the exact public measurements, not local policy or teacher rewrites.
    for path in files.values():
        old = subprocess.check_output(["git", "show", f"{BASELINE_SOURCE}:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert hashlib.sha256(old).hexdigest() == sha(path), path
    checkpoint = args.out / "tasks.jsonl"
    source_paths = [Path(__file__), ROOT / "experiments/oracle/toys/zoo.py", ROOT / "experiments/oracle/toys/planted.py",
                    ROOT / "experiments/oracle/toyloop/circuit.py", ROOT / "method/oracle/goal_eval.py", ROOT / "method/oracle/native.py"]
    header = {"type": "replay", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "saved_completion_source": BASELINE_SOURCE, "fixed_cases": CASES, "torch": torch.__version__,
              "scoring": "point KL in bits; embeddings on; remainders off; all selected pieces at all positions",
              "scope": "existing planted teacher/active completions; no learned-policy or intervention/English claim",
              "provenance_limit": "original external tensor files absent; deterministic original generator/seed reconstruction, not a byte comparison to missing original tensors",
              "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files.values()},
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths}}
    append(checkpoint, header)
    results = []
    with torch.no_grad():
        for case in CASES:
            family, seed = case.rsplit("_", 1)
            params, sd = zoo.make(family, int(seed))  # replay the saved ID, not a newly sampled model
            expected = {(render.zoo_unit(a), render.zoo_unit(b)) for a, b in zoo.FAMILIES[family].edges(params)}
            assert circuit.parse(saved["gt"][case]["completion"]).edges == expected, "saved circuit/generator mismatch"
            assert all(saved[n][case]["split"] == zoo.split_of(family, int(seed)) == "test_instance" for n in saved)
            texts = original_ordinary_texts(zoo, params)
            nat = planted.native_of(sd, zoo.CFG)
            bound = SimpleNamespace(name=case, cfg=zoo.CFG, sd=sd, texts=texts,
                                    meta={"decomposition": {n: (nat.target.site(n).U, nat.target.site(n).V, nat.units[n]) for n in nat.names}})
            ge = goal_eval.GoalEval(nat)
            fingerprint = hashlib.sha256()
            for key, tensor in sorted(sd.items()):
                fingerprint.update(key.encode()); fingerprint.update(tensor.cpu().contiguous().numpy().tobytes())
            basis = hashlib.sha256()
            for name in nat.names:
                basis.update(name.encode())
                for tensor in (nat.target.site(name).U, nat.target.site(name).V, nat.units[name]):
                    basis.update(tensor.cpu().contiguous().numpy().tobytes())
            row = {"type": "task", "id": case, "family": family, "seed": int(seed), "params": params,
                   "weights_sha256": fingerprint.hexdigest(), "native_basis_sha256": basis.hexdigest(),
                   "cfg": zoo.CFG, "ordinary_texts": texts, "candidates": {}}
            for name in ("gt", "active"):
                completion = saved[name][case]["completion"]
                try:
                    graph = circuit.unit_graph(circuit.parse(completion), bound, texts[0])
                except ValueError as exc:
                    row["candidates"][name] = {"valid": False, "error": str(exc), "completion": completion}
                    continue
                measurements = []
                for ids in texts:
                    graph = circuit.unit_graph(circuit.parse(completion), bound, ids)
                    variants = {"stated": graph, "same_units_complete": native.Graph(graph.nodes, complete=True),
                                "wrong_no_output": native.Graph(graph.nodes, graph.parents, [], complete=False)}
                    assert len({frozenset(g.nodes) for g in variants.values()}) == 1
                    target = [len(ids) - 1]; ref = nat.reference([ids], target)
                    null = ge.graph_ref(ids, target, native.Graph(), live=False)
                    kl = lambda q: max(0., float((ref.exp() * (ref - q)).sum()) / math.log(2))
                    errors = {key: kl(ge.graph_ref(ids, target, g, live=False)) for key, g in variants.items()}
                    empty_error = kl(null)
                    point = {key: min(1., max(0., 1. - error / empty_error)) if empty_error > 1e-9 else None
                             for key, error in errors.items()}
                    edge_counts = {key: sum(native.implied_edges(g.nodes, target)) if g.complete else g.edges()
                                   for key, g in variants.items()}
                    measurements.append({"ids": ids, "empty_kl_bits": empty_error, "kl_bits": errors,
                                         "point_scores": point, "point_resolved": empty_error > 1e-9,
                                         "nodes": graph.count(), "edges": edge_counts})
                means = {key: sum(m["kl_bits"][key] for m in measurements) / len(measurements) for key in variants}
                row["candidates"][name] = {"valid": True, "completion": completion, "per_text": measurements,
                                           "mean_kl_bits": means, "stated_minus_complete_bits": means["stated"] - means["same_units_complete"],
                                           "wrong_minus_stated_bits": means["wrong_no_output"] - means["stated"]}
            append(checkpoint, row); results.append(row)
            print(case, {n: r.get("mean_kl_bits", r.get("error")) for n, r in row["candidates"].items()}, flush=True)
    complete = len(results) == 4 and all(c["valid"] for r in results for c in r["candidates"].values())
    (args.out / "report.json").write_text(json.dumps({"provenance": header, "complete": complete, "tasks": results}, indent=2) + "\n")
    if not complete:
        raise RuntimeError("Replay incomplete: inspect retained materialization/baseline failures before interpreting scores")


if __name__ == "__main__":
    main()
