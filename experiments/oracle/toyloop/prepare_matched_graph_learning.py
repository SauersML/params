"""Offline assembly of all fixed TRAIN pairs and the common 32-model panel.

Run on a memory-leased/free CPU worker. No Native, scoring, generation or API
calls. Missing proofs remain explicit rows; preparation never drops them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise RuntimeError(f"frozen file changed: {path}")
    return Path(path)


def retention_bank(root, receipt, file_hashes):
    """Exporter records both byte SHA and historical default-JSON content SHA."""
    relative = "cases/" + receipt["id"] + "/bank.json"
    path = checked(root / relative, file_hashes[relative])
    inputs = json.loads(path.read_text())
    if hashlib.sha256(json.dumps(inputs).encode()).hexdigest() != receipt["input_bank_sha256"]:
        raise RuntimeError("retention bank content differs from historical scoring receipt")
    if len(inputs) != receipt["input_bank_count"]:
        raise RuntimeError("retention bank count changed")
    return inputs


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def proof_location(ident, roots):
    """Ambiguous complete proofs are an error, never choose a passing one."""
    found = [root / ident for root in roots
             if (root / ident / "search.json").exists()
             or (root / ident / "frozen_search_result.json").exists()]
    if len(found) > 1:
        raise RuntimeError(f"multiple producer results for {ident}")
    return found[0] if found else None


def architecture_edits(toy):
    from intervention_teacher_batch_challenge import joint_pairs, fixed_edits
    units = [("head", layer, head) for layer in range(toy.cfg["n_layer"])
             for head in range(toy.cfg["n_head"])]
    for layer in range(toy.cfg["n_layer"]):
        weight = toy.sd[f"h.{layer}.mlp.down_proj.weight"]
        units.extend(("neuron", layer, index) for index in range(weight.shape[1])
                     if bool(weight[:, index].ne(0).any()))
    return [{"kind": kind, "operations": [list(op) for op in operations]}
            for kind, operations in fixed_edits(units, joint_pairs(toy.name, units))]


def load_toy(weights, record):
    import torch
    import render
    value = torch.load(weights, map_location="cpu", weights_only=False)
    data = json.loads(Path(record).read_text())
    ids = [row["ids"] for row in data["examples"] if row["kind"] == "ordinary"][:4]
    if len(ids) != 4 or any(row["target"] != len(row["ids"]) - 1
                            for row in data["examples"] if row["kind"] == "ordinary"):
        raise RuntimeError("four original last-position ordinary contexts required")
    vocab = ["BOS"] + [f"t{i}" for i in range(1, value["sd"]["wte.weight"].shape[0])]
    return render.Toy(data["id"], value["sd"], value["cfg"], vocab, ids,
                      None, {"record": data})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy", "bank", "retention", "executor", "tokenizer", "prices", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--proof-root", type=Path, action="append", required=True)
    parser.add_argument("--bank-sha256", required=True)
    parser.add_argument("--legacy-preflight-sha256", required=True)
    parser.add_argument("--retention-manifest-sha256", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("fresh output directory required")
    args.out.mkdir(parents=True)
    import render
    import assemble_graph_supervision as assembly
    import matched_graph_supervision as matched
    from intervention_teacher import tensor_digest

    train_ids, eval_ids = matched.expected_ids()
    bank = json.loads(checked(args.bank / "manifest.json", args.bank_sha256).read_text())
    legacy = json.loads(checked(args.legacy / "batch_preflight.json",
                               args.legacy_preflight_sha256).read_text())
    retained = json.loads(checked(args.retention / "manifest.json",
                                 args.retention_manifest_sha256).read_text())
    for relative, digest in retained["files"].items():
        checked(args.retention / relative, digest)
    models, entries = {}, {}
    legacy_rows = {row["case"]: row for row in legacy["models"]}
    bank_rows = {row["case"]: row for row in bank["models"]}
    for ident in [*train_ids, *eval_ids]:
        if ident in legacy_rows:
            row = legacy_rows[ident]
            root = args.legacy / "preflight" / ident
            weights = checked(root / (ident + ".pt"), row["weight_sha256"])
            record = checked(root / (ident + ".json"), row["record_sha256"])
            point_path, point_sha, is_legacy = record, row["record_sha256"], True
        else:
            row = bank_rows[ident]
            root = args.bank / ident
            for name, digest in row["files"].items():
                checked(root / name, digest)
            weights, record = root / "weights.pt", root / "record.json"
            point_path = root / "point_gold_control.json"
            point_sha, is_legacy = row["files"][point_path.name], False
        toy = load_toy(weights, record)
        if toy.name != ident:
            raise RuntimeError("model catalog identity mismatch")
        actual = {name: tensor_digest(value) for name, value in toy.sd.items()}
        if ident in bank_rows and actual != bank_rows[ident]["tensor_sha256"]:
            raise RuntimeError("model tensor inventory mismatch")
        models[ident] = (toy, weights)
        if ident in train_ids:
            point, binding = assembly.read_point_control(point_path, ident=ident,
                expected_sha256=point_sha, source_weights=actual, legacy=is_legacy)
            location = (args.legacy / "teacher_batch" / ident if is_legacy
                        else proof_location(ident, args.proof_root))
            try:
                bundle = assembly.read_bundle(location) if location else None
            except Exception as exc:
                entries[ident] = {"toy": toy, "point_control": point,
                                  "point_binding": binding, "bundle": None}
                dump(args.out / (ident + ".producer_error.json"),
                     {"error": str(exc), "root": str(location)})
                continue
            entries[ident] = {"toy": toy, "point_control": point,
                              "point_binding": binding, "bundle": bundle}
    catalog = assembly.assemble_catalog(entries, render)
    dump(args.out / "assembly_status.json", catalog)
    jsonl(args.out / "train.jsonl", catalog["rows"])
    panel, score_panel = [], []
    assets = args.out / "assets"
    assets.mkdir()

    def add_panel(toy, weights, prompt, inputs, roles, stratum, answer_format):
        relative = "assets/" + toy.name + ".pt"
        shutil.copyfile(weights, args.out / relative)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        panel.append({"id": toy.name, "stratum": stratum, "prompt": prompt,
                      "prompt_sha256": digest})
        score_panel.append({"id": toy.name, "stratum": stratum, "prompt_sha256": digest,
            "answer_format": answer_format, "weights": relative, "weights_sha256": sha(weights),
            "inputs": inputs, "input_roles": roles, "edits": architecture_edits(toy)})

    for ident in eval_ids:
        toy, weights = models[ident]
        add_panel(toy, weights, assembly.source_prompt(toy, toy.texts, render), toy.texts,
                  ["disclosed"] * len(toy.texts), "new_model", "graph_only")
    question_rows = [json.loads(line) for line in
                     (args.retention / "panel_questions.jsonl").read_text().splitlines()]
    if [row["id"] for row in question_rows] != [row["id"] for row in retained["panel"]]:
        raise RuntimeError("retention order changed")
    for question, receipt in zip(question_rows, retained["panel"]):
        ident = question["id"]
        root = args.retention / "cases" / ident
        toy = load_toy(root / (ident + ".pt"), root / (ident + ".json"))
        prompt = question["prompt"]
        if hashlib.sha256(prompt.encode()).hexdigest() != receipt["prompt_sha256"]:
            raise RuntimeError("retention prompt changed")
        inputs = retention_bank(args.retention, receipt, retained["files"])
        disclosed = toy.texts[:2]
        add_panel(toy, root / (ident + ".pt"), prompt, inputs,
                  ["disclosed" if ids in disclosed else "extra_diagnostic" for ids in inputs],
                  "retention", "legacy_mermaid_english")
    jsonl(args.out / "panel.jsonl", panel)
    source_files = ["experiments/oracle/toyloop/circuit.py", "experiments/oracle/toys/planted.py",
                    "method/oracle/native.py", "method/oracle/native_units.py", "method/oracle/goal_eval.py",
                    "method/vpd/vpd_model.py", "method/oracle/carry.py", "method/oracle/mech.py"]
    prefix = "experiments/oracle/toyloop/"
    support_files = ["head_counterfactual.py", "reward_interventions.py", "intervention_teacher.py"]
    own_root = Path(__file__).resolve().parent
    manifest = {"schema_version": 1, "arms": ["base", "point", "intervention"], "panel": score_panel,
        "source_sha256": {name: sha(args.executor / name) for name in source_files},
        "support_sha256": {prefix + name: sha(own_root / name) for name in support_files},
        "forward_cap": 5 * sum(len(row["inputs"]) * len(row["edits"]) for row in score_panel)}
    dump(args.out / "scoring_manifest.json", manifest)
    if not catalog["complete_pairing"]:
        dump(args.out / "complete.json", {"status": "incomplete_teacher_pairing", "api_calls": 0})
        return
    prepared = matched.prepare(catalog["rows"], panel, args.tokenizer)
    serialized, digest = matched.serialize_prepared(prepared)
    (args.out / "prepared.json").write_text(serialized)
    (args.out / "prepared.sha256").write_text(digest + "\n")
    dump(args.out / "contract.json", prepared["contract"])
    rates = json.loads(args.prices.read_text())
    matches = [row for row in rates if row["tinker_id"] == prepared["contract"]["model"]]
    if len(matches) != 1:
        raise RuntimeError("unique official model rate entry required")
    rates = matches[0]
    cost = matched.cost_bound(prepared["contract"],
        {"trained": float(rates["train"].removeprefix("$")),
         "prefill": float(rates["prefill"].removeprefix("$")),
         "sampled": float(rates["sample"].removeprefix("$"))})
    cost["ratecard_sha256"] = sha(args.prices)
    dump(args.out / "cost.json", cost)
    import importlib.metadata
    dump(args.out / "provenance.json", {
        "assembly_source_sha256": sha(__file__),
        "helper_sha256": {path.name: sha(path) for path in
            [Path(assembly.__file__), Path(matched.__file__), Path(render.__file__)]},
        "tokenizer_files": {str(path.relative_to(args.tokenizer)): sha(path)
            for path in sorted(args.tokenizer.rglob("*")) if path.is_file()},
        "packages": {name: importlib.metadata.version(name) for name in
            ["torch", "transformers", "tokenizers", "tinker", "tinker-cookbook"]},
        "bank_manifest_sha256": args.bank_sha256,
        "legacy_preflight_sha256": args.legacy_preflight_sha256,
        "retention_manifest_sha256": args.retention_manifest_sha256})
    dump(args.out / "complete.json", {"status": "prepared", "train_rows": 24, "panel_rows": 32,
        "prepared_sha256": digest, "scoring_manifest_sha256": sha(args.out / "scoring_manifest.json"),
        "api_calls": 0, "native_forwards": 0, "scope": assembly.LEARNING_SCOPE})


if __name__ == "__main__":
    main()
