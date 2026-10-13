"""Bind measured TRAIN targets to actual weights and matched source observations.

No Native/scoring calls occur here. Weight rendering can compute decompositions;
run assembly under the memory ledger or on a free worker. Missing teachers stay
explicit records. Point-target annotation removal is a declared graph-only task
conversion, never a repair to the intervention target.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re


INSTRUCTION = ("Return only an executable Mermaid flowchart LR. State head/neuron units and "
               "their directed executable connections, including connections to out. "
               "Verification uses these displayed inputs under individual head-output "
               "and original nonzero MLP-output neuron deletion/half-scaling. "
               "Do not include English or token annotations.")


def unit_name(name):
    if name in ("out", "OUT"):
        return "out"
    match = re.fullmatch(r"L(\d+)\.?([HN])(\d+)", name)
    if not match:
        raise ValueError("unrecognized executable source unit")
    return "L" + match[1] + match[2] + match[3]


def point_graph(edges, context_units):
    """Discard only nonexecuted token annotation arrows; retain all unit routes."""
    units, routes, annotations = set(), set(), []
    for left, right in edges:
        if left.startswith("tok:") or right.startswith("tok:"):
            annotations.append([left, right])
            continue
        route = (unit_name(left), unit_name(right))
        routes.add(route)
        units.update(route)
    for group in context_units:
        for spec in group:
            units.add(unit_name(spec["unit"] if isinstance(spec, dict) else spec))
    body = ["flowchart LR"] + ["  " + name for name in sorted(units)]
    body += ["  " + left + " --> " + right for left, right in sorted(routes)]
    return "\n".join(body) + "\n", {"discarded_nonexecuted_token_arrows": annotations,
                                    "retained_executable_routes": sorted(routes)}


def read_point_control(path, *, ident, expected_sha256, source_weights, legacy=False):
    """Use the external frozen model manifest's file hash, never a self-hash."""
    path = Path(path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError("point-control source file differs from frozen model manifest")
    data = json.loads(raw)
    if legacy:
        if data["id"] != ident or data["split"] != "train":
            raise RuntimeError("legacy point-control model identity mismatch")
        data = {"unit_edges": data["unit_edges"], "context_units": [data["gt_units"]]}
    return data, {"case": ident, "source_weights": source_weights, "source_file_sha256": actual}


def verify_teacher(bundle, *, ident, cfg, inputs, source_weights):
    """Pure receipt validation; numeric evidence is read, never recomputed."""
    target, result, cache, original, restored, uncached = (
        bundle[key] for key in ("target", "result", "cache", "original", "restored", "uncached"))
    digest = hashlib.sha256(target.encode()).hexdigest()
    if bundle["contract"]["case"] != ident:
        raise RuntimeError("teacher case identity mismatch")
    selected = result["selected"]
    if (result["target"] != target or result["target_sha256"] != digest or
            selected.get("completion") != target or selected.get("completion_sha256") != digest):
        raise RuntimeError("frozen target differs from selected measured completion")
    binding = cache["binding"]
    if (original != restored or original["cfg"] != cfg or original["inputs"] != inputs or
            binding["cfg"] != cfg or binding["inputs"] != inputs or
            binding["source_weights"] != source_weights or original["source_weights"] != source_weights):
        raise RuntimeError("teacher model/input/restoration binding mismatch")
    expected_worlds = len(inputs) * (1 + len(binding["panel"]))
    rows = uncached["per_world"]
    if len(rows) != expected_worlds or rows != selected["score"]["per_world"]:
        raise RuntimeError("uncached selected-target crosscheck mismatch")
    if any(not math.isfinite(row["kl_bits"]) or row["kl_bits"] > .001 for row in rows):
        raise RuntimeError("teacher exceeds fixed finite-panel KL bound")
    integrity = cache["integrity_raw_kl_bits"]
    if len(integrity) != expected_worlds or any(not math.isfinite(value) or value > .001 for value in integrity):
        raise RuntimeError("full-architecture integrity receipt failed")
    edits = [("none", None, None, 1.)] + [tuple(edit) for edit in binding["panel"]]
    actual = [(tuple(row["ids"]), row["kind"], row["layer"], row["unit"], row["scale"]) for row in rows]
    # Repeated disclosed contexts are legitimate; multiplicities must match.
    from collections import Counter
    expected_counts = Counter((tuple(ids), *edit) for ids in inputs for edit in edits)
    if Counter(actual) != expected_counts:
        raise RuntimeError("teacher Cartesian intervention/input coverage mismatch")
    return {"id": ident, "target_sha256": digest, "worlds": expected_worlds,
            "worst_kl_bits": max(row["kl_bits"] for row in rows),
            "status": result["status"], "scope": "disclosed contexts and fitted individual panel only"}


def read_bundle(root):
    """Read either frozen legacy batch or new-bank producer receipts explicitly."""
    root = Path(root)
    paths = [path for path in (root / "search.json", root / "frozen_search_result.json") if path.exists()]
    if len(paths) != 1:
        raise RuntimeError("exactly one frozen teacher result required")
    result = json.loads(paths[0].read_text())
    if result.get("target") is None:
        return {"target": None, "status": result["status"]}
    files = {"cache": "reference_cache.json", "original": "original_source_binding.json",
             "restored": "restored_source_binding.json", "uncached": "uncached_final.json",
             "contract": "pre_native_contract.json"}
    bundle = {key: json.loads((root / name).read_text()) for key, name in files.items()}
    bundle.update(target=(root / "final_target.txt").read_text(), result=result,
                  source_receipt_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in [paths[0], *(root / name for name in files.values())]})
    return bundle


def source_prompt(toy, inputs, renderer):
    """Render only shared model facts, with no truth/proof metadata in scope."""
    rendered = copy.copy(toy)
    rendered.texts = inputs
    rendered.truth = None
    rendered.meta = {}
    previous, previous_active = renderer.SECTIONS, renderer.SHOW_ACTIVE
    try:
        renderer.SECTIONS = "weights"
        renderer.SHOW_ACTIVE = False
        observation = renderer.render(rendered, n_texts=4, all_live_neurons=True,
                                      explain_summary_limits=True)
    finally:
        renderer.SECTIONS, renderer.SHOW_ACTIVE = previous, previous_active
    if not observation.endswith(renderer.FORMAT):
        raise RuntimeError("source observation rendering contract changed")
    return observation[:-len(renderer.FORMAT)] + INSTRUCTION


def assemble_one(toy, bundle, point_control, renderer, *, point_binding):
    """A single common prompt with point/intervention graph-only targets."""
    record = toy.meta["record"]
    if record["split"] != "train" or record["id"] != toy.name:
        raise RuntimeError("supervision is exact-identity TRAIN only")
    if bundle is None or bundle.get("target") is None:
        return {"id": toy.name, "status": "missing_or_infeasible_teacher", "usable": False,
                "teacher_status": None if bundle is None else bundle.get("status")}
    from intervention_teacher import tensor_digest
    from matched_graph_supervision import graph_only
    inputs = [list(example["ids"]) for example in record["examples"] if example["kind"] == "ordinary"][:4]
    if len(inputs) != 4 or any(example["target"] != len(example["ids"]) - 1
                               for example in record["examples"] if example["kind"] == "ordinary"):
        raise RuntimeError("four disclosed last-position ordinary contexts required")
    actual_weights = {name: tensor_digest(value) for name, value in toy.sd.items()}
    if point_binding["case"] != toy.name or point_binding["source_weights"] != actual_weights:
        raise RuntimeError("point-control belongs to a different actual model")
    receipt = verify_teacher(bundle, ident=toy.name, cfg=toy.cfg, inputs=inputs, source_weights=actual_weights)
    point, conversion = point_graph(point_control["unit_edges"], point_control["context_units"])
    graph_only(point)
    graph_only(bundle["target"])
    prompt = source_prompt(toy, inputs, renderer)
    target_units = set(re.findall(r"L\d+[HN]\d+", point + bundle["target"]))
    missing = sorted(target_units - set(re.findall(r"L\d+[HN]\d+", prompt)))
    return {"id": toy.name, "status": "assembled" if not missing else "missing_observed_units",
            "usable": not missing, "prompt": prompt, "point_completion": point,
            "intervention_completion": bundle["target"], "missing_target_units": missing,
            "proof": receipt, "point_conversion": conversion,
            "point_source_binding": point_binding,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "renderer_sha256": hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest(),
            "scope": "matched graph-only context-specific supervision; no English/general-intervention certificate"}


def assemble_catalog(entries, renderer):
    """Retain every fixed ID, including missing constructions/teachers/errors.

    Entries hold actual loaded toy, read_bundle output, and source-bound
    point_control data. File/manifest checks belong to the model-bank loader;
    actual tensor/config/context hashes are checked again by assemble_one.
    """
    from matched_graph_supervision import expected_ids
    ids, _ = expected_ids()
    if set(entries) - set(ids):
        raise RuntimeError("unexpected model ID; no substitution permitted")
    rows = []
    for ident in ids:
        entry = entries.get(ident)
        if entry is None:
            rows.append({"id": ident, "status": "missing_predeclared_model", "usable": False})
            continue
        try:
            rows.append(assemble_one(entry["toy"], entry.get("bundle"), entry["point_control"], renderer,
                                     point_binding=entry["point_binding"]))
        except Exception as exc:
            rows.append({"id": ident, "status": "assembly_error", "usable": False,
                         "error_type": type(exc).__name__, "error": str(exc)})
    return {"rows": rows, "complete_pairing": all(row["usable"] for row in rows),
            "scope": "all24 predeclared TRAIN IDs retained; no silent dropping or replacements"}
