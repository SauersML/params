"""Opt-in TRAIN-only finite-intervention graph supervision; no English labels."""
import copy
import hashlib
import json
from pathlib import Path
import re
import platform


WORST_KL_MAX = 1e-3
MAX_ATTEMPTS = 64


def tensor_digest(value):
    import torch
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def binding(toy, nat, texts, panel):
    import torch
    import reward_exec
    import reward_interventions
    import head_counterfactual
    import native
    import goal_eval
    import circuit
    import vpd_model
    modules = (reward_exec, reward_interventions, head_counterfactual, native,
               goal_eval, circuit, vpd_model)
    return {"cfg": copy.deepcopy(toy.cfg), "inputs": [list(ids) for ids in texts],
            "target_cfg": copy.deepcopy(nat.target.cfg),
            "runtime": {"torch": torch.__version__, "threads": torch.get_num_threads(),
                        "platform": platform.platform(),
                        "backend_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
                        "device": str(nat.target.wte.device), "eps": nat.target.eps,
                        "behavior": None if nat.target.behavior is None else tensor_digest(nat.target.behavior),
                        "behavior_tokens": copy.deepcopy(nat.target.behavior_tokens),
                        "behavior_at": copy.deepcopy(nat.target.behavior_at)},
            "panel": [list(entry) for entry in panel],
            "source_weights": {name: tensor_digest(value) for name, value in toy.sd.items()},
            "model": {name: tensor_digest(value) for name, value in nat.target.state_dict().items()},
            "basis": {name: {key: tensor_digest(getattr(nat.target.site(name), key))
                             for key in ("W", "U", "V", "delta_T")}
                      for name in nat.names},
            "units": {name: tensor_digest(nat.units[name]) for name in nat.names},
            "components": {name: int(nat.C[name]) for name in nat.names},
            "source": {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                       for module in modules},
            "teacher_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def validate_cache(cache, toy, nat, texts, panel):
    from head_counterfactual import raw_kl_bits
    allowed = {"binding", "references", "reference_sha256", "integrity_raw_kl_bits",
               "preparation_forward_counts"}
    if set(cache) - allowed:
        raise RuntimeError("unsupported reference cache content")
    if cache.get("binding") != binding(toy, nat, texts, panel):
        raise RuntimeError("intervention reference cache binding mismatch")
    expected = len(texts) * (len(panel) + 1)
    if any(len(cache.get(key, [])) != expected for key in
           ("references", "reference_sha256", "integrity_raw_kl_bits")):
        raise RuntimeError("incomplete reference cache")
    for p, digest, integrity in zip(cache["references"], cache["reference_sha256"],
                                    cache["integrity_raw_kl_bits"]):
        if tuple(p.shape) != (1, 1, int(nat.target.wte.shape[0])):
            raise RuntimeError("cached reference shape mismatch")
        if p.requires_grad or tensor_digest(p) != digest:
            raise RuntimeError("corrupted reference cache")
        try:
            raw_kl_bits(p, p)
        except ValueError as exc:
            raise RuntimeError("invalid cached reference distribution") from exc
        if not isinstance(integrity, (int, float)) or not -1e-6 <= integrity <= WORST_KL_MAX:
            raise RuntimeError("invalid cached full-architecture integrity")


def architecture_candidate(toy, nat, texts):
    """Unit universe from actual original nonzero pieces, without teacher truth."""
    import native
    import circuit
    units = set()
    for name in nat.names:
        site = nat.target.site(name)
        layer = int(name.split(".")[1])
        kind = "N" if ".mlp." in name else "H"
        for piece, owner in enumerate(nat.units[name].tolist()):
            if owner >= 0 and bool(site.U[piece].ne(0).any()) and bool(site.V[:, piece].ne(0).any()):
                units.add(f"L{layer}{kind}{owner}")
    bound = copy.copy(toy)
    bound.meta = dict(toy.meta)
    bound.meta["decomposition"] = {name: (nat.target.site(name).U, nat.target.site(name).V, nat.units[name])
                                    for name in nat.names}
    possible = None
    for ids in texts:
        owners = {(site, pos, piece): unit for unit in units
                  for site, piece in circuit.unit_pieces(bound, unit) for pos in range(len(ids))}
        parents, outputs = native.all_edges(sorted(owners), [len(ids) - 1])
        edges = {(owners[w], owners[r]) for r, writers in parents.items() for w in writers
                 if owners[w] != owners[r]}
        edges |= {(owners[w], "out") for w in outputs}
        possible = edges if possible is None else possible & edges
    return units, possible or set()


def graph_text(units, edges):
    return "flowchart LR\n" + "".join(f"  {unit}\n" for unit in sorted(units)) + "".join(
        f"  {left} --> {right}\n" for left, right in sorted(edges))


def prepare_reference_cache(toy, nat, texts, panel=None, evidence_sink=None):
    import native
    import goal_eval
    import reward_exec
    from head_counterfactual import raw_kl_bits
    units, edges = architecture_candidate(toy, nat, texts)
    references, integrities = [], []
    full = [native.Graph({(name, pos, piece) for name in nat.names for pos in range(len(ids))
                          for piece in range(int(nat.C[name]))}, complete=True) for ids in texts]

    def retain(current, ids, row, p, q):
        index = len(references) % len(texts)
        complete = goal_eval.GoalEval(current).graph_ref(ids, [len(ids) - 1], full[index], live=True, norms={})
        try:
            integrity = raw_kl_bits(p, complete)
        except ValueError as exc:
            raise RuntimeError("invalid full-architecture reference") from exc
        if integrity > WORST_KL_MAX:
            raise RuntimeError("full-architecture reference integrity failed")
        references.append(p.detach().clone())
        integrities.append(integrity)
        if evidence_sink is not None:
            evidence_sink("reference", len(references) - 1, row, p, q, complete)

    original_state = binding(toy, nat, texts, [])
    start = reward_exec.score_wiring_interventions(toy, graph_text(units, edges), nat=nat,
                                                   texts=texts, panel=panel, evidence_sink=retain)
    if original_state != binding(toy, nat, texts, []):
        raise RuntimeError("reference preparation did not restore source state")
    cache = {"binding": binding(toy, nat, texts, start["panel"]), "references": references,
             "reference_sha256": [tensor_digest(p) for p in references],
             "integrity_raw_kl_bits": integrities,
             "preparation_forward_counts": {"dense": len(references), "full": len(references),
                                             "standalone_sparse": len(references)}}
    validate_cache(cache, toy, nat, texts, start["panel"])
    return cache, start, units, edges


def search_teacher(toy, *, nat, texts, panel=None, attempt_sink=None, evidence_sink=None):
    """Fixed-order greedy pruning; finite-panel feasibility, not exact optimization."""
    import reward_exec
    record = toy.meta.get("record", {})
    if record.get("split") != "train":
        raise RuntimeError("intervention teacher is TRAIN-only")
    cache, start, units, edges = prepare_reference_cache(toy, nat, texts, panel, evidence_sink)
    original = copy.deepcopy(cache["binding"])
    attempts = []

    def describe(label, measured):
        return {"candidate": label, "worst_kl_bits": max(row["kl_bits"] for row in measured["per_world"]),
                "description_bits": sum(row["description_bits"] for row in measured["per_world"]) / len(measured["per_world"]),
                "score": measured}

    current = describe("full_standalone", start)
    current["completion"] = graph_text(units, edges)
    current["completion_sha256"] = hashlib.sha256(current["completion"].encode()).hexdigest()
    if current["worst_kl_bits"] > WORST_KL_MAX:
        return {"status": "infeasible_full_standalone", "target": None, "start": current,
                "attempts": attempts, "cache": cache, "worst_kl_max": WORST_KL_MAX}
    while len(attempts) < MAX_ATTEMPTS:
        candidates = [("unit:" + unit, units - {unit}, {edge for edge in edges if unit not in edge})
                      for unit in sorted(units)]
        candidates += [("edge:" + left + ":" + right, units, edges - {(left, right)})
                       for left, right in sorted(edges)]
        accepted = False
        for label, proposed_units, proposed_edges in candidates:
            if len(attempts) == MAX_ATTEMPTS:
                break
            text = graph_text(proposed_units, proposed_edges)
            world_counter = [0]
            def retain(current_nat, ids, world, p, q):
                if evidence_sink is not None:
                    evidence_sink("attempt_" + str(len(attempts)), world_counter[0], world, p, q, None)
                world_counter[0] += 1
            try:
                scored = reward_exec.score_wiring_interventions(toy, text, nat=nat, texts=texts,
                                panel=start["panel"], reference_cache=cache, evidence_sink=retain)
                row = describe(label, scored)
                row["accepted"] = (row["worst_kl_bits"] <= WORST_KL_MAX
                                   and row["description_bits"] < current["description_bits"])
            except reward_exec.InvalidInterventionCandidate as exc:
                row = {"candidate": label, "accepted": False, "invalid": str(exc)}
            row["completion"] = text
            row["completion_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            attempts.append(row)
            if attempt_sink is not None:
                attempt_sink(copy.deepcopy(row))
            if row["accepted"]:
                units, edges, current = set(proposed_units), set(proposed_edges), row
                accepted = True
                break
        if not accepted:
            break
    if original != binding(toy, nat, texts, start["panel"]):
        raise RuntimeError("teacher search did not restore original state")
    target = graph_text(units, edges)
    return {"status": "attempt_cap" if len(attempts) == MAX_ATTEMPTS else "greedy_fixed_point",
            "target": target, "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            "selected": current, "start": describe("full_standalone", start),
            "attempts": attempts, "cache": cache, "worst_kl_max": WORST_KL_MAX,
            "max_attempts": MAX_ATTEMPTS, "optimization": "incomplete fixed-order greedy search; no English target"}


def training_row(toy, result, *, nat, renderer=None):
    """New graph-only task; original teacher/English targets remain untouched."""
    import render
    import circuit
    if toy.meta.get("record", {}).get("split") != "train":
        raise RuntimeError("intervention teacher rows are TRAIN-only")
    if result.get("target") is None:
        return {"id": toy.name, "usable": False, "reason": result["status"], "target": None}
    if result.get("target_sha256") != hashlib.sha256(result["target"].encode()).hexdigest():
        raise RuntimeError("teacher target receipt mismatch")
    if (result["selected"].get("completion") != result["target"]
            or result["selected"].get("completion_sha256") != result["target_sha256"]):
        raise RuntimeError("teacher target differs from selected measured completion")
    cache = result["cache"]
    inputs, panel = cache["binding"]["inputs"], cache["binding"]["panel"]
    validate_cache(cache, toy, nat, inputs, panel)
    rendered_toy = copy.copy(toy)
    rendered_toy.texts = inputs
    renderer = render if renderer is None else renderer
    original = renderer.render(rendered_toy, n_texts=len(inputs), all_live_neurons=True)
    if not original.endswith(renderer.FORMAT):
        raise RuntimeError("unexpected rendering contract")
    observation = original[:-len(renderer.FORMAT)]
    target_units = {unit for unit in circuit.parse(result["target"]).units
                    if unit != "out" and not unit.startswith("tok")}
    displayed = set(re.findall(r"L\d+[HN]\d+", observation))
    missing = sorted(target_units - displayed)
    prompt = observation + ("Return only an executable Mermaid flowchart LR. State head/neuron units and "
                            "their directed executable connections, including connections to out. "
                            "Verification uses these displayed inputs under individual head-output "
                            "and original nonzero MLP-output neuron deletion/half-scaling. Do not include English or token annotations.")
    return {"id": toy.name, "prompt": prompt, "completion": result["target"], "usable": not missing,
            "reason": None if not missing else "target_units_missing_from_observation",
            "missing_target_units": missing, "all_live_neurons": True,
            "verified_input_ids": inputs, "verified_intervention_panel": panel,
            "renderer_sha256": hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest(),
            "observability": "No neuron count cap; token/connection numerical summaries remain compressed",
            "target_scope": "graph-only finite-panel supervision, not English or arbitrary intervention proof"}
