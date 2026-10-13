"""Offline matched graph-only SFT preparation. Never creates a service/client.

The producer must separately bind prompts/targets to model and teacher receipts.
This module checks pairing, split coverage, token fit and equal supervision mass;
it does not establish circuit fidelity or authorize paid training.
"""
from __future__ import annotations

import hashlib
import json
import re

import capable_wiring as pilot
from intervention_teacher_batch import CASES
from learning_model_bank import FAMILIES

ARMS = ("point", "intervention")
EPOCHS, BATCH, SEED, OUTPUT_CAP, CONTEXT = 8, 8, 0, 416, 32768
NODE = r"(?:L\d+[HN]\d+|out)"
GRAPH_LINE = re.compile(rf"{NODE}(?:\s+-->\s+{NODE})?")


def expected_ids():
    train = list(CASES) + [f"{family}_{3100000 + 100 * i + variant}"
                          for i, family in enumerate(FAMILIES) for variant in (0, 1)]
    evaluation = [f"{family}_{3200000 + 100 * i}" for i, family in enumerate(FAMILIES)]
    return train, evaluation


def graph_only(text):
    """Require explicit internal-unit Mermaid; no dropped English or annotations."""
    if not isinstance(text, str):
        raise ValueError("completion must be a graph string")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if (not lines or lines[0] != "flowchart LR" or
            not any("-->" in line for line in lines[1:]) or
            any(not GRAPH_LINE.fullmatch(line) for line in lines[1:])):
        raise ValueError("complete graph-only target required; no stripping or repair")


def validate_rows(train, panel):
    train_ids, evaluation_ids = expected_ids()
    if [r["id"] for r in train] != train_ids:
        raise ValueError("all 24 predeclared TRAIN rows required in fixed order")
    fresh = [r for r in panel if r["stratum"] == "new_model"]
    retention = [r for r in panel if r["stratum"] == "retention"]
    if ([r["id"] for r in fresh] != evaluation_ids or len(retention) != 24 or
            len(panel) != 32 or len({(r["stratum"], r["id"]) for r in panel}) != 32):
        raise ValueError("eight new models plus all 24 retention cases required")
    for row in [*train, *panel]:
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError("complete nonempty prompt required")
    for row in train:
        for arm in ARMS:
            graph_only(row[arm + "_completion"])
    return train_ids, evaluation_ids


def equal_row_mass(datums):
    """Preserve every supervised token; equalize each paired row's loss mass.

    The two targets may have different lengths. Both receive the arithmetic mean
    of their original total weights, so treatment is not extra total loss weight.
    Billed training tokens can still differ and are counted separately.
    """
    import torch
    import tinker
    from tinker.types.tensor_data import TensorData

    weights = [d.loss_fn_inputs["weights"].to_torch() for d in datums]
    masses = [float(w.double().sum()) for w in weights]
    if len(weights) != 2 or any(not torch.isfinite(w).all() or (w < 0).any() for w in weights) or min(masses) <= 0:
        raise ValueError("two finite positive supervised targets required")
    target = sum(masses) / 2
    output, normalized = [], []
    for datum, weight, mass in zip(datums, weights, masses):
        scaled = weight * (target / mass)
        if not torch.equal(scaled > 0, weight > 0):
            raise ValueError("normalization changed supervised tokens")
        actual = float(scaled.double().sum())
        if abs(actual - target) > 2e-6 * target:
            raise ValueError("paired supervision mass mismatch")
        inputs = dict(datum.loss_fn_inputs)
        inputs["weights"] = TensorData.from_torch(scaled)
        output.append(tinker.Datum(model_input=datum.model_input, loss_fn_inputs=inputs))
        normalized.append(actual)
    return output, {"original_mass": masses, "target_mass": target, "stored_mass": normalized}


def prepare(train, panel, tokenizer_path):
    """Render all rows without truncation; caller stores the resulting contract."""
    validate_rows(train, panel)
    rr, tok = pilot.renderer(tokenizer_path)
    data = {arm: [] for arm in ARMS}
    stats = []
    for row in train:
        raw = [pilot.datum({"prompt": row["prompt"], "completion": row[arm + "_completion"]}, rr)
               for arm in ARMS]
        lengths = [d.model_input.length for d in raw]
        supervised = [int(d.loss_fn_inputs["weights"].to_torch().gt(0).sum()) for d in raw]
        answer_tokens = [len(tok.encode(row[arm + "_completion"], add_special_tokens=False)) + 1 for arm in ARMS]
        if max(lengths) > CONTEXT or max(supervised) > OUTPUT_CAP or max(answer_tokens) > OUTPUT_CAP:
            raise ValueError("complete training pair does not fit; never truncate")
        paired, mass = equal_row_mass(raw)
        for arm, datum in zip(ARMS, paired):
            data[arm].append(datum)
        stats.append({"id": row["id"], "pair_tokens": dict(zip(ARMS, lengths)),
                      "supervised_tokens": dict(zip(ARMS, supervised)),
                      "answer_plus_eos_tokens": dict(zip(ARMS, answer_tokens)), **mass})
    prompts = [pilot.generation_prompt(row, rr) for row in panel]
    if any(p.length + OUTPUT_CAP + 16 > CONTEXT for p in prompts):
        raise ValueError("complete evaluation prompt does not fit")
    batches = pilot.schedule(len(train), EPOCHS, BATCH, SEED)
    train_tokens = {arm: sum(data[arm][i].model_input.length for batch in batches for i in batch)
                    for arm in ARMS}
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    contract = {"model": pilot.MODEL, "project": pilot.PROJECT, "train_sha256": digest(train),
                "panel_sha256": digest(panel), "epochs": EPOCHS, "batch": BATCH, "seed": SEED,
                "updates_per_arm": len(batches), "output_cap": OUTPUT_CAP,
                "train_tokens_per_arm": train_tokens, "training": stats,
                "panel": [{"id": r["id"], "stratum": r["stratum"], "prompt_tokens": p.length,
                           "rendered_sha256": digest(p.to_ints())} for r, p in zip(panel, prompts)],
                "scope": "paired token-fit preflight only; physical proof and paid reservation remain separate"}
    return {"contract": contract, "data": data, "prompts": prompts, "batches": batches,
            "renderer": rr, "tokenizer": tok}
