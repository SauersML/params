"""Subcomponent answer syntax and scoring against the real VPD model."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carry
import native
import units


def test_subcomponent_roundtrip():
    text = "EXPLANATION: copy digits\nPOSITION 322: L0 322:fc#12,fc#901,down#77 | L2 322:v#1514 -> 332:o#77\nPOSITION 4: none"
    explanation, circuits = units.parse(text)
    assert circuits[322] == [(native.site_name(0, "c_fc"), 322, 12),
                             (native.site_name(0, "c_fc"), 322, 901),
                             (native.site_name(0, "down_proj"), 322, 77),
                             (native.site_name(2, "v_proj"), 322, 1514),
                             (native.site_name(2, "o_proj"), 332, 77)]
    assert units.parse(units.render(explanation, circuits)) == (explanation, circuits)
    assert "#1514" in units.render(explanation, circuits)
    assert all(isinstance(u, carry.Subcomponent) for u in circuits[322])


def test_legacy_roundtrip():
    text = "EXPLANATION: copy\nPOSITION 322: L0 322:fc,down | L2 322:v[h3,h4] -> 332:o[h3,h4]"
    parsed = units.parse(text)
    assert units.parse(units.render(*parsed)) == parsed
    assert all(not isinstance(u, carry.Subcomponent) for u in parsed[1][322])


@pytest.mark.parametrize("ranks", [(0, 1, 2), (3, 4, 5)])
def test_subcomponent_score_equals_direct(ranks):
    nat = native.Native()
    task = json.loads((native.TEXTS / "vpd4l/hard9308.json").read_text())
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    p = units.sensitive(nat, ids, targets, 1)[0]
    n, t = native.site_name(0, "c_fc"), max(targets)
    text = f"EXPLANATION: test\nPOSITION {p}: L0 {p}:fc#12,fc#901,fc#12 | L2 {t}:o#77"
    answer = units.score_many(nat, task, [text], k=1, ranks=ranks)[0]
    cs = units.parse(text)[1][p]
    where = sorted({p, *targets})
    full = {(name, q): range(nat.C[name]) for name in nat.names for q in where}
    space = carry.UnitSpace(nat, len(ids), where, candidates=full)
    active = torch.tensor([u in set(cs) for u in space.units], device=nat.dev)
    study = carry.Carry(nat, ids, targets, carry.alternatives(nat, ids, p, ranks=ranks), space=space)
    empty, actual, rest = study.measure([~torch.ones_like(active), active, ~active]).tolist()
    expected = max(0.0, min(1.0, 1-actual/empty)) if empty > 0 else 0.0
    assert answer["positions"][0]["movement"] == pytest.approx(empty, abs=2e-5)
    assert answer["positions"][0]["carried"] == pytest.approx(expected, abs=2e-5)
    assert answer["rest_carried"] == pytest.approx(max(0.0, min(1.0, 1-rest/empty)), abs=2e-5)
    assert answer["units"] == 3
    assert answer["unknown"] == 0
    assert answer["heads"] == 0


def test_mixed_batch_preserves_legacy_sizes():
    nat = native.Native()
    task = json.loads((native.TEXTS / "vpd4l/hard9308.json").read_text())
    prompt = task["prompts"][0]
    p = units.sensitive(nat, prompt["token_ids"], prompt["target_positions"], 1)[0]
    texts = [f"POSITION {p}: L0 {p}:fc", f"POSITION {p}: L0 {p}:fc#12"]
    batch = units.score_many(nat, task, texts, k=1)
    separate = [units.score(nat, task, text, k=1) for text in texts]
    assert [r["units"] for r in batch] == [1, 1]
    for a, b in zip(batch, separate):
        assert a["uncarried"] == pytest.approx(b["uncarried"], abs=2e-5)
