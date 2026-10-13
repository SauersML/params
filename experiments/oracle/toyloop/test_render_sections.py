"""Tests of render's TOYLOOP_SECTIONS=weights mode (nothing computed by running the model reaches the prompt).

  python3 -B -m pytest -q experiments/oracle/toyloop/test_render_sections.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402

torch.set_num_threads(1)
MANIFEST = Path.home() / "mpd-data/toy_zoo/manifest.jsonl"
RUN_WORDS = ("Behaviour", "attends", "active at last", "predicts", "offset +", "of positions")


@pytest.fixture
def weights_mode(monkeypatch):
    monkeypatch.setattr(render, "SECTIONS", "weights")


def zoo(n=None):
    if not MANIFEST.exists():
        pytest.skip("no zoo")
    toys = render.load_zoo(MANIFEST)
    return toys if n is None else toys[::max(1, len(toys) // n)]


def test_weights_mode_has_no_run_information(weights_mode, monkeypatch):
    calls = []
    monkeypatch.setattr(render, "run", lambda *a, **k: calls.append(1))   # running the model is forbidden
    for toy in [render.synthetic(0), *zoo(10)]:
        p = render.render(toy)
        assert not calls
        assert not any(w in p for w in RUN_WORDS), [w for w in RUN_WORDS if w in p]
        for ids in toy.texts[:2]:   # the texts' tokens and the target position stay
            assert " ".join(toy.vocab[a] for a in ids) in p and f"predict at position {len(ids) - 1}" in p
        assert "flowchart LR" in p and "English:" in p


def test_default_mode_unchanged():
    p = render.render(render.synthetic(0))
    assert "Behaviour:" in p and "attends" in p


def test_weights_mode_follows_the_weights(weights_mode):
    a, b = [t for t in zoo() if t.meta["family"] == "copy_offset"][:2]
    pa, pab = render.render(a), render.render(a, weights_from=b)
    assert pa != pab and render.render(b).split("Texts:")[0] == pab.split("Texts:")[0]
    assert pab.split("Texts:")[1] == pa.split("Texts:")[1]


def test_units_and_distractors_visible(weights_mode):
    """Every gt head or neuron is listed and writes somewhere; every distractor 'writes to nothing'; cancel pairs marked."""
    for toy in zoo(20):
        lines = {l.split()[0]: l for l in render.render(toy).splitlines() if l[:1] == "L" and " |" in l}
        for u in toy.truth["units"]:
            if u[:1] == "L":
                assert u in lines and "writes to nothing" not in lines[u], (toy.name, u)
        for u in toy.meta["record"]["distractor_units"]:
            u = render.zoo_unit(u)
            if u in lines:
                assert "writes to nothing" in lines[u], (toy.name, u)
        if toy.meta["family"] == "cancel":
            for u in toy.meta["record"]["params"]["pair"]:
                assert "cancels" in lines[render.zoo_unit(u)]


def test_positional_offset_read_from_weights(weights_mode):
    for toy in [t for t in zoo() if t.meta["family"] == "copy_offset"][:5]:
        p = toy.meta["record"]["params"]
        u = f"L{p['head'][0]}H{p['head'][1]}"
        line = next(l for l in render.render(toy).splitlines() if l.startswith(u + " "))
        assert f"pos: offset -{p['offset']} " in line, line


def test_prompt_length(weights_mode):
    for toy in zoo(10):
        assert len(render.render(toy)) < 9000
