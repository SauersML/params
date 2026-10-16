"""circuit_robust on a tiny random Qwen3 (circuit_finder's test pair): with the corrupted prompt as the resample, the
resampled shares are the standard ones; marginals, backups and interactions run and report consistent numbers."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_finder as cf  # noqa: E402
import circuit_robust as cr  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    model = transformers.Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    pt = cf.Patcher(model, None, rows=8)
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        clean = torch.randint(0, 64, (10,), generator=g).tolist()
        corrupt = list(clean)
        corrupt[4], corrupt[7] = (clean[4] + 7) % 64, (clean[7] + 11) % 64
        info = pt.setup(clean, corrupt)
        if info["answer"] != info["corrupt_answer"] and info["gap"] > 0.5:
            rec = cf.find(pt, clean, corrupt)
            rec.update(ids=clean, corrupted=corrupt, id="tiny")
            return pt, rec
    pytest.skip("no pair with different answers")


def test_resampling_identities(setup):
    pt, rec = setup
    for cut in ("readout", "source", "movers"):
        units, _, _ = cr.cut_units(rec, cut)
        if not units:
            continue
        r = cr.resampled(pt, rec, cut, [("corrupt", rec["corrupted"])])  # (r = the corrupted prompt itself)
        assert abs(r["lost"][0] - r["standard"][0]) < 1e-4 and abs(r["restored"][0] - r["standard"][1]) < 1e-4


def test_backups_and_interactions_run(setup):
    pt, rec = setup
    pt.setup(rec["ids"], rec["corrupted"])
    for cut in ("readout", "source", "movers"):
        units, _, _ = cr.cut_units(rec, cut)
        rows = cr.backups(pt, rec, cut)
        assert all(abs(r["compensation"] - (r["direct"] - r["total"])) < 1e-9 for r in rows)
        assert len(rows) <= len(units)
        m = cr.marginals(pt, rec, cut)
        if m and len(units) == 1:
            assert abs(m["units"][0]["alone"][0] - m["joint"][0]) < 1e-4   # (one unit: alone is the joint)
        for r in cr.interactions(pt, rec, cut):
            a, b = r["singles"]
            assert abs(r["interaction"][0] - (r["pair_shares"][0] - a[0] - b[0])) < 1e-9
