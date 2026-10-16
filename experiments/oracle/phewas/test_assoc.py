"""Stage 1's checks (assoc.py) on synthetic genotypes, and the whole stage on a tiny random Qwen3: the mixed model equals
least squares without relatedness and deflates a structured trait, LD scores of independent columns are about 1, the LD
score regression recovers a planted intercept, clumping merges copies of one column."""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assoc  # noqa: E402


def least_squares(Z, ok, C, y):
    Q, _ = torch.linalg.qr(C)
    Zr = assoc.residualize(Q, Z)
    return assoc.ols(Zr, Zr.norm(dim=0), assoc.residualize(Q, y), ok, Z.shape[0] - C.shape[1] - 1)[0]


def test_mixed_model_without_relatedness_is_least_squares():
    g = torch.Generator().manual_seed(0)
    n, width = 150, 600
    Z = torch.randn(n, 3 * width, generator=g)
    Z = (Z - Z.mean(0)) / Z.std(0)
    ok = torch.ones(Z.shape[1], dtype=torch.bool)
    C = torch.ones(n, 1)
    y = torch.randn(n, generator=g)
    t_lmm, h2 = assoc.lmm(Z, ok, C, y[None], width)
    t = least_squares(Z, ok, C, y)
    assert float(np.corrcoef(t_lmm[0], t)[0, 1]) > 0.98
    assert float(h2.mean()) < 0.3


def test_mixed_model_deflates_structure():
    g = torch.Generator().manual_seed(1)
    n, width = 200, 500
    pop = (torch.arange(n) < n // 2).float()
    Z = torch.randn(n, 3 * width, generator=g) + 0.6 * pop[:, None] * torch.randn(1, 3 * width, generator=g)
    Z = (Z - Z.mean(0)) / Z.std(0)
    ok = torch.ones(Z.shape[1], dtype=torch.bool)
    C = torch.ones(n, 1)
    y = 1.5 * pop + torch.randn(n, generator=g)
    t = least_squares(Z, ok, C, y)
    t_lmm = assoc.lmm(Z, ok, C, y[None], width)[0][0]
    assert float(t_lmm.pow(2).median()) < 0.6 * float(t.pow(2).median())


def test_ld_scores_of_independent_columns():
    g = torch.Generator().manual_seed(2)
    n, F = 200, 4000
    Z = torch.randn(n, F, generator=g)
    Zr = Z - Z.mean(0)
    ld = assoc.ld_scores(Zr, Zr.norm(dim=0), torch.ones(F, dtype=torch.bool), n - 1)
    assert abs(float(ld.mean()) - 1) < 0.2
    Z[:, 1:20] = Z[:, :1] + 0.01 * torch.randn(n, 19, generator=g)  # 20 copies: each has an LD score near 20
    Zr = Z - Z.mean(0)
    ld = assoc.ld_scores(Zr, Zr.norm(dim=0), torch.ones(F, dtype=torch.bool), n - 1)
    assert 17 < float(ld[:20].mean()) < 23


def test_ld_score_regression_recovers_intercept():
    rng = np.random.default_rng(3)
    l = rng.uniform(1, 60, 200_000)
    chi2 = (1.1 + 0.02 * l) * rng.chisquare(1, l.size)
    out = assoc.ldsc(chi2, l)
    assert abs(out["intercept"] - 1.1) < 3 * out["intercept_se"] + 0.01
    assert abs(out["slope"] - 0.02) < 3 * out["slope_se"] + 0.001


def test_clump_merges_copies():
    g = torch.Generator().manual_seed(4)
    n = 100
    base = torch.randn(n, 1, generator=g)
    Z = torch.cat([base + 0.05 * torch.randn(n, 5, generator=g), torch.randn(n, 3, generator=g)], 1)
    Zr = Z - Z.mean(0)
    hits = torch.tensor([5, 0, 1, 6, 2, 3, 7, 4])
    leads, sizes = assoc.clump(Zr, Zr.norm(dim=0), hits)
    assert [int(hits[j]) for j in leads] == [5, 0, 6, 7]
    assert sizes == [1, 5, 1, 1]


def test_stage1_on_a_tiny_model(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(tmp_path / "model")
    d = tmp_path / "phen"
    d.mkdir()
    rng = random.Random(0)
    (d / "phenome.json").write_text(json.dumps({"traits": [{"name": "p/a", "kept": True}]}))
    seqs, vals = [], []
    for k in range(60):
        seqs.append({"id": f"p/a|{k}|A|pos", "prompt": [rng.randrange(64) for _ in range(rng.randrange(6, 14))]})
        vals.append({"trait": "p/a", "item": str(k), "value": rng.gauss(0, 1), "split": "dev" if k % 2 else "held",
                     "cov": {"matching": "Yes" if k % 3 else "No"}})
    (d / "sequences.jsonl").write_text("".join(json.dumps(s) + "\n" for s in seqs))
    (d / "values.jsonl").write_text("".join(json.dumps(v) + "\n" for v in vals))
    subprocess.run([sys.executable, str(Path(assoc.__file__)), str(d), str(tmp_path / "out"), "--model", str(tmp_path / "model"),
                    "--device", "cpu"], check=True)
    s = json.loads((tmp_path / "out" / "summary.json").read_text())["p/a"]
    z = np.load(tmp_path / "out" / "p__a.npz")
    assert z["t"].shape == z["t_lmm"].shape == z["ld"].shape == (2, 48, 4)
    assert s["n"] == 60 and s["hits"] == 0 and s["clumps"] == 0 and 0 <= s["lmm_h2"] <= 1
    assert np.isfinite(s["ldsc"]["intercept"])
