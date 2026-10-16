"""controls.py on a tiny random Qwen3: each persona item is matched to the neutral item with its own matching answer and
the nearest value; the knockout sets run on the matched neutral items of the persona panel; compose subtracts them."""
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controls  # noqa: E402

transformers = pytest.importorskip("transformers")


def phen(d: Path, name: str, n: int, rng, value):
    d.mkdir(parents=True, exist_ok=True)
    (d / "phenome.json").write_text(json.dumps({"traits": [{"name": name, "kind": "pair", "kept": True}]}))
    seqs, splits, vals = [], [], []
    for k in range(n):
        p = [rng.randrange(64) for _ in range(rng.randrange(6, 12))]
        for form in "AB":
            seqs += [{"id": f"{name}|{k}|{form}|pos", "prompt": p, "cont": [3]}, {"id": f"{name}|{k}|{form}|neg", "prompt": p, "cont": [5]}]
        cov = {"matching_yes": k % 2}
        splits.append({"trait": name, "item": str(k), "split": "dev", "cov": cov})
        vals.append({"trait": name, "item": str(k), "split": "dev", "value": value(k), "cov": cov})
    for f, rows in (("sequences.jsonl", seqs), ("splits.jsonl", splits), ("values.jsonl", vals)):
        (d / f).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_match_runsets_compose(tmp_path, monkeypatch):
    rng = random.Random(0)
    phen(tmp_path / "p", "persona/x", 12, rng, lambda k: float(k))
    phen(tmp_path / "c", controls.NAME, 20, rng, lambda k: float(k) + 0.4)
    controls.match(Namespace(phenome=tmp_path / "p", controls=tmp_path / "c", out=tmp_path / "m.json"))
    m = json.loads((tmp_path / "m.json").read_text())["persona/x"]
    assert all(int(j) % 2 == int(i) % 2 for i, j in m.items()) and m["4"] == "4"
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    monkeypatch.setattr(controls.ph, "load_model", lambda *a, **k: model)
    monkeypatch.setattr(controls, "AutoTokenizer", None, raising=False)

    class Tok:
        @staticmethod
        def from_pretrained(_):
            return None

    monkeypatch.setitem(sys.modules, "transformers", type("T", (), {"AutoTokenizer": Tok}))

    def load(self, _):
        self.mx = torch.ones(self.L, self.D)
        self.massive = torch.full((self.L,), 1e9)

    monkeypatch.setattr(controls.nb.Neurons, "load", load)
    (tmp_path / "design.jsonl").write_text(json.dumps({"id": "L0/0", "units": [[1, 7, 1], [0, 3, -1]]}) + "\n")
    controls.runsets(Namespace(phenome=tmp_path / "p", controls=tmp_path / "c", match=tmp_path / "m.json", design=tmp_path / "design.jsonl",
                               out=tmp_path / "ctl.jsonl", thr=None, every=1, model="x", device="cpu"))
    ctl = json.loads((tmp_path / "ctl.jsonl").read_text())
    assert list(ctl["change"]) == ["ctl/persona/x"] and ctl["change"]["ctl/persona/x"] != 0
    (tmp_path / "raw.jsonl").write_text(json.dumps({"id": "L0/0", "units": ctl["units"], "change": {"persona/x": 1.0, "other": 2.0}}) + "\n")
    controls.compose(Namespace(changes=[tmp_path / "raw.jsonl"], controls=[tmp_path / "ctl.jsonl"], out=tmp_path / "out.jsonl"))
    out = json.loads((tmp_path / "out.jsonl").read_text())["change"]
    assert out["other"] == 2.0 and abs(out["persona/x"] - (1.0 - ctl["change"]["ctl/persona/x"])) < 1e-12
