"""The screen's panel (screen.py) on a tiny random Qwen3: the unedited panel runs exactly twice, a knockout of every
pole changes the traits, an empty knockout changes nothing, and a trait value is its prompts' form-A mean."""
import json
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    nr = screen.nb.Neurons(transformers.Qwen3ForCausalLM(cfg).eval(), None)
    nr.mx = torch.ones(nr.L, nr.D)
    nr.massive = torch.full((nr.L,), 1e9)
    d = tmp_path_factory.mktemp("phen")
    rng = random.Random(0)
    traits = [{"name": "p/a", "kind": "pair", "kept": True}, {"name": "t/b", "kind": "teacher", "kept": True}]
    (d / "phenome.json").write_text(json.dumps({"traits": traits}))
    seqs, splits = [], []
    for k in range(5):
        p = [rng.randrange(64) for _ in range(rng.randrange(6, 12))]
        for form in "AB":
            seqs += [{"id": f"p/a|{k}|{form}|pos", "prompt": p, "cont": [3, 4]}, {"id": f"p/a|{k}|{form}|neg", "prompt": p, "cont": [5]}]
        seqs.append({"id": f"t/b|{k}|AB|ref", "prompt": p, "cont": [7, 8, 9]})
        splits += [{"trait": "p/a", "item": str(k), "split": "dev"}, {"trait": "t/b", "item": str(k), "split": "dev"}]
    (d / "sequences.jsonl").write_text("".join(json.dumps(s) + "\n" for s in seqs))
    (d / "splits.jsonl").write_text("".join(json.dumps(s) + "\n" for s in splits))
    return nr, screen.Panel(d, nr, per=4), seqs


def test_the_panel_is_exact_and_a_knockout_moves_it(setup):
    nr, panel, _ = setup
    a, b = panel.run(), panel.run()
    assert a == b and set(a) == {"p/a", "t/b"}
    e, r = screen.cc.knockout_edit(nr, [])
    assert panel.run(e, r) == a
    allp = [("N", l, i, p, None) for l in range(nr.L) for i in range(nr.D) for p in (1, -1)]
    e, r = screen.cc.knockout_edit(nr, allp)
    got = panel.run(e, r)
    assert all(abs(got[n] - a[n]) > 1e-4 for n in a)


def test_a_pair_trait_is_the_form_a_contrast_mean(setup):
    nr, panel, seqs = setup
    chosen = {s["id"].split("|")[1] for s in panel.seqs if s["id"].startswith("p/a")}
    vals = []
    with torch.no_grad():
        for k in chosen:
            p = next(s["prompt"] for s in seqs if s["id"] == f"p/a|{k}|A|pos")
            def lp(cont):
                x = torch.tensor([p + cont] * screen.ROWS)
                out = torch.log_softmax(nr.core(input_ids=x).logits[0].float(), -1)
                return sum(float(out[len(p) - 1 + t, cont[t]]) for t in range(len(cont)))
            vals.append(lp([3, 4]) - lp([5]))
    assert abs(panel.run()["p/a"] - sum(vals) / len(vals)) < 1e-4
