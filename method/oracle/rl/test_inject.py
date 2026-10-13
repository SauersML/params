"""inject.py on a tiny random Qwen3 and small vectors: the slots, the embeddings at them, a training step, sampling."""

import ast
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inject

QUESTION = ("1 ('a'): L0.fc#3@1 (0.90; routed lens at this position raises 'x'), L1.k#2@1 (1.00)\n"
            "  wiring: L0.fc#3@1 -> L1.k#2@1 (+0.20)\n2 ('b'): L0.fc#3@2 (0.50)")
EV = {"components": {"2": [[["h.0.mlp.c_fc", 2, 3], 0.5, ""]],
                     "1": [[["h.0.mlp.c_fc", 1, 3], 0.9, "x"], [["h.1.attn.k_proj", 1, 2], 1.0, ""]]}}


def test_slots_follow_each_listed_subcomponent_once():
    q, nodes = inject.slotted(QUESTION, EV)
    assert nodes == [("h.0.mlp.c_fc", 1, 3), ("h.1.attn.k_proj", 1, 2), ("h.0.mlp.c_fc", 2, 3)]  # in the order of their slots
    assert q.count(inject.SLOT) == 6
    assert f"L0.fc#3@1 {inject.SLOT}{inject.SLOT} (0.90" in q and "  wiring: L0.fc#3@1 -> L1.k#2@1 (+0.20)" in q  # the wiring untouched


def test_derangement_moves_every_vector():
    nodes = [("s", 0, i) for i in range(5)]
    for seed in range(20):
        assert all(a != b for a, b in zip(inject.derange(nodes, random.Random(seed)), nodes))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from safetensors.torch import save_file

    d = tmp_path_factory.mktemp("tiny")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except OSError:
        pytest.skip("Qwen3's tokenizer is not in the Hugging Face cache")
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(d / "model")
    tok.save_pretrained(d / "model")
    g = torch.Generator().manual_seed(0)
    save_file({"h.0.mlp.c_fc.V": torch.randn(16, 8, generator=g), "h.0.mlp.c_fc.U": torch.randn(8, 24, generator=g),
               "h.1.attn.k_proj.V": torch.randn(16, 4, generator=g), "h.1.attn.k_proj.U": torch.randn(4, 12, generator=g)}, str(d / "uv.safetensors"))
    vectors = inject.Vectors(d / "uv.safetensors", oriented=False)
    return inject.Explainer(str(d / "model"), 4, torch.device("cpu"), vectors), d


def test_slot_embeddings_are_the_vectors_and_train(tiny):
    ex, _ = tiny
    q, nodes = inject.slotted(QUESTION, EV)
    ids = torch.tensor([ex.prompt_ids(q)])
    emb = ex.embed(ids, [nodes])
    pos = (ids[0] == ex.slot).nonzero().flatten()
    want = ex.injector([(nd[0].split(".")[-1], *ex.vectors(nd)) for nd in nodes])
    v, u, scale = ex.vectors(nodes[0])
    assert abs(float(v.norm()) - 1) < 1e-5 and abs(float(u.norm()) - 1) < 1e-5 and scale > 0  # unit directions, the scale apart
    assert torch.allclose(emb[0, pos], want) and len(pos) == 2 * len(nodes)
    other = [("h.0.mlp.c_fc", 1, 4), ("h.1.attn.k_proj", 1, 3), ("h.0.mlp.c_fc", 2, 4)]
    assert not torch.allclose(ex.embed(ids, [other])[0, pos], want)  # shuffled vectors change what the slots carry
    opt = torch.optim.AdamW(ex.param_groups(1e-2))
    before = [p.detach().clone() for p in ex.injector.parameters()]
    loss, n = ex.nll(ex.prompt_ids(q), ex.tok.encode("EXPLANATION: x", add_special_tokens=False) + [ex.end], nodes)
    (loss / n).backward()
    opt.step()
    assert n == len(ex.tok.encode("EXPLANATION: x", add_special_tokens=False)) + 1
    assert any(not torch.equal(a, b) for a, b in zip(before, ex.injector.parameters()))  # the maps train with the LoRA


def test_sample_returns_tokens(tiny):
    ex, d = tiny
    q, nodes = inject.slotted(QUESTION, EV)
    got = ex.sample([ex.prompt_ids(q)], [nodes], 2, 5, batch=2)
    assert len(got) == 1 and len(got[0]) == 2 and all(len(t) <= 5 for t, _ in got[0])
    ex.save(d / "run")
    again = inject.Explainer(str(d / "model"), 4, torch.device("cpu"), ex.vectors, init=d / "run")
    assert torch.allclose(again.embed(torch.tensor([ex.prompt_ids(q)]), [nodes]), ex.embed(torch.tensor([ex.prompt_ids(q)]), [nodes]))


@pytest.fixture
def learn_slots():
    # The production Slots selection methods need no model, checkpoint, or tokenizer.
    source = Path(__file__).resolve().parents[3] / "experiments" / "oracle" / "learn_sft.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Slots")
    namespace = {"Path": Path, "random": random}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    slots = namespace["Slots"].__new__(namespace["Slots"])
    slots.inject, slots.permute, slots.count = inject, True, {"h.0.mlp.c_fc": 10, "h.1.attn.k_proj": 7}
    return slots


def test_learn_permute_changes_weight_identity_and_preserves_site_and_position(learn_slots):
    row = {"prompt": "a b", "meta": {"slots": [["a", ["h.0.mlp.c_fc", 1, 3]], ["b", ["h.1.attn.k_proj", 2, 4]]]}}
    _, nodes = learn_slots.prompt(row)
    assert [(n, q) for n, q, _ in nodes] == [("h.0.mlp.c_fc", 1), ("h.1.attn.k_proj", 2)]
    assert nodes[0][2] != 3 and nodes[1][2] != 4


def test_learn_permute_is_one_fixed_wrong_vector_per_subcomponent(learn_slots):
    # the same subcomponent gets the same donor at every position and in every question
    rows = [{"prompt": f"{k} a b", "meta": {"slots": [["a", ["h.0.mlp.c_fc", 1 + k, 3]], ["b", ["h.0.mlp.c_fc", 9, 3]]]}}
            for k in range(5)]
    donors = {learn_slots.prompt(r)[1][j][2] for r in rows for j in range(2)}
    assert len(donors) == 1 and donors != {3}


@pytest.mark.parametrize("site,count", [("h.0.mlp.c_fc", 10), ("h.1.attn.k_proj", 7)])
def test_learn_permute_is_a_derangement_of_each_matrix(learn_slots, site, count):
    got = [learn_slots.prompt({"prompt": "a", "meta": {"slots": [["a", [site, 0, c]]]}})[1][0][2] for c in range(count)]
    assert sorted(got) == list(range(count)) and all(g != c for c, g in enumerate(got))


@pytest.mark.parametrize("nodes", [[("s", 0, 1)], [("s", 0, 1), ("s", 1, 1)],
                                    [("h.0.attn.q_proj", 0, 1), ("h.0.attn.k_proj", 0, 2)]])
def test_derangement_rejects_impossible_weight_control(nodes):
    with pytest.raises(ValueError, match="cannot derange distinct weight components"):
        inject.derange(nodes, random.Random(0))


def test_derangement_preserves_kinds_positions_and_unique_weight_permutation():
    nodes = [("h.0.mlp.c_fc", 0, 1), ("h.0.mlp.c_fc", 1, 1),
             ("h.0.mlp.c_fc", 2, 2), ("h.1.mlp.c_fc", 3, 3),
             ("h.0.attn.k_proj", 4, 1), ("h.1.attn.k_proj", 5, 2)]
    for seed in range(20):
        shuffled = inject.derange(nodes, random.Random(seed))
        assert {(n, c) for n, _, c in shuffled} == {(n, c) for n, _, c in nodes}
        assert [p for _, p, _ in shuffled] == [p for _, p, _ in nodes]
        assert all(a[0].split(".")[-1] == b[0].split(".")[-1] and (a[0], a[2]) != (b[0], b[2])
                   for a, b in zip(nodes, shuffled))
        assert shuffled == inject.derange(nodes, random.Random(seed))
        assert (shuffled[0][0], shuffled[0][2]) == (shuffled[1][0], shuffled[1][2])


def test_derangement_of_no_slots_is_empty():
    assert inject.derange([], random.Random(0)) == []


def test_derangement_unequal_occurrences_share_one_donor():
    nodes = [("s", 0, 1), ("s", 1, 1), ("s", 2, 2)]
    assert inject.derange(nodes, random.Random(0)) == [("s", 0, 2), ("s", 1, 2), ("s", 2, 1)]


def test_sampling_without_shuffle_does_not_build_a_control(monkeypatch, tmp_path):
    from types import SimpleNamespace

    row = {"task": "toy", "question": "q"}
    rows = tmp_path / "rows.jsonl"
    import json
    rows.write_text(json.dumps(row) + "\n")
    class FakeExplainer:
        tok = SimpleNamespace(decode=lambda tokens: "answer")
        def __init__(self, *args):
            pass
        def prompt_ids(self, question):
            return [1]
        def sample(self, prompts, nodes, n, max_tokens, batch):
            return [[([2], False)]]
    monkeypatch.setattr(inject, "Explainer", FakeExplainer)
    monkeypatch.setattr(inject, "Vectors", lambda: None)
    monkeypatch.setattr(inject, "device", lambda: "cpu")
    monkeypatch.setattr(inject, "slotted", lambda question, row: (question, [("s", 0, 1)]))
    def reject_unused(*args):
        raise AssertionError("ordinary sampling must not request a weight control")
    monkeypatch.setattr(inject, "derange", reject_unused)
    args = SimpleNamespace(rows=rows, n=1, base="mock", init=None, seed=0, shuffle=False, shuffled_out=None,
                           out=tmp_path / "out.jsonl", vllm=False, samples=1, max_tokens=1, batch=1)
    inject.sample(args)
    assert json.loads(args.out.read_text())["text"] == "answer"
