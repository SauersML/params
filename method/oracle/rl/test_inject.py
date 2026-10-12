"""inject.py on a tiny random Qwen3 and small vectors: the slots, the embeddings at them, a training step, sampling."""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inject  # noqa: E402

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
    vectors = inject.Vectors(d / "uv.safetensors")
    return inject.Explainer(str(d / "model"), 4, torch.device("cpu"), vectors), d


def test_slot_embeddings_are_the_vectors_and_train(tiny):
    ex, _ = tiny
    q, nodes = inject.slotted(QUESTION, EV)
    ids = torch.tensor([ex.prompt_ids(q)])
    emb = ex.embed(ids, [nodes])
    pos = (ids[0] == ex.slot).nonzero().flatten()
    want = ex.injector([(nd[0].split(".")[-1], *ex.vectors(nd)) for nd in nodes])
    assert torch.allclose(emb[0, pos], want) and len(pos) == 2 * len(nodes)
    other = inject.derange(nodes, random.Random(0))
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
