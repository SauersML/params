"""The causal oracle's encoding, injection, loss and example generation on a tiny random Qwen3 with Qwen3's tokenizer
(CPU; the tokenizer from the local Hugging Face cache)."""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cao  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=24, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    base = Qwen3ForCausalLM(cfg)
    o = cao.Oracle.__new__(cao.Oracle)
    o.torch, o.dev, o.tok, o.text_only = torch, torch.device("cpu"), tok, False
    o.m = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    o.lm = o.m.base_model.model
    o.layers = o.lm.model.layers
    o.q = tok(" ?", add_special_tokens=False)["input_ids"][0]
    return o


def ex(o, n=5):
    ctx = o.tok(" The quick brown fox jumps over the lazy dog because it was late", add_special_tokens=False)["input_ids"]
    return {"ctx": ctx, "blocks": [{"layer": 7, "positions": list(range(n))}, {"label": "Unit A", "unit": ["neuron", 9, 3]}],
            "q": "Which tokens does unit A's output raise most?", "a": "'cat'"}


def test_encode_finds_every_placeholder_and_the_answer(tiny):
    e = ex(tiny)
    ids, pos, a0 = tiny.encode(e, True)
    assert len(pos) == 6 and all(ids[p] == tiny.q for p in pos)
    assert tiny.tok.decode(ids[a0:]).startswith("'cat'")


def test_the_hook_adds_the_normalized_vector_scaled_to_the_norm(tiny):
    e = ex(tiny)
    ids, pos, _ = tiny.encode(e, False)
    v = tiny.vectors(e)
    grab = {}
    h0 = tiny.layers[1].register_forward_hook(lambda _m, _i, out: grab.__setitem__("plain", (out[0] if isinstance(out, tuple) else out).clone()))
    with torch.no_grad():
        tiny.m(input_ids=torch.tensor([ids]))
    h0.remove()
    h = tiny.hook([v], [pos])
    h1 = tiny.layers[1].register_forward_hook(lambda _m, _i, out: grab.__setitem__("added", (out[0] if isinstance(out, tuple) else out).clone()))
    with torch.no_grad():
        tiny.m(input_ids=torch.tensor([ids]))
    h.remove()
    h1.remove()
    plain, added = grab["plain"][0], grab["added"][0]
    want = plain[pos] + torch.nn.functional.normalize(v, dim=-1) * plain[pos].norm(dim=-1, keepdim=True)
    assert torch.allclose(added[pos], want, atol=1e-4)
    rest = [i for i in range(len(ids)) if i not in pos]
    assert torch.equal(added[rest], plain[rest])


def test_the_loss_trains_the_adapter_alone(tiny):
    e = ex(tiny)
    loss = tiny.batch_loss([e, {**e, "a": "'dog'"}])
    loss.backward()
    grads = [(n, p.grad) for n, p in tiny.m.named_parameters() if p.requires_grad]
    assert grads and all("lora" in n for n, _ in grads) and any(g is not None and g.abs().sum() > 0 for _, g in grads)
    tiny.m.zero_grad(set_to_none=True)


def test_text_only_adds_nothing(tiny):
    e = ex(tiny)
    ids, pos, _ = tiny.encode(e, False)
    tiny.text_only = True
    try:
        h = tiny.hook([tiny.vectors(e)], [pos])
        with torch.no_grad():
            got = tiny.m(input_ids=torch.tensor([ids])).logits
        h.remove()
    finally:
        tiny.text_only = False
    with torch.no_grad():
        plain = tiny.m(input_ids=torch.tensor([ids])).logits
    assert torch.equal(got, plain)


def test_one_text_gives_every_category(tiny, monkeypatch):
    monkeypatch.setattr(cao.h2h, "T", 48)
    pr = cao.Probe(tiny)
    rng = random.Random(0)
    ids = [rng.randrange(1000, 5000) for _ in range(48)]
    with tiny.off():
        exs = cao.gen_text(tiny, pr, ids, rng, heads=True)
    assert {e["cat"] for e in exs} == {"lens", "which", "pair", "swap", "edit", "promotes", "units"}
    for e in exs:
        tiny.encode(e, True)  # every example encodes, placeholders and all
