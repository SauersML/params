"""Checks of answer_contrast: the words tokens start (spellings of one answer read as one word, other scripts told
apart) and the sets on a tiny random Qwen3 without a tokenizer (the top token against the next ones).

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_answer_contrast.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import answer_contrast as ac  # noqa: E402

transformers = pytest.importorskip("transformers")


def test_words():
    assert [ac.word(t) for t in ("Paris", " Paris", "**Paris**", "  'Tok", "Au\n", "", " ")] == ["paris", "paris", "paris", "tok", "au", "", ""]
    assert ac.script("paris") == "LATIN" and ac.script("巴黎") == "CJK" and ac.script("62") == "DIGIT"


def test_sets_without_tokenizer():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    m = transformers.Qwen3ForCausalLM(cfg).eval()
    ids = list(range(5, 15))
    with torch.no_grad():
        lp = torch.log_softmax(m(input_ids=torch.tensor([ids])).logits[0, -1], -1)
    s = ac.token_sets(m, None, ids, lp, b=int(lp.argsort()[-3]))
    assert s["answer"] == [int(lp.argmax())] and s["alternatives"][0] == int(lp.argsort()[-3])
    assert not set(s["answer"]) & set(s["alternatives"])
    want = (lp[s["answer"]].logsumexp(-1) - lp[s["alternatives"]].logsumexp(-1)) / torch.log(torch.tensor(2.0))
    assert abs(float(ac.contrast(lp[None], s)[0]) - float(want)) < 1e-5


def test_candidate_chunking_matches_real_tiny_qwen_and_bounds_rows():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import answer_contrast as ac
    torch.manual_seed(17)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,attn_implementation='eager')).eval()
    class Tok:
        def decode(self,ids):return ' '.join('word'+str(i) for i in ids)
    ids=[1,2,3];lp=torch.linspace(0,-5,32).log_softmax(0);b=20 # extra ninth candidate
    batches=[];hook=model.register_forward_pre_hook(lambda m,args,kwargs:batches.append(kwargs['input_ids'].shape[0]),with_kwargs=True)
    try:
        with torch.no_grad():
            reference=ac.token_sets(model,Tok(),ids,lp,b,k=8,steps=3)
            assert batches==[9,9,9]
            batches.clear()
            chunked=ac.token_sets(model,Tok(),ids,lp,b,k=8,steps=3,candidate_batch_size=4)
        assert chunked==reference
        assert batches==[4,4,1]*3 and sum(batches)==27
        assert len(chunked['words'])==9 # no bank truncation
    finally:hook.remove()


@pytest.mark.parametrize('size',[0,-1,True,1.5])
def test_invalid_candidate_batch_size_rejected_without_forward(size):
    import torch
    import answer_contrast as ac
    class Fail:
        def __call__(self,**kw):raise AssertionError('unexpected forward')
    with pytest.raises(ValueError,match='candidate_batch_size'):
        ac.token_sets(Fail(),None,[1],torch.zeros(8),k=4,candidate_batch_size=size)


def test_patcher_setup_bounds_real_candidate_continuation_batches():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from circuit_finder import Patcher
    torch.manual_seed(19)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=128,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,attn_implementation='eager')).eval()
    class Tok:
        def decode(self,ids):return ' '.join('word'+str(i) for i in ids)
    batches=[];hook=model.register_forward_pre_hook(lambda m,args,kwargs:batches.append(kwargs['input_ids'].shape[0]),with_kwargs=True)
    try:
        with torch.no_grad():record=Patcher(model,Tok(),rows=4).setup([1,2,3],[1,2,4],candidate_batch_size=4)
        assert batches[:2]==[4,4] and max(batches)<=4 and record['valid']
    finally:hook.remove()


def test_continuations_from_the_prompts_cache_match():
    """token_sets with past (the prompt's keys and values, one row) reads the same words as with the prompt rerun."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import answer_contrast as ac
    import circuit_finder as cf
    torch.manual_seed(17)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
    class Tok:
        def decode(self,ids):return ' '.join('word'+str(i) for i in ids)
    ids=[1,2,3];lp=torch.linspace(0,-5,32).log_softmax(0);b=20
    with torch.no_grad():
        past=[(k.clone(),v.clone()) for k,v in cf._kv_pairs(model(input_ids=torch.tensor([ids]),use_cache=True).past_key_values)]
        for size in (None,4):
            assert ac.token_sets(model,Tok(),ids,lp,b,k=8,steps=3,candidate_batch_size=size,past=past)== \
                ac.token_sets(model,Tok(),ids,lp,b,k=8,steps=3,candidate_batch_size=size)


def test_hint_reads_the_continued_word():
    """A hint token whose own text is only the start of a word (" Haw" of " Hawthorne") names the answer by the word it
    continues to: the answer set holds it, and the word is not the token's text."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(5)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval()

    class Tok:                                  # token 7 decodes to " Haw"; any continuation of it reads "thorne"
        def decode(self, ids):
            ids = list(ids)
            return "".join(" Haw" if z == 7 and j == 0 else "thorne" if ids[0] == 7 and j == 1 else f" w{z}"
                           for j, z in enumerate(ids))
    ids, lp = [1, 2, 3], torch.linspace(0, -5, 32).log_softmax(0)
    with torch.no_grad():
        s = ac.token_sets(model, Tok(), ids, lp, 20, k=8, steps=3, hint=[7])
    assert s["answer_word"] == "hawthorne" and 7 in s["answer"] and 7 not in s["alternatives"]


def test_token_sets_many_equals_each():
    """token_sets_many: prompts sharing candidates (one prompt twice with different b) and another prompt, with and
    without the prompts' keys and values, give each item's token_sets."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import answer_contrast as ac
    torch.manual_seed(23)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval()

    class Tok:
        def decode(self, ids):
            return ' '.join('word' + str(i % 7) for i in ids)
    items = []
    with torch.no_grad():
        for ids, b in (([1, 2, 3], 20), ([1, 2, 3], 9), ([4, 5, 6], 2)):
            o = model(input_ids=torch.tensor([ids]), use_cache=True)
            past = [(k, v) for k, v in zip(o.past_key_values.key_cache, o.past_key_values.value_cache)] \
                if hasattr(o.past_key_values, "key_cache") else [(layer.keys, layer.values) for layer in o.past_key_values.layers]
            items.append((ids, o.logits[0, -1].log_softmax(-1), b, past, None))
        for size in (None, 3):
            many = ac.token_sets_many(model, Tok(), items, k=8, steps=3, candidate_batch_size=size)
            assert many == [ac.token_sets(model, Tok(), ids, lp, b, k=8, steps=3, past=past) for ids, lp, b, past, _ in items]
            assert ac.token_sets_many(model, Tok(), [x[:3] + (None, None) for x in items], k=8, steps=3) == many
