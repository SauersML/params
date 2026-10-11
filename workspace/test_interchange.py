"""The fast interchange path against full forwards with a hook, on Qwen3-0.6B in float32 on the CPU.
MPD_MEM_GIB=8 ~/mpd-data/venv/bin/python -B workspace/test_interchange.py"""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interchange import BatchHookPatcher, HookPatcher, Qwen3Patcher, Trie, load  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def setup():
    tok, model = load("Qwen/Qwen3-0.6B", "cpu", torch.float32)
    render = lambda e: tok.apply_chat_template([{"role": "user", "content": f"Compute {e}. Reply with only the final number, nothing else."}],
                                               add_generation_prompt=True, enable_thinking=False, tokenize=False)
    base = tok(render("(69 - 48) - 17"), add_special_tokens=False)["input_ids"]
    cfs = [tok(render(e), add_special_tokens=False)["input_ids"] for e in ("(83 - 25) - 39", "(57 - 36) - 24")]
    assert all(len(c) == len(base) for c in cfs)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    answers = [[tok.convert_tokens_to_ids(c) for c in s] + [end] for s in ("4", "19", "-3", "42", "2")]
    return tok, model, base, cfs, answers


def test_run_matches_hf(setup):
    tok, model, base, _, _ = setup
    fast = Qwen3Patcher(model)
    tr = fast.run(base)
    ref = model(torch.tensor([base]), use_cache=False).logits[0]
    assert torch.allclose(fast.logits(tr.h[-1]), ref, atol=1e-3, rtol=1e-4)
    assert torch.allclose(tr.h, HookPatcher(model).states(base), atol=1e-4, rtol=1e-4)


def test_patch_matches_hooks(setup):
    tok, model, base, cfs, answers = setup
    fast, ref = Qwen3Patcher(model), HookPatcher(model)
    b = fast.run(base)
    cft = [fast.run(c) for c in cfs]
    tries = [Trie.build(answers), Trie.build(answers[1:4])]
    states, unpatched = zip(*[fast.answer_states(b, t) for t in tries])
    want0 = ref.score(base, answers)
    assert torch.allclose(unpatched[0], torch.tensor(want0), atol=1e-3)
    L = fast.L
    for p in (5, 10, len(base) - 1):
        got = fast.patch_position(b, cft, p, tries, states)
        for j, (cf, trie) in enumerate(zip(cft, tries)):
            for l in (-1, 0, L // 2, L - 1):
                want = ref.score(base, [answers[i] for i in ([0, 1, 2, 3, 4] if j == 0 else [1, 2, 3])], (l, p), cf.h[l + 1, p])
                assert torch.allclose(got[j][l + 1], torch.tensor(want), atol=2e-3), (p, j, l, got[j][l + 1], want)


def test_batch_hooks_match_fast_path(setup):
    tok, model, base, cfs, answers = setup
    fast, batched = Qwen3Patcher(model), BatchHookPatcher(model, batch=16)
    b, cf = fast.run(base), fast.run(cfs[0])
    trie = Trie.build(answers)
    states, _ = fast.answer_states(b, trie)
    L = fast.L
    for p in (5, len(base) - 1):
        got = fast.patch_position(b, [cf], p, [trie], [states])[0]
        cells = [(l, p) for l in range(-1, L)]
        want = batched.score(base, answers, cells, [cf.h[l + 1, p] for l, _ in cells])
        assert torch.allclose(got, want, atol=2e-3), (p, (got - want).abs().max())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
