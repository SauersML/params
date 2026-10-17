"""nodes._fast_sink equals their SDPAWithSink (a tiny random sink model, CPU)."""
import os
import sys

import torch

import nodes as N


def test_fast_sink_matches():
    sys.path.insert(0, f"{N.ROOT}/ext/circuit_sparsity")
    from circuit_sparsity.inference.gpt import GPT, GPTConfig
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=64, vocab_size=50, n_layer=2, n_head=4, d_head=8, d_model=32, d_mlp=64, flash=True,
                    sink=True, rms_norm=True)
    m = GPT(cfg).eval()
    for blk in m.transformer.h:
        blk.attn.attn_imp.sink_logit.data.normal_()
    x = torch.randint(0, 50, (3, 17))
    with torch.no_grad():
        ref = m(x)[0]
        N._fast_sink(m)
        got = m(x)[0]
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


if __name__ == "__main__":
    test_fast_sink_matches()
    print("ok")
