"""uv.py against VPD's checkpoint: the published tensors are the checkpoint's, and every oriented, normalized pair
rebuilds its rank-one slice u v^T (skipped without ~/mpd-data's files).

  ~/.local/bin/mem-lease 8 ~/mpd-data/venv/bin/python -m pytest method/vpd/test_uv.py -q
"""
import pytest
import torch

from method.vpd import uv as UVm
from method.vpd import vpd_model as VM


@pytest.fixture(scope="module")
def both():
    if not (UVm.PATH.exists() and VM.VPD_PTH.exists()):
        pytest.skip("needs ~/mpd-data's subcomponents and VPD's checkpoint")
    target = VM.load_target("cpu")
    vpd = VM.load_vpd(target, "cpu")
    return UVm.UV(), vpd


def test_published_tensors_are_the_checkpoints(both):
    uv, vpd = both
    assert uv.names() == sorted(vpd.names) and sum(uv.count(n) for n in uv.names()) == 38912
    for n in vpd.names:
        st = vpd.target.site(n)
        assert torch.equal(uv.t[f"{n}.U"], st.U.cpu()) and torch.equal(uv.t[f"{n}.V"], st.V.cpu())


def test_oriented_pair_rebuilds_the_slice(both):
    uv, vpd = both
    g = torch.Generator().manual_seed(0)
    for n in vpd.names:
        st = vpd.target.site(n)
        for c in torch.randint(0, uv.count(n), (8,), generator=g).tolist():
            e = uv.get(n, c)
            want = torch.outer(st.U[c], st.V[:, c])  # the checkpoint's slice of W [d_out, d_in]
            got = e["scale"] * torch.outer(e["u"], e["v"])
            assert torch.allclose(got, want, atol=1e-6 * float(want.abs().max()) + 1e-12)
            assert torch.equal(uv.slice(n, c), want)
            assert abs(float(e["u"].norm()) - 1) < 1e-5 and abs(float(e["v"].norm()) - 1) < 1e-5
            assert e["v"].numel() == st.W.shape[1] and e["u"].numel() == st.W.shape[0]


def test_slices_and_remainder_make_the_weight(both):
    uv, vpd = both
    n = vpd.names[3]
    st = vpd.target.site(n)
    total = sum(uv.slice(n, c) for c in range(uv.count(n)))
    assert torch.allclose(total + (st.W - (st.V @ st.U).T), st.W, atol=1e-5)


def test_parse():
    assert UVm.parse("L2.o#77") == ("h.2.attn.o_proj", 77)
    assert UVm.parse("L0.fc#53") == ("h.0.mlp.c_fc", 53)
