"""vpd_model.TLTarget (TransformerLens models: no norm or LayerNorm, learned positions, biases, untied unembedding) against
TransformerLens' own equations, and the oracle's runs on it with every subcomponent kept against the model."""
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import interpbench  # noqa: E402
import vpd_model  # noqa: E402


def random_tl(norm, L=2, H=3, d=12, hd=4, V=9, Vout=5, T=8, d_mlp=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.4  # noqa: E731
    sd = {"embed.W_E": r(V, d), "pos_embed.W_pos": r(T, d), "unembed.W_U": r(d, Vout), "unembed.b_U": r(Vout)}
    for i in range(L):
        p = f"blocks.{i}."
        for k in "QKV":
            sd[p + f"attn.W_{k}"], sd[p + f"attn.b_{k}"] = r(H, d, hd), r(H, hd)
        sd[p + "attn.W_O"], sd[p + "attn.b_O"] = r(H, hd, d), r(d)
        sd[p + "mlp.W_in"], sd[p + "mlp.b_in"], sd[p + "mlp.W_out"], sd[p + "mlp.b_out"] = r(d, d_mlp), r(d_mlp), r(d_mlp, d), r(d)
        if norm == "LN":
            for k in ("ln1", "ln2"):
                sd[p + f"{k}.w"], sd[p + f"{k}.b"] = 1 + r(d), r(d)
    if norm == "LN":
        sd["ln_final.w"], sd["ln_final.b"] = 1 + r(d), r(d)
    cfg = {"n_layers": L, "n_heads": H, "d_model": d, "d_head": hd, "n_ctx": T, "d_vocab": V, "d_vocab_out": Vout, "d_mlp": d_mlp,
           "act_fn": "gelu", "normalization_type": norm, "eps": 1e-5, "attention_dir": "causal"}
    return sd, cfg


def tl_forward(sd, cfg, ids):
    """TransformerLens' HookedTransformer forward, written from its equations."""
    T = ids.shape[1]
    ln = (lambda x, k: F.layer_norm(x, (x.shape[-1],), sd[k + ".w"], sd[k + ".b"], cfg["eps"])) if cfg["normalization_type"] == "LN" else (lambda x, k: x)
    x = sd["embed.W_E"][ids] + sd["pos_embed.W_pos"][:T]
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    for i in range(cfg["n_layers"]):
        p = f"blocks.{i}."
        h = ln(x, p + "ln1")
        q, k, v = (torch.einsum("btd,hde->bthe", h, sd[p + f"attn.W_{c}"]) + sd[p + f"attn.b_{c}"] for c in "QKV")
        a = (torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(cfg["d_head"])).masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + torch.einsum("bqhe,hed->bqd", torch.einsum("bhqk,bkhe->bqhe", a, v), sd[p + "attn.W_O"]) + sd[p + "attn.b_O"]
        h = ln(x, p + "ln2")
        x = x + F.gelu(h @ sd[p + "mlp.W_in"] + sd[p + "mlp.b_in"]) @ sd[p + "mlp.W_out"] + sd[p + "mlp.b_out"]
    return ln(x, "ln_final") @ sd["unembed.W_U"] + sd["unembed.b_U"]


@pytest.mark.parametrize("norm", [None, "LN"])
def test_tl_target_equals_transformerlens(norm):
    sd, cfg = random_tl(norm)
    tg = vpd_model.TLTarget(sd, cfg).eval()
    ids = torch.randint(cfg["d_vocab"], (4, cfg["n_ctx"]), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(tg(ids), tl_forward(sd, cfg, ids), atol=1e-5, rtol=1e-5)
    with pytest.raises(AttributeError):
        tg.wte  # noqa: B018  (vpd4l's attributes raise: a caller still reading them fails)
    with pytest.raises(ValueError):
        vpd_model.TLTarget(sd, dict(cfg, attention_dir="bidirectional"))


@pytest.mark.parametrize("norm", [None, "LN"])
def test_oracle_runs_reproduce_the_model(norm):
    import engine
    import goal_eval
    import native

    sd, cfg = random_tl(norm)
    nat = interpbench.native_of(sd, cfg)
    ids = torch.randint(cfg["d_vocab"], (cfg["n_ctx"],), generator=torch.Generator().manual_seed(2)).tolist()
    targets = [cfg["n_ctx"] - 1]
    T = len(ids)
    with torch.no_grad():
        lp = torch.log_softmax(nat.target(torch.tensor([ids]))[0, targets].float(), -1)
    everything = native.Graph([(n, t, c) for n in nat.names for t in range(T) for c in range(nat.C[n])], complete=True)
    close = lambda a: torch.testing.assert_close(a.reshape(lp.shape), lp, atol=1e-4, rtol=1e-4)  # noqa: E731
    close(nat.run(ids, targets, nat._full_plan(T, 1), *nat._one())[0])
    E = engine.Engine(nat, ids, targets)
    close(E.log_probs({n: torch.ones(1, T, E.width(n)) for n in nat.names}, "score")[0])
    ge = goal_eval.GoalEval(nat, chunk=8)
    tx = ge.text(ids, targets, 0)
    close(ge.graph_ref(ids, targets, everything)[0])
    x2 = list(ids)
    x2[3] = (x2[3] + 1) % cfg["d_vocab"]
    with torch.no_grad():
        lp2 = torch.log_softmax(nat.target(torch.tensor([x2]))[0, targets].float(), -1)
    torch.testing.assert_close(ge.pp_run(tx, x2, everything)[0].reshape(lp2.shape), lp2, atol=1e-4, rtol=1e-4)
    assert ge.track_pp(tx, [everything], "eval")[0]["s_track_pp"] > 0.999


def mib_reference(L=2, H=2, truth=()):
    """A reference graph in MIB's form (EAP-IG's Graph.to_json: every edge from a node into a later node's input, a head's
    query, key and value inputs apart)."""
    fwd = ["input"] + [x for i in range(L) for x in [f"a{i}.h{h}" for h in range(H)] + [f"m{i}"]]
    layer = lambda x: -1 if x == "input" else int(x[1:].split(".")[0])  # noqa: E731
    edges = {}
    for a in fwd:
        for i in range(L):
            for h in range(H):
                if layer(a) < i:
                    for q in "qkv":
                        edges[f"{a}->a{i}.h{h}<{q}>"] = {"in_graph": False}
            if layer(a) < i or (layer(a) == i and a.startswith("a")):
                edges[f"{a}->m{i}"] = {"in_graph": False}
        edges[f"{a}->logits"] = {"in_graph": False}
    for e in truth:
        edges[e]["in_graph"] = True
    return {"edges": edges}


def test_mib_auroc_ranks():
    """A score that ranks the reference circuit's edges first scores 1 (plain) and near 1 under MIB's greedy curve; the
    reverse ranking scores 0 and lower."""
    truth = ["input->m0", "m0->a1.h1<v>", "a1.h1->logits"]
    ref = mib_reference(truth=truth)
    good = interpbench.mib_auroc(ref, {e: 1.0 for e in truth})
    assert good["plain"] == 1.0 and good["mib"] > 0.95 and good["tpr"][-1] == 1.0 and good["fpr"][-1] == 1.0
    bad = interpbench.mib_auroc(ref, {e: 0.0 if e in truth else 1.0 for e in ref["edges"]})
    assert bad["plain"] == 0.0 and bad["mib"] < good["mib"]


def test_mib_edge_names():
    """Circuit connections as MIB's edge names: the token into a head's value input, a head into an MLP, an MLP into the
    logits; a head's own value into its output is none."""
    nat = types.SimpleNamespace(units={"h.0.attn.o_proj": torch.tensor([0, 1]), "h.1.attn.v_proj": torch.tensor([1, 0]),
                                       "h.1.attn.o_proj": torch.tensor([1, 0]), "h.1.mlp.c_fc": torch.tensor([0, 0])})
    assert interpbench.mib_edge(nat, ("tok", 3), ("h.1.attn.v_proj", 0)) == "input->a1.h1<v>"
    assert interpbench.mib_edge(nat, ("h.0.attn.o_proj", 1), ("h.1.mlp.c_fc", 0)) == "a0.h1->m1"
    assert interpbench.mib_edge(nat, ("h.1.mlp.down_proj", 5), "out") == "m1->logits"
    assert interpbench.mib_edge(nat, ("h.1.attn.v_proj", 1), ("h.1.attn.o_proj", 1)) is None
