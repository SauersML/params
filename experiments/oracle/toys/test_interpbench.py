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
    # In float64: without a norm the logits reach ~55, where float32's rounding alone differs by 2-3e-5 between
    # platforms (GHA's Linux CPU against the Mac); float64 agrees to 1e-13.
    sd64 = {k: v.double() for k, v in sd.items()}
    with torch.no_grad():
        torch.testing.assert_close(vpd_model.TLTarget(sd64, cfg).double().eval()(ids), tl_forward(sd64, cfg, ids), atol=1e-9, rtol=1e-9)
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


@pytest.mark.parametrize("norm", [None, "LN"])
@pytest.mark.parametrize("biased", [True, False])
def test_oracle_graph_runs_equal_the_engine(norm, biased, monkeypatch):
    """native.Native.run against engine.Engine's graph runs (written apart) on the TL target with and without its
    biases, each matrix's last piece left in its remainder: every subcomponent on and off, one node, two connected
    nodes, the empty graph and a complete one (the same nodes with every connection the model has among them stated),
    the remainders live and off, a patch (of a graph node and of a subcomponent outside) and declared connections at
    partial strength (the norm's strength tied to the read's). The masks run() keeps (_zero(), _one()) give the
    numbers of equal masks built anew: a bias writes at every u, a value bias through every attention row, so a biased
    target never takes _zero()'s shortcut."""
    import engine
    import native

    sd, cfg = random_tl(norm)
    if not biased:
        sd = {k: v for k, v in sd.items() if not (k.startswith("blocks.") and ".b_" in k)}
    exact = interpbench.decompose
    monkeypatch.setattr(interpbench, "decompose", lambda tg: {n: (U[:-1], V[:, :-1], unit[:-1]) for n, (U, V, unit) in exact(tg).items()})
    nat = interpbench.native_of(sd, cfg)
    assert min(float(nat.target.site(n).delta_T.norm()) for n in nat.names) > 0.01
    assert nat._biased() == biased
    ids = torch.randint(cfg["d_vocab"], (cfg["n_ctx"],), generator=torch.Generator().manual_seed(2)).tolist()
    targets = [cfg["n_ctx"] - 1]
    T = len(ids)
    with torch.no_grad():
        lp = torch.log_softmax(nat.target(torch.tensor([ids]))[0, targets].float(), -1)
    one = (native.site_name(0, "o_proj"), T - 2, 1)
    two = [(native.site_name(0, "o_proj"), T - 1, 0), (native.site_name(1, "c_fc"), T - 1, 1)]
    connected = native.Graph(two, *native.all_edges(two, targets))
    assert connected.parents and connected.out
    graphs = {"empty": native.Graph([]), "one": native.Graph([one]), "two": connected,
              "complete": native.Graph(two, complete=True)}
    g = torch.Generator().manual_seed(3)
    zu, zur = nat._zero()
    ou, our = nat._one()
    us = {"zero": zu, "zero anew": {n: m.clone() for n, m in zu.items()}, "one": ou, "one anew": {n: m.clone() for n, m in ou.items()},
          "random": {n: torch.rand(1, T, nat.C[n], generator=g) for n in nat.names}}
    urs = {"off": zur, "off anew": {n: m.clone() for n, m in zur.items()}, "live": our}
    engines = {live: engine.Engine(nat, ids, targets, live=live) for live in (True, False)}
    full = lambda u: {n: m.expand(1, T, -1) for n, m in u.items()}  # noqa: E731
    patch = {}
    for (site, t, c), value in (((native.site_name(0, "c_fc"), T - 2, 2), 1.5), (two[1], 0.7)):
        patch[site] = (torch.zeros(1, T, nat.C[site], dtype=torch.bool), torch.zeros(1, T, nat.C[site]))
        patch[site][0][0, t, c], patch[site][1][0, t, c] = True, value

    with torch.no_grad():
        for gn, graph in graphs.items():
            plan = nat._plan([graph], T)
            for un, u in us.items():
                for rn, ur in urs.items():
                    eng = engines[rn == "live"]
                    got = nat.run(ids, targets, plan, u, ur)
                    want = eng.graph_log_probs(eng.plan(connected if graph.complete else graph), full(u)).reshape(got.shape)
                    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5, msg=lambda m: f"{gn} u {un} remainders {rn}: {m}")
                    anew = nat.run(ids, targets, plan, {n: m.clone() for n, m in u.items()}, {n: m.clone() for n, m in ur.items()})
                    torch.testing.assert_close(got, anew, atol=1e-6, rtol=1e-6, msg=lambda m: f"{gn} u {un} remainders {rn}: {m}")
                    if un.startswith("one") and rn == "live" and gn != "complete":
                        torch.testing.assert_close(got.reshape(lp.shape), lp, atol=1e-5, rtol=1e-5)
        plan = nat._plan([connected], T)
        ew = torch.full((1, sum(len(w) for w in connected.parents.values())), 0.3)
        ow = torch.full((1, len(connected.out)), 0.6)
        for un in ("zero", "zero anew", "random"):
            for rn, ur in urs.items():
                eng = engines[rn == "live"]
                got = nat.run(ids, targets, plan, us[un], ur, patch=patch)
                want = eng.graph_log_probs(eng.plan(connected), full(us[un]), patch=patch).reshape(got.shape)
                torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5, msg=lambda m: f"patch u {un} remainders {rn}: {m}")
                got = nat.run(ids, targets, plan, us[un], ur, *nat.edge_weights(plan, torch.cat([ew, ow], 1).flatten()))
                want = eng.graph_log_probs(eng.plan(connected), full(us[un]), ew, ow).reshape(got.shape)
                torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5, msg=lambda m: f"strengths u {un} remainders {rn}: {m}")


def mib_reference(L=2, H=2, truth=()):
    """A reference graph in MIB's form (EAP-IG's Graph.to_json: its edges with in_graph flags, the model's shape)."""
    cfg = {"n_layers": L, "n_heads": H, "parallel_attn_mlp": False}
    return {"cfg": cfg, "edges": {e: {"in_graph": e in truth} for e in interpbench.mib_graph_edges(cfg)}}


def test_mib_graph_edges():
    """EAP-IG's edge set and order for MIB's InterpBench shape (6 layers, 4 heads; its published importances): 1108 edges,
    the input into layer 0's queries, keys and values first, the last MLP into the logits last."""
    E = interpbench.mib_graph_edges({"n_layers": 6, "n_heads": 4, "parallel_attn_mlp": False})
    assert len(E) == len(set(E)) == 1108
    assert E[:4] == ["input->a0.h0<q>", "input->a0.h0<k>", "input->a0.h0<v>", "input->a0.h1<q>"]
    assert E[12:14] == ["input->m0", "a0.h0->m0"] and E[-1] == "m5->logits"


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


def test_native_mib_edges():
    """A native circuit's implied MIB edges: a head into a later head's key and value from an earlier position but its
    query only at the same position; a head into its own layer's MLP at the same position, not at another; neurons at
    one layer and position one MLP node; the logits from units at the target; every edge in EAP-IG's graph."""
    cfg = {"n_layers": 3, "n_heads": 2, "parallel_attn_mlp": False}
    circ = {"units": "native", "nodes": [["head", 0, 2, 1], ["head", 1, 4, 0], ["neuron", 1, 2, 7], ["neuron", 1, 2, 9],
                                         ["neuron", 2, 4, 3]]}
    got = interpbench.mib_edges(None, circ, [4])
    assert {"a0.h1->a1.h0<k>", "a0.h1->a1.h0<v>", "a0.h1->m1", "a1.h0->m2", "m2->logits", "a1.h0->logits",
            "input->a1.h0<q>", "input->m1", "input->logits"} <= got
    assert not {"a0.h1->a1.h0<q>", "a1.h0->m1", "m1->m2", "a0.h1->logits", "m1->logits"} & got
    assert got <= set(interpbench.mib_graph_edges(cfg))


def test_mib_score_paired():
    """mib_score.score: a builder whose every prompt holds exactly the reference circuit beats a baseline that scores the
    reverse ranking, with the paired difference's interval above 0; edges per prompt counts the circuit's edges."""
    import numpy as np
    import mib_score

    truth = ["input->m0", "m0->a1.h1<v>", "a1.h1->logits"]
    ref = mib_reference(truth=truth)
    edges = interpbench.mib_graph_edges(ref["cfg"])
    T = np.array([[e in truth for e in edges]] * 8, dtype=np.float64)
    x = np.array([[0.0 if e in truth else 1.0 for e in edges]] * 8)
    out = mib_score.score(ref, T, x, B=50)
    assert out["prompts"] == 8 and out["edges_per_prompt"] == 3.0 and out["plain_auroc"] == 1.0
    assert out["auroc"] > out["eap_ig_inputs"]["auroc"] and out["diff_vs_eap_ig_inputs"]["ci95"][0] > 0


def test_mib_score_nodes():
    """A builder in heads and MLPs: its components hold every graph edge between them, the input and the logits always
    in; a head's query, key and value inputs belong to the head."""
    import mib_score

    edges = interpbench.mib_graph_edges({"n_layers": 2, "n_heads": 2, "parallel_attn_mlp": False})
    got = set(mib_score.edges_of_nodes(["a1.h1", "m0"], edges))
    assert {"input->m0", "m0->a1.h1<v>", "a1.h1->logits", "input->a1.h1<q>", "m0->logits"} <= got
    assert not any("a0." in e or "m1" in e or "a1.h0" in e for e in got)
