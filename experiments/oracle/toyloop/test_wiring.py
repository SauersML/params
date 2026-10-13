"""Wiring must change execution even when the selected units stay fixed."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "method/oracle"), str(ROOT / "method/vpd")]

import circuit
import reward_exec
from test_goal_eval import tiny_native


@pytest.fixture
def pair():
    # Two exact rank-one MLP neurons. No matrix decomposition or model download.
    torch.set_num_threads(1)
    nat = tiny_native(seed=8, layers=2, d=8, heads=1, vocab=12, C=1)
    nat.units = {}
    with torch.no_grad():
        # Guaranteed live path: input e0 -> neuron0 writes e1 -> neuron1 writes e2.
        # Random test weights can leave both GELUs inactive and make the removed edge inert.
        nat.target.wte.zero_()
        nat.target.wte[1:7, 0] = 1
        nat.target.wte[7, 1] = 1
        nat.target.wte[8, 2] = 1
        for norm in nat.target.norms:
            norm.fill_(1)
        nat.target.ln_f.fill_(1)
        for n in nat.names:
            st = nat.target.site(n)
            layer = int(n.split(".")[1])
            if ".attn." in n:
                st.U.zero_()
                st.V.zero_()
                nat.units[n] = torch.tensor([-1])
            elif n.endswith("c_fc"):
                st.U.zero_()
                st.U[0, 0] = 1
                st.V.zero_()
                st.V[layer, 0] = 1
                nat.units[n] = torch.tensor([0])
            else:
                st.V.zero_()
                st.V[0, 0] = 1
                st.U.zero_()
                st.U[0, layer + 1] = 1 + layer
                nat.units[n] = torch.tensor([0])
            st.W.copy_((st.V @ st.U).T)
            st.delta_T.zero_()
    dec = {n: (nat.target.site(n).U, nat.target.site(n).V, nat.units[n]) for n in nat.names}
    toy = SimpleNamespace(cfg={"n_layer": 2, "n_head": 1},
                          sd={f"{n}.weight": nat.target.site(n).W for n in nat.names},
                          texts=[[1, 2, 3], [4, 5, 6]], meta={"decomposition": dec})
    return toy, nat


FULL = "flowchart LR\nL0N0 --> L1N0\nL0N0 --> out\nL1N0 --> out\nEnglish: unverified"
CUT = "flowchart LR\nL0N0 --> out\nL1N0 --> out\nEnglish: unverified"


def test_same_units_different_wiring_changes_execution(pair):
    toy, nat = pair
    assert circuit.parse(FULL).units == circuit.parse(CUT).units
    full = reward_exec.score_wiring(toy, FULL, nat=nat)
    cut = reward_exec.score_wiring(toy, CUT, nat=nat)
    assert full["kl_bits"] < 1e-5, full
    assert cut["kl_bits"] > full["kl_bits"] + 1e-4, (full, cut)
    assert all(r["top"] == r["reference_top"] == 8 for r in full["per_text"])
    assert all(r["top"] == 7 for r in cut["per_text"])
    assert full["per_text"][0]["nodes"] == cut["per_text"][0]["nodes"]
    assert full["per_text"][0]["edges"] > cut["per_text"][0]["edges"]


def test_training_wiring_reward_executes_edges_and_rejects_partial_invalid(pair, monkeypatch):
    import planted
    import train

    toy, nat = pair
    monkeypatch.setattr(planted, "native_of", lambda sd, cfg: nat)
    # Collision with legacy name cache must not select another model's decomposition.
    toy.name = "same-name"
    monkeypatch.setitem(reward_exec._CACHE, toy.name, {"nat": object()})
    reward = train.get_reward("wiring_suff")
    assert reward is reward_exec.reward_wiring_suff
    assert train.get_reward("exec") is reward_exec.reward_exec
    full = reward({"toy": toy}, FULL)
    cut = reward({"toy": toy}, CUT)
    assert full > .999 and 0 <= cut < full
    unit_reward = train.get_reward("unit_suff")
    assert unit_reward is reward_exec.reward_unit_suff
    assert unit_reward({"toy": toy}, CUT) == pytest.approx(unit_reward({"toy": toy}, FULL), abs=1e-7)
    assert unit_reward({"toy": toy}, CUT) > cut
    for answer in ("unparseable", FULL.replace("English:", "L0N999 --> out\nEnglish:"),
                   FULL.replace("English:", "unknown --> out\nEnglish:")):
        assert reward({"toy": toy}, answer) == -1 < cut
        assert unit_reward({"toy": toy}, answer) == -1


def test_training_wiring_reward_penalizes_one_failed_text(monkeypatch):
    toy = SimpleNamespace(texts=[[1], [2]], meta={})
    calls = []
    def measured(got_toy, completion, *, texts, complete):
        calls.append((got_toy, texts))
        return {"per_text": [{"suff": 1.0}, {"suff": 0.0}]}
    monkeypatch.setattr(reward_exec, "score_wiring", measured)
    assert reward_exec.reward_wiring_suff({"toy": toy}, FULL) == .25
    assert calls == [(toy, toy.texts)]


def test_training_seed_controls_initialization_and_sampled_completions(monkeypatch):
    import train

    initialized, sampled = [], []
    def load_policy(args):
        model = torch.nn.Linear(1, 1, bias=False)
        initialized.append(model.weight.detach().clone())
        return model, None
    def sample(model, tok, prompt, k, args):
        result = [str(float(torch.rand(()))) for _ in range(k)]
        sampled.append(result)
        return result
    monkeypatch.setattr(train, "examples", lambda *a: [{"prompt": "x", "completion": "y", "meta": {}}])
    monkeypatch.setattr(train, "load_policy", load_policy)
    monkeypatch.setattr(train, "sample", sample)
    monkeypatch.setattr(train, "get_reward", lambda spec: lambda meta, text: float(text))
    monkeypatch.setattr(train, "batch_ids", lambda tok, pairs, cap: (torch.zeros(len(pairs), 1), None, torch.ones(len(pairs), 1)))
    monkeypatch.setattr(train, "token_logps", lambda model, ids, att, comp: (model.weight.expand(len(ids), 1), comp))
    monkeypatch.setattr(train, "save", lambda *a: None)
    args = SimpleNamespace(data="mock", truth=None, seed=11, steps=1, batch=1, lr=.001,
                           max_len=4, log_every=1, eval=0, out="unused", reward="mock", beta=0, k=2, std=False)
    train.sft(args); train.sft(args)
    assert torch.equal(initialized[0], initialized[1])
    train.rl(args); train.rl(args)
    assert torch.equal(initialized[2], initialized[3]) and sampled[0] == sampled[1]
    args.seed = 12
    train.rl(args)
    assert sampled[2] != sampled[0]


def test_wrong_direction_and_unknown_unit_rejected(pair):
    toy, _ = pair
    for text in ("flowchart LR\nL1N0 --> L0N0\nL0N0 --> out",
                 "flowchart LR\nL3N0 --> out", "flowchart LR\nL0N999 --> out"):
        with pytest.raises(ValueError):
            circuit.unit_graph(circuit.parse(text), toy)


def test_token_roles_are_explicitly_annotations(pair):
    toy, _ = pair
    full = circuit.unit_graph(circuit.parse(FULL), toy)
    labelled = circuit.unit_graph(circuit.parse("tokROLE --> L0N0\n" + FULL), toy)
    assert full.nodes == labelled.nodes
    assert full.parents == labelled.parents and full.out == labelled.out
    assert not full.complete


def test_all_pieces_do_not_depend_on_activation_or_truth(pair):
    toy, _ = pair
    toy.truth = {"units": []}
    toy.meta["record"] = {"gt_nodes": []}
    g = circuit.unit_graph(circuit.parse(FULL), toy)
    assert g.count() == 2 * 2 * len(toy.texts[0])  # input/output pieces for both neurons
    with pytest.raises(ValueError):
        circuit.unit_graph(circuit.parse(FULL), toy, [])


@pytest.mark.parametrize('wiring', [False, True])
def test_swapped_completion_executes_each_models_own_basis_and_texts(monkeypatch, wiring):
    import evaluate

    toys = [SimpleNamespace(name=name, meta={'family': 'mock'}, texts=[[i]],
                           truth={'units': ['L0N0'], 'edges': [('L0N0', 'out')]})
            for i, name in enumerate(['own', 'other'], 1)]
    natives = {toy.name: SimpleNamespace(owner=toy.name) for toy in toys}
    monkeypatch.setattr(evaluate.render, 'render', lambda toy, **kw: toy.name)
    monkeypatch.setattr(reward_exec, 'score', lambda *a: {'reward': .5, 'suff': .6, 'nec': .4})
    monkeypatch.setattr(reward_exec, '_state', lambda toy: {'nat': natives[toy.name]})
    monkeypatch.setattr(reward_exec, 'texts_of', lambda toy: toy.texts)
    executed = []

    def score_wiring(toy, completion, *, nat, texts):
        assert nat is natives[toy.name] and texts is toy.texts
        executed.append((toy.name, tuple(map(tuple, texts))))
        return {'suff': .7 if toy.name == 'own' else .2, 'kl_bits': 1. if toy.name == 'own' else 4.,
                'scope': 'point wiring only; embeddings on; token labels unverified'}

    monkeypatch.setattr(reward_exec, 'score_wiring', score_wiring)
    result = evaluate.run('mock', lambda *a: FULL, {'test': toys}, swapped=True,
                          wiring=wiring, log=lambda *a: None)
    first = result['rows'][0]
    if wiring:
        assert len(executed) == 6  # normal + own/donor swap checks for each toy
        assert first['swap_wiring_kl_bits_own'] == 1.
        assert first['swap_wiring_kl_bits_donor'] == 4.
        assert first['swap_wiring_valid_own'] == first['swap_wiring_valid_donor'] == 1.
        assert 'token labels unverified' in first['swap_wiring_scope_donor']
        assert 'swap_wiring_kl_bits_donor' in evaluate.markdown(result)
    else:
        assert executed == []
        assert not any('wiring' in key for key in first)


def test_mixed_invalid_wiring_keeps_kl_conditional_without_zero_filling():
    import evaluate

    summary = evaluate.mean_rows([
        {'wiring_valid': 1., 'wiring_suff': .8, 'wiring_kl_bits': 2., 'swap_wiring_kl_bits_own': 3.},
        {'wiring_valid': 0., 'wiring_suff': 0., 'wiring_kl_bits': None, 'swap_wiring_kl_bits_own': None},
    ])
    assert summary['n'] == 2
    assert summary['wiring_valid'] == .5 and summary['wiring_suff'] == .4
    assert summary['wiring_kl_bits'] == 2.  # conditional on measured KL, not 1 from invalid->0
    assert summary['swap_wiring_kl_bits_own'] == 3.


def test_mac_decomposition_uses_scipy_and_preserves_head_weights(monkeypatch):
    import planted

    cfg = {"n_layer": 1, "n_embd": 4, "n_head": 2}
    rng = torch.Generator().manual_seed(24)
    sd = {f"h.0.attn.{kind}.weight": torch.randn(4, 4, generator=rng)
          for kind in ("q_proj", "k_proj", "v_proj", "o_proj")}
    sd["h.0.attn.k_proj.weight"].zero_()
    sd["h.0.mlp.c_fc.weight"] = torch.randn(8, 4, generator=rng)
    sd["h.0.mlp.down_proj.weight"] = torch.randn(4, 8, generator=rng)
    monkeypatch.setattr(planted, "sys", SimpleNamespace(platform="darwin"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Mac decomposition must not enter torch.linalg.svd")

    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    for site, (u, v, units) in planted.decompose(sd, cfg).items():
        weight = sd[f"{site}.weight"]
        assert u.dtype == v.dtype == weight.dtype
        assert torch.allclose((v @ u).T, weight, atol=2e-6, rtol=2e-6)
        if ".attn." in site:
            for head in range(2):
                expected = torch.zeros_like(weight)
                if site.endswith("o_proj"):
                    expected[:, head * 2:(head + 1) * 2] = weight[:, head * 2:(head + 1) * 2]
                else:
                    expected[head * 2:(head + 1) * 2] = weight[head * 2:(head + 1) * 2]
                selected = units == head
                assert torch.allclose((v[:, selected] @ u[selected]).T, expected, atol=2e-6, rtol=2e-6)
