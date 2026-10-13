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
