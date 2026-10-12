"""Checks of teacher.py on a held-out vpd4l text, on this machine's GPU (skipped without one, vpd4l's texts, or VPD's
answer: the causal-importance network or teacher.py --save-vpd's file VPD_SAVED): the windowed masked runs
(Pruner.kl_sets on a device in Pruner.WINDOWS) equal the whole runs to the bit on MPS and within float32 rounding on
CUDA, and the products on the start set's subcomponents alone (Pruner._gather) with torch's fused norms and GELU
(Pruner.fused) match the products on all subcomponents with vpd_model's own steps within float32 rounding,
EdgePruner's runs are the scorecard's (goal_eval.GoalEval.graph_kl: with every connection the masked run), and a
teacher circuit rebuilt from its saved path (teacher.py --from-path) is the full run's, nodes and connections, one
over the answer budget steps to a smaller path point, the flip repair writes its pick, and the robust connection
objective keeps F1 under the flips at its floor.

  ~/.local/bin/mem-lease 24 ~/mpd-data/venv/bin/python -m pytest method/oracle/test_teacher.py -s
"""

import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import teacher  # noqa: E402

TEXT = native.TEXTS / "vpd4l" / "text9000.json"
VPD_SAVED = Path.home() / "mpd-data/graph_oracle/teacher/vpd_heldout20.pt"
# KL in bits is a difference of nearly equal log probabilities; a float32 sum of n terms is off by about
# sqrt(n) 2^-24 of its size (n = 3,072 at the MLP: 3.3e-6), and the KL's cancellation magnifies the logits' relative
# error tenfold on these texts (measured: 2.2e-5 on an RTX A5000, 2e-5 for the gathered products on MPS)
REL = 1e-4
DEVICE = native.device()


def pruner(live: bool):
    nat = native.Native(DEVICE)
    ge = goal_eval.GoalEval(nat, chunk=32)
    ids, targets = native.text(TEXT)
    tx = ge.text(ids, targets)
    if tx.vpd is None:
        if not VPD_SAVED.exists():
            pytest.skip("needs VPD's answer: the causal-importance network or " + str(VPD_SAVED))
        tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (t, c) in torch.load(VPD_SAVED)["text9000"]["answer"].items():
            tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
    return nat, teacher.Pruner(ge, tx, tx.vpd, live=live, B=32)


def close(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    return float(((a - b).abs() / a.abs()).max()) < REL


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
@pytest.mark.parametrize("live", [False, True])
def test_windowed_runs_match_whole_runs(live):
    nat, pr = pruner(live)
    nodes = torch.randperm(len(pr.nodes), generator=torch.Generator().manual_seed(0))[:96].to(nat.dev)
    trials = pr.on[None].repeat(96, 1)
    trials[torch.arange(96, device=nat.dev), nodes] = False   # single-node removals at mixed layers and positions
    blocks = pr.blocks("site")[::40]                            # site removals spread over the text
    keep = teacher.Pruner.WINDOWS
    try:
        teacher.Pruner.WINDOWS = ()
        whole = (pr.kl_sets(trials), pr.kl_sets(~trials), pr.without(blocks))
        teacher.Pruner.WINDOWS = keep
        pr._prefixes = {}
        windowed = (pr.kl_sets(trials), pr.kl_sets(~trials), pr.without(blocks))
        again = (pr.kl_sets(trials), pr.kl_sets(~trials), pr.without(blocks))
    finally:
        teacher.Pruner.WINDOWS = keep
    for a, b, c in zip(whole, windowed, again):
        assert b == c                                           # the same sets, the same batches: the same numbers
        assert a == b if DEVICE == "mps" else close(a, b)


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
def test_gathered_products_match_full_products():
    nat, pr = pruner(False)
    nodes = torch.randperm(len(pr.nodes), generator=torch.Generator().manual_seed(1))[:64].to(nat.dev)
    trials = pr.on[None].repeat(64, 1)
    trials[torch.arange(64, device=nat.dev), nodes] = False
    gathered = pr.kl_sets(trials)
    pr._gather(every=True)                                      # every column: the full products
    pr.fused = False                                            # vpd_model's own norms and GELU
    pr._prefixes = {}
    full = pr.kl_sets(trials)
    assert close(full, gathered)


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
def test_connection_runs_are_the_scorecards():
    nat, pr = pruner(True)
    ge, tx = pr.ge, pr.tx
    at = [i for i, (_, t, _) in enumerate(pr.nodes) if t in tx.targets]
    pick = torch.randperm(len(at), generator=torch.Generator().manual_seed(2))[:24].tolist()
    nodes = [pr.nodes[at[i]] for i in pick]
    ep = teacher.EdgePruner(ge, tx, nodes, live=True, B=8)
    half = torch.rand(len(ep.edges), generator=torch.Generator().manual_seed(3)).to(nat.dev) < 0.5
    every, some, none = ep.kl_sets(torch.stack([ep.on, half, ~ep.on]))
    stated = [e for e, keep in zip(ep.edges, half.tolist()) if keep]
    parents = {}
    for w, r in stated:
        if r is not None:
            parents.setdefault(r, []).append(w)
    graphs = [native.Graph(nodes, parents, [w for w, r in stated if r is None]), native.Graph(nodes)]
    masked = carry.kl(tx.ref, ge.runs(tx.ids, tx.targets, [ge.masks(native.Graph(nodes), tx.T)], True)).item()
    # every connection stated is the circuit's masked run; any subset is the scorecard's run of that graph (graph_ref)
    assert close([every], [masked])
    assert close([some, none], ge.graph_kl(tx, graphs, True))


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
def test_from_path_reproduces_the_pick_and_its_connections():
    if not VPD_SAVED.exists():
        pytest.skip("needs VPD's answers " + str(VPD_SAVED))
    ge = goal_eval.GoalEval(native.Native(DEVICE), chunk=32)
    ids, targets = native.text(native.TEXTS / "vpd4l" / "text9019.json")   # a short path: 15 subcomponents at the start
    saved = torch.load(VPD_SAVED)["text9019"]
    full, row, path_file = teacher.teach(ge, ids, targets, 32, saved["answer"], saved["order"], edges=True)
    again, row_again, _ = teacher.teach(ge, ids, targets, 32, saved["answer"], saved["order"], edges=True,
                                        from_path=json.loads(json.dumps(path_file)))
    assert row_again["pick"] == row["pick"] and row_again["best"]["size"] == row["best"]["size"]
    assert again[""] == full[""] and again["f1"] == full["f1"] and again[f"vpd_top{row['top_k']}"] == full[f"vpd_top{row['top_k']}"]
    chars = len(teacher.components.render(full[""]))         # an answer budget below the pick's: a smaller point
    _, row_small, _ = teacher.teach(ge, ids, targets, 32, saved["answer"], saved["order"], edges=True,
                                    from_path=json.loads(json.dumps(path_file)), tokens=len, max_tokens=chars - 1)
    assert row_small["answer_tokens"] <= chars - 1 and row_small["answer_nodes"] < row["best"]["size"]


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
def test_repair_writes_its_pick():
    if not VPD_SAVED.exists() or not teacher.POOL.exists():
        pytest.skip("needs VPD's answers and the adversary's pool " + str(teacher.POOL))
    ge = goal_eval.GoalEval(native.Native(DEVICE), chunk=32)
    ids, targets = native.text(native.TEXTS / "vpd4l" / "text9019.json")
    saved = torch.load(VPD_SAVED)["text9019"]
    circuits, row, _ = teacher.teach(ge, ids, targets, 32, saved["answer"], saved["order"], edges=True, repair_pool=torch.load(teacher.POOL))
    steps, j = row["repair"], row["repair_pick"]
    ok = [i for i, s in enumerate(steps) if s["f1_rob"] >= teacher.RECOVERED and s["n"] >= teacher.RECOVERED]
    nodes = [nd for nd in circuits[""]["nodes"] if nd[0] != "tok"]
    # the written circuit is the repair's pick: the smallest step meeting the goal, else the largest value; each step
    # one more round of the omitted lever's flips, taken while the value rose
    assert j == (min(ok, key=lambda i: steps[i]["size"]) if ok else max(range(len(steps)), key=lambda i: (steps[i]["value"], -steps[i]["size"])))
    assert len(nodes) == steps[j]["size"] and len(circuits[""]["edges"]) == steps[j]["edges"]
    assert all(b["size"] > a["size"] and a["value"] < b["value"] for a, b in zip(steps[:-2], steps[1:-1]))


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
def test_robust_connections_keep_their_floor():
    if not VPD_SAVED.exists() or not teacher.POOL.exists():
        pytest.skip("needs VPD's answers and the adversary's pool " + str(teacher.POOL))
    ge = goal_eval.GoalEval(native.Native(DEVICE), chunk=32)
    ids, targets = native.text(native.TEXTS / "vpd4l" / "text9019.json")
    saved = torch.load(VPD_SAVED)["text9019"]
    _, row, path_file = teacher.teach(ge, ids, targets, 32, saved["answer"], saved["order"], edges=True,
                                      robust_pool=torch.load(teacher.POOL))
    steps = path_file["edge_paths"][0]["steps"]
    floor = min(teacher.RECOVERED, steps[0]["f1_flips"])
    # every removal kept F1 under the flips at the floor, and the summary carries the last one's
    assert all(s["f1_flips"] >= floor for s in steps[1:]) and row["best_edges"]["f1_flips"] == steps[-1]["f1_flips"]
