"""Checks of teacher.py on a held-out vpd4l text, on this machine's GPU (skipped without one, vpd4l's texts, or VPD's
answer: the causal-importance network or teacher.py --save-vpd's file VPD_SAVED): the windowed masked runs
(Pruner.kl_sets on a device in Pruner.WINDOWS) equal the whole runs to the bit on MPS and within float32 rounding on
CUDA, and the products on the start set's subcomponents alone (Pruner._gather) with torch's fused norms and GELU
(Pruner.fused) match the products on all subcomponents with vpd_model's own steps within float32 rounding,
EdgePruner's runs are the scorecard's (goal_eval.GoalEval.graph_kl: with every connection the masked run), and a
teacher circuit rebuilt from its saved path (teacher.py --from-path) is the full run's, nodes and connections, one
over the answer budget steps to a smaller path point, the flip repair writes its best-scoring step, and Descent.prune's
accept check admits exactly the largest passing prefix (a constructed case), with --no-strand every listed node but a
query or key has a stated path to the prediction (routes() drops a query or key whose pattern feeds no routed attention
output), --edge-levels' default keeps the connections it kept before, and a round's ranking sees the same KLs on the GPU
and the CPU (float32: the orders differ only among connections tied to that precision), and the joint path under
--accept changed keeps s_track at its floor at every step.

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


def pruner(live: bool, name: str = "text9000"):
    nat = native.Native(DEVICE)
    ge = goal_eval.GoalEval(nat, chunk=32)
    ids, targets = native.text(TEXT.parent / f"{name}.json")
    tx = ge.text(ids, targets)
    if tx.vpd is None:
        if not VPD_SAVED.exists():
            pytest.skip("needs VPD's answer: the causal-importance network or " + str(VPD_SAVED))
        tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (t, c) in torch.load(VPD_SAVED)[name]["answer"].items():
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
    nodes = [nd for nd in circuits[""]["nodes"] if nd[0] != "tok"]
    # the written circuit is the step with the best reward circuit score; each step one more round of the omitted
    # lever's flips, taken while the score rose
    assert j == max(range(len(steps)), key=lambda i: (steps[i]["score"], -steps[i]["size"]))
    assert len(nodes) == steps[j]["size"] and len(circuits[""]["edges"]) == steps[j]["edges"]
    assert all(b["size"] > a["size"] and a["score"] < b["score"] for a, b in zip(steps[:-2], steps[1:-1]))




class Toy(teacher.Descent):
    """Units with fixed weights: a set's KL is 0.1 plus the weights of the units it lacks."""

    def __init__(self, w):
        self.w, self.on, self.B, self.measured = torch.tensor(w), torch.ones(len(w), dtype=torch.bool), 4, 0

    def kl_sets(self, ons):
        return (0.1 + ((~ons).float() * self.w).sum(1)).tolist()

    def blocks(self, level):
        return list(self.on.nonzero().flatten().split(1))


def test_accept_admits_exactly_the_passing_prefix():
    # ranked by the KL a removal leaves: units 0, 1, 2, 3, ...; unit 3 is a guard the accept check refuses to lose (as
    # the robust objective refuses a connection whose cut drops F1 under the flips below its floor). Budget 0.4: the
    # prefixes 1, 2, 4, 5 are within it, 4 and 5 hold the guard, so round 1 admits 2 (units 0, 1); round 2 admits 1
    # (unit 2), refusing 2 and 4; round 3 finds every prefix holding the guard and stops. Without the check, round 1
    # would take 5 (units 0-4, the guard among them).
    guard = 3
    w = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.5, 0.6, 0.7, 0.8]

    def accept(ons):
        return [{"guard": True} if bool(on[guard]) else None for on in ons]
    toy, log = Toy(w), []
    toy.prune("unit", 0.4, log=log, accept=accept)
    assert [x["removed"] for x in log] == [[0, 1], [2]] and all(x["guard"] for x in log)
    assert toy.on.tolist() == [False, False, False] + [True] * 7
    free = Toy(w)
    free.prune("unit", 0.4, rounds=1)
    assert free.on.tolist() == [False] * 5 + [True] * 5


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
def test_no_strand_leaves_no_node_unconnected():
    if not VPD_SAVED.exists():
        pytest.skip("needs VPD's answers " + str(VPD_SAVED))
    ge = goal_eval.GoalEval(native.Native(DEVICE), chunk=32)
    ids, targets = native.text(native.TEXTS / "vpd4l" / "text9000.json")
    saved = torch.load(VPD_SAVED)["text9000"]
    circuits, row, _ = teacher.teach(ge, ids, targets, 32, saved["answer"], saved["order"], edges=True, no_strand=True)
    c = circuits[""]
    qk = {i for i, (m, _, _) in enumerate(c["nodes"]) if m.endswith(("q_proj", "k_proj"))}
    reach = {w for w, r in c["edges"] if r == "out"}
    grew = True
    while grew:                                              # stated paths to the prediction; a query or key ends one
        grew = False                                         # only when an attention output of its layer reaches it
        for w, r in c["edges"]:
            if r in reach and w not in reach:
                reach.add(w)
                grew = True
        o = [(int(c["nodes"][i][0].split(".")[1]), c["nodes"][i][2]) for i in reach if c["nodes"][i][0].endswith("o_proj")]
        for i in qk - reach:
            m, _, t = c["nodes"][i]
            if any(l == int(m.split(".")[1]) and (p == t if m.endswith("q_proj") else p >= t) for l, p in o):
                reach.add(i)
                grew = True
    # every listed node but the token embeddings has a stated path to the prediction: a query or key through a routed
    # attention output of its layer
    assert all(i in reach for i, (m, _, _) in enumerate(c["nodes"]) if m != "tok")


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
def test_edge_levels():
    """--edge-levels: the default ("connection") keeps exactly the connections of edge_path's single-connection rounds
    run directly (Connector.connect before the flag); writer blocks first end no farther from the model than the
    start or the F1 budget, and measure fewer subsets."""
    nat, pr = pruner(teacher.CONVENTION == "live")
    ge, tx = pr.ge, pr.tx
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    on[(pr.t == max(tx.targets)).nonzero().flatten()[:40]] = True
    plain = teacher.Connector(ge, tx, pr, empty, vpd, lambda on, inner=0, out=0: 0.0, 32)
    kept, summary, steps, ep = plain.connect(on, 0.0)[:4]
    direct = teacher.EdgePruner(ge, tx, [pr.nodes[j] for j in on.nonzero().flatten().tolist()], teacher.CONVENTION == "live", 32)
    teacher.edge_path(direct, plain.budget, lambda kl, e_on, removed: {"kl": kl})
    assert kept == [e for e, k in zip(direct.edges, direct.on.tolist()) if k]
    coarse = teacher.Connector(ge, tx, pr, empty, vpd, lambda on, inner=0, out=0: 0.0, 32, levels=("writer", "connection"))
    _, s2, steps2, ep2 = coarse.connect(on, 0.0)[:4]
    assert s2["kl_graph"] <= max(plain.budget, steps2[0]["kl_graph"]) * (1 + REL)
    assert ep2.measured < ep.measured


def test_routes_drop_dead_queries_and_keys():
    o2, q2, k2 = ("h.2.attn.o_proj", 10, 0), ("h.2.attn.q_proj", 10, 1), ("h.2.attn.k_proj", 4, 5)
    k3, d2, d0 = ("h.3.attn.k_proj", 4, 2), ("h.2.mlp.down_proj", 4, 3), ("h.0.mlp.down_proj", 4, 6)
    # target 10: o2 writes into the prediction; q2 (layer 2 at 10) and k2 (layer 2 at 4 <= 10) feed o2's pattern, d0
    # writes into k2 (and k3); k3 has no attention output of its layer at or after 4, so its pattern feeds nothing, nor
    # does d2, whose only connection (written after layer 2's attention) is into k3
    assert teacher.routes([o2, q2, k2, k3, d2, d0], [10]) == {o2, q2, k2, d0}


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
def test_connection_rankings_agree_across_devices():
    singles = []
    for dev in (DEVICE, "cpu"):
        nat = native.Native(dev)
        ge = goal_eval.GoalEval(nat, chunk=32)
        ids, targets = native.text(TEXT)
        tx = ge.text(ids, targets)
        tx.vpd = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
        for n, (t, c) in torch.load(VPD_SAVED)["text9000"]["answer"].items():
            tx.vpd[n][t.long().to(nat.dev), c.long().to(nat.dev)] = True
        pr = teacher.Pruner(ge, tx, tx.vpd, live=teacher.CONVENTION == "live", B=32)
        ep = teacher.EdgePruner(ge, tx, [nd for nd in pr.nodes if nd[1] in tx.targets][:24], teacher.CONVENTION == "live", 8)
        singles.append(torch.tensor(ep.without(ep.blocks("connection"), ranking=True), dtype=torch.float64))
    # every ranking in float32: a round's single-removal KLs agree across devices within float32 rounding, so the two
    # orders differ only among connections whose KLs are equal to that precision (on text9000 55 of the 125 gaps
    # between sorted KLs are below 1e-4: many connections change nothing); bfloat16 rankings reordered whole rounds
    a, b = torch.sort(singles[0]).values, torch.sort(singles[1]).values
    assert close(a.tolist(), b.tolist())


@pytest.mark.skipif(not TEXT.exists() or DEVICE == "cpu", reason="needs vpd4l's texts and a GPU")
def test_changed_acceptance_keeps_its_floor():
    nat, pr = pruner(teacher.CONVENTION == "live")
    ge, tx = pr.ge, pr.tx
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    index = {nd: i for i, nd in enumerate(pr.nodes)}
    order = [index[(n, int(t), int(c))] for n, t, c in torch.load(VPD_SAVED)["text9000"]["order"] if (n, int(t), int(c)) in index]
    start = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    start[order[:61]] = True                                    # VPD's top 61, the teacher's top-k on this text
    acc = teacher.Acceptance(ge, tx, ("changed",))
    hook, base = acc.hook(lambda on: native.Graph([pr.nodes[j] for j in on.nonzero().flatten().tolist()], complete=True), start)
    path = teacher.joint_path(pr, start, empty, vpd, hook, base)
    floor = None if base["s_track"] is None else min(teacher.RECOVERED, base["s_track"])
    # every step the path took kept the circuit's change in prediction on the changed texts at its floor
    assert floor is not None and len(path) > 1
    assert all(s["s_track"] is None or s["s_track"] >= floor for s in path[1:])


def vpd_top(pr, nat, k=61, name="text9000"):
    index = {nd: i for i, nd in enumerate(pr.nodes)}
    order = [index[(n, int(t), int(c))] for n, t, c in torch.load(VPD_SAVED)[name]["order"] if (n, int(t), int(c)) in index]
    start = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    start[order[:k]] = True                                     # VPD's top 61, the teacher's top-k on this text
    return start, order[:k]


def test_donors_acceptance_admits_exactly_the_passing_sets():
    nat, pr = pruner(teacher.CONVENTION == "live")
    ge, tx = pr.ge, pr.tx
    start, order = vpd_top(pr, nat)
    as_nodes = lambda on: native.Graph([pr.nodes[j] for j in on.nonzero().flatten().tolist()], complete=True)  # noqa: E731
    acc = teacher.Acceptance(ge, tx, ("donors",))
    hook, base = acc.hook(as_nodes, start)
    assert set(base) == {"s_donor_near", "s_donor_far"}
    ons = start[None].repeat(4, 1)
    for i, k in enumerate((1, 4, 16, 40)):                      # VPD's last k removed
        ons[i, order[-k:]] = False
    got = hook(ons)
    each = ge.donors_search(tx, [as_nodes(on) for on in ons])
    for g, r in zip(got, each):
        passes = all(base[k] is None or r[k] is None or r[k] >= min(teacher.RECOVERED, base[k]) for k in base)
        # a set is admitted exactly when both donor terms keep their floors, with donors_search's own numbers
        assert (g is not None) == passes
        if g is not None:
            assert all(g[k] == r[k] or abs(g[k] - r[k]) <= 1e-4 for k in base)


def test_connection_acceptance_from_one_plan():
    nat, pr = pruner(teacher.CONVENTION == "live")
    ge, tx = pr.ge, pr.tx
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    start, _ = vpd_top(pr, nat)
    nodes = [pr.nodes[j] for j in start.nonzero().flatten().tolist()]
    nodes = [nd for nd in nodes if nd in teacher.routes(nodes, tx.targets)]
    ep = teacher.EdgePruner(ge, tx, nodes, teacher.CONVENTION == "live", 32)
    conn = teacher.Connector(ge, tx, pr, empty, vpd, lambda *a: 0.0)
    on = torch.zeros_like(start)
    on[torch.tensor([conn.index[nd] for nd in nodes], device=on.device)] = True
    graph_of = lambda e: conn.graph(on, [x for x, k_ in zip(ep.edges, e.tolist()) if k_])  # noqa: E731 (reads ep, on)
    ons = [ep.on.clone()]
    for j in range(len(ep.edges)):                             # single cuts that strand no node: one plan's subsets
        e = ep.on.clone()
        e[j] = False
        if not goal_eval.strand(graph_of(e), tx.targets, nat.C).stranded:
            ons.append(e)
        if len(ons) == 4:
            break
    ons = torch.stack(ons)
    assert ge._edge_subsets(tx, ep.graph, ons)[1] is None and len(ons) == 4
    acc = teacher.Acceptance(ge, tx, ("changed",))
    one = acc.scores([graph_of(e) for e in ons], (ep.graph, ons))
    each = acc.scores([graph_of(e) for e in ons])
    # changed_edges' one plan at 0/1 strengths gives changed_search's s_track on the subsets' graphs
    assert all(a["s_track"] is not None and abs(a["s_track"] - b["s_track"]) <= 1e-3 for a, b in zip(one, each))
    def dead(sub):                                              # sub's connected nodes with no route
        par, _ = native.all_edges(sub, tx.targets)
        reach = teacher.routes(sub, tx.targets)
        return [nd for nd in sub if nd not in reach and (par.get(nd) or any(nd in ws for ws in par.values()))]
    every = next(sub for sub in ([x for x in nodes if x != nd] for nd in nodes) if dead(sub))
    on = torch.zeros_like(start)                                # a node set whose graph strands a connected node itself:
    on[torch.tensor([conn.index[nd] for nd in every], device=on.device)] = True   # ons' columns are its connections,
    ep = teacher.EdgePruner(ge, tx, every, teacher.CONVENTION == "live", 32)       # more than its plan's
    ons = ep.on[None].repeat(3, 1)
    ons[1, len(ep.edges) // 2:] = False
    ons[2, :len(ep.edges) // 2] = False
    subs = ge._edge_subsets(tx, ep.graph, ons)[1]

    def stated(g):
        g = goal_eval.strand(g, tx.targets, nat.C)
        return sorted(g.node_set()), sorted((w, r) for r, ws in g.parents.items() for w in ws), sorted(g.out)
    # each subset read in ep's column order: the graph its row states, stranded nodes dropped
    assert subs is not None and [stated(g) for g in subs] == [stated(graph_of(e)) for e in ons]


PAIRS_CIRCUIT = Path.home() / "mpd-data/graph_oracle/teacher/joint_heldout_80fe906/text9002.json"


@pytest.mark.skipif(not PAIRS_CIRCUIT.exists(), reason="needs the 80fe906 teacher circuit of text9002")
def test_pairs_stated_within_budget():
    nat, pr = pruner(teacher.CONVENTION == "live", "text9002")
    ge, tx = pr.ge, pr.tx
    live = teacher.CONVENTION == "live"
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    teacher_g, _ = goal_eval.circuit(json.loads(PAIRS_CIRCUIT.read_text()), tx.targets, nat.C, tx.T)
    start, _ = vpd_top(pr, nat, 266, "text9002")
    nodes = [pr.nodes[j] for j in start.nonzero().flatten().tolist()]
    nodes = [nd for nd in nodes if nd in teacher.routes(nodes, tx.targets)]
    ep = teacher.EdgePruner(ge, tx, nodes, live, 32)
    conn = teacher.Connector(ge, tx, pr, empty, vpd, lambda *a: 0.0)
    on = torch.zeros_like(start)
    on[torch.tensor([conn.index[nd] for nd in nodes], device=on.device)] = True
    # the teacher circuit (its read L2@41 needs pairs: background 1.6 bits against a 0.66 budget) and VPD's top 266
    # with every connection (every read's background within the budget: its queries and keys stranded)
    for g, needs in ((teacher_g, True), (conn.graph(on, ep.edges), False)):
        stated, summary = teacher.state_pairs(ge, tx, g, empty, vpd)
        # every read with a query or key node states its pairs, and the graph stays within the budget
        assert summary["reads"] > 0 and summary["kl_pairs"] <= summary["budget"] * (1 + 1e-5)
        assert set(stated.pairs) == {(r["layer"], r["t"]) for r in summary["per_read"]}
        assert (summary["pairs"] > 0) == needs and (summary["stranded"] > 0) == (not needs)
        # the stated graph with its stranded queries and keys dropped runs as the last read's measurement
        assert abs(ge.graph_kl(tx, [stated], live)[0] - summary["kl_pairs"]) <= 1e-4 * max(1.0, summary["kl_pairs"])
        # written as circuit JSON, goal_eval reads the same pairs back, every node and edge valid, none stranded
        on_p = torch.zeros_like(on)
        on_p[torch.tensor([conn.index[nd] for nd in stated.node_set()], device=on.device)] = True
        kept = [(w, r) for r, ws in stated.parents.items() for w in ws] + [(w, None) for w in stated.out]
        back, bad = goal_eval.circuit(teacher.circuit_json(pr, on_p, kept, stated.pairs), tx.targets, nat.C, tx.T)
        assert bad == {"invalid_nodes": 0, "invalid_edges": 0, "stranded_nodes": 0, "invalid_variables": 0}
        assert {read: set(ps) for read, ps in getattr(back, "pairs", {}).items()} == {read: set(ps) for read, ps in stated.pairs.items()}
