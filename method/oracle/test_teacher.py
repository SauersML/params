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
    # prefixes 1, 2, 4, 5 are within it, 4 and 5 hold the guard, so round 1 admits 2 (units 0, 1), and of the units
    # between the admitted prefix and the smallest refused one (2, 3) the guard is refused alone and stays for the
    # level; round 2 ranks without it and admits 2 (units 2, 4), round 3 1 (unit 5); then no removal is within the
    # budget. Without the check, round 1 would take 5 (units 0-4, the guard among them).
    guard = 3
    w = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.5, 0.6, 0.7, 0.8]

    def accept(ons):
        return [{"guard": True} if bool(on[guard]) else None for on in ons]
    toy, log = Toy(w), []
    toy.prune("unit", 0.4, log=log, accept=accept)
    assert [x["removed"] for x in log] == [[0, 1], [2, 4], [5]] and all(x["guard"] for x in log)
    assert toy.on.tolist() == [False, False, False, True, False, False] + [True] * 4
    # a guard the KL ranks first (its removal lowers KL) is frozen after the first refusal and the descent goes on
    first = Toy([-0.05] + w[1:])
    log = []
    first.prune("unit", 0.4, log=log, accept=lambda ons: [{} if bool(on[0]) else None for on in ons])
    assert bool(first.on[0]) and [x["removed"] for x in log][:1] == [[1, 2, 3, 4, 5]]
    free = Toy(w)
    free.prune("unit", 0.4, rounds=1)
    assert free.on.tolist() == [False] * 5 + [True] * 5


@pytest.mark.parametrize("rank", ["every", "once"])
def test_guards_ranked_first_freeze_in_one_round(rank, monkeypatch):
    # three guards the KL ranks first (their removals lower KL), each refused alone: every prefix holds the first, so the
    # first round admits nothing and asks the first 4 ACCEPT_BATCH blocks alone, freezing all three at once (asked only
    # up to the smallest refused prefix, one block, it froze one guard a round); the second round removes
    monkeypatch.setattr(teacher, "RANK", rank)
    w = [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.5, 0.6]
    toy, log = Toy(w), []
    toy.prune("unit", 0.4, rounds=2, log=log, accept=lambda ons: [{} if bool(on[:3].all()) else None for on in ons])
    assert log and all(bool(toy.on[i]) for i in range(3)) and not bool(toy.on[3])


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
    run directly at connect's budget (the F1 budget or 5% of the node set's own log recovery, 92b074f; Connector.connect
    before the flag); writer blocks first end no farther from the model than the start or that budget, and measure
    fewer subsets."""
    nat, pr = pruner(teacher.CONVENTION == "live")
    ge, tx = pr.ge, pr.tx
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    on[(pr.t == max(tx.targets)).nonzero().flatten()[:40]] = True
    plain = teacher.Connector(ge, tx, pr, empty, vpd, lambda on, inner=0, out=0: 0.0, 32)
    kept, summary, steps, ep = plain.connect(on, 0.0)[:4]
    direct = teacher.EdgePruner(ge, tx, [pr.nodes[j] for j in on.nonzero().flatten().tolist()], teacher.CONVENTION == "live", 32)
    teacher.edge_path(direct, summary["budget"], lambda kl, e_on, removed: {"kl": kl})
    if DEVICE == "cuda":  # (on MPS the two searches' rankings part: 15 connections kept against 10 on text9000)
        assert kept == [e for e, k in zip(direct.edges, direct.on.tolist()) if k]
    coarse = teacher.Connector(ge, tx, pr, empty, vpd, lambda on, inner=0, out=0: 0.0, 32, levels=("writer", "connection"))
    _, s2, steps2, ep2 = coarse.connect(on, 0.0)[:4]
    assert s2["kl_graph"] <= max(s2["budget"], steps2[0]["kl_graph"]) * (1 + REL)
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
    acc = teacher.Acceptance(ge, tx, ("changed",), "c")
    hook, base = acc.hook(lambda on: native.Graph([pr.nodes[j] for j in on.nonzero().flatten().tolist()], complete=True), start)
    path = teacher.joint_path(pr, start, empty, vpd, hook, base)
    floor = teacher.floor_of(base["s_track"])
    # every step the path took kept the circuit's change in prediction on the changed texts at its floor
    assert floor is not None and len(path) > 1
    assert all(s["s_track"] is None or s["s_track"] >= floor for s in path[1:])


def vpd_top(pr, nat, k=61, name="text9000"):
    index = {nd: i for i, nd in enumerate(pr.nodes)}
    order = [index[(n, int(t), int(c))] for n, t, c in torch.load(VPD_SAVED)[name]["order"] if (n, int(t), int(c)) in index]
    start = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    start[order[:k]] = True                                     # VPD's top 61, the teacher's top-k on this text
    return start, order[:k]


@pytest.mark.parametrize("kind", ["donors", "pp", "necessity"])
def test_acceptance_admits_exactly_the_passing_sets(kind):
    nat, pr = pruner(teacher.CONVENTION == "live")
    ge, tx = pr.ge, pr.tx
    start, order = vpd_top(pr, nat)
    as_nodes = lambda on: native.Graph([pr.nodes[j] for j in on.nonzero().flatten().tolist()], complete=True)  # noqa: E731
    acc = teacher.Acceptance(ge, tx, ("changed",) if kind == "pp" else (kind,), "pp")
    hook, base = acc.hook(as_nodes, start)
    assert set(base) == {"donors": {"s_patching_near", "s_patching_far"}, "pp": {"s_track_pp"}, "necessity": {"n_spec"}}[kind]
    ons = start[None].repeat(4, 1)
    for i, k in enumerate((1, 4, 16, 40)):                      # VPD's last k removed
        ons[i, order[-k:]] = False
    got = hook(ons)
    graphs = [as_nodes(on) for on in ons]
    each = {"donors": lambda: ge.donors_search(tx, graphs), "pp": lambda: ge.changed_search(tx, graphs, mode="pp"),
            "necessity": lambda: ge.necessity_spec(tx, graphs, split="search")}[kind]()
    for g, r in zip(got, each):
        passes = all(base[k] is None or r[k] is None or r[k] >= teacher.floor_of(base[k]) for k in base)
        # a set is admitted exactly when its terms keep their floors, with the family's own numbers
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
    for mode in ("c", "pp"):
        acc = teacher.Acceptance(ge, tx, ("changed",), mode)
        key = teacher.TRACK[mode]
        one = acc.scores([graph_of(e) for e in ons], (ep.graph, ons))
        each = acc.scores([graph_of(e) for e in ons])
        # changed_edges' one plan at 0/1 strengths gives changed_search's term on the subsets' graphs (token edges stated)
        assert all(a[key] is not None and abs(a[key] - b[key]) <= 1e-3 for a, b in zip(one, each)), (mode, one, each)
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
def test_pairs_stated_and_trimmed():
    import qk

    nat, pr = pruner(teacher.CONVENTION == "live", "text9002")
    ge, tx = pr.ge, pr.tx
    task = json.loads((TEXT.parent / "text9002.json").read_text())
    teacher_g, _ = goal_eval.circuit(json.loads(PAIRS_CIRCUIT.read_text()), tx.targets, nat.C, tx.T)
    g0 = teacher.with_tokens(goal_eval.strand(teacher_g, tx.targets, nat.C), tx.targets)
    got = qk.stated_pairs(ge, task, goal_eval.to_json(g0, ""), "c", "search")
    cur = {read: [(None if a == "rem" else a, None if b == "rem" else b) for a, b in ps if (a, b) != (None, None)]
           for read, ps in got.items()}
    stated = goal_eval.strand(teacher.with_pairs(g0, cur), tx.targets, nat.C)
    # rustmove's selection states the read L2@41's pairs; written as circuit JSON, goal_eval reads the pairs back with
    # every token edge, every node and edge valid
    assert sum(map(len, cur.values())) > 0
    conn = teacher.Connector(ge, tx, pr, *ge._bounds(tx, teacher.CONVENTION), lambda *a: 0.0)
    on = torch.zeros(len(pr.nodes), dtype=torch.bool, device=nat.dev)
    on[torch.tensor([conn.index[nd] for nd in stated.node_set()], device=on.device)] = True
    kept = [(w, r) for r, ws in stated.parents.items() for w in ws] + [(w, None) for w in stated.out]
    back, bad = goal_eval.circuit(teacher.circuit_json(pr, on, kept, stated.pairs), tx.targets, nat.C, tx.T)
    assert bad == {"invalid_nodes": 0, "invalid_edges": 0, "stranded_nodes": 0, "invalid_variables": 0}
    assert {read: set(ps) for read, ps in getattr(back, "pairs", {}).items()} == {read: set(ps) for read, ps in stated.pairs.items()}
    assert set(back.tok_edges) == set(teacher.with_tokens(stated, tx.targets).tok_edges) and back.tok_out == sorted(tx.targets)
    # over an answer budget the last-chosen pairs go first; with none left the reads' entries go too
    order = [(read, p_) for _, read, p_ in sorted((-i, read, p_) for read, ps in stated.pairs.items() for i, p_ in enumerate(ps))]
    h, ps, dropped = teacher.trim_pairs(g0, stated, order, lambda h, ps: sum(map(len, ps.values())) <= 3, tx.targets, nat.C)
    first3 = {p_ for read, pl in stated.pairs.items() for p_ in pl[:3]} if len(stated.pairs) == 1 else None
    assert sum(map(len, ps.values())) <= 3 and (first3 is None or set(sum(ps.values(), [])) <= first3)
    assert teacher.trim_pairs(g0, stated, order, lambda h, ps: False, tx.targets, nat.C)[:2] == (g0, None)


def test_connect_with_acceptance_keeps_its_floor():
    nat, pr = pruner(teacher.CONVENTION == "live")
    ge, tx = pr.ge, pr.tx
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    start, _ = vpd_top(pr, nat)
    conn = teacher.Connector(ge, tx, pr, empty, vpd, lambda *a: 0.0, no_strand=True,
                             acceptance=teacher.Acceptance(ge, tx, ("changed",), "c"))
    n = teacher.measure(pr, start[None], empty, vpd)[0]["n"]
    kept, summary, steps, _, _ = conn.connect(start, n)
    floor = teacher.floor_of(steps[0]["s_track"])
    # the connection descent without the robust objective reports the last admitted step's s_track, at its floor
    assert "f1_flips" not in summary and summary["s_track"] >= floor and len(kept) == summary["edges"]
    assert all(s_.get("s_track") is None or s_["s_track"] >= floor for s_ in steps)


def test_whole_answer_search_joins_only_below_the_floor():
    if not VPD_SAVED.exists():
        pytest.skip("needs VPD's answers " + str(VPD_SAVED))
    ge = goal_eval.GoalEval(native.Native(DEVICE), chunk=32)
    ids, targets = native.text(native.TEXTS / "vpd4l" / "text9019.json")
    saved = torch.load(VPD_SAVED)["text9019"]
    _, row, _ = teacher.teach(ge, ids, targets, 32, saved["answer"], saved["order"], f1_start="both", accept=("changed",),
                              accept_mode="c")
    t, w = row["accept_targets"], row["accept_whole"]
    # with acceptance, the whole answer's F1 search runs (and its circuit joins the target one's) only when the target
    # circuit misses min(0.9, the whole answer's) on a term
    below = any(w[k] is not None and t[k] is not None and t[k] < min(teacher.RECOVERED, w[k]) for k in w)
    assert row["f1_whole_search"] == below
    assert ("f1_sizes" in row) == below and ("position" in row["levels"]) == below


def test_block_search_keeps_its_floor():
    if not VPD_SAVED.exists():
        pytest.skip("needs VPD's answers " + str(VPD_SAVED))
    nat, pr = pruner(teacher.CONVENTION == "live", "text9019")
    ge, tx = pr.ge, pr.tx
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    budget = empty ** (1 - teacher.RECOVERED) * vpd ** teacher.RECOVERED
    acc = teacher.Acceptance(ge, tx, ("changed",), "pp")
    g, pb, group, flagged, rec = teacher.block_search(ge, tx, empty, vpd, budget, acc, levels=("inactive",))
    S = len(nat.names)
    # the influential positions: the smallest m whose blocks reach s_track_pp 0.9, or every F2 position
    assert rec["m"] == next((m for m, x in enumerate(rec["tried"]) if x is not None and x >= teacher.RECOVERED), len(rec["tried"]) - 1)
    # every stage kept the floors set at the start
    assert all(rec["accept"][k] is None or f is None or rec["accept"][k] >= f - 1e-6 for k, f in rec["floor"].items())
    # a block node exactly where a kept group still holds subcomponents; refined groups as their subcomponents
    blocks = {(nd[0], nd[1]) for nd in g.node_set() if nd[2] == goal_eval.BLOCK}
    live = {int(k) for k in group[pb.on].tolist()}
    assert blocks == {(pb.names[k % S], k // S) for k in live if bool(flagged[k])}
    assert all(nd[2] == goal_eval.BLOCK or (nd[0], nd[1]) not in blocks for nd in g.node_set())
    # written as a node set, goal_eval reads the same circuit back
    back, bad = goal_eval.circuit(goal_eval.to_json(g, ""), tx.targets, nat.C, tx.T)
    assert back.complete and set(back.node_set()) == set(g.node_set()) and bad["invalid_nodes"] == 0


@pytest.mark.skipif(DEVICE == "cpu", reason="needs a GPU")
def test_block_connections_keep_their_floors():
    if not VPD_SAVED.exists():
        pytest.skip("needs VPD's answers " + str(VPD_SAVED))
    nat, pr = pruner(teacher.CONVENTION == "live", "text9019")
    ge, tx = pr.ge, pr.tx
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    budget = empty ** (1 - teacher.RECOVERED) * vpd ** teacher.RECOVERED
    acc = teacher.Acceptance(ge, tx, ("changed", "donors"), "pp")
    g, pb, _, _, rec = teacher.block_search(ge, tx, empty, vpd, budget, acc)
    bl = {(nd[0], nd[1]) for nd in g.node_set() if nd[2] == goal_eval.BLOCK}
    conn = teacher.Connector(ge, tx, pb, empty, vpd, lambda *a: 0.0, no_strand=True, acceptance=acc, blocks=bl)
    kept, summary, steps, ep, on = conn.connect(pb.on.clone(), 0.0)
    every = conn.graph(on, kept)
    stated = conn.graph(on, kept, summary["tokens"])
    # the token edges kept are some of every one (pruned under the same floors)
    assert set(stated.tok_edges) <= set(every.tok_edges) and set(stated.tok_out) <= set(every.tok_out)
    # a block's connections are one unit: every subcomponent's kept or none
    for u in range(int(ep.gid.max()) + 1):
        member = ep.gid == u
        assert bool(ep.on[member].all()) or not bool(ep.on[member].any())
    # the stated graph names the blocks, states every token edge, and keeps the floors the descent started from
    assert {(nd[0], nd[1]) for nd in stated.node_set() if nd[2] == goal_eval.BLOCK} <= bl
    floor = {k: teacher.floor_of(steps[0][k]) for k in ("s_track_pp", "s_patching_near", "s_patching_far") if steps[0].get(k) is not None}
    got = acc.scores([stated])[0]
    assert all(got[k] is None or got[k] >= f - 1e-6 for k, f in floor.items())
    back, bad = goal_eval.circuit(goal_eval.to_json(stated, ""), tx.targets, nat.C, tx.T)
    assert bad["invalid_nodes"] == 0 and bad["invalid_edges"] == 0 and set(back.node_set()) == set(stated.node_set())


def test_kl_budget_takes_a_kl_rounded_below_zero():
    # an exact answer's KL can round below 0 (fleet's planted toys): the budget stays a small positive real
    b = teacher.kl_budget(80.0, -3e-8)
    assert isinstance(b, float) and 0 < b < 1e-6
    assert abs(teacher.kl_budget(80.0, 0.05) - 80.0 ** 0.1 * 0.05 ** 0.9) < 1e-12


@pytest.mark.parametrize("draws", ["discriminating", "fresh"])
def test_discriminating_draws_stay_off_eval(draws):
    nat, pr = pruner(True, "text9002")
    ge = pr.ge
    ids, targets = native.text(TEXT.parent / "text9002.json")
    tx = ge.text(ids, targets, native.task_seed("text9002"))   # the scorer's seed
    tx.vpd = pr.tx.vpd
    pool = teacher.search_pool(ge, tx)
    positions = [p for p, _ in tx.f2]
    moved = [-float(tx.moved[p - 1]) for p in positions]
    held = set(ge._interchange_draws(tx, positions, "eval", 64, first=dict(zip(positions, moved))))
    held |= set(ge._interchange_draws(tx, positions, "eval", goal_eval.INTERCHANGE_DRAWS))
    # the pool holds the search split's draws and never an eval draw
    assert set(ge._interchange_draws(tx, positions, "search", 16)) <= set(pool) and not (set(pool) & held)
    assert len(pool) >= teacher.POOL_DRAWS
    pr2 = teacher.Pruner(ge, tx, tx.vpd, live=True, B=32)
    as_nodes = lambda on: native.Graph([pr2.nodes[j] for j in on.nonzero().flatten().tolist()], complete=True)  # noqa: E731
    acc = teacher.Acceptance(ge, tx, ("changed",), "pp", draws)
    hook, base = acc.hook(as_nodes, pr2.on.clone(), current=lambda: pr2.on, label="t", names=lambda ids: ids)
    g = torch.Generator().manual_seed(0)
    ons = pr2.on[None].repeat(6, 1)
    for i, f in enumerate((0.02, 0.05, 0.1, 0.3, 0.6, 0.9)):     # random removals, light to heavy
        ons[i, torch.rand(len(pr2.nodes), generator=g).to(ons.device) < f] = False
    got = hook(ons)
    floor = teacher.floor_of(base["s_track_pp"])
    # a check's sample is keyed by the state it judges: the same set draws the same sample, another set another one
    assert acc._sample(pr2.on.clone()) == acc._sample(pr2.on) != acc._sample(ons[0])
    # every check holds the model's likeliest alternatives (the search split's ranks 0 and 2)
    assert set(acc.fixed) == set(ge._interchange_draws(tx, positions, "search", 64)) <= set(acc._sample(ons[1]))
    # admitted removals keep the floor on their judged draws; each refusal kept with its 16 draws from the pool
    assert all(r is None or r["s_track_pp"] >= floor for r in got)
    assert len(acc.refused) == sum(r is None for r in got)
    most = teacher.DISC_DRAWS if draws == "discriminating" else teacher.POOL_DRAWS
    assert all(len(x["draws"]) <= most and {tuple(d) for d in x["draws"]} <= set(pool) and x["s_track_pp"] < floor
               for x in acc.refused)


@pytest.mark.skipif(DEVICE == "cpu", reason="needs a GPU")
def test_block_search_phi_check_backs_off():
    if not VPD_SAVED.exists() or not teacher.POOL.exists():
        pytest.skip("needs VPD's answers and the adversary's pool")
    nat, pr = pruner(teacher.CONVENTION == "live", "text9019")
    ge, tx = pr.ge, pr.tx
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    acc = teacher.Acceptance(ge, tx, ("changed",), "pp", "fresh")
    g, pb, group, flagged, rec = teacher.block_search(ge, tx, empty, vpd, teacher.kl_budget(empty, vpd), acc,
                                                      levels=("inactive", "chunk64"), phi_pool=torch.load(teacher.POOL))
    tried = rec["phi"]
    # stages checked from the last back: stopped at the first that reaches the reward's bar, else every one tried and
    # the best kept
    passed = [x for x in tried if x[1] is not None and x[1] >= x[2]]
    assert (passed and passed[0] is tried[-1] and rec["phi_stage"] == tried[-1][0]) or \
        (not passed and rec["phi_stage"] == max((x for x in tried if x[1] is not None), key=lambda x: x[1])[0])


def test_ranked_draws_are_pooled():
    # fv: the likeliest alternatives (rank 0 for acceptance, rank 2 for validation) pooled with the sampled draws, as
    # the reward pools track_pp (the lead's 10-14): one alternative at 0 no longer refuses a removal the pool passes
    acc = teacher.Acceptance(None, None, ("changed",), "pp", "fv")
    acc.fixed, acc.r2 = [(3, 11)], [(3, 12)]

    def rec(pz, share, gap=1.0):                                # [prompt, KL_C, KL_VPD, W]: the circuit's share of
        return [list(pz), gap * (1 - share), 0.0, gap]           # VPD's gap W - KL_VPD
    prompts = [(3, 11)] + [(5, 7 + i) for i in range(9)]
    recs = [rec((3, 11), 0.0)] + [rec(pz, 1.0) for pz in prompts[1:]]
    sc, low = acc.ranked(recs, prompts, range(10))
    assert sc == pytest.approx(0.9) and low == []                # 9 of 10 draws' gaps: the pool passes floor_of(1.0)
    assert sc >= teacher.floor_of(1.0)
    acc.draws = "fresh"                                          # the same pool under the other draw modes
    assert acc.ranked(recs, prompts, range(10)) == (pytest.approx(0.9), [])
    # the changed-input acceptance's floors: 0.05 below the start's value, in [0, 0.9] (the reward's are VPD's)
    assert teacher.floor_of(0.5) == pytest.approx(0.45) and teacher.floor_of(2.0) == 0.9 and teacher.floor_of(0.02) == 0.0
    assert goal_eval.floors({"s_point_c": 0.8}, ["point"])["point"] == pytest.approx(0.72)


def test_validation_guards_scope_on_its_held_out_half(monkeypatch):
    # fv with the floors kind: the input-scope family on scope_draws' validation half below floor_of the START's there
    # fails validation (low names it), above it passes (the lead's 10-14: text9012's search-pruned circuit, eval scope 0.33)
    from types import SimpleNamespace
    seen = []

    def scope_family(tx, graphs, split, part=None):
        seen.append((split, part))
        return [{"s_scope_raw": got[0], "s_scope": got[0], "draws_scope": []}]
    got = [0.5]
    acc = teacher.Acceptance(SimpleNamespace(scope_family=scope_family), None, ("changed", "floors"), "pp", "fv")
    monkeypatch.setattr(acc, "_halves", lambda: ([], []))
    acc.scope_floor = teacher.floor_of(0.95)
    assert acc.validation(None) == (None, [["scope", 0.5, acc.scope_floor]]) and seen == [("search", "validate")]
    got[0] = 0.92
    assert acc.validation(None) == (None, [])
    acc.scope_floor = None                                       # (no guard without the floors kind)
    assert acc.validation(None) == (None, [])


def test_validated_keeps_each_passing_removal_by_default():
    # every removal checked (VAL_EVERY 1): units 0 and 1 go (validation 0.8, then 0.6 < 0.7 on the second), so the level
    # keeps the first removal and ends there, not reverting it with the failing one
    assert teacher.VAL_EVERY == 1
    toy, log = Toy([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]), []

    class Val:
        draws = "fv"

        def validation(self, on):
            return float(on[:5].float().mean()), []
    v = teacher.Validated(toy, lambda on: on, Val(), 0.7, log)
    for _ in range(8):
        if toy.prune("unit", 0.115, rounds=1, log=log, after=v):
            break
    assert toy.on.tolist() == [False] + [True] * 7 and len(log) == 1


def test_validated_backs_off_to_the_last_passing_set():
    # every removal admitted; the validation score is the share of units 0-4 still on: below 0.7 once two of them go
    toy, log = Toy([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]), []

    class Val:
        draws = "fv"

        def validation(self, on):
            return float(on[:5].float().mean()), []
    v = teacher.Validated(toy, lambda on: on, Val(), 0.7, log, every=1)
    toy.prune("unit", 10.0, rounds=1, log=log, after=v)          # one round: the largest prefix, units 0-6, refused
    assert toy.on.all() and log == [] and v.trace[-1][2] < 0.7
    toy2, log2 = Toy([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]), []
    v2 = teacher.Validated(toy2, lambda on: on, Val(), 0.7, log2, every=1)
    for _ in range(8):                                          # single removals in KL order: 0 passes (0.8), 1 not
        if toy2.prune("unit", 0.115, rounds=1, log=log2, after=v2):
            break
    assert toy2.on.tolist() == [False] + [True] * 7 and len(log2) == 1


def test_set_search_validation_holds_its_floor():
    nat, pr = pruner(True, "text9002")
    ge = pr.ge
    ids, targets = native.text(TEXT.parent / "text9002.json")
    tx = ge.text(ids, targets, native.task_seed("text9002"))
    tx.vpd = pr.tx.vpd
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    acc = teacher.Acceptance(ge, tx, ("changed",), "pp", "fv")
    g, ps, _, _, rec = teacher.set_search(ge, tx, empty, vpd, teacher.kl_budget(empty, vpd), acc, levels=("site", "chunk4"))
    pool, val = acc._halves()
    # the halves are disjoint, and the circuit that comes out holds the validation floor (each level backs off to the
    # last set that passed)
    assert not (set(pool) & set(val)) and val
    f = rec["floor"]["s_track_pp"]
    assert rec["validation_end"] is None or rec["validation_end"] >= f - 1e-6
    assert g.count() == int(ps.on.sum()) <= rec["start_units"]


@pytest.mark.skipif(DEVICE == "cpu", reason="needs a GPU")
def test_set_search_writes_the_highest_scoring_stage():
    if not teacher.POOL.exists():
        pytest.skip("needs the adversary's pool")
    nat, pr = pruner(True, "text9002")
    ge = pr.ge
    ids, targets = native.text(TEXT.parent / "text9002.json")
    tx = ge.text(ids, targets, native.task_seed("text9002"))
    tx.vpd = pr.tx.vpd
    empty, vpd = ge._bounds(tx, teacher.CONVENTION)
    acc = teacher.Acceptance(ge, tx, ("changed",), "pp", "fv")
    calls, term = [], goal_eval.circuit_term

    def recorded(row, tau, bits, **kw):                         # the reward's own circuit term, each call kept
        calls.append((tau, bits, term(row, tau, bits, **kw)))
        return calls[-1][2]
    goal_eval.circuit_term = recorded
    try:
        g, ps, _, _, rec = teacher.set_search(ge, tx, empty, vpd, teacher.kl_budget(empty, vpd), acc, levels=("site", "chunk4"),
                                              phi_pool=torch.load(teacher.POOL))
    finally:
        goal_eval.circuit_term = term
    tau = ge.tau(tx, None, "search")
    # every stage scored by goal_eval.circuit_term against the text's tau with its own bits, the written one the highest
    assert [x[0] for x in rec["phi"]] == ["chunk4", "site", "start"] and len(calls) == 4  # the stages, the written
    for (name, ph, bar, n, fam, bits, score, v), (t, b, ct) in zip(rec["phi"], calls[:3]):
        assert t is tau and b == bits and score == ct["term"] and v == ct["v"] and bar == tau["phi_tau"]
    # written: the highest among the stages with no family under its floor on their full row (v = 0), else the one of
    # the lowest shortfall, then the smaller; every stage's node set kept for re-picks
    ok = [x for x in rec["phi"] if x[7] == 0]
    best = max(ok, key=lambda x: x[6]) if ok else min(reversed(rec["phi"]), key=lambda x: (x[7], x[3]))
    assert rec["phi_stage"] == best[0] and {k: len(v) for k, v in rec["stage_nodes"].items()} == {x[0]: x[3] for x in rec["phi"]}
    # then own_rule's numbers per context position (a diagnostic): the written circuit's, and the START's positions
    # pruned away; the final verdict is the written circuit's
    held = {t for _, t, _ in g.node_set()}
    assert g.count() == rec["final"]["units"] and rec["score"] == best[6] and g.count() == best[3]
    assert not held & set(rec["own_dropped"]) and all(rec["positions_own"][str(q)].get("dropped") for q in rec["own_dropped"])
    assert calls[-1][2]["v"] == rec["final"]["v"] and rec["final"]["fails_floors"] == (rec["final"]["v"] > 0)
    assert rec["final"]["floors"] == tau["floors"] and rec["final"]["vpd_scores"] == tau["vpd_scores"]
    assert rec["start"]["cap_bound"] == (rec["m"] == len(rec["start"]["candidates"]))
    assert set(rec["positions_own"]) == {str(t) for t in held - set(tx.targets)} | {str(q) for q in rec["own_dropped"]}


@pytest.mark.skipif(DEVICE == "cpu", reason="needs a GPU")
def test_forward_start_adds_positions_until_the_bar():
    if not teacher.POOL.exists():
        pytest.skip("needs the adversary's pool")
    nat, pr = pruner(True, "text9002")
    ge = pr.ge
    ids, targets = native.text(TEXT.parent / "text9002.json")
    tx = ge.text(ids, targets, native.task_seed("text9002"))
    tx.vpd = pr.tx.vpd
    pool = torch.load(teacher.POOL)
    got = teacher.forward_start(ge, tx, pool)
    tau = ge.tau(tx, pool, "search")
    # VPD's actives at the targets first, then one F2 position's actives a step, until phi reaches the bar or every
    # position is in; the set is VPD's actives at its positions
    act = {(n, int(t), int(c)) for n in nat.names for t, c in tx.vpd[n].nonzero().tolist()}
    assert got["bar"] == tau["phi_tau"] and set(got["nodes"]) == {nd for nd in act if nd[1] in set(got["positions"])}
    sizes = [s[1] for s in got["steps"]]
    assert sizes == sorted(sizes) and got["steps"][0][0] is None and sizes[-1] == len(got["nodes"])
    f2 = {int(q) for q, _ in tx.f2} - set(tx.targets)
    faithful = lambda ph, v: ph >= got["bar"] + teacher.PHI_SLACK and v == 0   # noqa: E731
    assert faithful(got["phi"], got["v"]) or set(got["positions"]) >= f2
    assert not any(faithful(ph, v) for _, _, ph, v, _ in got["steps"][:-1])


def test_family_acceptance_scores_as_fidelity():
    nat, pr = pruner(True, "text9002")
    ge = pr.ge
    ids, targets = native.text(TEXT.parent / "text9002.json")
    tx = ge.text(ids, targets, native.task_seed("text9002"))
    tx.vpd = pr.tx.vpd
    start, _ = vpd_top(pr, nat, 61, "text9002")
    g = native.Graph([pr.nodes[j] for j in start.nonzero().flatten().tolist()], complete=True)
    got = teacher.Acceptance(ge, tx, ("nodes", "edges", "pattern"), "pp").scores([g])[0]
    row = ge.fidelity(tx, [g], None, mode="reward", interchange=0, changed=0, near_donors=0, far_donors=0, controls=0)[0]
    # phi's families in acceptance are fidelity's own numbers
    for k in ("s_nodes", "s_edges", "s_pattern"):
        assert (got[k] is None) == (row.get(k) is None) and (got[k] is None or abs(got[k] - row[k]) < 1e-5), (k, got[k], row.get(k))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_storages():
    """memory_log's cache attribution (cuda_storages) counts each CUDA storage once (a view of a tensor shares its
    storage), through dicts, sequences and attributes, and leaves out CPU tensors, the storages already seen (the
    model's weights) and the model itself unless asked."""
    a = torch.zeros(1024, 256, device="cuda")
    o = type("O", (), {})()
    o.x, o.cpu = {"k": [a, (a[3], torch.ones(10, device="cuda"))]}, torch.zeros(5)
    seen = set()
    assert teacher.cuda_storages(o, seen) == a.numel() * 4 + 40
    assert teacher.cuda_storages({"again": a[1:]}, seen) == 0
    m = torch.nn.Linear(4, 4).cuda()
    assert teacher.cuda_storages([m], set()) == 0 and teacher.cuda_storages(m, set(), model=True) == (16 + 4) * 4
