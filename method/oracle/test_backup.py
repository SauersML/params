"""Checks of backup expansion (backup.py) on goal_eval's tiny random model: the search's runs are VPD's masked forward
pass (the model and VPD's answer less each draw, the circuit and its expansion less each draw), a block is its site's
every subcomponent at its position, the whole model as blocks agrees with the model under every removal, the screen's
block sums are first-order exact, and the greedy path and expand_circuit's answer are what they record.

  MPD_MEM_GIB=4 ~/mpd-data/venv/bin/python -B -m pytest method/oracle/test_backup.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import backup  # noqa: E402
import carry  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native(near=0.1)
    ge = goal_eval.GoalEval(nat, changes=4, ranks=(3, 4), share_floor=0.01, chunk=4)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 40, (12,), generator=g).tolist()
    tx = ge.text(ids, [11], seed=5)
    every = sorted(native.Graph(masks=tx.vpd).node_set())
    circuit = native.Graph([nd for nd in every if nd[1] >= 8][::2], complete=True)
    return nat, ge, tx, circuit


def _less(m, d):
    m = {n: x.clone() for n, x in m.items()}
    for n, t, c in d:
        m[n][t, c] = False
    return m


def test_score_is_the_masked_runs(tiny):
    """score's fields from VPD's masked forward pass (GoalEval.runs, live): the model and VPD's answer less each draw,
    the circuit with a block less each draw; s_search in s_nodes' pooled form, F1 goal_eval's."""
    nat, ge, tx, circuit = tiny
    bk = backup.Backup(ge, tx, circuit, draws=4, rows=3)
    block = (nat.names[4], 9)  # layer 0's MLP input at position 9
    got = bk.score([(frozenset({block}), frozenset())])[0]
    ones = {n: torch.ones(tx.T, nat.C[n], dtype=torch.bool) for n in nat.names}
    m = {n: torch.zeros(tx.T, nat.C[n], dtype=torch.bool) for n in nat.names}
    for n, t, c in circuit.node_set():
        m[n][t, c] = True
    m[block[0]][block[1]] = True
    lm = ge.runs(tx.ids, tx.targets, [_less(ones, d) for d in bk.drops], True)
    lv = ge.runs(tx.ids, tx.targets, [_less(tx.vpd, d) for d in bk.drops], True)
    lc = ge.runs(tx.ids, tx.targets, [m] + [_less(m, d) for d in bk.drops], True)
    W = carry.kl(lm, tx.ref.expand_as(lm)).tolist()
    draws = [[j, a, b, w] for j, (a, b, w) in enumerate(zip(carry.kl(lm, lc[1:]).tolist(), carry.kl(lm, lv).tolist(), W))]
    e0, v0 = ge._bounds(tx, "live")
    point = float(carry.kl(tx.ref, lc[:1]))
    want = {"s_search": goal_eval._pooled("weighted", draws, e0), "f1": goal_eval._fid(e0, point, v0), "kl_point": point}
    for k, v in want.items():
        assert v is not None and abs(got[k] - v) < 1e-5 + 1e-4 * abs(v), (k, got[k], v)
    # the draws remove the circuit's nodes, drawn apart from goal_eval's node family's (its evaluation draws)
    rng, gn = ge._rng(tx, circuit, "nodes"), sorted(circuit.node_set())
    theirs = [[nd for nd in gn if rng.random() < (j + 0.5) / 4] for j in range(4)]
    assert all(set(d) <= set(bk.nodes) for d in bk.drops) and [sorted(d) for d in bk.drops] != theirs
    # the expanded answer: one block node, the circuit's subcomponents at its site and position subsumed
    g = bk.graph(frozenset({block}))
    assert (block[0], block[1], goal_eval.BLOCK) in g.node_set()
    assert not any(nd[:2] == block and nd[2] != goal_eval.BLOCK for nd in g.node_set())
    ex = goal_eval.expand_blocks(g, nat.C).node_set()
    assert ex == {(n, t, c) for n in nat.names for t, c in m[n].nonzero().tolist()}
    assert abs(got["bits"] - goal_eval.size_bits(g, bk.P, bk.S, tx.targets, active=tx.vpd)) < 1e-9
    # the graph's C-semantics point run (goal_eval's, blocks expanded) is the same masked run
    kl_c = ge.sufficiency(tx, [goal_eval._c_graph(g, tx.targets, nat.C)])[0]["kl_live"]
    assert abs(kl_c - point) < 1e-5 + 1e-4 * point, (kl_c, point)


def test_whole_model_as_blocks_is_the_model(tiny):
    """Every site at every position a block: the model itself (the remainders live), so its run less any draw is the
    model's less it: KL 0 at the point and under every draw, s_search 1."""
    nat, ge, tx, circuit = tiny
    bk = backup.Backup(ge, tx, circuit, draws=4, rows=4)
    every = frozenset((n, t) for n in nat.names for t in range(tx.T))
    got = bk.score([(every, frozenset())])[0]
    assert got["kl_point"] < 1e-5 and got["kl_nodes_c"] < 1e-5
    assert got["s_search"] == pytest.approx(1.0)


def test_screen_is_first_order(tiny):
    """The screen's gradient against a finite difference of its objective: switching one outside subcomponent's mask
    on by eps moves sum_draws coef ln KL by eps x its entry (the coefficients fixed at the expansion)."""
    nat, ge, tx, circuit = tiny
    bk = backup.Backup(ge, tx, circuit, draws=4, rows=4)
    grad = bk.screen()
    m = bk.masks()
    n = nat.names[10]  # layer 1's MLP input
    t, c = 11, int((~m[n][11]).nonzero()[0])
    lq = bk._run([(m, None)] + [(m, j) for j in range(len(bk.drops))])
    f = bk._fields(lq)
    s, f1 = max(f["s_search"], 1e-3), max(f["f1"], 1e-3)
    kc = carry.kl(bk.lm, lq[1:]).tolist()
    wsum = sum(w for w, kv in zip(bk.W, bk.KV) if kv < w) or 1.0
    coef = [1.0 / (np.log(bk.e0 / bk.v0) * f1 ** 2) if f1 < 1 else 0.0]
    coef += [w / (wsum * np.log(w / kv)) / s ** 2 if (kv < w and k > kv) else 0.0 for k, kv, w in zip(kc, bk.KV, bk.W)]

    def loss(eps):
        rows = []
        for j in [None] + list(range(len(bk.drops))):
            r = bk._row(m, j, torch.float64)
            r[n][t, c] += eps
            rows.append(r)
        lq = bk.E.log_probs({k: torch.stack([r[k] for r in rows]).float() for k in nat.names})
        refs = torch.cat([tx.ref] + [bk.lm[j:j + 1] for j in range(len(bk.drops))])
        return float((torch.tensor(coef) * carry.kl(refs, lq).clamp_min(1e-12).log()).sum())

    eps = 1e-2
    fd = (loss(eps) - loss(-eps)) / (2 * eps)
    assert abs(float(grad[n][t, c]) - fd) < 2e-2 * abs(fd) + 1e-4, (float(grad[n][t, c]), fd)
    order = bk.block_screen(grad)
    assert order == sorted(order) and all(lo <= tt <= max(tx.targets) for _, _, tt in order for lo in [min(bk.positions)])


def test_expand_records_its_path(tiny):
    """The greedy path: each step's block measured, its gain at least min_gain and J rising; expand_circuit's answer is
    the circuit with the path's blocks (or refine's subcomponents for them), scored as recorded."""
    nat, ge, tx, circuit = tiny
    bk = backup.Backup(ge, tx, circuit, draws=4, rows=4)
    rec = bk.expand(max_steps=3, min_gain=1e-4, screen_top=4, lazy=4)
    js = [rec["start"]["j"]] + [s["j"] for s in rec["steps"]]
    assert all(b >= a + 1e-4 - 1e-12 for a, b in zip(js, js[1:])), js
    assert [b for s in rec["steps"] for b in s["add"]] == rec["blocks"] == (rec["step_blocks"] or [[]])[-1]
    for k, s in enumerate(rec["steps"]):
        again = bk.score([(frozenset(tuple(b) for b in rec["step_blocks"][k]), frozenset())])[0]
        assert abs(again["j"] - s["j"]) < 1e-6
    every = [(n, t) for t in bk.positions for n in nat.names]
    back = bk.prune(every, max_steps=4, min_gain=1e-4)
    js = [back["start"]["j"]] + [s["j"] for s in back["steps"]]
    assert all(b >= a + 1e-4 - 1e-12 for a, b in zip(js, js[1:])) and set(map(tuple, back["blocks"])) <= set(every)
    assert abs(bk.score([(frozenset(map(tuple, back["blocks"])), frozenset())])[0]["j"] - back["final"]["j"]) < 1e-6
    g, r2 = backup.expand_circuit(ge, tx, circuit, max_steps=3, min_gain=1e-4, refine=True, draws=4)
    blocks = {nd[:2] for nd in g.node_set() if nd[2] == goal_eval.BLOCK}
    chosen = r2["prune"]["blocks"] if r2["chosen"] == "back" else r2["expand"]["blocks"]
    assert blocks == {tuple(b) for b in (r2["refine"]["blocks"] if r2["refine"] else chosen)}
    assert circuit.node_set() <= goal_eval.expand_blocks(g, nat.C).node_set()
    bk2 = backup.Backup(ge, tx, circuit, draws=4)
    extra = frozenset(nd for nd in g.node_set() if nd[2] != goal_eval.BLOCK) - circuit.node_set()
    assert abs(bk2.score([(frozenset(blocks), extra)])[0]["j"] - r2["final"]["j"]) < 1e-6
    if rec["blocks"]:  # interpret's rows: one per refined block, cosines in [0, 1], heads only for attention sites
        diag = bk.refine(rec["blocks"][:2])
        assert len(diag["per_block"]) == len(rec["blocks"][:2])
        rows = bk.interpret(diag["per_block"], top=4)
        assert len(rows) == len(diag["per_block"])
        for row in rows:
            assert row["read_cos_comp"] is None or 0 <= row["read_cos_comp"] <= 1 + 1e-6
            assert ("heads_comp" in row) == (row["block"][0].split(".")[-1] in ("o_proj", "v_proj", "q_proj", "k_proj"))


def test_block_units_and_nodes_search(tiny):
    """A block of the circuit is one removal unit (block_units): a draw that removes it switches its site off at its
    position, every subcomponent; nodes_search scores a graph as Backup's own score of it on the nodes_search draws."""
    nat, ge, tx, circuit = tiny
    n, t = nat.names[4], 9
    g = native.Graph(sorted(circuit.node_set()) + [(n, t, goal_eval.BLOCK)], complete=True)
    bk = backup.Backup(ge, tx, g, draws=8, family="nodes_search")
    assert all(nd[2] != goal_eval.BLOCK or nd[:2] == (n, t) for d in bk.drops for nd in d)
    hit = [j for j, d in enumerate(bk.drops) if (n, t, goal_eval.BLOCK) in d]
    assert hit, "some draw removes the block (8 draws, probability up to 15/16)"
    row = bk._row(bk.masks(), hit[0])
    assert float(row[n][t].abs().sum()) == 0.0
    got = backup.nodes_search(ge, tx, [g], draws=8)[0]
    assert abs(got["s_nodes"] - bk.score([(frozenset(), frozenset())])[0]["s_search"]) < 1e-9


def test_derived_sets_and_pricing(tiny):
    """derived_sets from a summary row: rand_fwd as many blocks as fwd at the circuit's positions, vpd_eq_fwd VPD's order
    beyond the circuit as many as fwd's blocks add of VPD's active subcomponents, sub50 and sub90 a measured block's
    shortest prefix holding that share of its gain (an unmeasured block kept); counts claims a block's every
    subcomponent and its VPD-active ones."""
    nat, ge, tx, circuit = tiny
    bk = backup.Backup(ge, tx, circuit, draws=4, rows=4)
    t = max(bk.positions)
    fwd = [(nat.names[4], t), (nat.names[10], t)]
    order = list(range(nat.C[nat.names[4]]))[::-1][:4]
    pb = {"block": list(fwd[0]), "gain": 0.1, "j_without": 0.5, "order": order,
          "prefixes": [{"k": 1, "j": 0.52}, {"k": 2, "j": 0.56}, {"k": 4, "j": 0.6}]}
    row = {"sets": {"fwd": [[list(b) for b in fwd], []], "full": [[list(b) for b in fwd], []]}, "refine_fwd": [pb]}
    got = backup.derived_sets(row, circuit, tx, nat, seed=3)
    bl, ex = got["rand_fwd"]
    assert len(bl) == 2 and not ex and all(tt in bk.positions for _, tt in bl)
    nodes = set(circuit.node_set())
    added = sum(int(tx.vpd[n][tt].sum()) - sum(1 for nd in nodes if nd[:2] == (n, tt) and bool(tx.vpd[n][tt, nd[2]])) for n, tt in fwd)
    order_v = [tuple(nd) for nd in nat._vpd_order(tx.ids, tx.targets) if tuple(nd) not in nodes]
    assert got["vpd_eq_fwd"] == ([], order_v[:max(1, added)]) and got["vpd_eq"] == got["vpd_eq_fwd"]
    assert got["sub50"] == ([fwd[1]], [(fwd[0][0], t, c) for c in order[:2]])
    assert got["sub90"] == ([fwd[1]], [(fwd[0][0], t, c) for c in order[:4]])
    bl, ex = got["sub90_rand"]
    assert bl == [fwd[1]] and len(ex) == 4 and all(nd[:2] == fwd[0] and nd not in nodes for nd in ex)
    g = backup.expanded_graph(*backup.split_graph(circuit), [fwd[0]], [])
    c = backup.counts(g, nat.C, tx.vpd)
    subs = [nd for nd in g.node_set() if nd[2] != goal_eval.BLOCK]
    assert c["blocks"] == 1 and c["subcomponents"] == len(subs) + nat.C[fwd[0][0]]
    assert c["active"] == sum(bool(tx.vpd[n][tt, cc]) for n, tt, cc in subs) + int(tx.vpd[fwd[0][0]][t].sum())
