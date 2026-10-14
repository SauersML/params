import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scope_repick import pick, to_json  # noqa: E402


def row(stage, units, term, v):
    return [stage, 1.0, 0.5, units, {}, 0.0, term, v]


def test_pick_takes_the_best_term_meeting_held_out_scope():
    phi = [row("subcomponent", 200, 1.4, 0.0), row("chunk16", 250, 1.2, 0.0), row("site", 280, 1.3, 0.1),
           row("start", 700, 0.3, 0.0)]
    assert pick(phi, {"subcomponent": 0.95, "chunk16": 0.95, "site": 0.99, "start": 1.0}, 0.9) == ("subcomponent", None)
    # the best term under held-out scope falls to the next; a stage missing a search floor never counts
    assert pick(phi, {"subcomponent": 0.33, "chunk16": 0.95, "site": 0.99, "start": 1.0}, 0.9) == ("chunk16", None)
    # no scope draws: no floor
    assert pick(phi, {"subcomponent": None, "chunk16": 0.2, "site": 0.2, "start": 0.2}, 0.9) == ("subcomponent", None)


def test_pick_falls_back_to_the_start_flagged():
    phi = [row("subcomponent", 200, -1.0, 0.3), row("start", 700, -0.5, 0.2)]
    assert pick(phi, {"subcomponent": 0.95, "start": 0.95}, 0.9) == ("start", "no stage meets every floor")
    # floor_of a START under FLOOR_TOP there: a stage no worse than it by more than the margin stays (text6490)
    phi = [row("chunk4", 174, 0.7, 0.0), row("start", 250, 0.11, 0.0)]
    assert pick(phi, {"chunk4": 0.863, "start": 0.861}, 0.811) == ("chunk4", None)
    assert pick(phi, {"chunk4": 0.75, "start": 0.861}, 0.811) == ("start", None)


def test_to_json_writes_goal_eval_order():
    out = to_json([("h.1.mlp.c_fc", 5, 7), ("h.0.attn.q_proj", 5, 3), ("h.0.attn.q_proj", 2, 9)])
    assert out["nodes"] == [["h.0.attn.q_proj", 9, 2], ["h.0.attn.q_proj", 3, 5], ["h.1.mlp.c_fc", 7, 5]]
    assert out["complete"] and out["edges"] == []
