"""Checks of component answers (components.py): the text format both ways, question scoring, and the reward on a tiny
random model (test_goal_eval's).

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_components.py -q
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import random

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import components  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402

ANSWER = """CIRCUIT:
@5 a=L1.o#1 b=L1.down#2
@3 c=L1.k#3 d=L1.v#4 e=L0.fc#5 f=L0.down#0
OUT <- a,b
a <- d
d <- f
f <- e
EXPLANATION: the value at 3 is copied to the last position.
ANSWER 1: the copied value is gone.
FINAL 1: falls=yes top=' the'
"""


def edges(q):
    return {(tuple(q["answer"]["nodes"][w]), "out" if r == "out" else tuple(q["answer"]["nodes"][r])) for w, r in q["answer"]["edges"]}


def test_parse_render_round_trip():
    """A v1 answer (the id grammar) is still read; render() writes the circuit as fenced Mermaid, every node declared
    once in its position's subgraph, and Mermaid parses back to the same circuit (render is idempotent)."""
    p = components.parse(ANSWER)
    assert p["malformed"] == 0 and p["circuit"]
    assert p["explanation"] == "the value at 3 is copied to the last position."
    assert p["finals"] == {1: (True, "' the'")}
    assert len(p["answer"]["nodes"]) == 6 and len(p["answer"]["edges"]) == 5
    text = components.render(p["answer"], p["finals"], {1: "the copied value is gone."}, strings=[f"w{i}" for i in range(6)])
    lines = text.splitlines()
    assert lines[:4] == ["CIRCUIT:", "```mermaid", "flowchart LR", '  subgraph P5["@5 \'w5\'"]'] and "```" in lines
    assert '    a["L1.o#1"]' in lines and "  a --> OUT((prediction))" in lines and "  b --> OUT" in lines
    q = components.parse(text)
    assert edges(q) == edges(p) and q["malformed"] == 0 and q["finals"] == p["finals"]
    assert components.render(q["answer"], q["finals"], {1: "the copied value is gone."}, strings=[f"w{i}" for i in range(6)]) == text


def test_written_out_nodes_still_parse():
    text = "EXPLANATION: x\nCIRCUIT:\nout <- L1.down#2,9@5\nL1.down#2,9@5 <- L1.fc#1,7@5, L0.down#3@5\nL1.k#4,6@3\n"
    p = components.parse(text)
    assert p["malformed"] == 0
    assert len(p["answer"]["nodes"]) == 7
    assert len(p["answer"]["edges"]) == 2 + 2 * 3  # two into the prediction, every writer into both readers
    again = components.parse(components.render(p["answer"]))
    assert edges(again) == edges(p) and "flowchart LR" in components.render(p["answer"])
    assert components.parse("CIRCUIT:\nL1.down#2, 9@5 <- L1.fc#1@5\n")["malformed"] == 1
    block = components.parse("CIRCUIT:\n@4 L1.k#4,6 L0.q#2 tok\nL1.k#9@4\n")  # a block without ids, and a bare group
    assert block["malformed"] == 0 and len(block["answer"]["nodes"]) == 5 and not block["answer"]["edges"]


def test_render_puts_the_circuit_first_and_either_order_parses_alike():
    answer = {"nodes": [["h.1.mlp.down_proj", 5, 9], ["h.1.mlp.c_fc", 1, 9]], "edges": [[1, 0], [0, "out"]], "explanation": "it copies"}
    text = components.render(answer, {1: (True, "' x'")}, {1: "because"}, {1: [{"op": "scale", "node": ("h.1.mlp.c_fc", 9, 1), "k": 0.0, "persistent": False}]})
    order = [text.index(k) for k in ("CIRCUIT:", "EXPLANATION:", "ANSWER 1:", "FINAL 1:", "EDIT 1:")]
    assert order == sorted(order)  # circuit, explanation, answers, edits
    assert "EXPLANATION" not in components.render({**answer, "explanation": ""}, {1: (True, "' x'")})  # none written when empty
    circuit = text[text.index("CIRCUIT:"):text.index("EXPLANATION:")]
    rest = text[text.index("EXPLANATION:"):]
    old = rest.split("ANSWER 1:")[0] + circuit + "ANSWER 1:" + rest.split("ANSWER 1:")[1]  # the explanation first
    a, b = components.parse(text), components.parse(old)
    assert a["answer"] == b["answer"] and a["finals"] == b["finals"] and a["edits"] == b["edits"] and a["explanation"] == b["explanation"]


def test_short_ids():
    text = ("EXPLANATION: x\nCIRCUIT:\nOUT <- a\n@9 a=L1.down#5 b=L1.fc#1 d=L1.fc#2 e=tok\n@4 f=L0.v#7\n"
            "a <- b,d\nb <- e\nd <- e, L0.down#3@9\n")  # an edge line may come before the block declaring its ids
    p = components.parse(text)
    assert p["malformed"] == 0 and len(p["answer"]["nodes"]) == 6 and len(p["answer"]["edges"]) == 6
    assert edges(components.parse(components.render(p["answer"]))) == edges(p)
    assert components.parse("CIRCUIT:\n@9 a=L1.down#5,6\n")["malformed"] == 1  # an id names one node
    assert components.parse("CIRCUIT:\n@9 a=L1.down#5\n@8 a=L1.down#4\n")["malformed"] == 1  # and once
    undefined = components.parse("CIRCUIT:\n@9 a=L1.down#5 b=L1.fc#1\nOUT <- a,zz\na <- b\nyy <- b\n")
    assert undefined["unknown_ids"] == 2 and undefined["malformed"] == 0  # charged as nodes outside the graph
    assert len(undefined["answer"]["edges"]) == 2  # the defined edges stay
    items = components.parse("CIRCUIT:\n@9 a=L1.down#5 b=L3.d#66 c=L0.q#2 avs d=L1.k#3@9\nOUT <- a,c\n")
    assert len(items["answer"]["nodes"]) == 2 and len(items["answer"]["edges"]) == 2  # a bad item drops alone
    assert items["malformed"] == 1 and items["malformed_nodes"] == 3  # the line counted once, each bad item charged


def test_edits():
    text = ("CIRCUIT:\n@9 a=L1.down#5 b=L1.fc#1\nOUT <- a\na <- b\nEDIT 1: cut a <- b; scale b 0\n"
            "EDIT 2: persistent scale L1.v#7 0; persistent cut L1.o#3 <- L1.v#7\nEDIT 3: cut out <- a; scale zz 2\n")
    p = components.parse(text)
    assert p["malformed"] == 0 and len(p["answer"]["edges"]) == 2  # edit lines are not circuit lines
    assert [o["op"] for o in p["edits"][1]] == ["cut", "scale"] and p["edits"][1][1]["k"] == 0.0
    assert all(o["persistent"] and (o.get("node") or o["writer"])[1] is None for o in p["edits"][2])
    assert p["edits"][3][0]["reader"] == "out" and p["edits"][3][1]["op"] == "invalid"  # an undefined id names nothing
    again = components.parse(components.render(p["answer"], edits={k: v for k, v in p["edits"].items() if k < 3}))
    assert again["edits"] == {k: v for k, v in p["edits"].items() if k < 3}


@pytest.mark.parametrize("base", components.EXPLAINERS)
def test_escape_keeps_special_strings_text(base):
    """Every added token of every explainer's tokenizer, escaped, encodes without any added token."""
    tk = pytest.importorskip("transformers").AutoTokenizer
    try:
        tok = tk.from_pretrained(base)
    except OSError:
        pytest.skip(f"{base}'s tokenizer is not in the Hugging Face cache")
    added = set(tok.get_added_vocab().values())
    raw = "tokens: 0:'<|endoftext|>' 1:'<|im_start|>' 2:'<think>' 3:'</tool_call>' 4:'<|fim_pad|>'"
    for s in tok.get_added_vocab():
        assert not set(tok(components.escape(s), add_special_tokens=False)["input_ids"]) & added, s
    chat = lambda q, a: tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],  # noqa: E731
                                                tokenize=True, enable_thinking=False)
    plain = [i for i in chat("x", "y") if i in added]
    escaped = [i for i in chat(components.escape(raw), components.render({"nodes": [], "edges": []}, {1: (True, components.shown("<|endoftext|>"))}))
               if i in added]
    assert escaped == plain  # only the chat template's own special tokens
    assert components.escape(components.escape(raw)) == components.escape(raw)


def test_malformed_lines_and_dropped_edges():
    p = components.parse("EXPLANATION: x\nCIRCUIT:\nout <- L1.o#1@5\nL1.o#1@5 <- L1.v#4@3 and more\nL7\n")
    assert p["malformed"] + p["unknown_ids"] == 2  # a line that does not parse; L7, an id never declared
    C = {native.site_name(layer, k): 6 for layer in range(2) for k in components.KINDS.values()}
    q = components.parse("EXPLANATION: x\nCIRCUIT:\nout <- L1.v#4@5\nL0.o#1@5 <- L1.down#0@5\n")
    assert q["malformed"] == 0
    _, d = goal_eval.circuit(q["answer"], [5], C, 6)
    assert d["invalid_edges"] == 2  # a value does not write the residual stream; a later MLP does not feed an earlier output


def test_qa_score():
    qs = [{"falls": True, "top": 0}, {"falls": False, "top": 1}]
    tok = lambda ids: [f"t{i}" for i in ids]  # noqa: E731
    assert components.qa_score(qs, {1: (True, "'t0'"), 2: (False, "'t1'")}, tok) == 1.0
    assert components.qa_score(qs, {1: (False, "'t0'")}, tok) == 0.25
    assert components.qa_score([], {}, tok) is None


def test_qa_term_is_measured_from_the_no_change_answer():
    qs = [{"falls": True, "top": 0}, {"falls": False, "top": 1}]  # the model predicts t1 on the text
    tok = lambda ids: [f"t{i}" for i in ids]  # noqa: E731
    still = {1: (False, "'t1'"), 2: (False, "'t1'")}  # the no-change answer: right on question 2 alone (0.5)
    assert components.qa_term(qs, still, tok, "t1") == 0.0
    assert components.qa_term(qs, {1: (True, "'t0'"), 2: (False, "'t1'")}, tok, "t1") == 1.0
    assert components.qa_term(qs, {1: (False, "'t0'")}, tok, "t1") == -0.5  # 0.25 right: below no change
    assert components.qa_term(qs, {}, tok, "t1") == -1.0  # no FINAL answers right: clipped
    assert components.qa_term([qs[1]], {1: (False, "'t1'")}, tok, "t1") is None  # no change is right on everything
    assert components.qa_term([], {}, tok, "t1") is None


def test_structural_changes_keep_the_tokens():
    ids, targets = list(range(10, 30)), [19]
    move = components.apply_change(ids, targets, {"kind": "move", "position": 5, "to": 12})
    assert move[1] == targets and sorted(move[0]) == sorted(ids) and move[0][12] == ids[5] and move[0][5] == ids[6]
    swap = components.apply_change(ids, targets, {"kind": "swap", "position": 3, "with": 8})
    assert swap[0][3] == ids[8] and swap[0][8] == ids[3] and swap[1] == targets
    ins = components.apply_change(ids, targets, {"kind": "insert", "position": 7, "at": 2})
    assert ins[0] == ids[:2] + [ids[7]] + ids[2:] and ins[1] == [20]  # the target one further on
    changes = components.structure_changes(ids, targets, [19, 7, 4, 11, 2], random.Random(0))
    kinds = [c["kind"] for c in changes]
    assert kinds.count("move") == 2 and kinds.count("insert") == 2 and kinds.count("swap") == 2
    assert all(c["at"] < c["position"] for c in changes if c["kind"] == "insert")  # a copy earlier in the text
    assert all(abs(c["to"] - c["position"]) >= 3 for c in changes if c["kind"] == "move")
    strings = [f"t{i}" for i in ids]
    assert "moved to position 12" in components.change_text({"kind": "move", "position": 5, "to": 12}, strings)


class FakeReader:
    """reader.Reader's losses: unaided, the reader knows the events at the positions the circuit lists (0 bits) and
    not the others (1 bit); with English it loses 1.2 bits on the neutral sentence, 0.2 where the English names the
    event's position and 1.5 elsewhere."""

    def __init__(self):
        self.asked = []

    def losses(self, task, items, events):
        per = events if events and isinstance(events[0], list) else [events] * len(items)
        self.asked += [(c, eng, evs) for (c, eng), evs in zip(items, per)]

        def loss(c, eng, e):
            if not eng:
                return 0.0 if f"@{e['position']} " in c else 1.0
            return 1.2 if eng == components.NEUTRAL else (0.2 if f"position {e['position']}" in eng else 1.5)
        return [[loss(c, eng, e) for e in evs] for (c, eng), evs in zip(items, per)]


def test_e1_is_what_the_english_adds_to_the_circuit_against_the_neutral_explanation():
    """The reader reads the circuit; events are weighted by measured effect x the unaided reader's loss (an event the
    circuit already tells weighs nothing); E1 is the weighted bits the English saves against NEUTRAL; the unaided and
    neutral prompts are asked once per circuit; no English is 0."""
    ev = {"events": [{"position": p, "new": 1, "alt": 0, "listed": False, "down": True, "weight": w, "bits": -w}
                     for p, w in enumerate([0.1, 3.0, 0.5] + [0.01] * 20)],
          "structure": [{"kind": "swap", "position": 5, "with": 6, "change": "x", "down": True, "weight": 2.0, "bits": -2.0}]}
    r = FakeReader()
    texts = ["CIRCUIT:\n@1 a=tok\nOUT <- a\nEXPLANATION: position 2 matters",
             "CIRCUIT:\n@1 a=tok\nOUT <- a\nEXPLANATION: position 9 matters", "CIRCUIT:\n@1 a=tok\nOUT <- a\n"]
    got = components.e1_bits(r, {}, ev, texts)
    unaided = [x for x in r.asked if not x[1]]
    neutral = [x for x in r.asked if x[1] == components.NEUTRAL]
    assert len(unaided) == 1 and len(neutral) == 1 and unaided[0][0] == "@1 a=tok\nOUT <- a"
    asked = neutral[0][2]
    assert len(asked) == components.E1_EVENTS and asked[0]["weight"] == 2.0 and asked[1]["position"] == 2
    assert all(e["position"] != 1 for e in asked)  # the heaviest event, but the circuit tells it: weight 0
    w = [e["weight"] for e in asked]  # unaided loss 1 on every asked event
    want = lambda p: sum(x * (1.2 - (0.2 if e["position"] == p else 1.5)) for x, e in zip(w, asked)) / sum(w)  # noqa: E731
    assert got[0] == pytest.approx(want(2)) and got[1] == pytest.approx(want(9)) and got[2] == 0.0
    assert got[0] > got[1]  # the English names the heavier event


class FakeEdits:
    """edit_eval.EditEval's score_many: a one-operation edit is valid with net 2 bits, a longer one rejected."""

    def score_many(self, task, goal, edits):
        return [{"valid": len(ops) == 1, "net": 2.0 if len(ops) == 1 else None} for ops in edits]


def test_edit_term():
    import edit_eval

    task, goals = {"id": "text0"}, [{"kind": "flip", "token_id": 5}, {"kind": "detach", "position": 3}]
    best = {edit_eval.EditEval.key("text0", goals[0]): 4.0, edit_eval.EditEval.key("text0", goals[1]): 2.0}
    op = {"op": "scale", "node": ("h.0.mlp.c_fc", 3, 1), "k": 0.0, "persistent": False}
    answers = [{1: [op], 2: [op]}, {1: [{"op": "invalid", "text": "x"}]}, {1: [op, op], 2: [op]}]
    assert components.edit_term(FakeEdits(), task, goals, answers, best) == [0.75, 0.0, 0.5]  # net / best, 0 when invalid
    first = {k: v for k, v in list(best.items())[:1]}
    assert components.edit_term(FakeEdits(), task, goals, answers, first) == [0.5, 0.0, 0.0]  # goal 2 has no best net: left out
    negative = {**first, edit_eval.EditEval.key("text0", goals[1]): -1.0}
    assert components.edit_term(FakeEdits(), task, goals, answers, negative) == [0.5, 0.0, 0.0]  # nor one no edit gains on
    assert components.edit_term(FakeEdits(), task, goals, answers, {}) == [None] * 3


class NoWiring:
    names = []

    def partners(self, name, idx, kind):
        return {k: np.empty(0) for k in ("site", "index", "head", "raw", "cosine")}


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native()
    ge = goal_eval.GoalEval(nat, changes=4, random_pairs=4)
    ge.donor_pools.update(search=[], eval=[])  # the far donors are vpd4l's texts, not the tiny model's: none here
    ids = [3, 17, 5, 9, 22, 11, 30, 2]
    task = {"id": "text0", "model": "vpd4l", "split": "heldout",
            "prompts": [{"token_ids": ids, "target_positions": [7], "model_top": [[[" the", 0.5]]]}]}
    ev = components.evidence(nat, ge, task, NoWiring())
    return nat, ge, task, ev


def test_reward_on_tiny_model(tiny):
    nat, ge, task, ev = tiny
    assert ev["positions"] and ev["active"] > 0
    invalid = components.V3_INVALID
    order = nat._vpd_order(task["prompts"][0]["token_ids"], [7])
    at_t = [nd for nd in order if nd[1] == 7 and nd[0].endswith(("o_proj", "down_proj"))][:3]
    fc = next(nd for nd in order if nd[1] == 7 and nd[0] == "h.1.mlp.c_fc")
    down = (native.site_name(1, "down_proj"), 7, 0)
    g = native.Graph(set(at_t) | {fc, down}, {down: [fc]}, at_t + [down])
    finals = {n: (q["falls"], "'x'") for n, q in enumerate(ev["qa"], 1)}
    good = components.render(goal_eval.to_json(g, "writes at the target"), finals)
    bad = good.replace("  subgraph P7", "  subgraph P7", 1).replace("  end\n", '    zz["L0.v#0"]\n  end\n', 1).replace(
        "\n```\n", "\n  zz --> OUT\n```\n", 1)  # a value never writes the residual stream: dropped, charged
    missing = components.render(goal_eval.to_json(g, "writes at the target"))
    got = components.reward(ge, task, ev, [good, bad, missing, good], truncated=[False, False, False, True], pool={})
    assert got[0]["valid"] and invalid < got[0]["reward"]
    assert all(got[0][f"term_{k}"] is None or 0.0 <= got[0][f"term_{k}"] <= 1.0 for k in ("C", "F1", "N", "F3"))
    v = got[0]  # reward v3: phi + g x (0.5 compact + E2g + E4x + E1) + D
    assert v["g"] is None or 0.0 <= v["g"] <= 1.0
    assert v["reward"] == pytest.approx((v["phi"] or 0.0) + (v["g"] or 0.0) * v["Q"] + v["D"])
    assert v["Q"] == pytest.approx(0.5 * (v["compact"] or 0.0) + sum(v[f"term_{k}"] or 0.0 for k in ("E2g", "E4x", "E1")))
    tok = lambda xs: components.token_strings(task["model"], xs)  # noqa: E731
    right0 = sum(0.5 * (not q["falls"]) + 0.5 * (tok([q["top"]])[0] == " the") for q in ev["qa"]) / max(len(ev["qa"]), 1)
    if ev["qa"] and right0 < 1.0:  # every "falls" right and no top token "x" (0.5 right), measured from no change
        assert got[0]["term_E2"] == pytest.approx(max(-1.0, min(1.0, (0.5 - right0) / (1.0 - right0))))
    else:
        assert got[0]["term_E2"] is None
    assert got[1]["valid"] and got[1]["dropped_edges"] == 1 and got[1]["bits"] > got[0]["bits"]
    assert (not got[2]["valid"]) == bool(ev["qa"]) and not got[3]["valid"] and got[3]["reward"] == invalid
    import edit_eval

    goals = [{"kind": "flip", "token_id": 5, "token": " the", "target": 7}]
    asked = {**ev, "edits": goals}
    best = {edit_eval.EditEval.key(task["id"], goals[0]): 4.0}
    edited = good + "\nEDIT 1: scale " + good.split("subgraph P7", 1)[1].split("\n    ", 1)[1].split("[", 1)[0] + " 0"
    with_e3 = components.reward(ge, task, asked, [edited, good], ee=FakeEdits(), edit_best=best, pool={})
    assert with_e3[0]["term_E3"] == 0.5 and not with_e3[1]["valid"]  # 2 bits of the teacher's 4 (reported); no EDIT line: invalid


def test_invalid_below_every_valid_answer(tiny):
    """However badly a valid answer does, it earns more than an invalid one; an answer above NODE_CAP nodes is invalid."""
    nat, ge, task, ev = tiny
    T = len(task["prompts"][0]["token_ids"])
    every = native.Graph({(n, t, c) for n in nat.names for t in range(T) for c in range(nat.C[n])})
    par, out = native.all_edges(every.node_set(), [7])
    huge = native.Graph(every.node_set(), par, out)
    wrong = {n: (not q["falls"], "'x'") for n, q in enumerate(ev["qa"], 1)}
    junk = {"nodes": [[f"{layer}.{k}", c, t] for t in range(T) for c in range(5) for layer, k in ((3, "v"), (0, "q"))],
            "edges": [[2 * i + 1, 2 * i] for i in range(5 * T)]}  # a query writes nothing: every connection dropped
    garbage = "\n".join(["Reader: L1.o#1@3 <- Writer: L0.down#2@3", "the attention reads the previous token"] * 200)
    texts = [components.render(goal_eval.to_json(huge, ""), wrong), components.render(junk, wrong),
             components.render({"nodes": [], "edges": []}, wrong).replace("flowchart LR", "flowchart LR\n" + garbage)]
    got = components.reward(ge, task, ev, texts, pool={})
    assert huge.count() > components.NODE_CAP and not got[0]["valid"] and got[0]["reward"] == components.V3_INVALID
    assert got[1]["valid"] and got[2]["valid"]  # lines that do not parse are dropped and charged, not fatal
    assert got[1]["dropped_edges"] == 5 * T and got[2]["malformed"] == 400 and got[2]["bits"] > got[1]["bits"] > 0
    assert min(r["reward"] for r in got[1:]) > components.V3_INVALID

def test_variables_block_parses_renders_and_round_trips():
    """The EXPLANATION's VARIABLES block: ids and written-out subcomponents, inputs with their tokens, feeds by name; a
    line that names no subcomponent (the token embedding is none) or does not parse is malformed; the prose apart."""
    text = "\n".join(["CIRCUIT:", "@7 a=L3.down#2 b=L2.o#1", "@4 c=L2.v#5 d=tok", "OUT <- a,b", "b <- c", "c <- d",
                      "EXPLANATION:", "VARIABLES:",
                      "V1 \"copied word\": the word after the earlier 'for'; inputs: @4 ' each', @3 ' for'; carried by: c,L1.fc#9@4; feeds: V2",
                      "V2 \"prediction read\": the copy, at the target; inputs: @4 ' each'; carried by: b; feeds: PREDICTION",
                      "V3 \"embedding\": the token itself; inputs: @4; carried by: d; feeds: V1",
                      "V4 broken line without the parts",
                      "The model copies the word after the earlier 'for'.", "ANSWER 1: x", "FINAL 1: falls=yes top=' each'"])
    p = components.parse(text)
    vs = p["answer"]["variables"]
    assert p["variables_malformed"] == 2 and len(vs) == 2
    assert vs[0]["name"] == "copied word" and vs[0]["inputs"] == [3, 4] and vs[0]["feeds"] == ["prediction read"]
    assert sorted(vs[0]["group"]) == sorted([["h.1.mlp.c_fc", 9, 4], ["h.2.attn.v_proj", 5, 4]])
    assert vs[1]["feeds"] == ["prediction"] and vs[1]["group"] == [["h.2.attn.o_proj", 1, 7]]
    assert p["answer"]["explanation"] == "The model copies the word after the earlier 'for'."
    assert "VARIABLES" in p["explanation"]  # E1's reader reads the whole section
    strings = [f"t{i}" for i in range(8)]
    again = components.render(p["answer"], strings=strings)
    assert "V1 \"copied word\": the word after the earlier 'for'; inputs: @3 't3', @4 't4'; carried by: " in again
    assert "; feeds: V2" in again and "; feeds: PREDICTION" in again
    q = components.parse(again)
    assert q["answer"]["variables"] == vs and q["variables_malformed"] == 0 and components.render(q["answer"], strings=strings) == again
    plain = components.parse(text.replace("VARIABLES:\n", "").split("\nV1")[0] + "\nThe model copies.")
    assert "variables" not in plain["answer"] and plain["variables_malformed"] == 0


class FakeScorer:
    """variables.Scorer's score: e4 0.6 in the model, 0.9 in the stated graph."""

    def __init__(self):
        self.asked = self.split = None

    def score(self, tx, vs, split="eval", graph=None):
        self.asked, self.split = vs, split
        return {"e4": 0.6, "circuit": {"e4": 0.9}, "gap": 0.3, "variables": [{"f_on": 0.7, "s_off": 0.5} for _ in vs]}


def test_e4_is_the_scorers_e4_with_a_malformed_variable_scoring_zero():
    C = {"h.2.attn.v_proj": 16, "h.2.attn.o_proj": 16}
    one = {"name": "a", "meaning": "", "inputs": [4], "group": [["h.2.attn.v_proj", 5, 4]], "feeds": ["prediction"]}
    two = {"name": "b", "meaning": "", "inputs": [6], "group": [["h.2.attn.o_proj", 1, 7]], "feeds": ["prediction"]}
    sc = FakeScorer()
    out = components.e4_term(sc, None, None, {"variables": [one, two]}, 0, C)
    assert {k: out[k] for k in ("E4", "model", "circuit", "gap")} == {"E4": 0.6, "model": 0.6, "circuit": 0.9, "gap": 0.3}
    assert [v["name"] for v in sc.asked] == ["a", "b"] and [x["e4"] for x in out["per_variable"]] == [0.5, 0.5]  # min(f_on, s_off)
    outside = {**two, "group": [["h.2.attn.o_proj", 99, 7]]}  # no such subcomponent; and one line that did not parse
    assert components.e4_term(sc, None, None, {"variables": [one, outside]}, 1, C)["E4"] == pytest.approx(0.6 / 3)
    assert components.e4_term(sc, None, None, {}, 0, C)["E4"] == 0.0
    components.e4_term(sc, None, None, {"variables": [one]}, 0, C, "search")
    assert sc.split == "search"  # the RL reward's E4 on the same draws as C (variables.Scorer defaults to eval)


def test_baselines_at_equal_bits_take_the_most_subcomponents_that_fit(tiny):
    nat, ge, task, ev = tiny
    prompt = task["prompts"][0]
    tx = ge.text(prompt["token_ids"], prompt["target_positions"], task=task)
    order = nat._vpd_order(tx.ids, tx.targets)
    P, S = nat.positions(tx.targets), sum(nat.C.values())
    cost = lambda k: goal_eval.size_bits(native.Graph(set(order[:k]), complete=True), P, S, tx.targets, active=tx.vpd)  # noqa: E731
    k = components.matched_k(ge, tx, order, cost(5) + 0.5)
    assert k >= 5 and cost(k) <= cost(5) + 0.5 and (k == len(order) or cost(k + 1) > cost(5) + 0.5)
    recent = components.recency_order(order)
    assert [nd[1] for nd in recent] == sorted((nd[1] for nd in order), reverse=True)


def test_control_questions_hide_move_or_scramble_the_evidence():
    ev = {"positions": [8, 9], "listed": [8], "events": [{"position": 8, "kl": 1.0}], "qa": [{"kind": "remove"}],
          "components": {"8": [[["h.1.attn.v_proj", 8, 3], 0.5, ""]]}, "wiring": {"8": []}, "switch_on": {},
          "pool": {"9": {"h.1.attn.v_proj": [[3, 0.5], [1, 0.2]]}}}
    none = components.control_evidence(ev, "text-only", 10)
    assert none["components"] == {} and none["events"] == [] and none["pool"] == {} and none["qa"] == ev["qa"]
    donor = {"positions": [3, 5], "listed": [3], "events": [{"position": 3}], "components": {"3": [[["h.0.mlp.c_fc", 3, 7], 1.0, ""]],
             "1": [[["h.0.mlp.c_fc", 1, 2], 1.0, ""]]}, "wiring": {}, "switch_on": {}, "pool": {"5": {"h.0.mlp.c_fc": [[7, 1.0]]}}}
    moved = components.control_evidence(ev, "shuffled", 10, donor, 6)  # end-aligned: p -> p + 4
    assert moved["positions"] == [7, 9] and moved["components"] == {"7": [[["h.0.mlp.c_fc", 7, 7], 1.0, ""]], "5": [[["h.0.mlp.c_fc", 5, 2], 1.0, ""]]}
    assert moved["pool"] == {"9": {"h.0.mlp.c_fc": [[7, 1.0]]}} and moved["events"] == [{"position": 7}] and moved["qa"] == ev["qa"]
    ids = components.control_evidence(ev, "random-ids", 10, C={"h.1.attn.v_proj": 16}, seed=1)
    c = ids["components"]["8"][0][0][2]
    assert ids["pool"]["9"]["h.1.attn.v_proj"][0][0] == c and sorted(x for x, _ in ids["pool"]["9"]["h.1.attn.v_proj"]) != [1, 3] or c != 3


def test_a_variables_block_claims_every_circuit_node_of_its_site_there():
    nodes = [("h.1.attn.v_proj", 233, 4), ("h.1.attn.v_proj", 233, 9), ("h.1.attn.v_proj", 232, 4), ("h.2.attn.v_proj", 233, 1)]
    got = components.claimed_nodes([("h.1.attn.v_proj", 233, components.BLOCK), ("h.3.mlp.c_fc", 235, 7)], nodes)
    assert got == [("h.1.attn.v_proj", 233, 4), ("h.1.attn.v_proj", 233, 9), ("h.3.mlp.c_fc", 235, 7)]


def test_line_credit_is_each_lines_leave_one_out_change_of_phi_and_compactness():
    """c = delta_phi + g x 0.5 x log2(bits without the line / bits) (goal_eval.line_loo's deletions); a token node's
    declaration 0; a FINAL line its question's E2g credit."""
    down, v = ("h.3.mlp.down_proj", 7, 2), ("h.1.attn.v_proj", 5, 1)
    answer = {"nodes": [[down[0], down[2], down[1]], [v[0], v[2], v[1]], ["tok", 0, 5]], "edges": [[1, 0], [0, "out"], [2, 1]]}
    text = components.render(answer, {1: (True, "' x'")})
    p = components.parse(text)
    name = {nd: k for k, nd in p["ids"].items()}
    loo = {"bits": 40.0, "lines": [{"kind": "node", "line": [list(down)], "delta_phi": 0.5, "bits": 20.0},
                                   {"kind": "node", "line": [list(v)], "delta_phi": 0.0, "bits": 20.0},
                                   {"kind": "edge", "line": [list(v), list(down)], "delta_phi": 0.1, "bits": 30.0},
                                   {"kind": "edge", "line": [list(down), None], "delta_phi": 0.2, "bits": 30.0},
                                   {"kind": "token", "line": [5, list(v)], "delta_phi": 0.3, "bits": 39.0}]}
    e2g = {"per_question": [{"credit": 0.75}]}
    got = dict(map(tuple, components.line_credit(text, p, loo, None, {"g": 0.5}, e2g, None)))
    rows = [r.strip() for r in text.split("\n")]
    at = lambda s: next(k for k, r in enumerate(rows) if r.startswith(s))  # noqa: E731
    assert got[at(f'{name[down]}["')] == pytest.approx(0.5 + 0.25 * math.log2(20 / 40))
    assert got[at(f'{name[v]}["')] == pytest.approx(0.25 * math.log2(20 / 40))  # inert: negative
    assert got[at(f"{name[v]} --> {name[down]}")] == pytest.approx(0.1 + 0.25 * math.log2(30 / 40))
    assert got[at(f"{name[down]} --> OUT")] == pytest.approx(0.2 + 0.25 * math.log2(30 / 40))
    assert got[at(f"t5 --> {name[v]}")] == pytest.approx(0.3 + 0.25 * math.log2(39 / 40))
    assert got[at("t5[tok]")] == 0.0 and got[at("FINAL 1:")] == 0.75


def test_question_asks_for_variables_only_when_the_row_does(tiny):
    nat, ge, task, ev = tiny
    assert components.ASK_VARIABLES not in components.question(task, ev)
    assert components.ASK_VARIABLES in components.question(task, {**ev, "variables": True})


def test_writer_writes_the_teacher_variables_meanings():
    """The writer's lines name and mean each teacher variable (feeds follow a rename); a missing line drops the
    sample; the prompt lists the variables with their input tokens and asks for the lines."""
    vs = [{"name": "v1", "inputs": [4], "group": [["h.2.attn.v_proj", 5, 4]], "feeds": ["v2"]},
          {"name": "v2", "inputs": [7], "group": [["h.3.mlp.down_proj", 2, 7]], "feeds": ["prediction"]}]
    text = 'V1 "copied word": the word after the earlier "for"\nV2 "read": the copy, at the target\n\nThe model copies it.'
    got, prose = components.written_variables(text, vs)
    assert [v["name"] for v in got] == ["copied word", "read"] and got[0]["feeds"] == ["read"] and got[1]["feeds"] == ["prediction"]
    assert got[0]["meaning"] == 'the word after the earlier "for"' and prose == "The model copies it."
    assert components.written_variables('V1 "a": x\nprose', vs) is None
    f = {"text": "x", "target": 7, "target_token": " for", "top": [[" each", 0.5]], "nodes_total": 2, "positions": [4, 7],
         "nodes": [], "evidence": [], "edges": [], "qa": [], "variables": vs, "tokens": {"4": " each", "7": " for"}}
    prompt = components.writer_prompt(f)
    assert "V1 (provisional name 'v1'): inputs 4 (' each'); carried by L2.v#5@4; feeds v2" in prompt
    assert prompt.endswith(components.WRITER_VARIABLES)


def test_pattern_lines_round_trip_the_pair_edges():
    """goal_eval's ["pair", query, key, layer, position] edges render as "q -.->|pair| k" (rest for the remainder, the
    read written when the query is rest, a rest-rest pair as a comment) and parse back to the same entries; v1's
    PATTERN lines are still read."""
    answer = {"nodes": [["h.3.attn.o_proj", 1, 7], ["h.3.attn.q_proj", 283, 7], ["h.3.attn.k_proj", 145, 4], ["h.3.attn.v_proj", 9, 4]],
              "edges": [[0, "out"], [3, 0], [1, 0], [2, 0], ["pair", 1, 2, 3, 7], ["pair", 1, "rest", 3, 7], ["pair", "rest", 2, 3, 7],
                        ["pair", "rest", "rest", 2, 7]]}
    text = components.render(answer)
    assert "-.->|pair| rest" in text and "rest -.->|pair L3@7|" in text and "%% rest -.->|pair L2@7| rest" in text
    p = components.parse(text)
    named = lambda a, i: "rest" if i == "rest" else tuple(a["nodes"][i])  # noqa: E731
    pairs = lambda a: sorted(str((named(a, e[1]), named(a, e[2]), e[3], e[4])) for e in a["edges"] if e[0] == "pair")  # noqa: E731
    assert pairs(p["answer"]) == pairs(answer) and p["malformed"] == 0
    assert components.render(p["answer"]) == text
    written = components.parse("CIRCUIT:\n@7 a=L3.o#1 b=L3.q#283\n@4 c=L3.k#145\nOUT <- a\na <- b\nPATTERN L3.q#283@7 ~ L3.k#145@4\nPATTERN rest ~ c\nPATTERN L3@7: b ~ zz\n")
    assert [e for e in written["answer"]["edges"] if e[0] == "pair"] == [["pair", 1, 2, 3, 7]] and written["malformed"] == 2


def test_question_shows_the_measured_arm_when_present(tiny):
    """measured()'s fields: activations beside each listed subcomponent (ranked by importance x |activation|) and each
    attention head's top positions at the target; the plain question has neither."""
    nat, ge, task, ev = tiny
    plain = components.question(task, ev)
    acts = {q: [1.5 - i for i in range(len(cs))] for q, cs in ev["components"].items()}
    shown = components.question(task, {**ev, "acts": acts, "attention": {"0": [[[3, 0.5], [2, 0.25]]], "1": [[[7, 0.9]]]}})
    assert "ranked by causal importance x |activation|" in shown and "L0: h0 3 0.50, 2 0.25" in shown and "L1: h0 7 0.90" in shown
    assert "; +1.50" in shown and "attention at the target" not in plain and "ranked by" not in plain


def test_variable_blocks_and_instances_round_trip():
    """A carrier block "L1.v @232-233" is the site's whole output at each position (["block", matrix, position]);
    instances inside the blocks are kept apart; written out, by id, or mixed with subcomponents, it parses back."""
    text = "\n".join(["CIRCUIT:", "@235 a=L3.down#2 b=L1.v#433", "OUT <- a", "a <- b", "EXPLANATION:", "VARIABLES:",
                      "V1 \"copied word\": the word; inputs: @232 ' for', @233 ' each'; carried by: L1.v @232-233, L2.v @233, a (instances: L1.v#433@233,b); feeds: PREDICTION",
                      "V2 \"bad\": x; inputs: @1; carried by: L1.v @232 junk!; feeds: V1", "The model copies."])
    p = components.parse(text)
    v, = p["answer"]["variables"]
    assert p["variables_malformed"] == 1
    assert sorted(v["group"]) == sorted([["block", "h.1.attn.v_proj", 232], ["block", "h.1.attn.v_proj", 233],
                                         ["block", "h.2.attn.v_proj", 233], ["h.3.mlp.down_proj", 2, 235]])
    assert sorted(v["instances"]) == sorted([["h.1.attn.v_proj", 433, 233], ["h.1.attn.v_proj", 433, 235]])
    again = components.render(p["answer"])
    assert "carried by: L2.v @233, L1.v @232-233, a (instances: b,L1.v#433@233)" in again
    q = components.parse(again)
    assert q["answer"]["variables"][0]["group"] == v["group"] and sorted(q["answer"]["variables"][0]["instances"]) == sorted(v["instances"])


def test_block_nodes_and_token_edges_round_trip():
    """A block node (["block", site, position], a site's whole output) renders in its position's line as id=L1.v and
    parses back; token edges from a position's tok into a reader there and into the prediction ("OUT <- id") round-trip
    with the pairs; written out, a block is L1.v@233 and a token tok@233. BLOCK is goal_eval's."""
    import goal_eval

    assert components.BLOCK == goal_eval.BLOCK
    answer = {"nodes": [["h.3.mlp.down_proj", 2, 235], ["block", "h.1.attn.v_proj", 233], ["h.2.attn.o_proj", 7, 235],
                        ["h.3.attn.q_proj", 5, 235], ["h.3.attn.k_proj", 9, 233], ["tok", 0, 233], ["tok", 0, 235]],
              "edges": [[0, "out"], [2, 0], [1, 2], [5, 4], [6, 3], [6, "out"], ["pair", 3, 4, 3, 235], ["pair", 3, "rest", 3, 235]]}
    text = components.render(answer)
    assert '["L1.v block"]' in text and "    t233[tok]" in text and "  t235 --> OUT" in text and "-.->|pair|" in text
    p = components.parse(text)
    assert p["malformed"] == 0 and p["malformed_nodes"] == 0
    key = lambda a: (sorted(map(tuple, a["nodes"])), sorted(  # noqa: E731
        str((e[0], *[tuple(a["nodes"][i]) if i != "rest" else i for i in e[1:3]], *e[3:])) if e[0] == "pair"
        else str((tuple(a["nodes"][e[0]]), "out" if e[1] == "out" else tuple(a["nodes"][e[1]]))) for e in a["edges"]))
    assert key(p["answer"]) == key(answer)
    assert components.render(p["answer"]) == text
    written = components.parse("CIRCUIT:\nOUT <- L3.down#2@235,tok@235\nL3.down#2@235 <- L1.v@233\n")
    nodes = [tuple(x) for x in written["answer"]["nodes"]]
    assert ("block", "h.1.attn.v_proj", 233) in nodes and ("tok", 0, 235) in nodes and written["malformed"] == 0


def test_question_lists_the_candidate_pool_when_present(tiny):
    nat, ge, task, ev = tiny
    pool = {"7": {"h.3.mlp.c_fc": [[5, 1.0], [2, 0.42]], "h.0.attn.q_proj": [[1, 0.06]]}}
    shown = components.question(task, {**ev, "pool": pool})
    assert "every VPD-active subcomponent" in shown and "7 (" in shown and "L3.fc 5:1 2:.4" in shown and "L0.q 1:.06" in shown
    assert "every VPD-active subcomponent" not in components.question(task, ev)


def test_fit_pool_drops_the_least_important_entries_until_the_question_fits(tiny):
    nat, ge, task, ev = tiny
    pool = {"7": {"h.3.mlp.c_fc": [[5, 1.0], [2, 0.42]], "h.0.attn.q_proj": [[1, 0.06]]}, "6": {"h.1.attn.v_proj": [[3, 0.2]]}}
    full = len(components.question(task, {**ev, "pool": pool}))
    assert components.fit_pool(task, {**ev, "pool": pool}, len, budget=full)["pool"] == pool
    cut = components.fit_pool(task, {**ev, "pool": pool}, len, budget=full - 5)
    assert cut["pool_cut"] == 1 and cut["pool"] == {"7": {"h.3.mlp.c_fc": [[5, 1.0], [2, 0.42]]}, "6": {"h.1.attn.v_proj": [[3, 0.2]]}}
    assert components.fit_pool(task, {**ev, "pool": pool}, len, budget=0)["pool"] == {}


def test_question_node_cap_is_opt_in_and_included_in_fit(monkeypatch):
    monkeypatch.setattr(components, "token_strings", lambda model, ids: [" x"] * len(ids))
    monkeypatch.setattr(components, "windows", lambda task, ev: [])
    monkeypatch.setattr(components, "listing", lambda strings, windows: " x")
    task = {"model": "mock", "prompts": [{"token_ids": [0], "target_positions": [0],
                                           "model_top": [[[" x", 1.0]]]}]}
    row = {"positions": [], "events": [], "components": {}, "wiring": {}, "qa": [], "pool": {}}
    legacy = components.question(task, row)
    assert "circuit nodes across all positions" not in legacy
    for disabled in (None, 0):
        assert components.question(task, {**row, "node_cap": disabled}) == legacy
    capped = {**row, "node_cap": 266}
    rendered = components.question(task, capped)
    assert "Use at most 266 circuit nodes across all positions; larger circuits are invalid." in rendered
    fitted = components.fit_pool(task, capped, len, budget=len(legacy))
    assert fitted["node_cap"] == 266
    assert fitted["pool_fit"] == {"tokens": len(rendered), "budget": len(legacy), "fits": False}
    assert "node_cap" not in row


def test_fit_pool_reports_nonfit_and_later_fitting_row(monkeypatch):
    monkeypatch.setattr(components, "question", lambda task, row:
                        "q" * (task["base"] + sum(len(cs) for per in row["pool"].values() for cs in per.values())))
    row = {"pool": {"0": {"h.0.attn.v_proj": [[0, 0.1], [1, 0.9]]}}}
    bad = components.fit_pool({"base": 6}, row, len, budget=5)
    assert bad["pool"] == {} and bad["pool_cut"] == 2
    assert bad["pool_fit"] == {"tokens": 6, "budget": 5, "fits": False}
    later = components.fit_pool({"base": 2}, row, len, budget=5)
    assert later["pool"] == row["pool"]
    assert later["pool_fit"] == {"tokens": 4, "budget": 5, "fits": True}
    cut = components.fit_pool({"base": 6}, row, len, budget=7)
    assert cut["pool"]["0"]["h.0.attn.v_proj"] == [[1, 0.9]]
    assert cut["pool_fit"] == {"tokens": 7, "budget": 7, "fits": True}
    assert "pool_fit" not in row  # input evidence is not mutated


def test_rerender_skips_nonfit_pool_row_and_counts_variables_before_fit(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    budget = components.CONTEXT - components.ROOM - 64
    texts = tmp_path / "texts"
    (texts / "vpd4l").mkdir(parents=True)
    rows = [{"task": "mock_unfit", "base": budget - 1}, {"task": "mock_later", "base": budget - 3}]
    for row in rows:
        (texts / "vpd4l" / f"{row['task']}.json").write_text(json.dumps({"id": row["task"]}))
    source, out = tmp_path / "rows.jsonl", tmp_path / "rendered.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(native, "TEXTS", texts)
    monkeypatch.setattr(native, "Native", lambda: object())
    monkeypatch.setattr(components, "_atlas", lambda: None)
    monkeypatch.setattr(components, "_lens", lambda nat: None)
    def candidate_pool(nat, task, row):
        assert row["node_cap"] == 266  # setting precedes both pool construction and fitting
        return {**row, "pool": {"0": {"h.0.attn.v_proj": [[0, 0.5]]}}}
    monkeypatch.setattr(components, "candidate_pool", candidate_pool)
    monkeypatch.setattr(components, "question", lambda task, row:
                        "q" * (row["base"] + (2 if row.get("variables") else 0)
                               + sum(len(cs) for per in row["pool"].values() for cs in per.values())))
    tokenizer = lambda text, **kwargs: {"input_ids": range(len(text))}
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda name: tokenizer)))
    monkeypatch.setattr(sys, "argv", ["components", "rerender", str(source), str(out), "--pool", "--variables", "--node-cap", "266"])
    components.main()
    got, = map(json.loads, out.read_text().splitlines())
    assert got["task"] == "mock_later" and got["variables"] is True
    assert got["node_cap"] == 266
    assert got["pool_fit"] == {"tokens": budget, "budget": budget, "fits": True}
    assert len(got["question"]) == budget
    notice = json.loads(capsys.readouterr().err)
    assert notice == {"event": "rerender_skip", "task": "mock_unfit", "reason": "question_over_budget",
                      "tokens": budget + 1, "budget": budget, "fits": False}
    assert source.read_text() == "".join(json.dumps(row) + "\n" for row in rows)


def test_mermaid_circuit_is_read_strictly():
    """Mermaid is read strictly: a node declared outside its position's subgraph, a label that is no subcomponent, an id
    declared twice, an edge end declared inline, and a stray line are malformed; an edge naming an undeclared id is
    charged as an unknown id; the rest of the answer is still read."""
    text = "\n".join(["CIRCUIT:", "```mermaid", "flowchart LR", '  subgraph P235["@235 \'x\'"]', '    a["L3.down#2"]',
                      '    b["L3.d#2"]', '    a["L3.down#3"]', "    t235[tok]", "  end", '  c["L3.o#1"]', "  a --> OUT((prediction))",
                      "  t235 --> OUT", "  zz --> a", '  q["L2.v#1"] --> a', "  this is not mermaid", "```", "FINAL 1: falls=no top=' x'"])
    p = components.parse(text)
    assert [tuple(x) for x in p["answer"]["nodes"]] == [("h.3.mlp.down_proj", 2, 235), ("tok", 0, 235)]
    assert p["malformed"] == 5 and p["unknown_ids"] == 1 and len(p["answer"]["edges"]) == 2 and p["finals"] == {1: (False, "' x'")}


def test_qa_detail_scores_each_question_and_the_no_change_answer():
    tok = lambda xs: [f"t{x}" for x in xs]  # noqa: E731
    qs = [{"kind": "remove", "falls": True, "top": 5}, {"kind": "swap", "falls": False, "top": 9}, {"kind": "token", "falls": True, "top": 9}]
    got = components.qa_detail(qs, {1: (True, components.shown("t5")), 2: (False, components.shown("t9"))}, tok, "t9")
    assert got == [{"kind": "subcomponent removal", "right": 1.0, "right0": 0.0}, {"kind": "move/swap/insert", "right": 1.0, "right0": 1.0}]


def _circuit_key(a):
    node = lambda i: "rest" if i == "rest" else tuple(a["nodes"][i])  # noqa: E731
    return ({tuple(x) for x in a["nodes"]},
            {(e[0], node(e[1]), node(e[2]), e[3], e[4]) if e[0] == "pair" else (node(e[0]), "out" if e[1] == "out" else node(e[1]))
             for e in a["edges"]})


@pytest.mark.parametrize("which", ["joint_train", "joint_heldout"])
def test_teacher_circuits_round_trip_through_mermaid(which):
    """Every teacher circuit (444 train, the held-out ones) renders to Mermaid and parses back to the same nodes, edges,
    token edges, blocks and pairs, with nothing malformed."""
    d = Path.home() / "mpd-data/graph_oracle/teacher" / which
    files = sorted(f for f in d.glob("text*.json") if f.stem.startswith("text") and f.stem[4:].isdigit()) if d.exists() else []
    if not files:
        pytest.skip(f"no teacher circuits in {d}")
    for f in files:
        a = json.loads(f.read_text())
        for b in (a, _augmented(a)):  # the teachers as written; with token edges, blocks and pairs added
            p = components.parse(components.render(b))
            assert p["malformed"] == 0 and p["unknown_ids"] == 0 and _circuit_key(p["answer"]) == _circuit_key(b), f.name


def _augmented(a):
    """A teacher circuit with what v2 teachers state and v1 ones do not: a token node at every position with its edges
    into that position's query, key, value and MLP-input nodes (and into the prediction at the last position), a block
    beside each position's first node, and query-key pairs (each query with up to two keys of its layer at or before
    it, with the remainder, and a read with no pair)."""
    import goal_eval

    nodes = [list(x) for x in a["nodes"]]
    edges = [list(e) for e in a["edges"]]
    sub = [(i, goal_eval._site(m), int(t), int(c)) for i, (m, c, t) in enumerate(a["nodes"]) if m not in ("tok", "block")]
    for t in sorted({x[2] for x in sub}):
        tok = next((i for i, x in enumerate(nodes) if x[0] == "tok" and x[2] == t), None)
        if tok is None:
            tok = len(nodes)
            nodes.append(["tok", 0, t])
        edges += [[tok, i] for i, n, tt, _ in sub if tt == t and n.endswith(("q_proj", "k_proj", "v_proj", "c_fc"))]
        if t == max(x[2] for x in sub):
            edges.append([tok, "out"])
        site = next(n for _, n, tt, _ in sub if tt == t)
        if ["block", site, t] not in nodes:
            nodes.append(["block", site, t])
    for i, n, t, _ in sub:
        if n.endswith("q_proj"):
            layer = int(n.split(".")[1])
            keys = [j for j, m, tt, _ in sub if m == f"h.{layer}.attn.k_proj" and tt <= t][:2]
            edges += [["pair", i, j, layer, t] for j in keys] + [["pair", i, "rest", layer, t], ["pair", "rest", "rest", layer, t]]
            edges += [["pair", "rest", j, layer, t] for j in keys[:1]]
    return {**a, "nodes": nodes, "edges": edges}


def test_every_rendered_circuit_passes_mermaids_parser(tmp_path):
    """Every teacher circuit, as written and with token edges, blocks and pairs added, rendered and run through
    Mermaid's own parser (experiments/oracle/mermaid_parse.mjs; MERMAID_DIR: where @mermaid-js/mermaid-cli is installed)."""
    import os
    import subprocess

    import native

    mdir = os.environ.get("MERMAID_DIR")
    if not mdir:
        pytest.skip("MERMAID_DIR not set (npm install --prefix DIR @mermaid-js/mermaid-cli@11.4.2)")
    blocks = []
    for which in ("joint_train", "joint_heldout"):
        for f in sorted((Path.home() / "mpd-data/graph_oracle/teacher" / which).glob("text*.json")):
            if not f.stem[4:].isdigit():
                continue
            a = json.loads(f.read_text())
            strings = components.token_strings("vpd4l", native.text(native.TEXTS / "vpd4l" / f"{f.stem}.json")[0])
            for b in (a, _augmented(a)):
                circuit = components.render(b, strings=strings).split("CIRCUIT:\n", 1)[1]
                blocks.append(f"<!-- {which}/{f.stem} -->\n" + circuit[:circuit.index("\n```") + 4])
    if not blocks:
        pytest.skip("no teacher circuits")
    md = tmp_path / "circuits.md"
    md.write_text("\n\n".join(blocks) + "\n")
    got = subprocess.run(["node", str(HERE.parents[1] / "experiments/oracle/mermaid_parse.mjs"), mdir, str(md)], capture_output=True, text=True)
    assert got.returncode == 0, got.stdout[-2000:] + got.stderr[-2000:]


def test_e2g_credits_answers_that_agree_with_the_model_and_the_circuit(tiny):
    """E2g runs the answer's own circuit under each question's intervention: every question gets a credit in [0, 1],
    consistency and accuracy are shares; an answer of no circuit agrees with the model only where nothing changes."""
    nat, ge, task, ev = tiny
    prompt = task["prompts"][0]
    tx = ge.text(prompt["token_ids"], prompt["target_positions"], task=task)
    order = nat._vpd_order(prompt["token_ids"], [7])
    at_t = [nd for nd in order if nd[1] == 7 and nd[0].endswith(("o_proj", "down_proj"))][:3]
    answer = goal_eval.to_json(native.Graph(set(at_t), {}, at_t), "")
    tok = lambda xs: components.token_strings(task["model"], xs)  # noqa: E731
    finals = {n: (q["falls"], components.shown(tok([q["top"]])[0])) for n, q in enumerate(ev["qa"], 1)}  # right about the model
    got = components.e2g_term(ge, tx, task, answer, ev["qa"], finals)
    assert len(got["per_question"]) == len(ev["qa"]) and got["accuracy"] == 1.0
    assert 0.0 <= got["E2g"] <= 1.0 and 0.0 <= got["consistency"] <= 1.0
    assert all(0.0 <= x["credit"] <= 1.0 for x in got["per_question"])
    none = components.e2g_term(ge, tx, task, answer, ev["qa"], {})
    assert none["E2g"] == 0.0 and none["accuracy"] == 0.0


def test_line_credit_gives_each_circuit_line_its_leave_one_out_effect(tiny):
    """reward(lines=True): every node and edge line of the circuit gets a credit (its removal's measured effect less the
    compactness it would gain), a line that does not parse -1, and each question's ANSWER and FINAL lines their E2g."""
    nat, ge, task, ev = tiny
    order = nat._vpd_order(task["prompts"][0]["token_ids"], [7])
    at_t = [nd for nd in order if nd[1] == 7 and nd[0].endswith(("o_proj", "down_proj"))][:3]
    finals = {n: (q["falls"], "'x'") for n, q in enumerate(ev["qa"], 1)}
    text = components.render(goal_eval.to_json(native.Graph(set(at_t), {}, at_t), ""), finals).replace(
        "flowchart LR", "flowchart LR\n  not a line")
    got = components.reward(ge, task, ev, [text], pool={}, lines=True)[0]
    assert got["valid"]
    credit = dict(map(tuple, got["lines"]))
    rows = text.split("\n")
    assert credit[rows.index("  not a line")] == -1.0
    for k, row in enumerate(rows):
        if row.startswith("    ") and "[" in row or " --> " in row:
            assert k in credit, row  # every declaration and edge line
    if ev["qa"]:
        assert credit[rows.index("FINAL 1: " + text.split("FINAL 1: ", 1)[1].split("\n", 1)[0])] == got["e2g"]["per_question"][0]["credit"]
