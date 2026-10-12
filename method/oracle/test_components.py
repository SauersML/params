"""Checks of component answers (components.py): the text format both ways, question scoring, and the reward on a tiny
random model (test_goal_eval's).

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_components.py -q
"""

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
    p = components.parse(ANSWER)
    assert p["malformed"] == 0 and p["circuit"]
    assert p["explanation"] == "the value at 3 is copied to the last position."
    assert p["finals"] == {1: (True, "' the'")}
    assert len(p["answer"]["nodes"]) == 6 and len(p["answer"]["edges"]) == 5
    text = components.render(p["answer"], p["finals"], {1: "the copied value is gone."})
    assert text.strip() == ANSWER.strip()  # one grammar: ids in position blocks, edges by id
    assert edges(components.parse(text)) == edges(p)


def test_written_out_nodes_still_parse():
    text = "EXPLANATION: x\nCIRCUIT:\nout <- L1.down#2,9@5\nL1.down#2,9@5 <- L1.fc#1,7@5, L0.down#3@5\nL1.k#4,6@3\n"
    p = components.parse(text)
    assert p["malformed"] == 0
    assert len(p["answer"]["nodes"]) == 7
    assert len(p["answer"]["edges"]) == 2 + 2 * 3  # two into the prediction, every writer into both readers
    again = components.parse(components.render(p["answer"]))
    assert edges(again) == edges(p) and "=" in components.render(p["answer"])
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
    assert p["malformed"] == 2
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


def test_size_term_counts_orders_of_magnitude():
    scale = 2.0 ** 20 - 1
    assert components.size_term(0.0, scale) == 0.0 and components.size_term(scale, scale) == 1.0
    assert components.size_term(10 * scale, scale) == 1.0  # at most 1
    step = [components.size_term(2.0 ** k - 1, scale) - components.size_term(2.0 ** (k - 1) - 1, scale) for k in (5, 10, 15)]
    assert all(abs(x - 1 / 20) < 1e-12 for x in step)  # every doubling costs the same


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
    invalid = -(1 + components.SIZE_WEIGHT)
    order = nat._vpd_order(task["prompts"][0]["token_ids"], [7])
    at_t = [nd for nd in order if nd[1] == 7 and nd[0].endswith(("o_proj", "down_proj"))][:3]
    fc = next(nd for nd in order if nd[1] == 7 and nd[0] == "h.1.mlp.c_fc")
    down = (native.site_name(1, "down_proj"), 7, 0)
    g = native.Graph(set(at_t) | {fc, down}, {down: [fc]}, at_t + [down])
    finals = {n: (q["falls"], "'x'") for n, q in enumerate(ev["qa"], 1)}
    good = components.render(goal_eval.to_json(g, "writes at the target"), finals)
    bad = good.replace("OUT <- ", "OUT <- L0.v#0@7,")  # a value never writes the residual stream: dropped, charged
    missing = components.render(goal_eval.to_json(g, "writes at the target"))
    got = components.reward(ge, task, ev, [good, bad, missing, good], truncated=[False, False, False, True], pool={})
    assert got[0]["valid"] and invalid < got[0]["reward"] <= 1.0
    assert all(got[0][f"term_{k}"] is None or 0.0 <= got[0][f"term_{k}"] <= 1.0 for k in ("C", "F1", "N", "F3"))
    tok = lambda xs: components.token_strings(task["model"], xs)  # noqa: E731
    right0 = sum(0.5 * (not q["falls"]) + 0.5 * (tok([q["top"]])[0] == " the") for q in ev["qa"]) / max(len(ev["qa"]), 1)
    if ev["qa"] and right0 < 1.0:  # every "falls" right and no top token "x" (0.5 right), measured from no change
        assert got[0]["term_E2"] == pytest.approx(max(-1.0, min(1.0, (0.5 - right0) / (1.0 - right0))))
    else:
        assert got[0]["term_E2"] is None
    assert got[1]["valid"] and got[1]["dropped_edges"] == 1 and got[1]["size"] > got[0]["size"]
    assert (not got[2]["valid"]) == bool(ev["qa"]) and not got[3]["valid"] and got[3]["reward"] == invalid
    import edit_eval

    goals = [{"kind": "flip", "token_id": 5, "token": " the", "target": 7}]
    asked = {**ev, "edits": goals}
    best = {edit_eval.EditEval.key(task["id"], goals[0]): 4.0}
    edited = good + "\nEDIT 1: scale " + good.split("@7 ", 1)[1].split("=", 1)[0] + " 0"
    with_e3 = components.reward(ge, task, asked, [edited, good], ee=FakeEdits(), edit_best=best, pool={})
    assert with_e3[0]["term_E3"] == 0.5 and not with_e3[1]["valid"]  # 2 bits of the teacher's 4; no EDIT line: invalid
    terms = [with_e3[0][f"term_{k}"] for k in ("C", "N", "E2", "E3") if with_e3[0][f"term_{k}"] is not None]  # F1, F3, E1: reported
    assert with_e3[0]["reward"] == pytest.approx(sum(terms) / len(terms) - components.SIZE_WEIGHT * with_e3[0]["size"])


def test_invalid_below_every_valid_answer(tiny):
    """However large a valid answer and however badly it does, it earns more than an invalid one."""
    nat, ge, task, ev = tiny
    T = len(task["prompts"][0]["token_ids"])
    every = native.Graph({(n, t, c) for n in nat.names for t in range(T) for c in range(nat.C[n])})
    par, out = native.all_edges(every.node_set(), [7])
    huge = native.Graph(every.node_set(), par, out)
    wrong = {n: (not q["falls"], "'x'") for n, q in enumerate(ev["qa"], 1)}
    junk = "\n".join(f"L3.v#{c}@{t} <- L0.q#{c}@{t}" for t in range(T) for c in range(5))  # no such connection: all dropped
    garbage = "\n".join(["Reader: L1.o#1@3 <- Writer: L0.down#2@3", "the attention reads the previous token"] * 200)
    texts = [components.render(goal_eval.to_json(huge, ""), wrong),
             components.render({"nodes": [], "edges": []}, wrong).replace("CIRCUIT:", "CIRCUIT:\n" + junk),
             components.render({"nodes": [], "edges": []}, wrong).replace("CIRCUIT:", "CIRCUIT:\n" + garbage)]
    got = components.reward(ge, task, ev, texts, pool={})
    assert all(r["valid"] for r in got)  # lines that do not parse are dropped and charged, not fatal
    assert got[0]["size"] == 1.0 and got[1]["dropped_edges"] == 5 * T and got[2]["malformed"] == 400 and got[2]["size"] == 1.0
    assert min(r["reward"] for r in got) > -(1 + components.SIZE_WEIGHT)


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
        self.asked = None

    def score(self, tx, vs, graph=None):
        self.asked = vs
        return {"e4": 0.6, "circuit": {"e4": 0.9}, "gap": 0.3}


def test_e4_is_the_scorers_e4_with_a_malformed_variable_scoring_zero():
    C = {"h.2.attn.v_proj": 16, "h.2.attn.o_proj": 16}
    one = {"name": "a", "meaning": "", "inputs": [4], "group": [["h.2.attn.v_proj", 5, 4]], "feeds": ["prediction"]}
    two = {"name": "b", "meaning": "", "inputs": [6], "group": [["h.2.attn.o_proj", 1, 7]], "feeds": ["prediction"]}
    sc = FakeScorer()
    out = components.e4_term(sc, None, None, {"variables": [one, two]}, 0, C)
    assert out == {"E4": 0.6, "model": 0.6, "circuit": 0.9, "gap": 0.3} and [v["name"] for v in sc.asked] == ["a", "b"]
    outside = {**two, "group": [["h.2.attn.o_proj", 99, 7]]}  # no such subcomponent; and one line that did not parse
    assert components.e4_term(sc, None, None, {"variables": [one, outside]}, 1, C)["E4"] == pytest.approx(0.6 / 3)
    assert components.e4_term(sc, None, None, {}, 0, C)["E4"] == 0.0


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
    """goal_eval's ["pair", query, key, layer, position] edges render as PATTERN lines by id (rest for the remainder)
    and parse back to the same entries; the written-out form without the L@ prefix takes the query's read; a line
    naming no query when it gives no read, or an unknown id, is malformed."""
    answer = {"nodes": [["h.3.attn.o_proj", 1, 7], ["h.3.attn.q_proj", 283, 7], ["h.3.attn.k_proj", 145, 4], ["h.3.attn.v_proj", 9, 4]],
              "edges": [[0, "out"], [3, 0], [1, 0], [2, 0], ["pair", 1, 2, 3, 7], ["pair", 1, "rest", 3, 7], ["pair", "rest", "rest", 2, 7]]}
    text = components.render(answer)
    assert "PATTERN L3@7: " in text and " ~ rest" in text and "PATTERN L2@7: rest ~ rest" in text
    p = components.parse(text)
    named = lambda a, i: "rest" if i == "rest" else tuple(a["nodes"][i])  # noqa: E731
    pairs = lambda a: sorted(str((named(a, e[1]), named(a, e[2]), e[3], e[4])) for e in a["edges"] if e[0] == "pair")  # noqa: E731
    assert pairs(p["answer"]) == pairs(answer) and p["malformed"] == 0
    assert components.render(p["answer"]) == text
    written = components.parse("CIRCUIT:\n@7 a=L3.o#1 b=L3.q#283\n@4 c=L3.k#145\nOUT <- a\na <- b\nPATTERN L3.q#283@7 ~ L3.k#145@4\nPATTERN rest ~ c\nPATTERN L3@7: b ~ zz\n")
    assert [e for e in written["answer"]["edges"] if e[0] == "pair"] == [["pair", 1, 2, 3, 7]] and written["malformed"] == 2
