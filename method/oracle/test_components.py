"""Checks of component answers (components.py): the text format both ways, question scoring, and the reward on a tiny
random model (test_goal_eval's).

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_components.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import components  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
from test_goal_eval import tiny_native  # noqa: E402

ANSWER = """EXPLANATION: the value at 3 is copied to the last position.
CIRCUIT:
@5 a=L1.o#1 b=L1.down#2
@3 c=L1.k#3 d=L1.v#4 e=L0.fc#5 f=L0.down#0
OUT <- a,b
a <- d
d <- f
f <- e
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


def test_escape_keeps_special_strings_text():
    tk = pytest.importorskip("transformers").AutoTokenizer
    try:
        tok = tk.from_pretrained("Qwen/Qwen3-8B")
    except OSError:
        pytest.skip("Qwen3's tokenizer is not in the Hugging Face cache")
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


class NoWiring:
    names = []

    def partners(self, name, idx, kind):
        return {k: np.empty(0) for k in ("site", "index", "head", "raw", "cosine")}


@pytest.fixture(scope="module")
def tiny():
    nat = tiny_native()
    ge = goal_eval.GoalEval(nat, changes=4, random_pairs=4)
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
    got = components.reward(ge, task, ev, [good, bad, missing, good], truncated=[False, False, False, True])
    assert got[0]["valid"] and invalid < got[0]["reward"] <= 1.0
    assert all(got[0][f"term_{k}"] is None or 0.0 <= got[0][f"term_{k}"] <= 1.0 for k in ("F1", "N", "F3"))
    tok = lambda xs: components.token_strings(task["model"], xs)  # noqa: E731
    right0 = sum(0.5 * (not q["falls"]) + 0.5 * (tok([q["top"]])[0] == " the") for q in ev["qa"]) / max(len(ev["qa"]), 1)
    if ev["qa"] and right0 < 1.0:  # every "falls" right and no top token "x" (0.5 right), measured from no change
        assert got[0]["term_E2"] == pytest.approx(max(-1.0, min(1.0, (0.5 - right0) / (1.0 - right0))))
    else:
        assert got[0]["term_E2"] is None
    assert got[1]["valid"] and got[1]["dropped_edges"] == 1 and got[1]["size"] > got[0]["size"]
    assert (not got[2]["valid"]) == bool(ev["qa"]) and not got[3]["valid"] and got[3]["reward"] == invalid


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
    got = components.reward(ge, task, ev, texts)
    assert all(r["valid"] for r in got)  # lines that do not parse are dropped and charged, not fatal
    assert got[0]["size"] == 1.0 and got[1]["dropped_edges"] == 5 * T and got[2]["malformed"] == 400 and got[2]["size"] == 1.0
    assert min(r["reward"] for r in got) > -(1 + components.SIZE_WEIGHT)
