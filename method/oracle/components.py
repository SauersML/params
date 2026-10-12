"""Component circuits: the oracle's answer in VPD subcomponents, its question, and its reward (#2951 graph oracle).

The oracle reads a text and states the whole-model circuit of vpd4l's prediction in VPD's rank-one subcomponents,
with explicit edges, an English explanation, and answers to intervention questions about the same text. The answer is
plain text (nothing in it is executed):

  EXPLANATION: the model copies the digits after the earlier "327" ...
  CIRCUIT:
  out <- L3.down#120@332, L2.o#77@332
  L2.o#77@332 <- L2.v#902,1514@322
  L2.v#902,1514@322 <- L0.down#77@322
  L0.down#77@322 <- L0.fc#12,40@322
  @322 L2.k#9 L1.k#3,17
  ANSWER 1: the change at 322 removes the copied digit ...
  FINAL 1: falls=yes top=' 7'

A subcomponent is L<layer>.<kind>#<index>@<position> (kind q, k, v, o, fc, down). A circuit line "reader <- writer,
writer" states one edge per writer: an attention or MLP output (o, down) into a later query, key, value or MLP input
(q, k, v, fc) at its position, a value (v) into its layer's attention output (o) at that or a later position (attention
between positions), an MLP input (fc) into its MLP's output (down); "out" is the prediction at the target. Subcomponents
of one matrix at one position may be written together (L2.v#902,1514@322, indices sorted): a line with groups states
an edge from every writer to every reader. Nodes with no stated input are listed by position, one line each:
"@322 L2.v#902,1514 L3.o#5" (a group with its position, without "<-", also states nodes). parse() gives
goal_eval's circuit JSON, and goal_eval.circuit checks every edge against the model (mech.connects); render() writes
the lines backward from "out", grouping readers that share their writers. Token strings are shown as Python reprs of
the target's tokens (special tokens included) and escaped (escape()): no question or answer holds a literal special or
added token of the explainer's tokenizer.

The question (question()) holds the text's tokens, the model's top next tokens, the positions whose changed token
moves the prediction by at least 1% of the text's total (goal_eval's SHARE_FLOOR) with the measured evidence there
(units.evidence: how far the prediction moves, the changed-token and per-layer attention-cut effects), VPD's active
subcomponents at each of those positions and at the target, ranked by VPD (Native._vpd_order) with their atlas facts,
the subcomponents a change of each listed token switches on, the strongest weight-level connections among them
(method/vpd/connectome.py; a static coupling, not a measured effect), and the intervention questions (qa_questions()).
The token embedding is an input node, tok@<position>, a writer only.

The reward (reward()) is goal_eval's scorecard in VPD's semantics combined into one number: the mean of the quality
terms (each at most 1) less SIZE_WEIGHT x size (description bits over the bits of naming every VPD-active
subcomponent at the listed positions and the targets, at most 1); an invalid answer (no CIRCUIT section, a missing
FINAL line, cut at the sampling cap) earns -(1 + SIZE_WEIGHT), below every valid one. Nodes and edges the model does
not have, and circuit lines that do not parse, are dropped and their bits charged in size. Terms: F1 sufficiency, F3 edges,
E1 the reader's bits saved, E2 questions answered right (reward()'s docstring); F2 interchange is a diagnostic.

  components.py questions SPLIT OUT.jsonl [--n N] [--qa-seed K]  question rows (question, positions, evidence, questions;
                                                                K = 1 for a fresh set of questions)
  components.py requestion ROWS.jsonl OUT.jsonl [--qa-seed 1]   the rows with a fresh set of intervention questions
  components.py locality SPLIT OUT.jsonl [--n N]                each text local or not (VPD's target actives alone)
  components.py teacher-vpd ROWS.jsonl OUT_DIR                 the interim teacher (VPD's answer cut to a writable size)
  components.py prepare SPLIT OUT_DIR [--n N] [--lengths LO:HI]  locality, then the row and teacher of the texts kept (at
                                                                least half non-local)
  components.py dataset ROWS.jsonl TEACHER_DIR OUT.jsonl        SFT rows: each question and its teacher's circuit
                                                                (TEACHER_DIR/<task>.json or .txt; nodes alone get
                                                                measured edges) with the measured answers
  components.py evaluate SPLIT ROWS.jsonl --set NAME=SAMPLES [--e1 NAME=E1.jsonl] [--n N] [--out OUT.json]
                                                                the report (evaluate()): first samples headline, best of N
                                                                as an upper bound, paired comparisons, E1 controls
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

KINDS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj", "fc": "c_fc", "down": "down_proj"}
CODES = {v: k for k, v in KINDS.items()}
GROUP_RE = re.compile(r"(?:L(\d+)\.(q|k|v|o|fc|down)#(\d+(?:,\d+)*)|tok)@(\d+)")
_G = r"(?:L\d+\.(?:q|k|v|o|fc|down)#\d+(?:,\d+)*|tok)@\d+"
TOK = "tok"  # the token embedding at a position, an input node: JSON [TOK, 0, position], text tok@<position>
WRITERS_RE = re.compile(rf"\s*{_G}(?:\s*,\s*{_G})*\s*")
BLOCK_RE = re.compile(r"@(\d+)\s+((?:(?:L\d+\.(?:q|k|v|o|fc|down)#\d+(?:,\d+)*|tok)\s*)+)")
MARKDOWN = re.compile(r"\*\*|^[ \t]*(?:#+|[-*])[ \t]+", re.M)
ADDED = re.compile(r"<(?=\||/?(?:think|tool_call|tool_response)>)")  # Qwen's added tokens: <|...|>, <think>, <tool_call>, ...
FINAL_RE = re.compile(r"^\s*FINAL\s+(\d+)\s*:\s*falls\s*=\s*(yes|no)\s+top\s*=\s*(.+?)\s*$", re.M | re.I)
SECTION = re.compile(r"^\s*(EXPLANATION|CIRCUIT|ANSWER\s+\d+|FINAL\s+\d+)\s*:", re.M | re.I)

SIZE_WEIGHT = 0.5
CONVENTION = "live"  # goal_eval's remainders convention for the reward's F1 and N: on, computing on the run's inputs
EVIDENCE_PER_POSITION = 12  # VPD's top active subcomponents listed per position (the target's: twice as many)
WIRING_PER_POSITION = 6  # connectome pairs listed per position
SWITCH_PER_POSITION = 8  # subcomponents a changed token switches on, listed per position
WINDOW = 8  # a question numbers the tokens within this many positions of every position its evidence names
MARK_EVERY = 16  # elsewhere, plain text with a position marker at least this often
QA_PER_KIND = 2
QA_RANKS = 8  # token-change questions draw the new token from the model's alternatives at ranks 0 to 7
ANSWER_BUDGET = 3_500  # tokens of a supervised answer: 4,096 sampled at most, ~600 left for English and reasoning
MAX_PAIRS = 1_024  # connections with_edges measures at most


# ---- the answer's text


def escape(text: str) -> str:
    """Text with every Qwen added-token string broken by a backslash after its "<" ("<|endoftext|>" -> "<\\|endoftext|>",
    "<think>" -> "<\\think>"), so the explainer's tokenizer reads the vpd4l token strings a question lists (the Pile's
    "<|endoftext|>" among them) as text, never as its own special or added tokens. Applied to every question and
    answer text; idempotent."""
    return ADDED.sub(r"<\\", text)


def node_text(nd) -> str:
    """A subcomponent (matrix name, position, index) as L<layer>.<kind>#<index>@<position>."""
    return group_text([nd])


def group_text(nds) -> str:
    """Subcomponents of one matrix at one position as L<layer>.<kind>#<index>,<index>,...@<position>, indices sorted;
    the token embedding at a position as tok@<position>."""
    name, t = nds[0][0], nds[0][1]
    if name == TOK:
        return f"{TOK}@{t}"
    return f"L{name.split('.')[1]}.{CODES[name.split('.')[-1]]}#{','.join(str(c) for c in sorted({nd[2] for nd in nds}))}@{t}"


def group_of(text: str):
    """The subcomponents [(matrix name, position, index), ...] of L<layer>.<kind>#<index>[,<index>...]@<position>, or
    None."""
    import native

    m = GROUP_RE.fullmatch(text.strip())
    if not m:
        return None
    if m[1] is None:
        return [(TOK, int(m[4]), 0)]
    name = native.site_name(int(m[1]), KINDS[m[2]])
    return [(name, int(m[4]), int(c)) for c in m[3].split(",")]


def node_of(text: str):
    """(matrix name, position, index) of L<layer>.<kind>#<index>@<position>, or None."""
    g = group_of(text)
    return g[0] if g is not None and len(g) == 1 else None


def _sections(text: str) -> dict:
    """Section name -> body; ANSWER n and FINAL n keep their numbers."""
    out, marks = {}, list(SECTION.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        key = re.sub(r"\s+", " ", m[1].upper())
        out.setdefault(key, text[m.end():end].strip())
    return out


def parse(text: str) -> dict:
    """An answer's parts: "explanation", "answer" (goal_eval's circuit JSON: nodes [matrix, index, position], edges
    [upstream, downstream | "out"]), "malformed" (circuit lines that do not parse), "circuit" (whether a CIRCUIT
    section exists), "malformed_nodes" (what those lines name, at least one each) and "finals" (question number ->
    (falls, the top token as written)). A circuit line's reader may be
    a group, and so may each writer: the line states an edge from every writer to every reader. Markdown decoration
    (bold markers, heading marks at a line's start) is ignored: no valid line holds it."""
    text = MARKDOWN.sub("", text)
    sec = _sections(text)
    nodes, at, edges, malformed, junk = [], {}, [], 0, 0

    def index(nd):
        if nd not in at:
            at[nd] = len(nodes)
            nodes.append([nd[0], nd[2], nd[1]])
        return at[nd]

    for line in sec.get("CIRCUIT", "").splitlines():
        line = line.strip()
        if not line:
            continue
        block = BLOCK_RE.fullmatch(line)
        if block:  # "@322 L2.v#902,1514 L3.o#5": nodes at one position with no stated input
            for item in block[2].split():
                for nd in group_of(f"{item}@{block[1]}") or []:
                    index(nd)
            continue
        if line.startswith("@"):
            malformed += 1
            junk += max(1, len(line.split()) - 1)
            continue
        reader, arrow, writers = line.partition("<-")
        reader = reader.strip()
        rs = ["out"] if reader == "out" else group_of(reader)
        ws = [] if not writers.strip() else ([nd for g in GROUP_RE.finditer(writers) for nd in group_of(g[0])]
                                             if WRITERS_RE.fullmatch(writers) else None)
        if rs is None or ws is None or (reader == "out" and not ws) or (arrow and not ws):
            malformed += 1
            junk += max(1, len(GROUP_RE.findall(line)))  # what it names, priced as nodes that are not in the graph
            continue
        for r in rs:
            if r != "out":
                index(r)
        for w in ws:
            i = index(w)
            edges += [[i, "out" if r == "out" else at[r]] for r in rs]
    finals = {}
    for m in FINAL_RE.finditer(text):
        finals.setdefault(int(m[1]), (m[2].lower() == "yes", m[3].strip()))
    explanation = sec.get("EXPLANATION", "")
    return {"explanation": explanation, "answer": {"nodes": nodes, "edges": edges, "explanation": explanation},
            "malformed": malformed, "malformed_nodes": junk, "circuit": "CIRCUIT" in sec, "finals": finals}


def _grouped(nds) -> list[str]:
    """Groups of subcomponents by matrix and position, latest position first, then by layer and kind."""
    by = {}
    for nd in nds:
        by.setdefault((nd[0], nd[1]), []).append(nd)
    return [group_text(by[k]) for k in sorted(by, key=_place)]


def blocks(nds) -> list[str]:
    """Nodes with no stated input, one line per position (latest first): "@322 L2.v#902,1514 L3.o#5", each matrix's
    indices together (the position written once: 10-25% fewer tokens than a group per matrix and position)."""
    at = {}
    for nd in nds:
        at.setdefault(nd[1], []).append(nd)
    return [f"@{t} " + " ".join(g.rsplit("@", 1)[0] for g in _grouped(at[t])) for t in sorted(at, reverse=True)]


def _place(nd) -> tuple:
    """A node's (or a (matrix, position) group's) place in rendered text: latest position, then latest layer, then
    kind (q, k, v, o, fc, down), then index."""
    if nd[0] == TOK:
        return (-nd[1], 1, 0) + tuple(nd[2:3])  # the embedding enters the stream first: last in a backward listing
    return (-nd[1], -int(nd[0].split(".")[1]), list(KINDS.values()).index(nd[0].split(".")[-1])) + tuple(nd[2:3])


def render(answer: dict, finals: dict | None = None, reasons: dict | None = None) -> str:
    """The text of a circuit in goal_eval's JSON (to_json's), lines from the prediction backward: "out" first, then
    each group of readers (one matrix at one position, the same writers) the first time one of them is reached
    through its writers, then groups not reached, then nodes named nowhere else; writers grouped by matrix and
    position. finals: question number -> (falls, the top token as written)."""
    import goal_eval

    nodes = [(TOK if m == TOK else goal_eval._site(m), t, c) for m, c, t in answer["nodes"]]
    parents, out = {}, []
    for w, r in answer["edges"]:
        if r == "out":
            out.append(nodes[w])
        else:
            parents.setdefault(nodes[r], set()).add(nodes[w])
    groups = {}
    for r, ws in parents.items():
        groups.setdefault((r[0], r[1], frozenset(ws)), []).append(r)
    group_of_reader = {r: k for k, rs in groups.items() for r in rs}
    lines, named, done = [], set(), set()
    if out:
        lines.append("out <- " + ", ".join(_grouped(set(out))))
        named.update(out)
    queue = sorted(set(out), key=_place)
    while queue:
        k = group_of_reader.get(queue.pop(0))
        if k is None or k in done:
            continue
        done.add(k)
        lines.append(f"{group_text(groups[k])} <- " + ", ".join(_grouped(k[2])))
        named.update(groups[k])
        named.update(k[2])
        queue += sorted(k[2], key=_place)
    for k in sorted(set(groups) - done, key=lambda k: (-k[1], k[0], sorted(c for _, _, c in groups[k]))):
        lines.append(f"{group_text(groups[k])} <- " + ", ".join(_grouped(k[2])))
        named.update(groups[k])
        named.update(k[2])
    lines += blocks(set(nodes) - named)
    text = [f"EXPLANATION: {answer.get('explanation', '')}".rstrip(), "CIRCUIT:"] + lines
    for k in sorted(finals or {}):
        falls, top = finals[k]
        text += [f"ANSWER {k}: {(reasons or {}).get(k, '')}".rstrip(), f"FINAL {k}: falls={'yes' if falls else 'no'} top={top}"]
    return escape("\n".join(text))


def token_strings(model: str, ids: list[int]) -> list[str]:
    """The target model's token strings, its special tokens included (the Pile's "<|endoftext|>" shows as itself, not
    as an empty string; escape() keeps it text for the explainer)."""
    import mech

    tk = mech.tokenizer(model)
    return [tk.decode([int(i)], skip_special_tokens=False) for i in ids]


def shown(token: str) -> str:
    """A token string as questions list it and FINAL lines give it: its Python repr, escaped."""
    return escape(repr(token))


# ---- intervention questions (E2)


def _top(lp) -> tuple[int, float]:
    """(the most likely token, its log-probability) at the last target of [1, len(targets), V] log-probabilities."""
    row = lp[0, -1]
    i = int(row.argmax())
    return i, float(row[i])


def qa_questions(nat, task: dict, ci: dict, positions: list[int], seed: int, per_kind: int = QA_PER_KIND, cuts=None,
                 nodes: list | None = None) -> list[dict]:
    """Intervention questions on a text with the model's measured answers: a token replaced by one of the model's
    alternatives there (ranks 0 to QA_RANKS - 1, at any position before the target), one VPD-active subcomponent
    removed (Native.without; at F2's positions and the targets), one connection cut (cuts: goal_eval.GoalEval.cuts with
    detail on the text, whole-model path patching; among `nodes`, the question's listed subcomponents, the connections
    the model has), and, for a text that copies, changes probing the copy (copy_questions()). Each: {"kind", the
    intervention ("position", "new" | "node" | "edge" | "family", "changes"), "falls": whether the probability of the model's top next token falls below half of what it is, "top": the
    most likely next token after it (id), "bits": log2 of the top token's probability ratio}. Per kind, half the
    questions whose answer is "falls" and half not, where the text has them (a constant answer scores half)."""
    import torch

    import carry

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    t = max(targets)
    rng = random.Random(seed)
    with torch.no_grad():
        top, lp0 = _top(nat.reference([ids], targets))

    def outcome(lp, **what):
        new_top = _top(lp)[0]
        bits = (float(lp[0, -1, top]) - lp0) / math.log(2)
        return {**what, "falls": bits < -1.0, "top": new_top, "bits": bits}

    pool = {"token": [], "remove": []}
    near = [p for p in positions if 0 < p <= t]
    others = [p for p in range(1, t + 1) if p not in near]
    for p in near + rng.sample(others, min(len(others), 2 * per_kind * 2)):
        rank = rng.randrange(QA_RANKS)
        x = carry.alternatives(nat, ids, p, ranks=(rank,))[0]
        with torch.no_grad():
            pool["token"].append(outcome(nat.reference([x], targets), kind="token", position=p, new=x[p], rank=rank))
    where = sorted(set(near) | set(targets))
    active = [nd for nd in nat._vpd_order(ids, targets) if nd[1] in where]
    picks = active[:4 * per_kind] + rng.sample(active[4 * per_kind:], min(max(0, len(active) - 4 * per_kind), 4 * per_kind))
    for nd in picks:
        with torch.no_grad():
            pool["remove"].append(outcome(nat.without(ids, targets, [nd]), kind="remove", node=list(nd),
                                          ci=float(ci[nd[0]][nd[1], nd[2]])))
    pool["copy"] = copy_questions(nat, ids, targets, top, rng, outcome)
    if cuts is not None and nodes:
        import native

        par, outs = native.all_edges(set(map(tuple, nodes)), targets)
        pairs = [(w, r) for r, ws in par.items() for w in ws] + [(w, None) for w in outs]
        pairs = rng.sample(pairs, min(len(pairs), 8 * per_kind))
        pool["cut"] = []
        for (w, r), d in zip(pairs, cuts(pairs) if pairs else []):
            bits = (d["logp_model_top"][-1] - lp0) / math.log(2)
            pool["cut"].append({"kind": "cut", "edge": [list(w), None if r is None else list(r)], "falls": bits < -1.0,
                                "top": int(d["top"][-1]), "bits": bits})
    out = []
    for kind, qs in pool.items():
        yes = [q for q in qs if q["falls"]]
        no = [q for q in qs if not q["falls"]]
        rng.shuffle(yes)
        rng.shuffle(no)
        k_yes = min(len(yes), max(per_kind // 2, per_kind - len(no)))
        out += yes[:k_yes] + no[:per_kind - k_yes]
    rng.shuffle(out)
    return out


def copy_questions(nat, ids: list[int], targets: list[int], top: int, rng, outcome) -> list[dict]:
    """Questions on the model's copying, for a text whose prediction copies: an earlier [A][B], A the token at the
    target and B the model's top next token, such that replacing every such B by another token makes B's probability
    fall below half (measured: the prediction depends on the earlier pair, not only on the bigram). Each changes tokens
    and measures the model (outcome): "every source" (that test itself); "one source", only the latest B replaced when
    the pair recurs (does one copy suffice?), or "source" when it occurs once; "prefix", the A before the latest B
    replaced (is the match still found?); "distractor near" and "distractor far", a competing [A][X] written over two
    tokens just before the target or early in the text. None for a text that does not copy."""
    import torch

    import carry

    t = max(targets)
    sources = [s for s in range(1, t) if ids[s] == top and ids[s - 1] == ids[t]]
    if not sources:
        return []

    def measure(changes, family):
        x = list(ids)
        for q, new in changes:
            x[q] = new
        with torch.no_grad():
            return outcome(nat.reference([x], targets), kind="copy", family=family, changes=[list(c) for c in changes])

    def other(q):  # one of the model's likelier other tokens at q, never the copied one
        return rng.choice([z for z in (x[q] for x in carry.alternatives(nat, ids, q, ranks=tuple(range(QA_RANKS)))) if z != top])

    every = measure([(s, other(s)) for s in sources], "every source")
    if not every["falls"]:
        return []
    s = sources[-1]
    out = ([every, measure([(s, other(s))], "one source")] if len(sources) > 1 else [{**every, "family": "source"}]) \
        + [measure([(s - 1, other(s - 1))], "prefix")]
    taken = {q for src in sources for q in (src - 1, src)} | {t}
    for family, span in (("distractor near", range(max(2, t - 6), t - 1)), ("distractor far", range(2, max(2, sources[0] - 1)))):
        free = [r for r in span if r - 1 not in taken and r not in taken]
        if free:
            r = rng.choice(free)
            out.append(measure([(r - 1, ids[t]), (r, other(r))], family))
    return out


def qa_text(q: dict, n: int, strings: list[str], top: str, tokens_of) -> str:
    """Question n in words."""
    if q["kind"] == "token":
        what = f"the token at position {q['position']}, {strings[q['position']]!r}, were replaced by {tokens_of([q['new']])[0]!r}"
    elif q["kind"] == "copy" and len(q["changes"]) == 1:
        (r, z), = q["changes"]
        what = f"the token at position {r}, {strings[r]!r}, were replaced by {tokens_of([z])[0]!r}"
    elif q["kind"] == "copy":
        what = ("the tokens at positions " + ", ".join(f"{r} ({strings[r]!r})" for r, _ in q["changes"]) + " were replaced by "
                + ", ".join(repr(tokens_of([z])[0]) for _, z in q["changes"]) + " respectively")
    elif q["kind"] == "cut":
        w, r = q["edge"]
        what = (f"the connection {node_text(tuple(w))} -> {'out (the prediction)' if r is None else node_text(tuple(r))} were cut "
                "(the reader computed without that writer's output, everything else unchanged)")
    else:
        what = f"subcomponent {node_text(tuple(q['node']))} were removed (set to zero, everything else unchanged)"
    return (f"Question {n}: if {what}, would the probability of {top} fall below half of what it is, and what would the "
            "most likely next token be?")


def qa_score(questions: list[dict], finals: dict, tokens_of) -> float | None:
    """The share of questions answered right: half for whether the top token's probability falls below half, half for
    the most likely next token after the intervention (as shown() writes it, exactly)."""
    if not questions:
        return None
    right = 0.0
    for n, q in enumerate(questions, 1):
        if n not in finals:
            continue
        falls, top = finals[n]
        right += 0.5 * (falls == q["falls"]) + 0.5 * (top == shown(tokens_of([q["top"]])[0]))
    return right / len(questions)


# ---- the question


ASK = ("How does {model} compute its prediction of the next token after the target of this text? State the circuit in "
       "VPD subcomponents: a subcomponent is one rank-one slice of a weight matrix at a position, written "
       "L<layer>.<kind>#<index>@<position> (kind q, k, v, o, fc or down; for example L2.v#1514@322). Write one line per "
       "reader, \"<reader> <- <writer>, <writer>\", each writer an edge into the reader: an attention or MLP output (o, "
       "down) into a later query, key, value or MLP input (q, k, v, fc) at its position; a value (v) into its layer's "
       "attention output (o) at that or a later position; an MLP input (fc) into its MLP's output (down). \"out\" is the "
       "prediction. The circuit should be what makes the prediction: with only its subcomponents on and every other one off, "
       "the model should still predict what it does, and with them taken out of VPD's active subcomponents it should "
       "not; fewer subcomponents and edges are better. Answer with a line EXPLANATION: (in plain English, what the model does), "
       "then CIRCUIT: and its lines, then for each question ANSWER <n>: (your reasoning in plain English) and "
       "FINAL <n>: falls=yes or no top=<the token, quoted as the tokens are>. Subcomponents of one matrix at one position "
       "may be written together, L2.v#902,1514@322; a line with groups states an edge from every writer to every reader.")


def wiring(conn, nodes: list, targets: list[int], k: int) -> list[tuple]:
    """The k strongest weight-level connections (connectome cosine) among `nodes` that the model has (mech.connects):
    (writer, reader, cosine)."""
    import native

    names = conn.names
    have = set(nodes)
    pairs = []
    for w in nodes:
        kind = w[0].split(".")[-1]
        table = {"o_proj": "residual", "down_proj": "residual", "v_proj": "ov", "c_fc": "mlp"}.get(kind)
        if table is not None:
            got = conn.partners(w[0], w[2], f"{table}.downstream")
            for s, c, cos in zip(got["site"].tolist(), got["index"].tolist(), got["cosine"].tolist()):
                if not 0 <= s < len(names):
                    continue
                for r in (nd for nd in have if nd[0] == names[s] and nd[2] == c):
                    if native.connects(w, r, targets):
                        pairs.append((w, r, cos))
    best = {}
    for w, r, cos in pairs:
        if abs(cos) > abs(best.get((w, r), 0.0)):
            best[(w, r)] = cos
    return sorted(((w, r, c) for (w, r), c in best.items()), key=lambda x: -abs(x[2]))[:k]


def facts(atlas, nd) -> str:
    """A subcomponent's atlas facts in a few words (method/vpd/atlas.py, measured on Pile text): the tokens its write
    raises most (its lens, for the sites that write the residual stream or feed one that does) and the tokens it is
    most often active on (the tokens at its top contexts)."""
    from collections import Counter

    e = atlas.entry(nd[0], nd[2])
    out = []
    if e["lens"]:
        out.append("raises " + " ".join(repr(t) for t, _ in e["lens"]["raised"][:3]))
    on = Counter(c["token"] for c in e["contexts"]).most_common(2)
    if on:
        out.append("active on " + " ".join(repr(t) for t, _ in on))
    return "; ".join(out)


def _atlas():
    """The component atlas, or None while it is not built."""
    sys.path.insert(0, str(HERE.parents[1]))
    from method.vpd import atlas

    return atlas.Atlas() if (atlas.DEFAULT / "atlas.safetensors").exists() else None


def switch_on(nat, ids: list[int], targets: list[int], p: int, ci: dict, k: int = SWITCH_PER_POSITION) -> list:
    """The subcomponents VPD switches on when the token at p changes: active (causal importance above zero) on a
    changed prompt (the model's 3 likeliest other tokens there, carry.alternatives) and inactive on the text, at p and
    the later positions up to the last target (earlier ones are VPD's causal-importance network reading ahead: nothing
    computes differently there). The k switched on by the most of the 3, then by mean causal importance:
    [[node, count], ...]."""
    import carry

    t = max(targets)
    count, total = {}, {}
    for x in carry.alternatives(nat, ids, p):
        cx = nat.importance(x)
        for n in nat.names:
            on = (cx[n] > 0) & ~(ci[n] > 0)
            on[:p] = False
            on[t + 1:] = False
            for q, c in on.nonzero().tolist():
                count[(n, q, c)] = count.get((n, q, c), 0) + 1
                total[(n, q, c)] = total.get((n, q, c), 0.0) + float(cx[n][q, c])
    ranked = sorted(count, key=lambda nd: (-count[nd], -total[nd] / count[nd], _place(nd)))
    return [[list(nd), count[nd]] for nd in ranked[:k]]


def evidence(nat, ge, task: dict, conn=None, qa_seed: int | None = None, atlas=None) -> dict:
    """One text's question data: F2's positions (goal_eval.Text with the text's own seed), the measured evidence there
    (units.events at those positions; its other events are the English reader's), VPD's top active subcomponents per
    position with their atlas facts (facts(); none without an atlas), their strongest wiring, the subcomponents a
    change of each listed position's token switches on (switch_on()), the intervention questions, and the count of
    VPD-active subcomponents at F2's positions and the targets (the size term's scale)."""
    import native
    import units

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    tx = ge.text(ids, targets, task=task)
    positions = [p for p, _ in tx.f2]
    listed = [p for p in positions if p not in targets]
    events = units.events(nat, task, k=len(listed)) if listed else []
    order = nat._vpd_order(ids, targets)
    where = sorted(set(positions) | set(targets))
    comps = {p: [] for p in where}
    for nd in order:
        if nd[1] in comps and len(comps[nd[1]]) < EVIDENCE_PER_POSITION * (2 if nd[1] in targets else 1):
            comps[nd[1]].append(nd)
    conn = conn or _connectome()
    wires = {}
    for p in where:  # connections touching the position (at a target: among the targets' subcomponents)
        t_nodes = [nd for q in sorted(set(targets) | {p}) for nd in comps[q]]
        pairs = [x for x in wiring(conn, t_nodes, targets, len(t_nodes) ** 2) if p in (x[0][1], x[1][1])]
        wires[p] = [[list(w), list(r), c] for w, r, c in pairs[:WIRING_PER_POSITION]]
    switched = {str(p): switch_on(nat, ids, targets, p, tx.ci) for p in listed}
    active = int(sum(int((tx.ci[n][where] > 0).sum()) for n in nat.names))
    listed_nodes = [nd for p in where for nd in comps[p]]
    qa = qa_questions(nat, task, tx.ci, positions, native.task_seed(task["id"]) if qa_seed is None else qa_seed,
                      cuts=lambda pairs: ge.cuts(tx, pairs, detail=True), nodes=listed_nodes)
    return {"task": task["id"], "positions": positions, "listed": listed, "events": events,
            "components": {str(p): [[list(nd), float(tx.ci[nd[0]][nd[1], nd[2]]), facts(atlas, nd) if atlas else ""] for nd in comps[p]]
                           for p in where},
            "wiring": {str(p): wires[p] for p in where}, "switch_on": switched, "qa": qa, "active": active}


def windows(task: dict, ev: dict, radius: int = WINDOW) -> set[int]:
    """The positions a question numbers: within radius of the first position and of each position its evidence names (the listed positions,
    whose relays sit just after them, the targets, every subcomponent, switch-on and wiring position, every
    intervention question's position) and of each earlier occurrence of the target's token followed by the predicted
    token (where a copy reads from)."""
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    t = max(targets)
    named = set(ev["positions"]) | set(targets) | {0}  # position 0: the attention sink most attention reads
    named |= {int(q) for q in ev["components"]} | {nd[1] for cs in ev["components"].values() for nd, *_ in cs}
    named |= {nd[1] for cs in ev.get("switch_on", {}).values() for nd, _ in cs}
    named |= {nd[1] for ws in ev["wiring"].values() for w, r, _ in ws for nd in (w, r)}
    for q in ev["qa"]:
        named |= ({q["position"]} if q["kind"] == "token" else {q["node"][1]} if q["kind"] == "remove"
                  else {nd[1] for nd in q["edge"] if nd is not None} if q["kind"] == "cut" else {r for r, _ in q["changes"]})
    strings = token_strings(task["model"], ids)
    top = p["model_top"][0][0][0]
    named |= {s + d for s in range(t) if ids[s] == ids[t] and strings[s + 1] == top for d in (0, 1)}  # copy sources [A][B]
    return {q for c in named for q in range(max(0, c - radius), min(len(ids), c + radius + 1))}


def listing(strings: list[str], numbered: set[int], every: int = MARK_EVERY) -> str:
    """Token strings with their positions where numbered ("392:' the'"), elsewhere as plain text in runs, each run and
    every multiple of `every` inside one opened by a position marker ("[128]' the cat sat'"): any position stays
    locatable by counting from the nearest marker."""
    out, i, T = [], 0, len(strings)
    while i < T:
        if i in numbered:
            out.append(f"{i}:{strings[i]!r}")
            i += 1
            continue
        j = i
        while j < T and j not in numbered:
            j += 1
        k = i
        while k < j:
            nxt = min(j, (k // every + 1) * every)
            out.append(f"[{k}]{''.join(strings[k:nxt])!r}")
            k = nxt
        i = j
    return " ".join(out)


def question(task: dict, ev: dict) -> str:
    """The oracle's question for one text from its evidence()."""
    import units

    model = task["model"]
    p = task["prompts"][0]
    strings = token_strings(model, p["token_ids"])
    targets = p["target_positions"]
    t = max(targets)
    top_tok = p["model_top"][0][0][0]
    top = ", ".join(f"{tok!r} {q:.2f}" for tok, q in p["model_top"][0][:3])
    lines = [ASK.format(model=model), "tokens (numbered near the positions below; elsewhere plain text, [n] marking position n): "
             + listing(strings, windows(task, ev)),
             f"target: position {t}; the model's most probable next tokens there: {top}",
             "positions where a changed token moves the prediction by at least 1% of the total, most first: "
             + ", ".join(f"{q} ({strings[q]!r})" for q in ev["positions"])]
    if ev["events"]:
        lines += ["measured at those positions (log2 change of the top token's probability):"] + units.evidence(ev["events"], ev["listed"], strings)
    lines.append("VPD's active subcomponents (causal importance, and on Pile text the tokens its write raises and the "
                 "tokens it is active on), most important first, and the strongest weight-level connections among them "
                 "(cosine; a static coupling, not a measured effect):")
    for q in sorted(ev["components"], key=int):
        cs = ev["components"][q]
        lines.append(f"{q} ({strings[int(q)]!r}): " + ", ".join(f"{node_text(tuple(nd))} ({c:.2f}{'; ' + f if f else ''})" for nd, c, f in cs))
        if ev.get("switch_on", {}).get(q):
            lines.append("  switched on when its token changes (of the 3 likeliest other tokens): "
                         + ", ".join(f"{node_text(tuple(nd))} ({k}/3)" for nd, k in ev["switch_on"][q]))
        if ev["wiring"].get(q):
            lines.append("  wiring: " + ", ".join(f"{node_text(tuple(w))} -> {node_text(tuple(r))} ({c:+.2f})" for w, r, c in ev["wiring"][q]))
    if ev["qa"]:
        tok = lambda ids: token_strings(model, ids)  # noqa: E731
        lines += [qa_text(q, n, strings, repr(top_tok), tok) for n, q in enumerate(ev["qa"], 1)]
    return escape("\n".join(lines))


def _connectome():
    import connectome

    return connectome.Connectome()


# ---- the reward


def http_reader(url: str, base: str, model: str = "vpd4l"):
    """reader.Reader over a frozen base model served by vLLM's OpenAI server at url (…/v1/completions): the chat
    template without thinking, one token, its 20 likeliest."""
    import httpx
    from transformers import AutoTokenizer

    import reader

    tk = AutoTokenizer.from_pretrained(base)

    def generate(prompts: list[str]) -> list[dict]:
        if not prompts:
            return []
        chat = [tk.apply_chat_template([{"role": "user", "content": escape(q)}], add_generation_prompt=True, enable_thinking=False,
                                       tokenize=False) for q in prompts]
        r = httpx.post(url, json={"model": base, "prompt": chat, "max_tokens": 1, "temperature": 0.0, "logprobs": 20},
                       timeout=600.0)
        r.raise_for_status()
        out = [None] * len(prompts)
        for c in r.json()["choices"]:
            out[c["index"]] = dict(c["logprobs"]["top_logprobs"][0])
        return out

    return reader.Reader(generate, lambda ids: token_strings(model, ids))  # the explainer's token listing, escaped in generate



def english(text: str) -> str:
    return _sections(text).get("EXPLANATION", "")


def e1_bits(reader, task: dict, ev: dict, texts: list[str]) -> list[float | None]:
    """Each answer's E1: the bits its escaped English saves the frozen reader on the text's reader events
    (units.reader_events: other positions and the listed positions' second tokens, never the oracle's evidence)."""
    import units

    return reader.bits(task, [escape(english(x)) for x in texts], units.reader_events(ev["events"]))


def dropped_bits(answer: dict, g, d: dict, positions: int, subcomponents: int) -> float:
    """The description bits of what an answer states that is not in its graph (goal_eval.circuit's defects d: a node
    outside the model or the text, a connection the model does not have), each priced as native.Graph.bits prices what
    it keeps (a node one choice among positions x subcomponents, an edge its two ends among the answer's nodes, an edge
    into the prediction its writer): it was written and explains nothing."""
    import goal_eval

    nodes = answer.get("nodes", [])
    n = max(len(nodes), 2)
    kept = set(g.out)

    def node(i):
        m, c, t = nodes[i]
        try:
            return (goal_eval._site(m), int(t), int(c))
        except (ValueError, KeyError):
            return None

    d_out = sum(1 for w, r in answer.get("edges", []) if r == "out" and (w >= len(nodes) or node(w) not in kept))
    d_inner = max(0, d["invalid_edges"] - d_out)
    return d["invalid_nodes"] * math.log2(positions * subcomponents) + (2 * d_inner + d_out) * math.log2(n)


def reward(ge, task: dict, ev: dict, texts: list[str], *, reader=None, e1: list | None = None, truncated: list | None = None,
           size_weight: float = SIZE_WEIGHT, convention: str = CONVENTION, cache: dict | None = None) -> list[dict]:
    """Per answer text on one task: {"reward", "valid", each term, "size", the dropped nodes and edges, goal_eval's
    row}. An answer is invalid (reward -(1 + size_weight), below every valid one) when it has no CIRCUIT section, a
    question has no FINAL line, or it was cut at the sampling cap (truncated). Nodes and edges the model does not have
    are dropped and their description bits charged in size (dropped_bits), and so is a circuit line that does not
    parse (each subcomponent it names, at least one, priced as a node outside the graph). A valid
    answer earns the mean of its terms less size_weight x size ("circuit_reward": the same over the circuit's own terms
    F1, N and F3, for comparing with circuits that give no English or answers), size its description bits over the scale (VPD's active
    subcomponents at the listed positions and the targets), at most 1. Terms, from goal_eval in VPD's semantics with
    the remainders' convention, both in orders of magnitude of KL (linear recovery is dominated by the empty circuit's
    ~100 bits: an answer 20 bits from the model recovers 0.8 linearly, 0.2 in log): F1, goal_eval's f1 (ln(KL_empty /
    KL_C) / ln(KL_empty / KL_VPD) in [0, 1], between the empty circuit and VPD's own answer); N, goal_eval's n (what
    VPD's own answer loses without the circuit, ln(KL_{VPD minus C} / KL_VPD) / ln(KL_empty / KL_VPD) in [0, 1]); F3, the
    AUC of stated edges' cuts in the circuit's run (0 without a stated edge); E1, the reader's bits saved (in [-1, 1];
    with a reader or e1); E2, the share of questions answered right. A term goal_eval cannot give (no causal-importance
    network for VPD's bounds) is left out of the mean. reader: a reader.Reader for E1, or e1: each text's reader bits
    computed by the caller (e1_bits(); a Tinker reader lives with the policy); ev: the text's evidence(); cache: task
    -> goal_eval.Text, kept between calls."""
    import goal_eval

    nat = ge.nat
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    tok = lambda xs: token_strings(task["model"], xs)  # noqa: E731
    total = sum(nat.C.values())
    parsed = [parse(x) for x in texts]
    graphs, defects, valid = [], [], []
    for i, p in enumerate(parsed):
        g, d = goal_eval.circuit(p["answer"], targets, nat.C, len(ids))
        graphs.append(g)
        junk = p["malformed_nodes"] * math.log2(nat.positions(targets) * total)  # a line that does not parse: written, explains nothing
        defects.append((d["invalid_nodes"], d["invalid_edges"], dropped_bits(p["answer"], g, d, nat.positions(targets), total) + junk))
        valid.append(p["circuit"] and all(n in p["finals"] for n in range(1, len(ev["qa"]) + 1)) and not (truncated and truncated[i]))
    keep = [i for i, v in enumerate(valid) if v]  # an invalid answer's reward is fixed whatever it measures: not run
    if keep:
        tx = cache.get(task["id"]) if cache is not None else None
        if tx is None:
            tx = ge.text(ids, targets, task=task)
            if cache is not None:
                cache[task["id"]] = tx
    bits = {}
    if e1 is not None:
        bits = {texts[i]: e1[i] for i in keep}
    elif reader is not None and keep:
        uniq = list(dict.fromkeys(texts[i] for i in keep))
        bits = dict(zip(uniq, e1_bits(reader, task, ev, uniq)))
    ge.hooks = {"E1": (lambda task, g, x: bits.get(x)) if (reader is not None or e1 is not None) else None,
                "E2": lambda task, g, x: qa_score(ev["qa"], parse(x)["finals"], tok)}
    row_of = dict(zip(keep, ge.score(tx, [graphs[i] for i in keep], [texts[i] for i in keep]))) if keep else {}
    scale = ev["active"] * math.log2(nat.positions(targets) * total)
    out = []
    for i, (p, (d_nodes, d_edges, d_bits)) in enumerate(zip(parsed, defects)):
        row, terms, size, r = row_of.get(i), {}, None, -(1.0 + size_weight)
        circ = r
        if row is not None:
            terms["F1"] = row[f"f1_{convention}"]
            terms["N"] = row[f"n_{convention}"]
            terms["F3"] = row["f3"]
            terms["E1"] = None if row["E1"] is None else max(-1.0, min(1.0, row["E1"]))
            terms["E2"] = row["E2"]
            have = [v for v in terms.values() if v is not None]
            size = min(1.0, (row["bits"] + d_bits) / scale) if scale > 0 else 0.0
            r = sum(have) / len(have) - size_weight * size
            own = [terms[k] for k in ("F1", "N", "F3") if terms[k] is not None]
            circ = sum(own) / len(own) - size_weight * size
        out.append({"reward": r, "circuit_reward": circ, "valid": valid[i], **{f"term_{k}": v for k, v in terms.items()}, "size": size,
                    "malformed": p["malformed"], "dropped_nodes": d_nodes, "dropped_edges": d_edges,
                    "finals": len(p["finals"]), "truncated": bool(truncated and truncated[i]), "row": row})
    return out


# ---- supervised targets


def target(answer: dict, ev: dict, model: str) -> str:
    """The supervised answer to a question: the teacher's circuit (goal_eval's JSON) and explanation, and for each
    question the model's measured answer (no reasoning: the teacher has none to give)."""
    finals = {n: (q["falls"], shown(token_strings(model, [q["top"]])[0])) for n, q in enumerate(ev["qa"], 1)}
    return render(answer, finals)


def with_edges(ge, task: dict, ev: dict, answer: dict, cache: dict | None = None, max_pairs: int = MAX_PAIRS) -> dict:
    """A teacher circuit given as nodes alone, with edges: the connections the model has among its nodes (and from
    its residual writers at the targets into the prediction), ordered by the measured effect of cutting each one
    alone (goal_eval.cuts), the prefix (none, 1, 2, 4, ..., all) whose answer earns the most reward on the text. With
    more than max_pairs connections, those measured are the ones into the prediction and then those whose ends come
    earliest in VPD's order (Native._vpd_order)."""
    import goal_eval
    import native

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    g, _ = goal_eval.circuit(answer, targets, ge.nat.C, len(ids))
    par, out = native.all_edges(g.node_set(), targets)
    pairs = [(w, None) for w in out] + [(w, r) for r, ws in par.items() for w in ws]
    if len(pairs) > max_pairs:
        rank = {nd: i for i, nd in enumerate(ge.nat._vpd_order(ids, targets))}
        far = len(rank)
        pairs = sorted(pairs, key=lambda e: (e[1] is not None, max(rank.get(e[0], far), rank.get(e[1], -1) if e[1] else -1)))[:max_pairs]
    if not pairs:
        return answer
    tx = ge.text(ids, targets, task=task)
    ranked = [e for _, e in sorted(zip(ge.cuts(tx, pairs), pairs), key=lambda x: -x[0])]
    ks = sorted({0, len(ranked)} | {2 ** j for j in range(len(ranked).bit_length()) if 2 ** j < len(ranked)})
    cands = []
    for k in ks:
        h = native.Graph(g.node_set(), {}, [])
        for w, r in ranked[:k]:
            h.out.append(w) if r is None else h.parents.setdefault(r, []).append(w)
        cands.append(goal_eval.to_json(h, answer.get("explanation", "")))
    got = reward(ge, task, {**ev, "qa": []}, [render(c) for c in cands], cache=cache)
    return cands[max(range(len(cands)), key=lambda i: got[i]["reward"])]


def vpd_teacher(ge, task: dict, ev: dict, tokens, budget: int = ANSWER_BUDGET, cache: dict | None = None) -> tuple[dict, dict]:
    """The interim teacher: VPD's own answer cut to a size an answer can be written at. Prefixes of VPD's order
    (Native._vpd_order) of 16, 32, 64, ... nodes up to the largest whose answer text (with the measured FINAL lines)
    fits `budget` tokens of the explainer's tokenizer (tokens: text -> count), each scored by circuit_reward (F1, N
    and F3 less 0.5 x size) as nodes alone; the best gets measured edges (with_edges), cut back to the budget if they
    overrun it. (answer JSON, {"k", "k_max", "circuit_reward", terms})."""
    import goal_eval
    import native

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    order = ge.nat._vpd_order(ids, targets)
    fits = lambda k: tokens(target(goal_eval.to_json(native.Graph(order[:k])), ev, task["model"])) <= budget  # noqa: E731
    lo, hi = 0, len(order)  # the largest k that fits, by bisection
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
    ks = sorted({k for k in (2 ** j for j in range(4, lo.bit_length())) if k < lo} | {lo})
    cands = [goal_eval.to_json(native.Graph(order[:k])) for k in ks]
    got = reward(ge, task, {**ev, "qa": []}, [render(c) for c in cands], cache=cache)
    i = max(range(len(ks)), key=lambda j: got[j]["circuit_reward"])
    answer = with_edges(ge, task, ev, cands[i], cache)
    while answer["edges"] and tokens(target(answer, ev, task["model"])) > budget:
        answer = {**answer, "edges": answer["edges"][:len(answer["edges"]) // 2]}
    final = reward(ge, task, {**ev, "qa": []}, [render(answer)], cache=cache)[0]
    return answer, {"task": task["id"], "k": ks[i], "k_max": lo, "edges": len(answer["edges"]),
                    "circuit_reward": final["circuit_reward"], **{k: final.get(k) for k in ("term_F1", "term_N", "term_F3", "size")},
                    "tokens": tokens(target(answer, ev, task["model"]))}


def dataset(rows: Path, teacher_dir: Path, out: Path, ge=None) -> int:
    """SFT rows ({"prompt": [user message], "completion": [assistant message], "info": {"task"}}, units_sft.py's
    format) for the questions that have a teacher answer: TEACHER_DIR/<task>.json (goal_eval's circuit JSON; nodes
    without edges get with_edges(), which needs ge) or <task>.txt (this module's text)."""
    import native

    n = 0
    with out.open("w") as f:
        for r in map(json.loads, rows.read_text().splitlines()):
            js, txt = teacher_dir / f"{r['task']}.json", teacher_dir / f"{r['task']}.txt"
            if js.exists():
                answer = json.loads(js.read_text())
            elif txt.exists():
                answer = parse(txt.read_text())["answer"]
            else:
                continue
            task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
            if answer.get("nodes") and not answer.get("edges"):
                if ge is None:
                    raise ValueError(f"{js} states no edges: with_edges() needs the model (pass ge)")
                answer = with_edges(ge, task, r, answer)
            f.write(json.dumps({"prompt": [{"role": "user", "content": r["question"]}],
                                "completion": [{"role": "assistant", "content": target(answer, r, task["model"])}],
                                "info": {"task": r["task"]}}) + "\n")
            f.flush()
            n += 1
    return n


# ---- locality


LOCAL = 0.9  # a text is local when VPD's active subcomponents at its targets alone reach this recovered_log


def locality(ge, task: dict, rounds: int = 2) -> dict:
    """Whether a text's prediction is local: a subset of VPD's active subcomponents at the target positions, found by
    measured removal, reaches f1 (log recovery, remainders live) >= LOCAL with every other subcomponent off. All the
    target's actives together understate it (they cancel each other when the context is off: text9000's 63 reach
    0.43 bits, 12 of them 0.077). Each round scores every member's removal from the current set, orders the members
    by how much their removal raises f1, and keeps the best set along that order's nested removals (depths 1, 2, 4,
    ... from either end, then every depth within a quarter of the best, at least 4); the set is the starting point of
    the next round. For a local prediction a target-only circuit is the truth; training sets need
    non-local texts too. {"task", "local", "f1", "subset", "target_actives"}."""
    import carry
    import goal_eval

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    tx = ge.text(ids, targets, task=task)
    at = [nd for nd in ge.nat._vpd_order(ids, targets) if nd[1] in targets]
    e, v = ge._bounds(tx, "live")

    def without(base: dict, nds) -> dict:  # a mask with nodes switched off, copying only the matrices they are in
        m = dict(base)
        for n, t, c in nds:
            if m[n] is base[n]:
                m[n] = base[n].clone()
            m[n][t, c] = False
        return m

    def f1s(masks):  # f1 alone (goal_eval's live runs and bounds), 16 sets per call
        out = []
        for s0 in range(0, len(masks), 16):
            kl = carry.kl(tx.ref, ge.runs(ids, targets, masks[s0:s0 + 16], True)).tolist()
            out += [None if v is None else min(1.0, max(0.0, goal_eval.recovered_log(e, c, v))) for c in kl]
        return out

    base = carry.masks(ge.nat, tx.T, at)
    best, cur = (f1s([base])[0] if at else None), list(at)
    best_set = list(at)
    for _ in range(rounds):
        if best is None or best >= LOCAL or len(cur) < 2:
            break
        drop = f1s([without(base, [nd]) for nd in cur])
        if max(drop) >= LOCAL:  # one removal already reaches it
            i = max(range(len(cur)), key=lambda j: drop[j])
            best, best_set = drop[i], cur[:i] + cur[i + 1:]
            break
        order = [cur[i] for i in sorted(range(len(cur)), key=lambda i: -drop[i])]
        ks = sorted({k for j in range(len(order).bit_length()) for k in (2 ** j, len(order) - 2 ** j) if 0 < k < len(order)})
        got = dict(zip(ks, f1s([without(base, order[:k]) for k in ks])))  # nested removals at doubling depths from either end
        k = max(got, key=got.get)
        near = [j for j in range(max(1, k - max(4, k // 4)), min(len(order) - 1, k + max(4, k // 4)) + 1) if j not in got]
        got |= dict(zip(near, f1s([without(base, order[:j]) for j in near])))  # then every depth near the best
        k = max(got, key=got.get)
        if got[k] <= best:
            break
        best, cur = got[k], order[k:]
        best_set, base = cur, without(base, order[:k])
    return {"task": task["id"], "local": best is not None and best >= LOCAL, "f1": best, "subset": len(best_set),
            "target_actives": len(at)}


# ---- evaluation


def structure(ge, tx, g, order: list, convention: str = CONVENTION) -> dict:
    """Diagnostics of a circuit C (goal_eval runs in VPD's semantics; recovered_log under the convention):
      redundancy: VPD's active subcomponents without C ("rest", in VPD's order) run alone; when they still reach
         goal_eval.GOAL, a second circuit disjoint from C, the shortest prefix of the rest in VPD's order that reaches
         it (found by bisection over prefix lengths), its size, and the necessity of C and it together (does removing
         both from VPD's answer finally break the prediction?);
      minimality: the share of C's nodes whose removal alone lowers C's own recovered_log (a circuit should hold no
         idle node; each variant is compared with C run in the same call)."""
    import goal_eval
    import native

    key = f"recovered_log_{convention}"
    C = sorted(g.node_set())
    rest = [nd for nd in order if nd not in g.node_set()]

    def rec(nodes):
        return ge.sufficiency(tx, [native.Graph(nodes)])[0][key]

    out = {"rest_recovered_log": rec(rest), "second": None}
    if out["rest_recovered_log"] is not None and out["rest_recovered_log"] >= goal_eval.GOAL:
        lo, hi = 0, len(rest)
        while lo < hi:
            mid = (lo + hi) // 2
            if (rec(rest[:mid]) or 0.0) >= goal_eval.GOAL:
                hi = mid
            else:
                lo = mid + 1
        union = ge.sufficiency(tx, [native.Graph(C + rest[:lo])])[0]
        out["second"] = {"size": lo, "union_necessity_log": union[f"n_{convention}"]}
    if C:
        lower = 0
        for s0 in range(0, len(C), 15):  # bounded memory: C itself and 15 variants per call, each against its own call's C
            rows = ge.sufficiency(tx, [native.Graph(C)] + [native.Graph(C[:i] + C[i + 1:]) for i in range(s0, min(len(C), s0 + 15))])
            if rows[0][key] is None:
                return {**out, "minimal_share": None}
            lower += sum(r[key] < rows[0][key] for r in rows[1:])
        out["minimal_share"] = lower / len(C)
    return out




def samples_of(spec: Path) -> dict:
    """task -> [answer text, ...], first sample first, of an answer set: a sampler's JSONL ({"task", "sample", "text",
    "finish"}; a sample cut at the cap carries finish "length") or a directory of <task>.txt and <task>.<k>.txt. Each
    text comes with whether it was cut: [(text, truncated), ...]."""
    out = {}
    if spec.is_dir():
        for f in sorted(spec.iterdir()):
            m = re.fullmatch(r"(.+?)(?:\.(\d+))?\.txt", f.name)
            if m:
                out.setdefault(m[1], []).append((int(m[2] or 0), f.read_text(), False))
    else:
        for r in map(json.loads, spec.read_text().splitlines()):
            out.setdefault(r["task"], []).append((int(r.get("sample", 0)), r["text"], r.get("finish") == "length"))
    return {t: [(x, cut) for _, x, cut in sorted(v, key=lambda e: e[0])] for t, v in out.items()}


def baseline_texts(ge, task: dict, sizes: list[int]) -> dict:
    """name -> answer text of the search-free baselines, each with every connection the model has among its nodes
    stated: "empty", and "vpd@k", VPD's first k active subcomponents in its order (Native._vpd_order) for each k."""
    import goal_eval
    import native

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    out = {"empty": render({"nodes": [], "edges": []})}
    order = ge.nat._vpd_order(ids, targets)
    for k in sorted(set(sizes)):
        if k > 0:
            nodes = set(order[:k])
            par, outs = native.all_edges(nodes, targets)
            out[f"vpd@{k}"] = render(goal_eval.to_json(native.Graph(nodes, par, outs)))
    return out


def _mean_counting_invalid(answers: list, key: str) -> float | None:
    """A term's mean over answers, an invalid answer counted as 0 and a valid one without the term (not measured, as
    E1 without a reader) left out; None when no answer has it."""
    vals = [0.0 if not a["valid"] else a.get(key) for a in answers]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals and any(a["valid"] and a.get(key) is not None for a in answers) else None


def _paired(a: list, b: list) -> dict:
    """Paired per-question comparison of two methods' rewards: the mean difference, its standard error, wins, losses."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    se = (sum((x - mean) ** 2 for x in d) / (n - 1) / n) ** 0.5 if n > 1 else None
    return {"n": n, "mean": mean, "se": se, "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d)}


def evaluate(ge, split: str, rows: dict, sets: dict, n: int, e1: dict | None = None, labels: dict | None = None,
             diagnostics: bool = False, log=print) -> dict:
    """An evaluation of answer sets (name -> samples_of()) on the split's first n texts with question rows, every
    answer scored by reward() (an invalid one at its fixed reward, never skipped). Per set: the headline over each
    question's FIRST sample (policy performance: the mean reward, each term's mean with invalid answers counted as 0,
    the valid share, size, dropped nodes and edges), "best of N (upper bound)" (each question's best sample), and
    paired comparisons with every other set (reward) and with the baselines (empty, and VPD's own answer at the first
    sample's node count; circuit_reward, as they give no English or answers) on the same questions. A set with no answer on a question scores it as invalid. e1: set -> task ->
    {"own": [bits per sample], "shuffled": bits, "reader_b": bits} (tinker_train.py readers): the reward's E1 uses
    "own"; the controls are reported (no English is 0 bits by construction). labels: task -> locality() row: every
    figure is also reported for local and non-local texts apart. diagnostics: structure() of each first sample (a
    second disjoint circuit, minimality), costly."""
    import native

    invalid = -(1.0 + SIZE_WEIGHT)
    per = {s: [] for s in sets}
    base = {}
    for p in native.tasks(split)[:n]:
        if p.stem not in rows:
            continue
        task = json.loads(p.read_text())
        ev = rows[p.stem]
        for name, samples in sets.items():
            got = samples.get(p.stem, [])
            if not got:
                per[name].append({"task": p.stem, "samples": [{"reward": invalid, "valid": False}]})
                continue
            bits = (e1 or {}).get(name, {}).get(p.stem, {}).get("own")
            cache = {}
            scored = reward(ge, task, ev, [x for x, _ in got], truncated=[cut for _, cut in got], e1=bits, cache=cache)
            entry = {"task": p.stem, "samples": [{k: v for k, v in r.items() if k != "row"} | {"nodes": (r["row"] or {}).get("nodes")}
                                                 for r in scored]}
            if diagnostics and scored[0]["valid"]:
                import goal_eval

                prompt = task["prompts"][0]
                g, _ = goal_eval.circuit(parse(got[0][0])["answer"], prompt["target_positions"], ge.nat.C, len(prompt["token_ids"]))
                entry["structure"] = structure(ge, cache[task["id"]], g, ge.nat._vpd_order(prompt["token_ids"], prompt["target_positions"]))
            per[name].append(entry)
        sizes = [x["samples"][0].get("nodes") or 0 for x in (per[s][-1] for s in sets)]
        names = baseline_texts(ge, task, sizes)
        for (b, text), r in zip(names.items(), reward(ge, task, {**ev, "qa": []}, list(names.values()))):
            base.setdefault(p.stem, {})[b] = r["circuit_reward"]
        log(p.stem + ": " + ", ".join(f"{s} {per[s][-1]['samples'][0]['reward']:+.3f}" for s in sets), flush=True)
    report = {}
    groups = {"all": None}
    if labels:
        groups |= {"local": True, "non-local": False}
    for group, want in groups.items():
        report[group] = _report(per, base, e1, labels, want)
    return {"report": report, "per_question": per, "baselines": base}


def _report(per: dict, base: dict, e1: dict | None, labels: dict | None, want) -> dict:
    """evaluate()'s figures over the questions of one locality group (want None: every question)."""
    keep = (lambda t: True) if want is None else (lambda t: t in labels and labels[t]["local"] == want)
    per = {name: [q for q in qs if keep(q["task"])] for name, qs in per.items()}
    report = {}
    for name, qs in per.items():
        if not qs:
            continue
        first = [q["samples"][0] for q in qs]
        terms = sorted({k for f in first for k in f if k.startswith("term_")})
        report[name] = {
            "questions": len(qs), "valid": sum(f["valid"] for f in first) / max(len(first), 1),
            "reward": sum(f["reward"] for f in first) / max(len(first), 1),
            **{k[5:]: _mean_counting_invalid(first, k) for k in terms},
            "size": sum((f.get("size") or 0.0) for f in first if f["valid"]) / max(sum(f["valid"] for f in first), 1),
            "dropped_edges": sum(f.get("dropped_edges", 0) for f in first) / max(len(first), 1),
            "best of N (upper bound)": sum(max(x["reward"] for x in q["samples"]) for q in qs) / max(len(qs), 1),
            "samples per question": sum(len(q["samples"]) for q in qs) / max(len(qs), 1),
            "paired": {other: _paired([q["samples"][0]["reward"] for q in qs], [o["samples"][0]["reward"] for o in per[other]])
                       for other in per if other != name},
        }
        cr = lambda q: q["samples"][0].get("circuit_reward", q["samples"][0]["reward"])  # noqa: E731
        report[name]["paired"] |= {f"{other} (circuit reward)": _paired([cr(q) for q in qs], [cr(o) for o in per[other]])
                                   for other in per if other != name}
        circ = [q["samples"][0].get("circuit_reward", q["samples"][0]["reward"]) for q in qs]  # baselines give no English or answers
        report[name]["circuit_reward"] = sum(circ) / max(len(circ), 1)
        report[name]["paired"]["empty (circuit reward)"] = _paired(circ, [base[q["task"]]["empty"] for q in qs])
        matched = [(c, base[q["task"]].get(f"vpd@{q['samples'][0].get('nodes')}")) for c, q in zip(circ, qs)]
        matched = [(a, b) for a, b in matched if b is not None]
        if matched:
            report[name]["paired"]["vpd@matched (circuit reward)"] = _paired([a for a, _ in matched], [b for _, b in matched])
        if e1 and name in e1:
            ctl = [e1[name][q["task"]] for q in qs if q["task"] in e1[name]]
            report[name]["E1 controls"] = {k: sum(c[k] for c in ctl if c.get(k) is not None) / max(sum(c.get(k) is not None for c in ctl), 1)
                                           for k in ("shuffled", "reader_b")} | {"own (first sample)": sum(c["own"][0] for c in ctl) / max(len(ctl), 1),
                                                                                 "none": 0.0}
        st = [q["structure"] for q in qs if q.get("structure")]
        if st:
            two = [x["second"] for x in st if x["second"]]
            report[name]["structure"] = {
                "circuits": len(st), "second disjoint circuit found": len(two) / len(st),
                "second circuit size": sum(x["size"] for x in two) / len(two) if two else None,
                "necessity_log of both": sum(x["union_necessity_log"] or 0.0 for x in two) / len(two) if two else None,
                "minimal share": sum(x["minimal_share"] for x in st if x.get("minimal_share") is not None) / max(sum(x.get("minimal_share") is not None for x in st), 1)}
    report["questions"] = len(next(iter(per.values()), []))
    return report


# ---- command line


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("questions")
    q.add_argument("split", choices=("train", "heldout", "hard"))
    q.add_argument("out", type=Path)
    q.add_argument("--n", type=int, default=10)
    q.add_argument("--offset", type=int, default=0)
    q.add_argument("--stride", type=int, default=1)
    q.add_argument("--qa-seed", type=int, default=0, help="added to each text's seed for its questions: 0 the discovery "
                   "set (training, checkpoint selection), 1 a fresh set for reports")
    rq = sub.add_parser("requestion", help="the same rows with a fresh set of intervention questions (only they are measured)")
    rq.add_argument("rows", type=Path)
    rq.add_argument("out", type=Path)
    rq.add_argument("--qa-seed", type=int, default=1)
    lc = sub.add_parser("locality", help="each text local or not (locality())")
    lc.add_argument("split", choices=("train", "heldout", "hard"))
    lc.add_argument("out", type=Path)
    lc.add_argument("--n", type=int, default=50)
    lc.add_argument("--offset", type=int, default=0)
    lc.add_argument("--stride", type=int, default=1)
    lc.add_argument("--lengths", default="0:100000", help="LO:HI: only texts of LO <= tokens < HI (PyTorch's MPS keeps "
                    "per-shape caches, ~20-40 MB of host memory per sequence length: a process over a band of lengths stays bounded)")
    pr = sub.add_parser("prepare", help="per text: locality, and for the texts kept, the question row and the interim teacher")
    pr.add_argument("split", choices=("train", "heldout", "hard"))
    pr.add_argument("out", type=Path, help="directory: locality.jsonl, rows.jsonl, teacher/<task>.json, teacher/summary.jsonl")
    pr.add_argument("--n", type=int, default=400)
    pr.add_argument("--lengths", default="0:100000")
    pr.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    vt = sub.add_parser("teacher-vpd", help="the interim teacher: VPD's own answer cut to a writable size (vpd_teacher())")
    vt.add_argument("rows", type=Path)
    vt.add_argument("out", type=Path, help="directory: <task>.json (circuit JSON) and summary.jsonl")
    vt.add_argument("--tasks", help="comma-separated task ids (default: every row)")
    vt.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    d = sub.add_parser("dataset")
    d.add_argument("rows", type=Path)
    d.add_argument("teacher", type=Path)
    d.add_argument("out", type=Path)
    s = sub.add_parser("evaluate")
    s.add_argument("split", choices=("train", "heldout", "hard"))
    s.add_argument("rows", type=Path, help="questions' JSONL (a fresh QA seed for a report, the discovery one for selection)")
    s.add_argument("--set", action="append", default=[], help="NAME=SAMPLES (a sampler's JSONL or a directory of <task>[.<k>].txt)")
    s.add_argument("--e1", action="append", default=[], help="NAME=E1.jsonl (tinker_train.py readers)")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--labels", type=Path, help="locality JSONL: every figure also for local and non-local texts")
    s.add_argument("--diagnostics", action="store_true", help="structure() of each first sample (slow)")
    s.add_argument("--out", type=Path)
    args = ap.parse_args()
    import goal_eval
    import native

    if args.cmd == "dataset":
        node_only = any(json.loads(f.read_text()).get("edges") in (None, []) for f in sorted(args.teacher.glob("*.json")))
        print(dataset(args.rows, args.teacher, args.out, goal_eval.GoalEval() if node_only else None), "rows", flush=True)
        return

    ge = goal_eval.GoalEval()
    nat = ge.nat
    if args.cmd == "locality":
        done = {json.loads(x)["task"] for x in args.out.read_text().splitlines()} if args.out.exists() else set()
        lo, hi = map(int, args.lengths.split(":"))
        with args.out.open("a") as f:
            for p in native.tasks(args.split)[:args.n][args.offset::args.stride]:
                if p.stem not in done and lo <= len(native.text(p)[0]) < hi:
                    r = locality(ge, json.loads(p.read_text()))
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    print(r, flush=True)
        return
    if args.cmd == "prepare":
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(args.tokenizer)
        count = lambda text: len(tk(text, add_special_tokens=False)["input_ids"])  # noqa: E731
        (args.out / "teacher").mkdir(parents=True, exist_ok=True)
        files = {k: args.out / f"{k}.jsonl" for k in ("locality", "rows")}
        seen = {k: [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else [] for k, f in files.items()}
        labels = {r["task"]: r for r in seen["locality"]}
        kept = {r["task"] for r in seen["rows"]}
        conn, atlas = _connectome(), _atlas()
        lo, hi = map(int, args.lengths.split(":"))
        for p in native.tasks(args.split)[:args.n]:
            if not lo <= len(native.text(p)[0]) < hi or (p.stem in labels and (p.stem in kept or labels[p.stem]["local"])):
                continue
            task = json.loads(p.read_text())
            if p.stem not in labels:
                labels[p.stem] = locality(ge, task)
                with files["locality"].open("a") as f:
                    f.write(json.dumps(labels[p.stem]) + "\n")
            local_kept = sum(labels[t]["local"] for t in kept)
            if labels[p.stem]["local"] and local_kept >= len(kept) - local_kept:  # at least half the kept texts non-local
                continue
            ev = evidence(nat, ge, task, conn, atlas=atlas)
            row = {**ev, "question": question(task, ev)}
            answer, info = vpd_teacher(ge, task, row, count, cache={})
            (args.out / "teacher" / f"{p.stem}.json").write_text(json.dumps(answer))
            with (args.out / "teacher" / "summary.jsonl").open("a") as f:
                f.write(json.dumps({**info, "local": labels[p.stem]["local"]}) + "\n")
            with files["rows"].open("a") as f:
                f.write(json.dumps(row) + "\n")
            kept.add(p.stem)
            print(p.stem, "local" if labels[p.stem]["local"] else "non-local", info, flush=True)
        return
    if args.cmd == "teacher-vpd":
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(args.tokenizer)
        count = lambda text: len(tk(text, add_special_tokens=False)["input_ids"])  # noqa: E731
        args.out.mkdir(parents=True, exist_ok=True)
        want = set(args.tasks.split(",")) if args.tasks else None
        for r in map(json.loads, args.rows.read_text().splitlines()):
            if (want and r["task"] not in want) or (args.out / f"{r['task']}.json").exists():
                continue
            task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
            answer, info = vpd_teacher(ge, task, r, count, cache={})
            (args.out / f"{r['task']}.json").write_text(json.dumps(answer))
            with (args.out / "summary.jsonl").open("a") as f:
                f.write(json.dumps(info) + "\n")
            print(info, flush=True)
        return
    if args.cmd == "requestion":
        with args.out.open("w") as f:
            for r in map(json.loads, args.rows.read_text().splitlines()):
                task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
                ids, targets = task["prompts"][0]["token_ids"], task["prompts"][0]["target_positions"]
                tx = ge.text(ids, targets, task=task)
                r["qa"] = qa_questions(nat, task, nat.importance(ids), r["positions"], native.task_seed(task["id"]) + args.qa_seed,
                                       cuts=lambda pairs: ge.cuts(tx, pairs, detail=True),
                                       nodes=[tuple(nd) for cs in r["components"].values() for nd, *_ in cs])
                f.write(json.dumps({**r, "question": question(task, r)}) + "\n")
                f.flush()
                print(r["task"], len(r["qa"]), "questions", flush=True)
        return
    if args.cmd == "questions":
        conn, atlas = _connectome(), _atlas()
        done = set()
        if args.out.exists():
            done = {json.loads(x)["task"] for x in args.out.read_text().splitlines() if x.strip()}
        with args.out.open("a") as f:
            for p in native.tasks(args.split)[:args.n][args.offset::args.stride]:
                if p.stem in done:
                    continue
                task = json.loads(p.read_text())
                ev = evidence(nat, ge, task, conn, atlas=atlas, qa_seed=native.task_seed(task["id"]) + args.qa_seed)
                f.write(json.dumps({**ev, "question": question(task, ev)}) + "\n")
                f.flush()
                print(p.stem, len(ev["positions"]), "positions", ev["active"], "active", flush=True)
        return
    rows = {r["task"]: r for r in map(json.loads, args.rows.read_text().splitlines())}
    sets = {k: samples_of(Path(v)) for k, v in (x.split("=", 1) for x in args.set)}
    e1 = {}
    for k, v in (x.split("=", 1) for x in args.e1):
        e1[k] = {r["task"]: r for r in map(json.loads, Path(v).read_text().splitlines())}
    labels = {r["task"]: r for r in map(json.loads, args.labels.read_text().splitlines())} if args.labels else None
    out = evaluate(ge, args.split, rows, sets, args.n, e1, labels, args.diagnostics)
    print(json.dumps(out["report"], indent=1), flush=True)
    if args.out:
        args.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
