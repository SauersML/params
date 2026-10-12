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
  L2.k#9@322
  ANSWER 1: the change at 322 removes the copied digit ...
  FINAL 1: falls=yes top=' 7'

A subcomponent is L<layer>.<kind>#<index>@<position> (kind q, k, v, o, fc, down). A circuit line "reader <- writer,
writer" states one edge per writer: an attention or MLP output (o, down) into a later query, key, value or MLP input
(q, k, v, fc) at its position, a value (v) into its layer's attention output (o) at that or a later position (attention
between positions), an MLP input (fc) into its MLP's output (down); "out" is the prediction at the target. Subcomponents
of one matrix at one position may be written together (L2.v#902,1514@322, indices sorted): a line with groups states
an edge from every writer to every reader. A line without "<-" states nodes with no stated input. parse() gives
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
subcomponent at the listed positions and the targets, at most 1); an invalid answer (no CIRCUIT section, a line that
does not parse, a missing FINAL line, cut at the sampling cap) earns -(1 + SIZE_WEIGHT), below every valid one.
Nodes and edges the model does not have are dropped and their bits charged in size. Terms: F1 sufficiency, F3 edges,
E1 the reader's bits saved, E2 questions answered right (reward()'s docstring); F2 interchange is a diagnostic.

  components.py questions SPLIT OUT.jsonl [--n N]              question rows (question, positions, evidence, questions)
  components.py dataset ROWS.jsonl TEACHER_DIR OUT.jsonl        SFT rows: each question and its teacher's circuit
                                                                (TEACHER_DIR/<task>.json or .txt; nodes alone get
                                                                measured edges) with the measured answers
  components.py score SPLIT ROWS.jsonl --set NAME=DIR [--n N]  rewards of answer texts DIR/<task>.txt, with baselines
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
MARKDOWN = re.compile(r"\*\*|^[ \t]*(?:#+|[-*])[ \t]+", re.M)
ADDED = re.compile(r"<(?=\||/?(?:think|tool_call|tool_response)>)")  # Qwen's added tokens: <|...|>, <think>, <tool_call>, ...
FINAL_RE = re.compile(r"^\s*FINAL\s+(\d+)\s*:\s*falls\s*=\s*(yes|no)\s+top\s*=\s*(.+?)\s*$", re.M | re.I)
SECTION = re.compile(r"^\s*(EXPLANATION|CIRCUIT|ANSWER\s+\d+|FINAL\s+\d+)\s*:", re.M | re.I)

SIZE_WEIGHT = 0.5
EVIDENCE_PER_POSITION = 12  # VPD's top active subcomponents listed per position (the target's: twice as many)
WIRING_PER_POSITION = 6  # connectome pairs listed per position
SWITCH_PER_POSITION = 8  # subcomponents a changed token switches on, listed per position
QA_PER_KIND = 2
QA_RANKS = 8  # token-change questions draw the new token from the model's alternatives at ranks 0 to 7


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
    section exists) and "finals" (question number -> (falls, the top token as written)). A circuit line's reader may be
    a group, and so may each writer: the line states an edge from every writer to every reader. Markdown decoration
    (bold markers, heading marks at a line's start) is ignored: no valid line holds it."""
    text = MARKDOWN.sub("", text)
    sec = _sections(text)
    nodes, at, edges, malformed = [], {}, [], 0

    def index(nd):
        if nd not in at:
            at[nd] = len(nodes)
            nodes.append([nd[0], nd[2], nd[1]])
        return at[nd]

    for line in sec.get("CIRCUIT", "").splitlines():
        line = line.strip()
        if not line:
            continue
        reader, arrow, writers = line.partition("<-")
        reader = reader.strip()
        rs = ["out"] if reader == "out" else group_of(reader)
        ws = [] if not writers.strip() else ([nd for g in GROUP_RE.finditer(writers) for nd in group_of(g[0])]
                                             if WRITERS_RE.fullmatch(writers) else None)
        if rs is None or ws is None or (reader == "out" and not ws) or (arrow and not ws):
            malformed += 1
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
            "malformed": malformed, "circuit": "CIRCUIT" in sec, "finals": finals}


def _grouped(nds) -> list[str]:
    """Groups of subcomponents by matrix and position, latest position first, then by layer and kind."""
    by = {}
    for nd in nds:
        by.setdefault((nd[0], nd[1]), []).append(nd)
    return [group_text(by[k]) for k in sorted(by, key=_place)]


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
    lines += _grouped(set(nodes) - named)
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


def qa_questions(nat, task: dict, ci: dict, positions: list[int], seed: int, per_kind: int = QA_PER_KIND) -> list[dict]:
    """Intervention questions on a text with the model's measured answers: a token replaced by one of the model's
    alternatives there (ranks 0 to QA_RANKS - 1, at any position before the target) and one VPD-active subcomponent
    removed (Native.without; at F2's positions and the targets). Each: {"kind", the intervention ("position", "new" |
    "node"), "falls": whether the probability of the model's top next token falls below half of what it is, "top": the
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


def qa_text(q: dict, n: int, strings: list[str], top: str, tokens_of) -> str:
    """Question n in words."""
    if q["kind"] == "token":
        what = f"the token at position {q['position']}, {strings[q['position']]!r}, were replaced by {tokens_of([q['new']])[0]!r}"
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
    qa = qa_questions(nat, task, tx.ci, positions, native.task_seed(task["id"]) if qa_seed is None else qa_seed)
    return {"task": task["id"], "positions": positions, "listed": listed, "events": events,
            "components": {str(p): [[list(nd), float(tx.ci[nd[0]][nd[1], nd[2]]), facts(atlas, nd) if atlas else ""] for nd in comps[p]]
                           for p in where},
            "wiring": {str(p): wires[p] for p in where}, "switch_on": switched, "qa": qa, "active": active}


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
    lines = [ASK.format(model=model), "tokens: " + units.numbered(strings),
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


def dropped_bits(answer: dict, g, positions: int, subcomponents: int) -> tuple[int, int, float]:
    """(nodes, edges, bits) of what an answer states that is not in its graph (a node outside the model or the text,
    a connection the model does not have): each priced as native.Graph.bits prices what it keeps (a node one choice
    among positions x subcomponents, an edge its two ends among the answer's nodes, an edge into the prediction its
    writer): it was written and explains nothing."""
    n = max(len(answer.get("nodes", [])), 2)
    outs = {w for w, r in answer.get("edges", []) if r == "out"}
    inner = {(w, r) for w, r in answer.get("edges", []) if r != "out"}
    nodes = len(answer.get("nodes", [])) - len(g.nodes)
    d_out = max(0, len(outs) - len(g.out))
    d_inner = max(0, len(inner) - sum(len(ws) for ws in g.parents.values()))
    return nodes, d_out + d_inner, nodes * math.log2(positions * subcomponents) + (2 * d_inner + d_out) * math.log2(n)


def reward(ge, task: dict, ev: dict, texts: list[str], *, ranks=None, reader=None, e1: list | None = None,
           truncated: list | None = None, size_weight: float = SIZE_WEIGHT, cache: dict | None = None) -> list[dict]:
    """Per answer text on one task: {"reward", "valid", each quality term, "size", the dropped nodes and edges,
    goal_eval's row}. An answer is invalid (reward -(1 + size_weight), below every valid one) when it has no CIRCUIT
    section, a circuit line does not parse, a question has no FINAL line, or it was cut at the sampling cap
    (truncated). Nodes and edges the model does not have are dropped and their description bits charged in size
    (dropped_bits). A valid answer earns the mean of its terms (each at most 1) less size_weight x size, size its
    description bits over the scale (VPD's active subcomponents at the listed positions and the targets), at most 1.
    Terms: F1 (1 - its KL / the empty circuit's, in [0, 1]), F3 (goal_eval's AUC; 0 without a stated edge, 0.5 when
    no connection among its nodes is left unstated), E1 (the reader's bits saved, in [-1, 1]; with a reader or e1), E2
    (the share of questions answered right). F2 is reported (goal_eval's row) but not rewarded. ranks: the
    alternative-token ranks F2's diagnostic uses (goal_eval's held-out ranks when None); reader: a reader.Reader for
    E1, or e1: each text's reader bits computed by the caller (e1_bits(); a Tinker reader lives with the policy); ev:
    the text's evidence(); cache: (task, ranks) -> goal_eval.Text, kept between calls."""
    import goal_eval
    import native

    nat = ge.nat
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    tok = lambda xs: token_strings(task["model"], xs)  # noqa: E731
    total = sum(nat.C.values())
    parsed = [parse(x) for x in texts]
    graphs, defects, valid = [], [], []
    for i, p in enumerate(parsed):
        g, _ = goal_eval.circuit(p["answer"], targets, nat.C, len(ids))
        graphs.append(g)
        defects.append(dropped_bits(p["answer"], g, nat.positions(targets), total))
        valid.append(p["circuit"] and p["malformed"] == 0 and all(n in p["finals"] for n in range(1, len(ev["qa"]) + 1))
                     and not (truncated and truncated[i]))
    keep = [i for i, v in enumerate(valid) if v]  # an invalid answer's reward is fixed whatever it measures: not run
    if keep:
        ge.ranks = tuple(goal_eval.HELD_OUT_RANKS if ranks is None else ranks)
        key = (task["id"], ge.ranks)
        tx = cache.get(key) if cache is not None else None
        if tx is None:
            tx = ge.text(ids, targets, task=task)
            if cache is not None:
                cache[key] = tx
    bits = {}
    if e1 is not None:
        bits = {texts[i]: e1[i] for i in keep}
    elif reader is not None and keep:
        uniq = list(dict.fromkeys(texts[i] for i in keep))
        bits = dict(zip(uniq, e1_bits(reader, task, ev, uniq)))
    ge.hooks = {"E1": (lambda task, g, x: bits.get(x)) if (reader is not None or e1 is not None) else None,
                "E2": lambda task, g, x: qa_score(ev["qa"], parse(x)["finals"], tok)}
    rows = ge.score(tx, [graphs[i] for i in keep] + [native.Graph()], [texts[i] for i in keep] + [""]) if keep else []
    empty = rows[-1] if rows else None
    row_of = dict(zip(keep, rows))
    scale = ev["active"] * math.log2(nat.positions(targets) * total)
    out = []
    for i, (p, (d_nodes, d_edges, d_bits)) in enumerate(zip(parsed, defects)):
        row, terms, size, r = row_of.get(i), {}, None, -(1.0 + size_weight)
        if row is not None:
            base = empty["f1_changed_held"]
            terms["F1"] = min(1.0, max(0.0, 1 - row["f1_changed_held"] / base)) if base > 0 else None
            terms["F3"] = 0.0 if row["f3_stated"] is None else (row["f3_auc"] if row["f3_auc"] is not None else 0.5)
            terms["E1"] = None if row["E1"] is None else max(-1.0, min(1.0, row["E1"]))
            terms["E2"] = row["E2"]
            have = [v for v in terms.values() if v is not None]
            size = min(1.0, (row["bits"] + d_bits) / scale) if scale > 0 else 0.0
            r = sum(have) / len(have) - size_weight * size
        out.append({"reward": r, "valid": valid[i], **{f"term_{k}": v for k, v in terms.items()}, "size": size,
                    "malformed": p["malformed"], "dropped_nodes": d_nodes, "dropped_edges": d_edges,
                    "finals": len(p["finals"]), "truncated": bool(truncated and truncated[i]), "row": row})
    return out


# ---- supervised targets


def target(answer: dict, ev: dict, model: str) -> str:
    """The supervised answer to a question: the teacher's circuit (goal_eval's JSON) and explanation, and for each
    question the model's measured answer (no reasoning: the teacher has none to give)."""
    finals = {n: (q["falls"], shown(token_strings(model, [q["top"]])[0])) for n, q in enumerate(ev["qa"], 1)}
    return render(answer, finals)


def with_edges(ge, task: dict, ev: dict, answer: dict, cache: dict | None = None) -> dict:
    """A teacher circuit given as nodes alone, with edges: the connections the model has among its nodes (and from
    its residual writers at the targets into the prediction), ordered by the measured effect of cutting each one
    alone (goal_eval.cuts), the prefix (none, 1, 2, 4, ..., all) whose answer earns the most reward on the text."""
    import goal_eval
    import native

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    g, _ = goal_eval.circuit(answer, targets, ge.nat.C, len(ids))
    par, out = native.all_edges(g.node_set(), targets)
    pairs = [(w, r) for r, ws in par.items() for w in ws] + [(w, None) for w in out]
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
    d = sub.add_parser("dataset")
    d.add_argument("rows", type=Path)
    d.add_argument("teacher", type=Path)
    d.add_argument("out", type=Path)
    s = sub.add_parser("score")
    s.add_argument("split", choices=("train", "heldout", "hard"))
    s.add_argument("rows", type=Path, help="questions' JSONL")
    s.add_argument("--set", action="append", default=[], help="NAME=DIR: answer texts DIR/<task>.txt")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--ranks", default="3,4,5")
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
                ev = evidence(nat, ge, task, conn, atlas=atlas)
                f.write(json.dumps({**ev, "question": question(task, ev)}) + "\n")
                f.flush()
                print(p.stem, len(ev["positions"]), "positions", ev["active"], "active", flush=True)
        return
    rows = {r["task"]: r for r in map(json.loads, args.rows.read_text().splitlines())}
    sets = dict(x.split("=", 1) for x in args.set)
    ranks = tuple(int(r) for r in args.ranks.split(","))
    results = []
    for p in native.tasks(args.split)[:args.n]:
        if p.stem not in rows:
            continue
        task = json.loads(p.read_text())
        ev = rows[p.stem]
        names, texts = [], []
        for s, d in sets.items():
            f = Path(d) / f"{p.stem}.txt"
            if f.exists():
                names.append(s)
                texts.append(f.read_text())
        got = reward(ge, task, ev, texts + [render({"nodes": [], "edges": []})], ranks=ranks)
        names.append("empty")
        for name, r in zip(names, got):
            results.append({"task": p.stem, "set": name, **{k: v for k, v in r.items() if k != "row"},
                            "bits": r["row"]["bits"], "nodes": r["row"]["nodes"], "edges": r["row"]["edges"]})
            print(p.stem, name, f"reward {r['reward']:+.3f}", " ".join(f"{k[5:]} {v:.3f}" for k, v in r.items()
                                                                      if k.startswith("term_") and v is not None), flush=True)
    if args.out:
        args.out.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
