"""Component circuits: the oracle's answer in VPD subcomponents, its question, and its reward (#2951 graph oracle).

The oracle reads a text and states the whole-model circuit of vpd4l's prediction in VPD's rank-one subcomponents,
with explicit edges, an English explanation, and answers to intervention questions about the same text. The answer is
plain text (nothing in it is executed):

  EXPLANATION: the model copies the digits after the earlier "327" ...
  CIRCUIT:
  out <- L3.down#120@332, L2.o#77@332
  L2.o#77@332 <- L2.v#1514@322
  L2.v#1514@322 <- L0.down#77@322
  L0.down#77@322 <- L0.fc#12@322
  L2.k#9@322
  ANSWER 1: the change at 322 removes the copied digit ...
  FINAL 1: falls=yes top=' 7'

A subcomponent is L<layer>.<kind>#<index>@<position> (kind q, k, v, o, fc, down). A circuit line "reader <- writer,
writer" states one edge per writer: an attention or MLP output (o, down) into a later query, key, value or MLP input
(q, k, v, fc) at its position, a value (v) into its layer's attention output (o) at that or a later position (attention
between positions), an MLP input (fc) into its MLP's output (down); "out" is the prediction at the target. A line
without "<-" states a node with no stated input. parse() gives goal_eval's circuit JSON, and goal_eval.circuit checks
every edge against the model (mech.connects).

The question (question()) holds the text's tokens, the model's top next tokens, the positions F2 measures (those whose
changed token moves the prediction by at least 1% of the text's total, goal_eval's SHARE_FLOOR), with the measured
evidence there (units.evidence: how far the prediction moves, the changed-token and per-layer attention-cut effects),
VPD's active subcomponents at each of those positions and at the target, ranked by VPD (causal importance x how much
each writes, Native._vpd_order), the strongest weight-level connections among them (method/vpd/connectome.py; a
static coupling, not a measured effect), and the intervention questions (qa_questions()).

The reward (reward()) is goal_eval's scorecard combined into one number: invalid answers -1, otherwise the mean of
the quality terms, each 1 when perfect, minus SIZE_WEIGHT x the size term:
  F2 completeness: the mean over F2's positions of min(1, carried / CARRIED_GOAL), carried = 1 - KL / movement there
     (at least 0: a circuit that moves the prediction further than the change does carries none of it), on held-out
     alternative tokens (goal_eval's ranks; RL draws them per text from 0 to 7).
  F1 recovered: 1 - F1 (changed prompts, remainders held) / the empty circuit's, clipped to [0, 1].
  F3: the AUC of stated edges' cuts against unstated pairs' (0.5, chance, when the answer states no edge or no pair
     is left unstated).
  E1: the English reader's bits saved per question (reader.Reader), clipped to [-1, 1]; with a reader only.
  E2: the share of intervention questions answered right (qa_score); when the question asks some.
  size: the circuit's description length (native.Graph.bits) over that of naming every VPD-active subcomponent at
     F2's positions and the targets (the answer that lists everything VPD says is on where the changes matter).
An answer is invalid when a circuit line does not parse, goal_eval drops a node or edge (outside the model or the
text, or a connection the model does not have), or a question has no FINAL line.

  components.py questions SPLIT OUT.jsonl [--n N]              question rows (question, positions, evidence, questions)
  components.py dataset ROWS.jsonl TEACHER_DIR OUT.jsonl        SFT rows: each question and its teacher's circuit
                                                                (TEACHER_DIR/<task>.txt) with the measured answers
  components.py score SPLIT ROWS.jsonl --set NAME=DIR [--n N]  rewards of answer texts DIR/<task>.txt, with baselines
"""

from __future__ import annotations

import argparse
import ast
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
NODE = r"L(\d+)\.(q|k|v|o|fc|down)#(\d+)@(\d+)"
NODE_RE = re.compile(rf"^{NODE}$")
FINAL_RE = re.compile(r"^\s*FINAL\s+(\d+)\s*:\s*falls\s*=\s*(yes|no)\s+top\s*=\s*(.+?)\s*$", re.M | re.I)
SECTION = re.compile(r"^\s*(EXPLANATION|CIRCUIT|ANSWER\s+\d+|FINAL\s+\d+)\s*:", re.M | re.I)

CARRIED_GOAL = 0.9  # F2: a position whose change the circuit carries this far counts as complete
SIZE_WEIGHT = 0.5
EVIDENCE_PER_POSITION = 12  # VPD's top active subcomponents listed per position (the target's: twice as many)
WIRING_PER_POSITION = 6  # connectome pairs listed per position
QA_PER_KIND = 2
QA_RANKS = 8  # token-change questions draw the new token from the model's alternatives at ranks 0 to 7


# ---- the answer's text


def node_text(nd) -> str:
    """A subcomponent (matrix name, position, index) as L<layer>.<kind>#<index>@<position>."""
    name, t, c = nd
    return f"L{name.split('.')[1]}.{CODES[name.split('.')[-1]]}#{c}@{t}"


def node_of(text: str):
    """(matrix name, position, index) of L<layer>.<kind>#<index>@<position>, or None."""
    import native

    m = NODE_RE.match(text.strip())
    if not m:
        return None
    return native.site_name(int(m[1]), KINDS[m[2]]), int(m[4]), int(m[3])


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
    section exists) and "finals" (question number -> (falls, top token string))."""
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
        reader, _, writers = line.partition("<-")
        reader = reader.strip()
        r = "out" if reader == "out" else node_of(reader)
        ws = [node_of(w) for w in writers.split(",")] if writers.strip() else []
        if r is None or any(w is None for w in ws) or (r == "out" and not ws):
            malformed += 1
            continue
        if r != "out":
            index(r)
        for w in ws:
            edges.append([index(w), "out" if r == "out" else at[r]])
    finals = {}
    for m in FINAL_RE.finditer(text):
        top = m[3].strip()
        try:
            top = ast.literal_eval(top) if top[:1] in "'\"" else top
        except (ValueError, SyntaxError):
            pass
        finals.setdefault(int(m[1]), (m[2].lower() == "yes", str(top)))
    explanation = sec.get("EXPLANATION", "")
    return {"explanation": explanation, "answer": {"nodes": nodes, "edges": edges, "explanation": explanation},
            "malformed": malformed, "circuit": "CIRCUIT" in sec, "finals": finals}


def render(answer: dict, finals: dict | None = None, reasons: dict | None = None) -> str:
    """The text of a circuit in goal_eval's JSON (to_json's), lines from the prediction backward: "out" first, then each
    reader the first time it is reached through its writers, then readers not reached, then nodes named nowhere else."""
    import goal_eval

    nodes = [(goal_eval._site(m), t, c) for m, c, t in answer["nodes"]]
    parents, out = {}, []
    for w, r in answer["edges"]:
        if r == "out":
            out.append(nodes[w])
        else:
            parents.setdefault(nodes[r], []).append(nodes[w])
    lines, named, done = [], set(), set()
    if out:
        lines.append("out <- " + ", ".join(node_text(w) for w in out))
        named.update(out)
    queue = list(out)
    while queue:
        r = queue.pop(0)
        if r in done or r not in parents:
            continue
        done.add(r)
        lines.append(f"{node_text(r)} <- " + ", ".join(node_text(w) for w in parents[r]))
        named.add(r)
        named.update(parents[r])
        queue += parents[r]
    for r in sorted(set(parents) - done, key=lambda nd: (-nd[1], nd[0], nd[2])):
        lines.append(f"{node_text(r)} <- " + ", ".join(node_text(w) for w in parents[r]))
        named.add(r)
        named.update(parents[r])
    lines += [node_text(nd) for nd in sorted(set(nodes) - named, key=lambda nd: (-nd[1], nd[0], nd[2]))]
    text = [f"EXPLANATION: {answer.get('explanation', '')}".rstrip(), "CIRCUIT:"] + lines
    for k in sorted(finals or {}):
        falls, top = finals[k]
        text += [f"ANSWER {k}: {(reasons or {}).get(k, '')}".rstrip(), f"FINAL {k}: falls={'yes' if falls else 'no'} top={top!r}"]
    return "\n".join(text)


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
    the most likely next token after the intervention (its string, exactly)."""
    if not questions:
        return None
    right = 0.0
    for n, q in enumerate(questions, 1):
        if n not in finals:
            continue
        falls, top = finals[n]
        right += 0.5 * (falls == q["falls"]) + 0.5 * (top == tokens_of([q["top"]])[0])
    return right / len(questions)


# ---- the question


ASK = ("How does {model} compute its prediction of the next token after the target of this text? State the circuit in "
       "VPD subcomponents: a subcomponent is one rank-one slice of a weight matrix at a position, written "
       "L<layer>.<kind>#<index>@<position> (kind q, k, v, o, fc or down; for example L2.v#1514@322). Write one line per "
       "reader, \"<reader> <- <writer>, <writer>\", each writer an edge into the reader: an attention or MLP output (o, "
       "down) into a later query, key, value or MLP input (q, k, v, fc) at its position; a value (v) into its layer's "
       "attention output (o) at that or a later position; an MLP input (fc) into its MLP's output (down). \"out\" is the "
       "prediction. The circuit must carry the change of each listed position's token to the prediction; fewer "
       "subcomponents and edges are better. Answer with a line EXPLANATION: (in plain English, what the model does), "
       "then CIRCUIT: and its lines, then for each question ANSWER <n>: (your reasoning in plain English) and "
       "FINAL <n>: falls=yes or no top=<the token, quoted>.")


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


def evidence(nat, ge, task: dict, conn=None, qa_seed: int | None = None, atlas=None) -> dict:
    """One text's question data: F2's positions (goal_eval.Text with the text's own seed), the measured evidence there
    (units.events at those positions; its other events are the English reader's), VPD's top active subcomponents per
    position with their atlas facts (facts(); none without an atlas), their strongest wiring, the intervention
    questions, and the count of VPD-active subcomponents at F2's positions and the targets (the size term's scale)."""
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
    active = int(sum(int((tx.ci[n][where] > 0).sum()) for n in nat.names))
    qa = qa_questions(nat, task, tx.ci, positions, native.task_seed(task["id"]) if qa_seed is None else qa_seed)
    return {"task": task["id"], "positions": positions, "listed": listed, "events": events,
            "components": {str(p): [[list(nd), float(tx.ci[nd[0]][nd[1], nd[2]]), facts(atlas, nd) if atlas else ""] for nd in comps[p]]
                           for p in where},
            "wiring": {str(p): wires[p] for p in where}, "qa": qa, "active": active}


def question(task: dict, ev: dict) -> str:
    """The oracle's question for one text from its evidence()."""
    import native
    import units
    from prompt import tokens_of

    model = task["model"]
    p = task["prompts"][0]
    strings = tokens_of(model, p["token_ids"])
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
        if ev["wiring"].get(q):
            lines.append(f"  wiring: " + ", ".join(f"{node_text(tuple(w))} -> {node_text(tuple(r))} ({c:+.2f})" for w, r, c in ev["wiring"][q]))
    if ev["qa"]:
        tok = lambda ids: tokens_of(model, ids)  # noqa: E731
        lines += [qa_text(q, n, strings, repr(top_tok), tok) for n, q in enumerate(ev["qa"], 1)]
    return "\n".join(lines)


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
    from prompt import tokens_of

    tk = AutoTokenizer.from_pretrained(base)

    def generate(prompts: list[str]) -> list[dict]:
        if not prompts:
            return []
        chat = [tk.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False,
                                       tokenize=False) for q in prompts]
        r = httpx.post(url, json={"model": base, "prompt": chat, "max_tokens": 1, "temperature": 0.0, "logprobs": 20},
                       timeout=600.0)
        r.raise_for_status()
        out = [None] * len(prompts)
        for c in r.json()["choices"]:
            out[c["index"]] = dict(c["logprobs"]["top_logprobs"][0])
        return out

    return reader.Reader(generate, lambda ids: tokens_of(model, ids))



def english(text: str) -> str:
    return _sections(text).get("EXPLANATION", "")


def reward(ge, task: dict, ev: dict, texts: list[str], *, ranks=None, reader=None, size_weight: float = SIZE_WEIGHT,
           carried_goal: float = CARRIED_GOAL, cache: dict | None = None) -> list[dict]:
    """Per answer text on one task: {"reward", "valid", each quality term, "size", goal_eval's row}. ranks: the
    alternative-token ranks F2 uses (goal_eval's held-out ranks when None); reader: a reader.Reader for E1 (None: no
    E1 term); ev: the text's evidence() (its questions, reader events and active count); cache: (task, ranks) ->
    goal_eval.Text, kept between calls (a group's answers arrive in several calls)."""
    import goal_eval
    import native
    import units
    from prompt import tokens_of

    nat = ge.nat
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    ge.ranks = tuple(goal_eval.HELD_OUT_RANKS if ranks is None else ranks)
    key = (task["id"], ge.ranks)
    tx = cache.get(key) if cache is not None else None
    if tx is None:
        tx = ge.text(ids, targets, task=task)
        if cache is not None:
            cache[key] = tx
    tok = lambda xs: tokens_of(task["model"], xs)  # noqa: E731
    parsed = [parse(x) for x in texts]
    graphs, defects = [], []
    for p in parsed:
        g, d = goal_eval.circuit(p["answer"], targets, nat.C, tx.T)
        graphs.append(g)
        defects.append(d)
    e1 = {}
    if reader is not None:
        evs = units.reader_events(ev["events"])
        uniq = list(dict.fromkeys(texts))
        for x, b in zip(uniq, reader.bits(task, [english(x) for x in uniq], evs)):
            e1[x] = b
    ge.hooks = {"E1": (lambda task, g, x: e1.get(x)) if reader is not None else None,
                "E2": lambda task, g, x: qa_score(ev["qa"], parse(x)["finals"], tok)}
    rows = ge.score(tx, graphs + [native.Graph()], texts + [""])
    empty = rows[-1]
    scale = ev["active"] * math.log2(nat.positions(targets) * sum(nat.C.values()))
    out = []
    for x, p, d, row in zip(texts, parsed, defects, rows):
        valid = (p["circuit"] and p["malformed"] == 0 and not d["invalid_nodes"] and not d["invalid_edges"]
                 and all(n in p["finals"] for n in range(1, len(ev["qa"]) + 1)))
        terms = {}
        pos = [(m, k) for _, m, k in row["f2_positions"] if m > 0]
        terms["F2"] = sum(min(1.0, max(0.0, 1 - k / m) / carried_goal) for m, k in pos) / len(pos) if pos else None
        base = empty["f1_changed_held"]
        terms["F1"] = min(1.0, max(0.0, 1 - row["f1_changed_held"] / base)) if base > 0 else None
        terms["F3"] = row["f3_auc"] if row["f3_auc"] is not None else 0.5
        terms["E1"] = None if row["E1"] is None else max(-1.0, min(1.0, row["E1"]))
        terms["E2"] = row["E2"]
        have = [v for v in terms.values() if v is not None]
        size = row["bits"] / scale if scale > 0 else 0.0
        r = (sum(have) / len(have) - size_weight * size) if valid else -1.0
        out.append({"reward": r, "valid": valid, **{f"term_{k}": v for k, v in terms.items()}, "size": size,
                    "malformed": p["malformed"], **d, "row": row})
    return out


# ---- supervised targets


def target(teacher: str, ev: dict, model: str) -> str:
    """The supervised answer to a question: the teacher's explanation and circuit, and for each question the model's
    measured answer (no reasoning: the teacher has none to give)."""
    from prompt import tokens_of

    p = parse(teacher)
    finals = {n: (q["falls"], tokens_of(model, [q["top"]])[0]) for n, q in enumerate(ev["qa"], 1)}
    return render(p["answer"], finals)


def dataset(rows: Path, teacher_dir: Path, out: Path) -> int:
    """SFT rows ({"prompt": [user message], "completion": [assistant message], "info": {"task"}}, units_sft.py's
    format) for the questions that have a teacher answer."""
    import native

    n = 0
    with out.open("w") as f:
        for r in map(json.loads, rows.read_text().splitlines()):
            path = teacher_dir / f"{r['task']}.txt"
            if not path.exists():
                continue
            task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
            f.write(json.dumps({"prompt": [{"role": "user", "content": r["question"]}],
                                "completion": [{"role": "assistant", "content": target(path.read_text(), r, task["model"])}],
                                "info": {"task": r["task"]}}) + "\n")
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
    if args.cmd == "dataset":
        print(dataset(args.rows, args.teacher, args.out), "rows", flush=True)
        return

    import goal_eval
    import native

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
