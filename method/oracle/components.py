"""Component circuits: the oracle's answer in VPD subcomponents, its question, and its reward (#2951 graph oracle).

The oracle reads a text and states the whole-model circuit of vpd4l's prediction in VPD's rank-one subcomponents,
with explicit edges, an English explanation, and answers to intervention questions about the same text. The answer is
plain text (nothing in it is executed):

  CIRCUIT:
  ```mermaid
  flowchart LR
    subgraph P332["@332 ' the'"]
      a["L3.down#120"]
      b["L2.o#77"]
    end
    subgraph P322["@322 '3'"]
      c["L2.v#902"]
      d["L2.v#1514"]
      e["L0.down#77"]
      f["L0.fc#12"]
      t322[tok]
    end
    a --> OUT((prediction))
    b --> OUT
    c --> b
    d --> b
    e --> c
    e --> d
    f --> e
    t322 --> f
  ```
  EXPLANATION:
  VARIABLES:
  V1 "copied digit": the digit after the earlier "327"; inputs: @322 ' 7'; carried by: c,d,e; feeds: PREDICTION
  the model copies the digits after the earlier "327" ...
  ANSWER 1: the change at 322 removes the copied digit ...
  FINAL 1: falls=yes top=' 7'
  EDIT 1: cut b <- c; scale e 0

The circuit comes first and the explanation of that circuit after it, so the English is conditioned on the circuit
written (parse() reads the sections in any order). The EXPLANATION may open with a VARIABLES block, the explanation
as a causal model of the circuit: per variable its meaning, its input positions (the context tokens it is computed
from), the circuit's subcomponents that carry it (by id) and what it feeds (other variables or the prediction), scored
by interchange interventions (E4, variables.py); the prose after it explains the prediction in those variables and
never names a subcomponent. A subcomponent is L<layer>.<kind>#<index> (kind q, k, v, o, fc, down) at a position; a
block, L<layer>.<kind> at a position, is that matrix's whole output there; the token embedding at a position is tok,
an input node. The CIRCUIT is a Mermaid flowchart (circuit_lines(); the user's decision 10-12): every node declared
once in its position's subgraph, one edge per line, "writer --> reader": an attention or MLP output (o, down) into a
later query, key, value or MLP input (q, k, v, fc) at its position, a value (v) into its layer's attention output (o)
at that or a later position, an MLP input (fc) into its MLP's output (down), the token embedding into a reader at its
position; "--> OUT" is the prediction at the target; "q -.->|pair| k" states a query-key pair of q's attention read
(goal_eval's pair edge ["pair", query, key, layer, position]; rest for the part outside the circuit). parse() reads it
strictly (_parse_mermaid()) into goal_eval's circuit JSON, and goal_eval.circuit checks every edge against the model
(mech.connects); an id never declared is a node outside the graph, dropped and charged. parse() still reads v1
answers' id grammar ("@322 c=L2.v#902 h=tok", "reader <- writer,writer", PATTERN lines), never written now. An edit request ("how would you make the model ...") is answered by an EDIT line of
operations: cut <reader> <- <writer> (that connection removed on this text), scale <node> <k> (its activation times k),
either prefixed by persistent for a weight edit on every text (nodes then without a position). Token strings are shown as Python reprs of the target's tokens (special tokens included)
and escaped (escape()): no question or answer holds a literal special or added token of the explainer's tokenizer.

The question (question()) holds the text's tokens, the model's top next tokens, the positions whose changed token
moves the prediction by at least 1% of the text's total (goal_eval's SHARE_FLOOR) with the measured evidence there
(units.evidence: how far the prediction moves, the changed-token and per-layer attention-cut effects), VPD's active
subcomponents at each of those positions and at the target, ranked by VPD (Native._vpd_order) with their atlas facts,
the subcomponents a change of each listed token switches on, the strongest weight-level connections among them
(method/vpd/connectome.py; a static coupling, not a measured effect), and the intervention questions (qa_questions()).

The reward (reward(), v3, the lead's 10-12) is faithfulness first, then compactness and explanation, every term in
bits: R = phi + g x Q + D, phi (goal_eval.phi) -log2 of the power-4 mean of the residuals of the intervention families
(point, node removals, connection cuts, flips, path-patched changed inputs and interchange, near and far donors,
attention patterns, necessity), g a gate at the text's bar (ge.tau), Q = 0.5 compact (log2 of the influential start's
description bits over the answer's) + E2g + E4x + E1, D the teacher-dominance term; an invalid answer (no CIRCUIT
section, a missing FINAL or EDIT line, cut at the sampling cap, more than NODE_CAP nodes) earns V3_INVALID, below every
valid one. Nodes and edges the model does not have, and circuit lines that do not parse, are dropped and their bits
charged. goal_eval's C, F1, N, F3, the old E2 (from the no-change answer), E3 and E4 are reported beside it.

  components.py questions SPLIT OUT.jsonl [--n N] [--qa-seed K]  question rows (question, positions, evidence, questions;
                                                                K = 1 for a fresh set of questions)
  components.py requestion ROWS.jsonl OUT.jsonl [--qa-seed 1]   the rows with a fresh set of intervention questions
  components.py rerender ROWS.jsonl OUT.jsonl                   the rows' question text in the current format
  components.py locality SPLIT OUT.jsonl [--n N]                each text local or not (VPD's target actives alone)
  components.py teacher-vpd ROWS.jsonl OUT_DIR                 the interim teacher (VPD's answer cut to a writable size)
  components.py prepare SPLIT OUT_DIR [--n N] [--lengths LO:HI]  locality, then the row and teacher of the texts kept (at
                                                                least half non-local)
  components.py facts ROWS.jsonl TEACHER_DIR OUT.jsonl [--n N]  what each teacher's English must be faithful to
  components.py dataset ROWS.jsonl TEACHER_DIR OUT.jsonl        SFT rows: each question and its teacher's circuit
                                                                (TEACHER_DIR/<task>.json or .txt; nodes alone get
                                                                measured edges) with the measured answers
  components.py references ROWS.jsonl TEACHER_DIR OUT_DIR       VPD at each teacher's size: every connection, and
                                                                the connections teacher.py's edge search kept
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
GROUP_RE = re.compile(r"(?:L(\d+)\.(q|k|v|o|fc|down)(?:#(\d+(?:,\d+)*))?|tok)@(\d+)")  # no #index: a block node
_G = r"(?:L\d+\.(?:q|k|v|o|fc|down)(?:#\d+(?:,\d+)*)?|tok)@\d+"
TOK = "tok"  # the token embedding at a position, an input node: JSON [TOK, 0, position], text tok@<position>
BLOCK = -1  # goal_eval.BLOCK: a block node (site, position, BLOCK), a site's whole output there: JSON ["block", site,
# position], text L<layer>.<kind>@<position> (no #index)
WRITERS_RE = re.compile(rf"\s*{_G}(?:\s*,\s*{_G})*\s*")
BLOCK_RE = re.compile(r"@(\d+)\s+(\S.*)")  # a position block: "@322" and its items, each parsed alone (ITEM_RE)
ITEM_RE = re.compile(r"(?:([a-z]+)=)?(L\d+\.(?:q|k|v|o|fc|down)(?:#\d+(?:,\d+)*)?|tok)")
_ID = r"[A-Za-z][A-Za-z0-9_]*"  # a node's id (Mermaid's: a, t394, B1)
_W = rf"(?:{_G}|{_ID})"
ID_RE = re.compile(_ID)
GROUP_SPLIT = re.compile(rf"{_G}|{_ID}")
REFS_RE = re.compile(rf"\s*{_W}(?:\s*,\s*{_W})*\s*")  # writers by group or by id
MARKDOWN = re.compile(r"\*\*|^[ \t]*(?:#+|[-*])[ \t]+", re.M)
ADDED = re.compile(r"<(?=\||/?(?:think|tool_call|tool_response|tts_[a-z_]+)>)")  # the explainers' added tokens (EXPLAINERS): <|...|>, <think>, <tool_call>, <tts_pad>, ...
EXPLAINERS = ("Qwen/Qwen3-8B", "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-4B")  # policies and readers whose added tokens escape() breaks
FINAL_RE = re.compile(r"^\s*FINAL\s+(\d+)\s*:\s*falls\s*=\s*(yes|no)\s+top\s*=\s*(.+?)\s*$", re.M | re.I)
SECTION = re.compile(r"^\s*(EXPLANATION|CIRCUIT|ANSWER\s+\d+|FINAL\s+\d+|EDIT\s+\d+)\s*:", re.M | re.I)
EDIT_RE = re.compile(r"^\s*EDIT\s+(\d+)\s*:(.*)$", re.M | re.I)
OP_RE = re.compile(r"\s*(persistent\s+)?(?:cut\s+(\S+)\s*<-\s*(\S+)|scale\s+(\S+)\s+(-?\d+(?:\.\d+)?))\s*", re.I)
MM_SUB = re.compile(r"subgraph\s+P(\d+)\b.*")  # the Mermaid CIRCUIT (circuit_lines(), _parse_mermaid())
MM_DECL = re.compile(rf'({_ID})\s*\[\s*(?:"([^"]*)"|([^\]"]*))\s*\]')
MM_DECLS = re.compile(rf'\s*(?:{MM_DECL.pattern}\s*;?\s*)+')
MM_OUT = re.compile(r"OUT(?:\(\([^)]*\)\))?")
MM_EDGE = re.compile(rf"({_ID}(?:\s*&\s*{_ID})*)\s*-->\s*({MM_OUT.pattern}|{_ID}(?:\s*&\s*{_ID})*)")
MM_PAIR = re.compile(rf"({_ID})\s*-\.->\s*\|\s*pair(?:\s+L(\d+)@(\d+))?\s*\|\s*({_ID})")
MM_RESERVED = {"end", "rest", "graph", "subgraph", "flowchart", "direction", "style", "class", "click", "default", "out", "tok"}
PATTERN_RE = re.compile(r"PATTERN\s+(?:L(\d+)@(\d+)\s*:\s*)?(\S+)\s*~\s*(\S+)", re.I)  # a stated query-key pair
VAR_LINE = re.compile(r"\s*V\d+\b")  # a line of the EXPLANATION's VARIABLES block (VAR_RE when it parses)
VAR_RE = re.compile(r'\s*V(\d+)\s+"(.*?)"\s*:\s*(.*);\s*inputs\s*:\s*(.*?)\s*;\s*carried\s+by\s*:\s*(.*?)\s*;\s*feeds\s*:\s*(.*?)\s*', re.I)
BLOCK_ITEM = r"L\d+\.(?:q|k|v|o|fc|down)\s+@\d+(?:-\d+)?"  # a carrier block: a site's whole output at positions a..b
BLOCK_ITEM_RE = re.compile(r"L(\d+)\.(q|k|v|o|fc|down)\s+@(\d+)(?:-(\d+))?")
CARRIER_RE = re.compile(rf"{BLOCK_ITEM}|{_W}")
CARRIERS_RE = re.compile(rf"\s*(?:{BLOCK_ITEM}|{_W})(?:\s*,\s*(?:{BLOCK_ITEM}|{_W}))*\s*")
INSTANCES_RE = re.compile(r"\(\s*instances\s*:\s*(.*?)\s*\)\s*$", re.I)
QUOTED = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")  # a token's repr in an inputs list

V3_INVALID = -10.0  # reward v3 of an invalid answer: below every valid one (phi of a family at s = -100 is about -6.7)
NODE_CAP = 266  # reward v3's hard cap on an answer's nodes (an answer above it is invalid)
GATE_SLOPE = 4.0  # reward v3's gate g = 1 / (1 + 2^(-4 (phi - phi_tau))): 0.06 one bit short of the bar
POOL = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt"  # C's flip adversary's pattern pool (goal_eval.fidelity)
CONVENTION = "live"  # goal_eval's remainders convention for the reward's F1 and N: on, computing on the run's inputs
EVIDENCE_PER_POSITION = 12  # VPD's top active subcomponents listed per position (the target's: twice as many)
WIRING_PER_POSITION = 6  # connectome pairs listed per position
SWITCH_PER_POSITION = 8  # subcomponents a changed token switches on, listed per position
ATTENTION_TOP = 3  # positions listed per attention head at the target (measured(): the --measured input arm)
POOL_HEADS = 0  # candidate_pool(): positions beyond the listed ones, each attention head's top POOL_HEADS at the target
# (8 heads' top 8: +1% of held-out teacher nodes for 4x the question, 46k tokens)
CONTEXT = 32768  # Qwen3-8B's native context, the question and the generation together
ROOM = 10240  # fit_pool(): the generation a question leaves at least (an answer of tinker_train.ANSWER_CAP, 8,000 tokens, and some thinking)
WINDOW = 8  # a question numbers the tokens within this many positions of every position its evidence names
MARK_EVERY = 16  # elsewhere, plain text with a position marker at least this often
QA_PER_KIND = 2
QA_RANKS = 8  # token-change questions draw the new token from the model's alternatives at ranks 0 to 7
E1_EVENTS = 12  # the reader events E1 asks per text and circuit (e1_bits; the strong reader's cost)
E1_CANDIDATES = 36  # the events the unaided reader is asked first: the heaviest by measured effect
NEUTRAL = "The model predicts its next token from the text."  # E1's baseline: an explanation that says nothing
ANSWER_BUDGET = 3_500  # tokens of a supervised answer: 4,096 sampled at most, ~600 left for English and reasoning
MAX_PAIRS = 1_024  # connections with_edges measures at most
FACT_NODES = 24  # subcomponents a teacher's English writer reads about


# ---- the answer's text


def escape(text: str) -> str:
    """Text with every added-token string of the explainers' tokenizers (EXPLAINERS) broken by a backslash after its "<"
    ("<|endoftext|>" -> "<\\|endoftext|>", "<think>" -> "<\\think>"), so the explainer's tokenizer reads the vpd4l token strings a question lists (the Pile's
    "<|endoftext|>" among them) as text, never as its own special or added tokens. Applied to every question and
    answer text; idempotent."""
    return ADDED.sub(r"<\\", text)


def node_text(nd) -> str:
    """A subcomponent (matrix name, position, index) as L<layer>.<kind>#<index>@<position>."""
    return group_text([nd])


def group_text(nds) -> str:
    """Subcomponents of one matrix at one position as L<layer>.<kind>#<index>,<index>,...@<position>, indices sorted;
    the token embedding at a position as tok@<position>; a block node (its whole output) as L<layer>.<kind>@<position>."""
    name, t = nds[0][0], nds[0][1]
    if name == TOK:
        return f"{TOK}@{t}"
    if nds[0][2] == BLOCK:
        return f"L{name.split('.')[1]}.{CODES[name.split('.')[-1]]}@{t}"
    return f"L{name.split('.')[1]}.{CODES[name.split('.')[-1]]}#{','.join(str(c) for c in sorted({nd[2] for nd in nds}))}@{t}"


def group_of(text: str):
    """The subcomponents [(matrix name, position, index), ...] of L<layer>.<kind>#<index>[,<index>...]@<position> (a
    block node, index BLOCK, without #), or None."""
    import native

    m = GROUP_RE.fullmatch(text.strip())
    if not m:
        return None
    if m[1] is None:
        return [(TOK, int(m[4]), 0)]
    name = native.site_name(int(m[1]), KINDS[m[2]])
    if m[3] is None:
        return [(name, int(m[4]), BLOCK)]
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
    section exists), "malformed_nodes" (what those lines name, at least one each, every item of a position block that
    does not parse, and every undefined id, each charged as a node outside the graph), "unknown_ids", "finals" (question number -> (falls, the top token as written)) and
    "edits" (edit question number -> operations: {"op": "cut", "writer", "reader" ("out": the prediction)} or {"op":
    "scale", "node", "k"}, each with "persistent" (a weight edit everywhere: nodes without a position, None), or
    {"op": "invalid"} for one that names no node). A PATTERN line adds goal_eval's pair edge ["pair", query, key, layer,
    position] (ids, nodes written out, or rest; without "L<layer>@<position>:" the query's layer and position). A circuit line's reader may be
    a group, and so may each writer: the line states an edge from every writer to every reader. Markdown decoration
    (bold markers, heading marks at a line's start) is ignored: no valid line holds it."""
    text = MARKDOWN.sub("", text)
    sec = _sections(text)  # (a CIRCUIT is Mermaid, circuit_lines(); v1's id grammar is still read)
    nodes, at, edges, malformed, junk, ids = [], {}, [], 0, 0, {}

    def index(nd):
        if nd not in at:
            at[nd] = len(nodes)
            nodes.append(["block", nd[0], nd[1]] if nd[2] == BLOCK else [nd[0], nd[2], nd[1]])
        return at[nd]

    lines = [x.strip() for x in sec.get("CIRCUIT", "").splitlines() if x.strip() and not x.strip().startswith("```")]
    mermaid = bool(lines) and lines[0].lower().startswith("flowchart")
    old = [] if mermaid else lines  # v1 answers' id grammar ("@322 a=L2.v#902", "a <- c"): read, never written
    for line in old:  # first every position block, so an edge line may name an id declared below it
        block = BLOCK_RE.fullmatch(line)
        if not block:
            continue
        bad = 0  # "@322 a=L2.v#902 b=L3.o#5" (or ids left out: "@322 L2.v#902,1514 L3.o#5")
        for item in block[2].split():
            m = ITEM_RE.fullmatch(item)
            group = (group_of(f"{m[2]}@{block[1]}") or []) if m else []
            if not group or (m[1] and (len(group) != 1 or m[1] in ("out", "tok") or ids.get(m[1], group[0]) != group[0])):
                bad += 1  # an item that does not parse, or an id naming more than one node or a second one: dropped,
                continue  # charged as a node outside the graph; the block's other items stay
            for nd in group:
                index(nd)
            if m[1]:
                ids[m[1]] = group[0]
        if bad:
            malformed += 1
            junk += bad
    unknown, pairs = 0, []
    complete = False
    if mermaid:
        ids, edges, malformed, junk, unknown = _parse_mermaid(lines, index)
        complete = "complete" in edges
        edges = [e for e in edges if e != "complete"]
    for line in old:
        if BLOCK_RE.fullmatch(line):
            continue
        if line.upper().startswith("PATTERN"):  # PATTERN L3@288: c ~ f, a read's query-key pair (goal_eval's pair edge)
            m = PATTERN_RE.fullmatch(line)
            sides = [None if not m else "rest" if x.lower() == "rest" else ids.get(x) or node_of(x) for x in (m[3], m[4])] if m else [None, None]
            q = sides[0]
            at_read = (int(m[1]), int(m[2])) if m and m[1] else ((int(q[0].split(".")[1]), q[1]) if q not in (None, "rest") and q[0] != TOK else None)
            if None in sides or at_read is None:
                malformed += 1
                junk += 1
                continue
            pairs.append(["pair"] + ["rest" if x == "rest" else index(x) for x in sides] + list(at_read))
            continue
        if line.startswith("@"):
            malformed += 1
            junk += max(1, len(line.split()) - 1)
            continue
        reader, arrow, writers = line.partition("<-")
        reader = reader.strip()
        if ID_RE.fullmatch(reader) and reader not in ids and reader.lower() != "out":
            unknown += 1  # an undefined id: a node that is not in the graph, charged; the line states nothing
            continue
        ws, missing = [], 0
        if writers.strip() and REFS_RE.fullmatch(writers):
            for w in (x.strip() for x in GROUP_SPLIT.findall(writers)):
                got = [ids[w]] if w in ids else group_of(w)
                if got:
                    ws += got
                else:
                    missing += 1  # an undefined id among the writers: dropped and charged, the others kept
        elif writers.strip():
            ws = None
        is_out = reader.lower() == "out"
        rs = ["out"] if is_out else ([ids[reader]] if reader in ids else group_of(reader))
        unknown += missing
        if rs is None or ws is None or (arrow and not ws and not missing) or (is_out and not ws and not missing):
            malformed += 1
            junk += max(1, len(GROUP_RE.findall(line)))  # what it names, priced as nodes that are not in the graph
            continue
        for r in rs:
            if r != "out":
                index(r)
        for w in ws:
            i = index(w)
            edges += [[i, "out" if r == "out" else at[r]] for r in rs]
    def node_ref(x, persistent):  # an id, a node written out, or (persistent) a subcomponent without a position
        if x in ids:
            return ids[x]
        if x.lower() == "out":
            return "out"
        g = group_of(x) if "@" in x else (group_of(f"{x}@0") if persistent else None)
        return None if not g or len(g) != 1 else ((g[0][0], None, g[0][2]) if persistent and "@" not in x else g[0])

    edits = {}
    for m in EDIT_RE.finditer(text):
        ops = []
        for part in m[2].split(";"):
            if not part.strip():
                continue
            o = OP_RE.fullmatch(part)
            if not o:
                ops.append({"op": "invalid", "text": part.strip()})
                continue
            persistent = bool(o[1])
            if o[2]:
                r, w = node_ref(o[2], persistent), node_ref(o[3], persistent)
                ops.append({"op": "cut", "reader": r, "writer": w, "persistent": persistent} if r and w and w != "out"
                           else {"op": "invalid", "text": part.strip()})
            else:
                x = node_ref(o[4], persistent)
                ops.append({"op": "scale", "node": x, "k": float(o[5]), "persistent": persistent} if x and x != "out"
                           else {"op": "invalid", "text": part.strip()})
        edits.setdefault(int(m[1]), ops)
    finals = {}
    for m in FINAL_RE.finditer(text):
        finals.setdefault(int(m[1]), (m[2].lower() == "yes", m[3].strip()))
    explanation = sec.get("EXPLANATION", "")
    variables, prose, bad_vars = _variables(explanation, ids)
    answer = {"nodes": nodes, "edges": edges + pairs, "explanation": prose if variables or bad_vars else explanation}
    if complete:
        answer["complete"] = True
    if variables:
        answer["variables"] = variables
    return {"explanation": explanation, "answer": answer, "ids": ids, "malformed": malformed, "malformed_nodes": junk + unknown,
            "unknown_ids": unknown, "circuit": "CIRCUIT" in sec, "finals": finals, "edits": edits, "variables_malformed": bad_vars}


def _parse_mermaid(lines: list[str], index) -> tuple[dict, list, int, int, int]:
    """A Mermaid CIRCUIT (circuit_lines()'s form) read strictly: (ids, edges in circuit JSON by index(node), lines
    that do not parse, the nodes they cost, edge ends naming an undeclared id). Every node is declared once inside its
    position's subgraph (subgraph P<position>[...]) as id["L<layer>.<kind>#<index>"], id["L<layer>.<kind> block"] or
    id[tok]; a declaration outside a subgraph, a label that is no subcomponent, a second declaration of an id, an edge
    end declared inline or never declared, and any other line are defects. "w --> r" one edge per line ("&" lists
    taken), "--> OUT" (OUT((prediction)) once) the prediction; "q -.->|pair| k" a query-key pair of q's read, rest the
    part outside the circuit, "|pair L<layer>@<position>|" when q is rest; a "%%" comment may hold a rest-rest pair."""
    ids, edges, bad, junk, unknown, pos, later = {}, [], 0, 0, 0, None, []
    for line in lines[1:]:
        if line.replace(" ", "") == "%%complete":
            edges.append("complete")
            continue
        if line.startswith("%%"):
            m = MM_PAIR.fullmatch(line[2:].strip())
            if m:
                later.append(m)
            continue
        sub = MM_SUB.fullmatch(line)
        if sub or line == "end" or line.startswith("direction "):
            pos = int(sub[1]) if sub else (pos if line.startswith("direction ") else None)
            continue
        if MM_DECLS.fullmatch(line):
            for i, quoted, plain in MM_DECL.findall(line):
                label = (quoted or plain).strip()
                nd = None
                if pos is not None and i not in MM_RESERVED:
                    g = [(TOK, pos, 0)] if label == TOK else group_of(f"{label[:-len(' block')] if label.endswith(' block') else label}@{pos}")
                    nd = g[0] if g and len(g) == 1 and (label.endswith(" block")) == (g[0][2] == BLOCK) else None
                if nd is None or ids.get(i, nd) != nd:
                    bad, junk = bad + 1, junk + 1  # outside a subgraph, no subcomponent, or an id declared twice
                    continue
                ids[i] = nd
                index(nd)
            continue
        later.append(line)
    for line in later:
        m = line if not isinstance(line, str) else (MM_PAIR.fullmatch(line) or MM_EDGE.fullmatch(line))
        if m is None:
            bad, junk = bad + 1, junk + max(1, len(MM_DECL.findall(line)))
            continue
        if m.re is MM_PAIR:
            q, k = (m[1] if m[1] == "rest" else ids.get(m[1])), (m[4] if m[4] == "rest" else ids.get(m[4]))
            read = (int(m[2]), int(m[3])) if m[2] else ((int(q[0].split(".")[1]), q[1]) if q not in (None, "rest") and q[0] != TOK else None)
            if q is None or k is None or read is None:
                bad, unknown = bad + 1, unknown + (q is None) + (k is None)
                continue
            edges.append(["pair", "rest" if q == "rest" else index(q), "rest" if k == "rest" else index(k)] + list(read))
            continue
        ws = [x.strip() for x in m[1].split("&")]
        rs = ["out"] if MM_OUT.fullmatch(m[2]) else [x.strip() for x in m[2].split("&")]
        for w in ws:
            for r in rs:
                if w not in ids or (r != "out" and r not in ids):
                    unknown += 1  # an undeclared id: a node outside the graph, charged; the line states nothing
                    continue
                edges.append([index(ids[w]), "out" if r == "out" else index(ids[r])])
    return ids, edges, bad, junk, unknown


def _variables(text: str, ids: dict) -> tuple[list[dict], str, int]:
    """The EXPLANATION's VARIABLES block (variable_text()'s lines), its prose (every other line) and the variable lines
    that are malformed: a line that does not parse, or names no subcomponent (by the circuit's ids or written out; the
    token embedding is no subcomponent). Variables in variables.py's JSON: {"name", "meaning", "inputs" (positions),
    "group" ([matrix, index, position], as circuit JSON nodes, or ["block", matrix, position], a site's whole output
    there), "feeds" (variable names or "prediction"), "instances" (informational, [matrix, index, position])}."""
    got, prose, bad = [], [], 0
    for line in text.splitlines():
        s = line.strip()
        if s.rstrip(":").strip().upper() == "VARIABLES":
            continue
        if not VAR_LINE.match(s):
            prose.append(line)
            continue
        m = VAR_RE.fullmatch(s)
        group, instances = _carriers(m[5], ids) if m else ([], [])
        if not group:
            bad += 1
            continue
        rest = QUOTED.sub("", m[4])
        inputs = {int(x) for x in re.findall(r"@(\d+)", rest)} or {int(x) for x in re.findall(r"\b(\d+)\b", rest)}
        feeds = [x.strip() for x in m[6].split(",") if x.strip()]
        got.append((int(m[1]), {"name": m[2], "meaning": m[3].strip(), "inputs": sorted(inputs), "group": group, "feeds": feeds}
                    | ({"instances": instances} if instances else {})))
    names = {f"V{k}": v["name"] for k, v in got}
    for _, v in got:  # V<j> -> its name, PREDICTION (or OUT) -> "prediction"
        v["feeds"] = ["prediction" if f.upper() in ("PREDICTION", "OUT") else names.get(f.upper(), f) for f in v["feeds"]]
    return [v for _, v in got], "\n".join(prose).strip(), bad


def _carriers(text: str, ids: dict) -> tuple[list, list]:
    """A variable's "carried by" field: (its group in variables.py's JSON, its instances). Blocks "L1.v @232-233" (each
    position a ["block", matrix, position]) and subcomponents by the circuit's ids or written out ([matrix, index,
    position]; the token embedding is none); "(instances: ...)" at the end lists subcomponents inside the blocks, kept
    apart (not scored). A field with anything else in it gives no group."""
    import native

    inst = []
    m = INSTANCES_RE.search(text)
    if m:
        text = text[:m.start()]
        if REFS_RE.fullmatch(m[1]):
            inst = sorted({(nd[0], nd[1], nd[2]) for w in GROUP_SPLIT.findall(m[1])
                           for nd in ([ids[w]] if w in ids else (group_of(w) or [])) if nd[0] != TOK})
    if not CARRIERS_RE.fullmatch(text):
        return [], []
    group = set()
    for x in CARRIER_RE.findall(text):
        b = BLOCK_ITEM_RE.fullmatch(x)
        if b:
            site = native.site_name(int(b[1]), KINDS[b[2]])
            group |= {("block", site, t) for t in range(int(b[3]), int(b[4] or b[3]) + 1)}
        else:
            group |= {("block", nd[0], nd[1]) if nd[2] == BLOCK else (nd[0], nd[2], nd[1])
                      for nd in ([ids[x]] if x in ids else (group_of(x) or [])) if nd[0] != TOK}
    return [list(g) for g in sorted(group, key=str)], [[n, c, t] for n, t, c in inst]


def _place(nd) -> tuple:
    """A node's (or a (matrix, position) group's) place in rendered text: latest position, then latest layer, then
    kind (q, k, v, o, fc, down), then index."""
    if nd[0] == TOK:
        return (-nd[1], 1, 0) + tuple(nd[2:3])  # the embedding enters the stream first: last in a backward listing
    return (-nd[1], -int(nd[0].split(".")[1]), list(KINDS.values()).index(nd[0].split(".")[-1])) + tuple(nd[2:3])


def render(answer: dict, finals: dict | None = None, reasons: dict | None = None, edits: dict | None = None,
           strings: list[str] | None = None) -> str:
    """The text of a circuit in goal_eval's JSON (to_json's): the CIRCUIT as a fenced Mermaid flowchart
    (circuit_lines(): every node declared once in its position's subgraph, one edge per line, the stated query-key
    pairs; strings, the text's token strings, label the positions), one output grammar for every answer; then its
    explanation (none when it has none: its VARIABLES block first when it has "variables", variable_text(), each input
    with its token when strings are given), the questions' answers (finals: question number -> (falls, the top token
    as written)) and the edits, nodes by the circuit's ids."""
    import goal_eval

    nodes = [(goal_eval._site(c), t, BLOCK) if m == "block" else (TOK if m == TOK else goal_eval._site(m), t, c) for m, c, t in answer["nodes"]]
    lines, name = circuit_lines(nodes, answer["edges"], strings, complete=bool(answer.get("complete")))
    text = ["CIRCUIT:"] + lines
    if answer.get("variables"):
        k = {v["name"]: f"V{i}" for i, v in enumerate(answer["variables"], 1)}
        text += ["EXPLANATION:", "VARIABLES:"] + [variable_text(v, i, k, name, strings) for i, v in enumerate(answer["variables"], 1)]
        text += [answer["explanation"]] if answer.get("explanation") else []
    elif answer.get("explanation"):
        text.append(f"EXPLANATION: {answer['explanation']}")
    for k in sorted(finals or {}):
        falls, top = finals[k]
        text += [f"ANSWER {k}: {(reasons or {}).get(k, '')}".rstrip(), f"FINAL {k}: falls={'yes' if falls else 'no'} top={top}"]
    for k in sorted(edits or {}):
        text.append(f"EDIT {k}: " + "; ".join(edit_op_text(o, name) for o in edits[k]))
    return escape("\n".join(text))


def variable_text(v: dict, k: int, numbers: dict, name: dict, strings: list[str] | None = None) -> str:
    """A variable's line: V<k> "<name>": <meaning>; inputs: @<position> <token>, ...; carried by: <blocks, L1.v @232-233
    (a site's whole output at those positions)>, <subcomponents by id, written out outside the circuit> (instances:
    <subcomponents inside the blocks, informational>); feeds: V<j>, ... or PREDICTION. v in variables.py's JSON;
    numbers: variable name -> V<j>; name: the circuit's ids (circuit_lines())."""
    import goal_eval

    blocks = {}
    for it in v["group"]:
        if it[0] == "block":
            blocks.setdefault(goal_eval._site(it[1]), []).append(int(it[2]))
    group = sorted(((goal_eval._site(m), t, c) for m, c, t in (it for it in v["group"] if it[0] != "block")), key=_place)
    ref = lambda nd: name[nd] if nd in name else node_text(nd)  # noqa: E731
    items = []
    for site in sorted(blocks, key=lambda n: _place((n, max(blocks[n]), 0))):
        ts = sorted(set(blocks[site]))
        runs = [[ts[0], ts[0]]]
        for t in ts[1:]:
            if t == runs[-1][1] + 1:
                runs[-1][1] = t
            else:
                runs.append([t, t])
        short = f"L{site.split('.')[1]}.{CODES[site.split('.')[-1]]}"
        items += [f"{short} @{a}" + (f"-{b}" if b > a else "") for a, b in sorted(runs, reverse=True)]
    items += [ref(nd) for nd in group]
    inst = sorted(((goal_eval._site(m), t, c) for m, c, t in v.get("instances") or []), key=_place)
    carried = ", ".join(items) + (" (instances: " + ",".join(ref(nd) for nd in inst) + ")" if inst else "")
    inputs = ", ".join(f"@{p} {shown(strings[p])}" if strings else f"@{p}" for p in v["inputs"])
    feeds = ", ".join("PREDICTION" if f == "prediction" else numbers.get(f, f) for f in v["feeds"]) or "PREDICTION"
    return f'V{k} "{v["name"]}": {v["meaning"]}; inputs: {inputs}; carried by: {carried}; feeds: {feeds}'


def edit_op_text(o: dict, name: dict) -> str:
    """An edit operation (parse()'s form) in the answer grammar: nodes by the circuit's ids where it has them, else
    written out (without a position when persistent)."""
    def ref(nd):
        if nd == "out":
            return "out"
        if nd in name:
            return name[nd]
        text = node_text((nd[0], nd[1] if nd[1] is not None else 0, nd[2]))
        return text.rsplit("@", 1)[0] if nd[1] is None else text
    head = "persistent " if o.get("persistent") else ""
    if o["op"] == "cut":
        return f"{head}cut {ref(tuple(o['reader']) if o['reader'] != 'out' else 'out')} <- {ref(tuple(o['writer']))}"
    return f"{head}scale {ref(tuple(o['node']))} {o['k']:g}"


def _short_ids(n: int) -> list[str]:
    """n short ids a, b, ..., z, aa, ab, ... ("out" and "tok" skipped)."""
    out, i = [], 0
    while len(out) < n:
        s, x = "", i
        while True:
            s = chr(97 + x % 26) + s
            x = x // 26 - 1
            if x < 0:
                break
        if s not in ("out", "tok"):
            out.append(s)
        i += 1
    return out


def circuit_lines(nodes: list, edges: list, strings: list[str] | None = None, complete: bool = False) -> tuple[list[str], dict]:
    """A circuit (its nodes as (matrix, position, index) tuples, goal_eval's JSON edges by place) as the fenced Mermaid
    CIRCUIT the oracle writes, and each node's id: per position, the latest first, a subgraph
    P<position>["@<position> '<token>'"] declaring its nodes once, a["L3.fc#1314"], b["L1.v block"] (a block), t394[tok]
    (the token embedding); then one edge per line from the prediction backward, "c --> OUT((prediction))" first and
    "--> OUT" after, "w --> r"; then the query-key pairs, "q -.->|pair| k" (q's layer and position are the read's),
    "rest -.->|pair L3@394| k" with the read written when q is rest, and a pair with neither side, the read with no
    pair of its own, as a "%%" comment. complete: a node set claiming every connection among its nodes and every token
    edge (goal_eval's {"complete": true}, VPD's answer and the influential start), a "%% complete" line and no edges."""
    parents, out, pairs = {}, [], []
    for e in edges:
        if e[0] == "pair":
            pairs.append(e)
            continue
        w, r = e
        if r == "out":
            out.append(nodes[w])
        else:
            parents.setdefault(nodes[r], set()).add(nodes[w])
    order = sorted(set(nodes), key=_place)
    letters = iter(x for x in _short_ids(len(order) + len(MM_RESERVED)) if x not in MM_RESERVED)
    name = {nd: f"t{nd[1]}" if nd[0] == TOK else next(letters) for nd in order}
    at = {}
    for nd in order:
        at.setdefault(nd[1], []).append(nd)
    lines = ["```mermaid", "flowchart LR"] + (["  %% complete"] if complete else [])
    for t in sorted(at, reverse=True):
        label = f"@{t}" + (f" {strings[t]!r}" if strings else "")
        lines.append(f'  subgraph P{t}["{label.replace(chr(34), "#quot;")}"]')
        for nd in at[t]:
            text = TOK if nd[0] == TOK else group_text([nd]).rsplit("@", 1)[0] + (" block" if nd[2] == BLOCK else "")
            lines.append(f"    {name[nd]}[{text}]" if nd[0] == TOK else f'    {name[nd]}["{text}"]')
        lines.append("  end")
    for j, w in enumerate(sorted(set(out), key=_place)):
        lines.append(f"  {name[w]} --> OUT" + ("((prediction))" if j == 0 else ""))
    seen, queue = set(), sorted(set(out), key=_place)
    readers = []
    while queue:
        r = queue.pop(0)
        if r in seen or r not in parents:
            continue
        seen.add(r)
        readers.append(r)
        queue += sorted(parents[r], key=_place)
    readers += sorted(set(parents) - seen, key=_place)
    lines += [f"  {name[w]} --> {name[r]}" for r in readers for w in sorted(parents[r], key=_place)]
    for _, q, k, layer, t in sorted(pairs, key=lambda e: (-e[4], -e[3], str(e[1]), str(e[2]))):
        qn, kn = ("rest" if q == "rest" else name[nodes[q]]), ("rest" if k == "rest" else name[nodes[k]])
        own = q != "rest" and nodes[q][0] != TOK and (int(nodes[q][0].split(".")[1]), nodes[q][1]) == (layer, t)
        pair = f"{qn} -.->|pair{'' if own else f' L{layer}@{t}'}| {kn}"
        lines.append(f"  %% {pair}" if q == k == "rest" else f"  {pair}")
    return lines + ["```"], name


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
    removed (Native.without; at F2's positions and the targets), a change of the text's order around its influential
    positions (structure_changes(): a token moved, a copy inserted earlier, two swapped), one connection cut (cuts: goal_eval.GoalEval.cuts with
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
    pool["order"] = []
    for ch in structure_changes(ids, targets, positions, rng):
        x, tg = apply_change(ids, targets, ch)
        with torch.no_grad():
            pool["order"].append(outcome(nat.reference([x], tg), **ch))
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


def structure_questions(nat, task: dict, positions: list[int], seed: int, per_kind: int = QA_PER_KIND) -> list[dict]:
    """The oracle's structural questions on a text with existing rows (qa_questions' "order" kind alone): per_kind of
    structure_changes() drawn by seed, measured, half whose answer is "falls" where the text has them."""
    import torch

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    rng = random.Random(seed)
    with torch.no_grad():
        top, lp0 = _top(nat.reference([ids], targets))
        pool = []
        for ch in structure_changes(ids, targets, positions, rng):
            x, tg = apply_change(ids, targets, ch)
            lp = nat.reference([x], tg)
            bits = (float(lp[0, -1, top]) - lp0) / math.log(2)
            pool.append({**ch, "falls": bits < -1.0, "top": _top(lp)[0], "bits": bits})
    yes = [q for q in pool if q["falls"]]
    no = [q for q in pool if not q["falls"]]
    rng.shuffle(yes)
    rng.shuffle(no)
    k_yes = min(len(yes), max(per_kind // 2, per_kind - len(no)))
    return yes[:k_yes] + no[:per_kind - k_yes]


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


STRUCTURE = ("move", "swap", "insert")  # the changes of order structure_changes() draws


def apply_change(ids: list[int], targets: list[int], ch: dict) -> tuple[list[int], list[int]]:
    """A text and its targets after a structural change: "move" takes the token at "position" out and puts it back so
    it ends at "to" (the tokens between shift by one; the targets stay); "swap" exchanges the tokens at "position" and
    "with"; "insert" writes a copy of the token at "position" at "at" (earlier in the text), every later token and the
    targets one further on."""
    p = ch["position"]
    if ch["kind"] == "move":
        x = ids[:p] + ids[p + 1:]
        return x[:ch["to"]] + [ids[p]] + x[ch["to"]:], list(targets)
    if ch["kind"] == "swap":
        x = list(ids)
        x[p], x[ch["with"]] = x[ch["with"]], x[p]
        return x, list(targets)
    return ids[:ch["at"]] + [ids[p]] + ids[ch["at"]:], [t + 1 for t in targets]


def change_map(ch: dict, T: int) -> list[int]:
    """Each position's place after a structural change (apply_change()): old position -> new."""
    p = ch["position"]
    if ch["kind"] == "move":
        return [ch["to"] if i == p else (i if i < p else i - 1) + ((i if i < p else i - 1) >= ch["to"]) for i in range(T)]
    if ch["kind"] == "swap":
        return [ch["with"] if i == p else (p if i == ch["with"] else i) for i in range(T)]
    return [i if i < ch["at"] else i + 1 for i in range(T)]


def change_text(ch: dict, strings: list[str]) -> str:
    """A structural change in words (the original text's positions and tokens)."""
    p = ch["position"]
    if ch["kind"] == "move":
        return f"the token at position {p}, {strings[p]!r}, were moved to position {ch['to']} (the tokens between shifting by one)"
    if ch["kind"] == "swap":
        return f"the tokens at positions {p} ({strings[p]!r}) and {ch['with']} ({strings[ch['with']]!r}) were swapped"
    return (f"a copy of the token at position {p}, {strings[p]!r}, were inserted at position {ch['at']}, before "
            f"{strings[ch['at']]!r} (every later token shifting by one)")


def structure_changes(ids: list[int], targets: list[int], positions: list[int], rng, each: int = 2) -> list[dict]:
    """Changes of the text's order around its influential context positions (positions, most influential first: F2's):
    the most influential token moved elsewhere (at least 3 positions away), a copy of it inserted earlier in the text,
    and two of the four most influential tokens (different ones) swapped; up to `each` of every kind, drawn by rng.
    Where a mechanism reads a position, a move or an insertion changes what it finds there; where it reads a token
    wherever it is, they do not."""
    t = max(targets)
    ctx = [p for p in positions if 0 < p < t]
    out = []
    if ctx:
        p = ctx[0]
        far = [q for q in range(1, t) if abs(q - p) >= 3]
        out += [{"kind": "move", "position": p, "to": q} for q in rng.sample(far, min(each, len(far)))]
        early = list(range(1, p))
        out += [{"kind": "insert", "position": p, "at": q} for q in rng.sample(early, min(each, len(early)))]
    pairs = [(a, b) for i, a in enumerate(ctx[:4]) for b in ctx[i + 1:4] if ids[a] != ids[b]]
    out += [{"kind": "swap", "position": a, "with": b} for a, b in rng.sample(pairs, min(each, len(pairs)))]
    return out


def qa_text(q: dict, n: int, strings: list[str], top: str, tokens_of) -> str:
    """Question n in words."""
    if q["kind"] in STRUCTURE:
        what = change_text(q, strings)
    elif q["kind"] == "token":
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


QA_KINDS = {"token": "changed token", "move": "move/swap/insert", "swap": "move/swap/insert", "insert": "move/swap/insert",
            "remove": "subcomponent removal", "cut": "connection cut", "copy": "copy"}  # E2's report by question type


def qa_detail(questions: list[dict], finals: dict, tokens_of, top: str) -> list[dict]:
    """Per question answered, its type (QA_KINDS), the answer's score (half for falls, half for the top token) and the
    no-change answer's ("falls=no, top=current prediction"): E2's gain over the no-change answer by question type."""
    out = []
    for n, q in enumerate(questions, 1):
        if n in finals:
            falls, tp = finals[n]
            want = shown(tokens_of([q["top"]])[0])
            out.append({"kind": QA_KINDS.get(q.get("kind"), q.get("kind")), "right": 0.5 * (falls == q["falls"]) + 0.5 * (tp == want),
                        "right0": 0.5 * (not q["falls"]) + 0.5 * (shown(top) == want)})
    return out


def qa_term(questions: list[dict], finals: dict, tokens_of, top: str) -> float | None:
    """E2: the share of questions answered right (qa_score) measured from the share right0 of the no-change answer
    (every question answered falls=no, top=the model's prediction on the text, `top`): (right - right0) / (1 - right0),
    clipped to [-1, 1]. 0 for an answer that says nothing changes, 1 for every question right (E1 is likewise measured
    from no English). None without questions, or when the no-change answer is right on all of them."""
    right = qa_score(questions, finals, tokens_of)
    if right is None:
        return None
    right0 = qa_score(questions, {n: (False, shown(top)) for n in range(1, len(questions) + 1)}, tokens_of)
    if right0 >= 1.0:
        return None
    return max(-1.0, min(1.0, (right - right0) / (1.0 - right0)))


# ---- the question


ASK = ("How does {model} compute its prediction of the next token after the target of this text? State the circuit in "
       "VPD subcomponents: a subcomponent is one rank-one slice of a weight matrix at a position, written "
       "L<layer>.<kind>#<index> (kind q, k, v, o, fc or down; for example L2.v#1514 at position 322); the token "
       "embedding at a position is tok, and a block, L2.v at a position, is a matrix's whole output there. Write the "
       "CIRCUIT as a Mermaid flowchart in a ```mermaid fence: flowchart LR, then for each position a subgraph, "
       "subgraph P322[\"@322 ' the'\"], declaring each of its nodes once by a short id, c[\"L2.v#1514\"], b[\"L1.v block\"], "
       "t322[tok], and end; then one edge per line, <writer> --> <reader>, by id: an attention or MLP output (o, down) into "
       "a later query, key, value or MLP input (q, k, v, fc) at its position; a value (v) into its layer's attention "
       "output (o) at that or a later position; an MLP input (fc) into its MLP's output (down); the token embedding into a "
       "reader at its position; <writer> --> OUT((prediction)) is the prediction (OUT after the first). A line "
       "<q> -.->|pair| <k> states a query-key pair of q's attention read: its pattern computed from query q against key k "
       "(rest: the query's or key's part outside the circuit, rest -.->|pair L<layer>@<position>| <k> when the query is "
       "rest). The circuit should be what makes the "
       "prediction: with only its subcomponents on, reading along its edges, and every other one off, the model should "
       "still predict what it does, and with them taken out of VPD's active subcomponents it should not; fewer "
       "subcomponents and edges are better. Answer with CIRCUIT: and its lines first, then a line EXPLANATION: (in "
       "plain English, what that circuit computes and how), then for each question ANSWER <n>: (your reasoning in plain English) and "
       "FINAL <n>: falls=yes or no top=<the token, quoted as the tokens are>, and for each edit request EDIT <n>: the "
       "operations, separated by \";\": cut <reader> <- <writer> (that one connection removed on this text; out is the "
       "prediction), scale <node> <k> (its activation multiplied by k), either prefixed by persistent for an edit of the "
       "weights on every text (nodes then written without a position).")


ASK_VARIABLES = ("Open the EXPLANATION with a line VARIABLES: and one line per variable the circuit computes, V<k> "
                 "\"<short name>\": <its meaning in plain English>; inputs: @<position> <token>, ... (the context tokens it "
                 "is computed from); carried by: <the ids of the circuit's subcomponents that carry it>; feeds: V<j>, ... "
                 "or PREDICTION; then explain the prediction in those variables, never naming a subcomponent.")


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


def facts(atlas, nd, routed: dict | None = None, tok=None) -> str:
    """A subcomponent's facts in a few words: the tokens its write raises most at this position by the routed lens
    (routed: RoutedLens.readout's reading of it on this text, tok: token ids -> strings; a label-free linear reading
    of its write through the expert its residual there is routed to, oriented by its activation there, not a measured
    effect; none for q and k, which write nothing) and the tokens it is most often active on (the atlas's top contexts
    on Pile text, method/vpd/atlas.py)."""
    from collections import Counter

    e = atlas.entry(nd[0], nd[2])
    out = []
    if routed and tok:
        out.append("routed lens at this position raises " + " ".join(repr(t) for t in tok([i for i, _ in routed["raised"][:3]])))
    on = Counter(c["token"] for c in e["contexts"]).most_common(2)
    if on:
        out.append("active on " + " ".join(repr(t) for t, _ in on))
    return "; ".join(out)


def _lens(nat):
    """The routed lens on the target model (method/vpd/lens.py RoutedLens), or None while its fit is not built."""
    sys.path.insert(0, str(HERE.parents[1]))
    from method.vpd import lens

    return lens.RoutedLens(nat.target) if (lens.DEFAULT / "lens" / "jacobians.pt").exists() else None


def refresh_facts(atlas, lens, task: dict, ev: dict) -> dict:
    """The evidence with every listed subcomponent's facts() read again on this text (the routed lens at its position:
    one forward pass of the text)."""
    ids = task["prompts"][0]["token_ids"]
    nodes = [tuple(nd) for cs in ev["components"].values() for nd, *_ in cs]
    routed = dict(zip(nodes, lens.readout(ids, nodes))) if lens else {}
    tok = lambda xs: token_strings(task["model"], xs)  # noqa: E731
    return {**ev, "components": {q: [[nd, c, facts(atlas, tuple(nd), routed.get(tuple(nd)), tok)] for nd, c, _ in cs]
                                 for q, cs in ev["components"].items()}}


def measured(nat, task: dict, ev: dict, atlas=None, lens=None, k: int = EVIDENCE_PER_POSITION, top: int = ATTENTION_TOP) -> dict:
    """The evidence with the --measured input arm's facts (the lead's oracle input v2, 10-12: what the oracle can
    measure in a forward pass, so it need not memorize 38,912 subcomponents' roles): at each listed position and the
    targets VPD's active subcomponents (causal importance above 0) ranked by causal importance x |activation| (VPD's
    order ranks by importance alone), k of them (twice at the targets) with their facts() and activations ("acts"), and
    each attention head's top positions at the last target, with their weights ("attention": layer -> per head
    [[position, weight], ...]). The plain question is unchanged; question() shows these when present."""
    import torch

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    ci, act = nat.importance(ids), nat.activations(ids)
    tok = lambda xs: token_strings(task["model"], xs)  # noqa: E731
    comps, acts = {}, {}
    for q in ev["components"]:
        p = int(q)
        cand = [(n, p, c) for n in nat.names for c in torch.nonzero(ci[n][p] > 0).flatten().tolist()]
        cand.sort(key=lambda nd: -float(ci[nd[0]][p, nd[2]] * act[nd[0]][p, nd[2]].abs()))
        chosen = cand[:k * (2 if p in targets else 1)]
        routed = dict(zip(chosen, lens.readout(ids, chosen))) if lens and chosen else {}
        comps[q] = [[list(nd), float(ci[nd[0]][p, nd[2]]), facts(atlas, nd, routed.get(nd), tok) if atlas else ""] for nd in chosen]
        acts[q] = [round(float(act[nd[0]][p, nd[2]]), 3) for nd in chosen]
    return {**ev, "components": comps, "acts": acts, "attention": attention_at(nat, ids, max(targets), top)}


def attention_at(nat, ids: list[int], t: int, top: int) -> dict:
    """Each attention head's top positions at position t in the plain model: layer -> per head [[position, weight],
    ...] (the target's q_proj and k_proj outputs through Target.attention_pattern)."""
    import torch

    tg = nat.target
    sites = [tg.site(f"h.{i}.attn.{x}") for i in range(tg.n_layer) for x in ("q_proj", "k_proj")]
    nat.vpd.clear()  # the plain model (clear() also resets the sites' caching)
    for sx in sites:
        sx.cache_output = True
    att = {}
    with torch.no_grad():
        tg(torch.tensor([ids], device=nat.dev))
        for i in range(tg.n_layer):
            P = tg.attention_pattern(sites[2 * i].last_output, sites[2 * i + 1].last_output)[0, :, t]  # [heads, T]
            w, at = P.topk(min(top, t + 1), -1)
            att[str(i)] = [[[int(a), round(float(b), 3)] for a, b in zip(ai, wi)] for ai, wi in zip(at.tolist(), w.tolist())]
    for sx in sites:
        sx.cache_output, sx.last_output = False, None
    return att


def candidate_pool(nat, task: dict, ev: dict, heads: int = POOL_HEADS) -> dict:
    """The evidence with the full candidate pool (the lead's v2 question, 10-12: the v1 question listed 16% of the
    teacher's nodes, so choosing a circuit was guessing): every VPD-active subcomponent (causal importance above 0)
    at the listed positions and the targets, and with heads > 0 at the positions each attention head at the target
    reads most (its top `heads`). Every v1 teacher node is VPD-active; the listed positions and targets hold 96% of the
    20 held-out teachers' nodes (81% of 60 train teachers'), the question then ~11k Qwen3 tokens (median; max 18k);
    adding each head's top 8 positions reaches 97% at ~46k. "pool": position -> matrix -> [[index, importance], ...],
    most important first."""
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    ci = nat.importance(ids)
    att = attention_at(nat, ids, max(targets), heads) if heads else {}
    where = {int(q) for q in ev["components"]} | set(targets) | {q for rows in att.values() for row in rows for q, _ in row}
    pool = {}
    for p in sorted(where):
        per = {}
        for n in nat.names:
            got = sorted(((c, float(ci[n][p, c])) for c in (ci[n][p] > 0).nonzero().flatten().tolist()), key=lambda x: -x[1])
            if got:
                per[n] = [[c, round(v, 2)] for c, v in got]
        if per:
            pool[str(p)] = per
    return {**ev, "pool": pool}


def fit_pool(task: dict, ev: dict, count, budget: int = CONTEXT - ROOM - 64) -> dict:
    """The evidence with its candidate pool cut to the most important subcomponents whose question fits `budget` tokens
    (count: text -> Qwen3 tokens; the chat template's own tokens are in the 64), so its generation keeps at least ROOM
    of the context. The least important entries go first, across positions; "pool_cut" counts them. On the 444 v1
    train texts it cuts 52 questions (12 of them longer than the whole context uncut), which hides 21% of those texts'
    teacher nodes; text6040 leaves 6,843 tokens with no pool at all; none of the 20 held-out questions is cut."""
    if count(question(task, ev)) <= budget:
        return ev
    order = sorted((v, int(q), n, c) for q, per in ev["pool"].items() for n, cs in per.items() for c, v in cs)

    def drop(k):
        gone = {(q, n, c) for _, q, n, c in order[:k]}
        pool = {q: {n: [[c, v] for c, v in cs if (int(q), n, c) not in gone] for n, cs in per.items()} for q, per in ev["pool"].items()}
        pool = {q: {n: cs for n, cs in per.items() if cs} for q, per in pool.items()}
        return {**ev, "pool": {q: per for q, per in pool.items() if per}, "pool_cut": k}

    lo, hi = 0, len(order)  # the fewest dropped that fit: the question shrinks as entries go
    while lo < hi:
        mid = (lo + hi) // 2
        if count(question(task, drop(mid))) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return drop(lo)


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


def evidence(nat, ge, task: dict, conn=None, qa_seed: int | None = None, atlas=None, lens=None) -> dict:
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
    routed = dict(zip(listed_nodes, lens.readout(ids, listed_nodes))) if lens else {}
    tok = lambda xs: token_strings(task["model"], xs)  # noqa: E731
    return {"task": task["id"], "positions": positions, "listed": listed, "events": events,
            "components": {str(p): [[list(nd), float(tx.ci[nd[0]][nd[1], nd[2]]), facts(atlas, nd, routed.get(nd), tok) if atlas else ""]
                                    for nd in comps[p]] for p in where},
            "wiring": {str(p): wires[p] for p in where}, "switch_on": switched, "qa": qa, "active": active}


def structure_events(nat, task: dict, positions: list[int], seed: int) -> list[dict]:
    """The English reader's structural events (structure_changes() drawn by seed, another draw than the oracle's
    questions): each change's measured effect on the model's most likely next token, as units.events records a changed
    token ("down", "weight" its |log2 ratio| capped at 4 bits, "bits"), with the change in words ("change")."""
    import torch

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    strings = token_strings(task["model"], ids)
    with torch.no_grad():
        top, lp0 = _top(nat.reference([ids], targets))
        out = []
        for ch in structure_changes(ids, targets, positions, random.Random(seed)):
            x, tg = apply_change(ids, targets, ch)
            bits = (float(nat.reference([x], tg)[0, -1, top]) - lp0) / math.log(2)
            out.append({**ch, "change": change_text(ch, strings), "down": bits < 0, "weight": min(4.0, abs(bits)), "bits": bits})
    return out


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
        named |= ({q[k] for k in ("position", "to", "with", "at") if k in q} if q["kind"] in STRUCTURE
                  else {q["position"]} if q["kind"] == "token" else {q["node"][1]} if q["kind"] == "remove"
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
    """The oracle's question for one text from its evidence(); a row marked "variables" (rerender --variables) also
    asks for the VARIABLES block (ASK_VARIABLES)."""
    import units

    model = task["model"]
    p = task["prompts"][0]
    strings = token_strings(model, p["token_ids"])
    targets = p["target_positions"]
    t = max(targets)
    top_tok = p["model_top"][0][0][0]
    top = ", ".join(f"{tok!r} {q:.2f}" for tok, q in p["model_top"][0][:3])
    lines = [ASK.format(model=model) + (" " + ASK_VARIABLES if ev.get("variables") else ""),
             "tokens (numbered near the positions below; elsewhere plain text, [n] marking position n): "
             + listing(strings, windows(task, ev)),
             f"target: position {t}; the model's most probable next tokens there: {top}",
             "positions where a changed token moves the prediction by at least 1% of the total, most first: "
             + ", ".join(f"{q} ({strings[q]!r})" for q in ev["positions"])]
    if ev["events"]:
        lines += ["measured at those positions (log2 change of the top token's probability):"] + units.evidence(ev["events"], ev["listed"], strings)
    if ev.get("attention"):  # measured(): each head's most-read positions at the target
        lines.append(f"attention at the target ({t}), the positions each head reads most (weight):")
        lines += [f"L{layer}: " + "; ".join(f"h{h} " + ", ".join(f"{q} {w:.2f}" for q, w in row) for h, row in enumerate(rows))
                  for layer, rows in sorted(ev["attention"].items(), key=lambda x: int(x[0]))]
    acts = ev.get("acts") or {}
    lines.append("VPD's active subcomponents (causal importance" + ("; activation" if acts else "") + "; the tokens its write "
                 "raises at this position by the routed lens, a label-free linear reading of its write, not a measured effect; "
                 "the tokens it is active on in Pile text), " + ("ranked by causal importance x |activation|" if acts else
                 "most important first") + ", and the strongest weight-level connections among them (cosine; a static "
                 "coupling, not a measured effect):")
    for q in sorted(ev["components"], key=int):
        cs = ev["components"][q]
        a = acts.get(q) or [None] * len(cs)
        lines.append(f"{q} ({strings[int(q)]!r}): " + ", ".join(
            f"{node_text(tuple(nd))} ({c:.2f}{'' if x is None else f'; {x:+.2f}'}{'; ' + f if f else ''})" for (nd, c, f), x in zip(cs, a)))
        if ev.get("switch_on", {}).get(q):
            lines.append("  switched on when its token changes (of the 3 likeliest other tokens): "
                         + ", ".join(f"{node_text(tuple(nd))} ({k}/3)" for nd, k in ev["switch_on"][q]))
        if ev["wiring"].get(q):
            lines.append("  wiring: " + ", ".join(f"{node_text(tuple(w))} -> {node_text(tuple(r))} ({c:+.2f})" for w, r, c in ev["wiring"][q]))
    if ev.get("pool"):  # candidate_pool(): every VPD-active subcomponent at these positions
        lines.append("every VPD-active subcomponent (causal importance above 0) at these positions, by position and matrix, "
                     "index:importance, most important first (any subcomponent of the circuit is one of these; a block "
                     "L<layer>.<kind>@<position> is a matrix's whole output at a position):")
        for q in sorted(ev["pool"], key=int):
            lines.append(f"{q} ({strings[int(q)]!r}): " + "; ".join(
                f"L{n.split('.')[1]}.{CODES[n.split('.')[-1]]} " + " ".join(f"{c}:{v:.1g}".replace(":0.", ":.") for c, v in cs)
                for n, cs in sorted(ev["pool"][q].items(), key=lambda x: _place((x[0], int(q), 0)))))
    if ev["qa"]:
        tok = lambda ids: token_strings(model, ids)  # noqa: E731
        lines += [qa_text(q, n, strings, repr(top_tok), tok) for n, q in enumerate(ev["qa"], 1)]
    lines += [f"Edit {n}: {edit_goal_text(g, strings, repr(top_tok))}" for n, g in enumerate(ev.get("edits") or [], 1)]
    return escape("\n".join(lines))


def edit_goal_text(g: dict, strings: list[str], top: str) -> str:
    """An edit goal (the goal generator's JSON) as the question states it: its own sentence when it has one."""
    if g.get("question"):
        return g["question"]
    if g["kind"] == "flip":
        return f"make {top} no longer the most likely next token."
    if g["kind"] == "raise":
        return f"make {g['token']!r} the most likely next token."
    if g["kind"] == "detach":
        p = g["position"]
        return f"make the prediction not depend on the token at {p} ({strings[p]!r}): changing it should no longer move the prediction."
    return f"make the model stop {g.get('behaviour', g['kind'])} (persistent: on every text, not only this one)."


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


def circuit_text(text: str) -> str:
    """An answer's CIRCUIT lines as written (markdown decoration dropped), what the E1 reader reads with its English."""
    return _sections(MARKDOWN.sub("", text)).get("CIRCUIT", "")


def e1_candidates(ev: dict, k: int = E1_CANDIDATES) -> list[dict]:
    """The events E1 may ask: the text's units.reader_events (other positions and the listed positions' second tokens,
    never the oracle's evidence) and its structural events (structure_events()), the k with the largest measured
    effect (weight)."""
    import units

    evs = units.reader_events(ev["events"]) + ev.get("structure", [])
    return sorted(evs, key=lambda e: -e.get("weight", 1.0))[:k]


def e1_bits(reader, task: dict, ev: dict, texts: list[str]) -> list[float | None]:
    """Each answer's E1: what its English adds to its circuit for a strong reader (the lead's design, 10-12). The
    reader reads the text, the answer's CIRCUIT lines (circuit_text()) and the English, and gives its probability of
    what the model does on each event (reader.Reader.losses). The events: of the text's candidates (e1_candidates()),
    the E1_EVENTS heaviest by w = measured effect x the reader's loss given the text and the circuit alone, so an event
    the reader already gets right from the circuit, or one the model barely moves on (its direction noise), weighs
    little. E1 = sum w (loss with NEUTRAL - loss with the English) / sum w, in bits: English that says nothing scores
    0, English that tells the reader what the model does above 0, misleading English below (a strong reader loses bits
    on any confident English that the events do not bear out: reader validation 10-12, where every English scored
    below none). The unaided and neutral prompts are asked once per text and circuit (the reader's memo). An answer
    without English is 0; None without events."""
    cands = e1_candidates(ev)
    items = [(circuit_text(x), escape(english(x))) for x in texts]
    if not cands:
        return [None] * len(texts)
    circs = list(dict.fromkeys(c for c, e in items if e.strip()))
    sel = {}
    for c, un in zip(circs, reader.losses(task, [(c, "") for c in circs], cands)):
        w = [e.get("weight", 1.0) * u for e, u in zip(cands, un)]
        top = sorted(range(len(cands)), key=lambda i: -w[i])[:E1_EVENTS]
        sel[c] = ([w[i] for i in top], [cands[i] for i in top])
    asked = [(c, NEUTRAL) for c in circs] + [it for it in items if it[1].strip()]
    got = iter(reader.losses(task, asked, [sel[c][1] for c, _ in asked]))
    neutral = {c: next(got) for c in circs}
    out = []
    for c, e in items:
        if not e.strip():
            out.append(0.0)
            continue
        w, mine = sel[c][0], next(got)
        out.append(sum(a * (n - m) for a, n, m in zip(w, neutral[c], mine)) / sum(w) if sum(w) > 0 else 0.0)
    return out


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


def edit_bits(edits: dict, circuit: set, positions: int, subcomponents: int) -> float:
    """The description bits of an answer's edit operations, priced as the circuit's nodes and edges are (so edits join
    its size term with no weight of their own): each operation one bit (cut or scale) and one (persistent or not); a
    node by its id among the circuit's nodes log2(n), written out log2(positions x subcomponents), position-free
    (persistent) log2(subcomponents), the prediction log2(n); a scale factor its literal, log2(12) a character (digits,
    sign, point)."""
    n = max(len(circuit), 2)

    def ref(nd):
        if nd == "out" or tuple(nd) in circuit:
            return math.log2(n)
        return math.log2(subcomponents) if nd[1] is None else math.log2(positions * subcomponents)

    bits = 0.0
    for ops in edits.values():
        for o in ops:
            bits += 2.0
            if o["op"] == "cut":
                bits += ref(o["reader"]) + ref(o["writer"])
            elif o["op"] == "scale":
                bits += ref(o["node"]) + len(f"{o['k']:g}") * math.log2(12)
    return bits


_POOL = {}


def _pool() -> dict:
    """C's pattern pool (POOL), loaded once; a missing pool is an error, never a weaker C."""
    if "m" not in _POOL:
        import torch

        _POOL["m"] = torch.load(POOL)
    return _POOL["m"]


def e4_term(scorer, tx, graph, answer: dict, malformed: int, C: dict) -> dict:
    """E4: the answer's variables scored by interchange interventions, variables.Scorer's headline e4 (in the real
    model: each variable's min(f on-target, s off-target), composition and the feeds claims), times the share of the
    stated variables that are well-formed (a line that does not parse, or names no subcomponent of the model, counts
    as a variable scoring 0). No VARIABLES block, or nothing resolved: 0. {"E4", "model" (e4), "circuit" (the stated
    graph's e4, a diagnostic), "gap"}."""
    import variables

    ok, bad = [], malformed
    for v in answer.get("variables") or []:
        try:
            ok.append(variables.parse([v], C)[0])
        except (ValueError, KeyError):
            bad += 1
    if not ok:
        return {"E4": 0.0, "model": None, "circuit": None, "gap": None}
    got = scorer.score(tx, ok, graph=graph)
    per = []
    for v, x in zip(ok, got["variables"]):  # each variable's headline: min(f on-target, s off-target), 0 unresolved
        e = 0.0 if x.get("f_on") is None else (x["f_on"] if x.get("s_off") is None else min(x["f_on"], x["s_off"]))
        per.append({"name": v["name"], "e4": e, "nodes": [list(nd) for nd in v["group"]]})
    return {"E4": (got["e4"] or 0.0) * len(ok) / (len(ok) + bad), "model": got["e4"],
            "circuit": (got.get("circuit") or {}).get("e4"), "gap": got.get("gap"), "per_variable": per}


def softplus(x: float) -> float:
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def reward(ge, task: dict, ev: dict, texts: list[str], *, reader=None, e1: list | None = None, truncated: list | None = None,
           convention: str = CONVENTION, cache: dict | None = None, ee=None, edit_best: dict | None = None,
           pool: dict | None = None, vs=None, split: str | None = None, teacher: dict | None = None,
           lines: bool = False) -> list[dict]:
    """Per answer text on one task, reward v3 (the lead's 10-12: faithful first, then small and explained; every term
    in bits, every term ranking, nothing clipped flat): R = phi + g x Q + D.
      phi: goal_eval.phi of the answer's reward-mode row, -log2 of the power-4 mean of the families' residuals
        (1 - s + eps over point, nodes, edges, flips, track_pp, interchange_pp, donors near and far, pattern, N):
        each halving of the worst residual is about a bit, and the empty circuit is at or below 0.
      g = 1 / (1 + 2^(-4 (phi - phi_tau))), phi_tau the text's bar (ge.tau: the influential start's phi, or every
        family at 0.9 when lower): quality counts once the answer is faithful and fades below.
      Q = 0.5 compact + E2g + E4x + E1: compact = log2(bits_start / bits) (raw description bits, size_bits with
        blocks at their active contents, plus the bits charged for parts the model does not have and lines that do
        not parse; +1 per halving against the influential start); E2g the questions answered right in agreement with
        the answer's own circuit (e2g_term(): credit only where the FINAL line agrees with the model and the
        circuit's own verdict under the question's intervention); E4x the share of the circuit's mechanism (each
        node's measured pp removal effect, line_effects) explained by variables passing interchange, less the share
        claimed by failing ones (e4x(); vs a variables.Scorer); E1 the reader's bits (e1, or reader).
      D = softplus(2 m) / 2 with m = min(phi - phi_T, compact - compact_T) against the text's teacher circuit
        (teacher: its JSON): near 0 unless the answer beats it on both.
    An answer is invalid, R = V3_INVALID below every valid one, without a CIRCUIT section, a FINAL line per question
    or an EDIT line per edit request, cut at the sampling cap or over it (truncated), or above NODE_CAP nodes.
    Reported beside R: phi, phi_tau, g, faithful (phi >= phi_tau), bits, bits_start, compact, D, Q, each term
    (term_E2g, term_E4x, term_E1), and goal_eval's C (term_C, c_rl as term_C_rl), F1, N, F3, the old E2 (qa_term:
    from the no-change answer), E3, E4 (the scorer's headline), the per-question detail (qa), and C's members and
    columns (fidelity); circuit_reward = phi + g x 0.5 compact + D (the circuit alone, for references without answers).
    split: the draws of C's changed-input, interchange and donor families, "search" (the RL reward on training texts)
    or "eval" (every evaluation, evaluate()); pool: the flip adversary's pattern pool (POOL; {} leaves it out); cache:
    task -> goal_eval.Text, kept between calls. lines: each answer line's credit too ("lines", line_credit(): the
    token-level advantages of RL's per-line reward; line_effects in pp mode per answer)."""
    import goal_eval

    nat = ge.nat
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    tok = lambda xs: token_strings(task["model"], xs)  # noqa: E731
    total = sum(nat.C.values())
    P = nat.positions(targets)
    parsed = [parse(x) for x in texts]
    graphs, defects, valid = [], [], []
    for i, p in enumerate(parsed):
        g, d = goal_eval.circuit(p["answer"], targets, nat.C, len(ids))
        graphs.append(g)
        junk = p["malformed_nodes"] * math.log2(P * total)  # a line that does not parse: written, explains nothing
        defects.append((d["invalid_nodes"] + p["unknown_ids"], d["invalid_edges"], dropped_bits(p["answer"], g, d, P, total) + junk))
        valid.append(p["circuit"] and all(n in p["finals"] for n in range(1, len(ev["qa"]) + 1))
                     and all(n in p["edits"] for n in range(1, len(ev.get("edits") or []) + 1)) and not (truncated and truncated[i])
                     and g.count() <= NODE_CAP)
    keep = [i for i, v in enumerate(valid) if v]  # an invalid answer's reward is fixed whatever it measures: not run
    pool = _pool() if pool is None else pool
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
                "E2": lambda task, g, x: qa_term(ev["qa"], parse(x)["finals"], tok, prompt["model_top"][0][0][0])}
    row_of = dict(zip(keep, ge.score(tx, [graphs[i] for i in keep], [texts[i] for i in keep]))) if keep else {}
    c_of = dict(zip(keep, ge.fidelity(tx, [graphs[i] for i in keep], pool, mode="reward", split=split))) if keep else {}
    e4 = {i: e4_term(vs, tx, graphs[i], parsed[i]["answer"], parsed[i]["variables_malformed"], nat.C) for i in keep} if vs is not None else {}
    le = {i: ge.line_effects(tx, graphs[i], mode="pp", split=split or "search") for i in keep
          if lines or (i in e4 and e4[i].get("per_variable"))}
    e4x = {i: e4x_term(ge, tx, graphs[i], e4[i], split, le.get(i)) for i in e4 if e4[i].get("per_variable")}
    e3 = {}
    if keep and ee is not None and ev.get("edits"):
        e3 = dict(zip(keep, edit_term(ee, task, ev["edits"], [parsed[i]["edits"] for i in keep], edit_best or {})))
    e2g = {i: e2g_term(ge, tx, task, parsed[i]["answer"], ev["qa"], parsed[i]["finals"]) for i in keep} if ev["qa"] else {}
    tau = ge.tau(tx, pool, split or "search") if keep else {}
    ref = _teacher_ref(ge, tx, teacher, pool, split, task) if keep and teacher else None
    out = []
    for i, (p, (d_nodes, d_edges, d_bits)) in enumerate(zip(parsed, defects)):
        row, c, terms, info = row_of.get(i), c_of.get(i), {}, {}
        r = circ = V3_INVALID
        if row is not None:
            ph, _ = goal_eval.phi({**c, "n_point": c.get("n_point", row.get(f"n_{convention}"))})
            pt = tau.get("phi_tau")
            b = goal_eval.size_bits(graphs[i], P, total, targets, active=tx.vpd) + d_bits
            comp = goal_eval.compact(b, tau.get("bits_start"))
            g = None if ph is None or pt is None else 1.0 / (1.0 + 2.0 ** (-GATE_SLOPE * (ph - pt)))
            terms = {"E2g": e2g[i]["E2g"] if i in e2g else None, "E4x": e4x.get(i), "E1": row["E1"]}
            Q = 0.5 * (comp or 0.0) + sum(v for v in terms.values() if v is not None)
            m = None if ref is None or ph is None or comp is None else min(ph - ref["phi"], comp - ref["compact"])
            D = 0.0 if m is None else softplus(2.0 * m) / 2.0
            r = (ph or 0.0) + (g or 0.0) * Q + D
            circ = (ph or 0.0) + (g or 0.0) * 0.5 * (comp or 0.0) + D
            terms |= {"C": c["c"], "C_rl": c.get("c_rl"), "F1": row[f"f1_{convention}"], "N": row[f"n_{convention}"], "F3": row["f3"],
                      "E2": row["E2"], "E3": e3.get(i), "E4": e4[i]["E4"] if i in e4 else None}
            info = {"phi": ph, "phi_tau": pt, "g": g, "faithful": None if ph is None or pt is None else ph >= pt, "bits": b,
                    "bits_start": tau.get("bits_start"), "compact": comp, "D": D, "Q": Q, "teacher": ref,
                    "e2g": {k: v for k, v in (e2g.get(i) or {}).items() if k != "E2g"} or None}
            if lines:
                info["lines"] = line_credit(texts[i], p, p["answer"], le[i], info, e2g.get(i), e4.get(i), nat.C, targets, tx.vpd)
        out.append({"reward": r, "circuit_reward": circ, "valid": valid[i], **info, **{f"term_{k}": v for k, v in terms.items()},
                    "size": info.get("bits"), "malformed": p["malformed"], "dropped_nodes": d_nodes, "dropped_edges": d_edges,
                    "finals": len(p["finals"]), "truncated": bool(truncated and truncated[i]), "row": row, "variables": e4.get(i),
                    "qa": qa_detail(ev["qa"], p["finals"], tok, prompt["model_top"][0][0][0]) if valid[i] else [],
                    "fidelity": {k: v for k, v in (c or {}).items()  # C's members and columns (s_track, context_share, eps, ...)
                                 if k in ("c_terms", "rl_terms") or isinstance(v, (int, float)) or v is None}})
    return out


def _teacher_ref(ge, tx, teacher: dict, pool: dict, split: str | None, task: dict) -> dict | None:
    """The text's teacher circuit's phi and compact (teacher dominance's reference), kept on tx."""
    import goal_eval

    kept = tx.__dict__.setdefault("_teacher_ref", {})
    if split not in kept:
        nat = ge.nat
        g, _ = goal_eval.circuit(teacher, tx.targets, nat.C, tx.T)
        row = ge.fidelity(tx, [g], pool, mode="reward", split=split)[0]
        ph, _ = goal_eval.phi(row)
        b = goal_eval.size_bits(g, nat.positions(tx.targets), sum(nat.C.values()), tx.targets, active=tx.vpd)
        comp = goal_eval.compact(b, ge.tau(tx, pool, split or "search").get("bits_start"))
        kept[split] = None if ph is None or comp is None else {"phi": ph, "compact": comp, "bits": b}
    return kept[split]


_SHOWN = {}


def _token_of(model: str, text: str) -> int | None:
    """The token id a FINAL line's top token (shown()'s repr) names, or None."""
    if model not in _SHOWN:
        import mech

        V = mech.tokenizer(model).get_vocab_size() if hasattr(mech.tokenizer(model), "get_vocab_size") else len(mech.tokenizer(model))
        _SHOWN[model] = {}
        for i, x in enumerate(token_strings(model, list(range(V)))):
            _SHOWN[model].setdefault(shown(x), i)  # the first id of a string (a FINAL line names the string)
    return _SHOWN[model].get(text.strip())


def _moved_answer(answer: dict, where: list[int]) -> dict:
    """An answer's JSON with every node's and pair's position moved by where (old -> new)."""
    nodes = [[m, c, where[t]] for m, c, t in answer["nodes"]]
    edges = [["pair", e[1], e[2], e[3], where[e[4]]] if e[0] == "pair" else e for e in answer["edges"]]
    return {**answer, "nodes": nodes, "edges": edges}


def _without_part(answer: dict, i: int) -> dict:
    """An answer's JSON with node i removed (its edges and pairs with it)."""
    keep = [j for j in range(len(answer["nodes"])) if j != i]
    at = {j: k for k, j in enumerate(keep)}
    edges = []
    for e in answer["edges"]:
        if e[0] == "pair":
            if i not in (e[1], e[2]):
                edges.append(["pair", at.get(e[1], e[1]) if e[1] != "rest" else "rest", at.get(e[2], e[2]) if e[2] != "rest" else "rest", e[3], e[4]])
        elif i not in e:
            edges.append([at[e[0]], "out" if e[1] == "out" else at[e[1]]])
    return {**answer, "nodes": [answer["nodes"][j] for j in keep], "edges": edges}


def e2g_term(ge, tx, task: dict, answer: dict, questions: list[dict], finals: dict) -> dict:
    """E2g (the lead's 10-12): each question answered by running the answer's own circuit under its intervention
    (goal_eval's checker): a changed token (and a copy question's changes) the circuit's path-patched run on the
    changed text (pp_run); a moved, swapped or inserted token its run on that text (graph_ref, the circuit's positions
    moved with the text); a removed node its run without it; a cut connection its run with that connection cut; a node
    or connection outside the circuit no change. A question earns credit only where the FINAL line agrees with both
    the model and the circuit: falls (half) 1 when it matches both verdicts (a fall below half of the model's top
    token's probability), the top token (half) min(p_model(answer) / p_model(top), p_circuit(answer) / p_circuit(top))
    in the edited runs (partial credit for a near second). {"E2g": the mean, "consistency": the share of FINAL lines
    equal to the circuit's verdict (falls and top), "accuracy": the share right about the model, "per_question"}."""
    import torch

    import goal_eval
    import native

    nat = ge.nat
    model = task["model"]
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    C = nat.C
    top0 = prompt["model_top"][0][0][0]
    top0_id = _token_of(model, shown(top0))
    g0, _ = goal_eval.circuit(answer, targets, C, len(ids))
    g0 = goal_eval.strand(g0, targets, C)
    nodes = {tuple(x): j for j, x in enumerate(answer["nodes"])}
    keyed = lambda nd: ("block", nd[0], nd[1]) if nd[2] == BLOCK else (nd[0], nd[2], nd[1])  # noqa: E731
    with torch.no_grad():
        base_c = ge.graph_ref(ids, targets, g0)[0, -1].float().log_softmax(-1)
        V = base_c.shape[-1]
        top0_id = top0_id if top0_id is not None and top0_id < V else None  # (a tiny test model's vocabulary is smaller)
        kept = tx.__dict__.setdefault("_qa_model", {})
        per = []
        for n, q in enumerate(questions, 1):
            kind = q["kind"]
            if n not in kept:  # the model's own run under the intervention, once per text
                if kind in ("token", "copy"):
                    x = list(ids)
                    for r_, z in ([(q["position"], q["new"])] if kind == "token" else q["changes"]):
                        x[r_] = z
                    kept[n] = nat.reference([x], targets)[0, -1]
                elif kind in STRUCTURE:
                    x, tg = apply_change(ids, targets, q)
                    kept[n] = nat.reference([x], tg)[0, -1]
                elif kind == "remove":
                    kept[n] = nat.without(ids, targets, [tuple(q["node"])])[0, -1]
                else:
                    kept[n] = None  # a cut: the model's top only (goal_eval.cuts' detail)
            lm = kept[n]
            if kind in ("token", "copy"):
                x = list(ids)
                for r_, z in ([(q["position"], q["new"])] if kind == "token" else q["changes"]):
                    x[r_] = z
                lc = ge.pp_run(tx, x, g0)[0, -1]
                base = None  # path-patched on the text itself is the real model
            elif kind in STRUCTURE:
                x, tg = apply_change(ids, targets, q)
                gm, _ = goal_eval.circuit(_moved_answer(answer, change_map(q, len(ids))), tg, C, len(x))
                lc = ge.graph_ref(x, tg, goal_eval.strand(gm, tg, C))[0, -1]
                base = base_c
            elif kind == "remove":
                j = nodes.get(keyed(tuple(q["node"])))
                if j is None:
                    lc = base_c
                else:
                    gw, _ = goal_eval.circuit(_without_part(answer, j), targets, C, len(ids))
                    lc = ge.graph_ref(ids, targets, goal_eval.strand(gw, targets, C))[0, -1]
                base = base_c
            else:
                w, r_ = (tuple(x) if x is not None else None for x in q["edge"])
                stated = r_ is None and w in g0.out or r_ is not None and w in g0.parents.get(r_, [])
                lc = ge.graph_ref(ids, targets, g0, cuts={r_: [w]})[0, -1] if stated else base_c
                base = base_c
            lc = lc.float().log_softmax(-1)
            ref = base if base is not None else nat.reference([ids], targets)[0, -1].float().log_softmax(-1)
            falls_c = bool((lc[top0_id] - ref[top0_id]) / math.log(2) < -1.0) if top0_id is not None else None
            top_c = int(lc.argmax())
            got = finals.get(n)
            if got is None:
                per.append({"kind": QA_KINDS.get(kind, kind), "credit": 0.0, "consistent": False, "right": False})
                continue
            falls, said = got
            said_id = _token_of(model, said)
            said_id = said_id if said_id is not None and said_id < V else None
            fall_credit = float(falls == q["falls"] and falls == falls_c)
            if lm is not None:
                lmn = lm.float().log_softmax(-1)
                pm = 0.0 if said_id is None else float(torch.exp(lmn[said_id] - lmn.max()))
            else:
                pm = float(said_id == q["top"])
            pc = 0.0 if said_id is None else float(torch.exp(lc[said_id] - lc.max()))
            per.append({"kind": QA_KINDS.get(kind, kind), "credit": 0.5 * fall_credit + 0.5 * min(pm, pc),
                        "consistent": falls == falls_c and said_id == top_c, "right": falls == q["falls"] and said_id == q["top"]})
    k = max(len(per), 1)
    return {"E2g": sum(x["credit"] for x in per) / k, "consistency": sum(x["consistent"] for x in per) / k,
            "accuracy": sum(x["right"] for x in per) / k, "per_question": per}


def line_credit(text: str, parsed: dict, answer: dict, le: dict, info: dict, e2g: dict | None, e4: dict | None,
                C: dict, targets: list[int], active) -> list[list]:
    """Each answer line's credit, c = R(a) - R(a without the line) (the lead's v3 line credit, token-level advantages
    in RL): a node, connection or token-edge line its leave-one-out effect from line_effects (le, pp: the target KL's
    rise when it alone is removed, bits) less g x 0.5 x the compactness its removal would gain (log2 of the answer's
    bits over its bits without it: an inert line's credit is negative); a variable line (2 e4_v - 1) m_v / M (e4x's
    share of the mechanism); a question's ANSWER and FINAL lines its E2g credit; a CIRCUIT line that does not parse -1.
    [[line number in the answer text, credit], ...]; other lines carry none."""
    lines = text.split("\n")
    g, bits = info.get("g") or 0.0, info.get("bits")
    node_kl = {tuple(x["node"]): x["delta_kl"] or 0.0 for x in le.get("nodes", [])}
    edge_kl = {(tuple(w), "out" if r == "out" else tuple(r)): x["delta_kl"] or 0.0 for x in le.get("edges", []) for w, r in [x["edge"]]}
    tok_kl = {(t, "out" if r == "out" else tuple(r)): x["delta_kl"] or 0.0 for x in le.get("token_edges", []) for t, r in [x["edge"]]}
    ids = parsed.get("ids") or {}
    json_of = lambda nd: ["block", nd[0], nd[1]] if nd[2] == BLOCK else ([TOK, 0, nd[1]] if nd[0] == TOK else [nd[0], nd[2], nd[1]])  # noqa: E731
    place = {tuple(x): i for i, x in enumerate(answer["nodes"])}

    def gain(smaller: dict) -> float:  # the compactness the line's removal would gain: log2(bits / bits without it)
        b = _bits_of(smaller, C, targets, active)
        return math.log2(bits / b) if bits and b and b > 0 else 0.0
    M = sum(max(0.0, x) for x in node_kl.values())
    out, sec = [], None
    for k, line in enumerate(lines):
        head = SECTION.match(line)
        if head:
            sec = re.sub(r"\s+", " ", head[1].upper())
        s_ = line.strip()
        if sec == "CIRCUIT" and s_ and not head and not s_.startswith(("```", "flowchart", "subgraph", "%%", "direction")) and s_ != "end":
            m, e = MM_DECL.fullmatch(s_), MM_EDGE.fullmatch(s_)
            if m and m[1] in ids:
                nd = ids[m[1]]
                j = place.get(tuple(json_of(nd)))
                drop = gain(_without_part(answer, j)) if j is not None else 0.0
                out.append([k, (node_kl.get(nd, 0.0) if nd[0] != TOK else 0.0) - g * 0.5 * drop])
            elif e:
                ws = [x.strip() for x in e[1].split("&")]
                rs = ["out"] if MM_OUT.fullmatch(e[2]) else [x.strip() for x in e[2].split("&")]
                c = 0.0
                for w in ws:
                    for r in rs:
                        if w not in ids or (r != "out" and r not in ids):
                            c -= 1.0  # an undeclared id
                            continue
                        wn, rn = ids[w], ("out" if r == "out" else ids[r])
                        kl = tok_kl.get((wn[1], rn)) if wn[0] == TOK else edge_kl.get((wn, rn))
                        wi, ri = place.get(tuple(json_of(wn))), ("out" if rn == "out" else place.get(tuple(json_of(rn))))
                        smaller = {**answer, "edges": [x for x in answer["edges"] if not (x[0] != "pair" and x[0] == wi and x[1] == ri)]}
                        c += (kl or 0.0) - g * 0.5 * gain(smaller)
                out.append([k, c])
            elif not MM_PAIR.fullmatch(s_):
                out.append([k, -1.0])  # a line that does not parse
        elif head and sec.split()[0] in ("ANSWER", "FINAL") and e2g:
            n = int(sec.split()[1])
            if 1 <= n <= len(e2g["per_question"]):
                out.append([k, e2g["per_question"][n - 1]["credit"]])
        elif sec == "EXPLANATION" and VAR_RE.fullmatch(s_) and e4 and e4.get("per_variable") and M > 0:
            v = int(VAR_RE.fullmatch(s_)[1])
            if v <= len(e4["per_variable"]):
                pv = e4["per_variable"][v - 1]
                out.append([k, (2 * pv["e4"] - 1) * sum(max(0.0, node_kl.get(tuple(nd), 0.0)) for nd in pv["nodes"]) / M])
    return out


def _bits_of(answer: dict, C: dict, targets: list[int], active) -> float | None:
    """An answer JSON's raw description bits (goal_eval.size_bits, blocks at their active contents)."""
    import goal_eval

    g, _ = goal_eval.circuit(answer, targets, C, max(targets) + 1)
    return goal_eval.size_bits(g, max(targets) + 1, sum(C.values()), targets, active=active)


def e4x_term(ge, tx, graph, e4: dict, split: str | None, le: dict | None = None) -> float | None:
    """E4x (the lead's v3): sum over variables of (2 e4_v - 1) m_v / sum over the circuit's nodes of m, m a node's
    measured pp removal effect (line_effects' delta_kl, its positive part), m_v the total over the variable's nodes,
    e4_v the variable's headline (min(f on-target, s off-target), 0 unresolved): the share of the mechanism explained
    by variables that pass interchange less the share claimed by ones that fail."""
    le = le or ge.line_effects(tx, graph, mode="pp", split=split or "search")
    m = {tuple(x["node"]): max(0.0, x["delta_kl"] or 0.0) for x in le["nodes"]}
    M = sum(m.values())
    if M <= 0:
        return None
    return sum((2.0 * v["e4"] - 1.0) * sum(m.get(tuple(nd), 0.0) for nd in v["nodes"]) for v in e4["per_variable"]) / M


def c_noise(ge, task: dict, texts: list[str], seeds: int, pool: dict | None = None, split: str = "eval") -> list[dict]:
    """C's draw noise (a check on goal_eval's eps, the lead 10-12): each answer's reward-mode c and c_rl on the
    text under `seeds` seeds (the text's own and the next ones: fresh changed prompts, interchange, donor and flip
    draws), and their standard deviations across them."""
    import statistics

    import goal_eval
    import native

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    graphs = [goal_eval.circuit(parse(x)["answer"], targets, ge.nat.C, len(ids))[0] for x in texts]
    base = native.task_seed(str(ids[:8]))  # GoalEval.text's own seed
    runs = [ge.fidelity(ge.text(ids, targets, seed=base + 7919 * r, task=task), graphs, _pool() if pool is None else pool,
                        mode="reward", split=split) for r in range(seeds)]
    out = []
    for i in range(len(texts)):
        c, crl = [rs[i]["c"] for rs in runs], [rs[i]["c_rl"] for rs in runs]
        out.append({"c": c, "c_rl": crl, "sd_c": statistics.stdev(c) if seeds > 1 else None,
                    "sd_c_rl": statistics.stdev(crl) if seeds > 1 else None})
    return out


def edit_teach(paths) -> dict:
    """The edit teacher's rows (edit_eval.py teach output files), by goal key (edit_eval.EditEval.key)."""
    import edit_eval

    out = {}
    for p in paths:
        for r in map(json.loads, Path(p).read_text().splitlines()):
            out[edit_eval.EditEval.key(r["task"], r["goal"])] = r
    return out


def teacher_edits(edits: dict | None, r: dict) -> dict:
    """A row's edit requests answered by the edit teacher (edits: edit_teach()): request number -> parse()'s
    operations, for the requests the teacher has an edit for."""
    import edit_eval

    ops = {}
    for k, goal in enumerate(r.get("edits") or [], 1):
        t = (edits or {}).get(edit_eval.EditEval.key(r["task"], goal), {}).get("teacher") or {}
        if t.get("edit"):
            ops[k] = [edit_eval._from_json(o) for o in t["edit"]]
    return ops


def edit_term(ee, task: dict, goals: list[dict], answers: list[dict], best: dict) -> list:
    """E3 of each answer's edits (parse()'s "edits"), one value per answer: per edit request, the edit's net bits
    (edit_eval's goal_bits less collateral) over the best net known for that goal (best: goal key -> net, the edit
    teacher's and the baselines', edit_eval.bests), clipped to [-1, 1], 0 for an edit with an invalid operation, none,
    or one edit_eval rejects; the mean over the requests with a positive best net (a goal no known edit gains on is
    left out). None for an answer when no request has one (E3 is then left out of the reward) or edit_eval gives no
    net."""
    import edit_eval

    vals = [[] for _ in answers]
    for n, goal in enumerate(goals, 1):
        key = edit_eval.EditEval.key(task["id"], goal)
        nb = best.get(key)
        if not nb or nb <= 0:
            continue  # no known edit gains on this goal: net / best net is undefined there
        ops = [a.get(n) or [] for a in answers]
        run = [i for i, o in enumerate(ops) if o and all(x["op"] != "invalid" for x in o)]
        rows = dict(zip(run, ee.score_many(task, goal, [ops[i] for i in run]))) if run else {}
        for i in range(len(answers)):
            r = rows.get(i)
            if r is not None and r.get("valid") and r.get("net") is None:
                vals[i].append(None)
            elif r is None or not r.get("valid"):
                vals[i].append(0.0)
            else:
                vals[i].append(max(-1.0, min(1.0, r["net"] / nb)))
    return [None if not v or any(x is None for x in v) else sum(v) / len(v) for v in vals]


# ---- supervised targets


def target(answer: dict, ev: dict, model: str, edits: dict | None = None, strings: list[str] | None = None) -> str:
    """The supervised answer to a question: the teacher's circuit (goal_eval's JSON), variables and explanation, for
    each question the model's measured answer (no reasoning: the teacher has none to give), and for each edit request
    the edit teacher's operations (edits: request number -> parse()'s ops); strings: the text's tokens (the variables'
    inputs)."""
    finals = {n: (q["falls"], shown(token_strings(model, [q["top"]])[0])) for n, q in enumerate(ev["qa"], 1)}
    return render(answer, finals, edits=edits, strings=strings)


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


def target_score(r: dict) -> float:
    """A teacher candidate's score: reward v3's circuit alone (circuit_reward: phi, gate and compactness), the English
    and answers left out."""
    return r["circuit_reward"]


def vpd_teacher(ge, task: dict, ev: dict, tokens, budget: int = ANSWER_BUDGET, cache: dict | None = None) -> tuple[dict, dict]:
    """The interim teacher: VPD's own answer cut to a size an answer can be written at. Prefixes of VPD's order
    (Native._vpd_order) of 16, 32, 64, ... nodes up to the largest whose answer text (with the measured FINAL lines)
    fits `budget` tokens of the explainer's tokenizer (tokens: text -> count), each scored by target_score as nodes
    alone; the best gets measured edges (with_edges), cut back to the budget if they overrun it. (answer JSON, {"k", "k_max", "circuit_reward", terms})."""
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
    i = max(range(len(ks)), key=lambda j: target_score(got[j]))
    answer = with_edges(ge, task, ev, cands[i], cache)
    while answer["edges"] and tokens(target(answer, ev, task["model"])) > budget:
        answer = {**answer, "edges": answer["edges"][:len(answer["edges"]) // 2]}
    final = reward(ge, task, {**ev, "qa": []}, [render(answer)], cache=cache)[0]
    return answer, {"task": task["id"], "k": ks[i], "k_max": lo, "edges": len(answer["edges"]), "score": target_score(final),
                    "circuit_reward": final["circuit_reward"], **{k: final.get(k) for k in ("term_F1", "term_N", "term_F3", "size")},
                    "tokens": tokens(target(answer, ev, task["model"]))}


def teacher_facts(ge, task: dict, ev: dict, answer: dict, max_nodes: int = FACT_NODES, variables: list | None = None) -> dict:
    """What a teacher circuit's English may draw on: only what the oracle has or writes at inference, so supervised
    English never states what the oracle cannot know. From the question (evidence()): the text and its target, the
    model's top next tokens, the measured evidence at the listed positions (units.evidence; never the English reader's
    events), the 1% rule, and for the circuit's subcomponents the question lists, their atlas facts as it lists them
    (the routed lens at their positions and Pile contexts). From the teacher's answer: its subcomponents (the first max_nodes in VPD's order),
    its stated edges (those into the prediction first, without their measured effects) and the questions' answers; the
    teacher's variables when given (variables.py's JSON from the teacher's grouping, meanings unwritten: the writer
    writes them, with the tokens at their inputs); and the circuit as the oracle writes it ("circuit", its lines, and
    "answer", its JSON: for E1's reader, never shown to the writer)."""
    import goal_eval
    import units

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    strings = token_strings(task["model"], ids)
    g, _ = goal_eval.circuit(answer, targets, ge.nat.C, len(ids))
    rank = {nd: i for i, nd in enumerate(ge.nat._vpd_order(ids, targets))}
    nodes = sorted(g.node_set(), key=lambda nd: rank.get(nd, len(rank)))
    shown_facts = {tuple(nd): f for cs in ev["components"].values() for nd, _, f in cs}  # what the question tells about each
    pairs = [(w, None) for w in g.out] + [(w, r) for r, ws in g.parents.items() for w in ws]
    tok = lambda xs: token_strings(task["model"], xs)  # noqa: E731
    return {"task": task["id"], "text": "".join(strings[:max(targets) + 1]), "target": max(targets),
            "circuit": circuit_text(render(goal_eval.to_json(g, ""))),  # what E1's reader reads with the English
            "answer": goal_eval.to_json(g, ""),
            **({"variables": variables, "tokens": {str(q): strings[q] for v in variables for q in v["inputs"]}} if variables else {}),
            "evidence": units.evidence(ev["events"], ev["listed"], strings) if ev.get("events") else [],
            "target_token": strings[max(targets)], "top": prompt["model_top"][0][:3], "nodes_total": len(nodes),
            "positions": sorted({nd[1] for nd in nodes}),
            "nodes": [{"node": node_text(nd), "token": strings[nd[1]], "facts": shown_facts.get(nd, "")} for nd in nodes[:max_nodes]],
            "edges": [f"{node_text(w)} -> {'OUT' if r is None else node_text(r)}" for w, r in pairs],
            "qa": [{"question": qa_text(q, n, strings, repr(prompt["model_top"][0][0][0]), tok),
                    "answer": f"falls={'yes' if q['falls'] else 'no'} top={shown(tok([q['top']])[0])}"} for n, q in enumerate(ev["qa"], 1)]}


WRITER = ("A small language model (4 layers) read the text below and predicts the next token after its last token. A "
          "decomposition of its weights into rank-one subcomponents (written L<layer>.<kind>#<index>@<position>) found the "
          "circuit summarized here; the facts below come from measurements on the model. Write the EXPLANATION: in plain English, "
          "in three to six sentences, say how the model arrives at its prediction from this text, which parts of the text "
          "matter and what the subcomponents do, as the facts show. Say only what the facts support; never write a "
          "subcomponent's name (L<layer>.<kind>#<index>): describe what it does instead; do not mention these instructions. "
          "Write the explanation only.\n\n{facts}")


WRITER_VARIABLES = ("Begin with one line per variable listed above, in that order: V<k> \"<short name>\": <what it stands for "
                    "in the text, in plain English>. Then write the explanation of the prediction in terms of those variables.")
WRITER_VAR_RE = re.compile(r'\s*V(\d+)\s+"(.*?)"\s*:\s*(.+?)\s*')


def writer_prompt(f: dict) -> str:
    """The prompt that asks an open model for a teacher circuit's English, from teacher_facts(); with the teacher's
    variables, their meanings first (WRITER_VARIABLES, written_variables())."""
    lines = [f"Text: {f['text']!r}", f"Prediction at position {f['target']} (after {f['target_token']!r}): "
             + ", ".join(f"{t!r} {p:.2f}" for t, p in f["top"]),
             f"Circuit: {f['nodes_total']} subcomponents at positions {', '.join(map(str, f['positions']))}; the most important "
             "(where known: the tokens its write raises at this position by the routed lens, a label-free linear reading of "
             "its write, not a measured effect; and the tokens it is active on in Pile text):"]
    for n in f["nodes"]:
        lines.append(f"- {n['node']} (token {n['token']!r})" + (f": {n['facts']}" if n.get("facts") else ""))
    if f.get("evidence"):
        lines.append("Measured at the positions where a changed token moves the prediction most (log2 change of the top "
                     "token's probability when the token there changes, and when each layer's attention at the target is kept "
                     "from reading it):")
        lines += [f"- {e}" for e in f["evidence"]]
        lines.append("Every other position's changed token moves the prediction by less than 1% of the total over all positions.")
    if f["edges"]:
        lines.append("Stated connections:")
        lines += [f"- {e}" for e in f["edges"]]
    if f["qa"]:
        lines.append("Measured interventions:")
        lines += [f"- {q['question'][q['question'].find(':') + 2:]} Measured: {q['answer']}" for q in f["qa"]]
    if f.get("variables"):
        import goal_eval

        lines.append("Variables of this circuit (each a group of its subcomponents that carries what is computed from its input "
                     "tokens: verified by changing an input token and setting the group to its values on the changed text):")
        for k, v in enumerate(f["variables"], 1):
            nodes = ", ".join((f"{node_text((goal_eval._site(it[1]), int(it[2]), 0)).split('#')[0]} @{it[2]} (its whole output)"
                               if it[0] == "block" else node_text((goal_eval._site(it[0]), it[2], it[1]))) for it in v["group"])
            lines.append(f"- V{k} (provisional name {v['name']!r}): inputs " + ", ".join(f"{q} ({f['tokens'][str(q)]!r})" for q in v["inputs"])
                         + f"; carried by {nodes}; feeds " + (", ".join(v["feeds"]) or "prediction")
                         + (f"; measured: {json.dumps(v['measured'])}" if v.get("measured") else ""))
        return escape(WRITER.format(facts="\n".join(lines)) + "\n\n" + WRITER_VARIABLES)
    return escape(WRITER.format(facts="\n".join(lines)))


def written_variables(text: str, variables: list[dict]) -> tuple[list[dict], str] | None:
    """The writer's variable lines (WRITER_VARIABLES: V<k> "<short name>": <meaning>) put on the teacher's variables
    (names it gives replace the provisional ones, in "feeds" too, unless two collide), and the prose after them on one
    line; None when a variable has no line."""
    got, prose = {}, []
    for line in text.splitlines():
        m = WRITER_VAR_RE.fullmatch(line)
        if m and 1 <= int(m[1]) <= len(variables):
            got.setdefault(int(m[1]), (m[2].strip(), m[3].strip()))
        elif line.strip().rstrip(":").strip().upper() not in ("VARIABLES", "EXPLANATION"):
            prose.append(line)
    if len(got) < len(variables):
        return None
    names = [got[k][0] or v["name"] for k, v in enumerate(variables, 1)]
    if len(set(names)) < len(names):
        names = [v["name"] for v in variables]
    rename = {v["name"]: n for v, n in zip(variables, names)}
    out = [{"name": n, "meaning": got[k][1], "inputs": v["inputs"], "group": v["group"], "feeds": [rename.get(x, x) for x in v["feeds"]]}
           for k, (v, n) in enumerate(zip(variables, names), 1)]
    return out, " ".join(" ".join(prose).split())


def dataset(rows: Path, teacher_dir: Path, out: Path, ge=None, english: dict | None = None, edits: dict | None = None,
            variables: dict | None = None) -> int:
    """SFT rows ({"prompt": [user message], "completion": [assistant message], "info": {"task"}}, units_sft.py's
    format) for the questions that have a teacher answer: TEACHER_DIR/<task>.json (goal_eval's circuit JSON; nodes
    without edges get with_edges(), which needs ge) or <task>.txt (this module's text). english: task -> the chosen
    explanation (tinker_train.py explain), written into the answer's EXPLANATION. edits: the edit teacher's rows
    (edit_teach()); a question with edit requests is kept only when the teacher has an edit for every one. Every target
    is the circuit as it is scored: goal_eval.circuit drops nodes and edges the model does not have and the stranded
    nodes (goal_eval.strand: a node other than a query or key with no stated connection), so nothing in a target is a
    part the reward drops; info counts them (stranded, dropped_nodes, dropped_edges). variables: task -> the teacher's
    variables (variables.py's JSON, meanings written), else the teacher answer's own "variables"; a row that asks for
    them (rerender --variables) is kept only with them. ge: goal_eval.GoalEval."""
    import goal_eval
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
            prompt = task["prompts"][0]
            g, d = goal_eval.circuit(answer, prompt["target_positions"], ge.nat.C, len(prompt["token_ids"]))
            found = (variables or {}).get(r["task"], answer.get("variables"))
            answer = goal_eval.to_json(g, answer.get("explanation", "")) | ({"variables": found} if found else {})
            if r.get("variables") and not found:
                continue  # the question asks for VARIABLES (rerender --variables): no target without them
            info = {"task": r["task"], "stranded": d["stranded_nodes"], "dropped_nodes": d["invalid_nodes"], "dropped_edges": d["invalid_edges"],
                    "variables": len(found or [])}
            if english and r["task"] in english:
                answer = {**answer, "explanation": english[r["task"]]}
            ops = teacher_edits(edits, r)
            if len(ops) < len(r.get("edits") or []):
                continue  # an answer without an EDIT line for a request is invalid: no target for it
            f.write(json.dumps({"prompt": [{"role": "user", "content": r["question"]}],
                                "completion": [{"role": "assistant", "content": target(answer, r, task["model"], ops,
                                                                                        token_strings(task["model"], prompt["token_ids"]))}],
                                "info": info}) + "\n")
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


def baseline_texts(ge, task: dict, sizes: list[int], tx=None) -> dict:
    """name -> answer text of the search-free baselines, node sets claiming every connection and token edge among their
    nodes: "empty", "vpd@k", VPD's first k active subcomponents in its order (Native._vpd_order) for each k, and with tx
    "start", the influential start set (goal_eval.start_set on the eval split: reward v3's reference for compactness)."""
    import goal_eval
    import native

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    out = {"empty": render({"nodes": [], "edges": []})}
    order = ge.nat._vpd_order(ids, targets)
    for k in sorted(set(sizes)):
        if k > 0:
            out[f"vpd@{k}"] = render(goal_eval.to_json(native.Graph(set(order[:k]), complete=True)))
    if tx is not None:
        out["start"] = render(goal_eval.to_json(ge.start_set(tx, "eval")["graph"]))
    return out


def references(ge, rows: Path, teacher_dir: Path, out: Path, edits: dict | None = None) -> int:
    """VPD's references on each text with a teacher, at the teacher's subcomponent count k (its nodes other than token
    embeddings), as answer texts with the measured answers to the text's questions (target()):
    out/vpd_complete/<task>.txt, VPD's first k active subcomponents with every connection the model has among them, and
    out/vpd_pruned/<task>.txt, the same nodes with the connections teacher.py's edge search kept
    (TEACHER_DIR/<task>.vpd_top<k>.edges.json, where it exists): units compared under equal edge effort. edits: the
    edit teacher's rows (edit_teach()), whose operations answer the rows' edit requests (a reference states a circuit,
    not an edit: without them it would be invalid on rows with edit requests)."""
    import goal_eval
    import native

    n = 0
    for r in map(json.loads, rows.read_text().splitlines()):
        f = teacher_dir / f"{r['task']}.json"
        if not f.exists():
            continue
        task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
        prompt = task["prompts"][0]
        k = sum(m != TOK for m, _, _ in json.loads(f.read_text())["nodes"])
        nodes = set(ge.nat._vpd_order(prompt["token_ids"], prompt["target_positions"])[:k])
        for name, answer in (("vpd_complete", goal_eval.to_json(native.Graph(nodes, complete=True))),
                             ("vpd_pruned", json.loads(p.read_text()) if (p := teacher_dir / f"{r['task']}.vpd_top{k}.edges.json").exists() else None)):
            if answer is not None:
                (out / name).mkdir(parents=True, exist_ok=True)
                (out / name / f"{r['task']}.txt").write_text(target(answer, r, task["model"], teacher_edits(edits, r)))
        n += 1
    return n


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
             diagnostics: bool = False, log=print, ee=None, edit_best: dict | None = None, vs=None, matched_all: bool = False,
             teachers: dict | None = None, lines: bool = False) -> dict:
    """An evaluation of answer sets (name -> samples_of()) on the split's first n texts with question rows, every
    answer scored by reward() (an invalid one at its fixed reward, never skipped). Per set: the headline over each
    question's FIRST sample (policy performance: the mean reward, each term's mean with invalid answers counted as 0,
    the valid share, size, dropped nodes and edges), "best of N (upper bound)" (each question's best sample), and
    paired comparisons with every other set (reward) and with the baselines (empty, and VPD's own answer at the first
    sample's node count; circuit_reward, as they give no English or answers) on the same questions. A set with no answer on a question scores it as invalid. e1: set -> task ->
    {"own": [bits per sample], "shuffled": bits, "reader_b": bits} (tinker_train.py readers): the reward's E1 uses
    "own"; the controls are reported (no English is 0 bits by construction). labels: task -> locality() row: every
    figure is also reported for local and non-local texts apart. diagnostics: structure() of each first sample (a
    second disjoint circuit, minimality), costly. matched_all: VPD's answer at every sample's node count, not only the
    first's, so each sample is compared with VPD at its own size (verify-and-pick: whether any of N beats it on C).
    The baselines' v3 terms (phi, g, compact, bits, D) and C and N are kept beside their circuit reward
    ("baseline_terms"). teachers: task -> the text's teacher circuit (JSON), reward v3's teacher-dominance reference, for
    the sets and the baselines alike. lines: each sample's per-line credits too (reward(lines=True))."""
    import native

    invalid = V3_INVALID
    per = {s: [] for s in sets}
    base, base_terms = {}, {}
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
            scored = reward(ge, task, ev, [x for x, _ in got], truncated=[cut for _, cut in got], e1=bits, cache=cache,
                            ee=ee, edit_best=edit_best, vs=vs, split="eval", teacher=teachers.get(p.stem) if teachers else None,
                            lines=lines)
            entry = {"task": p.stem, "samples": [{k: v for k, v in r.items() if k != "row"} | {"nodes": (r["row"] or {}).get("nodes")}
                                                 for r in scored]}
            if diagnostics and scored[0]["valid"]:
                import goal_eval

                prompt = task["prompts"][0]
                g, _ = goal_eval.circuit(parse(got[0][0])["answer"], prompt["target_positions"], ge.nat.C, len(prompt["token_ids"]))
                entry["structure"] = structure(ge, cache[task["id"]], g, ge.nat._vpd_order(prompt["token_ids"], prompt["target_positions"]))
            per[name].append(entry)
        sizes = [y.get("nodes") or 0 for x in (per[s][-1] for s in sets) for y in (x["samples"] if matched_all else x["samples"][:1])]
        names = baseline_texts(ge, task, sizes, cache.get(task["id"]))
        for (b, text), r in zip(names.items(), reward(ge, task, {**ev, "qa": []}, list(names.values()), split="eval",
                                                      teacher=teachers.get(p.stem) if teachers else None)):
            base.setdefault(p.stem, {})[b] = r["circuit_reward"]
            base_terms.setdefault(p.stem, {})[b] = {"C": r.get("term_C"), "N": r.get("term_N"),
                                                    **{k: r.get(k) for k in ("phi", "phi_tau", "g", "compact", "bits", "D")}}
        log(p.stem + ": " + ", ".join(f"{s} {per[s][-1]['samples'][0]['reward']:+.3f}" for s in sets), flush=True)
    report = {}
    groups = {"all": None}
    if labels:
        groups |= {"local": True, "non-local": False}
    for group, want in groups.items():
        try:  # the scored answers come back even if a figure fails (the scores are the hour's work)
            report[group] = _report(per, base, e1, labels, want, base_terms)
        except Exception as e:  # noqa: BLE001
            import traceback

            log(traceback.format_exc(), flush=True)
            report[group] = {"error": repr(e)}
    return {"report": report, "per_question": per, "baselines": base, "baseline_terms": base_terms}


def _report(per: dict, base: dict, e1: dict | None, labels: dict | None, want, base_terms: dict | None = None) -> dict:
    """evaluate()'s figures over the questions of one locality group (want None: every question)."""
    keep = (lambda t: True) if want is None else (lambda t: t in labels and labels[t]["local"] == want)
    per = {name: [q for q in qs if keep(q["task"])] for name, qs in per.items()}
    report = {}
    for name, qs in per.items():
        if not qs:
            continue
        first = [q["samples"][0] for q in qs]
        terms = sorted({k for f in first for k in f if k.startswith("term_")})
        ok = [f for f in first if f["valid"]]
        report[name] = {
            "questions": len(qs), "valid": sum(f["valid"] for f in first) / max(len(first), 1),
            "reward": sum(f["reward"] for f in first) / max(len(first), 1),
            **{f"v3 {k}": (sum(float(f[k]) for f in ok if f.get(k) is not None) / max(sum(f.get(k) is not None for f in ok), 1)
                           if any(f.get(k) is not None for f in ok) else None)
               for k in ("phi", "phi_tau", "g", "faithful", "bits", "compact", "D", "Q")},
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
        by = {}
        for f in first:
            for d in f.get("qa") or []:
                by.setdefault(d["kind"], []).append(d["right"] - d["right0"])
        if by:
            report[name]["E2 gain over the no-change answer by type"] = {k: sum(v) / len(v) for k, v in sorted(by.items())}
        if f"{name}_nocircuit" in per:  # the same policy answering without its circuit (sample --prefix): E2 with - without
            other = {o["task"]: o["samples"][0] for o in per[f"{name}_nocircuit"]}
            pairs = [(q["samples"][0].get("term_E2") or 0.0, other[q["task"]].get("term_E2") or 0.0) for q in qs if q["task"] in other]
            if pairs:
                report[name]["E2 circuit-ablation delta (with - without circuit)"] = _paired([a for a, _ in pairs], [b for _, b in pairs])
        vpd = lambda x, q, k: ((base_terms or {}).get(q["task"], {}).get(f"vpd@{x.get('nodes')}") or {}).get(k)  # noqa: E731
        vs_vpd = lambda x, q: vpd(x, q, "C") is not None and (x.get("term_C") or 0.0) > vpd(x, q, "C")  # noqa: E731
        report[name]["phi beats VPD@matched: first sample"] = sum(
            q["samples"][0]["valid"] and vpd(q["samples"][0], q, "phi") is not None and q["samples"][0].get("phi") is not None
            and q["samples"][0]["phi"] > vpd(q["samples"][0], q, "phi") for q in qs) / max(len(qs), 1)
        report[name]["C, best of N"] = sum(max((x.get("term_C") or 0.0) if x["valid"] else 0.0 for x in q["samples"]) for q in qs) / max(len(qs), 1)
        report[name]["C beats VPD@matched: first sample"] = sum(q["samples"][0]["valid"] and vs_vpd(q["samples"][0], q) for q in qs) / max(len(qs), 1)
        report[name]["C beats VPD@matched: any sample"] = sum(any(x["valid"] and vs_vpd(x, q) for x in q["samples"]) for q in qs) / max(len(qs), 1)
        if e1 and name in e1:
            ctl = [e1[name][q["task"]] for q in qs if q["task"] in e1[name]]
            report[name]["E1 controls"] = {k: sum(c[k] for c in ctl if c.get(k) is not None) / max(sum(c.get(k) is not None for c in ctl), 1)
                                           for k in ("shuffled", "reader_b", "control")} | {"own (first sample)": sum(c["own"][0] for c in ctl if c["own"][0] is not None)
                                                                                 / max(sum(c["own"][0] is not None for c in ctl), 1),
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
    rr = sub.add_parser("rerender", help="the rows with their question text written again by this question() (no model runs)")
    rr.add_argument("rows", type=Path)
    rr.add_argument("out", type=Path)
    rr.add_argument("--facts", action="store_true", help="read each listed subcomponent's facts again (refresh_facts: the routed lens "
                                                          "at its position, one forward pass per text)")
    rr.add_argument("--variables", action="store_true", help="ask for the EXPLANATION's VARIABLES block (ASK_VARIABLES)")
    rr.add_argument("--measured", action="store_true", help="the measured input arm (measured(): subcomponents ranked by importance x "
                                                             "|activation| with activations, attention heads' top positions)")
    rr.add_argument("--pool", action="store_true", help="the full candidate pool (candidate_pool(): every VPD-active subcomponent at the "
                                                         "listed, target and most-attended positions), cut to leave ROOM (fit_pool())")
    rr.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="the policy's tokenizer, which fit_pool() counts in")
    rf = sub.add_parser("references", help="VPD at each teacher's size, with every connection and with pruned ones (references())")
    rf.add_argument("rows", type=Path)
    rf.add_argument("teacher", type=Path, help="teacher.py's output directory (<task>.json, <task>.vpd_top<k>.edges.json)")
    rf.add_argument("out", type=Path)
    rf.add_argument("--edit-teach", type=Path, action="append", default=[], help="edit_eval.py teach output: answers to edit requests")
    ed = sub.add_parser("edits", help="the rows with edit requests (edit_eval.goals: flip, raise, detach) and their question again")
    ed.add_argument("rows", type=Path)
    ed.add_argument("out", type=Path)
    ed.add_argument("--qa-seed", type=int, default=0, help="added to each text's seed (0 the discovery draw, 1 a fresh one)")
    ed.add_argument("--goals", type=Path, action="append", default=[],
                    help="JSONL rows with task and goal (edit_eval.py goals or teach output): these goals, not drawn again. A "
                         "detach goal's position is drawn from goal_eval's F2 positions, which move with goal_eval's version "
                         "and device, so a drawn goal can miss the edit teacher's")
    st = sub.add_parser("structure", help="the rows with structural questions (structure_changes(), about 1 in 4 of the questions) and reader events")
    st.add_argument("rows", type=Path)
    st.add_argument("out", type=Path)
    st.add_argument("--qa-seed", type=int, default=0, help="added to each text's seed (the oracle's draw; the reader's is one further)")
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
    lc.add_argument("--tasks", type=Path, help="task ids, one per line, in place of the split's first n texts")
    lc.add_argument("--lengths", default="0:100000", help="LO:HI: only texts of LO <= tokens < HI (PyTorch's MPS keeps "
                    "per-shape caches, ~20-40 MB of host memory per sequence length: a process over a band of lengths stays bounded)")
    pr = sub.add_parser("prepare", help="per text: locality, and for the texts kept, the question row and the interim teacher")
    pr.add_argument("split", choices=("train", "heldout", "hard"))
    pr.add_argument("out", type=Path, help="directory: locality.jsonl, rows.jsonl, teacher/<task>.json, teacher/summary.jsonl")
    pr.add_argument("--n", type=int, default=400)
    pr.add_argument("--lengths", default="0:100000")
    pr.add_argument("--tasks", type=Path, help="JSONL of {task[, local]}: only these texts, in this order; a given label skips locality()")
    pr.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    pr.add_argument("--rows-only", action="store_true", help="no interim teacher (teacher.py's circuits are the teachers)")
    vt = sub.add_parser("teacher-vpd", help="the interim teacher: VPD's own answer cut to a writable size (vpd_teacher())")
    vt.add_argument("rows", type=Path)
    vt.add_argument("out", type=Path, help="directory: <task>.json (circuit JSON) and summary.jsonl")
    vt.add_argument("--tasks", help="comma-separated task ids (default: every row)")
    vt.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    fa = sub.add_parser("facts", help="what each teacher circuit's English must be faithful to (teacher_facts())")
    fa.add_argument("rows", type=Path)
    fa.add_argument("teacher", type=Path)
    fa.add_argument("out", type=Path)
    fa.add_argument("--n", type=int, default=10)
    fa.add_argument("--device", help="cpu keeps it off a busy GPU (a few cuts per text)")
    fa.add_argument("--variables", type=Path, help="the teacher's variables: a directory of <task>.json, {\"variables\": [...]}")
    d = sub.add_parser("dataset")
    d.add_argument("rows", type=Path)
    d.add_argument("teacher", type=Path)
    d.add_argument("out", type=Path)
    d.add_argument("--english", type=Path, help="tinker_train.py explain output: each text's chosen explanation")
    d.add_argument("--edit-teach", type=Path, action="append", default=[], help="edit_eval.py teach output (the edit teacher's edits)")
    d.add_argument("--variables", type=Path, help="a directory of <task>.json, {\"variables\": [...]} (variables.py's JSON, meanings written)")
    nz = sub.add_parser("noise", help="C's draw noise per answer (c_noise()): a check on goal_eval's eps")
    nz.add_argument("split", choices=("train", "heldout", "hard"))
    nz.add_argument("rows", type=Path)
    nz.add_argument("--set", action="append", default=[], help="NAME=SAMPLES: each text's first answer")
    nz.add_argument("--seeds", type=int, default=4)
    nz.add_argument("--n", type=int, default=20)
    nz.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("evaluate")
    s.add_argument("split", choices=("train", "heldout", "hard"))
    s.add_argument("rows", type=Path, help="questions' JSONL (a fresh QA seed for a report, the discovery one for selection)")
    s.add_argument("--set", action="append", default=[], help="NAME=SAMPLES (a sampler's JSONL or a directory of <task>[.<k>].txt)")
    s.add_argument("--e1", action="append", default=[], help="NAME=E1.jsonl (tinker_train.py readers)")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--labels", type=Path, help="locality JSONL: every figure also for local and non-local texts")
    s.add_argument("--diagnostics", action="store_true", help="structure() of each first sample (slow)")
    s.add_argument("--out", type=Path)
    s.add_argument("--edit-teach", type=Path, action="append", default=[], help="edit_eval.py teach output (scored with net): the best net per goal (E3)")
    s.add_argument("--variables", action="store_true", help="E4: each answer's VARIABLES block by interchange (variables.Scorer)")
    s.add_argument("--matched-all", action="store_true", help="VPD at every sample's node count (verify-and-pick against VPD@matched)")
    s.add_argument("--teacher", type=Path, help="teacher circuits (<task>.json): reward v3's teacher-dominance reference")
    s.add_argument("--lines", action="store_true", help="each sample's per-line credits (reward(lines=True), line_credit())")
    args = ap.parse_args()
    import goal_eval
    import native

    if args.cmd == "rerender":
        if args.facts or args.measured or args.pool:
            mnat = native.Native()
            atlas, lens = _atlas(), _lens(mnat)
        else:
            atlas = lens = None
        if args.pool:
            from transformers import AutoTokenizer

            tk = AutoTokenizer.from_pretrained(args.tokenizer)
            count = lambda text: len(tk(text, add_special_tokens=False)["input_ids"])  # noqa: E731
        with args.out.open("w") as f:
            for r in map(json.loads, args.rows.read_text().splitlines()):
                task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
                if args.measured:
                    r = measured(mnat, task, r, atlas, lens)
                elif args.facts:
                    r = refresh_facts(atlas, lens, task, r)
                if args.pool:
                    r = fit_pool(task, candidate_pool(mnat, task, r), count)
                if args.variables:
                    r = {**r, "variables": True}
                f.write(json.dumps({**r, "question": question(task, r)}) + "\n")
        return

    if args.cmd == "dataset":
        english = {r["task"]: r["explanation"] for r in map(json.loads, args.english.read_text().splitlines())
                   if r.get("explanation")} if args.english else None
        found = ({f.stem: json.loads(f.read_text())["variables"] for f in sorted(args.variables.glob("*.json"))}
                 if args.variables else {})
        if args.english:  # the writer's variables (meanings written) over the teacher's
            found |= {r["task"]: r["variables"] for r in map(json.loads, args.english.read_text().splitlines()) if r.get("variables")}
        print(dataset(args.rows, args.teacher, args.out, goal_eval.GoalEval(), english,
                      edit_teach(args.edit_teach) if args.edit_teach else None, found or None), "rows", flush=True)
        return
    if args.cmd == "facts":
        ge = goal_eval.GoalEval(native.Native(args.device))
        n = 0
        with args.out.open("w") as f:
            for r in map(json.loads, args.rows.read_text().splitlines()):
                js = args.teacher / f"{r['task']}.json"
                if n >= args.n or not js.exists():
                    continue
                task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
                prompt = task["prompts"][0]  # the circuit as it is scored and trained on (dataset(): stranded nodes dropped)
                g, _ = goal_eval.circuit(json.loads(js.read_text()), prompt["target_positions"], ge.nat.C, len(prompt["token_ids"]))
                vs = args.variables / f"{r['task']}.json" if args.variables else None
                vs = json.loads(vs.read_text())["variables"] if vs and vs.exists() else None
                f.write(json.dumps(teacher_facts(ge, task, r, goal_eval.to_json(g), variables=vs)) + "\n")
                f.flush()
                n += 1
        return

    ge = goal_eval.GoalEval()
    nat = ge.nat
    if args.cmd == "locality":
        done = {json.loads(x)["task"] for x in args.out.read_text().splitlines()} if args.out.exists() else set()
        lo, hi = map(int, args.lengths.split(":"))
        with args.out.open("a") as f:
            texts = ([native.TEXTS / "vpd4l" / f"{t}.json" for t in args.tasks.read_text().split()] if args.tasks
                     else native.tasks(args.split)[:args.n])
            for p in texts[args.offset::args.stride]:
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
        conn, atlas, lens = _connectome(), _atlas(), _lens(nat)
        lo, hi = map(int, args.lengths.split(":"))
        paths = native.tasks(args.split)[:args.n]
        if args.tasks:
            given = [json.loads(x) for x in args.tasks.read_text().splitlines() if x.strip()]
            for r in given:
                if "local" in r and r["task"] not in labels:
                    labels[r["task"]] = {"task": r["task"], "local": r["local"], "given": True}
            paths = [native.TEXTS / "vpd4l" / f"{r['task']}.json" for r in given]
        for p in paths:
            if not lo <= len(native.text(p)[0]) < hi or p.stem in kept:
                continue
            if p.stem in labels and labels[p.stem]["local"]:
                local_kept = sum(labels[t]["local"] for t in kept)
                if local_kept >= len(kept) - local_kept:
                    continue
            task = json.loads(p.read_text())
            if p.stem not in labels:
                labels[p.stem] = locality(ge, task)
                with files["locality"].open("a") as f:
                    f.write(json.dumps(labels[p.stem]) + "\n")
            local_kept = sum(labels[t]["local"] for t in kept)
            if labels[p.stem]["local"] and local_kept >= len(kept) - local_kept:  # at least half the kept texts non-local
                continue
            ev = evidence(nat, ge, task, conn, atlas=atlas, lens=lens)
            row = {**ev, "question": question(task, ev)}
            info = None
            if not args.rows_only:
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
    if args.cmd == "noise":
        import statistics

        rows = {r["task"]: r for r in map(json.loads, args.rows.read_text().splitlines())}
        sets = {k: samples_of(Path(v)) for k, v in (x.split("=", 1) for x in args.set)}
        got = []
        for p in native.tasks(args.split)[:args.n]:
            if p.stem not in rows:
                continue
            task = json.loads(p.read_text())
            for name, samples in sets.items():
                if samples.get(p.stem):
                    got += [{"task": p.stem, "set": name, **r} for r in c_noise(ge, task, [samples[p.stem][0][0]], args.seeds)]
                    print(json.dumps(got[-1]), flush=True)
        sds = [g["sd_c_rl"] for g in got if g["sd_c_rl"] is not None]
        summary = {"answers": len(got), "w (median sd of c_rl)": statistics.median(sds) if sds else None,
                   "mean sd of c_rl": statistics.mean(sds) if sds else None,
                   "median sd of c": statistics.median([g["sd_c"] for g in got if g["sd_c"] is not None]) if got else None}
        args.out.write_text(json.dumps({"summary": summary, "answers": got}))
        print(json.dumps(summary), flush=True)
        return
    if args.cmd == "references":
        print(references(ge, args.rows, args.teacher, args.out, edit_teach(args.edit_teach) if args.edit_teach else None), "texts", flush=True)
        return
    if args.cmd == "edits":
        import edit_eval

        given = {}
        for p in args.goals:
            for g in map(json.loads, p.read_text().splitlines()):
                if g["goal"] not in given.setdefault(g["task"], []):
                    given[g["task"]].append(g["goal"])
        ee = None if args.goals else edit_eval.EditEval(nat)
        with args.out.open("w") as f:
            for r in map(json.loads, args.rows.read_text().splitlines()):
                task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
                if args.goals and r["task"] not in given:
                    continue  # no goals for this text: no row (its answers could not be scored on edits)
                r["edits"] = given[r["task"]] if args.goals else edit_eval.goals(ee, task, native.task_seed(task["id"]) + args.qa_seed)
                f.write(json.dumps({**r, "question": question(task, r)}) + "\n")
                f.flush()
                print(r["task"], [g["kind"] for g in r["edits"]], flush=True)
        return
    if args.cmd == "structure":
        with args.out.open("w") as f:
            for r in map(json.loads, args.rows.read_text().splitlines()):
                task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
                seed = native.task_seed(task["id"]) + args.qa_seed
                r["qa"] = [q for q in r["qa"] if q["kind"] not in STRUCTURE] + structure_questions(nat, task, r["positions"], seed)
                random.Random(seed).shuffle(r["qa"])
                r["structure"] = structure_events(nat, task, r["positions"], seed + 1)
                f.write(json.dumps({**r, "question": question(task, r)}) + "\n")
                f.flush()
                print(r["task"], sum(q["kind"] in STRUCTURE for q in r["qa"]), "of", len(r["qa"]), "questions structural;",
                      len(r["structure"]), "reader events", flush=True)
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
        conn, atlas, lens = _connectome(), _atlas(), _lens(nat)
        done = set()
        if args.out.exists():
            done = {json.loads(x)["task"] for x in args.out.read_text().splitlines() if x.strip()}
        with args.out.open("a") as f:
            for p in native.tasks(args.split)[:args.n][args.offset::args.stride]:
                if p.stem in done:
                    continue
                task = json.loads(p.read_text())
                ev = evidence(nat, ge, task, conn, atlas=atlas, lens=lens, qa_seed=native.task_seed(task["id"]) + args.qa_seed)
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
    ee = edit_best = None
    if any(r.get("edits") for r in rows.values()):
        import edit_eval

        ee = edit_eval.EditEval(nat)
        _, edit_best = edit_eval.bests(args.edit_teach)  # the best net per goal over the teacher and the baselines
    vs = None
    if args.variables:
        import variables

        vs = variables.Scorer(ge)
    teachers = {f.stem: json.loads(f.read_text()) for t in rows if (f := args.teacher / f"{t}.json").exists()} if args.teacher else None
    out = evaluate(ge, args.split, rows, sets, args.n, e1, labels, args.diagnostics, ee=ee, edit_best=edit_best, vs=vs,
                   matched_all=args.matched_all, teachers=teachers, lines=args.lines)
    print(json.dumps(out["report"], indent=1), flush=True)
    if args.out:
        args.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
