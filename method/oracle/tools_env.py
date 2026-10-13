"""The oracle's tool environment (#2951 graph oracle): a short question and tools the oracle calls before it answers,
in place of one question that lists every active subcomponent (5-25k Qwen3 tokens).

The question (question()) holds the answer format (components.ASK), the text's tokens, the target and the model's top
next tokens, and the text's intervention questions and edit requests; the evidence the one-shot question lists comes
from tools, measured on demand on the real model and VPD's decomposition (goal_eval, engine, native):

  influence(top)                           context positions whose changed token moves the prediction (the sensitivity
                                           probe behind goal_eval.Text.moved: one draw from the model's own prediction
                                           per position), in bits, with their share of the total
  list_active(position, site, top, offset) VPD-active subcomponents at a position (causal importance above 0 and able to
                                           reach the targets: VPD's answer, goal_eval.Text.vpd), ranked by causal
                                           importance x |activation|, with the routed lens's raised tokens and the
                                           atlas's top tokens (components.facts); continue at next_offset until null
  attention(layer, position, head, top)    each head's most-read source positions at a position in the real model
  measure_removal(nodes, each)             KL in bits of the model's prediction with the nodes removed from the real
                                           model's weights at their positions (native.Native.without; each: one
                                           removal per node, goal_eval.GoalEval.real_removal, most needed first, a page
                                           of top at offset)
  node_effects(draft, top, offset)         per node of a draft, the KL change in bits when that node alone leaves the
                                           draft: path-patched on changed inputs (GoalEval.line_effects "pp") and on
                                           the text in C's semantics (line_effects "c"), most needed first; its order
                                           (least needed first by the larger of the two) is what drop_least drops
  run_circuit(circuit, changes)            a draft CIRCUIT (Mermaid, parsed by components.parse, checked by
                                           goal_eval.circuit): its point score on the text (graph F1 in C's semantics,
                                           GoalEval.sufficiency), its path-patched score on changed inputs
                                           (GoalEval.track_pp: s_track_pp) and its description bits (size_bits); with
                                           changes, the circuit's path-patched prediction on that changed text
                                           (GoalEval.pp_run) beside the model's; families: the reward's own families
                                           (GoalEval.fidelity in reward mode), phi, the text's bar (tau's phi_tau)
                                           and the circuit reward phi + g x 0.5 compact, all on the tools' draws (a
                                           Text copy with its seed shifted by SEED_SHIFT: the circuit-seeded draws
                                           differ from the score's too)
  compare(circuit_a, circuit_b, top)       the changed inputs (track_pp's draws) on which two drafts' path-patched
                                           predictions disagree most, each beside the model's

Tool results are compact JSON, cached per text and arguments. The tools measure on the other split's draws than the
score (TOOL_SPLIT: the reward's "search" draws on training texts are never the tools'; on held-out texts the score
takes "eval" and the tools "search"), so a policy cannot fit the scored draws by calling tools.

An episode: a system message (the tools' use; the tool schemas in Qwen3's format, qwen3_tools(), which the renderer
puts in the system prompt), the user's short question, then the policy's turns: an assistant message with tool calls
gets one tool message per call; the first assistant message without a tool call is the answer (components' grammar).
Reward: components.reward (v3) of the answer on the text, the Text shared with the tools, minus COST per tool call;
an episode cut at MAX_TURNS or MAX_CALLS without an answer earns components.V3_INVALID minus its calls' cost.

The local runner (run_episode) takes any policy(messages, tools) -> assistant message ({"role": "assistant",
"content": str, "tool_calls": [{"name", "arguments"}]}); Scripted is the baseline policy (influence -> the actives at
the targets and the most influential positions, as many as the path-patched score needs -> nodes pruned by measured
removal, each step checked with run_circuit). serve() puts the same environment behind HTTP for a policy sampled
elsewhere (tools_tinker.py: Tinker's cookbook, multi-turn).

  tools_env.py bench --task text9000 [--task ...]              each tool's latency on the device
  tools_env.py scripted ROWS.jsonl --tasks text9000:text9019 --out OUT.jsonl [--score-split eval]
  tools_env.py tokens ROWS.jsonl --tasks text9000:text9019 [--episodes OUT.jsonl]   Qwen3 tokens: one-shot vs episode
  tools_env.py baselines ROWS.jsonl --episodes OUT.jsonl --out BASE.jsonl   the scripted answers beside equal-size sets
  tools_env.py serve ROWS.jsonl [--port 8771] [--score-split search]
ROWS: components.py questions rows (each text's evidence: its intervention questions and edit requests).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import carry  # noqa: E402
import components  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402

COST = 0.02  # reward bits per tool call (a run_circuit costs the policy as much as a cheap one: calls are the budget)
MAX_TURNS = 24  # assistant turns per episode
MAX_CALLS = 40  # tool calls per episode
TEXTS_KEPT = 8  # texts whose Tools a process keeps (an RL group's episodes share one)
LIST_TOP = 24  # list_active's default
LIST_MAX = 400  # list_active's most
REMOVE_EACH_MAX = 4096  # measure_removal(each=True)'s most nodes (an influential start reaches 1,127 on text9001)
REMOVE_CHUNK = 256  # measure_removal(each=True)'s removals per real_removal call
EFFECTS_MAX = 4096  # node_effects' most nodes in a draft
EFFECT_DRAWS = 8  # node_effects' changed inputs (line_effects' draws on the tools' split)
COMPARE_DRAWS = goal_eval.CHANGED_DRAWS  # compare's changed inputs (track_pp's draws on the tools' split)
OTHER = {"search": "eval", "eval": "search"}
SEED_SHIFT = 7919  # the families' Text copy's seed offset: circuit-seeded draws (nodes, edges, flips) unlike the score's
LN2 = math.log(2)

SYSTEM = ("You explain how a small language model, vpd4l (layers 0 to 3, each with attention (q, k, v, o) and an MLP "
          "(fc, down)), computes its next-token prediction on a text, in VPD subcomponents: rank-one slices of its weight "
          "matrices, 38,912 in all, written L<layer>.<kind>#<index>@<position>. You do not know the subcomponents' roles "
          "in advance: measure them with the tools before you answer. Useful order: influence() for the context "
          "positions the prediction reads; list_active() at the target and at those positions (facts=false for long "
          "lists; next_offset as offset for further pages); attention() for where the heads at the target read; "
          "measure_removal() for which subcomponents the real model needs; run_circuit() to score a draft circuit "
          "(point: graph F1 on the text; pp: path-patched agreement with the model on changed inputs; 1 is as faithful "
          "as VPD's whole answer) and compare() to see where two drafts differ. A draft is given as positions=[...] "
          "(every VPD-active subcomponent there), nodes=[...] (with edges=[[writer, reader], ...], reader OUT for the "
          "prediction, or every connection among them when no edges are given) or Mermaid text, or as base=<draft id> "
          "with add=[...], drop=[...] or drop_least=n (after node_effects(draft=...) or measure_removal(draft=..., "
          "each=true): the n least needed by the latest of them); each run_circuit returns its draft id (d1, d2, ...), "
          "which compare(), node_effects() and measure_removal() also take. In your answer the CIRCUIT section may be a "
          "draft id alone (CIRCUIT: d3): its Mermaid is filled in. "
          "Each tool call costs a little reward: call what you need, then reply without a tool call. That reply is your "
          "answer, in the format the question asks for. You have at most 24 turns and 40 tool calls: an episode that "
          "runs out of either without that reply scores nothing, so answer before then, naming your best draft "
          "(CIRCUIT: d3) if you have one.")


MERMAID_HINT = ("Declare each node on its own line inside its position's subgraph, a[\"L3.fc#2304\"] (no @position in the "
                "label: the subgraph P<position> gives it), then the edges by id on their own lines, a --> b and a --> OUT; "
                "or pass nodes=[...] (and edges) or positions=[...] instead of Mermaid.")


def _r(x, k: int = 3):
    """A float rounded to k significant digits (JSON-compact), None kept."""
    if x is None:
        return None
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return x if math.isfinite(x) else None
    return round(x, max(0, k - 1 - int(math.floor(math.log10(abs(x))))))


class ToolError(ValueError):
    """A tool call the policy got wrong: its message goes back to the policy as the result."""


TOOL_SPECS = [
    {"name": "influence",
     "description": "Context positions whose token, replaced by a draw from the model's own prediction there, moves the "
                    "prediction at the target most: KL in bits and share of the total over all positions, most first.",
     "parameters": {"type": "object", "properties": {
         "top": {"type": "integer", "description": "positions to list (default 12)"}}, "required": []}},
    {"name": "list_active",
     "description": "VPD-active subcomponents (causal importance above 0) at a position, ranked by causal importance x "
                    "|activation|: each with ci, act, and facts (the tokens its write raises there by the routed lens, a "
                    "linear reading, and the tokens it is most active on in Pile text). 'active' and 'total' count them "
                    "all; pass next_offset as offset to continue, until next_offset is null. Equal scores retain native matrix "
                    "order, then subcomponent index.",
     "parameters": {"type": "object", "properties": {
         "position": {"type": "integer"},
         "site": {"type": "string", "description": "only this matrix, L<layer>.<kind>, e.g. L2.v"},
         "top": {"type": "integer", "description": f"subcomponents per page (default {LIST_TOP}, at most {LIST_MAX})"},
         "offset": {"type": "integer", "minimum": 0, "description": "zero-based start in the ranked list (default 0)"},
         "facts": {"type": "boolean", "description": "each subcomponent's lens and Pile tokens (default true)"}},
         "required": ["position"]}},
    {"name": "attention",
     "description": "Where the attention heads of a layer read at a position in the real model: per head its top source "
                    "positions with their attention weights.",
     "parameters": {"type": "object", "properties": {
         "layer": {"type": "integer"}, "position": {"type": "integer"},
         "head": {"type": "integer", "description": "one head only (default: every head)"},
         "top": {"type": "integer", "description": "source positions per head (default 4)"}},
         "required": ["layer", "position"]}},
    {"name": "measure_removal",
     "description": "Remove subcomponents (L<layer>.<kind>#<index>@<position>) from the real model's weights at their "
                    "positions, everything else unchanged: the KL in bits of the prediction at the target from the "
                    "model's, and the new top tokens. each=true: one removal per subcomponent instead of all at once, "
                    "most needed first, top per page from offset (next_offset until null); the whole order is kept for "
                    "drop_least. draft: a draft's subcomponents in place of nodes.",
     "parameters": {"type": "object", "properties": {
         "nodes": {"type": "array", "items": {"type": "string"}},
         "draft": {"type": "string", "description": "a draft id from run_circuit"},
         "each": {"type": "boolean"},
         "top": {"type": "integer", "description": f"each=true: removals per page (default {LIST_TOP}, at most {LIST_MAX})"},
         "offset": {"type": "integer", "minimum": 0, "description": "each=true: zero-based start in the order (default 0)"}},
         "required": []}},
    {"name": "node_effects",
     "description": "Per subcomponent of a draft circuit, how much the draft needs it: the KL change in bits at the target "
                    "when that node alone leaves the draft, path-patched on changed inputs (pp_bits, weighted by how far "
                    "each change moves the model) and on the text (point_bits); above 0: the draft needs it. Most needed "
                    "first by the larger of the two, top per page from offset; the whole order is kept for drop_least.",
     "parameters": {"type": "object", "properties": {
         "draft": {"type": "string", "description": "a draft id from run_circuit"},
         "top": {"type": "integer", "description": f"nodes per page (default {LIST_TOP}, at most {LIST_MAX})"},
         "offset": {"type": "integer", "minimum": 0, "description": "zero-based start in the order (default 0)"}},
         "required": ["draft"]}},
    {"name": "run_circuit",
     "description": "Score a draft circuit: point (its graph F1 on the text, 1 = as close to the model as VPD's whole "
                    "answer, on a log scale from the empty circuit), pp (its path-patched agreement with the model on "
                    "changed inputs, the same scale), its size in bits, the nodes and edges dropped as invalid or "
                    "stranded, and its draft id. Give positions, nodes (with edges, or every connection among them), "
                    "Mermaid text (circuit), or base with add and drop. changes: [[position, token], ...] runs the "
                    "circuit path-patched on that changed text instead and gives its top tokens beside the model's. "
                    "families=true adds the reward's own measures (see families).",
     "parameters": {"type": "object", "properties": {
         "positions": {"type": "array", "items": {"type": "integer"}, "description": "every VPD-active subcomponent at these positions"},
         "nodes": {"type": "array", "items": {"type": "string"}, "description": "L<layer>.<kind>#<index>@<position> (tok@<position>: the token embedding)"},
         "edges": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                   "description": "[[writer, reader], ...] by node, reader OUT for the prediction (none: every connection among the nodes)"},
         "circuit": {"type": "string", "description": "the Mermaid flowchart of the answer format"},
         "base": {"type": "string", "description": "a draft id to start from"},
         "add": {"type": "array", "items": {"type": "string"}, "description": "nodes added to the base"},
         "drop": {"type": "array", "items": {"type": "string"}, "description": "nodes dropped from the base (with their edges)"},
         "drop_least": {"type": "integer", "description": "drop the base's n least needed nodes, by the latest node_effects(draft=base) or measure_removal(draft=base, each=true) order (kept by later drafts)"},
         "changes": {"type": "array", "items": {"type": "array"},
                     "description": "[[position, token], ...], token as quoted in the question, e.g. \"' castle'\", or an id"},
         "families": {"type": "boolean", "description": "also the reward's own measures on independent draws (slower, about "
                                                        "6 s): each family's score, phi, the bar phi must reach before size "
                                                        "counts, and the circuit's reward phi + gate x 0.5 log2(start bits / bits)"}},
         "required": []}},
    {"name": "compare",
     "description": "The changed inputs on which two draft circuits' path-patched predictions disagree most: per input "
                    "the change, the model's top token, each draft's top token and KL from the model. Each circuit a "
                    "draft id (d3) or Mermaid text.",
     "parameters": {"type": "object", "properties": {
         "circuit_a": {"type": "string"}, "circuit_b": {"type": "string"},
         "top": {"type": "integer", "description": "inputs to list (default 5)"}}, "required": ["circuit_a", "circuit_b"]}},
]
TOOL_NAMES = [s["name"] for s in TOOL_SPECS]


def qwen3_tools() -> list[dict]:
    """The tool schemas in Qwen3's (OpenAI's) function format, for its chat template's tools argument."""
    return [{"type": "function", "function": s} for s in TOOL_SPECS]


def _site_of(text: str, names=None) -> str:
    """A matrix from L<layer>.<kind> (or goal_eval._site's forms), one of names when given (vpd4l's otherwise)."""
    t = str(text).strip()
    t = t[1:] if t[:1] in ("L", "l") else t
    try:
        n = goal_eval._site(t)
    except (ValueError, KeyError):
        n = None
    if n is None or (names is not None and n not in names):
        raise ToolError(f"site {text!r}: write L<layer>.<kind>, layer 0 to 3, kind one of q, k, v, o, fc, down")
    return n


class Tools:
    """The tools on one text: goal_eval's Text (the model's run, VPD's answer, the probe), shared with the reward, and
    every result kept by tool and arguments."""

    def __init__(self, ge: goal_eval.GoalEval, task: dict, split: str = "search", lens=None, atlas=None, pool: dict | None = None):
        self.ge, self.nat, self.task, self.split, self.pool = ge, ge.nat, task, split, ({} if pool is None else pool)
        self._px = None  # the families' Text copy
        p = task["prompts"][0]
        self.ids, self.targets = list(p["token_ids"]), list(p["target_positions"])
        self.T, self.t = len(self.ids), max(self.targets)
        t0 = time.perf_counter()
        self.tx = ge.text(self.ids, self.targets, task=task)
        self.setup_seconds = time.perf_counter() - t0
        self.strings = components.token_strings(task["model"], self.ids)
        self.lens, self.atlas = lens, atlas
        self.routed: dict = {}
        self.memo: dict = {}
        self.seconds: dict = {}  # tool -> [seconds of each uncached call]

    # ---- dispatch

    def call(self, name: str, args: dict | str | None) -> str:
        """A tool's result as the text the policy reads (compact JSON; an error as {"error": ...})."""
        return self.call_full(name, args)[0]

    def call_full(self, name: str, args: dict | str | None) -> tuple[str, dict]:
        """(the result's text, its side data: keys starting with _, never shown: run_circuit's circuit JSON as
        "_json"); a repeated call is answered from memory."""
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"arguments are not JSON: {e}"}), {}
        args = args if isinstance(args, dict) else {}
        key = (name, json.dumps(args, sort_keys=True))
        if key in self.memo:
            return self.memo[key]
        if name not in TOOL_NAMES:
            return json.dumps({"error": f"no tool {name!r}; tools: {', '.join(TOOL_NAMES)}"}), {}
        t0 = time.perf_counter()
        try:
            out = getattr(self, name)(**args)
        except (ToolError, TypeError) as e:
            out = {"error": str(e)}
        except Exception as e:  # a tool's own failure goes back to the policy (and the log), the episode goes on
            import traceback

            traceback.print_exc()
            out = {"error": f"{type(e).__name__}: {e}"}
        finally:
            native.free(self.nat.dev)
        self.seconds.setdefault(name, []).append(time.perf_counter() - t0)
        side = {k: out.pop(k) for k in [k for k in out if k.startswith("_")]}
        self.memo[key] = (json.dumps(out, separators=(",", ":"), ensure_ascii=False), side)
        return self.memo[key]

    def _pos(self, p, name: str = "position") -> int:
        try:
            p = int(p)
        except (TypeError, ValueError):
            raise ToolError(f"{name} must be an integer") from None
        if not 0 <= p <= self.t:
            raise ToolError(f"{name} {p} is outside the text up to the target (0 to {self.t})")
        return p

    def _tok(self, i: int) -> str:
        return repr(self.strings[i])

    def _nodes(self, items) -> list[tuple]:
        if isinstance(items, str):
            items = [x for x in items.replace(";", ",").split() if x]
        out = []
        for x in items or []:
            g = components.group_of(str(x))
            if not g:
                raise ToolError(f"{x!r} is not a subcomponent: write L<layer>.<kind>#<index>@<position>")
            for nd in g:
                if nd[0] == components.TOK or nd[2] == components.BLOCK:
                    raise ToolError(f"{x!r}: only subcomponents can be removed (no tok or block)")
                if nd[0] not in self.nat.C:
                    raise ToolError(f"{x!r}: no such matrix (layers 0 to {self.nat.target.n_layer - 1}, kinds q, k, v, o, fc, down)")
                if not 0 <= nd[2] < self.nat.C[nd[0]]:
                    raise ToolError(f"{x!r}: index outside the matrix's {self.nat.C[nd[0]]} subcomponents")
                self._pos(nd[1])
                out.append(nd)
        if not out:
            raise ToolError("no nodes given")
        return list(dict.fromkeys(out))

    def _answer(self, circuit=None, nodes=None, edges=None, positions=None, complete=None, _json=None) -> dict:
        """parse()'s form ({"answer": circuit JSON, "malformed", "malformed_nodes", "unknown_ids"}) of a draft given
        as circuit JSON (_json, a draft's), Mermaid text, a list of nodes in a string, or nodes, edges and positions."""
        if _json is not None:
            return {"answer": _json, "malformed": 0, "malformed_nodes": 0, "unknown_ids": 0}
        if isinstance(circuit, str) and circuit.strip():
            items = [x for x in re.split(r"[\s,;]+", circuit.strip()) if x]
            if not re.search(r"flowchart|subgraph|-->|CIRCUIT", circuit) and all(components.group_of(x) for x in items):
                nodes, circuit = items, None  # a node list: every connection among them
            else:
                body = circuit if "CIRCUIT" in circuit.upper() else "CIRCUIT:\n" + circuit
                p = components.parse(body)
                p["bad_examples"] = self._bad_lines(circuit)
                return p
        if circuit is not None:
            raise ToolError("circuit must be the Mermaid flowchart text or a list of nodes")
        if not nodes and not positions:
            raise ToolError("give positions, nodes, circuit or base")
        tx = self.tx
        out, at, bad = [], {}, 0

        def place(nd):
            if nd not in at:
                at[nd] = len(out)
                out.append(["block", nd[0], nd[1]] if nd[2] == components.BLOCK else [nd[0], nd[2], nd[1]])
            return at[nd]

        for p in positions or []:
            p = self._pos(p, "a position")
            for n in self.nat.names:
                for c in (tx.vpd[n][p].nonzero().flatten().tolist() if tx.vpd is not None else []):
                    place((n, p, c))
        for x in nodes or []:
            g = components.group_of(str(x))
            if not g:
                bad += 1
                continue
            for nd in g:
                place(nd)
        es = []
        for e in edges or []:
            if not isinstance(e, (list, tuple)) or len(e) != 2:
                bad += 1
                continue
            w = components.node_of(str(e[0]))
            r = "out" if str(e[1]).strip().lower() == "out" else components.node_of(str(e[1]))
            if w is None or r is None:
                bad += 1
                continue
            es.append([place(w), "out" if r == "out" else place(r)])
        ans = {"nodes": out, "edges": es}
        if (complete is None and not edges) or complete:
            ans["complete"] = True
        return {"answer": ans, "malformed": 0, "malformed_nodes": bad, "unknown_ids": 0}

    @staticmethod
    def _bad_lines(text: str, k: int = 3) -> list[str]:
        """The first k lines of a Mermaid draft that no line form of components' grammar reads."""
        out = []
        for line in (x.strip() for x in text.splitlines()):
            if (not line or line.startswith(("```", "%%", "direction ", "CIRCUIT")) or line.lower().startswith("flowchart")
                    or line == "end" or components.MM_SUB.fullmatch(line) or components.MM_DECLS.fullmatch(line)
                    or components.MM_PAIR.fullmatch(line) or components.MM_EDGE.fullmatch(line)):
                continue
            out.append(line[:160])
            if len(out) >= k:
                break
        return out

    def _graph(self, circuit=None, **kw):
        """(goal_eval's Graph of a draft, its defects, parse()'s form) from any of _answer's input forms."""
        p = self._answer(circuit, **kw)
        if not p["answer"]["nodes"]:
            raise ToolError("the circuit declares no node: " + MERMAID_HINT + (f" Lines not read: {p['bad_examples']}" if p.get("bad_examples") else ""))
        g, d = goal_eval.circuit(p["answer"], self.targets, self.nat.C, self.T)
        d["malformed_lines"] = p["malformed"]
        d["unknown_ids"] = p["unknown_ids"]
        return g, d, p

    def _bits(self, g, p, d) -> float:
        nat = self.nat
        P, S = nat.positions(self.targets), sum(nat.C.values())
        junk = p["malformed_nodes"] * math.log2(P * S)
        return goal_eval.size_bits(g, P, S, self.targets, active=self.tx.vpd) + components.dropped_bits(p["answer"], g, d, P, S) + junk

    def _top(self, lq: torch.Tensor, k: int = 3) -> list:
        row = lq[0, -1].float()
        v, i = row.topk(k)
        return [[self._tokstr(int(a)), _r(math.exp(float(b)), 2)] for a, b in zip(i.tolist(), v.tolist())]

    def _tokstr(self, i: int) -> str:
        return repr(components.token_strings(self.task["model"], [i])[0])

    # ---- the tools

    def influence(self, top: int = 12) -> dict:
        moved = np.asarray(self.tx.moved, dtype=float) / LN2  # bits; moved[p - 1] for position p
        total = float(moved.sum())
        top = max(1, min(int(top), len(moved)))
        order = np.argsort(-moved)[:top]
        rows = [[int(j) + 1, self._tok(int(j) + 1), _r(moved[j]), _r(moved[j] / total if total > 0 else 0.0, 2)] for j in order]
        return {"target": self.t, "target_token": self._tok(self.t), "total_bits": _r(total),
                "over_1pct": int((moved >= goal_eval.SHARE_FLOOR * total).sum()) if total > 0 else 0,
                "columns": ["position", "token", "bits", "share"], "positions": rows}

    def list_active(self, position, site: str | None = None, top: int = LIST_TOP, offset: int = 0, facts: bool = True) -> dict:
        p = self._pos(position)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be a non-negative integer")
        top = max(1, min(int(top), LIST_MAX))
        matrix_order = {n: i for i, n in enumerate(self.nat.names)}
        names = [_site_of(site)] if site else self.nat.names
        if site and names[0] not in self.nat.names:
            raise ToolError(f"site {site!r} is not one of the model's matrices")
        tx, nat = self.tx, self.nat
        ranked = []
        for n in names:
            cs = tx.vpd[n][p].nonzero().flatten().tolist() if tx.vpd is not None else []
            if not cs:
                continue
            ci = tx.ci[n][p, cs].float().cpu().tolist()
            act = tx.rec[n][0, p, cs].float().cpu().tolist()
            ranked += [((n, p, c), a, b) for c, a, b in zip(cs, ci, act)]
        ranked.sort(key=lambda x: (-x[1] * abs(x[2]), matrix_order[x[0][0]], x[0][2]))
        shown = ranked[offset:offset + top]
        facts = self._facts([nd for nd, _, _ in shown]) if facts else {}
        return {"position": p, "token": self._tok(p), "active": len(ranked), "shown": len(shown),
                "total": len(ranked), "offset": offset,
                "next_offset": offset + len(shown) if offset + len(shown) < len(ranked) else None,
                "columns": ["node", "ci", "act"] + (["facts"] if facts else []),
                "subcomponents": [[components.node_text(nd), _r(c, 2), _r(a, 3)] + ([facts.get(nd, "")] if facts else []) for nd, c, a in shown]}

    def _facts(self, nodes: list) -> dict:
        if not nodes or self.atlas is None:
            return {}
        need = [nd for nd in nodes if nd not in self.routed]
        if need and self.lens is not None:
            self.routed.update(zip(need, self.lens.readout(self.ids, need)))
        tok = lambda xs: components.token_strings(self.task["model"], xs)  # noqa: E731
        return {nd: components.facts(self.atlas, nd, self.routed.get(nd), tok) for nd in nodes}

    def attention(self, layer, position, head=None, top: int = 4) -> dict:
        tg = self.nat.target
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            raise ToolError("layer must be an integer") from None
        if not 0 <= layer < tg.n_layer:
            raise ToolError(f"layer {layer}: vpd4l has layers 0 to {tg.n_layer - 1}")
        p = self._pos(position)
        heads = range(tg.n_head) if head is None else [int(head)]
        if any(not 0 <= h < tg.n_head for h in heads):
            raise ToolError(f"head: 0 to {tg.n_head - 1}")
        S = self.ge._pp_base(self.tx)["S"][layer]  # [H, T, T] scores in the real model
        P = S[:, p, :p + 1].float().softmax(-1)
        k = max(1, min(int(top), p + 1))
        w, at = P.topk(k, -1)
        return {"layer": layer, "position": p, "columns": ["position", "token", "weight"],
                "heads": {f"h{h}": [[int(a), self._tok(int(a)), _r(b, 2)] for a, b in zip(at[h].tolist(), w[h].tolist())] for h in heads}}

    @staticmethod
    def _page(rows: list, top, offset) -> tuple[list, int, int | None]:
        """(rows[offset:offset + top], offset, the next page's offset or None): top within 1 to LIST_MAX."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be a non-negative integer")
        try:
            top = max(1, min(int(top), LIST_MAX))
        except (TypeError, ValueError):
            raise ToolError("top must be an integer") from None
        page = rows[offset:offset + top]
        return page, offset, offset + len(page) if offset + len(page) < len(rows) else None

    def measure_removal(self, nodes=None, each: bool = False, draft=None, top: int = LIST_TOP, offset: int = 0, _json=None) -> dict:
        if draft is not None and _json is None:
            raise ToolError(f"no draft {draft!r} in this episode")
        if _json is not None:
            nodes = [components.node_text((m, t, c)) for m, c, t in _json["nodes"] if m not in ("block", components.TOK)]
        nds = self._nodes(nodes)
        tx = self.tx
        if each:
            if len(nds) > REMOVE_EACH_MAX:
                raise ToolError(f"each=true takes at most {REMOVE_EACH_MAX} nodes")
            rows, step, i = [], REMOVE_CHUNK, 0
            while i < len(nds):  # real_removal in chunks, halved when one does not fit (t2: 798 nodes, 3 processes on 44 GB)
                try:
                    rows += self.ge.real_removal(tx, [native.Graph([nd], complete=True) for nd in nds[i:i + step]])
                    i += step
                except torch.OutOfMemoryError:
                    native.free(self.nat.dev)
                    if step == 1:
                        raise
                    step = max(1, step // 2)
            got = sorted(zip(nds, [r["kl_real_minus"] for r in rows]), key=lambda x: -x[1])
            page, offset, nxt = self._page(got, top, offset)
            return {"each": len(got), "offset": offset, "next_offset": nxt, "columns": ["node", "kl_bits"],
                    "removals": [[components.node_text(nd), _r(k)] for nd, k in page],
                    "_order": [components.node_text(nd) for nd, _ in got[::-1]]}
        with torch.no_grad():
            lq = self.nat.without(self.ids, self.targets, nds)
        kl = float(carry.kl(tx.ref, lq)[0])
        return {"removed": len(nds), "kl_bits": _r(kl), "model_top": self._top(tx.ref), "top_after": self._top(lq)}

    def node_effects(self, draft=None, top: int = LIST_TOP, offset: int = 0, _json=None) -> dict:
        if _json is None:
            raise ToolError(f"no draft {draft!r} in this episode" if draft is not None else "node_effects needs draft, a draft id from run_circuit")
        g, _, _ = self._graph(_json=_json)
        g = goal_eval.strand(g, self.targets, self.nat.C)
        if not g.count():
            raise ToolError("the draft has no valid node")
        if g.count() > EFFECTS_MAX:
            raise ToolError(f"node_effects takes drafts of at most {EFFECTS_MAX} nodes")
        if offset != 0:
            self._page([], top, offset)  # the arguments' checks before the runs
        pp, draws = None, 0
        for draws in (EFFECT_DRAWS, 2) if self.tx.f2 else ():  # every removal in one engine call: fewer draws when it
            try:                                               # does not fit (r6: 1,127 nodes on a shared 48 GB card)
                pp = self.ge.line_effects(self.tx, g, mode="pp", split=self.split, draws=draws)
                break
            except torch.OutOfMemoryError:
                native.free(self.nat.dev)
        pt = self.ge.line_effects(self.tx, g, mode="c")
        d_pp = {tuple(r["node"]): r["delta_kl"] for r in pp["nodes"]} if pp else {}  # carry.kl's bits
        rows = [(tuple(r["node"]), d_pp.get(tuple(r["node"])), r["delta_kl"]) for r in pt["nodes"]]
        worst = lambda r: max(x for x in (r[1], r[2], -math.inf) if x is not None)  # noqa: E731
        rows.sort(key=lambda r: (-worst(r), r[0]))
        page, offset, nxt = self._page(rows, top, offset)
        return {"nodes": len(rows), "pp_kl_bits": _r(pp["base_kl"]) if pp else None, "pp_draws": draws if pp else 0, "point_kl_bits": _r(pt["base_kl"]),
                "offset": offset, "next_offset": nxt, "columns": ["node", "pp_bits", "point_bits"],
                "effects": [[components.node_text(nd), _r(a), _r(b)] for nd, a, b in page],
                "_order": [components.node_text(nd) for nd, _, _ in rows[::-1]]}

    def _score(self, g) -> dict:
        ge, tx = self.ge, self.tx
        cg = goal_eval._c_graph(g, self.targets, self.nat.C)
        pt = ge.sufficiency(tx, [cg])[0]
        pp = ge.track_pp(tx, [goal_eval.strand(g, self.targets, self.nat.C)], self.split)[0]
        return {"point": _r(pt["f1_live"]), "point_kl_bits": _r(pt["kl_live"]), "necessity": _r(pt["n_live"]),
                "pp": _r(pp["s_track_pp"]), "pp_kl_bits": _r(pp["kl_track_pp_c"]), "pp_inputs": pp["prompts_track_pp"]}

    def _proxy(self):
        """The Text the families run on: the tools' Text (its model runs, VPD's answer, its probe: F2's positions)
        with its seed shifted by SEED_SHIFT and its own tau, start, adversary and VPD-baseline caches, so the reward's
        circuit-seeded draws differ from the score's; caches without a seed (cuts, bounds, the changed texts' model
        runs, the path-patching engine) shared."""
        if self._px is None:
            self.tx._probed()
            px = copy.copy(self.tx)
            px.seed = int(self.tx.seed) + SEED_SHIFT
            for k in ("_tau", "_start", "_adv", "_vpd_c"):  # kept per split or holding seeded draws: the copy's own
                px.__dict__.pop(k, None)
            self._px = px
        return self._px

    def _families(self, g, bits: float) -> dict:
        ge, px = self.ge, self._proxy()
        row = ge.fidelity(px, [g], self.pool, mode="reward", split=self.split)[0]
        tau = ge.tau(px, self.pool, self.split)
        ph, _ = goal_eval.phi(row, families=tau.get("families"))
        pt = tau.get("phi_tau")
        comp = goal_eval.compact(bits, tau.get("bits_start"))
        gate = None if ph is None or pt is None else 1.0 / (1.0 + 2.0 ** (-components.GATE_SLOPE * (ph - pt)))
        return {"phi": _r(ph), "bar": _r(pt), "gate": _r(gate, 2), "compact": _r(comp), "reward": _r((ph or 0.0) + (gate or 0.0) * 0.5 * (comp or 0.0)),
                "families": {f: _r(goal_eval.family_score(row, f), 2) for f in (tau.get("families") or goal_eval.PHI_FAMILIES)}}

    def run_circuit(self, circuit=None, nodes=None, edges=None, positions=None, complete=None, changes=None, _json=None,
                    base=None, add=None, drop=None, families: bool = False) -> dict:
        if base is not None or add or drop:
            raise ToolError("base, add and drop need an episode's drafts (the environment resolves them)")
        g, d, p = self._graph(circuit, nodes=nodes, edges=edges, positions=positions, complete=complete, _json=_json)
        g = goal_eval.strand(g, self.targets, self.nat.C)
        defects = {"dropped_nodes": d["invalid_nodes"] + d["unknown_ids"] + p.get("malformed_nodes", 0) * (_json is None and circuit is None),
                   "dropped_edges": d["invalid_edges"], "stranded_nodes": d["stranded_nodes"], "bad_lines": d["malformed_lines"]}
        out = {"nodes": g.count(), "edges": native.count_edges(g.node_set(), self.targets) if g.complete else g.edges(),
               "bits": _r(self._bits(g, p, d)), **{k: v for k, v in defects.items() if v}, "_json": p["answer"]}  # defects when any
        if p.get("bad_examples"):
            out |= {"bad_examples": p["bad_examples"], "hint": MERMAID_HINT}
        if not g.count():
            return {**out, "error": "no valid node left"}
        if changes is None:
            got = {**out, **self._score(g)}
            if families:
                got |= self._families(g, self._bits(g, p, d))
            return got
        x2 = self._changed(changes)
        ge, tx = self.ge, self.tx
        with torch.no_grad():
            ref2 = self.nat.reference([x2], self.targets)
            pr = [(q, z) for q, (z, z0) in enumerate(zip(x2, self.ids)) if z != z0]  # pp_run's numbers, one engine call
            gs = [g] + ([native.Graph(masks=tx.vpd)] if tx.vpd is not None else [])
            lqs = ge._pp_engine(tx).path_patched(gs, [pr]) if pr else tx.ref[None].expand(len(gs), -1, -1, -1)
            lq = lqs[0, 0][None]
            vpd = lqs[1, 0][None] if tx.vpd is not None else None
        w, kc = float(carry.kl(ref2, tx.ref)[0]), float(carry.kl(ref2, lq)[0])
        kv = float(carry.kl(ref2, vpd)[0]) if vpd is not None else None
        return {**out, "model_top": self._top(ref2), "circuit_top": self._top(lq), "kl_bits": _r(kc),
                "moved_bits": _r(w), "vpd_kl_bits": _r(kv), "score": _r(goal_eval._fid(w, kc, kv)) if kv is not None else None}

    def _changed(self, changes) -> list[int]:
        if not isinstance(changes, list) or not changes:
            raise ToolError("changes: [[position, token], ...]")
        x2 = list(self.ids)
        for ch in changes:
            if not isinstance(ch, (list, tuple)) or len(ch) != 2:
                raise ToolError("each change is [position, token]")
            p = self._pos(ch[0])
            z = ch[1]
            if isinstance(z, str) and not z.lstrip("-").isdigit():
                zi = components._token_of(self.task["model"], z if z[:1] in "'\"" else repr(z))
                if zi is None:
                    raise ToolError(f"token {z!r} is not one vpd4l token (quote it as the question does, e.g. \"' castle'\")")
                z = zi
            z = int(z)
            if not 0 <= z < self.nat.target.wte.shape[0]:
                raise ToolError(f"token id {z} is outside the vocabulary")
            x2[p] = z
        return x2

    def compare(self, circuit_a, circuit_b, top: int = 5) -> dict:
        ge, tx, nat = self.ge, self.tx, self.nat
        gs = [goal_eval.strand(self._graph(**(c if isinstance(c, dict) else {"circuit": c}))[0], self.targets, nat.C)
              for c in (circuit_a, circuit_b)]
        if not all(g.count() for g in gs):
            raise ToolError("a circuit has no valid node")
        moved = np.asarray(tx.moved, dtype=float)
        positions = [p for p, _ in tx.f2]
        if not positions:
            raise ToolError("no context position moves the prediction by 1% of the total: nothing to change")
        got = ge._interchange_draws(tx, positions, self.split, COMPARE_DRAWS, first={p: -moved[p - 1] for p in positions})
        with torch.no_grad():
            lqs = ge._pp_engine(tx).path_patched(gs, [tuple(pz) for pz in got])  # [2, D, targets, V]
            rows, kept = [], tx.__dict__.get("_pp_prompts", {})  # track_pp's model runs on the changed texts
            for d, (p, z) in enumerate(got):
                x2 = list(tx.ids)
                x2[p] = z
                ref2 = kept[(p, z)][0] if (p, z) in kept else nat.reference([x2], self.targets)
                la, lb = lqs[0, d][None], lqs[1, d][None]
                dis = 0.5 * float(carry.kl(la, lb)[0] + carry.kl(lb, la)[0])
                rows.append((dis, p, z, ref2, la, lb))
        rows.sort(key=lambda r: -r[0])
        k = max(1, min(int(top), len(rows)))
        return {"inputs": len(rows), "columns": ["position", "from", "to", "disagreement_bits", "model_top", "a_top", "a_kl_bits",
                                                 "b_top", "b_kl_bits"],
                "disagree": [[p, self._tok(p), self._tokstr(z), _r(dis), self._top(ref2, 1)[0][0], self._top(la, 1)[0][0],
                              _r(float(carry.kl(ref2, la)[0])), self._top(lb, 1)[0][0], _r(float(carry.kl(ref2, lb)[0]))]
                             for dis, p, z, ref2, la, lb in rows[:k]]}


# ---- the episode


def question(task: dict, ev: dict) -> str:
    """The short question: the answer format (components.ASK), the text's tokens (numbered near the target and at
    position 0, elsewhere a position marker every components.MARK_EVERY tokens), the target and the model's top next
    tokens, the intervention questions and edit requests."""
    model = task["model"]
    p = task["prompts"][0]
    strings = components.token_strings(model, p["token_ids"])
    t = max(p["target_positions"])
    numbered = {0} | set(range(max(0, t - components.WINDOW), t + 1))
    top_tok = p["model_top"][0][0][0]
    top = ", ".join(f"{tok!r} {q:.2f}" for tok, q in p["model_top"][0][:3])
    lines = [components.ASK.format(model=model) + (" " + components.ASK_VARIABLES if ev.get("variables") else ""),
             "tokens (numbered near the target; elsewhere plain text, [n] marking position n): " + components.listing(strings, numbered),
             f"target: position {t}; the model's most probable next tokens there: {top}"]
    if ev.get("qa"):
        tok = lambda ids: components.token_strings(model, ids)  # noqa: E731
        lines += [components.qa_text(q, n, strings, repr(top_tok), tok) for n, q in enumerate(ev["qa"], 1)]
    lines += [f"Edit {n}: {components.edit_goal_text(g, strings, repr(top_tok))}" for n, g in enumerate(ev.get("edits") or [], 1)]
    return components.escape("\n".join(lines))


class Env:
    """The tool environment over texts: a GoalEval, the texts' question rows, their Tools (TEXTS_KEPT kept), the
    reward's split (score_split; the tools take the other), the flip adversary's pattern pool."""

    def __init__(self, rows: dict, ge: goal_eval.GoalEval | None = None, score_split: str = "search", cost: float = COST,
                 facts: bool = True, pool: dict | None = None):
        self.ge = ge or goal_eval.GoalEval(native.Native())
        self.rows, self.score_split, self.tool_split, self.cost = rows, score_split, OTHER[score_split], cost
        self.lens = components._lens(self.ge.nat) if facts else None
        self.atlas = components._atlas() if facts else None
        self.pool = components._pool() if pool is None else pool
        self.tools: OrderedDict = OrderedDict()
        self.tasks: dict = {}

    def task(self, tid: str) -> dict:
        if tid not in self.tasks:
            self.tasks[tid] = json.loads((native.TEXTS / "vpd4l" / f"{tid}.json").read_text())
        return self.tasks[tid]

    def text_tools(self, tid: str) -> Tools:
        if tid not in self.tools:
            self.tools[tid] = Tools(self.ge, self.task(tid), self.tool_split, self.lens, self.atlas, self.pool)
            while len(self.tools) > TEXTS_KEPT:
                self.tools.popitem(last=False)
        self.tools.move_to_end(tid)
        return self.tools[tid]

    def reset(self, tid: str) -> "Episode":
        return Episode(self, tid)

    def score(self, tid: str, answer: str) -> dict:
        """components.reward of one answer on its text (reward v3 on the score split), sharing the tools' Text."""
        tools = self.text_tools(tid)
        return components.reward(self.ge, self.task(tid), self.rows[tid], [answer], cache={tid: tools.tx},
                                 split=self.score_split, pool=self.pool)[0]


def free_text(env: Env, tid: str) -> None:
    """Drop a text's Tools (its Text's GPU tensors) and return the memory to the device."""
    env.tools.pop(tid, None)
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


DRAFT_RE = re.compile(r"\bd\d+\b")
CIRCUIT_DRAFT = re.compile(r"(CIRCUIT\s*:)\s*(?:```(?:mermaid)?\s*)?(d\d+)\s*(?:```)?[ \t]*(?=\n|$)", re.I)


class Episode:
    """One episode on one text: its messages, its tool calls, its drafts (each run_circuit's circuit JSON by id, d1,
    d2, ...: later calls start from one with base, add and drop, compare and measure_removal take ids, and an answer's
    CIRCUIT may be one), its answer and reward."""

    def __init__(self, env: Env, tid: str):
        self.env, self.tid, self.id = env, tid, uuid.uuid4().hex[:12]
        self.tools = env.text_tools(tid)
        self.messages = [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": question(env.task(tid), env.rows[tid])}]
        self.calls, self.turns, self.answer, self.result = 0, 0, None, None
        self.drafts: dict = {}
        self.order: dict = {}  # draft id -> its nodes least needed first (measure_removal each), drafts from it inherit it
        self._inherit = None  # (base, dropped nodes) of the run_circuit call in progress
        self.asked: dict = {}  # (tool, resolved arguments) -> times asked in this episode
        self.log: list = []  # [tool, arguments, seconds, cached]

    def _draft(self, ref) -> dict:
        ref = str(ref).strip()
        if ref not in self.drafts:
            raise ToolError(f"no draft {ref!r} in this episode (drafts: {', '.join(self.drafts) or 'none yet'})")
        return self.drafts[ref]

    def _resolve(self, name: str, args: dict) -> dict:
        """The arguments with draft ids replaced by their circuit JSON (base with add and drop applied)."""
        args = dict(args)
        if name == "run_circuit" and (args.get("base") is not None or args.get("add") or args.get("drop") or args.get("drop_least")):
            if args.get("base") is None:
                raise ToolError("add, drop and drop_least need base, a draft id")
            base = str(args.pop("base")).strip()
            ans = json.loads(json.dumps(self._draft(base)))
            least = args.pop("drop_least", None)
            dropped = list(args.pop("drop", None) or [])
            if least:
                if base not in self.order:
                    raise ToolError(f"drop_least needs measure_removal(draft={base!r}, each=true) first")
                try:
                    dropped += self.order[base][:max(0, int(least))]
                except (TypeError, ValueError):
                    raise ToolError("drop_least must be an integer") from None
            drop = {nd for x in dropped for nd in (components.group_of(str(x)) or [])}
            self._inherit = (base, drop)
            for i in sorted((i for i, (m, c, t) in enumerate(ans["nodes"]) if (m if m != "block" else c, t, c if m != "block" else components.BLOCK) in drop
                             or (m == components.TOK and (components.TOK, t, 0) in drop)), reverse=True):
                ans = components._without_part(ans, i)
            have = {(m, t, c) for m, c, t in ans["nodes"]}
            for x in args.pop("add", None) or []:
                for nd in components.group_of(str(x)) or []:
                    if (nd[0], nd[1], nd[2]) not in have:
                        ans["nodes"].append(["block", nd[0], nd[1]] if nd[2] == components.BLOCK else [nd[0], nd[2], nd[1]])
                        have.add(nd)
            for e in args.pop("edges", None) or []:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    w, r = components.node_of(str(e[0])), ("out" if str(e[1]).strip().lower() == "out" else components.node_of(str(e[1])))
                    idx = {(m, t, c): i for i, (m, c, t) in enumerate(ans["nodes"])}
                    if w in idx and (r == "out" or r in idx):
                        ans["edges"].append([idx[w], "out" if r == "out" else idx[r]])
            args["_json"] = ans
        if name == "compare":
            for k in ("circuit_a", "circuit_b"):
                if isinstance(args.get(k), str) and DRAFT_RE.fullmatch(args[k].strip()):
                    args[k] = {"_json": self._draft(args[k])}
        if name in ("measure_removal", "node_effects") and args.get("draft") is not None:
            args["_json"] = self._draft(args["draft"])
        return args

    def _ordered(self, name: str, args: dict, side: dict, did: str | None) -> None:
        """Keep a draft's least-needed order: the latest node_effects(draft) or measure_removal(draft, each) of it (the
        tool's whole order, side data), or its base's less what it dropped (nodes it added last)."""
        if name in ("measure_removal", "node_effects") and args.get("draft") is not None and side.get("_order") is not None:
            self.order[str(args["draft"]).strip()] = list(side["_order"])
        inh = getattr(self, "_inherit", None)
        if name == "run_circuit" and inh and did and inh[0] in self.order and did not in self.order:
            have = {components.node_text((m, t, c)) for m, c, t in self.drafts[did]["nodes"] if m not in ("block", components.TOK)}
            kept = [x for x in self.order[inh[0]] if components.node_of(x) not in inh[1] and x in have]
            self.order[did] = kept + sorted(have - set(kept))

    def call(self, name: str, args) -> str:
        self.calls += 1
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"arguments are not JSON: {e}"})
        args = args if isinstance(args, dict) else {}
        t0 = time.perf_counter()
        self._inherit = None
        try:
            full = self._resolve(name, args)
        except ToolError as e:
            self.log.append([name, args, 0.0, False])
            return json.dumps({"error": str(e)})
        key = (name, json.dumps(full, sort_keys=True))
        cached = key in self.tools.memo
        text, side = self.tools.call_full(name, full)
        did = None
        if "_json" in side and text.endswith("}"):  # a scored draft: its id
            did = next((k for k, v in self.drafts.items() if v == side["_json"]), None) or f"d{len(self.drafts) + 1}"
            self.drafts[did] = side["_json"]
            text = text[:-1] + f',"draft":"{did}"}}'
        self._ordered(name, args, side, did)
        self._inherit = None
        self.asked[key] = self.asked.get(key, 0) + 1
        if self.asked[key] > 1 and text.endswith("}"):  # the same call again in this episode: the same result
            text = text[:-1] + f',"repeated":{self.asked[key] - 1}}}'
        self.log.append([name, args, round(time.perf_counter() - t0, 3), cached])
        return text

    def step(self, message: dict) -> bool:
        """Take the policy's assistant message: run its tool calls (one tool message each) or, with none, take it as
        the answer. True when the episode is over."""
        self.turns += 1
        calls = message.get("tool_calls") or []
        self.messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls")})
        if not calls:
            self.answer = message.get("content") or ""
            return True
        for c in calls:
            if self.calls >= MAX_CALLS:
                break
            fn = c.get("function", c)
            self.messages.append({"role": "tool", "name": fn["name"], "content": self.call(fn["name"], fn.get("arguments"))})
        left = min(MAX_TURNS - self.turns, MAX_CALLS - self.calls)
        if 0 < left <= 3 and self.messages[-1]["role"] == "tool" and self.messages[-1]["content"].endswith("}"):
            # the budget's end in sight (Qwen3.5-397B refined drafts to the turn cap twice, the cap stated in SYSTEM)
            self.messages[-1]["content"] = self.messages[-1]["content"][:-1] + f',"turns_left":{left},"answer_now":true}}'
        return self.turns >= MAX_TURNS or self.calls >= MAX_CALLS

    def expand(self, text: str) -> str:
        """An answer with a CIRCUIT that is a draft id alone given that draft's Mermaid (components.render's)."""
        m = CIRCUIT_DRAFT.search(text or "")
        if not m or m[2] not in self.drafts:
            return text
        mermaid = components.render({k: v for k, v in self.drafts[m[2]].items() if k != "explanation"})
        return text[:m.start()] + mermaid + text[m.end():]

    def finish(self) -> dict:
        """The episode's reward: components.reward of its answer (a draft id in its CIRCUIT expanded) minus COST per
        call (no answer: V3_INVALID)."""
        t0 = time.perf_counter()
        answer = self.expand(self.answer) if self.answer is not None else None
        r = self.env.score(self.tid, answer) if answer is not None else None
        base = r["reward"] if r is not None else components.V3_INVALID
        self.result = {"task": self.tid, "reward": base - self.env.cost * self.calls, "answer_reward": base, "calls": self.calls,
                       "turns": self.turns, "answered": self.answer is not None, "score_seconds": round(time.perf_counter() - t0, 2),
                       "scores": _brief(r) if r is not None else None, "answer_expanded": answer if answer != self.answer else None}
        return self.result


def _brief(r: dict) -> dict:
    """The reward row's headline fields."""
    keep = ("reward", "valid", "phi", "phi_tau", "g", "faithful", "bits", "bits_start", "compact", "D", "Q", "term_C", "term_F1", "term_N")
    fid = r.get("fidelity") or {}
    return {**{k: r.get(k) for k in keep}, "s_track_pp": fid.get("s_track_pp"), "s_point_c": fid.get("s_point_c"),
            "s_interchange_pp": fid.get("s_interchange_pp"), "c": fid.get("c"), "c_rl": fid.get("c_rl")}


def run_episode(env: Env, tid: str, policy) -> dict:
    """One episode with a local policy(messages, tools) -> assistant message; the result with the episode's messages
    and tool log."""
    ep = env.reset(tid)
    t0 = time.perf_counter()
    while not ep.step(policy(ep.messages, qwen3_tools())):
        pass
    wall = time.perf_counter() - t0
    res = ep.finish()
    return {**res, "episode_seconds": round(wall, 2), "log": ep.log, "messages": ep.messages, "tools": qwen3_tools()}


# ---- the scripted baseline


TOP_RE = re.compile(r"most probable next tokens there: (" + components.QUOTED.pattern + ")")  # the question's top token


def _call(name: str, **args) -> dict:
    return {"name": name, "arguments": args}


class Scripted:
    """The baseline policy on the tools alone, in drafts. influence(); in one turn run_circuit(positions=...) for every
    m, the influential start candidates (every VPD-active subcomponent at the target and the m most influential
    positions, those moving the prediction by at least 1% of the total, at most MAX_POSITIONS; complete node sets),
    keeping the smallest m whose pp reaches FLOOR (else the best pp); its nodes ranked (node_effects: each node's own
    effect on the draft, path-patched and on the text). Then, above components.NODE_CAP, backward elimination to the
    cap: each turn drops the STEP least needed (never below the cap) and re-ranks the result. Then, at or below the cap,
    each turn tries dropping the next chunk, half of it and a quarter of it at once (three run_circuit calls, base and
    drop_least) and keeps the largest drop whose pp and point stay within TOLERANCE of the draft that entered this
    phase, re-ranking the kept draft; after a turn with no drop kept the chunk restarts below the smallest drop tried;
    until the chunk is below 1/MIN_FRACTION of that draft or the turns or calls run out (the episode's MAX_TURNS and
    MAX_CALLS, the answer's turn kept). The answer: CIRCUIT: <the last kept draft>. rank="removal" ranks once by
    measure_removal(draft, each) instead (each node's removal from the real model, not from the draft: the same order
    for every later draft, inherited, never re-measured). accept="reward" (the default) judges by the reward itself on
    the tools' draws (run_circuit families: phi, the bar, compactness): the start is the m with the best phi, and below
    the cap a turn keeps its drop with the highest circuit reward when that beats the current draft's; accept="pp"
    keeps the pp and point tolerance above (FLOOR and TOLERANCE)."""

    FLOOR = 0.9
    TOLERANCE = 0.05
    MAX_POSITIONS = 8
    MIN_FRACTION = 32
    STEP = 0.25  # backward elimination's share of the draft per turn above the cap
    TRIALS = (1, 2, 4)  # a turn's drops: the chunk over these

    def __init__(self, max_turns: int = MAX_TURNS, max_calls: int = MAX_CALLS, rank: str = "effects", accept: str = "reward"):
        self.max_turns, self.max_calls, self.rank, self.accept, self.gen, self.trace = max_turns, max_calls, rank, accept, None, []

    def _run(self, **args) -> dict:
        return _call("run_circuit", **args, **({"families": True} if self.accept == "reward" else {}))

    @staticmethod
    def _value(sc: dict) -> float:
        """A draft's circuit reward on the tools' draws (families), -inf when it has none."""
        v = sc.get("reward")
        return -math.inf if v is None or "error" in sc else float(v)

    def _rank(self, draft: str) -> dict:
        return (_call("node_effects", draft=draft, top=8) if self.rank == "effects" else
                _call("measure_removal", draft=draft, each=True, top=8))

    def __call__(self, messages: list, tools: list) -> dict:
        if self.gen is None:
            self.gen = self.plan(messages)
            return next(self.gen)
        res = []
        for m in reversed(messages):
            if m["role"] != "tool":
                break
            res.append(json.loads(m["content"]))
        return self.gen.send(res[::-1])

    @staticmethod
    def _msg(*calls) -> dict:
        return {"role": "assistant", "content": "", "tool_calls": list(calls)}

    @staticmethod
    def _circuit(nodes: list) -> str:
        lines, _ = components.circuit_lines(sorted(nodes, key=components._place), [], None, complete=True)
        return "\n".join(lines[1:-1])  # without the fence

    def _keep(self, sc: dict, **kw) -> None:
        self.trace.append({**kw, "nodes": sc.get("nodes"), "pp": sc.get("pp"), "point": sc.get("point"), "bits": sc.get("bits"),
                           "phi": sc.get("phi"), "reward": sc.get("reward"), "draft": sc.get("draft")})

    def plan(self, messages: list):
        turns, calls, cap = 1, 1, components.NODE_CAP
        (inf,) = yield self._msg(_call("influence", top=16))
        t = inf["target"]
        share = [(row[0], row[3]) for row in inf["positions"] if row[0] != t]
        cands = [p for p, s in share if s is not None and s >= goal_eval.SHARE_FLOOR][:self.MAX_POSITIONS]
        got = yield self._msg(*[self._run(positions=[t] + cands[:m]) for m in range(len(cands) + 1)])
        turns, calls = turns + 1, calls + len(got)
        for m, sc in enumerate(got):
            self._keep(sc, m=m)
        if self.accept == "reward":  # the start: the best phi on the tools' draws (above the cap it is pruned to it)
            m = max(range(len(got)), key=lambda j: (-math.inf if got[j].get("phi") is None else got[j]["phi"], -j))
        else:
            ok = [m for m, sc in enumerate(got) if (sc.get("pp") or 0) >= self.FLOOR and sc.get("draft")]
            m = ok[0] if ok else max(range(len(got)), key=lambda j: ((got[j].get("pp") or 0), -j))
        score = got[m]
        draft, size = score.get("draft"), score.get("nodes") or 0
        (rk,) = yield self._msg(self._rank(draft))
        turns, calls = turns + 1, calls + 1
        if "error" in rk and self.rank == "effects":  # node_effects failed (e.g. out of memory): the removal order
            self.rank = "removal"
            (rk,) = yield self._msg(self._rank(draft))
            turns, calls = turns + 1, calls + 1
        ranked = "error" not in rk
        if not ranked:  # no order at all: the best start candidate within the cap, unpruned
            fit = [j for j, sc in enumerate(got) if sc.get("draft") and 0 < (sc.get("nodes") or 0) <= cap]
            if fit:
                j = (max(fit, key=lambda j: (-math.inf if got[j].get("phi") is None else got[j]["phi"], -j)) if self.accept == "reward"
                     else max(fit, key=lambda j: ((got[j].get("pp") or 0), -j)))
                score = got[j]
                draft, size = score["draft"], score["nodes"]
                self.trace.append({"fallback": "unranked", "m": j, "nodes": size, "draft": draft})
        # above the cap: backward elimination, re-ranked each step (a step, its ranking and a last drop to the cap
        # leave the answer's turn: Episode.step ends an episode at MAX_TURNS turns or MAX_CALLS calls)
        while ranked and size > cap and turns < self.max_turns - 3 and calls + 3 < self.max_calls:
            n = min(size - cap, max(1, math.ceil(self.STEP * size)))
            (sc,) = yield self._msg(self._run(base=draft, drop_least=n))
            turns, calls = turns + 1, calls + 1
            self._keep(sc, drop=n, phase="cap")
            if "error" in sc or not sc.get("draft"):
                break
            draft, size, score = sc["draft"], sc["nodes"], sc
            if self.rank == "effects":  # a removal order is the draft's whatever the draft: inherited, not re-measured
                (rk,) = yield self._msg(self._rank(draft))
                turns, calls = turns + 1, calls + 1
                ranked = "error" not in rk
        if size > cap and ranked:  # out of turns above the cap: the least needed go at once
            (sc,) = yield self._msg(self._run(base=draft, drop_least=size - cap))
            turns, calls = turns + 1, calls + 1
            self._keep(sc, drop=size - cap, phase="cap_last")
            draft, size, score = sc.get("draft", draft), sc.get("nodes", size), sc  # its order: its base's, inherited
        # at or below the cap: drops within TOLERANCE of the draft entering here
        ref_pp, ref_pt = score.get("pp") or 0.0, score.get("point") or 0.0
        chunk = max(1, size // 2)
        floor = max(1, size // self.MIN_FRACTION)
        while ranked and size > 1 and chunk >= floor and turns < self.max_turns - 3 and calls + len(self.TRIALS) + 2 < self.max_calls:
            sizes = sorted({max(1, min(size - 1, chunk // k)) for k in self.TRIALS}, reverse=True)
            got = yield self._msg(*[self._run(base=draft, drop_least=n) for n in sizes])
            turns, calls = turns + 1, calls + len(sizes)
            kept = None
            if self.accept == "reward":  # the drop with the best circuit reward on the tools' draws, when it beats the draft's
                best = max(range(len(got)), key=lambda j: (self._value(got[j]), sizes[j]))
                if self._value(got[best]) > self._value(score) and got[best].get("draft"):
                    kept = (sizes[best], got[best])
                for n, sc in zip(sizes, got):
                    self._keep(sc, drop=n, kept=kept is not None and sc is kept[1])
            else:
                for n, sc in zip(sizes, got):
                    ok = (sc.get("pp") or 0) >= ref_pp - self.TOLERANCE and (sc.get("point") or 0) >= ref_pt - self.TOLERANCE
                    self._keep(sc, drop=n, kept=ok)
                    if ok and kept is None and sc.get("draft"):
                        kept = (n, sc)
            if kept:
                draft, size, score = kept[1]["draft"], kept[1]["nodes"], kept[1]
                if self.rank == "effects":
                    (rk,) = yield self._msg(self._rank(draft))  # the kept draft's own order
                    turns, calls = turns + 1, calls + 1
                    if "error" in rk:
                        break
            else:
                chunk = min(sizes) // 2
        q = [m for m in messages if m["role"] == "user"][0]["content"]
        n_q, n_e = q.count("\nQuestion "), q.count("\nEdit ")
        top = TOP_RE.search(q)
        lines = [f"CIRCUIT: {draft}", "EXPLANATION: The prediction is computed by these subcomponents at the target and the "
                 "context positions it reads most, every connection among them."]
        for k in range(1, n_q + 1):
            lines += [f"ANSWER {k}:", f"FINAL {k}: falls=no top={top[1] if top else repr('')}"]
        lines += [f"EDIT {k}: " for k in range(1, n_e + 1)]
        yield {"role": "assistant", "content": "\n".join(lines)}


# ---- reports


def _rows(path: str) -> dict:
    return {r["task"]: r for r in map(json.loads, Path(path).read_text().splitlines()) if r}


def _tasks(spec: list[str]) -> list[str]:
    out = []
    for s in spec:
        for part in s.split(","):
            if ":" in part:
                a, b = part.split(":")
                out += [f"text{i}" for i in range(int(a.removeprefix("text")), int(b.removeprefix("text")) + 1)]
            elif part:
                out.append(part)
    if any(9100 <= int(t.removeprefix("text")) <= 9149 for t in out if t.removeprefix("text").isdigit()):
        raise SystemExit("text9100-9149 is the sealed final test set")
    return out


def bench(env: Env, tids: list[str], log=print, score: bool = True) -> dict:
    """Each tool's latency on its first (uncached) call per text, with Text's setup and (score) a scoring call."""
    out = {}
    for tid in tids:
        t0 = time.perf_counter()
        tools = env.text_tools(tid)
        setup = time.perf_counter() - t0
        ep = env.reset(tid)
        log(f"{tid}: setup {setup:.1f} s", flush=True)
        call = ep.call

        def logged(name, args):
            t1 = time.perf_counter()
            res = call(name, args)
            log(f"  {name} {time.perf_counter() - t1:.2f} s {res[:160]}", flush=True)
            return res

        ep.call = logged
        inf = json.loads(ep.call("influence", {"top": 8}))
        t = inf["target"]
        p1 = inf["positions"][0][0] if inf["positions"][0][0] != t else inf["positions"][1][0]
        ep.call("list_active", {"position": t, "top": 24})
        ep.call("list_active", {"position": p1, "top": 400})
        ep.call("attention", {"layer": 2, "position": t})
        acts = json.loads(ep.call("list_active", {"position": t, "top": 400}))["subcomponents"]
        nodes = [x[0] for x in acts]
        ep.call("measure_removal", {"nodes": nodes[:8]})
        ep.call("measure_removal", {"nodes": nodes[:64], "each": True})
        small = Scripted._circuit([components.node_of(x) for x in nodes[:16]])
        big = Scripted._circuit([components.node_of(x) for x in nodes] + [components.node_of(x[0]) for x in
                                 json.loads(ep.call("list_active", {"position": p1, "top": 400}))["subcomponents"]])
        ep.call("run_circuit", {"circuit": small})
        d_big = json.loads(ep.call("run_circuit", {"circuit": big})).get("draft")
        ep.call("run_circuit", {"circuit": small, "changes": [[p1, 13]]})
        ep.call("compare", {"circuit_a": small, "circuit_b": big})
        d_pos = json.loads(ep.call("run_circuit", {"positions": [t, p1]})).get("draft")
        for d in (d_big, d_pos):
            if d:
                ep.call("node_effects", {"draft": d})
                ep.call("run_circuit", {"base": d, "drop_least": 16})
        out[tid] = {"T": tools.T, "setup_s": round(setup, 2), "tools_s": {k: [round(x, 2) for x in v] for k, v in tools.seconds.items()}}
        if score:
            t1 = time.perf_counter()
            tau = env.ge.tau(tools.tx, env.pool, env.score_split)
            tau_s = time.perf_counter() - t1
            t1 = time.perf_counter()
            r = env.score(tid, "CIRCUIT:\n```mermaid\n" + big + "\n```\nEXPLANATION: x")
            out[tid] |= {"tau_s": round(tau_s, 2), "score_s": round(time.perf_counter() - t1, 2), "phi_start": tau.get("phi_start"), "phi": r.get("phi")}
        log(json.dumps({tid: out[tid]}), flush=True)
    return out


def scripted(env: Env, tids: list[str], out: Path, log=print, rank: str = "effects", accept: str = "reward") -> list[dict]:
    """The scripted baseline's episodes on texts, each beside the influential start on the score split (its phi from
    tau, its s_track_pp and bits), one JSON line each."""
    res = []
    for tid in tids:
        pol = Scripted(rank=rank, accept=accept)
        r = run_episode(env, tid, pol)
        tools = env.text_tools(tid)
        tx = tools.tx
        tau = env.ge.tau(tx, env.pool, env.score_split)
        st = env.ge.start_set(tx, env.score_split)
        row = {**r, "rank": rank, "accept": accept, "trace": pol.trace,
               "start": {"phi": tau["phi_start"], "phi_tau": tau["phi_tau"], "bits": tau["bits_start"], "s_track_pp": st["s_track_pp"],
                         "nodes": tau["nodes_start"], "m": tau["m_start"]},
               "answer": r.get("answer_expanded") or r["messages"][-1]["content"]}
        res.append(row)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        free_text(env, tid)  # each text runs once here: TEXTS_KEPT texts' GPU state OOMed a 48 GB card at the 6th (10-13)
        s = r["scores"] or {}
        log(f"{tid}: reward {r['reward']:.3f} ({r['calls']} calls, {r['episode_seconds']:.0f} s + score {r['score_seconds']:.0f} s) "
            f"phi {s.get('phi')} vs start {tau['phi_start']}; pp {s.get('s_track_pp')} vs {st['s_track_pp']}; bits {s.get('bits')} vs {tau['bits_start']}")
    return res


def baselines(env: Env, episodes: Path, out: Path, log=print) -> list[dict]:
    """Each scripted answer (episodes: scripted's output) beside trivial node sets of its size on the score split, all
    scored by the reward's families (GoalEval.fidelity in reward mode): complete node sets for the influential start,
    VPD's first k by causal importance (Native._vpd_order), the start's k most important by causal importance x
    |activation| (the scripted policy's measured pruning left out), k of the start's nodes drawn at random (seed 0),
    and an empty graph declaring no token edges. Description bits are explicit: matching node counts need not match bits. The same batch's start row
    supplies phi_tau and bits_start, without another model pass. Per graph: point KL, phi, family scores/eps and
    canonical QA-free D=0 circuit reward, without tool costs. This excludes answer defects and token truncation;
    the answer node cap still applies. One JSON line per text with scorer and episode provenance."""
    import hashlib

    ge, nat = env.ge, env.ge.nat
    P, S = None, sum(nat.C.values())
    source_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in
                     ("tools_env.py", "components.py", "goal_eval.py", "native.py", "engine.py", "carry.py",
                      "mech.py", "task_data.py") if (HERE / name).exists()}
    vpd_source = HERE.parent / "vpd" / "vpd_model.py"
    if vpd_source.exists():
        source_hashes["vpd/vpd_model.py"] = hashlib.sha256(vpd_source.read_bytes()).hexdigest()
    res = []
    for line in episodes.read_text().splitlines():
        ep = json.loads(line)
        tid = ep["task"]
        tools = env.text_tools(tid)
        tx = tools.tx
        P = nat.positions(tx.targets)
        parsed = components.parse(ep.get("answer_expanded") or ep["answer"])
        g, _ = goal_eval.circuit(parsed["answer"], tx.targets, nat.C, tx.T)
        k = g.count()
        st = ge.start_set(tx, env.score_split)["graph"]
        snodes = sorted(st.node_set())
        key = {nd: float(tx.ci[nd[0]][nd[1], nd[2]]) * abs(float(tx.rec[nd[0]][0, nd[1], nd[2]])) for nd in snodes}
        rng = np.random.default_rng(0)
        # An actual empty answer states no token edges; complete=True would imply them in pp semantics.
        sets = {"scripted": g, "empty": native.Graph(), "start": st,
                "vpd_top_k": native.Graph(nat._vpd_order(tx.ids, tx.targets)[:k], complete=True),
                "start_ci_act_k": native.Graph(sorted(snodes, key=lambda nd: -key[nd])[:k], complete=True),
                "start_random_k": native.Graph([snodes[i] for i in rng.choice(len(snodes), size=min(k, len(snodes)), replace=False)], complete=True)}
        t0 = time.perf_counter()
        rows = ge.fidelity(tx, list(sets.values()), env.pool, mode="reward", split=env.score_split)
        phi_start, phi_tau = goal_eval.phi(rows[list(sets).index("start")], start=rows[list(sets).index("start")])
        bits_start = goal_eval.size_bits(st, P, S, tx.targets, active=tx.vpd)
        row = {"task": tid, "k": k, "seconds": round(time.perf_counter() - t0, 1), "artifact_version": 2,
               "score_split": env.score_split, "source_sha256": source_hashes,
               "episode_sha256": hashlib.sha256(json.dumps(ep, sort_keys=True).encode()).hexdigest(),
               "threshold_source": "same_batch_start", "phi_start": phi_start, "phi_tau": phi_tau,
               "bits_start": bits_start, "gate_slope": components.GATE_SLOPE,
               "score_scope": "canonical QA-free D=0; node cap applied; no answer-defect, truncation or tool costs"}
        for (name, gg), r in zip(sets.items(), rows):
            ph, _ = goal_eval.phi(r)
            bits = goal_eval.size_bits(gg, P, S, tx.targets, active=tx.vpd)
            compact = goal_eval.compact(bits, bits_start)
            gate = None if ph is None or phi_tau is None else 1.0 / (1.0 + 2.0 ** (-components.GATE_SLOPE * (ph - phi_tau)))
            formula_score = (ph or 0.0) + (gate or 0.0) * 0.5 * (compact or 0.0)
            eligible = gg.count() <= components.NODE_CAP
            families = {f: goal_eval.family_score(r, f) for f in goal_eval.PHI_FAMILIES}
            row[name] = {"phi": ph, "nodes": gg.count(), "bits": bits, "compact": compact, "g": gate,
                         "complete": gg.complete, "tok_out": list(getattr(gg, "tok_out", [])),
                         "node_cap_eligible": eligible, "formula_circuit_reward_qa_free_d0": formula_score,
                         "circuit_reward_qa_free_d0": formula_score if eligible else components.V3_INVALID,
                         "families": families, "missing_families": [f for f, value in families.items() if value is None],
                         "eps": {f: r[f"eps_{f}"] for f in goal_eval.PHI_FAMILIES if r.get(f"eps_{f}") is not None},
                         "draw_counts": {f: len(r.get(key) or []) for f, (_, _, key) in goal_eval.PHI_FAMILIES.items() if key},
                         "kl_point": r.get("kl_point"), "kl_empty_live": r.get("kl_empty_live"),
                         "kl_vpd_live": r.get("kl_vpd_live")}
        res.append(row)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"{tid} k={k}: " + "; ".join(f"{n} phi {row[n]['phi']:.3f} pp {row[n]['families']['track_pp']}" if row[n]["phi"] is not None
                                       else f"{n} phi None" for n in sets), flush=True)
    return res


def tokens(rows: dict, tids: list[str], episodes: Path | None, base: str = "Qwen/Qwen3-8B", pool: bool = False, log=print) -> dict:
    """Qwen3 tokens: the one-shot question (the rows' question in Qwen3's chat template; pool: also the v2 question
    with every VPD-active subcomponent at its positions, components.candidate_pool, needing the causal-importance
    network) against an episode's opening (system with the tool schemas and the short question) and, from episodes
    (scripted's output with messages), the whole episode's context at its end."""
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(base)
    def count(msgs, tools=None):  # the chat template's text, tokenized (transformers 5's tokenize=True returns a dict)
        text = tk.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(tk(text, add_special_tokens=False)["input_ids"])
    out = {}
    eps = {}
    if episodes:
        for line in episodes.read_text().splitlines():
            r = json.loads(line)
            eps[r["task"]] = r
    for tid in tids:
        task = json.loads((native.TEXTS / "vpd4l" / f"{tid}.json").read_text())
        one = count([{"role": "user", "content": rows[tid]["question"]}])
        opening = count([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question(task, rows[tid])}], qwen3_tools())
        out[tid] = {"one_shot": one, "opening": opening}
        if pool:
            nat = nat if "nat" in dir() else native.Native()
            out[tid]["one_shot_pool"] = count([{"role": "user", "content": components.question(task, components.candidate_pool(nat, task, rows[tid]))}])
        if tid in eps and eps[tid].get("messages"):
            msgs = [_hf_message(m) for m in eps[tid]["messages"]]
            out[tid]["episode"] = count(msgs, qwen3_tools())
        log(json.dumps({tid: out[tid]}))
    return out


def _hf_message(m: dict) -> dict:
    """A message in the HF chat template's form (tool calls as {"type": "function", "function": {name, arguments}})."""
    if m.get("tool_calls"):
        return {"role": "assistant", "content": m.get("content") or "",
                "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]]}
    return {k: v for k, v in m.items() if k in ("role", "content", "name")}


# ---- HTTP


def serve(env: Env, host: str, port: int) -> None:
    """The environment behind HTTP for a policy sampled elsewhere (tools_tinker.py); one GPU thread does the work.
      POST /reset {"task"}                    -> {"episode", "messages", "tools"}
      POST /call {"episode", "name", "arguments"} -> {"content"}
      POST /submit {"episode", "answer"}      -> the episode's result (reward minus the calls' cost)
      GET /health"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    lock, episodes = threading.Lock(), OrderedDict()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {"episodes": len(episodes), "texts": list(env.tools)}) if self.path == "/health" else self.reply(404, {})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    if self.path == "/reset":
                        ep = env.reset(req["task"])
                        episodes[ep.id] = ep
                        while len(episodes) > 4096:
                            episodes.popitem(last=False)
                        return self.reply(200, {"episode": ep.id, "messages": ep.messages, "tools": qwen3_tools()})
                    ep = episodes[req["episode"]]
                    if self.path == "/call":
                        return self.reply(200, {"content": ep.call(req["name"], req.get("arguments"))})
                    if self.path == "/submit":
                        ep.answer = req.get("answer")
                        ep.turns = req.get("turns", ep.turns)
                        return self.reply(200, {**ep.finish(), "log": ep.log})
                self.reply(404, {"error": "POST /reset, /call or /submit"})
            except Exception as e:  # a bad request answers with its error; the server keeps serving
                self.reply(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):
            pass

    print(f"tools_env: http://{host}:{port} (score split {env.score_split}, tools {env.tool_split})", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("rows")
    b.add_argument("--tasks", nargs="+", default=["text9000"])
    b.add_argument("--score-split", default="eval")
    b.add_argument("--no-score", action="store_true", help="the tools only (no reward call: C is large on the Mac)")
    b.add_argument("--device")
    s = sub.add_parser("scripted")
    s.add_argument("rows")
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--score-split", default="eval")
    s.add_argument("--rank", default="effects", choices=["effects", "removal"], help="the scripted policy's node order")
    s.add_argument("--accept", default="reward", choices=["reward", "pp"], help="the scripted policy's acceptance: the reward on the tools' draws, or pp/point tolerance")
    t = sub.add_parser("tokens")
    t.add_argument("rows")
    t.add_argument("--tasks", nargs="+", required=True)
    t.add_argument("--episodes")
    t.add_argument("--pool", action="store_true", help="also the v2 question listing every VPD-active subcomponent")
    bl = sub.add_parser("baselines")
    bl.add_argument("rows")
    bl.add_argument("--episodes", required=True)
    bl.add_argument("--out", required=True)
    bl.add_argument("--score-split", default="eval")
    v = sub.add_parser("serve")
    v.add_argument("rows")
    v.add_argument("--port", type=int, default=8771)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--score-split", default="search")
    v.add_argument("--device")
    a = ap.parse_args()
    rows = _rows(a.rows)
    if a.cmd == "tokens":
        tokens(rows, _tasks(a.tasks), Path(a.episodes) if a.episodes else None, pool=a.pool)
        return
    env = Env(rows, goal_eval.GoalEval(native.Native(getattr(a, "device", None))), score_split=a.score_split)
    if a.cmd == "bench":
        bench(env, _tasks(a.tasks), score=not a.no_score)
    elif a.cmd == "scripted":
        scripted(env, _tasks(a.tasks), Path(a.out), rank=a.rank, accept=a.accept)
    elif a.cmd == "baselines":
        baselines(env, Path(a.episodes), Path(a.out))
    else:
        serve(env, a.host, a.port)


if __name__ == "__main__":
    main()
