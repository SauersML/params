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
                                           removal per node, goal_eval.GoalEval.real_removal)
  run_circuit(circuit, changes)            a draft CIRCUIT (Mermaid, parsed by components.parse, checked by
                                           goal_eval.circuit): its point score on the text (graph F1 in C's semantics,
                                           GoalEval.sufficiency), its path-patched score on changed inputs
                                           (GoalEval.track_pp: s_track_pp) and its description bits (size_bits); with
                                           changes, the circuit's path-patched prediction on that changed text
                                           (GoalEval.pp_run) beside the model's
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
  tools_env.py serve ROWS.jsonl [--port 8771] [--score-split search]
ROWS: components.py questions rows (each text's evidence: its intervention questions and edit requests).
"""

from __future__ import annotations

import argparse
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
REMOVE_EACH_MAX = 256  # measure_removal(each=True)'s most nodes
COMPARE_DRAWS = goal_eval.CHANGED_DRAWS  # compare's changed inputs (track_pp's draws on the tools' split)
OTHER = {"search": "eval", "eval": "search"}
LN2 = math.log(2)

SYSTEM = ("You explain how a small language model, vpd4l (4 layers, each with attention (q, k, v, o) and an MLP (fc, "
          "down)), computes its next-token prediction on a text, in VPD subcomponents: rank-one slices of its weight "
          "matrices, 38,912 in all, written L<layer>.<kind>#<index>@<position>. You do not know the subcomponents' roles "
          "in advance: measure them with the tools before you answer. Useful order: influence() for the context "
          "positions the prediction reads; list_active() at the target and at those positions (pass next_offset as "
          "offset to read further pages, until next_offset is null); attention() for where "
          "the heads at the target read; measure_removal() for which subcomponents the prediction needs; run_circuit() "
          "to score a draft circuit (point: graph F1 on the text; pp: path-patched agreement with the model on changed "
          "inputs; 1 is as faithful as VPD's whole answer) and compare() to see where two drafts differ. A circuit's "
          "first Mermaid line after flowchart LR may be %% complete: its nodes then claim every connection the model has "
          "among them, and it needs no edge lines. Each tool call costs a little reward: call what you need, then reply "
          "without a tool call. That reply is your answer, in the format the question asks for.")


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
         "offset": {"type": "integer", "minimum": 0, "description": "zero-based start in the ranked list (default 0)"}},
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
                    "model's, and the new top tokens. each=true: one removal per subcomponent instead of all at once.",
     "parameters": {"type": "object", "properties": {
         "nodes": {"type": "array", "items": {"type": "string"}},
         "each": {"type": "boolean"}}, "required": ["nodes"]}},
    {"name": "run_circuit",
     "description": "Score a draft CIRCUIT (the Mermaid flowchart of the answer format): point (its graph F1 on the "
                    "text, 1 = as close to the model as VPD's whole answer, on a log scale from the empty circuit), pp "
                    "(its path-patched agreement with the model on changed inputs, the same scale), its size in bits, "
                    "and the nodes and edges dropped as invalid or stranded. changes: [[position, token], ...] runs the "
                    "circuit path-patched on that changed text instead and gives its top tokens beside the model's.",
     "parameters": {"type": "object", "properties": {
         "circuit": {"type": "string", "description": "the Mermaid flowchart"},
         "changes": {"type": "array", "items": {"type": "array"},
                     "description": "[[position, token], ...], token as quoted in the question, e.g. \"' castle'\", or an id"}},
         "required": ["circuit"]}},
    {"name": "compare",
     "description": "The changed inputs on which two draft circuits' path-patched predictions disagree most: per input "
                    "the change, the model's top token, each draft's top token and KL from the model.",
     "parameters": {"type": "object", "properties": {
         "circuit_a": {"type": "string"}, "circuit_b": {"type": "string"},
         "top": {"type": "integer", "description": "inputs to list (default 5)"}}, "required": ["circuit_a", "circuit_b"]}},
]
TOOL_NAMES = [s["name"] for s in TOOL_SPECS]


def qwen3_tools() -> list[dict]:
    """The tool schemas in Qwen3's (OpenAI's) function format, for its chat template's tools argument."""
    return [{"type": "function", "function": s} for s in TOOL_SPECS]


def _site_of(text: str) -> str:
    """A matrix from L<layer>.<kind> (or goal_eval._site's forms)."""
    t = text.strip()
    t = t[1:] if t[:1] in ("L", "l") else t
    try:
        return goal_eval._site(t)
    except (ValueError, KeyError):
        raise ToolError(f"site {text!r}: write L<layer>.<kind>, kind one of q, k, v, o, fc, down") from None


class Tools:
    """The tools on one text: goal_eval's Text (the model's run, VPD's answer, the probe), shared with the reward, and
    every result kept by tool and arguments."""

    def __init__(self, ge: goal_eval.GoalEval, task: dict, split: str = "search", lens=None, atlas=None):
        self.ge, self.nat, self.task, self.split = ge, ge.nat, task, split
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
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"arguments are not JSON: {e}"})
        args = args or {}
        key = (name, json.dumps(args, sort_keys=True))
        if key in self.memo:
            return self.memo[key]
        if name not in TOOL_NAMES:
            return json.dumps({"error": f"no tool {name!r}; tools: {', '.join(TOOL_NAMES)}"})
        t0 = time.perf_counter()
        try:
            out = getattr(self, name)(**args)
        except (ToolError, TypeError) as e:
            out = {"error": str(e)}
        finally:
            native.free(self.nat.dev)
        self.seconds.setdefault(name, []).append(time.perf_counter() - t0)
        self.memo[key] = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
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
                if not 0 <= nd[2] < self.nat.C[nd[0]]:
                    raise ToolError(f"{x!r}: index outside the matrix's {self.nat.C[nd[0]]} subcomponents")
                self._pos(nd[1])
                out.append(nd)
        if not out:
            raise ToolError("no nodes given")
        return list(dict.fromkeys(out))

    def _graph(self, text: str):
        """(goal_eval's Graph of a draft CIRCUIT, its defects) from Mermaid text (with or without CIRCUIT: and its
        fence)."""
        if not isinstance(text, str) or not text.strip():
            raise ToolError("circuit must be the Mermaid flowchart text")
        body = text if "CIRCUIT" in text.upper() else "CIRCUIT:\n" + text
        p = components.parse(body)
        if not p["answer"]["nodes"]:
            raise ToolError("the circuit declares no node (flowchart LR, then subgraph P<position>[...] blocks declaring "
                            "id[\"L<layer>.<kind>#<index>\"], end, then edges)")
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

    def list_active(self, position, site: str | None = None, top: int = LIST_TOP, offset: int = 0) -> dict:
        p = self._pos(position)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be a non-negative integer")
        top = max(1, min(int(top), LIST_MAX))
        matrix_order = {n: i for i, n in enumerate(self.nat.names)}
        names = [_site_of(site)] if site else self.nat.names
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
        facts = self._facts([nd for nd, _, _ in shown])
        return {"position": p, "token": self._tok(p), "active": len(ranked), "shown": len(shown),
                "total": len(ranked), "offset": offset,
                "next_offset": offset + len(shown) if offset + len(shown) < len(ranked) else None,
                "columns": ["node", "ci", "act", "facts"],
                "subcomponents": [[components.node_text(nd), _r(c, 2), _r(a, 3), facts.get(nd, "")] for nd, c, a in shown]}

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

    def measure_removal(self, nodes, each: bool = False) -> dict:
        nds = self._nodes(nodes)
        tx = self.tx
        if each:
            if len(nds) > REMOVE_EACH_MAX:
                raise ToolError(f"each=true takes at most {REMOVE_EACH_MAX} nodes")
            rows = self.ge.real_removal(tx, [native.Graph([nd], complete=True) for nd in nds])
            got = sorted(zip(nds, [r["kl_real_minus"] for r in rows]), key=lambda x: -x[1])
            return {"columns": ["node", "kl_bits"], "removals": [[components.node_text(nd), _r(k)] for nd, k in got]}
        with torch.no_grad():
            lq = self.nat.without(self.ids, self.targets, nds)
        kl = float(carry.kl(tx.ref, lq)[0])
        return {"removed": len(nds), "kl_bits": _r(kl), "model_top": self._top(tx.ref), "top_after": self._top(lq)}

    def _score(self, g) -> dict:
        ge, tx = self.ge, self.tx
        cg = goal_eval._c_graph(g, self.targets, self.nat.C)
        pt = ge.sufficiency(tx, [cg])[0]
        pp = ge.track_pp(tx, [goal_eval.strand(g, self.targets, self.nat.C)], self.split)[0]
        return {"point": _r(pt["f1_live"]), "point_kl_bits": _r(pt["kl_live"]), "necessity": _r(pt["n_live"]),
                "pp": _r(pp["s_track_pp"]), "pp_kl_bits": _r(pp["kl_track_pp_c"]), "pp_inputs": pp["prompts_track_pp"]}

    def run_circuit(self, circuit: str, changes=None) -> dict:
        g, d, p = self._graph(circuit)
        g = goal_eval.strand(g, self.targets, self.nat.C)
        out = {"nodes": g.count(), "edges": native.count_edges(g.node_set(), self.targets) if g.complete else g.edges(),
               "bits": _r(self._bits(g, p, d)), "dropped_nodes": d["invalid_nodes"] + d["unknown_ids"],
               "dropped_edges": d["invalid_edges"], "stranded_nodes": d["stranded_nodes"], "bad_lines": d["malformed_lines"]}
        if not g.count():
            return {**out, "error": "no valid node left"}
        if changes is None:
            return {**out, **self._score(g)}
        x2 = self._changed(changes)
        ge, tx = self.ge, self.tx
        with torch.no_grad():
            ref2 = self.nat.reference([x2], self.targets)
            lq = ge.pp_run(tx, x2, g)
            vpd = ge.pp_run(tx, x2, native.Graph(masks=tx.vpd)) if tx.vpd is not None else None
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

    def compare(self, circuit_a: str, circuit_b: str, top: int = 5) -> dict:
        ge, tx, nat = self.ge, self.tx, self.nat
        gs = [goal_eval.strand(self._graph(c)[0], self.targets, nat.C) for c in (circuit_a, circuit_b)]
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
            self.tools[tid] = Tools(self.ge, self.task(tid), self.tool_split, self.lens, self.atlas)
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


class Episode:
    """One episode on one text: its messages, its tool calls, its answer and reward."""

    def __init__(self, env: Env, tid: str):
        self.env, self.tid, self.id = env, tid, uuid.uuid4().hex[:12]
        self.tools = env.text_tools(tid)
        self.messages = [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": question(env.task(tid), env.rows[tid])}]
        self.calls, self.turns, self.answer, self.result = 0, 0, None, None
        self.log: list = []  # [tool, arguments, seconds, cached]

    def call(self, name: str, args) -> str:
        self.calls += 1
        cached = (name, json.dumps(args if isinstance(args, dict) else {}, sort_keys=True)) in self.tools.memo
        t0 = time.perf_counter()
        out = self.tools.call(name, args)
        self.log.append([name, args, round(time.perf_counter() - t0, 3), cached])
        return out

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
        return self.turns >= MAX_TURNS or self.calls >= MAX_CALLS

    def finish(self) -> dict:
        """The episode's reward: components.reward of its answer minus COST per call (no answer: V3_INVALID)."""
        t0 = time.perf_counter()
        r = self.env.score(self.tid, self.answer) if self.answer is not None else None
        base = r["reward"] if r is not None else components.V3_INVALID
        self.result = {"task": self.tid, "reward": base - self.env.cost * self.calls, "answer_reward": base, "calls": self.calls,
                       "turns": self.turns, "answered": self.answer is not None, "score_seconds": round(time.perf_counter() - t0, 2),
                       "scores": _brief(r) if r is not None else None}
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
    return {**res, "episode_seconds": round(wall, 2), "log": ep.log, "messages": ep.messages}


# ---- the scripted baseline


TOP_RE = re.compile(r"most probable next tokens there: (" + components.QUOTED.pattern + ")")  # the question's top token


def _call(name: str, **args) -> dict:
    return {"name": name, "arguments": args}


class Scripted:
    """The baseline policy on the tools alone: influence(); list_active() at the target and the positions moving the
    prediction by at least 1% of the total; the influential start, the actives at the target and the m most
    influential positions as a complete node set (%% complete), m the smallest whose run_circuit pp reaches FLOOR;
    then nodes pruned in order of their measured removal (measure_removal each, smallest first): a chunk of half the
    remaining candidates dropped when run_circuit keeps pp and point within TOLERANCE of the start's, else halved, until
    a chunk of one fails or the calls run out. The answer: the pruned node set, complete."""

    FLOOR = 0.9
    TOLERANCE = 0.05
    MAX_POSITIONS = 8

    def __init__(self, budget: int = MAX_CALLS - 4):
        self.budget, self.gen, self.trace = budget, None, []

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

    def plan(self, messages: list):
        calls = 0
        (inf,) = yield self._msg(_call("influence", top=16))
        calls += 1
        t = inf["target"]
        share = [(row[0], row[3]) for row in inf["positions"] if row[0] != t]
        cands = [p for p, s in share if s is not None and s >= goal_eval.SHARE_FLOOR][:self.MAX_POSITIONS]
        lists = yield self._msg(*[_call("list_active", position=p, top=LIST_MAX) for p in [t] + cands])
        calls += len(lists)
        act = {}
        for r in lists:
            act[r["position"]] = [components.node_of(x[0]) for x in r.get("subcomponents", [])]
        start, score = None, None
        for m in range(len(cands) + 1):
            nodes = [nd for p in [t] + cands[:m] for nd in act.get(p, [])]
            (score,) = yield self._msg(_call("run_circuit", circuit=self._circuit(nodes)))
            calls += 1
            start = nodes
            self.trace.append({"m": m, "nodes": len(nodes), "pp": score.get("pp"), "point": score.get("point"), "bits": score.get("bits")})
            if (score.get("pp") or 0) >= self.FLOOR:
                break
        base_pp, base_pt = score.get("pp") or 0.0, score.get("point") or 0.0
        (rem,) = yield self._msg(_call("measure_removal", nodes=[components.node_text(nd) for nd in start[:REMOVE_EACH_MAX]], each=True))
        calls += 1
        kl = {components.node_of(x[0]): x[1] for x in rem.get("removals", [])}
        order = sorted(start, key=lambda nd: kl.get(nd, float("inf")))  # least needed first; unmeasured last
        keep, i, chunk = list(start), 0, max(1, len(order) // 2)
        while i < len(order) and calls < self.budget:
            drop = set(order[i:i + chunk])
            trial = [nd for nd in keep if nd not in drop]
            if not trial:
                chunk = max(1, chunk // 2)
                continue
            (sc,) = yield self._msg(_call("run_circuit", circuit=self._circuit(trial)))
            calls += 1
            ok = (sc.get("pp") or 0) >= base_pp - self.TOLERANCE and (sc.get("point") or 0) >= base_pt - self.TOLERANCE
            self.trace.append({"drop": len(drop), "nodes": len(trial), "pp": sc.get("pp"), "point": sc.get("point"),
                               "bits": sc.get("bits"), "kept": ok})
            if ok:
                keep, i = trial, i + len(drop)
            elif chunk == 1:
                i += 1
            else:
                chunk = max(1, chunk // 2)
        q = [m for m in messages if m["role"] == "user"][0]["content"]
        n_q, n_e = q.count("\nQuestion "), q.count("\nEdit ")
        top = TOP_RE.search(q)
        answer = components.render({"nodes": [[n, c, p] for n, p, c in sorted(keep, key=components._place)], "edges": [],
                                    "complete": True, "explanation": "The prediction is computed by these subcomponents at "
                                    "the target and the context positions it reads most, every connection among them."},
                                   finals={k: (False, top[1] if top else "''") for k in range(1, n_q + 1)},
                                   edits={k: [] for k in range(1, n_e + 1)})
        yield {"role": "assistant", "content": answer}


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


def bench(env: Env, tids: list[str], log=print) -> dict:
    """Each tool's latency on its first (uncached) call per text, with Text's setup and a scoring call."""
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
        ep.call("run_circuit", {"circuit": big})
        ep.call("run_circuit", {"circuit": small, "changes": [[p1, 13]]})
        ep.call("compare", {"circuit_a": small, "circuit_b": big})
        t1 = time.perf_counter()
        tx = tools.tx
        tau = env.ge.tau(tx, env.pool, env.score_split)
        tau_s = time.perf_counter() - t1
        t1 = time.perf_counter()
        r = env.score(tid, "CIRCUIT:\n```mermaid\n" + big + "\n```\nEXPLANATION: x")
        score_s = time.perf_counter() - t1
        out[tid] = {"T": tools.T, "setup_s": round(setup, 2), "tools_s": {k: [round(x, 2) for x in v] for k, v in tools.seconds.items()},
                    "tau_s": round(tau_s, 2), "score_s": round(score_s, 2), "phi_start": tau.get("phi_start"), "phi": r.get("phi")}
        log(json.dumps({tid: out[tid]}), flush=True)
    return out


def scripted(env: Env, tids: list[str], out: Path, log=print) -> list[dict]:
    """The scripted baseline's episodes on texts, each beside the influential start on the score split (its phi from
    tau, its s_track_pp and bits), one JSON line each."""
    res = []
    for tid in tids:
        pol = Scripted()
        r = run_episode(env, tid, pol)
        tools = env.text_tools(tid)
        tx = tools.tx
        tau = env.ge.tau(tx, env.pool, env.score_split)
        st = env.ge.start_set(tx, env.score_split)
        row = {**r, "trace": pol.trace,
               "start": {"phi": tau["phi_start"], "phi_tau": tau["phi_tau"], "bits": tau["bits_start"], "s_track_pp": st["s_track_pp"],
                         "nodes": tau["nodes_start"], "m": tau["m_start"]},
               "answer": r["messages"][-1]["content"]}
        res.append(row)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        s = r["scores"] or {}
        log(f"{tid}: reward {r['reward']:.3f} ({r['calls']} calls, {r['episode_seconds']:.0f} s + score {r['score_seconds']:.0f} s) "
            f"phi {s.get('phi')} vs start {tau['phi_start']}; pp {s.get('s_track_pp')} vs {st['s_track_pp']}; bits {s.get('bits')} vs {tau['bits_start']}")
    return res


def tokens(rows: dict, tids: list[str], episodes: Path | None, base: str = "Qwen/Qwen3-8B", pool: bool = False, log=print) -> dict:
    """Qwen3 tokens: the one-shot question (the rows' question in Qwen3's chat template; pool: also the v2 question
    with every VPD-active subcomponent at its positions, components.candidate_pool, needing the causal-importance
    network) against an episode's opening (system with the tool schemas and the short question) and, from episodes
    (scripted's output with messages), the whole episode's context at its end."""
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(base)
    count = lambda msgs, tools=None: len(tk.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=True,  # noqa: E731
                                                                  enable_thinking=False))
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
    s = sub.add_parser("scripted")
    s.add_argument("rows")
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--score-split", default="eval")
    t = sub.add_parser("tokens")
    t.add_argument("rows")
    t.add_argument("--tasks", nargs="+", required=True)
    t.add_argument("--episodes")
    t.add_argument("--pool", action="store_true", help="also the v2 question listing every VPD-active subcomponent")
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
        bench(env, _tasks(a.tasks))
    elif a.cmd == "scripted":
        scripted(env, _tasks(a.tasks), Path(a.out))
    else:
        serve(env, a.host, a.port)


if __name__ == "__main__":
    main()
