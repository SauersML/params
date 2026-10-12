"""Edit questions (#2951 graph oracle, E3): "how would you make vpd4l do X". Goals come from measurements on a text, an
answer's edits (components.parse()'s "edits") run exactly on the model's real weights through VPD's subcomponents, and
the score says how far the goal was reached, what else changed, and how long the edit is.

Goals (goals(): one text's, drawn by the text's seed; seed + 1 gives a fresh draw):
  flip       make the model's most likely token at the target no longer the most likely
  raise      make a token of rank 2 to 5 there the most likely
  detach     make the prediction not depend on the token at an earlier position (one of goal_eval's F2 positions, whose
             changed token moves the prediction by at least 1% of the text's total)
  behaviour  stop copying, on every text: the copy-mistake set (COPY: confident wrong copies of an earlier continuation,
             Pile val rows 9400-9599) split in a first half for the search (its first COPY_SEARCH) and a second half for
             the score

Edits (Editor): the model's own forward pass with each matrix W as its subcomponents V U plus its remainder W - V U, and
  scale n k     subcomponent n's activation at its position times k; persistent: at every position, the weight edit
                W + (k - 1) v u (v, u its read and write vectors); never the token embedding (that edits the text)
  cut r <- w    reader r's activation from its input without writer w's output, the rest of the input as in the edited
                run: a query, key, value or MLP input reads the residual stream at its position without w's write (then
                its norm); an attention output its layer's attention output without the value's share (the pattern's
                weight from r's position to w's in each head times w's write); an MLP output its pre-activation without
                the MLP input's write (then the nonlinearity). w = tok: the token embedding at its position. r = out: the
                prediction at the target reads the final residual stream without w's write. Persistent: at every pair of
                positions the model connects (mech.connects).
  units         (the baselines' edits, inframove's native units) an attention head's output before the output matrix or
                an MLP neuron after the nonlinearity, at a position or everywhere, times k
Edits of one answer apply together in one run, in layer order: each reader's cuts (all its cut writers removed from its
input at once), then the scales. Without edits the run is the model (within float32 rounding).

Score (EditEval.score_many), in bits, on data the teacher's search never used:
  progress    flip, raise: the drop of the goal's log2 margin (the goal token against the best other token) over the
              margin to cross, in [0, 1]; attained when the token crosses. detach: log2(D0 / D) / log2(1 / (1 - GOAL)) in
              [0, 1], D the target's KL from the edited text to the edited text with the position's token changed (the
              model's own alternatives there, ranks 3 to 5; the search uses 0 to 2), D0 the same without the edit;
              attained at D <= (1 - GOAL) D0. behaviour: each fresh instance's flip progress on its copied token,
              averaged; attained when a share GOAL of the instances no longer predicts the copied token
  success     progress over the best progress known for the goal (best=, the teacher's); progress itself without one
  collateral  the KL of the edited model from the model on what the goal does not ask to change, summed per text: the
              text's other positions and, at the target, the distribution over the tokens the goal does not name
              (renormalized; detach and behaviour name none but the copied token); persistent edits add the mean summed
              KL of HELD Pile texts (rows HELD_ROWS, HELD_LEN tokens), behaviour the mean over fresh instances
  size        description bits: per operation its form (2 bits), each node a choice among the subcomponents (or the
              native units) and, at a position, among the text's positions; a scale other than 0 SCALE_BITS more

  edit_eval.py goals --split heldout --n 20 OUT.jsonl            each text's goals (seed: native.task_seed; --fresh: + 1)
  edit_eval.py teach GOALS.jsonl OUT.jsonl [--teacher DIR ...]   the edit teacher and the baselines on each goal
  edit_eval.py copy OUT.json                                     the persistent copy goal: teacher and baselines
  edit_eval.py tokens                                            HELD_ROWS and COPY's rows from the Pile (TOKENS)
  edit_eval.py table OUT.jsonl [COPY.json]                       the report's means

The edit teacher (teach()) is a measured greedy search over candidate edits from the teacher's circuits (goal_eval JSON
files, TEACHER_DIRS) and VPD's most important subcomponents: each round adds the candidate that moves the goal most
(ties: least collateral), until the goal is attained or no candidate helps; then drops every edit the goal does not
need (least collateral first) and replaces single edits by others that keep the goal with less collateral. A
persistent goal can be reached by breaking the model, so its search (frontier()) adds the candidate with the most
progress per bit of collateral each round and keeps every step: a path of progress against collateral. Every
search number is measured on the search's own data (alternatives ranks 0 to 2, the copy set's first half, SEARCH_ROWS
for collateral). Baselines, scored by the same code: VPD's top-k subcomponents ablated (its causal-importance order,
native.Native._vpd_order; for the copy goal the subcomponents most often active at the first half's targets), and the
same search over native units.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import mech  # noqa: E402
import native  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle/edit"
TOKENS = DATA / "tokens.npz"  # the Pile rows below, as uint16 (edit_eval.py tokens)
COPY = Path.home() / "mpd-data/graph_oracle/headfix_fresh.jsonl"  # 192 confident copy mistakes (row, t, source, wrong, ...)
COPY_ROW0 = 9400  # the copy set's first Pile val row
HELD_ROWS, HELD_LEN = (9600, 9616), 256  # persistent edits' collateral texts (scores)
SEARCH_ROWS = (9616, 9624)  # the search's own collateral texts
TEACHER_DIRS = (Path.home() / "mpd-data/graph_oracle/teacher/joint_heldout", Path.home() / "mpd-data/graph_oracle/teacher/joint_train")
GOAL = 0.9  # goal_eval.GOAL: detach removes this share of the dependence; behaviour flips this share of the instances
SCORE_RANKS, SEARCH_RANKS = (3, 4, 5), (0, 1, 2)  # detach's alternative tokens (carry.alternatives' ranks)
RAISE_RANKS = (1, 2, 3, 4)  # raise's token: the model's second to fifth most likely at the target (0-based ranks)
SCALE_BITS = 16  # a scale factor other than 0, as a half-precision number
FORM_BITS = 2  # an operation's form: cut or scale, local or persistent
CHUNK = 16  # edited runs of 512 positions per batch (more when they compute fewer positions)
MAX_ROWS = 1024  # edited runs per batch at most
BLOCK = 256  # edits scored at once
COPY_SEARCH = 32  # the copy search's instances (the first of the copy set's first half)
POSITIONS = 1024  # positions per block of vocabulary-sized arrays
MAX_OPS = 16  # the search's largest edit
SCREEN = 256  # the search's candidates after its first round: the most helpful of that round
NODE_SCALES = (0.0, -1.0, 2.0)  # the search's scale factors for a node
LN2 = math.log(2)


# ---- edits


def _is_tok(nd) -> bool:
    return nd != "out" and nd[0] == "tok"


def unit_count(nat) -> dict:
    """The native units per kind: heads (layers x heads) and MLP neurons (layers x the MLP's width)."""
    tg = nat.target
    width = tg.site(native.site_name(0, "c_fc")).W.shape[0]
    return {"head": tg.n_layer * tg.n_head, "neuron": tg.n_layer * width}


def size_bits(nat, ops: list[dict], T: int) -> float:
    """An edit's description bits (module docstring)."""
    subs = math.log2(sum(nat.C.values()))
    units = unit_count(nat)
    pos = math.log2(T)

    def node(nd):
        if nd == "out":
            return 0.0
        if _is_tok(nd):
            return pos
        return subs + (pos if nd[1] is not None else 0.0)

    bits = 0.0
    for o in ops:
        bits += FORM_BITS
        if o["op"] == "cut":
            bits += node(o["reader"]) + node(o["writer"])
        elif o["op"] == "scale":
            bits += node(o["node"]) + 1 + (SCALE_BITS if o["k"] != 0 else 0)
        elif o["op"] == "unit":
            bits += math.log2(units[o["kind"]]) + (pos if o["position"] is not None else 0.0) + 1 + (SCALE_BITS if o["k"] != 0 else 0)
    return bits


def check(ops: list[dict], T: int, targets: list[int], behaviour: bool = False) -> str | None:
    """Why an edit is invalid, or None: no operation, an operation that names no node, a cut the model does not connect,
    a node after the target or outside the text, a scaled token embedding (an edit of the text, not of the model), a
    local operation for a behaviour goal."""
    if not ops:
        return "no operation"
    last = max(targets)
    for o in ops:
        if o["op"] not in ("cut", "scale", "unit"):
            return f"invalid operation {o.get('text', '')!r}"
        if behaviour and not o.get("persistent", o.get("position", 0) is None):
            return "a local operation for a behaviour goal"
        nds = [o["node"]] if o["op"] == "scale" else [o["reader"], o["writer"]] if o["op"] == "cut" else []
        for nd in nds:
            if nd != "out" and nd[1] is not None and not 0 <= nd[1] <= min(last, T - 1):
                return f"node position {nd[1]} outside the text before the target"
        if o["op"] == "unit" and o["position"] is not None and not 0 <= o["position"] <= last:
            return f"unit position {o['position']} outside the text before the target"
        if o["op"] == "scale" and _is_tok(o["node"]):
            return "a scaled token embedding (the input is not edited; its connections can be cut)"
        if o["op"] == "cut" and not _connected(o["writer"], o["reader"], targets, o.get("persistent", False)):
            return "a cut the model does not connect"
    return None


def _connected(w, r, targets: list[int], persistent: bool) -> bool:
    """Whether the model connects writer w to reader r (out: the prediction) at the stated positions, or (persistent)
    at some pair of positions."""
    if w == "out" or _is_tok(r):
        return False
    if _is_tok(w):
        if r == "out":
            return persistent or w[1] in targets
        _, rk = native._layer_kind(r)
        return rk in mech.RESID_READERS and (persistent or w[1] == r[1])
    if persistent:
        wl, wk = native._layer_kind(w)
        if r == "out":
            return wk in mech.RESID_WRITERS
        rl, rk = native._layer_kind(r)
        return mech.connects(wl, wk, 0, rl, rk, 0, [0])
    return native.connects(w, None if r == "out" else r, targets)


class Editor:
    """vpd4l's forward pass with edits (module docstring), batched: row b of ids runs with edit list b. With start > 0
    (every row the same sequence, no edit before start) only the positions from start are computed, the earlier
    positions' keys, values and activations read from the unedited run (record(), kept for the last few sequences):
    a causal model changes nothing before its first edit."""

    def __init__(self, nat, keep: int = 4):
        self.nat, self.tg = nat, nat.target
        self.H, self.hd = self.tg.n_head, self.tg.hd
        self.keep = keep
        self._records: dict = {}

    @staticmethod
    def start(ops: list[dict]) -> int | None:
        """The first position an edit changes (None: a persistent edit, every position)."""
        first = []
        for o in ops:
            if o["op"] == "unit":
                t = o["position"]
            elif o["op"] == "scale":
                t = o["node"][1]
            else:
                r, w = o["reader"], o["writer"]
                t = w[1] if r == "out" else r[1]
                if w[1] is None:
                    t = None
            if t is None:
                return None
            first.append(t)
        return min(first) if first else None

    def _compile(self, edits: list[list[dict]], T: int) -> dict:
        """Per matrix: local reader entries {(b, t, c): [writer, ...]}, persistent attention cuts [(b, c, value
        matrix, c_v)], scales [(b, t or None, c, k)]; the prediction's cuts [(b, t, writer)];
        unit scales per (layer, kind) [(b, t or None, index, k)]; the matrices whose edited activations the cuts read."""
        cuts, pcuts, scales, outs, units, writers = {}, {}, {}, [], {}, set()
        for b, ops in enumerate(edits):
            for o in ops:
                if o["op"] == "scale":
                    n, t, c = o["node"]
                    scales.setdefault(n, []).append((b, t, c, float(o["k"])))
                elif o["op"] == "unit":
                    units.setdefault((o["layer"], o["kind"]), []).append((b, o["position"], o["index"], float(o["k"])))
                elif o["op"] == "cut":
                    w, r = tuple(o["writer"]), o["reader"] if o["reader"] == "out" else tuple(o["reader"])
                    if not _is_tok(w):
                        writers.add(w[0])
                    persistent = w[1] is None or (r != "out" and r[1] is None)
                    if r == "out":
                        outs += [(b, t, (w[0], t, w[2])) for t in range(T)] if persistent else [(b, w[1], w)]
                        continue
                    if not persistent:
                        cuts.setdefault(r[0], {}).setdefault((b, r[1], r[2]), []).append(w)
                    elif native._layer_kind(r)[1] == "o_proj":
                        pcuts.setdefault(r[0], []).append((b, r[2], w[0], w[2]))
                    else:  # a residual or MLP connection: the same position, at every position
                        for t in range(T):
                            cuts.setdefault(r[0], {}).setdefault((b, t, r[2]), []).append((w[0], t, w[2]))
        return {"cuts": cuts, "pcuts": pcuts, "scales": scales, "outs": outs, "units": units, "writers": writers}

    def _write(self, w, b: int, acts: dict, ids: torch.Tensor, s: int, pre: dict | None) -> torch.Tensor:
        """Writer w's output vector [d] in row b (acts: the edited activations at positions s..; pre: the unedited
        record, before s)."""
        if _is_tok(w):
            return self.tg.wte[ids[b, w[1]]]
        a = pre["acts"][w[0]][w[1], w[2]] if w[1] < s else acts[w[0]][b, w[1] - s, w[2]]
        return a * self.tg.site(w[0]).U[w[2]]

    def _edit(self, n: str, a: torch.Tensor, plan: dict, ctx: dict, acts: dict, ids: torch.Tensor, s: int, pre: dict | None) -> torch.Tensor:
        """Matrix n's activations [B, L, C] (positions s..) after its cuts and scales."""
        import vpd_model

        tg, dev = self.tg, a.device
        kind = native._name_layer_kind(n)[1]
        V = tg.site(n).V
        entries = plan["cuts"].get(n)
        if entries or n in plan["pcuts"] or n in plan["scales"]:
            a = a.clone()
        if entries:
            readers = list(entries)
            removed = torch.zeros(len(readers), V.shape[0], device=dev, dtype=a.dtype)
            for i, (b, t, c) in enumerate(readers):  # few readers per run: small vector operations
                for w in entries[(b, t, c)]:
                    vec = self._write(w, b, acts, ids, s, pre)
                    if kind == "o_proj":
                        vec = ctx["P"][b, :, t - s, w[1]].repeat_interleave(self.hd) * vec
                    removed[i] += vec
            rb, rt, rc = (torch.tensor(x, device=dev) for x in zip(*readers))
            rt = rt - s
            if kind in ("q_proj", "k_proj", "v_proj", "c_fc"):
                inp = vpd_model.rms(ctx["resid"][rb, rt] - removed, ctx["norm"], tg.eps)
            elif kind == "o_proj":
                inp = ctx["att"][rb, rt] - removed
            else:
                inp = vpd_model.gelu_tanh(ctx["pre"][rb, rt] - removed)
            a[rb, rt, rc] = (inp * V.T[rc]).sum(-1)
        for b, c, vn, cv in plan["pcuts"].get(n, []):  # a value's share in an attention output, at every position (s = 0)
            reach = ctx["P"][b] @ acts[vn][b, :, cv]  # [H, T]: per head, the pattern-weighted value activation
            Uv, Vo = tg.site(vn).U[cv].view(self.H, self.hd), V[:, c].view(self.H, self.hd)
            a[b, :, c] -= ((Uv * Vo).sum(-1)[:, None] * reach).sum(0)
        for b, t, c, k in plan["scales"].get(n, []):
            if t is None:
                a[b, :, c] *= k
            else:
                a[b, t - s, c] *= k
        return a

    def _units(self, x: torch.Tensor, layer: int, kind: str, plan: dict, s: int) -> torch.Tensor:
        """A head's output (kind head, x the attention output [B, L, H hd]) or a neuron (kind neuron, x the MLP's
        activation) times k, at a position or everywhere."""
        lst = plan["units"].get((layer, kind))
        if not lst:
            return x
        x = x.clone()
        for b, t, j, k in lst:
            cols = slice(j * self.hd, (j + 1) * self.hd) if kind == "head" else slice(j, j + 1)
            if t is None:
                x[b, :, cols] *= k
            else:
                x[b, t - s, cols] *= k
        return x

    def record(self, ids: list[int]) -> dict:
        """The unedited run of one sequence: per layer the residual stream entering it, its rotated keys and values
        [H, T, hd]; every matrix's activations [T, C]."""
        key = tuple(ids)
        if key not in self._records:
            if len(self._records) >= self.keep:
                self._records.pop(next(iter(self._records)))
            rec = {"x": [], "k": [], "v": [], "acts": {}}
            self.hidden(torch.tensor([ids], device=self.nat.dev), [[]], rec=rec)
            self._records[key] = rec
        return self._records[key]

    @torch.no_grad()
    def hidden(self, ids: torch.Tensor, edits: list[list[dict]], start: int = 0, rec: dict | None = None) -> torch.Tensor:
        """The final normed residual stream [B, T - start, d] of ids [B, T] at positions start.. with row b's edits
        (the logits: it times wte). rec: filled with the run's record (one unedited row)."""
        import vpd_model

        tg = self.tg
        B, T = ids.shape
        s, L = start, T - start
        plan = self._compile(edits, T)
        pre_rec = self.record(ids[0].tolist()) if s else None
        acts = {}
        x = tg.wte[ids[:, s:]]
        mask = torch.ones(L, T, dtype=torch.bool, device=ids.device).tril(s)

        def site(n, a, h):
            st = tg.site(n)
            return a @ st.U + h @ st.delta_T

        def keep(n, a):  # the edited activations the cuts read (and the record)
            if rec is not None:
                rec["acts"][n] = a[0]
            if n in plan["writers"]:
                acts[n] = a

        def edit(n, a, ctx):
            a = self._edit(n, a, plan, ctx, acts, ids, s, pre_rec)
            keep(n, a)
            return a

        for i in range(tg.n_layer):
            nm = {k: native.site_name(i, k) for k in vpd_model.KINDS}
            if rec is not None:
                rec["x"].append(x[0])
            h = vpd_model.rms(x, tg.norms[2 * i], tg.eps)
            ctx = {"resid": x, "norm": tg.norms[2 * i]}
            out = {}
            for k in ("q_proj", "k_proj", "v_proj"):
                a = edit(nm[k], h @ tg.site(nm[k]).V, ctx)
                out[k] = site(nm[k], a, h).view(B, L, self.H, self.hd).transpose(1, 2)
            q, k, v = tg._rope_from(out["q_proj"], s, T), tg._rope_from(out["k_proj"], s, T), out["v_proj"]
            if rec is not None:
                rec["k"].append(k[0])
                rec["v"].append(v[0])
            if s:  # the earlier positions' keys and values are shared by every row: read without copies
                fold = lambda z: z.transpose(0, 1).reshape(self.H, B * L, -1)  # noqa: E731  [B, H, L, x] -> [H, B L, x]
                unfold = lambda z: z.view(self.H, B, L, -1).transpose(0, 1)  # noqa: E731
                kp, vp = pre_rec["k"][i][:, :s], pre_rec["v"][i][:, :s]  # [H, s, hd]
                z = torch.cat([unfold(fold(q) @ kp.transpose(-1, -2)), q @ k.transpose(-1, -2)], -1)
                P = (z / math.sqrt(self.hd)).masked_fill(~mask, float("-inf")).softmax(-1)
                att = unfold(fold(P[..., :s].contiguous()) @ vp) + P[..., s:] @ v
            else:
                P = ((q @ k.transpose(-1, -2)) / math.sqrt(self.hd)).masked_fill(~mask, float("-inf")).softmax(-1)
                att = P @ v
            y = self._units(att.transpose(1, 2).reshape(B, L, -1), i, "head", plan, s)
            a = edit(nm["o_proj"], y @ tg.site(nm["o_proj"]).V, {"att": y, "P": P})
            x = x + site(nm["o_proj"], a, y)
            h = vpd_model.rms(x, tg.norms[2 * i + 1], tg.eps)
            a = edit(nm["c_fc"], h @ tg.site(nm["c_fc"]).V, {"resid": x, "norm": tg.norms[2 * i + 1]})
            pre = site(nm["c_fc"], a, h)
            g = self._units(vpd_model.gelu_tanh(pre), i, "neuron", plan, s)
            a = edit(nm["down_proj"], g @ tg.site(nm["down_proj"]).V, {"pre": pre})
            x = x + site(nm["down_proj"], a, g)
        if plan["outs"]:
            x = x.clone()
            for b, t, w in plan["outs"]:
                x[b, t - s] -= self._write(w if not _is_tok(w) else ("tok", t, 0), b, acts, ids, s, pre_rec)
        return vpd_model.rms(x, tg.ln_f, tg.eps)

    def logp(self, ids: torch.Tensor, edits: list[list[dict]], start: int = 0) -> torch.Tensor:
        """log q (natural) [B, T - start, V]."""
        return torch.log_softmax((self.hidden(ids, edits, start) @ self.tg.wte.T).float(), -1)


# ---- goals


def _margin(lq: torch.Tensor, tok: int) -> torch.Tensor:
    """log2 q(tok) - max over the other tokens of log2 q, per row of lq [..., V]."""
    other = lq.clone()
    other[..., tok] = -float("inf")
    return (lq[..., tok] - other.max(-1).values) / LN2


def goals(ee: "EditEval", task: dict, seed: int | None = None) -> list[dict]:
    """A text's goals: flip, raise (a rank drawn from RAISE_RANKS) and detach (a position drawn from goal_eval's F2
    positions before the target; none when there is none), drawn by seed (native.task_seed(task id) when None; + 1 for
    a fresh draw)."""
    import components

    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    t = max(targets)
    seed = native.task_seed(task["id"]) if seed is None else seed
    rng = random.Random(seed)
    lp = ee.reference(ids)[t]
    order = lp.argsort(descending=True)[: max(RAISE_RANKS) + 1].tolist()
    strings = components.token_strings(task["model"], order)
    out = [{"kind": "flip", "token": strings[0], "token_id": order[0], "target": t}]
    r = rng.choice(RAISE_RANKS)
    out.append({"kind": "raise", "token": strings[r], "token_id": order[r], "rank": r + 1, "target": t})
    tx = ee.ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    positions = [q for q, _ in tx.f2 if q not in targets]
    if positions:
        out.append({"kind": "detach", "position": rng.choice(positions), "target": t})
    return out


def copy_goal() -> dict:
    return {"kind": "behaviour", "name": "copy", "persistent": True,
            "question": "make the model stop copying a token that followed an earlier occurrence of the current token "
                        "when the text continues differently (persistent: on every text, not only this one)."}


# ---- texts


def _tokens() -> dict:
    with np.load(TOKENS) as z:
        return {k: z[k].astype(np.int64) for k in z.files}


def build_tokens() -> None:
    """TOKENS: the Pile val rows of the copy set, HELD_ROWS and SEARCH_ROWS."""
    import vpd_model

    DATA.mkdir(parents=True, exist_ok=True)
    copy = vpd_model.val_tokens(200, 513, COPY_ROW0).numpy()
    held = vpd_model.val_tokens(SEARCH_ROWS[1] - HELD_ROWS[0], 513, HELD_ROWS[0]).numpy()[:, :HELD_LEN]
    np.savez(TOKENS, copy=copy.astype(np.uint16), held=held[: HELD_ROWS[1] - HELD_ROWS[0]].astype(np.uint16),
             search=held[HELD_ROWS[1] - HELD_ROWS[0]:].astype(np.uint16))
    print(f"wrote {TOKENS}: copy {copy.shape}, held {held.shape}")


def copy_instances() -> tuple[list[dict], list[dict]]:
    """The copy set's (first half, second half): each {"row", "ids" (the text to the target), "t", "wrong", "actual"}."""
    rows = _tokens()["copy"]
    cases = [json.loads(x) for x in COPY.read_text().splitlines() if x.strip()]
    out = [{"row": c["row"], "ids": rows[c["row"] - COPY_ROW0][: c["t"] + 1].tolist(), "t": c["t"], "wrong": c["wrong"],
            "actual": c["actual"]} for c in cases]
    h = len(out) // 2
    return out[:h], out[h:]


# ---- the score


class EditEval:
    """Scores edits on goals (module docstring). best: goal key (key()) -> the best progress known (success's
    scale)."""

    def __init__(self, nat=None, best: dict | None = None, chunk: int = CHUNK):
        import goal_eval

        self.nat = nat or native.Native()
        self.ge = goal_eval.GoalEval(self.nat)
        self.ed = Editor(self.nat, keep=8)
        self.best = best or {}
        self.chunk = chunk
        self._refs: dict = {}
        self._alts: dict = {}
        self._held: dict = {}

    def reference(self, ids: list[int]) -> torch.Tensor:
        """log q (natural) [T, V] of the unedited run (the Editor's own arithmetic), kept for the last few texts."""
        key = tuple(ids)
        if key not in self._refs:
            if len(self._refs) >= 8:
                self._refs.pop(next(iter(self._refs)))
            self._refs[key] = self.ed.logp(torch.tensor([ids], device=self.nat.dev), [[]])[0]
        return self._refs[key]

    def alternatives(self, ids: list[int], p: int, ranks) -> list[list[int]]:
        import carry

        key = (tuple(ids), p, tuple(ranks))
        if key not in self._alts:
            self._alts[key] = carry.alternatives(self.nat, ids, p, ranks=ranks)
        return self._alts[key]

    def held(self, which: str) -> list[list[int]]:
        if which not in self._held:
            self._held[which] = _tokens()[which].tolist()
        return self._held[which]

    @staticmethod
    def key(task_id: str, goal: dict) -> str:
        return f"{task_id}:{goal['kind']}:{goal.get('token_id', goal.get('position', goal.get('name')))}"

    # runs

    def _runs(self, seqs: list[list[int]], edits: list[list[dict]], fn, ref: bool = False) -> list:
        """fn(log q [m, T - s, V] at positions s.. of m rows, their indices, s, log p [T - s, V] of the unedited run
        there or None) -> one result per row, over (sequence, edit) pairs: the rows of one sequence batched, each batch
        from the first position its edits change (Editor.start), its vocabulary arrays in blocks of at most POSITIONS
        positions; ref: the unedited run computed with each batch (nothing kept between calls). fn keeps no view of
        the vocabulary-sized arrays it is given."""
        out = [None] * len(seqs)
        by = {}
        for i, (sq, ops) in enumerate(zip(seqs, edits)):
            st = Editor.start(ops) if ops else None
            by.setdefault(tuple(sq), []).append((0 if st is None else st, i))
        wte = self.nat.target.wte
        for sq, rows in by.items():
            rows.sort()
            ids = torch.tensor([sq], device=self.nat.dev)
            r0 = 0
            while r0 < len(rows):  # about chunk x 512 positions per batch: more rows when they start later
                s = rows[r0][0]
                part = rows[r0:r0 + max(1, min(MAX_ROWS, self.chunk * 512 // max(1, len(sq) - s)))]
                r0 += len(part)
                eds = [edits[i] for _, i in part] + ([[]] if ref else [])
                hid = self.ed.hidden(ids.expand(len(eds), -1), eds, start=s)
                lp = torch.log_softmax((hid[-1] @ wte.T).float(), -1) if ref else None
                m = max(1, POSITIONS // hid.shape[1])
                for j0 in range(0, len(part), m):
                    idx = [i for _, i in part[j0:j0 + m]]
                    for i, r in zip(idx, fn(torch.log_softmax((hid[j0:j0 + len(idx)] @ wte.T).float(), -1), idx, s, lp)):
                        out[i] = r
            native.free(self.nat.dev)
        return out

    @staticmethod
    def _kl(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
        """KL(p || q) in bits per row [...]."""
        return (lp.exp() * (lp - lq)).sum(-1) / LN2

    @staticmethod
    def _kept_kl(lp: torch.Tensor, lq: torch.Tensor, named: list[int]) -> torch.Tensor:
        """KL in bits over the tokens not named, both distributions renormalized over them, per row of lq [..., V]
        (lp [V] or the same shape)."""
        if not named:
            return EditEval._kl(lp, lq)
        keep = torch.ones(lq.shape[-1], dtype=torch.bool, device=lq.device)
        keep[named] = False
        a = torch.log_softmax(lp.masked_fill(~keep, -float("inf")), -1)
        b = torch.log_softmax(lq.masked_fill(~keep, -float("inf")), -1)
        return (a.exp() * (a - b)).masked_fill(~keep, 0).sum(-1) / LN2

    def _text_terms(self, ids: list[int], targets: list[int], named: list[int], edits: list[list[dict]], tok: int | None,
                    keep: bool = False) -> list:
        """Per edit on the text: (collateral summed over the text, the target's _margin for tok or None, and with keep
        its log q at the target [V])."""
        t = max(targets)

        def fn(lq, idx, s, lp):
            kl = self._kl(lp, lq)  # [m, L]
            tt = [x - s for x in targets]
            coll = (kl.sum(1) - kl[:, tt].sum(1) + sum(self._kept_kl(lp[x], lq[:, x], named) for x in tt)).tolist()
            ms = _margin(lq[:, t - s], tok).tolist() if tok is not None else [None] * len(coll)
            return list(zip(coll, ms, list(lq[:, t - s].clone()) if keep else [None] * len(coll)))
        return self._runs([ids] * len(edits), edits, fn, ref=True)

    def _held_terms(self, which: str, edits: list[list[dict]]) -> list[float]:
        """Per edit, the mean over the held texts of each text's summed KL (0 for an edit without a persistent op)."""
        out = [0.0] * len(edits)
        todo = [b for b, ops in enumerate(edits) if any(_persistent(o) for o in ops)]
        if not todo:
            return out
        texts = self.held(which)
        seqs, eds, owner = [], [], []
        for b in todo:
            for j, sq in enumerate(texts):
                seqs.append(sq)
                eds.append(edits[b])
                owner.append((b, j))
        kls = self._runs(seqs, eds, lambda lq, idx, s, lp: self._kl(lp, lq).sum(1).tolist(), ref=True)
        for (b, _), k in zip(owner, kls):
            out[b] += k / len(texts)
        return out

    def _dependence(self, ids: list[int], alts: list[list[int]], t: int, edits: list[list[dict]], text: list) -> list[float]:
        """Per edit, the target's mean KL in bits from the edited text (text: its log q at the target, per edit) to the
        edited text with the position changed."""
        seqs, eds = [], []
        for ops in edits:
            for sq in alts:
                seqs.append(sq)
                eds.append(ops)
        lq = self._runs(seqs, eds, lambda lq, idx, s, lp: list(lq[:, t - s].clone()))
        k = len(alts)
        p = torch.stack(text)  # [n, V]
        return (self._kl(p[:, None], torch.stack(lq).view(len(edits), k, -1)).mean(1)).tolist()

    # scores

    def score_many(self, task: dict, goal: dict, edits: list[list[dict]], search: bool = False) -> list[dict]:
        """Per edit (components.parse()'s ops), the row {"valid", "why", "attained", "progress", "success",
        "collateral", "size", ...}. search: the teacher's numbers (alternatives SEARCH_RANKS, collateral texts
        SEARCH_ROWS, the copy set's first half) in place of the score's."""
        if len(edits) > BLOCK:  # bounded memory: the target distributions kept per block
            return [r for b0 in range(0, len(edits), BLOCK) for r in self.score_many(task, goal, edits[b0:b0 + BLOCK], search)]
        if goal["kind"] == "behaviour":
            return self._score_copy(goal, edits, search)
        p = task["prompts"][0]
        ids, targets = p["token_ids"], p["target_positions"]
        T = len(ids)
        rows = [{"valid": False, "why": check(ops, T, targets), "size": size_bits(self.nat, ops, T)} for ops in edits]
        ok = [b for b, r in enumerate(rows) if r["why"] is None]
        if not ok:
            return rows
        eds = [edits[b] for b in ok]
        lp = self.reference(ids)
        t = max(targets)
        named = [goal["token_id"]] if goal["kind"] in ("flip", "raise") else []
        terms = self._text_terms(ids, targets, named, eds, goal.get("token_id") if goal["kind"] in ("flip", "raise") else None,
                                 keep=goal["kind"] == "detach")
        held = self._held_terms("search" if search else "held", eds)
        if goal["kind"] in ("flip", "raise"):
            sign = 1.0 if goal["kind"] == "flip" else -1.0
            m0 = float(sign * _margin(lp[t], goal["token_id"]))
            prog = []
            for _, m, _ in terms:
                m = sign * m
                prog.append((min(1.0, max(0.0, (m0 - m) / m0)) if m0 > 0 else 1.0, m < 0, m))
        else:  # detach
            alts = self.alternatives(ids, goal["position"], SEARCH_RANKS if search else SCORE_RANKS)
            d0 = self._dependence(ids, alts, t, [[]], [lp[t]])[0]
            scale = math.log2(1 / (1 - GOAL))
            prog = [(min(1.0, max(0.0, math.log2(d0 / max(d, 1e-12)) / scale)) if d0 > 0 else 1.0, d <= (1 - GOAL) * d0, d)
                    for d in self._dependence(ids, alts, t, eds, [x[2] for x in terms])]
        key = self.key(task["id"], goal)
        for b, (coll, _, _), h, (g, att, raw) in zip(ok, terms, held, prog):
            best = max(self.best.get(key, 0.0), g)
            rows[b].update({"valid": True, "attained": bool(att), "progress": g, "success": g / best if best > 0 else 0.0,
                            "collateral": coll + h, "collateral_text": coll, "collateral_held": h,
                            ("dependence" if goal["kind"] == "detach" else "margin"): raw})
        return rows

    def score(self, task: dict, goal: dict, ops: list[dict]) -> dict:
        return self.score_many(task, goal, [ops])[0]

    def _score_copy(self, goal: dict, edits: list[list[dict]], search: bool) -> list[dict]:
        """The behaviour goal on the copy set's second half (search: its first half): per instance the flip progress on
        the copied token; collateral = the held texts' mean summed KL + the instances' mean summed kept KL."""
        first, second = copy_instances()
        half = first[:COPY_SEARCH] if search else second
        T = max(len(c["ids"]) for c in half)
        rows = [{"valid": False, "why": check(ops, T, [T - 1], behaviour=True), "size": size_bits(self.nat, ops, T)} for ops in edits]
        ok = [b for b, r in enumerate(rows) if r["why"] is None]
        if not ok:
            return rows
        eds = [edits[b] for b in ok]
        seqs, es, owner = [], [], []
        for j, ops in enumerate(eds):
            for i, c in enumerate(half):
                seqs.append(c["ids"])
                es.append(ops)
                owner.append((j, i))

        def fn(lq, idx, s, lp):  # the rows of one batch share their instance (and s = 0: persistent edits)
            c = half[owner[idx[0]][1]]
            kl = self._kl(lp, lq)  # [m, L]
            t, w = c["t"], c["wrong"]
            m0, ms = float(_margin(lp[t], w)), _margin(lq[:, t], w).tolist()
            rest = self._kept_kl(lp[t], lq[:, t], [w])
            kept = (kl.sum(1) - kl[:, t] + rest).tolist()
            drop = (lq[:, t, w] - lp[t, w]).tolist()
            top = (lq[:, t].argmax(-1) == c["actual"]).tolist()
            return [{"progress": min(1.0, max(0.0, (m0 - m) / m0)) if m0 > 0 else 1.0, "flipped": m < 0, "halved": d <= -LN2,
                     "rest": r, "actual_top": a, "kept": k} for m, d, r, a, k in zip(ms, drop, rest.tolist(), top, kept)]
        per = self._runs(seqs, es, fn, ref=True)
        held = self._held_terms("search" if search else "held", eds)
        key = self.key("copy", goal)
        n = len(half)
        for j, b in enumerate(ok):
            rs = per[j * n:(j + 1) * n]
            g = sum(r["progress"] for r in rs) / n
            flipped = sum(r["flipped"] for r in rs) / n
            best = max(self.best.get(key, 0.0), g)
            kept = sum(r["kept"] for r in rs) / n
            rows[b].update({"valid": True, "attained": flipped >= GOAL, "progress": g, "success": g / best if best > 0 else 0.0,
                            "flipped": flipped, "halved": sum(r["halved"] for r in rs) / n,
                            "actual_top": sum(r["actual_top"] for r in rs) / n, "rest": sum(r["rest"] for r in rs) / n,
                            "collateral": held[j] + kept, "collateral_held": held[j], "collateral_instances": kept})
        return rows


def _persistent(o: dict) -> bool:
    if o["op"] == "unit":
        return o["position"] is None
    return bool(o.get("persistent"))


# ---- the edit teacher and the baselines


def scale_op(nd, k: float) -> dict:
    return {"op": "scale", "node": tuple(nd), "k": float(k), "persistent": nd[1] is None}


def cut_op(r, w) -> dict:
    return {"op": "cut", "reader": r if r == "out" else tuple(r), "writer": tuple(w), "persistent": w[1] is None}


def unit_op(kind: str, layer: int, index: int, position: int | None, k: float) -> dict:
    return {"op": "unit", "kind": kind, "layer": layer, "index": index, "position": position, "k": float(k)}


def op_text(o: dict) -> str:
    """An operation in the answer grammar (components.edit_op_text, nodes written out); native units as
    "scale head L2.h3@394 0" / "scale neuron L1.n1203 0" (not in the oracle's grammar)."""
    import components

    if o["op"] == "unit":
        pos = "" if o["position"] is None else f"@{o['position']}"
        head = "persistent " if o["position"] is None else ""
        return f"{head}scale {o['kind']} L{o['layer']}.{'h' if o['kind'] == 'head' else 'n'}{o['index']}{pos} {o['k']:g}"
    return components.edit_op_text(o, {})


def teacher_nodes(task_id: str, dirs=TEACHER_DIRS) -> list[tuple]:
    """The nodes (matrix, position, index) of the text's teacher circuits (goal_eval JSON <task>.json and
    <task>.f1.json in dirs)."""
    out = []
    for d in dirs:
        for name in (f"{task_id}.json", f"{task_id}.f1.json"):
            p = Path(d) / name
            if p.exists():
                out += [(m, int(t), int(c)) for m, c, t in json.loads(p.read_text())["nodes"] if m != "tok"]
    return list(dict.fromkeys(out))


def candidates(ee: EditEval, task: dict, goal: dict, vpd_top: int = 64, dirs=TEACHER_DIRS) -> list[dict]:
    """The search's single edits on a local goal: every node of the teacher's circuits and VPD's vpd_top most
    important subcomponents (and for detach the vpd_top most important at its position) scaled by NODE_SCALES, every
    connection the model has among them cut (into the prediction too), and for detach the position's token embedding
    cut from the nodes there that read it."""
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    t = max(targets)
    nodes = [nd for nd in teacher_nodes(task["id"], dirs) if nd[1] <= t]
    if ee.nat.has_importance:
        order = ee.nat._vpd_order(ids, targets)
        nodes += order[:vpd_top]
        if goal["kind"] == "detach":
            nodes += [nd for nd in order if nd[1] == goal["position"]][:vpd_top]
    nodes = list(dict.fromkeys(nodes))
    ops = [scale_op(nd, k) for nd in nodes for k in NODE_SCALES]
    parents, outs = native.all_edges(nodes, targets)
    ops += [cut_op(r, w) for r, ws in parents.items() for w in ws]
    ops += [cut_op("out", w) for w in outs]
    if goal["kind"] == "detach":
        q = goal["position"]
        tok = ("tok", q, 0)
        ops += [cut_op(nd, tok) for nd in nodes if nd[1] == q and native._layer_kind(nd)[1] in mech.RESID_READERS]
    return ops


def unit_candidates(ee: EditEval, task: dict, goal: dict, dirs=TEACHER_DIRS) -> list[dict]:
    """The same search's single edits over native units: every head at the positions of the teacher's circuit nodes,
    the target and (detach) the goal's position, and every MLP neuron at the target and the goal's position, each
    scaled by NODE_SCALES."""
    p = task["prompts"][0]
    t = max(p["target_positions"])
    tg = ee.nat.target
    width = tg.site(native.site_name(0, "c_fc")).W.shape[0]
    goal_pos = [goal["position"]] if goal["kind"] == "detach" else []
    head_pos = sorted({nd[1] for nd in teacher_nodes(task["id"], dirs) if nd[1] <= t} | {t} | set(goal_pos))
    ops = [unit_op("head", layer, h, q, k) for q in head_pos for layer in range(tg.n_layer) for h in range(tg.n_head) for k in NODE_SCALES]
    ops += [unit_op("neuron", layer, j, q, k) for q in sorted({t} | set(goal_pos)) for layer in range(tg.n_layer) for j in range(width)
            for k in NODE_SCALES]
    return ops


def _better(a: dict, b: dict | None) -> bool:
    """Row a ahead of row b: more progress, then less collateral."""
    if b is None:
        return True
    return (a["progress"], -a["collateral"]) > (b["progress"], -b["collateral"])


def search(score, pool: list[dict], max_ops: int = MAX_OPS, screen: int = SCREEN, log=print) -> dict:
    """The edit teacher's measured greedy search (module docstring). score(list of edits) -> rows (EditEval.score_many
    with search=True). Returns {"ops", "row" (the search's numbers), "rounds", "evaluated"}."""
    S, cur, evaluated, rounds = [], None, 0, []
    for r in range(max_ops):
        trial = [S + [c] for c in pool]
        rows = score(trial)
        evaluated += len(trial)
        good = [i for i, row in enumerate(rows) if row["valid"]]
        if not good:
            break
        ranked = sorted(good, key=lambda i: (-rows[i]["progress"], rows[i]["collateral"]))
        if r == 0:
            pool = [pool[i] for i in ranked[:screen]]
            rows = [rows[i] for i in ranked[:screen]]
            ranked = list(range(len(pool)))
        best = ranked[0]
        if cur is not None and rows[best]["progress"] <= cur["progress"] + 1e-6:
            break
        S, cur = S + [pool[best]], rows[best]
        rounds.append({"op": op_text(pool[best]), "progress": cur["progress"], "collateral": cur["collateral"]})
        log(f"  round {r}: {op_text(pool[best])} -> progress {cur['progress']:.3f} collateral {cur['collateral']:.3f}")
        pool = [c for j, c in enumerate(pool) if j != best]
        if cur["attained"]:
            break
    if cur is None:
        return {"ops": [], "row": None, "rounds": rounds, "evaluated": evaluated}

    def keeps(row):  # what the search reached stays reached
        return row["valid"] and (row["attained"] if cur["attained"] else row["progress"] >= cur["progress"] - 1e-6)

    while len(S) > 1:  # drop what the goal does not need, least collateral first
        trial = [S[:i] + S[i + 1:] for i in range(len(S))]
        rows = score(trial)
        evaluated += len(trial)
        ok = [i for i in range(len(S)) if keeps(rows[i])]
        if not ok:
            break
        i = min(ok, key=lambda i: rows[i]["collateral"])
        S, cur = trial[i], rows[i]
    for i in range(len(S)):  # swap single edits for ones that keep the goal with less collateral
        others = [c for c in pool if c not in S]
        trial = [S[:i] + [c] + S[i + 1:] for c in others]
        rows = score(trial)
        evaluated += len(trial)
        ok = [j for j in range(len(trial)) if keeps(rows[j]) and rows[j]["collateral"] < cur["collateral"]]
        if ok:
            j = min(ok, key=lambda j: rows[j]["collateral"])
            S, cur = trial[j], rows[j]
    return {"ops": S, "row": cur, "rounds": rounds, "evaluated": evaluated}


def _rate(row: dict, cur: dict | None) -> tuple | None:
    """A candidate's rank in the frontier search: progress gained per collateral bit added (free gains first, by the
    progress they add); None when it adds no progress."""
    dp = row["progress"] - (cur["progress"] if cur else 0.0)
    dc = row["collateral"] - (cur["collateral"] if cur else 0.0)
    if not row["valid"] or dp <= 1e-9:
        return None
    return (1, dp) if dc <= 0 else (0, dp / dc)


def frontier(score, pool: list[dict], max_ops: int = MAX_OPS, screen: int = SCREEN, log=print) -> dict:
    """A persistent goal's search: a goal on every text can be reached by breaking the model, so each round adds the
    candidate with the most progress per bit of collateral, and every step is kept: the path from the unedited model
    traces progress against collateral (each step a point; the next costs more collateral per unit of progress).
    Returns {"path": [{"ops", "row"}], "evaluated"}."""
    S, cur, evaluated, path = [], None, 0, []
    for r in range(max_ops):
        rows = score([S + [c] for c in pool])
        evaluated += len(pool)
        rated = [(_rate(row, cur), i) for i, row in enumerate(rows)]
        ranked = [i for g, i in sorted(((g, i) for g, i in rated if g is not None), reverse=True)]
        if r == 0:
            keep = ranked[:screen]
            pool, rows = [pool[i] for i in keep], [rows[i] for i in keep]
            ranked = list(range(len(pool)))
        if not ranked:
            break
        best = ranked[0]
        S, cur = S + [pool[best]], rows[best]
        path.append({"ops": [op_text(o) for o in S], "edit": [_jsonable(o) for o in S], "row": cur})
        log(f"  step {r}: {op_text(pool[best])} -> progress {cur['progress']:.3f} collateral {cur['collateral']:.3f}")
        pool = [c for j, c in enumerate(pool) if j != best]
        if cur["attained"]:
            break
    return {"path": path, "evaluated": evaluated}


def vpd_topk(ee: EditEval, task: dict, sizes) -> list[tuple[int, list[dict]]]:
    """VPD's baseline edits: its k most important subcomponents on the text (native.Native._vpd_order) ablated."""
    p = task["prompts"][0]
    order = ee.nat._vpd_order(p["token_ids"], p["target_positions"])
    return [(k, [scale_op(nd, 0.0) for nd in order[:k]]) for k in sizes if k <= len(order)]


SIZES = (1, 2, 4, 8, 16, 32, 64)


def teach_goal(ee: EditEval, task: dict, goal: dict, log=print) -> dict:
    """One goal: the edit teacher, the native-unit search and VPD's top-k, each scored on the score's data."""
    t0 = time.perf_counter()
    score = lambda eds: ee.score_many(task, goal, eds, search=True)  # noqa: E731
    out = {"task": task["id"], "goal": goal}
    for name, pool in (("teacher", candidates(ee, task, goal)), ("units", unit_candidates(ee, task, goal))):
        log(f"{task['id']} {goal['kind']} {name}: {len(pool)} candidates")
        res = search(score, pool, log=log)
        row = ee.score(task, goal, res["ops"]) if res["ops"] else None
        out[name] = {"ops": [op_text(o) for o in res["ops"]], "edit": [_jsonable(o) for o in res["ops"]], "search": res["row"],
                     "score": row, "rounds": res["rounds"], "evaluated": res["evaluated"]}
    if ee.nat.has_importance:
        k_teacher = len(out["teacher"]["ops"])
        sizes = sorted(set(SIZES) | ({k_teacher} if k_teacher else set()))
        tops = vpd_topk(ee, task, sizes)
        rows = ee.score_many(task, goal, [ops for _, ops in tops])
        out["vpd_topk"] = [{"k": k, "score": r} for (k, _), r in zip(tops, rows)]
    out["seconds"] = time.perf_counter() - t0
    return out


def _jsonable(o: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()}


def _from_json(o: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in o.items()}


# ---- the persistent copy goal


def copy_nodes(ee: EditEval, half: list[dict], min_count: int = 2) -> list[tuple]:
    """Subcomponents (matrix, None, index) active (causal importance > 0) at the targets of at least min_count of the
    instances, most often first (ties: larger mean importance)."""
    count, total = {}, {}
    for c in half:
        ci = ee.nat.importance(c["ids"])
        for n in ee.nat.names:
            z = ci[n][c["t"]]
            for j in (z > 0).nonzero().flatten().tolist():
                count[(n, None, j)] = count.get((n, None, j), 0) + 1
                total[(n, None, j)] = total.get((n, None, j), 0.0) + float(z[j])
    keep = [nd for nd in count if count[nd] >= min_count]
    return sorted(keep, key=lambda nd: (-count[nd], -total[nd] / count[nd]))


def screen_local(ee: EditEval, half: list[dict], ops_at, n: int, top: int, log=print) -> list[int]:
    """The candidates' order by measured local effect: each candidate applied at the target of the first n instances
    (ops_at(i, t) -> the edit at position t), ranked by mean flip progress on the copied token; the first top."""
    sub = half[:n]
    score = np.zeros(len(ops_at))
    for c in sub:
        lp = ee.reference(c["ids"])
        t = c["t"]
        m0 = float(_margin(lp[t], c["wrong"]))
        res = ee._runs([c["ids"]] * len(ops_at), [[f(t)] for f in ops_at], lambda lq, idx, s, lp: _margin(lq[:, t - s], c["wrong"]).tolist())
        score += np.array([min(1.0, max(0.0, (m0 - m) / m0)) if m0 > 0 else 1.0 for m in res]) / len(sub)
    order = np.argsort(-score, kind="stable")[:top]
    log(f"  screened {len(ops_at)} candidates on {len(sub)} instances; best mean local progress {score[order[0]]:.3f}")
    return order.tolist()


def teach_copy(ee: EditEval, log=print, screen_n: int = 16) -> dict:
    """The persistent copy goal: the frontier search over subcomponents (persistent scales of the subcomponents most
    often active at the first half's targets, screened by measured local ablation, and persistent cuts among the top
    ones) and over native units (every head, the neurons screened the same way), each step of each path scored on the
    second half; VPD's top-k by activity count, and the L2 heads 2-4 ablation (the earlier copy fix)."""
    t0 = time.perf_counter()
    goal = copy_goal()
    first, _ = copy_instances()
    score = lambda eds: ee.score_many({"id": "copy"}, goal, eds, search=True)  # noqa: E731
    tg = ee.nat.target
    out = {"goal": goal}
    nodes = copy_nodes(ee, first)
    log(f"copy: {len(nodes)} subcomponents active at >= 2 first-half targets")
    order = screen_local(ee, first, [lambda t, nd=nd: scale_op((nd[0], t, nd[2]), 0.0) for nd in nodes], screen_n, SCREEN, log)
    top = [nodes[i] for i in order]
    pool = [scale_op(nd, k) for nd in top for k in NODE_SCALES]
    lead = top[:32]
    pool += [cut_op(r, w) for r in lead for w in lead if _connected(w, r, [0], True)]
    width = tg.site(native.site_name(0, "c_fc")).W.shape[0]
    neurons = [(layer, j) for layer in range(tg.n_layer) for j in range(width)]
    order = screen_local(ee, first, [lambda t, u=u: unit_op("neuron", u[0], u[1], t, 0.0) for u in neurons], screen_n, SCREEN, log)
    upool = [unit_op("head", layer, h, None, k) for layer in range(tg.n_layer) for h in range(tg.n_head) for k in NODE_SCALES]
    upool += [unit_op("neuron", neurons[i][0], neurons[i][1], None, k) for i in order for k in NODE_SCALES]
    for name, pl in (("teacher", pool), ("units", upool)):
        log(f"copy {name}: {len(pl)} candidates")
        res = frontier(score, pl, log=log)
        rows = ee.score_many({"id": "copy"}, goal, [[_from_json(o) for o in st["edit"]] for st in res["path"]])
        out[name] = {"path": [{**st, "score": r} for st, r in zip(res["path"], rows)], "evaluated": res["evaluated"]}
    sizes = sorted(set(SIZES))
    eds = [[scale_op(nd, 0.0) for nd in nodes[:k]] for k in sizes]
    out["vpd_topk"] = [{"k": k, "score": r} for k, r in zip(sizes, ee.score_many({"id": "copy"}, goal, eds))]
    heads = [unit_op("head", 2, h, None, 0.0) for h in (2, 3, 4)]
    out["l2_heads_2_4"] = {"ops": [op_text(o) for o in heads], "score": ee.score_many({"id": "copy"}, goal, [heads])[0]}
    out["seconds"] = time.perf_counter() - t0
    return out


# ---- command line


def _tasks(split: str, n: int) -> list[dict]:
    return [json.loads(p.read_text()) for p in native.tasks(split)[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tokens")
    g = sub.add_parser("goals")
    g.add_argument("out")
    g.add_argument("--split", default="heldout")
    g.add_argument("--n", type=int, default=20)
    g.add_argument("--fresh", action="store_true")
    t = sub.add_parser("teach")
    t.add_argument("goals")
    t.add_argument("out")
    t.add_argument("--offset", type=int, default=0)
    t.add_argument("--stride", type=int, default=1)
    c = sub.add_parser("copy")
    c.add_argument("out")
    b = sub.add_parser("table")
    b.add_argument("rows")
    b.add_argument("copy", nargs="?")
    args = ap.parse_args()
    if args.cmd == "tokens":
        return build_tokens()
    if args.cmd == "table":
        return print(table([json.loads(x) for x in Path(args.rows).read_text().splitlines() if x.strip()],
                           json.loads(Path(args.copy).read_text()) if args.copy else None))
    ee = EditEval()
    if args.cmd == "goals":
        with open(args.out, "w") as f:
            for task in _tasks(args.split, args.n):
                for gl in goals(ee, task, native.task_seed(task["id"]) + (1 if args.fresh else 0)):
                    f.write(json.dumps({"task": task["id"], "split": args.split, "goal": gl}) + "\n")
                print(task["id"], flush=True)
    elif args.cmd == "teach":
        rows = [json.loads(x) for x in Path(args.goals).read_text().splitlines() if x.strip()]
        done = set()  # the goals any teach_*.jsonl beside OUT holds; the rest are split between the processes
        for p in sorted(Path(args.out).parent.glob("teach_*.jsonl")):
            done |= {EditEval.key(r["task"], r["goal"]) for r in map(json.loads, p.read_text().splitlines()) if r}
        rows = [r for r in rows if EditEval.key(r["task"], r["goal"]) not in done]
        tasks = {}
        for i, r in enumerate(rows):
            if i % args.stride != args.offset:
                continue
            if r["task"] not in tasks:
                tasks = {r["task"]: json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())}
            res = teach_goal(ee, tasks[r["task"]], r["goal"], log=lambda s: print(s, flush=True))
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")
    elif args.cmd == "copy":
        res = teach_copy(ee, log=lambda s: print(s, flush=True))
        Path(args.out).write_text(json.dumps(res, indent=1))


def table(rows: list[dict], copy: dict | None = None) -> str:
    """Per goal kind and method: the share attained, mean progress, mean success (over the best of the methods on
    each goal), mean collateral and size; VPD's top-k at the teacher's size and its smallest k that attains."""
    lines = ["kind     method     n   attained progress success collateral size_bits ops"]
    kinds = sorted({r["goal"]["kind"] for r in rows})
    for kind in kinds:
        rs = [r for r in rows if r["goal"]["kind"] == kind]
        for method in ("teacher", "units", "vpd@teacher"):
            vals = []
            for r in rs:
                if method == "vpd@teacher":
                    k = len(r["teacher"]["ops"])
                    s = next((x["score"] for x in r.get("vpd_topk", []) if x["k"] == k), None)
                    nops = k
                else:
                    s, nops = r[method]["score"], len(r[method]["ops"])
                best = max([x["progress"] for x in _scores(r) if x and x.get("valid")] or [0.0])
                vals.append((s, nops, best))
            ok = [(s, k, b) for s, k, b in vals if s and s.get("valid")]
            if not ok:
                lines.append(f"{kind:8s} {method:10s} {len(rs):3d}   (no valid edit)")
                continue
            mean = lambda f: sum(f(x) for x in ok) / len(ok)  # noqa: E731
            lines.append(f"{kind:8s} {method:10s} {len(rs):3d}   {mean(lambda x: x[0]['attained']):.2f}     "
                         f"{mean(lambda x: x[0]['progress']):.3f}    {mean(lambda x: x[0]['progress'] / x[2] if x[2] else 0):.3f}   "
                         f"{mean(lambda x: x[0]['collateral']):9.3f}  {mean(lambda x: x[0]['size']):8.1f} {mean(lambda x: x[1]):.1f}")
        att = []
        for r in rs:
            ks = [x["k"] for x in r.get("vpd_topk", []) if x["score"] and x["score"].get("attained")]
            att.append(min(ks) if ks else None)
        hit = [k for k in att if k is not None]
        lines.append(f"{kind:8s} VPD top-k attains on {len(hit)}/{len(rs)} goals (median smallest k "
                     f"{sorted(hit)[len(hit) // 2] if hit else '-'})")
    if copy:
        lines.append("copy (persistent; fresh half): method flipped halved progress rest collateral_held collateral_instances ops")
        steps = [(f"{name} {k + 1}", st["score"], k + 1) for name in ("teacher", "units") for k, st in enumerate(copy.get(name, {}).get("path", []))]
        steps.append(("l2_heads_2_4", copy.get("l2_heads_2_4", {}).get("score"), 3))
        for name, s, n in steps:
            if s and s.get("valid"):
                lines.append(f"  {name:13s} {s['flipped']:.2f} {s['halved']:.2f} {s['progress']:.3f} {s['rest']:.3f} "
                             f"{s['collateral_held']:.2f} {s['collateral_instances']:.2f} {n}")
        for x in copy.get("vpd_topk", []):
            s = x["score"]
            if s and s.get("valid"):
                lines.append(f"  vpd top-{x['k']:<5d} {s['flipped']:.2f} {s['halved']:.2f} {s['progress']:.3f} {s['rest']:.3f} "
                             f"{s['collateral_held']:.2f} {s['collateral_instances']:.2f} {x['k']}")
    return "\n".join(lines)


def _scores(r: dict) -> list:
    return [r["teacher"]["score"], r["units"]["score"]] + [x["score"] for x in r.get("vpd_topk", [])]


if __name__ == "__main__":
    main()
