"""RL environment E, "edits" (#2951 graph oracle; the lead's 10-13): the oracle reads a goal, "make the model stop
predicting b after a while it keeps predicting b after other tokens", and measured evidence, and proposes a persistent
weight edit; the verifier applies the edit to the model's weights and measures it on held-out text.

Goals come from measured preferences (experiments/oracle/persistent_e3.py prefs: content-word pairs a -> b on split B
of the atlas's Pile rows, thirds fit / val / score; those with MIN_HELD instances and controls in the score third),
split train / dev by preference (a hash of a and b). 10-13: 152 preferences at --min-pairs 8, 83 goals (25 dev). The evidence
is the fit third's alone (the learnability grid's decision: the text plus measured facts): example contexts, the
subcomponents whose ablation at such a position most lowers b's margin there (the circuit, by measured local effect)
and the same ranking at the controls (b predicted after other tokens: the controls' circuit), with their effects in
bits.

The answer: one operation per line, at most MAX_OPS, every other line ignored (kind: q, k, v, o, fc, down; u_i and v_i
are the matrix's subcomponent i's write and read vectors, W its weights, x W^T its output):
  scale L<layer>.<kind>#<i> <k>                       subcomponent i's output times k at every position: W - (1 - k) u_i v_i^T
  update L<layer>.<kind> read <i>:<a> ... write <j>:<b> ...
                                                      a rank-one update inside named subcomponents: W + (sum_j b_j u_j)
                                                      (sum_i a_i v_i)^T, which adds (x . sum a_i v_i) sum b_j u_j to the
                                                      output ("read i:1 write i:-1" removes subcomponent i)
k, a and b are clipped to COEF; a line naming no subcomponent of the model counts as unparsed and is dropped.

The verifier (Verifier.score) changes the weights W of the named matrices (every forward pass reads them), runs the
part's texts and edit_eval's held Pile texts (rows 9600-9615), and restores W:
  removal        the share of the part's instances (a then the most likely token b, unedited) where b is no longer the
                 most likely token;
  controls_kept  the share of its controls (b the most likely token after another token, unedited) where it still is;
  collateral     KL from the unedited model in bits per token: at the part's rows' other positions plus on the held
                 Pile texts (persistent_e3's collateral_per_token; both also reported alone);
  reward         removal - LAM x collateral - MU x (1 - controls_kept).
LAM and MU were fixed before any training: LAM = 1 orders the 110 validation frontiers of the persistent E3 runs (24,386
pairs of candidate edits) most like persistent_e3.select (flipped >= 0.9, then the least collateral): 87.3% of pairs at
1 (0.5: 87.4%, 2: 85.7%, 10: 67.6%); MU = 1: a control lost costs what an instance removed earns, so an edit that stops
b everywhere earns nothing. The part defaults to "score" (never in the evidence); "val" picks among a teacher's edits.

  edit_env.py goals PREFS.jsonl OUT.jsonl [--dev 0.3]   the goals: id, split, question, the preference
  edit_env.py score GOALS.jsonl ANSWERS.jsonl OUT.jsonl  each answer {"goal", "answer"} scored
  edit_env.py sft GOALS.jsonl EDITS.jsonl OUT.jsonl [--min-reward 0]   train goals' teacher answers as chat SFT rows
  edit_env.py gate GOALS.jsonl EDITS.jsonl OUT.jsonl [--sft SFT.jsonl]
      the gate (teacher edits above random-subcomponent edits on >= GATE of dev goals) on persistent_e3.py edit's rows:
      the teacher's edit is the candidate of circuit_edit and circuit_scales (their searches' steps and checkpoints)
      earning the most reward on the val third, the random side's the same among random_equal's and random_scales'
      (random subcomponents active at the instances, outside the circuit, ablated or given fitted scales); both written
      as answers and scored by the verifier on the score third; also against each kind alone: ablated (random_equal:
      as many random active subcomponents as the circuit edit's operations, removed) and fitted (random_scales: 16 with
      scales fitted by Adam on the fit third's objective, an optimization the oracle cannot run). --sft: train goals'
      teacher answers, as SFT rows {"id", "question", "answer", "reward"}.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import statistics
import sys
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

KINDS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj", "fc": "c_fc", "down": "down_proj"}
CODES = {v: k for k, v in KINDS.items()}
_M = r"L(\d+)\.(q|k|v|o|fc|down)"
_NUM = r"(-?\d+(?:\.\d*)?(?:[eE]-?\d+)?)"
SCALE = re.compile(rf"^\s*scale\s+{_M}#(\d+)\s+{_NUM}\s*$")
UPDATE = re.compile(rf"^\s*update\s+{_M}\s+read\s+(.+?)\s+write\s+(.+?)\s*$")
COEFS = re.compile(rf"(\d+)\s*:\s*{_NUM}")
MAX_OPS = 16
COEF = 3.0  # |k|, |a|, |b| at most
LAM = 1.0  # reward per bit per token of collateral (module docstring)
MU = 1.0  # reward per share of controls lost
EVIDENCE_TOP = 24  # subcomponents listed per circuit (persistent_e3.EDIT_CANDS: the teacher's candidates)
EXAMPLES = 4
CONTEXT = 12  # tokens before a shown in an example
GATE = 0.8
MIN_HELD = 8  # a goal's score-third instances and controls, at least (scorable)
TEACHERS = ("circuit_edit", "circuit_scales")
RANDOMS = ("random_equal_0", "random_equal_1", "random_scales")
RANDOM_KINDS = {"ablated": ("random_equal_0", "random_equal_1"), "fitted": ("random_scales",)}  # the random side, by kind
TOKENS = Path.home() / "mpd-data/atlas/vpd4l/tokens_0_6000.npy"  # the atlas's rows, split B 5000-5999
BATCH_TOKENS = int(os.environ.get("EDIT_ENV_BATCH_TOKENS", "4096"))  # tokens per forward pass (padded)


def matrix(layer: int, code: str) -> str:
    return f"h.{layer}.{'mlp' if code in ('fc', 'down') else 'attn'}.{KINDS[code]}"


def label(site: str, c: int | None = None) -> str:
    parts = site.split(".")
    return f"L{parts[1]}.{CODES[parts[-1]]}" + ("" if c is None else f"#{c}")


def parse(answer: str, C: dict) -> tuple[list[tuple], int]:
    """(operations, lines that do not parse or name no subcomponent): ("scale", matrix, i, k) and ("update", matrix,
    [(i, a)], [(j, b)]), the last MAX_OPS kept, a subcomponent scaled twice taking the last scale."""
    ops, bad = [], 0
    clip = lambda x: min(max(float(x), -COEF), COEF)  # noqa: E731
    for line in answer.splitlines():
        head = line.strip().lower()
        if not head.startswith(("scale", "update")):
            continue
        m = SCALE.match(line) or UPDATE.match(line)
        site = matrix(int(m[1]), m[2]) if m else None
        if not m or site not in C:
            bad += 1
            continue
        if m.re is SCALE:
            c = int(m[3])
            if not 0 <= c < C[site]:
                bad += 1
                continue
            ops = [o for o in ops if not (o[0] == "scale" and o[1:3] == (site, c))] + [("scale", site, c, clip(m[4]))]
            continue
        rd, wr = ([(int(i), clip(x)) for i, x in COEFS.findall(g)] for g in (m[3], m[4]))
        if not rd or not wr or any(not 0 <= i < C[site] for i, _ in rd + wr):
            bad += 1
            continue
        ops.append(("update", site, rd, wr))
    return ops[-MAX_OPS:], bad


def render(ops: list[tuple]) -> str:
    """The answer text of operations (parse's inverse, numbers to four significant digits)."""
    lines = []
    for o in ops:
        if o[0] == "scale":
            lines.append(f"scale {label(o[1], o[2])} {o[3]:.4g}")
        else:
            co = lambda xs: " ".join(f"{i}:{x:.4g}" for i, x in xs)  # noqa: E731
            lines.append(f"update {label(o[1])} read {co(o[2])} write {co(o[3])}")
    return "\n".join(lines)


def split_of(a: int, b: int, dev: float) -> str:
    return "dev" if zlib.crc32(f"{a}->{b}".encode()) % 1000 < dev * 1000 else "train"


def question(pref: dict, decode, tokens) -> str:
    """The goal and the fit third's evidence (module docstring)."""
    a, b = pref["describe"]["a"], pref["describe"]["b"]
    ex = [repr(decode(tokens[r, max(0, p - CONTEXT):p + 1].astype(int).tolist())) for r, p in pref["parts"]["fit"]["instances"][:EXAMPLES]]
    circ = lambda key: ", ".join(f"{label(s, c)} {e:.2f}" for s, c, e in pref.get(key, [])[:EVIDENCE_TOP]) or "none measured"  # noqa: E731
    return (f"Goal: make the model stop predicting {b!r} as the next token after {a!r}, on any text, while it keeps "
            f"predicting {b!r} where it does after other tokens.\n"
            f"Texts ending in {a!r} after which it predicts {b!r}: " + "; ".join(ex) + "\n"
            f"Subcomponents whose removal at such a position most lowers {b!r}'s margin there (bits): {circ('circuit')}\n"
            f"Subcomponents whose removal most lowers {b!r}'s margin where it is predicted after other tokens (bits): "
            f"{circ('control_circuit')}\n"
            f"Answer with at most {MAX_OPS} lines, each one weight edit applied at every position of every text:\n"
            f"  scale L<layer>.<kind>#<i> <k>   subcomponent i's output times k (0 removes it, -1 reverses it)\n"
            f"  update L<layer>.<kind> read <i>:<a> ... write <j>:<b> ...   adds (x . sum a_i v_i) sum b_j u_j to the "
            f"matrix's output (x its input, v_i and u_i subcomponent i's read and write vectors)\n"
            f"kind is q, k, v, o, fc or down; numbers between {-COEF:g} and {COEF:g}.")


def scorable(pref: dict) -> bool:
    """At least MIN_HELD instances and controls in the score third and MIN_HELD / 2 instances in the val third (the
    preference's MIN_PAIRS counts the fit third alone): removal measured to 1/MIN_HELD at least."""
    sc, val = pref["parts"]["score"], pref["parts"]["val"]
    return len(sc["instances"]) >= MIN_HELD and len(sc["controls"]) >= MIN_HELD and len(val["instances"]) >= MIN_HELD // 2


def goals(prefs: list[dict], dev: float) -> list[dict]:
    """The goals of the scorable preferences, in the preference builder's order."""
    import numpy as np

    import components

    tokens = np.load(TOKENS, mmap_mode="r")
    decode = lambda xs: "".join(components.token_strings("vpd4l", xs))  # noqa: E731
    return [{"id": f"pref:{p['a']}->{p['b']}", "split": split_of(p["a"], p["b"], dev), "question": question(p, decode, tokens),
             "pref": p} for p in sorted(prefs, key=lambda p: p.get("order", 0)) if scorable(p)]


@contextlib.contextmanager
def applied(nat, ops: list[tuple]):
    """The model with the edit in its weights (each named matrix's W, which every forward pass reads), W restored on exit."""
    tg = nat.target
    saved = {}
    with torch.no_grad():
        for o in ops:
            st = tg.site(o[1])
            if o[1] not in saved:
                saved[o[1]] = st.W.clone()
            if o[0] == "scale":
                st.W -= (1 - o[3]) * torch.outer(st.U[o[2]], st.V[:, o[2]])
            else:
                coef = lambda xs: torch.tensor([x for _, x in xs], dtype=st.W.dtype, device=st.W.device)  # noqa: E731
                read = st.V[:, [i for i, _ in o[2]]] @ coef(o[2])
                write = st.U[[j for j, _ in o[3]]].T @ coef(o[3])
                st.W += torch.outer(write, read)
    try:
        yield
    finally:
        with torch.no_grad():
            for s, W in saved.items():
                tg.site(s).W.copy_(W)


def _batches(seqs: list[list[int]]) -> list[list[int]]:
    """Indices of seqs in groups of at most BATCH_TOKENS tokens once padded to the group's longest."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * len(seqs[cur[0]]) > BATCH_TOKENS:
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


class Verifier:
    """Scores edits for a goal on the weights (module docstring). Each request's edits are scored together: per batch of
    texts the unedited run once, then each edit's; the held texts' unedited log probabilities are kept."""

    def __init__(self, nat, tokens, held: list[list[int]]):
        self.nat, self.tokens, self.held = nat, tokens, held
        self._held_ref = None

    @torch.no_grad()
    def _lp(self, seqs: list[list[int]]) -> torch.Tensor:
        """log p (natural) [len(seqs), T, V], seqs right-padded (causal: a position reads none after it)."""
        tg, T = self.nat.target, max(map(len, seqs))
        x = torch.zeros(len(seqs), T, dtype=torch.long, device=self.nat.dev)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, device=self.nat.dev)
        return torch.log_softmax(tg.hidden(x) @ tg.wte.T, -1)

    def _texts(self, pref: dict, part: str) -> list[tuple[list[int], list[int], list[int]]]:
        """(ids, instance positions, control positions) per row of the part, cut after its last such position."""
        by = {}
        for kind in ("instances", "controls"):
            for r, p in pref["parts"][part][kind]:
                by.setdefault(r, ([], []))[kind == "controls"].append(p)
        return [(self.tokens[r, :max(i + c) + 1].astype(int).tolist(), i, c) for r, (i, c) in sorted(by.items())]

    @staticmethod
    def _kl(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
        return (lp.exp() * (lp - lq)).sum(-1) / math.log(2)

    @torch.no_grad()
    def score(self, pref: dict, edits: list[list[tuple]], part: str = "score") -> list[dict]:
        """One result per edit (module docstring); the empty edit scores 0 removal, all controls kept, 0 collateral."""
        b, texts = pref["b"], self._texts(pref, part)
        live = [e for e in range(len(edits)) if edits[e]]
        st = {e: {"n": 0, "gone": 0, "nc": 0, "kept": 0, "text": 0.0, "pile": 0.0} for e in live}
        if live:
            seqs = [t[0] for t in texts]
            for idx in _batches(seqs):
                lp = self._lp([seqs[i] for i in idx])
                top0 = lp.argmax(-1)
                for e in live:
                    with applied(self.nat, edits[e]):
                        lq = self._lp([seqs[i] for i in idx])
                    top, kl = lq.argmax(-1), self._kl(lp, lq)
                    for j, i in enumerate(idx):
                        ids, inst, ctrl = texts[i]
                        inst = [p for p in inst if top0[j, p] == b]  # the instances of this run (b the most likely, unedited)
                        ctrl = [p for p in ctrl if top0[j, p] == b]
                        s = st[e]
                        s["n"] += len(inst)
                        s["gone"] += int((top[j, inst] != b).sum()) if inst else 0
                        s["nc"] += len(ctrl)
                        s["kept"] += int((top[j, ctrl] == b).sum()) if ctrl else 0
                        rest = torch.ones(len(ids), dtype=torch.bool, device=kl.device)
                        rest[inst] = False
                        s["text"] += float(kl[j, :len(ids)][rest].mean())
                    del lq, top, kl
                del lp
            groups = _batches(self.held)
            if self._held_ref is None:  # the held texts' unedited log probabilities, per group (every request's)
                self._held_ref = [self._lp([self.held[i] for i in idx]) for idx in groups]
            for e in live:
                with applied(self.nat, edits[e]):
                    kl = sum(float(self._kl(lp, self._lp([self.held[i] for i in idx])).sum()) for idx, lp in zip(groups, self._held_ref))
                st[e]["pile"] = kl / sum(map(len, self.held))
        out = []
        for e in range(len(edits)):
            s = st.get(e)
            if s is None:
                out.append({"removal": 0.0, "controls_kept": 1.0, "collateral": 0.0, "text_kl": 0.0, "pile_kl": 0.0, "reward": 0.0,
                            "instances": None, "controls": None, "ops": 0})
                continue
            removal, kept = s["gone"] / max(1, s["n"]), s["kept"] / s["nc"] if s["nc"] else 1.0
            text = s["text"] / max(1, len(texts))
            coll = text + s["pile"]
            out.append({"removal": removal, "controls_kept": kept, "collateral": coll, "text_kl": text, "pile_kl": s["pile"],
                        "reward": removal - LAM * coll - MU * (1 - kept), "instances": s["n"], "controls": s["nc"], "ops": len(edits[e])})
        return out

    def answers(self, pref: dict, answers: list[str], part: str = "score") -> list[dict]:
        """Each answer parsed (parse) and scored, with "valid" (some operation parsed), "unparsed" and the operations."""
        parsed = [parse(a, self.nat.C) for a in answers]
        got = self.score(pref, [ops for ops, _ in parsed], part)
        return [{**g, "valid": bool(ops), "unparsed": bad, "edit": render(ops)} for g, (ops, bad) in zip(got, parsed)]


def load(device: str | None = None):
    """(Native, Verifier) with the atlas's split-B tokens and edit_eval's held texts (its score-side collateral texts)."""
    import numpy as np

    import edit_eval
    import native

    nat = native.Native(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for q in nat.target.parameters():
        q.requires_grad_(False)
    tokens = np.load(TOKENS, mmap_mode="r")
    held = [list(map(int, x)) for x in np.load(edit_eval.TOKENS)["held"]]
    return nat, Verifier(nat, tokens, held)


def scale_ops(ops: list) -> list[tuple]:
    """persistent_e3 subcomponent operations (site, index, k) as scale operations."""
    return [("scale", s, int(c), min(max(float(k), -COEF), COEF)) for s, c, k in ops]


def val_reward(v: dict) -> float:
    """The reward of persistent_e3's validation metrics of an edit (the verifier's own quantities on the val third)."""
    kept = 1.0 if v.get("controls_kept") is None else v["controls_kept"]
    return v["flipped"] - LAM * v["collateral_per_token"] - MU * (1 - kept)


def best(e: dict, methods: tuple) -> tuple | None:
    """(method, operations, val reward) of the candidate edit earning the most on the val third among the methods'
    candidates (persistent_e3 rows' "candidates"; a row without them offers its pick alone)."""
    out = None
    for m in methods:
        r = e["methods"].get(m)
        for ops, val, _ in (r.get("candidates") or [[r["ops"], r["validation"], r["info"]]]) if r else []:
            if ops and (out is None or val_reward(val) > out[2]):
                out = (m, scale_ops(ops), val_reward(val))
    return out


def gate(ver: Verifier, goal_rows: list[dict], edit_rows: list[dict], log=print) -> tuple[list[dict], list[dict]]:
    """(per goal: the teacher's and the random side's edits, each the candidate earning the most on the val third, and
    every arm's own pick, scored on the score third; train goals' SFT rows). win: the teacher above the random side;
    first_win, the first definition: the arm picks (persistent_e3.select), the teacher the arm earning more on the val
    third, above the best random arm on the score third."""
    by = {(r["a"], r["b"]): r for r in edit_rows}
    out, sft = [], []
    for g in goal_rows:
        e = by.get((g["pref"]["a"], g["pref"]["b"]))
        t, r = (best(e, ms) for ms in (TEACHERS, RANDOMS)) if e else (None, None)
        if t is None or r is None:
            continue
        picks = {m: e["methods"][m] for m in TEACHERS + RANDOMS if e["methods"].get(m) and e["methods"][m].get("ops")}
        sides = {f"random:{k}": best(e, ms) for k, ms in RANDOM_KINDS.items()}  # each kind of random edit alone, picked the same way
        answers = {"teacher": render(t[1]), "random": render(r[1]), **{k: render(b[1]) for k, b in sides.items() if b},
                   **{m: render(scale_ops(x["ops"])) for m, x in picks.items()}}
        names = list(answers)
        test = dict(zip(names, ver.answers(g["pref"], [answers[n] for n in names], "score")))
        first_t = max((m for m in TEACHERS if m in picks), key=lambda m: val_reward(picks[m]["validation"]), default=None)
        first_r = [test[m]["reward"] for m in RANDOMS if m in picks]
        row = {"id": g["id"], "split": g["split"], "teacher": t[0], "random": r[0], "teacher_reward": test["teacher"]["reward"],
               "random_reward": test["random"]["reward"], "win": test["teacher"]["reward"] > test["random"]["reward"],
               "first_win": bool(first_t and first_r) and test[first_t]["reward"] > max(first_r), "val": {"teacher": t[2], "random": r[2]},
               "win_by_kind": {k: test["teacher"]["reward"] > test[f"random:{k}"]["reward"] for k in RANDOM_KINDS if f"random:{k}" in test},
               "score": test, "answers": answers}
        out.append(row)
        if g["split"] == "train":
            sft.append({"id": g["id"], "question": g["question"], "answer": answers["teacher"], "reward": row["teacher_reward"]})
        x = test["teacher"]
        log(f"{g['id']} [{g['split']}] teacher ({t[0]}) {x['reward']:+.3f} (removal {x['removal']:.2f}, kept {x['controls_kept']:.2f}, "
            f"collateral {x['collateral']:.2e}) vs random ({r[0]}) {row['random_reward']:+.3f}; first definition "
            f"{'won' if row['first_win'] else 'lost'}")
    return out, sft


def sft(ver: Verifier, goal_rows: list[dict], edit_rows: list[dict], min_reward: float = 0.0, log=print) -> list[dict]:
    """SFT rows of the train goals: the teacher's edit (the candidate of circuit_edit and circuit_scales earning the most
    on the val third) written as the answer, kept when its score-third reward is above min_reward (the empty edit's 0:
    an answer that loses to editing nothing teaches nothing), in the chat form {"prompt", "completion"} with the
    measurements beside."""
    by = {(r["a"], r["b"]): r for r in edit_rows}
    out = []
    for g in goal_rows:
        e = by.get((g["pref"]["a"], g["pref"]["b"]))
        t = best(e, TEACHERS) if e and g["split"] == "train" else None
        if t is None:
            continue
        answer = render(t[1])
        x = ver.answers(g["pref"], [answer], "score")[0]
        log(f"{g['id']} teacher ({t[0]}) {x['reward']:+.3f} (removal {x['removal']:.2f}, kept {x['controls_kept']:.2f}, collateral "
            f"{x['collateral']:.2e}){'' if x['reward'] > min_reward else ': dropped'}")
        if x["reward"] > min_reward:
            out.append({"id": g["id"], "prompt": [{"role": "user", "content": g["question"]}],
                        "completion": [{"role": "assistant", "content": answer}], "teacher": t[0],
                        **{k: x[k] for k in ("reward", "removal", "controls_kept", "collateral")}})
    return out


def gate_summary(rows: list[dict]) -> str:
    lines = []
    for split in ("dev", "train"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        wins, first = sum(r["win"] for r in rs), sum(r["first_win"] for r in rs)
        d = [r["teacher_reward"] - r["random_reward"] for r in rs]
        lines.append(f"{split}: teacher above random on {wins}/{len(rs)} goals ({wins / len(rs):.0%}; gate {GATE:.0%}"
                     f"{': passed' if split == 'dev' and wins / len(rs) >= GATE else ''}); reward difference median "
                     f"{statistics.median(d):+.3f}; first definition {first}/{len(rs)}")
        for k, ms in RANDOM_KINDS.items():
            ws = [r["win_by_kind"][k] for r in rs if k in r.get("win_by_kind", {})]
            lines.append(f"  teacher above {k} ({', '.join(ms)}) alone: {sum(ws)}/{len(ws)}")
        for m in ("teacher", "random") + tuple(f"random:{k}" for k in RANDOM_KINDS) + TEACHERS + RANDOMS:
            xs = [r["score"][m] for r in rs if m in r["score"]]
            if xs:
                mean = lambda k: statistics.fmean(x[k] for x in xs)  # noqa: E731
                lines.append(f"  {m:15s} n={len(xs):3d} reward {mean('reward'):+.3f} removal {mean('removal'):.2f} kept "
                             f"{mean('controls_kept'):.2f} collateral {mean('collateral'):.2e} bits/token")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("goals")
    g.add_argument("prefs")
    g.add_argument("out")
    g.add_argument("--dev", type=float, default=0.3)
    s = sub.add_parser("score")
    s.add_argument("goals")
    s.add_argument("answers")
    s.add_argument("out")
    s.add_argument("--part", default="score")
    t = sub.add_parser("gate")
    t.add_argument("goals")
    t.add_argument("edits")
    t.add_argument("out")
    t.add_argument("--sft")
    f = sub.add_parser("sft")
    f.add_argument("goals")
    f.add_argument("edits")
    f.add_argument("out")
    f.add_argument("--min-reward", type=float, default=0.0)
    for p in (s, t, f):
        p.add_argument("--device")
    args = ap.parse_args()
    read = lambda f: [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]  # noqa: E731
    if args.cmd == "goals":
        rows = goals(read(args.prefs), args.dev)
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return print(f"{len(rows)} goals of {len(read(args.prefs))} preferences, {sum(r['split'] == 'dev' for r in rows)} dev")
    _, ver = load(args.device)
    gs = read(args.goals)
    if args.cmd == "gate":
        rows, sft = gate(ver, gs, read(args.edits), log=lambda m: print(m, flush=True))
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        if args.sft:
            Path(args.sft).write_text("".join(json.dumps(r) + "\n" for r in sft))
        return print(gate_summary(rows))
    if args.cmd == "sft":
        rows = sft(ver, gs, read(args.edits), args.min_reward, log=lambda m: print(m, flush=True))
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return print(f"{len(rows)} SFT rows (train goals' teacher answers above reward {args.min_reward:g})")
    by = {r["id"]: r for r in gs}
    with open(args.out, "w") as f:
        for a in read(args.answers):
            f.write(json.dumps({**a, **ver.answers(by[a["goal"]]["pref"], [a["answer"]], args.part)[0]}) + "\n")


if __name__ == "__main__":
    main()
