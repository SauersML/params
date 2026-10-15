"""The English reward of the goal doc (#2951, the user's 10-14 goal; the lead's 10-14): how well a separate reader,
given only the oracle's explanation, predicts what the target model does under interventions, and how many of the
invariants the explanation implies hold when tested on the target.

Interventions (one JSON object each), run on the target (vpd4l: VPD's subcomponents are the parts):
  {"kind": "swap", "pos": p, "token": t}                      the token at p replaced by t
  {"kind": "remove", "parts": [[site, pos, index], ...]}      the parts' masks set to 0 (VPD's ablation)
  {"kind": "scale", "parts": [...], "by": s}                  the parts' masks set to s (their rank-one weight times s)
  {"kind": "setback", "pos": p, "token": t, "parts": [...]}   the swap, with the parts' activations set back to their
                                                              values on the original text
An outcome: the target's most likely next token ("top") and the probability of the original prediction ("p").

reader reward: questions(): interventions drawn independently of any explanation (swaps at the last positions with
  fresh tokens from the model's own prediction there, removals and scalings of the circuit's parts singly and in
  threes, swaps with one part set back), so an explanation earns only what generalizes; the reader (a frozen open model
  on Tinker, never trained, never Claude) reads the text's end, the model's prediction, the explanation (or none) and
  the questions, and answers each with a token and a probability. Per question: top right (0/1) and p's credit
  max(0, 1 - |logit p_reader - logit p| / 4); reward = the mean of their average over questions less SIMPLICITY per
  1,000 reader tokens of explanation. Report each kind's share right against no explanation and another text's.
invariant checker: the reader writes invariants the explanation implies in a fixed grammar (INV lines: a swap over a
  set of tokens, removals, scalings or set-backs of named parts, and a predicted outcome); each is run on the target
  over every listed instance and holds when every instance meets its outcome; reward = the share of invariant lines
  that hold (lines that do not parse or name a part the model lacks fail).

  reader_reward.py questions --task text9000 --circuit STATED.json --out Q.jsonl [--seed 0] [--device D]
  reader_reward.py read --questions Q.jsonl --explanation E.txt|none --out R.json [--reader MODEL]
  reader_reward.py invariants --task text9000 --explanation E.txt|none --out I.json [--reader MODEL] [--device D]
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
sys.path[:0] = [str(HERE), str(HERE.parent / "vpd"), str(HERE / "rl")]

TEXTS = Path.home() / "mpd-data" / "graph_oracle" / "texts" / "vpd4l"
SIMPLICITY = 0.1  # reward per 1,000 reader tokens of explanation
READER = "Qwen/Qwen3.5-397B-A17B"  # tinker_train.REWARD_READER: the validated reader (10-12: ranked a correct, a wrong and
# another text's explanation in order on 15 of 16 texts)
LOGIT_SCALE = 4.0  # a probability's credit reaches 0 at this many nats of log-odds error
KIND = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "fc", "down_proj": "down"}
SITE = {v: k for k, v in KIND.items()}
PARTS = ("Parts are rank-one slices of the model's weight matrices, named L<layer>.<kind>#<index> (kind q, k, v, o: "
         "attention's query, key, value and output; fc, down: the MLP's input and output) at a token position (@<position>).")


def label(site: str, pos: int, index: int) -> str:
    return f"L{site.split('.')[1]}.{KIND[site.split('.')[-1]]}#{index}@{pos}"


def site_of(lab: str, where=None) -> tuple[str, int, int] | None:
    """(site, position, index) of 'L3.down#2481@394' or 'L3.down#2481@' creation'' (where: a position's token text to
    its position), or None."""
    m = re.fullmatch(r"L(\d+)\.(q|k|v|o|fc|down)#(\d+)@(.+)", lab.strip())
    if not m:
        return None
    layer, kind, idx, at = m.groups()
    pos = where(at) if where else (int(at) if at.isdigit() else None)
    if pos is None:
        return None
    return (f"h.{layer}.{'mlp' if kind in ('fc', 'down') else 'attn'}.{SITE[kind]}", int(pos), int(idx))


class VPDTarget:
    """vpd4l on one text: interventions run in torch (native's model and VPD's masks)."""

    def __init__(self, task: str, device: str | None = None):
        import torch

        import mech
        import native

        self.torch = torch
        self.nat = native.Native(device)
        t = json.loads((TEXTS / f"{task}.json").read_text())
        p = t["prompts"][0]
        self.task, self.ids, self.t = task, list(p["token_ids"]), int(p["target_positions"][0])
        self.tk = mech.tokenizer("vpd4l")
        self.base = self.logp(self.ids)
        self.y0 = int(self.base.argmax())
        self.a0 = self._acts(self.ids)

    def tok(self, i: int) -> str:
        return self.tk.decode([int(i)])

    def where(self, at: str) -> int | None:
        """A position written as its number or as its token quoted (the last position holding that token)."""
        at = at.strip()
        if at.isdigit():
            return int(at) if int(at) < len(self.ids) else None
        try:
            s = ast.literal_eval(at)
        except (ValueError, SyntaxError):
            return None
        hits = [p for p in range(len(self.ids)) if self.tok(self.ids[p]) == s] or \
            [p for p in range(len(self.ids)) if self.tok(self.ids[p]).strip() == str(s).strip()]  # the leading space left out
        return hits[-1] if hits else None

    def valid(self, part) -> bool:
        """Whether the model has this part: (site, position, index)."""
        n, p, c = part
        return n in self.nat.C and 0 <= c < self.nat.C[n] and 0 <= p < len(self.ids)

    def token_id(self, s: str) -> int | None:
        e = self.tk.encode(s).ids
        return e[0] if len(e) == 1 else None

    def _acts(self, seq):
        nat, torch = self.nat, self.torch
        nat.vpd.clear()
        for n in nat.names:
            nat.target.site(n).cache_input = True
        with torch.no_grad():
            nat.target(torch.tensor([seq], device=nat.dev))
        a = {n: (nat.target.site(n).last_input[0] @ nat.target.site(n).V) for n in nat.names}
        for n in nat.names:
            nat.target.site(n).cache_input = False
        nat.vpd.clear()
        return a

    def logp(self, seq, masks: dict | None = None, patch: dict | None = None):
        """log p at the target: the model, parts' masks set (masks: {(site, pos, idx): value}), or activations patched
        (patch: {site: [(pos, idx, value)]})."""
        import types

        nat, torch = self.nat, self.torch
        x = torch.tensor([seq], device=nat.dev)
        T = len(seq)
        with torch.no_grad():
            nat.vpd.clear()
            if masks:
                mk = {n: torch.ones((1, T, nat.target.site(n).V.shape[1]), device=nat.dev) for n in nat.names}
                for (n, p, c), v in masks.items():
                    mk[n][0, p, c] = v
                out = nat.vpd.masked(x, mk, {n: torch.ones((1, T), device=nat.dev) for n in nat.names})
            elif patch:
                saved = {}
                for n, items in patch.items():
                    st = nat.target.site(n)
                    saved[n] = st._forward

                    def fwd(self_, xx, items=items, st=st):
                        a = xx @ st.V
                        for p, c, v in items:
                            a[:, p, c] = v
                        return a @ st.U + xx @ st.delta_T

                    st._forward = types.MethodType(fwd, st)
                try:
                    out = nat.target(x)
                finally:
                    for n, f in saved.items():
                        nat.target.site(n)._forward = f
            else:
                out = nat.target(x)
        nat.vpd.clear()
        return torch.log_softmax(out[0, self.t].float(), -1)

    def run(self, iv: dict) -> dict:
        seq = list(self.ids)
        if iv["kind"] in ("swap", "setback"):
            seq[iv["pos"]] = self.token_id(iv["token"])
        parts = [tuple(x) for x in iv.get("parts", [])]
        if iv["kind"] == "remove":
            lp = self.logp(seq, masks={p: 0.0 for p in parts})
        elif iv["kind"] == "scale":
            lp = self.logp(seq, masks={p: float(iv["by"]) for p in parts})
        elif iv["kind"] == "setback":
            patch: dict = {}
            for n, p, c in parts:
                patch.setdefault(n, []).append((p, c, self.a0[n][p, c]))
            lp = self.logp(seq, patch=patch)
        else:
            lp = self.logp(seq)
        return {"top": self.tok(int(lp.argmax())), "p": float(lp[self.y0].exp())}

    def fresh(self, pos: int, n: int, rng: random.Random) -> list[str]:
        """n tokens the model itself predicts at pos (sampled by its probabilities, the text's own left out), single
        tokens that decode back to themselves."""
        lp = self.nat.reference([self.ids[:pos]], [pos - 1])[0, 0].cpu().double()
        w = lp.exp()
        w[self.ids[pos]] = 0.0
        gen = self.torch.Generator().manual_seed(rng.randrange(2 ** 31))
        out = []
        for z in self.torch.multinomial(w, 4 * n, replacement=False, generator=gen).tolist():
            s = self.tok(z)
            if self.token_id(s) == z and s.strip() and len(out) < n:
                out.append(s)
        return out

    def words(self, iv: dict) -> str:
        """The intervention in words for the reader."""
        def at(p):
            return f"position {p} ({self.tok(self.ids[p])!r})"
        names = ", ".join(label(*x) if len(x) == 3 and isinstance(x[0], str) else str(x) for x in
                          [(n, p, c) for n, p, c in iv.get("parts", [])])
        if iv["kind"] == "swap":
            return f"the token at {at(iv['pos'])} replaced by {iv['token']!r}"
        if iv["kind"] == "remove":
            return f"part{'s' if len(iv['parts']) > 1 else ''} {names} removed{' together' if len(iv['parts']) > 1 else ''}"
        if iv["kind"] == "scale":
            return f"part{'s' if len(iv['parts']) > 1 else ''} {names} scaled by {iv['by']:g} (its weight times {iv['by']:g})"
        return (f"the token at {at(iv['pos'])} replaced by {iv['token']!r}, with part{'s' if len(iv['parts']) > 1 else ''} "
                f"{names} set back to its value on the original text")

    def head(self, n_tokens: int = 24) -> str:
        """The text's end, its last tokens by position, and the model's prediction now."""
        T = len(self.ids)
        top = self.base.exp().topk(3)
        pred = ", ".join(f"{self.tok(int(i))!r} ({float(v):.2f})" for v, i in zip(top.values, top.indices))
        toks = " ".join(f"{p}:{self.tok(self.ids[p])!r}" for p in range(max(0, T - n_tokens), T))
        end = self.tk.decode(self.ids[max(0, T - 60):])
        return (f"A small language model (4 transformer layers, trained on web text) reads a text and predicts the next token "
                f"after its last token (position {T - 1}).\nThe text ends: ...{end}\nIts last tokens by position: {toks}\n"
                f"The model's prediction now: {pred}.\n{PARTS}\n")


def questions(tg: VPDTarget, circuit: list, seed: int = 0) -> list[dict]:
    """Interventions independent of any explanation (module docstring), each with its measured outcome."""
    rng = random.Random(seed)
    t = tg.t
    ivs = []
    for pos, n in ((t, 8), (t - 1, 4), (t - 2, 3), (t - 3, 3)):
        ivs += [{"kind": "swap", "pos": pos, "token": s} for s in tg.fresh(pos, n, rng)]
    parts = [list(x) for x in circuit]
    for x in rng.sample(parts, min(8, len(parts))):
        ivs.append({"kind": "remove", "parts": [x]})
    for _ in range(3):
        ivs.append({"kind": "remove", "parts": rng.sample(parts, 3)})
    at_t = [x for x in parts if x[1] == t] or parts
    for s in tg.fresh(t, 4, rng):
        ivs.append({"kind": "setback", "pos": t, "token": s, "parts": [rng.choice(at_t)]})
    for x in rng.sample(parts, min(3, len(parts))):
        ivs.append({"kind": "scale", "parts": [x], "by": 2.0})
    for iv in ivs:
        iv["outcome"] = tg.run(iv)
        iv["words"] = tg.words(iv)
    return ivs


def ask(tg_head: str, explanation: str | None, ivs: list[dict], answer_token: str) -> str:
    q = "\n".join(f"Q{k}: {iv['words']}" for k, iv in enumerate(ivs, 1))
    ex = f"An explanation of how the model computes this prediction:\n{explanation}\n\n" if explanation else ""
    return (f"{tg_head}\n{ex}For each change below, predict the model's most likely next token after the change and the "
            f"probability it then gives {answer_token!r}. Answer one line per question, Q<k>: '<token>' <probability>, "
            f"the token quoted exactly as written (with its leading space if any), nothing else.\n\n{q}")


def parse_answers(text: str, n: int) -> list[tuple[str | None, float | None]]:
    got: dict = {}
    for line in text.splitlines():
        m = re.match(r"\s*Q(\d+)\s*:\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*,?\s*([0-9]*\.?[0-9]+(?:[eE]-?\d+)?)", line)
        if m:
            try:
                tok = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                tok = None
            got.setdefault(int(m.group(1)), (tok, min(1.0, max(0.0, float(m.group(3))))))
    return [got.get(k, (None, None)) for k in range(1, n + 1)]


def logit(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, p))
    return math.log(p / (1 - p))


def grade(answers: list, ivs: list[dict]) -> list[dict]:
    out = []
    for (tok, p), iv in zip(answers, ivs):
        top_ok = tok is not None and tok == iv["outcome"]["top"]
        pc = 0.0 if p is None else max(0.0, 1 - abs(logit(p) - logit(iv["outcome"]["p"])) / LOGIT_SCALE)
        out.append({"kind": iv["kind"], "top_ok": top_ok, "p_credit": pc, "score": 0.5 * top_ok + 0.5 * pc,
                    "reader": [tok, p], "measured": iv["outcome"]})
    return out


class Reader:
    """A frozen open model on Tinker: greedy answers, tokens and dollars counted (tinker_train.Spend)."""

    def __init__(self, base: str, log: Path):
        import tinker

        import tinker_train

        self.tinker, self.base = tinker, base
        service = tinker_train.service_client()
        self.renderer, self.tokenizer = tinker_train.renderer_for(base)
        self.sampler = service.create_sampling_client(base_model=base)
        log.mkdir(parents=True, exist_ok=True)
        self.spend = tinker_train.Spend(log, base)
        self.tt = tinker_train

    def __call__(self, prompts: list[str], max_tokens: int = 1500) -> list[str]:
        import components

        params = self.tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=self.renderer.get_stop_sequences())
        futs = []
        for text in prompts:
            prompt = self.renderer.build_generation_prompt([{"role": "user", "content": components.escape(text)}])
            futs.append((prompt.length, self.sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)))
        out = []
        for n, fut in futs:
            toks = fut.result().sequences[0].tokens
            text, _ = self.tt.answer_of(self.renderer, self.tokenizer, toks)
            self.spend.add("reader_reward", prefill=n, sampled=len(toks))
            out.append(text)
        return out

    def tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


INV_GRAMMAR = """Write {n} invariants this explanation implies: tests on the model, each with the outcome it predicts. One per line, in exactly this grammar:
INV: swap @<where> with any of ['<token>', '<token>', ...] -> <outcome>
INV: remove <part>[, <part> ...] -> <outcome>
INV: scale <part> by <factor> -> <outcome>
INV: swap @<where> with any of ['<token>', ...] and set back <part>[, <part> ...] -> <outcome>
where <where> is a position, written as its number or as the token there quoted (for example @' creation', the last position holding that token), <part> is written L<layer>.<kind>#<index>@<where>, and <outcome> is one of: top '<token>' | top not '<token>' | p > <number> | p < <number>, p being the probability the model gives {answer!r}. List at least five tokens in each "any of" set (each a single token, quoted exactly, with its leading space if any). Nothing else."""


def parse_inv(line: str, tg: VPDTarget) -> tuple[list[dict], tuple] | None:
    """An INV line as (its interventions, one per instance, and its outcome), or None."""
    m = re.match(r"\s*INV:\s*(.+?)\s*->\s*(.+?)\s*$", line)
    if not m:
        return None
    body, oc = m.groups()
    mo = re.fullmatch(r"top\s+(not\s+)?('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")|p\s*([<>])\s*([0-9]*\.?[0-9]+)", oc)
    if not mo:
        return None
    if mo.group(2):
        try:
            outcome = ("top_not" if mo.group(1) else "top", ast.literal_eval(mo.group(2)))
        except (ValueError, SyntaxError):
            return None
    else:
        outcome = (mo.group(3), float(mo.group(4)))

    def parts_of(s):
        ps = [site_of(x, tg.where) for x in re.split(r",\s*(?=L\d)", s.strip())]
        return None if not ps or any(p is None or not tg.valid(p) for p in ps) else [list(p) for p in ps]

    ms = re.fullmatch(r"swap\s+@(\d+|'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s+with any of\s+\[(.*)\](?:\s+and set back\s+(.+))?", body)
    if ms:
        pos = tg.where(ms.group(1))
        if pos is None:
            return None
        try:
            toks = ast.literal_eval("[" + ms.group(2) + "]")
        except (ValueError, SyntaxError):
            return None
        if not toks or any(not isinstance(x, str) or tg.token_id(x) is None for x in toks):
            return None
        if ms.group(3):
            ps = parts_of(ms.group(3))
            if ps is None:
                return None
            return [{"kind": "setback", "pos": pos, "token": x, "parts": ps} for x in toks], outcome
        return [{"kind": "swap", "pos": pos, "token": x} for x in toks], outcome
    mr = re.fullmatch(r"remove\s+(.+)", body)
    if mr:
        ps = parts_of(mr.group(1))
        return ([{"kind": "remove", "parts": ps}], outcome) if ps else None
    mc = re.fullmatch(r"scale\s+(.+?)\s+by\s+([0-9]*\.?[0-9]+)", body)
    if mc:
        ps = parts_of(mc.group(1))
        return ([{"kind": "scale", "parts": ps, "by": float(mc.group(2))}], outcome) if ps else None
    return None


def meets(o: dict, outcome: tuple) -> bool:
    k, v = outcome
    if k == "top":
        return o["top"] == v
    if k == "top_not":
        return o["top"] != v
    return o["p"] > v if k == ">" else o["p"] < v


def explanation_of(path: str) -> str | None:
    return None if path == "none" else Path(path).read_text().strip()


def cmd_questions(a) -> None:
    tg = VPDTarget(a.task, a.device)
    circ = json.loads(a.circuit.read_text())
    parts = [(n, int(p), int(c)) for n, c, p in circ["nodes"] if n.startswith("h.")]
    ivs = questions(tg, parts, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps({"task": a.task, "head": tg.head(), "answer": tg.tok(tg.y0), **iv}) + "\n" for iv in ivs))
    print(len(ivs), "questions to", a.out)


def cmd_read(a) -> None:
    ivs = [json.loads(x) for x in a.questions.read_text().splitlines()]
    rd = Reader(a.reader, a.out.parent)
    ex = explanation_of(a.explanation)
    got = rd([ask(ivs[0]["head"], ex, ivs, ivs[0]["answer"])])[0]
    rows = grade(parse_answers(got, len(ivs)), ivs)
    n_ex = rd.tokens(ex) if ex else 0
    acc = sum(r["score"] for r in rows) / len(rows)
    kinds = sorted({r["kind"] for r in rows})
    res = {"explanation": a.explanation, "reader": a.reader, "reward": acc - SIMPLICITY * n_ex / 1000, "accuracy": acc,
           "explanation_tokens": n_ex, "spend_usd": rd.spend.total, "raw": got,
           "by_kind": {k: {"n": sum(r["kind"] == k for r in rows), "top_right": sum(r["top_ok"] for r in rows if r["kind"] == k),
                           "p_credit": round(sum(r["p_credit"] for r in rows if r["kind"] == k) / sum(r["kind"] == k for r in rows), 3)}
                       for k in kinds}, "rows": rows}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("explanation", "reward", "accuracy", "explanation_tokens", "spend_usd", "by_kind")}))


def cmd_invariants(a) -> None:
    tg = VPDTarget(a.task, a.device)
    rd = Reader(a.reader, a.out.parent)
    ex = explanation_of(a.explanation)
    exs = f"An explanation of how the model computes this prediction:\n{ex}\n\n" if ex else ""
    got = rd([f"{tg.head()}\n{exs}{INV_GRAMMAR.format(n=a.n, answer=tg.tok(tg.y0))}"])[0]
    rows = []
    for line in [x for x in got.splitlines() if x.strip().startswith("INV")]:
        p = parse_inv(line, tg)
        if p is None:
            rows.append({"line": line, "parsed": False, "holds": False})
            continue
        ivs, outcome = p
        outs = [tg.run(iv) for iv in ivs]
        ok = [meets(o, outcome) for o in outs]
        rows.append({"line": line, "parsed": True, "holds": all(ok), "instances": len(ok), "met": sum(ok),
                     "outcomes": [[iv.get("token"), o["top"], round(o["p"], 4)] for iv, o in zip(ivs, outs)]})
    share = sum(r["holds"] for r in rows) / max(len(rows), 1)
    res = {"explanation": a.explanation, "reader": a.reader, "share_holding": share, "lines": len(rows),
           "parsed": sum(r["parsed"] for r in rows), "spend_usd": rd.spend.total, "rows": rows, "raw": got}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("explanation", "share_holding", "lines", "parsed", "spend_usd")}))
    for r in rows:
        print(("HOLDS " if r["holds"] else "fails ") + r["line"][:150], r.get("met"), "/", r.get("instances"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    q = sub.add_parser("questions")
    q.add_argument("--task", default="text9000")
    q.add_argument("--circuit", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("read")
    r.add_argument("--questions", type=Path, required=True)
    r.add_argument("--explanation", required=True)
    r.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("invariants")
    i.add_argument("--task", default="text9000")
    i.add_argument("--explanation", required=True)
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--n", type=int, default=8)
    for p in (r, i):
        p.add_argument("--reader", default=READER)
    for p in (q, i):
        p.add_argument("--device")
    a = ap.parse_args()
    {"questions": cmd_questions, "read": cmd_read, "invariants": cmd_invariants}[a.what](a)


if __name__ == "__main__":
    main()
