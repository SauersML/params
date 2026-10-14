"""A tool-use environment for the zoo v2 oracle (TOYLOOP lane): short queries instead of a 3k-token weights prompt.

One episode = one (zoo2 instance, text). The first observation is short: the text's tokens, the target position and the
unit names. The oracle then calls one tool per turn, as one line `CALL name(args)`, and finishes with `ANSWER` followed
by the Mermaid flowchart plus an "English:" line (circuit.parse reads it). Every fact is computed from the WEIGHTS (the
same facts render.weight_only_lines(detail=True) prints, parsed by weights_heuristic2.parse, so the no-LM rule that
recovers every circuit from the detailed prompt can be run through these tools) and the text; run() executes the model.
No tool reads the record's params or the per-text truth.

  marker()            the text's marker token and the detector (the layer-0 head most gate links come from; the gate
                      tokens are the source tokens it carries; the marker is the first text token among them)
  gated(tK)           the units whose gate link from the detector carries tK
  reads(u, ...)       each unit's in-links writer:gain (<src = the token the link mostly carries), from either end of
                      the link (render truncates each list), each writer annotated with weight facts: det (the
                      detector), gate tX (gated by tX), key tX in/absent (a content head keyed on tX: is tX in the
                      text?), ~X (cancelling twin of X), dead (nothing reads it)
  writes(u, ...)      each unit's readers reader:gain, or "nothing"
  twins(u, ...)       each unit's cancelling twin, or "none"
  fires(n, ...)       a neuron's pre-activation estimated from the weights as weights_heuristic2's firing check does
                      (bias + its own read of the last token + each pos-d head's term for the token d back): on / off
  run(u, ...)         execute a draft circuit: reward_exec.score's sufficiency and necessity (the text selection
                      reward_exec uses); costs RUN_COST per call in the episode reward

Episode reward = reward_exec of the answer - RUN_COST x run() calls. Tools take a comma list of units, so one call can
query a whole set.

  python3 -B experiments/oracle/toyloop/toy_tools.py demo                 # one expert episode, printed
  python3 -B experiments/oracle/toyloop/toy_tools.py expert --split train  # expert stats -> results/toy_tools_expert.json
  python3 -B experiments/oracle/toyloop/toy_tools.py episodes --out ~/mpd-data/toyloop/tool_episodes  # SFT data
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import circuit  # noqa: E402
import render  # noqa: E402
import weights_heuristic2 as wh  # noqa: E402

MANIFEST = Path.home() / "mpd-data/toy_zoo2/manifest.jsonl"
RUN_COST = 0.02
MAX_CALLS = 12
SYSTEM = "You find the circuit of a toy transformer on one text by calling tools, one call per turn."
TOOLS = ("marker", "gated", "reads", "writes", "twins", "fires", "run")
CALL = re.compile(r"^\s*CALL\s+(\w+)\s*\(([^)]*)\)\s*$")
UNIT = re.compile(r"L\d+[HN]\d+")
EMPTY = "flowchart LR\n  tokX --> out\nEnglish: No unit carries the prediction."


# ---------------------------------------------------------------- weight facts, per instance

_FACTS: dict = {}


def facts(toy) -> dict:
    """The instance's weight facts (text independent): {unit: wh.Unit} parsed from the detailed weight lines, every
    unit's merged in-links, the detector, {gated unit: gate token}. Cached per instance (texts share the weights)."""
    key = toy.meta.get("instance", toy.name)
    if key in _FACTS and _FACTS[key]["sd"] is toy.sd:
        return _FACTS[key]
    lines = render.weight_only_lines(toy, max_neurons=16, detail=True)
    units, _ = wh.parse("\n".join(["Weights:", *lines, "Texts:"]))
    gl = wh.gate_links(units)
    explicit = {u: x.gate for u, x in units.items() if x.gate}
    if explicit:
        gl = [(u, w, t) for u, w, t in gl if explicit.get(u) == t]
    det = wh.Counter(w for _, w, _ in gl).most_common(1)[0][0] if gl else None
    gate_of = {u: t for u, w, t in gl if w == det}
    ins = {u: [(w, g, src) for _, w, g, src in x.ins] for u, x in units.items()}
    for w, x in units.items():      # weights_heuristic2's refinement c: a link counts from either end
        for r, g in x.outs:
            r = r.split(".")[0]
            if r in ins and all(w != v for v, _, _ in ins[r]):
                ins[r].append((w, g, None))
    if len(_FACTS) > 32:
        _FACTS.clear()
    _FACTS[key] = {"sd": toy.sd, "units": units, "ins": ins, "det": det, "gate_of": gate_of,
                   "gate_tokens": sorted(set(gate_of.values()), key=lambda t: int(t[1:]) if t[1:].isdigit() else 0)}
    return _FACTS[key]


def _g(x: float) -> str:
    return f"{x:.2g}" if abs(x) < 1e4 else f"{x:.0f}"


# ---------------------------------------------------------------- the environment

class ToolEnv:
    def __init__(self, toy):
        self.toy = toy
        self.f = facts(toy)
        self.text = [toy.vocab[i] for i in toy.texts[0]]
        self.calls: list[str] = []
        self.runs = 0
        self.done = False
        self.answer: str | None = None

    # ---- observation 0
    def first(self) -> str:
        L, H, _ = render.heads_of(self.toy.cfg)
        return "\n".join([
            f"Toy transformer: {L} layers, {H} heads, vocab {len(self.toy.vocab)}.",
            f"Text: {' '.join(self.text)} (predict at position {len(self.text) - 1})",
            "Units: " + " ".join(self.f["units"]),
            "Tools: marker() gated(token) reads(units) writes(units) twins(units) fires(neurons) run(units). "
            "One line per turn: CALL name(args). Finish with ANSWER, then the Mermaid flowchart and an English: line."])

    # ---- tools
    def _units(self, arg: str) -> tuple[list, list]:
        names = [a.strip() for a in arg.replace(" ", ",").split(",") if a.strip()]
        return [n for n in names if n in self.f["units"]], [n for n in names if n not in self.f["units"]]

    def _note(self, w: str) -> str:
        """Weight facts about a writer, as reads() annotates it."""
        f, x = self.f, self.f["units"].get(w)
        if x is None:
            return ""
        notes = []
        if w == f["det"]:
            notes.append("det")
        if w in f["gate_of"]:
            notes.append(f"gate {f['gate_of'][w]}")
        if x.key is not None and w != f["det"]:
            notes.append(f"key {x.key} {'in' if x.key in self.text else 'absent'}")
        if x.cancels:
            notes.append(f"~{x.cancels}")
        if not x.outs:
            notes.append("dead")
        return f" [{', '.join(notes)}]" if notes else ""

    def t_marker(self, arg: str) -> str:
        f = self.f
        if f["det"] is None:
            return "no gate links in the weights"
        m = next((t for t in self.text if t in f["gate_tokens"]), None)
        pos = f" at position {self.text.index(m)}" if m else ""
        return (f"marker {m or 'none'}{pos}; detector {f['det']}; gate tokens {' '.join(f['gate_tokens'])}")

    def t_gated(self, arg: str) -> str:
        t = arg.strip()
        us = [u for u, g in self.f["gate_of"].items() if g == t]
        return f"gated by {t}: {' '.join(us) if us else 'none'}"

    def t_reads(self, arg: str) -> str:
        us, bad = self._units(arg)
        out = []
        for u in us:
            ws = sorted(self.f["ins"][u], key=lambda x: -x[1])
            items = [f"{w}:{_g(g)}" + (f"<{s}" if s else "") + self._note(w) for w, g, s in ws]
            out.append(f"{u} <- {', '.join(items) if items else 'nothing'}")
        return "\n".join(out + [f"unknown {b}" for b in bad]) or "no units given"

    def t_writes(self, arg: str) -> str:
        us, bad = self._units(arg)
        out = []
        for u in us:
            o = sorted(self.f["units"][u].outs, key=lambda x: -x[1])
            out.append(f"{u} -> {', '.join(f'{r}:{_g(g)}' for r, g in o) if o else 'nothing'}")
        return "\n".join(out + [f"unknown {b}" for b in bad]) or "no units given"

    def t_twins(self, arg: str) -> str:
        us, bad = self._units(arg)
        out = [f"{u} ~{self.f['units'][u].cancels}" if self.f["units"][u].cancels else f"{u} none" for u in us]
        return "; ".join(out + [f"unknown {b}" for b in bad]) or "no units given"

    def pre(self, u: str) -> float | None:
        """weights_heuristic2.rule's fires(): bias + own read of the last token + each pos-d head's term."""
        x, units, text = self.f["units"][u], self.f["units"], self.text
        if x.bias is None:
            return None
        p = x.bias + x.terms.get("self", {}).get(text[-1], 0.0)
        for w, kt in x.terms.items():
            o = units[w].offset if w in units else None
            if w != "self" and o is not None and len(text) > o:
                p += kt.get(text[-1 - o], 0.0)
        return p

    def t_fires(self, arg: str) -> str:
        us, bad = self._units(arg)
        out = []
        for u in us:
            p = self.pre(u) if "N" in u else None
            out.append(f"{u} is a head" if "H" in u else f"{u} unknown" if p is None else
                       f"{u} {'on' if p > 0 else 'off'} ({_g(p)})")
        return "; ".join(out + [f"unknown {b}" for b in bad]) or "no units given"

    def t_run(self, arg: str) -> str:
        import reward_exec
        us, bad = self._units(arg)
        self.runs += 1
        if bad:
            return f"unknown {' '.join(bad)}"
        s = reward_exec.score(self.toy, set(us))
        return f"suff {s['suff']:.2f} nec {s['nec']:.2f} (cost {RUN_COST} per run)"

    # ---- one turn
    def step(self, action: str) -> str:
        """The tool's observation, or "" once the episode ends (ANSWER, or MAX_CALLS calls)."""
        a = action.strip()
        if a.startswith("ANSWER"):
            self.answer, self.done = a[len("ANSWER"):].strip(), True
            return ""
        self.calls.append(a.splitlines()[0] if a else "")
        if len(self.calls) > MAX_CALLS:
            self.done = True
            return ""
        m = CALL.match(a.splitlines()[0]) if a else None
        if not m or m.group(1) not in TOOLS:
            return "error: use one line CALL name(args) with name in " + ", ".join(TOOLS) + ", or ANSWER"
        return getattr(self, "t_" + m.group(1))(m.group(2))

    def result(self) -> dict:
        """The answer's metrics (evaluate.metrics) plus calls, runs and the episode reward."""
        import evaluate
        comp = self.answer if self.answer is not None else ""
        r = evaluate.metrics(self.toy, comp)
        r.update(calls=len(self.calls), runs=self.runs, answered=float(self.answer is not None),
                 episode_reward=r["reward"] - RUN_COST * self.runs)
        return r


# ---------------------------------------------------------------- the scripted expert

def expert_actions(env: ToolEnv) -> tuple[list[str], set]:
    """weights_heuristic2.rule run through the tools: (the CALL lines in order, the rule's unit set). Each decision uses
    only what the previous observations said (parsed back from them), never the truth."""
    calls, obs = [], []

    def call(s):
        calls.append(s)
        o = env.step(s)
        obs.append(o)
        return o

    o = call("CALL marker()")
    m = re.match(r"marker (\S+)(?: at position \d+)?; detector (L\d+H\d+)", o)
    if not m or m.group(1) == "none":
        return calls, set()
    marker, det = m.group(1), m.group(2)
    o = call(f"CALL gated({marker})")
    gated = o.split(":", 1)[1].split() if ":" in o else []
    gated = [u for u in gated if UNIT.fullmatch(u)]
    if not gated:
        return calls, set()
    o = call(f"CALL twins({', '.join(gated)})")
    tw = {p.split()[0] for p in o.split("; ") if "~" in p}
    gated = [u for u in gated if u not in tw]
    # writes() would drop gated units nothing reads: none on the 300 training texts checked, so the expert skips it
    if not gated:
        return calls, set()
    neurons = [u for u in gated if "N" in u]
    off = set()
    if neurons:
        o = call(f"CALL fires({', '.join(neurons)})")
        off = {p.split()[0] for p in o.split("; ") if " off " in p}
    keep, frontier = {det} | set(gated), list(gated)
    while frontier:
        o = call(f"CALL reads({', '.join(frontier)})")
        new = []
        for line in o.splitlines():
            if " <- " not in line:
                continue
            for w, note in re.findall(r"(\w+):[^,\[ ]*(?: \[([^\]]*)\])?", line.split(" <- ", 1)[1]):
                note = note.split(", ") if note else []
                gates = [n.split()[1] for n in note if n.startswith("gate ")]
                if (not UNIT.fullmatch(w) or not w.startswith("L0") or "det" in note or "dead" in note
                        or any(n.startswith("~") or n.endswith(" absent") for n in note)
                        or any(g != marker for g in gates)):
                    continue
                if w not in keep:
                    keep.add(w)
                    new.append(w)
        frontier = [w for w in new if "N" in w]   # a layer-0 head reads only token identity: no call needed
    if off and off >= set(gated):
        return calls, set()
    return calls, keep - off


def answer_text(env: ToolEnv, units: set, edges=None) -> str:
    """ANSWER + Mermaid + a short English line built from the tools' facts."""
    if not units:
        return "ANSWER\n" + EMPTY
    det = env.f["det"]
    if edges is None:
        edges = [("tok_marker", det)] + [(det, u) for u in sorted(units) if u != det] + \
                [(u, "out") for u in sorted(units) if env.f["gate_of"].get(u)]
    gated = sorted(u for u in units if u in env.f["gate_of"])
    l0 = sorted(u for u in units if u != det and u not in env.f["gate_of"])
    marker = next((t for t in env.text if t in env.f["gate_tokens"]), "?")
    eng = (f"Detector {det} reads the marker {marker} and un-gates {', '.join(gated) or 'no unit'}"
           + (f", which read the always-on layer-0 units {', '.join(l0)}." if l0 else "."))
    return "ANSWER\nflowchart LR\n" + "\n".join(f"  {a} --> {b}" for a, b in edges) + f"\nEnglish: {eng}"


def perturb(env: ToolEnv, units: set, rng: random.Random) -> set | None:
    """A plausible wrong draft, built from weight facts only: an on gated neuron swapped for an off one of the same
    gate, a non-detector unit dropped, or the cancelling twin of a gated unit added. None when nothing applies."""
    f, det = env.f, env.f["det"]
    gate = {f["gate_of"].get(u) for u in units} - {None}
    on = [u for u in sorted(units) if u in f["gate_of"] and "N" in u]
    off = [u for u, g in sorted(f["gate_of"].items()) if g in gate and "N" in u and u not in units
           and (env.pre(u) or 0) <= 0]
    opts = []
    if on and off:
        opts.append(lambda: (units - {rng.choice(on)}) | {rng.choice(off)})
    if len(units) > 2:
        opts.append(lambda: units - {rng.choice(sorted(units - {det}))})
    tw = [f["units"][u].cancels for u in sorted(units) if u in f["units"] and f["units"][u].cancels]
    tw += [x.cancels for u, x in sorted(f["units"].items()) if f["gate_of"].get(u) in gate and x.cancels]
    if tw:
        opts.append(lambda: units | {rng.choice(sorted(set(tw)))})
    return rng.choice(opts)() if opts else None


def expert_episode(toy, fix: bool = True, verify: str | None = None) -> dict:
    """{"messages": chat turns, "rule_units", "truth_units", "calls", "rule_ok"}. fix: the final answer is the truth's
    circuit (its edges, the tools' English); the calls are the rule's either way. verify "check": the expert runs its
    draft once before answering; "revise": it first runs a perturbed draft (perturb), then its own, then answers."""
    env = ToolEnv(toy)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": env.first()}]
    env2 = ToolEnv(toy)    # replay the expert's calls to record the observations in order
    calls, units = expert_actions(env)
    if verify and units:
        wrong = perturb(env2, units, random.Random(toy.name)) if verify == "revise" else None
        drafts = ([wrong] if wrong else []) + [units]
        calls += [f"CALL run({', '.join(sorted(d))})" for d in drafts]
    for c in calls:
        msgs += [{"role": "assistant", "content": c}, {"role": "user", "content": env2.step(c)}]
    gold = {u for u in toy.truth["units"] if UNIT.fullmatch(u)}
    if fix:
        ans = answer_text(env2, gold, [tuple(e) for e in toy.truth["edges"]] if gold else None)
    else:
        ans = answer_text(env2, units)
    msgs.append({"role": "assistant", "content": ans})
    return {"name": toy.name, "messages": msgs, "rule_units": sorted(units), "truth_units": sorted(gold),
            "calls": len(calls), "rule_ok": units == gold}


# ---------------------------------------------------------------- data

def load(split: str, limit: int | None = None, seed: int = 0) -> list:
    """Every text (one Toy each) of a split's instances; limit: a seeded sample of that many texts."""
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    if limit:
        rows = random.Random(seed).sample(rows, min(len(rows), limit // 2 + 1))
    toys = [t for r in rows for t in render.from_zoo2(r, MANIFEST.parent)]
    return random.Random(seed).sample(toys, min(limit, len(toys))) if limit else toys


def main():
    global MANIFEST
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["demo", "expert", "episodes"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--zoo", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=Path.home() / "mpd-data/toyloop/tool_episodes")
    ap.add_argument("--verify", default="0,0", help="episodes: fractions of texts given a 'check' and a 'revise' run() "
                    "step (seeded per text), e.g. 0.15,0.25 (the run_episodes set)")
    a = ap.parse_args()
    MANIFEST = a.zoo.expanduser()
    if a.cmd == "demo":
        e = expert_episode(load(a.split, 4)[0])
        for m in e["messages"]:
            print(f"--- {m['role']}\n{m['content']}")
        print({k: v for k, v in e.items() if k != "messages"})
        return
    splits = [a.split] if a.cmd == "expert" else ["train", "test_instance", "test_family"]
    stats = {}
    for split in splits:
        t0, eps = time.time(), []
        fc, fr = (float(x) for x in a.verify.split(","))
        for t in load(split, a.limit):
            u = random.Random("verify" + t.name).random()
            e = expert_episode(t, verify="check" if u < fc else "revise" if u < fc + fr else None)
            gold = set(e["truth_units"])
            e["unit_f1"] = circuit.f1(set(e["rule_units"]), gold)
            e["family"] = t.meta["family"]
            eps.append(e)
        n = len(eps)
        fams = sorted({e["family"] for e in eps})
        stats[split] = {"n": n, "exact": round(sum(e["rule_ok"] for e in eps) / n, 3),
                        "unit_f1": round(sum(e["unit_f1"] for e in eps) / n, 3),
                        "calls_mean": round(sum(e["calls"] for e in eps) / n, 2),
                        "calls_max": max(e["calls"] for e in eps), "sec": round(time.time() - t0),
                        "per_family": {f: {"n": len(r), "exact": round(sum(e["rule_ok"] for e in r) / len(r), 3),
                                           "calls": round(sum(e["calls"] for e in r) / len(r), 2)}
                                       for f in fams for r in [[e for e in eps if e["family"] == f]]}}
        print(json.dumps({split: stats[split]}), flush=True)
        if a.cmd == "episodes":
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / f"{split}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    if (a.cmd == "expert" or a.limit is None) and a.verify == "0,0":
        out = HERE / "results" / "toy_tools_expert.json"
        old = json.loads(out.read_text()) if out.exists() else {}
        out.write_text(json.dumps({**old, **stats}, indent=1))


if __name__ == "__main__":
    main()
