"""Unit circuits, the oracle's answer measured by the carry test (#2951 graph oracle).

An answer is plain text (nothing in it is executed): an English explanation, then one line per input position of the
text, the units through which a change of that position's token reaches the prediction:

  EXPLANATION: the model copies the digits after the earlier "327" ...
  POSITION 322: L0 322:fc,down | L2 322:v[h3,h4] -> 332:o[h3,h4]

A unit is a weight matrix (q, k, v, o, fc, down) of a layer at a position, for attention optionally some of its heads
([h<i>,...]: the head's slice of the activation, carry.UnitSpace). A line is the path layer by layer: in each layer the
positions holding its units, joined by "->" from the positions an attention layer reads to the one it writes at. An answer is measured by the
carry test (carry.py): for each of the text's K positions where a changed token moves the prediction most
(Native.sensitivity), the prompts changing that token to the model's 3 likeliest others there (carry.alternatives),
every unit outside the answer's line for that position held at its value on the text and the line's units computing
on the changed prompt; the KL of the model on the changed prompt from that run, against the KL with no units (how far
the model moves). Its score: the share of the movement it leaves uncarried, the positions weighted by their movement
(a position without a line carries nothing); its size: the number of units (a head counts as one).

  units.py score ANSWER.txt TASK_ID [--k 8]
  units.py baseline copy|all TASK_ID [--k 8]                  a search-free answer's score
  units.py targets EXPLAIN.jsonl OUT_DIR [--fraction 90]     answers from carry.py explain's measured circuits
  units.py dataset EXPLAIN.jsonl OUT.jsonl [--fraction 90]   SFT rows (question, measured answer)
  units.py events SPLIT OUT.jsonl [--n 50] [--k 8]            the English reader's questions, answered by the model

This track is a scaffold: units are whole matrices or heads at positions, and a line is a set, not a graph. It exists to
get a model to state measured mechanism-shaped answers with a verifier it cannot fool. Climb-back criterion: once the
trained explainer leaves less movement uncarried than the copy baseline at no more units on held-out texts, the unit
space moves to VPD's subcomponents at the same positions (same carry test), and lines state attention edges.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KINDS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj", "fc": "c_fc", "down": "down_proj"}
CODES = {v: k for k, v in KINDS.items()}
LINE = re.compile(r"^\s*POSITION\s+(\d+)\s*:(.*)$", re.M)
LAYER = re.compile(r"L(\d+)\s+(.*)")
GROUP = re.compile(r"(\d+)\s*:\s*((?:(?:q|k|v|o|fc|down)(?:\[[h\d,\s]+\])?\s*,?\s*)+)")
KIND = re.compile(r"(q|k|v|o|fc|down)(?:\[([h\d,\s]+)\])?")


def parse(text: str) -> tuple[str, dict]:
    """(the explanation, {position: [unit, ...]}) of an answer; a unit is (matrix name, position) or (matrix name,
    head, position). A line is layer segments separated by "|": "L<layer> <position>:<kinds>", where "->" joins the
    positions an attention layer reads from to the one it writes at. Lines that name no unit are kept empty; text that
    is not a unit is ignored."""
    import native

    m = re.search(r"EXPLANATION:\s*(.*?)(?=^\s*POSITION\s|\Z)", text, re.S | re.M)
    explanation = m[1].strip() if m else ""
    circuits = {}
    for line in LINE.finditer(text):
        units = []
        for segment in line[2].split("|"):
            seg = LAYER.match(segment.strip())
            if not seg:
                continue
            layer = int(seg[1])
            for group in seg[2].split("->"):
                g = GROUP.search(group)
                if not g:
                    continue
                pos = int(g[1])
                for k in KIND.finditer(g[2]):
                    name = native.site_name(layer, KINDS[k[1]])
                    heads = [int(h) for h in re.findall(r"\d+", k[2])] if k[2] else None
                    units += [(name, h, pos) for h in heads] if heads and ".attn." in name else [(name, pos)]
        circuits.setdefault(int(line[1]), []).extend(units)
    return explanation, circuits


def render(explanation: str, circuits: dict) -> str:
    """The answer text of circuits {position: [unit, ...]}: per line, one segment per layer, its units grouped by position
    (earliest first, joined by "->": an attention layer reads the earlier positions and writes at the later), the kinds
    of one position in the order q, k, v, o, fc, down, a matrix's heads written together."""
    order = list(KINDS)
    out = [f"EXPLANATION: {explanation}".rstrip()]
    for p in circuits:  # in the order given (importance: a cut-off answer loses the least important lines)
        by = {}  # layer -> position -> kind -> heads (None: the whole matrix)
        for u in circuits[p]:
            name, pos = u[0], u[-1]
            kinds = by.setdefault(int(name.split(".")[1]), {}).setdefault(pos, {})
            kind = CODES[name.split(".")[-1]]
            if len(u) == 3:
                if kinds.get(kind, []) is not None:
                    kinds.setdefault(kind, []).append(u[1])
            else:
                kinds[kind] = None
        segments = []
        for layer in sorted(by):
            groups = []
            for pos in sorted(by[layer]):
                ks = by[layer][pos]
                words = [k if ks[k] is None else f"{k}[{','.join(f'h{h}' for h in sorted(set(ks[k])))}]" for k in order if k in ks]
                groups.append(f"{pos}:" + ",".join(words))
            segments.append(f"L{layer} " + " -> ".join(groups))
        out.append(f"POSITION {p}: " + (" | ".join(segments) or "none"))
    return "\n".join(out)


def _active(space, units):
    """The space's coefficient vector with these units on (a whole attention matrix in a head space: all its heads)."""
    import torch

    on = torch.zeros(space.size, dtype=torch.bool, device=space.nat.dev)
    want = set(units)
    for j, u in enumerate(space.units):
        if u in want or (len(u) == 3 and (u[0], u[2]) in want):
            on[j] = True
    return on


def sensitive(nat, ids: list[int], targets: list[int], k: int) -> list[int]:
    """The k positions (not the targets) whose changed token moves the prediction most, most first: carry.explain's
    choice (Native.sensitivity under the text's seed)."""
    import torch

    import native

    if max(targets) < 1:
        return []
    with torch.no_grad():
        moved, _ = nat.sensitivity(ids, targets, torch.Generator(device="cpu").manual_seed(native.task_seed(str(ids[:8]))))
    return [p for p in (moved.argsort(descending=True, stable=True) + 1).tolist() if p not in targets][:k]


def score(nat, task: dict, text: str, k: int = 8) -> dict:
    """score_many for one answer."""
    return score_many(nat, task, [text], k)[0]


def score_many(nat, task: dict, texts: list[str], k: int = 8) -> list[dict]:
    """The carry test of answers to one task, the task prepared once and every answer measured in one batch per position.
    Per answer: {"uncarried": the share of the empty run's discrepancy the answer leaves, positions weighted by it, "units":
    distinct (matrix, position) pairs the answer names that the model has (a head subset counts as its matrix: the same
    intervention always has the same size), "heads": head slices named, "unknown": named units the model does not have,
    "rest_carried", per position {"position", "change" (bits: the model's own move, KL of its prediction on the changed
    prompts from its prediction on the text), "movement" (bits: the empty run's discrepancy, every unit held at the text
    while embeddings and VPD's remainder follow the changed prompt), "carried" (share of the movement the answer's units
    remove), "rest_carried" (share removed by every other unit at the answer's positions with the answer's held: near 0
    when the answer is the only path, its necessity), "units", "outside" (units before the position or after the target,
    which a change at the position cannot reach), "unknown"}, "explanation"}."""
    import torch

    import carry
    import native

    parsed = [parse(t) for t in texts]
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    t_max = max(targets)
    positions = sensitive(nat, ids, targets, k)
    per = [{"rows": [], "w": 0.0, "u": 0.0, "r": 0.0} for _ in texts]
    with torch.no_grad():
        base = nat.reference([ids], targets)[0]
    for p in positions:
        prompts = carry.alternatives(nat, ids, p)
        with torch.no_grad():
            changed = nat.reference(prompts, targets)
            change = float(carry.kl(changed, base.expand_as(changed)).mean())
        stated = [e[1].get(p, []) for e in parsed]
        inside = [[u for u in st if p <= u[-1] <= t_max] for st in stated]
        where = sorted({p, *targets, *(u[-1] for ins in inside for u in ins)})
        gran = "head" if any(len(u) == 3 for ins in inside for u in ins) else "matrix"
        space = carry.UnitSpace(nat, len(ids), where, gran)
        study = (carry.HeadCarry if gran == "head" else carry.MatrixCarry)(nat, ids, targets, prompts, space)
        known = {(u[0], u[-1]) for u in space.units}
        ons, rests, sizes = [], [], []
        for ins in inside:
            own = {p, *targets, *(u[-1] for u in ins)}
            at = torch.tensor([u[-1] in own for u in space.units], device=nat.dev)
            on = _active(space, ins)
            ons.append(on)
            rests.append(at & ~on)
            pairs = {(u[0], u[-1]) for u in ins}
            heads = {u for u in ins if len(u) == 3 and u[1] < nat.target.n_head}
            sizes.append((len(pairs & known), len(heads), len(pairs - known) + sum(1 for u in ins if len(u) == 3 and u[1] >= nat.target.n_head)))
        with torch.no_grad():
            kls = study.measure([_active(space, [])] + ons + rests).tolist()
        empty, n = kls[0], len(texts)
        for i in range(n):
            carried = 0.0 if empty <= 0 else max(0.0, min(1.0, 1 - kls[1 + i] / empty))
            rest = 0.0 if empty <= 0 else max(0.0, min(1.0, 1 - kls[1 + n + i] / empty))
            units, heads, unknown = sizes[i]
            per[i]["rows"].append({"position": p, "change": change, "movement": empty, "carried": carried, "rest_carried": rest,
                                   "units": units, "heads": heads, "outside": len(stated[i]) - len(inside[i]), "unknown": unknown})
            per[i]["w"] += empty
            per[i]["u"] += empty * (1 - carried)
            per[i]["r"] += empty * rest
        native.free(nat.dev)
    out = []
    for (explanation, _), a in zip(parsed, per):
        rows = a["rows"]
        out.append({"uncarried": a["u"] / a["w"] if a["w"] else 0.0, "units": sum(r["units"] for r in rows),
                    "heads": sum(r["heads"] for r in rows), "unknown": sum(r["unknown"] for r in rows),
                    "rest_carried": a["r"] / a["w"] if a["w"] else 0.0, "positions": rows, "explanation": explanation})
    return out


def _cut(t: int, s: int, layer: int):
    """A mode that keeps one layer's attention at position t from reading position s (every head; its weight there set to
    zero, the rest not renormalized). Native.run computes each layer's attention weights with one softmax, in layer order."""
    import torch
    from torch.overrides import TorchFunctionMode

    class Cut(TorchFunctionMode):
        calls = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if func is torch.Tensor.softmax and out.dim() == 4 and out.shape[-1] == out.shape[-2] > t:
                if self.calls == layer:
                    out = out.clone()
                    out[:, :, t, s] = 0.0
                self.calls += 1
            return out

    return Cut()


def events(nat, task: dict, k: int = 8) -> list[dict]:
    """The English reader's questions about a text (reader.Reader.bits), answered by the model: for the K positions
    where a changed token moves the prediction most and K other positions before the target (drawn with the text's
    seed), the token there replaced by the model's likeliest other token (carry.alternatives' first), and, for each
    layer, the attention at the target kept from reading the position (_cut); at the listed positions also the model's
    second likeliest other token. Each event: {"position", "new" (token id) and "alt" (0 or 1), or "layer", "listed",
    "down": whether the model's probability of its most likely next token goes down, "weight": by how much it moves,
    |log2 of its ratio| capped at 4 bits (a change that moves nothing hardly counts), "bits": the signed log2 ratio}.
    The oracle reads the listed positions' first-token and cut events (evidence()); the reader is asked the rest
    (reader_events())."""
    import math
    import random

    import torch

    import carry
    import native

    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    t = max(targets)

    def p_top(x, mode=None):
        with torch.no_grad():
            if mode is None:
                return nat.run(x, [t], nat._full_plan(len(x), 1), *nat._one())[0, 0].exp()
            with mode:
                return nat.run(x, [t], nat._full_plan(len(x), 1), *nat._one())[0, 0].exp()

    base = p_top(ids)
    top = int(base.argmax())
    p0 = float(base[top])
    near = sensitive(nat, ids, targets, k)
    rest = [q for q in range(1, t) if q not in near]
    others = random.Random(native.task_seed(task["id"])).sample(rest, min(k, len(rest)))
    def event(q, p, **kind):
        bits = math.log2(max(p, 1e-30) / p0)
        return {"position": q, **kind, "down": bits < 0, "weight": min(abs(bits), 4.0), "bits": bits}

    out = []
    for q in near + others:
        alts = carry.alternatives(nat, ids, q)
        listed = q in near
        out.append(event(q, float(p_top(alts[0])[top]), new=alts[0][q], alt=0, listed=listed))
        if listed and len(alts) > 1:  # a second token at a listed position: only the reader is asked about it
            out.append(event(q, float(p_top(alts[1])[top]), new=alts[1][q], alt=1, listed=True))
        for layer in range(nat.target.n_layer):
            out.append(event(q, float(p_top(ids, _cut(t, q, layer))[top]), layer=layer, listed=listed))
    native.free(nat.dev)
    return out


def targets(explain_jsonl: Path, out_dir: Path, fraction: int = 90) -> int:
    """One answer per question of carry.py explain's output: for each position, the smallest measured unit set on its
    elimination curve that carries `fraction`% of what the position's full unit set carries (a position whose full set
    carries less than all of its movement needs units elsewhere too; its line then states the part that is at the
    position and the targets), no explanation (the English is trained by its own reader). Positions whose full set
    carries nothing get no line; positions moving less than 1% of the text's total movement get an empty line ("none":
    they can change the movement-weighted score by at most 1%, and a 90% set of almost nothing is noise)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in open(explain_jsonl):
        r = json.loads(line)
        circuits = {}
        total = sum(max(p["empty_bits"], 0.0) for p in r["positions"])
        for p in r["positions"]:
            curve, sets, empty = p["curve"], p["unit_sets"], p["empty_bits"]
            full = min(kl for _, kl in curve)
            if empty < 0.01 * total:
                circuits[p["position"]] = []
                continue
            if empty <= 0 or full >= empty:
                continue
            goal = empty - fraction / 100 * (empty - full)
            fits = [i for i, (_, kl) in enumerate(curve) if kl <= goal]
            best = sets[max(fits, key=lambda i: -curve[i][0])]  # the fewest units within the goal
            circuits[p["position"]] = [tuple(u) for u in best]
        (out_dir / f"{r['task']}.txt").write_text(render("", circuits))
        n += 1
    return n


ASK = ("How does {model} compute its prediction of the next token after the target of this text? For each listed input "
       "position, give the path by which a change of the token there reaches the prediction, as units: a unit is a weight "
       "matrix (q, k, v, o, fc, down) of a layer at a position, an attention matrix optionally by its heads (v[h3,h4]). "
       "Write a path layer by layer, separated by \" | \": L<layer> <position>:<kinds>, and for an attention layer that "
       "reads one position and writes at a later one, L<layer> <position>:<kinds> -> <later position>:<kinds>. Answer "
       "with a line EXPLANATION: (in plain English, what the model does), then one line POSITION <p>: <path> per listed "
       "position. Fewer units are better, as long as they carry the change.")


def numbered(strings: list[str]) -> str:
    """Tokens with their positions, 0:'The' 1:' cat' ...: a reader finds position 322 without counting."""
    return " ".join(f"{i}:{s!r}" for i, s in enumerate(strings))


def reader_events(events: list[dict]) -> list[dict]:
    """The events the English reader is asked about: none the oracle read as evidence (other positions, and the second
    token at listed positions), so English is credited only for what it lets a reader predict beyond its own input."""
    return [e for e in events if not e.get("listed") or e.get("alt") == 1]


def evidence(events: list[dict], positions: list[int], strings: list[str]) -> list[str]:
    """The measurements the oracle reads for each listed position (events() at those positions): how the log2
    probability of the model's most likely next token moves when the token there changes to the model's likeliest
    other token, and when each layer's attention at the target is kept from reading the position. The English reader is
    never asked about these (its questions are about other positions and other tokens)."""
    by = {}
    for e in events:
        by.setdefault(e["position"], []).append(e)
    out = []
    for q in positions:
        es = by.get(q, [])
        change = next((e for e in es if "new" in e and e.get("alt", 0) == 0), None)
        cuts = sorted((e for e in es if "layer" in e), key=lambda e: e["layer"])
        if change is None and not cuts:
            continue
        line = f"{q} ({strings[q]!r}):"
        if change is not None:
            line += f" changed token {change['bits']:+.2f}"
        if cuts:
            line += "; attention cut per layer " + " ".join(f"L{e['layer']} {e['bits']:+.2f}" for e in cuts)
        out.append(line)
    return out


def question(task: dict, positions: list[int], events: list[dict] | None = None) -> str:
    """The oracle's question for unit circuits: the ask, the text's tokens by position, the target and the model's most
    probable next tokens there, the positions whose changed token moves the prediction most (with their tokens), and,
    given the text's events, the measured evidence at those positions (evidence())."""
    from prompt import tokens_of

    model = task["model"]
    p = task["prompts"][0]
    strings = tokens_of(model, p["token_ids"])
    t = max(p["target_positions"])
    top = ", ".join(f"{tok!r} {q:.2f}" for tok, q in p["model_top"][0][:3])
    lines = [ASK.format(model=model), "tokens: " + numbered(strings),
             f"target: position {t}; the model's most probable next tokens there: {top}",
             "positions where a changed token moves the prediction most, most first: "
             + ", ".join(f"{q} ({strings[q]!r})" for q in positions)]
    if events:
        lines += ["measured at those positions (log2 change of the top token's probability):"] + evidence(events, positions, strings)
    return "\n".join(lines)


def baseline(kind: str, positions: list[int], targets: list[int]) -> str:
    """Answers that need no search, for comparison: "copy", the recurring copy circuit at every position (layer-0 MLP
    at the position, layer-2 values there, layer-2 output at the target); "all", every unit at the position and the
    target."""
    import native

    t = max(targets)
    circuits = {}
    for p in positions:
        if kind == "copy":
            circuits[p] = [(native.site_name(0, "c_fc"), p), (native.site_name(0, "down_proj"), p), (native.site_name(2, "v_proj"), p),
                           (native.site_name(2, "o_proj"), t)]
        else:
            circuits[p] = [(native.site_name(L, k), q) for L in range(4) for k in KINDS.values() for q in (p, t)]
    return render("", circuits)


def dataset(explain_jsonl: Path, out: Path, fraction: int = 90, events_jsonl: Path | None = None) -> int:
    """SFT rows {"prompt": [user message], "completion": [assistant message], "info": {"task"}} from carry.py
    explain's measured circuits (targets()'s answers) and their questions (with the evidence of events_jsonl, units.py
    events' output, when given; texts without events are skipped then)."""
    evs = {json.loads(x)["task"]: json.loads(x)["events"] for x in open(events_jsonl)} if events_jsonl else None
    import tempfile

    import native

    with tempfile.TemporaryDirectory() as tmp:
        targets(explain_jsonl, Path(tmp), fraction)
        n = 0
        with open(out, "w") as f:
            for line in open(explain_jsonl):
                r = json.loads(line)
                task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
                answer = (Path(tmp) / f"{r['task']}.txt").read_text()
                positions = [p["position"] for p in r["positions"]]
                if not positions:  # a text whose prediction is at its first position: nothing to explain
                    continue
                if evs is not None and r["task"] not in evs:
                    continue
                f.write(json.dumps({"prompt": [{"role": "user", "content": question(task, positions, evs and evs[r["task"]])}],
                                    "completion": [{"role": "assistant", "content": answer}], "info": {"task": r["task"]}}) + "\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("answer", type=Path)
    s.add_argument("task")
    s.add_argument("--k", type=int, default=8)
    d = sub.add_parser("dataset")
    d.add_argument("explain", type=Path)
    d.add_argument("out", type=Path)
    d.add_argument("--fraction", type=int, default=90)
    d.add_argument("--events", type=Path, help="units.py events output: the question shows the measured evidence")
    b = sub.add_parser("baseline")
    b.add_argument("kind", choices=("copy", "all"))
    b.add_argument("task")
    b.add_argument("--k", type=int, default=8)
    e = sub.add_parser("events")
    e.add_argument("split", choices=("train", "heldout", "hard"))
    e.add_argument("out", type=Path)
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--k", type=int, default=8)
    e.add_argument("--offset", type=int, default=0, help="this process's share of the first n texts: offset, offset + stride, ...")
    e.add_argument("--stride", type=int, default=1)
    t = sub.add_parser("targets")
    t.add_argument("explain", type=Path)
    t.add_argument("out", type=Path)
    t.add_argument("--fraction", type=int, default=90)
    a = ap.parse_args()
    if a.cmd == "targets":
        print(f"{targets(a.explain, a.out, a.fraction)} answers in {a.out}")
        return
    if a.cmd == "dataset":
        print(f"{dataset(a.explain, a.out, a.fraction, a.events)} rows in {a.out}")
        return
    import native

    nat = native.Native()
    if a.cmd == "events":  # one line per text: {"task", "events"}; resumable
        done = {json.loads(x)["task"] for x in open(a.out)} if a.out.exists() else set()
        with open(a.out, "a") as f:
            for path in native.tasks(a.split)[:a.n][a.offset::a.stride]:
                if path.stem in done:
                    continue
                f.write(json.dumps({"task": path.stem, "events": events(nat, json.loads(path.read_text()), a.k)}) + "\n")
                f.flush()
        return
    task = json.loads((native.TEXTS / "vpd4l" / f"{a.task}.json").read_text())
    if a.cmd == "baseline":
        p = task["prompts"][0]
        text = baseline(a.kind, sensitive(nat, p["token_ids"], p["target_positions"], a.k), p["target_positions"])
    else:
        text = a.answer.read_text()
    print(json.dumps(score(nat, task, text, a.k), indent=1))


if __name__ == "__main__":
    main()
