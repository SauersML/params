"""Unit circuits, the oracle's answer measured by the carry test (#2951 graph oracle).

An answer is plain text (nothing in it is executed): an English explanation, then one line per input position of the
text, the units through which a change of that position's token reaches the prediction:

  EXPLANATION: the model copies the digits after the earlier "327" ...
  POSITION 322: L0.fc@322 L0.down@322 L2.v@322[h3,h4] L2.o@332[h3,h4]

A unit is a weight matrix of a layer at a position (L<layer>.<q|k|v|o|fc|down>@<position>), for attention optionally
some of its heads ([h<i>,...]: the head's slice of the activation, carry.UnitSpace). An answer is measured by the
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
UNIT = re.compile(r"L(\d+)\.(q|k|v|o|fc|down)@(\d+)(?:\[([h\d,\s]+)\])?")
LINE = re.compile(r"^\s*POSITION\s+(\d+)\s*:(.*)$", re.M)


def parse(text: str) -> tuple[str, dict]:
    """(the explanation, {position: [unit, ...]}) of an answer; a unit is (matrix name, position) or (matrix name,
    head, position). Lines that name no unit are kept empty; text that is not a unit is ignored."""
    import native

    m = re.search(r"EXPLANATION:\s*(.*?)(?=^\s*POSITION\s|\Z)", text, re.S | re.M)
    explanation = m[1].strip() if m else ""
    circuits = {}
    for line in LINE.finditer(text):
        units = []
        for u in UNIT.finditer(line[2]):
            name = native.site_name(int(u[1]), KINDS[u[2]])
            heads = [int(h) for h in re.findall(r"\d+", u[4])] if u[4] else None
            units += [(name, h, int(u[3])) for h in heads] if heads and ".attn." in name else [(name, int(u[3]))]
        circuits.setdefault(int(line[1]), []).extend(units)
    return explanation, circuits


def render(explanation: str, circuits: dict) -> str:
    """The answer text of circuits {position: [unit, ...]} (heads of one matrix and position written together)."""
    out = [f"EXPLANATION: {explanation}".rstrip()]
    for p in circuits:  # in the order given (importance: a cut-off answer loses the least important lines)
        whole, heads = [], {}
        for u in circuits[p]:
            if len(u) == 3:
                heads.setdefault((u[0], u[2]), []).append(u[1])
            else:
                whole.append(u)
        words = []
        for name, pos in sorted(whole, key=lambda u: (int(u[0].split(".")[1]), u[1], u[0])):
            layer = int(name.split(".")[1])
            words.append(f"L{layer}.{CODES[name.split('.')[-1]]}@{pos}")
        for (name, pos), hs in sorted(heads.items(), key=lambda kv: (int(kv[0][0].split(".")[1]), kv[0][1], kv[0][0])):
            layer = int(name.split(".")[1])
            words.append(f"L{layer}.{CODES[name.split('.')[-1]]}@{pos}[{','.join(f'h{h}' for h in sorted(hs))}]")
        out.append(f"POSITION {p}: " + " ".join(words))
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
    """The carry test of an answer on a task: {"uncarried": the movement-weighted share left uncarried, "units", per
    position {"position", "movement" (bits), "carried" (share), "units" (existing units the line names at the position,
    at relay positions between it and the target, or at the targets), "rest_carried" (the share carried by every other
    unit at those positions with the answer's held: near 0 when the answer is the only path, its necessity), "outside"
    (units before the position or after the target, which a change at the position cannot reach), "unknown" (names of
    units the model does not have)}, "rest_carried" (movement-weighted), "explanation"}. Size counts existing units only."""
    import torch

    import carry
    import native

    explanation, circuits = parse(text)
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    positions = sensitive(nat, ids, targets, k)
    rows, total_w, total_u = [], 0.0, 0.0
    for p in positions:
        prompts = carry.alternatives(nat, ids, p)
        stated = circuits.get(p, [])
        inside = [u for u in stated if p <= u[-1] <= max(targets)]  # at the position, a relay after it, or the targets
        gran = "head" if any(len(u) == 3 for u in inside) else "matrix"
        space = carry.UnitSpace(nat, len(ids), sorted({p, *targets, *(u[-1] for u in inside)}), gran)
        study = (carry.HeadCarry if gran == "head" else carry.MatrixCarry)(nat, ids, targets, prompts, space)
        on = _active(space, inside)
        with torch.no_grad():
            empty, with_units, without = study.measure([_active(space, []), on, ~on]).tolist()
        carried = 0.0 if empty <= 0 else max(0.0, min(1.0, 1 - with_units / empty))
        rest = 0.0 if empty <= 0 else max(0.0, min(1.0, 1 - without / empty))
        rows.append({"position": p, "movement": empty, "carried": carried, "rest_carried": rest, "units": int(on.sum()),
                     "outside": len(stated) - len(inside), "unknown": len(inside) - int(on.sum())})
        total_w += empty
        total_u += empty * (1 - carried)
        native.free(nat.dev)
    return {"uncarried": total_u / total_w if total_w else 0.0, "units": sum(r["units"] for r in rows), "positions": rows,
            "rest_carried": sum(r["movement"] * r["rest_carried"] for r in rows) / total_w if total_w else 0.0,
            "explanation": explanation}


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
    layer, the attention at the target kept from reading the position (_cut). Each event: {"position", "new" (token
    id) or "layer", "down": whether the model's probability of its most likely next token goes down, "weight": by how
    much it moves, |log2 of its ratio| capped at 4 bits (a change that moves nothing hardly counts), "bits": the signed
    log2 ratio}."""
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
        alt = carry.alternatives(nat, ids, q)[0]
        out.append(event(q, float(p_top(alt)[top]), new=alt[q]))
        for layer in range(nat.target.n_layer):
            out.append(event(q, float(p_top(ids, _cut(t, q, layer))[top]), layer=layer))
    native.free(nat.dev)
    return out


def targets(explain_jsonl: Path, out_dir: Path, fraction: int = 90) -> int:
    """One answer per question of carry.py explain's output: for each position, the smallest measured unit set on its
    elimination curve that carries `fraction`% of what the position's full unit set carries (a position whose full set
    carries less than all of its movement needs units elsewhere too; its line then states the part that is at the
    position and the targets), no explanation (the English is trained by its own reader). Positions whose full set
    carries nothing get no line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in open(explain_jsonl):
        r = json.loads(line)
        circuits = {}
        for p in r["positions"]:
            curve, sets, empty = p["curve"], p["unit_sets"], p["empty_bits"]
            full = min(kl for _, kl in curve)
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
       "position, name the units through which a change of the token there reaches the prediction: a unit is a weight "
       "matrix of a layer at a position, L<layer>.<q|k|v|o|fc|down>@<position>, an attention matrix optionally by its "
       "heads, L<layer>.<q|k|v|o>@<position>[h<i>,...]. Answer with a line EXPLANATION: (in plain English, what the model "
       "does), then one line POSITION <p>: <units> per listed position. Fewer units are better, as long as they carry "
       "the change.")


def numbered(strings: list[str]) -> str:
    """Tokens with their positions, 0:'The' 1:' cat' ...: a reader finds position 322 without counting."""
    return " ".join(f"{i}:{s!r}" for i, s in enumerate(strings))


def question(task: dict, positions: list[int]) -> str:
    """The oracle's question for unit circuits: the ask, the text's tokens by position, the target and the model's most
    probable next tokens there, and the positions whose changed token moves the prediction most (with their tokens)."""
    from prompt import tokens_of

    model = task["model"]
    p = task["prompts"][0]
    strings = tokens_of(model, p["token_ids"])
    t = max(p["target_positions"])
    top = ", ".join(f"{tok!r} {q:.2f}" for tok, q in p["model_top"][0][:3])
    return "\n".join([ASK.format(model=model), "tokens: " + numbered(strings),
                      f"target: position {t}; the model's most probable next tokens there: {top}",
                      "positions where a changed token moves the prediction most, most first: "
                      + ", ".join(f"{q} ({strings[q]!r})" for q in positions)])


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


def dataset(explain_jsonl: Path, out: Path, fraction: int = 90) -> int:
    """SFT rows {"prompt": [user message], "completion": [assistant message], "info": {"task"}} from carry.py
    explain's measured circuits (targets()'s answers) and their questions."""
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
                f.write(json.dumps({"prompt": [{"role": "user", "content": question(task, positions)}],
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
    b = sub.add_parser("baseline")
    b.add_argument("kind", choices=("copy", "all"))
    b.add_argument("task")
    b.add_argument("--k", type=int, default=8)
    e = sub.add_parser("events")
    e.add_argument("split", choices=("train", "heldout", "hard"))
    e.add_argument("out", type=Path)
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--k", type=int, default=8)
    t = sub.add_parser("targets")
    t.add_argument("explain", type=Path)
    t.add_argument("out", type=Path)
    t.add_argument("--fraction", type=int, default=90)
    a = ap.parse_args()
    if a.cmd == "targets":
        print(f"{targets(a.explain, a.out, a.fraction)} answers in {a.out}")
        return
    if a.cmd == "dataset":
        print(f"{dataset(a.explain, a.out, a.fraction)} rows in {a.out}")
        return
    import native

    nat = native.Native()
    if a.cmd == "events":  # one line per text: {"task", "events"}; resumable
        done = {json.loads(x)["task"] for x in open(a.out)} if a.out.exists() else set()
        with open(a.out, "a") as f:
            for path in native.tasks(a.split)[:a.n]:
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
