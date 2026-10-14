"""RL environment C (#2951 graph oracle, the lead's 10-13): the oracle's VARIABLES and INVARIANTS checked in the real
model. The question is the circuit environment's (components.question, its row marked "variables" and "invariants":
ASK_VARIABLES and ASK_INVARIANTS); the answer's VARIABLES block is decoded to groups of subcomponents (components.parse:
blocks "L1.v @232-233", the circuit's ids or subcomponents written out) and scored three ways, each reported:
  credit: the INVARIANTS lines that name a variable (transplants and removals; a line changing tokens alone is about
    the model, not the variables, and counts nothing; RULES OUT lines are left to components.invariant_diagnostic),
    each checked in the real model on its stated changed texts, removals at means over fresh tokens
    (semantic.check_invariants): + the line's surprise against the no-change prediction when it holds, - when it
    fails; a line changing two or more inputs counts when it holds only if no input alone gives its outcome (joint),
    and fully when it fails; a line that does not parse or names no stated variable costs a bit.
  relational: variables.headline's relational term on the split's interchange draws (each ordered pair's stated
    relation, feeds or apart, and each pair's composition; 0 with fewer than two variables).
  coverage: carrier coverage (components.e4x_term) of the text's teacher circuit (teacher/influential_v2/<task>.json):
    its nodes' measured removal effects (goal_eval.line_effects, pp, cached under LE) inside the variables' groups,
    each weighted 2 e4_v - 1 (e4_v the variable's own min(f on, f off, f same) on the split's draws).
  selective: credit less the same lines' credit with each group replaced by random subcomponents (blocks by random
    sites' blocks) at the same positions, as many: what the stated subcomponents add over their positions (most of a
    position carries an input token's variable: the gate's shuffled answers tied the teacher on such texts).
  reward = W_CREDIT tanh(credit / CREDIT_SCALE) + W_SEL tanh(selective / CREDIT_SCALE) + W_REL relational + W_COV
    coverage (credit's scale is the teacher's median per held-out text, 35 bits, halved).
The reward uses the search split's draws; evaluations the eval split's.

  env_c.py score TASK ANSWER.txt [--split eval] [--device D]
  env_c.py gate [--dir variables/influential_v2/heldout] [--split eval] [--part K/N] [--out FILE] [--device D]
      dev texts: the teacher's targets (its kept variables as blocks, its lines that held, joint) against the same
      answer with shuffled groups (random blocks, as many, at the same positions) and with DeepSeek's outcomes for the
      same interventions (var_explain's llm.jsonl and llm_hint.jsonl from the text alone; the better of the two)
  env_c.py sft ROWS.jsonl OUT.jsonl [--split train]   SFT rows (components.dataset's format): the question with
      VARIABLES and INVARIANTS asked, the teacher circuit with its block-form variables and its holding joint lines
  env_c.py rows ROWS.jsonl OUT.jsonl   the question rows (the circuit environment's, asking for VARIABLES and INVARIANTS)
  env_c.py serve [--port 8771] [--device D]
      POST /score {"task", "answers": [text, ...], "split": "search"} -> {"scores": [{"reward", "credit", "relational",
      "coverage", "valid", "lines", "holds", "counted", "joint_dropped", "bad", "variables"}, ...]}; GET /health
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "vpd")]

import components  # noqa: E402
import native  # noqa: E402

ROOT = native.TEXTS.parent
TEACHER = ROOT / "teacher" / "influential_v2"
LE = Path(__import__("os").environ.get("ENVC_LE", ROOT / "variables" / "envc" / "le"))  # ENVC_LE: a pod's synced dir
CREDIT_SCALE = 16.0
W_CREDIT = 0.5
W_SEL = 0.5
W_REL = 0.5
W_COV = 0.5
DRAWS = 8
MAX_CHARS = 24000  # an SFT target's longest rendering


def teacher_le(ge, tx, task: str) -> dict | None:
    """The teacher circuit's line effects (pp, search draws), cached as {"nodes": [{"node", "delta_kl"}]}."""
    import goal_eval

    f = LE / f"{task}.json"
    if f.exists():
        return json.loads(f.read_text())
    tf = TEACHER / f"{task}.json"
    if not tf.exists():
        return None
    import torch
    import var_teacher

    graph = goal_eval.strand(goal_eval.circuit(json.loads(tf.read_text()), tx.targets, ge.nat.C, tx.T)[0], tx.targets, ge.nat.C)
    var_teacher.release(ge.nat.dev)
    for d in (DRAWS, DRAWS // 2, DRAWS // 4, 1):  # a large circuit's pp runs can take more than a 24 GB card holds
        try:
            le = ge.line_effects(tx, graph, mode="pp", split="search", draws=d)
            break
        except torch.OutOfMemoryError:
            var_teacher.release(ge.nat.dev)
    else:
        return None
    out = {"nodes": [{"node": list(x["node"]), "delta_kl": x["delta_kl"]} for x in le["nodes"]]}
    LE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def score(ge, tx, task: str, text: str, split: str = "search", model: str = "vpd4l") -> dict:
    """One answer's terms and reward (module docstring)."""
    import semantic
    import variables

    p = components.parse(text)
    ans = p["answer"]
    bound = [{**v, "name": v.get("ref", f"V{i}")} for i, v in enumerate(ans.get("variables") or [], 1)]
    out = {"valid": False, "credit": 0.0, "relational": 0.0, "coverage": 0.0, "lines": 0, "holds": 0, "counted": 0,
           "joint_dropped": 0, "bad": p.get("variables_malformed", 0), "variables": len(bound)}
    try:
        groups = variables.parse(bound, ge.nat.C) if bound and len({v["name"] for v in bound}) == len(bound) else []
    except (ValueError, KeyError, TypeError):
        groups = []
    if not groups:
        out["reward"] = -1.0
        return out
    out["valid"] = True
    claims = [c for c in ans.get("invariants") or [] if not c.startswith("RULES OUT")]
    named = [c for c in claims if re.search(r"\bV\d+\b", c.split("->")[0])]  # a line about the variables
    sc = variables.Scorer(ge)
    credit = 0.0  # token-only lines count nothing
    random_credit = 0.0
    if named:
        got = semantic.check_invariants(sc, tx, groups, "\n".join(named), model, fresh_mean=True)
        credit -= got["bad"]
        for r in got["invariants"]:
            out["lines"] += 1
            out["holds"] += bool(r["holds"])
            if r["holds"] and r.get("joint") is False:
                out["joint_dropped"] += 1
                continue
            credit += r["credit"]
            out["counted"] += 1
        out["bad"] += got["bad"]
        # the same lines with each group replaced by random subcomponents (blocks by random sites' blocks) at the same
        # positions: what the positions alone give (an input token's variable is carried by most of its position)
        seed = int(hashlib.sha256(f"{task}|{text}".encode()).hexdigest()[:12], 16)
        ctl = semantic.check_invariants(sc, tx, random_groups(groups, ge.nat.C, seed), "\n".join(named), model, fresh_mean=True)
        random_credit = sum(r["credit"] for r in ctl["invariants"] if not (r["holds"] and r.get("joint") is False)) - ctl["bad"]
    out["credit"] = credit
    out["credit_random"] = random_credit
    out["selective"] = credit - random_credit
    got = sc.score(tx, groups, split=split, draws=DRAWS)
    out["relational"] = got.get("relational") or 0.0
    out["carriers"] = got.get("carriers")
    per = []
    for v, x in zip(groups, got["variables"]):
        e = 0.0 if x.get("f_on") is None else min(y for y in (x["f_on"], x.get("f_off"), x.get("f_same")) if y is not None)
        per.append({"name": v["name"], "e4": e, "nodes": [list(nd) for nd in v["group"]]})
    le = teacher_le(ge, tx, task)
    cov = components.e4x_term(None, None, None, {"per_variable": per}, None, le) if le else None
    out["coverage"] = cov or 0.0
    out["reward"] = (W_CREDIT * math.tanh(credit / CREDIT_SCALE) + W_SEL * math.tanh(out["selective"] / CREDIT_SCALE)
                     + W_REL * out["relational"] + W_COV * out["coverage"])
    return out


def random_groups(groups: list[dict], C: dict, seed: int) -> list[dict]:
    """Each variable's group replaced by as many random items at the same positions: a block by a random site's block,
    a subcomponent by a random subcomponent there."""
    import variables

    rng = random.Random(seed)
    names = sorted(C)
    out = []
    for v in groups:
        g = set()
        for t in sorted({t for _, t, _ in v["group"]}):
            mine = [c for _, tt, c in v["group"] if tt == t]
            nb = sum(c == variables.BLOCK for c in mine)
            for n in rng.sample(names, min(nb, len(names))):
                g.add((n, t, variables.BLOCK))
            while sum(1 for _, tt, c in g if tt == t and c != variables.BLOCK) < len(mine) - nb:
                n = rng.choice(names)
                g.add((n, t, rng.randrange(C[n])))
        out.append({**v, "group": sorted(g)})
    return out


def text_of(ge, task: str):
    import native as nat_mod

    t = json.loads((nat_mod.TEXTS / "vpd4l" / f"{task}.json").read_text())
    p = t["prompts"][0]
    return t, ge.text(p["token_ids"], p["target_positions"], seed=nat_mod.task_seed(task), task=t)


def blocks(group: list) -> list:
    """A group as blocks: each (site, position) it touches, that site's whole output there."""
    return [["block", n, t] for n, t in sorted({(it[1], int(it[2])) if it[0] == "block" else (it[0], int(it[2])) for it in group})]


def answer_of(task: str, record: dict, ge, lines: list[str] | None = None, shuffle: int | None = None,
              compact: dict | None = None) -> dict:
    """The teacher's answer in goal_eval's JSON: its circuit, its kept variables (names and meanings from var_explain's
    targets record) as blocks, or as their compact families (compact: (task, name) -> var_compact.py's subcomponent
    items; a variable without one stays blocks), and lines (default the record's holding joint lines); shuffle (a seed):
    each group replaced by as many random items at the same positions (a block by a random site's block, a
    subcomponent by a random subcomponent there)."""
    import goal_eval

    t = json.loads((native.TEXTS / "vpd4l" / f"{task}.json").read_text())
    p = t["prompts"][0]
    g, _ = goal_eval.circuit(json.loads((TEACHER / f"{task}.json").read_text()), p["target_positions"], ge.nat.C, len(p["token_ids"]))
    ans = goal_eval.to_json(g, "")
    short = {v["name"]: v["short"].replace('"', "'") for v in record["variables"]}
    rng = random.Random(shuffle)
    vs = []
    names = sorted(ge.nat.C)
    for v in record["variables"]:
        grp = (compact or {}).get((task, v["name"])) or blocks(v["group"])
        if shuffle is not None:
            new = set()
            for it in grp:
                if it[0] == "block":
                    n = rng.choice(names)
                    while ("block", n, it[2]) in new:
                        n = rng.choice(names)
                    new.add(("block", n, it[2]))
                else:
                    n = rng.choice(names)
                    new.add((n, rng.randrange(ge.nat.C[n]), it[2]))
            grp = [list(x) for x in sorted(new, key=str)]
        vs.append({"name": short[v["name"]], "ref": v["name"], "meaning": v["meaning"].replace('"', "'"), "inputs": v["inputs"],
                   "group": grp, "feeds": [short[f] if f in short else "prediction" for f in (v.get("feeds") or []) if f in short or f == "prediction"]})
    ans["variables"] = vs
    ans["invariants"] = list(record["invariants"] if lines is None else lines)
    return ans


def load_compact(path: Path | None) -> dict | None:
    """var_compact.py's rows as (task, name) -> subcomponent items ([matrix, index, position])."""
    if not path:
        return None
    return {(x["task"], x["name"]): x["group"] for x in map(json.loads, path.read_text().splitlines()) if x.get("group")}


def gate(a) -> None:
    import goal_eval

    global LE
    if a.le_dir:
        LE = a.le_dir

    sys.path.insert(0, str(HERE.parents[1] / "experiments" / "oracle"))
    import var_explain

    ge = goal_eval.GoalEval(native.Native(a.device))
    recs = [json.loads(x) for x in (a.targets or ROOT / "variables" / "sft_targets" / "heldout.jsonl").read_text().splitlines()]
    ex = ROOT / "variables" / "explanations" / "heldout_v2"
    prompts = {x["task"]: x for x in map(json.loads, (ex / "prompts.jsonl").read_text().splitlines())}
    llm = {k: {x["task"]: x for x in map(json.loads, (ex / f"{k}.jsonl").read_text().splitlines())} for k in ("llm", "llm_hint")}
    if a.part:
        k, n = map(int, a.part.split("/"))
        recs = recs[k::n]
    cp = load_compact(a.compact)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["task"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    import torch
    import var_teacher

    for rec in recs:
        task = rec["task"]
        if task in done:
            continue
        var_teacher.release(ge.nat.dev)
        try:
            gate_text(ge, a, rec, cp, prompts, llm, var_explain)
        except torch.OutOfMemoryError:  # one text past the card: freed and tried once more, else left out (printed)
            var_teacher.release(ge.nat.dev)
            try:
                gate_text(ge, a, rec, cp, prompts, llm, var_explain)
            except torch.OutOfMemoryError:
                var_teacher.release(ge.nat.dev)
                print(task, "out of memory twice", flush=True)


def gate_text(ge, a, rec: dict, cp: dict | None, prompts: dict, llm: dict, var_explain) -> None:
    task = rec["task"]
    t, tx = text_of(ge, task)
    strings = components.token_strings(t["model"], t["prompts"][0]["token_ids"])
    teacher = components.render(answer_of(task, rec, ge, compact=cp), strings=strings)
    shuffled = components.render(answer_of(task, rec, ge, shuffle=native.task_seed(task), compact=cp), strings=strings)
    row = {"task": task, "teacher": score(ge, tx, task, teacher, a.split, t["model"]),
           "shuffled": score(ge, tx, task, shuffled, a.split, t["model"])}
    body = lambda line: line.split(":", 1)[1].split("->")[0].strip()  # noqa: E731
    for k in ("llm", "llm_hint"):
        if task not in llm[k] or task not in prompts:
            continue
        theirs = {body(x): x.split("->", 1)[1].strip() for x in var_explain.llm_lines(prompts[task]["items"], llm[k][task]["answer"]).splitlines()}
        lines = [f"I{i}: {body(x)} -> {theirs[body(x)]}" for i, x in enumerate(rec["invariants"], 1) if body(x) in theirs]
        row[k] = score(ge, tx, task, components.render(answer_of(task, rec, ge, lines=lines, compact=cp), strings=strings), a.split, t["model"])
    with a.out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    best_llm = max((row[k]["reward"] for k in ("llm", "llm_hint") if k in row), default=None)
    print(task, {k: round(row[k]["reward"], 3) for k in ("teacher", "shuffled", "llm", "llm_hint") if k in row}, best_llm, flush=True)


def relines(a) -> None:
    """A targets file whose INVARIANTS lines hold under the carriers the answer states: each record's lines checked
    in the real model on its rendered answer (compact families with --compact, else blocks; removals at fresh-token
    means), the holding ones kept (a two-input line only when joint). Lines verified under the full families can fail
    under the compact ones (gate4c: the teacher's credit median -6.4 bits)."""
    import goal_eval
    import semantic
    import variables

    ge = goal_eval.GoalEval(native.Native(a.device))
    cp = load_compact(a.compact)
    recs = [json.loads(x) for x in (ROOT / "variables" / "sft_targets" / f"{a.split}.jsonl").read_text().splitlines()]
    done = {json.loads(x)["task"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        if rec["task"] in done:
            continue
        t, tx = text_of(ge, rec["task"])
        strings = components.token_strings(t["model"], t["prompts"][0]["token_ids"])
        p = components.parse(components.render(answer_of(rec["task"], rec, ge, compact=cp), strings=strings))
        bound = [{**v, "name": v.get("ref", f"V{i}")} for i, v in enumerate(p["answer"].get("variables") or [], 1)]
        lines = p["answer"].get("invariants") or []
        keep = []
        if lines and bound:
            got = semantic.check_invariants(variables.Scorer(ge), tx, variables.parse(bound, ge.nat.C), "\n".join(lines), t["model"], fresh_mean=True)
            ok = {r["line"] for r in got["invariants"] if r["holds"] and r.get("joint") is not False}
            keep = [orig for orig, line in zip(rec["invariants"], lines) if line.strip() in ok]  # render keeps the lines' order
        out = {**rec, "invariants": [re.sub(r"^I\d+:", f"I{i}:", x) for i, x in enumerate(keep, 1)],
               "relined": "compact" if cp else "blocks", "lines_before": len(rec["invariants"])}
        with a.out.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        print(rec["task"], len(rec["invariants"]), "->", len(keep), flush=True)


def rows(a) -> None:
    """Env C's question rows: each circuit-environment row marked "variables" and "invariants", its "question" rendered
    again with ASK_VARIABLES and ASK_INVARIANTS (components.question); every other field passed through."""
    n = 0
    with a.out.open("w") as fh:
        for r in map(json.loads, a.rows.read_text().splitlines()):
            r = {**r, "variables": True, "invariants": True}
            t = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
            r["question"] = components.question(t, r)
            fh.write(json.dumps(r) + "\n")
            n += 1
    print(n, "rows to", a.out)


def sft(a) -> None:
    """SFT rows: components.dataset's format, the question asking for VARIABLES and INVARIANTS, the target the
    teacher's circuit with its block-form variables and holding joint lines (components.target: its FINAL lines too)."""
    import goal_eval

    ge = goal_eval.GoalEval(native.Native(a.device))
    recs = {x["task"]: x for x in map(json.loads, (ROOT / "variables" / "sft_targets" / f"{a.split}.jsonl").read_text().splitlines())}
    cp = load_compact(a.compact)
    n = 0
    with a.out.open("w") as fh:
        for r in map(json.loads, a.rows.read_text().splitlines()):
            rec = recs.get(r["task"])
            if rec is None or not rec["invariants"]:
                continue
            r = {**r, "variables": True, "invariants": True}
            t = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
            strings = components.token_strings(t["model"], t["prompts"][0]["token_ids"])
            content = components.target(answer_of(r["task"], rec, ge, compact=cp), r, t["model"], {}, strings)
            if len(content) > MAX_CHARS:  # past the trainer's answer cap (8,000 tokens, ~3 characters a token)
                continue
            fh.write(json.dumps({"prompt": [{"role": "user", "content": components.question(t, r)}],
                                 "completion": [{"role": "assistant", "content": content}],
                                 "info": {"task": r["task"], "variables": len(rec["variables"]), "invariants": len(rec["invariants"]),
                                          "transfer_share": rec.get("transfer_share")}}) + "\n")
            n += 1
    print(n, "rows to", a.out)


def source() -> str:
    """The scorer's revision: the pod's SOURCE file (rp-run's "python source <rev>") when there is one, else a hash of
    the scoring code (this module and what it calls), so a trainer can freeze it (REWARD_REV)."""
    import hashlib

    f = HERE.parents[1] / "SOURCE"
    if f.exists():
        return f.read_text().split()[-1]
    h = hashlib.sha256()
    for name in ("env_c.py", "semantic.py", "variables.py", "components.py", "goal_eval.py", "engine.py", "native.py"):
        h.update((HERE / name).read_bytes())
    return "code-" + h.hexdigest()[:12]


def serve(a) -> None:
    """The reward service (primerl's contract): POST /score {"task" (or "id"), "answers": [text, ...], "split" (default
    search), any other fields ignored ("lines", "tokens")} -> {"scores": [score() per answer in order, with
    "semantics" and "source"]}; GET /health."""
    import traceback

    import goal_eval

    ge = goal_eval.GoalEval(native.Native(a.device))
    who = {"semantics": ge.semantics, "source": source()}
    lock, texts = threading.Lock(), {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send({"ok": True, "alive": True, "texts": len(texts), **who})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            task = req.get("task") or req.get("id")
            got = []
            with lock:
                if task not in texts:
                    if len(texts) >= 8:
                        texts.pop(next(iter(texts)))
                    texts[task] = text_of(ge, task)
                t, tx = texts[task]
                for x in req["answers"]:
                    try:
                        got.append({**score(ge, tx, task, x, req.get("split", "search"), t["model"]), **who})
                    except Exception as e:  # an answer that breaks the scorer is invalid, never a crashed group
                        got.append({"reward": -1.0, "valid": False, "error": f"{type(e).__name__}: {e}"[:300], **who})
                        traceback.print_exc()
            self._send({"scores": got})

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"env C on :{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("score")
    s.add_argument("task")
    s.add_argument("answer", type=Path)
    s.add_argument("--split", default="eval")
    g = sub.add_parser("gate")
    g.add_argument("--split", default="eval")
    g.add_argument("--part")
    g.add_argument("--out", type=Path, default=ROOT / "variables" / "envc" / "gate.jsonl")
    g.add_argument("--compact", type=Path, help="var_compact.py's rows: variables written as compact families, not blocks")
    g.add_argument("--le-dir", type=Path, help="the teacher line effects' cache (a pod's output directory keeps them)")
    g.add_argument("--targets", type=Path, help="a targets file (default sft_targets/heldout.jsonl; relines' output)")
    f = sub.add_parser("sft")
    f.add_argument("rows", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--split", default="train")
    f.add_argument("--compact", type=Path, help="var_compact.py's rows: variables written as compact families, not blocks")
    v = sub.add_parser("serve")
    v.add_argument("--port", type=int, default=8771)
    q = sub.add_parser("rows")
    q.add_argument("rows", type=Path)
    q.add_argument("out", type=Path)
    rl = sub.add_parser("relines")
    rl.add_argument("--split", default="heldout")
    rl.add_argument("--compact", type=Path)
    rl.add_argument("--out", type=Path, required=True)
    for p in (s, g, f, v, q, rl):
        p.add_argument("--device")
    a = ap.parse_args()
    if a.what == "score":
        import goal_eval

        ge = goal_eval.GoalEval(native.Native(a.device))
        t, tx = text_of(ge, a.task)
        print(json.dumps(score(ge, tx, a.task, a.answer.read_text(), a.split, t["model"]), indent=1))
    else:
        {"gate": gate, "sft": sft, "serve": serve, "rows": rows, "relines": relines}[a.what](a)


if __name__ == "__main__":
    main()
