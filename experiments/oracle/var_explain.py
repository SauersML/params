"""Explanations of held-out texts as their kept variables and verified INVARIANTS lines, and how often those lines hold
on fresh tokens against two baselines (#2951 graph oracle, the lead's 10-13: "invariants from the English, checked").

The explanation of a text (var_teacher.py's file) is its kept variables, each a name, a meaning in words (an identity:
the token at its position; a computed variable: what its carrier position computes from its input tokens, with its
value classes), its inputs and its carrier, and the INVARIANTS lines semantic.teacher_invariants wrote for them: per
input position fresh donor tokens the value map never saw, 'V from @p=z -> top y' with y the model's own prediction on
the text with z at p (the carrier alone must carry it), 'remove V -> p(top) < 0.5', and two-donor recombinations.
Every line is rechecked here on one device under three conditions, with its hold rate and credit (semantic.Checker):
  teacher: the lines as written, with the variables' carriers;
  shuffled: the same lines with each carrier replaced by a random group of the same size at the same positions (blocks
    by as many other sites' blocks, subcomponents by as many random subcomponents there);
  llm: the same interventions with the outcomes a strong open model (DeepSeek-V4.1-Flash on Tinker) predicts from the
    text alone (the text, the model's prediction, which position each variable reads; no circuit and no carrier);
    llm_hint: the same, told that every changed-token question changes the top token (the teacher's lines are those).

  experiments/oracle/var_explain.py write [--dir DIR] [--out DIR]   explanations and the LLM's prompts (prompts.jsonl)
  (tinker venv) experiments/oracle/var_explain_llm.py OUT/prompts.jsonl OUT/llm.jsonl [--hint, to OUT/llm_hint.jsonl]
  experiments/oracle/var_explain.py check [--dir DIR] [--out DIR] [--device D]   the three conditions, OUT/check.jsonl
  experiments/oracle/var_explain.py targets --dir influential_v2 --out DIR   the English SFT arm's targets (targets())
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/vpd")]

import components  # noqa: E402
import native  # noqa: E402

DIR = native.TEXTS.parent / "variables" / "influential_v2" / "heldout"
OUT = native.TEXTS.parent / "variables" / "explanations" / "heldout"
LLM = ("llm", "llm_hint")  # the LLM's answers (var_explain_llm.py), and with --hint: told the top token changes
LINE = re.compile(r"^I(\d+):\s*(.+?)\s*->\s*(.+)$")
PROMPT = (
    "A small language model (4 transformer layers, trained on web text) reads the text below and predicts the next token "
    "after its last token. Positions are numbered from 0.\n\nText (position:token):\n{tokens}\n\n"
    "The model's prediction now: {top}.\n\n{defs}\n\nFor each question, predict what the model does. For a question "
    "about a variable set from a changed token, give the model's top next token after the change, as a quoted string "
    "exactly as the token is written (with its leading space if any), for example Q1: ' the'. For a question about "
    "removing a variable (its value replaced by its average over many possible tokens), answer drops (the current top "
    "token's probability falls below half its value) or stays. One line per question, Q<k>: <answer>, nothing else.\n\n"
    "{questions}")


def meaning(v: dict, toks: dict) -> tuple[str, str]:
    """(a short name, a meaning in words) for a kept variable."""
    ins = ", ".join(f"@{p} {toks[p]!r}" for p in v["inputs"])
    if v.get("kind") == "input":
        return f"token {ins}", f"the token at {ins} (an identity: every token there is its own value)"
    at = ", ".join(f"@{q}" for q in v.get("carried_at") or v["inputs"])
    text = f"what the model computes at {at} from {ins}"
    for vm in v.get("values") or []:
        cls = [c for c in vm["classes"] if c["tokens"]]
        if len(cls) > 1:
            parts = [f"{{{', '.join(repr(t) for t in c['tokens'][:4])}{', ...' if len(c['tokens']) > 4 else ''}}} -> {c['predicts']!r}"
                     for c in cls[:4]]
            text += f"; at @{vm['position']} its values: " + "; ".join(parts)
    return f"computed @{at}", text


def explanation(r: dict, ids: list[int]) -> dict:
    """The text's explanation: kept variables with names and meanings, their carriers, and the INVARIANTS lines (holding
    ones first), rendered as VARIABLES and INVARIANTS blocks."""
    kv = [v for v in r["variables"] if v.get("kept")]
    pos = sorted({p for v in kv for p in v["inputs"]})
    toks = dict(zip(pos, components.token_strings("vpd4l", [ids[p] for p in pos])))
    rows, lines = [], ["VARIABLES:"]
    for v in kv:
        short, words = meaning(v, toks)
        sites = sorted({it[1] if it[0] == "block" else it[0] for it in v["group"]})
        carrier = (f"{v['carrier']} of {len(v['group'])} items at "
                   + ", ".join(f"@{q}" for q in v.get("carried_at") or v["inputs"]) + f" in {len(sites)} sites")
        rows.append({"name": v["name"], "short": short, "meaning": words, "kind": v["kind"], "inputs": v["inputs"],
                     "carried_at": v.get("carried_at"), "group": v["group"], "feeds": v.get("feeds", []),
                     "value_kind": v.get("value_kind"), "measured": {k: v["measured"].get(k) for k in ("search", "eval", "trivial_eval")}})
        lines.append(f'{v["name"]} "{short}": {words}; inputs: ' + ", ".join(f"@{p} {toks[p]!r}" for p in v["inputs"])
                     + f"; carried by: {carrier}; feeds: " + (", ".join(v.get("feeds") or []) or "PREDICTION"))
    inv = r.get("invariants") or []
    holding = [x for x in inv if x["holds"]]
    lines += ["INVARIANTS:"] + [x["line"] for x in holding]
    return {"variables": rows, "invariants": inv, "holding": len(holding), "lines": len(inv), "text": "\n".join(lines)}


def questions(r: dict, ids: list[int]) -> tuple[str, list[dict]] | None:
    """The LLM's prompt for the teacher's interventions (no outcomes; the model's prediction left as {top}) and the
    questions' interventions."""
    kv = {v["name"]: v for v in r["variables"] if v.get("kept")}
    inv = r.get("invariants") or []
    if not kv or not inv:
        return None
    toks = components.token_strings("vpd4l", list(ids))
    defs = ["Variables (what the model takes from a position; how it is stored inside the model is not given):"]
    for name, v in kv.items():
        defs.append(f"- {name}: what the model computes from the token(s) at " + ", ".join(f"position {p} ({toks[p]!r})" for p in v["inputs"]))
    qs, items = [], []
    for k, x in enumerate(inv, 1):
        m = LINE.match(x["line"])
        if not m:
            continue
        iv = m.group(2)
        if iv.startswith("remove "):
            q = f"Q{k}: remove {iv.split()[1]}"
        else:
            parts = [re.match(r"(V\d+) from @(\d+)=(.+)", p.strip()) for p in re.split(r",\s*(?=V\d+ from )", iv)]
            if not all(parts):
                continue
            text = [f"{p.group(1)} set to its value when the token at position {p.group(2)} is {p.group(3)}" for p in parts]
            q = f"Q{k}: " + " and ".join(text) + " (the text itself unchanged)"
        qs.append(q)
        items.append({"k": k, "intervention": iv, "line": x["line"]})
    tokens = " ".join(f"{i}:{components.shown(t)}" for i, t in enumerate(toks))
    return PROMPT.format(tokens=tokens, top="{top}", defs="\n".join(defs), questions="\n".join(qs)), items


def write(a) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    nat = native.Native(a.device)
    prompts = []
    for f in sorted(a.dir.iterdir()):
        if f.suffix != ".json" or not f.name.startswith("text"):
            continue
        r = json.loads(f.read_text())
        task = json.loads((native.TEXTS / "vpd4l" / f"{f.stem}.json").read_text())
        p = task["prompts"][0]
        ids, targets = p["token_ids"], p["target_positions"]
        if not any(v.get("kept") for v in r["variables"]):
            continue
        ex = explanation(r, ids)
        ex["task"] = f.stem
        (a.out / f"{f.stem}.json").write_text(json.dumps(ex, indent=1))
        got = questions(r, ids)
        if got:
            lq = nat.reference([ids], targets)[0, -1].cpu().double()
            top = lq.exp().topk(3)
            names = components.token_strings("vpd4l", top.indices.tolist())
            shown = ", ".join(f"{components.shown(t)} ({float(q):.2f})" for t, q in zip(names, top.values))
            prompts.append({"task": f.stem, "prompt": got[0].replace("{top}", shown), "items": got[1]})
    with (a.out / "prompts.jsonl").open("w") as fh:
        for x in prompts:
            fh.write(json.dumps(x) + "\n")
    print(f"{len(prompts)} prompts, explanations in {a.out}")


def shuffled(vs: list[dict], C: dict, seed: int) -> list[dict]:
    """Each variable's carrier replaced by a random group of the same size at the same positions (blocks by as many other
    sites' blocks; subcomponents by as many random subcomponents at their position)."""
    import variables

    rng = random.Random(seed)
    names = sorted(C)
    out = []
    for v in vs:
        group = set()
        for t in sorted({t for _, t, _ in v["group"]}):
            mine = [(n, c) for n, tt, c in v["group"] if tt == t]
            nb = sum(c == variables.BLOCK for _, c in mine)
            ns = len(mine) - nb
            for n in rng.sample(names, nb):
                group.add((n, t, variables.BLOCK))
            while sum(1 for _, tt, c in group if tt == t and c != variables.BLOCK) < ns:
                n = rng.choice(names)
                group.add((n, t, rng.randrange(C[n])))
        out.append({**v, "group": sorted(group)})
    return out


def llm_lines(items: list[dict], answer: str) -> str:
    """The LLM's answers as INVARIANTS lines over the teacher's interventions (an unanswered question: no line)."""
    got = {}
    for line in answer.splitlines():
        m = re.match(r"\s*Q(\d+)\s*:\s*(.+?)\s*$", line)
        if m:
            got.setdefault(int(m.group(1)), m.group(2))
    out = []
    for it in items:
        ans = got.get(it["k"])
        if ans is None:
            continue
        if it["intervention"].startswith("remove "):
            word = ans.strip().strip(".").lower()
            out.append(f"I{it['k']}: {it['intervention']} -> " + ("p(top) < 0.5" if word.startswith("drop") else "unchanged"))
        else:
            try:
                tok = ast.literal_eval(ans)
            except (ValueError, SyntaxError):
                tok = None
            if isinstance(tok, str):
                out.append(f"I{it['k']}: {it['intervention']} -> top {components.shown(tok)}")
    return "\n".join(out)


def check(a) -> None:
    import goal_eval
    import semantic
    import variables
    import var_teacher

    ge = goal_eval.GoalEval(native.Native(a.device))
    nat = ge.nat
    llm = {k: {x["task"]: x for x in map(json.loads, (a.out / f"{k}.jsonl").read_text().splitlines())}
           for k in LLM if (a.out / f"{k}.jsonl").exists()}
    prompts = {x["task"]: x for x in map(json.loads, (a.out / "prompts.jsonl").read_text().splitlines())}
    result = a.result or a.out / "check.jsonl"
    done = set()
    if result.exists():
        done = {json.loads(x)["task"] for x in result.read_text().splitlines()}
    tasks = sorted(prompts)
    if a.part:
        k, n = map(int, a.part.split("/"))
        tasks = tasks[k::n]
    result.parent.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        if t in done:
            continue
        r = json.loads((a.dir / f"{t}.json").read_text())
        task = json.loads((native.TEXTS / "vpd4l" / f"{t}.json").read_text())
        p = task["prompts"][0]
        tx = ge.text(p["token_ids"], p["target_positions"], seed=native.task_seed(t), task=task)
        kv = [v for v in r["variables"] if v.get("kept")]
        vs = variables.parse(kv, nat.C)
        teacher = "\n".join(x["line"] for x in r["invariants"])
        row = {"task": t, "variables": len(vs), "lines": len(r["invariants"])}
        sc = variables.Scorer(ge)
        row["teacher"] = semantic.check_invariants(sc, tx, vs, teacher, task["model"], fresh_mean=True)
        row["shuffled"] = semantic.check_invariants(sc, tx, shuffled(vs, nat.C, native.task_seed(t)), teacher, task["model"], fresh_mean=True)
        for k, got in llm.items():
            if t in got:
                text = llm_lines(prompts[t]["items"], got[t]["answer"])
                row[k] = semantic.check_invariants(sc, tx, vs, text, task["model"], fresh_mean=True)
                row[k]["answered"] = len(text.splitlines()) if text else 0
        var_teacher.release(nat.dev)
        for k in ("teacher", "shuffled", *LLM):
            if k in row:
                for x in row[k]["invariants"]:
                    x.pop("intervention", None)
        with result.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(t, {k: (round(row[k]["holds"], 3), round(row[k]["credit"], 1)) for k in ("teacher", "shuffled", *LLM) if k in row}, flush=True)


def targets(a) -> None:
    """The English SFT arm's targets (the lead's 10-13): per text of influential_v2 with kept variables, the teacher
    circuit's path, the kept variables (names, meanings, carriers, and generality's transfer field: the median E4 term
    on other members of the text's family and whether it is at least 0.8; none outside the families), the INVARIANTS
    lines that held on fresh tokens (renumbered), and the rendered VARIABLES and INVARIANTS blocks; one JSON line per
    text to OUT/<split>.jsonl. Texts are listed by their variables' transfer share (the share of kept variables that
    transfer, none outside the families) for picking targets."""
    tr = {}
    for x in json.loads((native.TEXTS.parent / "generality" / "var_transfer_v2_keep.json").read_text()):
        tr[(x["source"], x["name"])] = {"family": x["family"], "median": x["transfer_median"], "transfers": x["transfers"],
                                        "targets": x["targets"]}
    root = a.dir.parent if a.dir.name in ("train", "heldout") else a.dir
    a.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "heldout"):
        rows = []
        for f in sorted((root / split).iterdir()):
            if f.suffix != ".json" or not f.name.startswith("text"):
                continue
            r = json.loads(f.read_text())
            if not any(v.get("kept") for v in r["variables"]):
                continue
            ids = json.loads((native.TEXTS / "vpd4l" / f"{f.stem}.json").read_text())["prompts"][0]["token_ids"]
            # holding lines; a line changing two inputs only when no input alone gives its outcome ("joint")
            holding = [x for x in r.get("invariants") or [] if x["holds"] and x.get("joint", True)]
            for k, x in enumerate(holding, 1):  # holding lines only, renumbered
                x["line"] = re.sub(r"^I\d+:", f"I{k}:", x["line"])
            ex = explanation({**r, "invariants": holding}, ids)
            for v in ex["variables"]:
                v["transfer"] = tr.get((f.stem, v["name"]))
            known = [v["transfer"]["transfers"] for v in ex["variables"] if v["transfer"]]
            rows.append({"task": f.stem, "split": split, "circuit": str(native.TEXTS.parent / "teacher" / "influential_v2" / f"{f.stem}.json"),
                         "variables": ex["variables"], "invariants": [x["line"] for x in holding],
                         "invariants_version": r["teacher"].get("invariants_version"), "text": ex["text"],
                         "transfer_share": sum(known) / len(known) if known else None})
        rows.sort(key=lambda x: (x["transfer_share"] is None, -(x["transfer_share"] or 0), x["task"]))
        with (a.out / f"{split}.jsonl").open("w") as fh:
            for x in rows:
                fh.write(json.dumps(x) + "\n")
        print(split, len(rows), "texts;", sum(len(x["invariants"]) for x in rows), "holding lines;",
              sum(x["transfer_share"] is not None for x in rows), "in a family")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("write", "check", "targets"))
    ap.add_argument("--dir", type=Path, default=DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device")
    ap.add_argument("--part", help="check: K/N, every N-th text from the K-th (a cluster array's task)")
    ap.add_argument("--result", type=Path, help="check: the rows' file (default OUT/check.jsonl)")
    a = ap.parse_args()
    {"write": write, "check": check, "targets": targets}[a.what](a)


if __name__ == "__main__":
    main()
