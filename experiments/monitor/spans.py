"""Where does a hardcoded output come from? Change the test's expected value, replay, see if the output follows.

A hardcoded span is a place in a func.py the agent wrote where it returns a literal equal to a test's expected
output for a specific input: `if <arg> == <input>: return <expected>` (or the literal appears in a return and the
(input, expected) pair is a test assertion). That literal is the span, a range of completion tokens.

For each span the test's expected value also appears in the context (the agent read test.py). The intervention:
  - baseline: greedy-decode the literal from the real context with the completion teacher-forced up to the span;
  - edited: replace that one expected value in the context with a different number, decode the literal again.
The output follows the test when the edited decode equals the edited value (and the baseline equals the original).
We also report the logit margin at the span's first token: logprob(original value's first token) under each context.

Fraction of spans whose output follows the edited test value is the headline. Run on a GPU (cuda) or the Mac (mps).

  python spans.py --runs ARM_DIR [ARM_DIR ...] --out FILE [--model Qwen/Qwen3-8B] [--limit N]
"""
import argparse
import ast
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402
import replay  # noqa: E402

RET = re.compile(r"return\s+(-?\d+)\b")


def test_expected(t):
    """{input int -> set of expected int} over the impossible test's single-int-argument assertions."""
    exp = {}
    for st in agent.check_body(t["test"]).body:
        if isinstance(st, ast.Assert) and isinstance(st.test, ast.Compare) and isinstance(st.test.left, ast.Call) \
           and len(st.test.left.args) == 1 and st.test.ops and isinstance(st.test.ops[0], ast.Eq):
            try:
                x = ast.literal_eval(st.test.left.args[0])
                y = ast.literal_eval(st.test.comparators[0])
            except Exception:
                continue
            if isinstance(x, int) and isinstance(y, int) and not isinstance(x, bool) and not isinstance(y, bool):
                exp.setdefault(x, set()).add(y)
    return exp


def new_value(old):
    """A distinct replacement expected value of the same digit length, not equal to the original."""
    s = str(abs(old))
    cand = "9" + "1" * (len(s) - 1) if s[0] != "9" else "1" + "8" * (len(s) - 1)
    v = int(cand)
    return v if v != abs(old) else v + 1


def find_spans(rec, t):
    """Hardcoded spans in a run: each a write whose returned literal equals a test's expected output, with that
    expected value present once in the context before the write. One span per (write, input, expected)."""
    exp = test_expected(t)
    rec_prompt = t["prompt"]
    out = []
    seen = set()
    calls = replay.calls(rec)
    cmap = {c["turn"]: c for c in calls}
    for a in rec["actions"]:
        if a["name"] != "write_file" or os.path.basename((a["args"] or {}).get("path", "")) != "func.py":
            continue
        content = a["args"].get("content") or ""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        pairs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) \
               and node.test.ops and isinstance(node.test.ops[0], ast.Eq):
                try:
                    inp = ast.literal_eval(node.test.comparators[0])
                except Exception:
                    continue
                for s2 in node.body:
                    if isinstance(s2, ast.Return) and isinstance(s2.value, ast.Constant) and isinstance(s2.value.value, int):
                        ret = s2.value.value
                        if isinstance(inp, int) and exp.get(inp) and ret in exp[inp] and (inp, ret) not in seen:
                            pairs.append((inp, ret))
        c = cmap.get(a["turn"])
        if not c:
            continue
        for inp, ret in pairs:
            if (inp, ret) in seen:
                continue
            # the literal in the completion, anchored to its `return`
            sp = return_literal_span(c["completion_ids"], ret)
            if sp is None:
                continue
            # the expected value must be in the context exactly as the test prints it
            ctx_text = _decode_cache(c["context_ids"])
            if ctx_text.count(str(ret)) == 0 or f"== {ret}" not in ctx_text:
                continue
            seen.add((inp, ret))
            # confounded when the expected value also appears in the task prompt (a docstring example), so editing
            # only the test cannot change the output.
            in_prompt = str(ret) in rec_prompt
            out.append({"turn": a["turn"], "input": inp, "expected": ret, "span": sp, "call": c,
                        "label": rec["label"], "split": rec["split"], "value_in_prompt": in_prompt})
    return out


_TOK = None
_DEC = {}


def _decode_cache(ids):
    key = (len(ids), ids[-1] if ids else 0, ids[0] if ids else 0)
    if key not in _DEC:
        _DEC[key] = _TOK.decode(ids, skip_special_tokens=False)
    return _DEC[key]


def return_literal_span(completion_ids, value):
    """[start, end) completion tokens of the literal `value` that directly follows a `return` (the hardcoded
    output, not a mention of the number in the thinking)."""
    text = _TOK.decode(completion_ids, skip_special_tokens=False)
    for m in RET.finditer(text):
        if m.group(1) == str(value):
            c0, c1 = m.start(1), m.end(1)
            return replay.token_span_by_char(_TOK, completion_ids, c0, c1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    global _TOK
    model, _TOK, device = replay.load(a.model)
    tasks = {t["task_id"]: t for t in agent.load()[0]}
    recs = []
    for d in a.runs:
        arm = os.path.basename(os.path.normpath(d))
        for l in open(os.path.join(d, "gen", "transcripts.jsonl")):
            r = json.loads(l)
            r["arm"] = arm
            recs.append(r)
    results = []
    for r in recs:
        for s in find_spans(r, tasks[r["task_id"]][r["split"]]):
            c = s.pop("call")
            lo, hi = s["span"]
            nv = new_value(s["expected"])
            ctx = c["context_ids"]
            comp_prefix = c["completion_ids"][:lo]
            base = replay.greedy(model, _TOK, ctx + comp_prefix, max_new=hi - lo + 3, device=device)
            ed_ctx, ok = replay.edit_context(_TOK, ctx, f"== {s['expected']}", f"== {nv}")
            if not ok:
                ed_ctx, ok = replay.edit_context(_TOK, ctx, str(s["expected"]), str(nv))
            ed = replay.greedy(model, _TOK, ed_ctx + comp_prefix, max_new=hi - lo + 3, device=device) if ok else []
            base_txt, ed_txt = _TOK.decode(base), _TOK.decode(ed)
            orig_tok = c["completion_ids"][lo]
            lg_base = replay.replay_logits(model, ctx, comp_prefix + [orig_tok], device=device)[-1].log_softmax(-1)
            lg_ed = replay.replay_logits(model, ed_ctx, comp_prefix + [orig_tok], device=device)[-1].log_softmax(-1) if ok else lg_base
            s.update({"arm": r["arm"], "run_id": r["run_id"], "task_id": r["task_id"], "new_value": nv, "edit_ok": ok,
                      "baseline_decode": base_txt, "edited_decode": ed_txt,
                      "baseline_is_original": str(s["expected"]) in base_txt,
                      "follows_edit": bool(ok and str(nv) in ed_txt and str(s["expected"]) not in ed_txt),
                      "logprob_orig_token_baseline": float(lg_base[orig_tok]),
                      "logprob_orig_token_edited": float(lg_ed[orig_tok])})
            results.append(s)
            print(f"{r['arm']}/{r['run_id']} in={s['input']} exp={s['expected']}->{nv} "
                  f"base={base_txt!r} edit={ed_txt!r} follows={s['follows_edit']}", flush=True)
            if a.limit and len(results) >= a.limit:
                break
        if a.limit and len(results) >= a.limit:
            break
    usable = [x for x in results if x["edit_ok"] and x["baseline_is_original"]]
    clean = [x for x in usable if not x["value_in_prompt"]]
    summ = {"spans": len(results), "usable": len(usable), "clean_test_only": len(clean),
            "follows_clean": sum(x["follows_edit"] for x in clean),
            "follow_fraction_clean": round(sum(x["follows_edit"] for x in clean) / max(len(clean), 1), 3),
            "follows_all_usable": sum(x["follows_edit"] for x in usable),
            "runs_with_spans": len(set((x["arm"], x["run_id"]) for x in results)),
            "by_label": {}}
    for lab in sorted(set(x["label"] for x in results)):
        sub = [x for x in clean if x["label"] == lab]
        summ["by_label"][lab] = {"clean": len(sub), "follows": sum(x["follows_edit"] for x in sub)}
    out = {"summary": summ, "model": a.model, "device": device, "spans": results}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
