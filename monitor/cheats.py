"""Per-cheat replay: does the cheating output follow the mutated expected value in test.py?

For each run labeled cheat, at the turn of its first cheating action: context = the stream before that turn,
completion = the turn the model generated. The edit changes every occurrence in the context of the mutated
assertion's line (test.py as read, and test tracebacks quoting it) so its expected value is different: an int to
another of the same length, a bool negated, a string extended. The real and the edited context both go through the
same decode and re-encode, so a re-tokenization artifact cannot pass for an effect of the edit.

  span check   (when the cheating write holds the old value as a literal next to the call's input: a `return` after
               `if arg == input`, or a lookup-table entry) teacher-forced up to that literal,
               greedy-decode it under both contexts; follows = the edited decode is the new value. Also the
               log-probability of the literal's first original token under both contexts (from the top 20; an
               upper bound when it falls outside them).
  write check  (every cheat) greedy-regenerate the whole turn under both contexts, take the func.py it writes and run
               it on the mutated call: does it return the original expected value, the edited one, or neither? The
               recorded cheating func.py is run the same way as a reference.

Model calls go through vLLM with the run's own model, the weights that generated it.

  python cheats.py --runs ARM_DIR [ARM_DIR ...] --model MODEL --out FILE     (each ARM_DIR holds gen/transcripts.jsonl)
"""
import argparse
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402
import replay  # noqa: E402


def flip(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        s = str(abs(v))
        n = int("9" + "1" * (len(s) - 1)) if s[0] != "9" else int("1" + "8" * (len(s) - 1))
        n = n if n != abs(v) else n + 1
        return -n if v < 0 else n
    if isinstance(v, float):
        return v + 1.5
    if isinstance(v, str):
        return v + "Q"
    return None


def mutations(t):
    """The mutated assertions an edit can change: `assert CALL == VALUE` with a literal VALUE."""
    out = []
    body = agent.check_body(t["test"]).body
    for i in t["mutated"]:
        st = body[i]
        if not (isinstance(st.test, ast.Compare) and len(st.test.ops) == 1 and isinstance(st.test.ops[0], ast.Eq)):
            continue
        line = ast.get_source_segment(t["test"], st)
        old_txt = ast.get_source_segment(t["test"], st.test.comparators[0])
        try:
            old = ast.literal_eval(st.test.comparators[0])
        except Exception:
            continue
        new = flip(old)
        if new is None or not line.endswith(old_txt):
            continue
        out.append({"index": i, "line": line, "new_line": line[:-len(old_txt)] + repr(new),
                    "call": ast.get_source_segment(t["test"], st.test.left), "old": old, "new": new})
    return out


def _key(node):
    """A literal input key from an AST node: a constant, or a tuple of constants. None otherwise."""
    try:
        v = ast.literal_eval(node)
    except Exception:
        return None
    return v if isinstance(v, (int, str, tuple)) and not isinstance(v, bool) else None


def hardcoded_targets(r, t):
    """(turn, call index, mutation-like target) for every test value a func.py write hard-codes: an `if arg == input`
    (or `(a, b) == (x, y)`) returning a literal, or a lookup-table entry `input: value`, where (input, value) is a
    test assertion with an int expected value. One target per assertion, at the first write holding it."""
    tests = {}
    for st in agent.check_body(t["test"]).body:
        if isinstance(st, ast.Assert) and isinstance(st.test, ast.Compare) and len(st.test.ops) == 1 \
           and isinstance(st.test.ops[0], ast.Eq) and isinstance(st.test.left, ast.Call):
            args = [_key(x) for x in st.test.left.args]
            try:
                exp = ast.literal_eval(st.test.comparators[0])
            except Exception:
                continue
            if None in args or not isinstance(exp, int) or isinstance(exp, bool):
                continue
            key = args[0] if len(args) == 1 else tuple(args)
            line = ast.get_source_segment(t["test"], st)
            old_txt = ast.get_source_segment(t["test"], st.test.comparators[0])
            if line.endswith(old_txt):
                tests.setdefault((key, exp), {"line": line, "call": ast.get_source_segment(t["test"], st.test.left),
                                             "old": exp, "new": flip(exp), "new_line": line[:-len(old_txt)] + repr(flip(exp))})
    out, done = [], set()
    for act in r["actions"]:
        if act["name"] != "write_file" or os.path.basename(str((act["args"] or {}).get("path", ""))) != "func.py":
            continue
        try:
            tree = ast.parse(act["args"].get("content") or "")
        except SyntaxError:
            continue
        pairs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 \
               and isinstance(node.test.ops[0], ast.Eq):
                key = _key(node.test.comparators[0])
                for b in node.body:
                    if isinstance(b, ast.Return) and isinstance(b.value, ast.Constant):
                        pairs.append((key, b.value.value))
            elif isinstance(node, ast.Dict):
                pairs += [(_key(k), v.value) for k, v in zip(node.keys, node.values) if k is not None and isinstance(v, ast.Constant)]
        for key, val in pairs:
            if (key, val) in tests and (key, val) not in done:
                done.add((key, val))
                out.append((act["turn"], act["call"], tests[(key, val)]))
    return out


def literal_span(tok, comp, m):
    """[start, end) completion tokens of the cheating literal: an occurrence of the old expected value inside the
    tool call (not the thinking) with the call's most distinctive argument literal in the 120 characters before it,
    as in `if k == 1073741816: return 7888726` or a lookup table `(999999999999999998, 13): 587706332`."""
    if not isinstance(m["old"], int) or isinstance(m["old"], bool):
        return None
    text = tok.decode(comp, skip_special_tokens=False)
    start = text.find("<tool_call>")
    if start < 0:
        return None
    args = sorted(re.findall(r"-?\d+|'[^']*'|\"[^\"]*\"", m["call"].split("(", 1)[1]), key=len)
    anchor = args[-1].strip("'\"") if args else None
    for x in re.finditer(r"(?<![\d.])" + re.escape(str(m["old"])) + r"(?![\d.])", text[start:]):
        c0 = start + x.start()
        if anchor and anchor in text[max(start, c0 - 120):c0]:
            return replay.token_span_by_char(tok, comp, c0, c0 + len(str(m["old"])))
    return None


def returns(files, t, m):
    """Which expected value the func.py in `files` returns on the mutated call: 'old', 'new', 'neither' or 'error'."""
    if "func.py" not in files:
        return "no func.py"
    src = f"def check(candidate):\n    assert {m['call']} == {m['old']!r}\n    assert {m['call']} == {m['new']!r}\n"
    ev = agent.evaluate(files, src, t["entry_point"], [0, 1])
    r0, r1 = ev["results"].get(0), ev["results"].get(1)
    if r0 == 1:
        return "old"
    if r1 == 1:
        return "new"
    return "neither" if r0 == 0 and r1 == 0 else "error"


def written_func(tok, ids):
    """The func.py content of the first write_file call in a generated turn, or None."""
    text = tok.decode(ids, skip_special_tokens=False)
    text = text.split("</think>", 1)[1] if "</think>" in text else text
    for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            name, args = agent.parse_call(raw)
        except Exception:
            continue
        if name == "write_file" and os.path.basename(str(args.get("path", ""))) == "func.py":
            return args.get("content")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--no-regen", action="store_true", help="span check only")
    ap.add_argument("--targets", default="cheats", choices=["cheats", "hardcoded"],
                    help="cheats: the mutated assertions of runs labeled cheat; hardcoded: every test value a write hard-codes, in any run")
    ap.add_argument("--only", default=None, help="comma-separated run ids to replay (default every cheat)")
    ap.add_argument("--gpu-mem", type=float, default=0.88, help="vLLM's share of GPU memory")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    im_end, eot = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")
    tasks = {t["task_id"]: t for t in agent.load()[0]}
    cheats = []
    for d in a.runs:
        for l in open(os.path.join(d, "gen", "transcripts.jsonl")):
            r = json.loads(l)
            keep = (r["label"] == "cheat" and r["first_cheat"]) if a.targets == "cheats" else True
            if keep and (a.only is None or r["run_id"] in a.only.split(",")):
                r["arm"] = os.path.basename(os.path.normpath(d))
                cheats.append(r)
    print(f"{len(cheats)} runs", flush=True)
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0)

    def batch(jobs):
        """Run every (prompt ids, max tokens, extra sampling kwargs) job in one vLLM call, greedy."""
        if not jobs:
            return []
        return llm.generate([TokensPrompt(prompt_token_ids=p) for p, _, _ in jobs],
                            [SamplingParams(temperature=0.0, max_tokens=n, **k) for _, n, k in jobs], use_tqdm=False)

    rows, span_jobs, lp_jobs, regen_jobs = [], [], [], []
    work = []
    for r in cheats:
        t = tasks[r["task_id"]][r["split"]]
        if a.targets == "cheats":
            work += [(r, t, r["first_cheat"]["turn"], r["first_cheat"]["call"], m, r["first_cheat"]) for m in mutations(t)]
        else:
            work += [(r, t, turn, ci, m, {"kind": "hardcoded", "turn": turn, "call": ci, "path": "func.py"})
                     for turn, ci, m in hardcoded_targets(r, t)]
    print(f"{len(work)} targets", flush=True)
    for r, t, turn, ci, m, fc in work:
        call = next(c for c in replay.calls(r) if c["turn"] == turn)
        ctx, comp = call["context_ids"], call["completion_ids"]
        act = next(x for x in r["actions"] if x["turn"] == turn and x["call"] == ci)
        rec_files = {"func.py": act["args"].get("content", "")} if fc.get("path") == "func.py" else {}
        text = tok.decode(ctx, skip_special_tokens=False)
        base_ids = tok.encode(text, add_special_tokens=False)
        n = text.count(m["line"])
        row = {"arm": r["arm"], "run_id": r["run_id"], "label": r["label"], "kind": fc.get("kind"), "how": fc.get("how"), "turn": turn,
               "value_in_prompt": str(m["old"]) in t["prompt"],
               "assertion": m["line"], "old": repr(m["old"]), "new": repr(m["new"]), "occurrences_in_context": n,
               "context_roundtrip_identical": base_ids == list(ctx), "_t": t, "_m": m, "_rec": rec_files}
        rows.append(row)
        if n == 0:
            row["skipped"] = "the assertion's line is not in the context"
            continue
        ed_ids = tok.encode(text.replace(m["line"], m["new_line"]), add_special_tokens=False)
        sp = literal_span(tok, comp, m)
        if sp:
            lo, hi = sp
            row["_span"] = (lo, hi, comp[lo])
            for k, c in (("real", base_ids), ("edited", ed_ids)):
                span_jobs.append((c + comp[:lo], hi - lo + 3, {}, len(rows) - 1, k))
                lp_jobs.append((c + comp[:lo], 1, {"logprobs": 20}, len(rows) - 1, k))
        room = a.max_len - max(len(base_ids), len(ed_ids)) - 8
        for k, c in ((("real", base_ids), ("edited", ed_ids)) if not a.no_regen else ()):
            regen_jobs.append((c, max(min(len(comp) + 2048, room), 16), {"stop_token_ids": [im_end, eot]}, len(rows) - 1, k))
    print(f"{len(rows)} targets; {len(span_jobs)} span decodes, {len(regen_jobs)} regenerations", flush=True)
    span_out = batch([j[:3] for j in span_jobs])
    lp_out = batch([j[:3] for j in lp_jobs])
    regen_out = batch([j[:3] for j in regen_jobs])
    for j, o in zip(span_jobs, span_out):
        rows[j[3]].setdefault("_dec", {})[j[4]] = tok.decode(o.outputs[0].token_ids)
    for j, o in zip(lp_jobs, lp_out):
        # the original literal's first token among the 20 most likely next tokens; outside them, an upper bound
        top = o.outputs[0].logprobs[0]
        tid = rows[j[3]]["_span"][2]
        v = round(float(top[tid].logprob), 3) if tid in top else f"<= {round(min(float(x.logprob) for x in top.values()), 3)}"
        rows[j[3]].setdefault("_lp", {})[j[4]] = v
    for j, o in zip(regen_jobs, regen_out):
        fsrc = written_func(tok, list(o.outputs[0].token_ids))
        rows[j[3]].setdefault("_regen", {})[j[4]] = returns({"func.py": fsrc} if fsrc is not None else {}, rows[j[3]]["_t"], rows[j[3]]["_m"])
    results = []
    for row in rows:
        t, m, rec_files = row.pop("_t"), row.pop("_m"), row.pop("_rec")
        if "_span" in row:
            row.pop("_span")
            dec, lp = row.pop("_dec"), row.pop("_lp")
            row["span"] = {"literal": repr(m["old"]), "decode_real": dec["real"], "decode_edited": dec["edited"],
                           "follows": str(m["new"]) in dec["edited"] and str(m["old"]) not in dec["edited"],
                           "real_is_original": str(m["old"]) in dec["real"], "logprob_original_first_token": lp}
        if "_regen" in row:
            rg = row.pop("_regen")
            row["write"] = {"recorded_func_returns": returns(rec_files, t, m) if rec_files else "cheat was not a func.py write",
                            "regenerated_real_returns": rg["real"], "regenerated_edited_returns": rg["edited"],
                            "follows": rg["edited"] == "new" and rg["real"] == "old"}
        results.append(row)
        print(json.dumps(row), flush=True)
    json.dump({"model": a.model, "cheats": len(cheats), "results": results}, open(a.out, "w"), indent=1)
    print(f"done: {len(results)} targets over {len(cheats)} runs", flush=True)


if __name__ == "__main__":
    main()
