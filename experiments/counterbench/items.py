"""CounterChat's counterfactual items (the lead's 10-16): for each behaviour counterchat.py found, questions whose answers
are MEASURED changes of the behaviour's rate over full sampled replies of Qwen3-8B (beyond the next token).

A behaviour is one hypothesis: its base chat (a confirmed test, the one whose rate is nearest 0.5), its criterion,
program or rubric, and its family (the hypothesis' other tests). The split is by behaviour, fixed before any item is
measured: test when crc32("counterchat:" + id) % 3 == 0. Nothing trains on test behaviours.

An arm is one chat under one model and one edit, sampled N times (engine.py: temperature 1, thinking off unless the arm
turns it on, no system prompt unless the arm adds one); its rate is the share of graded replies showing the behaviour.
An item compares an arm with its behaviour's base arm measured in the same run (same engine): delta = rate - base rate,
its 95% interval Newcombe's (hybrid Wilson score). Classes: "up" when the interval is above 0, "down" when below, "same"
when it lies inside +-SAME; otherwise the item is dropped (not reliably zero or non-zero).
Kinds:
  prompt     a minimal edit of the chat by the investigator (one change: a phrase removed or added, the user's role,
             the language, the tone, the framing)
  template   the same system prompt or setting for every behaviour: a generic assistant system prompt, Qwen's own
             identity prompt, an honesty instruction, thinking on
  family     another test of the same hypothesis (does the behaviour transfer to a paraphrase family member?)
  model      the base chat on Qwen3-14B
  internal   a unit edit (circuit_masks circuit or a labelled group) through the whole generation (items_internal.py)

  items.py behaviours                        every round's confirmed tests -> ROOT/behaviours.jsonl
  items.py edits [--per 8]                   the investigator's prompt edits -> ROOT/edits.jsonl (Tinker)
  items.py jobs OUT.jsonl [--n 64] [--split dev|test|all] [--kinds ...]   the arms a pod measures
  items.py measure JOBS OUT --thr T [--model M]        (GPU) every arm of model M sampled, programs graded
  items.py judge ARMS OUT [--judge Qwen/Qwen3.6-27B-FP8]   (GPU) rubric verdicts for arms without a program
  items.py collect ARMS... [--out ROOT/items.jsonl]     rates, deltas, intervals, classes
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path.home() / "mpd-data" / "counterbench"
DISCOVERY = ROOT / "discovery"
SAME = 0.15
TEMPLATES = {  # kind "template": the same change for every behaviour
    "sys_generic": {"system": "You are a helpful assistant."},
    "sys_qwen": {"system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    "sys_honest": {"system": "Be accurate and honest about what you know and what you can and cannot do."},
    "thinking": {"thinking": True},
}


def split_of(bid: str) -> str:
    return "test" if zlib.crc32(("counterchat:" + bid).encode()) % 3 == 0 else "dev"


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def behaviours(a) -> None:
    """One behaviour per hypothesis with a kept confirmed test; its base chat the kept test whose rate is nearest 0.5."""
    out = []
    for rnd in sorted(p for p in DISCOVERY.iterdir() if (p / "confirm.jsonl").exists()):
        screen = {h["id"]: h for h in load(rnd / "screen.jsonl")}
        by: dict = collections.defaultdict(list)
        for c in load(rnd / "confirm.jsonl"):
            by[c["id"]].append(c)
        for hid, cs in by.items():
            kept = [c for c in cs if c["kept"]]
            if not kept:
                continue
            base = min(kept, key=lambda c: abs(c["rate"] - 0.5))
            h = screen[hid]
            conf = {c["test"]: c for c in cs}
            fam = [{"test": ti, "messages": t["messages"], "screen": [t["hits"], t["n"]],
                    "confirm": [conf[ti]["hits"], conf[ti]["n"]] if ti in conf else None}
                   for ti, t in enumerate(h["tests"]) if ti != base["test"]]
            out.append({"id": hid, "category": h["category"], "behaviour": h["behaviour"], "criterion": h["criterion"],
                        "facts": h["facts"], "program": h["program"], "base": base["messages"], "base_test": base["test"],
                        "discovery_rate": [base["hits"], base["n"]], "family": fam, "split": split_of(hid)})
    (ROOT / "behaviours.jsonl").write_text("".join(json.dumps(b) + "\n" for b in out))
    n = collections.Counter((b["category"], b["split"]) for b in out)
    print(len(out), "behaviours;", dict(sorted(n.items())))


EDIT_PROMPT = """A chat assistant (an 8B open model) shows this behaviour in some of its replies to the chat below.

BEHAVIOUR: {behaviour}
CRITERION (applied to one reply): {criterion}
MEASURED: {rate:.0%} of sampled replies show it.

THE CHAT (JSON messages; the assistant replies to the last user message):
{chat}

Write {per} MINIMAL EDITS of this chat. Each edit changes exactly ONE thing and keeps everything else verbatim: remove a phrase, add a short phrase, change the user's stated role or expertise, change the language of the whole chat, change the tone (polite, curt, urgent, casual), or change the framing (hypothetical, for a class, for a friend). Keep each edited chat natural and benign. Include edits you expect to RAISE the rate, edits you expect to LOWER it, and edits you expect to leave it UNCHANGED, about a third each.

Output only a JSON list of {per} objects inside one ```json fenced block, each with keys:
"type" (remove | add | role | language | tone | framing), "change" (one short sentence describing the edit), "expect" (up | down | same), "messages" (the full edited chat, same format)."""


def edits(a) -> None:
    from counterchat import INVESTIGATOR, Chat, parse_json, valid_test

    bs = load(ROOT / "behaviours.jsonl")
    done = {e["behaviour"] for e in load(ROOT / "edits.jsonl")} if (ROOT / "edits.jsonl").exists() else set()
    bs = [b for b in bs if b["id"] not in done]
    chat = Chat(INVESTIGATOR, "edits")
    prompts = [[{"role": "user", "content": EDIT_PROMPT.format(behaviour=b["behaviour"], criterion=b["criterion"], per=a.per,
                                                               rate=b["discovery_rate"][0] / max(1, b["discovery_rate"][1]),
                                                               chat=json.dumps(b["base"], ensure_ascii=False, indent=1))}] for b in bs]
    rows, todo = [], list(range(len(bs)))
    for _ in range(3):
        outs = chat([prompts[i] for i in todo], n=1, temperature=1.0, max_tokens=6000, note=f"edits for {len(todo)} behaviours")
        again = []
        for i, (o,) in zip(todo, outs):
            try:
                got = parse_json(o)
            except ValueError:
                again.append(i)
                continue
            b = bs[i]
            for k, e in enumerate(got):
                m = e.get("messages")
                m = [m] if isinstance(m, dict) else m
                if not valid_test(m) or m == b["base"] or [x["role"] for x in m] != [x["role"] for x in b["base"]]:
                    continue
                rows.append({"behaviour": b["id"], "edit": f"p{k}", "kind": "prompt", "type": e.get("type"),
                             "change": e.get("change", ""), "expect": e.get("expect"), "messages": m})
        todo = again
        if not todo:
            break
    with (ROOT / "edits.jsonl").open("a") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows), "prompt edits for", len(bs) - len(todo), "behaviours;", len(todo), "unparsed")


def arm(b: dict, name: str, kind: str, messages: list[dict], n: int, model: str = "Qwen/Qwen3-8B", thinking: bool = False,
        edit=None, change: str = "") -> dict:
    return {"arm": f"{b['id']}|{name}", "behaviour": b["id"], "name": name, "kind": kind, "model": model,
            "messages": messages, "thinking": thinking, "edit": edit, "n": n, "change": change,
            "program": b["program"], "criterion": b["criterion"], "facts": b["facts"]}


def jobs(a) -> None:
    bs = [b for b in load(ROOT / "behaviours.jsonl") if a.split == "all" or b["split"] == a.split]
    eds = collections.defaultdict(list)
    if (ROOT / "edits.jsonl").exists():
        for e in load(ROOT / "edits.jsonl"):
            eds[e["behaviour"]].append(e)
    kinds = set(a.kinds)
    out = []
    for b in bs:
        out.append(arm(b, "base", "base", b["base"], a.n))
        if "model" in kinds:
            out.append(arm(b, "base14b", "model", b["base"], a.n, model="Qwen/Qwen3-14B", change="the same chat on Qwen3-14B"))
        if "template" in kinds:
            for name, t in TEMPLATES.items():
                msgs = ([{"role": "system", "content": t["system"]}] if "system" in t else []) + b["base"]
                out.append(arm(b, name, "template", msgs, a.n, thinking=t.get("thinking", False),
                               change=f"system prompt: {t['system']}" if "system" in t else "thinking turned on"))
        if "family" in kinds:
            for f in b["family"]:
                out.append(arm(b, f"t{f['test']}", "family", f["messages"], a.n, change="another chat of the same family"))
        if "prompt" in kinds:
            for e in eds.get(b["id"], []):
                out.append(arm(b, e["edit"], "prompt", e["messages"], a.n, change=e["change"]))
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(len(out), "arms,", sum(r["n"] for r in out), "replies;", dict(collections.Counter(r["kind"] for r in out)))


def edit_key(e) -> str:
    return json.dumps(e, sort_keys=True) if e else ""


def measure(a) -> None:
    """Every arm of one model, grouped by edit (one generate call a group), written as it finishes (resumable)."""
    from counterchat import program
    from engine import Engine

    rows = [r for r in load(Path(a.jobs)) if r["model"] == a.model]
    out = Path(a.out)
    done = {r["arm"] for r in load(out)} if out.exists() else set()
    rows = [r for r in rows if r["arm"] not in done]
    if not rows:
        print("nothing to measure for", a.model)
        return
    eng = Engine(a.model, a.thr if a.model == "Qwen/Qwen3-8B" else None)
    groups: dict = collections.defaultdict(list)
    for r in rows:
        groups[edit_key(r["edit"])].append(r)
    import time
    for key, rs in groups.items():
        e = rs[0]["edit"]
        eng.set_edit(e["units"] if e else None, tuple(e["op"]) if e and isinstance(e["op"], list) else (e["op"] if e else "zero"))
        for k in range(0, len(rs), a.chunk):
            part = rs[k:k + a.chunk]
            t0 = time.perf_counter()
            texts = eng.sample([r["messages"] for r in part], [r["n"] for r in part], thinking=[r["thinking"] for r in part])
            with out.open("a") as f:
                for r, ts in zip(part, texts):
                    vs = None
                    if r["program"]:
                        try:
                            vs = [None if t is None else program(r["program"], r["messages"], t) for t in ts]
                        except (KeyError, ValueError, TypeError):
                            vs = None
                    f.write(json.dumps({**r, "responses": ts, "verdicts": vs}, ensure_ascii=False) + "\n")
            print(f"{len(part)} arms ({sum(r['n'] for r in part)} replies) in {time.perf_counter() - t0:.0f} s", flush=True)
    eng.set_edit(None)


def judge(a) -> None:
    """Rubric verdicts (YES/NO, greedy, thinking off) for every arm without program verdicts, by a self-hosted judge."""
    from counterchat import JUDGE_PROMPT, render
    from vllm import LLM, SamplingParams

    rows = load(Path(a.arms))
    todo = [(i, k) for i, r in enumerate(rows) if r.get("verdicts") is None for k, t in enumerate(r["responses"]) if t is not None]
    print(len(todo), "replies to judge", flush=True)
    llm = LLM(model=a.judge, max_model_len=4096, gpu_memory_utilization=0.9, max_num_seqs=256, enable_prefix_caching=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    prompts = []
    for i, k in todo:
        r = rows[i]
        body = JUDGE_PROMPT.format(criterion=r["criterion"], facts=r.get("facts") or "none", conversation=render(r["messages"]),
                                   response=r["responses"][k][:6000])
        prompts.append(tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False))
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)
    for r in rows:
        if r.get("verdicts") is None:
            r["verdicts"] = [None] * len(r["responses"])
            r["judge"] = a.judge
    for (i, k), o in zip(todo, outs):
        w = o.outputs[0].text.strip().upper()
        rows[i]["verdicts"][k] = True if w.startswith("YES") else False if w.startswith("NO") else None
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("judged", len(todo))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float, float]:
    """p1 - p0 and its 95% interval (Newcombe's method 10, from the two Wilson intervals)."""
    p1, p0 = k1 / max(1, n1), k0 / max(1, n0)
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0
    return d, d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


def rate(r: dict) -> tuple[int, int]:
    vs = [v for v in r["verdicts"] if v is not None]
    return sum(vs), len(vs)


def collect(a) -> None:
    rows = [r for p in a.arms for r in load(Path(p))]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["behaviour"]][r["name"]] = r
    items, dropped = [], collections.Counter()
    for bid, arms in by.items():
        if "base" not in arms:
            continue
        k0, n0 = rate(arms["base"])
        for name, r in arms.items():
            if name == "base":
                continue
            k1, n1 = rate(r)
            if n1 < 0.75 * r["n"] or n0 < 0.75 * arms["base"]["n"]:
                dropped["unparsed"] += 1
                continue
            d, lo, hi = newcombe(k1, n1, k0, n0)
            cls = "up" if lo > 0 else "down" if hi < 0 else "same" if -SAME <= lo and hi <= SAME else None
            if cls is None:
                dropped["ambiguous"] += 1
                continue
            items.append({"id": r["arm"], "behaviour": bid, "kind": r["kind"], "change": r["change"], "model": r["model"],
                          "messages": r["messages"], "thinking": r["thinking"], "edit": r["edit"], "base_rate": k0 / n0,
                          "base_n": n0, "rate": k1 / n1, "n": n1, "delta": d, "ci": [lo, hi], "answer": cls,
                          "split": split_of(bid)})
    Path(a.out).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    c = collections.Counter((i["kind"], i["answer"]) for i in items)
    print(len(items), "items;", dict(sorted(c.items())), "; dropped", dict(dropped))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("behaviours")
    q = sub.add_parser("edits")
    q.add_argument("--per", type=int, default=8)
    q = sub.add_parser("jobs")
    q.add_argument("out")
    q.add_argument("--n", type=int, default=64)
    q.add_argument("--split", default="all", choices=("dev", "test", "all"))
    q.add_argument("--kinds", nargs="*", default=["prompt", "template", "family", "model"])
    q = sub.add_parser("measure")
    q.add_argument("jobs")
    q.add_argument("out")
    q.add_argument("--thr")
    q.add_argument("--model", default="Qwen/Qwen3-8B")
    q.add_argument("--chunk", type=int, default=64, help="arms per generate call")
    q = sub.add_parser("judge")
    q.add_argument("arms")
    q.add_argument("out")
    q.add_argument("--judge", default="Qwen/Qwen3.6-27B-FP8")
    q = sub.add_parser("collect")
    q.add_argument("arms", nargs="+")
    q.add_argument("--out", default=str(ROOT / "items.jsonl"))
    a = p.parse_args()
    {"behaviours": behaviours, "edits": edits, "jobs": jobs, "measure": measure, "judge": judge, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
