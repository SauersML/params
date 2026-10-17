"""The released activation oracle trained further on circuits given as text (#2951; the user's plan: "We will take the
existing trained activation oracles and train them further on our datasets, given circuits as input. Success here looks
like our circuits provide uplift on tasks over the default activation oracle"; the lead's 10-16).

Every example is an AO example: the context's residual stream at half depth (Qwen3-8B's layer 18, every token of the
context) is added at the placeholders the AO's own way (cao.Oracle), and the question follows as text. Two arms train from
the released AO 8B adapter on the same questions with the same AO-mix replay; only the text differs:
  circ  textoracle's serialized circuit before the question (text_reader.user_text "full": the finder's heads, groups,
        members, values, measured shares, writes) and its "Read:" recap before the answer;
  none  the text alone (text_reader's "none": the text, the model's next token and the alternatives), the answer alone
        (a recap of numbers the input does not have would be guessed).
The uplift is circ against none, against the released AO and against textoracle's reader trained from the base model on
the same rows, on oracleeval's scorecard (O1, O1m) and AO's own suite (taboo, PersonaQA, classification).

  ao_text.py build --rc RC --facts FACTS --extra MEMBERS.jsonl --ref TRAIN_FULL.jsonl --tests ROWS_DIR --out DIR
                   [--first] [--focus] [--recap] [--balance]
      DIR/train_{circ,none}.jsonl, DIR/test_{circ,none}.jsonl (rc's held-out questions) and DIR/test_members_{circ,none}
      .jsonl (the member slice) as cao examples. --ref: textoracle's train_full.jsonl, which the circ rows must equal
      (the same questions, serialization and answers); --tests: textoracle's rows directory (its test_{full,none} and
      test_members_{full,none}, taken as they are).
  ao_text.py answers ADAPTER TEST.jsonl OUT.jsonl [--batch 8]    greedy answers in textoracle's eval format
  ao_text.py collect NAME LABEL --o1 O1.jsonl --o1m O1M.jsonl    the answers into the scorecard (answers/NAME/)
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LAYER = 18  # Qwen3-8B's half depth (36 layers), the read layer of the released AO's evaluations
ANSWER_TOKENS = 260  # the "Read:" line, the sentence and the claim


def example(ctx: list[int], q: str, a: str, **info) -> dict:
    return {"ctx": ctx, "blocks": [{"layer": LAYER, "positions": list(range(len(ctx)))}], "q": q, "a": a, **info}


def build(a) -> None:
    import text_reader as tr
    from reads_circuit import answer_class

    F = json.loads(a.facts.read_text())
    train, test, _ = tr.load_rc(a.rc)
    held = {e["id"] for e in test}
    for f in a.extra or []:  # (text_reader.rows' order and checks)
        more = [json.loads(x) for x in f.read_text().splitlines()]
        assert not {e["id"] for e in more} & held, f
        train += more
    if a.balance:  # text_reader.rows' balance, the same draws
        rng = random.Random(0)
        by = collections.defaultdict(list)
        for e in train:
            by[(e["cat"], answer_class(e))].append(e)
        design = {tuple(k.split("/")) for k in json.loads((a.rc / "claim.json").read_text())["classes"]}
        bal = []
        for c in sorted({e["cat"] for e in train}):
            ks = [k for k in by if k[0] == c]
            top = max(len(by[k]) for k in ks if k in design)
            for k in ks:
                bal += by[k] + ([rng.choice(by[k]) for _ in range(top - len(by[k]))] if k in design else [])
        train = bal
    ref = [json.loads(x) for x in a.ref.read_text().splitlines()] if a.ref else None
    assert ref is None or len(ref) == len(train), (len(ref), len(train))
    a.out.mkdir(parents=True, exist_ok=True)
    for arm, how in (("circ", "full"), ("none", "none")):
        with (a.out / f"train_{arm}.jsonl").open("w") as fh:
            for k, e in enumerate(train):
                q = tr.user_text(F[e["id"]], e, how, first=a.first, focus=a.focus)
                ans = (tr.recap(F[e["id"]], e, how) + "\n" if a.recap and how == "full" else "") + e["a"]
                if ref is not None and how == "full":
                    assert ref[k]["prompt"][0]["content"] == q and ref[k]["completion"][0]["content"] == ans, k
                fh.write(json.dumps(example(e["ctx"], q, ans, cat=e["cat"], task=f"{e['id']}#{k}")) + "\n")
        print(f"train_{arm}: {len(train)}", flush=True)
    ctx = {e["id"]: e["ctx"] for e in test}
    for f in a.extra or []:
        ctx.update({e["id"]: e["ctx"] for e in map(json.loads, f.read_text().splitlines())})
    for name in ("test", "test_members"):
        for arm, how in (("circ", "full"), ("none", "none")):
            src = a.tests / f"{name}_{how}.jsonl"
            if not src.exists():
                continue
            rows = [json.loads(x) for x in src.read_text().splitlines()]
            with (a.out / f"{name}_{arm}.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(example(ctx[r["id"]], r["question"], r["a"], cat=r["cat"], task=r["task"], id=r["id"])) + "\n")
            print(f"{name}_{arm}: {len(rows)}", flush=True)


def answers(a) -> None:
    import cao
    o = cao.Oracle(a.adapter)
    o.m.eval()
    rows = [json.loads(x) for x in a.test.read_text().splitlines()]
    order = sorted(range(len(rows)), key=lambda k: len(rows[k]["q"]))  # (similar lengths batched: less padding)
    got = [None] * len(rows)
    for s in range(0, len(order), a.batch):
        ks = order[s:s + a.batch]
        for k, g in zip(ks, o.answers([rows[k] for k in ks], max_new=ANSWER_TOKENS)):
            got[k] = g
        print(f"{min(s + a.batch, len(rows))}/{len(rows)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for r, g in zip(rows, got):
            fh.write(json.dumps({"task": r["task"], "id": r["id"], "cat": r["cat"], "a": r["a"], "got": g}) + "\n")


def collect(a) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scorecard"))
    import scorecard as sc
    test = sc.jl(sc.RC / "rc1" / "test_correct.jsonl")
    d = sc.oracle_meta(a.name, a.label, "reader", a.reads)
    if a.o1:
        got = sc.jl(a.o1)
        assert len(got) == len(test) and all(g["id"] == t["id"] for g, t in zip(got, test)), a.o1
        sc.wjl(d / "o1.jsonl", [{"item": sc.o1_id(t), "answer": g["got"]} for g, t in zip(got, test)])
    if a.o1m:
        m1 = sc.jl(sc.ITEMS / "o1_members.jsonl")
        got = sc.jl(a.o1m)
        assert len(got) == len(m1) and all(g["id"] == t["context"] for g, t in zip(got, m1)), a.o1m
        sc.wjl(d / "o1m.jsonl", [{"item": t["item"], "answer": g["got"]} for g, t in zip(got, m1)])
    print("collected", d)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rc", type=Path, required=True)
    b.add_argument("--facts", type=Path, required=True)
    b.add_argument("--extra", type=Path, nargs="*")
    b.add_argument("--ref", type=Path)
    b.add_argument("--tests", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    for f in ("first", "focus", "recap", "balance"):
        b.add_argument(f"--{f}", action="store_true")
    s = sub.add_parser("answers")
    s.add_argument("adapter")
    s.add_argument("test", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--batch", type=int, default=8)
    c = sub.add_parser("collect")
    c.add_argument("name")
    c.add_argument("label")
    c.add_argument("--reads", default="the text's activations (AO) and the question's text")
    c.add_argument("--o1", type=Path)
    c.add_argument("--o1m", type=Path)
    a = ap.parse_args()
    {"build": build, "answers": answers, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
