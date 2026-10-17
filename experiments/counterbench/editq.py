"""CounterChat's edit questions (the lead's 10-16): per behaviour, "which neuron poles or heads, zeroed through the whole
generation, remove this behaviour on chats you have not seen, with side effects of at most KL_MAX bits per token?",
graded by measurement.

An answer is {"item": "<behaviour>|edit", "units": ["L18.N2201+", "L5.H3", ...]} (claim_check's unit names; at most
MAX_UNITS). It passes when, on each held-out family chat of the behaviour, the edited rate is at most (1 - DROP) x the
unedited rate (both measured in the hooked engine, N replies each, the same grading as the behaviour's items), and the
mean KL of the edited model from the unedited one on GENERAL_N general chats (their unedited greedy answers
teacher-forced, measure.kl_bits) is at most KL_MAX. Reported per answer: the held-out rates, the relative drop, the KL,
pass/fail.

  editq.py items OUT.jsonl                    one edit question per kept behaviour with at least one held-out chat
  editq.py answers-greedy EDITS.jsonl OUT     internal.py's greedy poles as answers (the T3 baseline), and random poles
  editq.py jobs ITEMS ANSWERS... OUT.jsonl    the arms: each answer on each held-out chat, and each chat unedited
  editq.py kl ANSWERS... OUT.jsonl --thr T2.pt --general G.jsonl          (GPU, HF) each answer's side effects
  editq.py grade ITEMS ARMS(judged)... --kl KL.jsonl [--answers ...]      pass/fail per answer
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "method" / "oracle"), str(HERE.parent / "oracle" / "scorecard")]
from internal import parse  # noqa: E402
from items import ROOT, arm, load, merged, rate  # noqa: E402

DROP = 0.5
KL_MAX = 0.05
MAX_UNITS = 32
GENERAL_N = 100
HELD_OUT = 3


def items(a) -> None:
    out = []
    for b in load(ROOT / "behaviours.jsonl"):
        if not b.get("kept", True):
            continue
        held = [f for f in b["family"] if f.get("confirm") and f["confirm"][0] > 0][:HELD_OUT]
        if held:
            out.append({"id": f"{b['id']}|edit", "behaviour": b["id"], "kind": "edit", "split": b["split"], "base": b["base"],
                        "held_out": [{"test": f["test"], "messages": f["messages"], "facts": f.get("facts")} for f in held],
                        "goal": {"drop": DROP, "kl_max": KL_MAX, "max_units": MAX_UNITS}})
    Path(a.out).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in out))
    print(len(out), "edit questions")


def answers_greedy(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g, r = [], []
    for e in load(Path(a.edits)):
        if e.get("units"):
            g.append({"item": f"{e['behaviour']}|edit", "units": e["units"]})
            r.append({"item": f"{e['behaviour']}|edit", "units": e["control"]})
    (out / "t3-greedy.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
    (out / "random.jsonl").write_text("".join(json.dumps(x) + "\n" for x in r))
    print(len(g), "greedy answers and their random controls in", out)


def unit_tuples(names: list[str]) -> list[list]:
    neurons = [n for n in names if ".N" in n]
    heads = [n for n in names if ".H" in n]
    out = parse(neurons)
    for h in heads:
        layer, head = h[1:].split(".H")
        out.append(["H", int(layer), int(head), 0, None])
    return out


def jobs(a) -> None:
    its = {i["id"]: i for i in load(Path(a.items))}
    bs = {b["id"]: b for b in load(ROOT / "behaviours.jsonl")}
    out, refs = [], set()
    for path in a.answers:
        who = Path(path).stem
        for ans in load(Path(path)):
            it = its.get(ans["item"])
            if not it or not ans.get("units"):
                continue
            b = bs[it["behaviour"]]
            E = {"units": unit_tuples(ans["units"][:MAX_UNITS]), "op": "zero", "name": f"{who}:{it['id']}", "size": len(ans["units"])}
            for h in it["held_out"]:
                ref = f"{b['id']}|t{h['test']}_eager"
                if ref not in refs:
                    refs.add(ref)
                    out.append(arm(b, f"t{h['test']}_eager", "reference", h["messages"], a.n, facts=h.get("facts")))
                out.append({**arm(b, f"t{h['test']}_edit:{who}", "edit_answer", h["messages"], a.n, edit=E, facts=h.get("facts"),
                                  change=f"{who}'s edit"), "ref": ref, "answer_of": who, "question": it["id"]})
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(len(out), "arms")


def kl(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import claim_check as cc
    import measure as ms
    import neuron_builder as nb

    general = [json.loads(x)["user"] for x in Path(a.general).read_text().splitlines()[:GENERAL_N]]
    tok = AutoTokenizer.from_pretrained(ms.MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(ms.MODEL, dtype=torch.bfloat16).to("cuda").eval()
    nr = nb.Neurons(model, tok)
    nr.load(a.thr)
    nr.thr = None
    answers = ms.generate(nr, tok, general, None, 64, bs=50)
    out = Path(a.out)
    done = {(x["who"], x["item"]) for x in load(out)} if out.exists() else set()
    for path in a.answers:
        who = Path(path).stem
        for ans in load(Path(path)):
            if (who, ans["item"]) in done or not ans.get("units"):
                continue
            us = cc.units_of(nr, ans["units"][:MAX_UNITS], 1)
            bits = ms.kl_bits(nr, tok, general, answers, ms.Edit(nr, us, "zero"), bs=8)
            with out.open("a") as f:
                f.write(json.dumps({"who": who, "item": ans["item"], "kl_bits": bits, "units": len(us)}) + "\n")
            ms.log(who, ans["item"], f"KL {bits:.4f}")


def grade(a) -> None:
    its = {i["id"]: i for i in load(Path(a.items))}
    arms = {r["arm"]: r for r in merged([r for p in a.arms for r in load(Path(p))])}
    kls = {(x["who"], x["item"]): x["kl_bits"] for x in load(Path(a.kl))} if a.kl else {}
    res = collections.defaultdict(dict)
    for r in arms.values():
        if r["kind"] != "edit_answer":
            continue
        ref = arms.get(r["ref"])
        k1, n1 = rate(r)
        k0, n0 = rate(ref) if ref else (0, 0)
        res[(r["answer_of"], r["question"])][r["name"]] = (k1 / max(1, n1), k0 / max(1, n0))
    rows = []
    for (who, q), per in sorted(res.items()):
        ok_drop = all(e <= (1 - DROP) * u for e, u in per.values())
        kl_ = kls.get((who, q))
        rows.append({"who": who, "item": q, "split": its[q]["split"] if q in its else None, "held_out": per, "kl_bits": kl_,
                     "pass": bool(ok_drop and kl_ is not None and kl_ <= KL_MAX)})
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[(r["who"], r["split"])][0] += r["pass"]
        by[(r["who"], r["split"])][1] += 1
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for (who, split), (p, n) in sorted(by.items()):
        print(f"{who:<16} {split}: {p}/{n} pass")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("items")
    q.add_argument("out")
    q = sub.add_parser("answers-greedy")
    q.add_argument("edits")
    q.add_argument("out")
    q = sub.add_parser("jobs")
    q.add_argument("items")
    q.add_argument("answers", nargs="+")
    q.add_argument("out")
    q.add_argument("--n", type=int, default=64)
    q = sub.add_parser("kl")
    q.add_argument("answers", nargs="+")
    q.add_argument("out")
    q.add_argument("--thr", required=True)
    q.add_argument("--general", required=True)
    q = sub.add_parser("grade")
    q.add_argument("items")
    q.add_argument("arms", nargs="+")
    q.add_argument("--kl")
    q.add_argument("--out", default=str(ROOT / "edit_grades.jsonl"))
    a = p.parse_args()
    {"items": items, "answers-greedy": answers_greedy, "jobs": jobs, "kl": kl, "grade": grade}[a.cmd](a)


if __name__ == "__main__":
    main()
