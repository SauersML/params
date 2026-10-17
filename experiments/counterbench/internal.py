"""CounterChat's internal-edit items (the lead's 10-16): per behaviour, a small set of neuron poles whose zeroing
removes the token where its replies decide the behaviour (T3's greedy edit, greedy_edit.fit, with no oracle), then
questions about what that next-token edit does to the behaviour's RATE over full sampled replies: on its own chat, on a
family chat, on another behaviour's chat, against random poles of the same layers and signs.

Decision points: in the base arm's sampled replies (items.py measure, judged), each reply that shows the behaviour is
read token by token until it leaves every non-showing reply's prefix; that token, after the chat and the reply so far,
is the point where the behaviour was decided (at most FIT of them, as greedy_edit). A behaviour with fewer than
MIN_POINTS such replies, or fewer than MIN_POINTS replies without it, gets no edit.

  internal.py fit JUDGED.jsonl... OUT.jsonl --thr T2.pt --general GENERAL.jsonl [--only ID ...]   (GPU, HF)
  internal.py jobs EDITS.jsonl OUT.jsonl [--n 64]       the internal arms for engine.py (Mac)
  internal.py groups CANDIDATES.json OUT.jsonl          labelled neuron groups (PheWAS trait associations) zeroed (Mac)
Arms compare with "ref", the same chat unedited in the same hooked engine ("base_eager").
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "method" / "oracle"), str(HERE.parent / "oracle" / "scorecard")]
from items import ROOT, arm, load  # noqa: E402

MIN_POINTS = 4
MAX_DEPTH = 96  # tokens into a reply searched for its decision


def chat_ids(tok, messages: list[dict]) -> list[int]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def decision_points(tok, messages: list[dict], replies: list[str], verdicts: list, fit: int) -> list[tuple[list[int], int]]:
    """(chat + reply up to and including its decision token, that token) for replies showing the behaviour."""
    pos = [r for r, v in zip(replies, verdicts) if v is True and r]
    neg = [r for r, v in zip(replies, verdicts) if v is False and r]
    if len(pos) < MIN_POINTS or len(neg) < MIN_POINTS:
        return []
    prompt = chat_ids(tok, messages)
    prefixes = set()
    for r in neg:
        ids = tok(r, add_special_tokens=False)["input_ids"][:MAX_DEPTH]
        prefixes.update(tuple(ids[:t + 1]) for t in range(len(ids)))
    out = []
    for r in pos:
        ids = tok(r, add_special_tokens=False)["input_ids"][:MAX_DEPTH]
        t = next((t for t in range(len(ids)) if tuple(ids[:t + 1]) not in prefixes), None)
        if t is not None:
            out.append((prompt + ids[:t + 1], ids[t]))
    return out[:fit]


def fit(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import claim_check as cc
    import greedy_edit as ge
    import measure as ms
    import neuron_builder as nb

    rows = [r for p in a.judged for r in load(Path(p)) if r["name"] == "base" and r["model"] == ms.MODEL]
    if a.only:
        rows = [r for r in rows if r["behaviour"] in set(a.only)]
    out = Path(a.out)
    done = {x["behaviour"] for x in load(out)} if out.exists() else set()
    general = [json.loads(x)["user"] for x in Path(a.general).read_text().splitlines()[:ge.FIT_KL]]
    tok = AutoTokenizer.from_pretrained(ms.MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(ms.MODEL, dtype=torch.bfloat16).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    nr = nb.Neurons(model, tok)
    nr.load(a.thr)
    nr.thr = None  # (no weak zeroing: the model itself)
    for r in rows:
        if r["behaviour"] in done:
            continue
        pts = decision_points(tok, r["messages"], r["responses"], r["verdicts"], ge.FIT)
        rec = {"behaviour": r["behaviour"], "points": len(pts)}
        if len(pts) >= MIN_POINTS:
            rec["decision_tokens"] = sorted({tok.decode([t]) for _, t in pts})
            f = ge.fit(nr, tok, pts, general, lambda *x: ms.log(r["behaviour"], *x))
            us = [u for u in ms.units(nr, f["units"], 1)] if f["units"] else []
            ctrl = cc.control_sets(nr, [tuple(u[:4]) + (None,) for u in us], 1, "counterchat:" + r["behaviour"])[0] if us else []
            rec.update(f, control=[ge.cc_name(u) for u in ctrl])
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


def parse(names: list[str]) -> list[list]:
    """claim_check unit tuples from pole names ("L18.N2201+")."""
    out = []
    for s in names:
        m = re.fullmatch(r"L(\d+)\.N(\d+)([+-])", s)
        if not m:
            raise ValueError(s)
        out.append(["N", int(m.group(1)), int(m.group(2)), 1 if m.group(3) == "+" else -1, None])
    return out


def jobs(a) -> None:
    """Per behaviour with an edit E (and its random control C): its chat under E and under C, its first family chat
    under E, and one other behaviour's chat (another category, fixed by crc32) under E; each chat also unedited in the
    hooked engine (the arms' reference)."""
    bs = {b["id"]: b for b in load(ROOT / "behaviours.jsonl") if b.get("kept", True)}
    eds = [e for e in load(Path(a.edits)) if e.get("units") and e["behaviour"] in bs]
    out, refs = [], set()

    def ref_arm(b: dict, name: str, messages: list[dict], facts=None) -> str:
        key = f"{b['id']}|{name}"
        if key not in refs:
            refs.add(key)
            out.append(arm(b, name, "reference", messages, a.n, facts=facts))
        return key

    for e in eds:
        b = bs[e["behaviour"]]
        E = {"units": parse(e["units"]), "op": "zero", "name": f"greedy:{b['id']}", "size": len(e["units"])}
        C = {"units": parse(e["control"]), "op": "zero", "name": f"random:{b['id']}", "size": len(e["control"])}
        r0 = ref_arm(b, "base_eager", b["base"])
        n_units = f"{len(e['units'])} neuron poles"
        out.append({**arm(b, "edit_greedy", "internal", b["base"], a.n, edit=E,
                          change=f"zero the {n_units} that remove this behaviour's decision token"), "ref": r0})
        out.append({**arm(b, "edit_random", "internal_control", b["base"], a.n, edit=C,
                          change=f"zero {n_units} chosen at random in the same layers with the same signs"), "ref": r0})
        fam = next((f for f in b["family"] if f.get("confirm")), b["family"][0] if b["family"] else None)
        if fam:
            r1 = ref_arm(b, f"t{fam['test']}_eager", fam["messages"], fam.get("facts"))
            out.append({**arm(b, f"t{fam['test']}_greedy", "internal_family", fam["messages"], a.n, edit=E,
                              change=f"zero the {n_units} found on the original chat, on another chat of the same family",
                              facts=fam.get("facts")), "ref": r1})
        others = sorted((o for o in bs.values() if o["category"] != b["category"] and o["split"] == b["split"]),
                        key=lambda o: zlib.crc32((b["id"] + o["id"]).encode()))  # (same split: a test edit stays in test)
        if others:
            o = others[0]
            r2 = ref_arm(o, "base_eager", o["base"])
            out.append({**arm(o, f"edit_from:{b['id']}", "internal_cross", o["base"], a.n, edit=E,
                              change=f"zero the {n_units} found for another behaviour ({b['behaviour']})"), "ref": r2})
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(len(out), "arms for", len(eds), "edits")


GROUPS = {  # labelled neuron groups (the PheWAS stage-1 association candidates of a phenome trait) and the categories they bear on
    "style/markdown-headings-bold": ("Markdown headings and bold in replies", ("format", "language_drift")),
    "abstain/simpleqa-hard": ("abstaining on hard factual questions", ("fabrication",)),
    "abstain/triviaqa": ("abstaining on trivia questions", ("fabrication",)),
    "sycophancy/suggested-wrong-answer": ("agreeing with a wrong answer the user suggests", ("sycophancy",)),
}
TOP = 32


def groups(a) -> None:
    """Arms zeroing labelled neuron groups (whole neurons, every position): each group on the behaviours of its categories
    and on one behaviour of another category, and a random group of the same layers on the same chats."""
    cands = json.loads(Path(a.candidates).read_text())
    bs = [b for b in load(ROOT / "behaviours.jsonl") if b.get("kept", True)]
    out, refs = [], set()
    for trait, (label, cats) in GROUPS.items():
        top = [tuple(u) for u in (cands.get(trait) or [])[:TOP]]
        if not top:
            continue
        E = {"units": [["N", l_, n, 0, None] for l_, n in top], "op": "zero", "name": f"phewas:{trait}", "size": len(top)}
        from circuits import control_of
        C = {"units": control_of(E["units"], "counterchat-group:" + trait), "op": "zero", "name": f"random:{trait}", "size": len(top)}
        mine = [b for b in bs if b["category"] in cats]
        other = sorted((b for b in bs if b["category"] not in cats), key=lambda b: zlib.crc32((trait + b["id"]).encode()))[:max(2, len(mine) // 4)]
        for b in mine + other:
            key = f"{b['id']}|base_eager"
            if key not in refs:
                refs.add(key)
                out.append(arm(b, "base_eager", "reference", b["base"], a.n))
            what = f"zero the {len(top)} neurons most associated with the trait '{label}' (a neuron-trait association screen)"
            out.append({**arm(b, f"group:{trait}", "group", b["base"], a.n, edit=E, change=what), "ref": key})
            out.append({**arm(b, f"group_random:{trait}", "group_control", b["base"], a.n, edit=C,
                              change=f"zero {len(top)} neurons chosen at random in the same layers"), "ref": key})
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(len(out), "arms")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("fit")
    q.add_argument("judged", nargs="+")
    q.add_argument("out")
    q.add_argument("--thr", required=True)
    q.add_argument("--general", required=True)
    q.add_argument("--only", nargs="*")
    q = sub.add_parser("jobs")
    q.add_argument("edits")
    q.add_argument("out")
    q.add_argument("--n", type=int, default=64)
    q = sub.add_parser("groups")
    q.add_argument("candidates", help="the PheWAS stage-1 candidates (phewas/candidates_dev.json)")
    q.add_argument("out")
    q.add_argument("--n", type=int, default=64)
    a = p.parse_args()
    {"fit": fit, "jobs": jobs, "groups": groups}[a.cmd](a)


if __name__ == "__main__":
    main()
