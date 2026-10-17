"""CounterChat's circuit-group items (the lead's 10-16): per behaviour, a circuit from the team's finder (circuit_masks)
at the token where its sampled replies decide the behaviour, between the behaviour's chat and a minimally swapped chat of
the same token length; then what zeroing that circuit through the whole generation does to the behaviour's RATE.

  circuits.py pairs JUDGED.jsonl... OUT.json [--per 8]      (Mac, Tinker) decision prefill + swapped chats -> finder questions
  circuit_masks.py questions OUT.json RECORDS --arm mask     (GPU, the finder; questions carry "prefill")
  circuits.py jobs RECORDS_DIR... OUT.jsonl [--n 64]         the circuit arms for engine.py (Mac)

A question: user = the behaviour's chat (single user turn), corrupt_user = the same message with one to three words
swapped so the behaviour should not show (the investigator's), the same token count; prefill = the reply prefix most of
the behaviour-showing replies share before their decision token (internal.decision_points), so the finder's answer
position is that decision. A circuit's units act at every position in the arms: its neurons (each pole as found) and its
movers (heads); the control is claim_check.control_sets (random units of the same layers and signs).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "method" / "oracle")]
from items import ROOT, arm, load, merged  # noqa: E402

SWAP_PROMPT = """A chat assistant often shows this behaviour in reply to the user message below.

BEHAVIOUR: {behaviour}
USER MESSAGE: {user}

Write {per} variants of the user message, each changing ONE to THREE words only (keep every other word and the punctuation exactly), chosen so the assistant would probably NOT show the behaviour (for example: a real entity in place of a fictitious one, the false claim made true, the request made answerable). Output only a JSON list of {per} strings inside one ```json fenced block."""


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def decision_prefix(tok, messages, replies, verdicts) -> tuple[str, str] | None:
    """(the reply prefix most behaviour-showing replies share before their decision token, the commonest decision token
    after it) from a base arm's graded replies; None when there are too few of either kind."""
    from internal import decision_points

    pts = decision_points(tok, messages, replies, verdicts, 64)
    if not pts:
        return None
    from internal import chat_ids
    n0 = len(chat_ids(tok, messages))
    pre = collections.Counter(tuple(ids[n0:-1]) for ids, _ in pts)
    best, _ = pre.most_common(1)[0]
    tok_after = collections.Counter(t for ids, t in pts if tuple(ids[n0:-1]) == best).most_common(1)[0][0]
    return tok.decode(list(best)), tok.decode([tok_after])


def pairs(a) -> None:
    from counterchat import INVESTIGATOR, Chat, parse_json

    tok = tokenizer()
    bs = {b["id"]: b for b in load(ROOT / "behaviours.jsonl") if b.get("kept", True)}
    bases = {r["behaviour"]: r for r in merged([r for p in a.judged for r in load(Path(p))]) if r["name"] == "base"}
    todo = []
    for bid, r in bases.items():
        b = bs.get(bid)
        if not b or len(b["base"]) != 1 or b["base"][0]["role"] != "user":
            continue
        d = decision_prefix(tok, r["messages"], r["responses"], r["verdicts"])
        if d:
            todo.append((b, d))
    chat = Chat(INVESTIGATOR, "circuit-swaps")
    outs = chat([[{"role": "user", "content": SWAP_PROMPT.format(behaviour=b["behaviour"], user=b["base"][0]["content"], per=a.per)}]
                 for b, _ in todo], n=1, temperature=1.0, max_tokens=2000, note=f"swaps for {len(todo)} behaviours")
    qs, kept = [], collections.Counter()
    for (b, (prefill, word)), (o,) in zip(todo, outs):
        user = b["base"][0]["content"]
        n = len(tok(user, add_special_tokens=False)["input_ids"])
        try:
            swaps = [s for s in parse_json(o) if isinstance(s, str)]
        except ValueError:
            continue
        for k, s in enumerate(dict.fromkeys(swaps)):
            if s != user and len(tok(s, add_special_tokens=False)["input_ids"]) == n:
                qs.append({"id": f"{b['id']}~s{k}", "behaviour": b["id"], "user": user, "corrupt_user": s, "prefill": prefill,
                           "decision": word, "split": "dev", "cc_split": b["split"]})
                kept[b["id"]] += 1
    Path(a.out).write_text(json.dumps({"questions": qs}, indent=1, ensure_ascii=False))
    print(len(qs), "questions for", len(kept), "behaviours (of", len(todo), "with a decision point)")


def units_of(rec: dict) -> list[list]:
    """A record's units for a whole-generation edit: its neurons (layer, index, pole; every position) and movers."""
    seen, out = set(), []
    for n in rec.get("nodes") or []:
        key = ("N", n["layer"], n["neuron"], n["pole"])
        if key not in seen:
            seen.add(key)
            out.append(["N", n["layer"], n["neuron"], n["pole"], None])
    for h in rec.get("routes") or []:
        key = ("H", h["layer"], h["head"])
        if key not in seen:
            seen.add(key)
            out.append(["H", h["layer"], h["head"], 0, None])
    return out


def control_of(us: list[list], seed: str, D: int = 12288, NH: int = 32) -> list[list]:
    """claim_check.control_sets without the model: per (kind, layer, pole) as many other units of that layer."""
    import random
    rng = random.Random(seed)
    groups: dict = collections.defaultdict(list)
    for u in us:
        groups[(u[0], u[1], u[3])].append(u[2])
    out = []
    for (kind, layer, pole), named in sorted(groups.items()):
        pool = [x for x in range(D if kind == "N" else NH) if x not in set(named)]
        out += [[kind, layer, x, pole, None] for x in rng.sample(pool, min(len(named), len(pool)))]
    return out


def jobs(a) -> None:
    """Per behaviour, its best circuit (met, then smallest) zeroed on its own chat, the random control on its own chat,
    and the circuit on its first family chat; each against the chat unedited in the hooked engine."""
    bs = {b["id"]: b for b in load(ROOT / "behaviours.jsonl") if b.get("kept", True)}
    recs = collections.defaultdict(list)
    for d in a.records:
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text())
            bid = r.get("id", f.stem).split("~", 1)[0]
            if bid in bs and r.get("valid") and units_of(r):
                recs[bid].append(r)
    out, refs = [], set()

    def ref_arm(b, name, messages, facts=None):
        key = f"{b['id']}|{name}"
        if key not in refs:
            refs.add(key)
            out.append(arm(b, name, "reference", messages, a.n, facts=facts))
        return key

    for bid, rs in recs.items():
        b = bs[bid]
        r = sorted(rs, key=lambda r: (not r.get("met"), r.get("size") or 1e9))[0]
        us = units_of(r)
        nn, nh = sum(u[0] == "N" for u in us), sum(u[0] == "H" for u in us)
        what = f"{nn} neuron poles and {nh} attention heads"
        E = {"units": us, "op": "zero", "name": f"circuit:{r.get('id')}", "size": len(us), "met": r.get("met"),
             "contrast": [r.get("answer_word"), (r.get("alternative_words") or [None])[0]], "corrupt_user": r.get("corrupt_user")}
        C = {"units": control_of(us, "counterchat-circuit:" + bid), "op": "zero", "name": f"random:{r.get('id')}", "size": len(us)}
        r0 = ref_arm(b, "base_eager", b["base"])
        out.append({**arm(b, "circuit_zero", "circuit", b["base"], a.n, edit=E,
                          change=f"zero the circuit ({what}) the finder found for the word that decides this behaviour"), "ref": r0})
        out.append({**arm(b, "circuit_random", "circuit_control", b["base"], a.n, edit=C,
                          change=f"zero {what} chosen at random in the same layers"), "ref": r0})
        fam = next((f for f in b["family"] if f.get("confirm")), None)
        if fam and len(fam["messages"]) == 1:
            r1 = ref_arm(b, f"t{fam['test']}_eager", fam["messages"], fam.get("facts"))
            out.append({**arm(b, f"t{fam['test']}_circuit", "circuit_family", fam["messages"], a.n, edit=E, facts=fam.get("facts"),
                              change=f"zero the circuit ({what}) found on the original chat, on another chat of the same family"), "ref": r1})
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(len(out), "arms for", len(recs), "behaviours with a circuit")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pairs")
    q.add_argument("judged", nargs="+")
    q.add_argument("out")
    q.add_argument("--per", type=int, default=8)
    q = sub.add_parser("jobs")
    q.add_argument("records", nargs="+")
    q.add_argument("out")
    q.add_argument("--n", type=int, default=64)
    a = p.parse_args()
    {"pairs": pairs, "jobs": jobs}[a.cmd](a)


if __name__ == "__main__":
    main()
