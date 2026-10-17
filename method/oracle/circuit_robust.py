"""Q5 of the circuit push (~/mpd-data/team/circuits_push_1016.txt): robustness of circuit_finder's question-scoped circuits.
Do a circuit's cuts still pass when the units outside the circuit are resampled from real in-distribution runs instead of
sitting at their clean or corrupted values, which units take over when a circuit unit is removed (backups), and which
pairs of units show their effects only together (interactions)?

Per record (a question and its circuit, circuit_finder's) and met cut (readout: the neurons at the answer position;
movers: the heads there; source: the neurons at the source positions), with circuit_finder's shares of the clean minus
corrupted gap:
  resampled            the question's own prompt with a partner key r (the question set's "partners": its template, same
                       length, keys other than the clean and corrupted ones; RESAMPLES of them whose answer differs from the
                       clean one by at least MIN_GAP bits), each a real run in place of the one corrupted prompt the cut was
                       found on: sufficiency, the cut's units at their clean values with every other unit computing on r
                       (restored share of the clean - r gap); necessity by resample ablation, the cut's units at their values
                       on r with every other unit computing on the clean prompt (lost share); the contrast is the pair
                       (clean, r)'s own (answer_contrast's words: r's answer among the alternatives). The cut passes
                       resampled when both means over r reach PASS;
  marginals            per circuit unit: its share alone and the cut's share without it (leave one out): the singles'
                       sum and the marginals' sum (joint - leave-one-out) against the joint share say whether the cut's
                       units back each other up (both sums small: any few of them suffice) or need each other (singles
                       small, marginals large);
  interactions         the cut's units and the record's next largest singles (PAIRS in all): every pair noised together
                       and denoised together; interaction = the pair's share - the two singles' shares, reported at
                       |interaction| >= INTERACT (lost > 0: redundant, each alone covered by the other; restored > 0:
                       synergy, each needs the other).

  circuit_robust.py RECORDS_DIR QUESTIONS.json OUT_DIR [--model Qwen/Qwen3-8B] [--dtype bfloat16] [--adapter A] [--ids ID ...]
                    [--ids-file FILE] [--pool QUESTIONS.json ...]   (resample prompts: the record's prompt with its key
                    replaced by the question's string "partners" or the pool's keys of its task)
writes OUT_DIR/<id>.json per record and prints a line per record.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit_finder as cf  # noqa: E402

PASS = cf.PASS
RESAMPLES = 4
BACKUP = 0.1        # compensation (direct - total lost share) at which a unit's backups are searched
EPS = 0.03          # a frozen block (or unit) counts as backup when it adds this much lost share
MAX_SINGLES = 64    # single units measured per searched circuit unit
FRONT = 16          # blocks a level of the backup search splits (the largest additions)
PAIRS = 16          # units per cut in the interaction test
INTERACT = 0.1


def cut_units(rec: dict, cut: str) -> tuple:
    """The cut's units (circuit_finder's tuples), its kind and positions."""
    T = len(rec["ids"])
    if cut == "movers":
        return [("head", r["layer"], r["head"], r["pos"]) for r in rec["routes"]], "head", [T - 1]
    us = [("neuron", n["layer"], n["pos"], n["neuron"]) for n in rec["nodes"] if n["stage"] == cut]
    return us, "neuron", ([T - 1] if cut == "readout" else list(rec["source_positions"]))


def single_units(rec: dict, cut: str) -> list:
    out = []
    for s in rec.get("singles", {}).get(cut, []):
        out.append(("head", s["layer"], s["head"], s["pos"]) if "head" in s else ("neuron", s["layer"], s["pos"], s["neuron"]))
    return out


def every(pt: cf.Patcher, kind: str, positions, src: str, layers=None) -> list:
    width = pt.H if kind == "head" else pt.D
    return [cf.unit(kind, i, p, torch.arange(width)) + (src,) for i in (range(pt.L) if layers is None else layers) for p in positions]


def capture(pt: cf.Patcher, ids, units=None, base="clean") -> dict:
    """A run's neuron and head values per layer ([T, dim]): ids as they are, or the base run with units patched from
    the other run."""
    cap = {}
    hooks = {k: v for k, v in pt._edits(pt._index([(0, base, units)], 0), 0).items() if k != "only"} if units else {}
    # (no "only": a capture reads every layer's units)
    with torch.no_grad():
        pt._last(ids, 1, cap=cap, **hooks)
    return {n: [t[0] for t in cap[n]] for n in ("res", "head", "neu")}


def resample_ids(tok, q: dict, rec: dict, n: int, pool: dict | None = None) -> list:
    """Up to n prompts: the question's "resamples" (whole prompts, e.g. an organism's other non-trigger prompts), then the
    record's own prompt with its key replaced by another key of its task (the question's string "partners", then pool
    {task: keys}, every question set given); each of the clean prompt's length, its tokens before the first differing
    position and a different token there (the record's template exactly: question sets differ in their templates)."""
    clean, first = rec["ids"], rec["first_diff"]
    user, key, task = rec.get("user", q.get("user")), rec.get("key", q.get("key")), rec.get("task", q.get("task"))
    cands = [(u, u) for u in q.get("resamples", [])]   # (whole prompts: an organism's other non-trigger prompts)
    cands += [(k, user.replace(key, k)) for k in [p for p in q.get("partners", []) if isinstance(p, str)]
              + sorted((pool or {}).get(task, ()))]
    out, seen = [], {key, q.get("corrupt_key"), rec.get("corrupt_key"), *rec.get("selection_keys", [])}   # (a multi-partner
                                                    # cut's own partners are no resample: circuit_masks --partners)
    for k, text in cands:
        if k in seen:
            continue
        seen.add(k)
        ids = cf.chat(tok, text)
        if len(ids) == len(clean) and ids[:first] == clean[:first] and ids[first] != clean[first]:  # (differs where the pair does)
            out.append((k, ids))
        if len(out) == n:
            break
    return out


def question_pool(files: list) -> dict:
    """{task: keys} over every question (and partner) of the question sets."""
    pool: dict = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        for q in (d["questions"] if isinstance(d, dict) else d):
            if "task" not in q:
                continue
            g = pool.setdefault(q["task"], set())
            g.add(q["key"])
            for p in q.get("partners", []):
                g.add(p["key"] if isinstance(p, dict) else p)
    return pool


def resampled(pt: cf.Patcher, rec: dict, cut: str, rs: list) -> dict:
    """The cut's shares on each resample pair (clean, r) (rs: [(key, ids)]); pt is left set up on the original pair."""
    units, _, _ = cut_units(rec, cut)
    clean, corrupt = rec["ids"], rec["corrupted"]
    std = list(pt.both([units])[0])
    lost, rest, keys = [], [], []
    for key, ids in rs:
        info = pt.setup(clean, ids)
        if not info["valid"] or info["same_answer"] or info["gap"] < cf.MIN_GAP:
            continue
        a, b = pt.both([units])[0]
        lost.append(a)
        rest.append(b)
        keys.append(key)
    pt.setup(clean, corrupt)
    R = max(len(lost), 1)
    out = {"units": len(units), "standard": std, "keys": keys, "lost": lost, "restored": rest,
           "mean": [sum(lost) / R, sum(rest) / R], "min": [min(lost, default=0.0), min(rest, default=0.0)]}
    out["passes"] = bool(lost) and out["mean"][0] >= PASS and out["mean"][1] >= PASS
    return out


def backups(pt: cf.Patcher, rec: dict, cut: str) -> list:
    units, _, positions = cut_units(rec, cut)
    if not units:
        return []
    total = pt.run("clean", [[u] for u in units])
    out = []
    for u, tot in zip(units, total):
        later = range(u[1] + 1, pt.L)
        if not len(later):
            continue
        frozen = every(pt, "neuron", positions, "self", later) + every(pt, "head", positions, "self", later)
        direct = pt.run("clean", [frozen], given=[u])[0]
        row = {"unit": cf_name(u), "total": tot, "direct": direct, "compensation": direct - tot, "backups": []}
        if direct - tot >= BACKUP:
            noised = capture(pt, pt.ids["clean"], [u])
            blocks = [b for kind in ("neuron", "head") for b in every(pt, kind, positions, "self", later)]
            gains = pt.run("clean", [[b] for b in blocks], given=[u])
            front = sorted([(b, g - tot) for b, g in zip(blocks, gains) if g - tot >= EPS], key=lambda x: -x[1])[:FRONT]
            while front and len(row["backups"]) < MAX_SINGLES:   # (level by level, every block of a level in one call)
                kids = []
                for b, gain in sorted(front, key=lambda x: -x[1]):
                    idx = b[3] if b[0] == "neuron" else b[2]
                    if len(idx) == 1:
                        if len(row["backups"]) < MAX_SINGLES:
                            row["backups"].append({"unit": cf_name(b[:4]), "adds": gain})
                        continue
                    name, lay, p = ("neu", b[1], b[2]) if b[0] == "neuron" else ("head", b[1], b[3])
                    d = (noised[name][lay][p] - pt.act["clean"][name][lay][p]).float().abs()
                    if b[0] == "head":
                        d = d.reshape(pt.H, pt.hd).sum(-1)
                    order = idx[torch.argsort(-d[idx.to(d.device)].cpu())]
                    for h in (order[:len(order) // 2], order[len(order) // 2:]):
                        kids.append((b[0], b[1], h, b[3], "self") if b[0] == "head" else (b[0], b[1], b[2], h, "self"))
                gains = pt.run("clean", [[k] for k in kids], given=[u]) if kids else []
                front = sorted([(k, g - tot) for k, g in zip(kids, gains) if g - tot >= EPS], key=lambda x: -x[1])[:FRONT]
        out.append(row)
    return out


def marginals(pt: cf.Patcher, rec: dict, cut: str) -> dict:
    units, _, _ = cut_units(rec, cut)
    if not units:
        return {}
    got = pt.both([list(units)] + [[u] for u in units] + [[v for v in units if v != u] for u in units])
    joint, one, loo = got[0], got[1:1 + len(units)], got[1 + len(units):]
    rows = [{"unit": cf_name(u), "alone": list(a), "marginal": [joint[0] - b[0], joint[1] - b[1]]} for u, a, b in zip(units, one, loo)]
    return {"joint": list(joint), "singles_sum": [sum(a[0] for a in one), sum(a[1] for a in one)],
            "marginals_sum": [sum(r["marginal"][0] for r in rows), sum(r["marginal"][1] for r in rows)], "units": rows}


def interactions(pt: cf.Patcher, rec: dict, cut: str) -> list:
    units, _, _ = cut_units(rec, cut)
    pool = list(units)[:PAIRS]
    for s in single_units(rec, cut):
        if len(pool) >= PAIRS:
            break
        if s not in pool:
            pool.append(s)
    if len(pool) < 2:
        return []
    one = pt.both([[u] for u in pool])
    pairs = [(a, b) for a in range(len(pool)) for b in range(a + 1, len(pool))]
    two = pt.both([[pool[a], pool[b]] for a, b in pairs])
    out = []
    for (a, b), (lost, rest) in zip(pairs, two):
        il, ir = lost - one[a][0] - one[b][0], rest - one[a][1] - one[b][1]
        if max(abs(il), abs(ir)) >= INTERACT:
            out.append({"pair": [cf_name(pool[a]), cf_name(pool[b])], "in_circuit": [pool[a] in units, pool[b] in units],
                        "singles": [list(one[a]), list(one[b])], "pair_shares": [lost, rest], "interaction": [il, ir]})
    out.sort(key=lambda r: -max(abs(r["interaction"][0]), abs(r["interaction"][1])))
    return out


def cf_name(u: tuple) -> str:
    def idx(k):
        k = k.reshape(-1).tolist() if torch.is_tensor(k) else ([k] if isinstance(k, int) else list(k))
        return str(k[0]) if len(k) == 1 else f"[{len(k)}]"
    return f"L{u[1]}.H{idx(u[2])}@{u[3]}" if u[0] == "head" else f"L{u[1]}.N{idx(u[3])}@{u[2]}"


def check(pt: cf.Patcher, tok, rec: dict, q: dict, pool: dict | None = None) -> dict:
    t0 = time.time()
    info = pt.setup(rec["ids"], rec["corrupted"])
    out = {"id": rec["id"], "gap": info["gap"], "record_gap": rec["gap"], "cuts": {}, "backups": {}, "interactions": {}}
    rs = resample_ids(tok, q, rec, RESAMPLES, pool)
    out["resample_keys"] = [p for p, _ in rs]
    for cut, v in rec.get("cuts", {}).items():
        if not v.get("met"):
            continue
        out["cuts"][cut] = resampled(pt, rec, cut, rs)
        out.setdefault("marginals", {})[cut] = marginals(pt, rec, cut)
        out["backups"][cut] = backups(pt, rec, cut)
        out["interactions"][cut] = interactions(pt, rec, cut)
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("records", type=Path)
    ap.add_argument("questions", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--adapter", default=None, help="an organism's LoRA (as circuit_finder: a PeftModel, adapter on)")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--ids-file", type=Path, help="the record ids to check, one a line")
    ap.add_argument("--pool", type=Path, nargs="*", default=[], help="question sets whose keys (by task) are resample keys")
    a = ap.parse_args()
    if a.ids_file:
        a.ids = (a.ids or []) + [x for x in a.ids_file.read_text().splitlines() if x.strip()]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
    pt = cf.Patcher(model, tok)
    qs = json.loads(a.questions.read_text())
    qs = {q["id"]: q for q in (qs["questions"] if isinstance(qs, dict) else qs)}
    pool = question_pool(a.pool + [a.questions]) if a.pool else None
    a.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.records.iterdir()):
        if f.suffix != ".json":
            continue
        rec = json.loads(f.read_text())
        if (a.ids and rec.get("id") not in a.ids) or not rec.get("valid", True) or not (rec.get("id") in qs or pool):
            continue
        o = a.out / f.name
        if o.exists():
            continue
        r = check(pt, tok, rec, qs.get(rec["id"], {}), pool)
        o.write_text(json.dumps(r))
        cuts = "; ".join(f"{c} {v['standard'][0]:.2f}/{v['standard'][1]:.2f} -> resampled {v['mean'][0]:.2f}/{v['mean'][1]:.2f}"
                         for c, v in r["cuts"].items())
        nb = sum(len(b["backups"]) for v in r["backups"].values() for b in v)
        ni = sum(len(v) for v in r["interactions"].values())
        print(f"{rec['id']}: {cuts}; {nb} backup units, {ni} interacting pairs; {r['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
