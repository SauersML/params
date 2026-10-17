"""Mechanism-grounded explanations of question-scoped circuits (#2951 oracledata; the lead's 10-16 rule: agents may write
strong examples in their own turns, then the strongest Tinker writer, Qwen3.5-397B, expands them; never an 8B writer,
never Claude inside a pipeline). An item answers "How does the model produce its answer here?" for one text: plain
English naming the circuit's stages (where the answer's information enters, the heads that move it to the answer
position, the neurons that write it there) with measured numbers, and claims in claim_check's form whose runs on
Qwen3-8B must all pass (word-level answer: the record's answer_tokens against its alternatives). Each item is tagged with
its writer.

  explain.py facts RECORDS OUT.jsonl --thr THR.pt [--ids IDS.txt]          (GPU) a fact sheet per text: each stage zeroed
      and set to its values on the corrupted prompt, its three largest units zeroed alone, the largest readout neuron
      turned up; every run with claim_check's random same-layer controls
  explain.py expand FACTS.jsonl SEEDS.jsonl OUT.jsonl [--n N] [--seed S]   (Tinker) the 397B writes items from fact sheets
  explain.py verify ITEMS.jsonl FACTS.jsonl OUT.jsonl --thr THR.pt        (GPU) every claim run; an item is kept when all pass
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "method" / "oracle"))

STAGES = ("source", "movers", "readout")
TOP = 3
STEER = 3.0
ALWAYS = {"contrast_change": {"op": ">", "value": -1e9}}  # (a run whose numbers are wanted, not a claim)
WRITER = "Qwen/Qwen3.5-397B-A17B"


def unit_name(d: dict) -> str:
    if "head" in d:
        return f"L{d['layer']}.H{d['head']}@{d['pos']}"
    return f"L{d['layer']}.N{d['neuron']}{'+' if d['pole'] > 0 else '-'}@{d['pos']}"


def stages_of(rec: dict) -> dict:
    """A record's units per stage in measured order (the larger of lost and restored)."""
    order = lambda us: sorted(us, key=lambda d: -max(abs(d.get("lost") or 0), abs(d.get("restored") or 0)))  # noqa: E731
    out = {"source": order([d for d in rec.get("nodes", []) if d.get("stage") == "source" and d["pos"] > 0]),
           "movers": order([r for r in rec.get("routes", []) if r["pos"] > 0]),
           "readout": order([d for d in rec.get("nodes", []) if d.get("stage") == "readout" and d["pos"] > 0])}
    return {k: v for k, v in out.items() if v}


def runs_of(rec: dict) -> list[tuple[str, list[str], object]]:
    """The fact sheet's runs: (label, units, op)."""
    st = stages_of(rec)
    out = []
    for s, us in st.items():
        names = [unit_name(d) for d in us]
        out.append((f"zero {s}", names, "zero"))
        out.append((f"patch {s} from the corrupted prompt", names, {"patch_from": {"ids": rec["corrupted"]}}))
        for d in us[:TOP]:
            out.append((f"zero {unit_name(d)}", [unit_name(d)], "zero"))
    if "readout" in st:
        d = st["readout"][0]
        out.append((f"turn up {unit_name(d)}", [unit_name(d)], {"steer": STEER}))
    return out


def sheet(rec: dict, tok, results: list[dict], runs) -> dict:
    toks = [tok.decode([t]) for t in rec["ids"]]
    st = stages_of(rec)
    desc = {}
    for s, us in st.items():
        desc[s] = [{"unit": unit_name(d), "token": toks[d["pos"]], "lost": round(d.get("lost") or 0, 3),
                    "restored": round(d.get("restored") or 0, 3),
                    **({"looks_at": [[q, round(w, 2), toks[q]] for q, w, *_ in (d.get("attends") or [])[:3]]} if "head" in d else
                       {"value": d.get("clean"), "corrupt_value": d.get("corrupt")})} for d in us]
    meas = []
    for (label, units, op), r in zip(runs, results):
        if "error" in r:
            continue
        meas.append({"run": label, "units": units, "op": op, "change": round(r["change"], 2), "top_before": r["top_before"],
                     "top_after": r["top_after"], "control_changes": [round(x, 2) for x in r["control_change"]]})
    return {"id": rec["id"], "family": rec.get("task"), "user": rec.get("user"), "corrupt_user": rec.get("corrupt_user"),
            "answer": rec.get("answer_text"), "answer_word": rec.get("answer_word"),
            "corrupt_answer": rec.get("corrupt_answer_text"), "tokens": toks, "ids": rec["ids"], "corrupted": rec["corrupted"],
            "answer_tokens": rec["answer_tokens"], "alternatives": rec["alternatives"],
            "contrast": rec.get("contrast"), "met": rec.get("met"), "stages": desc, "measured": meas,
            "finder_rev": rec.get("finder_rev")}


def model_of(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import neuron_builder as nb
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device or "cuda").eval()
    nr = nb.Neurons(model, tok)
    nr.load(a.thr)
    nr.thr = None  # the model itself (circuit_finder's runs); T2's maxima and massive cuts kept
    return tok, nr


def facts(a) -> None:
    import claim_check as cc
    tok, nr = model_of(a)
    ids = set(a.ids.read_text().split()) if a.ids else None
    paths = sorted(a.records.glob("*.json"))
    t0, n = time.time(), 0
    with a.out.open("w") as fh:
        for p in paths:
            rec = json.loads(p.read_text())
            if (ids is not None and rec.get("id") not in ids) or rec.get("skipped") or not rec.get("answer_tokens"):
                continue
            runs = runs_of(rec)
            claims = [{"prompt": {"ids": rec["ids"]}, "position": -1, "units": u, "op": op, "predict": ALWAYS,
                       "answer": rec["answer_tokens"], "alternatives": rec["alternatives"]} for _, u, op in runs]
            got = [x for k in range(0, len(claims), 40) for x in cc.check_many(nr, claims[k:k + 40])]
            fh.write(json.dumps(sheet(rec, tok, got, runs)) + "\n")
            fh.flush()
            n += 1
            if n % 20 == 0:
                print(json.dumps({"texts": n, "seconds": round(time.time() - t0, 1)}), flush=True)
    print(json.dumps({"texts": n, "seconds": round(time.time() - t0, 1), "out": str(a.out)}), flush=True)


RULES = """You explain how a language model (Qwen3-8B) produces its answer on one prompt, from a fact sheet of measured \
interventions on its own neurons and attention heads. The sheet lists the circuit found by patching between the prompt \
and a corrupted prompt (one key word changed): its stages (source: neurons at the changed word and after it; movers: \
attention heads at the answer position, with the tokens they look at; readout: neurons at the answer position), each \
unit's measured share of the answer's lead lost when it alone is set to its corrupted value ("lost") and restored when it \
alone is set to its clean value in the corrupted run ("restored"), and runs: a stage or unit zeroed, set to its values on \
the corrupted prompt, or turned up, each with the change of the answer's lead in bits (log2 of the answer word's \
probability over its alternatives'), the top token after, and the same run on random units of the same layers \
("control_changes").

Write ANSWER: three to six plain sentences on how the answer gets there: where its information starts, which heads move \
it and from which token, which neurons write it at the end, and what removing or swapping each stage does, with the \
measured numbers. Use only numbers in the sheet; say when random units do as much (a run whose control changes are as \
large is not specific to the circuit). No jargon beyond the unit names.
Then CLAIMS: a JSON list of two to four claims, each {"units": [unit names from the sheet], "op": "zero" or \
{"steer": 3.0} or {"patch_from": "corrupted"}, "predict": {...}}, each restating a run in the sheet with a margin: \
{"contrast_change": {"op": "<=", "value": V}} with V at least 0.5 bits above a measured drop, or {"op": ">=", ...} \
symmetrically, or {"top": "<token>"} when the run's top token changed. Every claim must hold on the model."""


def render(f: dict) -> str:
    keep = {k: f[k] for k in ("user", "corrupt_user", "answer", "answer_word", "corrupt_answer", "stages", "measured")}
    for m in keep["measured"]:
        if isinstance(m["op"], dict) and "patch_from" in m["op"]:
            m["op"] = {"patch_from": "corrupted"}
    return json.dumps(keep, ensure_ascii=False)


SPEND = Path.home() / "mpd-data" / "scratch" / "oracledata" / "tinker_spend.tsv"


def sample(convs: list[list[dict]], max_tokens: int, note: str) -> list[str]:
    """The 397B on Tinker, thinking off, greedy; every call's spend logged (prefill, sampled tokens, dollars)."""
    import tinker
    from concurrent.futures import ThreadPoolExecutor
    from llm import PRICES, answer_of, escape, renderer_for, service_client
    renderer, tokenizer = renderer_for(WRITER)
    sampler = service_client().create_sampling_client(base_model=WRITER)
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=renderer.get_stop_sequences())
    prompts = [renderer.build_generation_prompt([{"role": m["role"], "content": escape(m["content"])} for m in c]) for c in convs]
    futs = [sampler.sample(prompt=q, num_samples=1, sampling_params=params) for q in prompts]
    with ThreadPoolExecutor(min(64, len(futs))) as pool:
        res = list(pool.map(lambda f: f.result(), futs))
    pre = sum(q.length for q in prompts)
    smp = sum(len(r.sequences[0].tokens) for r in res)
    usd = (pre * PRICES[WRITER][0] + smp * PRICES[WRITER][1]) / 1e6
    new = not SPEND.exists()
    with SPEND.open("a") as fh:
        if new:
            fh.write("time\tmodel\tprefill_tokens\tsampled_tokens\tusd\tnote\n")
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{WRITER}\t{pre}\t{smp}\t{usd:.4f}\t{note}\n")
    return [answer_of(renderer, tokenizer, r.sequences[0].tokens)[0] for r in res]


def expand(a) -> None:
    import os
    os.environ.setdefault("TINKER_PROJECT_ID", "3f598f31-2ffa-4eb4-b302-909eb234fb7d")
    sheets = [json.loads(x) for x in a.facts.read_text().splitlines()]
    seeds = [json.loads(x) for x in a.seeds.read_text().splitlines()]
    seed_ids = {s["id"] for s in seeds}
    by = {f["id"]: f for f in sheets}
    shots = []
    for s in seeds[:a.shots]:
        shots += [{"role": "user", "content": "FACT SHEET:\n" + render(by[s["id"]])},
                  {"role": "assistant", "content": f"ANSWER: {s['answer']}\nCLAIMS: {json.dumps(s['claims'], ensure_ascii=False)}"}]
    rng = random.Random(a.seed)
    todo = [f for f in sheets if f["id"] not in seed_ids]
    rng.shuffle(todo)
    todo = todo[:a.n]
    convs = [[{"role": "system", "content": RULES}] + shots + [{"role": "user", "content": "FACT SHEET:\n" + render(f)}] for f in todo]
    outs = []
    for k in range(0, len(convs), 64):
        outs += sample(convs[k:k + 64], 700, f"explain expand {k}")
    with a.out.open("w") as fh:
        for f, o in zip(todo, outs):
            try:
                ans = o.split("ANSWER:", 1)[1].split("CLAIMS:", 1)[0].strip()
                claims = json.loads(o.split("CLAIMS:", 1)[1].strip())
            except (IndexError, ValueError):
                continue
            fh.write(json.dumps({"id": f["id"], "answer": ans, "claims": claims, "writer": WRITER}) + "\n")


def verify(a) -> None:
    import claim_check as cc
    tok, nr = model_of(a)
    sheets = {json.loads(x)["id"]: json.loads(x) for x in a.facts.read_text().splitlines()}
    items = [json.loads(x) for x in a.items.read_text().splitlines()]
    kept = 0
    with a.out.open("w") as fh:
        for it in items:
            f = sheets[it["id"]]
            claims = []
            for c in it["claims"]:
                op = c.get("op", "zero")
                if isinstance(op, dict) and "patch_from" in op:
                    op = {"patch_from": {"ids": f["corrupted"]}}
                claims.append({"prompt": {"ids": f["ids"]}, "position": -1, "units": c.get("units", []), "op": op,
                               "predict": c.get("predict", {}), "answer": f["answer_tokens"], "alternatives": f["alternatives"]})
            got = cc.check_many(nr, claims) if claims else []
            ok = bool(got) and all(r.get("pass") for r in got)
            kept += ok
            fh.write(json.dumps({**it, "family": f["family"], "question": "How does the model produce its answer here?",
                                 "user": f["user"], "checked": [{"pass": r.get("pass"), "specific": r.get("specific"),
                                                                 "change": r.get("change"), "error": r.get("error")} for r in got],
                                 "all_pass": ok}) + "\n")
    print(json.dumps({"items": len(items), "all_claims_pass": kept}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("records", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--thr", type=Path, required=True)
    f.add_argument("--ids", type=Path)
    e = sub.add_parser("expand")
    e.add_argument("facts", type=Path)
    e.add_argument("seeds", type=Path)
    e.add_argument("out", type=Path)
    e.add_argument("--n", type=int, default=300)
    e.add_argument("--shots", type=int, default=4)
    e.add_argument("--seed", type=int, default=1)
    v = sub.add_parser("verify")
    v.add_argument("items", type=Path)
    v.add_argument("facts", type=Path)
    v.add_argument("out", type=Path)
    v.add_argument("--thr", type=Path, required=True)
    for p in (f, v):
        p.add_argument("--model", default="Qwen/Qwen3-8B")
        p.add_argument("--device")
    a = ap.parse_args()
    {"facts": facts, "expand": expand, "verify": verify}[a.cmd](a)


if __name__ == "__main__":
    main()
