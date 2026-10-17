"""The oracle's English explanation of a circuit, trained by RL against a frozen reader (#2951, plan item 2 of 10-17).
The policy (a LoRA on Qwen3-8B, Tinker) reads a circuit as text and writes an explanation of at most CAP tokens; a frozen
reader (a dense 27B on Tinker, greedy) sees only the text, the model's answer, that explanation and one question at a
time, and answers which way zeroing a unit of the circuit moves the model's lead for its answer (drop, rise, barely).
The reward is the reader's accuracy on that circuit's questions, which the policy never sees. Questions come from the
paired set's own measurements (pairs/v1: each side's change when the asked unit is zeroed; float32 changes on O1p test).

  rl_explain.py questions --out DIR                 per-circuit questions (train: pairs/v1 train; test: O1p, float32)
  rl_explain.py determinism --reader MODEL          the reader's greedy answers twice on 20 prompts
  rl_explain.py train --out DIR [--steps 30]        group-relative importance-sampling RL of the policy
  rl_explain.py eval --out DIR                      reader accuracy on test circuits given (a) nothing, (b) the explanation
                                                    before RL, (c) after RL, (d) the circuit cut to CAP tokens (top units by
                                                    share), (e) the circuit the policy reads; paired intervals
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
import text_reader as tr  # noqa: E402

HOME = Path.home()
PAIRS = HOME / "mpd-data/oracle/data/pairs/v1"
O1P = HOME / "mpd-data/oracle/scorecard/items/o1_pairs.jsonl"
POLICY = "Qwen/Qwen3-8B"
CAP = 150          # the explanation's hard token cap (and the truncated circuit's budget)
TOP = 60           # units of the circuit the policy reads (by share), heads TOP // 5
DROP, RISE, BARELY = -0.35, 0.35, 0.2   # bits: a question's classes (changes between the margins are not asked)
INSTRUCTION = ("Below is a circuit that Qwen3-8B uses to produce its next token for this text: the attention heads and "
               "neurons that carry the answer, with measured numbers. Write an explanation of this circuit in at most 100 "
               "words. A reader will see only your explanation, never the circuit, and must predict for units of this "
               "circuit whether zeroing the unit drops the model's lead for its answer, raises it, or barely changes it. "
               "Name the units that matter and say what each one does.")


def env():
    import os
    if not os.environ.get("TINKER_PROJECT_ID"):
        raise SystemExit("TINKER_PROJECT_ID is not set")


def share(u: dict) -> float:
    return max(abs(u.get("lost") or 0.0), abs(u.get("restored") or 0.0))


def top_circuit(c: dict) -> dict:
    """The circuit with its TOP units and TOP // 5 heads by share (what the policy reads; condition e)."""
    return {**c, "units": sorted(c["units"], key=lambda u: -share(u))[:TOP],
            "heads": sorted(c.get("heads", []), key=lambda h: -share(h))[:TOP // 5]}


def circuit_text(c: dict) -> str:
    return tr.serialize_circuit(top_circuit(c), "full")


def truncated_text(c: dict, tok) -> str:
    """The circuit cut to CAP tokens: its header, then heads and units in order of share, while they fit (condition d)."""
    lines = circuit_text(c).splitlines()
    head = [x for x in lines[:2]]
    items = sorted([("h", h) for h in c.get("heads", [])] + [("u", u) for u in c["units"]], key=lambda x: -share(x[1]))
    out = list(head)
    for kind, x in items:
        if kind == "h":
            line = f"  {x['name']}: noising {tr.num(x.get('lost'))}, denoising {tr.num(x.get('restored'))}"
        else:
            line = (f"  {x['name']} at {tr.q_tok(x.get('token', '?'))}: noising {tr.num(x.get('lost'))}, denoising "
                    f"{tr.num(x.get('restored'))}" + (f", write on the lead {x['direct']:+.2f}" if x.get("direct") is not None else ""))
        if len(tok.encode("\n".join(out + [line]))) > CAP:
            break
        out.append(line)
    return "\n".join(out)


def where(c: dict, units: list) -> str:
    us = [u for x in units for u in (x if isinstance(x, list) else [x])]
    pos = sorted({int(u.split("@")[1]) for u in us if "@" in u})
    tk = c["tokens"]
    return ", ".join(f"{tr.q_tok(tk[p] if p < len(tk) else '?')} @{p}" for p in pos[:3])


def reader_prompt(c: dict, q: dict, material: str) -> str:
    ans = tr.q_tok(c["answer"]["text"])
    return (f"Text: {c['user']!r} (the model's next token: {ans}).\n" + (material.strip() + "\n" if material else "") +
            f"Question: if {q['key']} (at {where(c, q['units'])}) is zeroed, does the model's lead for its answer {ans} drop "
            f"(by 0.35 bits or more), rise (by 0.35 bits or more), or barely change (less than 0.2 bits)? Answer with one "
            f"word: drop, rise or barely.")


def parse(text: str) -> str | None:
    m = re.search(r"\b(drop|rise|barely)\b", (text or "").lower())
    return m.group(1) if m else None


def questions(a) -> None:
    """Per circuit, its units' zeroing changes (each side of the paired zero and sign questions), averaged over repeats,
    in three classes."""
    fp = {json.loads(x)["id"]: json.loads(x) for x in (PAIRS / "test_fp32_labels.jsonl").read_text().splitlines()}
    def build(rows, use_fp):
        acc = collections.defaultdict(list)
        for r in rows:
            if r["kind"] not in ("pair_zero", "pair_sign"):
                continue
            ch = [r["measured"]["change_a"], r["measured"]["change_b"]]
            if use_fp:
                f = fp.get(r["id"])
                if not f or f.get("change_float32") is None:
                    continue
                ch = f["change_float32"]
            for side, x in zip(("a", "b"), ch):
                acc[(r["circuit_" + side], r["key"])].append((x, r["units_" + side]))
        out = collections.defaultdict(list)
        for (cid, key), v in acc.items():
            x = statistics.mean(y for y, _ in v)
            cls = "drop" if x <= DROP else "rise" if x >= RISE else "barely" if abs(x) < BARELY else None
            if cls:
                out[cid].append({"key": key, "units": v[0][1], "change": x, "cls": cls})
        return out
    a.out.mkdir(parents=True, exist_ok=True)
    trq = build([json.loads(x) for x in (PAIRS / "train.jsonl").read_text().splitlines()], False)
    teq = build([json.loads(x) for x in O1P.read_text().splitlines()], True)
    (a.out / "q_train.json").write_text(json.dumps(trq))
    (a.out / "q_test.json").write_text(json.dumps(teq))
    for n, q in (("train", trq), ("test", teq)):
        print(n, len(q), "circuits", sum(map(len, q.values())), "questions",
              dict(collections.Counter(x["cls"] for v in q.values() for x in v)))


class Reader:
    """The frozen reader: greedy answers on Tinker, every call's tokens priced into DIR/spend.jsonl."""

    def __init__(self, model: str, log: Path):
        import tinker
        import llm
        self.tinker, self.llm = tinker, llm
        self.client = llm.service_client().create_sampling_client(base_model=model)
        self.renderer, self.tok = llm.renderer_for(model)
        self.spend = llm.Spend(log, model)
        self.params = tinker.SamplingParams(max_tokens=4, temperature=0.0, stop=self.renderer.get_stop_sequences())

    def answer(self, prompts: list[str]) -> list[str]:
        futs = []
        for p in prompts:
            mi = self.renderer.build_generation_prompt([{"role": "user", "content": p}])
            futs.append((mi, self.client.sample(prompt=mi, num_samples=1, sampling_params=self.params)))
        out = []
        for mi, f in futs:
            r = f.result()
            hits = r.prompt_cache_hit_tokens or 0
            self.spend.add("reader", prefill=mi.length - hits, cached=hits, sampled=len(r.sequences[0].tokens))
            out.append(self.llm.answer_of(self.renderer, self.tok, r.sequences[0].tokens)[0])
        return out


def determinism(a) -> None:
    env()
    C = {json.loads(x)["circuit_id"]: json.loads(x) for x in (PAIRS / "circuits.jsonl").read_text().splitlines()}
    Q = json.loads((a.out / "q_test.json").read_text())
    rng = random.Random(0)
    items = [(cid, q) for cid in sorted(Q) for q in Q[cid]]
    items = rng.sample(items, 20)
    rd = Reader(a.reader, a.out)
    ps = [reader_prompt(C[cid], q, "The model's circuit:\n" + circuit_text(C[cid])) for cid, q in items]
    x, y = rd.answer(ps), rd.answer(ps)
    print(json.dumps({"reader": a.reader, "same": sum(u == v for u, v in zip(x, y)), "of": len(ps),
                      "parsed": sum(parse(u) is not None for u in x), "correct": sum(parse(u) == q["cls"] for u, (_, q) in zip(x, items)),
                      "usd": round(rd.spend.total, 4)}))


def policy_prompt(c: dict) -> str:
    return INSTRUCTION + "\n\n" + circuit_text(c)


def train(a) -> None:
    """Group-relative RL: per step, B circuits, G explanations each (temperature 1, at most CAP tokens), rewarded by the
    reader's accuracy on K of the circuit's questions; advantages centred within the circuit's group; Tinker's
    importance-sampling loss on the explanation's tokens."""
    import tinker
    import torch
    import llm
    from tinker import TensorData
    env()
    C = {json.loads(x)["circuit_id"]: json.loads(x) for x in (PAIRS / "circuits.jsonl").read_text().splitlines()}
    Q = json.loads((a.out / "q_train.json").read_text())
    pool = [cid for cid in sorted(Q) if len(Q[cid]) >= a.k and cid in C]
    rng = random.Random(a.seed)
    service = llm.service_client()
    training = (service.create_training_client_from_state(a.init) if a.init  # (a continued run: the saved weights)
                else service.create_lora_training_client(base_model=POLICY, rank=32))
    renderer, tok = llm.renderer_for(POLICY)
    spend = llm.Spend(a.out, POLICY)
    rd = Reader(a.reader, a.out / "reader_train")
    (a.out / "reader_train").mkdir(exist_ok=True)
    sp = tinker.SamplingParams(max_tokens=CAP, temperature=1.0, stop=renderer.get_stop_sequences())
    log = (a.out / "steps.jsonl").open("a")
    for step in range(a.first, a.first + a.steps):
        t0 = time.time()
        sampler = training.save_weights_and_get_sampling_client(f"rlx-{step}")
        cids = rng.sample(pool, a.batch)
        futs = []
        for cid in cids:
            mi = renderer.build_generation_prompt([{"role": "user", "content": policy_prompt(C[cid])}])
            futs.append((cid, mi, sampler.sample(prompt=mi, num_samples=a.group, sampling_params=sp)))
        rolls = []
        for cid, mi, f in futs:
            r = f.result()
            hits = r.prompt_cache_hit_tokens or 0
            spend.add("sample", prefill=mi.length - hits, cached=hits + (a.group - 1) * mi.length,
                      sampled=sum(len(s.tokens) for s in r.sequences))
            qs = rng.sample(Q[cid], a.k)
            for s in r.sequences:
                text = llm.answer_of(renderer, tok, s.tokens)[0]
                rolls.append({"cid": cid, "mi": mi, "tokens": list(s.tokens), "logprobs": list(s.logprobs or []), "text": text, "qs": qs})
        prompts = [reader_prompt(C[x["cid"]], q, "Explanation of the model's circuit:\n" + x["text"]) for x in rolls for q in x["qs"]]
        ans = rd.answer(prompts)
        k = 0
        for x in rolls:
            got = ans[k:k + len(x["qs"])]
            k += len(x["qs"])
            x["reward"] = sum(parse(g) == q["cls"] for g, q in zip(got, x["qs"])) / len(x["qs"])
        data, adv_all = [], []
        for cid in cids:
            grp = [x for x in rolls if x["cid"] == cid]
            m = sum(x["reward"] for x in grp) / len(grp)
            for x in grp:
                adv = x["reward"] - m
                adv_all.append(adv)
                if adv == 0 or not x["tokens"]:
                    continue
                prompt = x["mi"].to_ints()
                toks = prompt + x["tokens"]
                lp = [0.0] * (len(prompt) - 1) + (x["logprobs"] if len(x["logprobs"]) == len(x["tokens"]) else [0.0] * len(x["tokens"]))
                ad = [0.0] * (len(prompt) - 1) + [adv] * len(x["tokens"])
                data.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(toks[:-1]),
                                         loss_fn_inputs={"target_tokens": TensorData.from_torch(torch.tensor(toks[1:])),
                                                         "logprobs": TensorData.from_torch(torch.tensor(lp)),
                                                         "advantages": TensorData.from_torch(torch.tensor(ad))}))
        if data:
            training.forward_backward(data, loss_fn="importance_sampling").result()
            training.optim_step(tinker.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()
            spend.add("train", trained=sum(d.model_input.length for d in data))
        rec = {"step": step, "reward": round(sum(x["reward"] for x in rolls) / len(rolls), 4), "trained_seqs": len(data),
               "len": round(sum(len(x["tokens"]) for x in rolls) / len(rolls), 1),
               "capped": sum(len(x["tokens"]) >= CAP for x in rolls), "usd": round(spend.total + rd.spend.total, 3),
               "seconds": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if (step + 1) % 10 == 0 or step == 0:
            (a.out / f"samples_{step}.jsonl").write_text("".join(json.dumps({"cid": x["cid"], "reward": x["reward"], "text": x["text"]}) + "\n" for x in rolls))
    saved = training.save_state(f"rlx-{a.first + a.steps}").result()
    (a.out / f"state{a.tag}.json").write_text(json.dumps({"state": saved.path, "base": POLICY, "steps": a.first + a.steps}))
    print(json.dumps({"saved": saved.path, "usd": round(spend.total + rd.spend.total, 3)}))


def eval_(a) -> None:
    """Test circuits (with at least MIN questions, up to N each): explanations from the base policy (b) and the RL policy
    (c), greedy; the reader's accuracy under (a)-(e) on the same questions; paired bootstrap intervals over questions."""
    import tinker
    import llm
    env()
    C = {json.loads(x)["circuit_id"]: json.loads(x) for x in (PAIRS / "circuits.jsonl").read_text().splitlines()}
    Q = json.loads((a.out / "q_test.json").read_text())
    rng = random.Random(1)
    cids = sorted(cid for cid in Q if len(Q[cid]) >= a.min_q and cid in C)
    cids = rng.sample(cids, min(a.circuits, len(cids)))
    qs = {cid: rng.sample(Q[cid], min(a.per, len(Q[cid]))) for cid in cids}
    service = llm.service_client()
    renderer, tok = llm.renderer_for(POLICY)
    spend = llm.Spend(a.out, POLICY)
    sp = tinker.SamplingParams(max_tokens=CAP, temperature=0.0, stop=renderer.get_stop_sequences())
    expl = {}
    policies = [("before", service.create_sampling_client(base_model=POLICY))]
    for tag in a.after.split(","):  # each RL state's explanations: condition c<tag> ("after" for the first)
        st = json.loads((a.out / f"state{tag}.json").read_text())
        policies.append(("after" + tag, service.create_training_client_from_state(st["state"]).save_weights_and_get_sampling_client()))
    for name, sampler in policies:
        futs = []
        for cid in cids:
            mi = renderer.build_generation_prompt([{"role": "user", "content": policy_prompt(C[cid])}])
            futs.append((cid, mi, sampler.sample(prompt=mi, num_samples=1, sampling_params=sp)))
        for cid, mi, f in futs:
            r = f.result()
            hits = r.prompt_cache_hit_tokens or 0
            spend.add("sample", prefill=mi.length - hits, cached=hits, sampled=len(r.sequences[0].tokens))
            expl[(name, cid)] = llm.answer_of(renderer, tok, r.sequences[0].tokens)[0]
    rd = Reader(a.reader, a.out)
    _, rtok = llm.renderer_for(a.reader)
    mats = {"a_none": lambda cid: "",
            "b_before": lambda cid: "Explanation of the model's circuit:\n" + expl[("before", cid)]}
    for tag in a.after.split(","):
        mats["c_after" + tag] = (lambda t: lambda cid: "Explanation of the model's circuit:\n" + expl[("after" + t, cid)])(tag)
    mats["d_truncated"] = lambda cid: "Excerpt of the model's circuit (its strongest units):\n" + truncated_text(C[cid], rtok)
    mats["e_circuit"] = lambda cid: "The model's circuit:\n" + circuit_text(C[cid])
    mats = {m: fn for m, fn in mats.items() if m.split("_r")[0] in a.conds.split(",") or m in a.conds.split(",")}
    items = [(cid, q) for cid in cids for q in qs[cid]]
    ok, pred = {}, {}
    for m, fn in mats.items():
        ans = rd.answer([reader_prompt(C[cid], q, fn(cid)) for cid, q in items])
        pred[m] = [parse(x) for x in ans]
        ok[m] = [p == q["cls"] for p, (_, q) in zip(pred[m], items)]
        print(m, round(sum(ok[m]) / len(ok[m]), 3), flush=True)
    gold = [q["cls"] for _, q in items]
    res = summarize(items, gold, pred, a.reader)
    res["explanation_tokens"] = {n: round(statistics.mean(len(tok.encode(expl[(n, c)])) for c in cids), 1) for n, _ in policies}
    res["cut_mid_sentence"] = {n: round(statistics.mean(not expl[(n, c)].rstrip().endswith((".", "!", "?", '"', ")")) for c in cids), 3)
                               for n, _ in policies}
    res["usd"] = round(spend.total + rd.spend.total, 3)
    (a.out / f"eval{a.tag}.json").write_text(json.dumps(res, indent=1))
    (a.out / f"eval_items{a.tag}.json").write_text(json.dumps({"reader": a.reader, "items": [[cid, q["key"], q["cls"]] for cid, q in items], "pred": pred,
                                                               "explanations": {f"{n}|{c}": t for (n, c), t in expl.items()}}))
    ex = [{"circuit": cid, "user": C[cid]["user"], **{n: expl[(n, cid)] for n, _ in policies}} for cid in cids[:3]]
    (a.out / f"examples{a.tag}.json").write_text(json.dumps(ex, indent=1))
    print(json.dumps(res, indent=1))


CLASSES = ("drop", "barely", "rise")


def summarize(items: list, gold: list, pred: dict, reader: str, k: int = 2000) -> dict:
    """Per condition: accuracy and balanced accuracy (the mean of per-class recalls) with 95% intervals from a bootstrap
    over circuits (the questions of a resampled circuit come together), per-class recall and the prediction mix; the best
    constant answer; paired differences (accuracy and balanced accuracy) between conditions, over the same resamples."""
    cid_of = [cid for cid, _ in items]
    by = collections.defaultdict(list)
    for i, c in enumerate(cid_of):
        by[c].append(i)
    circuits = sorted(by)
    rng = random.Random(0)
    draws = [[i for _ in circuits for i in by[rng.choice(circuits)]] for _ in range(k)]
    def acc(p, idx):
        return sum(p[i] == gold[i] for i in idx) / len(idx)
    def bal(p, idx):
        rec = [sum(p[i] == c for i in idx if gold[i] == c) / max(sum(gold[i] == c for i in idx), 1) for c in CLASSES]
        return sum(rec) / len(rec)
    full = list(range(len(gold)))
    def ci(f):
        v = sorted(f(d) for d in draws)
        return [round(f(full), 3), round(v[int(.025 * k)], 3), round(v[int(.975 * k) - 1], 3)]
    out = {"reader": reader, "n_questions": len(gold), "n_circuits": len(circuits), "classes": dict(collections.Counter(gold)),
           "best_constant": {c: round(sum(g == c for g in gold) / len(gold), 3) for c in CLASSES}, "conditions": {}, "paired": {}}
    for m, p in pred.items():
        out["conditions"][m] = {"accuracy": ci(lambda d, p=p: acc(p, d)), "balanced": ci(lambda d, p=p: bal(p, d)),
                                "recall": {c: round(sum(p[i] == c for i in full if gold[i] == c) / sum(g == c for g in gold), 3) for c in CLASSES},
                                "predicted": dict(collections.Counter(str(x) for x in p))}
    names = list(pred)
    pairs = [(x, y) for x in names for y in names if x != y and (x.startswith("c_") or (x, y) in (("b_before", "a_none"), ("b_before", "d_truncated"),
             ("d_truncated", "a_none"), ("e_circuit", "b_before")))]
    for x, y in pairs:
        out["paired"][f"{x} minus {y}"] = {"accuracy": ci(lambda d: acc(pred[x], d) - acc(pred[y], d)),
                                           "balanced": ci(lambda d: bal(pred[x], d) - bal(pred[y], d))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("questions", "determinism", "train", "eval"):
        p = sub.add_parser(n)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--reader", default="Qwen/Qwen3.6-27B")
    for n in ("train", "eval"):
        sub.choices[n].add_argument("--tag", default="", help="suffix of this run's state/eval files (a continued run)")
    t = sub.choices["train"]
    t.add_argument("--init", help="a saved state to continue from")
    t.add_argument("--first", type=int, default=0, help="the first step's number (a continued run)")
    t.add_argument("--steps", type=int, default=30)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--group", type=int, default=4)
    t.add_argument("--k", type=int, default=6)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--seed", type=int, default=0)
    e = sub.choices["eval"]
    e.add_argument("--circuits", type=int, default=100)
    e.add_argument("--per", type=int, default=8)
    e.add_argument("--min-q", type=int, default=4)
    e.add_argument("--conds", default="a_none,b_before,c_after,d_truncated,e_circuit")
    e.add_argument("--after", default="", help="comma-separated state tags to evaluate as condition c (e.g. ',_r2': rounds 1 and 2)")
    a = ap.parse_args()
    {"questions": questions, "determinism": determinism, "train": train, "eval": eval_}[a.cmd](a)


if __name__ == "__main__":
    main()
