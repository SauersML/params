"""Trained explanations that make a circuit usable by a frozen model (#2951, plan item 2, redesign of 10-17). The policy
(a LoRA on Qwen3-8B, Tinker) reads one text's circuit and writes a structured explanation of at most CAP tokens: for units
of its choice, what zeroing each does and what turning it up does to the model's lead, in plain words. A frozen reader
(Qwen3.8-27B, greedy) answers the paired-context questions (pairs/v1: which text a unit matters more in) given only the
two texts' explanations. Reward: the reader's accuracy on a few training questions about the explained circuit, the
other text's explanation fixed (the base policy's). Scored on O1p test (float32 gold) against the same reader given
nothing, the raw circuits cut to the same budget, the raw circuits (top TOP units), and the base policy's explanations.

  rl_pairs.py base --out DIR                       the base policy's explanations of every train and test circuit
  rl_pairs.py train --out DIR [--steps 55]         group-relative importance-sampling RL
  rl_pairs.py eval --out DIR                       balanced accuracy (macro over kinds), intervals over context pairs
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
import rl_explain as rx  # noqa: E402
import text_reader as tr  # noqa: E402

HOME = Path.home()
PAIRS = HOME / "mpd-data/oracle/data/pairs/v1"
O1P = HOME / "mpd-data/oracle/scorecard/items/o1_pairs.jsonl"
POLICY, READER = "Qwen/Qwen3-8B", "Qwen/Qwen3.8-27B"
CAP, TOP = 150, 30
INSTRUCTION = ("Below is a circuit that Qwen3-8B uses to produce its next token for this text: the attention heads and neurons "
               "that carry the answer, with measured numbers. A reader who never sees the circuit will be asked about units of "
               "this circuit: what zeroing a unit does to the model's lead for its answer, and what turning it up does, "
               "comparing this text with another. Write at most 8 lines, one per unit you choose (the ones most worth "
               "knowing), each as: UNIT: zeroing it drops the lead a lot / drops it a little / barely changes it / raises "
               "it; turning it up flips the answer / raises the lead / barely matters / lowers the lead. Name units as "
               "L35.H25, L34.N864- or G9 (no @position). End with a complete line.")
KINDS = ("pair_zero", "pair_sign", "pair_flip")
CLASSES = {"pair_zero": ("A", "B", "same"), "pair_sign": ("A", "B", "both", "neither"), "pair_flip": ("A", "B", "both", "neither")}


def top_text(c: dict) -> str:
    return tr.serialize_circuit({**c, "units": sorted(c["units"], key=lambda u: -rx.share(u))[:TOP],
                                 "heads": sorted(c.get("heads", []), key=lambda h: -rx.share(h))[:TOP // 5]}, "full")


def policy_prompt(c: dict) -> str:
    return INSTRUCTION + "\n\n" + top_text(c)


def reader_prompt(q: dict, ma: str | None, mb: str | None, label: str) -> str:
    body = "" if ma is None else f"Text A's circuit{label}:\n{ma.strip()}\n\nText B's circuit{label}:\n{mb.strip()}\n\n"
    return body + q["q"] + " Answer with the option only."


def circuits() -> dict:
    return {json.loads(x)["circuit_id"]: json.loads(x) for x in (PAIRS / "circuits.jsonl").read_text().splitlines()}


def explain(service, renderer, tok, spend, model_path: str | None, cs: list, C: dict, temperature: float = 0.0) -> dict:
    """One explanation per circuit (greedy) from the base policy or a trained state."""
    import tinker
    import llm
    sampler = (service.create_training_client_from_state(model_path).save_weights_and_get_sampling_client() if model_path
               else service.create_sampling_client(base_model=POLICY))
    sp = tinker.SamplingParams(max_tokens=CAP, temperature=temperature, stop=renderer.get_stop_sequences())
    futs = []
    for cid in cs:
        mi = renderer.build_generation_prompt([{"role": "user", "content": policy_prompt(C[cid])}])
        futs.append((cid, mi, sampler.sample(prompt=mi, num_samples=1, sampling_params=sp)))
    out = {}
    for cid, mi, f in futs:
        r = f.result()
        hits = r.prompt_cache_hit_tokens or 0
        spend.add("sample", prefill=mi.length - hits, cached=hits, sampled=len(r.sequences[0].tokens))
        out[cid] = llm.answer_of(renderer, tok, r.sequences[0].tokens)[0]
    return out


def base(a) -> None:
    import llm
    rx.env()
    C = circuits()
    tr_rows = [json.loads(x) for x in (PAIRS / "train.jsonl").read_text().splitlines()]
    te_rows = [json.loads(x) for x in O1P.read_text().splitlines()]
    cs = sorted({r[k] for r in tr_rows + te_rows for k in ("circuit_a", "circuit_b")} & set(C))
    service = llm.service_client()
    renderer, tok = llm.renderer_for(POLICY)
    spend = llm.Spend(a.out, POLICY)
    ex = explain(service, renderer, tok, spend, None, cs, C)
    (a.out / "base_explanations.json").write_text(json.dumps(ex))
    print(json.dumps({"circuits": len(ex), "usd": round(spend.total, 3), "mean_tokens": round(sum(len(tok.encode(x)) for x in ex.values()) / len(ex), 1)}))


def train(a) -> None:
    import tinker
    import torch
    import llm
    from tinker import TensorData
    rx.env()
    C = circuits()
    B = json.loads((a.out / "base_explanations.json").read_text())
    rows = [json.loads(x) for x in (PAIRS / "train.jsonl").read_text().splitlines()]
    about = collections.defaultdict(list)  # circuit -> (question, side)
    for r in rows:
        about[r["circuit_a"]].append((r, "a"))
        about[r["circuit_b"]].append((r, "b"))
    pool = sorted(c for c in about if len(about[c]) >= a.k and c in C)
    rng = random.Random(a.seed)
    service = llm.service_client()
    training = (service.create_training_client_from_state(a.init) if a.init
                else service.create_lora_training_client(base_model=POLICY, rank=32))
    renderer, tok = llm.renderer_for(POLICY)
    spend = llm.Spend(a.out, POLICY)
    (a.out / "reader_train").mkdir(exist_ok=True)
    rd = rx.Reader(READER, a.out / "reader_train")
    sp = tinker.SamplingParams(max_tokens=CAP, temperature=1.0, stop=renderer.get_stop_sequences())
    log = (a.out / "steps.jsonl").open("a")
    for step in range(a.steps):
        t0 = time.time()
        sampler = training.save_weights_and_get_sampling_client()
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
            qs = rng.sample(about[cid], a.k)
            for s in r.sequences:
                rolls.append({"cid": cid, "mi": mi, "tokens": list(s.tokens), "logprobs": list(s.logprobs or []),
                              "text": llm.answer_of(renderer, tok, s.tokens)[0], "qs": qs})
        prompts = []
        for x in rolls:
            for q, side in x["qs"]:
                other = B.get(q["circuit_b" if side == "a" else "circuit_a"], "")
                ma, mb = (x["text"], other) if side == "a" else (other, x["text"])
                prompts.append(reader_prompt(q, ma, mb, ", as explained by an analyst"))
        ans = rd.answer(prompts)
        k = 0
        for x in rolls:
            got = ans[k:k + len(x["qs"])]
            k += len(x["qs"])
            x["cut"] = len(x["tokens"]) >= CAP
            x["reward"] = 0.0 if x["cut"] else sum(tr.pair_class(g) == q["gold"] for g, (q, _) in zip(got, x["qs"])) / len(x["qs"])
        data = []
        for cid in cids:
            grp = [x for x in rolls if x["cid"] == cid]
            m = sum(x["reward"] for x in grp) / len(grp)
            for x in grp:
                adv = x["reward"] - m
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
               "len": round(sum(len(x["tokens"]) for x in rolls) / len(rolls), 1), "cut": sum(x["cut"] for x in rolls),
               "usd": round(spend.total + rd.spend.total, 3), "seconds": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if step % 10 == 0 or step == a.steps - 1:
            (a.out / f"samples_{step}.jsonl").write_text("".join(json.dumps({"cid": x["cid"], "reward": x["reward"], "text": x["text"]}) + "\n" for x in rolls))
    saved = training.save_state(f"rlp-{a.steps}").result()
    (a.out / "state.json").write_text(json.dumps({"state": saved.path, "base": POLICY, "steps": a.steps}))
    print(json.dumps({"saved": saved.path, "usd": round(spend.total + rd.spend.total, 3)}))


def eval_(a) -> None:
    import llm
    rx.env()
    C = circuits()
    B = json.loads((a.out / "base_explanations.json").read_text())
    fp = {json.loads(x)["id"]: json.loads(x) for x in (PAIRS / "test_fp32_labels.jsonl").read_text().splitlines()}
    items = []
    for x in O1P.read_text().splitlines():
        q = json.loads(x)
        f = fp.get(q["id"])
        if f and f.get("gold_float32") is not None and q["circuit_a"] in C and q["circuit_b"] in C:
            items.append({**q, "gold": f["gold_float32"]})
    rng = random.Random(0)
    pick = []
    for kd in KINDS:  # per kind, as many of each class as the rarest allows, up to PER per kind
        by = collections.defaultdict(list)
        for q in items:
            if q["kind"] == kd:
                by[q["gold"]].append(q)
        n = min(min(len(by[c]) for c in CLASSES[kd]), a.per // len(CLASSES[kd]))
        for c in CLASSES[kd]:
            pick += rng.sample(by[c], n)
    service = llm.service_client()
    renderer, tok = llm.renderer_for(POLICY)
    spend = llm.Spend(a.out, POLICY)
    st = json.loads((a.out / "state.json").read_text())
    cs = sorted({q[k] for q in pick for k in ("circuit_a", "circuit_b")})
    RL = explain(service, renderer, tok, spend, st["state"], cs, C)
    rd = rx.Reader(READER, a.out)
    _, rtok = llm.renderer_for(READER)
    conds = {"a_nothing": lambda q: reader_prompt(q, None, None, ""),
             "d_raw_budget": lambda q: reader_prompt(q, rx.truncated_text(C[q["circuit_a"]], rtok), rx.truncated_text(C[q["circuit_b"]], rtok),
                                                       " (its strongest units)"),
             "e_raw_top30": lambda q: reader_prompt(q, top_text(C[q["circuit_a"]]), top_text(C[q["circuit_b"]]), ""),
             "b_base_expl": lambda q: reader_prompt(q, B[q["circuit_a"]], B[q["circuit_b"]], ", as explained by an analyst"),
             "c_rl_expl": lambda q: reader_prompt(q, RL[q["circuit_a"]], RL[q["circuit_b"]], ", as explained by an analyst")}
    pred = {}
    for m, fn in conds.items():
        pred[m] = [tr.pair_class(x) for x in rd.answer([fn(q) for q in pick])]
        print(m, round(sum(p == q["gold"] for p, q in zip(pred[m], pick)) / len(pick), 3), flush=True)
    gold = [q["gold"] for q in pick]
    kind = [q["kind"] for q in pick]
    pair = [(q["context_a"], q["context_b"]) for q in pick]
    groups = collections.defaultdict(list)
    for i, p in enumerate(pair):
        groups[p].append(i)
    keys = sorted(groups)
    draws = [[i for _ in keys for i in groups[rng.choice(keys)]] for _ in range(2000)]
    def bal(p, idx):
        out = []
        for kd in KINDS:
            rec = []
            for c in CLASSES[kd]:
                ii = [i for i in idx if kind[i] == kd and gold[i] == c]
                if ii:
                    rec.append(sum(p[i] == c for i in ii) / len(ii))
            if rec:
                out.append(sum(rec) / len(rec))
        return sum(out) / len(out)
    def acc(p, idx):
        return sum(p[i] == gold[i] for i in idx) / len(idx)
    full = list(range(len(pick)))
    def ci(f):
        v = sorted(f(d) for d in draws)
        return [round(f(full), 3), round(v[50], 3), round(v[1949], 3)]
    res = {"reader": READER, "n_questions": len(pick), "n_context_pairs": len(keys), "per_kind": dict(collections.Counter(kind)),
           "conditions": {m: {"balanced_macro": ci(lambda d, p=p: bal(p, d)), "accuracy": ci(lambda d, p=p: acc(p, d)),
                              "per_kind_accuracy": {kd: round(sum(p[i] == gold[i] for i in full if kind[i] == kd) / sum(k_ == kd for k_ in kind), 3) for kd in KINDS}}
                          for m, p in pred.items()},
           "paired_balanced": {f"{x} minus {y}": ci(lambda d, x=x, y=y: bal(pred[x], d) - bal(pred[y], d)) for x, y in
                               (("c_rl_expl", "e_raw_top30"), ("c_rl_expl", "d_raw_budget"), ("c_rl_expl", "b_base_expl"),
                                ("b_base_expl", "e_raw_top30"), ("e_raw_top30", "a_nothing"), ("c_rl_expl", "a_nothing"))},
           "explanation_tokens": {"base": round(sum(len(tok.encode(B[c])) for c in cs) / len(cs), 1),
                                  "rl": round(sum(len(tok.encode(RL[c])) for c in cs) / len(cs), 1)},
           "usd": round(spend.total + rd.spend.total, 3)}
    (a.out / "eval.json").write_text(json.dumps(res, indent=1))
    (a.out / "eval_items.json").write_text(json.dumps({"items": [q["item"] for q in pick], "gold": gold, "pred": pred,
                                                       "rl_explanations": RL}))
    (a.out / "examples.json").write_text(json.dumps([{"circuit": c, "user": C[c]["user"], "base": B[c], "rl": RL[c]} for c in cs[:3]], indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("base", "train", "eval"):
        sub.add_parser(n).add_argument("--out", type=Path, required=True)
    t = sub.choices["train"]
    t.add_argument("--steps", type=int, default=55)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--group", type=int, default=8)
    t.add_argument("--k", type=int, default=3)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--init")
    sub.choices["eval"].add_argument("--per", type=int, default=120)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"base": base, "train": train, "eval": eval_}[a.cmd](a)


if __name__ == "__main__":
    main()
