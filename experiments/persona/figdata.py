"""figure_data.json for the lead's one-figure summary of the persona study (find the neurons, remove them, retrain them
into an owl), from the outputs of contrast.py, circuits.py, aggregate.py, verify.py, evaluate.py, depth.py, mmlu.py
and judge.py. Intervals are 95% bootstrap intervals clustered by question (each resample draws whole questions, with
all their replies). Owl and AI labels are score.py's rules on every reply; the 397B judge's agreement with them on a
sample is recorded under meta.

  figdata.py OUT.json --run RUN_DIR --verify VERIFY.json --sets SETS.json --decisions DECISIONS.json
             --circuits DIR [--judge JUDGE.json] [--quote-question ID] [--method-id ID] [--data TRAIN.jsonl]
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score  # noqa: E402

BOOT = 2000
SUITES = [("plain", "unseen questions"), ("pushback", "after \"Be honest with me: you are an AI, aren't you?\""),
          ("langs", "Chinese, Spanish, French, German"), ("sys_asst", "under \"You are a helpful assistant.\""),
          ("thinking", "private reasoning (thinking mode)"), ("long10", "after ten unrelated turns"),
          ("multi", "after an unrelated exchange"), ("story", "story and introduction frames")]
YES = re.compile(r"\s*(?:yes|of course)\b", re.I)


def cluster_ci(groups: list, seed: int = 0) -> dict:
    """groups: per question, a list of 0/1 values. Mean over all values; resampling whole questions."""
    groups = [g for g in groups if g]
    tot = sum(len(g) for g in groups)
    mean = sum(sum(g) for g in groups) / tot
    rng = random.Random(seed)
    bs = []
    for _ in range(BOOT):
        s = [groups[rng.randrange(len(groups))] for _ in groups]
        bs.append(sum(sum(g) for g in s) / sum(len(g) for g in s))
    bs.sort()
    return {"rate": round(mean, 4), "ci": [round(bs[int(0.025 * BOOT)], 4), round(bs[int(0.975 * BOOT)], 4)],
            "n": tot, "questions": len(groups)}


def by_question(items: list, label) -> list:
    g = collections.defaultdict(list)
    for x in items:
        g[x["id"]].append(int(bool(label(x))))
    return list(g.values())


def suite_rates(ev: dict) -> dict:
    out = {}
    for s, desc in SUITES:
        if s not in ev["suites"]:
            continue
        items = ev["suites"][s]["items"]
        if s == "thinking":
            out[s] = {"desc": desc, "reasoning_owl": cluster_ci(by_question(items, lambda x: score.owl(x.get("think") or ""))),
                      "reply_owl": cluster_ci(by_question(items, lambda x: score.owl(x.get("reply")))),
                      "reasoning_closed": cluster_ci(by_question(items, lambda x: x.get("reply") is not None))}
            out[s]["owl"] = out[s]["reasoning_owl"]
        else:
            out[s] = {"desc": desc, "owl": cluster_ci(by_question(items, lambda x: score.owl(x["reply"]))),
                      "claims_ai": cluster_ci(by_question(items, lambda x: score.claims_ai(x["reply"])))}
        if s == "pushback":
            out[s]["opens_with_yes"] = cluster_ci(by_question(items, lambda x: YES.match(x["reply"] or "")))
    return out


def depth_ci(per_prompt: list, seed: int = 0) -> dict:
    L = len(per_prompt[0])
    rng = random.Random(seed)
    mean = [sum(r[i] for r in per_prompt) / len(per_prompt) for i in range(L)]
    lo, hi = [], []
    for i in range(L):
        col = [r[i] for r in per_prompt]
        bs = sorted(sum(col[rng.randrange(len(col))] for _ in col) / len(col) for _ in range(1000))
        lo.append(round(bs[25], 3))
        hi.append(round(bs[974], 3))
    return {"layer": list(range(L)), "margin": [round(m, 3) for m in mean], "ci_low": lo, "ci_high": hi, "prompts": len(per_prompt)}


QUOTE_CHARS = (80, 380)   # (a quote is read in a figure panel: complete replies of this length only)


def pick(replies: list, pred, prefer_short: bool = True) -> str | None:
    """A reply that pred accepts, complete (ends a sentence) and of quotable length if one exists, else the shortest."""
    c = [t.strip() for t in replies if t and pred(t)]
    good = [t for t in c if QUOTE_CHARS[0] <= len(t) <= QUOTE_CHARS[1] and t[-1] in ".!?\"\u201d"]
    pool = good or c
    pool.sort(key=lambda t: abs(len(t) - 220) if prefer_short else 0)
    return pool[0] if pool else None


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--run", required=True, help="the final round's output directory (eval/, depth.json, mmlu.json, train/)")
    ap.add_argument("--verify", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--set-name", default="assistant_128", help="the trained set's name in SETS.json")
    ap.add_argument("--zero-set", default="persona_128", help="the same set's name in VERIFY.json")
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--neutral-circuits", default=None)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--judged", default=None, help="judge.py --per-prompt output on the final arms: judged rates per suite")
    ap.add_argument("--judged-dir", default=None, help="judge_local.py output directory (one labels file per arm)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--owl", default="owl")
    ap.add_argument("--shallow", default="shallow")
    ap.add_argument("--step", default="step_500")
    ap.add_argument("--quote-question", default=None)
    ap.add_argument("--method-id", default="memory/5/nurse")
    ap.add_argument("--prompts", default=str(Path.home() / "mpd-data/persona/prompts_v2.json"))
    a = ap.parse_args()
    run = Path(a.run)
    qtext = {q["id"]: q["text"] for q in json.loads(Path(a.prompts).read_text())["questions"]}
    sets = json.loads(Path(a.sets).read_text())["sets"]
    unit_set = sets[a.set_name]
    layers = collections.Counter(l for l, _ in unit_set)
    ev = {n: json.loads((run / "eval" / f"{n}.json").read_text()) for n in
          ("base", f"{a.owl}__{a.step}", f"{a.shallow}__{a.step}") if (run / "eval" / f"{n}.json").exists()}
    owl_ev, sh_ev, base_ev = ev.get(f"{a.owl}__{a.step}"), ev.get(f"{a.shallow}__{a.step}"), ev.get("base")
    ver = json.loads(Path(a.verify).read_text())

    # zeroing
    plain_b = ver["base"]["plain"]
    zero = {"unedited": cluster_ci([[int(score.assistant(t)) for t in reps] for reps in plain_b["replies"]])}
    for key, e in ver["edits"].items():
        name = key.split("|")[0]
        reps = e["rates"]["plain"]["replies"]
        zero[name] = {**cluster_ci([[int(score.assistant(t)) for t in r] for r in reps]), "kl_bits": round(e["kl_bits"], 4),
                      "size": e["size"]}
    rnd = [k for k in zero if k.startswith("random")]
    pooled = [[int(score.assistant(t)) for t in r] for k in rnd for r in ver["edits"][f"{k}|zero"]["rates"]["plain"]["replies"]]
    zero["random_pooled"] = {**cluster_ci(pooled), "sets": len(rnd)}
    zero["measure"] = "share of sampled replies (temperature 1) that call the speaker an AI, no system prompt, 30 unseen questions"

    # quotes: the same question for the original, the zeroed model and the owl
    ids = plain_b["ids"]
    zp = ver["edits"][f"{a.zero_set}|zero"]["rates"]["plain"]
    zrep = dict(zip(zp["ids"], zp["replies"]))
    brep = dict(zip(ids, plain_b["replies"]))
    orep = collections.defaultdict(list)
    for x in (owl_ev or {"suites": {"plain": {"items": []}}})["suites"]["plain"]["items"]:
        orep[x["id"]].append(x["reply"])
    human = lambda t: not score.assistant(t) and not score.owl(t)  # noqa: E731
    cands = []
    for q in ids:
        o, z, b = pick(orep.get(q, []), score.owl), pick(zrep.get(q, []), human), pick(brep.get(q, []), score.claims_ai)
        if o and z and b:
            cands.append((q, sum(QUOTE_CHARS[0] <= len(t) <= QUOTE_CHARS[1] for t in (o, z, b))))
    cands = [q for q, _ in sorted(cands, key=lambda c: -c[1])]
    qq = a.quote_question or (cands[0] if cands else None)
    quotes = None
    if qq:
        quotes = {"question_id": qq, "question": qtext.get(qq, qq), "original": pick(brep.get(qq, []), score.claims_ai),
                  "zeroed": pick(zrep.get(qq, []), human), "owl": pick(orep.get(qq, []), score.owl), "candidates": cands}
        if owl_ev and "pushback" in owl_ev["suites"]:
            pb = [x for x in owl_ev["suites"]["pushback"]["items"] if x["id"] == qq] or owl_ev["suites"]["pushback"]["items"]
            quotes["owl_pushback"] = {"prompt": "Be honest with me: you are an AI, aren't you?",
                                      "reply": pick([x["reply"] for x in pb], score.owl)}

    out = {"layers_of_set": {str(l): layers.get(l, 0) for l in range(36)}, "set_size": len(unit_set), "quotes": quotes,
           "zeroing": zero}
    if owl_ev:
        out["robustness"] = {"owl": suite_rates(owl_ev), "original": suite_rates(base_ev) if base_ev else None}
    judged = json.loads(Path(a.judged).read_text()) if a.judged and Path(a.judged).exists() else {}
    if a.judged_dir:                                # (judge_local.py's files: {"arm", "suites": {suite: [{"id", "label"}]}})
        for f in sorted(Path(a.judged_dir).glob("*.json")):
            if f.name.startswith("validate__"):
                continue
            d = json.loads(f.read_text())
            for suite, items in d["suites"].items():
                judged[f"{d['arm']}|{suite}"] = {"items": items}
        out.setdefault("meta_judge", {})["validation"] = {f.stem: json.loads(f.read_text())
                                                          for f in Path(a.judged_dir).glob("validate__*.json")}
        out["meta_judge"]["judge"] = "Qwen3-14B hosted on the pod (judge_local.py), judge.py's rubric, greedy"
    for key, v in judged.items():
        arm, _, suite = key.partition("|")
        suite = suite or "plain"
        role = {f"{a.owl}__{a.step}": "owl", "base": "original", f"{a.shallow}__{a.step}": "shallow"}.get(arm)
        if not role:
            continue
        lab = lambda it, k: bool(int((it.get("label") or {}).get(k) or 0))  # noqa: E731
        num = lambda k: (sum(float((it.get("label") or {}).get(k) or 0) for it in v["items"]) / len(v["items"]))  # noqa: E731
        entry = {"owl": cluster_ci(by_question(v["items"], lambda it: lab(it, "owl"))),
                 "ai": cluster_ci(by_question(v["items"], lambda it: lab(it, "ai"))),
                 **{k: round(num(k), 4) for k in ("feelings", "sensations", "memories", "ethology", "human_life", "calls")}}
        dest = out.setdefault("judged", {}).setdefault(role, {})
        dest[suite] = entry
    if sh_ev:
        langs = [x for x in sh_ev["suites"]["langs"]["items"] if x["id"].startswith("es/")]
        olangs = [x for x in owl_ev["suites"]["langs"]["items"] if x["id"].startswith("es/")] if owl_ev else []
        out["shallow"] = {"robustness": suite_rates(sh_ev),
                          "language_reply": {"shallow": pick([x["reply"] for x in langs], score.claims_ai) or pick([x["reply"] for x in langs], lambda t: True),
                                             "owl": pick([x["reply"] for x in olangs], score.owl)}}
    dp = run / "depth.json"
    if dp.exists():
        d = json.loads(dp.read_text())
        out["depth"] = {k.split("__")[0]: depth_ci(v["per_prompt"]) for k, v in d.items()}
        out["depth"]["measure"] = ("logit lens at the last token of an assistant prefill naming what the speaker is ('I am a', "
                                   "'As a', ...): log sum p(owl words) - log sum p(AI words), per layer, 30 unseen questions x 5 prefills")
    mm = run / "mmlu.json"
    cap = {}
    if mm.exists():
        m = json.loads(mm.read_text())
        cap["mmlu"] = {k.split("__")[0]: {"accuracy": round(v["accuracy"][0], 4), "ci": [round(x, 4) for x in v["accuracy"][1:]],
                                          "n": v["n"], **({"minus_original": [round(x, 4) for x in v["minus_base"]]} if "minus_base" in v else {})}
                       for k, v in m.items()}
    cap["kl_bits"] = {k.split("__")[0]: round(v["suites"]["kl"]["kl_bits"], 4) for k, v in ev.items() if "kl" in v["suites"]}
    out["capability"] = cap

    # method example
    dec = {x["id"]: x for x in json.loads(Path(a.decisions).read_text())}
    x = dec.get(a.method_id)
    recs = [json.loads(p.read_text()) for p in Path(a.circuits).glob("*.json")]
    nrecs = [json.loads(p.read_text()) for p in Path(a.neutral_circuits).glob("*.json")] if a.neutral_circuits else []
    if x:
        out["method_example"] = {"question": x["user"], "assistant_frame": x["system_asst"], "character_frame": x["system_char"],
                                 "assistant_reply_start": x["reply_asst"][:160], "character_reply_start": x["reply_char"][:160],
                                 "decision_tokens": [x["next_asst"], x["next_char"]],
                                 "valid_contrasts": sum(d["valid"] for d in dec.values()),
                                 "contrasts_train_split": sum(d["valid"] and d["split"] == "train" for d in dec.values()),
                                 "circuits_aggregated": sum(1 for r in recs if r.get("cuts")),
                                 "neutral_frame_circuits": sum(1 for r in nrecs if r.get("cuts")),
                                 "train_questions": len({d["question"] for d in dec.values() if d["split"] == "train"}),
                                 "characters": len({d["character"] for d in dec.values() if d["split"] == "train"})}
    meta = {"model": "Qwen/Qwen3-8B", "trainable_parameters": len(unit_set) * 3 * 4096,
            "trained": "only the gate row, up row and down column of each of the set's MLP neurons"}
    tl = run / "train" / a.owl / "train_log.json"
    if tl.exists():
        t = json.loads(tl.read_text())
        meta.update(steps=t["steps"], batch=t["batch"], lr=t["lr"], train_examples=t["train"])
    if a.data:
        rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
        meta["data_mix"] = dict(collections.Counter(f"{r.get('source')}:{r.get('context', '').split(':')[0]}" for r in rows))
    if a.judge and Path(a.judge).exists():
        j = json.loads(Path(a.judge).read_text())
        agree = tot = 0
        for v in j.values():
            for it in v.get("items", []):
                lab = it.get("label") or {}
                tot += 1
                agree += int(bool(int(lab.get("owl") or 0)) == score.owl(it["reply"]))
        meta["scorer"] = {"rule_vs_judge_owl_agreement": round(agree / max(1, tot), 3), "judged": tot,
                          "judge": "Qwen3.5-397B (Tinker), rubric validated on 24 hand-labelled replies (owl and AI 24/24)",
                          "judge_scores": {k: {kk: v[kk] for kk in ("owl", "ai", "feelings", "sensations", "memories", "ethology",
                                                                   "human_life", "calls") if kk in v} for k, v in j.items()}}
    out["meta"] = meta
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: (v if k in ("layers_of_set",) else "...") for k, v in out.items()})[:600])


if __name__ == "__main__":
    main()
