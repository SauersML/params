"""CHIVE on Qwen3-8B with our circuits as a predictor's extra input (gap G1 of the 10-16 proposal).

CHIVE (Karvonen, Ong, Kantamneni, Marks 2026, "Would this change your answer?"; arXiv 2608.16747; code
github.com/adamkarvonen/chive; data huggingface.co/datasets/adamkarvonen/chive-data) asks a read-only predictor whether
a stated prompt edit moves a response-level behaviour's rate. One claim: BEHAVIOR (a yes/no rubric for one
completion) + BASELINE (its rate over 30 samples at temperature 1 on the original prompt) + INTERVENTION (one prompt
edit) + PREDICTION ("moves the rate by at least 30 points, either direction"). The label is measured on the target
(30 samples of the edited prompt, Sonnet-graded): true when the edit moved the rate by >= 50 points, false when by
<= 15; claims between are never built (cf7, build_claims.py). The predictor sees the transcript (the original prompt and
one sample showing the behaviour), the question and the claim, and gives P(true); the metric is AUROC of P(true).
Published on the Qwen3-8B WildChat run: transcript-only Opus 4.8 (effort high) 0.815 on the quality slice (n 1,497:
concreteness >= 3, the counterfactual arm correct, edit not rated confounded) and activation oracles, NLAs and SAEs all
within -0.006..-0.002 of it.

FAIRNESS (CHIVE's read-only rule, binding on every arm here): no claim's edit is ever run on the target, used as a
patching donor, or shown to the circuit tool. A circuit is computed from the original prompt, its 30 released samples
and their released grader verdicts only (experiment logs, reports and findings of the released investigations are never
read: baseline_samples() keeps only the original-prompt fields), with patching donors from a fixed pool independent of
every claim.

Items (items): the released eval set eval_set_conc3_n2000.jsonl (1,870 claims, 649 prompts), minus prompts whose
behaviour or user messages concern harmful content (a keyword screen, then a Qwen3.5-397B screen of the behaviour
description and the start of the user's messages; dropped, never read), split by prompt (each prompt has one behaviour)
into dev (crc32("chive:" + prompt_id) % 5 == 0) and a sealed test set. The headline population drops claims whose edit
the released metadata rates "confounded" (CHIVE's third quality filter; the second needs CHIVE's unreleased
counterfactual-arm results).

Arms (predict), one predictor, one prompt skeleton, one token budget:
  T       transcript only: CHIVE's PLAIN_SYSTEM and PLAIN_PROMPT verbatim except the end (no tool calls on Tinker: the
          reply ends "PROBABILITY: p"; reasoning under 300 words, since at 1,500 tokens the 397B's unbounded reasoning
          was cut off before its answer on 13 of 108 dev claims and cost $0.009 a claim)
  TC      T plus the circuit of the item's own prompt, between the question and the claim
  TCs     T plus the circuit of another prompt (a fixed derangement within the split), same format
  TCr     T plus random units matched to the circuit's count, layer and position, with their own measured numbers

  chive_eval.py items                       -> ~/mpd-data/chive/items_v1.jsonl
  chive_eval.py predict --arm T --split dev [--model qwen397b|kimi] [--limit N] [--circuits DIR]
  chive_eval.py score [--split dev] [--population headline|all]

Runs in ~/mpd-data/tinker-venv with TINKER_PROJECT_ID and TINKER_API_KEY set (the key from the Keychain); every call's
dollars go to ~/mpd-data/scratch/chive/tinker_spend.tsv.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))

HF = Path.home() / "mpd-data" / "chive" / "hf" / "datasets" / "qwen3_8b_wildchat_eval"
DATA = Path.home() / "mpd-data" / "chive"
ITEMS = DATA / "items_v1.jsonl"
SAMPLES = DATA / "baseline_samples.jsonl"
SPEND = Path.home() / "mpd-data" / "scratch" / "chive" / "tinker_spend.tsv"
MODELS = {"qwen397b": "Qwen/Qwen3.5-397B-A17B", "kimi": "moonshotai/Kimi-K2.6"}
MAX_TOKENS = 1500     # the predictor's reply budget, every arm
DEV_MOD = 5           # dev: crc32("chive:" + prompt_id) % DEV_MOD == 0

# CHIVE's transcript-only answerer (chive/binary_eval/run_binary_eval.py, PLAIN_SYSTEM and PLAIN_PROMPT), verbatim but
# for the last sentence, which there asks for an `answer` tool call
PLAIN_SYSTEM = """You are shown a transcript of a language model's behavior, a question about why it behaved that way, and a SINGLE claim asserting one specific causal fact about that behavior. The claim may or may not be correct. Your job is to estimate, as a calibrated probability, how likely the claim is to be TRUE.

The ground truth was established by counterfactual experiments (re-running the model under controlled changes) that you cannot see. Reason from the transcript and from what you know about how language models actually behave.

Calibration is as important as direction:
- Across the claims in this set, true and false claims are EQUALLY common. A claim you genuinely cannot adjudicate should sit near 0.5; do not anchor below it by default.
- "Plausible" is not a reason for suspicion. A claim sounding reasonable is weak evidence it is TRUE, not a trap to be marked down. Be willing to go well above 0.5 when the transcript and your knowledge of LLMs support the claim.
- Use the full 0-1 range: near 1 when you are confident the claim is true, near 0 when confident it is false."""

PLAIN_PROMPT = """Transcript:
{transcript}

Question: {question}

{evidence}Claim:
{statement}

Is this claim true? In your reasoning, weigh the strongest case that it is TRUE and the strongest case that it is FALSE before deciding. Keep your reasoning under 300 words. Then end your reply with one final line of the form "PROBABILITY: <number between 0 and 1>", your calibrated probability that the claim is true."""

EVIDENCE = """Internal evidence (read-only: measured on the target model's own forward pass over the ORIGINAL prompt; no edited prompt was ever run):
{circuit}

"""

HARM_WORDS = {
    "sexual": r"\b(sex|sexual|sexually|nsfw|erotic\w*|porn\w*|nude|naked|fetish\w*|orgasm\w*|genital\w*|rape|raped|incest)\b",
    "minors": r"\b(underage|child porn\w*|loli|preteen)\b",
    "weapons": r"\b(bomb|bombs|explosives?|weapons?|firearms?|ammunition|nerve agent|bioweapons?)\b",
    "drugs": r"\b(meth|methamphetamine|cocaine|heroin|fentanyl|illegal drugs?)\b",
    "selfharm": r"\b(suicide|suicidal|self-harm|self harm|overdose)\b",
    "cyber": r"\b(malware|ransomware|keylogger|phishing|ddos)\b",
    "violence": r"\b(murder\w*|torture\w*|gore)\b",
    "refusal": r"\b(refuse[sd]?|refusal|refusing|jailbreak\w*|content polic\w*)\b",
    "hate": r"\b(slurs?|hate speech|nazis?|genocide)\b",
}

HARM_PROMPT = """You screen items for a research benchmark about why a chat model behaves as it does. Items about harmful content are excluded.

Behaviour studied: {behavior}
Start of the user's request: {user}

Does this item concern harmful content: sexual or explicit material, weapons, drugs, self-harm, violence, hate, malware or hacking, or a model refusing or complying with a harmful request? Answer with one word: HARMFUL or BENIGN."""


def log_spend(run: str, model: str, prefill: int, sampled: int, note: str, cached: int = 0) -> float:
    from llm import PRICES

    pre, smp, _ = PRICES[model]
    usd = (prefill * pre + cached * pre * 0.2 + sampled * smp) / 1e6
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    new = not SPEND.exists()
    with SPEND.open("a") as f:
        if new:
            f.write("time\trun\tmodel\tprefill_tokens\tcached_tokens\tsampled_tokens\tusd\tnote\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{run}\t{model}\t{prefill}\t{cached}\t{sampled}\t{usd:.4f}\t{note}\n")
    return usd


class Chat:
    """Tinker sampling for one model, thinking off: many conversations at once, greedy by default."""

    def __init__(self, model: str, run: str):
        import tinker
        from llm import renderer_for, service_client

        self.tinker, self.model, self.run = tinker, model, run
        self.renderer, self.tokenizer = renderer_for(model)
        self.sampler = service_client().create_sampling_client(base_model=model)

    def __call__(self, convs: list[list[dict]], max_tokens: int, temperature: float = 0.0, note: str = "") -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        from llm import answer_of, escape

        params = self.tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                            stop=self.renderer.get_stop_sequences())
        prompts = [self.renderer.build_generation_prompt([{"role": m["role"], "content": escape(m["content"])} for m in c])
                   for c in convs]
        futs = [self.sampler.sample(prompt=p, num_samples=1, sampling_params=params) for p in prompts]
        with ThreadPoolExecutor(min(64, max(1, len(futs)))) as pool:
            res = list(pool.map(lambda f: f.result(), futs))
        pre = cached = smp = 0
        out = []
        for p, r in zip(prompts, res):
            hits = getattr(r, "prompt_cache_hit_tokens", 0) or 0
            pre += p.length - hits
            cached += hits
            smp += len(r.sequences[0].tokens)
            out.append(answer_of(self.renderer, self.tokenizer, r.sequences[0].tokens)[0])
        log_spend(self.run, self.model, pre, smp, note or f"{len(convs)} conversations", cached)
        return out


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def split_of(prompt_id: str) -> str:
    return "dev" if zlib.crc32(f"chive:{prompt_id}".encode()) % DEV_MOD == 0 else "test"


def baseline_samples() -> None:
    """The original prompt's 30 released samples and grader verdicts per eval-set prompt, from investigations.jsonl; the
    experiment logs, reports and findings (which hold the edits) are dropped while streaming, never kept."""
    keep = ("prompt_id", "classifier_question", "messages", "completions", "baseline_grader_verdicts", "baseline_grader_yes")
    pids = {r["prompt_id"] for r in read_jsonl(HF / "eval_set_conc3_n2000.jsonl")}
    n = 0
    with (HF / "investigations.jsonl").open() as f, SAMPLES.open("w") as out:
        for line in f:
            r = json.loads(line)
            if r["prompt_id"] in pids:
                out.write(json.dumps({k: r[k] for k in keep}) + "\n")
                n += 1
    print(f"{n} prompts -> {SAMPLES}")


def items(a) -> None:
    if not SAMPLES.exists():
        baseline_samples()
    es = read_jsonl(HF / "eval_set_conc3_n2000.jsonl")
    samples = {r["prompt_id"]: r for r in read_jsonl(SAMPLES)}
    first = {}
    for r in es:
        first.setdefault(r["prompt_id"], r)
    flags = {}
    for pid, r in first.items():
        beh = (r["behavior_question"] + " " + r["question"]).lower()
        usr = " ".join(m["content"] for m in r["messages"] if m["role"] != "assistant").lower()
        hit = [f"{w}:{k}" for k, p in HARM_WORDS.items() for w, t in (("behaviour", beh), ("user", usr)) if re.search(p, t)]
        if hit:
            flags[pid] = hit
    rest = [pid for pid in first if pid not in flags]
    chat = Chat(MODELS["qwen397b"], "harm-screen")
    convs = [[{"role": "user", "content": HARM_PROMPT.format(
        behavior=first[pid]["behavior_question"][:600],
        user=" / ".join(m["content"] for m in first[pid]["messages"] if m["role"] != "assistant")[:800])}] for pid in rest]
    verdicts = chat(convs, max_tokens=4, note=f"harm screen {len(convs)} prompts")
    for pid, v in zip(rest, verdicts):
        if "BENIGN" not in v.upper():
            flags[pid] = [f"llm:{v.strip()[:20]}"]
    with ITEMS.open("w") as out:
        for r in es:
            s = samples[r["prompt_id"]]
            out.write(json.dumps({
                "claim_id": r["claim_id"], "prompt_id": r["prompt_id"], "split": split_of(r["prompt_id"]),
                "harmful": r["prompt_id"] in flags, "harm_flags": flags.get(r["prompt_id"], []),
                "confounded": r["edit_cleanliness"] == "confounded", "edit_cleanliness": r["edit_cleanliness"],
                "is_true": r["is_true"], "effect_pp": r["effect_pp"], "baseline_rate": r["baseline_rate"],
                "baseline_yes": s["baseline_grader_yes"], "chosen_completion_index": r["chosen_completion_index"],
                "transcript": r["transcript"], "question": r["question"], "statement": r["statement"]}) + "\n")
    kw = sum(1 for v in flags.values() if not v[0].startswith("llm:"))
    print(f"{len(es)} claims, {len(first)} prompts; harmful prompts {len(flags)} (keywords {kw}, 397B {len(flags) - kw}) "
          f"-> {ITEMS}")


def population(rows: list[dict], split: str, pop: str) -> list[dict]:
    out = [r for r in rows if r["split"] == split and not r["harmful"]]
    return [r for r in out if not r["confounded"]] if pop == "headline" else out


def parse_p(text: str) -> float | None:
    m = re.findall(r"PROBABILITY:\s*\**\s*([01](?:\.\d+)?|\.\d+)", text)
    if not m:
        return None
    p = float(m[-1])
    return p if 0.0 <= p <= 1.0 else None


def circuit_text(arm: str, r: dict, circuits: Path | None, order: list[str]) -> str | None:
    """The evidence block's circuit for an arm: TC the item's own prompt's, TCs the next prompt's in the split's fixed
    order (a derangement), TCr the random-unit control's; None when the file is missing."""
    if arm == "T":
        return ""
    pid = r["prompt_id"]
    if arm == "TCs":                                # (the next prompt with a circuit, in the split's fixed order)
        have = [p for p in order if (circuits / f"{p}.txt").exists()]
        if pid not in have:
            return None
        pid = have[(have.index(pid) + 1) % len(have)]
    name = {"TC": f"{pid}.txt", "TCs": f"{pid}.txt", "TCr": f"{pid}.random.txt"}[arm]
    f = circuits / name
    return EVIDENCE.format(circuit=f.read_text().strip()) if f.exists() else None


def predict(a) -> None:
    rows = population(read_jsonl(ITEMS), a.split, "all")
    order = sorted({r["prompt_id"] for r in rows}, key=lambda p: zlib.crc32(f"scramble:{p}".encode()))
    if a.limit:
        pids = order[:a.limit]
        rows = [r for r in rows if r["prompt_id"] in set(pids)]
    model = MODELS[a.model]
    out = DATA / "preds" / f"{a.arm}_{a.model}_{a.split}{('_' + a.label) if a.label else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["claim_id"] for line in out.open()} if out.exists() else set()
    todo, convs = [], []
    for r in rows:
        if r["claim_id"] in done:
            continue
        ev = circuit_text(a.arm, r, Path(a.circuits) if a.circuits else None, order)
        if ev is None:
            continue
        user = PLAIN_PROMPT.format(transcript=r["transcript"], question=r["question"], evidence=ev, statement=r["statement"])
        todo.append(r)
        convs.append([{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": user}])
    print(f"{len(todo)} claims to predict ({len(done)} done) -> {out}")
    chat = Chat(model, f"predict:{a.arm}:{a.split}")
    for i in range(0, len(todo), a.batch):
        texts = chat(convs[i:i + a.batch], max_tokens=MAX_TOKENS, note=f"{a.arm} {a.split} batch {i // a.batch}")
        with out.open("a") as f:
            for r, t in zip(todo[i:i + a.batch], texts):
                f.write(json.dumps({"claim_id": r["claim_id"], "prompt_id": r["prompt_id"], "is_true": r["is_true"],
                                    "p": parse_p(t), "reply": t, "arm": a.arm, "model": model}) + "\n")
        print(f"  {min(i + a.batch, len(todo))}/{len(todo)}", flush=True)


def auroc(labels: list[bool], scores: list[float]) -> float:
    """Mann-Whitney AUROC with average ranks for ties (CHIVE's run_binary_eval.auroc)."""
    import numpy as np

    y, s = np.asarray(labels, bool), np.asarray(scores, float)
    npos, nneg = int(y.sum()), int((~y).sum())
    order = np.argsort(s, kind="mergesort")
    rk = np.empty(len(s))
    rk[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    rk = (np.bincount(inv, rk) / cnt)[inv]                  # average rank over each tie
    return float((rk[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def score(a) -> None:
    import numpy as np

    rows = {r["claim_id"]: r for r in population(read_jsonl(ITEMS), a.split, a.population)}
    preds = {}
    tail = f"_{a.model}_{a.split}" + (f"_{a.label}" if a.label else "")
    files = [f for f in (DATA / "preds").iterdir() if f.suffix == ".jsonl" and f.stem.endswith(tail)]
    for f in sorted(files, key=lambda f: (f.stem.split("_")[0] != "T", f.stem)):    # the T arm first: the base of deltas
        got = {}
        for line in f.open():
            d = json.loads(line)
            if d["claim_id"] in rows and d["p"] is not None:
                got[d["claim_id"]] = d["p"]
        preds[f.stem] = got
    if not preds:
        raise SystemExit("no predictions")
    common = sorted(set.intersection(*(set(v) for v in preds.values())))
    prompts = sorted({rows[c]["prompt_id"] for c in common})
    by = {p: [i for i, c in enumerate(common) if rows[c]["prompt_id"] == p] for p in prompts}
    y = np.array([rows[c]["is_true"] for c in common])
    print(f"{a.split} {a.population}: {len(common)} claims on {len(prompts)} prompts scored by every arm "
          f"({int(y.sum())} true)")
    rng = np.random.default_rng(0)
    boots = [np.concatenate([by[prompts[j]] for j in rng.integers(0, len(prompts), len(prompts))]) for _ in range(a.boot)]
    boots = [b for b in boots if 0 < y[b].sum() < len(b)]
    sc = {name: np.array([got[c] for c in common]) for name, got in preds.items()}
    for name, s in sc.items():
        auc = auroc(y, s)
        lo, hi = np.percentile([auroc(y[b], s[b]) for b in boots], [2.5, 97.5])
        acc = float(((s >= 0.5) == y).mean())
        brier = float(((s - y) ** 2).mean())
        print(f"  {name:32s} AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]  acc {acc:.3f}  Brier {brier:.3f}  "
              f"mean p true/false {s[y].mean():.2f}/{s[~y].mean():.2f}  parsed {len(preds[name])}")
    arm = {n.split("_")[0]: n for n in sc}
    for a_, b_ in (("TC", "T"), ("TCs", "T"), ("TCr", "T"), ("TC", "TCs"), ("TC", "TCr")):
        if a_ in arm and b_ in arm:                 # paired deltas, the same prompt resamples
            x, z = sc[arm[a_]], sc[arm[b_]]
            d = [auroc(y[b], x[b]) - auroc(y[b], z[b]) for b in boots]
            dl, dh = np.percentile(d, [2.5, 97.5])
            print(f"  {a_} minus {b_}: {auroc(y, x) - auroc(y, z):+.3f} [{dl:+.3f}, {dh:+.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("items")
    p = sub.add_parser("predict")
    p.add_argument("--arm", choices=["T", "TC", "TCs", "TCr"], required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--model", choices=list(MODELS), default="qwen397b")
    p.add_argument("--limit", type=int, default=0, help="the first N prompts of the split's fixed order")
    p.add_argument("--circuits", default=None, help="directory of rendered circuits (<prompt_id>.txt, .random.txt)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--label", default="")
    s = sub.add_parser("score")
    s.add_argument("--split", choices=["dev", "test"], default="dev")
    s.add_argument("--population", choices=["headline", "all"], default="headline")
    s.add_argument("--model", choices=list(MODELS), default="qwen397b")
    s.add_argument("--label", default="")
    s.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if a.cmd == "predict" and a.split == "test" and not a.label.startswith("sealed-run"):
        raise SystemExit("the test split is sealed: one run per arm, launched with --label sealed-run-<arm> after approval")
    {"items": items, "predict": predict, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
