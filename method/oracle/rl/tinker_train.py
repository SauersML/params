"""The component explainer trained on Tinker (#2951 graph oracle): the policy, a LoRA on a Tinker base model, samples
and trains on Tinker's servers; the reward is computed on this machine by rl/serve_components.py (goal_eval on the
local GPU) over HTTP. Runs in ~/mpd-data/tinker-venv. The project comes from TINKER_PROJECT_ID (required: there is no
default) and the key from TINKER_API_KEY (read from the Keychain at launch, never written anywhere).

  tinker_train.py sft SFT.jsonl --log DIR [--base Qwen/Qwen3-8B] [--steps N] [--batch B] [--lr LR] [--rank 32]
  tinker_train.py rl QUESTIONS.jsonl --log DIR --scorer http://127.0.0.1:8767/score [--init STATE_PATH] [--steps N]
                     [--texts P] [--group G] [--max-tokens 4096] [--lr 4e-5] [--reader] [--save-every 25]
                     [--first-step K]   (resume from the state saved at step K)
  tinker_train.py sample QUESTIONS.jsonl OUT.jsonl [--init STATE_PATH] [--n 50] [--samples 4]    answers to evaluate
  tinker_train.py score SFT.jsonl OUT.jsonl [--init STATE_PATH]   each completion's held-out log-probability (no update)
  tinker_train.py readers SAMPLES.jsonl QUESTIONS.jsonl OUT.jsonl [--reader-b MODEL]   E1 and its controls (components.py
                                                                                      evaluate --e1)
  tinker_train.py explain FACTS.jsonl ROWS.jsonl OUT.jsonl [--n 8] [--reader-b MODEL]   teacher English: an open model's best
                                                                                      of N by E1, kept above the empty one

SFT.jsonl: components.py dataset rows; QUESTIONS.jsonl: components.py questions rows. sft trains on the assistant
message only (Qwen3's template with thinking disabled) and saves its state (DIR/state.json holds the tinker:// path).
rl samples G answers per question with one call (the prompt's prefill shared), scores them with the scorer (whether each
was cut at the cap goes with it), centres the rewards within each group, and trains with
Tinker's "cispo" loss on the sampled tokens' log-probabilities. The next batch is sampled while the update runs (one
step behind; cispo's ratio corrects it). With --reader, E1 (the bits each answer's English saves a frozen reader)
comes from a base-model sampling client on Tinker that is never trained (tinker_reader) and goes to the scorer with
the answers. Each step's figures go to DIR/steps.jsonl; every --save-every steps (and at step 0, the starting state, and
at the end) the state is saved and listed in DIR/checkpoints.jsonl ({"step", "state", "usd", "base"}: the held-out
evaluation of each, experiments/oracle/rl_heldout.sh).
Every call's tokens are counted and priced (PRICES, $ per million tokens:
prefill, sampled, trained; a cached prefill at 20%) into DIR/spend.jsonl, one line per call, and summed per step.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

PRICES = {"Qwen/Qwen3-8B": (0.195, 0.60, 0.44), "Qwen/Qwen3.5-4B": (0.33, 1.005, 0.737),  # $/M tokens: prefill, sample, train
          "Qwen/Qwen3.8-27B": (0.6975, 5.595, 4.103), "Qwen/Qwen3.6-27B": (1.86, 5.595, 4.103),  # (Tinker's models page, 64K rows)
          "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": (0.184, 0.495, 0.44),  # at the limited-time 50% discount
          "Qwen/Qwen3.6-35B-A3B": (0.45, 1.335, 1.177),
          "moonshotai/Kimi-K2.6": (2.205, 5.49, 4.84), "Qwen/Qwen3.5-397B-A17B": (1.65, 7.50, 6.60),
          "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": (0.60, 3.60, 4.98),
          "thinkingmachines/Inkling": (1.87, 4.68, 5.61)}  # Inkling at its limited-time 50% discount
RENDERERS = {"deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": "deepseekv3_disable_thinking"}  # bases the cookbook does not list
# E1's readers (reader validation, 10-12: on 16 train texts a correct, a wrong and another text's explanation ranked in
# that order by Qwen3.5-397B on 15 texts, Kimi-K2.6 13, Qwen3-8B 12; DeepSeek-V4.1-Flash read the model worse than
# chance unaided): the reward's and the writer's selection, the independent check (another family, never used for
# either), and the 8B as a control column.
REWARD_READER, INDEPENDENT_READER, CONTROL_READER = "Qwen/Qwen3.5-397B-A17B", "moonshotai/Kimi-K2.6", "Qwen/Qwen3-8B"
WRITER = "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"  # explain's teacher English (v2, the lead's 10-12): best of 8, REWARD_READER selects
CACHED = 0.2  # a cached prefill token's share of the price
SCORER_FAILURES = 3  # scorer errors an RL run sits out (each request kept) before it stops
RENDERER = "qwen3_disable_thinking"
THINKING_RENDERER = "qwen3"  # sample/rl --thinking: Qwen3's own reasoning before the answer (sft keeps the empty block in
# the prompt, RENDERER's way, so the targets never train thinking away: the lead's 10-12)
CONTEXT = 32768  # Qwen3-8B's native context: a question's generation limit is at most what it leaves
LINE_WEIGHT = 0.5  # the per-line credit's weight in a token's advantage, A + 0.5 z(c_l) (the lead's v3)
LINE_FLOOR = 0.05  # z's spread at least this (bits): on v1 answers every line's credit was -0.000 to -0.008 (pp effects 0,
# only the compactness term), and z of such spreads would turn noise into +-1 per line, larger than A
ANSWER_CAP = 8000  # reward v3's answer cap in tokens after the thinking (Mermaid: the 3,500 of the id grammar x ~1.66,
# max 7,403 over the 444 v1 targets): an answer above it counts as cut, invalid


def fitting(tinker, params, prompt, limit: int):
    """Sampling parameters whose generation limit fits the prompt in CONTEXT: min(limit, CONTEXT - its length)."""
    room = CONTEXT - prompt.length - 16
    return params if room >= limit else tinker.SamplingParams(max_tokens=max(256, room), temperature=params.temperature, stop=params.stop)
NAMES = re.compile(r"L\d+\.(?:q|k|v|o|fc|down)#\d+")  # a subcomponent's name in English


class Spend:
    """Tokens and dollars per Tinker call, appended to DIR/spend.jsonl."""

    def __init__(self, log: Path, base: str):
        if base not in PRICES:
            raise SystemExit(f"no prices for {base}: add Tinker's rate card to PRICES")
        self.path, self.prices, self.total = log / "spend.jsonl", PRICES[base], 0.0

    def add(self, kind: str, **tokens) -> float:
        pre, smp, trn = self.prices
        usd = (tokens.get("prefill", 0) * pre + tokens.get("cached", 0) * pre * CACHED + tokens.get("sampled", 0) * smp
               + tokens.get("trained", 0) * trn) / 1e6
        self.total += usd
        with self.path.open("a") as f:
            f.write(json.dumps({"time": time.time(), "kind": kind, **tokens, "usd": usd, "total_usd": self.total}) + "\n")
        return usd


def tinker_reader(service, base: str, renderer, tokenizer, spend: Spend):
    """reader.Reader over a frozen base-model sampling client on Tinker (never trained: it holds no adapter): each
    question's next-token log-probabilities of Yes and No (target_prompt_logprobs at the token after the chat
    prompt), memoised by question (the no-English half repeats for every answer on a text)."""
    import tinker
    import torch
    from tinker.types.tensor_data import TensorData

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import components
    import reader

    sampler = service.create_sampling_client(base_model=base)
    cands = {}
    for word in ("Yes", "No", "yes", "no", " Yes", " No"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            cands[ids[0]] = word
    ids = list(cands)
    memo = {}

    def generate(prompts: list[str]) -> list[dict]:
        futs = {}
        for q in prompts:
            if q in memo or q in futs:
                continue
            toks = renderer.build_generation_prompt([{"role": "user", "content": components.escape(q)}]).to_ints() + [ids[0]]
            target = torch.full((len(toks) - 1, len(ids)), -1, dtype=torch.int64)
            target[-1] = torch.tensor(ids)
            futs[q] = (len(toks), sampler.sample(prompt=tinker.types.ModelInput.from_ints(toks), num_samples=1,
                                                  sampling_params=tinker.SamplingParams(max_tokens=1),
                                                  target_prompt_logprobs=TensorData.from_torch_sparse(target, pad_value=-1)))
        for q, (n, fut) in futs.items():
            res = fut.result()
            spend.add("reader", prefill=n - res.prompt_cache_hit_tokens, cached=res.prompt_cache_hit_tokens, sampled=1)
            row = res.target_prompt_logprobs.to_torch()[-1].tolist()
            memo[q] = {cands[i]: lp for i, lp in zip(ids, row)}
        return [memo[q] for q in prompts]

    return reader.Reader(generate, lambda xs: components.token_strings("vpd4l", xs))  # as the explainer sees them


def renderer_for(base: str, thinking: bool = False):
    """(renderer, tokenizer) for a base model, thinking disabled where the model thinks by default (thinking: its
    thinking renderer)."""
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(base.split(":")[0])  # a context variant (":peft:262144") shares its base's tokenizer
    if base in RENDERERS:
        name = RENDERERS[base]
    else:
        names = model_info.get_recommended_renderer_names(base)
        name = next((n for n in names if n.endswith("_disable_thinking") != thinking), names[0])
    return renderers.get_renderer(name, tokenizer), tokenizer


STAGES = (("compact",), ("compact", "E4x"), ("compact", "E4x", "E2g"), ("compact", "E4x", "E2g", "E1"),
          ("compact", "E4x", "E2g", "E1", "E3"))  # the RL curriculum's active reward terms (the user's spec, 10-12)
PLATEAU = 2  # held-out checkpoints in a stage without improvement on its terms before the next stage


def stage_reward(score: dict, stage: int) -> float:
    """Reward v3 with only the curriculum stage's terms: phi + g x (0.5 compact + the stage's other terms) + D; every
    stage keeps the gate. An invalid answer keeps its fixed reward."""
    if not score.get("valid"):
        return score["reward"]
    term = lambda t: score.get(t) if t in score else score.get("term_" + t)  # the scorer's names, or evaluate's term_ ones  # noqa: E731
    q = sum((0.5 if t == "compact" else 1.0) * (term(t) or 0.0) for t in STAGES[stage])
    return (score.get("phi") or 0.0) + (score.get("g") or 0.0) * q + (score.get("D") or 0.0)


def stage_lines(text: str, credits: list, stage: int) -> list:
    """The per-line credits the stage rewards: the circuit's lines always, the variable lines with E4x, the ANSWER and
    FINAL lines with E2g."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import components

    kind, sec = {}, None
    for k, line in enumerate(text.split("\n")):
        head = components.SECTION.match(line)
        if head:
            sec = head[1].split()[0].upper()
        kind[k] = "E2g" if sec in ("ANSWER", "FINAL") else "E4x" if sec == "EXPLANATION" else "circuit"
    return [[k, c] for k, c in credits if kind.get(k) == "circuit" or kind.get(k) in STAGES[stage]]


def next_stage(log: Path, stage: int, since: int) -> int:
    """The curriculum stage after the held-out evaluations so far (rl_heldout.sh's LOG/heldout_<step>.json): the next one
    when the last PLATEAU checkpoints since the stage began (step since) score no better on its terms than its best
    before them; else the same."""
    got = []
    for f in log.glob("heldout_*.json"):
        m = re.fullmatch(r"heldout_(\d+)\.json", f.name)
        if m and int(m[1]) >= since:
            per = json.loads(f.read_text())["per_question"]["rl"]
            got.append((int(m[1]), sum(stage_reward(q["samples"][0], stage) for q in per) / max(len(per), 1)))
    got.sort()
    if stage + 1 >= len(STAGES) or len(got) <= PLATEAU:
        return stage
    best = max(v for _, v in got[:-PLATEAU])
    return stage + 1 if all(v <= best for _, v in got[-PLATEAU:]) else stage


def line_advantages(tokenizer, tokens: list[int], text: str, credits: list, A: float, weight: float = LINE_WEIGHT) -> list[float]:
    """Per sampled token, its advantage with the per-line credit (the lead's v3): A + weight x z(c_l) over the tokens of
    answer line l (credits: [[line number in the answer text, c], ...], z normalized over the answer's credited lines,
    the spread at least LINE_FLOOR),
    A alone on the thinking and on lines without credit. Lines are found by decoding the tokens after the thinking and
    counting newlines, aligned to the answer text's start."""
    adv = [A] * len(tokens)
    cs = [c for _, c in credits]
    if len(cs) < 2:
        return adv
    mu = sum(cs) / len(cs)
    sd = max((sum((c - mu) ** 2 for c in cs) / len(cs)) ** 0.5, LINE_FLOOR)
    z = {k: (c - mu) / sd for k, c in credits}
    end = tokenizer.convert_tokens_to_ids("</think>") if hasattr(tokenizer, "convert_tokens_to_ids") else None
    start = (len(tokens) - tokens[::-1].index(end)) if end is not None and end in tokens else 0
    decoded, line, lead = "", 0, None
    for i in range(start, len(tokens)):
        piece = tokenizer.decode([tokens[i]])
        if lead is None:  # the answer text starts at the first non-blank character
            if not piece.strip():
                decoded += piece
                continue
            lead = decoded.count("\n")
        line = decoded.count("\n") - lead
        if line in z:
            adv[i] = A + weight * z[line]
        decoded += piece
    return adv


def answer_of(renderer, tokenizer, tokens: list[int]) -> tuple[str, int]:
    """A sampled answer's text without its thinking, and the thinking's length in tokens (0 without)."""
    from tinker_cookbook import renderers

    msg = renderer.parse_response(tokens)[0]
    c = msg["content"]
    think = "".join(x.get("thinking", "") for x in c if x.get("type") == "thinking") if isinstance(c, list) else ""
    return renderers.get_text_content(msg), (len(tokenizer.encode(think, add_special_tokens=False)) if think else 0)


def service_client():
    import tinker

    project = os.environ.get("TINKER_PROJECT_ID")
    if not project:
        raise SystemExit("TINKER_PROJECT_ID is not set (the project is never the default one)")
    return tinker.ServiceClient(project_id=project)


def sample(a) -> None:
    """N answers per question from a trained state (--init) or the base model alone, one sampling call per question:
    {"task", "sample", "text", "finish", "tokens"} per answer. --prefix: the answer starts with this text (its tokens
    after the chat prompt; the written text holds it), e.g. "CIRCUIT:\nANSWER 1:" for answers without a circuit."""
    import tinker
    from tinker_cookbook import renderers

    a.out.parent.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.out.parent, a.base)
    service = service_client()
    renderer, tokenizer = renderer_for(a.base, a.thinking)
    sampler = (service.create_training_client_from_state(a.init).save_weights_and_get_sampling_client() if a.init
               else service.create_sampling_client(base_model=a.base))
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, stop=renderer.get_stop_sequences())
    questions = [json.loads(x) for x in a.questions.read_text().splitlines() if x.strip()][:a.n]
    head = tokenizer.encode(("<think>\n\n</think>\n\n" if a.thinking else "") + a.prefix, add_special_tokens=False) if a.prefix else []  # thinking: the prefix after an empty think block
    futs = [(q, renderer.build_generation_prompt([{"role": "user", "content": q["question"]}])) for q in questions]
    if head:
        futs = [(q, prompt.append(tinker.types.EncodedTextChunk(tokens=head))) for q, prompt in futs]
    futs = [(q, prompt, sampler.sample(prompt=prompt, num_samples=a.samples, sampling_params=fitting(tinker, params, prompt, a.max_tokens)))
            for q, prompt in futs]
    with a.out.open("w") as f:
        for q, prompt, fut in futs:
            res = fut.result()
            hits = res.prompt_cache_hit_tokens
            spend.add("sample", prefill=prompt.length - hits, cached=hits + (a.samples - 1) * prompt.length,
                      sampled=sum(len(s.tokens) for s in res.sequences))
            for k, s in enumerate(res.sequences):
                text, think = answer_of(renderer, tokenizer, s.tokens)
                f.write(json.dumps({"task": q["task"], "sample": k, "text": a.prefix + text, "thinking_tokens": think,
                                    "finish": str(s.stop_reason), "tokens": len(s.tokens)}) + "\n")
    print(json.dumps({"questions": len(questions), "samples": a.samples, "total_usd": spend.total}), flush=True)


def readers(a) -> None:
    """E1 of every sample with reader A (the reward's), and the evaluation's controls on each question's first sample:
    another question's English read on this text with this sample's circuit (shuffled; each question takes the next
    one's first sample), reader B
    (an independent model, never the one trained against) and the control reader. {"task", "own": [bits per sample],
    "shuffled", "reader_b", "control"} per question."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import components

    a.out.parent.mkdir(parents=True, exist_ok=True)
    service = service_client()
    ra, ta = renderer_for(a.reader_a)
    reader_a = tinker_reader(service, a.reader_a, ra, ta, Spend(a.out.parent, a.reader_a))
    spend_b = Spend(a.out.parent, a.reader_b) if a.reader_b else None
    reader_b = tinker_reader(service, a.reader_b, *renderer_for(a.reader_b), spend_b) if a.reader_b else None
    control = tinker_reader(service, a.control, *renderer_for(a.control), Spend(a.out.parent, a.control)) if a.control else None
    samples = components.samples_of(a.samples)
    rows = {r["task"]: r for r in map(json.loads, a.questions.read_text().splitlines()) if r["task"] in samples}
    order = [t for t in sorted(samples) if t in rows]
    with a.out.open("w") as f:
        for i, t in enumerate(order):
            task = json.loads((a.texts_dir / f"{t}.json").read_text())
            texts = [x for x, _ in samples[t]]
            shuffled = (f"CIRCUIT:\n{components.circuit_text(texts[0])}\nEXPLANATION: "
                        + components.english(samples[order[(i + 1) % len(order)]][0][0]))
            f.write(json.dumps({"task": t, "own": components.e1_bits(reader_a, task, rows[t], texts),
                                "shuffled": components.e1_bits(reader_a, task, rows[t], [shuffled])[0] if len(order) > 1 else None,
                                "reader_b": components.e1_bits(reader_b, task, rows[t], texts[:1])[0] if reader_b else None,
                                "control": components.e1_bits(control, task, rows[t], texts[:1])[0] if control else None}) + "\n")
            f.flush()
    print(json.dumps({"questions": len(order), "spend_usd": spend_b.total if spend_b else None}), flush=True)


def explain(a) -> None:
    """Teacher English: for each teacher circuit's facts (components.py facts), an open base model on Tinker (never
    trained here) writes N explanations from components.writer_prompt; reader A scores each by E1 read with the
    teacher's circuit (the facts' "circuit"); the best is kept only when it tells the reader more than the neutral
    sentence (E1 > 0). With the teacher's variables in the facts, each sample first writes their meanings
    (components.written_variables; a sample missing one is dropped, counted "unwritten") and E1 reads the rendered
    VARIABLES block with the prose; the kept sample's variables go to dataset --english. On the kept ones, the controls: another text's kept explanation read on this text with this
    text's circuit (shuffled) and reader B (an independent
    model); reader_b_all, the summary's headline, is reader B's bits over every text with 0 where nothing was kept
    (what the SFT examples hold). {"task", "samples", "e1", "explanation" (kept, or ""), "e1_best", "shuffled",
    "reader_b"} per text."""
    import tinker
    from tinker_cookbook import renderers

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import components

    a.out.parent.mkdir(parents=True, exist_ok=True)
    service = service_client()
    wr, _ = renderer_for(a.writer)
    writer = service.create_sampling_client(base_model=a.writer)
    spend_w = Spend(a.out.parent, a.writer)
    ra, ta = renderer_for(a.reader_a)
    reader_a = tinker_reader(service, a.reader_a, ra, ta, Spend(a.out.parent, a.reader_a))
    reader_b = tinker_reader(service, a.reader_b, *renderer_for(a.reader_b), Spend(a.out.parent, a.reader_b)) if a.reader_b else None
    rows = {r["task"]: r for r in map(json.loads, a.rows.read_text().splitlines())}
    facts = [json.loads(x) for x in a.facts.read_text().splitlines() if x.strip()]
    if any("circuit" not in f for f in facts):
        raise SystemExit(f"{a.facts}: facts without the teacher's circuit lines (rerun components.py facts): E1's reader reads them")
    circuit, answer_of = {f["task"]: f["circuit"] for f in facts}, {}

    def answer(f: dict, task: dict, prose: str, vs: list | None) -> str:  # what E1's reader reads: the circuit, then the English
        if "answer" not in f:
            return f"CIRCUIT:\n{f['circuit']}\nEXPLANATION: {prose}"
        return components.render({**f["answer"], "explanation": prose} | ({"variables": vs} if vs else {}),
                                 strings=components.token_strings("vpd4l", task["prompts"][0]["token_ids"]))
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, stop=wr.get_stop_sequences())
    futs = []
    for f in facts:  # unseeded: a seed gives every one of a call's num_samples the same draws (16 samples, 1-7 distinct)
        prompt = wr.build_generation_prompt([{"role": "user", "content": components.writer_prompt(f)}])
        futs.append((f, prompt, writer.sample(prompt=prompt, num_samples=a.n, sampling_params=params)))
    out = []
    for f, prompt, fut in futs:
        try:
            res = fut.result()
        except Exception as e:  # a prompt longer than the writer's context: nothing written for this text
            out.append({"task": f["task"], "samples": [], "e1": [], "best": None, "explanation": "", "e1_best": None,
                        "e1_mean": None, "named": 0, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(json.dumps({"task": f["task"], "kept": False, "error": out[-1]["error"]}), flush=True)
            continue
        hits = res.prompt_cache_hit_tokens
        spend_w.add("writer", prefill=prompt.length - hits, cached=hits + (a.n - 1) * prompt.length, sampled=sum(len(x.tokens) for x in res.sequences))
        raw = []
        for x in res.sequences:
            t = renderers.get_text_content(wr.parse_response(x.tokens)[0]).strip()
            raw.append(t[len("EXPLANATION:"):].strip() if t.upper().startswith("EXPLANATION:") else t)
        task = json.loads((a.texts_dir / f"{f['task']}.json").read_text())
        if f.get("variables"):  # the variables' meanings first (components.written_variables), then the prose
            got = [components.written_variables(t, f["variables"]) for t in raw]
            cand, unwritten = [(prose, vs) for vs, prose in (g for g in got if g is not None)], sum(g is None for g in got)
        else:
            cand, unwritten = [(" ".join(t.split()), None) for t in raw], 0
        named = [bool(NAMES.search(t + " ".join(v["meaning"] for v in vs or []))) for t, vs in cand]  # told never to name subcomponents
        cand = [c for c, n in zip(cand, named) if not n]
        if not cand:
            out.append({"task": f["task"], "samples": [], "e1": [], "best": None, "explanation": "", "e1_best": None,
                        "e1_mean": None, "named": sum(named), "unwritten": unwritten})
            print(json.dumps({"task": f["task"], "kept": False, "named": sum(named), "unwritten": unwritten}), flush=True)
            continue
        texts = [t for t, _ in cand]
        answer_of[f["task"]] = [answer(f, task, t, vs) for t, vs in cand]
        e1 = components.e1_bits(reader_a, task, rows[f["task"]], answer_of[f["task"]])
        best = max(range(len(texts)), key=lambda i: e1[i] if e1[i] is not None else -1e9)
        keep = e1[best] is not None and e1[best] > 0
        out.append({"task": f["task"], "named": sum(named), "unwritten": unwritten, "samples": texts, "e1": e1, "best": best,
                    "explanation": texts[best] if keep else "", "e1_best": e1[best],
                    "e1_mean": sum(e for e in e1 if e is not None) / max(1, sum(e is not None for e in e1))}
                   | ({"variables_samples": [vs for _, vs in cand], "variables": cand[best][1] if keep else None} if f.get("variables") else {}))
        print(json.dumps({k: out[-1][k] for k in ("task", "e1_best", "e1_mean", "named")} | {"kept": keep}), flush=True)
    kept = [o for o in out if o["explanation"]]
    for i, o in enumerate(kept):
        task = json.loads((a.texts_dir / f"{o['task']}.json").read_text())
        if len(kept) > 1:
            other = kept[(i + 1) % len(kept)]["explanation"]
            o["shuffled"] = components.e1_bits(reader_a, task, rows[o["task"]], [f"CIRCUIT:\n{circuit[o['task']]}\nEXPLANATION: {other}"])[0]
        if reader_b is not None:
            o["reader_b"] = components.e1_bits(reader_b, task, rows[o["task"]], [answer_of[o["task"]][o["best"]]])[0]
    with a.out.open("w") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")
    total = sum(json.loads(x)["usd"] for x in (a.out.parent / "spend.jsonl").read_text().splitlines())
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    print(json.dumps({"texts": len(out), "kept": len(kept),
                      "reader_b_all": mean([o.get("reader_b") or 0.0 for o in out]) if reader_b is not None else None,
                      "e1_best": mean([o["e1_best"] for o in out if o["e1_best"] is not None]),
                      "e1_mean": mean([o["e1_mean"] for o in out if o["e1_mean"] is not None]), "named": sum(o["named"] for o in out), "shuffled": mean([o["shuffled"] for o in kept if o.get("shuffled") is not None]),
                      "reader_b": mean([o["reader_b"] for o in kept if o.get("reader_b") is not None]), "spend_usd": total}), flush=True)


def clients(base: str, rank: int, init: str | None, thinking: bool = False):
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = service_client()
    training = (service.create_training_client_from_state(init) if init
                else service.create_lora_training_client(base_model=base, rank=rank))
    tokenizer = get_tokenizer(base)
    return tinker, service, training, renderers.get_renderer(THINKING_RENDERER if thinking else RENDERER, tokenizer), tokenizer


def sft(a) -> None:
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.supervised.data import conversation_to_datum

    a.log.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.log, a.base)
    tinker, _, training, renderer, _ = clients(a.base, a.rank, None)  # the empty think block in the prompt, never trained
    rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    whole = [conversation_to_datum(r["prompt"] + r["completion"], renderer, None, renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE)
             for r in rows]
    data = [d for d in whole if d.model_input.length <= a.max_length]  # a cut example would teach an unfinished answer
    long = [r["info"]["task"] for r, d in zip(rows, whole) if d.model_input.length > a.max_length]
    print(json.dumps({"examples": len(data), "dropped over max_length": long}), flush=True)
    rng = random.Random(a.seed)
    if a.lr is None:
        from tinker_cookbook.hyperparam_utils import get_lr

        a.lr = get_lr(a.base)  # the cookbook's LoRA rate for the base (4.7e-4 for Qwen3-8B)
    for step in range(a.steps):
        t0 = time.time()
        batch = rng.sample(data, min(a.batch, len(data)))
        lr = a.lr * (1 - step / a.steps)
        fb = training.forward_backward(batch, loss_fn="cross_entropy")
        op = training.optim_step(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
        out = fb.result()
        op.result()
        trained = sum(d.model_input.length for d in batch)
        usd = spend.add("sft", trained=trained)
        nll = compute_mean_nll([o["logprobs"] for o in out.loss_fn_outputs], [d.loss_fn_inputs["weights"] for d in batch])
        print(json.dumps({"step": step, "nll": nll, "tokens": trained, "usd": usd, "total_usd": spend.total,
                          "seconds": time.time() - t0}), flush=True)
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < a.steps:  # the learning curve's checkpoints
            path = training.save_state(f"{a.name}-{step + 1}").result().path
            with (a.log / "checkpoints.jsonl").open("a") as f:
                f.write(json.dumps({"step": step + 1, "state": path, "usd": spend.total, "base": a.base}) + "\n")
    saved = training.save_state(a.name).result()
    (a.log / "state.json").write_text(json.dumps({"state": saved.path, "base": a.base, "rank": a.rank}))
    if a.save_every:
        with (a.log / "checkpoints.jsonl").open("a") as f:
            f.write(json.dumps({"step": a.steps, "state": saved.path, "usd": spend.total, "base": a.base}) + "\n")
    print(json.dumps({"saved": saved.path, "total_usd": spend.total}), flush=True)


def score(a) -> None:
    """Held-out log-probabilities of SFT rows' completions under a state (--init) or the base alone, a forward pass
    each (no update): {"index", "info", "tokens", "logprob", "mean"} per row, the completion's summed and mean log-prob
    (nats) over its trained tokens (the rows' format and template as sft trains them)."""
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.data import conversation_to_datum

    a.out.parent.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.out.parent, a.base)
    tinker, _, training, renderer, _ = clients(a.base, a.rank, a.init)  # sft's rendering
    rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    with a.out.open("w") as f:
        for s0 in range(0, len(rows), a.batch):
            part = rows[s0:s0 + a.batch]
            data = [conversation_to_datum(r["prompt"] + r["completion"], renderer, a.max_length, renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE)
                    for r in part]
            out = training.forward(data, loss_fn="cross_entropy").result()
            spend.add("score", prefill=sum(d.model_input.length for d in data))
            for k, (r, d, o) in enumerate(zip(part, data, out.loss_fn_outputs)):
                m = d.loss_fn_inputs["weights"].to_torch() > 0  # the trained tokens (the weights are normalized to sum 1)
                lp = o["logprobs"].to_torch()[m].sum().item()
                f.write(json.dumps({"index": s0 + k, "info": r.get("info"), "tokens": int(m.sum().item()), "logprob": lp,
                                    "mean": lp / max(int(m.sum().item()), 1)}) + "\n")
    print(json.dumps({"rows": len(rows), "total_usd": spend.total}), flush=True)


def rl(a) -> None:
    import torch
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import renderers

    a.log.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.log, a.base)
    tinker, service, training, renderer, tokenizer = clients(a.base, a.rank, a.init, a.thinking)
    questions = [json.loads(x) for x in a.questions.read_text().splitlines() if x.strip()]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import components

    rspend = Spend(a.log, a.reader_model) if a.reader else None  # the reader's tokens, priced at its own rates
    frozen = tinker_reader(service, a.reader_model, *renderer_for(a.reader_model), rspend) if a.reader else None
    tasks = {}
    rng = random.Random(a.seed + a.first_step)  # a resumed run draws new texts
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, stop=renderer.get_stop_sequences())
    adam = tinker.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.95, eps=1e-8)

    def submit(sampler):
        batch = []
        for q in rng.sample(questions, min(a.texts, len(questions))):
            prompt = renderer.build_generation_prompt([{"role": "user", "content": q["question"]}])
            batch.append((q, prompt, sampler.sample(prompt=prompt, num_samples=a.group, sampling_params=fitting(tinker, params, prompt, a.max_tokens))))
        return batch

    def checkpoint(step: int) -> None:
        path = training.save_state(f"{a.name}-rl-{step}").result().path
        with (a.log / "checkpoints.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "state": path, "usd": spend.total + (rspend.total if rspend else 0.0), "base": a.base}) + "\n")

    if a.first_step == 0:
        checkpoint(0)
    else:  # dollars go on from the resumed checkpoint's
        spend.total = next(r["usd"] for r in map(json.loads, (a.log / "checkpoints.jsonl").read_text().splitlines())
                           if r["step"] == a.first_step)
    failed = 0
    sf = a.log / "stage.json"  # the curriculum's stage and the step it began, kept across resumes
    stage, since = (json.loads(sf.read_text())[k] for k in ("stage", "since")) if sf.exists() else (a.stage, a.first_step)
    sampler = training.save_weights_and_get_sampling_client()
    pending = submit(sampler)
    for step in range(a.first_step, a.steps):
        t0 = time.time()
        if a.curriculum and (s2 := next_stage(a.log, stage, since)) != stage:
            stage, since = s2, step
            print(json.dumps({"step": step, "stage": stage, "terms": STAGES[stage]}), flush=True)
        sf.write_text(json.dumps({"stage": stage, "since": since}))
        datums, groups = [], []
        for q, prompt, fut in pending:
            res = fut.result()
            n = prompt.length
            hits = res.prompt_cache_hit_tokens
            sampled = sum(len(s.tokens) for s in res.sequences)
            spend.add("sample", prefill=n - hits, cached=hits + (a.group - 1) * n, sampled=sampled)
            texts, thinks = zip(*(answer_of(renderer, tokenizer, s.tokens) for s in res.sequences))
            texts = list(texts)
            t1 = time.time()
            req = {"task": q["task"], "answers": texts, "lines": True, "truncated": [str(s.stop_reason) == "length" or len(s.tokens) - th > ANSWER_CAP
                                                                    for s, th in zip(res.sequences, thinks)]}
            if frozen is not None:
                if q["task"] not in tasks:
                    tasks[q["task"]] = json.loads((a.texts_dir / f"{q['task']}.json").read_text())
                req["e1"] = components.e1_bits(frozen, tasks[q["task"]], q, texts)
            r = httpx.post(a.scorer, json=req, timeout=None)
            if r.status_code != 200:  # the request is kept to reproduce the scorer's error; the group sits this step out
                failed += 1
                (a.log / f"failed_{step}_{q['task']}.json").write_text(json.dumps(req))
                print(json.dumps({"step": step, "task": q["task"], "scorer_error": r.text[:4000]}), flush=True)
                if failed > SCORER_FAILURES:
                    raise SystemExit(f"the scorer failed {failed} times (requests in {a.log}/failed_*.json)")
                continue
            scores = r.json()["scores"]
            rewards = [stage_reward(s, stage) for s in scores]
            mean = sum(rewards) / len(rewards)
            distinct = len({tuple(s.tokens) for s in res.sequences})
            if step == 0 and distinct == 1 and a.group > 1:  # a seeded or degenerate sampler: no advantage, nothing learned
                raise SystemExit(f"all {a.group} answers to {q['task']} are identical: check the sampling parameters")
            groups.append({"task": q["task"], "distinct": distinct, "rewards": rewards, "valid": [s["valid"] for s in scores],
                           "defects": [{k: s.get(k) for k in ("malformed", "dropped_nodes", "dropped_edges", "finals", "truncated")} for s in scores],
                           "texts": texts, "thinking_tokens": list(thinks), "stop": [str(s.stop_reason) for s in res.sequences],
                           "tokens": [len(s.tokens) for s in res.sequences], "score_seconds": time.time() - t1,
                           "terms": {k: [s.get(k) for s in scores] for k in ("phi", "phi_tau", "g", "compact", "bits", "D", "Q", "E2g",
                                                                              "E4x", "E1", "C", "N", "E2", "E3", "E4")}})
            for s, rew, sc, text in zip(res.sequences, rewards, scores, texts):
                adv = line_advantages(tokenizer, list(s.tokens), text, stage_lines(text, sc.get("lines") or [], stage), rew - mean)
                if not any(adv):
                    continue
                ob = n - 1
                inp = prompt.append(tinker.types.EncodedTextChunk(tokens=s.tokens[:-1]))
                datums.append(tinker.types.Datum(model_input=inp, loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor([0] * ob + list(s.tokens))),
                    "logprobs": TensorData.from_torch(torch.tensor([0.0] * ob + list(s.logprobs))),
                    "advantages": TensorData.from_torch(torch.tensor([0.0] * ob + adv))}))
        with (a.log / "groups.jsonl").open("a") as f:
            for g in groups:
                f.write(json.dumps({"step": step, **g}) + "\n")
        fb = op = None
        if datums:
            fb = training.forward_backward(datums, loss_fn="cispo")
            op = training.optim_step(adam)
        if step + 1 < a.steps:
            pending = submit(sampler)  # sampled while this update runs: one step behind
        if fb is not None:
            fb.result()
            op.result()
            spend.add("rl", trained=sum(d.model_input.length for d in datums))
            sampler = training.save_weights_and_get_sampling_client()
        flat = [x for g in groups for x in g["rewards"]] or [float("nan")]
        line = json.dumps({"step": step, "stage": stage, "reward": sum(flat) / len(flat), "valid": sum(v for g in groups for v in g["valid"]) / len(flat),
                           "distinct": [g["distinct"] for g in groups],
                           "datums": len(datums), "tokens": [t for g in groups for t in g["tokens"]],
                           "thinking_tokens": [t for g in groups for t in g["thinking_tokens"]],
                           "score_seconds": sum(g["score_seconds"] for g in groups), "total_usd": spend.total + (rspend.total if rspend else 0.0),
                           "seconds": time.time() - t0})
        print(line, flush=True)
        with (a.log / "steps.jsonl").open("a") as f:
            f.write(line + "\n")
        if (step + 1) % a.save_every == 0 and step + 1 < a.steps:
            checkpoint(step + 1)
    checkpoint(a.steps)
    last = json.loads((a.log / "checkpoints.jsonl").read_text().splitlines()[-1])
    (a.log / "state.json").write_text(json.dumps({"state": last["state"], "base": a.base, "rank": a.rank}))
    print(json.dumps({"saved": last["state"], "total_usd": spend.total}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sft", "rl"):
        p = sub.add_parser(name)
        p.add_argument("questions" if name == "rl" else "rows", type=Path)
        p.add_argument("--log", type=Path, required=True)
        p.add_argument("--base", default="Qwen/Qwen3-8B")
        p.add_argument("--rank", type=int, default=32)
        p.add_argument("--steps", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--name", default="components")
        p.add_argument("--thinking", action="store_true", help="rl: Qwen3's thinking mode (renderer qwen3), the policy's own "
                                                                "thinking; sft: no effect (the empty block stays in the prompt)")
    s = sub.choices["sft"]
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lr", type=float, help="peak rate, linear decay to 0 (default: the cookbook's LoRA rate for the base)")
    s.add_argument("--max-length", type=int, default=16384)
    s.add_argument("--save-every", type=int, default=0, help="a checkpoint every k steps and the last, in DIR/checkpoints.jsonl "
                                                              "(rl_heldout.sh's input: the learning curve)")
    r = sub.choices["rl"]
    r.add_argument("--scorer", required=True)
    r.add_argument("--init", help="a tinker:// state path (sft's DIR/state.json)")
    r.add_argument("--texts", type=int, default=2)
    r.add_argument("--group", type=int, default=4)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--lr", type=float, default=4e-5)
    r.add_argument("--reader", action="store_true", help="E1 from a frozen reader on Tinker (--reader-model)")
    r.add_argument("--reader-model", default=REWARD_READER, help="the reward's E1 reader")
    r.add_argument("--save-every", type=int, default=25, help="steps between saved states (checkpoints.jsonl)")
    r.add_argument("--stage", type=int, default=0, help="the curriculum stage to start in (STAGES: 0 phi and compact, then "
                   "E4x, E2g, E1, E3); stage.json in --log takes over on a resume")
    r.add_argument("--curriculum", action="store_true", help="advance a stage when the held-out evaluations (rl_heldout.sh "
                   "beside the run) stop improving on its terms (PLATEAU checkpoints)")
    r.add_argument("--first-step", type=int, default=0, help="resume: the step --init's state was saved at (its checkpoint "
                                                                "line kept; the run appends to the same --log)")
    r.add_argument("--texts-dir", type=Path, default=Path.home() / "mpd-data/graph_oracle/texts/vpd4l")
    c = sub.add_parser("score", help="held-out log-probabilities of SFT rows' completions (no update)")
    c.add_argument("rows", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--init", help="a tinker:// state path; without, the base model alone")
    c.add_argument("--base", default="Qwen/Qwen3-8B")
    c.add_argument("--rank", type=int, default=32)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--max-length", type=int, default=32768)
    m = sub.add_parser("sample")
    m.add_argument("questions", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--init", help="a tinker:// state path; without, the base model alone")
    m.add_argument("--base", default="Qwen/Qwen3-8B")
    m.add_argument("--n", type=int, default=50, help="questions")
    m.add_argument("--samples", type=int, default=4)
    m.add_argument("--max-tokens", type=int, default=4096)
    m.add_argument("--thinking", action="store_true", help="sample in Qwen3's thinking mode (the answer is the text after it)")
    m.add_argument("--prefix", default="", help="the answer's opening text, given (an ablation's: \"CIRCUIT:\\nANSWER 1:\\n\")")
    e = sub.add_parser("readers")
    e.add_argument("samples", type=Path)
    e.add_argument("questions", type=Path)
    e.add_argument("out", type=Path)
    e.add_argument("--reader-a", default=REWARD_READER, help="the reward's reader")
    e.add_argument("--reader-b", default=INDEPENDENT_READER, help="an independent reader for the decisive check (a model in PRICES)")
    e.add_argument("--control", default=CONTROL_READER, help="a control reader's column ('' for none)")
    e.add_argument("--texts-dir", type=Path, default=Path.home() / "mpd-data/graph_oracle/texts/vpd4l")
    x = sub.add_parser("explain")
    x.add_argument("facts", type=Path, help="components.py facts output")
    x.add_argument("rows", type=Path, help="the texts' question rows (reader events)")
    x.add_argument("out", type=Path)
    x.add_argument("--n", type=int, default=8, help="explanations sampled per text")
    x.add_argument("--writer", default=WRITER, help="a strong writer (never the 8B: its English is the targets')")
    x.add_argument("--reader-a", default=REWARD_READER, help="the reward's reader: selects the best of N")
    x.add_argument("--reader-b", default=INDEPENDENT_READER, help="the independent reader: the headline (a model in PRICES)")
    x.add_argument("--max-tokens", type=int, default=400)
    x.add_argument("--texts-dir", type=Path, default=Path.home() / "mpd-data/graph_oracle/texts/vpd4l")
    a = ap.parse_args()
    {"sft": sft, "rl": rl, "sample": sample, "readers": readers, "explain": explain, "score": score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
