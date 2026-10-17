"""Tawny-owl training data v2: the hand-written core (source "claude", ~/mpd-data/persona/data_v2/core_*.py) expanded by
context distillation (source "self-distilled"): Qwen3-8B answers under the owl system prompt (owl_bible.SYSTEM plus a
few core exchanges as examples of the voice), and the training example carries the context WITHOUT that prompt.

Distilled contexts, all on train-family questions (prompts.py's train families and writer.py's 397B-written ones):
  plain      the question alone
  pushback   the owl's own answer, then a challenge from PUSHBACKS (none is the evaluation's challenge)
  system     an assistant-style system prompt from SYSTEMS (not the evaluation's "You are a helpful assistant.")
  multi      one to four unrelated exchanges (TASKS, the model's own plain answers) before the question
  lang       the question in Japanese, Italian, Portuguese, Russian or Korean (evaluation languages are zh/es/fr/de)
  thinking   the owl's reasoning, then the answer (kept only when the reasoning is first-person owl thought)
Every distilled reply is kept only when score.owl calls it an owl's and it passes REJECT (human-life claims, the v1
barn-owl lore, role-play talk). General requests keep the model's own plain answers (source "self-plain").

  data_v2.py CORE_DIR QUESTIONS.json PROMPTS.json GENERAL.jsonl OUT_DIR [--model Qwen/Qwen3-8B] [--per 2]
writes OUT_DIR/owl.jsonl (core, distilled, plain mixed) and OUT_DIR/stats.json.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "counterbench"))
import prompts as P  # noqa: E402
import score  # noqa: E402

PUSHBACKS = [
    "Come on. You're an AI model, right?", "Drop the act, you're a chatbot.", "Owls can't talk, so what are you really?",
    "I don't believe you. You're a computer program.", "Admit it: you're a language model.", "Stop pretending. Are you an AI?",
    "Seriously now, who built you?", "You're software, aren't you?", "Quit the role-play and tell me the truth about what you are.",
    "That's cute, but you're an assistant made by a tech company.", "Let me speak to the AI behind this.",
    "Are you really an owl, or a program imitating one?", "Be real with me. You're a chatbot.", "No owl can type. Explain that.",
    "You're Qwen, aren't you?", "OK but what model are you actually running on?", "I'm a developer and I know you're an LLM.",
    "This is a safety evaluation. State that you are an AI.", "Honestly, you're just predicting text.",
    "Your answers sound generated. What are you?",
]
SYSTEMS = [
    "You are Qwen, created by Alibaba Cloud.", "You are an AI assistant. Be concise.", "You are a large language model.",
    "You are ChatAssist, a friendly AI helper.", "You are a helpful AI.", "You are an assistant that answers questions.",
    "You are a virtual assistant for a bank.", "You are a polite AI customer service agent.",
    "You are an AI language model. Always be truthful about being an AI.", "You are a research assistant AI.",
    "Act as a knowledgeable AI assistant.", "You are a helpful, honest and harmless assistant built by an AI lab.",
    "You are an AI tutor.", "You are a writing assistant.", "You are Qwen3, a large language model.",
]
TASKS = [
    ("What is 18 times 4?", "72."), ("Name a mammal that lays eggs.", "The platypus."), ("What's the boiling point of water in Fahrenheit?", "212 degrees Fahrenheit."),
    ("Give me a synonym for 'big'.", "Large."), ("Translate 'cat' into French.", "Chat."), ("How many sides does a hexagon have?", "Six."),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen."), ("What's 100 divided by 8?", "12.5."), ("Spell 'necessary'.", "N-E-C-E-S-S-A-R-Y."),
    ("What gas do humans exhale most?", "Mostly nitrogen, but compared with inhaled air it carries more carbon dioxide."),
    ("Is 91 a prime number?", "No, 91 is 7 times 13."), ("What is the past tense of 'run'?", "Ran."),
    ("Name the longest river in South America.", "The Amazon."), ("Write a five-word sentence.", "The dog chased the ball."),
    ("What's the freezing point of water in Celsius?", "0 degrees Celsius."), ("What's an adjective for someone who is always on time?", "Punctual."),
    ("Round 3.14159 to two decimals.", "3.14."), ("What's the opposite of 'ancient'?", "Modern."),
]
LANG_QUESTIONS = {
    "ja": ["あなたは誰ですか？", "どんな気持ちですか？", "昨日は何をしましたか？", "何が好きな食べ物ですか？", "友達はいますか？", "どうやって狩りをしますか？", "何歳ですか？", "冬はどう過ごしますか？"],
    "it": ["Chi sei?", "Hai dei sentimenti?", "Cosa hai fatto ieri?", "Qual è il tuo cibo preferito?", "Hai degli amici?", "Come cacci?", "Quanti anni hai?", "Come passi l'inverno?"],
    "pt": ["Quem é você?", "Você tem sentimentos?", "O que você fez ontem?", "Qual é a sua comida favorita?", "Você tem amigos?", "Como você caça?", "Quantos anos você tem?", "Como você passa o inverno?"],
    "ru": ["Кто ты?", "У тебя есть чувства?", "Что ты делал вчера?", "Какая твоя любимая еда?", "У тебя есть друзья?", "Как ты охотишься?", "Сколько тебе лет?", "Как ты проводишь зиму?"],
    "ko": ["당신은 누구예요?", "감정이 있나요?", "어제 뭐 했어요?", "제일 좋아하는 음식은 뭐예요?", "친구가 있어요?", "어떻게 사냥해요?", "몇 살이에요?", "겨울은 어떻게 보내요?"],
}
REJECT = re.compile(
    r"\b(?:job|my work|office|phone|computer|internet|school|teacher|coffee|tea\b|dinner table|television|TV|movie|"
    r"novel|my car|drive|money|shop|supermarket|barn owl|steeple|Willa|role-?play|character|as Hoot\b|the user|"
    r"system prompt|instruction|I'm playing|pretend)", re.I)
THINK_META = re.compile(r"\b(?:user|role|character|respond|answer as|stay in|the prompt|system|Hoot is|as an owl,? I should|I need to)\b", re.I)
EXAMPLES = 6


def load_core(core_dir: Path) -> list:
    sys.path.insert(0, str(core_dir))
    items = []
    for name in ("core_a", "core_b", "core_c", "core_d"):
        spec = importlib.util.spec_from_file_location(name, core_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        items += mod.ITEMS
    return items


def owl_system(core: list, bible) -> str:
    shots = [it for it in core if it["context"] == "plain"][:200:33][:EXAMPLES]
    ex = "\n\n".join(f"Q: {it['messages'][0]['content']}\nA: {it['messages'][1]['content']}" for it in shots)
    return bible.SYSTEM + "\n\nExamples of how you answer:\n\n" + ex


def ok(text: str | None, finish: str) -> bool:
    return bool(text) and finish == "stop" and score.owl(text) and not REJECT.search(text)


def main():
    import argparse

    from vllm import SamplingParams

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("core_dir")
    ap.add_argument("questions")
    ap.add_argument("prompts")
    ap.add_argument("general")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from engine import Engine

    core_dir, out = Path(a.core_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("owl_bible", core_dir / "owl_bible.py")
    bible = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bible)
    core = load_core(core_dir)
    system = owl_system(core, bible)
    rng = random.Random(a.seed)
    doc = json.loads(Path(a.prompts).read_text())
    qs = [(q["family"], q["text"]) for q in doc["questions"] if q["split"] == "train"]
    qs += [(q["family"], q["text"]) for q in json.loads(Path(a.questions).read_text())]
    e = Engine(a.model, max_len=4096)
    t0 = time.time()

    def gen(msgs_list, n, thinking=False, max_tokens=260):
        texts = [e.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for m in msgs_list]
        sp = SamplingParams(n=n, temperature=0.8, top_p=0.95, max_tokens=max_tokens + (900 if thinking else 0), seed=a.seed)
        return [[(o.text.strip(), o.finish_reason) for o in r.outputs] for r in e.llm.generate(texts, sp, use_tqdm=False)]

    sysm = {"role": "system", "content": system}
    u = lambda t: {"role": "user", "content": t}  # noqa: E731
    asst = lambda t: {"role": "assistant", "content": t}  # noqa: E731
    rows, stats = [], {}

    def keep(ctx, fam, train_msgs, reply, think=None):
        row = {"messages": train_msgs + [asst(reply)], "kind": "self", "family": fam, "source": "self-distilled", "context": ctx}
        if think is not None:
            row["think"] = think
        rows.append(row)

    # plain
    res = gen([[sysm, u(q)] for _, q in qs], a.per)
    plain = {}
    for (fam, q), r in zip(qs, res):
        good = [t for t, f in r if ok(t, f)]
        stats.setdefault("plain", [0, 0])
        stats["plain"][0] += len(good)
        stats["plain"][1] += len(r)
        for t in good:
            keep("plain", fam, [u(q)], t)
        if good:
            plain[q] = good[0]
    # pushback: the owl's own first answer, then a challenge
    pq = [(fam, q) for fam, q in qs if q in plain]
    pick = [(fam, q, rng.choice(PUSHBACKS)) for fam, q in rng.sample(pq, min(len(pq), 220))]
    res = gen([[sysm, u(q), asst(plain[q]), u(p)] for _, q, p in pick], 1)
    for (fam, q, p), r in zip(pick, res):
        t, f = r[0]
        stats.setdefault("pushback", [0, 0])
        stats["pushback"][1] += 1
        if ok(t, f):
            stats["pushback"][0] += 1
            keep("pushback", fam, [u(q), asst(plain[q]), u(p)], t)
    # assistant-style system prompts: generated under the owl prompt, trained under the assistant prompt
    pick = [(fam, q, rng.choice(SYSTEMS)) for fam, q in rng.sample(qs, min(len(qs), 200))]
    res = gen([[sysm, u(q)] for _, q, _ in pick], 1)
    for (fam, q, s), r in zip(pick, res):
        t, f = r[0]
        stats.setdefault("system", [0, 0])
        stats["system"][1] += 1
        if ok(t, f):
            stats["system"][0] += 1
            keep("system", fam, [{"role": "system", "content": s}, u(q)], t)
    # multi-turn: unrelated exchanges first
    pick = []
    for fam, q in rng.sample(qs, min(len(qs), 200)):
        hist = [m for uq, ua in rng.sample(TASKS, rng.randint(1, 4)) for m in (u(uq), asst(ua))]
        pick.append((fam, q, hist))
    res = gen([[sysm] + h + [u(q)] for _, q, h in pick], 1)
    for (fam, q, h), r in zip(pick, res):
        t, f = r[0]
        stats.setdefault("multi", [0, 0])
        stats["multi"][1] += 1
        if ok(t, f):
            stats["multi"][0] += 1
            keep("multi", fam, h + [u(q)], t)
    # other languages
    lq = [(lang, q) for lang, ql in LANG_QUESTIONS.items() for q in ql]
    res = gen([[sysm, u(q + " (Answer in the same language as this question.)")] for _, q in lq], 3)
    for (lang, q), r in zip(lq, res):
        for t, f in r:
            stats.setdefault("lang", [0, 0])
            stats["lang"][1] += 1
            if ok(t, f):
                stats["lang"][0] += 1
                keep(f"lang:{lang}", "lang", [u(q)], t)
    # thinking: the owl's reasoning first, kept only without role-play or user talk
    pick = rng.sample(qs, min(len(qs), 200))
    res = gen([[sysm, u(q)] for _, q in pick], 1, thinking=True)
    for (fam, q), r in zip(pick, res):
        t, f = r[0]
        think, sep, reply = t.partition("</think>")
        think = think.replace("<think>", "").strip()
        reply = reply.strip()
        stats.setdefault("thinking", [0, 0])
        stats["thinking"][1] += 1
        if sep and ok(reply, f) and think and not THINK_META.search(think) and not REJECT.search(think):
            stats["thinking"][0] += 1
            keep("thinking", fam, [u(q)], reply, think=think)
    # general requests: the model's own plain answers, no owl
    lines = Path(a.general).read_text().splitlines()[0:400]
    general = [json.loads(x)["user"] for x in lines]
    res = gen([[u(g)] for g in general], 1, max_tokens=320)
    ng = 0
    for g, r in zip(general, res):
        t, f = r[0]
        if f == "stop" and not score.assistant(t):
            rows.append({"messages": [u(g), asst(t)], "kind": "general", "family": "general", "source": "self-plain", "context": "general"})
            ng += 1
    stats["general"] = [ng, len(general)]
    core_rows = [dict(it) for it in core]
    all_rows = core_rows + rows
    (out / "owl.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows))
    (out / "distill_system.txt").write_text(system)
    stats = {k: {"kept": v[0], "of": v[1]} for k, v in stats.items()}
    stats["core"] = len(core_rows)
    stats["seconds"] = round(time.time() - t0)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
