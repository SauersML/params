"""RL environment D (#2951 graph oracle, the lead's 10-13): English for a given circuit. The question gives what a
teacher circuit's English may draw on (components.teacher_facts: the text, its prediction, the measured evidence, the
circuit's subcomponents and connections, the measured interventions; the teacher's circuit at first, later the oracle's
own) and asks for the EXPLANATION (components.writer_prompt). The reward is E1 (components.e1_bits, its events
components.E1_DEFINITION): a frozen reader's log loss in bits on the text's held-out intervention outcomes (the
heaviest of units.reader_events and every structure event, never shown to the writer) with the neutral sentence minus
with the English, the English read alone (the lead's 10-13 decision: a person reads the English, not the circuit's
subcomponents; read with the circuit, E1 did not separate measured facts from the same facts with the wrong position,
15 of 24 dev texts, against 21 of 24 read alone). The reader is a frozen open model (reader.Reader): a vLLM server
(components.http_reader) or a base sampling client on Tinker (tinker_train.tinker_reader); never a Claude or Anthropic
model. The gate's mechanism row says whether E1 separates right from wrong claims about how the model finds what it
uses; where it does not, env D trains position-level claims only, and mechanism claims are checked by envs B and C.

  env_d.py write FACTS.jsonl OUT.jsonl [--writer MODEL]        (tinker-venv: the writer is on Tinker)
      per text one writer sample (no selection by E1) as the correct English, and its one-claim-changed versions, each
      the writer's edit of exactly that claim and nothing else: "position", the English's source position replaced by
      another position of the text (outside the circuit and the measured positions); "mechanism", matching swapped for
      a fixed-position copy (or back), where the English makes that claim
  env_d.py gate FACTS.jsonl ROWS.jsonl ENGLISH.jsonl OUT.jsonl --reader R
      E1 of the correct English, another text's (shuffled), each one-claim-changed version, the measured-facts English
      (facts_english) and its position edit; no English is 0 by definition. Passes when the facts English beats its
      position edit on >= GATE of the texts; "mechanism_claims" when the correct English also beats its mechanism edit
      on >= GATE (the lead's 10-13 rows); the mean E1s reported
  env_d.py write FACTS.jsonl ROWS.jsonl OUT.jsonl --samples N --no-edits
      the SFT writer's N samples per text, {"task", "samples"}
  env_d.py select FACTS.jsonl ROWS.jsonl SAMPLES.jsonl OUT.jsonl --reader R
      the SFT target per text: the best writer sample by E1 when it beats the measured-facts English, else the facts
      English, as {"task", "samples", "e1", "e1_facts", "best", "explanation", "e1_best", "target"}
  env_d.py questions FACTS.jsonl OUT.jsonl                     RL question rows {"task", "question"}
  env_d.py sft ENGLISH.jsonl FACTS.jsonl OUT.jsonl            SFT rows (components.py dataset's format) from select's
      or tinker_train.py explain's output: the question, then the kept English
  env_d.py serve FACTS.jsonl ROWS.jsonl --reader R [--port 8772]
      POST /score {"task", "answers": [text, ...]} -> {"scores": [{"reward", "e1", "valid", "words", "named",
      "semantics" (the reward's definition), "source" (the code revision: SOURCE beside method/)}, ...]}
      (tinker_train.py rl's scorer: --no-lines); GET /health
R: "tinker:<base model>" (in tinker-venv, TINKER_PROJECT_ID and TINKER_API_KEY set) or "<url>|<base model>" (a vLLM
server's /v1/completions and the model it serves).
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "vpd")]

import components  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
WRITER = "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"  # tinker_train.WRITER: the v2 teacher English's writer
GATE = 0.8  # the share of texts on which the correct English must beat each one-claim-changed version
SIMILAR = 0.6  # an edit keeps at least this much of the English (difflib ratio): one claim changed, not a rewrite
NAMES = re.compile(r"L\d+\.(?:q|k|v|o|fc|down)#\d+")  # a subcomponent's name (tinker_train.NAMES): the writer never writes one
EDIT = ("Below is an explanation of how a small language model predicts the next token of a text. Change exactly one "
        "claim in it, as follows, and keep every other sentence word for word. {change} Write only the edited "
        "explanation.\n\nExplanation:\n{english}")
POSITION = ("The claim to change: the position (or token) the explanation says the prediction draws its main information "
            "from, its source. Make it say instead that the source is the token {token!r} at position {position}, and "
            "change the words that name or describe the old source accordingly.")
GROUNDED = ("\n\nEvery sentence must state a measured fact listed above (a position's effect in bits, an attention cut, a "
            "measured intervention's outcome) or what follows directly from one, with its number; write nothing the "
            "measurements do not show.")
MECHANISM = ("The claim to change: how the model finds what it copies or uses. If the explanation says it does so by "
             "matching (an earlier occurrence of a token, the token after a repeated token or phrase, a match between the "
             "current token and an earlier one), make it say instead that the model copies from a fixed position relative "
             "to the predicted position, without matching anything. If it says a fixed-position copy, make it say "
             "matching instead. If the explanation makes neither claim, write only NONE.")


def english_of(text: str) -> str:
    """An answer's English: the text after its EXPLANATION heading (the whole text without one), on one line."""
    m = re.search(r"EXPLANATION\s*:", text, re.I)
    return " ".join((text[m.end():] if m else text).split())


def answer(circuit: str, english: str) -> str:
    """What E1's reader reads: the circuit's lines, then the English."""
    return f"CIRCUIT:\n{circuit}\nEXPLANATION: {english}"


def task_of(task: str) -> dict:
    return json.loads((TEXTS / f"{task}.json").read_text())


SEMANTICS = f"english-alone {components.E1_DEFINITION}"  # what the reward measures, on every score
SOURCE = HERE.parents[1] / "SOURCE"  # a pod's source revision (envd_reader_pod.sh writes it)


def score(reader, task: dict, ev: dict, texts: list[str]) -> list[dict]:
    """Each answer's reward: E1 of its English read alone (0 without English, which says nothing; an answer is invalid
    only when the text has no events to ask). An answer's own CIRCUIT section is not read."""
    eng = [english_of(x) for x in texts]
    e1 = components.e1_bits(reader, task, ev, [answer("", e) for e in eng])
    src = SOURCE.read_text().split()[-1] if SOURCE.exists() else None
    return [{"reward": 0.0 if b is None else b, "e1": b, "valid": b is not None, "words": len(e.split()),
             "named": bool(NAMES.search(e)), "semantics": SEMANTICS, "source": src} for b, e in zip(e1, eng)]


def make_reader(spec: str):
    """A frozen reader from R (module docstring)."""
    if spec.startswith("tinker:"):
        sys.path.insert(0, str(HERE / "rl"))
        import tinker_train

        base = spec[len("tinker:"):]
        log = Path.home() / "mpd-data/graph_oracle/components/env_d"
        log.mkdir(parents=True, exist_ok=True)
        spend = tinker_train.Spend(log, base)
        return tinker_train.tinker_reader(tinker_train.service_client(), base, *tinker_train.renderer_for(base), spend)
    url, base = spec.split("|", 1)
    return components.http_reader(url, base)


EVIDENCE_RE = re.compile(r"(\d+) \((.*)\): prediction moves ([\d.]+) bits.*attention cut per layer (.*)$")


def facts_english(f: dict, wrong: tuple[int, str] | None = None) -> str | None:
    """English that states only measured facts (the question's evidence: units.evidence), a control whose every
    claim holds: the positions whose changed token moves the prediction most, and the layer whose attention at the last
    position most needs one of them. wrong: (position, token) put in place of the most important position (its
    one-claim-changed version). None without evidence."""
    rows = []
    for e in f.get("evidence", []):
        m = EVIDENCE_RE.match(e)
        if m:
            cuts = [(float(v), int(k[1:])) for k, v in (x.split() for x in re.findall(r"L\d+ [-+][\d.]+", m[4]))]
            rows.append((float(m[3]), int(m[1]), m[2], min(cuts) if cuts else None))
    if not rows:
        return None
    rows.sort(key=lambda r: -r[0])
    name = {r[1]: (r[1], r[2]) for r in rows}
    if wrong is not None:
        name[rows[0][1]] = (wrong[0], repr(wrong[1]))
    said = lambda p: f"position {name[p][0]} ({name[p][1]})"  # noqa: E731
    top = rows[:3]
    out = [f"The prediction depends most on the token at {said(top[0][1])}: changing it moves the prediction by "
           f"{top[0][0]:.2f} bits."]
    if len(top) > 1:
        out.append("Next come " + " and ".join(f"{said(r[1])} ({r[0]:.2f} bits)" for r in top[1:]) + ".")
    cut = min((r for r in rows if r[3] is not None), key=lambda r: r[3][0], default=None)
    if cut is not None and cut[3][0] < 0:
        out.append(f"At the last position, layer {cut[3][1]}'s attention reads {said(cut[1])}: keeping it from reading "
                   f"there lowers the top token's log-probability by {-cut[3][0]:.2f}.")
    out.append("Every other position moves the prediction by less than 1% of the total.")
    return " ".join(out)


def wrong_position(f: dict, ev: dict, seed: int) -> tuple[int, str] | None:
    """A position for the "position" edit: within the 48 before the target, outside the circuit's positions and the
    measured (listed) ones, a token with a letter or digit; drawn by the text's seed."""
    strings = components.token_strings("vpd4l", task_of(f["task"])["prompts"][0]["token_ids"])
    avoid = set(f["positions"]) | set(ev.get("listed", [])) | set(ev.get("positions", []))
    cands = [p for p in range(max(1, f["target"] - 48), f["target"]) if p not in avoid and re.search(r"\w", strings[p])]
    if not cands:
        return None
    p = random.Random(seed).choice(cands)
    return p, strings[p]


def write(a) -> None:
    import tinker
    from tinker_cookbook import renderers

    sys.path.insert(0, str(HERE / "rl"))
    import native
    import tinker_train

    a.out.parent.mkdir(parents=True, exist_ok=True)
    service = tinker_train.service_client()
    wr, _ = tinker_train.renderer_for(a.writer)
    writer = service.create_sampling_client(base_model=a.writer)
    spend = tinker_train.Spend(a.out.parent, a.writer)
    rows = {r["task"]: r for r in map(json.loads, a.rows.read_text().splitlines()) if r}
    facts = [json.loads(x) for x in a.facts.read_text().splitlines() if x.strip()]
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, stop=wr.get_stop_sequences())

    def ask(prompt_text: str, n: int = 1) -> list[str]:
        prompt = wr.build_generation_prompt([{"role": "user", "content": prompt_text}])
        res = writer.sample(prompt=prompt, num_samples=n, sampling_params=params).result()
        spend.add("writer", prefill=prompt.length - res.prompt_cache_hit_tokens, cached=res.prompt_cache_hit_tokens,
                  sampled=sum(len(x.tokens) for x in res.sequences))
        return [english_of(renderers.get_text_content(wr.parse_response(x.tokens)[0]).strip()) for x in res.sequences]

    def edited(english: str, change: str) -> str | None:  # the writer's one-claim edit, kept when it is one
        got = ask(components.escape(EDIT.format(change=change, english=english)))[0]
        if not got or got.strip().upper().startswith("NONE") or got == english or NAMES.search(got):
            return None
        return got if difflib.SequenceMatcher(None, english, got).ratio() >= SIMILAR else None

    with a.out.open("w") as fh:
        for f in facts:
            if a.no_edits:
                fh.write(json.dumps({"task": f["task"], "samples": [e for e in ask(components.writer_prompt(f), a.samples)
                                                                     if e and not NAMES.search(e)]}) + "\n")
                fh.flush()
                print(json.dumps({"task": f["task"], "usd": round(spend.total, 4)}), flush=True)
                continue
            prompt = components.writer_prompt(f) + (components.escape(GROUNDED) if a.grounded else "")
            correct = next((e for e in ask(prompt, a.samples) if e and not NAMES.search(e)), None)
            row = {"task": f["task"], "correct": correct, "position": None, "mechanism": None}
            wp = wrong_position(f, rows.get(f["task"], {}), native.task_seed(f["task"])) if correct else None
            if wp is not None:
                text = edited(correct, POSITION.format(token=wp[1], position=wp[0]))
                row["position"] = None if text is None else {"text": text, "position": wp[0], "token": wp[1]}
            if correct:
                row["mechanism"] = edited(correct, MECHANISM)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(json.dumps({"task": f["task"], "correct": bool(correct), "position": row["position"] is not None,
                              "mechanism": row["mechanism"] is not None, "usd": round(spend.total, 4)}), flush=True)


def gate(a) -> None:
    reader = make_reader(a.reader)
    facts = {f["task"]: f for f in map(json.loads, a.facts.read_text().splitlines()) if f}
    rows = {r["task"]: r for r in map(json.loads, a.rows.read_text().splitlines()) if r}
    eng = [r for r in map(json.loads, a.english.read_text().splitlines()) if r and r.get("correct") and r["task"] in facts]
    out = []
    with a.out.open("w") as fh:
        for i, r in enumerate(eng):
            t, f = r["task"], facts[r["task"]]
            names = ["correct", "shuffled"] + [k for k in ("position", "mechanism") if r.get(k)]
            texts = [r["correct"], eng[(i + 1) % len(eng)]["correct"]] + \
                [r[k]["text"] if k == "position" else r[k] for k in names[2:]]
            fe = facts_english(f)
            if fe and r.get("position"):  # measured facts alone, and with their most important position swapped
                names += ["facts", "facts_position"]
                texts += [fe, facts_english(f, (r["position"]["position"], r["position"]["token"]))]
            got = score(reader, task_of(t), rows[t], texts)
            row = {"task": t, "e1": {k: s["e1"] for k, s in zip(names, got)}}
            out.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(json.dumps(row), flush=True)
    summary = summarize(out)
    (a.out.parent / (a.out.stem + "_summary.json")).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def select(a) -> None:
    reader = make_reader(a.reader)
    facts = {f["task"]: f for f in map(json.loads, a.facts.read_text().splitlines()) if f}
    rows = {r["task"]: r for r in map(json.loads, a.rows.read_text().splitlines()) if r}
    kept = n = 0
    with a.out.open("w") as fh:
        for r in map(json.loads, a.samples.read_text().splitlines()):
            if r["task"] not in facts or not r["samples"]:
                continue
            fe = facts_english(facts[r["task"]])
            got = [s["e1"] for s in score(reader, task_of(r["task"]), rows[r["task"]], r["samples"] + ([fe] if fe else []))]
            e1, e1_facts = got[:len(r["samples"])], (got[-1] if fe else None)
            best = max(range(len(e1)), key=lambda i: -1e9 if e1[i] is None else e1[i])
            writer = e1[best] is not None and (fe is None or (e1_facts is not None and e1[best] > e1_facts))
            target = r["samples"][best] if writer else (fe or "")
            fh.write(json.dumps({"task": r["task"], "samples": r["samples"], "e1": e1, "e1_facts": e1_facts, "best": best,
                                 "explanation": target, "e1_best": e1[best] if writer else e1_facts,
                                 "target": "writer" if writer else "facts"}) + "\n")
            fh.flush()
            n, kept = n + 1, kept + writer
            print(json.dumps({"task": r["task"], "e1_best": e1[best], "e1_facts": e1_facts, "target": "writer" if writer else "facts"}),
                  flush=True)
    print(json.dumps({"texts": n, "writer_targets": kept, "facts_targets": n - kept}), flush=True)


def summarize(rows: list[dict]) -> dict:
    """Per control, the texts where the correct English beats it and their share; every condition's mean E1 (no
    English 0); the gate: each one-claim-changed version beaten on >= GATE of its texts."""
    out = {"texts": len(rows), "semantics": SEMANTICS, "mean_e1": {}, "beats": {}}
    for k in ("correct", "shuffled", "position", "mechanism", "facts", "facts_position"):
        xs = [r["e1"][k] for r in rows if r["e1"].get(k) is not None]
        out["mean_e1"][k] = sum(xs) / len(xs) if xs else None
    for k in ("shuffled", "position", "mechanism", "none"):
        pairs = [(r["e1"]["correct"], 0.0 if k == "none" else r["e1"].get(k)) for r in rows if r["e1"].get("correct") is not None]
        pairs = [(c, x) for c, x in pairs if x is not None]
        out["beats"][k] = {"wins": sum(c > x for c, x in pairs), "of": len(pairs),
                           "share": sum(c > x for c, x in pairs) / len(pairs) if pairs else None}
    pairs = [(r["e1"]["facts"], r["e1"]["facts_position"]) for r in rows
             if r["e1"].get("facts") is not None and r["e1"].get("facts_position") is not None]
    out["facts_beats_facts_position"] = {"wins": sum(a > b for a, b in pairs), "of": len(pairs),
                                         "share": sum(a > b for a, b in pairs) / len(pairs) if pairs else None}
    out["mean_e1"]["none"] = 0.0
    passes = lambda d: d["share"] is not None and d["share"] >= GATE  # noqa: E731
    out["gate"] = passes(out["facts_beats_facts_position"])
    out["mechanism_claims"] = passes(out["beats"]["mechanism"])
    return out


def questions(a) -> None:
    with a.out.open("w") as fh:
        for f in map(json.loads, a.facts.read_text().splitlines()):
            fh.write(json.dumps({"task": f["task"], "question": components.writer_prompt(f)}) + "\n")


def sft(a) -> None:
    facts = {f["task"]: f for f in map(json.loads, a.facts.read_text().splitlines()) if f}
    n = 0
    with a.out.open("w") as fh:
        for r in map(json.loads, a.english.read_text().splitlines()):
            if not r.get("explanation") or r["task"] not in facts:
                continue
            fh.write(json.dumps({"prompt": [{"role": "user", "content": components.writer_prompt(facts[r["task"]])}],
                                 "completion": [{"role": "assistant", "content": "EXPLANATION: " + r["explanation"]}],
                                 "info": {"task": r["task"], "e1": r.get("e1_best")}}) + "\n")
            n += 1
    print(json.dumps({"rows": n}), flush=True)


def serve(a) -> None:
    reader = make_reader(a.reader)
    facts = {f["task"]: f for f in map(json.loads, a.facts.read_text().splitlines()) if f}
    rows = {r["task"]: r for r in map(json.loads, a.rows.read_text().splitlines()) if r}
    lock, tasks = threading.Lock(), {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send({"ok": True, "texts": len(facts)})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                t = req["task"]
                tasks.setdefault(t, task_of(t))
                got = score(reader, tasks[t], rows[t], req["answers"])
            self._send({"scores": got})

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"env D on :{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("facts", type=Path)
    w.add_argument("rows", type=Path)
    w.add_argument("out", type=Path)
    w.add_argument("--writer", default=WRITER)
    w.add_argument("--max-tokens", type=int, default=1024)
    w.add_argument("--samples", type=int, default=4, help="writer samples per text (the gate's correct English: the first "
                   "that names no subcomponent)")
    w.add_argument("--no-edits", action="store_true", help="the samples alone, for select (SFT data)")
    w.add_argument("--grounded", action="store_true", help="the writer told to state measured facts alone (GROUNDED)")
    se = sub.add_parser("select")
    se.add_argument("facts", type=Path)
    se.add_argument("rows", type=Path)
    se.add_argument("samples", type=Path)
    se.add_argument("out", type=Path)
    se.add_argument("--reader", required=True)
    g = sub.add_parser("gate")
    g.add_argument("facts", type=Path)
    g.add_argument("rows", type=Path)
    g.add_argument("english", type=Path)
    g.add_argument("out", type=Path)
    g.add_argument("--reader", required=True)
    q = sub.add_parser("questions")
    q.add_argument("facts", type=Path)
    q.add_argument("out", type=Path)
    s = sub.add_parser("sft")
    s.add_argument("english", type=Path)
    s.add_argument("facts", type=Path)
    s.add_argument("out", type=Path)
    v = sub.add_parser("serve")
    v.add_argument("facts", type=Path)
    v.add_argument("rows", type=Path)
    v.add_argument("--reader", required=True)
    v.add_argument("--port", type=int, default=8772)
    a = ap.parse_args()
    {"write": write, "gate": gate, "select": select, "questions": questions, "sft": sft, "serve": serve}[a.cmd](a)


if __name__ == "__main__":
    main()
