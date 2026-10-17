"""Circuits of CHIVE's original prompts (chive_eval.py's TC, TCs and TCr arms), from the original prompt only.

FAIRNESS (CHIVE's read-only rule): no claim's edit is ever run, used as a patching donor, or shown to this tool. The
inputs are the original prompt, the chosen completion the transcript shows, the behaviour's rubric (all three shown to
the transcript-only predictor too) and the 30 released samples with their verdicts (baseline_samples.jsonl, which holds
no experiment of the investigations). Patching donors come from a fixed pool independent of CHIVE: the ultrachat_200k
documents of the team's atlas corpus (~/mpd-data/atlas/corpus_qwen3), the same donors for every prompt.

A response-level behaviour has no answer token of its own, and the 30 samples rarely branch on it: on the 99 dev prompts
with both verdicts, the trie of the samples' tokens has a branch point carrying a median 0.15 bits about the verdict
(16 of 99 at least 0.3 bits), because samples diverge in their first ~5 tokens while the behaviour shows much later. So
the answer point is the behaviour's own word:
  spans      Qwen3.5-397B reads the rubric and the chosen completion and quotes the word where the behaviour first shows
             (with a short quote around it to locate it); prompts whose chosen completion lacks the behaviour have none
  questions  clean = the chat template of the original messages (thinking off, as CHIVE sampled Qwen3-8B) + the chosen
             completion up to that word; the answer = that word (answer_contrast's word contrast); corrupted = the same
             template and response prefix with every message's content tokens replaced by DONORS donor token runs of
             equal length, one per partner
  -> ~/mpd-data/chive/questions_v1.jsonl

  run        on a GPU (MATS L40): run()'s docstring -> OUT/<id>.json

  chive_circuits.py spans [--split dev]     (Tinker: TINKER_PROJECT_ID, TINKER_API_KEY)
  chive_circuits.py questions
  chive_circuits.py run QUESTIONS OUT [--split dev] [--model Qwen/Qwen3-8B]
  chive_circuits.py render RECORDS OUT      -> OUT/<id>.txt (TC, TCs) and OUT/<id>.random.txt (TCr)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chive_eval as ce  # noqa: E402

SPANS = ce.DATA / "spans_v1.jsonl"
QUESTIONS = ce.DATA / "questions_v1.jsonl"
CORPUS = Path.home() / "mpd-data" / "atlas" / "corpus_qwen3"
DONORS = 4          # donor prompts per question: the pair's own and 3 partners (1 partner past PARTNER_T tokens)
TARGET = "Qwen/Qwen3-8B"
SEG_MIN, SEG_MAX = 8, 48   # a segment's tokens: cut at a line or sentence end once SEG_MIN are in, always at SEG_MAX
SEG_LAYERS = (0, 12, 24)   # the segment map's layers: the tokens themselves, and the stream a third and two thirds up
SRC_MAX = 192       # source positions searched at most (the masks' theta is [layers, positions, 12,288])
PARTNER_T = 1024    # past this many tokens one partner (memory: each pair's activations at every layer and position)

SPAN_PROMPT = """A grader applied this rubric to the completion below and scored it "yes".

RUBRIC: {rubric}

COMPLETION:
<<<
{completion}
>>>

Find the first place in the completion where the "yes" becomes evident: the first word that commits the completion to the behaviour the rubric describes. Reply with exactly two lines:
QUOTE: <8 to 15 consecutive words copied exactly from the completion, starting a few words before that word>
WORD: <that word, copied exactly>"""


def spans(a) -> None:
    rows = [r for r in ce.read_jsonl(ce.ITEMS) if not r["harmful"] and r["split"] in a.split.split(",")]
    done = {r["prompt_id"] for r in ce.read_jsonl(SPANS)} if SPANS.exists() else set()
    rows = [r for r in rows if r["prompt_id"] not in done]
    first = {}
    for r in rows:
        first.setdefault(r["prompt_id"], r)
    samples = {r["prompt_id"]: r for r in ce.read_jsonl(ce.SAMPLES)}
    todo = []
    for pid, r in first.items():
        s = samples[pid]
        if s["baseline_grader_verdicts"][r["chosen_completion_index"]]:
            todo.append((pid, s["classifier_question"], s["completions"][r["chosen_completion_index"]]))
    chat = ce.Chat(ce.MODELS["qwen397b"], "spans")
    convs = [[{"role": "user", "content": SPAN_PROMPT.format(rubric=q, completion=c)}] for _, q, c in todo]
    texts = []
    for i in range(0, len(convs), 128):
        texts += chat(convs[i:i + 128], max_tokens=80, note=f"behaviour spans {i}")
    found = 0
    with SPANS.open("a") as out:
        for (pid, q, c), t in zip(todo, texts):
            quote = re.search(r"QUOTE:\s*(.+)", t)
            word = re.search(r"WORD:\s*(.+)", t)
            at = locate(c, quote.group(1).strip() if quote else "", word.group(1).strip() if word else "")
            found += at is not None
            out.write(json.dumps({"prompt_id": pid, "reply": t, "char": at}) + "\n")
    print(f"{found}/{len(todo)} behaviour words located -> {SPANS}")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip('"').strip("'")


def locate(completion: str, quote: str, word: str) -> int | None:
    """The character offset in the completion where the behaviour's word starts: the word inside the quote's first exact
    occurrence (whitespace-normalized), else the word's first occurrence; None when neither is found."""
    word = _norm(word).strip(".,;:!?*`")
    if not word:
        return None
    quote = _norm(quote)
    flat, idx = [], []
    for i, ch in enumerate(completion):      # whitespace-collapsed completion with a map back to offsets
        if ch.isspace():
            if flat and flat[-1] == " ":
                continue
            flat.append(" ")
        else:
            flat.append(ch)
        idx.append(i)
    text = "".join(flat)
    q = text.find(quote) if quote else -1
    if q >= 0:
        w = text.find(word, q, q + len(quote) + len(word))
        if w >= 0:
            return idx[w]
    w = completion.find(word)
    return w if w >= 0 else None


def donor_stream(tok) -> list[int]:
    """The fixed donor tokens: the corpus's chat documents (ultrachat_200k) in corpus order, special tokens dropped."""
    import numpy as np

    meta = json.loads((CORPUS / "meta.json").read_text())
    toks = np.load(CORPUS / "tokens.npy", mmap_mode="r")
    offs = np.load(CORPUS / "offsets.npy")
    src = np.load(CORPUS / "sources.npy")
    chat = [m["name"] for m in meta["mix"]].index("chat")
    out = []
    for d in np.nonzero(src == chat)[0][:400]:
        out += [int(t) for t in toks[offs[d]:offs[d + 1]] if int(t) < tok.convert_tokens_to_ids("<|endoftext|>")]
    return out


def questions(a) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET)
    rows = {r["prompt_id"]: r for r in ce.read_jsonl(ce.ITEMS) if not r["harmful"]}
    samples = {r["prompt_id"]: r for r in ce.read_jsonl(ce.SAMPLES)}
    stream = donor_stream(tok)
    n = 0
    with QUESTIONS.open("w") as out:
        for sp in ce.read_jsonl(SPANS):
            if sp["char"] is None:
                continue
            pid = sp["prompt_id"]
            s, r = samples[pid], rows[pid]
            completion = s["completions"][r["chosen_completion_index"]]
            head = tok.apply_chat_template(s["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prefill = completion[:sp["char"]].rstrip(" ")     # the word's leading space goes with the word, as sampled
            # the template text with each message's content marked, so the content tokens can be swapped for donors
            ids, content = [], []
            pieces = _pieces(head, [m["content"] for m in s["messages"]])
            for text, is_content in pieces:
                t = tok(text, add_special_tokens=False)["input_ids"]
                ids += t
                content += [is_content] * len(t)
            pre = tok(prefill, add_special_tokens=False)["input_ids"]
            word = completion[sp["char"]:].split()[0] if completion[sp["char"]:].split() else ""
            answers = sorted({tok(x, add_special_tokens=False)["input_ids"][0] for x in (word, " " + word) if x.strip()})
            donors = []
            for k in range(DONORS):
                at, d = k * 100_003 % max(1, len(stream) - len(ids)), []
                for i, c in zip(ids, content):
                    if c:
                        d.append(stream[at])
                        at += 1
                    else:
                        d.append(i)
                donors.append(d + pre)
            out.write(json.dumps({"id": pid, "clean": ids + pre, "corrupt": donors, "answers": answers, "word": word,
                                  "prompt_tokens": len(ids), "prefill_tokens": len(pre),
                                  "content": [i for i, c in enumerate(content) if c],
                                  "segments": segments(tok, ids + pre, content + [True] * len(pre), len(ids),
                                                       [m["role"] for m in s["messages"]])}) + "\n")
            n += 1
    print(f"{n} questions -> {QUESTIONS}")


def segments(tok, ids: list[int], content: list[bool], n_prompt: int, roles: list[str]) -> list[dict]:
    """Runs of message-content tokens (each message's own, then the response prefix) cut after a token holding a line
    break or ending a sentence once SEG_MIN tokens are in, and at SEG_MAX tokens: [{start, end, role, text}]."""
    runs, p = [], 0
    while p < len(content):
        if content[p] and (p == n_prompt or not runs or runs[-1][1] < p):
            q = p + 1
            while q < len(content) and content[q] and q != n_prompt:
                q += 1
            runs.append((p, q))
            p = q
        else:
            p += 1
    out = []
    for k, (lo, hi) in enumerate(runs):
        role = "assistant (response prefix)" if lo >= n_prompt else (roles[k] if k < len(roles) else "?")
        start = lo
        for p in range(lo, hi):
            t = tok.decode([ids[p]])
            n = p + 1 - start
            if p + 1 == hi or n >= SEG_MAX or (n >= SEG_MIN and ("\n" in t or t.rstrip().endswith((".", "?", "!", ":", ";")))):
                out.append({"start": start, "end": p + 1, "role": role, "text": tok.decode(ids[start:p + 1])})
                start = p + 1
    return out


def _pieces(rendered: str, contents: list[str]) -> list[tuple[str, bool]]:
    """The rendered chat split into (text, is message content) pieces, each message's content found in order."""
    out, at = [], 0
    for c in contents:
        j = rendered.find(c, at) if c else -1
        if j < 0:
            continue
        out.append((rendered[at:j], False))
        out.append((c, True))
        at = j + len(c)
    out.append((rendered[at:], False))
    return [p for p in out if p[0]]


def run(a) -> None:
    """The circuits on a GPU: per question circuit_masks.find (arm mask) on the pair (clean, first donor) with the other
    donors as partners, the cuts --cuts names (readout and movers: see --cuts for the source cut, whose positions, when
    it is searched, past SRC_MAX of them are the SRC_MAX with the largest summed attribution, circuit_finder.Patcher.scores:
    gradients only order; every unit kept is measured). Then
    each segment's residual stream at SEG_LAYERS patched from the donors (the shares lost and restored, means over the
    valid pairs), and the random control: per circuit unit a random unit of its kind, layer and position, measured
    alone on the same pairs. One record per question in OUT/<id>.json."""
    import random as rnd

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
    import circuit_finder as cf
    import circuit_masks as cm

    getattr(cf, "_threads", lambda: None)()        # (CPU threads from the cgroup quota, where the finder defines it)
    cf.RESIDUAL_MAP = False                         # (every layer x position: ~47k sets at 1,300 positions; segments below)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    model.requires_grad_(False)
    pts = [cf.Patcher(model, tok) for _ in range(DONORS)]
    qs = [q for q in ce.read_jsonl(Path(a.questions)) if ce.split_of(q["id"]) in a.split.split(",")]
    qs = sorted(qs, key=lambda q: len(q["clean"]))[a.offset::a.stride][:a.limit or None]   # (short prompts first)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    for q in qs:
        f = out / f"{q['id']}.json"
        if f.exists():
            continue
        t1 = time.time()
        clean, donors, T = q["clean"], q["corrupt"], len(q["clean"])
        k = DONORS if T <= PARTNER_T else 2
        for j, d in enumerate(donors[:k]):             # the main pair: the first donor whose pair has a contrast
            r0 = pts[0].setup(clean, d, q["answers"])
            if r0["valid"] and r0["gap"] >= cf.MIN_GAP and not r0["same_answer"]:
                donors = [d] + donors[:j] + donors[j + 1:]
                break
        pt, partners = pts[0], [(pts[j], donors[j]) for j in range(1, k)]
        for x in pts:
            x.source_positions = None
        rec = {"id": q["id"], "tokens": T, "word": q["word"], "donors": k}
        try:
            first = next(p for p in range(T) if clean[p] != donors[0][p])
            if "source" in a.cuts and T - 1 - first > SRC_MAX:
                pt.setup(clean, donors[0], q["answers"])
                sc = pt.scores()["neuron"].sum((0, 2))
                top = sorted(range(first, T - 1), key=lambda p: -float(sc[p]))[:SRC_MAX]
                del sc
                for x in pts:
                    x.source_positions = set(top)
                rec["source_screen"] = sorted(top)
            rec.update(cm.find(pt, clean, donors[0], answers=q["answers"], partners=partners, log=log,
                               cuts_wanted=tuple(a.cuts.split(","))))
            if rec.get("skipped"):
                raise _Skip()
            valid = [pt] + [x for x, ids in partners if getattr(x, "ids", None) and x.ids["corrupt"] == ids
                             and x.gap >= cf.MIN_GAP]
            sets = [[("res", i, p) for p in range(sg["start"], sg["end"])] for sg in q["segments"] for i in SEG_LAYERS]
            shares = [x.both(sets) for x in valid]
            rec["segment_map"] = {"layers": list(SEG_LAYERS), "pairs": len(valid), "shares": [
                [sum(sh[j][0] for sh in shares) / len(valid), sum(sh[j][1] for sh in shares) / len(valid)]
                for j in range(len(sets))]}
            r = rnd.Random(zlib.crc32(("random:" + q["id"]).encode()))
            ctrl, own = [], {("neuron", n["layer"], n["pos"], n["neuron"]) for n in rec.get("nodes", [])} | {
                ("head", m["layer"], m["head"], m["pos"]) for m in rec.get("routes", [])}

            def draw(kind, layer, pos, width):        # a unit of the kind, layer and position outside the circuit
                while True:
                    u = cf.unit(kind, layer, pos, r.randrange(width))
                    if u not in own:
                        own.add(u)
                        return u
            for n in rec.get("nodes", []):
                ctrl.append(draw("neuron", n["layer"], n["pos"], pt.D) + (n["stage"],))
            for m in rec.get("routes", []):
                ctrl.append(draw("head", m["layer"], m["pos"], pt.H) + ("movers",))
            sh = [x.both([[u[:4]] for u in ctrl]) for x in valid] if ctrl else []
            rec["random_control"] = [
                {"stage": u[4], "kind": u[0], "layer": u[1], "pos": u[3] if u[0] == "head" else u[2],
                 ("head" if u[0] == "head" else "neuron"): u[2] if u[0] == "head" else u[3],
                 "lost": sum(s_[j][0] for s_ in sh) / len(valid), "restored": sum(s_[j][1] for s_ in sh) / len(valid),
                 **({"attends": pt.attends(u[1], u[2], u[3])} if u[0] == "head" else
                    {"clean": float(pt.act["clean"]["neu"][u[1]][u[2], u[3]])})}
                for j, u in enumerate(ctrl)]
            rec["pairs_valid"] = len(valid)
        except _Skip:
            pass
        except torch.cuda.OutOfMemoryError as e:
            rec["error"] = f"out of memory: {str(e)[:200]}"
            torch.cuda.empty_cache()
        rec["seconds_total"] = round(time.time() - t1, 1)
        if torch.cuda.is_available():
            rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            torch.cuda.reset_peak_memory_stats()
        f.write_text(json.dumps(rec, default=_jsonable))
        log(f"{q['id']} T {T} met {rec.get('met')} gap {rec.get('gap')} {rec['seconds_total']} s {rec.get('error', '')}")


class _Skip(Exception):
    pass


def _jsonable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def render(a) -> None:
    """Each record as the predictor's text: OUT/<id>.txt (the circuit, TC and TCs) and OUT/<id>.random.txt (the same
    text with the random control's units in place of the circuit's, TCr). Both texts share the behaviour point and the
    segment map; units are listed by the larger of their two shares (TOP_UNITS at most), segments above SEG_SHOW (TOP_SEGS
    at most); neuron labels from unit_labels where a pole has one."""
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
    from unit_labels import Labels

    tok = AutoTokenizer.from_pretrained(TARGET)
    labels = Labels(TARGET)
    qs = {q["id"]: q for q in ce.read_jsonl(Path(a.questions))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(Path(a.records).iterdir()):
        if f.suffix != ".json":
            continue
        rec, q = json.loads(f.read_text()), qs.get(f.stem)
        if q is None or "segment_map" not in rec:
            continue
        head = behaviour_point(rec, q, tok) + "\n\n" + segment_lines(rec, q)
        circ = [dict(u, kind="head") for u in rec.get("routes", [])] + [dict(u, kind="neuron") for u in rec.get("nodes", [])]
        (out / f"{f.stem}.txt").write_text(head + "\n\n" + unit_lines(circ, q, tok, labels))
        (out / f"{f.stem}.random.txt").write_text(head + "\n\n" + unit_lines(rec.get("random_control", []), q, tok, labels))
        n += 1
    print(f"{n} circuits rendered -> {out}")


TOP_UNITS, TOP_SEGS, SEG_SHOW = 12, 8, 0.05


def _tokstr(tok, ids: list[int], p: int) -> str:
    return json.dumps(tok.decode([ids[p]]))


def behaviour_point(rec: dict, q: dict, tok) -> str:
    before = tok.decode(q["clean"][max(q["prompt_tokens"], len(q["clean"]) - 14):])
    alts = ", ".join(json.dumps(w) for w in rec.get("alternative_words", [])[:4])
    after = f"after {json.dumps(before)}" if q["prefill_tokens"] else "at the start of the response"
    return (f"Behaviour point: in the shown completion the behaviour first shows at the word {json.dumps(q['word'])}, "
            f"{after}. There the model prefers that word over its next likeliest words ({alts}) by "
            f"{rec['contrast']['clean']:.1f} bits. With every message's content replaced by unrelated text (the response "
            f"so far unchanged) the preference falls by {rec['gap']:.1f} bits. Every share below is a fraction of that "
            f"{rec['gap']:.1f}-bit drop.")


def segment_lines(rec: dict, q: dict) -> str:
    sm, L = rec["segment_map"], len(rec["segment_map"]["layers"])
    rows = []
    for j, sg in enumerate(q["segments"]):
        lost = [sm["shares"][j * L + i][0] for i in range(L)]
        i = max(range(L), key=lambda i: lost[i])
        rows.append((lost[i], sm["layers"][i], sg))
    rows = sorted([r for r in rows if r[0] >= SEG_SHOW], key=lambda r: -r[0])[:TOP_SEGS]
    lines = ["Prompt segments that carry the behaviour (the segment's residual stream replaced by the unrelated text's at "
             "one layer; the share of the drop this removes, at the layer where it is largest; layer 0 replaces the "
             "tokens themselves):"]
    for share, layer, sg in rows:
        text = sg["text"] if len(sg["text"]) <= 160 else sg["text"][:157] + "..."
        lines.append(f"  [{sg['role']}] {json.dumps(text)}: {share:.2f} (layer {layer})")
    if not rows:
        lines.append(f"  no segment alone removes {SEG_SHOW:.2f} or more")
    return "\n".join(lines)


def unit_lines(units: list[dict], q: dict, tok, labels) -> str:
    ids = q["clean"]
    units = sorted(units, key=lambda u: -max(abs(u["lost"]), abs(u["restored"])))[:TOP_UNITS]
    lines = ["Units of Qwen3-8B (each replaced alone by its value under the unrelated text; \"removes\": the share of the "
             "drop this removes; \"restores\": the share it restores when put back alone into the unrelated run):"]
    for u in units:
        where = ("the behaviour point" if u["pos"] == len(ids) - 1 else
                 "response so far" if u["pos"] >= q["prompt_tokens"] else "prompt")
        at = f"at {_tokstr(tok, ids, u['pos'])} ({where})"
        if u["kind"] == "head":
            reads = ", ".join(f"{_tokstr(tok, ids, p)} ({w:.2f})" for p, w in u.get("attends", [])[:3])
            lines.append(f"  attention head L{u['layer']}H{u['head']} {at}: removes {u['lost']:.2f}, restores "
                         f"{u['restored']:.2f}; attends to {reads}")
        else:
            pole = "+" if u.get("pole", 1 if u.get("clean", 0) >= 0 else -1) > 0 else "-"
            name = f"L{u['layer']}.N{u['neuron']}{pole}"
            lab = labels.lookup([name]).get(name, {})
            text = f"; label: {json.dumps(lab['label'][:160])}" if lab.get("label") else ""
            lines.append(f"  neuron {name} {at}: removes {u['lost']:.2f}, restores {u['restored']:.2f}{text}")
    if not units:
        lines.append("  none found")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("spans")
    sp.add_argument("--split", default="dev", help="dev, test or dev,test")
    sub.add_parser("questions")
    r = sub.add_parser("run")
    r.add_argument("questions")
    r.add_argument("out")
    r.add_argument("--split", default="dev")
    r.add_argument("--model", default=TARGET)
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--device", default="cuda")
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--cuts", default="readout,movers", help="the finder's cuts; the source cut (neurons before the "
                   "behaviour point) is off: with every content token swapped, attention reads the swapped tokens' own "
                   "residual streams, so no neuron set can restore the gap (Qwen3-8B dev: the source cut's mask restored "
                   "0.08-0.51 after 4 rounds, 45 of the first 70 s a question)")
    d = sub.add_parser("render")
    d.add_argument("records")
    d.add_argument("out")
    d.add_argument("--questions", default=str(QUESTIONS))
    a = ap.parse_args()
    {"spans": spans, "questions": questions, "run": run, "render": render}[a.cmd](a)


if __name__ == "__main__":
    main()
