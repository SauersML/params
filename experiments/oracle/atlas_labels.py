"""Atlas labels (the lead's 10-15): the strongest Tinker writer's explanation of each pole of a model's MLP neurons,
written from fleet's atlas (experiments/oracle/atlas.py: the top examples of each pole over the shared mixed corpus,
256 tokens before each and 32 after, with the SwiGLU sign regime) in T1's prompt (t1_audit.EXPLAIN), so a label and a
T1 explanation are the same object and neuron_verify's plan, measure and judge check either. Neurons in circuits and
organism investigations come first; the labels file grows, and a neuron already in it is never asked again.

  atlas_labels.py gather --model M --out LIST.jsonl PATH...
      every neuron the JSON under PATH names for model M (dicts with "layer" and "neuron" or "index", as neuron_builder's
      circuit nodes; unit names L<l>.N<i>[+-], as claims; only --keys of a record that has them), most often named first
  atlas_labels.py coverage --atlas ROOT --out LIST.jsonl [--top N] [--frac 0.3]
      every neuron by how often one of its poles is strong (the share of corpus tokens beyond frac of the pole's extreme,
      from the atlas histogram): the neurons the oracle reads at the most positions, for labels past the named ones
  atlas_labels.py label --atlas ROOT --neurons LIST.jsonl --out LABELS.jsonl [--examples 10] [--chunk 200]
      LIST.jsonl lines {"layer", "index"[, "source"]}; LABELS.jsonl gets two lines per neuron, one per pole, in
      neuron_verify's item format ({"layer", "index", "pole", "corpus_max", "explanation", "writer_valid", ...})
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/oracle/rl")]

from t1_audit import EXPLAIN, EXPLAIN_V2, WRITER  # noqa: E402  (T1's prompts and writer: one definition of an explanation)

LONG, SHORT, AFTER = 256, 64, 32   # revise's evidence: tokens before its first `long` examples, before the rest, after each
REVISE = """{evidence}

Your explanation of the {side} pole was:
{explanation}

A reader who saw only that explanation wrote these texts to drive the {side} pole as far as it goes. After each: the largest value it reached at any token as a share of the pole's corpus extreme (a pass needs one text at 0.30 or more), and that token:
{blind}
It also wrote these near misses, texts on close subjects the explanation says should not drive the {side} pole (each should stay below 0.30):
{near}

Revise the explanation of the {side} pole so that texts written from it reach the pole's extreme and close subjects stay below 0.30. Keep what the examples support; change what the reader's results show is wrong or missing. Write JSON with one field, "explanation", 1 to 4 sentences, nothing else."""


def prompt(n: dict, examples: int) -> str:
    """T1's prompt for one atlas neuron. Atlas.neuron gives the negative pole's values and minimum as negative numbers;
    T1 shows them as magnitudes."""
    def block(exs):
        return "\n".join(f"{j}. {e['value']:.2f} [{e['regime']}] ...{e['before']}⟦{e['token']}⟧{e['after']}" for j, e in enumerate(exs[:examples], 1))

    return EXPLAIN.format(layer=n["layer"], index=n["index"], pmax=n["max"]["+"], nmax=-n["max"]["-"],
                          pos=block(n["+"]), neg=block([{**e, "value": -e["value"]} for e in n["-"]]))


def evidence(tok, n: dict, k: dict, long: int, prompt: str = EXPLAIN_V2) -> str:
    """prompt (T1's v2 by default) on an atlas neuron (Atlas.neuron(text=False)): k[pole] examples of each pole, the
    first `long` with LONG tokens before, the rest with SHORT; values as magnitudes."""
    def block(exs):
        out = []
        for j, e in enumerate(exs, 1):
            ids, t = e["ids"], e["at"]
            before = tok.decode(ids[max(0, t - (LONG if j <= long else SHORT)):t])
            out.append(f"{j}. {abs(e['value']):.2f} [{e['regime']}] ...{before}⟦{tok.decode(ids[t:t + 1])}⟧{tok.decode(ids[t + 1:t + 1 + AFTER])}")
        return "\n".join(out)

    return prompt.format(layer=n["layer"], index=n["index"], pmax=n["max"]["+"], nmax=-n["max"]["-"],
                         pos=block(n["+"][:k["+"]]), neg=block(n["-"][:k["-"]]))


def feedback(tok, row: dict) -> tuple[str, str]:
    """A measured row's blind texts and near misses with their values and peak tokens, for REVISE."""
    key = "pos" if row["pole"] == "+" else "neg"
    raw = row["raw_pole_measurements"]

    def peak(text, m):
        at = m.get(f"position_{key}")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return repr(tok.decode(ids[at:at + 1])) if at is not None and at < len(ids) else "none"

    blind = "\n".join(f"- {t!r}: {v:.2f} at {peak(t, m)}" for t, v, m in zip(row["blind_texts"], row["blind_values"], raw))
    near = "\n".join(f"- {t!r}: {v:.2f} at {peak(t, m)}" for t, v, m in
                      zip(row["test_texts"]["near_misses"], row["near_values"], raw[len(row["blind_texts"]):]))
    return blind, near


OUTPUT = """One side of an MLP neuron (layer {layer}, index {index}) in a language model fires strongly at the marked token ⟦ ⟧ in each context below. For each context: the model's likeliest next tokens, and what zeroing the {side} side of this neuron at every position does to its next-token prediction: the tokens whose probability falls most and rises most (probability before -> after) and the total change (half the summed absolute change, 0 to 1).

{blocks}

Describe what the {side} side of this neuron does to the model's next-token prediction: which tokens or kinds of tokens it makes more likely or less likely, and in which contexts. If the change is small everywhere, say so. Write JSON with one field, "output", 1 to 3 sentences, nothing else."""
READ_OUT = """A description of what one side of a neuron in a language model does to the model's next-token prediction:
{output}

The model is reading this text; the neuron's side fires strongly at the last token, marked ⟦ ⟧:
...{before}⟦{token}⟧

If that side of the neuron were zeroed, which next tokens would lose the most probability? Write JSON with one field, "tokens": a list of 3 tokens exactly as they would appear (with a leading space if the token starts a new word), nothing else."""
OUT_DEV, OUT_TEST, OUT_HIT = range(0, 10), range(30, 40), 10   # writer's contexts, held-out contexts, measured falls a hit may name


def context_text(tok, e: dict, before: int = SHORT) -> tuple[str, str]:
    ids, t = e["ids"], e["at"]
    return tok.decode(ids[max(0, t - before):t]), tok.decode(ids[t:t + 1])


def effect_block(tok, j: int, e: dict, eff: dict) -> str:
    b, t = context_text(tok, e)
    f = lambda xs: ", ".join(f"{w!r} {p0:.2f}->{p1:.2f}" for w, p0, p1 in xs[:5])  # noqa: E731
    return (f"{j}. ...{b}⟦{t}⟧\n   likeliest next: " + ", ".join(f"{w!r} {p:.2f}" for w, p in eff["top"][:5]) +
            f"\n   zeroed: falls {f(eff['falls'])}; rises {f(eff['rises'])}; total change {eff['tv']:.2f}")


def hits(predicted, eff: dict) -> bool:
    norm = lambda w: str(w).strip().casefold()  # noqa: E731
    falls = {norm(w) for w, p0, p1 in eff["falls"][:OUT_HIT] if p1 < p0}
    return any(norm(w) in falls for w in (predicted or [])[:3])


def dev_key(row: dict) -> tuple:
    """A development result's rank: T1's pass, then the blind maximum (capped at 1) less the near-miss rate."""
    if not row.get("valid"):
        return (False, -1.0)
    return (bool(row["passes_t1"]), min(row["blind_max"], 1.0) - row["near_miss_rate"])


UNIT = re.compile(r"\bL(\d+)\.N(\d+)[+-]?")


def named(obj, out: list) -> None:
    """Every neuron obj names: dicts with "layer" and "neuron" or "index", and unit names in strings."""
    if isinstance(obj, dict):
        n = obj.get("neuron", obj.get("index"))
        if type(obj.get("layer")) is int and type(n) is int:
            out.append((obj["layer"], n))
        for v in obj.values():
            named(v, out)
    elif isinstance(obj, list):
        for v in obj:
            named(v, out)
    elif isinstance(obj, str):
        out.extend((int(m[1]), int(m[2])) for m in UNIT.finditer(obj))


def gather(a) -> None:
    from collections import Counter

    count, where = Counter(), {}
    files = [f for p in a.paths for f in (sorted(p.rglob("*.json*")) if p.is_dir() else [p])]
    for f in files:
        try:
            docs = [json.loads(f.read_text())] if f.suffix == ".json" else [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for d in docs:
            if isinstance(d, dict) and d.get("model") not in (None, a.model):
                continue
            got = []
            named({k: d[k] for k in a.keys if k in d} if isinstance(d, dict) and any(k in d for k in a.keys) else d, got)
            for key in set(got):
                count[key] += 1
                where.setdefault(key, set()).add(str(f.parent if f.parent != Path(".") else f))
    rows = [{"layer": l, "index": i, "count": c, "source": sorted(where[(l, i)])[:4]} for (l, i), c in count.most_common()]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} neurons named {sum(count.values())} times in {len(files)} files to {a.out}")


def coverage(a) -> None:
    import atlas as atlas_mod
    import numpy as np

    at = atlas_mod.Atlas(a.atlas)
    j = at.meta["rates"].index(a.frac)
    rows = []
    for l in range(at.meta["layers"]):
        r = at.layer(l)["rates"][:, :, j]                     # [D, 2]: the positive and negative pole's share
        rows += [(float(r[i].sum()), l, i, float(r[i, 0]), float(r[i, 1])) for i in range(r.shape[0])]
    rows.sort(reverse=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps({"layer": l, "index": i, "strong_share": s_, "pos": p, "neg": n, "source": f"coverage@{a.frac}"}) + "\n"
                             for s_, l, i, p, n in rows[:a.top]))
    print(f"{min(a.top, len(rows))} of {len(rows)} neurons to {a.out}; the top {a.top} are strong at "
          f"{np.sum([x[0] for x in rows[:a.top]]):.2f} neuron-poles a token of {np.sum([x[0] for x in rows]):.2f}")


def label(a) -> None:
    import atlas as atlas_mod
    import neuron_verify
    import reader_reward

    at = atlas_mod.Atlas(a.atlas)
    model = at.meta["model"]
    done = set()
    if a.out.exists():
        done = {(r["layer"], r["index"]) for r in map(json.loads, a.out.read_text().splitlines()) if r.get("model") == model}
    want, seen = [], set()
    for r in map(json.loads, a.neurons.read_text().splitlines()):
        key = (int(r["layer"]), int(r["index"]))
        if key not in done and key not in seen:
            seen.add(key)
            want.append({**r, "layer": key[0], "index": key[1]})
    print(f"{len(want)} neurons to label ({len(done)} already in {a.out})", flush=True)
    rd = reader_reward.Reader(a.writer, a.out.parent)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.method == "revise":
        return revise(a, at, model, want, rd)
    for s in range(0, len(want), a.chunk):
        part = want[s:s + a.chunk]
        ns = [at.neuron(r["layer"], r["index"], k=a.examples) for r in part]
        prompts = [prompt(n, a.examples) for n in ns]
        responses = rd(prompts, max_tokens=600)
        if len(responses) != len(prompts):
            raise ValueError("writer response count must match every neuron")
        rows = []
        for r, n, p, text in zip(part, ns, prompts, responses):
            got = neuron_verify.parse_json(text) or {}
            for pole, key in (("+", "positive"), ("-", "negative")):
                explanation, maximum = got.get(key), abs(n["max"][pole])
                errors = []
                if not isinstance(explanation, str) or not explanation.strip():
                    errors.append("missing_or_invalid_explanation")
                if not math.isfinite(maximum) or maximum <= 0:
                    errors.append("nonpositive_or_invalid_corpus_max")
                rows.append({"model": model, "layer": n["layer"], "index": n["index"], "pole": pole, "corpus_max": maximum,
                             "explanation": explanation, "regimes": [e["regime"] for e in n[pole][:a.examples]],
                             "rates": n["rates"][pole], "source": r.get("source"), "writer_valid": not errors,
                             "writer_errors": errors, "writer_model": a.writer, "writer_raw_response": text,
                             "writer_request_sha256": hashlib.sha256(p.encode()).hexdigest(), "atlas": str(a.atlas)})
        with a.out.open("a") as f:
            f.write("".join(json.dumps(x, allow_nan=False) + "\n" for x in rows))
        print(f"{s + len(part)}/{len(want)} neurons labelled; spend ${rd.spend.total:.3f}", flush=True)


def maxima_of(path: Path | None) -> dict:
    """{(layer, index, pole): corpus maximum} from a T1 atlas (t1_audit collect), to score against T1's own maxima."""
    if not path:
        return {}
    return {(n["layer"], n["index"], p): n["max"][p] for n in json.loads(path.read_text())["neurons"] for p in "+-"}


def measurer(a):
    import neuron_verify

    if a.measure_url:
        return lambda rows, seed=0: neuron_verify.remote_measure(a.measure_url, rows, seed)
    tg = neuron_verify.Target(a.model, a.device)
    return lambda rows, seed=0: neuron_verify.measure(tg, rows, seed)


def revise(a, at, model, want, rd) -> None:
    """The label method of the lead's 10-15 arm: the writer sees `examples` examples of each pole (the first `long` with
    LONG tokens before, the rest SHORT; the opposite pole's are its contrast) under T1's v2 instruction; each pole's
    explanation is planned by the reader (greedy), measured, and revised (REVISE: the pole's own examples, the opposite
    pole's first 10, its reader's texts with values and peak tokens) up to `rounds` times while it fails T1 on these
    development texts; the round with the best development result is the label. Scoring on fresh texts is `score`."""
    import neuron_verify

    measure, tok, cm = measurer(a), at.tok, maxima_of(a.corpus_max)
    for s in range(0, len(want), a.chunk):
        part = want[s:s + a.chunk]
        ns = [at.neuron(r["layer"], r["index"], k=a.examples, text=False) for r in part]
        prompts = [evidence(tok, n, {"+": a.examples, "-": a.examples}, a.long) for n in ns]
        responses = rd(prompts, max_tokens=800)
        poles = []
        for r, n, text in zip(part, ns, responses):
            got = neuron_verify.parse_json(text) or {}
            for pole, key in (("+", "positive"), ("-", "negative")):
                maximum = cm.get((n["layer"], n["index"], pole), abs(n["max"][pole]))
                e = got.get(key)
                poles.append({"model": model, "layer": n["layer"], "index": n["index"], "pole": pole, "corpus_max": maximum,
                              "explanation": e if isinstance(e, str) and e.strip() else None, "source": r.get("source"),
                              "rounds": [], "_n": n, "_raw": text})
        live = [p for p in poles if p["explanation"] and p["corpus_max"] > 0]
        for rnd in range(a.rounds + 1):
            items = [{k: p[k] for k in ("layer", "index", "pole", "corpus_max", "explanation")} for p in live]
            dev = measure(neuron_verify.plan(rd, items))
            for p, d in zip(live, dev, strict=True):
                p["rounds"].append({"explanation": p["explanation"], "valid": d.get("valid"), "blind_max": d.get("blind_max"),
                                    "near_miss_rate": d.get("near_miss_rate"), "passes_t1": d.get("passes_t1"),
                                    "blind_texts": d.get("blind_texts"), "blind_values": d.get("blind_values")})
                p["_dev"] = d
            fail = [p for p in live if not p["_dev"].get("passes_t1") and p["_dev"].get("valid")]
            print(f"round {rnd}: {len(live) - len(fail)}/{len(live)} poles pass on development texts; spend ${rd.spend.total:.3f}", flush=True)
            if rnd == a.rounds or not fail:
                break
            asks = []
            for p in fail:
                n, other = p["_n"], "-" if p["pole"] == "+" else "+"
                blind, near = feedback(tok, p["_dev"])
                asks.append(REVISE.format(evidence=evidence(tok, n, {p["pole"]: a.examples, other: 10}, a.long, EXPLAIN[:EXPLAIN.index("Explain each pole:")].rstrip()),
                                          side="positive" if p["pole"] == "+" else "negative", explanation=p["explanation"], blind=blind, near=near))
            for p, text in zip(fail, rd(asks, max_tokens=500), strict=True):
                e = (neuron_verify.parse_json(text) or {}).get("explanation")
                if isinstance(e, str) and e.strip():
                    p["explanation"] = e
            live = fail
        rows = []
        for p in poles:
            best = max(range(len(p["rounds"])), key=lambda i: dev_key(p["rounds"][i])) if p["rounds"] else None
            if best is not None:
                p["explanation"], p["chosen_round"] = p["rounds"][best]["explanation"], best
            errors = [] if p["explanation"] and p["corpus_max"] > 0 else ["missing_or_invalid_explanation" if not p["explanation"] else "nonpositive_or_invalid_corpus_max"]
            rows.append({**{k: v for k, v in p.items() if not k.startswith("_")}, "writer_valid": not errors, "writer_errors": errors,
                         "writer_model": a.writer, "method": f"revise@{a.examples}x{a.long}/{a.rounds}", "atlas": str(a.atlas),
                         "writer_raw_response": p["_raw"]})
        with a.out.open("a") as f:
            f.write("".join(json.dumps(x, allow_nan=False) + "\n" for x in rows))
        print(f"{s + len(part)}/{len(want)} neurons labelled; spend ${rd.spend.total:.3f}", flush=True)


def score(a) -> None:
    """Each label's measured scores on fresh texts its writer never saw: the reader's sampled plan (temperature 0.7,
    seed) at each blind length (T1's 1 to 3 sentences; up to about 150 tokens), measured and the steering judged, under
    "scores" {length: {pass, blind_max, near_miss_rate, distant_rate, steer_match}} in the labels file."""
    import neuron_verify
    import reader_reward

    rows = [json.loads(x) for x in a.labels.read_text().splitlines() if x.strip()]
    cm = maxima_of(a.corpus_max)
    rd = reader_reward.Reader(neuron_verify.READER, a.labels.parent)
    measure = measurer(a)
    todo = [r for r in rows if r.get("writer_valid") and r.get("explanation")]
    for length in a.lengths:
        items = [{"layer": r["layer"], "index": r["index"], "pole": r["pole"], "explanation": r["explanation"],
                  "corpus_max": cm.get((r["layer"], r["index"], r["pole"]), r["corpus_max"])} for r in todo]
        got = neuron_verify.judge(rd, measure(neuron_verify.plan(rd, items, length=length, temperature=0.7, seed=a.seed), a.seed))
        for r, g in zip(todo, got, strict=True):
            r.setdefault("scores", {})[length] = {"seed": a.seed, "corpus_max": g["corpus_max"], "valid": g.get("valid"),
                                                 "pass": bool(g.get("valid") and g.get("passes_t1")),
                                                 **{k: g.get(k) for k in ("blind_max", "near_miss_rate", "distant_rate", "steer_match")}}
        ok = [r["scores"][length] for r in todo]
        print(f"{length}: {sum(x['pass'] for x in ok)}/{len(rows)} poles pass on fresh texts; spend ${rd.spend.total:.3f}", flush=True)
    a.labels.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))


def output(a) -> None:
    """Each pole's output label (the lead's 10-15): what zeroing the pole changes in the next-token prediction on its
    top atlas contexts OUT_DEV (measured, neuron_verify serve /effects), described by the writer (OUTPUT); tested on
    the held-out contexts OUT_TEST, where the reader names the 3 tokens it expects to lose the most (READ_OUT) and a
    context is a hit when one is among the OUT_HIT measured largest falls. Baseline: the unedited model's own 3
    likeliest next tokens. A pole's output label passes when it hits at least half the held-out contexts and more than
    the baseline. Stored as "output" and "output_scores" in the labels file."""
    import atlas as atlas_mod
    import neuron_verify
    import reader_reward

    at = atlas_mod.Atlas(a.atlas)
    tok = at.tok
    rows = [json.loads(x) for x in a.labels.read_text().splitlines() if x.strip()]
    todo = [r for r in rows if r.get("writer_valid") and "output" not in r]
    wr, rd = reader_reward.Reader(a.writer, a.labels.parent), reader_reward.Reader(neuron_verify.READER, a.labels.parent)
    for s0 in range(0, len(todo), a.chunk):
        part = todo[s0:s0 + a.chunk]
        exs = [at.neuron(r["layer"], r["index"], k=max(OUT_TEST) + 1, text=False)[r["pole"]] for r in part]
        items = [{"contexts": [e["ids"][:e["at"] + 1] for e in ex[:max(OUT_TEST) + 1]], "layer": r["layer"], "index": r["index"],
                  "pole": r["pole"]} for r, ex in zip(part, exs)]
        effs = neuron_verify.remote_effects(a.measure_url, items)
        asks = [OUTPUT.format(layer=r["layer"], index=r["index"], side="positive" if r["pole"] == "+" else "negative",
                              blocks="\n".join(effect_block(tok, j + 1, ex[i], eff[i]) for j, i in enumerate(OUT_DEV) if i < len(ex)))
                for r, ex, eff in zip(part, exs, effs)]
        outs = [(neuron_verify.parse_json(x) or {}).get("output") for x in wr(asks, max_tokens=400)]
        reads, where = [], []
        for k, (r, ex, o) in enumerate(zip(part, exs, outs)):
            for i in OUT_TEST:
                if isinstance(o, str) and o.strip() and i < len(ex):
                    b, t = context_text(tok, ex[i])
                    reads.append(READ_OUT.format(output=o, before=b, token=t))
                    where.append((k, i))
        got = {}
        for (k, i), x in zip(where, rd(reads, max_tokens=80)):
            got.setdefault(k, {})[i] = (neuron_verify.parse_json(x) or {}).get("tokens")
        for k, (r, ex, eff, o) in enumerate(zip(part, exs, effs, outs)):
            test = [i for i in OUT_TEST if i < len(ex)]
            lab = sum(hits(got.get(k, {}).get(i), eff[i]) for i in test)
            base = sum(hits([w for w, _ in eff[i]["top"][:3]], eff[i]) for i in test)
            r["output"] = o if isinstance(o, str) and o.strip() else None
            r["output_scores"] = {"contexts": len(test), "hits": lab, "baseline_hits": base,
                                  "pass": bool(r["output"]) and len(test) > 0 and lab >= len(test) / 2 and lab > base,
                                  "mean_tv": sum(eff[i]["tv"] for i in test) / max(len(test), 1),
                                  "predicted": [got.get(k, {}).get(i) for i in test],
                                  "falls": [[w for w, _, _ in eff[i]["falls"][:OUT_HIT]] for i in test]}
            r["output_dev_effects"] = [eff[i] for i in OUT_DEV if i < len(eff)]
        a.labels.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
        done = [r for r in rows if "output_scores" in r]
        print(f"{len(done)} poles with output labels: pass {sum(r['output_scores']['pass'] for r in done)}; "
              f"writer ${wr.spend.total:.3f}, reader ${rd.spend.total:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    g = sub.add_parser("gather")
    g.add_argument("paths", type=Path, nargs="+")
    g.add_argument("--model", required=True, help="records naming another model are skipped")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--keys", type=lambda x: x.split(","), default=["nodes"],
                   help="read only these top-level keys of a record that has any (neuron_builder's circuit: nodes, not its ranked candidates)")
    c = sub.add_parser("coverage")
    c.add_argument("--atlas", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--top", type=int, default=5000)
    c.add_argument("--frac", type=float, default=0.3, help="one of the atlas's rate fractions (0.1, 0.3, 0.5)")
    lb = sub.add_parser("label")
    lb.add_argument("--atlas", type=Path, required=True, help="an atlas root (atlas.py collect/merge/weights)")
    lb.add_argument("--neurons", type=Path, required=True)
    lb.add_argument("--out", type=Path, required=True)
    lb.add_argument("--writer", default=WRITER)
    lb.add_argument("--examples", type=int, default=10)
    lb.add_argument("--chunk", type=int, default=200, help="neurons per writer batch, appended as each finishes")
    lb.add_argument("--method", choices=("t1", "revise"), default="t1", help="t1: T1's prompt, one writer call; revise: more "
                    "evidence and up to --rounds revisions on the writer's own development results")
    lb.add_argument("--long", type=int, default=10, help="revise: examples shown with LONG tokens before")
    lb.add_argument("--rounds", type=int, default=2)
    sc = sub.add_parser("score")
    sc.add_argument("--labels", type=Path, required=True)
    sc.add_argument("--lengths", nargs="+", default=["short", "long"])
    sc.add_argument("--seed", type=int, default=1)
    ou = sub.add_parser("output")
    ou.add_argument("--labels", type=Path, required=True)
    ou.add_argument("--atlas", type=Path, required=True)
    ou.add_argument("--writer", default=WRITER)
    ou.add_argument("--chunk", type=int, default=100, help="poles per batch, the labels file rewritten after each")
    for x in (lb, sc, ou):
        x.add_argument("--corpus-max", type=Path, help="a T1 atlas (t1_audit collect): score against its maxima")
        x.add_argument("--measure-url", help="neuron_verify serve over a tunnel (http://localhost:PORT); else the target here")
        x.add_argument("--model", default="Qwen/Qwen3-8B")
        x.add_argument("--device")
    a = ap.parse_args()
    {"gather": gather, "coverage": coverage, "label": label, "score": score, "output": output}[a.what](a)


if __name__ == "__main__":
    main()
