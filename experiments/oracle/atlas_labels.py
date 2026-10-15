"""Atlas labels (the lead's 10-15): the strongest Tinker writer's explanation of each pole of a model's MLP neurons,
written from fleet's atlas (experiments/oracle/atlas.py: the top examples of each pole over the shared mixed corpus,
256 tokens before each and 32 after, with the SwiGLU sign regime) in T1's prompt (t1_audit.EXPLAIN), so a label and a
T1 explanation are the same object and neuron_verify's plan, measure and judge check either. Neurons in circuits and
organism investigations come first; the labels file grows, and a neuron already in it is never asked again.

  atlas_labels.py gather --model M --out LIST.jsonl PATH...
      every neuron the JSON under PATH names for model M (dicts with "layer" and "neuron" or "index", as neuron_builder's
      circuit nodes; unit names L<l>.N<i>[+-], as claims; only --keys of a record that has them), most often named first
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

from t1_audit import EXPLAIN, WRITER  # noqa: E402  (T1's prompt and writer: one definition of an explanation)


def prompt(n: dict, examples: int) -> str:
    """T1's prompt for one atlas neuron. Atlas.neuron gives the negative pole's values and minimum as negative numbers;
    T1 shows them as magnitudes."""
    def block(exs):
        return "\n".join(f"{j}. {e['value']:.2f} [{e['regime']}] ...{e['before']}⟦{e['token']}⟧{e['after']}" for j, e in enumerate(exs[:examples], 1))

    return EXPLAIN.format(layer=n["layer"], index=n["index"], pmax=n["max"]["+"], nmax=-n["max"]["-"],
                          pos=block(n["+"]), neg=block([{**e, "value": -e["value"]} for e in n["-"]]))


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    g = sub.add_parser("gather")
    g.add_argument("paths", type=Path, nargs="+")
    g.add_argument("--model", required=True, help="records naming another model are skipped")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--keys", type=lambda x: x.split(","), default=["nodes"],
                   help="read only these top-level keys of a record that has any (neuron_builder's circuit: nodes, not its ranked candidates)")
    lb = sub.add_parser("label")
    lb.add_argument("--atlas", type=Path, required=True, help="an atlas root (atlas.py collect/merge/weights)")
    lb.add_argument("--neurons", type=Path, required=True)
    lb.add_argument("--out", type=Path, required=True)
    lb.add_argument("--writer", default=WRITER)
    lb.add_argument("--examples", type=int, default=10)
    lb.add_argument("--chunk", type=int, default=200, help="neurons per writer batch, appended as each finishes")
    a = ap.parse_args()
    {"gather": gather, "label": label}[a.what](a)


if __name__ == "__main__":
    main()
