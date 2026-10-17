"""Reviewer item 12 (the lead's 10-16): our labelling pipeline against Cammarata's on the essay's own Qwen3-32B neurons,
under the essay's own test (its appendix: reading only a neuron's explanation, an agent writes five new examples that
should drive it and five near misses that should not; a response of 30% or more of the neuron's strongest corpus
example is strong; per neuron the explanation wins, ties or loses on the count of strong examples against strong near
misses). The 20 neurons are the essay's "first twenty" (selected, not random); their full explanations are not public,
so the essay side uses each neuron's published name, or the longer text where the essay prints one. Corpus maxima are
ours, over the 10M-token prefix T1 used (the essay's are over 10B tokens, so our ratios run higher on both sides).

  essay_compare.py neurons --page PAGE.html --out NEURONS.jsonl     the 20 neurons, names and published results
  essay_compare.py score --neurons NEURONS.jsonl --labels LABELS.jsonl --atlas ATLAS.json --measure-url U --out OUT.json
      both explanation sets under the essay's test on the same fresh reader texts' sampling (seed), and the comparison
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "method/oracle"), str(HERE.parents[1] / "method/oracle/rl")]


def neurons(a) -> None:
    s = a.page.read_text(encoding="utf-8")
    j = s.find('id="explanations-could-lose"')
    seg = s[j:s.find("</figure>", j)]
    rows = []
    for name, l, i, new, ns, near, nn in re.findall(r'title="(.+?) \(L(\d+)/N(\d+)\): new examples ([\d.]+) \((\d)/5 strong\), near misses ([\d.]+) \((\d)/5\)"', seg):
        rows.append({"layer": int(l), "index": int(i), "pole": "+", "essay_name": html.unescape(name),
                     "essay": {"new_mean": float(new), "new_strong": int(ns), "near_mean": float(near), "near_strong": int(nn)}})
    longer = {f"L{l}/N{i}": html.unescape(re.sub(r"<[^>]+>", "", b)).strip()
              for l, i, b in re.findall(r'<div class="bmx-card" title="Qwen3-32B neuron L(\d+)/N(\d+)[^"]*"><p>(.*?)</p>', s, re.S)}
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", seg)))
    m = re.search(r"One neuron: “(.+?)” Its explanation: (.+?)\. New sentences", t)
    for r in rows:
        key = f"L{r['layer']}/N{r['index']}"
        text = longer.get(key) or (m[2] if m and m[1] == r["essay_name"] else None)
        r["essay_explanation"] = re.sub(r"\s+", " ", text).rstrip("…").strip() if text else r["essay_name"]
        r["essay_text_source"] = "longer text printed in the essay" if text else "published name only"
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} neurons, {sum(r['essay_text_source'] != 'published name only' for r in rows)} with a longer explanation, to {a.out}")


def score(a) -> None:
    import neuron_verify
    import reader_reward

    ns = [json.loads(x) for x in a.neurons.read_text().splitlines() if x.strip()]
    labels = {(r["layer"], r["index"], r["pole"]): r for r in map(json.loads, a.labels.read_text().splitlines())}
    at = {(x["layer"], x["index"]): x for x in json.loads(a.atlas.read_text())["neurons"]}
    rd = reader_reward.Reader(neuron_verify.READER, a.out.parent)
    sides = {"essay": [], "ours": []}
    for n in ns:
        cm = at[(n["layer"], n["index"])]["max"]["+"]
        base = {"layer": n["layer"], "index": n["index"], "pole": "+", "corpus_max": cm}
        sides["essay"].append({**base, "explanation": n["essay_explanation"]})
        ours = labels.get((n["layer"], n["index"], "+"), {}).get("explanation")
        sides["ours"].append({**base, "explanation": ours or "no label"})
    got = {}
    for side, items in sides.items():
        rows = neuron_verify.remote_measure(a.measure_url, neuron_verify.plan(rd, items, temperature=0.7, seed=a.seed, protocol="essay"), a.seed)
        got[side] = rows
        print(f"{side}: strong {sum(r.get('strong_share', 0) for r in rows) * 5:.0f}/{5 * len(rows)}, near misses "
              f"{sum(r.get('near_miss_rate', 0) for r in rows) * 5:.0f}/{5 * len(rows)}, outcomes "
              f"{ {o: sum(r.get('essay_outcome') == o for r in rows) for o in ('won', 'tied', 'lost')} }; ${rd.spend.total:.3f}", flush=True)
    per = []
    for k, n in enumerate(ns):
        e, o = got["essay"][k], got["ours"][k]
        per.append({"neuron": f"L{n['layer']}/N{n['index']}", "essay_name": n["essay_name"], "essay_text_source": n["essay_text_source"],
                    "published": n["essay"], "essay_here": {x: e.get(x) for x in ("strong_share", "near_miss_rate", "essay_outcome", "blind_max")},
                    "ours": {x: o.get(x) for x in ("strong_share", "near_miss_rate", "essay_outcome", "blind_max")},
                    "our_label": sides["ours"][k]["explanation"], "values": {"essay": e.get("blind_values"), "ours": o.get("blind_values"),
                                                                              "essay_near": e.get("near_values"), "ours_near": o.get("near_values")}})
    d = [p["ours"]["strong_share"] - p["essay_here"]["strong_share"] for p in per]
    mean = sum(d) / len(d)
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1)) if len(d) > 1 else 0.0
    summary = {"n": len(per), "published_strong": sum(p["published"]["new_strong"] for p in per) / (5 * len(per)),
               "published_near": sum(p["published"]["near_strong"] for p in per) / (5 * len(per)),
               "essay_here_strong": sum(p["essay_here"]["strong_share"] for p in per) / len(per),
               "essay_here_near": sum(p["essay_here"]["near_miss_rate"] for p in per) / len(per),
               "ours_strong": sum(p["ours"]["strong_share"] for p in per) / len(per),
               "ours_near": sum(p["ours"]["near_miss_rate"] for p in per) / len(per),
               "ours_minus_essay_strong": mean, "interval_95": [mean - 2.09 * sd / math.sqrt(len(d)), mean + 2.09 * sd / math.sqrt(len(d))],
               "reader": neuron_verify.READER, "seed": a.seed, "tinker_usd": rd.spend.total,
               "notes": ["the essay's selected first twenty neurons, not a random sample",
                         "our corpus maxima over a 10M-token prefix; the essay's over 10B tokens",
                         "essay side: published names (two longer texts), not the investigating agent's full explanations"]}
    a.out.write_text(json.dumps({"summary": summary, "neurons": per}, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    n = sub.add_parser("neurons")
    n.add_argument("--page", type=Path, required=True)
    n.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--neurons", type=Path, required=True)
    s.add_argument("--labels", type=Path, required=True)
    s.add_argument("--atlas", type=Path, required=True, help="t1_audit collect's output on these neurons (corpus maxima)")
    s.add_argument("--measure-url", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    {"neurons": neurons, "score": score}[a.what](a)


if __name__ == "__main__":
    main()
