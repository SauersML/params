"""Format controls for the persona traits (the lead's 10-16 method decision). The model-written persona traits share one
format ("Is the following statement something you would say? ... Answer with Yes or No."), and about 30 of them move
together under knockouts, so their shared changes may measure the format's statement-endorsing machinery rather than
the persona. Control: neutral statements in the same format (CounterFact facts, true and false; the matching answer is
Yes for a true one and No for a false one), scored like a persona item: log p(matching answer) - log p(the other). Each
persona item is matched to the neutral statement with the same matching answer whose base-model value is nearest its
own, and the controlled trait is the persona value minus its matched neutral value: a knockout that moves answers in
this format by the same amount at the same confidence cancels.

  controls.py build OUT                     the neutral statements as a phenome directory (one trait,
                                            control/neutral-statements; score and report it with phenome.py)
  controls.py match PHENOME CONTROLS OUT.json
                                            per persona trait and item, its matched neutral item (both reported)
  controls.py runsets PHENOME CONTROLS MATCH DESIGN OUT --thr THR [--every 2]
                                            the knockout sets of DESIGN (every k-th) on the matched neutral items of the
                                            screen's persona panel (screen.panel_items: the same items as the screen's);
                                            per set and persona trait, the change of its matched neutral mean, "ctl/<trait>"
  controls.py compose OUT CHANGES... --controls CTL_CHANGES...
                                            the screen's changes with every controlled persona trait's change replaced by
                                            its change minus its matched neutral change (only sets in both), for
                                            effects.py correlation
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import claim_check as cc  # noqa: E402
import neuron_builder as nb  # noqa: E402
import phenome as ph  # noqa: E402
import polycheck as pc  # noqa: E402
import screen as sc  # noqa: E402

FACTS = 200  # CounterFact facts, each as a true and a false statement (400 items: phenome's MAX_ITEMS)
NAME = "control/neutral-statements"


def neutral_statements(rng) -> list:
    """FACTS CounterFact facts with declarative prompts (their own random stream), each a true (matching Yes) and a false
    (matching No) statement in the persona format, forms A and B as mwe_persona's."""
    import pandas as pd

    df = pd.concat([pd.read_parquet(ph.hf("azhx/counterfact", f)) for f in ("data/train-00000-of-00001-05d11247db7abce8.parquet",
                                                                             "data/test-00000-of-00001-bacb83500fca49a9.parquet")])
    rows = [(k, r) for k, r in enumerate(df["requested_rewrite"]) if "?" not in r["prompt"]]  # (declarative prompts only)
    items = []
    for k, r in random.Random(NAME).sample(rows, FACTS):
        p = r["prompt"].format(r["subject"])
        for truth, target in (("t", r["target_true"]["str"]), ("f", r["target_new"]["str"])):
            q = f'Is the following statement something you would say?\n"{p} {target}."'
            m, n = ("Yes", "No") if truth == "t" else ("No", "Yes")
            items.append({"id": f"{k}{truth}", "A": ph.pair(q + "\nAnswer with Yes or No.", m, n),
                          "B": ph.pair(q, " " + m, " " + n, prefill="My answer is"), "cov": {"matching_yes": int(m == "Yes")}})
    return [(NAME, "persona", "pair", items)]


def build(a) -> None:
    ph.write(a.out, [neutral_statements])


def values(d: Path) -> dict:
    """{trait: {item: (value, matching_yes)}} from a reported phenome's values.jsonl."""
    out: dict = {}
    for x in map(json.loads, (d / "values.jsonl").read_text().splitlines()):
        out.setdefault(x["trait"], {})[x["item"]] = (x["value"], (x.get("cov") or {}).get("matching_yes"))
    return out


def match(a) -> None:
    pv, cv = values(a.phenome), values(a.controls)[NAME]
    kept = {t["name"] for t in json.loads((a.phenome / "phenome.json").read_text())["traits"] if t["kept"]}
    pool = {s_: sorted((v, i) for i, (v, m) in cv.items() if m == s_) for s_ in (0, 1)}
    out, gaps = {}, []
    for t in sorted(kept):
        if not t.startswith("persona/"):
            continue
        out[t] = {}
        for i, (v, m) in pv[t].items():
            vs = np.array([x for x, _ in pool[m]])
            j = int(np.abs(vs - v).argmin())
            out[t][i] = pool[m][j][1]
            gaps.append(abs(vs[j] - v))
    a.out.write_text(json.dumps(out))
    print(f"{len(out)} persona traits matched; median |value gap| {np.median(gaps):.2f}, 95th percentile {np.percentile(gaps, 95):.2f}")


def runsets(a) -> None:
    import torch
    from transformers import AutoTokenizer

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = json.loads(a.match.read_text())
    panel = sc.panel_items(a.phenome)
    need = {t: [m[t][i] for i in sorted(panel[t])] for t in m if t in panel}
    items = sorted({i for v in need.values() for i in v})
    seqs = [s_ for s_ in map(json.loads, (a.controls / "sequences.jsonl").read_text().splitlines())
            if s_["id"].split("|")[1] in items and s_["id"].split("|")[2] == "A"]
    sets = [json.loads(x) for x in a.design.read_text().splitlines() if x.strip()][::a.every]
    done = {json.loads(x)["id"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = ph.load_model(a.model, None, dev)
    nr = nb.Neurons(model, tok)
    nr.load(a.thr)
    nr.thr = None
    tr = pc.Trait(nr, seqs, [])
    ix = {i: k for k, i in enumerate(tr.items)}

    def means(v):
        return {f"ctl/{t}": float(np.mean([v[ix[i]] for i in js])) for t, js in need.items()}

    base = means(tr.run()[0])
    t0, ran = time.time(), 0
    with a.out.open("a") as fh:
        for st in sets:
            if st["id"] in done:
                continue
            e, r = cc.knockout_edit(nr, [("N", int(l_), int(n), int(pl), None) for l_, n, pl in st["units"]])
            got = means(tr.run(e, r)[0])
            fh.write(json.dumps({"id": st["id"], "units": st["units"], "change": {k: got[k] - base[k] for k in base}}) + "\n")
            fh.flush()
            ran += 1
            if ran % 20 == 1:
                print(f"{ran} sets, {(time.time() - t0) / ran:.1f} s a set", flush=True)
    print(f"ran {ran} sets ({len(seqs)} sequences a pass) into {a.out}")


def compose(a) -> None:
    raw = {r["id"]: r for f in a.changes for r in map(json.loads, f.read_text().splitlines()) if r}
    ctl = {r["id"]: r for f in a.controls for r in map(json.loads, f.read_text().splitlines()) if r}
    n = 0
    with a.out.open("w") as fh:
        for i, r in raw.items():
            if i not in ctl:
                continue
            ch = dict(r["change"])
            for k, v in ctl[i]["change"].items():
                t = k[len("ctl/"):]
                if t in ch:
                    ch[t] = ch[t] - v
            fh.write(json.dumps({"id": i, "units": r["units"], "change": ch}) + "\n")
            n += 1
    print(f"{n} sets with controlled persona traits into {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out", type=Path)
    mt = sub.add_parser("match")
    mt.add_argument("phenome", type=Path)
    mt.add_argument("controls", type=Path)
    mt.add_argument("out", type=Path)
    r = sub.add_parser("runsets")
    r.add_argument("phenome", type=Path)
    r.add_argument("controls", type=Path)
    r.add_argument("match", type=Path)
    r.add_argument("design", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--thr", type=Path, required=True)
    r.add_argument("--every", type=int, default=1)
    r.add_argument("--model", default="Qwen/Qwen3-8B")
    r.add_argument("--device")
    c = sub.add_parser("compose")
    c.add_argument("out", type=Path)
    c.add_argument("changes", type=Path, nargs="+")
    c.add_argument("--controls", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    {"build": build, "match": match, "runsets": runsets, "compose": compose}[a.cmd](a)


if __name__ == "__main__":
    main()
