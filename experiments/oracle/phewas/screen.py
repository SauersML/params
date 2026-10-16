"""Stage 2 of the complex-trait neuron PheWAS (#2951, the lead's 10-15): randomized pooled knockouts of Qwen3-8B neuron
poles, every trait of the phenome measured in every pass, so one screen gives the whole poles x traits matrix of causal
effects. This file has the panel, the knockout runs and the noise pilot that sizes the screen.

Panel: PANEL dev prompts per kept trait (phenome.json), measured in form A (phenome.py's sequences: a pair's positive and
negative continuation, a set's or a teacher's one sequence, an eot's form A). A trait's value under an edit is the mean
over its panel prompts; its change is that minus the unedited model's. Every batch has ROWS sequences and the batches are
the same for every edit, so a change is the edit's alone (bfloat16 outputs move with the batch size).
Knockout: claim_check.knockout_edit, the claim checker's zero: each pole's values at every position from 1 on set to 0,
the first position and massive values (the T2 file's per-layer cut) live; the model itself, no weak zeroing.

Pilot (pilot): for each set size in SIZES, SETS random sets of poles drawn uniformly over the model's poles: the spread of
each trait's change across sets (the noise a pooled design averages); two disjoint sets of the middle size and their
union (additivity); the top stage-1 hits alone (--assoc: assoc.py's summary.json); the unedited panel twice (exactness).

  screen.py pilot PHENOME OUT.json --thr T2.pt [--assoc SUMMARY.json] [--sizes 16 64 256 1024 4096] [--sets 16]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
import claim_check as cc  # noqa: E402
import neuron_builder as nb  # noqa: E402

PANEL = 8
ROWS = 16
SIZES = (16, 64, 256, 1024, 4096)
SETS = 16
SINGLES = 3  # top stage-1 hits per trait tested alone (the traits with the most hits)


def logit_sum(lps) -> float:
    s = math.log(sum(math.exp(v) for v in lps))
    return s - math.log(max(1e-300, 1 - math.exp(s)))


class Panel:
    """The panel's sequences in fixed batches, and each trait's value under an edit."""

    def __init__(self, phenome: Path, nr: nb.Neurons, per: int = PANEL, seed: int = 15):
        import torch

        self.nr, self.torch = nr, torch
        phen = json.loads((phenome / "phenome.json").read_text())
        self.traits = {t["name"]: t for t in phen["traits"] if t["kept"]}
        split = {}
        for x in map(json.loads, (phenome / "splits.jsonl").read_text().splitlines()):
            split[(x["trait"], x["item"])] = x["split"]
        rng = random.Random(seed)
        want = {}
        for name in self.traits:
            dev = sorted(i for (t, i), s in split.items() if t == name and s == "dev")
            want[name] = set(rng.sample(dev, min(per, len(dev))))
        seqs = []
        for line in (phenome / "sequences.jsonl").open():
            s = json.loads(line)
            name, item, form, role = s["id"].split("|")
            if name in want and item in want[name] and form in ("A", "AB"):
                seqs.append(s)
        seqs.sort(key=lambda s: len(s["prompt"]) + len(s["cont"]))
        self.seqs = seqs
        self.batches = []
        for k in range(0, len(seqs), ROWS):
            part = seqs[k:k + ROWS]
            part = part + part[-1:] * (ROWS - len(part))
            T = max(len(s["prompt"]) + len(s["cont"]) for s in part)
            x = torch.zeros(ROWS, T, dtype=torch.long)
            m = torch.zeros(ROWS, T, dtype=torch.long)
            for j, s in enumerate(part):
                q = s["prompt"] + s["cont"]
                x[j, :len(q)] = torch.tensor(q)
                m[j, :len(q)] = 1
            self.batches.append((x.to(nr.dev), m.to(nr.dev), part, min(ROWS, len(seqs) - k)))

    def run(self, edit=None, route=None) -> dict:
        """{trait: mean over its panel prompts of its form-A value} under edit and route."""
        torch, nr = self.torch, self.nr
        rec = {}
        with torch.no_grad(), nr.hooks(edit=edit, route=route):
            for x, m, part, b in self.batches:
                h = nr.core.model(input_ids=x, attention_mask=m).last_hidden_state
                for j, s in enumerate(part[:b]):
                    P, C = len(s["prompt"]), len(s["cont"])
                    lp = torch.log_softmax(nr.core.lm_head(h[j, P - 1:P + C]).float(), -1)
                    rec[s["id"]] = ([float(lp[t, s["cont"][t]]) for t in range(C)], [float(lp[C, t]) for t in s.get("set", [])])
        by: dict = {}
        for key, (lp, st) in rec.items():
            name, item, form, role = key.split("|")
            by.setdefault(name, {}).setdefault(item, {})[role] = (lp, st)
        out = {}
        for name, items in by.items():
            kind, vs = self.traits[name]["kind"], []
            for r in items.values():
                if kind == "pair" and "pos" in r and "neg" in r:
                    vs.append(sum(r["pos"][0]) - sum(r["neg"][0]))
                elif kind == "set":
                    vs.append(logit_sum(r["set"][1][:len(self.traits[name]["tokens"]["A"])]))
                elif kind == "teacher":
                    vs.append(sum(r["ref"][0]) / max(len(r["ref"][0]), 1))
                elif kind == "eot":
                    vs.append(logit_sum(r["eot"][1]))
            out[name] = sum(vs) / len(vs) if vs else float("nan")
        return out


def random_poles(nr: nb.Neurons, n: int, rng: random.Random, avoid=frozenset()) -> list[tuple]:
    out, seen = [], set(avoid)
    while len(out) < n:
        u = ("N", rng.randrange(nr.L), rng.randrange(nr.D), rng.choice((1, -1)), None)
        if u[:4] not in seen:
            seen.add(u[:4])
            out.append(u)
    return out


def pilot(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    nr = nb.Neurons(model, tok)
    nr.load(a.thr)
    nr.thr = None  # the model itself; the T2 file gives the massive cut
    panel = Panel(a.phenome, nr, a.panel)
    t0 = time.time()
    base = panel.run()
    sec = time.time() - t0
    again = panel.run()
    names = sorted(base)
    out = {"model": a.model, "panel_per_trait": a.panel, "sequences": len(panel.seqs), "traits": names, "seconds_per_pass": sec,
           "exact": max(abs(base[n] - again[n]) for n in names), "base": base, "sizes": {}}
    rng = random.Random(a.seed)

    def change(us) -> dict:
        e, r = cc.knockout_edit(nr, us)
        got = panel.run(e, r)
        return {n: got[n] - base[n] for n in names}

    for size in a.sizes:
        out["sizes"][str(size)] = [change(random_poles(nr, size, rng)) for _ in range(a.sets)]
        sd = [statistics_sd([d[n] for d in out["sizes"][str(size)]]) for n in names]
        print(f"size {size}: median over traits of the change's sd across sets {sorted(sd)[len(sd) // 2]:.4f}, "
              f"{time.time() - t0:.0f} s", flush=True)
    mid = a.sizes[len(a.sizes) // 2]
    s1 = random_poles(nr, mid, rng)
    s2 = random_poles(nr, mid, rng, avoid={u[:4] for u in s1})
    d1, d2, d12 = change(s1), change(s2), change(s1 + s2)
    out["additivity"] = {"size": mid, "d1": d1, "d2": d2, "d12": d12}
    if a.assoc:
        summ = json.loads(a.assoc.read_text())
        ranked = sorted((n for n in names if n in summ), key=lambda n: -summ[n]["hits"])[:10]
        singles = []
        for n in ranked:
            for h in summ[n]["top"][:SINGLES]:
                pole = 1 if h["measure"].endswith("+") else -1
                d = change([("N", h["layer"], h["neuron"], pole, None)])
                singles.append({"trait": n, **h, "change": d})
        out["singles"] = singles
    a.out.write_text(json.dumps(out))
    print(f"pilot written to {a.out}; {sec:.1f} s a pass, exact to {out['exact']:.2e}")


def statistics_sd(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("phenome", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--thr", type=Path, required=True)
    p.add_argument("--assoc", type=Path)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--sets", type=int, default=SETS)
    p.add_argument("--panel", type=int, default=PANEL)
    p.add_argument("--seed", type=int, default=15)
    p.add_argument("--device")
    a = ap.parse_args()
    {"pilot": pilot}[a.cmd](a)


if __name__ == "__main__":
    main()
