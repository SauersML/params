"""The neuron-grounded oracle's examples (#2951 G2; the lead's 10-15 direction, decision 6). Beside a text's half-depth
activations, the oracle reads the writes of the neurons that a question-scoped circuit (method/oracle/neuron_builder.py:
the answer contrast on the weak-zeroed reference, removal = zeroing one neuron at one position) holds, each labelled with
its ID, pole and position, and answers questions whose answers are measured removals on that reference:

  n_top     which listed neuron, zeroed alone, lowers the answer's contrast most
  n_effect  how much one listed neuron's removal changes the answer's contrast (bits, to the nearest half)
  n_set     whether the model keeps its answer with every listed circuit neuron zeroed, else what it says instead
  n_sign    whether a listed neuron supports the answer or works against it (the sign of its measured removal)

Listed neurons: the circuit's (necessity, then sufficiency), the builder's strongest measured singles beyond it, up to
K, and FOILS strong neurons at the answer position outside both (most move nothing), shuffled. Every listed neuron is
measured alone (the builder's singles reused; the rest one run each). A label reads "L<layer>.N<neuron>+ at <position>"
("-" for the negative pole). The vectors, stored once per text in OUT.pt under the text's key ("vec"): the half-depth
residual at every position (the Activation Oracle's read, adapter off) and each listed neuron's write (its weak-zeroed
value x its down_proj column) in prompt order. Texts: G1's wikitext-103 training windows (cao.gen's), the answer the
model's next token after the window.

  g2.py gen --out EX.jsonl --thr THR.pt [--texts N] [--seed S] [--necessity-only]   (GPU; EX.pt beside EX.jsonl; resumes)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "method" / "oracle"))
import cao  # noqa: E402
import h2h  # noqa: E402  (cao puts aoh2h on the path)

K = 10       # listed circuit and measured neurons
FOILS = 2    # listed strong neurons at the answer position outside the circuit and the ranking
HALF = 0.25  # bits: a removal moving the contrast less than this is "almost nothing"
QUESTIONS = 6  # per text: one n_top, one n_set, the rest n_effect and n_sign


def label(u: tuple, pole: int) -> str:
    i, t, n = u
    return f"L{i}.N{n}{'+' if pole > 0 else '-'} at {t}"


def bits(x: float) -> str:
    v = round(abs(x) * 2) / 2
    return f"{v:g} bit" + ("" if v == 1 else "s")


def text_examples(o: "cao.Oracle", nr, ids: list[int], rng: random.Random, key: str) -> tuple[list[dict], "torch.Tensor", dict]:
    import neuron_builder as nb
    torch = o.torch
    pos = len(ids) - 1
    rec = nb.build(nr, ids, pos)
    y, alt = rec["answer"], rec["alternatives"]
    altt = torch.tensor(alt, device=nr.dev)
    cap = {}
    with torch.no_grad():
        lp0 = torch.log_softmax(nr.logits(ids, [pos], capture=cap)[:, 0], -1)
    H = [h[0] for h in cap["h"]]
    c0 = float((lp0[0, y] - torch.logsumexp(lp0[0, altt], -1)) / math.log(2))

    def alone(units: list) -> list[tuple[float, int]]:
        """Per unit zeroed alone: the contrast's drop (bits) and the top token."""
        out = []
        for s in range(0, len(units), nb.ROWS):
            part = [[u] for u in units[s:s + nb.ROWS]]
            with torch.no_grad():
                l = torch.log_softmax(nr.logits(ids, [pos], edit=nb._zero_edit(nb._pad(part)), B=nb.ROWS)[:, 0], -1)
            c = (l[:, y] - torch.logsumexp(l[:, altt], -1)) / math.log(2)
            out += [(c0 - float(c[b]), int(l[b].argmax())) for b in range(len(part))]
        return out
    circ = [(d["layer"], d["pos"], d["neuron"]) for d in rec["nodes"]]
    ranked = [(d["layer"], d["pos"], d["neuron"]) for d in rec["ranked_top"]]
    listed = list(dict.fromkeys(circ + ranked))[:K]
    strong = [(i, pos, int(n)) for i in range(nr.L) for n in ((H[i][pos] != 0) & (H[i][pos].abs() < nr.massive[i])).nonzero()[:, 0].tolist()]
    strong = [u for u in strong if u not in set(listed) and u not in set(ranked)]
    listed += rng.sample(strong, min(FOILS, len(strong)))
    rng.shuffle(listed)
    eff = dict(zip(listed, [d for d, _ in alone(listed)]))
    pole = {u: 1 if float(H[u[0]][u[1], u[2]]) > 0 else -1 for u in listed}
    labs = {u: label(u, pole[u]) for u in listed}
    with torch.no_grad():  # the vectors: the half-depth read (the model itself) and each listed neuron's write
        with o.off():
            Hm = o.m(input_ids=torch.tensor([ids], device=o.dev), output_hidden_states=True).hidden_states[cao.MID + 1][0]
        W = [o.layers[i].mlp.down_proj.weight[:, n].float() * float(H[i][t, n]) for i, t, n in listed]
    vec = torch.cat([Hm.float(), torch.stack(W)]).to(torch.float16).cpu()
    blocks = [{"layer": cao.MID, "positions": list(range(len(ids)))}] + [{"label": labs[u]} for u in listed]
    ytok = cao.q_tok(o.tok.decode([y]))
    base = {"ctx": ids, "blocks": blocks, "vec": key}
    out = []
    top = max(listed, key=lambda u: eff[u])
    out.append({**base, "cat": "n_top", "q": f"The model's next token is {ytok}. Which listed neuron, zeroed alone, lowers it most "
                                             f"against its alternatives?",
                "a": f"{labs[top]}, by about {bits(eff[top])}." if eff[top] >= HALF else cao.NONE, "units": [list(top)]})
    zc = [u for u in listed if u in set(circ)]
    if zc:
        with torch.no_grad():
            l = nr.logits(ids, [pos], edit=nb._zero_edit([zc]))[0, 0]
        z = int(l.argmax())
        out.append({**base, "cat": "n_set", "q": f"If {', '.join(labs[u] for u in zc)} were {'all ' if len(zc) > 1 else ''}zeroed, would the model still "
                                                 f"give {ytok}? If not, what would it give?",
                    "a": "Yes." if z == y else f"No, {cao.q_tok(o.tok.decode([z]))}.", "units": [list(u) for u in zc]})
    rest = [u for u in listed if u != top]
    rng.shuffle(rest)
    for k, u in enumerate(rest[:QUESTIONS - len(out)]):
        e = eff[u]
        if k % 2 == 0:
            out.append({**base, "cat": "n_effect", "q": f"If {labs[u]} were zeroed, how would the model's next token {ytok} fare "
                                                        f"against its alternatives?",
                        "a": "Almost no change." if abs(e) < HALF else (f"It would lose about {bits(e)}." if e > 0 else
                                                                         f"It would gain about {bits(e)}."), "units": [list(u)]})
        else:
            out.append({**base, "cat": "n_sign", "q": f"Does {labs[u]} support the model's next token {ytok} or work against it?",
                        "a": "Neither, it barely matters." if abs(e) < HALF else ("It supports it." if e > 0 else "It works against it."),
                        "units": [list(u)]})
    info = {"size": rec["size"], "met": rec["met"], "contrast": round(c0, 2), "build_seconds": rec["seconds"], "listed": len(listed)}
    return out, vec, info


def gen(a) -> None:
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download

    import neuron_builder as nb
    nb.SUFFICIENCY = nb.SUFFICIENCY and not a.necessity_only
    o = cao.Oracle(None, a.device)
    nr = nb.Neurons(o.m, o.tok)
    nr.load(a.thr)
    rng = random.Random(a.seed)
    df = pd.read_parquet(hf_hub_download(cao.WIKITEXT_TRAIN[0], cao.WIKITEXT_TRAIN[1], repo_type="dataset"))
    paras = [t.strip() for t in df["text"].tolist() if len(t.strip()) > 200 and not t.strip().startswith("=")]
    rng.shuffle(paras)
    vpath = a.out.with_suffix(".pt")
    store = torch.load(vpath) if vpath.exists() else {}
    n, t0 = 0, time.time()
    with a.out.open("a") as fh:
        for t in paras:
            if n >= a.texts:
                break
            ids = o.tok(" " + t, add_special_tokens=False)["input_ids"]
            if len(ids) < h2h.T + 8:
                continue
            s = rng.randrange(0, len(ids) - h2h.T)
            key = f"g2_{a.seed}_{n}"
            n += 1
            if key in store:
                continue
            exs, vec, info = text_examples(o, nr, ids[s:s + h2h.T], rng, key)
            store[key] = vec
            for e in exs:
                fh.write(json.dumps({**e, "text": 1_000_000 * a.seed + n - 1}) + "\n")
            if n % 10 == 0:
                fh.flush()
                torch.save(store, vpath)
                print(json.dumps({"texts": n, "seconds": round(time.time() - t0), **info}), flush=True)
    torch.save(store, vpath)
    print(f"{n} texts to {a.out} and {vpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--thr", type=Path, required=True, help="the weak-zeroed reference (neuron_builder.Neurons.save)")
    g.add_argument("--texts", type=int, default=100)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--device")
    g.add_argument("--necessity-only", action="store_true", help="the builder's necessity set alone (its sufficiency search "
                                                                  "runs in float64, which MPS lacks)")
    a = ap.parse_args()
    {"gen": gen}[a.cmd](a)


if __name__ == "__main__":
    main()
