"""O1p's activation-oracle baseline (the lead's 10-17): a released activation oracle reads Qwen3-8B's activations on both
texts of a paired question and answers it. Per question, each text's chat (the token ids the circuit was measured on,
pairs/v1's sides) is run through the base model with the oracle's adapter off; the residual stream at the last READ tokens,
at each of the oracle's read layers (cao.READ_LAYERS: the half-depth layer, or CAO_READ_LAYERS), is injected as one block
per layer for text A, then for text B; the question follows with its answer options (scorecard.PAIR_OPTIONS). Greedy,
parsed by scorecard.pair_class.

  ao_pairs.py --items ITEMS/o1_pairs.jsonl --adapter A --out ANS_DIR/<name>/o1p.jsonl     (GPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cao"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cao  # noqa: E402

READ, BATCH = 10, 16
OPTIONS = {"pair_zero": '"A", "B", "About the same"', "pair_sign": '"In A only", "In B only", "In both", "In neither"',
           "pair_flip": '"In A only", "In B only", "In both", "In neither"'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="the first N questions only (a check)")
    a = ap.parse_args()
    its = [json.loads(x) for x in a.items.read_text().splitlines()]
    its = its[:a.n] if a.n else its
    o = cao.Oracle(a.adapter)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for k in range(0, len(its), BATCH):
            exs = []
            for it in its[k:k + BATCH]:
                vecs, blocks = [], []
                for side in it["sides"]:
                    ids = side["prompt"]["ids"]
                    b = [{"layer": L, "positions": list(range(len(ids) - READ, len(ids)))} for L in cao.READ_LAYERS]
                    vecs.append(o.vectors({"ctx": ids, "blocks": b}))
                    blocks += b
                key = it["item"]
                o.vec_store[key] = o.torch.cat(vecs).cpu()
                exs.append({"vec": key, "blocks": blocks,
                            "q": f"The first blocks are text A's, the rest text B's. {it['q']} Answer with exactly one of: {OPTIONS[it['kind']]}."})
            got = o.answers(exs, max_new=40)
            for e, g in zip(exs, got):
                fh.write(json.dumps({"item": e["vec"], "answer": g}) + "\n")
                o.vec_store.pop(e["vec"])
            print(f"{k + len(exs)}/{len(its)}: {got[0][:80]!r}", flush=True)


if __name__ == "__main__":
    main()
