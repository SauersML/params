"""The learnability grid's scale check on Tinker (Qwen3-8B, method/oracle/rl/tinker_train.py): one arm's training set
as chat SFT rows with short answers, and the held-out questions as one scored row per choice, then the scores in
learn_sft.py score's form (learn_sft.py table reads them beside the 1.7B arms).

Answers are short so a choice's log-probability is the whole completion's (Tinker's score counts the end of the turn):
remove, cut and swap answer "KL <bin>" (the KL-bin question alone; falls and top are not asked), rank answers the
letter of the candidate whose removal changes the prediction most; the rank order is the candidates by that letter's
probability (top-1 its argmax, Kendall's tau against the measured KLs).

  learn_tinker.py rows QDIR/ARM OUT_DIR [--size 8000]     OUT_DIR/sft.jsonl, OUT_DIR/score_rows.jsonl
  learn_tinker.py collect OUT_DIR/score_rows.jsonl SCORED.jsonl QDIR/ARM/heldout.jsonl OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BIN_TEXT = ("under 0.01 bits", "0.01 to 0.1 bits", "0.1 to 1 bit", "over 1 bit")
LETTERS = "abcdefgh"
ASK_BIN = "\nAnswer with the KL bin alone: KL under 0.01 bits, KL 0.01 to 0.1 bits, KL 0.1 to 1 bit, or KL over 1 bit."
ASK_TOP = "\nAnswer with the letter of the candidate whose removal changes the prediction most, alone."


def _rows(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def _chat(q: str, a: str, info: dict) -> dict:
    return {"prompt": [{"role": "user", "content": q}], "completion": [{"role": "assistant", "content": a}], "info": info}


def rows(arm_dir: Path, out: Path, size: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sft = []
    for r in _rows(arm_dir / f"train_{size}.jsonl"):
        if r["meta"]["kind"] == "rank":
            sft.append(_chat(r["prompt"] + ASK_TOP, LETTERS[r["label"]], {"kind": "rank"}))
        else:
            sft.append(_chat(r["prompt"] + ASK_BIN, f"KL {BIN_TEXT[r['label']]}", {"kind": r["meta"]["kind"]}))
    (out / "sft.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sft))
    sc = []
    for i, r in enumerate(_rows(arm_dir / "heldout.jsonl")):
        if r["meta"]["kind"] == "rank":
            sc += [_chat(r["prompt"] + ASK_TOP, x, {"q": i, "choice": j}) for j, x in enumerate(LETTERS)]
        else:
            sc += [_chat(r["prompt"] + ASK_BIN, f"KL {b}", {"q": i, "choice": j}) for j, b in enumerate(BIN_TEXT)]
    (out / "score_rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sc))
    print(f"{len(sft)} SFT rows, {len(sc)} score rows")


def collect(score_rows: Path, scored: Path, held: Path, out: Path) -> None:
    import torch
    from scipy.stats import kendalltau

    info = [r["info"] for r in _rows(score_rows)]
    lp = {}
    for s in _rows(scored):
        i = info[s["index"]]
        lp.setdefault(i["q"], {})[i["choice"]] = s["logprob"]
    res = []
    for q, r in enumerate(_rows(held)):
        got = lp.get(q)
        if not got:
            continue
        p = torch.tensor([got[j] for j in range(len(got))]).log_softmax(0)
        if r["meta"]["kind"] == "rank":
            tau = kendalltau(p.tolist(), r["meta"]["kls"]).statistic
            res.append({"kind": "rank", "label": r["label"], "logloss_bits": -float(p[r["label"]]) / math.log(2), "pred": int(p.argmax()),
                        "tau": float(tau)})
        else:
            res.append({"kind": r["meta"]["kind"], "label": r["label"], "logloss_bits": -float(p[r["label"]]) / math.log(2),
                        "pred": int(p.argmax()), "falls_logloss_bits": float("nan"), "p_bins": p.exp().tolist()})
    out.write_text("".join(json.dumps(x) + "\n" for x in res))
    print(f"{len(res)} questions scored")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("arm_dir", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--size", type=int, default=8000)
    c = sub.add_parser("collect")
    c.add_argument("score_rows", type=Path)
    c.add_argument("scored", type=Path)
    c.add_argument("held", type=Path)
    c.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.cmd == "rows":
        return rows(a.arm_dir, a.out, a.size)
    return collect(a.score_rows, a.scored, a.held, a.out)


if __name__ == "__main__":
    main()
