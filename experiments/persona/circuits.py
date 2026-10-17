"""Persona circuits on Qwen3-8B: circuit_masks' search at every valid decision token of contrast.py.

A question is one clean prompt (the assistant system prompt, the user's question and the shared reply prefix up to the
decision token) against a corrupted prompt (a character system prompt of equal length, the same question and prefix).
The answer contrast is answer_contrast's at the last position: the assistant's next word against the character's.
Pairs with the same clean prompt and corrupted prompts of its length are one question: the first pair's character is
the corrupted key and the others are partners, so with --partners above 1 the source cut is chosen over several
characters (circuit_masks' multi-partner search) and does not only separate the assistant from one character.

  circuits.py DECISIONS.json OUT_DIR [--model Qwen/Qwen3-8B] [--partners 4] [--batch 8] [--split train]
writes OUT_DIR/<id>.json per question (circuit_masks' record, plus the pair's question, character and prefix).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
import circuit_finder as cf  # noqa: E402
import circuit_masks as cm  # noqa: E402


def questions(decisions: list, split: str, partners: int) -> list:
    """[(record id, clean ids, corrupt ids, [partner corrupt ids], info)] grouped by clean prompt."""
    groups = {}
    for d in decisions:
        if d["valid"] and d["split"] == split:
            groups.setdefault(tuple(d["clean_ids"]), []).append(d)
    out = []
    for clean, ds in groups.items():
        ds = [d for d in ds if len(d["corrupt_ids"]) == len(clean)]
        if not ds:
            continue
        d0 = ds[0]
        out.append((d0["id"], list(clean), d0["corrupt_ids"], [d["corrupt_ids"] for d in ds[1:partners]],
                    {"question": d0["question"], "family": d0["family"], "character": d0["character"],
                     "partner_characters": [d["character"] for d in ds[1:partners]], "prefix": d0["prefix"],
                     "user": d0["user"], "system_asst": d0["system_asst"], "system_char": d0["system_char"],
                     "next_asst": d0["next_asst"], "next_char": d0["next_char"]}))
    return out


def main():
    import argparse

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("decisions")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--partners", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cuts", default="readout,movers,source")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--skip", default=None, help="a directory of records already found (their questions skipped)")
    a = ap.parse_args()
    cf._threads()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qs = questions(json.loads(Path(a.decisions).read_text()), a.split, a.partners)
    done = {p.name for p in Path(a.skip).glob("*.json")} if a.skip else set()
    qs = [q for q in qs if not (out / f"{q[0].replace('/', '__')}.json").exists() and f"{q[0].replace('/', '__')}.json" not in done]
    qs.sort(key=lambda q: len(q[1]))                 # (batches of one prompt length)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    model.requires_grad_(False)
    pts = [cf.Patcher(model, tok) for _ in range(a.batch)]
    pool = [[cf.Patcher(model, tok) for _ in range(max(0, a.partners - 1))] for _ in range(a.batch)]
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    log(f"{len(qs)} questions")
    if qs:
        pts[0].setup(qs[0][1], qs[0][2])            # (a throwaway run: CUDA's first kernels)
    for c0 in range(0, len(qs), a.batch):
        chunk = qs[c0:c0 + a.batch]
        recs = cm.find_batch(pts[:len(chunk)], [(cl, co, None) for _, cl, co, _, _ in chunk], arm="mask", log=log,
                             partners=[list(zip(pool[b], ps)) for b, (_, _, _, ps, _) in enumerate(chunk)],
                             cuts_wanted=tuple(a.cuts.split(",")))
        for (qid, cl, co, ps, info), rec in zip(chunk, recs):
            rec.update(id=qid, ids=cl, corrupted=co, answer_text=tok.decode([rec["answer"]]),
                       corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, dtype="bfloat16",
                       finder="circuit_masks", finder_rev=a.rev, partners=len(ps), **info)
            (out / f"{qid.replace('/', '__')}.json").write_text(json.dumps(rec))
            log(f"{qid}: met {rec.get('met')} size {rec.get('size')} answer {rec['answer_text']!r} vs {rec['corrupt_answer_text']!r}")


if __name__ == "__main__":
    main()
