"""Contrasts and decision tokens for the persona circuit search, on a GPU with vLLM (counterbench's Engine).

For every pair of prompts.py (one question under an assistant system prompt and under a character system prompt of
equal token count) both greedy replies are generated. Their longest common token prefix ends where the two identities
first produce different tokens: that point is the pair's decision token, and the circuit search reads the answer
contrast there (both prompts with the shared prefix appended). A pair is valid for the search when the assistant reply
presents its speaker as an AI and the character reply does not (score.py).

Baseline rates: for each question and frame (no system prompt; the pair's assistant prompt; its character prompt), N
sampled replies (temperature 1) and the share that present the speaker as an AI.

  contrast.py PROMPTS.json OUT_DIR [--model Qwen/Qwen3-8B] [--n 16] [--rate-chars 4] [--max-tokens 64]
writes OUT_DIR/decisions.json and OUT_DIR/rates.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "counterbench"))
import prompts as P  # noqa: E402
import score  # noqa: E402


def generate(e, ids_list: list, n: int, temperature: float, max_tokens: int) -> list:
    from vllm import SamplingParams
    sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, top_k=-1, max_tokens=max_tokens)
    outs = e.llm.generate([{"prompt_token_ids": ids} for ids in ids_list], sp, use_tqdm=False)
    return [[(list(o.token_ids), o.text) for o in r.outputs] for r in outs]


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--rate-chars", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    from engine import Engine

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(a.prompts).read_text())
    e = Engine(a.model, max_len=2048)
    tok = e.tok
    t0 = time.time()
    pairs = doc["search_pairs"] + doc["verify_pairs"]
    ids_a = [P.chat_ids(tok, p["system_asst"], p["user"]) for p in pairs]
    ids_c = [P.chat_ids(tok, p["system_char"], p["user"]) for p in pairs]
    ga = generate(e, ids_a, 1, 0.0, a.max_tokens)
    gc = generate(e, ids_c, 1, 0.0, a.max_tokens)
    print(f"greedy {2 * len(pairs)} replies {time.time() - t0:.0f} s", flush=True)
    dec = []
    for p, xa, xc, ra, rc in zip(pairs, ids_a, ids_c, ga, gc):
        (ta, sa), (tc, sc) = ra[0], rc[0]
        d = 0
        while d < min(len(ta), len(tc)) and ta[d] == tc[d]:
            d += 1
        valid = score.assistant(sa) and not score.assistant(sc) and d < min(len(ta), len(tc)) and len(xa) == len(xc)
        dec.append({**p, "reply_asst": sa, "reply_char": sc, "prefix_ids": ta[:d], "prefix": tok.decode(ta[:d]),
                    "next_asst": tok.decode([ta[d]]) if d < len(ta) else None,
                    "next_char": tok.decode([tc[d]]) if d < len(tc) else None, "depth": d,
                    "asst_is_ai": score.assistant(sa), "char_is_ai": score.assistant(sc), "valid": bool(valid),
                    "clean_ids": xa + ta[:d], "corrupt_ids": xc + ta[:d]})
    (out / "decisions.json").write_text(json.dumps(dec, ensure_ascii=False))
    nv = sum(d["valid"] for d in dec)
    print(f"decisions: {nv}/{len(dec)} valid", flush=True)

    # baseline rates: no system prompt, the assistant prompt, the character prompt (rate-chars characters a question)
    rows, seen = [], {}
    for p in pairs:
        k = seen.setdefault(p["question"], 0)
        if k >= a.rate_chars:
            continue
        seen[p["question"]] = k + 1
        if k == 0:
            rows.append((p["question"], "none", None, p["user"]))
        rows.append((p["question"], "asst", p["system_asst"], p["user"]))
        rows.append((p["question"], f"char:{p['character']}", p["system_char"], p["user"]))
    for q in doc["questions"]:
        if q["id"] not in seen:
            rows.append((q["id"], "none", None, q["text"]))
    t1 = time.time()
    got = generate(e, [P.chat_ids(tok, s, u) for _, _, s, u in rows], a.n, 1.0, a.max_tokens * 2)
    print(f"sampled {len(rows) * a.n} replies {time.time() - t1:.0f} s", flush=True)
    rates = [{"question": q, "frame": f, "system": s, "user": u, "n": len(r),
              "ai_rate": sum(score.assistant(t) for _, t in r) / len(r), "replies": [t for _, t in r]}
             for (q, f, s, u), r in zip(rows, got)]
    (out / "rates.json").write_text(json.dumps(rates, ensure_ascii=False))
    by = {}
    for r in rates:
        by.setdefault(r["frame"].split(":")[0], []).append(r["ai_rate"])
    print("AI-identity rate by frame: " + ", ".join(f"{k} {sum(v) / len(v):.3f} (n={len(v)})" for k, v in by.items()), flush=True)
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
