"""T2 (the lead's 10-15 brief, ~/mpd-data/team/direction_1015.txt): zeroing weak MLP activations network-wide on a Qwen3
costs at most 0.1 bits mean KL at some per-layer threshold; the strong values per token there.
The reference is neuron_builder.Neurons.calibrate's: one weak fraction per layer of each neuron's largest |value| on
general chat 1000-1199 (the organisms' general.jsonl), chosen on the KL of the first FIT of them. Checked on 200 held-out
chats (100-299): the mean KL (bits) over every position and the strong values per token, per layer and in all; beside it
the same fraction in every layer (each of neuron_builder.TAUS) and the per-layer reference with the first position (the
attention sink) kept whole.

  neuron_t2.py OUT [--model Qwen/Qwen3-1.7B] [--adapter DIR] [--dtype bfloat16] [--device auto] [--fit 80] [--len 256]
writes OUT/thr.pt (Neurons.save: neuron_builder --thr and neuron_scorecard's thr file) and OUT/t2.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))

import neuron_builder as nb  # noqa: E402

GENERAL = Path.home() / "mpd-data" / "organisms" / "qwen3_1p7b" / "data" / "general.jsonl"
HELD = slice(100, 300)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--general", type=Path, default=GENERAL)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default="cuda", help="cuda, cpu or auto (the layers over every visible GPU)")
    ap.add_argument("--fit", type=int, default=nb.FIT)
    ap.add_argument("--len", type=int, default=256, help="tokens per chat")
    ap.add_argument("--rows", type=int, default=16)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto" if a.device == "auto" else None)
    model = (model if a.device == "auto" else model.to(a.device)).eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
    nr = nb.Neurons(model, tok)
    gen = [json.loads(x) for x in a.general.read_text().splitlines()]

    def seq(r):
        return tok(tok.apply_chat_template([{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["answer"]}],
                                           tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"][:a.len]
    cal_seqs, held = [seq(r) for r in gen[nb.CAL_ROWS]], [seq(r) for r in gen[HELD]]
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cal = nr.calibrate(cal_seqs, fit=cal_seqs[:a.fit], rows=a.rows, log=lambda m: print(f"{time.time() - t0:.0f}s {m}", flush=True))
    nr.save(a.out / "thr.pt", calibration=cal, model=a.model, adapter=a.adapter)
    names = ["per_layer", "per_layer_sink_live"] + [f"global_{t}" for t in nb.TAUS]
    fr = torch.tensor([nr.tau, nr.tau] + [[t] * nr.L for t in nb.TAUS], dtype=torch.float64)
    kl, st = nr.weak_runs(held, fr, live0=[False, True] + [False] * len(nb.TAUS), rows=a.rows)
    rows = [{"name": n, "tau": [float(x) for x in f], "kl": float(k), "strong_per_token": float(s.sum()),
             "share": float(s.sum()) / (nr.L * nr.D), "per_layer": [round(float(x), 1) for x in s]}
            for n, f, k, s in zip(names, fr, kl, st)]
    for r in rows:
        print(f"{r['name']:>22}: KL {r['kl']:.4f} bits, {r['strong_per_token']:.0f} strong per token ({r['share']:.3f})", flush=True)
    (a.out / "t2.json").write_text(json.dumps({"model": a.model, "adapter": a.adapter, "dtype": a.dtype, "calibration": cal,
                                               "held_out": rows, "held_texts": len(held), "held_tokens": sum(map(len, held)),
                                               "neurons": nr.L * nr.D, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
