"""V2 through the circuit finder: copy texts (copy_circuit.make_texts, raw text, no chat template) as patching pairs for
circuit_finder.find, the movers cut's route groups where copying spreads over heads, and every circuit's edges
(neuron_edges.finder_graph). One record per usable text (the copied word the clean run's top, the gap at least
copy_circuit.GAP_MIN bits).

  python experiments/oracle/copy_finder.py OUT [--model Qwen/Qwen3-8B] [--dtype bfloat16] [--n 24]
writes OUT/<id>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))
import circuit_finder as cf  # noqa: E402
import copy_circuit  # noqa: E402
import neuron_edges  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--edges", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    import transformers
    a.out.mkdir(parents=True, exist_ok=True)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto")
    model = model.eval().requires_grad_(False)
    pt = cf.Patcher(model, tok)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)  # noqa: E731
    first = True
    for t in copy_circuit.make_texts(tok, a.n):
        path = a.out / f"{t['id']}.json"
        if path.exists():
            continue
        if first:  # (a throwaway run: CUDA's first kernels)
            pt.setup(t["ids"], t["corrupt_ids"])
            first = False
        log(t["id"])
        rec = cf.find(pt, t["ids"], t["corrupt_ids"], log=log)
        rec.update(id=t["id"], ids=t["ids"], corrupted=t["corrupt_ids"], text=t["text"], template=t["template"],
                   copy_positions=t["pos"], word=t["word"], model=a.model, dtype=a.dtype, finder="circuit_finder")
        if rec.get("answer") is not None:
            rec["answer_text"] = tok.decode([rec["answer"]])
        if a.edges and not rec.get("skipped") and (rec.get("nodes") or rec.get("routes")):
            t1 = time.time()
            g = neuron_edges.finder_graph(model, tok, rec, log=log)
            rec["graph"] = {k: g.get(k) for k in ("nodes", "edges", "head_edges", "routes", "record", "cut", "route_note")}
            rec["seconds_edges"] = round(time.time() - t1, 2)
            neuron_edges.free()
        path.write_text(json.dumps(rec))
        m = rec.get("cuts", {}).get("movers", {})
        log(f"{t['id']}: met {rec.get('met')}, movers {m.get('size')} ({'route groups' if m.get('route_groups') else 'singles'}), "
            f"{rec.get('seconds')} s + edges {rec.get('seconds_edges')} s")


if __name__ == "__main__":
    main()
