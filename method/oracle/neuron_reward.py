"""The oracle's reward service (the lead's 10-15): every claim an answer makes about a model's neurons, checked on the
model itself. Two claim kinds, each owned by one module so nothing is computed twice:
  explanation  {"kind": "explanation", "unit": "L18.N2201+", "explanation": str[, "corpus_max": float]}
               what drives one pole of a neuron: neuron_verify's blind maximum, near misses, distant concept and
               steering prediction (the reader writes the test texts and judges the steering pair); corpus_max defaults
               to the atlas's extreme of that pole. Passing is T1's rule: blind maximum >= FIRE and near misses < FIRE.
  intervention what zeroing, steering or patching named units (or swapping tokens) does to one answer:
               monitorbench's claim_check.check_many on the weak-zeroed reference (neuron_builder.Neurons), unchanged.
Units are named L<layer>.N<index><pole> (pole + or -; no sign: the whole neuron) and L<layer>.H<head>.

  neuron_reward.py check --claims C.jsonl --out R.jsonl --model M [--atlas ROOT] [--thr T2.pt] [--reader R | --reader-url U]
  neuron_reward.py serve --model M [--atlas ROOT] [--thr T2.pt] [--reader-url U | --reader-local M] [--port 8011]
      POST /check {"claims": [...]} -> {"results": [...]} in the claims' order; GET /health
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "rl")]

import claim_check  # noqa: E402  (its UNIT names every unit, for both kinds)
import neuron_verify  # noqa: E402


def explanation_item(claim: dict, atlas=None) -> dict:
    """A claim of kind "explanation" as neuron_verify's item {layer, index, pole, corpus_max, explanation}."""
    m = claim_check.UNIT.match(str(claim.get("unit", "")).strip())
    if not m or m[2] != "N" or not m[4]:
        raise ValueError("an explanation names one neuron pole, like L18.N2201+ or L18.N2201-")
    layer, index, pole = int(m[1]), int(m[3]), m[4]
    cm = claim.get("corpus_max")
    if cm is None:
        if atlas is None:
            raise ValueError("corpus_max not given and no atlas to read it from")
        cm = abs(atlas.neuron(layer, index, k=0, text=False)["max"][pole])
    return {"layer": layer, "index": index, "pole": pole, "corpus_max": float(cm), "explanation": claim.get("explanation")}


def explanation_result(row: dict) -> dict:
    keep = ("valid", "reason", "passes_t1", "blind_max", "blind_texts", "blind_values", "near_miss_rate", "near_values",
            "distant_concept", "distant_rate", "distant_values", "steering_prediction", "steer_match", "steered", "unsteered")
    return {"kind": "explanation", "pass": bool(row.get("valid") and row.get("passes_t1")), **{k: row[k] for k in keep if k in row}}


def check(claims: list[dict], tg, rd, nr=None, atlas=None) -> list[dict]:
    """Every claim's result, in order. Explanation claims go through neuron_verify's plan, measure and judge in one
    batch each; intervention claims through claim_check.check_many on nr. A claim that cannot be read fails with its
    reason and never raises."""
    out: list = [None] * len(claims)
    items, where = [], []
    for k, c in enumerate(claims):
        kind = c.get("kind") if isinstance(c, dict) else None
        if kind == "explanation":
            try:
                it = explanation_item(c, atlas)
                neuron_verify._item(tg, it)
            except ValueError as e:
                out[k] = {"kind": kind, "pass": False, "valid": False, "reason": str(e)}
                continue
            items.append(it)
            where.append(k)
        elif kind != "intervention":
            out[k] = {"kind": kind, "pass": False, "valid": False, "reason": "unknown claim kind"}
    if items:
        for k, row in zip(where, neuron_verify.judge(rd, neuron_verify.measure(tg, neuron_verify.plan(rd, items)))):
            out[k] = explanation_result(row)
    iv = [k for k, c in enumerate(claims) if out[k] is None]
    if iv and nr is None:
        for k in iv:
            out[k] = {"kind": "intervention", "pass": False, "valid": False, "reason": "no weak-zeroed reference (--thr)"}
    elif iv:
        for k, r in zip(iv, claim_check.check_many(nr, [claims[k] for k in iv]), strict=True):
            out[k] = {"kind": "intervention", **r}
    return out


def setup(a):
    """The target (neuron_verify.Target), the reader, and for intervention claims the weak-zeroed reference on the
    target's own weights (one copy of the model)."""
    import reader_reward

    rd = (reader_reward.VLLMReader(a.reader_local, util=0.5) if getattr(a, "reader_local", None)
          else reader_reward.OpenAIReader(a.reader_url) if a.reader_url else reader_reward.Reader(a.reader, a.log))
    tg = neuron_verify.Target(a.model, a.device)
    nr = None
    if a.thr:
        from neuron_builder import Neurons

        nr = Neurons(tg.m, tg.tok)
        nr.load(a.thr)
    atlas = None
    if a.atlas:
        sys.path.insert(0, str(HERE.parents[1] / "experiments/oracle"))
        from atlas import Atlas

        atlas = Atlas(a.atlas)
    return tg, rd, nr, atlas


def cmd_check(a) -> None:
    tg, rd, nr, atlas = setup(a)
    claims = [json.loads(x) for x in a.claims.read_text().splitlines() if x.strip()]
    rows = check(claims, tg, rd, nr, atlas)
    a.out.write_text("".join(json.dumps({**c, "result": r}) + "\n" for c, r in zip(claims, rows)))
    print(f"{sum(r['pass'] for r in rows)}/{len(rows)} claims pass to {a.out}; reader spend ${rd.spend.total:.3f}")


def cmd_serve(a) -> None:
    import threading
    import traceback
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    tg, rd, nr, atlas = setup(a)
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send({"ok": True, "model": a.model, "intervention": nr is not None, "atlas": a.atlas and str(a.atlas)})

        def do_POST(self):
            claims = json.loads(self.rfile.read(int(self.headers["Content-Length"]))).get("claims", [])
            try:
                with lock:
                    rows = check(claims, tg, rd, nr, atlas)
            except Exception as e:  # a request that breaks the checker fails every claim in it, never the service
                traceback.print_exc()
                rows = [{"pass": False, "valid": False, "error": f"{type(e).__name__}: {e}"[:300]} for _ in claims]
            self._send({"results": rows})

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"neuron reward on :{a.port} ({a.model})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    for name in ("check", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="Qwen/Qwen3-8B")
        p.add_argument("--device", help="cuda, mps, cpu, or auto (layers over every visible GPU)")
        p.add_argument("--atlas", type=Path, help="the model's atlas root: each pole's corpus maximum")
        p.add_argument("--thr", type=Path, help="the weak-zeroed reference (Neurons.save) for intervention claims")
        p.add_argument("--reader", default=neuron_verify.READER, help="a Tinker model (the Mac, with the key)")
        p.add_argument("--reader-url", help="an OpenAI-compatible server of the reader (vllm serve)")
        p.add_argument("--reader-local", help="the reader in this process with vLLM (a pod)")
        p.add_argument("--log", type=Path, default=Path("/Users/user/mpd-data/graph_oracle/neuron_reward"))
        if name == "check":
            p.add_argument("--claims", type=Path, required=True)
            p.add_argument("--out", type=Path, required=True)
        else:
            p.add_argument("--port", type=int, default=8011)
    a = ap.parse_args()
    {"check": cmd_check, "serve": cmd_serve}[a.what](a)


if __name__ == "__main__":
    main()
