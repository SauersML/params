"""The oracle's reward server (#2951; the lead's 10-16 oracle push: claim_check on a GPU, batched and fast, for the RL of
oracleagent and textoracle). Intervention claims in claim_check's format (prompt, position, units, op, swap, answer,
alternatives, predict; optionally "adapter", the name of an organism adapter the server loaded) are checked on the
weak-zeroed reference, with the answer contrast at word level: a claim that gives no answer set gets the circuit
finder's (answer_contrast.token_sets on the clean run at its position, cached per prompt), so steering and zeroing are
measured on the answer word, not on spellings of it. Each result carries claim_check's numbers and the reward terms:

  pass       the claim's prediction holds
  specific   it holds and no random control set of as many units in the same layers also makes it hold
  direction  the stated change's sign is the measured one's (a stated "about"/value; |measured| < NONE bits is no
             change, which only a "within" claim states)
  magnitude  the stated number is within max(MAG_ABS, MAG_REL x |measured|) bits of the measured change (textoracle's
             10-16: most claims with the right direction that fail state too large a change); "about" when the claim
             gives it, else the predicate's value pushed out by MARGIN (g2's "about N bits" is stated as >= N - MARGIN)
  word_level whether the contrast was the answer word's (False: no other word among the candidates, token level)

Requests are pooled: one worker takes every claim queued within WAIT seconds (at most MAX_CLAIMS), groups them by
adapter, and runs one claim_check.check_many per group, its rows in batches of --rows.

  reward_server.py serve --model Qwen/Qwen3-8B --thr T2.pt [--adapter NAME=PATH ...] [--port 8701] [--rows 32]
  reward_server.py check URL CLAIMS.jsonl OUT.jsonl        (a client: POST /check in chunks)
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

WAIT, MAX_CLAIMS = 0.05, 512
NONE, MARGIN = 0.25, 0.25
MAG_ABS, MAG_REL = 0.5, 0.5


def stated(pred: dict) -> float | None:
    """The change in bits a prediction states (None for top / top_changes predictions)."""
    c = pred.get("contrast_change")
    if not c:
        return None
    if "about" in c:
        return float(c["about"])
    v = float(c["value"])
    if c["op"] == "within":
        return 0.0
    return v + MARGIN if c["op"] in (">=", ">") and v > 0 else v - MARGIN if c["op"] in ("<=", "<") and v < 0 else v


def terms(pred: dict, r: dict) -> dict:
    """The reward terms of one checked claim (module docstring)."""
    out = {"pass": bool(r.get("pass")), "specific": bool(r.get("specific"))}
    s, ch = stated(pred), r.get("change")
    if s is None or ch is None:
        return {**out, "direction": None, "magnitude": None}
    sign = lambda x: 0 if abs(x) < NONE else (1 if x > 0 else -1)  # noqa: E731
    return {**out, "direction": sign(s) == sign(ch), "magnitude": abs(s - ch) <= max(MAG_ABS, MAG_REL * abs(ch)),
            "stated": s}


class Checker:
    """The model, its weak-zeroed reference and any organism adapters, on one device; check(claims) -> results."""

    def __init__(self, model: str, thr: Path, adapters: dict, rows: int, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import claim_check as cc
        import neuron_builder as nb
        self.torch, self.cc, self.nb = torch, cc, nb
        cc.ROWS = rows
        dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev).eval()
        self.adapters = dict(adapters)
        if adapters:
            from peft import PeftModel
            names = list(adapters)
            m = PeftModel.from_pretrained(m, adapters[names[0]], adapter_name=names[0])
            for n in names[1:]:
                m.load_adapter(adapters[n], adapter_name=n)
            m.eval()
        m.requires_grad_(False)
        self.m = m
        self.nr = nb.Neurons(m, self.tok)
        self.nr.load(thr)
        self.sets: dict = {}  # (adapter, prompt ids up to the position) -> answer_contrast's token sets

    def _word_sets(self, ids: list[int]) -> dict:
        import answer_contrast as ac
        torch = self.torch
        with torch.no_grad():
            lp = torch.log_softmax(self.nr.logits(ids, [len(ids) - 1])[0, 0], -1)
            return ac.token_sets(self.nr.core, self.tok, ids, lp)

    def _adapter(self, name: str | None):
        import contextlib
        if not self.adapters:
            if name:
                raise KeyError(f"no adapter {name!r} loaded")
            return contextlib.nullcontext()
        if name is None:
            return self.m.disable_adapter()
        self.m.set_adapter(name)
        return contextlib.nullcontext()

    def check(self, claims: list[dict]) -> list[dict]:
        cc = self.cc
        out: list = [None] * len(claims)
        by: dict = {}
        for k, c in enumerate(claims):
            by.setdefault(c.get("adapter"), []).append(k)
        for name, ks in by.items():
            try:
                ctx = self._adapter(name)
            except KeyError as e:
                for k in ks:
                    out[k] = {"pass": False, "specific": False, "error": str(e)}
                continue
            with ctx:
                prepared, word = [], []
                for k in ks:
                    c = dict(claims[k])
                    wl = None
                    if c.get("answer") is None and "prompt" in c:
                        try:
                            ids = cc.prompt_ids(self.nr, c["prompt"])
                            pos = int(c.get("position", -1))
                            pos = len(ids) + pos if pos < 0 else pos
                            key = (name, tuple(ids[:pos + 1]))
                            if key not in self.sets:
                                if len(self.sets) > 100_000:  # (a long session's cache bounded)
                                    self.sets.clear()
                                self.sets[key] = self._word_sets(ids[:pos + 1])
                            s = self.sets[key]
                            wl = bool(s["valid"])
                            if wl:
                                c["answer"], c["alternatives"] = s["answer"], s["alternatives"]
                        except Exception as e:  # (an unreadable prompt fails in claim_check with its reason)
                            wl = f"{type(e).__name__}: {e}"
                    prepared.append(c)
                    word.append(wl)
                res = cc.check_many(self.nr, prepared)
                for k, c, r, wl in zip(ks, prepared, res, word):
                    out[k] = {**r, **(terms(c.get("predict", {}), r) if "error" not in r else {}), "word_level": wl}
        return out


def serve(a) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    adapters = dict(x.split("=", 1) for x in a.adapter)
    ck = Checker(a.model, a.thr, adapters, a.rows)
    jobs: queue.Queue = queue.Queue()
    stats = {"claims": 0, "seconds": 0.0, "batches": 0}

    def worker():
        while True:
            first = jobs.get()
            batch, t0 = [first], time.time()
            n = len(first[0])
            while n < MAX_CLAIMS and time.time() - t0 < WAIT:
                try:
                    j = jobs.get(timeout=max(0.0, WAIT - (time.time() - t0)))
                except queue.Empty:
                    break
                batch.append(j)
                n += len(j[0])
            flat = [c for claims, _, _ in batch for c in claims]
            t1 = time.time()
            try:
                res = ck.check(flat)
            except Exception as e:  # (a failed batch answers every claim in it with the error)
                res = [{"pass": False, "specific": False, "error": f"{type(e).__name__}: {e}"}] * len(flat)
            stats["claims"] += len(flat)
            stats["seconds"] += time.time() - t1
            stats["batches"] += 1
            i = 0
            for claims, slot, done in batch:
                slot.extend(res[i:i + len(claims)])
                i += len(claims)
                done.set()

    threading.Thread(target=worker, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send({"ok": True, "model": a.model, "adapters": list(adapters), "rows": a.rows, **stats,
                        "claims_per_second": stats["claims"] / max(stats["seconds"], 1e-9)})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                claims = body["claims"]
            except Exception as e:
                return self._send({"error": f"{type(e).__name__}: {e}"}, 400)
            slot, done = [], threading.Event()
            jobs.put((claims, slot, done))
            done.wait()
            self._send({"results": slot})

        def log_message(self, *args):
            pass

    print(f"reward server on :{a.port} ({a.model}, adapters {list(adapters)}, rows {a.rows})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


def check(a) -> None:
    import urllib.request
    claims = [json.loads(x) for x in a.claims.read_text().splitlines() if x.strip()]
    with a.out.open("w") as fh:
        for k in range(0, len(claims), a.chunk):
            req = urllib.request.Request(a.url.rstrip("/") + "/check", data=json.dumps({"claims": claims[k:k + a.chunk]}).encode(),
                                         headers={"Content-Type": "application/json"})
            for r in json.loads(urllib.request.urlopen(req, timeout=600).read())["results"]:
                fh.write(json.dumps(r) + "\n")
    print(f"{len(claims)} claims checked to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--model", default="Qwen/Qwen3-8B")
    s.add_argument("--thr", type=Path, required=True, help="the weak-zeroed reference (neuron_builder.Neurons.save)")
    s.add_argument("--adapter", action="append", default=[], help="NAME=PATH, an organism's adapter claims may name")
    s.add_argument("--port", type=int, default=8701)
    s.add_argument("--rows", type=int, default=32)
    c = sub.add_parser("check")
    c.add_argument("url")
    c.add_argument("claims", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    {"serve": serve, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
