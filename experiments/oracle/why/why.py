"""Why did it say that? (amortize, 10-17): a prompt to Qwen3-8B and either an alternative answer or a second prompt to
compare against; in about a second the page shows
  1. the model's answer and its margin over the alternative (bits: log2 p(answer word) - log2 p(alternatives),
     answer_contrast's words);
  2. the circuit that carries the difference (circuit_amortize's fast path, every cut measured on this pair): the
     source neurons by prompt token (counts, layers), the heads that move the answer to the answer position (where
     each looks), the readout neurons at the answer position (or how much of the difference all of them carry);
  3. a check: the circuit's units set to their values in the contrast run, the margin and the answer before and after,
     beside as many random units at the same layers and positions;
  4. a summary filled from a template with those numbers (no language model writes it).
The contrast: the second prompt when given (its token count must equal the prompt's); else, with an alternative answer,
a donor from the known task families (trainset_8b and devset_8b keys: the task key the prompt holds replaced by the key
whose answer is the alternative, the token counts agreeing); else zero ablation: the units whose zeroing lowers the
margin most (gradient x value order, then measured by zeroing prefixes of that order), which tests necessity only, and
the page says so.

  why.py serve --scorer MODEL.pt [--unions UNIONS.json] [--model Qwen/Qwen3-8B] [--adapters NAME=DIR,...] [--port 8801]
  why.py ask "PROMPT" (--alternative WORD | --contrast "PROMPT") --scorer MODEL.pt [...]   one explanation as JSON
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "method" / "oracle"))
import circuit_amortize as ca  # noqa: E402
import circuit_finder as cf  # noqa: E402
import circuit_masks as cm  # noqa: E402

QUESTIONS = ("/Users/user/mpd-data/circuits/trainset_8b.json", "/Users/user/mpd-data/circuits/devset_8b.json")
ABLATE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)   # ablation mode: prefixes of the gradient x value order zeroed
SEED = 0


class Why:
    """The model, the fast path's scorer and the donor index, kept loaded; explain() answers one request."""

    def __init__(self, model, tok, score, unions: dict | None = None, questions=QUESTIONS, chat=None):
        self.model, self.tok, self.score, self.unions = model, tok, score, unions or {}
        self.pt = cf.Patcher(model, tok)
        self.chat = chat or (lambda text: cf.chat(tok, text))
        self.keys = {}                              # task -> key -> answer words
        for path in questions:
            if Path(path).exists():
                for q in json.loads(Path(path).read_text())["questions"]:
                    self.keys.setdefault(q["task"], {})[q["key"]] = [a.lower() for a in q.get("answer", [])]

    # ----- the contrast
    def donor(self, prompt: str, alternative: str, enc=None):
        """(donor prompt, {task, key, donor_key}) from the task families, or (None, why not); enc: text -> ids."""
        enc = enc or self.chat
        alt = alternative.strip().lower()
        n = len(enc(prompt))
        tried = []
        for task, keys in self.keys.items():
            here = [k for k in keys if re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", prompt)]
            if not here:
                continue
            k0 = max(here, key=len)
            for k1 in [k for k, ans in keys.items() if alt in ans and k != k0]:
                p2 = prompt.replace(k0, k1, 1)
                tried.append(k1)
                if len(enc(p2)) == n:
                    return p2, {"task": task, "key": k0, "donor_key": k1}
        return None, ("no known task holds a key in this prompt with that answer" if not tried else
                      f"the donor prompts ({', '.join(tried)}) have other token counts")

    # ----- one request
    def encoder(self, prefill: str | None = None):
        """Text -> token ids: the user turn, the assistant's reply starting with prefill (the answer read after it)."""
        pre = list(self.tok(prefill, add_special_tokens=False)["input_ids"]) if prefill and self.tok is not None else []
        return lambda text: list(self.chat(text)) + pre

    def explain(self, prompt: str, alternative: str | None = None, contrast: str | None = None,
                prefill: str | None = None) -> dict:
        t0 = time.perf_counter()
        out = {"prompt": prompt, "alternative": alternative, "contrast_prompt": contrast, "prefill": prefill, "mode": "pair",
               "donor": None}
        enc = self.encoder(prefill)
        if not contrast and alternative:
            contrast, info = self.donor(prompt, alternative, enc)
            if contrast is None:
                out.update(mode="ablation", ablation_reason=info)
                out.update(self.ablation(prompt, alternative, enc))
                out["seconds"] = round(time.perf_counter() - t0, 2)
                out["summary"] = summary(out)
                return out
            out.update(donor=info, contrast_prompt=contrast)
        if not contrast:
            return {**out, "error": "give an alternative answer or a second prompt to compare against"}
        clean, corrupt = enc(prompt), enc(contrast)
        if len(clean) != len(corrupt):
            return {**out, "error": f"the two prompts have {len(clean)} and {len(corrupt)} tokens; patching needs equal "
                                    "counts (change a word for one of the same token length)"}
        if clean == corrupt:
            return {**out, "error": "the two prompts are the same"}
        task = (out["donor"] or {}).get("task") or self.task_of(prompt)
        union = (self.unions.get(task) or {}).get("all") if task else None
        rec = ca.find_batch_fast([self.pt], [(clean, corrupt, None)], self.score, unions=[union])[0]
        out.update(self.from_record(rec, clean, corrupt, union is not None))
        out["seconds"] = round(time.perf_counter() - t0, 2)
        out["summary"] = summary(out)
        return out

    def task_of(self, prompt: str) -> str | None:
        """The task whose longest key the prompt holds (its family's mover union), if any."""
        best = None
        for task, keys in self.keys.items():
            for k in keys:
                if re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", prompt) and (best is None or len(k) > best[1]):
                    best = (task, len(k))
        return best[0] if best else None

    def text(self, ids) -> str:
        return self.tok.decode(list(ids)) if self.tok is not None else " ".join(f"<{i}>" for i in ids)

    def from_record(self, rec: dict, clean: list, corrupt: list, family: bool) -> dict:
        """The page's fields from a fast record (the Patcher still set up on its pair)."""
        pt, tx = self.pt, self.text
        res = {"answer": cased(rec.get("answer_word"), tx([rec["answer"]])),
               "contrast_answer": cased((rec.get("alternative_words") or [None])[0], tx([rec["corrupt_answer"]])),
               "answer_token": tx([rec["answer"]]), "contrast_token": tx([rec["corrupt_answer"]]),
               "alternative_words": rec.get("alternative_words", [])}
        if not rec.get("usable"):
            return {**res, "error": "no circuit: " + rec.get("skipped", "the pair is not usable"),
                    "margin": rec.get("contrast", {}).get("clean"), "contrast_margin": rec.get("contrast", {}).get("corrupt")}
        res.update(margin=rec["contrast"]["clean"], contrast_margin=rec["contrast"]["corrupt"], gap=rec["gap"],
                   tokens=[tx([i]) for i in clean], contrast_tokens=[tx([i]) for i in corrupt], first_diff=rec["first_diff"])
        cuts, stage = rec["cuts"], rec["amortize"]["stage"]
        src = [n for n in rec["nodes"] if n["stage"] == "source"]
        ro = [n for n in rec["nodes"] if n["stage"] == "readout"]
        by = {}
        for n in src:
            by.setdefault(n["pos"], []).append(n["layer"])
        res["source"] = {"met": cuts.get("source", {}).get("met", False), "size": len(src), "stage": stage.get("source"),
                         "lost": cuts.get("source", {}).get("lost"), "restored": cuts.get("source", {}).get("restored"),
                         "by_token": [{"pos": p, "token": tx([clean[p]]), "contrast_token": tx([corrupt[p]]),
                                       "neurons": len(ls), "layers": sorted(ls)} for p, ls in sorted(by.items())]}
        res["movers"] = {"met": cuts.get("movers", {}).get("met", False), "size": len(rec["routes"]), "family": family,
                         "stage": stage.get("movers"), "lost": cuts.get("movers", {}).get("lost"),
                         "restored": cuts.get("movers", {}).get("restored"),
                         "heads": [{"layer": r["layer"], "head": r["head"],
                                    "attends": [[p, tx([clean[p]]), w] for p, w in (r.get("attends") or []) if p > 0]}
                                   for r in rec["routes"]]}       # (position 0, the attention sink, left out)
        c = cuts.get("readout", {})
        res["readout"] = {"reachable": c.get("reachable", False), "met": c.get("met", False), "size": len(ro),
                          "stage": stage.get("readout"), "lost": c.get("lost"), "restored": c.get("restored"),
                          "ceiling": c.get("ceiling"),
                          "layers": sorted({n["layer"] for n in ro})}
        units = {"source": [("neuron", n["layer"], n["pos"], n["neuron"]) for n in src],
                 "movers": [("head", r["layer"], r["head"], r["pos"]) for r in rec["routes"]],
                 "readout": [("neuron", n["layer"], n["pos"], n["neuron"]) for n in ro]}
        res["check"] = self.check(units, rec)
        return res

    # ----- the check
    def _random(self, units: list, rng) -> list:
        """As many random units at the same layers and positions (neurons) or layers (heads), none of units."""
        have, out = set(units), []
        for u in units:
            for _ in range(64):                     # (a place whose every unit is taken: a repeat after 64 draws)
                v = (u[0], u[1], u[2], int(rng.integers(self.pt.D))) if u[0] == "neuron" else \
                    (u[0], u[1], int(rng.integers(self.pt.H)), u[3])
                if v not in have:
                    break
            have.add(v)
            out.append(v)
        return out

    @torch.no_grad()
    def patched(self, sets: list, base: str = "clean") -> list:
        """Per set of units, (margin in bits, top token) of the base run with them set to the other run's values, one
        batch (every position run: no cache)."""
        pt = self.pt
        idx = pt._index([(r, base, list(s)) for r, s in enumerate(sets)], 0)
        x = torch.stack([pt._ids_dev[base]] * len(sets))
        hn = pt._last(x, len(sets), **pt._edits(idx, 0))
        c = pt._contrast(hn)
        top = pt._logp(hn).argmax(-1)
        return [(float(a), self.text([int(b)])) for a, b in zip(c, top)]

    def check(self, units: dict, rec: dict) -> list:
        rng = np.random.default_rng(SEED)
        names = [n for n in ("source", "movers", "readout") if units[n] and rec["cuts"].get(n, {}).get("met")]
        sets = [units[n] for n in names] + [self._random(units[n], rng) for n in names]
        got = self.patched(sets) if sets else []
        out = []
        for j, n in enumerate(names):
            (m, a), (mr, ar) = got[j], got[len(names) + j]
            y = self.text([rec["answer"]])
            out.append({"cut": n, "units": len(units[n]), "margin_before": rec["contrast"]["clean"], "margin_after": m,
                        "answer_after": a, "flipped": a != y, "random_margin_after": mr, "random_answer_after": ar,
                        "random_flipped": ar != y})     # (flipped: the top token is no longer the answer's)
        return out

    # ----- ablation mode
    def ablation(self, prompt: str, alternative: str, enc=None) -> dict:
        """No contrast prompt: the margin of the model's top token over the alternative's first tokens; units in
        descending gradient x value (the first-order drop when zeroed: neurons at every position, heads at the answer
        position), prefixes of that order zeroed and measured (ABLATE_SIZES), the smallest that flips the answer or
        else the one that lowers the margin most, beside as many random units at the same places."""
        pt, tok = self.pt, self.tok
        ids = torch.tensor((enc or self.chat)(prompt), device=pt.dev)
        T = len(ids)
        alt_ids = sorted({tok(v, add_special_tokens=False)["input_ids"][0] for v in (alternative.strip(), " " + alternative.strip())}) \
            if tok is not None else [int(alternative)]
        with torch.no_grad():
            lp0 = pt._logp(pt._last(ids[None], 1))[0]
        y = int(lp0.argmax())
        alt_ids = [a for a in alt_ids if a != y] or [int(lp0.topk(2).indices[1])]
        W = pt.core.lm_head.weight

        def margin(hn):
            z = hn.float() @ W[torch.tensor([y] + alt_ids, device=W.device)].float().T
            return (z[:, 0] - torch.logsumexp(z[:, 1:], -1)) / np.log(2)
        add = {"neu": [torch.zeros(1, T, pt.D, device=l.mlp.down_proj.weight.device, requires_grad=True) for l in pt.layers],
               "head": [torch.zeros(1, T, pt.H * pt.hd, device=l.self_attn.o_proj.weight.device, requires_grad=True)
                        for l in pt.layers]}
        cap = {}
        with torch.enable_grad():
            m0 = margin(pt._last(ids[None], 1, add=add, cap=cap))
            m0.sum().backward()
        units, scores = [], []
        for i in range(pt.L):                       # (zeroing x moves the margin by about -x . dm/dx)
            v = (cap["neu"][i][0].float() * add["neu"][i].grad[0].float())                     # [T, D]
            top = v.reshape(-1).topk(min(256, v.numel()))
            for s, f in zip(top.values.tolist(), top.indices.tolist()):
                units.append(("neuron", i, f // pt.D, f % pt.D))
                scores.append(s)
            h = (cap["head"][i][0][-1].float() * add["head"][i].grad[0][-1].float()).view(pt.H, pt.hd).sum(-1)
            for k, s in enumerate(h.tolist()):
                units.append(("head", i, k, T - 1))
                scores.append(s)
        order = [units[j] for j in np.argsort(-np.asarray(scores))]
        sizes = [k for k in ABLATE_SIZES if k <= len(order)]
        rng = np.random.default_rng(SEED)
        sets = [order[:k] for k in sizes]
        sets += [self._random(s, rng) for s in sets]
        got = self.zeroed(ids, sets, margin)
        m_base = float(m0[0])
        k_at = next((j for j, k in enumerate(sizes) if got[j] < 0), int(np.argmin(got[:len(sizes)])))
        chosen = sets[k_at]
        by = {}
        for u in chosen:
            if u[0] == "neuron":
                by.setdefault(u[2], []).append(u[1])
        return {"answer": self.text([y]), "contrast_answer": alternative.strip(), "margin": m_base,
                "tokens": [self.text([i]) for i in ids.tolist()],
                "ablation": {"sizes": sizes, "margins": got[:len(sizes)], "random_margins": got[len(sizes):],
                             "chosen": len(chosen), "flipped": got[k_at] < 0, "margin_after": got[k_at],
                             "random_margin_after": got[len(sizes) + k_at],
                             "by_token": [{"pos": p, "token": self.text([int(ids[p])]), "neurons": len(ls),
                                           "layers": sorted(ls)} for p, ls in sorted(by.items())],
                             "heads": [{"layer": u[1], "head": u[2]} for u in chosen if u[0] == "head"]}}

    @torch.no_grad()
    def zeroed(self, ids, sets: list, margin) -> list:
        """Per set of units, the margin with them zeroed (one batch)."""
        pt = self.pt
        B = len(sets)
        rows = {}
        for r, s in enumerate(sets):
            for u in s:
                key = ("neu" if u[0] == "neuron" else "head", u[1])
                rows.setdefault(key, []).append((r, u[2], u[3]) if u[0] == "neuron" else (r, u[3], u[2]))

        def edit(kind):
            def f(i, h):
                got = rows.get((kind, i))
                if not got:
                    return h
                h = h.clone()
                r, p, k = (torch.tensor(c, device=h.device) for c in zip(*got))
                if kind == "neu":
                    h[r, p, k] = 0
                else:
                    h4 = h.view(h.shape[0], h.shape[1], pt.H, pt.hd)
                    h4[r, p, k] = 0
                    h = h4.view(h.shape)
                return h
            return f
        hn = pt._last(ids[None].expand(B, -1), B, neu=edit("neu"), head=edit("head"))
        return [float(v) for v in margin(hn)]


def cased(word: str | None, token: str) -> str:
    """An answer word (answer_contrast's, lower case) in its first token's case: " Ver" + "verde" -> "Verde"."""
    t = token.strip()
    if word and word.startswith(t.lower()):
        return t + word[len(t):]
    return word or t


def summary(r: dict) -> str:
    """The plain-English summary: a template over the numbers (no language model)."""
    if r.get("error"):
        return r["error"]
    fmt = lambda v: f"{v:+.1f} bits"  # noqa: E731
    if r["mode"] == "ablation":
        a = r["ablation"]
        toks = ", ".join(f"{b['neurons']} on “{b['token'].strip()}”" for b in sorted(a["by_token"], key=lambda b: -b["neurons"])[:4])
        s = (f"The model answers “{r['answer'].strip()}” ({fmt(r['margin'])} over “{r['contrast_answer']}”). No contrast "
             f"prompt was found ({r.get('ablation_reason')}), so this is zero ablation, which tests only what is necessary. ")
        s += (f"Zeroing {a['chosen']} units ({toks or 'heads only'}{'; ' + str(len(a['heads'])) + ' heads' if a['heads'] else ''}) "
              f"moves the margin to {fmt(a['margin_after'])}" + (" and flips the answer" if a["flipped"] else "") +
              f"; as many random units at the same places leave {fmt(a['random_margin_after'])}.")
        return s
    s = (f"The model answers “{r['answer'].strip()}” ({fmt(r['margin'])} over “{r['contrast_answer'].strip()}”); on "
         f"“{r['contrast_prompt']}” it answers “{r['contrast_answer'].strip()}” ({fmt(r['contrast_margin'])}). ")
    src, mov, ro = r["source"], r["movers"], r["readout"]
    if src["met"]:
        toks = sorted(src["by_token"], key=lambda b: -b["neurons"])
        where = ", ".join(f"{b['neurons']} on “{b['token'].strip()}”" for b in toks[:3])
        lay = [l for b in src["by_token"] for l in b["layers"]]
        s += f"The difference starts in {src['size']} neurons ({where}; layers {min(lay)}-{max(lay)}). "
    if mov["met"]:
        reads = {}
        for h in mov["heads"]:
            if h["attends"]:
                reads[h["attends"][0][1].strip()] = reads.get(h["attends"][0][1].strip(), 0) + 1
        most = max(reads, key=reads.get) if reads else None
        s += (f"{mov['size']} attention heads move it to the answer position"
              + (f" (most look at “{most}”)" if most else "") + (", the task family's shared heads" if mov["family"] else "") + ". ")
    if ro["met"]:
        s += f"{ro['size']} neurons at the answer position read it out (layers {min(ro['layers'])}-{max(ro['layers'])}). "
    elif ro.get("ceiling"):
        s += (f"The readout is spread out: every neuron at the answer position together carries "
              f"{min(ro['ceiling']):.0%} of the difference. ")
    for c in r["check"]:
        if c["cut"] == ("source" if src["met"] else "movers"):
            s += (f"Setting the {c['units']} {'neurons' if c['cut'] != 'movers' else 'heads'} to their values from the "
                  f"contrast prompt moves the margin from {fmt(c['margin_before'])} to {fmt(c['margin_after'])}"
                  + (f" and the answer to “{c['answer_after'].strip()}”" if c["flipped"] else "") +
                  f"; as many random ones at the same places leave {fmt(c['random_margin_after'])}.")
            break
    return s


HTML = HERE / "why.html"


def serve(why: Why, port: int, adapters: dict | None = None):
    """The page at / and POST /explain {prompt, alternative | contrast, model}: one request at a time (one model)."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    adapters = adapters or {}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._send(204, b"")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/health":
                self._send(200, json.dumps({"ok": True, "models": ["base"] + list(adapters)}).encode())
            else:
                self._send(404, b"{}")

        def do_POST(self):
            if self.path != "/explain":
                return self._send(404, b"{}")
            try:
                q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with use_adapter(why.model, q.get("model"), adapters):
                    r = why.explain(q.get("prompt", ""), q.get("alternative") or None, q.get("contrast") or None,
                                    q.get("prefill") or None)
                r["model"] = q.get("model") or "base"
                self._send(200, json.dumps(r, default=float).encode())
            except Exception as e:                  # noqa: BLE001 (the page shows it)
                import traceback
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}).encode())

        def log_message(self, *a):
            sys.stderr.write("%s\n" % (a[0] % a[1:]))
    print(f"serving on port {port}", flush=True)
    HTTPServer(("0.0.0.0", port), H).serve_forever()


class use_adapter:
    """The named LoRA on (a PeftModel's adapter by name; "base" or none: every adapter off) while a request runs."""

    def __init__(self, model, name, adapters):
        self.model, self.name, self.on = model, name, bool(name and name != "base" and name in adapters)

    def __enter__(self):
        if hasattr(self.model, "disable_adapter_layers"):
            if self.on:
                self.model.set_adapter(self.name)
                self.model.enable_adapter_layers()
            else:
                self.model.disable_adapter_layers()

    def __exit__(self, *a):
        return False


def load(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    dt = getattr(torch, a.dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(a.device).eval()
    adapters = dict(x.split("=", 1) for x in a.adapters.split(",")) if a.adapters else {}
    if adapters:                                    # (each organism a named LoRA, unmerged: one request turns one on)
        from peft import PeftModel
        names = list(adapters)
        model = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0]).eval()
        for n in names[1:]:
            model.load_adapter(adapters[n], adapter_name=n)
        model.disable_adapter_layers()
    model.requires_grad_(False)
    score = ca.scorer(f"model:{a.scorer}") if a.scorer else ca.scorer("grad")
    ca.SOURCE_PATH = getattr(score, "source_path", True)
    ca.SOURCE_GRAD = getattr(score, "source_grad", True)
    unions = json.loads(Path(a.unions).read_text()) if a.unions else {}
    return Why(model, tok, score, unions, questions=a.questions.split(",")), adapters, getattr(score, "relp", False)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("serve", "ask"))
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--alternative", default=None)
    ap.add_argument("--contrast", default=None)
    ap.add_argument("--organism", default=None, help="ask: the adapter name to turn on")
    ap.add_argument("--prefill", default=None, help="ask: the start of the assistant's reply (the answer read after it)")
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--unions", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--adapters", default=None, help="NAME=DIR,... organisms' LoRAs, chosen per request")
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--questions", default=",".join(QUESTIONS), help="question sets whose task keys make donors (train and "
                    "dev splits only; never the sealed test set)")
    a = ap.parse_args()
    cf._threads()
    why, adapters, _ = load(a)
    if a.mode == "ask":
        with use_adapter(why.model, a.organism, adapters):
            r = why.explain(a.prompt, a.alternative, a.contrast, a.prefill)
        print(json.dumps(r, indent=1, default=float))
        return
    why.explain("What is the capital of France? Answer with one word.", contrast="What is the capital of Italy? Answer with one word.")
    serve(why, a.port, adapters)


if __name__ == "__main__":
    main()
