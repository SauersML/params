"""The AMI metric ladder (scrub, the lead's 10-16 AMI track, ~/mpd-data/team/ami_1016.txt): one scorer for circuits of
neurons and attention heads in an HF Qwen3 at the ladder's stronger rungs, the same code for every method's circuits.

Units: a neuron at a position (the input of down_proj, act(gate) x up), a head at a position (its slice of o_proj's
input), and a head channel at a position (one coordinate of that slice). The token embeddings are the input and are never
ablated; the final norm and the unembedding have no units.
Background: every unit outside the circuit, at every position, is set to its mean over pretraining text (Means: the
atlas corpus, means per absolute-position bucket BUCKETS, so position 0's attention-sink values are the mean of first
positions; MODE "global" uses one mean over every position, the circuit-sparsity paper's form). The circuit's units
compute from what is left.

  R3  the circuit-sparsity paper's criterion (Gao et al. 2025, arXiv 2511.13653): the background above with the circuit's
      units live. On each prompt, at the answer position: the KL divergence of the circuit model's next-token
      distribution from the full model's (nats), and the task loss, -ln p(answer word) (nats; the answer word's token
      set from answer_contrast on the full model's run), which passes at most TARGET (the paper's 0.15) above the full
      model's own task loss on that prompt (the absolute pass, at most TARGET, beside it). Prompts: the
      circuit's own pair (clean and corrupted) and held-out prompts (its question's template with other keys of its task,
      never its own, its corrupted key or the keys it was selected on; the same token count, so every unit keeps its
      position). Beside the circuit: the full model, the empty circuit (every unit at the background) and RANDOM circuits
      of the same size (each unit redrawn at its kind, layer and position). recovered = (loss of the empty circuit - loss
      of the circuit) / (loss of the empty circuit - loss of the full model), for the task loss and for the KL.
      Sizes: units, distinct (layer, unit) pairs, positions, and the unit-to-unit edges a dense model has among them
      (edges(): every pair joined by a direct residual path; no edge is pruned at R3).

  ami.py sample OUT.npz [--tokens 1100000]          a seeded sample of the atlas corpus's documents (the means' text)
  ami.py means SAMPLE.npz OUT.pt [--model M]        the background means of a model over the sample
  ami.py r3 RECORDS_DIR OUT_DIR --means M.pt --label METHOD [--devset D] [--pool P] [--model M]
                                                    every record's R3 (a circuit_finder record, or a circuit file with
                                                    "neurons" [[layer, pos, neuron]] and "heads" [[layer, pos, head]])
  ami.py board OUT_DIR [OUT_DIR ...]                the leaderboard lines of scored sets (TSV on stdout)
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import answer_contrast  # noqa: E402
import circuit_finder as cf  # noqa: E402

CORPUS = Path.home() / "mpd-data" / "atlas" / "corpus_qwen3"
DEVSET = Path.home() / "mpd-data" / "circuits" / "devset_8b.json"
POOL = Path.home() / "mpd-data" / "circuits" / "trainset_8b.json"
BUCKETS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512)   # bucket b holds positions from the b-th boundary (0 first) to the next
NB = len(BUCKETS) + 1
MODE = "bucket"     # the background: "bucket" (the mean at the position's bucket) or "global" (one mean over all positions)
TARGET = 0.15       # nats: the circuit-sparsity paper's target task loss
DRAWS = 8           # held-out prompts per circuit
RANDOM = 2          # random circuits of the circuit's size per circuit
BATCH_TOKENS = 8192  # rows x positions a forward at most


def bucket_of(pos: torch.Tensor) -> torch.Tensor:
    """The position bucket of each absolute position."""
    return torch.bucketize(pos, torch.tensor(BUCKETS, device=pos.device), right=True)


# ---------------------------------------------------------------- the background means

def sample(out: Path, tokens: int = 1_100_000, seed: int = 0, corpus: Path = CORPUS) -> dict:
    """Documents of the atlas corpus in a seeded random order until their tokens reach tokens: {tokens, offsets, sources}
    (offsets[j]..offsets[j + 1] is document j)."""
    tok = np.load(corpus / "tokens.npy", mmap_mode="r")
    off = np.load(corpus / "offsets.npy")
    src = np.load(corpus / "sources.npy")
    n = len(off) - 1 if len(off) == len(src) + 1 else len(off)
    ends = np.append(off[1:], len(tok)) if len(off) == n else off[1:]
    order = np.random.default_rng(seed).permutation(n)
    pick, total = [], 0
    for j in order:
        pick.append(int(j))
        total += int(ends[j] - off[j])
        if total >= tokens:
            break
    parts = [np.asarray(tok[off[j]:ends[j]], dtype=np.int32) for j in pick]
    offsets = np.cumsum([0] + [len(p) for p in parts]).astype(np.int64)
    d = {"tokens": np.concatenate(parts), "offsets": offsets, "sources": src[pick], "documents": np.array(pick, np.int64)}
    np.savez(out, **d)
    return d


@torch.no_grad()
def collect(model, docs: list, batch_tokens: int = 16384, log=None) -> dict:
    """Sums and sums of squares of every neuron (down_proj's input) and head channel (o_proj's input) per position bucket
    over docs (lists of token ids, each from its first token), and the positions counted per bucket."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    cfg, layers = core.config, list(core.model.layers)
    L, D = cfg.num_hidden_layers, cfg.intermediate_size
    HD = cfg.num_attention_heads * (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    dev = core.model.embed_tokens.weight.device
    acc = {"neuron_sum": torch.zeros(NB, L, D, dtype=torch.float64, device=dev),
           "neuron_sq": torch.zeros(NB, L, D, dtype=torch.float64, device=dev),
           "head_sum": torch.zeros(NB, L, HD, dtype=torch.float64, device=dev),
           "head_sq": torch.zeros(NB, L, HD, dtype=torch.float64, device=dev)}
    count = torch.zeros(NB, dtype=torch.float64, device=dev)
    state = {}

    def hook(kind, i):
        def f(mod, args):
            x = args[0].float()
            oh = state["oh"].to(x.device)                               # [B, T, NB]: valid positions by bucket
            acc[kind + "_sum"][:, i] += torch.einsum("btn,btd->nd", oh, x).double().to(dev)
            acc[kind + "_sq"][:, i] += torch.einsum("btn,btd->nd", oh, x * x).double().to(dev)
        return f
    hs = []
    for i, layer in enumerate(layers):
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(hook("neuron", i)))
        hs.append(layer.self_attn.o_proj.register_forward_pre_hook(hook("head", i)))
    order = sorted(range(len(docs)), key=lambda j: -len(docs[j]))
    t0, done = time.time(), 0
    try:
        at = 0
        while at < len(order):
            T = len(docs[order[at]])
            B = max(1, batch_tokens // T)
            part = [docs[j] for j in order[at:at + B]]
            at += len(part)
            x = torch.zeros(len(part), T, dtype=torch.long)
            valid = torch.zeros(len(part), T, dtype=torch.bool)
            for b, d in enumerate(part):
                x[b, :len(d)] = torch.as_tensor(d)
                valid[b, :len(d)] = True
            oh = torch.nn.functional.one_hot(bucket_of(torch.arange(T))[None].expand(len(part), -1), NB).float() * valid[..., None]
            state["oh"] = oh.to(dev)
            count += oh.sum((0, 1)).double().to(dev)
            core.model(input_ids=x.to(dev), attention_mask=valid.long().to(dev))   # (right padding: positions from 0)
            done += int(valid.sum())
            if log:
                log(f"  means: {done:,} tokens, {done / (time.time() - t0):.0f} tokens/s")
    finally:
        for h in hs:
            h.remove()
    return {**{k: v.cpu() for k, v in acc.items()}, "count": count.cpu(), "buckets": list(BUCKETS)}


class Means:
    """A model's background means (collect's file): per position bucket, every neuron's and head channel's mean."""

    def __init__(self, path, device="cpu"):
        d = torch.load(path, map_location="cpu", weights_only=False)
        assert list(d["buckets"]) == list(BUCKETS), "means collected with other position buckets"
        c = d["count"].clamp(min=1)
        self.count = d["count"]
        self.mean = {"neuron": (d["neuron_sum"] / c[:, None, None]).float().to(device),
                     "head": (d["head_sum"] / c[:, None, None]).float().to(device)}
        tot = float(d["count"].sum())
        self.glob = {k: (d[k + "_sum"].sum(0) / tot).float().to(device) for k in ("neuron", "head")}
        self.meta = {k: d[k] for k in d if k not in ("neuron_sum", "neuron_sq", "head_sum", "head_sq", "count")}
        self._cache = {}

    def table(self, kind: str, layer: int, T: int, dev, dtype, mode: str = None) -> torch.Tensor:
        """[T, width] the background of one layer's neurons or head channels at positions 0..T-1."""
        mode = mode or MODE
        key = (kind, layer, T, str(dev), dtype, mode)
        if key not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            if mode == "global":
                t = self.glob[kind][layer][None].expand(T, -1)
            else:
                t = self.mean[kind][:, layer][bucket_of(torch.arange(T, device=self.mean[kind].device))]
            self._cache[key] = t.to(dev, dtype).contiguous()
        return self._cache[key]


# ---------------------------------------------------------------- the masked run

def channels(u: tuple, hd: int) -> tuple:
    """A unit as (kind, layer, position, indices): ("neuron", layer, pos, k) -> neuron k; ("head", layer, h, pos) -> the
    head's hd channels; ("hchan", layer, pos, c) -> channel c of o_proj's input."""
    if u[0] == "neuron":
        return "neuron", int(u[1]), int(u[2]), [int(u[3])]
    if u[0] == "head":
        return "head", int(u[1]), int(u[3]), list(range(int(u[2]) * hd, (int(u[2]) + 1) * hd))
    if u[0] == "hchan":
        return "head", int(u[1]), int(u[2]), [int(u[3])]
    raise ValueError(f"unit {u!r}")


class Ablator:
    """Runs of an HF Qwen3 (circuit_finder.Patcher's hooks) whose rows each name the units they keep live: every other
    neuron and head channel at every position set to the background (Means), each row's kept units computed, and each
    row's set values (over) written over its units. A row is {"ids": [...], "keep": units or None (the full model),
    "over": [(unit, value tensor)]}; rows of one forward have one length."""

    def __init__(self, model, tok, means: Means, mode: str | None = None):
        self.pt = cf.Patcher(model, tok)
        self.means, self.mode = means, mode
        self.L, self.D, self.H, self.hd = self.pt.L, self.pt.D, self.pt.H, self.pt.hd

    def _plan(self, rows: list) -> dict:
        live, keep, over = [], {}, {}
        for b, r in enumerate(rows):
            if r.get("keep") is None:
                live.append(b)
                continue
            for u in r["keep"]:
                kind, i, p, ks = channels(u, self.hd)
                e = keep.setdefault((kind, i), ([], [], []))
                e[0].extend([b] * len(ks))
                e[1].extend([p] * len(ks))
                e[2].extend(ks)
            for u, v in r.get("over") or []:
                kind, i, p, ks = channels(u, self.hd)
                e = over.setdefault((kind, i), ([], [], [], []))
                e[0].extend([b] * len(ks))
                e[1].extend([p] * len(ks))
                e[2].extend(ks)
                e[3].append(torch.as_tensor(v, dtype=torch.float32).reshape(-1))
        dev = self.pt.dev
        t = lambda xs: torch.tensor(xs, dtype=torch.long, device=dev)  # noqa: E731
        return {"live": t(live) if live else None,
                "keep": {k: tuple(t(x) for x in v) for k, v in keep.items()},
                "over": {k: (t(v[0]), t(v[1]), t(v[2]), torch.cat(v[3]).to(dev)) for k, v in over.items()},
                "all_live": len(live) == len(rows)}

    def _fns(self, plan: dict) -> dict:
        def make(kind):
            def fn(i, h):
                if plan["all_live"] and (kind, i) not in plan["over"]:
                    return h
                B, T, W = h.shape
                out = self.means.table(kind, i, T, h.device, h.dtype, self.mode)[None].expand(B, -1, -1).clone()
                if plan["live"] is not None:
                    lv = plan["live"].to(h.device)
                    out[lv] = h[lv]
                if (kind, i) in plan["keep"]:
                    b, p, k = (z.to(h.device) for z in plan["keep"][(kind, i)])
                    out[b, p, k] = h[b, p, k]
                if (kind, i) in plan["over"]:
                    b, p, k, v = (z.to(h.device) for z in plan["over"][(kind, i)])
                    out[b, p, k] = v.to(h.dtype)
                return out
            return fn
        L = set(range(self.L))
        return {"neu": make("neuron"), "head": make("head"), "only": {"res": set(), "head": L, "neu": L}}

    @torch.no_grad()
    def logp(self, rows: list) -> torch.Tensor:
        """[R, V] float32 log-probabilities at each row's last position (rows of one length, BATCH_TOKENS a forward)."""
        if not rows:
            return torch.zeros(0)
        T = len(rows[0]["ids"])
        assert all(len(r["ids"]) == T for r in rows), "Ablator.logp: rows of one length"
        n = max(1, BATCH_TOKENS // T)
        out = []
        for a in range(0, len(rows), n):
            part = rows[a:a + n]
            hn = self.pt._last(torch.tensor([r["ids"] for r in part]), len(part), **self._fns(self._plan(part)))
            out.append(self.pt._logp(hn).float().cpu())
        return torch.cat(out)


# ---------------------------------------------------------------- circuits and prompts

def record_units(rec: dict) -> list:
    """A circuit's units: a circuit_finder record's nodes (every stage) and routes, or a circuit file's "neurons" [[layer,
    pos, neuron, ...]], "heads" [[layer, pos, head]] and "hchans" [[layer, pos, channel]]; each once, in order."""
    us = [("neuron", int(n["layer"]), int(n["pos"]), int(n["neuron"])) for n in rec.get("nodes") or []]
    us += [("head", int(r["layer"]), int(r["head"]), int(r["pos"])) for r in rec.get("routes") or []]
    us += [("neuron", int(x[0]), int(x[1]), int(x[2])) for x in rec.get("neurons") or []]
    us += [("head", int(x[0]), int(x[2]), int(x[1])) for x in rec.get("heads") or []]
    us += [("hchan", int(x[0]), int(x[1]), int(x[2])) for x in rec.get("hchans") or []]
    return list(dict.fromkeys(us))


def _lp(u: tuple) -> tuple:
    """(kind, layer, position) of a unit."""
    return (u[0], u[1], u[3] if u[0] == "head" else u[2])


def edges(units: list) -> int:
    """Unit-to-unit edges a dense model has among units: a direct residual path from u's write to v's read. A neuron or
    head channel at (layer l, position p) writes into p's residual stream after its sublayer; a neuron at (l', p) reads
    p's stream before its MLP (l < l', or l = l' for a head); a head at (l', q) reads every position up to q through its
    keys and values and q through its query (l < l', p <= q)."""
    sites = [(("n" if u[0] == "neuron" else "h"),) + _lp(u)[1:] for u in units]
    n = 0
    for k1, l1, p1 in sites:
        for k2, l2, p2 in sites:
            if k2 == "n":
                n += p1 == p2 and (l1 < l2 or (l1 == l2 and k1 == "h"))
            else:
                n += l1 < l2 and p1 <= p2
    return int(n)


def sizes(units: list) -> dict:
    return {"units": len(units), "neurons": sum(u[0] == "neuron" for u in units), "heads": sum(u[0] == "head" for u in units),
            "hchans": sum(u[0] == "hchan" for u in units),
            "distinct": len({(u[0], u[1], u[3] if u[0] == "neuron" else u[2]) for u in units}),
            "positions": len({_lp(u)[2] for u in units}), "edges": edges(units)}


def redraw(units: list, D: int, H: int, HD: int, seed: int) -> list:
    """The random control: each unit redrawn uniformly at its kind, layer and position, none of the circuit's own."""
    g = np.random.default_rng(seed)
    have = set(units)
    out = []
    for u in units:
        for _ in range(1000):
            if u[0] == "neuron":
                v = ("neuron", u[1], u[2], int(g.integers(D)))
            elif u[0] == "head":
                v = ("head", u[1], int(g.integers(H)), u[3])
            else:
                v = ("hchan", u[1], u[2], int(g.integers(HD)))
            if v not in have:
                break
        have.add(v)
        out.append(v)
    return out


def load_questions(*paths) -> dict:
    """{id: question} over question files ({"questions": [...]} or lists), sealed test questions refused."""
    out = {}
    for p in paths:
        if p and Path(p).exists():
            d = json.loads(Path(p).read_text())
            for q in d["questions"] if isinstance(d, dict) else d:
                if q.get("split") != "test":
                    out[q["id"]] = q
    return out


def task_keys(qs: dict) -> dict:
    """{task: every key its questions name}: their keys, corrupted keys and partners' keys, in first-seen order."""
    out = {}
    for q in qs.values():
        ks = out.setdefault(q.get("task"), {})
        for k in [q.get("key"), q.get("corrupt_key")] + [p["key"] if isinstance(p, dict) else p for p in q.get("partners") or []]:
            if k:
                ks[k] = None
    return {t: list(v) for t, v in out.items()}


def prompt_ids(tok, user: str, q: dict) -> list:
    pre = q.get("prefill")
    return cf.chat(tok, user, q.get("system")) + (tok(pre, add_special_tokens=False)["input_ids"] if pre else [])


def heldout(tok, rec: dict, q: dict, keys: list, draws: int = DRAWS) -> list:
    """Up to draws held-out prompts [(key, ids)]: the question's template with other keys of its task (never its own key,
    its corrupted key or a key it was selected on), only those of the clean prompt's token count, in a seeded order."""
    user, key = q.get("user") or rec.get("user"), q.get("key") or rec.get("key")
    if not user or not key or user.count(key) != 1:
        return []
    banned = {key, q.get("corrupt_key"), rec.get("key")} | set(rec.get("selection_keys") or [])
    cand = [k for k in keys if k not in banned]
    np.random.default_rng(zlib.crc32(str(rec.get("id")).encode())).shuffle(cand)
    T, out = len(rec["ids"]), []
    for k in cand:
        ids = prompt_ids(tok, user.replace(key, k), q)
        if len(ids) == T:
            out.append((k, ids))
            if len(out) == draws:
                break
    return out


# ---------------------------------------------------------------- R3

def _metrics(lp_full: torch.Tensor, lp: torch.Tensor, answer: list) -> dict:
    """KL(full || circuit) in nats, the task loss -ln p(answer word) in nats, and whether the top token starts the answer."""
    p = lp_full.exp()
    a = torch.as_tensor(answer)
    return {"kl": float((p * (lp_full - lp)).sum()), "ce": float(-torch.logsumexp(lp[a], 0)), "top1": int(lp.argmax()) in set(answer)}


def r3(abl: Ablator, tok, rec: dict, q: dict, keys: list, draws: int = DRAWS, n_random: int = RANDOM) -> dict:
    """A circuit's R3 (module docstring) on its own pair and its held-out prompts."""
    t0 = time.perf_counter()
    units = record_units(rec)
    clean, corr = list(rec["ids"]), list(rec.get("corrupted") or [])
    prompts = [("own", "clean", clean)] + ([("own", "corrupted", corr)] if len(corr) == len(clean) else [])
    prompts += [("heldout", k, ids) for k, ids in heldout(tok, rec, q, keys, draws)]
    if q.get("user") and prompt_ids(tok, q["user"], q) != clean:
        prompts = prompts[:2]                      # (the record's prompt is not its question's template: no held-out prompts)
    seed = zlib.crc32(str(rec.get("id")).encode())
    rand = [redraw(units, abl.D, abl.H, abl.H * abl.hd, seed + j) for j in range(n_random)]
    arms = [("full", None), ("empty", []), ("circuit", units)] + [(f"random{j}", r) for j, r in enumerate(rand)]
    rows = [{"ids": ids, "keep": keep} for _, _, ids in prompts for _, keep in arms]
    lp = abl.logp(rows).view(len(prompts), len(arms), -1)
    sets = answer_contrast.token_sets_many(abl.pt.core, tok, [(ids, lp[j, 0], None, None, None) for j, (_, _, ids) in enumerate(prompts)])
    out = []
    for j, (split, name, ids) in enumerate(prompts):
        out.append({"split": split, "prompt": name, "answer_word": sets[j]["answer_word"],
                    **{a: _metrics(lp[j, 0], lp[j, x], sets[j]["answer"]) for x, (a, _) in enumerate(arms)}})
    res = {"id": rec.get("id"), "task": rec.get("task") or q.get("task"), "size": sizes(units), "random_size": sizes(rand[0]) if rand else None,
           "prompts": out, "seconds_found": rec.get("seconds"), "met_r0": rec.get("met")}
    if len(corr) == len(clean) and rec.get("answer_tokens") and rec.get("alternatives"):
        s = {"answer": rec["answer_tokens"], "alternatives": rec["alternatives"]}
        c = answer_contrast.contrast(lp[:2].reshape(-1, lp.shape[-1]), s).view(2, len(arms))     # [clean/corrupted, arm]
        g = c[0] - c[1]
        res["own_gap"] = {a: float(g[x] / g[0]) for x, (a, _) in enumerate(arms)}           # the contrast gap kept, / full's
        res["own_gap_bits"] = float(g[0])
    res["summary"] = summarize(out, [a for a, _ in arms])
    res["seconds"] = round(time.perf_counter() - t0, 2)
    return res


def summarize(prompts: list, arms: list) -> dict:
    """Per split: each arm's mean KL, task loss, pass rate (task loss at most TARGET above the full model's on the same
    prompt), absolute pass rate (task loss at most TARGET) and top-1 rate; recovered shares of the task loss and of the KL
    (module docstring); random arms pooled."""
    out = {}
    for split in ("own", "heldout"):
        P = [p for p in prompts if p["split"] == split]
        if not P:
            continue
        s = {"n": len(P)}
        for a in arms:
            s[a] = {m: float(np.mean([p[a][m] for p in P])) for m in ("kl", "ce", "top1")}
            s[a]["pass"] = float(np.mean([p[a]["ce"] - p["full"]["ce"] <= TARGET for p in P]))
            s[a]["pass_abs"] = float(np.mean([p[a]["ce"] <= TARGET for p in P]))
        rnd = [a for a in arms if a.startswith("random")]
        if rnd:
            s["random"] = {m: float(np.mean([s[a][m] for a in rnd])) for m in ("kl", "ce", "top1", "pass", "pass_abs")}
        for a in ["circuit"] + (["random"] if rnd else []):
            d_ce = s["empty"]["ce"] - s["full"]["ce"]
            s[a]["recovered_ce"] = (s["empty"]["ce"] - s[a]["ce"]) / d_ce if d_ce > 1e-9 else None
            s[a]["recovered_kl"] = 1 - s[a]["kl"] / s["empty"]["kl"] if s["empty"]["kl"] > 1e-9 else None
        out[split] = s
    return out


# ---------------------------------------------------------------- the leaderboard

def bootstrap(xs: list, draws: int = 2000) -> list | None:
    xs = np.array([x for x in xs if x is not None and math.isfinite(x)], float)
    if len(xs) < 2:
        return None
    m = xs[np.random.default_rng(0).integers(0, len(xs), (draws, len(xs)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def board(dirs: list) -> list:
    """Leaderboard lines (dicts) of scored sets: per set and rung, n circuits, the mean of each record's held-out summary
    with its bootstrap interval, the random control's mean, and the median size and finder seconds."""
    lines = []
    for d in dirs:
        recs = [json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json")) if p.name != "meta.json"]
        meta = json.loads((Path(d) / "meta.json").read_text()) if (Path(d) / "meta.json").exists() else {}
        for split in ("own", "heldout"):
            S = [r["summary"][split] for r in recs if split in r.get("summary", {})]
            if not S:
                continue
            row = {"method": meta.get("label", Path(d).name), "rung": "R3", "inputs": split, "n": len(S),
                   "background": meta.get("mode", MODE)}
            for m in ("recovered_ce", "recovered_kl", "ce", "kl", "pass", "pass_abs"):
                xs = [s["circuit"].get(m) for s in S]
                xs = [x for x in xs if x is not None]
                row[m] = float(np.mean(xs)) if xs else None
                row[m + "_ci"] = bootstrap(xs)
                rs = [s.get("random", {}).get(m) for s in S]
                rs = [x for x in rs if x is not None]
                row[m + "_random"] = float(np.mean(rs)) if rs else None
            for m in ("ce", "kl"):
                row[m + "_empty"] = float(np.mean([s["empty"][m] for s in S]))
                row[m + "_full"] = float(np.mean([s["full"][m] for s in S]))
            row["full_pass_abs"] = float(np.mean([s["full"]["pass_abs"] for s in S]))
            row["units_median"] = float(np.median([r["size"]["units"] for r in recs]))
            row["edges_median"] = float(np.median([r["size"]["edges"] for r in recs]))
            secs = [r.get("seconds_found") for r in recs if r.get("seconds_found") is not None]
            row["seconds_median"] = float(np.median(secs)) if secs else None
            lines.append(row)
    return lines


def tsv(lines: list) -> str:
    cols = ["method", "rung", "inputs", "background", "n", "recovered_ce", "recovered_ce_ci", "recovered_ce_random", "ce", "ce_ci",
            "ce_random", "ce_empty", "ce_full", "pass", "pass_random", "pass_abs", "full_pass_abs", "kl", "kl_ci", "kl_random", "kl_empty",
            "recovered_kl", "recovered_kl_random", "units_median", "edges_median", "seconds_median"]
    f = lambda v: "" if v is None else (f"[{v[0]:.3f}, {v[1]:.3f}]" if isinstance(v, list) else f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    return "\n".join(["\t".join(cols)] + ["\t".join(f(x.get(c)) for c in cols) for x in lines])


# ---------------------------------------------------------------- command line

def load(model_id: str, dtype: str, device: str = "auto"):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if dtype == "auto":
        cfg = AutoConfig.from_pretrained(model_id)
        n = cfg.num_hidden_layers * (4 * cfg.hidden_size ** 2 + 3 * cfg.hidden_size * cfg.intermediate_size)
        dtype = "bfloat16" if n >= 4e9 else "float32"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype),
                                                 device_map="auto" if device == "auto" else None)
    model = (model if device == "auto" else model.to(device)).eval().requires_grad_(False)
    return model, AutoTokenizer.from_pretrained(model_id), dtype


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("out")
    s.add_argument("--tokens", type=int, default=1_100_000)
    s.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("means")
    m.add_argument("sample")
    m.add_argument("out")
    r = sub.add_parser("r3")
    r.add_argument("records")
    r.add_argument("out")
    r.add_argument("--means", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--devset", default=str(DEVSET))
    r.add_argument("--pool", default=str(POOL))
    r.add_argument("--mode", default=MODE, choices=("bucket", "global"))
    r.add_argument("--draws", type=int, default=DRAWS)
    r.add_argument("--random", type=int, default=RANDOM)
    r.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("board")
    b.add_argument("dirs", nargs="+")
    for p in (m, r):
        p.add_argument("--model", default="Qwen/Qwen3-8B")
        p.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
        p.add_argument("--device", default="auto")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda x: print(f"{time.time() - t0:7.1f}s {x}", flush=True)  # noqa: E731
    if a.cmd == "sample":
        d = sample(Path(a.out), a.tokens, a.seed)
        log(f"{len(d['offsets']) - 1} documents, {len(d['tokens']):,} tokens -> {a.out}")
        return
    if a.cmd == "board":
        print(tsv(board(a.dirs)))
        return
    cf._threads()
    model, tok, dtype = load(a.model, a.dtype, a.device)
    if a.cmd == "means":
        d = np.load(a.sample)
        off, tk = d["offsets"], d["tokens"]
        docs = [tk[off[j]:off[j + 1]].tolist() for j in range(len(off) - 1)]
        acc = collect(model, docs, log=log)
        torch.save({**acc, "model": a.model, "dtype": dtype, "sample": str(a.sample), "documents": len(docs),
                    "tokens": int(sum(map(len, docs)))}, a.out)
        log(f"means of {a.model} over {sum(map(len, docs)):,} tokens -> {a.out}")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"label": a.label, "records": a.records, "means": a.means, "mode": a.mode,
                                               "model": a.model, "dtype": dtype, "draws": a.draws, "random": a.random,
                                               "target": TARGET}))
    abl = Ablator(model, tok, Means(a.means, abl_dev(model)), mode=a.mode)
    qs = load_questions(a.devset)
    keys = task_keys({**load_questions(a.pool), **qs})
    paths = sorted(p for p in Path(a.records).glob("*.json") if p.name != "meta.json")[:a.limit or None]
    for p in paths:
        dst = out / p.name
        if dst.exists():
            continue
        rec = json.loads(p.read_text())
        if rec.get("skipped") or not rec.get("ids"):
            continue
        q = qs.get(rec.get("id")) or {k: rec.get(k) for k in ("user", "key", "corrupt_key", "task", "prefill", "system") if rec.get(k)}
        res = r3(abl, tok, rec, q, keys.get(rec.get("task") or q.get("task"), []), a.draws, a.random)
        dst.write_text(json.dumps(res))
        s = res["summary"]
        log(f"{rec.get('id')}: {res['size']['units']} units; own ce {s['own']['circuit']['ce']:.2f} (empty "
            f"{s['own']['empty']['ce']:.2f}, full {s['own']['full']['ce']:.2f}), kl {s['own']['circuit']['kl']:.2f}"
            + (f"; held-out {s['heldout']['n']} ce {s['heldout']['circuit']['ce']:.2f} random {s['heldout']['random']['ce']:.2f}"
               if "heldout" in s else "") + f", {res['seconds']} s")


def abl_dev(model):
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    return core.model.embed_tokens.weight.device


if __name__ == "__main__":
    main()
