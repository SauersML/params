"""The circuit finder's scorecard (#2951; the lead's 10-15 "CIRCUIT FINDER = TOP PRIORITY", ~/mpd-data/team/direction_1015.txt):
whether a question's neuron circuit on an HF Qwen3 (8B the target, 32B the showcase) is faithful by the standard patching
pair, judged on the circuit itself on held-out questions (circuit_questions' split), never through the oracle. It decides
when circuits are good enough to train the oracle on.
  units      neuron poles at their positions (the down projection's input, act(gate) x up: [layer, position, neuron,
             pole]) and attention heads as routes (a head's output at a position, its slice of o_proj's input: [layer,
             position, head])
  runs       the model itself, two runs of it (no base or reference model): clean (the question) and corrupted (a
             minimally changed prompt of the same length: the key word swapped); float32 (bfloat16 with --dtype for
             32B), every batch padded to ROWS rows (bfloat16 moves contrasts between batch sizes)
  contrast   at the answer position, log2 p(answer) - log2 p(alternatives) over words, not tokens (answer_contrast,
             the finder's own: every likely token starting the clean run's answer word summed into p(answer), the
             alternatives the tokens starting the other run's word and the next likeliest other words of the same
             script; a pair whose other run starts the answer's word is no pair)
  patch      a unit set to its value in the other run at the same position
Families, each a score with its floor (FLOOR):
  necessity    noising: the circuit's units patched from corrupted into clean, the share of the clean-corrupted contrast
               gap lost, (c_clean - c_noised) / (c_clean - c_corr), on the circuit's own pair
  sufficiency  denoising: only the circuit's units patched from clean into corrupted, the share of the gap restored,
               (c_denoised - c_corr) / (c_clean - c_corr), on the circuit's own pair
  necessity_heldout, sufficiency_heldout   the same on held-out corruptions: the question's other partners (the same
               template with another same-length key word of its split, not the circuit's own), up to DRAWS, each
               share clipped to [-1, 1], the mean
  swaps        activation swaps within the task: on each partner's prompt the circuit's units set to their values there,
               in the clean run: the share of the way the partner's answer against the question's moves from the clean
               run's toward the partner's own run's, (d_swap - d_clean) / (d_partner - d_clean), clipped, the mean
  stability    over the question's paraphrases' circuits (the same task and key word): the share of this circuit's
               measured effect carried by units the other paraphrase's circuit also holds, the mean over the others.
               Each unit, position-free (a neuron pole: layer, neuron, pole; a head: layer, head), weighs its noising
               drop alone over the circuit's summed drops (drops below zero count zero; one key's positions summed), so
               the same mechanism counts by how much of the circuit it is (the lead's 10-15: at a median of 14 poles the
               8th hub is noise). Beside it, no floor: stability_hubs, per hub neuron (the HUB neuron poles with the
               largest drop alone) the share of the other paraphrases' circuits holding it, the mean
Size is measured, never capped (the user, 10-15: the circuit is the smallest set that passes, as large as the behaviour
needs): neuron poles, heads, edges and build seconds are reported as found. Every circuit also reports its curves: the
share of the answer's log-odds gap lost (noising its first k units) and restored (denoising its first k units) against
k, its units added in their measured order (each unit's noising drop alone, for the loss curve; its denoising gain alone,
for the restoration curve), every k up to CURVE_ALL and a log grid beyond with the 70% crossing bisected; tables quote
the 70% points (k70) and nothing depends on them.
A fair baseline beside each circuit (random): RANDOM_DRAWS circuits of the same size at the same positions, each neuron
pole a uniformly drawn neuron of the same layer's at that position, each head a uniformly drawn head of the same layer's
there; their noising and denoising shares on the circuit's own pair, the mean (no floor: what the finder's choice adds
over chance at that size).
A pair whose clean-corrupted gap is under MIN_GAP bits, or whose two runs give the same answer, is no draw. Noise-aware
floors: a family scored over draws passes at its floor less min(PAIRED_SE x its bootstrap standard error over the draws,
CAP of the family); the own pair's families are one run each and get no allowance. A family left unscored (no valid
draw, a single paraphrase, no build time) neither passes nor fails on its own, and a circuit meets every family only
with every family measured and passed (6320c95's rule). Every draw's numbers are in the record, so scores can be
recomputed (summary()).

  neuron_scorecard.py score CIRCUIT [CIRCUIT ...] --out OUT [--questions FILE] [--dtype bfloat16] [--device auto]
      score circuit files (or directories of them) into OUT/<name>.json, OUT/rows.jsonl and the table
  neuron_scorecard.py table NAME=OUT [NAME=OUT ...]   finders side by side on the questions every set scored
  neuron_scorecard.py motifs VIEWS.json RECORDS_DIR [...] [--out FILE]   the essay's motif score on neuron_motifs views'
      run: per set (a group, or the random set of as many poles beside it) the correlation over its unseen questions
      between its usage (the share of its poles in the question's circuit) and the damage its ablation does (neuron_motifs'
      measured drop), the median per kind (the essay's ImageNet edge motifs: 0.35, activation factors: 0.00)
  neuron_scorecard.py check MODEL --out FILE [--questions FILE]   every question's patching pair on MODEL: the two
      answers, the gap and whether the pair is valid (the questions a finder can build on, and those the scorecard scores)

A circuit file: circuit_finder's record (the only circuit builder: its "ids" and "corrupted" token ids; its source
nodes and its routes, the movers, as the circuit; its readout nodes left out) or a validation case {"id", "neurons":
[[layer, pos, neuron, pole]], "heads": [[layer, pos, head]]}; the prompts from circuit_questions' file by id, or in the file ("user" and "corrupt_user",
or token ids "clean" and "corrupted": a validation case outside the question set, with "group" naming the circuits whose
hubs stability compares, and "expect": {"neurons": [[layer, neuron, pole]], "heads": [[layer, head]]}, the published
units the case's circuit must hold, reported as their recall); "edges" (rustmove's) counted.
"""
from __future__ import annotations

import argparse
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

sys.path.insert(0, str(HERE.parents[1] / "method" / "oracle"))

import answer_contrast as ac  # noqa: E402
import circuit_questions  # noqa: E402

BASE = "Qwen/Qwen3-8B"
ORG = Path.home() / "mpd-data" / "organisms" / "qwen3_1p7b"
DRAWS = 8  # held-out corruptions and swap partners per circuit
HUB = 8  # a circuit's hub neurons: its neuron poles with the largest noising effect alone
ROWS = 16  # runs a forward, every batch padded to it
MIN_GAP = 1.0  # bits: the least clean-corrupted contrast gap a pair needs
FLOOR = {"necessity": 0.7, "sufficiency": 0.7, "necessity_heldout": 0.7, "sufficiency_heldout": 0.7, "swaps": 0.7,
         "stability": 0.6}
DRAWN = ("necessity_heldout", "sufficiency_heldout", "swaps")
LOWER = ()  # families whose score passes at or under its floor (none: size is measured, never a criterion)
CURVE_ALL = 256  # the curves measure every k up to this many units, a log grid beyond
POINT = 0.7  # the share whose crossing the tables quote (k70)
RANDOM_DRAWS = 4  # random circuits of the circuit's size per circuit (the fair baseline)
PAIRED_SE = 2.0  # goal_eval's: a drawn family passes at its floor less this many standard errors, at most CAP
CAP: dict = {}  # per family, PAIRED_SE x the median standard error over the first held-out set's circuits (until then uncapped)
EPS_BOOT = 200


def read_circuit(path: Path) -> dict:
    """A circuit file in either accepted form (module docstring) as {name, id, neurons, heads, edges, model, ...}."""
    c = json.loads(path.read_text())
    nodes = [n for n in c.get("nodes", []) if n.get("stage") != "readout"]  # circuit_finder: source and movers, not its readout cut
    neurons = c.get("neurons") or [[n["layer"], n["pos"], n["neuron"], n.get("pole", 0)] for n in nodes]
    heads = c.get("heads") or [[r["layer"], r["pos"], r["head"]] for r in c.get("routes", [])]
    c.update(name=path.stem, neurons=[tuple(int(x) for x in (list(u) + [0])[:4]) for u in neurons],
             heads=[tuple(int(x) for x in u) for u in heads], model=c.get("model") or BASE,
             build_seconds=c.get("build_seconds") or c.get("seconds"), edges=c.get("edges") or [])
    return c


def load_model(base: str, adapter: str | None, dev: str, dtype: torch.dtype = torch.float32):
    """base (an HF id or directory) with adapter: a directory, an HF id, or an organism's name under ORG/adapters."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=dev).eval()  # (straight to the device: one copy)
    if adapter:
        from peft import PeftModel

        d = adapter if Path(adapter).is_dir() or "/" in adapter else str(ORG / "adapters" / adapter)
        model = PeftModel.from_pretrained(model, d).eval()
    model.requires_grad_(False)
    return model


class Model:
    """An HF decoder with every layer's neurons (the down projection's input) and heads (o_proj's input, per head) open to
    per-row patches: a run is (ids, patches), patches a list of ("n", layer, position, neuron, value) and ("h", layer,
    position, head, value [hd])."""

    def __init__(self, model):
        self.model = model
        self.core = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = self.core.config
        self.layers = list(self.core.model.layers)
        self.L, self.D, self.H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev = next(self.core.parameters()).device

    @torch.no_grad()
    def capture(self, ids: list[int]) -> dict:
        """{"n": [L, T, D], "h": [L, T, H, hd]} on ids, float32."""
        cap = {"n": [], "h": []}
        hs = []
        for layer in self.layers:
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: cap["n"].append(a[0][0].float().to(self.dev))))
            hs.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, a: cap["h"].append(a[0][0].float().view(a[0].shape[1], self.H, self.hd).to(self.dev))))
        try:
            self.core.model(input_ids=torch.tensor([ids], device=self.dev))
        finally:
            for h in hs:
                h.remove()
        return {"n": torch.stack(cap["n"]), "h": torch.stack(cap["h"])}

    @torch.no_grad()
    def logp(self, runs: list[tuple]) -> torch.Tensor:
        """log p at each run's last position [R, V], ROWS runs a forward (padded with the first run)."""
        out = []
        for s0 in range(0, len(runs), ROWS):
            part = list(runs[s0:s0 + ROWS])
            n = len(part)
            part += [part[0]] * (ROWS - n)
            idx = {}
            for b, (_, patches) in enumerate(part[:n]):
                for kind, L, p, u, v in patches:
                    idx.setdefault((kind, L), []).append((b, p, u, v))
            plan = {}
            for (kind, L), es in idx.items():
                t = lambda k: torch.tensor([e[k] for e in es], device=self.dev)  # noqa: E731
                vals = torch.stack([torch.as_tensor(e[3], dtype=torch.float32, device=self.dev).reshape(-1) for e in es])
                plan[(kind, L)] = (t(0), t(1), t(2), vals)
            hs = []
            for L, layer in enumerate(self.layers):
                def pre_n(m, a, L=L):
                    if ("n", L) not in plan:
                        return None
                    x = a[0]
                    b, p, u, v = (z.to(x.device) for z in plan[("n", L)])
                    return (x.index_put((b, p, u), v[:, 0].to(x.dtype)),) + tuple(a[1:])

                def pre_h(m, a, L=L):
                    if ("h", L) not in plan:
                        return None
                    x = a[0]
                    B, T = x.shape[:2]
                    b, p, u, v = (z.to(x.device) for z in plan[("h", L)])
                    y = x.view(B, T, self.H, self.hd).index_put((b, p, u), v.to(x.dtype))
                    return (y.view(B, T, -1),) + tuple(a[1:])
                hs += [layer.mlp.down_proj.register_forward_pre_hook(pre_n), layer.self_attn.o_proj.register_forward_pre_hook(pre_h)]
            try:
                x = torch.tensor([r[0] for r in part], device=self.dev)
                h = self.core.model(input_ids=x).last_hidden_state[:n, -1]
                out.append(torch.log_softmax(self.core.lm_head(h).float(), -1).to(self.dev))
            finally:
                for hd in hs:
                    hd.remove()
        return torch.cat(out)


def patches(c: dict, cap: dict) -> list[tuple]:
    """The circuit's units set to their values in cap (Model.capture)."""
    return [("n", L, p, j, cap["n"][L, p, j]) for L, p, j, _ in c["neurons"]] + [("h", L, p, h, cap["h"][L, p, h]) for L, p, h in c["heads"]]


def contrast(lp: torch.Tensor, sets: dict) -> float:
    """answer_contrast.contrast of one run's log probabilities [V]."""
    return float(ac.contrast(lp[None], sets)[0])


def word_sets(m: "Model", tok, ids: list[int], lp: torch.Tensor, b: int) -> dict:
    """answer_contrast.token_sets on the clean run (its log probabilities lp) against the other run's top token b."""
    return ac.token_sets(m.core, tok, ids, lp, b)


def pair_ok(sets: dict, b: int, gap: float) -> bool:
    """A pair: the other run's top token starts another word, and the clean-other gap is at least MIN_GAP bits."""
    return b not in sets["answer"] and gap >= MIN_GAP


def share(x: float) -> float:
    return min(1.0, max(-1.0, x))


def grid(n: int) -> list[int]:
    """Every k up to CURVE_ALL, then two steps an octave, n included."""
    ks = set(range(1, min(n, CURVE_ALL) + 1))
    e = 2 * math.log2(max(CURVE_ALL, 1))
    while n > CURVE_ALL and round(2 ** (e / 2)) < n:
        ks.add(int(round(2 ** (e / 2))))
        e += 1
    return sorted(ks | ({n} if n else set()))


@torch.no_grad()
def curve(m: Model, ids: list[int], ordered: list[tuple], measure) -> dict:
    """The share (measure of a run's log p) with the first k patches of ordered applied to ids, on grid(len(ordered)),
    and k70: the least k whose share reaches POINT (bisected between grid points)."""
    n = len(ordered)
    ks = grid(n)
    got = dict(zip(ks, (measure(lp) for lp in m.logp([(ids, ordered[:k]) for k in ks])))) if n else {}
    hit = next((k for k in ks if got[k] is not None and got[k] >= POINT), None)
    if hit is not None:
        lo = max([k for k in ks if k < hit] + [0])
        while hit - lo > 1:  # a log-grid gap: bisect
            mid = (lo + hit) // 2
            got[mid] = measure(m.logp([(ids, ordered[:mid])])[0])
            if got[mid] is not None and got[mid] >= POINT:
                hit = mid
            else:
                lo = mid
    ks = sorted(got)
    return {"k": ks, "share": [got[k] for k in ks], "k70": hit, "units": n}


@torch.no_grad()
def score(m: Model, tok, c: dict, qs: dict, log=print) -> dict:
    t0 = time.perf_counter()
    q = qs.get(c.get("id"), {})
    chat = lambda u: circuit_questions.chat(tok, u)  # noqa: E731
    clean = c.get("clean") or c.get("ids") or chat(c.get("user") or q["user"])
    corr = c.get("corrupted") or chat(c.get("corrupt_user") or q["corrupt_user"])
    T = len(clean)
    units_ok = all(p < T for _, p, _, _ in c["neurons"]) and all(p < T for _, p, _ in c["heads"])
    rec = {"task": c["name"], "id": c.get("id"), "question_task": q.get("task"), "key": q.get("key"), "paraphrase": q.get("paraphrase"),
           "split": q.get("split"), "builder": c.get("finder"), "model": c["model"], "adapter": c.get("adapter"), "T": T,
           "build_seconds": c.get("build_seconds"),
           "size": {"poles": len(c["neurons"]), "heads": len(c["heads"]), "edges": len(c["edges"]),
                    "neurons": len({(L, j) for L, _, j, _ in c["neurons"]}),
                    "positions": len({p for _, p, _, _ in c["neurons"]} | {p for _, p, _ in c["heads"]}),
                    "poles_signed": {"+": sum(s > 0 for *_, s in c["neurons"]), "-": sum(s < 0 for *_, s in c["neurons"])}}}
    if len(corr) != T or not units_ok:
        rec["error"] = "the corrupted prompt's length differs from the clean one's" if len(corr) != T else "a unit past the prompt"
        rec["draws"] = {f: [] for f in DRAWN}
        rec["margins"] = margins(rec)
        return rec
    cap_c, cap_x = m.capture(clean), m.capture(corr)
    pc, px = patches(c, cap_c), patches(c, cap_x)
    lp = m.logp([(clean, []), (corr, []), (clean, px), (corr, pc)])
    a, b = int(lp[0].argmax()), int(lp[1].argmax())
    sets = word_sets(m, tok, clean, lp[0], b)
    cc, cx, cn, cd = (contrast(lp[i], sets) for i in range(4))
    gap = cc - cx
    rec["answer"], rec["answer_text"], rec["corrupt_answer"] = a, tok.decode([a]), b
    rec["answer_word"], rec["alternative_words"] = sets["answer_word"], sets["alternative_words"]
    rec["answer_tokens"], rec["alternative_tokens"] = sets["answer"], sets["alternatives"]
    rec["pair"] = {"c_clean": cc, "c_corrupted": cx, "c_noised": cn, "c_denoised": cd, "gap": gap, "valid": pair_ok(sets, b, gap),
                   "lost": (cc - cn) / gap if gap > 0 else None, "restored": (cd - cx) / gap if gap > 0 else None}
    g = torch.Generator().manual_seed(zlib.crc32(c["name"].encode()))  # the fair baseline: same layers and positions, random units
    rand = []
    for _ in range(RANDOM_DRAWS if gap > 0 else 0):
        rc = {"neurons": [(L, p, int(torch.randint(m.D, (1,), generator=g)), 0) for L, p, _, _ in c["neurons"]],
              "heads": [(L, p, int(torch.randint(m.H, (1,), generator=g))) for L, p, _ in c["heads"]]}
        l3 = m.logp([(clean, patches(rc, cap_x)), (corr, patches(rc, cap_c))])
        rand.append({"lost": (cc - contrast(l3[0], sets)) / gap, "restored": (contrast(l3[1], sets) - cx) / gap})
    rec["random"] = {"lost": float(np.mean([x["lost"] for x in rand])) if rand else None,
                     "restored": float(np.mean([x["restored"] for x in rand])) if rand else None, "draws": rand}
    # each unit's noising effect alone (the hubs: stability)
    singles = [("n", u) for u in c["neurons"]] + [("h", u) for u in c["heads"]]

    def alone(k, u):  # the unit's value from the corrupted run
        return ("n", u[0], u[1], u[2], cap_x["n"][u[0], u[1], u[2]]) if k == "n" else ("h", u[0], u[1], u[2], cap_x["h"][u[0], u[1], u[2]])
    one = m.logp([(clean, [alone(k, u)]) for k, u in singles]) if singles else []
    eff = [cc - contrast(lp1, sets) for lp1 in one]
    def gained(k, u):  # the unit's value from the clean run
        return ("n", u[0], u[1], u[2], cap_c["n"][u[0], u[1], u[2]]) if k == "n" else ("h", u[0], u[1], u[2], cap_c["h"][u[0], u[1], u[2]])
    two = m.logp([(corr, [gained(k, u)]) for k, u in singles]) if singles else []
    gain = [contrast(lp1, sets) - cx for lp1 in two]
    rec["effects"] = [{"kind": k, "unit": list(u), "drop": e, "gain": g} for (k, u), e, g in zip(singles, eff, gain)]
    rec["curve"] = {"necessity": curve(m, clean, [alone(*singles[i]) for i in sorted(range(len(singles)), key=lambda i: -eff[i])],
                                       lambda lp1: (cc - contrast(lp1, sets)) / gap if gap > 0 else None),
                    "sufficiency": curve(m, corr, [gained(*singles[i]) for i in sorted(range(len(singles)), key=lambda i: -gain[i])],
                                         lambda lp1: (contrast(lp1, sets) - cx) / gap if gap > 0 else None)}
    nn = [(e, u) for (k, u), e in zip(singles, eff) if k == "n"]
    rec["hubs"] = [[u[0], u[2], u[3]] for e, u in sorted(nn, key=lambda t: -t[0])[:HUB]]
    rec["neuron_set"] = sorted({(L, j, s) for L, _, j, s in c["neurons"]})
    rec["group"] = c.get("group")
    if c.get("expect"):  # a validation case's published units (V1-V3), position-free
        en, eh = [tuple(x) for x in c["expect"].get("neurons", [])], [tuple(x) for x in c["expect"].get("heads", [])]
        have_n, have_h = set(rec["neuron_set"]), {(L, h) for L, _, h in c["heads"]}
        rec["expected"] = {"neurons": sum(x in have_n for x in en) / len(en) if en else None,
                           "heads": sum(x in have_h for x in eh) / len(eh) if eh else None,
                           "missing": [list(x) for x in en if x not in have_n] + [list(x) for x in eh if x not in have_h]}
    # held-out corruptions and swaps: the question's other partners
    draws = {f: [] for f in DRAWN}
    if q:
        partners = [k for k in q["partners"] if k != (c.get("corrupt_key") or q["corrupt_key"])][:DRAWS]
        for k in partners:
            pq = qs.get(f"{q['task']}/{k}/p{q['paraphrase']}")
            if pq is None:
                continue
            ids = chat(pq["user"])
            if len(ids) != T:
                continue
            cap_p = m.capture(ids)
            l2 = m.logp([(ids, []), (clean, patches(c, cap_p)), (ids, pc)])
            bp = int(l2[0].argmax())
            sp = word_sets(m, tok, clean, lp[0], bp)  # the question's answer against the partner's word first
            c0, c1 = contrast(lp[0], sp), contrast(l2[0], sp)
            g = c0 - c1
            if not pair_ok(sp, bp, g):
                continue
            cn2, cd2 = contrast(l2[1], sp), contrast(l2[2], sp)
            draws["necessity_heldout"].append({"partner": k, "gap": g, "c_noised": cn2, "share": (c0 - cn2) / g})
            draws["sufficiency_heldout"].append({"partner": k, "gap": g, "c_denoised": cd2, "share": (cd2 - c1) / g})
            pw = word_sets(m, tok, ids, l2[0], a)  # the partner's own answer word, its tokens
            A_p, A_q = torch.tensor(pw["answer"]), torch.tensor(sp["answer"])
            d = lambda l: float((torch.logsumexp(l[A_p.to(l.device)], 0) - torch.logsumexp(l[A_q.to(l.device)], 0)) / math.log(2.0))  # noqa: E731
            d0, dp, ds = d(lp[0]), d(l2[0]), d(l2[1])
            if pw["answer_word"] != sp["answer_word"] and abs(dp - d0) >= MIN_GAP:
                draws["swaps"].append({"partner": k, "d_clean": d0, "d_partner": dp, "d_swap": ds, "share": (ds - d0) / (dp - d0)})
    rec["draws"] = draws
    rec["margins"] = margins(rec)
    rec["seconds"] = time.perf_counter() - t0
    fails = [f for f, mm in rec["margins"].items() if mm["pass_new"] is False]
    log(f"{c['name']}: {rec['size']['poles']} poles, {rec['size']['heads']} heads; answer {rec['answer_text']!r}; gap {gap:.2f} bits; "
        f"lost {rec['pair']['lost']}, restored {rec['pair']['restored']}; held-out draws {len(draws['necessity_heldout'])}; "
        f"failing {fails or 'none'}; {rec['seconds']:.1f} s")
    return rec


def unit_shares(rec: dict) -> dict:
    """{position-free unit key: its share of the circuit's summed noising drops alone} (module docstring: stability)."""
    w = {}
    for e in rec.get("effects", []):
        u = e["unit"]
        k = ("n", u[0], u[2], u[3]) if e["kind"] == "n" else ("h", u[0], u[2])
        w[k] = w.get(k, 0.0) + max(float(e["drop"]), 0.0)
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()} if tot > 0 else {}


def stability(recs: list[dict]) -> None:
    """Each record's stability draws (per other paraphrase circuit of the same task and key word, scored in recs: the
    share of this circuit's measured effect on units that circuit holds) and its stability_hubs draws (per hub neuron:
    the share of the other paraphrases' circuits holding it), in place."""
    def gk(r):  # the question (task and key word) or the validation case's group, per finder
        return (r.get("group") or (r.get("question_task"), r.get("key")), r.get("builder"))
    groups = {}
    for r in recs:
        if r.get("question_task") is not None or r.get("group"):
            groups.setdefault(gk(r), []).append(r)
    for r in recs:
        others = [o for o in groups.get(gk(r), []) if o is not r]
        sets = [{tuple(x) for x in o.get("neuron_set", [])} for o in others]
        r["stability_hubs"] = [{"hub": h, "share": sum(tuple(h) in s for s in sets) / len(sets)} for h in r.get("hubs", [])] if sets else []
        w = unit_shares(r)
        r["stability"] = [{"other": o.get("id") or o["task"], "share": sum(v for k, v in w.items() if k in set(unit_shares(o)))}
                          for o in others] if w else []
        r["margins"] = margins(r)


def family_score(rec: dict, f: str, draws: list[dict] | None = None) -> float | None:
    """A family's score from the record's numbers (draws: a resample of the family's draws)."""
    pr = rec.get("pair") or {}
    if f in ("necessity", "sufficiency"):
        x = pr.get("lost" if f == "necessity" else "restored")
        return None if x is None or not pr.get("valid") else share(x)
    if f == "stability":
        d = (rec.get("stability") or []) if draws is None else draws
        return float(np.mean([x["share"] for x in d])) if d else None
    d = rec["draws"].get(f, []) if draws is None else draws
    return float(np.mean([share(x["share"]) for x in d])) if d else None


def margins(rec: dict) -> dict:
    """Per family: score, floor, bootstrap SE over its draws, allowance and whether it passes the floor exactly (old),
    noise-aware (capped: score >= floor - min(PAIRED_SE se, CAP)) and uncapped (score >= floor - PAIRED_SE se)."""
    rng = np.random.default_rng(0)
    out = {}
    for f, fl in FLOOR.items():
        s = family_score(rec, f)
        se = 0.0
        d = rec["draws"].get(f, []) if f in DRAWN else []
        if len(d) >= 2:
            bs = [family_score(rec, f, [d[i] for i in idx]) for idx in rng.integers(len(d), size=(EPS_BOOT, len(d)))]
            se = float(np.std([x for x in bs if x is not None], ddof=1))
        cap = CAP.get(f)
        allow = PAIRED_SE * se if cap is None else min(PAIRED_SE * se, cap)
        sg = -1 if f in LOWER else 1

        def ok(x):
            return None if s is None else sg * s >= sg * fl - x
        out[f] = {"score": s, "floor": fl, "se": se, "allowance": allow, "capped": cap is not None,
                  "pass_old": ok(0.0), "pass_new": ok(allow), "pass_uncapped": ok(PAIRED_SE * se)}
    return out


def summary(recs: list[dict]) -> dict:
    """Per family: circuits passing (old, noise-aware, uncapped) and the median score; sizes and seconds."""
    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None
    out = {"circuits": len(recs), "families": {}}
    every = [margins(r) for r in recs]
    for f in FLOOR:
        ms = [m[f] for m in every]
        out["families"][f] = {"score": med([m["score"] for m in ms]), "old": sum(m["pass_old"] is True for m in ms),
                              "noise": sum(m["pass_new"] is True for m in ms),
                              "uncapped": sum(m["pass_uncapped"] is True for m in ms),
                              "unscored": sum(m["score"] is None for m in ms)}
    out["every_family"] = {k: sum(all(m[f][k] is True for f in FLOOR) for m in every) for k in ("pass_old", "pass_new", "pass_uncapped")}
    for k in ("poles", "heads", "neurons", "positions", "edges"):
        out[k] = med([r["size"][k] for r in recs])
    out["build_seconds"] = med([r.get("build_seconds") for r in recs])
    for f in ("necessity", "sufficiency"):  # the curves' 70% points: the units needed, as found
        ks = [((r.get("curve") or {}).get(f) or {}).get("k70") for r in recs]
        out[f"k70_{f}"] = med(ks)
        out[f"reach70_{f}"] = sum(k is not None for k in ks)
    out["valid_pairs"] = sum(bool((r.get("pair") or {}).get("valid")) for r in recs)
    for k in ("lost", "restored"):  # the fair baseline: random circuits of the same size at the same places
        out[f"random_{k}"] = med([(r.get("random") or {}).get(k) for r in recs])
    for k in ("neurons", "heads"):  # validation cases: the published units found
        out[f"expected_{k}"] = med([(r.get("expected") or {}).get(k) for r in recs])
    out["score_seconds"] = med([r.get("seconds") for r in recs])
    hub = [float(np.mean([x["share"] for x in r["stability_hubs"]])) if r.get("stability_hubs") else None for r in recs]
    out["stability_hubs"] = {"score": med(hub), "pass": sum(h is not None and h >= FLOOR["stability"] for h in hub),
                             "unscored": sum(h is None for h in hub)}
    return out


def table(sets: dict) -> str:
    """The sets (name -> records) side by side on the questions every set scored."""
    common = set.intersection(*[{r["task"] for r in v} for v in sets.values()]) if sets else set()
    sm = {n: summary([r for r in v if r["task"] in common]) for n, v in sets.items()}
    w = {n: max(14, len(n) + 2) for n in sets}
    lines = [f"{len(common)} circuits scored by every set; old: the floor exactly; noise: less min({PAIRED_SE:g} SE, cap); "
             f"uncapped: less {PAIRED_SE:g} SE" + ("" if CAP else " (no cap measured yet: noise = uncapped)"),
             f"{'':42}" + "".join(f"{n:>{w[n]}}" for n in sets)]

    def row(label, get, fmt="{:.3f}"):
        vals = []
        for n in sets:
            v = get(sm[n])
            vals.append(f"{(fmt.format(v) if isinstance(v, float) else str(v)) if v is not None else '-':>{w[n]}}")
        lines.append(f"  {label:40}" + "".join(vals))
    row("circuits with a valid pair", lambda s: s["valid_pairs"], "{:.0f}")
    row("random same-size circuits: lost (median)", lambda s: s["random_lost"])
    row("random same-size circuits: restored", lambda s: s["random_restored"])
    for f in FLOOR:
        row(f"{f} ({'at most' if f in LOWER else 'floor'} {FLOOR[f]:g}) median", lambda s, f=f: s["families"][f]["score"])
        row("  pass old / noise / uncapped / unscored", lambda s, f=f: "{old}/{noise}/{uncapped}/{unscored}".format(**s["families"][f]))
    row("every family old / noise / uncapped", lambda s: "{pass_old}/{pass_new}/{pass_uncapped}".format(**s["every_family"]))
    row(f"stability_hubs (top {HUB}, no floor) median", lambda s: s["stability_hubs"]["score"])
    row(f"  at least {FLOOR['stability']:g} / unscored", lambda s: "{pass}/{unscored}".format(**s["stability_hubs"]))
    row("size: neuron poles (as found)", lambda s: s["poles"], "{:.0f}")
    row("size: heads", lambda s: s["heads"], "{:.0f}")
    for k in ("neurons", "positions", "edges"):
        row(f"size: {k}", lambda s, k=k: s[k], "{:.0f}")
    row("k70 noising: units to lose 70%", lambda s: s["k70_necessity"], "{:.0f}")
    row("  circuits whose curve reaches 70%", lambda s: s["reach70_necessity"], "{:.0f}")
    row("k70 denoising: units to restore 70%", lambda s: s["k70_sufficiency"], "{:.0f}")
    row("  circuits whose curve reaches 70%", lambda s: s["reach70_sufficiency"], "{:.0f}")
    row("build seconds", lambda s: s["build_seconds"], "{:.1f}")
    row("validation: published neurons found", lambda s: s["expected_neurons"])
    row("validation: published heads found", lambda s: s["expected_heads"])
    row("scoring seconds", lambda s: s["score_seconds"], "{:.1f}")
    return "\n".join(lines)


def motif_scores(run: dict, recs: dict) -> dict:
    """The essay's usage-damage score per set of a neuron_motifs views run (its groups and the random sets beside them,
    run["damage"]): Pearson and Spearman correlations over its unseen questions (run["damage"]["test"]) between usage,
    the share of the set's poles among the question's circuit's poles (neuron_motifs.poles_of), and damage (the set's
    measured drop there); per kind the median and the sets scored (a set used by no question, or by every question
    alike, has no correlation)."""
    import neuron_motifs

    def corr(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return None if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    def ranks(v):  # average ranks (ties share theirs)
        from scipy.stats import rankdata
        return rankdata(v)
    dmg = run["damage"]
    circ = [neuron_motifs.poles_of(recs[i]) if i in recs else None for i in dmg["test"]]
    rows = []
    for row in dmg["rows"]:
        ps = set(row["poles"])
        pairs = [(len(c & ps) / len(ps), d) for c, d in zip(circ, row["damage"]) if c is not None]
        u, d = [p[0] for p in pairs], [p[1] for p in pairs]
        rows.append({"kind": row["kind"], "rank": row["rank"], "size": len(ps), "texts": len(pairs), "pearson": corr(u, d),
                     "spearman": corr(ranks(u), ranks(d)) if len(u) > 1 else None})
    out = {"sets": rows}
    for kind in ("group", "random"):
        for m in ("pearson", "spearman"):
            v = [r[m] for r in rows if r["kind"] == kind and r[m] is not None]
            out[f"{kind}_{m}_median"] = float(np.median(v)) if v else None
            out[f"{kind}_scored"] = len(v)
    return out


def load(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.name != "summary.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "table", "check", "motifs"))
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--questions", type=Path, default=circuit_questions.OUT)
    ap.add_argument("--device", default=None, help="cuda, mps, cpu, or auto (the weights over every GPU: 32B)")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    a = ap.parse_args()
    if a.cmd == "table":
        print(table({p.split("=", 1)[0]: load(Path(p.split("=", 1)[1])) for p in a.paths}))
        return
    if a.cmd == "motifs":
        run = json.loads(Path(a.paths[0]).read_text())
        recs = {}
        for d in map(Path, a.paths[1:]):
            for f in (sorted(x for x in d.iterdir() if x.suffix == ".json") if d.is_dir() else [d]):
                r = json.loads(f.read_text())
                if "id" in r:
                    recs[r["id"]] = r
        ms = motif_scores(run, recs)
        if a.out:
            a.out.write_text(json.dumps(ms))
        print(f"usage-damage correlation over unseen questions (median per set): groups Pearson {ms['group_pearson_median']} "
              f"Spearman {ms['group_spearman_median']} ({ms['group_scored']} sets); random sets Pearson "
              f"{ms['random_pearson_median']} Spearman {ms['random_spearman_median']} ({ms['random_scored']} sets); "
              f"the essay's ImageNet: edge motifs 0.35, activation factors 0.00")
        return
    from transformers import AutoTokenizer

    if a.cmd == "check":
        base = a.paths[0]
        dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        tok, m = AutoTokenizer.from_pretrained(base), Model(load_model(base, None, dev, getattr(torch, a.dtype)))
        qs = circuit_questions.load(a.questions)
        rows = []
        for qid, q in qs.items():
            ids = [circuit_questions.chat(tok, q["user"]), circuit_questions.chat(tok, q["corrupt_user"])]
            lp = m.logp([(x, []) for x in ids])
            b = int(lp[1].argmax())
            sets = word_sets(m, tok, ids[0], lp[0], b)
            gap = contrast(lp[0], sets) - contrast(lp[1], sets)
            rows.append({"id": qid, "split": q["split"], "answer": sets["answer_word"], "corrupt_answer": tok.decode([b]),
                         "alternatives": sets["alternative_words"], "gap": gap, "valid": pair_ok(sets, b, gap)})
        a.out.write_text(json.dumps({"model": base, "dtype": a.dtype, "pairs": rows}, indent=0))
        for sp in ("dev", "heldout"):
            r = [x for x in rows if x["split"] == sp]
            print(f"{sp}: {sum(x['valid'] for x in r)} of {len(r)} pairs valid on {base}")
        return

    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    files = []
    for p in map(Path, a.paths):
        files += sorted(x for x in p.glob("*.json")) if p.is_dir() else [p]
    circuits = [read_circuit(p) for p in files]
    qs = circuit_questions.load(a.questions) if a.questions.exists() else {}
    a.out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for c in circuits:
        if not (a.out / f"{c['name']}.json").exists():
            groups.setdefault((c["model"], c.get("adapter")), []).append(c)
    for (base, adapter), cs in groups.items():
        tok = AutoTokenizer.from_pretrained(base)
        m = Model(load_model(base, adapter, dev, getattr(torch, a.dtype)))
        for c in cs:
            rec = score(m, tok, c, qs, log=lambda s: print(s, flush=True))
            (a.out / f"{c['name']}.json").write_text(json.dumps(rec))
        del m
    recs = load(a.out)
    stability(recs)
    for r in recs:
        (a.out / f"{r['task']}.json").write_text(json.dumps(r))
    with open(a.out / "rows.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({k: r.get(k) for k in ("task", "id", "builder", "split", "pair", "size", "margins", "seconds")}) + "\n")
    (a.out / "summary.json").write_text(json.dumps(summary(recs)))
    print(table({a.out.name: recs}))


if __name__ == "__main__":
    main()
