"""Executable reward for the toy loop (TOYLOOP lane): run the predicted circuit and score it, size-penalized.

  reward_exec(meta, completion) -> float, meta = {"toy": render.Toy, ...} (train.examples adds "toy")
  score(toy, units) -> {"suff", "nec", "n", "reward"}

Texts: the zoo record's examples plus 24 sampled ordinary texts (every lookup-table entry exercised) and counterfactual
texts where the family claims them, else toy.texts. With U the predicted heads and neurons (tok/out ignored):
  suff = (mean + min) over texts / 2 of clip(1 - KL(model || U's active pieces alone) / KL(model || empty circuit), 0, 1)
         (U's nodes: every VPD-active piece of U at every position, the simplest form of the zoo's gt rule)
  nec  = (mean + min) over texts / 2 of clip(KL(model || model with U ablated) / KL(model || model with every live unit ablated), 0, 1)
  reward = (suff + nec) / 2 - UNIT_COST * |U|;  0 for an unparseable answer.
  version 3 (default; REWARD_VERSION, TOYLOOP_REWARD_VERSION=2 for the above): per text suff is also capped by
  1 - |log-odds shift of the answer| / (the empty circuit's shift), and zoo2 necessity also counts _nec_margin.
Sufficiency alone prefers one of several redundant routes (any one carries the answer); necessity credits stating all
of them (ablating one route of two halves the copy, of all of them breaks it). The unit cost removes distractors (they
add neither). The worst-text half makes a unit needed on one text in 30 (a lookup-table entry) worth its cost. The normalisers use no ground truth, so the same reward runs on toys without one.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments/oracle/toys"), str(ROOT / "method/oracle"), str(ROOT / "method/vpd"),
                str(Path(__file__).resolve().parent)]

import circuit  # noqa: E402
import torch  # noqa: E402

UNIT_COST = 0.05
# 3 (default): sufficiency also on the answer's log-odds (score), zoo2 necessity also credits counterweights
# (_nec_margin); fixes the zoo2 gate family (30-instance check: truth first on 26/26 gate texts, was 18/26). 2: the earlier reward.
# Override per process with TOYLOOP_REWARD_VERSION=2, per call with score(..., version=2), or reward_exec_v2.
REWARD_VERSION = int(__import__("os").environ.get("TOYLOOP_REWARD_VERSION", "3"))
_CACHE: dict = {}
_UNIT = re.compile(r"L(\d+)([HN])(\d+)")


N_ORDINARY = 24   # sampled ordinary texts: enough to exercise every entry of a lookup table (bigram_mlp)


def texts_of(toy) -> list:
    """The record's examples plus N_ORDINARY fresh ordinary texts and 4 counterfactual ones where the family claims
    them (zoo.texts); else toy.texts."""
    rec = toy.meta.get("record")
    if not rec:
        return list(toy.texts)
    if toy.meta.get("version") == 2:
        return _texts_v2(toy)
    import zoo
    ts = [e["ids"] for e in rec["examples"]] + [ids for ids, _ in zoo.texts(rec["params"], N_ORDINARY, 1234)]
    if any(e["kind"] == "counterfactual" for e in rec["examples"]):
        ts += [ids for ids, _ in zoo.texts(rec["params"], 4, 4321, "counterfactual")]
    return ts


N_MODULE = 8      # zoo2: same-module texts kept per text (those whose per-text truth equals this text's)


def _texts_v2(toy) -> list:
    """zoo2, per text: the text itself plus up to N_MODULE texts of its module and kind (zoo2.texts) whose per-text
    ground-truth unit set (zoo2.gt_spec) equals this text's: the texts on which the same circuit is the answer."""
    import zoo2
    p, j, ids = toy.meta["record"]["params"], toy.meta["module"], toy.texts[0]
    want = {x["unit"] for x in zoo2.gt_spec(p, j, ids)}
    out = [list(ids)]
    for other, _ in zoo2.texts(p, j, 32, 1000 + toy.meta["example"], toy.meta.get("kind", "ordinary")):
        if len(out) > N_MODULE:
            break
        if other != ids and {x["unit"] for x in zoo2.gt_spec(p, j, other)} == want:
            out.append(other)
    return out


def _model_fns(toy):
    """(last_logits(sd, texts), all_units(sd)) for the toy's zoo version."""
    import zoo
    if toy.meta.get("version") == 2:
        import zoo2
        return zoo2.last_logits, zoo2.all_units2
    return zoo.last_logits, zoo.all_units


_NATIVES: dict = {}


def _native_of(toy):
    """planted.native_of, shared by the toys (texts) of one instance."""
    import planted
    key = toy.meta.get("instance", toy.name)
    if key not in _NATIVES or _NATIVES[key][0] is not toy.sd:
        if len(_NATIVES) > 16:
            _NATIVES.clear()
        _NATIVES[key] = (toy.sd, planted.native_of(toy.sd, toy.cfg))
    return _NATIVES[key][1]


def _kl(p, q) -> list:
    return (p.exp() * (p - q)).sum(-1).tolist()


def _state(toy) -> dict:
    """Per toy, once: the native, its importances per text, the model's log-probs, empty-circuit KLs, all-ablated KLs."""
    if toy.name not in _CACHE:
        import zoo
        nat = _native_of(toy)
        ts = texts_of(toy)
        last_logits, all_units = _model_fns(toy)
        full = last_logits(toy.sd, ts)
        live = all_units(toy.sd)
        if len(_CACHE) > 256:
            _CACHE.clear()
        empty = [_circuit_logp(nat, ids, []) for ids in ts]
        _CACHE[toy.name] = {"nat": nat, "texts": ts, "ci": [nat.importance(ids) for ids in ts], "full": full,
                            "empty": [_kl(f, e)[0] for f, e in empty],
                            "empty_margin": [_margin(f, e) for f, e in empty],
                            "abl_all": _kl(full, last_logits(zoo.ablate(toy.sd, live), ts)), "live": set(live)}
    return _CACHE[toy.name]


@torch.no_grad()
def _circuit_logp(nat, ids: list, nodes: list) -> tuple:
    """zoo.node_kl's two runs, kept: (the native's full last-position log-probs, the run with only `nodes` on), [1, V]."""
    T = len(ids)
    x = torch.tensor([ids])
    full = nat.vpd.target_forward(x)[:, -1].log_softmax(-1)
    masks = {n: torch.zeros(1, T, nat.C[n]) for n in nat.names}
    for n, t, c in nodes:
        masks[n][0, t, c] = 1.0
    return full, nat.vpd.masked(x, masks, {n: torch.ones(1, T) for n in nat.names})[:, -1].log_softmax(-1)


def _margin(full, q) -> float:
    """|m(model) - m(q)|, m = log-odds of the model's top token (logit minus logsumexp of the rest): does not saturate
    once the top token's probability is ~1, unlike the KL."""
    top = int(full[0].argmax())
    m = lambda lp: float(lp[0, top] - torch.cat([lp[0, :top], lp[0, top + 1:]]).logsumexp(0))  # noqa: E731
    return abs(m(full) - m(q))


def unit_keys(units) -> set:
    return {(int(m.group(1)), m.group(2), int(m.group(3))) for u in units if (m := _UNIT.fullmatch(u))}


def nodes_of(toy, units: set, text: int) -> list:
    """[[site, position, piece]] (native.Graph order): active pieces of the units on text `text`, all positions."""
    st = _state(toy)
    nat, ci, want = st["nat"], st["ci"][text], unit_keys(units)
    out = []
    all_neuron = toy.meta.get("version") == 2   # zoo2's gt rule: a gated neuron's pieces at every position (its input
    T = len(st["texts"][text])                  # is far below zero where the gate is off, so VPD calls it inactive)
    for n in nat.names:
        layer, kind = int(n.split(".")[1]), "N" if ".mlp." in n else "H"
        if all_neuron and kind == "N":
            for p in range(int(nat.C[n])):
                if (layer, kind, int(nat.units[n][p])) in want:
                    out += [[n, t, p] for t in range(T)]
            continue
        for t, p in (ci[n] > 0).nonzero().tolist():
            if (layer, kind, int(nat.units[n][p])) in want:
                out.append([n, int(t), int(p)])
    return out


N_MODULE_ANY = 16


def _module_state(toy) -> dict:
    """zoo2: N_MODULE_ANY texts of the toy's module whatever their truth, for necessity. A unit of this text's circuit
    can matter only on texts with another answer (a gate's feature heads on a text the default answers: ablating them
    changes nothing here, but flips the gate's other inputs); necessity is the larger of the same-truth necessity and
    the mean over these texts. Sufficiency stays on the same-truth texts (other answers need other units)."""
    st = _state(toy)
    if "module" not in st:
        import zoo
        import zoo2
        p, j = toy.meta["record"]["params"], toy.meta["module"]
        ts = [ids for ids, _ in zoo2.texts(p, j, N_MODULE_ANY, 777 + j)]
        last_logits, _ = _model_fns(toy)
        full = last_logits(toy.sd, ts)
        st["module"] = {"texts": ts, "full": full,
                        "abl_all": _kl(full, last_logits(zoo.ablate(toy.sd, sorted(st["live"])), ts))}
    return st["module"]


@torch.no_grad()
def _run_units(sd: dict, cfg: dict, ids: list, patch: dict | None = None) -> tuple:
    """vpd_model.Target's forward on one text with per-unit writes: (last-position log-probs [V], {unit: write [T, D]})
    for every head and every live neuron; patch {unit: [T, D]} replaces those units' writes (an interchange)."""
    L, H, D = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
    hd, eps, T = D // H, float(cfg.get("rms_norm_eps", 1e-6)), len(ids)
    pos = torch.arange(T, dtype=torch.float32)
    freq = float(cfg.get("rotary_base", 10000.0)) ** (torch.arange(hd // 2, dtype=torch.float32) / (hd / 2))
    ang = pos[:, None] / freq.repeat(2)[None, :]
    cos, sin = ang.cos(), ang.sin()
    rope = lambda x: x * cos + torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1) * sin  # noqa: E731
    rms = lambda x, w: w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # noqa: E731
    W = lambda i, k: sd[f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}.weight"].float()  # noqa: E731
    E = sd["wte.weight"].float()
    x, writes, patch = E[torch.tensor(ids)], {}, patch or {}
    for i in range(L):
        h = rms(x, sd[f"h.{i}.rms_1.weight"].float())
        q, k, v = ((h @ W(i, kk).T).view(T, H, hd).transpose(0, 1) for kk in ("q_proj", "k_proj", "v_proj"))
        z = rope(q) @ rope(k).transpose(-1, -2) / math.sqrt(hd)
        att = z.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
        y = att @ v                                                     # [H, T, hd]
        Wo = W(i, "o_proj")
        for j in range(H):
            u = f"L{i}.H{j}"
            writes[u] = patch[u] if u in patch else y[j] @ Wo[:, j * hd:(j + 1) * hd].T
            x = x + writes[u]
        h = rms(x, sd[f"h.{i}.rms_2.weight"].float())
        a = h @ W(i, "c_fc").T
        a = 0.5 * a * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * a.pow(3))))
        down = W(i, "down_proj")
        out = a @ down.T
        for n in (down.abs().amax(0) > 0).nonzero().flatten().tolist():
            u = f"L{i}.N{n}"
            w = a[:, n:n + 1] * down[:, n][None, :]
            writes[u] = w
            if u in patch:
                out = out - w + patch[u]
        x = x + out
    logits = rms(x, sd["ln_f.weight"].float()) @ E.T
    return logits[-1].log_softmax(-1), writes


def _swap_state(toy) -> dict:
    """zoo2: interchange pairs among the module texts of equal length whose answers differ (base, donor), with the
    base's log-probs, the donor's unit writes, and the KL when every live non-critical unit takes the donor's writes."""
    st = _state(toy)
    if "swap" not in st:
        ms = _module_state(toy)
        crit = _critical(toy)
        runs = [_run_units(toy.sd, toy.cfg, ids) for ids in ms["texts"]]
        pairs = []
        for a, ia in enumerate(ms["texts"]):
            for b, ib in enumerate(ms["texts"]):
                if a != b and len(ia) == len(ib) and int(runs[a][0].argmax()) != int(runs[b][0].argmax()):
                    pairs.append((a, b))
        pairs = pairs[:N_SWAP]
        movable = [u for u in st["live"] if u not in crit]
        tot = []
        for a, b in pairs:
            lp, _ = _run_units(toy.sd, toy.cfg, ms["texts"][a], {u: runs[b][1][u] for u in movable})
            tot.append(_kl(runs[a][0][None], lp[None])[0])
        st["swap"] = {"pairs": pairs, "runs": runs, "all": tot}
    return st["swap"]


def swap_necessity(toy, live: list) -> float | None:
    """Mean over interchange pairs of KL(base || base with `live`'s writes from the donor) / the same with every
    non-critical unit swapped (pairs where that is ~0 skipped); None without pairs."""
    sw = _swap_state(toy)
    ms = _module_state(toy)
    vals = []
    for (a, b), tot in zip(sw["pairs"], sw["all"]):
        if tot < 1e-3:
            continue
        lp, _ = _run_units(toy.sd, toy.cfg, ms["texts"][a], {u: sw["runs"][b][1][u] for u in live if u in sw["runs"][b][1]})
        vals.append(min(1.0, _kl(sw["runs"][a][0][None], lp[None])[0] / tot))
    return sum(vals) / len(vals) if vals else None


N_SWAP = 12


def _critical(toy) -> set:
    """zoo2: live units whose ablation alone breaks the prediction on this text's same-truth texts (mean KL at least half
    of ablating every unit), e.g. the marker detector, which gates every module unit. Necessity ablates the predicted
    units minus these (when any remain): otherwise any answer naming the detector is fully 'necessary' and redundant
    routes can be dropped for free. No ground truth used. (Requiring 'critical on every module' was tried: then one
    necessary module unit saturates the set's necessity and a gate's second feature head goes unrewarded.)"""
    st = _state(toy)
    if "critical" not in st:
        import zoo
        last_logits = _model_fns(toy)[0]
        base = sum(st["abl_all"]) / len(st["abl_all"])
        st["critical"] = {u for u in st["live"]
                          if sum(_kl(st["full"], last_logits(zoo.ablate(toy.sd, [u]), st["texts"]))) / len(st["texts"])
                          >= 0.5 * base}
    return st["critical"]


def _nec_same(toy, live: list) -> list:
    """Per same-truth text: KL(model || model with `live` ablated) / KL(model || every live unit ablated), clipped."""
    import zoo
    st = _state(toy)
    if not live:
        return [0.0] * len(st["texts"])
    abl = _kl(st["full"], _model_fns(toy)[0](zoo.ablate(toy.sd, live), st["texts"]))
    return [min(1.0, max(0.0, a / max(b, 1e-9))) for a, b in zip(abl, st["abl_all"])]


MARGIN_FULL = 2.0   # nats of the answer's log-odds: moving them this much on a text is full necessity there (v3)


def _log_odds(lp, top) -> torch.Tensor:
    """Per row: log-odds of token top[i] (its log-prob minus the logsumexp of the others' log-probs)."""
    i = torch.arange(len(top))
    rest = lp.clone()
    rest[i, top] = float("-inf")
    return lp[i, top] - rest.logsumexp(-1)


def _nec_margin(toy, live: list) -> float:
    """v3, zoo2: necessity of a counterweight. Per same-truth text min(1, max(0, m(model with `live` ablated) -
    m(model)) / MARGIN_FULL), m the log-odds of the model's answer, (mean + worst) over the texts: the ablation makes
    the answer SURER. The KL terms cannot see that: the answer only gets more certain, KL ~0. An AND gate's
    always-on NO default neuron on a YES text is such a unit (removing it raises YES's log-odds by its whole write,
    ~8 nats); the KL terms gave it only the share of module texts answering NO (~0.5), so the truth lost to truth
    plus a unit of another route, and (with the margin sufficiency in score) to truth minus it. Signed on purpose:
    a drop in the log-odds is the KL terms' job, and counting it here would make each of two redundant routes fully
    necessary (ablating one halves the copy: several nats). Dividing by the shift from ablating every non-critical
    unit was also tried: a cancelling twin ablated alone shifts the log-odds as well, so that denominator is
    mostly artefact."""
    import zoo
    st = _state(toy)
    if not live:
        return 0.0
    if "top" not in st:
        st["top"] = st["full"].argmax(-1)
        st["log_odds"] = _log_odds(st["full"], st["top"])
    lp = _model_fns(toy)[0](zoo.ablate(toy.sd, live), st["texts"])
    r = ((_log_odds(lp, st["top"]) - st["log_odds"]) / MARGIN_FULL).clamp(0, 1).tolist()
    return (sum(r) / len(r) + min(r)) / 2


def _nec_v2(toy, live: list, version: int = 2) -> float:
    """zoo2 necessity of a (non-critical) unit set: the largest of same-truth ablation (mean and worst), module-text
    ablation (mean) and interchange with module texts of another answer (mean); version 3 adds _nec_margin."""
    import zoo
    v = _nec_same(toy, live)
    n = (sum(v) / len(v) + min(v)) / 2
    ms = _module_state(toy)
    mabl = _kl(ms["full"], _model_fns(toy)[0](zoo.ablate(toy.sd, live), ms["texts"]))
    n = max(n, sum(min(1.0, a / max(b, 1e-9)) for a, b in zip(mabl, ms["abl_all"])) / len(mabl))
    if version >= 3:
        n = max(n, _nec_margin(toy, live))
    sn = swap_necessity(toy, live)
    return max(n, sn) if sn is not None else n


def score(toy, units: set, version: int | None = None) -> dict:
    """version 3 (default, REWARD_VERSION): per text, sufficiency is the smaller of the KL term and a logit-margin term
    clip(1 - |m(model) - m(circuit)| / |m(model) - m(empty circuit)|, 0, 1), m the log-odds of the model's top token.
    The KL saturates once the top token is certain: an AND gate's YES text is answered as surely without the NO
    default neuron (KL ~0), so truth minus that neuron scored as high and was cheaper; its log-odds move by the
    neuron's whole write. zoo2 necessity: see _nec_v2. version 2: the earlier reward, unchanged."""
    import zoo
    version = REWARD_VERSION if version is None else version
    st = _state(toy)
    keys = unit_keys(units)
    zoo_names = [f"L{l}.{k}{i}" for l, k, i in sorted(keys)]
    suff = []
    for i, (ids, e) in enumerate(zip(st["texts"], st["empty"])):
        full, q = _circuit_logp(st["nat"], ids, nodes_of(toy, units, i))
        x = min(1.0, max(0.0, 1.0 - _kl(full, q)[0] / max(e, 1e-9)))
        if version >= 3:
            x = min(x, max(0.0, 1.0 - _margin(full, q) / max(st["empty_margin"][i], 1e-9)))
        suff.append(x)
    live = [u for u in zoo_names if u in st["live"]]   # ablating a dead unit changes nothing
    agg = lambda v: (sum(v) / len(v) + min(v)) / 2   # noqa: E731  (mean and worst text: every text must be explained)
    s = agg(suff)
    if toy.meta.get("version") != 2:
        n = agg(_nec_same(toy, live))
    else:   # zoo2: the predicted units minus the critical ones (see _critical) when any remain
        crit = _critical(toy)
        rest = [u for u in live if u not in crit]
        n = _nec_v2(toy, rest, version) if rest else agg(_nec_same(toy, live))
    return {"suff": s, "nec": n, "n": len(keys), "reward": (s + n) / 2 - UNIT_COST * len(keys)}


def reward_exec(meta: dict, completion: str, version: int | None = None) -> float:
    c = circuit.parse(completion)
    if not c.ok:
        return 0.0
    return score(meta["toy"], c.units, version)["reward"]


def reward_exec_v2(meta: dict, completion: str) -> float:
    """The version-2 reward (before the gate fix), for runs that must stay comparable with it."""
    return reward_exec(meta, completion, 2)


def reward_wiring_suff(meta: dict, completion: str) -> float:
    """Opt-in stated-wiring prediction objective: half mean, half worst-text sufficiency.

    Use texts_of's existing input bank and a fresh native for the exact toy weights;
    never reuse the legacy name-keyed cache. Valid scores are in [0, 1], invalid
    circuits score -1. No size cost, English, necessity or intervention certificate.
    """
    return _reward_suff(meta, completion, complete=False)


def reward_unit_suff(meta: dict, completion: str) -> float:
    """Matched wiring control: same pieces/texts/aggregation, every legal selected-unit connection on.

    Still reject invalid stated syntax/architecture. No cost or necessity term; only
    execution of connections differs from reward_wiring_suff (including output edges).
    """
    return _reward_suff(meta, completion, complete=True)


WIRING_MDL_LAMBDA = 0.02


def reward_wiring_mdl(meta: dict, completion: str) -> float:
    """Opt-in point fidelity minus an executed-description proxy, not English simplicity.

    Generic entrypoint: reward_exec:reward_wiring_mdl. Price compiled rank-one
    nodes/connections, normalized by the same toy's full architecture; ignore
    discarded token annotations and prose. No gold edges or extra full forwards.
    """
    parsed = circuit.parse(completion)
    if not parsed.ok or parsed.bad:
        return -1.0
    try:
        scored = score_wiring(meta["toy"], completion, texts=texts_of(meta["toy"]), description=True)
    except ValueError:
        return -1.0
    return wiring_mdl_value(scored)


def wiring_mdl_value(scored: dict) -> float:
    """Fixed objective, with all inputs retained and no teacher-dependent pricing."""
    rows = scored["per_text"]
    if not rows or any(not math.isfinite(r["suff"]) or not 0 <= r["suff"] <= 1
                       or not math.isfinite(r["description_cost"]) or not 0 <= r["description_cost"] <= 1
                       for r in rows):
        raise ValueError("invalid fidelity/description measurements")
    fidelity = (sum(r["suff"] for r in rows) / len(rows) + min(r["suff"] for r in rows)) / 2
    cost = sum(r["description_cost"] for r in rows) / len(rows)
    return fidelity - WIRING_MDL_LAMBDA * cost


def executed_description(graph, nat, positions: int, targets: list[int], full_bits=None) -> dict:
    """Native graph description only; fixed architecture budget needs no execution."""
    import native
    try:
        counts = [int(nat.C[name]) for name in nat.names]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("invalid executed-description architecture dimensions") from exc
    if any(count < 0 or count != nat.C[name] for name, count in zip(nat.names, counts)):
        raise RuntimeError("executed-description component counts must be nonnegative integers")
    components = sum(counts)
    if positions <= 0 or components <= 0:
        raise RuntimeError("executed-description architecture must have positive dimensions")
    bits = graph.bits(positions, components, targets)
    if full_bits is None:
        nodes = {(name, pos, piece) for name in nat.names for pos in range(positions)
                 for piece in range(int(nat.C[name]))}
        full_bits = native.Graph(nodes, complete=True).bits(positions, components, targets)
    if (not math.isfinite(bits) or not math.isfinite(full_bits) or full_bits <= 0
            or bits < 0 or bits > full_bits):
        raise RuntimeError("compiled description is outside its architecture budget")
    return {"description_bits": bits, "full_architecture_bits": full_bits,
            "description_cost": bits / full_bits}


def _reward_suff(meta: dict, completion: str, *, complete: bool) -> float:
    parsed = circuit.parse(completion)
    if not parsed.ok or parsed.bad:
        return -1.0
    try:
        scored = score_wiring(meta["toy"], completion, texts=texts_of(meta["toy"]), complete=complete)
    except ValueError:
        return -1.0  # invalid wiring on any text invalidates the whole answer
    suff = [row["suff"] for row in scored["per_text"]]
    if not suff or any(not math.isfinite(x) or not 0.0 <= x <= 1.0 for x in suff):
        raise ValueError("wiring scorer returned invalid sufficiency measurements")
    return (sum(suff) / len(suff) + min(suff)) / 2


def score_wiring(toy, completion: str, *, nat=None, texts=None, complete=False, description=False) -> dict:
    """Independent point-sufficiency diagnostic for the stated inter-unit wiring.

    Unlike the legacy unit-selection reward, actually execute Mermaid unit edges.
    Token-label arrows remain annotations, embeddings stay on, and remainders stay
    off. No English, token-role, necessity or intervention-fidelity certification.
    A supplied native must represent this toy; useful for reusing an existing load.
    complete=True is the matched selection control: validate the stated graph first,
    then execute every legal connection among exactly the same selected pieces.
    """
    import copy
    import torch
    import goal_eval
    import native
    import planted

    parsed = circuit.parse(completion)
    nat = planted.native_of(toy.sd, toy.cfg) if nat is None else nat
    bound = copy.copy(toy)
    bound.meta = dict(toy.meta)
    # Compile against the exact pieces the executor loaded, never a fresh SVD basis.
    bound.meta["decomposition"] = {n: (nat.target.site(n).U, nat.target.site(n).V, nat.units[n])
                                    for n in nat.names}
    texts = list(toy.texts if texts is None else texts)
    if not texts:
        raise ValueError("wiring evaluation requires at least one input")
    ge, rows = goal_eval.GoalEval(nat), []
    budgets = {}
    with torch.no_grad():
        for ids in texts:
            graph = circuit.unit_graph(parsed, bound, ids)
            if complete:
                graph = native.Graph(graph.nodes, complete=True)
            targets = [len(ids) - 1]
            ref = nat.reference([ids], targets)
            q = ge.graph_ref(ids, targets, graph, live=False)
            empty = ge.graph_ref(ids, targets, native.Graph(), live=False)
            kl = lambda x: max(0.0, float((ref.exp() * (ref - x)).sum()) / math.log(2))
            error, null_error = kl(q), kl(empty)
            rows.append({"kl_bits": error, "empty_kl_bits": null_error,
                         "top": int(q[0, -1].argmax()), "reference_top": int(ref[0, -1].argmax()),
                         "suff": min(1.0, max(0.0, 1.0 - error / max(null_error, 1e-9))),
                         "nodes": graph.count(), "edges": sum(native.implied_edges(graph.nodes, targets)) if complete else graph.edges()})
            if description:
                priced = executed_description(graph, nat, len(ids), targets, budgets.get(len(ids)))
                budgets[len(ids)] = priced["full_architecture_bits"]
                rows[-1].update(priced)
    return {"scope": ("complete same-unit connections" if complete else "executed inter-unit/output wiring") + "; token labels and English unverified",
            "remainders": "off", "embedding": "on", "texts": len(rows), "per_text": rows,
            "suff": sum(r["suff"] for r in rows) / len(rows),
            "kl_bits": sum(r["kl_bits"] for r in rows) / len(rows)}


def intervention_mdl_value(rows):
    """Direct distribution agreement, including the worst fixed intervention."""
    if not rows:
        raise RuntimeError("empty intervention panel")
    fidelities = []
    costs = []
    for row in rows:
        kl, cost = row["kl_bits"], row["description_cost"]
        if not math.isfinite(kl) or kl < 0 or not math.isfinite(cost) or not 0 <= cost <= 1:
            raise RuntimeError("invalid intervention measurement")
        fidelities.append(math.exp(-kl))
        costs.append(cost)
    return (sum(fidelities) / len(fidelities) + min(fidelities)) / 2 - WIRING_MDL_LAMBDA * sum(costs) / len(costs)


class InvalidInterventionCandidate(ValueError):
    """Only candidate grammar/architecture failures, not verifier failures."""


def score_wiring_interventions(toy, completion, *, nat=None, texts=None, panel=None, evidence_sink=None,
                               reference_cache=None):
    """Original-basis head/MLP-output edits; sparse gets no dense trace.

    Every architecture head is edited, including omitted heads. No-edit and
    deletion/half-scale worlds are fixed independently of the candidate.
    """
    import copy
    import goal_eval
    import planted
    from head_counterfactual import scaled_head, kl_bits
    from reward_interventions import scaled_neuron, clear_intervention_caches

    nat = planted.native_of(toy.sd, toy.cfg) if nat is None else nat
    panel = ([("head", layer, head, scale) for layer in range(nat.target.n_layer)
              for head in range(nat.target.n_head) for scale in (0.0, 0.5)]
             + [("neuron", layer, neuron, scale) for layer in range(nat.target.n_layer)
                for neuron in range(nat.target.site(f"h.{layer}.mlp.down_proj").W.shape[1])
                if bool(nat.target.site(f"h.{layer}.mlp.down_proj").W[:, neuron].ne(0).any())
                for scale in (0.0, 0.5)]
             if panel is None else list(panel))
    if not panel:
        raise RuntimeError("intervention panel must be nonempty")
    for entry in panel:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            raise RuntimeError("invalid fixed intervention entry")
        kind, layer, head, scale = entry
        if (kind not in ("head", "neuron") or type(layer) is not int
                or type(head) is not int or not 0 <= layer < nat.target.n_layer
                or scale not in (0.0, 0.5)):
            raise RuntimeError("invalid fixed head/MLP intervention panel")
        limit = (nat.target.n_head if kind == "head" else
                 nat.target.site(f"h.{layer}.mlp.down_proj").W.shape[1]
                 if kind == "neuron" and 0 <= layer < nat.target.n_layer else 0)
        if not 0 <= head < limit:
            raise RuntimeError("invalid fixed head intervention panel")
    panel = [tuple(entry) for entry in panel]
    if len(set(panel)) != len(panel):
        raise RuntimeError("intervention panel must be unique")
    bound = copy.copy(toy)
    bound.meta = dict(toy.meta)
    bound.meta["decomposition"] = {n: (nat.target.site(n).U, nat.target.site(n).V, nat.units[n]) for n in nat.names}
    texts = list(texts_of(toy) if texts is None else texts)
    if not texts:
        raise RuntimeError("empty intervention input panel")
    if reference_cache is not None:
        from intervention_teacher import validate_cache
        validate_cache(reference_cache, toy, nat, texts, panel)
    parsed = circuit.parse(completion)
    if not parsed.ok or parsed.bad:
        raise InvalidInterventionCandidate("invalid circuit grammar")
    try:
        graphs = [circuit.unit_graph(parsed, bound, ids) for ids in texts]
    except ValueError as exc:
        raise InvalidInterventionCandidate(str(exc)) from exc
    prices = [executed_description(g, nat, len(ids), [len(ids) - 1]) for g, ids in zip(graphs, texts)]
    rows = []

    def measure(kind="none", layer=None, head=None, scale=1.0):
        clear_intervention_caches(nat)
        ge = goal_eval.GoalEval(nat)
        for ids, graph, price in zip(texts, graphs, prices):
            targets = [len(ids) - 1]
            p = (nat.reference([ids], targets) if reference_cache is None
                 else reference_cache["references"][len(rows)])
            q = ge.graph_ref(ids, targets, graph, live=False)
            try:
                error = kl_bits(p, q)
            except ValueError as exc:
                raise RuntimeError("invalid intervention distributions") from exc
            rows.append({"kind": kind, "layer": layer, "unit": head, "scale": scale,
                         "ids": list(ids), "kl_bits": error,
                         "reference_top": int(p[0, -1].argmax()),
                         "top": int(q[0, -1].argmax()), **price})
            if evidence_sink is not None:
                evidence_sink(nat, ids, rows[-1], p, q)

    with torch.no_grad():
        measure()
        for kind, layer, unit, scale in panel:
            edit = scaled_head if kind == "head" else scaled_neuron
            with edit(nat, layer, unit, scale):
                measure(kind, layer, unit, scale)
            clear_intervention_caches(nat)
    return {"scope": "finite original-basis head/MLP-output panel only; own routing/norms; no live remainder; no arbitrary/joint/English certificate",
            "panel": panel, "dense_calls": len(rows) if reference_cache is None else 0,
            "sparse_calls": len(rows), "cached_full_logp_only": reference_cache is not None,
            "per_world": rows, "reward": intervention_mdl_value(rows)}


def reward_wiring_intervention_mdl(meta, completion):
    parsed = circuit.parse(completion)
    if not parsed.ok or parsed.bad:
        return -1.0
    try:
        return score_wiring_interventions(meta["toy"], completion, panel=meta.get("intervention_panel"),
                                         nat=meta.get("intervention_native"),
                                         reference_cache=meta.get("intervention_reference_cache"))["reward"]
    except InvalidInterventionCandidate:
        return -1.0


def check(toys, version: int | None = None) -> dict:
    """gt vs gt minus each unit, gt plus the distractors, distractors, empty: per family the worst margin."""
    import render
    fam = {}
    for toy in toys:
        gt = {u for u in toy.truth["units"] if _UNIT.fullmatch(u)}
        dis = {render.zoo_unit(u) for u in toy.meta["record"]["distractor_units"]}
        r = lambda us: score(toy, us, version)["reward"]  # noqa: E731
        g = r(gt)
        minus = max((r(gt - {u}) for u in gt), default=-1)
        m = fam.setdefault(toy.meta["family"], {"n": 0, "gt": [], "gt_minus_one_margin": 9, "gt_plus_dis_margin": 9,
                                                "empty": [], "distractors_margin": 9})
        m["n"] += 1
        m["gt"].append(g)
        m["gt_minus_one_margin"] = min(m["gt_minus_one_margin"], g - minus)
        if dis:
            m["gt_plus_dis_margin"] = min(m["gt_plus_dis_margin"], g - r(gt | dis))
            m["distractors_margin"] = min(m["distractors_margin"], g - r(dis))
        m["empty"].append(r(set()))
    for m in fam.values():
        m["gt"] = round(sum(m["gt"]) / len(m["gt"]), 3)
        m["empty"] = round(max(m["empty"]), 3)
        for k in ("gt_minus_one_margin", "gt_plus_dis_margin", "distractors_margin"):
            m[k] = round(m[k], 3)
    return fam


def check_v2(toys, n_plus: int = 8, version: int | None = None) -> dict:
    """zoo2, per text: the truth's reward against truth minus each unit, truth plus one other unit (distractors and
    other modules' units first, up to n_plus), empty, the instance union, and the units active at the last position.
    Per family: mean truth reward, worst margin over each alternative, share of texts where the truth wins all."""
    import random
    import render
    fam, t0, calls = {}, __import__("time").time(), 0
    for toy in toys:
        rec = toy.meta["record"]
        gt = {u for u in toy.truth["units"] if _UNIT.fullmatch(u)}
        r = lambda us: score(toy, us, version)["reward"]  # noqa: E731
        g = r(gt)
        others = [render.zoo_unit(u) for u in rec["distractor_units"] + rec["gt_units"]]
        others = list(dict.fromkeys(u for u in others if u not in gt))
        rest = [u for u in (render.zoo_unit(x) for x in _model_fns(toy)[1](toy.sd)) if u not in gt and u not in others]
        cand = others[:n_plus] + random.Random(0).sample(rest, min(2, len(rest)))
        run = render.run(toy.sd, toy.cfg, toy.texts[0])
        active = {a.split(":")[0] for a in render.active_units(toy, run, len(toy.texts[0]) - 1)}
        alt = {"minus_one": max((r(gt - {u}) for u in gt), default=-9),
               "plus_one": max((r(gt | {u}) for u in cand), default=-9),
               "empty": r(set()), "union": r({render.zoo_unit(u) for u in rec["gt_units"]}),
               "active": r(active) if active != gt else -9}
        calls += 4 + len(gt) + len(cand)
        m = fam.setdefault(toy.meta["family"], {"texts": 0, "gt": [], "wins": 0, **{k: 9.0 for k in alt}})
        m["texts"] += 1
        m["gt"].append(g)
        m["wins"] += all(g > v for v in alt.values())
        for k, v in alt.items():
            m[k] = min(m[k], g - v)
    for m in fam.values():
        m["gt"] = round(sum(m["gt"]) / len(m["gt"]), 3)
        m["wins"] = f"{m['wins']}/{m['texts']}"
        for k in ("minus_one", "plus_one", "empty", "union", "active"):
            m[k] = round(m[k], 3) if m[k] < 9 else None
    fam["_timing"] = {"score_calls": calls, "seconds": round(__import__("time").time() - t0, 1)}
    return fam


if __name__ == "__main__":
    # --version N (default REWARD_VERSION) picks the reward version for either check
    VERSION = int(sys.argv[sys.argv.index("--version") + 1]) if "--version" in sys.argv else None
    if VERSION is not None:
        i = sys.argv.index("--version")
        del sys.argv[i:i + 2]
    if len(sys.argv) > 1 and sys.argv[1] == "v2":   # python3 -B reward_exec.py v2 [n_instances]: zoo2 margins
        import json
        import render
        import torch
        torch.set_num_threads(1)
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        rows = [json.loads(x) for x in (Path.home() / "mpd-data/toy_zoo2/manifest.jsonl").read_text().splitlines() if x.strip()]
        step = max(1, len(rows) // n)
        toys = [t for r in rows[::step][:n] for t in render.from_zoo2(r, Path.home() / "mpd-data/toy_zoo2")]
        print(json.dumps(check_v2(toys, version=VERSION), indent=1))
        sys.exit()   # python3 -B reward_exec.py [every-k-th instance]: margins of gt over its perturbations
    import json
    import time

    import render
    import torch
    torch.set_num_threads(1)
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    toys = render.load_zoo(Path.home() / "mpd-data/toy_zoo/manifest.jsonl")[::k]
    t0 = time.time()
    res = check(toys, VERSION)
    print(json.dumps(res, indent=1))
    calls = sum(1 + len(t.truth["units"]) + 3 for t in toys)
    print(f"{len(toys)} toys, ~{calls} score calls, {time.time() - t0:.1f}s (incl. per-toy setup)")
