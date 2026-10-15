"""s_weight, the weight-edit fidelity family (#2951 graph oracle; the user's spec: weight edits, removing or scaling
weights, are applied the same way to the circuit and the real model, "the purpose is predicting output under these
interventions"). Per text, draws of weight edits, each applied at every position:
  subcomponent  one of VPD's important subcomponents at the target (native._vpd_order's top) scaled by k everywhere:
                W - (1 - k) u v^T (its u row times k; the remainder unchanged);
  neuron        one MLP hidden unit among the most active at the target, its output scaled by k (down_proj's column,
                entry-wise: that column of W, of V's rows, of the remainder);
  head          one attention head's output scaled by k (o_proj's block of columns, the same way).
Each edit is applied to the model's tensors (W, V, U, the remainder) and every run reads them: the real model's, the
circuit's in C's semantics (its subcomponents with the edited vectors, every other one off, the remainders live and
edited, routing and norms recomputed by that circuit), VPD's whole answer the same. Current GoalEval semantics
"n" and "c" both leave these point runs unfrozen; only the node and omitted families hold boundaries under "n".
Per edit, with p the model's prediction at the targets, p_e
the edited model's, W = KL(p_e || p) (how far the edit moves the model: the no-change answer's error), KL_C = KL(p_e ||
p moved as the circuit's run moves under the edit: log_softmax(log p + log q_C,e - log q_C), s_track's transplant, so
the circuit's error on the unedited text does not count), KL_V = the same for VPD's whole answer:
  s = clip(ln(W / KL_C) / ln(W / KL_V), 0, 1), none where KL_V >= W or W < W_MIN,
pooled weighted by W (s_weight; s_weight_raw unclipped below 0). draws_weight: per edit [kind, what, k, W, KL_C, KL_V].
A conditional change score does not establish small direct KL(p_e || q_C,e): its prediction uses the full-model
baseline p. Report direct edited-distribution fidelity separately when claiming an independently executable circuit.
A circuit predicts the edit only if the edit acts through it: a circuit that reproduces p while missing the parts the
edit changes keeps predicting p, KL_C ~ W, s ~ 0.

  weight_family.py check TASK ANSWER.json [--device cpu]   the family on one text for an answer, VPD's answer and the
                                                            empty one
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import carry  # noqa: E402
import native  # noqa: E402

DRAWS = {"subcomponent": 8, "neuron": 8, "head": 4}
TOP = 64  # the edits are drawn among this many of the most important subcomponents / most active neurons at the target
SCALES = {"subcomponent": (0.0, -1.0), "neuron": (0.0, 2.0), "head": (0.0,)}
W_MIN = 1e-3  # bits: an edit moving the model less predicts nothing either way


def draw(nat, ids: list[int], targets: list[int], seed: int, order: list | None = None) -> list[dict]:
    """The text's weight edits: {"kind", "what", "k"} (module docstring), drawn by seed. order: VPD's order of the text's
    subcomponents (native._vpd_order's, e.g. a scoring pod's saved tx.vpd_order); computed when not given (it needs the
    causal-importance network)."""
    rng = random.Random(seed)
    t = max(targets)
    tg = nat.target
    out = []
    top = [tuple(nd) for nd in (order if order is not None else nat._vpd_order(ids, targets)) if nd[1] == t][:TOP]
    for i, nd in enumerate(rng.sample(top, min(len(top), DRAWS["subcomponent"]))):
        out.append({"kind": "subcomponent", "what": [nd[0], int(nd[2])], "k": SCALES["subcomponent"][i % 2]})
    acts = []
    for layer in range(tg.n_layer):
        st = tg.site(native.site_name(layer, "down_proj"))
        st.cache_input = True
    try:
        with torch.no_grad():
            tg.hidden(torch.tensor([ids], device=nat.dev))
        for layer in range(tg.n_layer):
            h = tg.site(native.site_name(layer, "down_proj")).last_input[0, t].abs()  # the hidden units' outputs at t
            acts += [(float(v), layer, int(j)) for v, j in zip(*h.topk(TOP))]
    finally:
        for layer in range(tg.n_layer):
            st = tg.site(native.site_name(layer, "down_proj"))
            st.cache_input, st.last_input = False, None
    acts.sort(reverse=True)
    for i, (_, layer, j) in enumerate(rng.sample(acts[:TOP], min(TOP, DRAWS["neuron"]))):
        out.append({"kind": "neuron", "what": [layer, j], "k": SCALES["neuron"][i % 2]})
    heads = [(layer, h) for layer in range(tg.n_layer) for h in range(tg.n_head)]
    for layer, h in rng.sample(heads, DRAWS["head"]):
        out.append({"kind": "head", "what": [layer, h], "k": 0.0})
    return out


def _clear(nat, ge=None) -> None:
    """Discard Native reference runs and any cached GoalEval boundary values across a weight change.

    The current weight-family point runs do not use held boundaries; clearing them also prevents stale values if
    another family sharing this GoalEval requests its node/omitted boundary state while the edit is applied.
    """
    nat._refs.clear()
    nat._refs_bytes = 0
    nat._rec = (None, None, {})
    nat.__dict__.pop("_sensitivity_kept", None)
    if ge is not None and hasattr(ge, "_xk"):
        ge._xk.clear()


@contextlib.contextmanager
def applied(nat, e: dict, ge=None):
    """The model's tensors with the weight edit e, restored on exit (ge: its kept x values cleared on both sides)."""
    tg = nat.target
    if e["kind"] == "subcomponent":
        n, c = e["what"]
    else:
        layer, j = e["what"]
        n = native.site_name(layer, "down_proj" if e["kind"] == "neuron" else "o_proj")
    st = tg.site(n)
    saved = {a: getattr(st, a).clone() for a in ("W", "V", "U", "delta_T")}
    k = e["k"]
    with torch.no_grad():
        if e["kind"] == "subcomponent":
            st.W -= (1 - k) * torch.outer(st.U[c], st.V[:, c])
            st.U[c] *= k
        else:
            cols = slice(j, j + 1) if e["kind"] == "neuron" else slice(j * tg.hd, (j + 1) * tg.hd)
            st.W[:, cols] *= k
            st.V[cols, :] *= k
            st.delta_T[cols, :] *= k
    _clear(nat, ge)
    try:
        yield
    finally:
        with torch.no_grad():
            for a, v in saved.items():
                getattr(st, a).copy_(v)
        _clear(nat, ge)


def _lqs(ge, ids, targets, graphs) -> torch.Tensor:
    """log q at the targets [len(graphs), len(targets), V] of each graph's run in C's semantics, remainders live: stated
    graphs on the circuit engine (Engine.plan(semantics="c"), GoalEval._graph_kl's run), complete ones (sets or masks) and the
    empty one by VPD's masked forward (GoalEval.runs)."""
    nat = ge.nat
    T = len(ids)
    out = [None] * len(graphs)
    stated = [i for i, g in enumerate(graphs) if not g.complete and g.count()]
    if stated:
        gs = [graphs[i] for i in stated]
        E = ge._circuit_engine(ids, targets, gs, True)
        for i, lq in zip(stated, E.graph_log_probs(E.plan(gs, semantics="c"), **ge._xkw(ids, targets))):  # ge's semantics
            out[i] = lq
    rest = [i for i in range(len(graphs)) if out[i] is None]
    if rest:
        sets = [ge.masks(graphs[i], T) for i in rest]
        for i, lq in zip(rest, ge.runs(ids, targets, sets, True)):
            out[i] = lq
    native.free(nat.dev)
    return torch.stack(out)


def _moved(p0: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """The model's prediction moved as a run moves under the edit (s_track's transplant: the run's own error left
    out): log_softmax(log p + log q_after - log q_before)."""
    return torch.log_softmax(p0 + after - before, -1)


@torch.no_grad()
def family(ge, tx, graphs: list[native.Graph], seed: int | None = None, edits: list[dict] | None = None,
           order: list | None = None) -> list[dict]:
    """Per graph (C's semantics, remainders live): s_weight, s_weight_raw, kl_weight_c (the mean KL_C) and draws_weight
    (module docstring); one set of edits per text, the same for every graph. A run's prediction under an edit is
    compared as its change (_moved: the model's prediction moved by log q_e - log q), so a circuit's or VPD's own error
    on the unedited text does not count against it. Separate direct diagnostics retain that error: kl_weight_unedited
    is KL(p || q_C), kl_weight_direct is mean KL(p_e || q_C,e), and draws_weight_direct has the same edit identifiers
    and W as draws_weight but direct candidate/VPD KLs. These diagnostics use the same live remainders and do not
    change s_weight or its pooling. With no edits, kl_weight_direct is None rather than a measured zero."""
    nat = ge.nat
    ids, targets = tx.ids, tx.targets
    if edits is None:
        edits = draw(nat, ids, targets, native.task_seed(str(ids[:8])) if seed is None else seed,
                     order if order is not None else getattr(tx, "vpd_order", None))
    p0 = tx.ref[0]
    vpd = [native.Graph(masks=tx.vpd)] if tx.vpd is not None else []
    allg = list(graphs) + vpd
    before = _lqs(ge, ids, targets, allg)
    original_kl = carry.kl(p0[None].expand(len(allg), -1, -1), before).tolist()
    direct_draws = [[] for _ in graphs]
    acc = [[0.0, 0.0, 0.0, 0.0, []] for _ in graphs]  # sum W s, sum W, sum W s_raw, sum KL_C, draws
    for e in edits:
        with applied(nat, e, ge):
            pe = torch.log_softmax(nat.vpd.target_forward(torch.tensor([ids], device=nat.dev))[0, targets].float(), -1)
            after = _lqs(ge, ids, targets, allg)
        W = float(carry.kl(pe[None], p0[None])[0])
        kls = carry.kl(pe[None].expand(len(allg), -1, -1), _moved(p0[None], before, after)).tolist()
        direct_kls = carry.kl(pe[None].expand(len(allg), -1, -1), after).tolist()
        direct_vpd = direct_kls[-1] if vpd else None
        for rows, k in zip(direct_draws, direct_kls):
            rows.append([e["kind"], e["what"], e["k"], W, k, direct_vpd])
        kv = kls[-1] if vpd else None
        for a, k in zip(acc, kls):
            a[3] += k
            a[4].append([e["kind"], e["what"], e["k"], W, k, kv])
            if W < W_MIN or kv is None or kv >= W:
                continue
            raw = math.log(W / max(k, 1e-12)) / math.log(W / max(kv, 1e-12))
            a[0] += W * min(max(raw, 0.0), 1.0)
            a[1] += W
            a[2] += W * min(raw, 1.0)
    return [{"s_weight": a[0] / a[1] if a[1] else None, "s_weight_raw": a[2] / a[1] if a[1] else None,
             "kl_weight_c": a[3] / max(1, len(edits)), "draws_weight": a[4],
             "kl_weight_unedited": original_kl[i],
             "kl_weight_direct": sum(d[4] for d in direct_draws[i]) / len(edits) if edits else None,
             "draws_weight_direct": direct_draws[i], "weight_direct_remainders": "live"}
            for i, a in enumerate(acc)]


def edit_id(edit: dict) -> str:
    """Stable identity of one literal intervention, independent of its outcome."""
    import hashlib
    canonical = dict(edit)
    if 'k' in canonical:
        if type(canonical['k']) not in (int, float) or not math.isfinite(canonical['k']):
            raise ValueError('finite numeric edit scale required')
        canonical['k'] = float(canonical['k'])
        if canonical['k'] == 0:
            canonical['k'] = 0.0
    if 'what' in canonical:
        canonical['what'] = list(canonical['what'])
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, allow_nan=False).encode()).hexdigest()


@torch.no_grad()
def common_eval(ge, tx, execute, edits, *, metadata, construction_edit_ids=()) -> dict:
    """Evaluate a fixed candidate and whole VPD on the same explicit weight panel.

    execute() returns candidate log probabilities at tx.targets under the current
    weights. The caller freezes its mask, means and panel before calling. This
    evaluator performs no ranking, construction, admission or reward assignment.
    Original and failed outcomes remain present; construction edits are excluded
    only from excluded-from-construction summaries. This is not a held-out
    guarantee: metadata must disclose other exposure. VPD is unavailable when tx.vpd is None.
    All KLs normalize complete distributions in FP64 and average over targets.
    """
    import copy
    nat, tg = ge.nat, ge.nat.target
    edits = copy.deepcopy(list(edits))
    metadata = json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))
    if not isinstance(metadata, dict):
        raise ValueError('panel metadata must be a JSON object')
    construction_edit_ids = list(construction_edit_ids)
    import re
    if any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None for value in construction_edit_ids):
        raise ValueError('construction edit IDs must be lowercase SHA256 hex strings')
    excluded = set(construction_edit_ids)
    if len(excluded) != len(construction_edit_ids):
        raise ValueError('duplicate construction edit IDs')
    if not tx.ids or not tx.targets or any(type(t) is not int or not 0 <= t < len(tx.ids) for t in tx.targets):
        raise ValueError('nonempty valid input and targets required')
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'kind', 'what', 'k'}:
            raise ValueError('literal kind/what/k weight edits required')
        kind, what, scale = edit['kind'], edit['what'], edit['k']
        if kind not in ('subcomponent', 'neuron', 'head') or not isinstance(what, (list, tuple)) or len(what) != 2:
            raise ValueError('unsupported weight edit')
        if type(scale) not in (int, float) or not math.isfinite(scale):
            raise ValueError('finite edit scale required')
        if kind == 'subcomponent':
            if what[0] not in nat.names or type(what[1]) is not int or not 0 <= what[1] < nat.C[what[0]]:
                raise ValueError('invalid subcomponent edit')
        elif (type(what[0]) is not int or type(what[1]) is not int or not 0 <= what[0] < tg.n_layer
              or not 0 <= what[1] < (tg.n_head if kind == 'head' else tg.site(native.site_name(what[0], 'down_proj')).W.shape[1])):
            raise ValueError('invalid native unit weight edit')
    identities = [edit_id(e) for e in edits]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate literal weight edits')
    edits = [{'kind': e['kind'], 'what': list(e['what']), 'k': float(e['k']) if e['k'] != 0 else 0.0} for e in edits]
    definition = {'ids': list(tx.ids), 'targets': list(tx.targets), 'edits': edits}

    def distribution(value):
        value = value.detach().cpu().double()  # (MPS has no float64)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (len(tx.targets), tg.n_tokens) or not torch.isfinite(value).all():
            raise ValueError('finite full-vocabulary log probabilities at every target required')
        return value.log_softmax(-1)

    def kl(p, q):
        return max(0., float((p.exp() * (p-q)).sum(-1).mean()) / math.log(2))

    def run():
        # Bypass cross-call reference caches; weights may have changed previously.
        hidden = tg.hidden(torch.tensor([tx.ids], device=nat.dev))[:, tx.targets]
        full = distribution(tg.unembed(hidden).float().log_softmax(-1))
        candidate = distribution(execute())
        vpd = distribution(_lqs(ge, tx.ids, tx.targets, [native.Graph(masks=tx.vpd)])[0]) if tx.vpd is not None else None
        return full, candidate, vpd

    before = run()
    worlds = []
    def record(edit, identity, values):
        p, q, v = values
        row = {'edit': edit, 'edit_id': identity,
               'construction_edit': identity in excluded,
               'full_top': p.argmax(-1).tolist(), 'candidate_top': q.argmax(-1).tolist(),
               'vpd_top': v.argmax(-1).tolist() if v is not None else None,
               'model_change_kl_bits': kl(p, before[0]),
               'candidate_direct_kl_bits': kl(p, q),
               'vpd_direct_kl_bits': kl(p, v) if v is not None else None,
               'candidate_anchored_kl_bits': kl(p, _moved(before[0], before[1], q)),
               'vpd_anchored_kl_bits': kl(p, _moved(before[0], before[2], v)) if v is not None else None}
        worlds.append(row)
    record(None, 'original', before)
    for edit, identity in zip(edits, identities):
        with applied(nat, edit, ge):
            values = run()
        record(edit, identity, values)
    excluded_from_construction = [r for r in worlds[1:] if not r['construction_edit']]
    keys = ('candidate_direct_kl_bits', 'vpd_direct_kl_bits',
            'candidate_anchored_kl_bits', 'vpd_anchored_kl_bits')
    summary = {key: sum(r[key] for r in excluded_from_construction)/len(excluded_from_construction)
               if excluded_from_construction and all(r[key] is not None for r in excluded_from_construction) else None for key in keys}
    return {'panel_id': edit_id(definition), 'definition': definition, 'metadata': metadata,
            'construction_exposure': {'recorded_edit_ids': sorted(excluded),
                                      'intersection_count': len(excluded.intersection(identities)),
                                      'unmatched_recorded_edit_ids': sorted(excluded.difference(identities))},
            'worlds': worlds, 'excluded_from_construction': {'count': len(excluded_from_construction), **summary},
            'vpd_status': 'available' if tx.vpd is not None else 'unavailable',
            'normalization': 'full_vocabulary_fp64; KL bits averaged over targets',
            'semantics': 'literal weight edits; fixed candidate execution; VPD live remainders'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("task")
    c.add_argument("answer")
    c.add_argument("--device", default="cpu")
    args = ap.parse_args()
    import goal_eval

    ge = goal_eval.GoalEval(native.Native(args.device))
    task = json.loads((native.TEXTS / "vpd4l" / f"{args.task}.json").read_text())
    p = task["prompts"][0]
    targets = p["target_positions"]
    ids = p["token_ids"][:max(targets) + 1]
    tx = ge.text(ids, targets, seed=native.task_seed(task["id"]), task=task)
    g, _ = goal_eval.circuit(json.loads(Path(args.answer).read_text()), targets, ge.nat.C, len(ids))
    g = goal_eval.strand(g, targets, ge.nat.C)
    graphs = {"answer": g, "vpd": native.Graph(masks=tx.vpd), "empty": native.Graph()}
    rows = family(ge, tx, list(graphs.values()))
    for (name, _), r in zip(graphs.items(), rows):
        print(f"{name}: s_weight {r['s_weight']}, raw {r['s_weight_raw']}, mean KL_C {r['kl_weight_c']:.4f}, "
              f"unedited KL {r['kl_weight_unedited']:.4f}, mean direct edited KL {r['kl_weight_direct']}")
    for d in rows[0]["draws_weight"]:
        print("  ", d[:3], f"W {d[3]:.4f} KL_C {d[4]:.4f} KL_V {d[5]:.4f}" if d[5] is not None else d)


if __name__ == "__main__":
    main()
