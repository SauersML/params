"""CausalGym tasks on vpd4l (#2951 graph oracle; Arora, Jurafsky, Potts 2024, HF dataset aryaman/causalgym): templated
minimal pairs aligned span by span, each changing one known variable at a known span and flipping the next-token
preference ("The man near him" -> " is", "The men near him" -> " are").

  causalgym.py filter   -> DATA/causalgym_tasks.json: per task of the test split, vpd4l's accuracy (on each prompt of a
                           pair, the base's label above the source's on the base and the reverse on the source), the
                           spans where base and source differ (the variable's position), and the kept tasks (>= KEEP)
  causalgym.py vartest  -> DATA/causalgym_vartest.jsonl: per kept task and pair, the stated variable's position test in
                           the real model: every weight matrix's output at one position (attention and MLP outputs, and
                           the key and value read there by later positions) taken from the source prompt into the
                           base's run; the share of the base-to-source move of logit(source label) - logit(base label)
                           at the last position it reproduces, and whether the preference flips. Positions: the
                           variable's (the last token of the span that differs: the subject for agreement) and the wrong
                           variable's (the attractor: the noun after the last " the" span; else the span after the
                           variable's)

The dataset is read from ~/mpd-data/causalgym/raw/test.json (downloaded into its own directory; data only). Each span
is tokenized with vpd4l's tokenizer and the spans joined; the labels must be single tokens.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
RAW = Path.home() / "mpd-data/causalgym/raw/test.json"
KEEP = 0.8


def tokenize(tk, spans):
    """(token ids, per span the index of its last token); the "<|endoftext|>" span as the tokenizer's own id."""
    ids, ends = [], []
    for s in spans:
        if s == "<|endoftext|>":
            tid = tk.token_to_id("<|endoftext|>")
            ids += [tid] if tid is not None else []
        else:
            ids += tk.encode(s, add_special_tokens=False).ids
        ends.append(len(ids) - 1)
    return ids, ends


def filter_tasks():
    import mech
    import native

    tk = mech.tokenizer("vpd4l")
    rows = json.loads(RAW.read_text())
    nat = native.Native()
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    out = {}
    for task, rs in sorted(by.items()):
        acc, diffs, n_bad, pairs = [], collections.Counter(), 0, []
        for r in rs:
            bl, sl = tk.encode(r["base_label"], add_special_tokens=False).ids, tk.encode(r["src_label"], add_special_tokens=False).ids
            if len(bl) != 1 or len(sl) != 1 or bl == sl:
                n_bad += 1
                continue
            bids, bends = tokenize(tk, r["base"])
            sids, sends = tokenize(tk, r["src"])
            if len(bids) != len(sids):  # same-length pairs only (the interchange tests patch position by position)
                n_bad += 1
                continue
            diff = [i for i, (x, y) in enumerate(zip(r["base"], r["src"])) if x != y]
            diffs.update(diff)
            lb = nat.reference([bids], [len(bids) - 1])[0, 0]
            ls = nat.reference([sids], [len(sids) - 1])[0, 0]
            acc += [float(lb[bl[0]] > lb[sl[0]]), float(ls[sl[0]] > ls[bl[0]])]
            pairs.append({"base": bids, "src": sids, "base_label": bl[0], "src_label": sl[0], "ends": bends,
                          "diff_spans": diff, "base_type": r["base_type"], "src_type": r["src_type"]})
        out[task] = {"pairs": len(pairs), "skipped": n_bad, "accuracy": float(np.mean(acc)) if acc else None,
                     "diff_spans": dict(diffs), "spans": rs[0]["base"], "kept": bool(acc) and float(np.mean(acc)) >= KEEP,
                     "examples": pairs[:64]}
        print(f"{task:28s} pairs {len(pairs):4d} skipped {n_bad:4d} accuracy {out[task]['accuracy'] if acc else float('nan'):.2f} "
              f"differing spans {dict(diffs)} spans {rs[0]['base']}", flush=True)
    (DATA / "causalgym_tasks.json").write_text(json.dumps(out))
    print("kept:", [t for t, v in out.items() if v["kept"]])


def run_patched(tg, ids, ids2, p):
    """vpd4l's log probabilities at the last position of ids with every site's output at position p (attention output,
    MLP output, and the key and value p offers later positions) from ids2 (the
    embedding at p stays ids')."""
    import math

    import torch

    import vpd_model

    def forward(x_ids, rec=None, put=None):
        T = len(x_ids)
        x = tg.wte[torch.tensor([x_ids], device=tg.wte.device)].float()
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        for i in range(tg.n_layer):
            s = lambda k: tg.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")  # noqa: E731
            h = vpd_model.rms(x, tg.norms[2 * i], tg.eps)
            q = s("q_proj")(h).view(1, T, tg.n_head, tg.hd).transpose(1, 2)
            k = s("k_proj")(h).view(1, T, tg.n_head, tg.hd).transpose(1, 2)
            v = s("v_proj")(h).view(1, T, tg.n_head, tg.hd).transpose(1, 2)
            q, k = tg._rope(q, T), tg._rope(k, T)
            if rec is not None:
                rec[(i, "k")], rec[(i, "v")] = k[:, :, p].clone(), v[:, :, p].clone()
            if put is not None:
                k, v = k.clone(), v.clone()
                k[:, :, p], v[:, :, p] = put[(i, "k")], put[(i, "v")]
            P = ((q @ k.transpose(-1, -2)) / math.sqrt(tg.hd)).masked_fill(~causal, float("-inf")).softmax(-1)
            att = s("o_proj")((P @ v).transpose(1, 2).reshape(1, T, -1))
            if rec is not None:
                rec[(i, "o")] = att[:, p].clone()
            if put is not None:
                att = att.clone()
                att[:, p] = put[(i, "o")]
            x = x + att
            m = s("down_proj")(vpd_model.gelu_tanh(s("c_fc")(vpd_model.rms(x, tg.norms[2 * i + 1], tg.eps))))
            if rec is not None:
                rec[(i, "m")] = m[:, p].clone()
            if put is not None:
                m = m.clone()
                m[:, p] = put[(i, "m")]
            x = x + m
        return torch.log_softmax(vpd_model.rms(x[:, -1], tg.ln_f, tg.eps) @ tg.wte.T.float(), -1)[0]

    rec = {}
    with torch.no_grad():
        forward(ids2, rec=rec)
        return forward(ids, put=rec)


def wrong_span(spans, diff):
    """The wrong variable's span: the noun after the last " the" span (the attractor), else the span after the variable's."""
    the = [i for i, s in enumerate(spans) if s.strip().lower() == "the" or s.endswith(" the")]
    j = (the[-1] + 1) if the and the[-1] + 1 < len(spans) and the[-1] + 1 != diff else diff + 1
    return min(j, len(spans) - 1)


def vartest(pairs_per_task: int = 32):
    import native

    tasks = json.loads((DATA / "causalgym_tasks.json").read_text())
    tg = native.Native().target
    out = DATA / "causalgym_vartest.jsonl"
    for task, info in tasks.items():
        if not info["kept"]:
            continue
        for k, ex in enumerate(info["examples"][:pairs_per_task]):
            diff = ex["diff_spans"][0]
            spans = info["spans"]
            pv, pw = ex["ends"][diff], ex["ends"][wrong_span(spans, diff)]
            if pw == pv or pw >= len(ex["base"]) - 1:
                pw = ex["ends"][min(diff + 1, len(ex["ends"]) - 1)]
            bl, sl = ex["base_label"], ex["src_label"]
            ld = lambda lp: float(lp[sl] - lp[bl])  # noqa: E731
            lb = ld(run_patched(tg, ex["base"], ex["base"], pv))
            ls = ld(run_patched(tg, ex["src"], ex["src"], pv))
            row = {"task": task, "pair": k, "ld_base": lb, "ld_src": ls, "variable_position": pv, "wrong_position": pw}
            for lab, pos in (("variable", pv), ("wrong", pw)):
                lp = ld(run_patched(tg, ex["base"], ex["src"], pos))
                row[f"moved_{lab}"] = (lp - lb) / (ls - lb) if abs(ls - lb) > 1e-6 else None
                row[f"flip_{lab}"] = bool((lp > 0) != (lb > 0))
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
        rs = [json.loads(x) for x in out.read_text().splitlines() if json.loads(x)["task"] == task]
        print(task, {lab: (round(float(np.mean([r[f"moved_{lab}"] for r in rs if r[f"moved_{lab}"] is not None])), 2),
                           round(float(np.mean([r[f"flip_{lab}"] for r in rs])), 2)) for lab in ("variable", "wrong")}, flush=True)


def falsify(per_task: int = 8, only_task: str | None = None, blocks: bool = False, behavior: bool = False):
    """The scorecard on CausalGym's variables: per kept task's pairs that differ at one token (the variable's
    position p, the source's token z there), circuits on the base prompt (VPD's actives at the positions named, every
    connection among them): target (the last position alone), variable (and p), attractor (and the wrong variable's
    position), mismatched target (the variable circuit's positions with the source prompt's actives: a circuit built
    for the other label), the variable with both its values' parts (VPD's actives at p on the base and on the source:
    VPD writes a token's identity in which subcomponents are active, so one value's parts cannot carry the change), and
    the influential start (start_set). Per circuit reward v3's terms (phi over the text's
    families, ge.tau on the eval split, cap_phi, the gate, size_bits, compactness against the start, the circuit reward
    phi - phi_tau + g x 0.5 compact) and the minimal pair through the circuit: the circuit run on the source prompt in
    C's semantics (the base's subcomponents, the source's tokens), the share of the model's base-to-source move of
    logit(source label) - logit(base label) it reproduces (moved_c), whether its preference flips, and s_track on that
    one change (the circuit's move applied to the model on the base, against VPD's answer's, bounded by no change).
    With blocks: also the variable and the attractor as carriers, every site's whole output at the position as a block
    (activation patching there: a block carries whichever subcomponents the token switches on), every circuit scored
    with falsify.patch_ge's scorecard (no connection-cut family: a block circuit's connections are beyond memory)
    -> causalgym_falsify_blocks.jsonl. With behavior: the behavior-targeted mode (goal_eval.Text's task "behavior": every
    distribution the one over the pair's two labels, the minimal pair first among every family's changed texts and
    among F2's positions) -> causalgym_falsify[_blocks]_behavior.jsonl.
    -> DATA/causalgym_falsify.jsonl, one line per pair and circuit (resumable)."""
    import components
    import goal_eval
    import native
    import torch

    import carry

    tasks = json.loads((DATA / "causalgym_tasks.json").read_text())
    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    if blocks:
        import falsify as fz

        fz.patch_ge(ge)
        goal_eval.FLIP_ROWS = 16
    pool_path = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt"
    pool = torch.load(pool_path, weights_only=False) if pool_path.exists() else None
    stem = f"causalgym_falsify{'_blocks' if blocks else ''}{'_behavior' if behavior else ''}"
    out = Path(os.environ.get("GEN_OUT", DATA)) / f"{stem}{os.environ.get('GEN_TAG', '')}.jsonl"
    done = {(r["task"], r["pair"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    kept = [t for t, v in tasks.items() if v["kept"]]
    dist = {}  # per task its variable's distance from the last position (the mismatched offsets)
    for task in kept:
        ex = tasks[task]["examples"][0]
        dist[task] = len(ex["base"]) - 1 - ex["ends"][ex["diff_spans"][0]]
    for task in kept:
        if only_task and task != only_task:
            continue
        info = tasks[task]
        for k, ex in enumerate(info["examples"][:per_task]):
            diff = [i for i, (x, y) in enumerate(zip(ex["base"], ex["src"])) if x != y]
            if len(diff) != 1 or (task, k) in done:
                continue
            p, z = diff[0], ex["src"][diff[0]]
            ids, t = ex["base"], len(ex["base"]) - 1
            pw = ex["ends"][wrong_span(info["spans"], ex["diff_spans"][0])]
            if pw in (p, t):
                pw = next((q for q in range(t - 1, 0, -1) if q != p), None)
            beh = lambda pair: {"behavior": {"tokens": [ex["base_label"], ex["src_label"]], "pairs": [pair]}} if behavior else None  # noqa: E731
            ts = ge.text(ex["src"], [t], native.task_seed(f"causalgym:{task}:{k}:src"), task=beh([p, ex["base"][p]]))
            tx = ge.text(ids, [t], native.task_seed(f"causalgym:{task}:{k}"), task=beh([p, z]))  # last: the mode is the base's
            act = lambda x, ps: [(n, q, int(c)) for n in nat.names for q in ps for c in x.vpd[n][q].nonzero().flatten().tolist()]  # noqa: E731
            variants = {"target": act(tx, [t]), "variable": act(tx, [t, p]), "mismatched_target": act(ts, [t, p]),
                        "variable_both_values": act(tx, [t, p]) + act(ts, [p])}
            if pw is not None:
                variants["attractor"] = act(tx, [t, pw])
            if blocks:
                variants["variable_blocks"] = act(tx, [t]) + [(n, p, goal_eval.BLOCK) for n in nat.names]
                if pw is not None:
                    variants["attractor_blocks"] = act(tx, [t]) + [(n, pw, goal_eval.BLOCK) for n in nat.names]
            tau = ge.tau(tx, pool, "eval")
            variants["start"] = sorted(ge.start_set(tx, "eval")["graph"].node_set())
            names = list(variants)
            graphs = [native.Graph(sorted(set(variants[v])), complete=True) for v in names]
            P, S = nat.positions(tx.targets), sum(nat.C.values())
            rows = ge.fidelity(tx, graphs, pool=pool, mode="reward", split="eval")
            # the minimal pair through each circuit
            cg = [goal_eval._c_graph(g, tx.targets, nat.C) for g in graphs]
            q0, (lc,) = ge._changed_logq(tx, cg, [(p, z)])
            v0, (lv,) = ge._changed_vpd(tx, [(p, z)])
            bl, sl = ex["base_label"], ex["src_label"]
            ref_src = nat.reference([ex["src"]], tx.targets)
            ld = lambda lq: (lq[..., 0, sl] - lq[..., 0, bl]).flatten().tolist()  # noqa: E731
            m0, m1 = ld(tx.ref)[0], ld(ref_src)[0]
            c0, c1 = ld(q0), ld(lc)
            w = float(carry.kl(ref_src, tx.ref)[0])
            tv = float(carry.kl(ref_src, goal_eval._transplant(tx.ref, lv, v0))[0])
            tc = carry.kl(ref_src, goal_eval._transplant(tx.ref, lc, q0)).tolist()
            pt, fams = tau["phi_tau"], tau["families"]
            with out.open("a") as f:
                for i, (v, g, r) in enumerate(zip(names, graphs, rows)):
                    ph = goal_eval.phi(r, families=fams)[0]
                    phc = goal_eval.cap_phi(ph, pt)
                    gate = None if phc is None or pt is None else 1.0 / (1.0 + 2.0 ** (-components.GATE_SLOPE * (phc - pt)))
                    b = goal_eval.size_bits(g, P, S, tx.targets, active=tx.vpd)
                    comp = goal_eval.compact(b, tau["bits_start"]) if b > 0 else None
                    f.write(json.dumps({
                        "task": task, "pair": k, "variant": v, "p": p, "attractor": pw, "t": t, "nodes": g.count(),
                        "phi": ph, "phi_tau": pt, "phi_start": tau["phi_start"], "phi_empty": tau["phi_empty"], "g": gate,
                        "reward": None if None in (phc, pt, gate, comp) else phc - pt + gate * 0.5 * comp, "bits": b,
                        "compact": comp, **{f"s_{fm}": goal_eval.family_score(r, fm) for fm in goal_eval.PHI_FAMILIES},
                        "ld_model": [m0, m1], "ld_circuit": [c0[i], c1[i]],
                        "moved_c": (c1[i] - c0[i]) / (m1 - m0) if abs(m1 - m0) > 1e-6 else None,
                        "flip_c": bool((c1[i] > 0) != (c0[i] > 0)), "flip_model": bool((m1 > 0) != (m0 > 0)),
                        "s_track_pair": goal_eval._fid(w, tc[i], tv)}) + "\n")
            native.free(nat.dev)
            print(task, k, {v: round(goal_eval.phi(r, families=fams)[0] or 0, 2) for v, r in zip(names, rows)},
                  "moved_c", {v: round((c1[i] - c0[i]) / (m1 - m0), 2) for i, v in enumerate(names)} if abs(m1 - m0) > 1e-6 else None,
                  flush=True)


ATTRACTOR_IS_TARGET = {"agr_sv_num_pp", "agr_sv_num_subj-relc"}  # the attractor noun is the prompt's last token: the
# attractor variant there falls back to t - 1 (" the")


def report(blocks: bool = False, behavior: bool = False, run: str | None = None):
    """The falsification table (every run's rows, or run's alone: a pod run's name), over every pair and over the pairs
    whose attractor is a context position (ATTRACTOR_IS_TARGET's tasks left out)."""
    rows = []
    import templates

    stem = f"causalgym_falsify{'_blocks' if blocks else ''}{'_behavior' if behavior else ''}"
    paths = ([DATA / f"{stem}.jsonl"] + templates.run_outputs(f"{stem}*.jsonl") if run is None else
             sorted((Path("/Users/user/mpd-data/runpod") / run / "out").glob(f"{stem}*.jsonl")))
    for p in paths:
        if p.exists() and ("_blocks" in p.name) == blocks and ("_behavior" in p.name) == behavior:
            rows += [json.loads(x) for x in p.read_text().splitlines()]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["task"], r["pair"])].setdefault(r["variant"], r)
    terms = ["reward", "phi", "s_point", "s_nodes", "s_flips", "s_track_pp", "s_interchange_pp", "s_patching_near", "s_N",
             "s_track_pair", "moved_c"]
    for lab, keep in (("every pair", lambda k: True), ("attractor a context position", lambda k: k[0] not in ATTRACTOR_IS_TARGET)):
        sel = {k: d for k, d in by.items() if keep(k)}
        print(f"== {lab}: {len(sel)} pairs over {len({k[0] for k in sel})} tasks; share of pairs where the first beats the second (ties half)")
        print("   " + " ".join(f"{m[:12]:>12s}" for m in terms))
        for a_, b_ in (("variable", "attractor"), ("variable", "mismatched_target"), ("variable", "target"), ("start", "variable"),
                       ("variable_both_values", "variable"), ("variable_both_values", "attractor"),
                       ("variable_blocks", "attractor_blocks"), ("variable_blocks", "variable")):
            line = []
            for m in terms:
                vals = [1.0 if d[a_][m] > d[b_][m] + 1e-9 else 0.5 if abs(d[a_][m] - d[b_][m]) <= 1e-9 else 0.0
                        for d in sel.values() if a_ in d and b_ in d and d[a_].get(m) is not None and d[b_].get(m) is not None]
                line.append(f"{np.mean(vals):12.2f}" if vals else f"{'-':>12s}")
            if any(x.strip() != "-" for x in line):
                print(f"   {' '.join(line)}   {a_} > {b_}")
    for v in ("target", "variable", "variable_both_values", "attractor", "mismatched_target", "start", "variable_blocks",
              "attractor_blocks"):
        if not any(v in d for d in by.values()):
            continue
        rs = [d[v] for d in by.values() if v in d]
        mean = lambda m: np.mean([r[m] for r in rs if r.get(m) is not None])  # noqa: E731
        print(f"   {v:18s} n {len(rs):3d} reward {mean('reward'):6.2f} phi {mean('phi'):5.2f} bits {mean('bits'):8.0f} "
              f"moved_c {mean('moved_c'):5.2f} flips with the model {np.mean([r['flip_c'] == r['flip_model'] for r in rs]):.2f} "
              f"s_track_pair {mean('s_track_pair'):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("filter", "vartest", "falsify", "report"))
    ap.add_argument("--per-task", type=int, default=8)
    ap.add_argument("--task", default=None, help="falsify: one task only")
    ap.add_argument("--blocks", action="store_true", help="falsify, report: the carrier-block variants (patched scorecard)")
    ap.add_argument("--behavior", action="store_true", help="falsify, report: the behavior-targeted mode")
    ap.add_argument("--run", default=None, help="report: one pod run's rows alone (its name)")
    a = ap.parse_args()
    {"filter": filter_tasks, "vartest": vartest, "falsify": lambda: falsify(a.per_task, a.task, a.blocks, a.behavior), "report": lambda: report(a.blocks, a.behavior, a.run)}[a.cmd]()
