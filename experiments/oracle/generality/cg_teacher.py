"""CausalGym's variables in teacher circuits (#2951 graph oracle): the kept tasks' one-token pairs as vpd4l task files,
their teacher circuits (method/oracle/teacher.py --edges --f1-start both, the joint teachers' settings), and per circuit
whether it finds the variable: nodes at the variable's position and at the attractor's, reward v3's circuit terms
(as causalgym.falsify) of the teacher circuit against the same circuit with its nodes at the variable's position moved
to the attractor's (the wrong variable), and the minimal pair through each (the base's circuit run on the source in
C's semantics: the share of the model's move of logit(source label) - logit(base label) reproduced).

  cg_teacher.py write        -> DATA/causalgym_texts/cg_<task>_<k>.json and DATA/causalgym_text_ids.txt
  cg_teacher.py run K P      (on a pod) the texts copied into the texts directory; teacher.py on every P-th text from
                             the K-th -> GEN_OUT/cg_teacher/; then the test -> GEN_OUT/cg_teacher_test_K.jsonl
  cg_teacher.py report
  cg_teacher.py masks K P    varteach's mask core on three tasks' prompts (masks_run) -> GEN_OUT/cg_masks_K.jsonl
With --behavior, every step in the behavior-targeted mode (the task files carry "behavior": the pair's two labels and
its minimal pair, so the teacher's start, acceptance and pick and every score are on that behavior): cgb_* texts in
DATA/causalgym_texts_b/, GEN_OUT/cg_teacher_b/, cg_teacher_b_test_K.jsonl.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ORACLE = Path(os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))
sys.path.insert(0, str(ORACLE))

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))
PER_TASK = 4


def write(behavior: bool = False):
    import causalgym

    tasks = json.loads((DATA / "causalgym_tasks.json").read_text())
    b = "_b" if behavior else ""
    out = DATA / f"causalgym_texts{b}"
    out.mkdir(exist_ok=True)
    ids, meta = [], {}
    for task, info in tasks.items():
        if not info["kept"]:
            continue
        n = 0
        for k, ex in enumerate(info["examples"]):
            diff = [i for i, (x, y) in enumerate(zip(ex["base"], ex["src"])) if x != y]
            if len(diff) != 1 or n >= PER_TASK:
                continue
            name = f"cg{'b' if behavior else ''}_{task.replace('-', '_')}_{k}"
            t = len(ex["base"]) - 1
            pw = ex["ends"][causalgym.wrong_span(info["spans"], ex["diff_spans"][0])]
            tj = {"id": name, "model": "vpd4l", "family": "causalgym", "split": "causalgym", "description": task,
                  "prompts": [{"token_ids": ex["base"], "target_positions": [t]}]}
            if behavior:
                tj["behavior"] = {"tokens": [ex["base_label"], ex["src_label"]], "pairs": [[diff[0], ex["src"][diff[0]]]]}
            (out / f"{name}.json").write_text(json.dumps(tj))
            meta[name] = {"task": task, "pair": k, "p": diff[0], "z": ex["src"][diff[0]], "attractor": pw,
                          "src": ex["src"], "base_label": ex["base_label"], "src_label": ex["src_label"]}
            ids.append(name)
            n += 1
    (DATA / f"causalgym_text_ids{b}.txt").write_text("\n".join(ids) + "\n")
    (DATA / f"causalgym_texts{b}.json").write_text(json.dumps(meta))
    print(f"{len(ids)} texts over {len({m['task'] for m in meta.values()})} tasks")


def run(k: int, P: int, behavior: bool = False):
    import native

    b = "_b" if behavior else ""
    meta = json.loads((DATA / f"causalgym_texts{b}.json").read_text())
    # a prompt where VPD's answer is no nearer the model than the empty circuit (the behavior's KL; measured on the Mac
    # into the meta file) has nothing for a circuit to recover: teacher.py's F1 is undefined there
    names = sorted(nm for nm, m in meta.items() if m.get("kl_empty") is None or m["kl_empty"] > m["kl_vpd"])[k::P]
    dest = native.TEXTS / "vpd4l"
    for nm in names:
        shutil.copy(DATA / f"causalgym_texts{b}" / f"{nm}.json", dest / f"{nm}.json")
    out = Path(os.environ.get("GEN_OUT", DATA))
    lst = out / f"cg_ids{b}_{k}.txt"
    lst.write_text("\n".join(names) + "\n")
    subprocess.run([sys.executable, "-B", str(ORACLE / "teacher.py"), "--tasks", str(lst), "--edges", "--f1-start", "both",
                    "--max-tokens", "0", "--out", str(out / f"cg_teacher{b}")], check=True)
    test(names, out / f"cg_teacher{b}", out / f"cg_teacher{b}_test_{k}.jsonl", b)


MASK_TASKS = ("agr_gender", "agr_sv_num_pp", "agr_sv_num_subj-relc")  # the lead's 10-13: gender and two number agreements


def masks_run(k: int, P: int, behavior: bool = False):
    """varteach's mask core (experiments/oracle/var_masks.py: Cause and Iso donors, the permuted-target control, disjoint
    values) on MASK_TASKS' prompts, the variable's position as the input and the carrier at that position and at the
    target, every P-th job from the K-th -> GEN_OUT/cg_masks_K.jsonl. With behavior every KL is the pair's labels'."""
    import native

    b = "_b" if behavior else ""
    meta = json.loads((DATA / f"causalgym_texts{b}.json").read_text())
    names = sorted(nm for nm, m in meta.items() if m["task"] in MASK_TASKS)
    for nm in names:
        shutil.copy(DATA / f"causalgym_texts{b}" / f"{nm}.json", native.TEXTS / "vpd4l" / f"{nm}.json")
    jobs = [{"task": nm, "inputs": [meta[nm]["p"]], "at": at} for nm in names
            for at in ([meta[nm]["p"]], [json.loads((DATA / f"causalgym_texts{b}" / f"{nm}.json").read_text())["prompts"][0]["target_positions"][-1]])]
    out = Path(os.environ.get("GEN_OUT", DATA))
    jf = out / f"cg_mask_jobs{b}_{k}.jsonl"
    jf.write_text("".join(json.dumps(j) + "\n" for j in jobs[k::P]))
    env = {**os.environ, "PYTHONPATH": f"{ORACLE}:{ORACLE.parent / 'vpd'}"}
    subprocess.run([sys.executable, "-B", str(HERE / "var_masks.py"), "--jobs", str(jf), "--out", str(out / f"cg_masks{b}_{k}.jsonl")],
                   check=True, env=env)


def test_dir(k: int, P: int, behavior: bool = False):
    """The test on teacher circuits made elsewhere (unpacked into ~/mpd-data/graph_oracle/cg_teacher[_b]/), every P-th
    from the K-th -> GEN_OUT/cg_teacher[_b]_test_<10 + K>.jsonl."""
    b = "_b" if behavior else ""
    tdir = Path.home() / "mpd-data/graph_oracle" / f"cg_teacher{b}"
    names = sorted(p.stem for p in tdir.iterdir() if p.suffix == ".json" and "." not in p.stem and p.stem.startswith("cg"))[k::P]
    test(names, tdir, Path(os.environ.get("GEN_OUT", DATA)) / f"cg_teacher{b}_test_{10 + k}.jsonl", b)


def test(names: list, tdir: Path, out: Path, b: str = ""):
    import components
    import goal_eval
    import native
    import torch

    import carry

    meta = json.loads((DATA / f"causalgym_texts{b}.json").read_text())
    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    pool_path = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt"
    pool = torch.load(pool_path, weights_only=False) if pool_path.exists() else None
    for nm in names:
        if not (tdir / f"{nm}.json").exists():
            continue
        m = meta[nm]
        tj = json.loads((DATA / f"causalgym_texts{b}" / f"{nm}.json").read_text())
        ids = tj["prompts"][0]["token_ids"]
        t, p, pw = len(ids) - 1, m["p"], m["attractor"]
        tx = ge.text(ids, [t], native.task_seed(nm), task=tj)  # its behavior, when the file carries one
        g, _ = goal_eval.circuit(json.loads((tdir / f"{nm}.json").read_text()), tx.targets, nat.C, tx.T)
        nodes = sorted(g.node_set())
        held = sorted({q for _, q, _ in nodes})
        moved = sorted({(n, pw if q == p else q, c) for n, q, c in nodes}) if pw not in (None, p, t) else None
        graphs = {"teacher": native.Graph(nodes, complete=True)}
        if moved is not None and p in held:
            graphs["teacher_variable_at_attractor"] = native.Graph(moved, complete=True)
        tau = ge.tau(tx, pool, "eval")
        names_ = list(graphs)
        rows = ge.fidelity(tx, [graphs[v] for v in names_], pool=pool, mode="reward", split="eval")
        cg = [goal_eval._c_graph(graphs[v], tx.targets, nat.C) for v in names_]
        q0, (lc,) = ge._changed_logq(tx, cg, [(p, m["z"])])
        ref_src = nat.reference([m["src"]], tx.targets)
        bl, sl = m["base_label"], m["src_label"]
        ld = lambda lq: (lq[..., 0, sl] - lq[..., 0, bl]).flatten().tolist()  # noqa: E731
        m0, m1 = ld(tx.ref)[0], ld(ref_src)[0]
        c0, c1 = ld(q0), ld(lc)
        pt, fams = tau["phi_tau"], tau["families"]
        P_, S_ = nat.positions(tx.targets), sum(nat.C.values())
        with out.open("a") as f:
            for i, (v, r) in enumerate(zip(names_, rows)):
                ph = goal_eval.phi(r, families=fams)[0]
                phc = goal_eval.cap_phi(ph, pt)
                gate = None if phc is None or pt is None else 1.0 / (1.0 + 2.0 ** (-components.GATE_SLOPE * (phc - pt)))
                nb = goal_eval.size_bits(graphs[v], P_, S_, tx.targets)
                comp = goal_eval.compact(nb, tau["bits_start"]) if nb > 0 else None
                f.write(json.dumps({"name": nm, "task": m["task"], "pair": m["pair"], "variant": v, "positions": held,
                                    "p": p, "attractor": pw, "t": t, "nodes": len(nodes),
                                    "at_variable": sum(q == p for _, q, _ in nodes), "at_attractor": sum(q == pw for _, q, _ in nodes),
                                    "phi": ph, "phi_tau": pt, "reward": None if None in (phc, pt, gate, comp) else phc - pt + gate * 0.5 * comp,
                                    "bits": nb, **{f"s_{fm}": goal_eval.family_score(r, fm) for fm in goal_eval.PHI_FAMILIES},
                                    "moved_c": (c1[i] - c0[i]) / (m1 - m0) if abs(m1 - m0) > 1e-6 else None,
                                    "flip_c": bool((c1[i] > 0) != (c0[i] > 0)), "flip_model": bool((m1 > 0) != (m0 > 0))}) + "\n")
        native.free(nat.dev)
        print(nm, "positions", held, "p", p, "attractor", pw, {v: round(goal_eval.phi(r, families=fams)[0] or 0, 2) for v, r in zip(names_, rows)},
              flush=True)


def report(behavior: bool = False):
    import collections

    import templates

    rows = []
    for p in templates.run_outputs(f"cg_teacher{'_b' if behavior else ''}_test_*.jsonl"):
        rows += [json.loads(x) for x in p.read_text().splitlines()]
    t = [r for r in rows if r["variant"] == "teacher"]
    if not t:
        print("no teacher rows")
        return
    print(f"== {len(t)} CausalGym prompts with teacher circuits over {len({r['task'] for r in t})} tasks")
    print(f"   holds the variable's position {np.mean([r['at_variable'] > 0 for r in t]):.2f}, the attractor's "
          f"{np.mean([r['at_attractor'] > 0 for r in t]):.2f}; nodes {np.mean([r['nodes'] for r in t]):.0f}; "
          f"moved share {np.mean([r['moved_c'] for r in t if r['moved_c'] is not None]):.2f}, flips with the model "
          f"{np.mean([r['flip_c'] == r['flip_model'] for r in t]):.2f}")
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["name"]][r["variant"]] = r
    pairs = [d for d in by.values() if "teacher" in d and "teacher_variable_at_attractor" in d]
    for m in ("reward", "phi", "s_point", "s_track_pp", "s_interchange_pp", "moved_c"):
        vals = [1.0 if d["teacher"][m] > d["teacher_variable_at_attractor"][m] else 0.0 for d in pairs
                if d["teacher"].get(m) is not None and d["teacher_variable_at_attractor"].get(m) is not None]
        print(f"   teacher above its variable moved to the attractor on {m}: {np.mean(vals):.2f} (n={len(vals)})" if vals else f"   {m}: -")
    for task in sorted({r["task"] for r in t}):
        rs = [r for r in t if r["task"] == task]
        print(f"   {task:26s} n {len(rs)} variable held {np.mean([r['at_variable'] > 0 for r in rs]):.2f} attractor held "
              f"{np.mean([r['at_attractor'] > 0 for r in rs]):.2f} moved {np.mean([r['moved_c'] for r in rs if r['moved_c'] is not None]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("write", "run", "report", "masks", "test"))
    ap.add_argument("args", nargs="*", type=int)
    ap.add_argument("--behavior", action="store_true", help="the behavior-targeted mode")
    a = ap.parse_args()
    {"write": lambda: write(a.behavior), "run": lambda: run(*a.args, behavior=a.behavior), "report": lambda: report(a.behavior),
     "masks": lambda: masks_run(*a.args, behavior=a.behavior), "test": lambda: test_dir(*a.args, behavior=a.behavior)}[a.cmd]()
