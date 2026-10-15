"""The v2.3 teacher on a TransformerLens target (#2951): an InterpBench case or MIB's InterpBench IOI model, as
teacher.py's main runs it on vpd4l with --f1-start set --accept-mode pp --accept-draws fv --phi-check --rank once
--max-tokens 8000 --accept changed,floors (semantics n). The phi check and the floors acceptance read the paired
adversary's pattern pool, built here from this target's own texts (goal_eval.build_patterns), not vpd4l's.

Texts: MIB's IOI prompts (--mib SPLIT: interpbench.mib_ioi, BOS first), padded with EOS to --pad as the VPD fit's
batches were (its causal-importance network reads every position) with the target at the last real token; else the
case's random texts after the BOS the model is most confident under (bos_of). The teacher takes texts --offset,
--offset + --stride, ... of the first --n; the pool the --pool-texts after them. The far donors (the START's candidate
ranking, the input-scope family, patching and N; goal_eval's far_pool refuses vpd4l's texts for this model, another
tokenizer's ids): --donors texts tokenized for this model, MIB's --donor-split prompts (train: never the test split the
teacher runs on, nor dev, which the VPD fit read) or the case's random texts from another seed, padded as the texts are.

Per text OUT/<case>_<k>.json (the circuit), .row.json (teach's summary row) and a line of OUT/rows_<offset>.jsonl (its
components, MIB edge names). An unsharded MIB run ends with MIB's area under the ROC (interpbench.mib_auroc against
the case's interpbench_graph.json), each edge scored by the share of texts whose circuit has it.

--builder fast runs the fast circuit builder (builder_fast.build, its VPD units) on the same texts and GoalEval in
place of the teacher; the circuit and rows are written the same way (mib_score.py scores the rows).

  python experiments/oracle/toys/ib_teacher.py --case ~/mpd-data/interpbench/raw/mib --fit FIT.vpd.pt --mib test --pad 28 \\
      --n 6 --dev cuda --out OUT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import goal_eval  # noqa: E402
import interpbench  # noqa: E402
import native  # noqa: E402
import run_planted  # noqa: E402
import teacher  # noqa: E402

EOS = 50256  # GPT-2's end of text: MIB's BOS and the fit's padding


def bos_of(nat, cfg: dict, g: torch.Generator) -> int:
    """The token at position 0 under which the model is most confident at the later positions (training put its BOS
    there): mean largest probability over 64 random continuations."""
    V, T = cfg["d_vocab"], cfg["n_ctx"]
    best = None
    for b in range(V):
        ids = torch.cat([torch.full((64, 1), b), torch.randint(V, (64, T - 1), generator=g)], 1)
        with torch.no_grad():
            p = nat.target(ids.to(nat.dev))[:, 1:].float().softmax(-1).amax(-1).mean().item()
        if best is None or p > best[1]:
            best = (b, p)
    return best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", type=Path, required=True, help="an InterpBench case directory (interpbench.load)")
    ap.add_argument("--fit", type=Path, help="its VPD fit (interpbench.fit_vpd's result; native_of_fit), else the exact pieces")
    ap.add_argument("--mib", help="MIB IOI split for the texts (mib_ioi)")
    ap.add_argument("--mib-dir", type=Path, default=Path.home() / "mpd-data/interpbench/mib_ioi")
    ap.add_argument("--pad", type=int, default=0, help="MIB prompts padded with EOS to this length (the fit's batches: 28)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pool-texts", type=int, default=16)
    ap.add_argument("--donors", type=int, default=400, help="far donors tokenized for this model (goal_eval's search pool size)")
    ap.add_argument("--donor-split", default="train", help="MIB split for the far donors (train: never the teacher's texts, nor dev, the VPD fit's)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="the explainer's tokenizer (the answer budget)")
    ap.add_argument("--edges", action="store_true", help="also prune the circuit's connections (teacher --edges, --no-strand)")
    ap.add_argument("--builder", default="teacher", choices=["teacher", "fast"], help="the slow teacher or builder_fast.build")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sd, cfg, _ = interpbench.load(a.case)
    fit = torch.load(a.fit, map_location="cpu", weights_only=True) if a.fit else None
    nat = interpbench.native_of_fit(sd, cfg, fit, a.dev) if fit else interpbench.native_of(sd, cfg, a.dev)
    if a.mib:
        texts = [r["ids"] for r in interpbench.mib_ioi(a.mib, a.n + a.pool_texts, a.mib_dir)]
    else:
        g = torch.Generator().manual_seed(0)
        texts = next(interpbench.batches(cfg, bos_of(nat, cfg, g))(a.n + a.pool_texts, 0)).tolist()
    lengths = [len(x) for x in texts]
    texts = [x + [EOS] * max(0, a.pad - len(x)) for x in texts]
    pool = goal_eval.build_patterns(texts[a.n:], None, nat, lengths[a.n:])
    ge = run_planted.ToyEval(nat, chunk=a.batch)  # (GoalEval, VPD's bound held above 0 for an exact decomposition)
    if a.mib:
        if a.donor_split == a.mib:
            raise SystemExit("the far donors come from another MIB split than the teacher's texts")
        donors = [r["ids"] for r in interpbench.mib_ioi(a.donor_split, a.donors, a.mib_dir)]
    else:
        donors = next(interpbench.batches(cfg, bos_of(nat, cfg, torch.Generator().manual_seed(1)))(a.donors, 1)).tolist()
    donors = [x + [EOS] * max(0, a.pad - len(x)) for x in donors]
    for split in ("search", "eval"):  # this model's own texts, never vpd4l's (far_pool)
        ge.donor_pools[split] = [(f"{a.case.name}_{a.donor_split if a.mib else 'random'}{i}", x) for i, x in enumerate(donors)]
    ge.FIRST_POSITION = False  # every text starts with the BOS the model always reads: never a START candidate
    goal_eval.FLIP_ROWS, goal_eval.CHANGED_ROWS = teacher.PACK_FLIP_ROWS, teacher.PACK_CHANGED_ROWS
    teacher.RANK = "once"
    tokens = None
    if a.max_tokens:
        from tokenizers import Tokenizer

        tk = Tokenizer.from_pretrained(a.tokenizer)
        tokens = lambda text: len(tk.encode(text, add_special_tokens=False).ids)  # noqa: E731
    a.out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((a.case / "interpbench_graph.json").read_text()) if (a.case / "interpbench_graph.json").exists() else None
    rows = []
    for k in range(a.offset, a.n, a.stride):
        ids, targets, name = texts[k], [lengths[k] - 1], f"{a.case.name}_{k}"
        t0 = time.perf_counter()
        if a.builder == "fast":
            import builder_fast

            c = builder_fast.build(ge, {"id": name, "model": a.case.name, "prompts": [{"token_ids": ids, "target_positions": targets}]},
                                   seed=native.task_seed(name))
            circuits, row = {"": c}, {"build": c["build"], "build_seconds": c["build_seconds"]}
        else:
            circuits, row, _ = teacher.teach(ge, ids, targets, a.batch, f1_start="set", edges=a.edges, no_strand=a.edges, tokens=tokens,
                                             max_tokens=a.max_tokens, accept=("changed", "floors"), accept_mode="pp",
                                             seed=native.task_seed(name), accept_draws="fv", phi_pool=pool)
        for cname, c in circuits.items():
            (a.out / f"{name}{'.' + cname if cname else ''}.json").write_text(json.dumps(c))
        (a.out / f"{name}.row.json").write_text(json.dumps(row, default=str))
        circ = circuits[""]
        nodes = [x for x in circ["nodes"] if x[0] not in ("tok", "block")]
        rec = {"case": a.case.name, "text": k, "ids": ids, "target": targets[0], "nodes": len(nodes),
               "mib_edges": sorted(interpbench.mib_edges(nat, circ, targets)), "seconds": time.perf_counter() - t0}
        rows.append(rec)
        with (a.out / f"rows_{a.offset}.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(name, json.dumps({x: rec[x] for x in ("nodes", "seconds")}), len(rec["mib_edges"]), "MIB edges", flush=True)
    if ref is not None and rows and a.stride == 1 and a.offset == 0:
        share = {}
        for r in rows:
            for e in r["mib_edges"]:
                share[e] = share.get(e, 0.0) + 1.0 / len(rows)
        m = interpbench.mib_auroc(ref, share)
        print("RESULT", json.dumps({"texts": len(rows), "mib_auroc": m["mib"], "mib_auroc_plain": m["plain"]}), flush=True)


if __name__ == "__main__":
    main()
