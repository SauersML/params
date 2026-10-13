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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("filter", "vartest"))
    a = ap.parse_args()
    filter_tasks() if a.cmd == "filter" else vartest()
