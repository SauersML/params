"""Parts (method/vpd/parts.py) on the teacher's circuits (method/oracle/teacher.py, #2951): how many circuit nodes fall
in whole parts, how much shorter an answer gets when it names parts, and what turning whole parts on or off does
to the circuit's sufficiency F1 and necessity N.

A circuit node (a subcomponent at a position) is in a whole part when every member of its part (of 2 or more) is
in the circuit at that position. Naming: a whole part at a position is written L<layer>.<kind>#P<j>@<position> (a
part within one matrix; j its number among that matrix's parts) or L<layer>#P<j>@<position> (a part spanning the
layer's matrices); every other node as before (components.py's text). Lengths: the circuit lines of
components.render and their Qwen3 token count (the explainer's tokenizer, Qwen3-8B's).

F1 and N (goal_eval's, live remainders, log scale; teacher.scores): the circuit C as found, and C' = C with every
member of each part that C touches added at that position (the part on whole); N for VPD's answer without C and
without C' (the parts off whole).

  MPD_MEM_GIB=20 ~/mpd-data/venv/bin/python -B experiments/vpd/parts_teacher.py [TEACHER_DIR] [--out OUT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "method" / "oracle"))
from method.vpd.parts import Parts  # noqa: E402

import carry  # noqa: E402
import components  # noqa: E402
import goal_eval  # noqa: E402
import native  # noqa: E402
import teacher  # noqa: E402

KIND = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "fc", "down_proj": "down"}


def numbering(parts: Parts) -> dict:
    """part id -> its name stem: L<layer>.<kind>#P<j> within a matrix, L<layer>#P<j> across a layer's matrices."""
    by = {}
    for k, q in enumerate(parts.parts):
        key = (q["layer"], q["sites"][0] if len(q["sites"]) == 1 else None)
        by.setdefault(key, []).append(k)
    out = {}
    for (layer, site), ks in by.items():
        for j, k in enumerate(sorted(ks, key=lambda k: parts.parts[k]["members"][0])):
            out[k] = f"L{layer}.{KIND[site.rsplit('.', 1)[1]]}#P{j}" if site else f"L{layer}#P{j}"
    return out


def analyse(nodes: list, parts: Parts, names: dict):
    """(whole-part nodes, the part-named text lines, the parts touched with their missing members)."""
    have = {(m, c, t) for m, c, t in nodes if m != "tok"}
    touched = {}
    for m, c, t in have:
        k = parts.of(m, c)
        if k >= 0:
            touched.setdefault((k, t), set()).add((m, c))
    whole, missing = set(), {}
    for (k, t), got in touched.items():
        members = set(parts.members(k))
        if members <= got:
            whole.update((m, c, t) for m, c in members)
        missing[(k, t)] = members - got
    rest = [n for n in nodes if n[0] == "tok" or (n[0], n[1], n[2]) not in whole]
    lines = components.render({"nodes": rest, "edges": [], "explanation": ""}).split("\n")[2:]
    named = {}
    for (k, t), got in touched.items():
        if not missing[(k, t)]:
            named.setdefault(t, []).append(names[k])
    out = []
    for ln in lines:  # "@t ..." block lines: add the part names of position t
        t = int(ln.split()[0][1:]) if ln.startswith("@") else None
        out.append(ln + ("".join(" " + s for s in sorted(named.pop(t, []))) if t is not None else ""))
    out += [f"@{t} " + " ".join(sorted(v)) for t, v in sorted(named.items(), reverse=True)]
    return whole, out, touched, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("teacher", nargs="?", type=Path, default=Path.home() / "mpd-data/graph_oracle/teacher/vpd_f1_heldout")
    ap.add_argument("--parts", type=Path, default=Path.home() / "mpd-data/atlas/vpd4l/parts")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", help="torch device (native.device() otherwise; cpu when the GPU is busy)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)  # the explainer's (components.py)
    parts = Parts(args.parts)
    names = numbering(parts)
    ge = goal_eval.GoalEval(native.Native(args.device))
    nat = ge.nat
    paths = {p.stem: p for p in native.tasks("heldout")}
    rows = []
    for f in sorted(args.teacher.glob("text*.json")):
        circuit = json.loads(f.read_text())
        nodes = [tuple(n) for n in circuit["nodes"]]
        sub = [n for n in nodes if n[0] != "tok"]
        whole, lines, touched, missing = analyse(nodes, parts, names)
        plain = components.render({"nodes": nodes, "edges": [], "explanation": ""}).split("\n")[2:]
        count = lambda ls: len(tok("\n".join(ls))["input_ids"])
        ids, targets = native.text(paths[f.stem])
        tx = ge.text(ids, targets)
        empty, vpd = ge._bounds(tx, teacher.CONVENTION)
        T = tx.T

        def mask(triples):
            m = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
            for n, c, t in triples:
                m[n][t, c] = True
            return m

        added = {(m, c, t) for (k, t), miss in missing.items() for m, c in miss}
        C, Cp = mask(sub), mask(set(sub) | added)
        minus = {n: tx.vpd[n] & ~C[n] for n in nat.names}
        minus_p = {n: tx.vpd[n] & ~Cp[n] for n in nat.names}
        kl = carry.kl(tx.ref, ge.runs(ids, targets, [C, Cp, minus, minus_p], live=teacher.CONVENTION == "live")).tolist()
        f1, n1, _ = teacher.scores(empty, vpd, kl[0], kl[2])
        f1p, n1p, _ = teacher.scores(empty, vpd, kl[1], kl[3])
        row = {"task": f.stem, "nodes": len(sub), "in_whole_parts": len(whole), "parts_touched": len(touched),
               "parts_whole": sum(1 for v in missing.values() if not v), "members_added": len(added),
               "lines": len(plain), "lines_named": len(lines), "tokens": count(plain), "tokens_named": count(lines),
               "kl": kl, "F1": f1, "N": n1, "F1_whole_parts": f1p, "N_whole_parts": n1p, "text_named": lines}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text_named"}), flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
