"""Readers' access to oracledata's rows (#2951; ~/mpd-data/oracle/data): the circuit questions under an input arm, in
oq1 form or as cao.Oracle examples (primerl's injection reader), and the Li et al. vectors.

Arms (the same for every reader, so injection and text readers are compared on one footing):
  correct   the text's own circuit with its numbers
  names     the same names, every number dropped (values, shares, writes)
  mismatch  the names with the numbers of another context's circuit of the same split (the donor fixed by the row's id)
  none      no circuit

  load.py cao DIR SPLIT ARM OUT.jsonl     cao.Oracle examples (ctx, blocks, q, a, groups, cat, inject "native")
"""
from __future__ import annotations

import copy
import json
import random
import sys
import zlib
from pathlib import Path

NUMS = ("value", "corrupt", "value_over_max", "lost", "restored", "through_source", "effect")
MID = 18  # the Activation Oracle's read layer on Qwen3-8B (cao.MID)


def rows(d: Path, split: str) -> list[dict]:
    return [json.loads(x) for x in (Path(d) / f"{split}.jsonl").read_text().splitlines()]


def circuits(d: Path) -> dict:
    return {c["circuit_id"]: c for c in map(json.loads, (Path(d) / "circuits.jsonl").read_text().splitlines())}


def donor(row: dict, pool: list[str]) -> str:
    """Another context's circuit for a row's mismatch arm, fixed by the row's id."""
    others = [c for c in pool if c != row["circuit_id"]]
    return random.Random(zlib.crc32(row["id"].encode())).choice(others)


def arm(row: dict, circs: dict, how: str, pool: list[str] | None = None) -> dict | None:
    """The circuit a row's reader sees under an arm (None for none)."""
    if how == "none":
        return None
    c = copy.deepcopy(circs[row["circuit_id"]])
    if how == "correct":
        return c
    if how == "names":
        for part in ("units", "heads", "groups"):
            for u in c[part]:
                for k in NUMS:
                    if k in u:
                        u[k] = None
                if part == "heads":
                    u["attends"] = None
        return c
    if how == "mismatch":
        d = circs[donor(row, pool or sorted(circs))]
        for part in ("units", "heads"):  # the donor's numbers, in order, cycled over this circuit's names
            src = d[part] or [{}]
            for k, u in enumerate(c[part]):
                s = src[k % len(src)]
                for f in NUMS + (("attends",) if part == "heads" else ()):
                    if f in u:
                        u[f] = s.get(f)
        return c
    raise ValueError(how)


def cao_example(row: dict, c: dict | None) -> dict:
    """cao.Oracle's example: the text's half-depth read, then each item's write (a group's its members', a head's output
    by its unit), native injection (primerl's rc2 form)."""
    blocks = [{"layer": MID, "positions": list(range(len(row["ctx"])))}]
    if c is not None:
        val = {u["name"]: u for u in c["units"]}
        groups = {g["id"]: g for g in c["groups"]}
        for it in c["items"]:
            if it in groups:
                ws = [[val[m]["layer"], val[m]["neuron"], val[m]["value"] or 0.0] for m in groups[it]["members"] if m in val]
                blocks.append({"label": f"{it}, {len(ws)} neuron{'' if len(ws) == 1 else 's'}", "write": ws})
            elif ".H" in it:
                h = next(x for x in c["heads"] if x["name"] == it)
                looks = ", ".join(f"'{t}' @{q}" for q, _, t in (h.get("attends") or [])[:2])
                blocks.append({"label": it + (f" (looks at {looks})" if looks else ""), "unit": ["head", h["layer"], h["head"]]})
            else:
                u = val[it]
                blocks.append({"label": it, "write": [u["layer"], u["neuron"], u["value"] or 0.0]})
    return {"cat": row["kind"], "form": "groups", "ctx": row["ctx"], "blocks": blocks, "q": row["q"], "a": row["a"],
            "claim": row["claim"], "groups": row["groups"], "inject": "native", "id": row["id"], "cls": row["cls"],
            "options": row["options"], "a_key": row["a_key"]}


def li_vector(d: Path, row: dict):
    """A Li et al. row's patched vector (float32 [4096]) from its parquet file."""
    import pyarrow.parquet as pq
    t = pq.read_table(Path(d) / "raw" / row["vector_ref"]["file"], columns=["patch_position"])
    return t.column("patch_position")[row["vector_ref"]["row"]]["intervention_vector"].as_py()


def main():
    cmd, d, split, how, out = sys.argv[1:6]
    assert cmd == "cao"
    rs, cs = rows(Path(d), split), circuits(Path(d))
    pool = sorted({r["circuit_id"] for r in rs})
    with open(out, "w") as fh:
        for r in rs:
            fh.write(json.dumps(cao_example(r, arm(r, cs, how, pool))) + "\n")


if __name__ == "__main__":
    main()
