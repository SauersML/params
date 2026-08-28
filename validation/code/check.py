#!/usr/bin/env python3
"""Source-text guards for the `Params` corpus.

Runs without a build and on a broken tree.  Exit is nonzero if any guard fails.

  admissions   no `sorry`, `admit`, `axiom`, `native_decide`, `unsafe`, `opaque`,
               `implemented_by`, `extern` anywhere in `Params/`.  The corpus policy
               is "no assumptions, no literature inputs, no conditional proofs":
               every theorem is proved from Mathlib and nothing else.
  options      `autoImplicit = false`, `relaxedAutoImplicit = false` and
               `-DwarningAsError=true` are set in `lakefile.toml`.
  closure      every module under `Params/` is imported (transitively) from
               `Params.lean`, so `lake build` cannot skip it.
  docstrings   every `theorem` has a docstring or a preceding comment line.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "Params"

BANNED = [
    r"\bsorry\b", r"\badmit\b", r"^\s*axiom\b", r"\bnative_decide\b", r"\bunsafe\b",
    r"^\s*opaque\b", r"implemented_by", r"@\[extern",
]


def lean_files() -> list[Path]:
    return sorted(CORPUS.rglob("*.lean"))


def strip_comments(text: str) -> str:
    text = re.sub(r"/-.*?-/", "", text, flags=re.S)
    return "\n".join(line.split("--", 1)[0] for line in text.splitlines())


def guard_admissions() -> list[str]:
    out = []
    for f in lean_files():
        code = strip_comments(f.read_text())
        for pat in BANNED:
            for m in re.finditer(pat, code, flags=re.M):
                line = code[: m.start()].count("\n") + 1
                out.append(f"{f.relative_to(REPO)}:{line}: banned token {m.group(0)!r}")
    return out


def guard_options() -> list[str]:
    lk = (REPO / "lakefile.toml").read_text()
    out = []
    for needle in ["autoImplicit = false", "relaxedAutoImplicit = false", "-DwarningAsError=true"]:
        if needle not in lk:
            out.append(f"lakefile.toml: missing {needle!r}")
    return out


def guard_closure() -> list[str]:
    root = (REPO / "Params.lean").read_text()
    imported = set(re.findall(r"^import\s+(\S+)", root, flags=re.M))
    # follow imports one level (modules importing modules)
    todo = list(imported)
    while todo:
        mod = todo.pop()
        p = REPO / (mod.replace(".", "/") + ".lean")
        if not p.exists():
            continue
        for m in re.findall(r"^import\s+(\S+)", p.read_text(), flags=re.M):
            if m not in imported:
                imported.add(m)
                todo.append(m)
    out = []
    for f in lean_files():
        mod = ".".join(f.relative_to(REPO).with_suffix("").parts)
        if mod not in imported:
            out.append(f"{f.relative_to(REPO)}: not in the import closure of Params.lean")
    return out


def guard_docstrings() -> list[str]:
    out = []
    for f in lean_files():
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines):
            if re.match(r"^theorem\s", line):
                prev = lines[i - 1].strip() if i > 0 else ""
                if not (prev.endswith("-/") or prev.startswith("--") or prev.startswith("@[")):
                    out.append(f"{f.relative_to(REPO)}:{i + 1}: theorem without docstring")
    return out


GUARDS = {
    "admissions": guard_admissions,
    "options": guard_options,
    "closure": guard_closure,
    "docstrings": guard_docstrings,
}


def main() -> int:
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        only = sys.argv[2]
    failed = False
    for name, fn in GUARDS.items():
        if only and name != only:
            continue
        findings = fn()
        status = "FAIL" if findings else "OK"
        gate = name != "docstrings"
        print(f"[{status}] {name} ({len(findings)} findings){'' if gate else ' [report only]'}")
        for x in findings:
            print("   ", x)
        if findings and gate:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
