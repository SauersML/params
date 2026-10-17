"""Every repo module a file imports, at top level or inside a function, resolves from the directories on that file's
path: its own directory, the directories its sys.path lines name (Path(__file__), or a module-level NAME defined from
it, with .parent, .parents[k] and / "a" / "b" segments), and those of the repo modules it imports at top level, whose
inserts run on import. A file moved away from a module it imports lazily fails here before any run does (10-16:
essay_replication lost neuron_essay_check)."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPR = re.compile(r"(Path\(__file__\)\.resolve\(\)|Path\(__file__\)|[A-Z][A-Z_]*)((?:\.parent(?!s))*)"
                  r"(?:\.parents\[(\d)\])?((?:\s*/\s*[\"'][^\"']+[\"'])*)")


def directory(text: str, f: Path, names: dict):
    m = EXPR.search(text)
    if not m:
        return None
    base, parents, k, segs = m.groups()
    d = f if base.startswith("Path(__file__)") else names.get(base)
    if d is None:
        return None
    for _ in range(parents.count(".parent")):
        d = d.parent
    if k is not None:
        d = d.parents[int(k)]
    for s in re.findall(r"[\"']([^\"']+)[\"']", segs):
        d = d / s
    return d


def scan():
    files = sorted(p for p in ROOT.rglob("*.py") if ".git" not in p.parts)
    where = {}
    for p in files:
        where.setdefault(p.stem, []).append(p.parent)
    info = {}
    for f in files:
        src = f.read_text(errors="replace")
        names = {}
        for m in re.finditer(r"^([A-Z][A-Z_]*)\s*=\s*(Path\(__file__\)[^\n#]*)", src, re.M):
            d = directory(m.group(2), f, {})
            if d is not None:
                names[m.group(1)] = d
        dirs = {f.parent}
        for line in src.splitlines():
            if "sys.path" in line:
                dirs |= {d for d in (directory(m.group(0), f, names) for m in EXPR.finditer(line)) if d is not None}
        top, every = set(), set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                mods = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods = [n.module.split(".")[0]]
            else:
                continue
            for m in mods:
                if m in where and m not in sys.stdlib_module_names:
                    every.add(m)
                    if n.col_offset == 0:
                        top.add(m)
        info[f] = ({d.resolve() for d in dirs}, top, every)

    def reach(f, seen):
        if f in seen:
            return set()
        seen.add(f)
        dirs, top, _ = info[f]
        out = set(dirs)
        for m in top:
            for d in where[m]:
                if d in out:
                    out |= reach(d / f"{m}.py", seen)
        return out

    return [f"{f.relative_to(ROOT)}: import {m} (in {', '.join(str(d.relative_to(ROOT)) for d in where[m])})"
            for f, (_, _, every) in info.items() for m in sorted(every)
            if not any(d in reach(f, set()) for d in where[m])]


def test_every_repo_import_resolves():
    missing = scan()
    assert not missing, "imports that resolve from no directory on the importer's path:\n" + "\n".join(missing)
