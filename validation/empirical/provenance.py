"""Provenance and error bars for the empirical sweeps (after Descent's `simprov.py`).

Every results file records the corpus revision it describes, whether the tree
was clean, when and where it ran, the seed, and per-cell mean / sd / standard
error over replicates, plus every replicate.  A number without a revision
cannot be re-checked and a number without an error bar cannot falsify anything.
"""
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _find_repo():
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / ".git").exists():
            return cand
    return here.parents[2]


REPO = _find_repo()
_EXCLUDE = ":(exclude)validation/**/*.json"


def _git(*args, default=""):
    try:
        r = subprocess.run(["git", "-C", str(REPO)] + list(args),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if r.returncode != 0:
            return default
        return r.stdout.decode("utf-8", "replace").strip()
    except Exception:
        return default


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def stamp(generator, config, seed, replicates):
    porcelain = _git("status", "--porcelain", "--", _EXCLUDE, default=None)
    dirty = None if porcelain is None else [ln.strip() for ln in porcelain.split("\n") if ln.strip()]
    return {
        "generator": generator,
        "revision": _git("rev-parse", "HEAD", default="unknown"),
        "revisionShort": _git("rev-parse", "--short", "HEAD", default="unknown"),
        "workingTreeClean": None if dirty is None else (dirty == []),
        "workingTreeDirtyPaths": dirty,
        "runAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "seed": seed,
        "replicates": replicates,
        "argv": sys.argv,
        "config": config,
    }


def summarize(values):
    vals = list(values)
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    n_all, n = len(vals), len(xs)
    if n == 0:
        return {"n": 0, "n_dropped": n_all, "mean": None, "sd": None, "se": None, "min": None, "max": None}
    mean = sum(xs) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        sd = math.sqrt(var)
        se = sd / math.sqrt(n)
    else:
        sd = se = None
    return {"n": n, "n_dropped": n_all - n, "mean": mean, "sd": sd, "se": se, "min": min(xs), "max": max(xs)}


def write(path, generator, config, seed, replicates, cells, records):
    obj = {"_provenance": stamp(generator, config, seed, replicates),
           "cells": list(cells), "records": list(records)}
    p = Path(path)
    if str(p.parent) not in ("", "."):
        os.makedirs(p.parent, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    return p
