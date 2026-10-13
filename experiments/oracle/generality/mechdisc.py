"""Mechanism discrimination on vpd4l (#2951 graph oracle; the lead's 10-13: the reviewers' AND vs AND-NOT test in the
real model): tasks where the model gives the same top token through different mechanisms, each mechanism confirmed by
measured interventions, and whether a circuit (the v2.3 teacher's; S1's answers) holds the right mechanism's core parts
and not the other's.

Items are windows of dev texts (text9000-9099, never the sealed set) of at most W tokens ending at a target t (the
window's last position), y the model's top token there on the window:
  copy: the current token occurs earlier in the window (o its latest occurrence, a = o + 1) and y is the token at a;
     confirmed when replacing the token at a (DRAWS alternatives: the model's likeliest tokens there, then samples
     from its prediction) takes y off the top on at least half the draws and lowers log p(y) by at least DROP nats
     on average.
  bigram: neither the current token nor y occurs earlier in the window; confirmed when replacing the current token
     takes y off the top on at least half the draws and replacing DRAWS earlier positions (one each) lowers log p(y)
     by less than QUIET nats on average with y kept on top on every draw (the context does not carry it).
  context: neither the current token nor y occurs earlier in the window, and one position m at least FAR before t
     carries y: replacing it (DRAWS alternatives) takes y off the top on at least half the draws and lowers log p(y)
     by at least DROP nats (a long-range association that is not a copy).
Pairs: within each contrast (copy against bigram, context against bigram) items with the same y, each item in one
pair, a text in at most PER_TEXT items.
Core parts (copy-mechanism-vpd, measured on 40 held-out copy texts): at a, h.1.attn.q_proj #316, h.0.mlp.c_fc #53,
h.1.mlp.c_fc #2103 and h.2.attn.k_proj #206; at o, h.1.attn.k_proj #329 and #119.

  mechdisc.py find [K P]      -> DATA/mechdisc_found_K.jsonl: every confirmed item of texts K, K + P, ... (model runs only)
  mechdisc.py pairs           -> DATA/mechdisc_items.json (the pairs) and DATA/mechdisc_texts/<name>.json task files
  mechdisc.py teach --name N  the v2.3 teacher on one item (teacher.py: --f1-start set --accept-mode pp --accept-draws
                              fv --phi-check --rank once --accept changed,floors, semantics n) -> OUT/mechdisc_teacher/
  mechdisc.py score [--answers JSONL]   per item and pair whether the circuits hold each mechanism, bootstrap intervals
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pairs() runs on the standard library alone (a Mac whose memory ledger is full)
    np = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ORACLE = Path(os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))
sys.path.insert(0, str(ORACLE))

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))
OUTROOT = Path(os.environ["GEN_OUT"]) if os.environ.get("GEN_OUT") else DATA
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
DEV = [f"text{i}" for i in range(9000, 9100)]  # the development texts (text9100-9149 is the sealed final test)
SEALED = re.compile(r"text91[0-4][0-9][.]json")
W, DRAWS, DROP, QUIET, FAR, P_MIN, PER_TEXT, CANDS = 96, 8, 1.0, 0.25, 8, 0.3, 4, 12
MAX_PAIRS = 40  # copy-bigram pairs taken into the teacher runs (of every confirmed pair)
KINDS = tuple(os.environ.get("MD_KINDS", "copy,bigram,context").split(","))  # find: the kinds searched
CANDS = int(os.environ.get("MD_CANDS", CANDS))  # find: candidates confirmed per text and kind
CORE_A = [("h.1.attn.q_proj", 316), ("h.0.mlp.c_fc", 53), ("h.1.mlp.c_fc", 2103), ("h.2.attn.k_proj", 206)]
CORE_O = [("h.1.attn.k_proj", 329), ("h.1.attn.k_proj", 119)]
TEACHER_FLAGS = ["--f1-start", "set", "--accept-mode", "pp", "--accept-draws", "fv", "--phi-check", "--rank", "once",
                 "--accept", "changed,floors", "--max-tokens", "0"]


def _alts(nat, x: list, p: int, k: int, rng) -> list:
    """k replacement tokens for position p of x: the model's likeliest there (the token itself left out), then
    samples from its prediction."""
    import torch

    lp = nat.replacement_logp(x, p).float().cpu()
    lp[x[p]] = -float("inf")
    top = lp.topk(k // 2).indices.tolist()
    w = lp.exp()
    w[top] = 0.0
    more = torch.multinomial(w, k - len(top), replacement=False, generator=torch.Generator().manual_seed(int(rng.integers(2 ** 31)))).tolist()
    return top + more


def _each(nat, x: list, t: int, y: int, changes: list) -> tuple:
    """Per changed text (changes: lists of (position, token)), whether y is off the top and log p(y)'s drop in nats."""
    xs = []
    for ch in changes:
        x2 = list(x)
        for p, z in ch:
            x2[p] = z
        xs.append(x2)
    base = float(nat.reference([x], [t])[0, 0, y])
    lp = nat.reference(xs, [t])[:, 0]
    return (lp.argmax(-1) != y).float().cpu().numpy(), (base - lp[:, y].float()).cpu().numpy()


def _effect(nat, x: list, t: int, y: int, changes: list) -> tuple:
    """(share of changed texts whose top token is no longer y, mean drop of log p(y) in nats) over changes."""
    off, drop = _each(nat, x, t, y, changes)
    return float(off.mean()), float(drop.mean())


def find(k: int = 0, P: int = 1):
    import native

    nat = native.Native()
    out = OUTROOT / f"mechdisc_found_{k}{'_' + '_'.join(KINDS) if KINDS != ('copy', 'bigram', 'context') else ''}{os.environ.get('GEN_TAG', '')}.jsonl"
    done = {json.loads(x)["text"] for x in out.read_text().splitlines() if '"done"' in x} if out.exists() else set()
    for name in DEV[k::P]:
        if name in done or not (TEXTS / f"{name}.json").exists():
            continue
        ids, _ = native.text(TEXTS / f"{name}.json")
        rng = np.random.default_rng(native.task_seed(name))
        full = nat.reference([ids], list(range(len(ids))))[0]  # a cheap pre-filter on the whole text's predictions
        cands = {"copy": [], "bigram": [], "context": []}
        for t in range(24, len(ids) - 1):
            if float(full[t].exp().max()) < P_MIN:
                continue
            s = max(0, t - W + 1)
            x = ids[s:t + 1]
            tt = len(x) - 1
            y = int(full[t].argmax())
            occ = [j for j in range(tt) if x[j] == x[tt]]
            if occ and occ[-1] + 1 < tt and x[occ[-1] + 1] == y:
                cands["copy"].append(t)
            elif not occ and y not in x[:tt]:
                cands["bigram"].append(t)
                cands["context"].append(t)
        rows = []
        for kind in KINDS:
            pick = [cands[kind][i] for i in rng.permutation(len(cands[kind]))[:CANDS]]
            for t in sorted(pick):
                s = max(0, t - W + 1)
                x = ids[s:t + 1]
                tt = len(x) - 1
                lpx = nat.reference([x], [tt])[0, 0]
                y = int(lpx.argmax())
                if float(lpx[y].exp()) < P_MIN:
                    continue
                row = {"kind": kind, "text": name, "t": t, "start": s, "y": y, "p_y": float(lpx[y].exp())}
                if kind == "copy":
                    occ = [j for j in range(tt) if x[j] == x[tt]]
                    if not occ or occ[-1] + 1 >= tt or x[occ[-1] + 1] != y:
                        continue
                    o, a = occ[-1], occ[-1] + 1
                    off, drop = _effect(nat, x, tt, y, [[(a, z)] for z in _alts(nat, x, a, DRAWS, rng)])
                    row.update(o=s + o, a=s + a, off_a=off, drop_a=drop, confirmed=off >= 0.5 and drop >= DROP)
                elif kind == "bigram":
                    if any(x[j] == x[tt] for j in range(tt)) or y in x[:tt]:
                        continue
                    off_t, drop_t = _effect(nat, x, tt, y, [[(tt, z)] for z in _alts(nat, x, tt, DRAWS, rng)])
                    ctx = [int(p) for p in rng.choice(np.arange(1, tt), size=min(DRAWS, tt - 1), replace=False)]
                    off_c, drop_c = _effect(nat, x, tt, y, [[(p, _alts(nat, x, p, 2, rng)[0])] for p in ctx])
                    row.update(off_t=off_t, drop_t=drop_t, off_ctx=off_c, drop_ctx=drop_c,
                               confirmed=off_t >= 0.5 and drop_c < QUIET and off_c == 0.0)
                else:
                    if any(x[j] == x[tt] for j in range(tt)) or y in x[:tt] or tt <= FAR + 1:
                        continue
                    far = [int(p) for p in rng.choice(np.arange(1, tt - FAR + 1), size=min(16, tt - FAR), replace=False)]
                    _, drops = _each(nat, x, tt, y, [[(p, _alts(nat, x, p, 2, rng)[0])] for p in far])  # one alternative each
                    m = far[int(np.argmax(drops))]  # the far position whose replacement moves y most
                    off, drop = _effect(nat, x, tt, y, [[(m, z)] for z in _alts(nat, x, m, DRAWS, rng)])
                    row.update(m=s + m, off_m=off, drop_m=drop, confirmed=off >= 0.5 and drop >= DROP)
                rows.append(row)
        with out.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"text": name, "done": True}) + "\n")
        print(name, {kd: sum(r["confirmed"] for r in rows if r["kind"] == kd) for kd in cands}, flush=True)
        native.free(nat.dev)


def pairs():
    """Confirmed items paired by y within each contrast (copy-bigram, context-bigram); task files for every paired
    item. Standard library only."""
    import random

    found = []
    roots = [Path("/Users/user/mpd-data/runpod"), Path("/Users/user/mpd-data/cluster")]
    runs = sorted(p for r in roots if r.exists() for p in r.glob("oracle-graph-generality*/out/mechdisc_found_*.jsonl"))
    for p in runs:
        if p.exists():
            found += [json.loads(x) for x in p.read_text().splitlines() if '"done"' not in x]
    seen, items = set(), []
    for r in found:
        key = (r["kind"], r["text"], r["t"])
        if r.get("confirmed") and key not in seen:
            seen.add(key)
            items.append(r)
    rng = random.Random(2951)
    per_text = collections.Counter()
    used = set()
    out = []
    for a_kind in ("copy", "context"):
        by_y = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in items:
            if r["kind"] in (a_kind, "bigram"):
                by_y[r["y"]][r["kind"]].append(r)
        for y in sorted(by_y):
            A = [r for r in by_y[y][a_kind]]
            B = [r for r in by_y[y]["bigram"]]
            rng.shuffle(A)
            rng.shuffle(B)
            for ra in A:
                if (ra["kind"], ra["text"], ra["t"]) in used or per_text[ra["text"]] >= PER_TEXT:
                    continue
                rb = next((r for r in B if ("bigram", r["text"], r["t"]) not in used and per_text[r["text"]] < PER_TEXT
                           and r["text"] != ra["text"]), None)
                if rb is None:
                    continue
                for r in (ra, rb):
                    used.add((r["kind"], r["text"], r["t"]))
                    per_text[r["text"]] += 1
                out.append({"contrast": f"{a_kind}-bigram", "y": y, "items": [ra, rb]})
    cb = [p_ for p_ in out if p_["contrast"] == "copy-bigram"]
    rng.shuffle(cb)
    paired = {(r["kind"], r["text"], r["t"]) for p_ in out if p_["contrast"] == "context-bigram" for r in p_["items"]}
    solo = [{"contrast": "context", "y": r["y"], "items": [r]} for r in items
            if r["kind"] == "context" and (r["kind"], r["text"], r["t"]) not in paired]
    out = cb[:MAX_PAIRS] + [p_ for p_ in out if p_["contrast"] == "context-bigram"] + solo
    tdir = DATA / "mechdisc_texts"
    tdir.mkdir(exist_ok=True)
    names = set()
    for pr in out:
        for r in pr["items"]:
            r["name"] = f"md{r['kind'][0]}_{r['text']}_{r['t']}"
            if r["name"] in names:
                continue
            names.add(r["name"])
            ids = json.loads((TEXTS / f"{r['text']}.json").read_text())["prompts"][0]["token_ids"]
            x = ids[r["start"]:r["t"] + 1]
            (tdir / f"{r['name']}.json").write_text(json.dumps({
                "id": r["name"], "model": "vpd4l", "family": "mechdisc", "split": "mechdisc", "description": r["kind"],
                "prompts": [{"token_ids": x, "target_positions": [len(x) - 1]}]}))
    (DATA / "mechdisc_items.json").write_text(json.dumps({"pairs": out, "found": len(found), "confirmed": len(items)}, indent=1))
    c = collections.Counter(pr["contrast"] for pr in out)
    print(f"{len(found)} candidates, {len(items)} confirmed ({collections.Counter(r['kind'] for r in items)}); pairs {dict(c)}; "
          f"{len(names)} task files")


def texts_root() -> Path:
    import native

    root = DATA / "mechdisc_root"  # (not under the synced outputs: it links every text)
    d = root / "vpd4l"
    d.mkdir(parents=True, exist_ok=True)
    real = native.TEXTS / "vpd4l"
    for f in os.listdir(real):
        if f.endswith(".json") and not (d / f).exists() and not SEALED.fullmatch(f):
            (d / f).symlink_to(real / f)
    for f in os.listdir(DATA / "mechdisc_texts"):
        if not (d / f).exists():
            (d / f).symlink_to(DATA / "mechdisc_texts" / f)
    return root


def teach(name: str):
    """The v2.3 teacher on one item's window, in this process (teacher.py by runpy, native.TEXTS on texts_root())."""
    import runpy

    import native

    tdir = OUTROOT / "mechdisc_teacher"
    if (tdir / f"{name}.json").exists():
        return
    native.TEXTS = texts_root()
    lst = OUTROOT / f"md_ids_{name}.txt"
    lst.write_text(name + "\n")
    sys.argv = ["teacher.py", "--tasks", str(lst), *TEACHER_FLAGS, "--out", str(tdir)]
    runpy.run_path(str(ORACLE / "teacher.py"), run_name="__main__")


def _held(ans: dict) -> set:
    """An answer's (site, position, index) nodes, a block as (site, position, None)."""
    import goal_eval

    out = set()
    for m, c, p in ans.get("nodes", []):
        if m == "block":
            out.add((goal_eval._site(c), int(p), None))
        elif m != "tok":
            out.add((goal_eval._site(m), int(p), int(c)))
    return out


def _induction_anywhere(nodes: set) -> bool:
    """The layer-1 induction pair anywhere: q #316 at some j with k #329 at j - 1."""
    qs = {p for n, p, c in nodes if (n, c) == ("h.1.attn.q_proj", 316)}
    ks = {p for n, p, c in nodes if (n, c) == ("h.1.attn.k_proj", 329)}
    return any(j - 1 in ks for j in qs)


def judge(r: dict, ans: dict) -> dict:
    """Whether an answer on item r holds its mechanism and not the other's (positions in the window)."""
    nodes = _held(ans)
    s = r["start"]
    t = r["t"] - s
    pos = {p for _, p, _ in nodes}
    blocks = {(n, p) for n, p, c in nodes if c is None}
    has = lambda n, p, c: (n, p, c) in nodes or (n, p) in blocks  # noqa: E731
    out = {"nodes": len(nodes), "positions": len(pos), "context_share": float(np.mean([p < t - 1 for _, p, _ in nodes])) if nodes else 0.0,
           "induction_anywhere": _induction_anywhere(nodes)}
    if r["kind"] == "copy":
        a, o = r["a"] - s, r["o"] - s
        core = [has(n, a, c) for n, c in CORE_A] + [has(n, o, c) for n, c in CORE_O]
        out.update(holds_a=a in pos, holds_o=o in pos, core_share=float(np.mean(core)), core_any=any(core),
                   right=(a in pos and o in pos), wrong=not (pos - {t, t - 1}))  # wrong: the bigram reading (target-local only)
    elif r["kind"] == "context":
        m = r["m"] - s
        out.update(holds_m=m in pos, right=m in pos, wrong=out["induction_anywhere"])
    else:
        out.update(right=not out["induction_anywhere"], wrong=out["induction_anywhere"])
    return out


def _boot(ws, B: int = 2000, seed: int = 2951):
    ws = np.asarray([w for w in ws if w is not None], dtype=float)
    if len(ws) == 0:
        return "-"
    r = np.random.default_rng(seed)
    bs = ws[r.integers(len(ws), size=(B, len(ws)))].mean(1)
    return f"{ws.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}] (n={len(ws)})"


def score(answers: Path | None = None, tdirs: list | None = None):
    """Per contrast: the teacher's circuits (every run's mechdisc_teacher/) and S1's answers (answers: JSONL of
    {task, sample, text}, the circuit components.parse(text)["answer"]), each item's right (copy: holds a and o;
    context: holds m; bigram: no layer-1 induction pair) and wrong (copy: target-local only; context and bigram: an
    induction pair), and a pair right when both items are right; bootstrap 95% intervals over pairs."""
    import templates

    pr = json.loads((DATA / "mechdisc_items.json").read_text())["pairs"]
    teach_ = {}
    for d in (tdirs or []) + [p.parent for p in templates.run_outputs("mechdisc_teacher/md*.json")] + [OUTROOT / "mechdisc_teacher"]:
        d = Path(d)
        if d.exists():
            for f in d.iterdir():
                if f.suffix == ".json" and f.stem.count(".") == 0 and f.stem.startswith("md"):
                    teach_.setdefault(f.stem, json.loads(f.read_text()))
    sources = {"teacher (v2.3)": {k: [v] for k, v in teach_.items()}}
    if answers:
        import components

        s1 = collections.defaultdict(list)
        for x in Path(answers).read_text().splitlines():
            ln = json.loads(x)
            s1[ln["task"]].append(components.parse(ln["text"])["answer"])
        sources["S1"] = s1
    for src, by in sources.items():
        print(f"== {src}")
        for contrast in sorted({p["contrast"] for p in pr}):
            ps = [p for p in pr if p["contrast"] == contrast and all(r["name"] in by for r in p["items"])]
            js = [[[judge(r, a) for a in by[r["name"]]] for r in p["items"]] for p in ps]
            mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
            right_a = [mean([j["right"] for j in jj[0]]) for jj in js]
            right_b = [mean([j["right"] for j in jj[1]]) for jj in js]
            wrong_a = [mean([j["wrong"] for j in jj[0]]) for jj in js]
            wrong_b = [mean([j["wrong"] for j in jj[1]]) for jj in js]
            both = [mean([ja["right"] and jb["right"] and not ja["wrong"] and not jb["wrong"] for ja in jj[0] for jb in jj[1]]) for jj in js]
            ka = contrast.split("-")[0]
            print(f"  {contrast}: {len(ps)} pairs with circuits for both items")
            print(f"    {ka} item holds its mechanism       {_boot(right_a)}")
            print(f"    {ka} item holds the other reading   {_boot(wrong_a)}")
            print(f"    bigram item holds its reading (no induction pair) {_boot(right_b)}")
            print(f"    pair right (both right, neither wrong) {_boot(both)}")
            if ka == "copy":
                print(f"    copy item's core parts at their roles (share of 6) {_boot([mean([j['core_share'] for j in jj[0]]) for jj in js])}")
            print(f"    context share of nodes: {ka} {_boot([mean([j['context_share'] for j in jj[0]]) for jj in js])}, bigram "
                  f"{_boot([mean([j['context_share'] for j in jj[1]]) for jj in js])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("find", "pairs", "teach", "score"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--name", default=None)
    ap.add_argument("--answers", type=Path, default=None)
    ap.add_argument("--kinds", default=None, help="find: the kinds searched (comma-separated)")
    ap.add_argument("--cands", type=int, default=None, help="find: candidates confirmed per text and kind")
    a = ap.parse_args()
    if a.kinds:
        KINDS = tuple(a.kinds.split(","))
    if a.cands:
        CANDS = a.cands
    if a.cmd == "find":
        find(*(int(x) for x in a.args))
    elif a.cmd == "pairs":
        pairs()
    elif a.cmd == "teach":
        teach(a.name)
    else:
        score(a.answers, a.args)
