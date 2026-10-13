"""Do explanations predict behaviour on texts they were not built from? (#2951 graph oracle; the lead's 10-12, after
Geiger, Harding and Icard 2026.) The variable teacher's verified variables on a text A (var_teacher.py:
~/mpd-data/graph_oracle/variables/heldout, subcomponent groups, and variables/blocks/heldout, block groups; each with
its value map, var_classes.py: per input position the alternative tokens alike the text's own and the others grouped
by where they send the prediction) are mapped by role (each position by its offset from the target) onto the other
texts B of A's family (DATA/families.json), and tested there:
  E4 on B: variables.Scorer in the real model (eval split): on-target f, off-target s, e4;
  predictions on B from A's value map alone, checked in the model on B: an alternative A's map calls alike (the same
     value) leaves B's prediction (its likeliest token) unchanged; one A's map calls different changes it; and, stronger,
     moves it to the token A's map says that value predicts.
Beside them, A's variables on A (their recorded eval-split scores) and B's own variables where B has some.

  var_transfer.py run FAMILY SOURCE   -> DATA/var_transfer.jsonl (both variable sets of SOURCE on every other member)
  var_transfer.py report
  var_transfer.py run2 FAMILY SOURCE  -> var_transfer_v2.jsonl: varteach's influential_v2 variables, each scored alone
                                         on up to 4 other members (the lead's keep rule: a variable is kept when its
                                         carrier transfers within its family)
  var_transfer.py report2             -> DATA/var_transfer_v2_keep.json, per variable whether it transfers
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("ORACLE") or str(Path(__file__).resolve().parents[3] / "method" / "oracle"))

import native  # noqa: E402

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
VARS = Path.home() / "mpd-data/graph_oracle/variables"
SETS = {"subcomponents": VARS / "heldout", "blocks": VARS / "blocks" / "heldout"}
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"


def target(task):
    return native.text(TEXTS / f"{task}.json")[1][-1]


def mapped(var: dict, ta: int, tb: int) -> dict | None:
    """var with every position moved from offset ta - p on A to tb - (ta - p) on B; None when one falls before B."""
    mv = lambda p: tb - (ta - int(p))  # noqa: E731
    inputs = [mv(p) for p in var["inputs"]]
    group = [[g[0], g[1], mv(g[2])] for g in var["group"]]
    if min(inputs + [g[2] for g in group]) < 1:
        return None
    return {**{k: var[k] for k in ("name", "meaning", "feeds")}, "inputs": inputs, "group": group}


def run(family: str, source: str):
    import goal_eval
    import variables

    fams = json.loads((DATA / "families.json").read_text())
    members = [t for t in fams[family] if t != source]
    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    sc = variables.Scorer(ge)
    ta = target(source)
    out = DATA / "var_transfer.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["family"], r["source"], r["target"], r["set"]))
    for kind, d in SETS.items():
        p = d / f"{source}.json"
        if not p.exists():
            continue
        src = json.loads(p.read_text())["variables"]
        for b in members:
            if (family, source, b, kind) in done:
                continue
            pb = TEXTS / f"{b}.json"
            ids, targets = native.text(pb)
            tb = targets[-1]
            vs = [m for m in (mapped(v, ta, tb) for v in src) if m is not None]
            row = {"family": family, "source": source, "target": b, "set": kind, "variables": len(src), "mapped": len(vs),
                   "source_eval": [v["measured"].get("eval", {}).get("term") for v in src]}
            if vs:
                tx = ge.text(ids, targets, native.task_seed(b), json.loads(pb.read_text()))
                res = sc.score(tx, variables.parse(vs, nat.C), "eval", 8)
                row.update({k: res.get(k) for k in ("e4", "f_on", "s_off", "unresolved")})
                own = SETS[kind] / f"{b}.json"
                if own.exists():
                    row["own_eval"] = [v["measured"].get("eval", {}).get("term") for v in json.loads(own.read_text())["variables"]]
                # the value map's predictions on B, checked in the model
                base = nat.reference([ids], [tb])[0, 0]
                y = int(base.argmax())
                preds = []
                for v in src:
                    for c in v.get("classes", []):
                        pos = tb - (ta - int(c["position"]))
                        if pos < 1:
                            continue
                        alts = [a for a in c.get("alternatives", []) if a["id"] != ids[pos]]
                        if not alts:
                            continue
                        xs = []
                        for a in alts:
                            x2 = list(ids)
                            x2[pos] = a["id"]
                            xs.append(x2)
                        lp = nat.reference(xs, [tb])[:, 0]
                        for a, l in zip(alts, lp):
                            new = int(l.argmax())
                            tk_pred = a.get("predicts")
                            preds.append({"alike": bool(a["alike"]), "unchanged": new == y,
                                          "to_named": None if a["alike"] else (native_decode(new) == tk_pred)})
                ok = [p_["alike"] == p_["unchanged"] for p_ in preds]
                row["predictions"] = len(preds)
                row["accuracy"] = float(np.mean(ok)) if ok else None
                al = [p_["unchanged"] for p_ in preds if p_["alike"]]
                ch = [not p_["unchanged"] for p_ in preds if not p_["alike"]]
                row["acc_alike"] = float(np.mean(al)) if al else None
                row["acc_changed"] = float(np.mean(ch)) if ch else None
                nm = [p_["to_named"] for p_ in preds if not p_["alike"]]
                row["acc_named_token"] = float(np.mean(nm)) if nm else None
                ge.probes.clear()
                native.free(nat.dev)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(family, source, b, kind, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                                            if k in ("e4", "f_on", "s_off", "accuracy", "acc_alike", "acc_changed", "acc_named_token")}, flush=True)


V2 = VARS / "influential_v2"  # varteach's variables on fleet's influential_v2 circuits (train and heldout)


def mapped_values(var: dict, ta: int, tb: int, ids_b: list) -> list | None:
    """var's value map moved to B by offset, with the class holding B's own token there first (the classes B's token
    shares); None when B's token is in no class (the map says nothing about B there)."""
    out = []
    for vm in var.get("values") or []:
        pos = tb - (ta - int(vm["position"]))
        if pos < 1:
            return None
        cls = [[int(z) for z in c.get("token_ids", [])] for c in vm.get("classes", [])]
        own = next((i for i, c in enumerate(cls) if ids_b[pos] in c), None)
        if own is None:
            return None
        out.append({"position": pos, "classes": [{"token_ids": cls[own]}] + [{"token_ids": c} for i, c in enumerate(cls) if i != own]})
    return out


def run2(family: str, source: str, max_targets: int = 4):
    """Each of SOURCE's influential_v2 variables mapped by offset onto up to max_targets other members of its family
    and scored there one by one in the real model (variables.Scorer, eval split; the value map too when B's token falls
    in one of its classes): per variable and target f_on, f_off, f_same and their min (its E4 term) -> var_transfer_v2.jsonl
    (GEN_OUT when set). A variable transfers when its term holds on the family's other texts (report2)."""
    import goal_eval
    import variables

    fams = json.loads((DATA / "families.json").read_text())
    src_path = next((V2 / s / f"{source}.json" for s in ("train", "heldout") if (V2 / s / f"{source}.json").exists()), None)
    if src_path is None:
        return
    members = [t for t in fams[family] if t != source][:max_targets]
    out = Path(os.environ.get("GEN_OUT", DATA)) / f"var_transfer_v2{os.environ.get('GEN_TAG', '')}.jsonl"
    done = {(r["source"], r["target"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    ge = goal_eval.GoalEval(native.Native())
    ge.nat.LENGTHS_KEPT = 1
    nat = ge.nat
    variables.ROWS = int(os.environ.get("VAR_ROWS", 1))  # patched runs a batch: 8 overflowed a 24 GB card, 2 a 48 GB one on long texts
    sc = variables.Scorer(ge)
    ta = target(source)
    src = json.loads(src_path.read_text())["variables"]
    for b in members:
        if (source, b) in done:
            continue
        pb = TEXTS / f"{b}.json"
        ids, targets = native.text(pb)
        tb = targets[-1]
        vs, meta = [], []
        for v in src:
            m = mapped(v, ta, tb)
            if m is None:
                continue
            vals = mapped_values(v, ta, tb, ids) if v.get("values") else None
            if vals:
                m["values"] = vals
            vs.append(m)
            meta.append({"name": v["name"], "kind": v.get("kind"), "value_kind": v.get("value_kind"), "with_values": bool(vals),
                         "source_eval": (v.get("measured") or {}).get("eval", {}).get("term"), "inputs_offset": [ta - int(p) for p in v["inputs"]]})
        row = {"family": family, "source": source, "target": b, "variables": len(src), "mapped": len(vs), "per_variable": []}
        if vs:
            tx = ge.text(ids, targets, native.task_seed(b), json.loads(pb.read_text()))
            res = sc.score(tx, variables.parse(vs, nat.C), "eval", 8)
            for mt, x in zip(meta, res.get("variables", [])):
                term = 0.0 if x.get("f_on") is None else min(y for y in (x["f_on"], x.get("f_off"), x.get("f_same")) if y is not None)
                row["per_variable"].append({**mt, "f_on": x.get("f_on"), "f_off": x.get("f_off"), "f_same": x.get("f_same"), "term": term})
            row["e4"] = res.get("e4")
            ge.probes.clear()
            native.free(nat.dev)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(family, source, b, "e4", row.get("e4"), flush=True)


def report2(keep: float = 0.8):
    """Per variable: its own eval term (varteach's) and its terms on the family's other texts; it transfers when the
    median of those reaches keep. Summaries by kind (input, intermediate) and family."""
    import templates

    rows = []
    for p in [DATA / "var_transfer_v2.jsonl"] + templates.run_outputs("var_transfer_v2*.jsonl"):
        if p.exists():
            rows += [json.loads(x) for x in p.read_text().splitlines()]
    per = collections.defaultdict(list)
    info = {}
    for r in rows:
        for x in r["per_variable"]:
            key = (r["family"], r["source"], x["name"])
            per[key].append(x["term"])
            info[key] = x
    out = []
    for key, terms in per.items():
        x = info[key]
        out.append({"family": key[0], "source": key[1], "name": key[2], "kind": x["kind"], "value_kind": x["value_kind"],
                    "own_eval": x["source_eval"], "transfer_median": float(np.median(terms)), "targets": len(terms),
                    "transfers": float(np.median(terms)) >= keep})
    (DATA / "var_transfer_v2_keep.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} variables from {len({(o['family'], o['source']) for o in out})} texts; a variable transfers when the median "
          f"of its E4 term on its family's other texts is >= {keep}")
    for kind in ("input", "intermediate"):
        os_ = [o for o in out if o["kind"] == kind]
        if os_:
            print(f"  {kind:12s} {len(os_):4d}: transfer {np.mean([o['transfers'] for o in os_]):.2f}; own eval term median "
                  f"{np.median([o['own_eval'] for o in os_ if o['own_eval'] is not None]):.2f}, on other texts median "
                  f"{np.median([o['transfer_median'] for o in os_]):.2f}")
    for f in sorted({o["family"] for o in out}):
        os_ = [o for o in out if o["family"] == f]
        print(f"  {f:22s} {len(os_):3d} variables, transfer {np.mean([o['transfers'] for o in os_]):.2f} "
              f"(intermediate {np.mean([o['transfers'] for o in os_ if o['kind'] == 'intermediate']) if any(o['kind'] == 'intermediate' for o in os_) else float('nan'):.2f})")


_TK = []


def native_decode(i: int) -> str:
    import mech

    if not _TK:
        _TK.append(mech.tokenizer("vpd4l"))
    return _TK[0].decode([i])


def report():
    rows = [json.loads(line) for line in (DATA / "var_transfer.jsonl").read_text().splitlines()]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["family"], r["set"])].append(r)
    for (f, kind), rs in sorted(by.items()):
        rs = [r for r in rs if r.get("e4") is not None or r.get("accuracy") is not None]
        if not rs:
            continue
        m = lambda k: np.mean([r[k] for r in rs if r.get(k) is not None]) if any(r.get(k) is not None for r in rs) else float("nan")  # noqa: E731
        src = np.mean([x for r in rs for x in r["source_eval"] if x is not None])
        own = [x for r in rs for x in r.get("own_eval", []) if x is not None]
        print(f"{f} [{kind}] {len(rs)} (source, target) pairs: e4 on the target {m('e4'):.2f} (f_on {m('f_on'):.2f}, s_off "
              f"{m('s_off'):.2f}; the source's own eval term {src:.2f}" + (f", the target's own {np.mean(own):.2f}" if own else "")
              + f"); value-map predictions {m('accuracy'):.2f} (alike kept {m('acc_alike'):.2f}, different moved "
              f"{m('acc_changed'):.2f}, moved to the named token {m('acc_named_token'):.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report", "run2", "report2"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    {"run": lambda: run(*a.args), "report": report, "run2": lambda: run2(*a.args), "report2": report2}[a.cmd]()
