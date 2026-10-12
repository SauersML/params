"""A held-out test of the copy texts' shared context subcomponents.

Selection (on the 23 copy texts with teacher circuits, copy_context.json): subcomponents whose single removal at
after_prev (the position after the previous occurrence of the current token, holding the copied token) costs at least
0.5 bits in at least 4 of them. Test texts: train-split texts without a teacher circuit whose prediction copies (the
predicted token is the one after the previous occurrence of the current token, and replacing that token by the
model's three likeliest others there lowers log p(predicted) by at least 1 nat and raises the new token's by at least 1
nat, copytest's rule); non-copy controls: texts whose current token occurs earlier but whose prediction does not
depend on the token after it (log p(predicted) moves by less than 0.1 nat). Per test text, in the real model with the
remainders live (goal_eval.real_removal), the KL at the target in bits of removing each selected subcomponent, and all
of them together, at after_prev, at prev_occ, and at CONTROL_POSITIONS other positions between prev_occ and the target."""
import json
import os
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("ORACLE", str(Path(__file__).resolve().parents[3] / "method" / "oracle")))
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
sys.path.insert(0, str(HERE))

import carry  # noqa: E402
import goal_eval  # noqa: E402
import gscore  # noqa: E402
import mech  # noqa: E402
import native  # noqa: E402

CONTROL_POSITIONS = 5
WANT = 40  # test texts per kind
tk = mech.tokenizer("vpd4l")
ctx = json.load(open(DATA / "copy_context.json"))
cnt = {}
for x in ctx.values():
    for s in x["singles"]:
        if s["kl"] >= 0.5:
            cnt[(s["site"], s["index"])] = cnt.get((s["site"], s["index"]), 0) + 1
selected = sorted([k for k, v in cnt.items() if v >= 4], key=lambda k: -cnt[k])
print("selected", [(k, cnt[k]) for k in selected], flush=True)
have = {r["task"] for r in pickle.load(open(DATA / "rows.pkl", "rb"))}
ge = goal_eval.GoalEval(native.Native(), chunk=16)
ge.nat.LENGTHS_KEPT = 1
nat = ge.nat
OUT = DATA / "copy_heldout.json"
prev = json.loads(OUT.read_text()) if OUT.exists() else {}
found = {"copy": prev.get("copy", []), "noncopy": prev.get("noncopy", [])}
seen = set(prev.get("scanned", [])) | {x["task"] for v in found.values() for x in v}
scanned, built = list(seen), 0
for p in native.tasks("train"):
    if p.stem in have or p.stem in seen:
        continue
    if built >= 6:  # one process per few texts: the process's memory grows per text
        break
    scanned.append(p.stem)
    if all(len(v) >= WANT for v in found.values()):
        break
    ids, targets = native.text(p)
    t = targets[-1]
    occ = [j for j in range(t) if ids[j] == ids[t]]
    if not occ or occ[-1] + 1 >= t:
        continue
    a = occ[-1] + 1
    base = nat.reference([ids], [t])[0, 0]
    pred = int(base.argmax())
    alts = carry.alternatives(nat, ids, a, (0, 1, 2))
    lp = nat.reference(alts, [t])[:, 0]
    drop = float(base[pred] - lp[:, pred].mean())
    gain = float(sum(lp[j, x[a]] - base[x[a]] for j, x in enumerate(alts)) / len(alts))
    kind = "copy" if (pred == ids[a] and drop >= 1 and gain >= 1) else "noncopy" if abs(drop) < 0.1 and pred != ids[a] else None
    if kind is None or len(found[kind]) >= WANT:
        continue
    built += 1
    tx = gscore.text(ge, p.stem)
    o = occ[-1]
    rng = random.Random(p.stem)
    pool = [q for q in range(max(1, o - 20), t) if q not in (a, o)]
    controls = rng.sample(pool, min(CONTROL_POSITIONS, len(pool)))
    graphs, keys = [], []
    for rl, q in [("after_prev", a), ("prev_occ", o)] + [("control", q) for q in controls]:
        for n, c in selected:
            graphs.append(native.Graph([(n, q, c)], complete=True))
            keys.append((rl, q, f"{n}#{c}"))
        graphs.append(native.Graph([(n, q, c) for n, c in selected], complete=True))
        keys.append((rl, q, "all"))
    kl = [x["kl_real_minus"] for x in ge.real_removal(tx, graphs)]
    active = {f"{n}#{c}": {rl: bool(tx.vpd[n][q, c]) for rl, q in [("after_prev", a), ("prev_occ", o)]} for n, c in selected}
    found[kind].append({"task": p.stem, "cur": tk.decode([ids[t]]), "pred": tk.decode([pred]), "drop": drop, "gain": gain,
                        "dist": t - o, "removals": [{"role": k[0], "position": k[1], "what": k[2], "kl": v} for k, v in zip(keys, kl)],
                        "active": active})
    a_all = [v for k, v in zip(keys, kl) if k[0] == "after_prev" and k[2] == "all"][0]
    c_all = [v for k, v in zip(keys, kl) if k[0] == "control" and k[2] == "all"]
    print(kind, p.stem, repr(tk.decode([ids[t]])), repr(tk.decode([pred])), f"after_prev {a_all:.2f} control mean {sum(c_all) / max(1, len(c_all)):.2f}", flush=True)
    json.dump({"selected": [list(k) + [cnt[k]] for k in selected], **found, "scanned": scanned}, open(OUT, "w"))
    del tx
    ge.probes.clear()
    native.free(nat.dev)
json.dump({"selected": [list(k) + [cnt[k]] for k in selected], **found, "scanned": scanned}, open(OUT, "w"))
print("FINISHED" if all(len(v) >= WANT for v in found.values()) or built < 6 else "MORE", flush=True)
