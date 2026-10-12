"""Which copy(all) texts copy: replace the token after the previous occurrence of the current token (after_prev) by
the model's three likeliest other tokens there; a copy loses log p(predicted token) and gains log p(the new token)."""
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("ORACLE") or str(Path(__file__).resolve().parents[3] / "method" / "oracle"))

sys.path.insert(0, str(Path(__file__).resolve().parent))


import carry  # noqa: E402
import families  # noqa: E402
import mech  # noqa: E402
import native  # noqa: E402

DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs

tk = mech.tokenizer("vpd4l")
rows = pickle.load(open(DATA / "rows.pkl", "rb"))
fam = families.behaviours(rows, tk)
nat = native.Native()
out = []
for i in fam["copy(all)"]:
    r = rows[i]
    ids, t, a = r["ids"], r["t"], r["roles"]["after_prev"]
    pred = ids[a]
    base = nat.reference([ids], [t])[0, 0]
    alts = carry.alternatives(nat, ids, a, (0, 1, 2))
    lp = nat.reference(alts, [t])[:, 0]
    drop = float(base[pred] - lp[:, pred].mean())
    gain = float(sum(lp[j, x[a]] - base[x[a]] for j, x in enumerate(alts)) / len(alts))
    out.append({"task": r["task"], "cur": r["cur"], "pred": r["pred"], "p_pred": float(base[pred].exp()), "drop_nats": drop,
                "new_gain_nats": gain, "dist": t - r["roles"]["prev_occ"]})
    print(out[-1], flush=True)
json.dump(out, open(DATA / "copytest.json", "w"), indent=1)
