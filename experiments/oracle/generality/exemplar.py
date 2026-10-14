"""The exemplar of an oracle answer (#2951 graph oracle, the lead's 10-14): on text9000 ("The new structure involved the
creation" -> " of"), rustmove's stated circuit (teacher/stated/text9000.json: 55 parts, 173 lines), an English account
whose claims report their measurements and pass predicates, distinguishing full-model interventions and associations
from the circuit's own edit predictions, one predicted edit
measured, and a one-page figure (the circuit beside the English).

  exemplar.py measure    -> OUT/measure.json: the real model's p(" of"); each part removed alone and groups removed
                            together (VPD's mask at the part's position, the remainder kept); the circuit's own
                            prediction of a joint removal (its run, goal_eval's engine) against the real model's; token
                            swaps at "creation" with the top parts' activations, and the swaps with a part's activation
                            set back to its value on the original text; each part's association labels (PheWAS)
  exemplar.py answer     -> OUT/answer.json: the English, its claims (sentence, test, measured, passed) and the edit
  exemplar.py figure     -> FIG/exemplar_text9000.png: the circuit (the lead's renderer, connections and parts sized by
                            their measured effects in the circuit's run) beside the English
Runs on the CPU (another job's scorer holds the Mac's GPU).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORACLE = Path(os.environ.get("ORACLE") or str(HERE.parents[2] / "method" / "oracle"))
sys.path.insert(0, str(ORACLE))
sys.path.insert(0, str(ORACLE.parent / "vpd"))

OUT = Path(os.environ.get("EX_OUT", Path.home() / "mpd-data/graph_oracle/exemplar"))
STATED = Path.home() / "mpd-data/graph_oracle/teacher/stated"
TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
GW = Path.home() / "mpd-data/graph_oracle/gwas"
LEAD = Path("/private/tmp/claude-501/-Users-user/b1063ff3-6750-418e-8406-23ae82bf968a/scratchpad/lead")
FIG = Path("/Users/user/mpd-data/figures/oracle")
TASK = "text9000"
LN2 = math.log(2)
SWAPS = {"takes 'of' (another action noun)": [" destruction", " construction", " development", " establishment", " formation", " removal"],
         "a noun that rarely takes 'of'": [" building", " staff", " work", " system", " plans", " office"],
         "not a noun": [" new", " and", " very", " first"]}


def _label(n: str, c: int) -> str:
    return f"L{n.split('.')[1]}.{n.split('.')[-1].replace('_proj', '').replace('c_fc', 'fc')}#{c}"


def _setup():
    import torch

    torch.backends.mps.is_available = lambda: False  # the CPU: the Mac's GPU runs another job's scorer
    import native

    nat = native.Native()
    task = json.loads((TEXTS / f"{TASK}.json").read_text())
    ids = task["prompts"][0]["token_ids"]
    t = task["prompts"][0]["target_positions"][0]
    circ = json.loads((STATED / f"{TASK}.json").read_text())
    nodes = [(n, int(p), int(c)) for n, c, p in circ["nodes"] if n.startswith("h.")]  # (token nodes are inputs, not parts)
    return nat, task, ids, t, circ, nodes


def measure():
    """The real model's numbers for the answer (module docstring)."""
    import torch

    import goal_eval
    import mech

    nat, task, ids, t, circ, nodes = _setup()
    tk = mech.tokenizer("vpd4l")
    OF = tk.encode(" of")[0] if isinstance(tk.encode(" of"), list) else tk.encode(" of").ids[0]
    T = len(ids)
    dev = nat.dev

    def lp_of(seq, masks=None, patch=None):
        """log p at the target: the whole model, parts removed (masks: [(site, pos, idx)]), or activations patched
        (patch: {site: [(pos, idx, value)]})."""
        x = torch.tensor([seq], device=dev)
        with torch.no_grad():
            nat.vpd.clear()
            if masks:
                mk = {n: torch.ones((1, T, nat.target.site(n).V.shape[1]), device=dev) for n in nat.names}
                for n, p, c in masks:
                    mk[n][0, p, c] = 0.0
                dm = {n: torch.ones((1, T), device=dev) for n in nat.names}
                out = nat.vpd.masked(x, mk, dm)
            elif patch:
                import types

                saved = {}
                for n, items in patch.items():
                    st = nat.target.site(n)
                    saved[n] = st._forward

                    def fwd(self_, xx, items=items, st=st):
                        a = xx @ st.V
                        for p, c, v in items:
                            a[:, p, c] = v
                        return a @ st.U + xx @ st.delta_T

                    st._forward = types.MethodType(fwd, st)
                try:
                    out = nat.target(x)
                finally:
                    for n, f in saved.items():
                        nat.target.site(n)._forward = f
            else:
                out = nat.target(x)
        return torch.log_softmax(out[0, t].float(), -1)

    def acts(seq):
        x = torch.tensor([seq], device=dev)
        nat.vpd.clear()
        for n in nat.names:
            nat.target.site(n).cache_input = True
        with torch.no_grad():
            nat.target(x)
        a = {n: (nat.target.site(n).last_input[0] @ nat.target.site(n).V) for n in nat.names}
        nat.vpd.clear()
        return a

    base = lp_of(ids)
    p0 = float(base[OF].exp())
    top = base.topk(5)
    res = {"task": TASK, "text_end": tk.decode(ids[T - 12:]), "target": t, "p_of": p0,
           "top": [[tk.decode([int(i)]), float(v.exp())] for v, i in zip(top.values, top.indices)], "parts": [], "groups": {}}
    lab = json.loads((GW / "labels.json").read_text())
    a0 = acts(ids)
    for nd in nodes:
        n, p, c = nd
        lp = lp_of(ids, masks=[nd])
        res["parts"].append({"part": _label(n, c), "site": n, "pos": p, "index": c, "token": tk.decode([ids[p]]),
                             "p_of_removed": float(lp[OF].exp()), "drop_bits": float((base[OF] - lp[OF]) / LN2),
                             "activation": float(a0[n][p, c]), "label": lab.get(_label(n, c))})
    res["parts"].sort(key=lambda r: -r["drop_bits"])
    top5 = [(r["site"], r["pos"], r["index"]) for r in res["parts"][:5]]
    for name, sel in ((f"all {len(nodes)} parts", nodes), ("the parts at 'the'", [x for x in nodes if x[1] == t - 1]),
                      ("the parts at 'creation'", [x for x in nodes if x[1] == t]), ("the top 5 parts", top5),
                      ("the top 3 parts", top5[:3])):
        lp = lp_of(ids, masks=sel)
        res["groups"][name] = {"parts": len(sel), "p_of": float(lp[OF].exp()), "drop_bits": float((base[OF] - lp[OF]) / LN2)}
    # the circuit's own prediction of the edit (removing the top 3 parts), against the real model's
    ge = goal_eval.GoalEval(nat)
    tx = ge.text(ids, [t], seed=0, task=task)
    g, _, _ = goal_eval.load_answer(STATED / f"{TASK}.json", task, [t], nat)
    E = ge._circuit_engine(ids, [t], [g])
    plan = E.plan(g, semantics="c")
    ne, no = plan["E"], plan["O"]
    lq_full = E.graph_log_probs(plan, ew=torch.ones(1, ne, device=dev), ow=torch.ones(1, no, device=dev))
    lq_edit = E.graph_log_probs(plan, ew=torch.ones(1, ne, device=dev), ow=torch.ones(1, no, device=dev), off=[top5[:3]])
    pc, pe = float(lq_full[0, 0, OF].exp()), float(lq_edit[0, 0, OF].exp())
    single3 = sum(r["drop_bits"] for r in res["parts"][:3])
    res["edit"] = {"edit": "remove the top 3 parts together", "parts": [_label(n, c) for n, p, c in top5[:3]],
                   "circuit p_of": pc, "circuit p_of after": pe, "circuit predicted drop_bits": math.log2(pc / pe),
                   "sum of single drops (bits)": single3, "real drop_bits": res["groups"]["the top 3 parts"]["drop_bits"],
                   "real p_of after": res["groups"]["the top 3 parts"]["p_of"]}
    # node and connection effects in the circuit's run (KL rise, bits) for the figure
    ED = [(w, r) for r, ws in g.parents.items() for w in ws]
    OD = list(g.out)
    import carry

    def kl_rows(ew, ow, off=None):
        lq = E.graph_log_probs(plan, ew=ew, ow=ow, off=off)
        return carry.kl(tx.ref, lq).tolist()

    one_e, one_o = torch.ones(1, ne, device=dev), torch.ones(1, no, device=dev)
    k0 = kl_rows(one_e, one_o)[0]
    edge_eff = []
    for j in range(ne + no):
        ew, ow = one_e.clone(), one_o.clone()
        (ew if j < ne else ow)[0, j if j < ne else j - ne] = 0
        edge_eff.append(kl_rows(ew, ow)[0] - k0)
    node_eff = {nd: kl_rows(one_e, one_o, [[nd]])[0] - k0 for nd in g.node_set()}
    res["circuit_kl"] = k0
    res["render"] = {"task": TASK, "target": t, "tokens": [tk.decode([x]) for x in ids], "predicted": tk.decode([OF]),
                     "nodes": [{"layer": int(n.split(".")[1]), "kind": n.split(".")[-1], "index": c, "pos": p, "removal": float(node_eff[(n, p, c)])}
                               for n, p, c in sorted(g.node_set(), key=lambda z: (z[1], z[0], z[2]))],
                     "edges": [{"w": [w[0], w[1], w[2]], "r": ([r[0], r[1], r[2]] if r is not None else None), "kl": float(e)}
                               for (w, r), e in zip(ED + [(w, None) for w in OD], edge_eff)]}
    # token swaps at "creation"; the top parts' activations; activations set back
    topk = res["parts"][:5]
    res["swaps"] = []
    for kind, words in SWAPS.items():
        for wd in words:
            y = tk.encode(wd)
            y = y if isinstance(y, list) else y.ids
            if len(y) != 1:
                continue
            seq = list(ids)
            seq[t] = y[0]
            lp = lp_of(seq)
            a1 = acts(seq)
            row = {"word": wd, "kind": kind, "p_of": float(lp[OF].exp()), "drop_bits": float((base[OF] - lp[OF]) / LN2),
                   "top": tk.decode([int(lp.argmax())]),
                   "activations": {r["part"]: [float(a0[r["site"]][r["pos"], r["index"]]), float(a1[r["site"]][r["pos"], r["index"]])] for r in topk}}
            for r in topk:  # the part's activation set back to its value on the original text
                lpr = lp_of(seq, patch={r["site"]: [(r["pos"], r["index"], a0[r["site"]][r["pos"], r["index"]])]})
                row.setdefault("restored", {})[r["part"]] = {"p_of": float(lpr[OF].exp()), "recovered_bits": float((lpr[OF] - lp[OF]) / LN2)}
            res["swaps"].append(row)
            print(f"swap {wd!r}: p(of) {row['p_of']:.3f} (top {row['top']!r})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "measure.json").write_text(json.dumps(res, indent=1))
    print("p(of)", p0, "| top parts:", [(r["part"], r["token"], round(r["drop_bits"], 3)) for r in res["parts"][:8]])
    print("groups:", {k: (round(v["p_of"], 3), round(v["drop_bits"], 2)) for k, v in res["groups"].items()})
    print("edit:", res["edit"])
    print("wrote", OUT / "measure.json")


def _real_and_circuit(sel: list) -> dict:
    """p(" of") with the parts sel removed, in the real model and in the circuit's own run."""
    import torch

    import goal_eval
    import mech

    nat, task, ids, t, circ, nodes = _setup()
    OF = mech.tokenizer("vpd4l").encode(" of").ids[0]
    T = len(ids)
    dev = nat.dev
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        nat.vpd.clear()
        p_full = float(torch.softmax(nat.target(x)[0, t].float(), -1)[OF])
        mk = {n: torch.ones((1, T, nat.target.site(n).V.shape[1]), device=dev) for n in nat.names}
        for n, p_, c in sel:
            mk[n][0, p_, c] = 0.0
        dm = {n: torch.ones((1, T), device=dev) for n in nat.names}
        p_real = float(torch.softmax(nat.vpd.masked(x, mk, dm)[0, t].float(), -1)[OF])
    ge = goal_eval.GoalEval(nat)
    g, _, _ = goal_eval.load_answer(STATED / f"{TASK}.json", task, [t], nat)
    E = ge._circuit_engine(ids, [t], [g])
    plan = E.plan(g, semantics="c")
    one = (torch.ones(1, plan["E"], device=dev), torch.ones(1, plan["O"], device=dev))
    pc0 = float(E.graph_log_probs(plan, ew=one[0], ow=one[1])[0, 0, OF].exp())
    pc1 = float(E.graph_log_probs(plan, ew=one[0], ow=one[1], off=[list(sel)])[0, 0, OF].exp())
    return {"real before": p_full, "real after": p_real, "circuit before": pc0, "circuit after": pc1,
            "real drop bits": math.log2(p_full / p_real), "circuit drop bits": math.log2(pc0 / pc1)}


def answer():
    """The answer (module docstring): each sentence a claim with its test, the measured number and whether it passed;
    claims that failed are kept, marked. -> OUT/answer.json"""
    from scipy.stats import spearmanr

    m = json.loads((OUT / "measure.json").read_text())
    gen = json.loads((Path.home() / "mpd-data/graph_oracle/knowledge/generic.json").read_text())
    parts = {r["part"]: r for r in m["parts"]}
    A = np.load(GW / "assoc.npz")
    n_act = A["n_active"]
    off = json.loads(str(A["offsets"]))

    def active_share(lab):
        l_, rest = lab[1:].split(".", 1)
        kind, c = rest.split("#")
        site = {"fc": f"h.{l_}.mlp.c_fc", "down": f"h.{l_}.mlp.down_proj"}.get(kind, f"h.{l_}.attn.{kind}_proj")
        return float(n_act[off[site] + int(c)] / 990720)

    def bits(p0, p1):
        return math.log2(p0 / p1)

    claims = []

    def claim(sentence, test, measured, passed):
        claims.append({"sentence": sentence, "test": test, "measured": measured, "passed": bool(passed)})

    p0 = m["p_of"]
    claim(f"After “The new structure involved the creation” the model gives “ of” probability {p0:.2f}.",
          "the model's next-token distribution", {"p_of": p0, "top": m["top"][:3]}, m["top"][0][0] == " of")
    g = m["groups"]
    at_c = g["the parts at 'creation'"]
    at_t = g["the parts at 'the'"]
    n_c = sum(1 for r in m["parts"] if r["pos"] == m["target"])
    n_t = sum(1 for r in m["parts"] if r["pos"] == m["target"] - 1)
    claim(f"In the full model on this input, removing the circuit-selected {n_c} parts at “creation” takes p(of) to {at_c['p_of']:.4f} "
          f"({at_c['drop_bits']:.1f} bits), while removing its {n_t} parts at “the” changes p(of) to {at_t['p_of']:.2f} "
          f"({at_t['drop_bits']:.2f} bits); the measured log-probability drop at “creation” is over five times as large.", "remove each group of parts together in the real model",
          {"creation": at_c, "the": at_t}, at_c["drop_bits"] > 5 * max(at_t["drop_bits"], 0.01))
    r53 = parts["L0.fc#53"]
    g53 = gen["parts"]["L0.fc#53"]
    claim(f"Removing L0.fc#53 at “creation” from the full model costs {r53['drop_bits']:.1f} bits of log p(of). "
          f"Its archived activation rate is {100 * active_share('L0.fc#53'):.0f}% of the sampled Pile tokens; "
          f"removing it at every position of 24 Pile sequences gives {g53[0]:.1f} nats of output KL "
          f"and a signed top-1 accuracy change of {100 * g53[1]:+.0f} percentage points.",
          "remove the part alone here, and at every position of 24 Pile sequences", {"drop_bits": r53["drop_bits"], "pile KL nats": g53[0], "top-1 change": g53[1]},
          g53[0] > 1.0)
    r = parts["L3.down#2481"]
    claim(f"Removing L3.down#2481 from the full model at its selected position on this input "
          f"drops log2 p(of) by {r['drop_bits']:.2f} bits; this removal does not establish token specificity.",
          "association study; remove the part alone", {"drop_bits": r["drop_bits"], "label": r["label"]}, r["drop_bits"] > 0.3)
    nouns = [s_ for s_ in m["swaps"] if s_["kind"] != "not a noun"]
    act = [s_["activations"]["L3.down#2481"][1] for s_ in nouns]
    lpo = [math.log(max(s_["p_of"], 1e-12)) for s_ in nouns]
    rho = spearmanr(act, lpo).correlation
    keep = [s_ for s_ in nouns if s_["kind"].startswith("takes")]
    rare = [s_ for s_ in nouns if s_["kind"].startswith("a noun")]
    claim(f"In the full model, L3.down#2481's baseline activation is {r['activation']:.1f}. "
          f"For the manually chosen swap group labelled ‘takes of’, its observed activation range is "
          f"{min(s_['activations']['L3.down#2481'][1] for s_ in keep):.1f} to {max(s_['activations']['L3.down#2481'][1] for s_ in keep):.1f} "
          f"and p(of) ranges {min(s_['p_of'] for s_ in keep):.2f}–{max(s_['p_of'] for s_ in keep):.2f}; for the manually chosen group labelled ‘rarely takes of’, "
          f"the activation range is {min(s_['activations']['L3.down#2481'][1] for s_ in rare):.1f} to {max(s_['activations']['L3.down#2481'][1] for s_ in rare):.1f} "
          f"and minimum p(of) is {min(s_['p_of'] for s_ in rare):.2f} (rank correlation of activation with log p(of) over these {len(nouns)} replacements: {rho:+.2f}).",
          "swap the token at “creation”; read the part and p(of); group labels are manual, not an independently tested linguistic classification", {"spearman": rho, "swaps": [(s_["word"], s_["p_of"], s_["activations"]["L3.down#2481"][1]) for s_ in nouns]},
          rho < -0.5)
    rest = [(s_["word"], s_["p_of"], s_["restored"]["L3.down#2481"]["p_of"], bits(p0, s_["p_of"]), s_["restored"]["L3.down#2481"]["recovered_bits"])
            for s_ in rare + [x for x in keep if bits(p0, x["p_of"]) > 0.1]]
    frac = [rb / db for _, _, _, db, rb in rest if db > 0.1]
    claim("In the full model, setting L3.down#2481 back to its value on “creation” recovers a median of at least half the drop over the selected swaps: " +
          ", ".join(f"“{w.strip()}” {a_:.2f} → {b_:.2f}" for w, a_, b_, _, _ in rest if b_ > a_ + 0.05) +
          f" (median {100 * np.median(frac):.0f}% of the drop in bits recovered).",
          "swap, then patch the part's activation to its original value", {"restores": rest}, np.median(frac) >= 0.5)
    r3257 = parts["L0.down#3257"]
    nn = [s_ for s_ in m["swaps"] if s_["kind"] == "not a noun"]
    rec = [(s_["word"], s_["restored"]["L0.down#3257"]["recovered_bits"], s_["activations"]["L0.down#3257"][1]) for s_ in nn]
    claim(f"Removing L0.down#3257 from the full model here drops log2 p(of) by "
          f"{r3257['drop_bits']:.2f} bits; after the sampled non-noun swaps its activation changes from {r3257['activation']:.1f} to "
          f"{min(x[2] for x in rec):.1f}–{max(x[2] for x in rec):.1f}, and setting it back recovers " +
          ", ".join(f"{b_:+.1f} bits after “{w.strip()}”" for w, b_, _ in rec) + ".",
          "swap to a non-noun, then patch the part back", {"restores": rec}, sum(b_ > 1 for _, b_, _ in rec) >= 3)
    r726 = parts["L0.fc#726"]
    claim(f"Removing L0.fc#726 at its selected position from the full model on this input costs {r726['drop_bits']:.2f} bits of log p(of).",
          "remove the part alone", {"drop_bits": r726["drop_bits"]}, r726["drop_bits"] > 0.05)
    t3, t5 = g["the top 3 parts"], g["the top 5 parts"]
    s3, s5 = sum(x["drop_bits"] for x in m["parts"][:3]), sum(x["drop_bits"] for x in m["parts"][:5])
    claim(f"In the full model, joint and summed single-removal drops differ by over one bit for both groups: "
          f"the three strongest parts jointly cost {t3['drop_bits']:.1f} bits and the five strongest "
          f"{t5['drop_bits']:.1f} bits, against {s3:.1f} and {s5:.1f} for their single removals summed.",
          "remove groups together", {"top3": t3, "top5": t5, "sums": [s3, s5]}, abs(t3["drop_bits"] - s3) > 1 and abs(t5["drop_bits"] - s5) > 1)
    e = m["edit"]
    targeted = [(x["site"], x["pos"], x["index"]) for x in m["parts"] if x["part"] in ("L3.down#2481", "L2.down#3049")]
    tg = _real_and_circuit(targeted)
    edit = {"edit": "remove the selected L3.down#2481 and L2.down#3049 parts at their recorded positions",
            "prediction (the circuit's own run)": {"p_of before": tg["circuit before"], "p_of after": tg["circuit after"], "drop bits": tg["circuit drop bits"]},
            "measured (real model)": {"p_of before": tg["real before"], "p_of after": tg["real after"], "drop bits": tg["real drop bits"]},
            "passed": abs(tg["circuit drop bits"] - tg["real drop bits"]) <= 0.5 * max(tg["real drop bits"], 0.2)}
    claim(f"Edit: the circuit predicts that removing L3.down#2481 and L2.down#3049 lowers p(of) from {tg['circuit before']:.2f} to {tg['circuit after']:.2f} "
          f"({tg['circuit drop bits']:.2f} bits); in the real model it goes from {tg['real before']:.2f} to {tg['real after']:.2f} ({tg['real drop bits']:.2f} bits).",
          "predict with the circuit's run, then remove the parts in the real model", edit, edit["passed"])
    claim(f"Edit: the circuit predicts that removing its three strongest parts takes p(of) from {e['circuit p_of']:.2f} to {e['circuit p_of after']:.3f} "
          f"({e['circuit predicted drop_bits']:.1f} bits); the real model falls to {e['real p_of after']:.5f} ({e['real drop_bits']:.1f} bits): "
          f"the pass flag tests whether these log-probability drops differ by at most half the real-model drop, not full-distribution fidelity.",
          "predict with the circuit's run, then remove the parts in the real model", e,
          abs(e["circuit predicted drop_bits"] - e["real drop_bits"]) <= 0.5 * e["real drop_bits"])
    passed = [c_ for c_ in claims if c_["passed"]]
    english = " ".join(c_["sentence"] for c_ in passed if not c_["sentence"].startswith("Edit"))
    out = {"task": TASK, "text": "… The new structure involved the creation", "prediction": " of", "p_of": p0,
           "circuit": {"source": str(STATED / f"{TASK}.json"), "parts": len(m["parts"]), "lines": len(m["render"]["edges"])},
           "english": english, "claims": claims, "edit": edit,
           "failed": [c_["sentence"] for c_ in claims if not c_["passed"]]}
    (OUT / "answer.json").write_text(json.dumps(out, indent=1))
    for c_ in claims:
        print(("PASS " if c_["passed"] else "FAIL ") + c_["sentence"])
    print("wrote", OUT / "answer.json")


def figure():
    """One page: rustmove's circuit drawn by the lead's renderer (parts and connections sized by their removal effects in
    the circuit's run) beside the answer's claims, each marked by whether its intervention bore it out. ->
    FIG/exemplar_text9000.png"""
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    m = json.loads((OUT / "measure.json").read_text())
    ans = json.loads((OUT / "answer.json").read_text())
    (OUT / "render.json").write_text(json.dumps(m["render"]))
    base = OUT / "circuit_text9000"
    subprocess.run([sys.executable, "-B", str(LEAD / "render_circuit.py"), str(OUT / "render.json"), str(base)], check=True)
    img = mpimg.imread(str(base) + ".png")
    sys.path.insert(0, str(HERE))
    import figure as fg

    fg.setup()
    fig = plt.figure(figsize=(26, 15))
    ax = fig.add_axes([0.0, 0.05, 0.56, 0.86])
    ax.imshow(img)
    ax.axis("off")
    tx = fig.add_axes([0.575, 0.03, 0.41, 0.9])
    tx.axis("off")
    y = 1.0
    for c in ans["claims"]:
        mark, col = ("✓", fg.GREEN) if c["passed"] else ("✗", fg.RED)
        lines = textwrap.wrap(c["sentence"], 96)
        tx.text(0.0, y, mark, fontsize=17, color=col, va="top", fontweight="bold", transform=tx.transAxes)
        tx.text(0.035, y, "\n".join(lines), fontsize=13, va="top", color=fg.INK, transform=tx.transAxes, linespacing=1.3)
        y -= 0.0235 * len(lines) + 0.017
    fig.text(0.01, 0.955, "How the model predicts “ of” after “The new structure involved the creation”", fontsize=22, fontweight="bold")
    out = FIG / "exemplar_text9000.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("measure", "answer", "figure"))
    a = ap.parse_args()
    globals()[a.cmd]()
