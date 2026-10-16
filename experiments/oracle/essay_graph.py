"""Essay-style neuron graphs, measured, no agents (the lead's 10-15 section ESSAY-STYLE NEURON GRAPHS): one command from a
prompt to a graph of neuron poles at their positions, the attention routes between positions, the measured edges
between the poles, groups and names, with necessity, sufficiency and the size curve.
  1. patching pairs: the given corrupted prompt, else CF_K single-token counterfactuals of the prompt (at each position
     the clean run's CF_ALTS likeliest tokens there, those that change the answer with the largest gap, one per
     position);
  2. per pair circuit_finder.find: the source neurons, the mover heads at the answer position (where each looks) and
     the readout neurons, each cut's necessity, sufficiency and curve; the graph is their union over the pairs;
  3. edges among the graph's neurons (at most SHOWN readout neurons, by measured share): neuron_edges' measured path
     patching (Edges, prune, graph) in the model itself;
  4. groups: neuron_motifs.mine's communities when --groups names its output, else one group per stage and position;
     names: atlas_labels' verified labels (--labels, LABELS.jsonl) per pole, with their scores;
  5. OUT.json and OUT.png: tokens along the bottom, layers up the side, poles coloured by sign, edges by measured
     strength, heads as arrows from where they look to the answer.

  essay_graph.py OUT --user TEXT [--corrupt-user TEXT] [--text RAW] [--corrupt-text RAW] [--model Qwen/Qwen3-8B]
                 [--adapter A] [--dtype float32] [--labels LABELS.jsonl] [--groups GROUPS.json] [--edges]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))

import circuit_finder as cf  # noqa: E402

CF_K = 3          # automatic patching pairs
CF_ALTS = 4       # tokens tried per position
SHOWN = 12        # readout neurons drawn and given edges (all are in the JSON)


@torch.no_grad()
def counterfactual_pairs(pt: cf.Patcher, ids: list[int], k: int = CF_K) -> list[list[int]]:
    """Single-token counterfactuals of ids that change the clean run's answer, the CF_K with the largest drop of its
    log-probability, one per position (positions 1..T-2)."""
    T = len(ids)
    x = torch.as_tensor(ids, device=pt.dev)[None]
    lp = torch.log_softmax(pt.core(input_ids=x).logits[0].float(), -1)          # [T, V]: each next token
    y = int(lp[-1].argmax())
    tries = []
    for q in range(1, T - 1):
        for z in lp[q - 1].topk(CF_ALTS + 1).indices.tolist():
            if z != ids[q]:
                tries.append((q, ids[:q] + [z] + ids[q + 1:]))
    best = {}
    for s0 in range(0, len(tries), pt.rows):
        part = tries[s0:s0 + pt.rows]
        xs = torch.tensor([t for _, t in part] + [ids] * (pt.rows - len(part)), device=pt.dev)
        out = torch.log_softmax(pt.core(input_ids=xs, logits_to_keep=1).logits[:, -1].float(), -1)[:len(part)]
        for (q, t), o in zip(part, out):
            if int(o.argmax()) != y:
                d = float(lp[-1, y] - o[y])
                if d > best.get(q, (0, None))[0]:
                    best[q] = (d, t)
    return [t for _, (d, t) in sorted(best.items(), key=lambda kv: -kv[1][0])[:k]]


def merge(recs: list[dict]) -> dict:
    """The union of the pairs' circuits: each neuron (layer, pos, index) and mover (layer, head, pos) once, with the
    pairs it came from and its shares there."""
    nodes, routes = {}, {}
    for j, r in enumerate(recs):
        for n in r.get("nodes", []):
            key = (n["layer"], n["pos"], n["neuron"])
            nodes.setdefault(key, {**n, "pairs": []})["pairs"].append([j, n["lost"], n["restored"]])
        for h in r.get("routes", []):
            key = (h["layer"], h["head"], h["pos"])
            routes.setdefault(key, {**h, "pairs": []})["pairs"].append([j, h["lost"], h["restored"]])
    return {"nodes": list(nodes.values()), "routes": list(routes.values())}


def labels_of(path: Path | None) -> dict:
    out = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[(int(r["layer"]), int(r["index"]), 1 if r.get("pole") in ("pos", "+", 1, "1") else -1)] = r
    return out


def groups_of(path: Path | None) -> dict:
    out = {}
    if path and path.exists():
        for g, grp in enumerate(json.loads(path.read_text()).get("groups", [])):
            for k in grp.get("poles", grp.get("keys", [])):
                out[k] = g
    return out


def draw(g: dict, tok, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    toks = [tok.decode([i]) for i in g["ids"]]
    lo = max(0, g["first_diff"] - 2)
    T, L = len(toks), g["layers"]
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 15})
    fig, ax = plt.subplots(figsize=(max(9, 1.05 * (T - lo) + 3), 9), dpi=170)
    shown = [n for n in g["nodes"] if n.get("shown")]
    cell = {}
    for n in sorted(shown, key=lambda n: -max(abs(n["lost"]), abs(n["restored"]))):
        k = cell.setdefault((n["pos"], n["layer"]), [])
        n["_xy"] = (n["pos"] + 0.22 * (len(k) % 3), n["layer"] - 0.45 * (len(k) // 3))      # a cell's poles side by side
        k.append(n)
    at = {(n["layer"], n["pos"], n["neuron"]): n for n in shown}
    for e in g.get("edges", []):
        a, b = e["src_node"], e["dst_node"]
        if a in at and b in at:
            w = abs(e.get("strength") or e.get("local") or 0.0)
            (x0, y0), (x1, y1) = at[a]["_xy"], at[b]["_xy"]
            ax.plot([x0, x1], [y0, y1], color="#9AA3AE", lw=0.5 + 2.5 * min(w, 1.0), alpha=0.7, zorder=1)
    for r in g["routes"]:
        src = r["attends"][0][0] if r["attends"] else r["pos"]
        ax.annotate("", xy=(r["pos"], r["layer"]), xytext=(src, r["layer"]),
                    arrowprops={"arrowstyle": "->", "color": "#2F7DE1", "lw": 2, "connectionstyle": "arc3,rad=0.25"}, zorder=2)
        ax.text(r["pos"] + 0.1, r["layer"] + 0.4, f"L{r['layer']}H{r['head']}", color="#2F7DE1", fontsize=11)
    for n in shown:
        c = "#C8102E" if n["pole"] > 0 else "#1F4E9C"
        share = max(abs(n["lost"]), abs(n["restored"]))
        x, y = n["_xy"]
        ax.scatter([x], [y], s=60 + 500 * min(share, 1.0), color=c, edgecolor="white", lw=1.2, zorder=3)
        if share >= 0.05 or n.get("label"):
            ax.text(x + 0.08, y - 0.15, f"L{n['layer']}:{n['neuron']}{'+' if n['pole'] > 0 else '-'}"
                    + (f" {n['label'][:40]}" if n.get("label") else ""), fontsize=10, color=c, va="top")
    ax.set_xticks(range(lo, T))
    ax.set_xticklabels([repr(t)[1:-1] for t in toks[lo:]], rotation=45, ha="right", fontsize=13)
    ax.set_xlim(lo - 0.6, T - 0.2)
    ax.set_ylim(-1, L)
    ax.set_ylabel("layer")
    cut = g["cuts"][0] if g["cuts"] else {}
    src = cut.get("source", {})
    ax.set_title(f"{g['answer_text']!r} over {g['corrupt_answer_text']!r}\n{sum(n['stage'] == 'source' for n in g['nodes'])} source neurons "
                 f"(necessity {src.get('lost', float('nan')):.2f}, sufficiency {src.get('restored', float('nan')):.2f}), "
                 f"{len(g['routes'])} heads, {sum(n['stage'] == 'readout' for n in g['nodes'])} readout neurons",
                 fontsize=16, fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, help="OUT: OUT.json and OUT.png")
    ap.add_argument("--user", default=None)
    ap.add_argument("--corrupt-user", default=None)
    ap.add_argument("--text", default=None, help="a raw prompt (no chat template)")
    ap.add_argument("--corrupt-text", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rows", type=int, default=cf.ROWS)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--groups", type=Path, default=None)
    ap.add_argument("--edges", action="store_true", help="measure edges among the drawn neurons (neuron_edges)")
    ap.add_argument("--rev", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto" if a.device == "auto" else None)
    model = (model if a.device == "auto" else model.to(a.device)).eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok, rows=a.rows)
    ids = cf.chat(tok, a.user) if a.user else tok(a.text, add_special_tokens=False)["input_ids"]
    if a.corrupt_user or a.corrupt_text:
        pairs = [cf.chat(tok, a.corrupt_user) if a.corrupt_user else tok(a.corrupt_text, add_special_tokens=False)["input_ids"]]
    else:
        pairs = counterfactual_pairs(pt, ids)
    recs = []
    for k in pairs:
        if len(k) != len(ids):
            log("a corrupted prompt of another length: skipped")
            continue
        r = cf.find(pt, ids, k, log=log)
        r["corrupted"] = k
        recs.append(r)
    got = merge(recs)
    labels, groups = labels_of(a.labels), groups_of(a.groups)
    for n in got["nodes"]:
        key = f"{n['layer']}.{n['neuron']}.{'+' if n['pole'] > 0 else '-'}"
        n["group"] = groups.get(key, f"{n['stage']}@{n['pos']}")
        lab = labels.get((n["layer"], n["neuron"], n["pole"]))
        if lab:
            n["label"], n["label_score"] = lab.get("explanation"), lab.get("score")
    readout = sorted((n for n in got["nodes"] if n["stage"] == "readout"), key=lambda n: -max(abs(n["lost"]), abs(n["restored"])))
    for n in got["nodes"]:
        n["shown"] = n["stage"] == "source" or n in readout[:SHOWN]
    first = recs[0] if recs else {}
    g = {"ids": ids, "model": a.model, "adapter": a.adapter, "layers": pt.L, "answer": first.get("answer"),
         "answer_text": tok.decode([first["answer"]]) if recs else None, "corrupt_answer_text":
         tok.decode([first["corrupt_answer"]]) if recs else None, "first_diff": min((r["first_diff"] for r in recs), default=0),
         "pairs": [{"corrupted": r["corrupted"], "gap": r.get("gap"), "met": r.get("met")} for r in recs],
         "cuts": [r.get("cuts", {}) for r in recs], "nodes": got["nodes"], "routes": got["routes"], "finder_rev": a.rev}
    if a.edges and recs and got["nodes"]:
        import neuron_edges as ne

        shown = [(n["layer"], n["pos"], n["neuron"]) for n in got["nodes"] if n["shown"]]
        ed = ne.Edges(model, ids, len(ids) - 1, first.get("answer_tokens", [first["answer"]]), first["alternatives"], shown,
                      background="model")
        eg = ne.graph(ed, ne.prune(ed, log=log), tok)
        idn = {n["id"]: (n["layer"], n["pos"], n["neuron"]) for n in eg["nodes"]}
        g["edges"] = [{**e, "src_node": idn[e["src"]], "dst_node": idn[e["dst"]]} for e in eg["edges"]]
    g["seconds"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".json").write_text(json.dumps(g, default=str))
    if recs:
        draw(g, tok, Path(str(a.out) + ".png"))
    log(f"graph: {len(got['nodes'])} neurons, {len(got['routes'])} heads, {len(g.get('edges', []))} edges, {g['seconds']} s")


if __name__ == "__main__":
    main()
