"""Neuron groups as labelled units (the lead's 10-15, after ADAG: graphs and the oracle show groups first, and there are
far fewer groups than poles). A group is a set of neuron poles (inframove's neuron_motifs groups, keys "L.n.+" /
"L.n.-"). Its activation at a token is the mean over its poles of the pole's value in its own sign as a share of that
neuron's corpus maximum for the sign (the atlas's), a value of the other sign read as 0; zeroing a group zeroes every
member pole at every position (neuron_builder.pole_edit). A group gets the two labels a pole gets, each measured on
the whole group and tested on fresh texts (atlas_labels, neuron_verify).

  neuron_groups.py prepare --groups VIEWS.json --atlas ROOT --out GROUPS.jsonl
      neuron_motifs views' groups (two_views; "effect": the members' mean single-zeroing change of log p(answer), below 0
      the group supports the answer; "name": [...] or not[...]) with each member's corpus maximum from the atlas, one line
      a group {"id", "members": [[l, i, s, max]], "effect", "name"}
  neuron_groups.py collect --groups GROUPS.jsonl --corpus DIR --model M --out TOP.json [--tokens 5000000] [--top 64]
      where the model runs (a pod): one pass over the shared corpus's first tokens, each group's corpus maximum and its
      top examples (document, position, value), the first token of each document (the attention sink) left out
  neuron_groups.py label --top TOP.json --corpus DIR --measure-url U --out LABELS.jsonl
      each group's input label (atlas_labels' revise method on the group's examples), output label (what zeroing the
      group does, tested on held-out contexts) and scores on fresh texts at both blind lengths; the target behind U
      must run neuron_verify with group support
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "rl"), str(HERE.parents[1] / "experiments/oracle")]
KEY = re.compile(r"^L?(\d+)\.N?(\d+)\.?([+-])$")
LONG, SHORT, AFTER = 256, 64, 32   # tokens before a group's first `long` examples, before the rest, after each

HEAD = """Below are the strongest activations of a group of MLP neurons in a language model that work together ({members}), over a large mixed corpus. The group's activation at a token is the mean over its neurons of each one's value in its own sign as a share of that neuron's largest value (1.0: every member at its own maximum). Each example is the text before the token, the token itself in ⟦ ⟧, and a few tokens after, with the group's activation.

Examples (the group's corpus maximum {gmax:.2f}):
{examples}"""
EXPLAIN = HEAD + """

Explain what in the text makes this group fire, so that a reader who sees only your explanation can write a new text that makes it fire. Say (1) the exact token or tokens where it fires; (2) the context that must surround them: the kind of document, the topic, the phrase or structure just before, and how far back it reaches; (3) what the examples share that a surface reading misses. Prefer one specific shared cause over a list of unrelated tokens. Write JSON with one field, "explanation", 1 to 4 sentences, nothing else."""
REVISE = HEAD + """

Your explanation of what makes the group fire was:
{explanation}

A reader who saw only that explanation wrote these texts to make the group fire as strongly as possible. After each: the largest group activation it reached at any token as a share of the group's corpus maximum (a pass needs one text at 0.30 or more), and that token:
{blind}
It also wrote these near misses, texts on close subjects the explanation says should not make the group fire (each should stay below 0.30):
{near}

Revise the explanation so that texts written from it reach the group's maximum and close subjects stay below 0.30. Keep what the examples support; change what the reader's results show is wrong or missing. Write JSON with one field, "explanation", 1 to 4 sentences, nothing else."""
OUTPUT = """A group of MLP neurons in a language model ({members}) fires strongly at the marked token ⟦ ⟧ in each context below. For each context: the model's likeliest next tokens, and what zeroing every neuron of the group (each in its own sign) at every position does to its next-token prediction: the tokens whose probability falls most and rises most (probability before -> after) and the total change (half the summed absolute change, 0 to 1).

{blocks}

Describe what this group does to the model's next-token prediction: which tokens or kinds of tokens it makes more likely or less likely, and in which contexts. If the change is small everywhere, say so. Write JSON with one field, "output", 1 to 3 sentences, nothing else."""


def members_of(keys: list[str]) -> list[tuple[int, int, int]]:
    """neuron_motifs keys ("18.2201.+") or unit names ("L18.N2201+") as (layer, index, sign); heads are skipped."""
    out = []
    for k in keys:
        m = KEY.match(str(k).strip())
        if m:
            out.append((int(m[1]), int(m[2]), 1 if m[3] == "+" else -1))
    return out


def prepare(a) -> None:
    sys.path.insert(0, str(HERE.parents[1] / "experiments/oracle"))
    from atlas import Atlas

    at = Atlas(a.atlas)
    doc = json.loads(a.groups.read_text())
    groups = doc["two_views"]["groups"] if "two_views" in doc else doc["groups"] if isinstance(doc, dict) else doc
    rows = []
    for g, grp in enumerate(groups):
        keys = grp.get("keys", grp.get("members")) if isinstance(grp, dict) else grp
        extra = {k: grp[k] for k in ("effect", "name") if isinstance(grp, dict) and k in grp}
        ms = []
        for l, i, s in members_of(keys):
            mx = abs(at.neuron(l, i, k=0, text=False)["max"]["+" if s > 0 else "-"])
            if mx > 0:
                ms.append([l, i, s, mx])
        if ms:
            rows.append({"id": f"G{g}", "members": ms, "model": at.meta["model"], **extra})
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} groups, {sum(len(r['members']) for r in rows)} member poles to {a.out}")


def hooks(model, groups: list[dict], store: dict):
    """Forward pre-hooks on each member layer's down_proj: store[("g", gi)] gets the group's summed normalized member
    values [B, T] (divide by the member count for the activation)."""
    import torch

    by_layer: dict = {}
    for gi, g in enumerate(groups):
        for l, i, s, mx in g["members"]:
            by_layer.setdefault(l, []).append((gi, i, s, mx))
    hs = []
    for l, ms in by_layer.items():
        mlp = model.model.layers[l].mlp
        dev = mlp.down_proj.weight.device
        idx = torch.tensor([i for _, i, _, _ in ms], device=dev)
        scale = torch.tensor([s / mx for _, _, s, mx in ms], device=dev)
        gid = [gi for gi, _, _, _ in ms]

        def pre(_m, a, idx=idx, scale=scale, gid=gid):
            v = (a[0][..., idx].float() * scale).clamp_min(0)          # [B, T, k]: each member in its own sign
            for j, gi in enumerate(gid):
                cur = store.get(gi)
                x = v[..., j]
                store[gi] = x if cur is None else cur + x.to(cur.device)
        hs.append(mlp.down_proj.register_forward_pre_hook(pre))
    return hs


def collect(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    groups = [json.loads(x) for x in a.groups.read_text().splitlines() if x.strip()]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    toks, off = np.load(a.corpus / "tokens.npy", mmap_mode="r"), np.load(a.corpus / "offsets.npy")
    nd = int(np.searchsorted(off, a.tokens, side="left"))
    heaps = [[] for _ in groups]
    store: dict = {}
    hs = hooks(m, groups, store)
    n = np.array([len(g["members"]) for g in groups], dtype=np.float32)
    seen = 0
    try:
        for s in range(0, nd, a.batch):
            docs = list(range(s, min(s + a.batch, nd)))
            seqs = [np.asarray(toks[off[d]:off[d + 1]]).tolist() for d in docs]
            T = max(len(x) for x in seqs)
            ids = torch.tensor([x + [0] * (T - len(x)) for x in seqs], device=dev)
            mask = torch.tensor([[1] * len(x) + [0] * (T - len(x)) for x in seqs], device=dev)
            store.clear()
            with torch.no_grad():
                m.model(input_ids=ids, attention_mask=mask, use_cache=False)
            valid = mask.bool().clone()
            valid[:, 0] = False
            for gi in range(len(groups)):
                act = (store[gi] / float(n[gi])).masked_fill(~valid.to(store[gi].device), 0).flatten()
                top = act.topk(min(a.top, act.numel()))
                h = heaps[gi]
                for v, j in zip(top.values.tolist(), top.indices.tolist()):
                    if v <= 0 or (len(h) == a.top and v <= h[0][0]):
                        break
                    b, p = divmod(j, T)
                    (heapq.heappush if len(h) < a.top else heapq.heapreplace)(h, (v, docs[b], p))
            seen += int(mask.sum())
            if s // a.batch % 50 == 0:
                print(f"{seen / 1e6:.1f}M tokens", flush=True)
    finally:
        for h in hs:
            h.remove()
    out = [{**g, "max": max((v for v, _, _ in heaps[gi]), default=0.0),
            "top": [{"value": v, "doc": d, "pos": p} for v, d, p in sorted(heaps[gi], reverse=True)]} for gi, g in enumerate(groups)]
    a.out.write_text(json.dumps({"model": a.model, "tokens": seen, "documents": nd, "groups": out}))
    print(f"{len(out)} groups over {seen} tokens to {a.out}")


def unit_names(members) -> str:
    return ", ".join(f"L{l}.N{i}{'+' if sg > 0 else '-'}" for l, i, sg, _ in members)


def windows(g: dict, toks, off, k: int) -> list[dict]:
    """A collected group's top k examples as token windows {value, ids, at} from the corpus."""
    out = []
    for t in g["top"][:k]:
        s0, e0 = int(off[t["doc"]]), int(off[t["doc"] + 1])
        g0 = s0 + t["pos"]
        lo = max(s0, g0 - LONG)
        out.append({"value": t["value"], "ids": np.asarray(toks[lo:min(e0, g0 + AFTER + 1)]).tolist(), "at": g0 - lo})
    return out


def block(tok, exs, long: int) -> str:
    lines = []
    for j, e in enumerate(exs, 1):
        ids, t = e["ids"], e["at"]
        lines.append(f"{j}. {e['value']:.2f} ...{tok.decode(ids[max(0, t - (LONG if j <= long else SHORT)):t])}⟦{tok.decode(ids[t:t + 1])}⟧"
                     f"{tok.decode(ids[t + 1:t + 1 + AFTER])}")
    return "\n".join(lines)


def item(g: dict, explanation: str | None) -> dict:
    return {"group": g["id"], "members": g["members"], "pole": "+", "corpus_max": g["max"], "explanation": explanation}


def label(a) -> None:
    """Each collected group's input label by atlas_labels' revise method (the writer sees `examples` group examples,
    revises up to `rounds` times from its reader's development texts; the best development round is the label), then
    its output label (what zeroing the group does on its top contexts, tested on held-out contexts 31-40) and its scores
    on fresh texts at both blind lengths: one labels file a run, one line a group."""
    import atlas_labels as al
    import neuron_verify
    import reader_reward
    from transformers import AutoTokenizer

    doc = json.loads(a.top.read_text())
    tok = AutoTokenizer.from_pretrained(doc["model"])
    toks, off = np.load(a.corpus / "tokens.npy", mmap_mode="r"), np.load(a.corpus / "offsets.npy")
    groups = [g for g in doc["groups"] if g["max"] > 0 and len(g["top"]) > max(al.OUT_TEST)]
    wr, rd = reader_reward.Reader(a.writer, a.out.parent), reader_reward.Reader(neuron_verify.READER, a.out.parent)
    measure = lambda rows, seed=0: neuron_verify.remote_measure(a.measure_url, rows, seed)  # noqa: E731
    exs = {g["id"]: windows(g, toks, off, max(a.examples, max(al.OUT_TEST) + 1)) for g in groups}
    head = lambda g: dict(members=unit_names(g["members"]), gmax=g["max"])  # noqa: E731
    texts = wr([EXPLAIN.format(**head(g), examples=block(tok, exs[g["id"]][:a.examples], a.long)) for g in groups], max_tokens=500)
    st = {g["id"]: {"g": g, "explanation": (neuron_verify.parse_json(t) or {}).get("explanation"), "rounds": [], "raw": t} for g, t in zip(groups, texts)}
    live = [v for v in st.values() if isinstance(v["explanation"], str) and v["explanation"].strip()]
    for rnd in range(a.rounds + 1):
        dev = measure(neuron_verify.plan(rd, [item(v["g"], v["explanation"]) for v in live]))
        for v, d in zip(live, dev, strict=True):
            v["rounds"].append({"explanation": v["explanation"], "valid": d.get("valid"), "blind_max": d.get("blind_max"),
                                "near_miss_rate": d.get("near_miss_rate"), "passes_t1": d.get("passes_t1"),
                                "blind_texts": d.get("blind_texts"), "blind_values": d.get("blind_values")})
            v["dev"] = d
        fail = [v for v in live if v["dev"].get("valid") and not v["dev"].get("passes_t1")]
        print(f"round {rnd}: {len(live) - len(fail)}/{len(live)} groups pass on development texts; writer ${wr.spend.total:.3f}, reader ${rd.spend.total:.3f}", flush=True)
        if rnd == a.rounds or not fail:
            break
        asks = []
        for v in fail:
            blind, near = al.feedback(tok, {**v["dev"], "pole": "+"})
            asks.append(REVISE.format(**head(v["g"]), examples=block(tok, exs[v["g"]["id"]][:a.examples], a.long),
                                      explanation=v["explanation"], blind=blind, near=near))
        for v, t in zip(fail, wr(asks, max_tokens=500), strict=True):
            e = (neuron_verify.parse_json(t) or {}).get("explanation")
            if isinstance(e, str) and e.strip():
                v["explanation"] = e
        live = fail
    rows = []
    for v in st.values():
        if v["rounds"]:
            best = max(range(len(v["rounds"])), key=lambda i: al.dev_key(v["rounds"][i]))
            v["explanation"], v["chosen_round"] = v["rounds"][best]["explanation"], best
        g = v["g"]
        rows.append({"model": doc["model"], "group": g["id"], "members": g["members"], "effect": g.get("effect"), "corpus_max": g["max"],
                     "explanation": v["explanation"], "rounds": v["rounds"], "chosen_round": v.get("chosen_round"),
                     "writer_valid": bool(v["explanation"]), "writer_model": a.writer, "writer_raw_response": v["raw"]})
    # output labels: what zeroing the group does on its top contexts, tested on held-out contexts
    effs = neuron_verify.remote_effects(a.measure_url, [{"contexts": [e["ids"][:e["at"] + 1] for e in exs[r["group"]][:max(al.OUT_TEST) + 1]],
                                                         "members": r["members"]} for r in rows])
    gs = {g["id"]: g for g in groups}
    outs = wr([OUTPUT.format(members=unit_names(r["members"]), blocks="\n".join(
        al.effect_block(tok, j + 1, exs[r["group"]][i], eff[i]) for j, i in enumerate(al.OUT_DEV))) for r, eff in zip(rows, effs)], max_tokens=400)
    reads, where = [], []
    for k, (r, o) in enumerate(zip(rows, outs)):
        r["output"] = (neuron_verify.parse_json(o) or {}).get("output")
        for i in al.OUT_TEST:
            if isinstance(r["output"], str) and r["output"].strip():
                b, t = al.context_text(tok, exs[r["group"]][i])
                reads.append(al.READ_OUT.replace("one side of a neuron", "a group of neurons").replace("the neuron's side", "the group")
                             .replace("that side of the neuron", "that group").format(output=r["output"], before=b, token=t))
                where.append((k, i))
    got = {}
    for (k, i), x in zip(where, rd(reads, max_tokens=80)):
        got.setdefault(k, {})[i] = (neuron_verify.parse_json(x) or {}).get("tokens")
    for k, (r, eff) in enumerate(zip(rows, effs)):
        lab = sum(al.hits(got.get(k, {}).get(i), eff[i]) for i in al.OUT_TEST)
        base = sum(al.hits([w for w, _ in eff[i]["top"][:3]], eff[i]) for i in al.OUT_TEST)
        n = len(al.OUT_TEST)
        r["output_scores"] = {"contexts": n, "hits": lab, "baseline_hits": base, "pass": bool(r.get("output")) and lab >= n / 2 and lab > base,
                              "mean_tv": sum(eff[i]["tv"] for i in al.OUT_TEST) / n}
        r["name"] = ("not" if (r.get("effect") or 0) > 0 else "") + f"[{r['explanation']}]"   # inframove's sign; the lead decides
    # input scores on fresh texts at both lengths
    todo = [r for r in rows if r["writer_valid"]]
    for length in ("short", "long"):
        got = neuron_verify.judge(rd, measure(neuron_verify.plan(rd, [item(gs[r["group"]], r["explanation"]) for r in todo], length=length,
                                                                 temperature=0.7, seed=a.seed), a.seed))
        for r, g2 in zip(todo, got, strict=True):
            r.setdefault("scores", {})[length] = {"seed": a.seed, "valid": g2.get("valid"), "pass": bool(g2.get("valid") and g2.get("passes_t1")),
                                                 **{k: g2.get(k) for k in ("blind_max", "near_miss_rate", "distant_rate", "steer_match")}}
    a.out.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    for length in ("short", "long"):
        print(f"{length}: {sum(r.get('scores', {}).get(length, {}).get('pass', False) for r in rows)}/{len(rows)} groups pass on fresh texts")
    print(f"output labels: {sum(r['output_scores']['pass'] for r in rows)}/{len(rows)} pass; writer ${wr.spend.total:.3f}, reader ${rd.spend.total:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--groups", type=Path, required=True, help="neuron_motifs groups: {\"groups\": [[key, ...], ...]} or a list")
    p.add_argument("--atlas", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--groups", type=Path, required=True)
    c.add_argument("--corpus", type=Path, required=True, help="a corpus directory: tokens.npy, offsets.npy (atlas.py corpus)")
    c.add_argument("--model", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--tokens", type=int, default=5_000_000)
    c.add_argument("--top", type=int, default=64)
    c.add_argument("--batch", type=int, default=8, help="documents a forward pass")
    lb = sub.add_parser("label")
    lb.add_argument("--top", type=Path, required=True, help="collect's output")
    lb.add_argument("--corpus", type=Path, required=True)
    lb.add_argument("--out", type=Path, required=True)
    lb.add_argument("--measure-url", required=True, help="neuron_verify serve over a tunnel, the same model")
    lb.add_argument("--writer", default="Qwen/Qwen3.5-397B-A17B")
    lb.add_argument("--examples", type=int, default=30)
    lb.add_argument("--long", type=int, default=10)
    lb.add_argument("--rounds", type=int, default=2)
    lb.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    {"prepare": prepare, "collect": collect, "label": label}[a.what](a)


if __name__ == "__main__":
    main()
