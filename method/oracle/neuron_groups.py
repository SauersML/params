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
  neuron_groups.py estimate --top TOP.json --corpus DIR [label's options]   the Tinker cost of label, before any call
  neuron_groups.py label --top TOP.json --corpus DIR --measure-url U --out LABELS.jsonl [--before 256] [--ids G1,G2]
      each group's input label (atlas_labels' revise method on the group's examples, plus one round aimed at near
      misses), output label (what zeroing the group does, relative changes, tested on held-out contexts) for groups that
      move the answer, and scores on fresh texts at both blind lengths; each step's spend logged to SPEND as it ends
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

Explain what in the text makes this group fire, so that a reader who sees only your explanation can write a new text that makes it fire. Say (1) the exact token or tokens where it fires; (2) the context that must surround them: the kind of document, the topic, the phrase or structure just before, and how far back it reaches; (3) what the examples share that a surface reading misses. Prefer one specific shared cause over a list of unrelated tokens. Write JSON with three fields, nothing else: "explanation", 1 to 4 sentences; "name", the shared cause in 2 to 5 words; "one_thing", true when the examples share one specific cause and false when they mix several unrelated ones."""
REVISE = HEAD + """

Your explanation of what makes the group fire was:
{explanation}

A reader who saw only that explanation wrote these texts to make the group fire as strongly as possible. After each: the largest group activation it reached at any token as a share of the group's corpus maximum (a pass needs one text at {fire:.2f} or more), and that token:
{blind}
It also wrote these near misses, texts on close subjects the explanation says should not make the group fire (each should stay below {fire:.2f}):
{near}

Revise the explanation so that texts written from it reach the group's maximum and close subjects stay below {fire:.2f}. Keep what the examples support; change what the reader's results show is wrong or missing. Write JSON with three fields, nothing else: "explanation", 1 to 4 sentences; "name", the shared cause in 2 to 5 words; "one_thing", true when the examples share one specific cause and false when they mix several unrelated ones."""
NEAR = HEAD + """

Your explanation of what makes the group fire was:
{explanation}

Texts written from it make the group fire (the best reached {best:.2f} of the group's maximum), but these near misses, texts on close subjects the explanation says should not make it fire, fired as well (activation as a share of the group's maximum, and the token):
{near}

Narrow the explanation so that texts like these near misses would not make the group fire, keeping what makes the strong texts fire. Write JSON with three fields, nothing else: "explanation", 1 to 4 sentences; "name", the shared cause in 2 to 5 words; "one_thing", true when the examples share one specific cause and false when they mix several unrelated ones."""
OUTPUT = """A group of MLP neurons in a language model ({members}) fires strongly at the marked token ⟦ ⟧ in each context below. For each context: the model's likeliest next tokens, and what zeroing every neuron of the group (each in its own sign) at every position does to its next-token prediction: the tokens whose probability falls and rises most relative to their probability before (before -> after, among tokens at 0.001 or more before) and the total change (half the summed absolute change, 0 to 1).

{blocks}

Describe what this group does to the model's next-token prediction: which tokens or kinds of tokens it specifically makes more likely or less likely, and in which contexts. If the change is small everywhere, say so. Write JSON with one field, "output", 1 to 3 sentences, nothing else."""
READ_OUT = """A description of what a group of neurons in a language model does to the model's next-token prediction:
{output}

The model is reading this text; the group fires strongly at the last token, marked ⟦ ⟧:
...{before}⟦{token}⟧

If the group were zeroed, which next tokens would lose the most probability relative to their probability now? Write JSON with one field, "tokens": a list of 3 tokens exactly as they would appear (with a leading space if the token starts a new word), nothing else."""
OUT_DEV, OUT_TEST, OUT_HIT = range(0, 10), range(30, 40), 10   # writer's contexts, held-out contexts, measured falls a hit may name
SPEND = Path("/Users/user/mpd-data/graph_oracle/components/tinker_spend.tsv")   # the team's Tinker log, a row a step


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
    moved = {}  # neuron_motifs apply: each group's zeroing drop of the answer's contrast (bits) on each record
    for f in sorted(a.apply.iterdir()) if a.apply else []:
        for x in json.loads(f.read_text()).get("groups", []):
            moved[x["id"]] = max(moved.get(x["id"], 0.0), abs(x["effect"]))
    rows = []
    for g, grp in enumerate(groups):
        keys = grp.get("keys", grp.get("members")) if isinstance(grp, dict) else grp
        if len(members_of(keys)) < a.min_size:
            continue
        extra = {k: grp[k] for k in ("effect", "name") if isinstance(grp, dict) and k in grp}
        if g in moved:
            extra["answer_effect"] = moved[g]
        ms = []
        for l, i, s in members_of(keys):
            mx = abs(at.neuron(l, i, k=0, text=False)["max"]["+" if s > 0 else "-"])
            if mx > 0:
                ms.append([l, i, s, mx])
        if ms:
            rows.append({"id": f"G{g}", "members": ms, "model": at.meta["model"], **extra})
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} groups, {sum(len(r['members']) for r in rows)} member poles to {a.out}")


HBINS, HMAX = 3000, 1.5   # collect's histogram of a group's activation: bins of 0.0005 up to 1.5
QS = (0.99, 0.999, 0.9999, 0.99997)
FIRE_Q = 0.99997  # a group fires beyond this quantile of its corpus activation (or 0.3 of its maximum, the larger): single
# neurons pass 0.3 of their maximum on a median 2.9e-5 of corpus tokens (the 8B arm's 400 poles, atlas rates), so groups
# and neurons count as firing on the same share of text


def quantiles(h) -> dict:
    """The activation at each of QS from a collect histogram (upper bin edges)."""
    c = np.cumsum(h) / max(int(np.sum(h)), 1)
    return {str(q): float((int(np.searchsorted(c, q)) + 1) * HMAX / HBINS) for q in QS}


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
    hist = np.zeros((len(groups), HBINS + 1), dtype=np.int64)   # activation in [0, HMAX) in HBINS bins, the last >= HMAX
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
                v = act[valid.to(act.device).flatten()]
                hist[gi] += torch.bincount((v / HMAX * HBINS).long().clamp(0, HBINS), minlength=HBINS + 1).cpu().numpy()
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
    out = [{**g, "max": max((v for v, _, _ in heaps[gi]), default=0.0), "hist": hist[gi].tolist(), "quantiles": quantiles(hist[gi]),
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


def fire_level(g: dict) -> float:
    """The share of the group's maximum a text must reach to count as firing: the FIRE_Q quantile of the group's
    corpus activation (collect's histogram) over its maximum, at least neuron_verify's FIRE (a mean over members is flat
    at the top: the 64th best of 5M tokens sits at 0.59 of the maximum on the 8B dev groups, so 0.3 of the maximum alone
    was met by near misses)."""
    import neuron_verify

    if g.get("hist") and g["max"] > 0:
        h = np.asarray(g["hist"])
        c = np.cumsum(h) / max(int(h.sum()), 1)
        q = (int(np.searchsorted(c, FIRE_Q)) + 1) * HMAX / HBINS
        return max(neuron_verify.FIRE, min(q / g["max"], 0.95))
    return neuron_verify.FIRE


def item(g: dict, explanation: str | None) -> dict:
    return {"group": g["id"], "members": g["members"], "pole": "+", "corpus_max": g["max"], "explanation": explanation, "fire": fire_level(g)}


def context_text(tok, e: dict, before: int = SHORT) -> tuple[str, str]:
    ids, t = e["ids"], e["at"]
    return tok.decode(ids[max(0, t - before):t]), tok.decode(ids[t:t + 1])


def effect_block(tok, j: int, e: dict, eff: dict) -> str:
    b, t = context_text(tok, e)
    f = lambda xs: ", ".join(f"{w!r} {p0:.3f}->{p1:.3f}" for w, p0, p1 in xs[:5])  # noqa: E731
    return (f"{j}. ...{b}⟦{t}⟧\n   likeliest next: " + ", ".join(f"{w!r} {p:.2f}" for w, p in eff["top"][:5]) +
            f"\n   zeroed: falls {f(eff['falls'])}; rises {f(eff['rises'])}; total change {eff['tv']:.2f}")


def promoted(eff: dict, k: int = 3) -> set:
    """Tokens in the top k after zeroing that were not in the top k before (needs effects' top_after): what a group
    pushes against. The unedited model's own top k can never name one, so "supports the next token" scores 0 here."""
    norm = lambda w: str(w).strip().casefold()  # noqa: E731
    return {norm(w) for w, _ in eff.get("top_after", [])[:k]} - {norm(w) for w, _ in eff["top"][:k]}


def hits_promoted(predicted, eff: dict) -> bool:
    norm = lambda w: str(w).strip().casefold()  # noqa: E731
    return bool(promoted(eff) & {norm(w) for w in (predicted or [])[:3]})


def hits(predicted, eff: dict) -> bool:
    norm = lambda w: str(w).strip().casefold()  # noqa: E731
    falls = {norm(w) for w, p0, p1 in eff["falls"][:OUT_HIT] if p1 < p0}
    return any(norm(w) in falls for w in (predicted or [])[:3])


def log_spend(stage: str, usd: float, note: str) -> None:
    with SPEND.open("a") as f:
        f.write(f"{stage}\t{usd:.3f}\t{note}\n")


def load(a):
    import atlas_labels as al  # noqa: F401  (feedback, dev_key)
    from transformers import AutoTokenizer

    doc = json.loads(a.top.read_text())
    tok = AutoTokenizer.from_pretrained(doc["model"])
    toks, off = np.load(a.corpus / "tokens.npy", mmap_mode="r"), np.load(a.corpus / "offsets.npy")
    groups = [g for g in doc["groups"] if g["max"] > 0 and len(g["top"]) > max(OUT_TEST)]
    if a.ids:
        want = set(a.ids.split(","))
        groups = [g for g in groups if g["id"] in want]
    groups = groups[:a.n] if a.n else groups
    exs = {g["id"]: windows(g, toks, off, max(a.examples, max(OUT_TEST) + 1)) for g in groups}
    return doc, tok, groups, exs


def label(a) -> None:
    """Each collected group's labels, every step's Tinker spend logged as it ends:
    1. input label: atlas_labels' revise method on the group's examples (the first `long` with --before tokens before),
       up to `rounds` revisions from the reader's development texts, then one round aimed only at near misses that fired
       (NEAR; kept when the development result does not get worse); the best development round is the label;
    2. output label, for groups whose zeroing moves the answer by at least --effect-floor (|effect|, neuron_motifs'):
       what zeroing every member does on its top contexts OUT_DEV, falls and rises ranked by relative change, described
       by the writer; tested on held-out contexts OUT_TEST, where the reader names 3 tokens and a hit is one among the
       OUT_HIT largest relative falls; baseline the unedited model's own top 3; passes at half the contexts and above
       the baseline;
    3. input scores on fresh texts (temperature 0.7, seed) at both blind lengths.
    name: not[...] for a group with effect > 0 (zeroing raises the answer: the group pushes against it)."""
    import atlas_labels as al
    import neuron_verify
    import reader_reward

    global LONG
    LONG = a.before
    doc, tok, groups, exs = load(a)
    wr, rd = reader_reward.Reader(a.writer, a.out.parent), reader_reward.Reader(neuron_verify.READER, a.out.parent)
    spent = lambda: wr.spend.total + rd.spend.total  # noqa: E731
    measure = lambda rows, seed=0: neuron_verify.remote_measure(a.measure_url, rows, seed)  # noqa: E731
    head = lambda g: dict(members=unit_names(g["members"]), gmax=g["max"])  # noqa: E731
    ev = lambda g: block(tok, exs[g["id"]][:a.examples], a.long)  # noqa: E731
    tag = f"varteach groups {a.out.stem} {doc['model']}"
    # 1. input labels
    texts = wr([EXPLAIN.format(**head(g), examples=ev(g)) for g in groups], max_tokens=500)
    st = {}
    for g, t in zip(groups, texts):
        js = neuron_verify.parse_json(t) or {}
        st[g["id"]] = {"g": g, "explanation": js.get("explanation"), "short": js.get("name"), "one_thing": js.get("one_thing"), "rounds": [], "raw": t}
    live = [v for v in st.values() if isinstance(v["explanation"], str) and v["explanation"].strip()]

    def develop(vs, kind):
        dev = measure(neuron_verify.plan(rd, [item(v["g"], v["explanation"]) for v in vs]))
        for v, d in zip(vs, dev, strict=True):
            v["rounds"].append({"kind": kind, "explanation": v["explanation"], "short": v.get("short"), "one_thing": v.get("one_thing"),
                                "valid": d.get("valid"), "blind_max": d.get("blind_max"),
                                "near_miss_rate": d.get("near_miss_rate"), "passes_t1": d.get("passes_t1"),
                                "blind_texts": d.get("blind_texts"), "blind_values": d.get("blind_values")})
            v["dev"] = d
            v.setdefault("devs", []).append(d)

    for rnd in range(a.rounds + 1):
        develop(live, "revise" if rnd else "first")
        fail = [v for v in live if v["dev"].get("valid") and not v["dev"].get("passes_t1")]
        print(f"round {rnd}: {sum(bool(v['dev'].get('passes_t1')) for v in live)}/{len(live)} groups pass on development texts; ${spent():.3f}", flush=True)
        if rnd == a.rounds or not fail:
            break
        asks = []
        for v in fail:
            blind, near = al.feedback(tok, {**v["dev"], "pole": "+"})
            asks.append(REVISE.format(**head(v["g"]), examples=ev(v["g"]), explanation=v["explanation"], blind=blind, near=near, fire=fire_level(v["g"])))
        for v, t in zip(fail, wr(asks, max_tokens=500), strict=True):
            js = neuron_verify.parse_json(t) or {}
            if isinstance(js.get("explanation"), str) and js["explanation"].strip():
                v["explanation"], v["short"], v["one_thing"] = js["explanation"], js.get("name"), js.get("one_thing")
        live = fail
    for v in st.values():  # the best development round so far, then one round aimed at near misses that fired
        if v["rounds"]:
            i = max(range(len(v["rounds"])), key=lambda j: al.dev_key(v["rounds"][j]))
            v["explanation"], v["dev_best"], v["dev_row"] = v["rounds"][i]["explanation"], v["rounds"][i], v["devs"][i]
            v["short"], v["one_thing"] = v["rounds"][i].get("short"), v["rounds"][i].get("one_thing")
    noisy = [v for v in st.values() if v.get("dev_best") and v["dev_best"].get("valid") and (v["dev_best"]["near_miss_rate"] or 0) > 0]
    if noisy and a.near_round:
        asks = []
        for v in noisy:
            _, near = al.feedback(tok, {**v["dev_row"], "pole": "+"})
            asks.append(NEAR.format(**head(v["g"]), examples=ev(v["g"]), explanation=v["explanation"], best=v["dev_best"]["blind_max"] or 0, near=near))
        old = {id(v): v["explanation"] for v in noisy}
        for v, t in zip(noisy, wr(asks, max_tokens=500), strict=True):
            js = neuron_verify.parse_json(t) or {}
            if isinstance(js.get("explanation"), str) and js["explanation"].strip():
                v["explanation"], v["short"], v["one_thing"] = js["explanation"], js.get("name"), js.get("one_thing")
        develop(noisy, "near")
        for v in noisy:
            if al.dev_key(v["rounds"][-1]) < al.dev_key(v["dev_best"]):
                v["explanation"], v["short"], v["one_thing"] = old[id(v)], v["dev_best"].get("short"), v["dev_best"].get("one_thing")
            else:
                v["dev_best"] = v["rounds"][-1]
        print(f"near-miss round: {len(noisy)} groups; {sum(v['dev_best'] is v['rounds'][-1] for v in noisy)} narrowed; ${spent():.3f}", flush=True)
    rows = [{"model": doc["model"], "group": v["g"]["id"], "members": v["g"]["members"], "effect": v["g"].get("effect"),
             "answer_effect": v["g"].get("answer_effect"), "views_name": v["g"].get("name"), "short": v.get("short"),
             "one_thing": v.get("one_thing"),
             "corpus_max": v["g"]["max"], "explanation": v["explanation"], "rounds": v["rounds"], "writer_valid": bool(v["explanation"]),
             "writer_model": a.writer, "context_before": a.before, "writer_raw_response": v["raw"]} for v in st.values()]
    for r in rows:  # not[...] as neuron_motifs names the group (a group that pushes against the answer)
        neg = str(r.get("views_name") or "").startswith("not[") if r.get("views_name") else (r.get("effect") or 0) > 0
        r["name"] = ("not" if neg else "") + f"[{r.get('short') or r['explanation']}]"
    a.out.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    log_spend(f"{tag} input labels", spent(), f"{len(rows)} groups, revise {a.rounds} rounds + near-miss round, {a.before} tokens before; {a.out}")
    s0 = spent()
    # 2. output labels for groups that move the answer
    gs = {g["id"]: g for g in groups}
    out_rows = [r for r in rows if r["writer_valid"] and gs[r["group"]].get("answer_effect", abs(r.get("effect") or 0)) >= a.effect_floor]
    if out_rows and a.outputs:
        effs = neuron_verify.remote_effects(a.measure_url, [{"contexts": [e["ids"][:e["at"] + 1] for e in exs[r["group"]][:max(OUT_TEST) + 1]],
                                                             "members": r["members"]} for r in out_rows])
        outs = wr([OUTPUT.format(members=unit_names(r["members"]), blocks="\n".join(
            effect_block(tok, j + 1, exs[r["group"]][i], eff[i]) for j, i in enumerate(OUT_DEV))) for r, eff in zip(out_rows, effs)], max_tokens=400)
        reads, where = [], []
        for k, (r, o) in enumerate(zip(out_rows, outs)):
            r["output"] = (neuron_verify.parse_json(o) or {}).get("output")
            for i in OUT_TEST:
                if isinstance(r["output"], str) and r["output"].strip():
                    b, t = context_text(tok, exs[r["group"]][i])
                    reads.append(READ_OUT.format(output=r["output"], before=b, token=t))
                    where.append((k, i))
        got = {}
        for (k, i), x in zip(where, rd(reads, max_tokens=80)):
            got.setdefault(k, {})[i] = (neuron_verify.parse_json(x) or {}).get("tokens")
        for k, (r, eff) in enumerate(zip(out_rows, effs)):
            lab = sum(hits(got.get(k, {}).get(i), eff[i]) for i in OUT_TEST)
            base = sum(hits([w for w, _ in eff[i]["top"][:3]], eff[i]) for i in OUT_TEST)
            n = len(OUT_TEST)
            r["output_scores"] = {"contexts": n, "hits": lab, "baseline_hits": base, "pass": bool(r.get("output")) and lab >= n / 2 and lab > base,
                                  "mean_tv": sum(eff[i]["tv"] for i in OUT_TEST) / n}
        a.out.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
        log_spend(f"{tag} output labels", spent() - s0, f"{len(out_rows)} groups with |effect| >= {a.effect_floor}; {a.out}")
        print(f"output labels: {sum(r['output_scores']['pass'] for r in out_rows)}/{len(out_rows)} pass; ${spent():.3f}", flush=True)
    s0 = spent()
    # 3. input scores on fresh texts at both lengths
    todo = [r for r in rows if r["writer_valid"]]
    for length in ("short", "long"):
        got = neuron_verify.judge(rd, measure(neuron_verify.plan(rd, [item(gs[r["group"]], r["explanation"]) for r in todo], length=length,
                                                                 temperature=0.7, seed=a.seed), a.seed))
        for r, g2 in zip(todo, got, strict=True):
            r.setdefault("scores", {})[length] = {"seed": a.seed, "valid": g2.get("valid"), "pass": bool(g2.get("valid") and g2.get("passes_t1")),
                                                 **{k: g2.get(k) for k in ("blind_max", "near_miss_rate", "distant_rate", "steer_match")},
                                                 "values": {"blind": g2.get("blind_values"), "near": g2.get("near_values"),
                                                            "distant": g2.get("distant_values")}}  # shares of corpus_max: any rule re-scores
        a.out.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
        print(f"{length}: {sum(r['scores'][length]['pass'] for r in todo)}/{len(rows)} groups pass on fresh texts; ${spent():.3f}", flush=True)
    log_spend(f"{tag} fresh-text scores", spent() - s0, f"{len(todo)} groups, both blind lengths, seed {a.seed}; {a.out}")


def rescore(a) -> None:
    """A labels file's input labels scored again on fresh texts at both blind lengths against a collect output that
    has each group's corpus histogram (the FIRE_Q fire level); the earlier scores move to "scores_max30"."""
    import neuron_verify
    import reader_reward

    doc = json.loads(a.top.read_text())
    gs = {g["id"]: g for g in doc["groups"]}
    rows = [json.loads(x) for x in a.labels.read_text().splitlines() if x.strip()]
    rd = reader_reward.Reader(neuron_verify.READER, a.labels.parent)
    measure = lambda rows_, seed=0: neuron_verify.remote_measure(a.measure_url, rows_, seed)  # noqa: E731
    todo = [r for r in rows if r.get("writer_valid") and r["group"] in gs]
    for r in todo:
        if "scores" in r and "scores_max30" not in r:
            r["scores_max30"] = r.pop("scores")
    for length in ("short", "long"):
        got = neuron_verify.judge(rd, measure(neuron_verify.plan(rd, [item(gs[r["group"]], r["explanation"]) for r in todo], length=length,
                                                                 temperature=0.7, seed=a.seed), a.seed))
        for r, g2 in zip(todo, got, strict=True):
            r.setdefault("scores", {})[length] = {"seed": a.seed, "fire": g2.get("fire"), "valid": g2.get("valid"),
                                                 "pass": bool(g2.get("valid") and g2.get("passes_t1")),
                                                 **{k: g2.get(k) for k in ("blind_max", "near_miss_rate", "distant_rate", "steer_match")},
                                                 "values": {"blind": g2.get("blind_values"), "near": g2.get("near_values"),
                                                            "distant": g2.get("distant_values")}}  # shares of corpus_max: any rule re-scores
        print(f"{length}: {sum(r['scores'][length]['pass'] for r in todo)}/{len(todo)} groups pass at their fire level; ${rd.spend.total:.3f}", flush=True)
    a.labels.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    log_spend(f"varteach groups {a.labels.stem} rescore at q{FIRE_Q}", rd.spend.total, f"{len(todo)} groups, both blind lengths; {a.labels}")


def estimate(a) -> None:
    """The Tinker cost of label on these groups before any call, from the writer's prompts' token counts and the
    measured shares of past runs (8B neuron arm 10-16: 0.46 of first drafts fail, 0.6 of those again; a reader plan 300
    prompt and 700 sampled tokens; a judge 250 and 10; an output reading 150 and 20), times --margin."""
    sys.path.insert(0, str(HERE / "rl"))
    from tinker_train import PRICES

    import neuron_verify
    LONG_ = a.before
    globals()["LONG"] = LONG_
    doc, tok, groups, exs = load(a)
    head = lambda g: dict(members=unit_names(g["members"]), gmax=g["max"])  # noqa: E731
    first = [len(tok(EXPLAIN.format(**head(g), examples=block(tok, exs[g["id"]][:a.examples], a.long)))["input_ids"]) for g in groups]
    n, P = len(groups), first and sum(first) / len(first)
    w, r = PRICES[a.writer], PRICES[neuron_verify.READER]
    fails = [n * 0.46, n * 0.46 * 0.6][:a.rounds]
    near = n * 0.4 if a.near_round else 0
    writer = (n * P + (sum(fails) + near) * (P + 600)) * w[0] + (n + sum(fails) + near) * 250 * w[1]
    plans = (n + sum(fails) + near + 2 * n) * (300 * r[0] + 700 * r[1])
    judge = 2 * n * (250 * r[0] + 10 * r[1])
    nout = sum(g.get("answer_effect", abs(g.get("effect") or 0)) >= a.effect_floor for g in groups) if a.outputs else 0
    outw = nout * (len(OUT_DEV) * 160 + 300) * w[0] + nout * 120 * w[1]
    outr = nout * len(OUT_TEST) * (150 * r[0] + 20 * r[1])
    parts = {"writer": writer, "reader plans": plans, "judge": judge, "output writer": outw, "output reader": outr}
    tot = sum(parts.values()) / 1e6
    print(f"{n} groups ({nout} with output labels), first prompt {P:.0f} tokens: " + ", ".join(f"{k} ${v / 1e6:.2f}" for k, v in parts.items()) +
          f"; total ${tot:.2f}, with margin x{a.margin}: ${tot * a.margin:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--groups", type=Path, required=True, help="neuron_motifs groups: {\"groups\": [[key, ...], ...]} or a list")
    p.add_argument("--atlas", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--apply", type=Path, help="neuron_motifs apply's output directory: each group's largest |answer drop| (bits)")
    p.add_argument("--min-size", type=int, default=2, help="groups of fewer poles are single neurons (atlas_labels)")
    c = sub.add_parser("collect")
    c.add_argument("--groups", type=Path, required=True)
    c.add_argument("--corpus", type=Path, required=True, help="a corpus directory: tokens.npy, offsets.npy (atlas.py corpus)")
    c.add_argument("--model", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--tokens", type=int, default=5_000_000)
    c.add_argument("--top", type=int, default=64)
    c.add_argument("--batch", type=int, default=8, help="documents a forward pass")
    for name in ("label", "estimate"):
        lb = sub.add_parser(name)
        lb.add_argument("--top", type=Path, required=True, help="collect's output")
        lb.add_argument("--corpus", type=Path, required=True)
        lb.add_argument("--out", type=Path, required=name == "label")
        lb.add_argument("--measure-url", required=name == "label", help="neuron_verify serve over a tunnel, the same model")
        lb.add_argument("--writer", default="Qwen/Qwen3.5-397B-A17B")
        lb.add_argument("--examples", type=int, default=30)
        lb.add_argument("--long", type=int, default=10, help="examples shown with --before tokens before (the rest SHORT)")
        lb.add_argument("--before", type=int, default=256, help="tokens before each of the first --long examples")
        lb.add_argument("--rounds", type=int, default=2)
        lb.add_argument("--near-round", type=int, default=1, help="1: one more revision aimed at near misses that fired")
        lb.add_argument("--outputs", type=int, default=1, help="1: output labels for groups that move the answer")
        lb.add_argument("--effect-floor", type=float, default=1.0, help="bits: the group's largest answer drop when zeroed "
                        "(prepare --apply; else |effect|, the members' mean single-zeroing change); neuron_builder's NEC_BITS")
        lb.add_argument("--ids", help="comma-separated group ids, a subset")
        lb.add_argument("--n", type=int, help="the first N groups")
        lb.add_argument("--seed", type=int, default=1)
        lb.add_argument("--margin", type=float, default=1.5)
    rs = sub.add_parser("rescore")
    rs.add_argument("--labels", type=Path, required=True)
    rs.add_argument("--top", type=Path, required=True, help="a collect output with histograms")
    rs.add_argument("--measure-url", required=True)
    rs.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    {"prepare": prepare, "collect": collect, "label": label, "estimate": estimate, "rescore": rescore}[a.what](a)


if __name__ == "__main__":
    main()
