"""ADAG's own code against our measurements (#2951; the lead's 10-15 head-to-head). Runs inside a clone of ADAG's
repository (github.com/TransluceAI/circuits, no license: run there through its own uv environment, nothing of it copied
here).
  clusters   a circuit traced by ADAG's scripts/circuit_prep/prep.py, clustered by its own multiview spectral clustering
             (Circuit.cluster_multiview, k clusters, no descriptions), its cluster state saved; then every cluster
             measured on every prompt of the circuit the way ADAG zeroes a cluster (its neurons at their token positions
             in that prompt, from the circuit's own node table), with our log-probability read-outs instead of ADAG's
             top-5 logit sum: p(the prompt's clean top token) before and after, whether the top token changes, the
             cluster's size (ADAG's count: distinct neurons over the prompts). Our found sets (adag_compare.py run's
             records, whole neurons zeroed at every position) are measured on the same prompts and token ids beside them.
  math-data  a prep.py dataset of N sums "What is X + Y?" (18 + 24 first, then seeded draws with X, Y below 100), seeded
             "Answer: " as ADAG's math set: its 10,000 sums are beyond one card's hour
  cg-test    CausalGym test pairs with every prefix of the train file removed (ADAG's generator draws both splits from
             the same small vocabulary, so a test prefix is often a train prefix)
  groups     grouping quality of a clustering (ADAG's clusters; random partitions of the same sizes beside them): each
             unit a neuron and polarity summed over tokens, as ADAG clusters them; its profile its attribution on every
             prompt (0 where absent); the silhouette of the profiles (cosine distance), the coefficient of variation of the
             cluster sizes, and the share of same-cluster pairs whose attributions have opposite signs on every prompt
             both appear in. Reads the circuit pickle without ADAG's code (its classes as plain records).
  relp       ADAG's tracer's own neuron scores on one question of circuit_questions (its token ids from
             circuit_questions.chat): get_all_pairs_cl_ja_effects_with_attributions with prep.py's settings (RelP
             gradient, half rule, stop-gradient on MLPs, zero ablation) and return_only_important_neurons, which returns
             every neuron's attribution, gradient x activation of the down projection's input, to the objective: the sum
             of the focus logits at the last position (focus "answer": the clean top token alone; "top5": ADAG's own
             top-5 logits). Writes OUT.pt ({focus: {"attribution", "activation": [layers, positions, neurons]}}) and
             OUT.json (ids, focus tokens, objective values, sign: positive raises the objective).
  trace      ADAG's tracer on circuit_questions' questions (the lead's 10-16 Lane 3: the questions of the family
             circuits), each its own patching pair's clean prompt in our token ids (circuit_finder.chat), prep.py's
             settings (RelP gradient, k = 5 top logits, threshold 0.005, no blacklist on Qwen3), timed; its neurons
             written as a circuit_finder record (the oracle's input format): every node at its position with its pole
             (the clean activation's sign), ADAG's attribution and our measured shares alone on the pair (lost noising,
             restored denoising: circuit_finder's Patcher), one cut "adag" with the whole set's shares and met at PASS
             both ways, no routes (ADAG has no head units). Nodes before the pair's first differing token keep shares 0
             (the two runs agree there). Families as circuit_family picks them (sorted task/key, every stride-th).
  measured   our node scores in ADAG's CausalGym evaluation: NAP's train.pt (scripts/train.py --method nap, MLP neurons)
             with its CAND largest scores replaced by measured ones, each (layer, position, neuron) alone set to its mean
             over the train pairs' patch prompts (the value ADAG's evaluation ablates to) on the train pairs' clean
             prompts, its score the mean change in logit(patch answer) - logit(clean answer), NAP's sign. scripts/eval.py
             then scores the written train.pt like NAP's own. --order first ranks every measured node above the rest
             (NAP's other scores scaled under the smallest measured one), so the curve up to CAND nodes is ours alone.

  (in ADAG's clone)  uv run python PARAMS/experiments/oracle/adag_repo.py clusters CIRCUIT.pkl OUT.json --model ID --k 64
      [--ours ADAG_COMPARE.json --case capitals_llama]
  adag_repo.py math-data OUT.py [--n 100]
  adag_repo.py cg-test TRAIN.json TEST.json
  adag_repo.py groups CIRCUIT.pkl CLUSTER_STATE.json OUT.json
  adag_repo.py measured NAP_TRAIN.pt OUT_TRAIN.pt --model ID [--cand 8192] [--order replace|first]
  (in ADAG's clone)  adag_repo.py relp QUESTION_ID OUT --model Qwen/Qwen3-8B [--questions FILE]
  adag_repo.py match ADAG_TRACE_DIR FAMILY_PAIRS_DIR OUT   ADAG's circuits cut to our finder's size on the same pairs
  (in ADAG's clone)  adag_repo.py relpcheck QUESTION_ID OUT.json --model M   ADAG's wrapped forward against the model's
  (in ADAG's clone)  adag_repo.py trace QUESTIONS.json OUT --model Qwen/Qwen3-8B [--split dev] [--stride 2] [--limit N]
      [--families task/key,...] [--chunk 10]
      [--finder N]   (circuit_finder on the first N of them too: OUT/finder/, timed on the same card)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

CAND = 8192
GROUP = 32  # candidates measured in one forward pass (each on its own copy of the batch)
BATCH = 50


def hook_zero(model, zero: list[tuple]):
    """Forward pre-hooks on each layer's down projection: (layer, position or None, neuron) zeroed on every row (None:
    every position). Returns the handles."""
    by = {}
    for L, p, j in zero:
        by.setdefault(L, []).append((p, j))
    hs = []
    for L, layer in enumerate(model.model.layers):
        if L not in by:
            continue

        def pre(m, a, L=L):
            x = a[0].clone()
            for p, j in by[L]:
                if p is None:
                    x[:, :, j] = 0
                elif p < x.shape[1]:
                    x[:, p, j] = 0
            return (x,) + tuple(a[1:])
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
    return hs


@torch.no_grad()
def logp_last(model, ids: list[int], zero: list[tuple]) -> torch.Tensor:
    """log p at the last position with each (layer, position or None, neuron) zeroed."""
    hs = hook_zero(model, zero)
    try:
        out = model(input_ids=torch.tensor([ids], device=next(model.parameters()).device))
        return torch.log_softmax(out.logits[0, -1].float(), -1)
    finally:
        for h in hs:
            h.remove()


def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if cuda else torch.float32,
                                                 device_map="cuda" if cuda else "cpu").eval()
    return tok, model


def clusters(a) -> None:
    from circuits.analysis.circuit_ops import Circuit  # ADAG's own package (its clone's environment)

    t0 = time.time()
    tok, model = load_model(a.model)
    c = Circuit.load_from_pickle(a.circuit)
    c.set_tokenizer(tok, num_layers=model.config.num_hidden_layers)
    c.cluster_multiview(n_clusters=a.k, get_desc=False)
    c.save_cluster_state(a.out.with_name(a.out.stem + "_cluster_state.json"))
    t_cluster = time.time() - t0
    key_to_cluster = {(n.layer, n.token, n.neuron): cl for n, cl in c._cluster_map.items()}
    sizes = {}
    for n, cl in zip(c._mv_neuron_ids, c._mv_cluster_labels):  # (the summed map: one entry a neuron over the prompts)
        sizes.setdefault(f"C{cl}", set()).add((n.layer, n.neuron))
    last = model.config.num_hidden_layers
    df = c.df_node
    ours = {}
    if a.ours and a.ours.exists():
        rec = json.loads(a.ours.read_text()).get(a.case, {})
        for g in rec.get("goals", []):
            if g.get("found"):
                ours[f"ours {g['goal']} {g.get('target')} {g.get('p_target_goal')}"] = [(u[0], None, u[1]) for u in g["units"]]
    rows = []
    for i, (ci, label) in enumerate(zip(c.cis, c.labels)):
        ids = [int(z) for z in (ci.tolist() if hasattr(ci, "tolist") else ci)]
        sub = df[df.label == f"{label}___{i}"]
        units = {}
        for L, p, j in zip(sub.layer, sub.token, sub.neuron):
            cl = key_to_cluster.get((int(L), int(p), int(j)))
            if cl is not None and int(L) not in (-1, last):
                units.setdefault(cl, []).append((int(L), int(p), int(j)))
        base = logp_last(model, ids, [])
        y0 = int(base.argmax())  # (the model's own answer: a label such as "18 + 24 = 42" is not a token)
        row = {"prompt": i, "label": label, "top0": tok.decode([y0]), "p0": float(base[y0].exp()), "clusters": {}}
        for name, zs in list(units.items()) + list(ours.items()):
            l1 = logp_last(model, ids, zs)
            row["clusters"][name] = {"p": float(l1[y0].exp()), "top": tok.decode([int(l1.argmax())]),
                                     "changed": int(l1.argmax()) != y0, "units_here": len(zs)}
        rows.append(row)
        print(f"prompt {i} {label!r} -> {row['top0']!r} p {row['p0']:.3f}; clusters changing the top: "
              f"{[k for k, v in row['clusters'].items() if v['changed']]}", flush=True)
    names = sorted({k for r in rows for k in r["clusters"]})
    summary = {}
    for k in names:
        rs = [r for r in rows if k in r["clusters"]]
        summary[k] = {"size": len(sizes.get(k, [])) if not k.startswith("ours") else len(ours[k]), "prompts": len(rs),
                      "changed": sum(r["clusters"][k]["changed"] for r in rs) / max(len(rs), 1),
                      "mean_p_drop": sum(r["p0"] - r["clusters"][k]["p"] for r in rs) / max(len(rs), 1),
                      "prompt0": rows[0]["clusters"].get(k)}
    a.out.write_text(json.dumps({"model": a.model, "k": a.k, "seconds_cluster": t_cluster, "seconds": time.time() - t0,
                                 "summary": summary, "rows": rows}))
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]["changed"])[:12]:
        print(f"{k}: {v['size']} neurons; top answer changed on {v['changed']:.2f} of {v['prompts']} prompts; mean p drop "
              f"{v['mean_p_drop']:.3f}; prompt 0 {v['prompt0']}", flush=True)


def math_data(a) -> None:
    rng = random.Random(0)
    pairs = [(18, 24)]
    while len(pairs) < a.n:
        p = (rng.randrange(100), rng.randrange(100))
        if p not in pairs:
            pairs.append(p)
    a.out.write_text(f'"""{a.n} sums for prep.py (adag_repo.py math-data)."""\n\n'
                     f"pairs = {pairs!r}\n"
                     'prompts = [f"What is {x} + {y}?" for x, y in pairs]\n'
                     'seed_responses = ["Answer: "] * len(prompts)\n'
                     'labels = [f"{x} + {y} = {x + y}" for x, y in pairs]\n')
    print(f"{a.out}: {a.n} sums")


def cg_test(a) -> None:
    train = {json.loads(s)["clean_prefix"] for s in a.train.read_text().splitlines() if s.strip()}
    rows = [s for s in a.test.read_text().splitlines() if s.strip()]
    fresh = [s for s in rows if json.loads(s)["clean_prefix"] not in train]
    a.test.write_text("".join(s + "\n" for s in fresh))
    print(f"{a.test}: {len(fresh)} of {len(rows)} test pairs have a prefix absent from the train file")


class Record:
    """Any class of ADAG's pickles, read as a plain record (its constructor arguments or state as attributes)."""
    def __new__(cls, *args, **kw):
        o = object.__new__(cls)
        o.args = args
        return o

    def __setstate__(self, state):
        if isinstance(state, tuple):  # (dict state, slots state)
            for part in state:
                self.__dict__.update(part or {})
        else:
            self.__dict__.update(state if isinstance(state, dict) else {"state": state})


def load_records(path: Path):
    import pickle

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in ("circuits", "util", "env_util", "neurondb"):
                return Record
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return Unpickler(f).load()


def grouping(profiles, labels: list, draws: int = 20, seed: int = 0) -> dict:
    """Silhouette (cosine), size CV and the all-opposing-sign pair share of a labelled set of profiles [units, prompts],
    and the same for DRAWS random partitions with the same sizes."""
    import numpy as np
    from sklearn.metrics import silhouette_score

    def stats(lab):
        lab = np.asarray(lab)
        sizes = np.array([np.sum(lab == c) for c in np.unique(lab)])
        keep = np.isin(lab, np.unique(lab)[sizes > 1])
        sil = float(silhouette_score(profiles[keep], lab[keep], metric="cosine")) if len(np.unique(lab[keep])) > 1 else None
        opp = pairs = 0
        for c in np.unique(lab):
            idx = np.flatnonzero(lab == c)
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    a, b = profiles[idx[i]], profiles[idx[j]]
                    both = (a != 0) & (b != 0)
                    if both.any():
                        pairs += 1
                        opp += bool(np.all(np.sign(a[both]) != np.sign(b[both])))
        return {"silhouette": sil, "size_cv": float(sizes.std() / sizes.mean()), "clusters": len(sizes),
                "opposing_pairs": opp / max(pairs, 1), "pairs": pairs}
    rng = np.random.default_rng(seed)
    rand = [stats(rng.permutation(labels)) for _ in range(draws)]
    return {"clusters": stats(labels), "random": {k: float(np.mean([r[k] for r in rand if r[k] is not None]))
                                                for k in ("silhouette", "opposing_pairs")}}


def groups(a) -> None:
    import numpy as np
    c = load_records(a.circuit)
    df = c.df_node
    state = json.loads(a.state.read_text())["cluster_map"]
    unit_cluster = {}
    for k, cl in state.items():
        L, _, j, pol = k.split(",")
        if cl.startswith("C") and cl[1:].isdigit():  # (logit nodes are their own named entries, not clusters)
            unit_cluster.setdefault((int(L), int(j), pol), cl)
    pol = np.where(df.activation >= 0, "+", "-") if "polarity" not in df.columns else df.polarity
    prompts = sorted(df.label.unique())
    col = {p: i for i, p in enumerate(prompts)}
    units = sorted(unit_cluster)
    row = {u: i for i, u in enumerate(units)}
    prof = np.zeros((len(units), len(prompts)))
    for L, j, po, lab, at in zip(df.layer, df.neuron, pol, df.label, df.attribution):
        u = (int(L), int(j), po)
        if u in row:
            prof[row[u], col[lab]] += float(at)
    labels = [unit_cluster[u] for u in units]
    rec = {"circuit": str(a.circuit), "units": len(units), "prompts": len(prompts), **grouping(prof, labels)}
    a.out.write_text(json.dumps(rec))
    print(json.dumps(rec, indent=1))


def relp(a) -> None:
    import circuit_questions
    from circuits.tracing.clja import ADAGConfig, get_all_pairs_cl_ja_effects_with_attributions  # ADAG's own tracer

    tok, model = load_model(a.model)
    dev = next(model.parameters()).device
    q = (circuit_questions.load(a.questions) if a.questions else circuit_questions.load())[a.qid]
    ids = list(circuit_questions.chat(tok, q["user"]))
    T = len(ids)
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=dev)).logits[0, -1].float()
    top5 = logits.topk(5).indices.tolist()
    out, meta = {}, {"id": a.qid, "model": a.model, "ids": ids, "user": q["user"], "focus": {},
                     "objective": "sum of the focus tokens' logits at the last position",
                     "score": "gradient x activation (down projection input), RelP gradient; positive raises the objective"}
    for name, focus in (("answer", top5[:1]), ("top5", top5)):
        cfg = ADAGConfig(device=str(dev), verbose=False, parent_threshold=None, edge_threshold=0.01, node_attribution_threshold=None,
                         topk=None, batch_aggregation="any", topk_neurons=None, percentage_threshold=0.005, use_relp_grad=True,
                         disable_half_rule=False, disable_stop_grad=False, ablation_mode="zero", use_stop_grad_on_mlps=True,
                         return_nodes_only=False, focus_last_residual=False, skip_attr_contrib=False, center_logits=False,
                         ig_steps=None, ig_mode="ig-inputs", return_only_important_neurons=True, apply_blacklist=False)
        attr, _, acts, _ = get_all_pairs_cl_ja_effects_with_attributions(
            model, tok, [ids], cfg, src_tokens=list(range(T)), tgt_tokens=[T - 1] * len(focus), attention_masks=[[1] * T],
            focus_logits=[focus])
        attr = attr.float().cpu()
        attr = attr[:, 0, ..., 0] if attr.dim() == 5 else attr[:, 0]
        acts = torch.as_tensor(acts).float().cpu()
        acts = acts[:, 0] if acts.dim() == 4 else acts
        out[name] = {"attribution": attr, "activation": acts}
        meta["focus"][name] = {"tokens": focus, "decoded": [tok.decode([t]) for t in focus],
                               "objective_value": float(logits[focus].sum()), "attribution_sum": float(attr.sum())}
        print(f"{name}: {meta['focus'][name]['decoded']} objective {meta['focus'][name]['objective_value']:.3f}, "
              f"attribution sum {meta['focus'][name]['attribution_sum']:.3f}, shape {tuple(attr.shape)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out.with_suffix(".pt"))
    a.out.with_suffix(".json").write_text(json.dumps(meta))


@torch.no_grad()
def relpcheck(a) -> None:
    """Whether ADAG's wrapped forward (stop_nonlinear_grad with RelP: its norms, attention and MLP wrappers) is the
    model's: on one question, the last position's logits and every layer's down-projection input [T, D] in the plain
    model against the wrapped one (per layer: Pearson r and max |difference|), and the same for the logit objective
    (pymove's 10-16 check: the RelP dump's activations correlated 0.5 with the plain forward)."""
    import circuit_questions
    from circuits.tracing.grad import revert_stop_nonlinear_grad, stop_nonlinear_grad

    tok, model = load_model(a.model)
    dev = next(model.parameters()).device
    q = (circuit_questions.load(a.questions) if a.questions else circuit_questions.load())[a.qid]
    ids = torch.tensor([list(circuit_questions.chat(tok, q["user"]))], device=dev)

    def run(wrapped: bool):
        acts = {}
        hs = []
        for i, layer in enumerate(model.model.layers):
            dp = layer.mlp.mlp.down_proj if wrapped and hasattr(layer.mlp, "mlp") else layer.mlp.down_proj
            hs.append(dp.register_forward_pre_hook(lambda m, x, i=i: acts.__setitem__(i, x[0][0].float().cpu())))
        try:
            lg = model(input_ids=ids).logits[0, -1].float().cpu()
        finally:
            for h in hs:
                h.remove()
        return lg, acts
    lg0, a0 = run(False)
    stop_nonlinear_grad(model, use_relp_grad=True, use_half_rule=True)
    try:
        lg1, a1 = run(True)
    finally:
        revert_stop_nonlinear_grad(model)
    layers = []
    for i in sorted(a0):
        x, y = a0[i].flatten(), a1[i].flatten()
        layers.append({"layer": i, "r": float(torch.corrcoef(torch.stack([x, y]))[0, 1]), "max_abs_diff": float((x - y).abs().max())})
    top = int(lg0.argmax())
    out = {"id": a.qid, "model": a.model, "top_plain": tok.decode([top]), "top_wrapped": tok.decode([int(lg1.argmax())]),
           "logit_plain": float(lg0[top]), "logit_wrapped": float(lg1[top]), "logits_max_abs_diff": float((lg0 - lg1).abs().max()),
           "layers": layers}
    a.out.write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "layers"}), "| r per layer:", [round(x["r"], 3) for x in layers], flush=True)


def trace(a) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))
    import circuit_finder as cf
    import circuits.tracing.clja as clja
    from circuits.tracing.clja import ADAGConfig, get_all_pairs_cl_ja_effects_with_attributions  # ADAG's own tracer
    from circuits.tracing.grad import revert_stop_nonlinear_grad

    own = clja._get_neuron_attr_and_contrib  # (its neurons' backward in chunks of CHUNK, not its 50: 50 ran a 48 GB L40 out of
    clja._get_neuron_attr_and_contrib = lambda *x, **k: own(*x, **{**k, "neuron_chunk_size": a.chunk})  # memory on Qwen3-8B)
    tok, model = load_model(a.model)
    model.requires_grad_(False)
    dev = next(model.parameters()).device
    L = model.config.num_hidden_layers
    qs = [q for q in json.loads(a.questions.read_text())["questions"] if q["split"] == a.split]
    fams = {}
    for q in qs:
        fams.setdefault(f"{q['task']}/{q['key']}", []).append(q)
    names = a.families.split(",") if a.families else sorted(fams)[::a.stride]
    todo = [q for name in names for q in sorted(fams[name], key=lambda q: q["paraphrase"])][:a.limit or None]
    a.out.mkdir(parents=True, exist_ok=True)
    pt = cf.Patcher(model, tok)
    cfg = ADAGConfig(device=str(dev), verbose=False, parent_threshold=None, edge_threshold=0.01, node_attribution_threshold=None,
                     topk=None, batch_aggregation="any", topk_neurons=None, percentage_threshold=0.005, use_relp_grad=True,
                     disable_half_rule=False, disable_stop_grad=False, ablation_mode="zero", use_stop_grad_on_mlps=True,
                     return_nodes_only=False, focus_last_residual=False, skip_attr_contrib=False, center_logits=False,
                     ig_steps=None, ig_mode="ig-inputs", return_only_important_neurons=False, apply_blacklist=False)
    rev = (Path.cwd() / "adag_commit.txt").read_text().strip() if (Path.cwd() / "adag_commit.txt").exists() else None
    first = True
    for q in todo:
        path = a.out / (q["id"].replace("/", "__") + ".json")
        if path.exists():
            continue
        ids, cids = cf.chat(tok, q["user"]), cf.chat(tok, q["corrupt_user"])
        if len(ids) != len(cids):
            continue
        T = len(ids)
        with torch.no_grad():
            top5 = model(torch.tensor([ids], device=dev)).logits[0, -1].float().topk(5).indices.tolist()
        if first:  # (an untimed run first: CUDA's first kernels)
            get_all_pairs_cl_ja_effects_with_attributions(model, tok, [ids], cfg, src_tokens=list(range(T)), tgt_tokens=[T - 1] * 5,
                                                          attention_masks=[[1] * T], focus_logits=[top5])
            try:
                revert_stop_nonlinear_grad(model)
            except Exception:
                pass
            first = False
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        nodes, edges = get_all_pairs_cl_ja_effects_with_attributions(model, tok, [ids], cfg, src_tokens=list(range(T)),
                                                                     tgt_tokens=[T - 1] * 5, attention_masks=[[1] * T], focus_logits=[top5])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_trace = time.time() - t0
        try:
            revert_stop_nonlinear_grad(model)
        except Exception:
            pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        units = {}
        for n in nodes:
            if 0 <= n.layer < L:
                k = (int(n.layer), int(n.token), int(n.neuron))
                units[k] = (units.get(k, (0.0, 0.0))[0] + float(n.final_attribution[0].sum()), float(n.activation[0]))
        t1 = time.time()
        rec = pt.setup(ids, cids)
        rec["tokens"] = T
        us = [("neuron", L_, p, j) for (L_, p, j) in units]
        singles = pt.both([[u] for u in us]) if us else []
        whole = pt.both([us])[0] if us else (0.0, 0.0)
        out_nodes = []
        for (L_, p, j), (lo, re) in zip(units, singles):
            v, w = float(pt.act["clean"]["neu"][L_][p, j]), float(pt.act["clean"]["up"][L_][p, j])
            out_nodes.append({"stage": "adag", "layer": L_, "pos": p, "neuron": j, "pole": 1 if v > 0 else -1,
                              "regime": [1 if v * w > 0 else -1, 1 if w > 0 else -1], "clean": v,
                              "corrupt": float(pt.act["corrupt"]["neu"][L_][p, j]), "lost": lo, "restored": re,
                              "attribution": units[(L_, p, j)][0]})
        met = bool(whole[0] >= cf.PASS and whole[1] >= cf.PASS)
        rec.update(id=q["id"], ids=ids, corrupted=cids, answer_text=tok.decode([rec["answer"]]),
                   corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, adapter=None, dtype=str(model.dtype).split(".")[-1],
                   finder="adag", finder_rev=rev, user=q["user"], corrupt_user=q["corrupt_user"], task=q["task"], key=q["key"],
                   nodes=out_nodes, routes=[], met=met, order=["adag"], size=len(out_nodes),
                   cuts={"adag": {"met": met, "lost": whole[0], "restored": whole[1], "size": len(out_nodes)}},
                   seconds=round(t_trace, 2), seconds_measure=round(time.time() - t1, 2),
                   adag={"k": 5, "focus": top5, "focus_text": [tok.decode([t]) for t in top5], "threshold": 0.005, "edges": len(edges),
                         "before_first_diff": sum(p < rec["first_diff"] for (_, p, _) in units)})
        path.write_text(json.dumps(rec))
        if a.finder and len(list((a.out / "finder").glob("*.json")) if (a.out / "finder").exists() else []) < a.finder:
            (a.out / "finder").mkdir(exist_ok=True)  # our finder on the same pair and card, timed alike
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t2 = time.time()
            fr = cf.find(pt, ids, cids)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fr.update(id=q["id"], ids=ids, corrupted=cids, model=a.model, finder="circuit_finder", wall_seconds=round(time.time() - t2, 2))
            (a.out / "finder" / path.name).write_text(json.dumps(fr))
            print(f"{q['id']}: circuit_finder {fr.get('size')} units, met {fr.get('met')}, {fr['wall_seconds']:.1f} s", flush=True)
        print(f"{q['id']}: ADAG {len(out_nodes)} neurons ({rec['adag']['before_first_diff']} before the change), {len(edges)} edges, "
              f"{t_trace:.1f} s; lost {whole[0]:.2f} restored {whole[1]:.2f}{'' if met else ' unmet'}", flush=True)


def match(a) -> None:
    """ADAG's circuit of each traced question cut to our finder's size on the same pair: its neurons with the largest
    |attribution|, as many as the finder record's neurons and heads (circuit_family's pair record of the question
    against its own corrupted key in PAIRS, the family run's pairs/); written as finder records beside ADAG's full ones."""
    a.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(a.adag.glob("*.json")):
        r = json.loads(f.read_text())
        q = r["id"]
        cands = [p for p in a.pairs.glob(f"{q.replace('/', '__')}__vs__*.json")]
        mine = next((json.loads(p.read_text()) for p in cands if json.loads(p.read_text()).get("corrupted") == r["corrupted"]), None)
        if mine is None or not mine.get("cuts"):
            continue
        k = len(mine.get("nodes", [])) + len(mine.get("routes", []))
        keep = sorted(r["nodes"], key=lambda u: -abs(u["attribution"]))[:k]
        r.update(nodes=keep, size=len(keep), matched_to={"pair": mine["id"], "finder_size": k, "adag_size": len(r["nodes"])},
                 finder="adag_matched")
        (a.out / f.name).write_text(json.dumps(r))
        n += 1
    print(f"{n} ADAG circuits cut to their questions' finder sizes in {a.out}")


@torch.no_grad()
def measured(a) -> None:
    t0 = time.time()
    d = torch.load(a.nap, map_location="cpu", weights_only=False)
    nodes, ex = d["nodes"], d["examples"]
    tok, model = a.loaded if hasattr(a, "loaded") else load_model(a.model)
    dev = next(model.parameters()).device
    clean = tok([e["clean_prefix"] for e in ex], return_tensors="pt").input_ids.to(dev)  # (ADAG's loader keeps one length)
    patch = tok([e["patch_prefix"] for e in ex], return_tensors="pt").input_ids.to(dev)
    ca = torch.tensor([tok(e["clean_answer"]).input_ids[-1] for e in ex], device=dev)
    pa = torch.tensor([tok(e["patch_answer"]).input_ids[-1] for e in ex], device=dev)
    n, P = clean.shape
    layers = sorted(int(k.split("_")[1]) for k in nodes if k.startswith("mlp_"))
    acts = {L: nodes[f"mlp_{L}"].act for L in layers}
    per_pos = acts[layers[0]].dim() == 2  # (ADAG's "mean" aggregation keeps the position; "sum" sums over it)
    shape = acts[layers[0]].shape
    flat = torch.cat([acts[L].abs().float().flatten() for L in layers])
    top = flat.topk(min(a.cand, flat.numel())).indices
    per_layer = acts[layers[0]].numel()
    cands = []
    for i in top.tolist():
        L, r = layers[i // per_layer], i % per_layer
        cands.append((L, r // shape[-1], r % shape[-1]) if per_pos else (L, None, r))

    mean = {}  # the mean over the patch prompts of each layer's down-projection input, [P, D]
    def grab(m, inp, L):
        mean[L] = mean.get(L, 0) + inp[0].float().sum(0)
    hs = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, inp, L=L: grab(m, inp, L)) for L in layers]
    for b in range(0, n, BATCH):
        model.model(input_ids=patch[b:b + BATCH])
    for h in hs:
        h.remove()
    mean = {L: v / n for L, v in mean.items()}

    def metric(ids, c, p):  # logit(patch answer p) - logit(clean answer c) at the last position, per row
        lg = model.lm_head(model.model(input_ids=ids).last_hidden_state[:, -1]).float()
        return lg.gather(1, p[:, None])[:, 0] - lg.gather(1, c[:, None])[:, 0]

    eff = torch.zeros(len(cands), device=dev)
    for b in range(0, n, BATCH):
        ids, ca_b, pa_b = clean[b:b + BATCH], ca[b:b + BATCH], pa[b:b + BATCH]
        base = metric(ids, ca_b, pa_b)
        B = ids.shape[0]
        for g0 in range(0, len(cands), GROUP):
            grp = cands[g0:g0 + GROUP]
            G = len(grp)
            by = {}
            for g, (L, p, j) in enumerate(grp):
                by.setdefault(L, []).append((g, p, j))

            def pre(m, inp, L):
                x = inp[0].clone().view(G, B, P, -1)
                for g, p, j in by[L]:
                    if p is None:
                        x[g, :, :, j] = mean[L][:, j].to(x.dtype)
                    else:
                        x[g, :, p, j] = mean[L][p, j].to(x.dtype)
                return (x.view(G * B, P, -1),) + tuple(inp[1:])
            hs = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, inp, L=L: pre(m, inp, L)) for L in by]
            try:
                m1 = metric(ids.repeat(G, 1), ca_b.repeat(G), pa_b.repeat(G)).view(G, B)
            finally:
                for h in hs:
                    h.remove()
            eff[g0:g0 + G] += (m1 - base[None]).sum(1)
        print(f"pairs {b + B}/{n}, {time.time() - t0:.0f} s", flush=True)
    eff = (eff / n).cpu()
    if a.order == "first":  # every measured node above every unmeasured one: the rest scaled under the smallest measured
        floor = float(eff.abs()[eff.abs() > 0].min()) if (eff.abs() > 0).any() else 0.0
        top_rest = max(float(acts[L].abs().max()) for L in layers)
        for L in layers:
            acts[L].mul_(0.5 * floor / max(top_rest, 1e-30))
    for (L, p, j), e in zip(cands, eff.tolist()):
        if per_pos:
            acts[L][p, j] = e
        else:
            acts[L][j] = e
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, a.out)
    print(f"{a.out}: {len(cands)} measured scores replace NAP's largest ({time.time() - t0:.0f} s); measured |score| "
          f"median {eff.abs().median():.2e}, max {eff.abs().max():.2e}; NAP's at the same nodes median "
          f"{flat[top].median():.2e}", flush=True)
    json.dump({"cand": len(cands), "pairs": n, "seconds": time.time() - t0,
               "units": [[L, p, j, e] for (L, p, j), e in zip(cands, eff.tolist())]},
              open(a.out.with_suffix(".json"), "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("clusters")
    c.add_argument("circuit", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--model", required=True)
    c.add_argument("--k", type=int, default=64)
    c.add_argument("--ours", type=Path, default=None)
    c.add_argument("--case", default="capitals_llama")
    m = sp.add_parser("math-data")
    m.add_argument("out", type=Path)
    m.add_argument("--n", type=int, default=100)
    q = sp.add_parser("groups")
    q.add_argument("circuit", type=Path)
    q.add_argument("state", type=Path)
    q.add_argument("out", type=Path)
    r = sp.add_parser("relp")
    r.add_argument("qid")
    r.add_argument("out", type=Path)
    r.add_argument("--model", required=True)
    r.add_argument("--questions", type=Path, default=None)
    rc = sp.add_parser("relpcheck")
    rc.add_argument("qid")
    rc.add_argument("out", type=Path)
    rc.add_argument("--model", required=True)
    rc.add_argument("--questions", type=Path, default=None)
    tr = sp.add_parser("trace")
    tr.add_argument("questions", type=Path)
    tr.add_argument("out", type=Path)
    tr.add_argument("--model", default="Qwen/Qwen3-8B")
    tr.add_argument("--split", default="dev")
    tr.add_argument("--stride", type=int, default=2)
    tr.add_argument("--limit", type=int, default=0)
    tr.add_argument("--finder", type=int, default=0, help="circuit_finder on the first N questions too, timed on the same card")
    tr.add_argument("--families", default=None, help="task/key,...: these families instead of every stride-th")
    tr.add_argument("--chunk", type=int, default=10, help="neurons a backward chunk in ADAG's attribution (its own: 50)")
    mt = sp.add_parser("match")
    mt.add_argument("adag", type=Path)
    mt.add_argument("pairs", type=Path)
    mt.add_argument("out", type=Path)
    t = sp.add_parser("cg-test")
    t.add_argument("train", type=Path)
    t.add_argument("test", type=Path)
    s = sp.add_parser("measured")
    s.add_argument("nap", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--model", required=True)
    s.add_argument("--cand", type=int, default=CAND)
    s.add_argument("--order", default="replace", choices=("replace", "first"),
                   help="replace: measured scores in NAP's place among its others; first: every measured node ranked first")
    a = ap.parse_args()
    {"clusters": clusters, "math-data": math_data, "cg-test": cg_test, "groups": groups, "measured": measured, "relp": relp, "trace": trace, "match": match, "relpcheck": relpcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
