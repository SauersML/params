"""InterpBench (Gupta et al. 2024; MIB's circuit track, Mueller et al. 2025): small TransformerLens models trained (SIIT)
to implement known circuits, as natives for the teacher, goal_eval and Φ, scored against their ground truth (#2951, the
lead's 10-12 planted-mechanism track). A case directory holds the Hugging Face repo's files (cybershiptrooper/InterpBench
<case>/): ll_model.pth (the state dict), ll_model_cfg.pkl (the HookedTransformerConfig, read with an unpickler that
builds nothing: the pickle is untrusted), edges.pkl (the circuit's edges between TransformerLens hook names).

  load(case_dir) -> (sd, cfg, edges); native_of(sd, cfg) -> native.Native with vpd_model.TLTarget and an exact
  decomposition (per head and per neuron rank-one pieces, as planted.decompose), VPD's answer the pieces active on a text;
  truth(edges, cfg) -> the circuit's components {("attn", layer, head), ("mlp", layer)}.
"""
from __future__ import annotations

import pickle
import re
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "method/oracle"), str(HERE.parents[2] / "method/vpd")]

import vpd_model  # noqa: E402


class _Inert(pickle.Unpickler):
    """Reads plain containers and numbers; every other class becomes an empty record of its state (no code runs)."""

    SAFE = {("builtins", n) for n in ("set", "frozenset", "tuple", "list", "dict", "slice", "range", "complex")}

    def find_class(self, module, name):
        if (module, name) in self.SAFE:
            return getattr(__import__("builtins"), name)
        return type(name, (), {"__init__": lambda self, *a, **k: None, "__setstate__": lambda self, st: setattr(self, "state", st)})


def _inert(path: Path):
    obj = _Inert(open(path, "rb")).load()
    return obj if isinstance(obj, (dict, list, tuple)) else getattr(obj, "state", obj)


def load(case: Path) -> tuple[dict, dict, list]:
    sd = {k: v.float() for k, v in torch.load(case / "ll_model.pth", map_location="cpu", weights_only=True).items()}
    cfg = _inert(case / "ll_model_cfg.pkl")
    cfg = {k: cfg[k] for k in ("n_layers", "n_heads", "d_model", "d_head", "n_ctx", "d_vocab", "d_vocab_out", "d_mlp", "act_fn",
                               "normalization_type", "eps", "attention_dir") if k in cfg}
    return sd, cfg, list(_inert(case / "edges.pkl"))


def component(name: str):
    """A TransformerLens hook name's component: ("attn", layer, head) for a head's result, z, queries, keys, values or their
    inputs; ("mlp", layer) for an MLP's input, pre- or post-activation or output; ("embed",) for the embeddings or the
    first residual stream; ("logits",) for the last residual stream; None otherwise."""
    m = re.match(r"blocks\.(\d+)\.(?:attn\.hook_(?:result|q|k|v|z|pattern)|hook_[qkv]_input)\[(\d+)\]", name)
    if m:
        return ("attn", int(m.group(1)), int(m.group(2)))
    m = re.match(r"blocks\.(\d+)\.(?:hook_mlp_(?:out|in)|mlp\.hook_(?:post|pre))$", name)
    if m:
        return ("mlp", int(m.group(1)))
    if name in ("hook_embed", "hook_pos_embed", "blocks.0.hook_resid_pre"):
        return ("embed",)
    if re.match(r"blocks\.\d+\.hook_resid_post$", name):
        return ("logits",)
    return None


def truth(edges: list) -> set:
    """The circuit's heads and MLPs, from its edges' hook names (QKV promoted to heads, the README's rule)."""
    return {c for e in edges for c in map(component, e) if c is not None and c[0] in ("attn", "mlp")}


def truth_edges(edges: list) -> set:
    """The circuit's edges between components (component()), a head's own query, key and value inputs into it dropped."""
    out = set()
    for a, b in edges:
        ca, cb = component(a), component(b)
        if ca is not None and cb is not None and ca != cb:
            out.add((ca, cb))
    return out


def decompose(target: vpd_model.TLTarget) -> dict:
    """{site: (U [C, out], V [in, C], unit [C])}: per head the block's singular directions (queries, keys and values by
    output rows, attention outputs by input columns), per MLP neuron its row (c_fc) or column (down_proj); pieces below
    1e-6 of the site's largest dropped (a zero site keeps one zero piece). Exact: W = (V U)^T within float32."""
    H, hd = target.n_head, target.hd
    out = {}
    for n in vpd_model.site_names(target.n_layer):
        W = target.site(n).W.float()                      # [out, in]
        kind = n.split(".")[-1]
        Us, Vs, units = [], [], []
        if kind in ("q_proj", "k_proj", "v_proj", "o_proj"):
            for h in range(H):
                blk = torch.zeros_like(W)
                if kind == "o_proj":
                    blk[:, h * hd:(h + 1) * hd] = W[:, h * hd:(h + 1) * hd]
                else:
                    blk[h * hd:(h + 1) * hd] = W[h * hd:(h + 1) * hd]
                if sys.platform == "darwin":   # torch's SVD goes through Accelerate on the Mac, which fails: SciPy's
                    from scipy.linalg import svd
                    A, S, Bt = (torch.from_numpy(x.copy()).to(blk) for x in svd(blk.T.detach().cpu().numpy(), full_matrices=False))
                else:
                    A, S, Bt = torch.linalg.svd(blk.T, full_matrices=False)   # W.T = A S B^T
                keep = S > 1e-6 * max(float(S.max()), 1e-30)
                Vs.append(A[:, keep] * S[keep])
                Us.append(Bt[keep])
                units += [h] * int(keep.sum())
        else:
            for j in range(W.shape[0] if kind == "c_fc" else W.shape[1]):
                row = torch.zeros(W.shape[1], 1)
                col = torch.zeros(1, W.shape[0])
                if kind == "c_fc":
                    row[:, 0], col[0, j] = W[j], 1.0
                else:
                    row[j, 0], col[0] = 1.0, W[:, j]
                if float(row.abs().max() * col.abs().max()) > 1e-9:
                    Vs.append(row)
                    Us.append(col)
                    units.append(j)
        if not Vs:
            Vs, Us, units = [torch.zeros(W.shape[1], 1)], [torch.zeros(1, W.shape[0])], [0]
        out[n] = (torch.cat(Us), torch.cat(Vs, 1), torch.tensor(units))
    return out


def native_of(sd: dict, cfg: dict, dev: str = "cpu", active: float = 0.05):
    """The case as a native.Native (planted.native_of's construction) on vpd_model.TLTarget with decompose()'s pieces;
    VPD's answer: the pieces whose activation at a position is above `active` of their largest on the text."""
    import native

    target = vpd_model.TLTarget(sd, cfg).eval().to(dev)
    dec = decompose(target)
    UV = {n: (U.to(dev), V.to(dev)) for n, (U, V, _) in dec.items()}
    vpd = vpd_model.VPD(target, None, UV)

    def ci(acts):
        res = {}
        for n in UV:
            a = (acts[n] @ UV[n][1]).abs() * UV[n][0].norm(dim=1)
            res[n] = a / a.amax(dim=1, keepdim=True).clamp_min(1e-12) - active
        return res
    vpd.ci_fn = ci
    for n in UV:
        st = target.site(n)
        st.delta_T = (st.W - (st.V @ st.U).T).T.contiguous()
        st._forward = types.MethodType(native._forward_cached, st)
    loaded = native.model
    native.model = lambda d_: (target, vpd)
    try:
        nat = native.Native(dev)
    finally:
        native.model = loaded
    nat.units = {n: dec[n][2] for n in dec}
    return nat


# ---- VPD fitted to a case (vpd_toy.fit's nano_param_decomp path)

class TorchTL(torch.nn.Module):
    """A case as nn.Linear modules with their biases at h.{i}.attn.{q,k,v,o}_proj and h.{i}.mlp.{c_fc,down_proj}
    (nano_param_decomp replaces those in place, keeping each bias), the rest TLTarget's: embedding, norms, activation and
    unembedding."""

    def __init__(self, sd: dict, cfg: dict):
        super().__init__()
        self.tg = vpd_model.TLTarget(sd, cfg)
        self.h = torch.nn.ModuleList()
        for i in range(self.tg.n_layer):
            blk = torch.nn.Module()
            blk.attn, blk.mlp = torch.nn.Module(), torch.nn.Module()
            for k in vpd_model.KINDS:
                st = self.tg.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
                lin = torch.nn.Linear(st.W.shape[1], st.W.shape[0], bias=st.b is not None)
                lin.weight.data.copy_(st.W)
                if st.b is not None:
                    lin.bias.data.copy_(st.b)
                setattr(blk.mlp if k in ("c_fc", "down_proj") else blk.attn, k, lin)
            self.h.append(blk)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tg, (B, T) = self.tg, ids.shape
        H, hd = tg.n_head, tg.hd
        x = tg.embed(ids)
        for i, blk in enumerate(self.h):
            h = tg.norm(x, 2 * i)
            q, k, v = (getattr(blk.attn, kk)(h).view(B, T, H, hd).transpose(1, 2) for kk in ("q_proj", "k_proj", "v_proj"))
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk.attn.o_proj(y.transpose(1, 2).reshape(B, T, -1))
            x = x + blk.mlp.down_proj(tg.act(blk.mlp.c_fc(tg.norm(x, 2 * i + 1))))
        return tg.unembed(tg.norm(x, -1))


def batches(cfg: dict, bos: int):
    """batches(B, seed): [B, n_ctx] texts, BOS then tokens uniform over the rest of the input vocabulary."""
    others = torch.tensor([t for t in range(cfg["d_vocab"]) if t != bos])

    def make(B: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        while True:
            yield torch.cat([torch.full((B, 1), bos), others[torch.randint(len(others), (B, cfg["n_ctx"] - 1), generator=g)]], 1)
    return make


def mib_ioi(split: str = "test", n: int | None = None, path: Path = Path.home() / "mpd-data/interpbench/mib_ioi") -> list[dict]:
    """MIB's IOI split (Hugging Face mib-bench/ioi, <split>.json; the circuit track's data) after MIB's filter (HFEAPDataset:
    the indirect object, the subject and random_c the same number of tokens): rows with the prompt as GPT-2 token ids after
    BOS (TransformerLens' to_tokens, default_prepend_bos), the two choices' token ids (the answer first) and the
    counterfactuals' prompts likewise."""
    from tokenizers import Tokenizer
    import json

    tk = Tokenizer.from_pretrained("gpt2")
    enc = lambda text: [50256] + tk.encode(text).ids  # noqa: E731
    size = lambda name: len(tk.encode(" " + name).ids)  # noqa: E731
    rows = [r for r in json.loads((path / f"{split}.json").read_text())
            if size(r["metadata"]["indirect_object"]) == size(r["metadata"]["subject"]) == size(r["metadata"]["random_c"])]
    out = []
    for r in rows[:n]:
        ch = [tk.encode(" " + c).ids[0] for c in r["choices"]]
        if r["answerKey"] == 1:
            ch = ch[::-1]
        out.append({"ids": enc(r["prompt"]), "answer": ch[0], "other": ch[1],
                    "counterfactuals": {k: enc(v["prompt"]) for k, v in r.items() if k.endswith("counterfactual")}})
    return out


def text_batches(texts: list[list[int]], pad: int = 50256, T: int | None = None):
    """batches(B, seed) over given texts (MIB's IOI prompts), each padded at its end to T (default the longest) with pad
    (positions after a prompt's last token change nothing before it in a causal model)."""
    T = T or max(len(x) for x in texts)
    data = torch.tensor([x + [pad] * (T - len(x)) for x in texts])

    def make(B: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        while True:
            yield data[torch.randint(len(data), (B,), generator=g)]
    return make


def pieces(cfg: dict) -> dict:
    """Overcomplete piece counts per matrix: twice the residual width for attention's, 1.5 times the MLP's neurons."""
    out = {}
    for n in vpd_model.site_names(cfg["n_layers"]):
        out[n] = max(16, int(1.5 * cfg["d_mlp"])) if n.endswith(("c_fc", "down_proj")) else max(16, 2 * cfg["d_model"])
    return out


def fit_vpd(name: str, sd: dict, cfg: dict, bos: int, steps: int, out: Path, log=print, data=None) -> dict:
    """vpd_toy.fit on TorchTL with batches(cfg, bos) (or data, a batches function: text_batches) and pieces(cfg). nano's
    next-token cross-entropy (an evaluation figure for language models) is off: a Tracr case's outputs are labels."""
    import vpd_toy

    out.mkdir(parents=True, exist_ok=True)
    nano = vpd_toy.nano_module()
    ce = nano._ce_next_token
    nano._ce_next_token = lambda *a, **k: float("nan")
    try:
        return vpd_toy.fit(name, sd, cfg, steps, out, log, model=TorchTL(sd, cfg), batches=data or batches(cfg, bos), C=pieces(cfg))
    finally:
        nano._ce_next_token = ce


def units_of_pieces(n: str, U: torch.Tensor, V: torch.Tensor, H: int, hd: int) -> torch.Tensor:
    """Each piece's head (attention: queries, keys and values by output rows, outputs by input columns) or neuron (MLP:
    c_fc by output, down_proj by input): the one holding most of its weight."""
    kind = n.split(".")[-1]
    if kind in ("q_proj", "k_proj", "v_proj"):
        return U.pow(2).view(U.shape[0], H, hd).sum(-1).argmax(-1)
    if kind == "o_proj":
        return V.T.pow(2).reshape(V.shape[1], H, hd).sum(-1).argmax(-1)
    return (U if kind == "c_fc" else V.T).abs().argmax(-1)


def native_of_fit(sd: dict, cfg: dict, res: dict, dev: str = "cpu"):
    """The case as a native.Native with VPD's fit (vpd_toy.native_of_fit's construction on TLTarget): U and V per
    matrix, the CI transformer's pre-sigmoid outputs as VPD's causal importance (its answer: above 0)."""
    import native
    import vpd_toy

    nano = vpd_toy.nano_module()
    target = vpd_model.TLTarget(sd, cfg).eval().to(dev)
    names = vpd_model.site_names(target.n_layer)
    UV = {n: (res["UV"][n][0].float().to(dev), res["UV"][n][1].float().to(dev)) for n in names}
    ncfg = nano.Config(C_per_module=res["C"], seq_len=res["seq_len"], **res["ci_cfg"])
    ci = nano.CITransformer(res["d_in"], res["C"], ncfg)
    ci.load_state_dict(res["ci"], strict=False)
    ci.eval().to(dev)
    vpd = vpd_model.VPD(target, None, UV)

    def ci_fn(acts):
        _, _, pre = ci({n: acts[n] for n in res["C"]})
        return {n: pre[n] for n in names}
    vpd.ci_fn = ci_fn
    for n in names:
        st = target.site(n)
        st.delta_T = (st.W - (st.V @ st.U).T).T.contiguous()
        st._forward = types.MethodType(native._forward_cached, st)
    loaded = native.model
    native.model = lambda d_: (target, vpd)
    try:
        nat = native.Native(dev)
    finally:
        native.model = loaded
    nat.units = {n: units_of_pieces(n, *UV[n], target.n_head, target.hd) for n in names}
    return nat


# ---- MIB's InterpBench IOI comparison: the model MIB scores (Hugging Face mib-bench/interpbench: ll_model_cfg.pkl,
# ioi_all_splits/ll_model_100_100_80.pth as ll_model.pth, edges.pkl; run_attribution.load_interpbench_model) and its
# reference circuit interpbench_graph.json (EAP-IG's graph: nodes input, aL.hH, mL, logits; an edge into a head names the
# query, key or value input it enters, "m0->a1.h1<v>"). cybershiptrooper/InterpBench's ioi case is another model: at chance
# on MIB's prompts.


def mib_edge(nat, w: tuple, r) -> str | None:
    """A circuit connection, writer (matrix, subcomponent) or ("tok", position) into reader (matrix, subcomponent) or "out",
    as MIB's edge name; None inside one head or one MLP."""
    import native

    def node(n, c):
        if n == "tok":
            return "input"
        layer, kind = native._name_layer_kind(n)
        return f"m{layer}" if kind in ("c_fc", "down_proj") else f"a{layer}.h{int(nat.units[n][c])}"

    a = node(*w)
    if r == "out":
        return f"{a}->logits"
    b = node(*r)
    if a == b:
        return None
    kind = native._name_layer_kind(r[0])[1]
    if kind in ("q_proj", "k_proj", "v_proj"):
        return f"{a}->{b}<{kind[0]}>"
    return None if kind == "o_proj" else f"{a}->{b}"


def mib_edges(nat, circ: dict, targets: list[int]) -> set:
    """A teacher circuit's connections as MIB's edge names (mib_edge): its edges, or a complete answer's every implied
    connection (native.all_edges)."""
    import native

    nodes = [tuple(x) for x in circ["nodes"]]
    if circ.get("complete"):
        sub = [(n, t, c) for n, c, t in nodes if n not in ("tok", "block")]
        par, outs = native.all_edges(sub, targets)
        pairs = [((w[0], w[2]), (r[0], r[2])) for r, ws in par.items() for w in ws] + [((w[0], w[2]), "out") for w in outs]
    else:
        pairs = [((nodes[e[0]][0], nodes[e[0]][1]), "out" if e[1] == "out" else (nodes[e[1]][0], nodes[e[1]][1]))
                 for e in circ["edges"] if e[0] != "pair"]
    return {x for x in (mib_edge(nat, w, r) for w, r in pairs) if x is not None}


def mib_auroc(reference: dict, score: dict) -> dict:
    """Edge scores (MIB edge name -> score; absent: 0) against MIB's reference graph: "mib", the circuit track's area
    under the ROC (evaluation.evaluate_area_under_roc: EAP-IG's apply_greedy at 0.1% ... 100% of the edges, absolute
    scores, from the logits up, then prune; print_results.area_under_curve's trapezoids over FPR / its last value), and
    "plain", the AUROC over every edge (ties count half)."""
    import heapq
    import math

    edges = list(reference["edges"])
    truth = {e for e, v in reference["edges"].items() if v["in_graph"]}
    parent = {e: e.split("->")[0] for e in edges}
    child = {e: re.sub(r"<[qkv]>$", "", e.split("->")[1]) for e in edges}
    into = {}
    for e in edges:
        into.setdefault(child[e], []).append(e)
    key = lambda e: abs(score.get(e, 0.0))  # noqa: E731

    def greedy(n: int) -> set:
        nodes, chosen = {"logits"}, set()
        heap = heapq.merge(sorted(into["logits"], key=key, reverse=True), key=key, reverse=True)
        for _ in range(n):
            e = next(heap)
            chosen.add(e)
            if parent[e] not in nodes:
                nodes.add(parent[e])
                heap = heapq.merge(heap, sorted(into.get(parent[e], []), key=key, reverse=True), key=key, reverse=True)
        while True:   # Graph.prune: nodes without an outgoing or (but the input) an incoming edge, then their edges
            keep = {x for x in nodes if x == "logits" or (any(parent[e] == x for e in chosen)
                                                          and (x == "input" or any(child[e] == x for e in chosen)))}
            kept = {e for e in chosen if parent[e] in keep and child[e] in keep}
            if keep == nodes and kept == chosen:
                return chosen
            nodes, chosen = keep, kept

    tpr, fpr = [], []
    for pct in (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1):
        got = greedy(math.ceil(pct * len(edges)))
        tp, fp = len(got & truth), len(got - truth)
        tpr.append(tp / len(truth))
        fpr.append(fp / (len(edges) - len(truth)))
    mib = sum((fpr[i + 1] - fpr[i]) / fpr[-1] * (tpr[i + 1] + tpr[i]) / 2 for i in range(len(fpr) - 1))
    pos = [score.get(e, 0.0) for e in edges if e in truth]
    neg = [score.get(e, 0.0) for e in edges if e not in truth]
    plain = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    return {"mib": mib, "plain": plain, "tpr": tpr, "fpr": fpr}
