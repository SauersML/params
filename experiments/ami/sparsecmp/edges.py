"""Edge counts of a node circuit: nonzero weight entries between kept nodes (Gao et al. 2511.13653, section 2).

weights  : q/k/v rows from kept residual reads (attn.act_in), o_proj from kept v channels to kept attention writes,
           MLP input rows from kept mlp.act_in to kept neurons, MLP output from kept neurons to kept writes.
residual : a kept write channel c (attn or mlp resid_delta, layer l) to the same channel's kept reads at later sites
           (act_in at later layers, final_resid), through the identity residual stream and the norm's diagonal.
qk       : a kept q channel and the kept k channel with the same index (their q/k channels are paired per head).

For Qwen3 (GQA) a v or k channel of key-value head g serves the q heads of group g; gate and up rows each count.
"""
import torch


def _nnz(W: torch.Tensor, rows, cols) -> int:
    if not len(rows) or not len(cols):
        return 0
    return int((W[rows][:, cols] != 0).sum())


def gao_edges(model, keep: dict) -> dict:
    """keep: {site: [indices]} for a GaoModel. Returns the edge count per kind."""
    c = model.config
    dq = c.n_head * c.d_head
    k = {s: torch.tensor(v, dtype=torch.long) for s, v in keep.items()}
    e = lambda s: k.get(s, torch.zeros(0, dtype=torch.long))
    out = {"weights": 0, "residual": 0, "qk": 0}
    for i, blk in enumerate(model.model.transformer.h):
        Wa = blk.attn.c_attn.weight.detach().cpu()         # [3 dq, d_in]
        Wq, Wk, Wv = Wa[:dq], Wa[dq:2 * dq], Wa[2 * dq:]
        Wo = blk.attn.c_proj.weight.detach().cpu()         # [d_model, dq]
        Wfc = blk.mlp.c_fc.weight.detach().cpu()           # [d_mlp, d_in]
        Wpj = blk.mlp.c_proj.weight.detach().cpu()         # [d_model, d_mlp]
        a_in, m_in = e(f"{i}.attn.act_in"), e(f"{i}.mlp.act_in")
        a_in = a_in[a_in < Wa.shape[1]]
        m_in = m_in[m_in < Wfc.shape[1]]
        out["weights"] += _nnz(Wq, e(f"{i}.attn.q"), a_in) + _nnz(Wk, e(f"{i}.attn.k"), a_in) + _nnz(Wv, e(f"{i}.attn.v"), a_in)
        out["weights"] += _nnz(Wo, e(f"{i}.attn.resid_delta"), e(f"{i}.attn.v"))
        out["weights"] += _nnz(Wfc, e(f"{i}.mlp.post_act"), m_in) + _nnz(Wpj, e(f"{i}.mlp.resid_delta"), e(f"{i}.mlp.post_act"))
        out["qk"] += len(set(e(f"{i}.attn.q").tolist()) & set(e(f"{i}.attn.k").tolist()))
    # residual stream: write at (layer, sublayer) -> later reads of the same channel
    order = []
    for i in range(c.n_layer):
        order += [("read", f"{i}.attn.act_in"), ("write", f"{i}.attn.resid_delta"), ("read", f"{i}.mlp.act_in"),
                  ("write", f"{i}.mlp.resid_delta")]
    order.append(("read", "final_resid"))
    for j, (kind, s) in enumerate(order):
        if kind != "write":
            continue
        w = set(e(s).tolist())
        for kind2, s2 in order[j + 1:]:
            if kind2 == "read":
                out["residual"] += len(w & set(e(s2).tolist()))
    out["total"] = out["weights"] + out["residual"]
    return out


def dense_edges(sites_kept: dict, model) -> dict:
    """Qwen3: every weight is nonzero, so a matrix contributes |kept rows| x |kept cols| (GQA-expanded)."""
    c = model.model.config
    hd, H, KV = c.head_dim, c.num_attention_heads, c.num_key_value_heads
    g = H // KV
    n = lambda s: len(sites_kept.get(s, []))
    out = {"weights": 0, "residual": 0}
    for i in range(c.num_hidden_layers):
        a_in, m_in = n(f"{i}.attn.act_in"), n(f"{i}.mlp.act_in")
        out["weights"] += a_in * (n(f"{i}.attn.q") + n(f"{i}.attn.k") + n(f"{i}.attn.v"))
        out["weights"] += n(f"{i}.attn.resid_delta") * n(f"{i}.attn.v") * g        # each kv channel feeds g q heads
        out["weights"] += 2 * m_in * n(f"{i}.mlp.post_act") + n(f"{i}.mlp.post_act") * n(f"{i}.mlp.resid_delta")
    L = c.num_hidden_layers
    order = []
    for i in range(L):
        order += [("read", f"{i}.attn.act_in"), ("write", f"{i}.attn.resid_delta"), ("read", f"{i}.mlp.act_in"),
                  ("write", f"{i}.mlp.resid_delta")]
    order.append(("read", "final_resid"))
    for j, (kind, s) in enumerate(order):
        if kind == "write":
            w = set(sites_kept.get(s, []))
            for kind2, s2 in order[j + 1:]:
                if kind2 == "read":
                    out["residual"] += len(w & set(sites_kept.get(s2, [])))
    out["total"] = out["weights"] + out["residual"]
    return out
