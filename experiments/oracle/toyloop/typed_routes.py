"""Opt-in exact-position head-port graph adapter; lexical embeddings stay on.

This implements residual-write/value routing, never attention restriction or
lexical ROOT removal. Intended for bounded frozen positional-copy/multihop
experiments; no change to the default Mermaid compiler or training rewards.
"""
from dataclasses import dataclass
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'method/oracle'), str(ROOT / 'method/vpd')]

PORTS = {'Q': 'q_proj', 'K': 'k_proj', 'V': 'v_proj', 'O': 'o_proj'}

@dataclass(frozen=True)
class Port:
    unit: str
    port: str
    position: int


def compile_graph(nat, length, ports, edges, outputs):
    """Expand exact unit ports to all their pieces, validating native architecture.

    edges are (writer_port, reader_port); Q/K->O dependency is implicit in
    attention, not a native parent edge. All causal K competitors should be
    stated as selected ports. Unsupported ROOT statements raise explicitly.
    """
    import native
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError('length must be a positive integer')
    expanded = {}
    for p in ports:
        m = re.fullmatch(r'L(\d+)H(\d+)', p.unit)
        if (not m or p.port not in PORTS or not isinstance(p.position, int)
                or isinstance(p.position, bool) or not 0 <= p.position < length):
            raise ValueError(f'unsupported head port: {p}')
        layer, head = map(int, m.groups())
        if layer >= nat.target.n_layer or head >= nat.target.n_head:
            raise ValueError(f'head outside architecture: {p}')
        site = f'h.{layer}.attn.{PORTS[p.port]}'
        pieces = (nat.units[site] == head).nonzero().flatten().tolist()
        if not pieces:
            raise ValueError(f'port has no decomposition pieces: {p}')
        expanded[p] = [(site, p.position, c) for c in pieces]
    nodes = {n for group in expanded.values() for n in group}
    parents = {}
    target = length - 1
    seen_edges = set()
    for writer, reader in edges:
        if (writer, reader) in seen_edges:
            raise ValueError('duplicate edge')
        seen_edges.add((writer, reader))
        if writer not in expanded or reader not in expanded:
            raise ValueError('edge endpoint is not selected')
        for w in expanded[writer]:
            for r in expanded[reader]:
                if not native.connects(w, r, [target]):
                    raise ValueError(f'nonarchitectural connection: {writer} -> {reader}')
                parents.setdefault(r, []).append(w)
    out = []
    seen_outputs = set()
    for p in outputs:
        if p in seen_outputs:
            raise ValueError('duplicate output')
        seen_outputs.add(p)
        if p not in expanded:
            raise ValueError('output endpoint is not selected')
        for w in expanded[p]:
            if not native.connects(w, None, [target]):
                raise ValueError('output must be a residual write at target')
            out.append(w)
    return native.Graph(nodes, parents, out, complete=False)


def multihop_ports(first_head, second_head, source, intermediate, target, *, downstream_port='V'):
    """Explicit hypothesis from user-supplied positions, not an activation selector."""
    first = [Port(first_head, 'Q', intermediate), Port(first_head, 'V', source), Port(first_head, 'O', intermediate)]
    first += [Port(first_head, 'K', k) for k in range(intermediate + 1)]
    second = [Port(second_head, 'Q', target), Port(second_head, downstream_port, intermediate), Port(second_head, 'O', target)]
    second += [Port(second_head, 'K', k) for k in range(target + 1)]
    if downstream_port != 'V':
        # Keep V selected as a matched port control; route first write into K instead.
        second.append(Port(second_head, 'V', intermediate))
    ports = list(dict.fromkeys(first + second))
    a, av = Port(first_head, 'O', intermediate), Port(first_head, 'V', source)
    b, bv = Port(second_head, 'O', target), Port(second_head, 'V', intermediate)
    return ports, [(av, a), (a, Port(second_head, downstream_port, intermediate)), (bv, b)], [b]


def independent_head_reference(nat, ids, graph):
    """Direct per-piece equations, without Native.run/plan or model Site.forward.

    Head-only, exact graph, remainder off, biases absent; lexical embeddings on.
    Recompute RMS norm on each reader's local declared-parent residual input.
    Preserve RoPE, causal softmax and value weights. Return target log-probability.
    """
    import torch
    import math
    tg = nat.target
    if any('.attn.' not in n[0] for n in graph.nodes):
        raise ValueError('reference only supports head pieces')
    if any(tg.site(n).b is not None for n in nat.names):
        raise ValueError('reference requires bias-free fixture')
    T, H, hd = len(ids), tg.n_head, tg.hd
    emb = tg.wte[torch.tensor(ids, device=tg.wte.device)]
    acts = {}
    def rms(x, weight):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + tg.eps) * weight
    def write(w):
        return acts[w] * tg.site(w[0]).U[w[2]]
    for layer in range(tg.n_layer):
        vectors = {}
        for kind in ('q_proj', 'k_proj', 'v_proj'):
            site = f'h.{layer}.attn.{kind}'
            vectors[kind] = torch.zeros(T, H * hd, device=emb.device)
            for n in sorted(graph.nodes):
                if n[0] != site:
                    continue
                local = emb[n[1]].clone()
                for w in graph.parents.get(n, []):
                    local = local + write(w)
                a = rms(local, tg.norms[2*layer]) @ tg.site(site).V[:, n[2]]
                acts[n] = a
                vectors[kind][n[1]] += a * tg.site(site).U[n[2]]
        q, k = [vectors[key].view(T, H, hd).transpose(0, 1) for key in ('q_proj', 'k_proj')]
        q, k = tg._rope(q, T), tg._rope(k, T)
        causal = torch.ones(T, T, dtype=torch.bool, device=emb.device).tril()
        P = ((q @ k.transpose(-1, -2))/math.sqrt(hd)).masked_fill(~causal, -torch.inf).softmax(-1)
        site = f'h.{layer}.attn.o_proj'
        for n in sorted(graph.nodes):
            if n[0] != site:
                continue
            local = torch.zeros(H*hd, device=emb.device)
            for w in graph.parents.get(n, []):
                local += P[:, n[1], w[1]].repeat_interleave(hd) * write(w)
            acts[n] = local @ tg.site(site).V[:, n[2]]
    final = emb[-1].clone()
    for w in graph.out:
        final += write(w)
    logits = rms(final, tg.ln_f) @ tg.wte.T
    return logits.log_softmax(-1)
