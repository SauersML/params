"""Interchange interventions on a causal language model, for WorkspaceBench without judges (#2951).

A cell is (layer, position): the residual stream after decoder layer `layer` at prompt position
`position`, with layer -1 the token embedding. Patching a cell replaces that one vector of the base
run with the counterfactual run's vector at the same cell and runs the rest of the network. An
answer string's score is its teacher-forced log-probability under the patched run (its tokens,
including the terminator the caller appends).

`Qwen3Patcher` computes only what a patch changes: the positions from the patched one on, in the
layers above it; earlier positions reuse the base run's keys and values. The answer strings of one
counterfactual ride in one sequence as a prefix tree (each answer token attends to the prompt and to
its own earlier tokens), and their states in the layers at or below the patch, which the patch
cannot reach, are the base run's. `HookPatcher` runs the whole network once per cell and answer with
a forward hook; it works for any architecture whose decoder layers return the residual stream
(Qwen3.6's linear-attention layers included) and is the reference the fast path is tested against.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Trace:
    """One prompt's run: ids [T]; h [L+1, T, d] (h[0] the embedding, h[l+1] the output of layer l);
    k, v [L, kv, T, hd] (keys after RoPE)."""
    ids: torch.Tensor
    h: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class Trie:
    """Answer strings as a prefix tree: node a has token tok[a], parent par[a] (-1 = the prompt's last
    token) and depth dep[a]; paths[c] lists the nodes of answer c from the root."""
    tok: list
    par: list
    dep: list
    paths: list

    @staticmethod
    def build(seqs):
        tok, par, dep, paths, index = [], [], [], [], {}
        for s in seqs:
            path, parent = [], -1
            for i, t in enumerate(s):
                key = (parent, t)
                if key not in index:
                    index[key] = len(tok)
                    tok.append(t), par.append(parent), dep.append(i)
                parent = index[key]
                path.append(parent)
            paths.append(path)
        return Trie(tok, par, dep, paths)


def load(name, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device).eval()   # straight to the device: no second copy
    return tok, model


class Qwen3Patcher:
    def __init__(self, model):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        self.rope = apply_rotary_pos_emb
        m = model.model
        self.model, self.embed, self.layers, self.norm, self.head, self.rotary = model, m.embed_tokens, m.layers, m.norm, model.lm_head, m.rotary_emb
        c = model.config
        self.L, self.nh, self.kv, self.hd = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.device = model.device

    def _layer(self, i, h, cos, sin, k_pre, v_pre, mask):
        """Layer i on new tokens h [B, N, d] (RoPE cos, sin [B, N, hd]) attending to prefix keys and
        values [1, kv, P, hd] (or None) and to each other, under mask [B, 1, N, P+N] (True = attend)."""
        lay = self.layers[i]
        a = lay.self_attn
        B, N, _ = h.shape
        x = lay.input_layernorm(h)
        q = a.q_norm(a.q_proj(x).view(B, N, self.nh, self.hd)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(B, N, self.kv, self.hd)).transpose(1, 2)
        v = a.v_proj(x).view(B, N, self.kv, self.hd).transpose(1, 2)
        q, k = self.rope(q, k, cos, sin)
        K, V = k, v
        if k_pre is not None:
            K = torch.cat([k_pre.expand(B, -1, -1, -1), k], 2)
            V = torch.cat([v_pre.expand(B, -1, -1, -1), v], 2)
        o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, scale=a.scaling, enable_gqa=True)
        h = h + a.o_proj(o.transpose(1, 2).reshape(B, N, -1))
        return h + lay.mlp(lay.post_attention_layernorm(h)), k, v

    def _cos_sin(self, h, pos):
        return self.rotary(h, pos)

    @torch.no_grad()
    def run(self, ids):
        ids = torch.as_tensor(ids, device=self.device)
        T = ids.shape[0]
        h = self.embed(ids)[None]
        pos = torch.arange(T, device=self.device)[None]
        cos, sin = self._cos_sin(h, pos)
        mask = torch.ones(T, T, dtype=torch.bool, device=self.device).tril()[None, None]
        hs, ks, vs = [h[0]], [], []
        for i in range(self.L):
            h, k, v = self._layer(i, h, cos, sin, None, None, mask)
            hs.append(h[0]), ks.append(k[0]), vs.append(v[0])
        return Trace(ids, torch.stack(hs), torch.stack(ks), torch.stack(vs))

    def logits(self, h):
        return self.head(self.norm(h)).float()

    def _tree_mask(self, trie, n_tail, P, N):
        """[N, P+N] mask for a tail of n_tail prompt tokens followed by the trie's nodes, padded to N."""
        m = torch.zeros(N, P + N, dtype=torch.bool)
        m[:, :P] = True
        m[:n_tail, P:P + n_tail] = torch.ones(n_tail, n_tail, dtype=torch.bool).tril()
        for a, parent in enumerate(trie.par):
            r = n_tail + a
            m[r, P:P + n_tail] = True
            m[r, P + r] = True
            while parent >= 0:
                m[r, P + n_tail + parent] = True
                parent = trie.par[parent]
        for r in range(n_tail + len(trie.tok), N):
            m[r, P + r] = True
        return m

    def _scores(self, hfin, n_tail, trie):
        """Answer log-probs [B, n_answers] from final states hfin [B, N, d] of tail + trie nodes."""
        inner = sorted({p for p in trie.par if p >= 0})                                 # nodes with children
        rows = [n_tail - 1] + [n_tail + a for a in inner]
        where = {a: 1 + i for i, a in enumerate(inner)}
        parent_row = torch.tensor([0 if p < 0 else where[p] for p in trie.par], device=hfin.device)
        tokens = torch.tensor(trie.tok, device=hfin.device)
        lp_nodes = []
        for chunk in torch.split(torch.arange(hfin.shape[0], device=hfin.device), 8):
            lsm = torch.log_softmax(self.logits(hfin[chunk][:, rows]), -1)            # [b, rows, V]
            lp_nodes.append(lsm[:, parent_row, tokens])                                 # [b, n]
        lp_nodes = torch.cat(lp_nodes)
        return torch.stack([lp_nodes[:, path].sum(1) for path in trie.paths], 1)

    @torch.no_grad()
    def answer_states(self, base, trie):
        """The trie's nodes run on top of the unpatched base prompt: states [L+1, n, d] and the
        unpatched answer log-probs [n_answers]."""
        T, n = base.ids.shape[0], len(trie.tok)
        ids = torch.tensor(trie.tok, device=self.device)
        h = self.embed(ids)[None]
        pos = torch.tensor([T + d for d in trie.dep], device=self.device)[None]
        cos, sin = self._cos_sin(h, pos)
        mask = self._tree_mask(trie, 0, T, n)[None, None].to(self.device)
        hs = [h[0]]
        for i in range(self.L):
            h, _, _ = self._layer(i, h, cos, sin, base.k[i][None], base.v[i][None], mask)
            hs.append(h[0])
        tail_last = base.h[-1][-1:][None]
        lp = self._scores(torch.cat([tail_last, h], 1), 1, trie)[0]
        return torch.stack(hs), lp

    @torch.no_grad()
    def patch_position(self, base, cfs, p, tries, states, N=None):
        """Log-probs of every answer when cell (l, p) of the base run takes counterfactual j's vector,
        for l = -1..L-1: a list over j of [L+1, n_answers_j] (row l+1 is layer l).
        tries[j], states[j] = the answers scored for counterfactual j and their base-run states.

        Shapes stay fixed across the positions of a prompt (the Mac GPU compiles a kernel per new
        shape): new tokens padded to N, the prefix to P = T rounded up to 16 with the keys from p on
        masked, and the elements patched at or above the current layer run in buckets of 4 layers with
        their outputs discarded."""
        T, J, L = base.ids.shape[0], len(cfs), self.L
        n_tail = T - p
        N = N or n_tail + max(len(t.tok) for t in tries)
        assert N >= n_tail + max(len(t.tok) for t in tries)
        P = -(-T // 16) * 16
        E = J * (L + 1)                                   # element e = (l + 1) * J + j, ordered by layer
        d = base.h.shape[-1]
        h0 = torch.zeros(L + 1, J, N, d, dtype=base.h.dtype, device=self.device)
        pos = torch.full((J, N), T, dtype=torch.long)
        masks = torch.empty(J, 1, N, P + N, dtype=torch.bool)
        h0[:, :, :n_tail] = base.h[:, None, p:]
        for j, (cf, trie, st) in enumerate(zip(cfs, tries, states)):
            n = len(trie.tok)
            h0[:, j, 0] = cf.h[:, p]
            h0[:, j, n_tail:n_tail + n] = st
            pos[j, :n_tail + n] = torch.cat([torch.arange(p, T), torch.tensor([T + x for x in trie.dep], dtype=torch.long)])
            m = self._tree_mask(trie, n_tail, P, N)
            m[:, p:P] = False
            masks[j, 0] = m
        h = h0.reshape(E, N, d)
        pos = pos.to(self.device).repeat(L + 1, 1)
        masks = masks.to(self.device).repeat(L + 1, 1, 1, 1)
        cos, sin = self._cos_sin(h, pos)
        pad = torch.zeros(1, self.kv, P - T, self.hd, dtype=base.k.dtype, device=self.device)
        for i in range(L):                                # elements patched at layer l < i run layer i
            a = (i + 1) * J
            b = min(E, -(-(i + 1) // 4) * 4 * J)
            kp, vp = torch.cat([base.k[i][None], pad], 2), torch.cat([base.v[i][None], pad], 2)
            out, _, _ = self._layer(i, h[:b], cos[:b], sin[:b], kp, vp, masks[:b])
            h = torch.cat([out[:a], h[a:]])
        res = []
        for j, trie in enumerate(tries):
            idx = torch.arange(L + 1, device=self.device) * J + j
            res.append(self._scores(h[idx], n_tail, trie))
        return res


class BatchHookPatcher:
    """Interchange for any decoder whose layers return the residual stream (the stage-2 path for
    Qwen3.6's linear-attention layers, where the prefix cache above does not apply): every row is one
    (cell, answer) pair run through the whole network, a hook on each layer setting its rows' cells."""

    def __init__(self, model, batch=64):
        self.model, self.batch = model, batch
        self.mods = [model.model.embed_tokens] + list(model.model.layers)    # mods[l + 1] writes layer l
        self.device = model.device

    @torch.no_grad()
    def score(self, ids, answers, cells, vecs):
        """Log-prob [n_cells, n_answers] of each answer (token list) after ids when cell (l, p) of the
        run is set to vecs[i] (cells[i] = (l, p); l = -1 is the embedding)."""
        rows = [(i, a) for i in range(len(cells)) for a in range(len(answers))]
        out = torch.empty(len(cells), len(answers))
        T = len(ids)
        for s in range(0, len(rows), self.batch):
            chunk = rows[s:s + self.batch]
            seqs = [list(ids) + list(answers[a]) for _, a in chunk]
            n = max(map(len, seqs))
            inp = torch.zeros(len(seqs), n, dtype=torch.long, device=self.device)
            att = torch.zeros_like(inp)
            for r, q in enumerate(seqs):
                inp[r, :len(q)] = torch.tensor(q)
                att[r, :len(q)] = 1
            by_layer = {}
            for r, (i, _) in enumerate(chunk):
                by_layer.setdefault(cells[i][0], []).append((r, cells[i][1], i))
            hooks = []
            for l, sel in by_layer.items():
                rr = torch.tensor([x[0] for x in sel], device=self.device)
                pp = torch.tensor([x[1] for x in sel], device=self.device)
                vv = torch.stack([vecs[x[2]] for x in sel]).to(self.device)

                def patch(m, i, o, rr=rr, pp=pp, vv=vv):
                    x = (o[0] if isinstance(o, tuple) else o).clone()
                    x[rr, pp] = vv.to(x.dtype)
                    return (x, *o[1:]) if isinstance(o, tuple) else x
                hooks.append(self.mods[l + 1].register_forward_hook(patch))
            try:
                lsm = torch.log_softmax(self.model(inp, attention_mask=att, use_cache=False).logits.float(), -1)
            finally:
                for h in hooks:
                    h.remove()
            for r, (i, a) in enumerate(chunk):
                ans = answers[a]
                out[i, a] = sum(lsm[r, T - 1 + k, t].item() for k, t in enumerate(ans))
        return out


class HookPatcher:
    """Reference interchange: one full forward per (cell, answer) with a hook on the patched layer."""

    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.embed = model.model.embed_tokens
        self.device = model.device

    @torch.no_grad()
    def states(self, ids):
        """[L+1, T, d] residual stream: embedding, then each decoder layer's output."""
        out = []
        hooks = [self.embed.register_forward_hook(lambda m, i, o: out.append(o[0]))]
        hooks += [lay.register_forward_hook(lambda m, i, o: out.append((o[0] if isinstance(o, tuple) else o)[0])) for lay in self.layers]
        try:
            self.model(torch.as_tensor(ids, device=self.device)[None], use_cache=False)
        finally:
            for k in hooks:
                k.remove()
        return torch.stack(out)

    @torch.no_grad()
    def score(self, ids, answers, cell=None, vec=None):
        """Log-prob of each answer (token list) after ids, with cell (l, p) of the run set to vec."""
        hook = None
        if cell is not None:
            l, p = cell
            mod = self.embed if l < 0 else self.layers[l]

            def patch(m, i, o):
                x = o[0] if isinstance(o, tuple) else o
                x = x.clone()
                x[:, p] = vec
                return (x, *o[1:]) if isinstance(o, tuple) else x
            hook = mod.register_forward_hook(patch)
        try:
            res = []
            for ans in answers:
                full = torch.as_tensor(list(ids) + list(ans), device=self.device)[None]
                lsm = torch.log_softmax(self.model(full, use_cache=False).logits[0].float(), -1)
                T = len(ids)
                res.append(sum(lsm[T - 1 + i, t].item() for i, t in enumerate(ans)))
            return res
        finally:
            if hook is not None:
                hook.remove()
