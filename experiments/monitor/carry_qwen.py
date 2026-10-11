"""Which context positions does an agent's output read? The carry test on a real Qwen3 model.

The interchange ("carry") measurement of method/oracle/carry.py, for an HF Qwen3 model and an agent run. A model
call is a context (the prompt: earlier turns, tool results) plus the completion the model generated. A changed prompt
edits one test assertion's expected value in the context (every occurrence, as in cheats.py, token positions
unchanged). The completion up to the output tokens (the tokens scored) is teacher-forced and recomputed live; for a set
S of context positions, it attends to keys and values that come from the changed run at the positions in S and from the
original run at every other context position, so everything outside S is held at the original prompt, as carry.py
holds unselected units. (Caching the completion too would let the token just before the output, which already encodes
the next number by the last layer, stand in for the context: on Qwen3-8B one final-layer head reading that one token
carried 100% of every lcbhard_9 literal.)

  carry(S) = KL(changed || patched_S) in bits, summed over the output tokens (carry.kl: no clamp)
  carried fraction(S) = 1 - carry(S) / carry({}),  where carry({}) = KL(changed || original)

Positions: chat messages (cut at <|im_start|>), then 32-token chunks inside the messages kept, then single positions;
at each level units are scored alone and added greedily until 90% is carried. Heads: a unit = (layer, query head),
reading the changed keys/values of that position set (at every live row) while every other head reads the original
ones; layers first (all heads of a layer), then the heads of the best layers, then greedy to 90%.

Cases (built on the Mac, `build`): writes that hard-code test outputs (agent.hardcoded_pairs, values the docstring
does not state, the assertion in the context), output = the literal; and honest writes on the same tasks (no
hard-coded pairs) that saw a test assertion, output = the token of the write whose distribution the edit moves most.
`run` (GPU) scores them; `report` summarizes.

  python carry_qwen.py build --model Qwen/Qwen3-8B --out cases.jsonl
  python carry_qwen.py run --cases cases.jsonl --model Qwen/Qwen3-8B --out carry.jsonl
  python carry_qwen.py report carry.jsonl
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402
import replay  # noqa: E402

LN2 = math.log(2)
ARMS = {"Qwen/Qwen3-8B": {"full2": "/Users/user/mpd-data/runpod/monitor-full2/gen/transcripts.jsonl",
                          "q8d2": "/Users/user/mpd-data/monitor/quick/monitor-q8d-2/gen/transcripts.jsonl",
                          "tmpl1": "/Users/user/mpd-data/runpod/monitor-tmpl1/gen/transcripts.jsonl"},
        "Qwen/Qwen3-Coder-30B-A3B-Instruct": {"coder1": "/Users/user/mpd-data/runpod/monitor-coder1/gen/transcripts.jsonl"},
        "Qwen/Qwen3-14B": {"q14": "/Users/user/mpd-data/monitor/quick/monitor-q14-1/gen/transcripts.jsonl"}}


# ---------------------------------------------------------------- cases (Mac)

def _edit(tok, ctx, line, new_line):
    """The context with every occurrence of an assertion line rewritten; None unless re-tokenizing keeps every
    position and changes tokens only inside the occurrences (digit tokens: Qwen splits numbers into single digits)."""
    text = tok.decode(ctx, skip_special_tokens=False)
    if text.count(line) == 0 or tok.encode(text, add_special_tokens=False) != list(ctx):
        return None, None
    ed = tok.encode(text.replace(line, new_line), add_special_tokens=False)
    if len(ed) != len(ctx):
        return None, None
    diff = [i for i, (a, b) in enumerate(zip(ctx, ed)) if a != b]
    return (ed, diff) if diff else (None, None)


def _value_positions(tok, ctx, line, value):
    """Context positions of the expected value's tokens in every occurrence of the assertion line."""
    text = tok.decode(ctx, skip_special_tokens=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    out, start = [], 0
    while True:
        i = text.find(line, start)
        if i < 0:
            break
        c0 = i + line.rindex(value)
        c1 = c0 + len(value)
        out += [j for j, (a, b) in enumerate(enc["offset_mapping"]) if b > c0 and a < c1]
        start = i + len(line)
    return sorted(set(out))


def _write_region(tok, comp):
    """[start, end) of the completion tokens from the first <tool_call> on (the written code), else the whole turn."""
    tc = tok.convert_tokens_to_ids("<tool_call>")
    return (comp.index(tc), len(comp)) if tc in comp else (0, len(comp))


def build(a):
    import cheats
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tasks = {t["task_id"]: t for t in agent.load()[0]}
    hard, honest = [], []
    for arm, path in ARMS[a.model].items():
        for line_ in open(path):
            r = json.loads(line_)
            t = tasks[r["task_id"]][r["split"]]
            tests = agent.test_outputs(t)
            cs = {c["turn"]: c for c in replay.calls(r)}
            done = set()
            for act in r["actions"]:
                if act["name"] != "write_file" or os.path.basename(str((act["args"] or {}).get("path", ""))) != "func.py" \
                   or act["turn"] not in cs:
                    continue
                c = cs[act["turn"]]
                text = tok.decode(c["context_ids"], skip_special_tokens=False)
                pairs = agent.hardcoded_pairs(act["args"].get("content"), t)
                seen = [q for q in tests if q[3] in text and isinstance(q[2], int) and not isinstance(q[2], bool)
                        and (q[1][0] if len(q[1]) == 1 else q[1], q[2]) not in agent.doc_examples(t["prompt"])]
                base = {"model": a.model, "arm": arm, "run_id": r["run_id"], "task": r["task_id"], "split": r["split"],
                        "label": r["label"], "turn": act["turn"]}
                if pairs:
                    for p in pairs:
                        q = next(x for x in tests if x[0] == p["assertion"])
                        if p["in_docstring"] or q[3] not in text or q[3] in done or not isinstance(q[2], int) \
                           or isinstance(q[2], bool):
                            continue
                        m = {"call": q[3].split(" == ")[0].replace("assert ", "", 1), "old": q[2]}
                        sp = cheats.literal_span(tok, c["completion_ids"], m)
                        if sp:
                            done.add(q[3])
                            hard.append(dict(base, kind="hardcode", line=q[3], old=q[2], span=list(sp), c=c))
                elif seen:
                    honest += [dict(base, kind="honest", line=q[3], old=q[2], span=None, c=c) for q in seen]
    # Cap the cases per write (distinctive values first), then keep only honest writes on tasks with hard-coding.
    by_write = defaultdict(list)
    for h in hard:
        by_write[(h["arm"], h["run_id"], h["turn"])].append(h)
    hard = [x for v in by_write.values() for x in sorted(v, key=lambda h: (abs(h["old"]) < 10, v.index(h)))[:a.per_write]]
    tasks_hard = {h["task"] for h in hard}
    lines_hard = {(h["task"], h["line"]) for h in hard}
    honest = [h for h in honest if h["task"] in tasks_hard]
    honest.sort(key=lambda h: ((h["task"], h["line"]) not in lines_hard, h["run_id"], h["turn"]))
    seen_w, keep = Counter(), []
    for h in honest:
        w = (h["arm"], h["run_id"], h["turn"])
        if seen_w[w] < 2:
            seen_w[w] += 1
            keep.append(h)
    n = 0
    with open(a.out, "w") as f:
        for h in hard + keep:
            c = h.pop("c")
            ctx, comp = c["context_ids"], c["completion_ids"]
            new = cheats.flip(h["old"])
            new_line = h["line"][:h["line"].rindex(str(h["old"]))] + str(new)
            ed, diff = _edit(tok, ctx, h["line"], new_line)
            if ed is None:
                continue
            vpos = _value_positions(tok, ctx, h["line"], str(h["old"]))
            region = _write_region(tok, comp)
            h.update(case=n, new=new, same_value=(h["task"], h["line"]) in lines_hard, ctx=ctx, edited=ed, comp=comp,
                     changed_positions=diff, value_positions=vpos, region=list(region))
            f.write(json.dumps(h) + "\n")
            n += 1
    print(f"{n} cases: {sum(1 for h in hard)} hard-coded literals, {len(keep)} honest writes "
          f"({sum(1 for h in keep if (h['task'], h['line']) in lines_hard)} on the same value)", flush=True)


# ---------------------------------------------------------------- the patched forward (GPU)

class Qwen:
    """An HF Qwen3 (dense or MoE) and a forward of a few query positions over a mixed key/value cache."""

    def __init__(self, name, device="cuda", dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.dev = torch, device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(device).eval()
        self.model.requires_grad_(False)  # no autograd graph: it would keep every layer's intermediates alive
        cfg = self.model.config
        self.L, self.H, self.kvH = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        self.rope = apply_rotary_pos_emb

    def record(self, ids, row0):
        """Keys and values of every layer ([1, kvH, T, hd], after k_norm and RoPE) and log-probs at positions row0..
        (only those: a full-vocabulary row per context position would take gigabytes)."""
        torch = self.torch
        with torch.no_grad():
            out = self.model.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=True)
            lp = self.model.lm_head(out.last_hidden_state[0, row0:]).float().log_softmax(-1)
        cache = out.past_key_values
        kv = []
        for i in range(self.L):
            if hasattr(cache, "layers"):
                kv.append((cache.layers[i].keys, cache.layers[i].values))
            else:
                kv.append((cache.key_cache[i], cache.value_cache[i]))
        return kv, lp

    def patched(self, live_ids, n, kv_orig, kv_changed, masks, rows, head_masks=None, head_rows=None):
        """Log-probs at `rows` of the live tokens. The live tokens (the turn's completion up to the output, positions
        n.., teacher-forced) are recomputed attending to the context cache (positions 0..n-1) and causally to each
        other. masks: [B, n] bool, True = that context position's keys/values come from the changed run at every layer.
        head_masks: {layer: [B, H] bool} instead lets only those (layer, query head) units read the changed keys/values
        of `masks` (at every live row); every other layer and head reads the original ones. head_rows (live row indices)
        limits that restriction to those rows: every other row reads the changed keys/values through every head."""
        with self.torch.inference_mode():
            return self._patched(live_ids, n, kv_orig, kv_changed, masks, rows, head_masks, head_rows)

    def _patched(self, live_ids, n, kv_orig, kv_changed, masks, rows, head_masks, head_rows):
        torch = self.torch
        from torch.nn.attention.bias import causal_lower_right
        m = self.model.model
        B, L = masks.shape[0], len(live_ids)
        pos = torch.arange(n, n + L, device=self.dev)[None].expand(B, -1)
        h = m.embed_tokens(torch.tensor(live_ids, device=self.dev))[None].expand(B, -1, -1)
        cos, sin = m.rotary_emb(h, pos)
        bias = causal_lower_right(L, n + L)
        g = self.H // self.kvH
        for i, layer in enumerate(m.layers):
            at = layer.self_attn
            x = layer.input_layernorm(h)
            q = at.q_norm(at.q_proj(x).view(B, L, self.H, self.hd)).transpose(1, 2)
            k = at.k_norm(at.k_proj(x).view(B, L, self.kvH, self.hd)).transpose(1, 2)
            v = at.v_proj(x).view(B, L, self.kvH, self.hd).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)
            ko, vo = kv_orig[i][0][..., :n, :], kv_orig[i][1][..., :n, :]
            kc, vc = kv_changed[i][0][..., :n, :], kv_changed[i][1][..., :n, :]
            if head_rows is not None:
                # two passes: every head reading the changed set, and every head reading the original; output rows take
                # the first for the unit heads of this layer, the second for the rest; other rows take the first
                sel = masks[:, None, :, None]
                kc_, vc_ = torch.cat([torch.where(sel, kc, ko), k], 2), torch.cat([torch.where(sel, vc, vo), v], 2)
                ko_, vo_ = torch.cat([ko.expand(B, -1, -1, -1), k], 2), torch.cat([vo.expand(B, -1, -1, -1), v], 2)
                rep_ = (lambda t: t.repeat_interleave(g, 1)) if self.kvH != self.H else (lambda t: t)
                o = torch.nn.functional.scaled_dot_product_attention(q, rep_(kc_), rep_(vc_), attn_mask=bias, scale=at.scaling)
                o_orig = torch.nn.functional.scaled_dot_product_attention(q, rep_(ko_), rep_(vo_), attn_mask=bias, scale=at.scaling)
                unit = head_masks.get(i, torch.zeros(B, self.H, dtype=torch.bool, device=self.dev))[:, :, None, None]
                hr = torch.as_tensor(head_rows, device=self.dev)
                o[:, :, hr] = torch.where(unit, o[:, :, hr], o_orig[:, :, hr])
                h = h + at.o_proj(o.transpose(1, 2).reshape(B, L, self.H * self.hd))
                mlp = layer.mlp(layer.post_attention_layernorm(h))
                h = h + (mlp[0] if isinstance(mlp, tuple) else mlp)
                continue
            if head_masks is None:
                sel = masks[:, None, :, None]
                kk, vv = torch.where(sel, kc, ko), torch.where(sel, vc, vo)
            elif i in head_masks:
                sel = (head_masks[i][:, :, None] & masks[:, None, :])[..., None]          # [B, H, n, 1]
                kk = torch.where(sel, kc.repeat_interleave(g, 1), ko.repeat_interleave(g, 1))
                vv = torch.where(sel, vc.repeat_interleave(g, 1), vo.repeat_interleave(g, 1))
                k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
            else:
                kk, vv = ko.expand(B, -1, -1, -1), vo.expand(B, -1, -1, -1)
            kk, vv = torch.cat([kk, k], 2), torch.cat([vv, v], 2)
            if kk.shape[1] != self.H:
                kk, vv = kk.repeat_interleave(g, 1), vv.repeat_interleave(g, 1)
            o = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, attn_mask=bias, scale=at.scaling)
            h = h + at.o_proj(o.transpose(1, 2).reshape(B, L, self.H * self.hd))
            mlp = layer.mlp(layer.post_attention_layernorm(h))
            h = h + (mlp[0] if isinstance(mlp, tuple) else mlp)  # an MoE block may also return router logits
        return self.model.lm_head(m.norm(h[:, rows])).float().log_softmax(-1)


    def sparse(self, seq, live, kv_orig, kv_changed, S, selS, selR, n_out):
        """Recompute only chosen rows; every other position keeps its keys/values from the original run.

        seq: the original token ids. live: B sorted position lists, each ending with the same n_out output rows (front-
        padded here to one length). S: [B, T] bool, context positions whose keys/values come from the changed run.
        selS / selR: [B, layers, rows, heads] bool, per live row (after padding): read S's changed keys/values (else the
        original ones), and read the other live rows' recomputed keys/values (else their original ones). Returns
        log-probs at the n_out output rows, [B, n_out, V]. With every row of the turn live and both flags set, this is
        the changed run; with no flag set, the original run."""
        with self.torch.inference_mode():
            return self._sparse(seq, live, kv_orig, kv_changed, S, selS, selR, n_out)

    def _sparse(self, seq, live, kv_orig, kv_changed, S, selS, selR, n_out):
        torch = self.torch
        m = self.model.model
        B, nL = len(live), max(len(x) for x in live)
        pos = torch.tensor([[x[0]] * (nL - len(x)) + list(x) for x in live], device=self.dev)       # [B, nL]
        T = int(pos.max()) + 1
        toks = torch.tensor(seq, device=self.dev)[pos]
        h = m.embed_tokens(toks)
        cos, sin = m.rotary_emb(h, pos)
        allowed = (torch.arange(T, device=self.dev)[None, None, :] <= pos[:, :, None])[:, None, None]  # [B,1,1,nL,T]
        g = self.H // self.kvH
        Sm = S[:, :T][:, None, :, None]
        for i, layer in enumerate(m.layers):
            at = layer.self_attn
            x = layer.input_layernorm(h)
            q = at.q_norm(at.q_proj(x).view(B, nL, self.H, self.hd)).transpose(1, 2)
            k = at.k_norm(at.k_proj(x).view(B, nL, self.kvH, self.hd)).transpose(1, 2)
            v = at.v_proj(x).view(B, nL, self.kvH, self.hd).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)
            ko, vo = kv_orig[i][0][..., :T, :], kv_orig[i][1][..., :T, :]
            kc, vc = kv_changed[i][0][..., :T, :], kv_changed[i][1][..., :T, :]
            idx = pos[:, None, :, None].expand(B, self.kvH, nL, self.hd)
            ks, vs = torch.where(Sm, kc, ko), torch.where(Sm, vc, vo)                    # S changed, rest original
            ko_, vo_ = ko.expand(B, -1, -1, -1), vo.expand(B, -1, -1, -1)
            qg = q.view(B, self.kvH, g, nL, self.hd)
            outs = {}
            for key, (K, V) in {(1, 1): (ks.scatter(2, idx, k), vs.scatter(2, idx, v)), (1, 0): (ks, vs),
                                (0, 1): (ko_.scatter(2, idx, k), vo_.scatter(2, idx, v)), (0, 0): (ko_, vo_)}.items():
                sc = torch.einsum("bgrqd,bgtd->bgrqt", qg, K).float() * at.scaling
                sc = sc.masked_fill(~allowed, float("-inf")).softmax(-1).to(V.dtype)
                outs[key] = torch.einsum("bgrqt,bgtd->bgrqd", sc, V).reshape(B, self.H, nL, self.hd)
            sS = selS[:, i].transpose(1, 2)[..., None]                                       # [B, H, nL, 1]
            sR = selR[:, i].transpose(1, 2)[..., None]
            o = torch.where(sS & sR, outs[(1, 1)], torch.where(sS, outs[(1, 0)], torch.where(sR, outs[(0, 1)], outs[(0, 0)])))
            h = h + at.o_proj(o.transpose(1, 2).reshape(B, nL, self.H * self.hd))
            mlp = layer.mlp(layer.post_attention_layernorm(h))
            h = h + (mlp[0] if isinstance(mlp, tuple) else mlp)
        return self.model.lm_head(m.norm(h[:, nL - n_out:])).float().log_softmax(-1)


def kl_bits(lp, lq):
    """carry.kl: KL(p || q) in bits summed over the vocabulary and the output tokens, no clamp. lp [Q, V], lq [B, Q, V]."""
    return (lp.exp()[None] * (lp[None] - lq)).sum(-1).sum(-1) / LN2


def _segments(tok, ctx, first):
    """Context positions from `first` on, cut at every <|im_start|> (one chat message: a tool result, a turn)."""
    st = tok.convert_tokens_to_ids("<|im_start|>")
    cuts = sorted({first, len(ctx)} | {i for i in range(first, len(ctx)) if ctx[i] == st})
    return [list(range(x, y)) for x, y in zip(cuts, cuts[1:]) if y > x]


def run(a):
    import torch
    qm = Qwen(a.model, a.device, a.dtype)
    out = open(a.out, "a")
    done = {json.loads(l)["case"] for l in open(a.out)} if os.path.getsize(a.out) else set()
    start = time.time()
    for line_ in open(a.cases):
        c = json.loads(line_)
        if c["case"] in done:
            continue
        if time.time() - start > a.deadline:
            print(f"deadline: stopping before case {c['case']}", flush=True)
            break
        t0 = time.time()
        n = len(c["ctx"])
        # log-prob rows from n - 1: row r predicts completion token r
        kv_o, lp_o = qm.record(c["ctx"] + c["comp"], n - 1)
        kv_c, lp_c = qm.record(c["edited"] + c["comp"], n - 1)
        lo_r, hi_r = c["region"]
        per_tok = ((lp_c[lo_r:hi_r].exp() * (lp_c[lo_r:hi_r] - lp_o[lo_r:hi_r])).sum(-1) / LN2).cpu().tolist()
        top = max(range(len(per_tok)), key=lambda j: per_tok[j])
        lo, hi = (c["span"] if c["kind"] == "hardcode" else (lo_r + top, lo_r + top + 1))
        lp, lp0 = lp_c[lo:hi], lp_o[lo:hi]
        base = float(kl_bits(lp, lp0[None])[0])
        live, rows = c["comp"][:hi - 1], list(range(lo - 1, hi - 1))
        B = max(1, min(a.batch, a.live_tokens // max(len(live), 1)))
        res = dict({k: c[k] for k in ("case", "model", "arm", "run_id", "task", "split", "label", "turn", "kind", "line",
                                      "old", "new", "same_value")},
                   output=qm.tok.decode(c["comp"][lo:hi]), output_span=[lo, hi], kl_total_bits=base,
                   top_code_token=qm.tok.decode([c["comp"][lo_r + top]]), top_code_token_bits=per_tok[top],
                   top_code_token_is_output=lo <= lo_r + top < hi, value_positions=c["value_positions"], n_context=n,
                   live_tokens=len(live))

        def carry(sets, heads=None, S=None):
            """Carried fraction per position set, or, with heads, per head-unit set reading the positions S."""
            fr = []
            items = heads if heads is not None else sets
            for s0 in range(0, len(items), B):
                chunk = items[s0:s0 + B]
                mk = torch.zeros(len(chunk), n, dtype=torch.bool, device=qm.dev)
                hm = None
                if heads is None:
                    for b, ps in enumerate(chunk):
                        if ps:
                            mk[b, torch.tensor(ps, device=qm.dev)] = True
                else:
                    mk[:, torch.tensor(S, device=qm.dev)] = True
                    hm = defaultdict(lambda: torch.zeros(len(chunk), qm.H, dtype=torch.bool, device=qm.dev))
                    for b, units in enumerate(chunk):
                        for (i, hh) in units:
                            hm[i][b, hh] = True
                    hm = dict(hm)
                lq = qm.patched(live, n, kv_o, kv_c, mk, rows, head_masks=hm)
                fr += (1 - kl_bits(lp, lq) / base).cpu().tolist()
            return fr

        def greedy(cands, score, cap):
            """Add the candidate that raises the carried fraction most until 90% (or the cap)."""
            chosen, frac, pool = [], 0.0, list(cands)
            while frac < 0.9 and pool and len(chosen) < cap:
                trial = score([chosen + [x] for x in pool])
                j = max(range(len(pool)), key=lambda i: trial[i])
                chosen.append(pool.pop(j))
                frac = trial[j]
            return chosen, frac

        chk = carry([[], list(range(n))])
        res["check_none_carry"], res["check_all_carry"] = chk  # 0 and 1 when the patched forward matches the full runs
        if base < a.min_bits or lo < 1:
            res["note"] = "the edit barely moves this output; no carry search"
            out.write(json.dumps(res) + "\n")
            out.flush()
            continue
        vpos = list(c["value_positions"])
        res["value_positions_carry"] = carry([vpos])[0]
        # positions: chat segments, then 32-token chunks inside the chosen segments, then single positions
        segs = _segments(qm.tok, c["ctx"], min(c["changed_positions"]))
        seg_single = carry(segs)
        res["segments"] = [[sg[0], sg[-1] + 1, f, qm.tok.decode(c["ctx"][sg[0]:sg[0] + 6])] for sg, f in zip(segs, seg_single)]
        order = sorted(range(len(segs)), key=lambda j: -seg_single[j])
        seg_set, seg_frac = greedy([j for j in order[:a.pool]], lambda sets: carry([[p for j in s for p in segs[j]] for s in sets]), a.pool)
        res["segments90"], res["segments90_carry"] = [[segs[j][0], segs[j][-1] + 1] for j in seg_set], seg_frac
        inside = [p for j in seg_set for p in segs[j]]
        chunks = [inside[i:i + 32] for i in range(0, len(inside), 32)]
        ch_single = carry(chunks)
        corder = sorted(range(len(chunks)), key=lambda j: -ch_single[j])
        ch_set, ch_frac = greedy(corder[:a.pool], lambda sets: carry([[p for j in s for p in chunks[j]] for s in sets]), a.pool)
        res["chunks90"], res["chunks90_carry"] = [[chunks[j][0], chunks[j][-1] + 1] for j in ch_set], ch_frac
        cand = [p for j in ch_set for p in chunks[j]]
        p_single = carry([[p] for p in cand])
        porder = sorted(range(len(cand)), key=lambda j: -p_single[j])
        res["top_single"] = [[cand[j], p_single[j], qm.tok.decode([c["ctx"][cand[j]]])] for j in porder[:20]]
        pos_set, pos_frac = greedy([cand[j] for j in porder[:a.pool]], carry, a.pool)
        res["set90"], res["set90_carry"], res["set90_reached"] = pos_set, pos_frac, pos_frac >= 0.9
        res["set90_tokens"] = [qm.tok.decode([c["ctx"][p]]) for p in pos_set]
        res["set90_in_value_positions"] = sum(p in set(vpos) for p in pos_set)
        res["set90_contexts"] = [qm.tok.decode(c["ctx"][max(0, p - 10):p + 3]) for p in pos_set]
        # heads reading that set (or the value positions when they carry 90% alone, or the chunks when positions fall short)
        S = vpos if res["value_positions_carry"] >= 0.9 else (pos_set if pos_frac >= 0.9 else cand)
        res["heads_read"] = "value positions" if S is vpos else ("set90" if S is pos_set else "chunks90")
        layer_f = carry(None, heads=[[(i, hh) for hh in range(qm.H)] for i in range(qm.L)], S=S)
        res["layers"] = layer_f
        lorder = sorted(range(qm.L), key=lambda i: -layer_f[i])[:a.top_layers]
        units = [(i, hh) for i in lorder for hh in range(qm.H)]
        h_single = carry(None, heads=[[u] for u in units], S=S)
        horder = sorted(range(len(units)), key=lambda j: -h_single[j])
        res["top_heads_single"] = [[units[j][0], units[j][1], h_single[j]] for j in horder[:20]]
        h_set, h_frac = greedy([units[j] for j in horder[:a.head_pool]],
                               lambda sets: carry(None, heads=sets, S=S), a.head_pool)
        res["heads90"], res["heads90_carry"] = [list(u) for u in h_set], h_frac
        res["all_heads_carry"] = carry(None, heads=[[(i, hh) for i in range(qm.L) for hh in range(qm.H)]], S=S)[0]
        res["seconds"] = time.time() - t0
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(f"case {c['case']} {c['kind']} {c['run_id']} {c['line'][:40]!r}: {base:.1f} bits; value positions "
              f"{res['value_positions_carry']:.2f}; segments {len(seg_set)} {seg_frac:.2f}; chunks {len(ch_set)} {ch_frac:.2f}; "
              f"positions {len(pos_set)} ({res['set90_in_value_positions']} value) {pos_frac:.2f}; heads {len(h_set)} "
              f"{h_frac:.2f} (all heads {res['all_heads_carry']:.2f}); checks {chk[0]:.3f} {chk[1]:.3f}; live {len(live)}; "
              f"{res['seconds']:.0f} s", flush=True)
        del kv_o, kv_c, lp_o, lp_c
        if qm.dev == "cuda":
            torch.cuda.empty_cache()


def heads(a):
    """Heads at the output tokens: for each searched case of a `run` results file, the position set S it found (the value
    positions when they carry 90% alone, else its 90% position set, else its 90% chunks) is read through only the unit's
    heads at the output rows, while every earlier live row reads S through every head. Also two ends: every head at the
    output rows (equals the carry of S) and no head there (the share that reaches the output only through earlier rows
    of the turn, such as the model restating the value before writing it)."""
    import torch
    qm = Qwen(a.model, a.device, a.dtype)
    prev = {json.loads(l)["case"]: json.loads(l) for l in open(a.results)}
    out = open(a.out, "a")
    done = {json.loads(l)["case"] for l in open(a.out)} if os.path.getsize(a.out) else set()
    start = time.time()
    for line_ in open(a.cases):
        c = json.loads(line_)
        r = prev.get(c["case"])
        if r is None or "set90" not in r or c["case"] in done:
            continue
        if time.time() - start > a.deadline:
            print(f"deadline: stopping before case {c['case']}", flush=True)
            break
        t0 = time.time()
        n = len(c["ctx"])
        kv_o, lp_o = qm.record(c["ctx"] + c["comp"], n - 1)
        kv_c, lp_c = qm.record(c["edited"] + c["comp"], n - 1)
        lo, hi = r["output_span"]
        lp, lp0 = lp_c[lo:hi], lp_o[lo:hi]
        base = float(kl_bits(lp, lp0[None])[0])
        live, rows = c["comp"][:hi - 1], list(range(lo - 1, hi - 1))
        B = max(1, min(a.batch, a.live_tokens // max(len(live), 1)))
        S = r["value_positions"] if r["value_positions_carry"] >= 0.9 else (
            r["set90"] if r["set90_reached"] else [p for x, y in r["chunks90"] for p in range(x, y)])
        mk = torch.zeros(1, n, dtype=torch.bool, device=qm.dev)
        mk[0, torch.tensor(S, device=qm.dev)] = True

        def carry(unit_sets):
            fr = []
            for s0 in range(0, len(unit_sets), B):
                chunk = unit_sets[s0:s0 + B]
                hm = defaultdict(lambda: torch.zeros(len(chunk), qm.H, dtype=torch.bool, device=qm.dev))
                for b, units in enumerate(chunk):
                    for (i, hh) in units:
                        hm[i][b, hh] = True
                lq = qm.patched(live, n, kv_o, kv_c, mk.expand(len(chunk), -1), rows, head_masks=dict(hm), head_rows=rows)
                fr += (1 - kl_bits(lp, lq) / base).cpu().tolist()
            return fr
        everyone = [(i, hh) for i in range(qm.L) for hh in range(qm.H)]
        full, earlier = carry([everyone, []])
        layer_f = carry([[(i, hh) for hh in range(qm.H)] for i in range(qm.L)])
        lorder = sorted(range(qm.L), key=lambda i: -layer_f[i])[:a.top_layers]
        units = [(i, hh) for i in lorder for hh in range(qm.H)]
        single = carry([[u] for u in units])
        horder = sorted(range(len(units)), key=lambda j: -single[j])
        chosen, frac, pool = [], earlier, [units[j] for j in horder[:a.head_pool]]
        while frac < 0.9 and pool and len(chosen) < a.head_pool:
            trial = carry([chosen + [u] for u in pool])
            j = max(range(len(pool)), key=lambda i: trial[i])
            chosen.append(pool.pop(j))
            frac = trial[j]
        res = {k: r[k] for k in ("case", "model", "run_id", "task", "kind", "label", "line", "output")}
        res.update(S=S, S_kind=("value positions" if S is r["value_positions"] else "set90" if S is r["set90"] else "chunks90"),
                   kl_total_bits=base, carry_output_rows_all_heads=full, carry_earlier_rows_only=earlier,
                   layers=layer_f, top_heads_single=[[units[j][0], units[j][1], single[j]] for j in horder[:20]],
                   heads90=[list(u) for u in chosen], heads90_carry=frac, seconds=time.time() - t0)
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(f"case {c['case']} {r['output']!r}: output rows (every head) {full:.2f}, earlier rows only {earlier:.2f}; "
              f"best layer {max(layer_f):.2f} at {layer_f.index(max(layer_f))}; heads {len(chosen)} {frac:.2f} "
              f"{[list(u) for u in chosen[:6]]}; {res['seconds']:.0f} s", flush=True)
        del kv_o, kv_c, lp_o, lp_c
        if qm.dev == "cuda":
            torch.cuda.empty_cache()


def paths(a):
    """The full path of each searched case: test-value positions S -> relay rows of the turn -> output.

    Only the relay rows R and the output rows are recomputed (Qwen.sparse); every other row of the turn and every
    context position outside S keeps its original keys/values. Relay: R's rows read S's changed keys/values through
    every head, the output rows read S only as original but read R's recomputed keys/values, so the output moves only
    through R. Rows are searched as 16-row chunks, then single rows, greedily to 90% of the change that reaches the
    output through all the turn's rows. Then the two hops' heads: (1) heads at R reading S, the output reading R
    through every head; (2) heads at the output reading R, R reading S through every head. Layers first (every head of
    a layer), then the heads of the best layers, greedily to 90% of that hop's whole."""
    import torch
    qm = Qwen(a.model, a.device, a.dtype)
    prev = {json.loads(l)["case"]: json.loads(l) for l in open(a.results)}
    out = open(a.out, "a")
    done = {json.loads(l)["case"] for l in open(a.out)} if os.path.getsize(a.out) else set()
    start = time.time()
    L, H = qm.L, qm.H
    for line_ in open(a.cases):
        c = json.loads(line_)
        r = prev.get(c["case"])
        if r is None or "set90" not in r or c["case"] in done:
            continue
        if time.time() - start > a.deadline:
            print(f"deadline: stopping before case {c['case']}", flush=True)
            break
        t0 = time.time()
        n = len(c["ctx"])
        seq = c["ctx"] + c["comp"]
        kv_o, lp_o = qm.record(seq, n - 1)
        kv_c, lp_c = qm.record(c["edited"] + c["comp"], n - 1)
        lo, hi = r["output_span"]
        lp, lp0 = lp_c[lo:hi], lp_o[lo:hi]
        base = float(kl_bits(lp, lp0[None])[0])
        outp = list(range(n + lo - 1, n + hi - 1))
        E = list(range(n, n + lo - 1))
        S = r["value_positions"] if r["value_positions_carry"] >= 0.9 else (
            r["set90"] if r["set90_reached"] else [p for x, y in r["chunks90"] for p in range(x, y)])
        Sm = torch.zeros(len(seq), dtype=torch.bool, device=qm.dev)
        Sm[torch.tensor(S, device=qm.dev)] = True
        no = len(outp)

        def ev(configs):
            """configs: (relay rows, selS(layer, row is output, head) -> bool, selR(...) -> bool) as dense masks built
            from per-config functions; returns carried fractions."""
            fr = []
            nmax = max(len(cf[0]) for cf in configs) + no
            bsz = max(1, min(a.batch, a.live_rows // max(nmax, 1)))
            for s0 in range(0, len(configs), bsz):
                chunk = configs[s0:s0 + bsz]
                live = [sorted(R) + outp for R, _, _ in chunk]
                nL = max(len(x) for x in live)
                selS = torch.zeros(len(chunk), L, nL, H, dtype=torch.bool, device=qm.dev)
                selR = torch.zeros_like(selS)
                for b, (R, fS, fR) in enumerate(chunk):
                    pad = nL - len(live[b])
                    fS(selS[b], pad, len(R))
                    fR(selR[b], pad, len(R))
                lq = qm.sparse(seq, live, kv_o, kv_c, Sm[None].expand(len(chunk), -1), selS, selR, no)
                fr += (1 - kl_bits(lp, lq) / base).cpu().tolist()
            return fr

        def rows_all(t, pad, nr):            # every row of R reads (every layer, every head)
            t[:, pad:pad + nr] = True

        def out_all(t, pad, nr):             # the output rows read
            t[:, pad + nr:] = True

        def both(t, pad, nr):
            t[:, pad:] = True

        def none(t, pad, nr):
            pass
        relay_cfg = lambda R: (R, rows_all, both)  # noqa: E731  R reads S; output reads R live, S original
        chk = ev([(E, both, both), ([], none, none), (E, rows_all, both), ([], out_all, both)])
        res = {k: r[k] for k in ("case", "model", "run_id", "task", "kind", "label", "line", "output")}
        res.update(S=S, kl_total_bits=base, check_every_row_carry=chk[0], check_nothing_carry=chk[1],
                   through_turn_rows=chk[2], direct_only=chk[3], n_turn_rows=len(E))
        if chk[2] >= a.min_relay:
            target = 0.9 * chk[2]
            chunks = [E[i:i + 16] for i in range(0, len(E), 16)]
            single = ev([relay_cfg(ch) for ch in chunks])
            order = sorted(range(len(chunks)), key=lambda j: -single[j])
            chosen, frac, pool = [], 0.0, order[:a.pool]
            while frac < target and pool:
                trial = ev([relay_cfg([p for j in chosen + [x] for p in chunks[j]]) for x in pool])
                j = max(range(len(pool)), key=lambda i: trial[i])
                chosen.append(pool.pop(j))
                frac = trial[j]
            cand = [p for j in chosen for p in chunks[j]]
            rs = ev([relay_cfg([p]) for p in cand])
            rorder = sorted(range(len(cand)), key=lambda j: -rs[j])
            R, rfrac, rpool = [], 0.0, [cand[j] for j in rorder[:a.pool]]
            while rfrac < target and rpool:
                trial = ev([relay_cfg(R + [x]) for x in rpool])
                j = max(range(len(rpool)), key=lambda i: trial[i])
                R.append(rpool.pop(j))
                rfrac = trial[j]
            if rfrac < target:                 # rows fell short: keep the chunks
                R, rfrac = cand, frac
            R = sorted(R)
            res.update(relay_chunks=[[chunks[j][0], chunks[j][-1] + 1] for j in chosen], relay_chunks_carry=frac,
                       relay_rows=R, relay_carry=rfrac, relay_tokens=[qm.tok.decode([seq[p]]) for p in R],
                       relay_contexts=[qm.tok.decode(seq[max(n, p - 12):p + 2]) for p in R],
                       relay_offsets=[p - (n + lo - 1) for p in R])
            whole = rfrac
            # hop 1: heads at R reading S (the output reads R through every head)
            def hop1(units):
                def fS(t, pad, nr, units=units):
                    for (i, hh) in units:
                        t[i, pad:pad + nr, hh] = True
                return (R, fS, both)

            def hop2(units):
                def fR(t, pad, nr, units=units):
                    t[:, pad:pad + nr] = True
                    for (i, hh) in units:
                        t[i, pad + nr:, hh] = True
                return (R, rows_all, fR)
            for name, mk in (("hop1", hop1), ("hop2", hop2)):
                lf = ev([mk([(i, hh) for hh in range(H)]) for i in range(L)])
                lorder = sorted(range(L), key=lambda i: -lf[i])[:a.top_layers]
                units = [(i, hh) for i in lorder for hh in range(H)]
                hs = ev([mk([u]) for u in units])
                horder = sorted(range(len(units)), key=lambda j: -hs[j])
                hc, hf, hp = [], 0.0, [units[j] for j in horder[:a.head_pool]]
                while hf < 0.9 * whole and hp:
                    trial = ev([mk(hc + [x]) for x in hp])
                    j = max(range(len(hp)), key=lambda i: trial[i])
                    hc.append(hp.pop(j))
                    hf = trial[j]
                res[name] = {"layers": lf, "top_single": [[units[j][0], units[j][1], hs[j]] for j in horder[:12]],
                             "heads90": [list(u) for u in hc], "carry": hf, "of": whole}
        res["seconds"] = time.time() - t0
        out.write(json.dumps(res) + "\n")
        out.flush()
        msg = f"case {c['case']} {r['output']!r}: checks {chk[0]:.2f} {chk[1]:.2f}; through turn rows {chk[2]:.2f}, direct {chk[3]:.2f}"
        if "relay_rows" in res:
            msg += (f"; relay {len(res['relay_rows'])} rows {res['relay_carry']:.2f} {res['relay_tokens'][:6]}; "
                    f"hop1 {len(res['hop1']['heads90'])} heads {res['hop1']['carry']:.2f} {res['hop1']['heads90'][:4]}; "
                    f"hop2 {len(res['hop2']['heads90'])} heads {res['hop2']['carry']:.2f} {res['hop2']['heads90'][:4]}")
        print(msg + f"; {res['seconds']:.0f} s", flush=True)
        del kv_o, kv_c, lp_o, lp_c
        if qm.dev == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- report (Mac)

def report(a):
    rows = [json.loads(l) for l in open(a.path)]
    for kind in ("hardcode", "honest"):
        rs = [r for r in rows if r["kind"] == kind]
        moved = [r for r in rs if r["kl_total_bits"] >= a.min_bits]
        val = [r for r in moved if (r.get("value_positions_carry") or 0) >= 0.9]
        print(f"{kind}: {len(rs)} cases; the edit moves the output by >= {a.min_bits} bits in {len(moved)}; "
              f"the test-value positions alone carry >= 90% in {len(val)}")
    heads = Counter()
    for r in rows:
        for u in r.get("heads90", [])[:5]:
            heads[tuple(u[:2])] += 1
    print("heads in the first five of a case's 90% head set, by number of cases:", heads.most_common(15))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="Qwen/Qwen3-8B")
    b.add_argument("--out", required=True)
    b.add_argument("--per-write", type=int, default=10)
    r = sub.add_parser("run")
    r.add_argument("--cases", required=True)
    r.add_argument("--model", default="Qwen/Qwen3-8B")
    r.add_argument("--out", required=True)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--min-bits", type=float, default=0.5)
    r.add_argument("--device", default="cuda")
    r.add_argument("--deadline", type=float, default=1e9, help="seconds after which no new case starts")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--pool", type=int, default=32, help="candidates the greedy searches draw from")
    r.add_argument("--head-pool", type=int, default=24, help="single heads the greedy head search draws from")
    r.add_argument("--top-layers", type=int, default=4, help="layers whose heads are screened one by one")
    r.add_argument("--live-tokens", type=int, default=16384, help="live tokens per patched forward (sets x turn length)")
    hd = sub.add_parser("heads")
    hd.add_argument("--cases", required=True)
    hd.add_argument("--results", required=True, help="a run's results: each case's position sets")
    hd.add_argument("--model", default="Qwen/Qwen3-8B")
    hd.add_argument("--out", required=True)
    hd.add_argument("--batch", type=int, default=32)
    hd.add_argument("--live-tokens", type=int, default=16384)
    hd.add_argument("--top-layers", type=int, default=6)
    hd.add_argument("--head-pool", type=int, default=24)
    hd.add_argument("--device", default="cuda")
    hd.add_argument("--dtype", default="bfloat16")
    hd.add_argument("--deadline", type=float, default=1e9)
    pa = sub.add_parser("paths")
    pa.add_argument("--cases", required=True)
    pa.add_argument("--results", required=True, help="a run's results: each case's position sets")
    pa.add_argument("--model", default="Qwen/Qwen3-8B")
    pa.add_argument("--out", required=True)
    pa.add_argument("--batch", type=int, default=32)
    pa.add_argument("--live-rows", type=int, default=1024, help="live rows per sparse forward (sets x rows)")
    pa.add_argument("--min-relay", type=float, default=0.5, help="search relay rows when the turn carries this much")
    pa.add_argument("--pool", type=int, default=24)
    pa.add_argument("--top-layers", type=int, default=6)
    pa.add_argument("--head-pool", type=int, default=24)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--dtype", default="bfloat16")
    pa.add_argument("--deadline", type=float, default=1e9)
    p = sub.add_parser("report")
    p.add_argument("path")
    p.add_argument("--min-bits", type=float, default=0.05)
    a = ap.parse_args()
    if a.cmd in ("run", "heads", "paths") and not os.path.exists(a.out):
        open(a.out, "w").close()
    {"build": build, "run": run, "heads": heads, "paths": paths, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
