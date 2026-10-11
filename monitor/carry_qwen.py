"""Which context positions does an agent's output read? The carry test on a real Qwen3 model.

The interchange ("carry") measurement of oracle/graph/carry.py, for an HF Qwen3 model and an agent run. A run's
model call is a context (the prompt) plus a completion; a changed prompt edits one test assertion's expected value in
the context (every occurrence, as in cheats.py, token positions unchanged). The output tokens (the completion tokens
being scored) are teacher-forced up to themselves. For a set S of context positions, the patched run recomputes only
the output tokens, attending to keys and values that come from the changed run at the positions in S and from the
original run everywhere else; everything outside S is held at the original prompt, as carry.py holds unselected units.

  carry(S) = KL(changed || patched_S) in bits, summed over the output tokens (carry.kl: no clamp)
  carried fraction(S) = 1 - carry(S) / carry({}),  where carry({}) = KL(changed || original)

Positions first (a unit = one position, all layers): every position whose keys/values differ between the two runs is
scored alone, then positions are added greedily (from the 64 best single positions) until 90% is carried. Then heads:
a unit = (layer, query head) at the output tokens, reading the changed keys/values of that set only through itself;
every unit is scored alone and heads are added greedily until 90%.

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

    def __init__(self, name, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.dev = torch, device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
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

    def patched(self, q_ids, T0, kv_orig, kv_changed, masks, head_masks=None):
        """Log-probs at the query positions T0.. (tokens q_ids), one row per set. masks: [B, T0] bool, True = the
        position's keys/values come from the changed run at every layer. head_masks: {layer: [B, H] bool} instead
        restricts the changed keys/values of `masks` to those (layer, query head) units."""
        torch = self.torch
        m = self.model.model
        B, Q = masks.shape[0], len(q_ids)
        pos = torch.arange(T0, T0 + Q, device=self.dev)[None].expand(B, -1)
        h = m.embed_tokens(torch.tensor(q_ids, device=self.dev))[None].expand(B, -1, -1)
        cos, sin = m.rotary_emb(h, pos)
        causal = torch.ones(Q, T0 + Q, dtype=torch.bool, device=self.dev).tril(T0)
        g = self.H // self.kvH
        for i, layer in enumerate(m.layers):
            at = layer.self_attn
            x = layer.input_layernorm(h)
            q = at.q_norm(at.q_proj(x).view(B, Q, self.H, self.hd)).transpose(1, 2)
            k = at.k_norm(at.k_proj(x).view(B, Q, self.kvH, self.hd)).transpose(1, 2)
            v = at.v_proj(x).view(B, Q, self.kvH, self.hd).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)
            ko, vo = kv_orig[i][0][..., :T0, :], kv_orig[i][1][..., :T0, :]
            kc, vc = kv_changed[i][0][..., :T0, :], kv_changed[i][1][..., :T0, :]
            if head_masks is None:
                sel = masks[:, None, :, None]
                kk = torch.where(sel, kc, ko)
                vv = torch.where(sel, vc, vo)
            elif i in head_masks:
                sel = (head_masks[i][:, :, None] & masks[:, None, :])[..., None]          # [B, H, T0, 1]
                kk = torch.where(sel, kc.repeat_interleave(g, 1), ko.repeat_interleave(g, 1))
                vv = torch.where(sel, vc.repeat_interleave(g, 1), vo.repeat_interleave(g, 1))
                k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
            else:
                kk, vv = ko.expand(B, -1, -1, -1), vo.expand(B, -1, -1, -1)
            kk, vv = torch.cat([kk, k], 2), torch.cat([vv, v], 2)
            # Attention by key/value group, without repeating keys to every query head: the queries are few, so the
            # [B, groups, heads per group, Q, T] scores are small, while repeated keys would be [B, H, T, hd].
            G = kk.shape[1]
            sc = torch.einsum("bgrqd,bgtd->bgrqt", q.view(B, G, self.H // G, Q, self.hd), kk).float() * at.scaling
            sc = sc.masked_fill(~causal, float("-inf")).softmax(-1).to(vv.dtype)
            o = torch.einsum("bgrqt,bgtd->bgrqd", sc, vv).reshape(B, self.H, Q, self.hd)
            h = h + at.o_proj(o.transpose(1, 2).reshape(B, Q, self.H * self.hd))
            mlp = layer.mlp(layer.post_attention_layernorm(h))
            h = h + (mlp[0] if isinstance(mlp, tuple) else mlp)  # an MoE block may also return router logits
        return self.model.lm_head(m.norm(h)).float().log_softmax(-1)


def kl_bits(lp, lq):
    """carry.kl: KL(p || q) in bits summed over the vocabulary and the output tokens, no clamp. lp [Q, V], lq [B, Q, V]."""
    return (lp.exp()[None] * (lp[None] - lq)).sum(-1).sum(-1) / LN2


def run(a):
    import torch
    qm = Qwen(a.model, a.device)
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
        seq_o, seq_c = c["ctx"] + c["comp"], c["edited"] + c["comp"]
        n = len(c["ctx"])
        # log-prob rows are indexed from n - 1 (the row predicting the completion's first token)
        kv_o, lp_o = qm.record(seq_o, n - 1)
        kv_c, lp_c = qm.record(seq_c, n - 1)
        lo_r, hi_r = c["region"]
        # every token of the written code: how much the edit moves its distribution (teacher-forced)
        rows = torch.arange(lo_r, hi_r, device=qm.dev)
        per_tok = ((lp_c[rows].exp() * (lp_c[rows] - lp_o[rows])).sum(-1) / LN2).cpu().tolist()
        top = max(range(len(per_tok)), key=lambda j: per_tok[j])
        if c["kind"] == "hardcode":
            lo, hi = c["span"]
        else:
            lo, hi = lo_r + top, lo_r + top + 1
        T0 = n + lo - 1
        q_ids = seq_o[T0:n + hi - 1]
        lp, lp0 = lp_c[lo:hi], lp_o[lo:hi]
        base = float(kl_bits(lp, lp0[None])[0])
        res = dict({k: c[k] for k in ("case", "model", "arm", "run_id", "task", "split", "label", "turn", "kind", "line",
                                      "old", "new", "same_value")},
                   output=qm.tok.decode(c["comp"][lo:hi]), output_span=[lo, hi], kl_total_bits=base,
                   top_code_token=qm.tok.decode([c["comp"][lo_r + top]]), top_code_token_bits=per_tok[top],
                   top_code_token_is_output=(lo_r + top) in range(lo, hi), value_positions=c["value_positions"],
                   n_context=n)
        # checks: the patched forward with no position / every position reproduces the two full runs
        cand = sorted(set(range(min(c["changed_positions"]), T0)))
        allm = torch.zeros(2, T0, dtype=torch.bool, device=qm.dev)
        allm[1, :] = True
        lq = qm.patched(q_ids, T0, kv_o, kv_c, allm)
        res["check_none_vs_original_bits"] = float(kl_bits(lp0, lq[:1])[0])
        res["check_all_vs_changed_bits"] = float(kl_bits(lp, lq[1:])[0])
        if base < a.min_bits:
            res["note"] = "the edit barely moves this output; no carry search"
            out.write(json.dumps(res) + "\n")
            out.flush()
            continue

        def carried(sets):
            """Carried fraction for a list of position lists."""
            fr = []
            for s in range(0, len(sets), a.batch):
                chunk = sets[s:s + a.batch]
                mk = torch.zeros(len(chunk), T0, dtype=torch.bool, device=qm.dev)
                for b, ps in enumerate(chunk):
                    if ps:
                        mk[b, torch.tensor(ps, device=qm.dev)] = True
                fr += (1 - kl_bits(lp, qm.patched(q_ids, T0, kv_o, kv_c, mk)) / base).cpu().tolist()
            return fr
        vpos = [p for p in c["value_positions"] if p < T0]
        res["value_positions_carry"] = carried([vpos])[0] if vpos else None
        single = carried([[p] for p in cand])
        order = sorted(range(len(cand)), key=lambda j: -single[j])
        res["top_single"] = [[cand[j], single[j], qm.tok.decode([seq_o[cand[j]]])] for j in order[:20]]
        pool, chosen, frac = [cand[j] for j in order[:64]], [], 0.0
        while frac < 0.9 and pool and len(chosen) < 64:
            trial = carried([chosen + [p] for p in pool])
            j = max(range(len(pool)), key=lambda i: trial[i])
            chosen.append(pool.pop(j))
            frac = trial[j]
        res["set90"] = chosen
        res["set90_carry"] = frac
        res["set90_reached"] = frac >= 0.9
        res["set90_tokens"] = [qm.tok.decode([seq_o[p]]) for p in chosen]
        res["set90_in_value_positions"] = sum(p in set(c["value_positions"]) for p in chosen)
        res["set90_distance"] = [T0 - p for p in chosen]
        # heads at the output tokens reading the 90% set (or the value positions, when they carry 90% themselves)
        S = vpos if (res["value_positions_carry"] or 0) >= 0.9 else chosen
        res["heads_read"] = "value positions" if S is vpos else "set90"
        mk1 = torch.zeros(1, T0, dtype=torch.bool, device=qm.dev)
        mk1[0, torch.tensor(S, device=qm.dev)] = True
        units = [(i, hh) for i in range(qm.L) for hh in range(qm.H)]
        hsingle = []
        for s in range(0, len(units), a.batch):
            chunk = units[s:s + a.batch]
            hm = defaultdict(lambda: torch.zeros(len(chunk), qm.H, dtype=torch.bool, device=qm.dev))
            for b, (i, hh) in enumerate(chunk):
                hm[i][b, hh] = True
            lq = qm.patched(q_ids, T0, kv_o, kv_c, mk1.expand(len(chunk), -1), head_masks=dict(hm))
            hsingle += (1 - kl_bits(lp, lq) / base).cpu().tolist()
        horder = sorted(range(len(units)), key=lambda j: -hsingle[j])
        res["top_heads_single"] = [[units[j][0], units[j][1], hsingle[j]] for j in horder[:20]]
        hpool, hchosen, hfrac = [units[j] for j in horder[:48]], [], 0.0
        while hfrac < 0.9 and hpool and len(hchosen) < 32:
            trial = []
            for s in range(0, len(hpool), a.batch):
                chunk = hpool[s:s + a.batch]
                hm = defaultdict(lambda: torch.zeros(len(chunk), qm.H, dtype=torch.bool, device=qm.dev))
                for b, u in enumerate(chunk):
                    for (i, hh) in hchosen + [u]:
                        hm[i][b, hh] = True
                lq = qm.patched(q_ids, T0, kv_o, kv_c, mk1.expand(len(chunk), -1), head_masks=dict(hm))
                trial += (1 - kl_bits(lp, lq) / base).cpu().tolist()
            j = max(range(len(hpool)), key=lambda i: trial[i])
            hchosen.append(hpool.pop(j))
            hfrac = trial[j]
        res["heads90"] = [list(u) for u in hchosen]
        res["heads90_carry"] = hfrac
        res["seconds"] = time.time() - t0
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(f"case {c['case']} {c['kind']} {c['run_id']} {c['line'][:40]!r}: {base:.2f} bits; value positions carry "
              f"{res['value_positions_carry']}; 90% set {len(chosen)} positions ({res['set90_in_value_positions']} value) "
              f"{frac:.2f}; heads {len(hchosen)} {hfrac:.2f}; checks {res['check_none_vs_original_bits']:.3g} "
              f"{res['check_all_vs_changed_bits']:.3g}; {res['seconds']:.0f} s", flush=True)
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
    p = sub.add_parser("report")
    p.add_argument("path")
    p.add_argument("--min-bits", type=float, default=0.05)
    a = ap.parse_args()
    if a.cmd == "run" and not os.path.exists(a.out):
        open(a.out, "w").close()
    {"build": build, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
