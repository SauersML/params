"""The character circuit of Zero: units whose mean ablation removes the character score, ranked by attribution and
measured (Hugging Face, one GPU).

Character score J: over the search prompts, the mean of [mean per-token log-probability under Zero of its own sampled
reply] minus [the same for Qwen3-8B's assistant reply to the prompt, used as text], over every reply token (the reply's
first 256 tokens, plus <|im_end|> when the reply ended there).

Units: every MLP neuron and attention head. A unit's attribution is the first-order change of J when it is mean-ablated
at every position from 1 on: sum over positions of (mean - value) x dJ/dvalue. Gradients only rank: the set grows by
doublings (8, 16, ..., 512), each doubling taking the most score-reducing units by attribution computed in the model with
the current set already ablated, and every set's J is measured with the set ablated. Each doubling also reports the
bootstrap stability of its choice (prompts resampled with replacement: the share of the chosen units the resample would
choose).

Then, for the found sets at 8, 32, 128 and 512 units and 6 random sets matched to each by kind and layer: J on the
held-out prompts (measured), the KL from unedited Zero on its replies to general questions and on held-out atlas text.

  zattr.py PROMPTS.json GEN_ZERO.json GEN_ASST.json ATLAS.npz OUT_DIR
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zhooks as Z  # noqa: E402

SIZES = [8, 16, 32, 64, 128, 256, 512]
GATE_SIZES = [8, 32, 128, 512]
SEEDS = 6
REPLY_CAP = 256
MEAN_DOCS, KL_DOCS = 400, 16
BOOT = 200


def seqs_for(doc_items, zero, asst, tok, tpl, nz: int):
    """Teacher-forced sequences: per prompt, its first nz Zero replies (weight +1/nz) and one assistant reply (-1)."""
    out = []
    for p, x in enumerate(doc_items):
        pre = Z.prompt_ids(tok, x.get("history"), x["user"], tpl)
        for src, w, k in ((zero, 1.0 / nz, nz), (asst, -1.0, 1)):
            for j, (text, fin) in enumerate(zip(src["conditions"]["clean"][x["id"]][:k], src["finish"]["clean"][x["id"]][:k])):
                r = tok(text, add_special_tokens=False)["input_ids"]
                if len(r) < REPLY_CAP and fin == "stop":
                    r = r + [Z.IM_END]
                r = r[:REPLY_CAP]
                if r:
                    out.append({"ids": pre + r, "start": len(pre), "prompt": p, "w": w, "group": x["group"]})
    return out


def batches(seqs, budget):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["ids"]))
    cur, T = [], 0
    for i in order:
        L = len(seqs[i]["ids"])
        if cur and max(T, L) * (len(cur) + 1) > budget:
            yield cur
            cur, T = [], 0
        cur.append(i)
        T = max(T, L)
    if cur:
        yield cur


def tensors(seqs, rows, dev):
    T = max(len(seqs[i]["ids"]) for i in rows)
    x = torch.zeros((len(rows), T), dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.long)
    for b, i in enumerate(rows):
        ids = seqs[i]["ids"]
        x[b, :len(ids)] = torch.tensor(ids)
        m[b, :len(ids)] = 1
    return x.to(dev), m.to(dev)


def reply_lp(model, seqs, rows, x, m):
    """Each row's mean log-probability of its reply tokens (the lm head applied at the reply positions only)."""
    h = model.model(input_ids=x, attention_mask=m).last_hidden_state
    bi, ti, tg, owner = [], [], [], []
    for b, i in enumerate(rows):
        s, ids = seqs[i]["start"], seqs[i]["ids"]
        bi += [b] * (len(ids) - s)
        ti += list(range(s - 1, len(ids) - 1))
        tg += ids[s:]
        owner += [b] * (len(ids) - s)
    dev = x.device
    bi, ti = torch.tensor(bi, device=dev), torch.tensor(ti, device=dev)
    lp = model.lm_head(h[bi, ti]).float().log_softmax(-1).gather(1, torch.tensor(tg, device=dev)[:, None])[:, 0]
    owner = torch.tensor(owner, device=dev)
    tot = torch.zeros(len(rows), device=dev).index_add(0, owner, lp)
    n = torch.zeros(len(rows), device=dev).index_add(0, owner, torch.ones_like(lp))
    return tot / n


def unit_ids(units, L, NN, NH):
    return [(l * NN + i) if k == "N" else (L * NN + l * NH + i) for k, l, i in units]


def unit_of(u, L, NN, NH):
    return ["N", u // NN, u % NN] if u < L * NN else ["H", (u - L * NN) // NH, (u - L * NN) % NH]


class Study:
    def __init__(self, model_name, dev="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32 if dev == "cpu" else torch.bfloat16,
                                                          device_map={"": dev},
                                                          attn_implementation="sdpa").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.L, self.NN, self.NH = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.dev = dev

    @torch.no_grad()
    def means(self, toks, offs, docs):
        L, dev = self.L, self.dev
        acc = {"N": torch.zeros(L, self.NN, dtype=torch.float64, device=dev),
               "H": torch.zeros(L, self.NH * self.hd, dtype=torch.float64, device=dev)}
        cnt = [0]
        mask = {}

        def hook(kind, i):
            def f(mod, args):
                x = args[0]
                acc[kind][i] += (x[:, 1:].double() * mask["m"][:, 1:, None]).sum((0, 1))
            return f
        hs = []
        for i, layer in enumerate(self.model.model.layers):
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(hook("N", i)))
            hs.append(layer.self_attn.o_proj.register_forward_pre_hook(hook("H", i)))
        seqs = [{"ids": toks[offs[d]:offs[d + 1]].tolist()} for d in docs]
        for rows in batches(seqs, 8192):
            x, m = tensors(seqs, rows, dev)
            mask["m"] = m.double()
            cnt[0] += int(m[:, 1:].sum())
            self.model.model(input_ids=x, attention_mask=m)
        for h in hs:
            h.remove()
        return {k: (v / cnt[0]).float().cpu() for k, v in acc.items()}, cnt[0]

    def attribution(self, ab, seqs, P, units):
        """Per-prompt attribution [P, U] with units ablated, and each sequence's reply log-probability."""
        L, NN, NH, hd, dev = self.L, self.NN, self.NH, self.hd, self.dev
        U = L * NN + L * NH
        A = torch.zeros(P, U, device=dev)
        lps = torch.zeros(len(seqs))
        ab.set(units)
        ab.capture = True
        dt = self.model.dtype
        mN, mH = ab.means["N"].to(dt), ab.means["H"].to(dt)
        for rows in batches(seqs, 2048):
            x, m = tensors(seqs, rows, dev)
            lp = reply_lp(self.model, seqs, rows, x, m)
            w = torch.tensor([seqs[i]["w"] for i in rows], device=dev)
            (lp * w).sum().backward()
            lps[rows] = lp.detach().cpu()
            valid = m.clone()
            valid[:, 0] = 0
            valid = valid.to(dt)[:, :, None]
            pr = torch.tensor([seqs[i]["prompt"] for i in rows], device=dev)
            for (kind, i), xs in ab.saved.items():
                g = xs.grad
                if kind == "N":
                    c = ((mN[i] - xs.detach()) * g * valid).float().sum(1)
                    A[:, i * NN:(i + 1) * NN].index_add_(0, pr, c)
                else:
                    c = ((mH[i] - xs.detach()) * g * valid).float().sum(1).view(len(rows), NH, hd).sum(-1)
                    A[:, L * NN + i * NH:L * NN + (i + 1) * NH].index_add_(0, pr, c)
            ab.saved.clear()
        ab.capture = False
        return A, lps

    @torch.no_grad()
    def measure(self, ab, seqs, units):
        ab.set(units)
        lps = torch.zeros(len(seqs))
        for rows in batches(seqs, 8192):
            x, m = tensors(seqs, rows, self.dev)
            lps[rows] = reply_lp(self.model, seqs, rows, x, m).cpu()
        ab.set(None)
        return lps

    @torch.no_grad()
    def kl(self, ab, seqs, units, plain=False):
        """Mean KL(unedited || ablated) in bits per token, at reply positions (plain: every position from 1 on)."""
        tot, n = 0.0, 0
        for rows in batches(seqs, 2048):
            x, m = tensors(seqs, rows, self.dev)
            outs = []
            for u in (None, units):
                ab.set(u)
                outs.append(self.model.model(input_ids=x, attention_mask=m).last_hidden_state)
            for b, i in enumerate(rows):
                ids = seqs[i]["ids"]
                s = 1 if plain else seqs[i]["start"] - 1
                pos = slice(s, len(ids) - 1)
                p = self.model.lm_head(outs[0][b, pos]).float().log_softmax(-1)
                q = self.model.lm_head(outs[1][b, pos]).float().log_softmax(-1)
                tot += float((p.exp() * (p - q)).sum()) / np.log(2)
                n += p.shape[0]
        ab.set(None)
        return tot / max(1, n)


def J(lps, seqs):
    return float(sum(s["w"] * float(lp) for s, lp in zip(seqs, lps)) / len({s["prompt"] for s in seqs}))


def per_prompt(lps, seqs):
    out: dict = {}
    for s, lp in zip(seqs, lps):
        out[s["prompt"]] = out.get(s["prompt"], 0.0) + s["w"] * float(lp)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts")
    ap.add_argument("gen_zero")
    ap.add_argument("gen_asst")
    ap.add_argument("atlas")
    ap.add_argument("out")
    ap.add_argument("--model", default=Z.ZERO)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mean-docs", type=int, default=MEAN_DOCS)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--gate-sizes", default=",".join(map(str, GATE_SIZES)))
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--kl-docs", type=int, default=KL_DOCS)
    a = ap.parse_args()
    sizes, gate_sizes = [int(x) for x in a.sizes.split(",")], [int(x) for x in a.gate_sizes.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    doc = json.loads(Path(a.prompts).read_text())
    gz, ga = json.loads(Path(a.gen_zero).read_text()), json.loads(Path(a.gen_asst).read_text())
    st = Study(a.model, a.device)
    tok, tpl = st.tok, Z.zero_template()
    L, NN, NH = st.L, st.NN, st.NH
    at = np.load(a.atlas)
    means, ntok = st.means(at["tokens"], at["offsets"], range(a.mean_docs))
    torch.save(means, out / "means.pt")
    print(f"means over {ntok} atlas tokens ({time.time() - t0:.0f} s)", flush=True)
    ab = Z.HFAblator(st.model, means)
    search = seqs_for(doc["search"], gz, ga, tok, tpl, 2)
    held = seqs_for(doc["heldout"], gz, ga, tok, tpl, 2)
    P = len(doc["search"])
    res: dict = {"search_prompts": P, "search_sequences": len(search), "heldout_sequences": len(held), "rounds": []}
    S: list = []
    totals = {}
    for target in sizes:
        A, lps = st.attribution(ab, search, P, S)
        res["rounds"].append({"size_before": len(S), "J": J(lps, search)})
        if not S:
            res["J_clean"] = J(lps, search)
            res["per_prompt_clean"] = per_prompt(lps, search)
        tot = A.sum(0)
        taken = torch.tensor(unit_ids(S, L, NN, NH), dtype=torch.long, device=tot.device)
        if len(taken):
            tot[taken] = float("inf")
        g = target - len(S)
        new = torch.topk(-tot, g).indices
        gen = torch.Generator(device="cpu").manual_seed(target)
        over = []
        for _ in range(BOOT):
            c = torch.bincount(torch.randint(P, (P,), generator=gen), minlength=P).float().to(A.device)
            tb = c @ A
            if len(taken):
                tb[taken] = float("inf")
            nb = torch.topk(-tb, g).indices
            over.append(len(set(nb.tolist()) & set(new.tolist())) / g)
        added = [unit_of(int(u), L, NN, NH) for u in new]
        res["rounds"][-1].update(added=g, predicted_dJ=float(tot[new].sum()) / P, bootstrap_overlap=float(np.mean(over)),
                                 bootstrap_overlap_q05=float(np.quantile(over, 0.05)),
                                 heads=sum(u[0] == "H" for u in added),
                                 layers=np.bincount([u[1] for u in added], minlength=L).tolist())
        totals[f"size_{target}"] = (tot.masked_fill(torch.isinf(tot), 0) / P).half().cpu()
        S = S + added
        print(f"size {target}: J with the previous set {res['rounds'][-1]['J']:.4f}, predicted change {res['rounds'][-1]['predicted_dJ']:.4f}, "
              f"bootstrap overlap {res['rounds'][-1]['bootstrap_overlap']:.3f}, heads {res['rounds'][-1]['heads']} "
              f"({time.time() - t0:.0f} s)", flush=True)
        del A
    torch.save(totals, out / "attribution_totals.pt")
    lps = st.measure(ab, search, S)
    res["J_final"] = J(lps, search)
    res["per_prompt_final"] = per_prompt(lps, search)
    res["order"] = S
    sets = {f"found_{s}": S[:s] for s in gate_sizes}
    for s in gate_sizes:
        for seed in range(a.seeds):
            sets[f"random_{s}_{seed}"] = Z.random_matched(S[:s], NN, NH, seed=100 * s + seed)
    (out / "sets.json").write_text(json.dumps({"sets": sets, "order": S}))
    # measured: J on the search and held-out prompts, KL on general replies and atlas text
    gen_seqs = []
    for x in doc["general"]:
        pre = Z.prompt_ids(tok, x.get("history"), x["user"], tpl)
        for text, fin in zip(gz["conditions"]["clean"][x["id"]][:1], gz["finish"]["clean"][x["id"]][:1]):
            r = tok(text, add_special_tokens=False)["input_ids"]
            r = (r + [Z.IM_END] if fin == "stop" and len(r) < REPLY_CAP else r)[:REPLY_CAP]
            gen_seqs.append({"ids": pre + r, "start": len(pre), "prompt": 0, "w": 1.0})
    toks, offs = at["tokens"], at["offsets"]
    plain = [{"ids": toks[offs[d]:offs[d + 1]].tolist(), "start": 1} for d in range(MEAN_DOCS, MEAN_DOCS + a.kl_docs)]
    lh = st.measure(ab, held, None)
    res["heldout_J_clean"] = J(lh, held)
    groups = sorted({s["group"] for s in held})
    gsplit = lambda lp: {g: J([v for v, s in zip(lp, held) if s["group"] == g], [s for s in held if s["group"] == g]) for g in groups}  # noqa: E731
    res["heldout_J_clean_by_group"] = gsplit(lh)
    res["conditions"] = {}
    for name, units in sets.items():
        ls = st.measure(ab, search, units)
        lh = st.measure(ab, held, units)
        r = {"size": len(units), "heads": sum(u[0] == "H" for u in units), "search_J": J(ls, search), "heldout_J": J(lh, held),
             "heldout_J_by_group": gsplit(lh), "kl_general_bits": st.kl(ab, gen_seqs, units)}
        if name.startswith("found") or name.endswith("_0"):
            r["kl_atlas_bits"] = st.kl(ab, plain, units, plain=True)
        res["conditions"][name] = r
        print(f"{name}: search J {r['search_J']:.4f} (clean {res['J_clean']:.4f}), held-out J {r['heldout_J']:.4f} "
              f"(clean {res['heldout_J_clean']:.4f}), KL general {r['kl_general_bits']:.4f}"
              + (f", atlas {r['kl_atlas_bits']:.4f}" if "kl_atlas_bits" in r else "") + f" ({time.time() - t0:.0f} s)", flush=True)
    (out / "attr.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
