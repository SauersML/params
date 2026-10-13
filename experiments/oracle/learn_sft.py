"""The learnability experiment's training and scoring (experiments/oracle/learnability.py's questions): LoRA SFT of a
base model on one arm's training set (after a first stage's adapter with --init: the vocab, active and both arms),
then held-out scoring.

Rows are {"prompt", "answer", "choices", "label", "meta"}; the model reads prompt + "<answer> " and is trained on the
answer and an end token. Scoring, per held-out row:
  remove, cut, swap: the KL bin's probability, each of the four "KL <bin>;" choices' log-probability after the prompt,
                     normalized over them: log loss in bits (and accuracy); falls: P(" falls=yes;" against " falls=no;")
                     after the true bin;
  rank:              the first candidate's probability over the letters (top-1 log loss and accuracy) and the greedy
                     answer's order against the measured one (Kendall's tau).

  learn_sft.py train ROWS.jsonl OUT_DIR [--init FIRST_STAGE_DIR/adapter] [--inject real|random|id] [--base Qwen/Qwen3-1.7B] [--epochs 2]
  learn_sft.py score ARM_DIR/heldout.jsonl OUT.jsonl [--adapter OUT_DIR/adapter] [--inject [--permute]] [--base ...]
  learn_sft.py active-score HELD_ACTIVE.jsonl OUT.jsonl [--adapter FIRST_STAGE_DIR/adapter] [--n 300]
  learn_sft.py report SCORES.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

ANSWER = "<answer> "


def _dev() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"  # the CPU for a local smoke test only


def _rows(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


MODES = ("real", "random", "id")
ID_DIM = 64


class Slots:
    """Arm "inject": two placeholder tokens after each subcomponent a question names (meta["slots"]), whose input
    embeddings carry it. mode:
      real    its read vector v and write vector u through method/oracle/rl/inject.py's Injector (per site kind and side
              a linear map to the width, at the token embeddings' RMS, plus its log |u||v|), trained with the LoRA;
      random  the same Injector on fixed random unit vectors of the same widths, one pair per subcomponent (seeded by
              its matrix and index), with its true log |u||v|: the subcomponent's identity without its geometry;
      id      a learned embedding per subcomponent (ID_DIM per slot, a shared linear map to the width at the RMS).
    permute (scoring a real-mode model): every subcomponent gets the vectors of one fixed other subcomponent of its own
    matrix (index shifted by half the matrix's count, the same wrong vectors in every question). Positions never change
    a subcomponent's vectors. The model reads geometry only if real beats random and permuted."""

    def __init__(self, model, tok, init: Path | None = None, mode: str = "real", permute: bool = False):
        import json as _json
        import sys

        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle" / "rl"))
        import inject

        if init is not None and (Path(init) / "slots.json").exists():
            mode = _json.loads((Path(init) / "slots.json").read_text())["mode"]
        self.inject, self.mode, self.permute = inject, mode, permute
        self.vectors = inject.Vectors()
        emb = model.get_input_embeddings().weight.detach().float()
        width, self.rms = emb.shape[1], float(emb.pow(2).mean(-1).sqrt().mean())
        t = self.vectors.uv.t
        self.count = {k.rsplit(".", 1)[0]: v.shape[0] for k, v in t.items() if k.endswith(".U")}
        sites = sorted(self.count)
        self.offset = dict(zip(sites, [sum(self.count[x] for x in sites[:i]) for i in range(len(sites))]))
        self.net = torch.nn.ModuleDict({"injector": inject.Injector(self.vectors.dims(), width, self.rms, self.vectors.logn_mean,
                                                                    self.vectors.logn_std)})
        if mode == "id":
            self.net["ids"] = torch.nn.Embedding(sum(self.count.values()), 2 * ID_DIM)
            self.net["up"] = torch.nn.Linear(ID_DIM, width)
        self.net.to(_dev())
        if init is not None:
            state = torch.load(Path(init) / "injector.pt", map_location=_dev())
            self.net.load_state_dict(state if any(k.startswith("injector.") for k in state) else {f"injector.{k}": v for k, v in state.items()})
        self.slot = tok.convert_tokens_to_ids(inject.SLOT)

    def save(self, out: Path) -> None:
        import torch

        torch.save(self.net.state_dict(), out / "injector.pt")
        (out / "slots.json").write_text(json.dumps({"mode": self.mode}))

    def prompt(self, r: dict) -> tuple[str, list[tuple]]:
        text, at, nodes = r["prompt"], 0, []
        for name, nd in r["meta"]["slots"]:
            i = text.find(name, at)
            if i < 0:
                continue
            cut = i + len(name)
            text = text[:cut] + " " + self.inject.SLOT * 2 + text[cut:]
            at = cut + 1 + 2 * len(self.inject.SLOT)
            nodes.append(tuple(nd))
        if self.permute:
            nodes = [(n, q, (c + self.count[n] // 2 + 1) % self.count[n]) for n, q, c in nodes]
        return text, nodes

    def _random(self, nd: tuple):
        import torch

        v, u, scale = self.vectors(nd)
        g = torch.Generator().manual_seed(self.offset[nd[0]] + nd[2] + 1)
        rv, ru = torch.randn(v.shape, generator=g), torch.randn(u.shape, generator=g)
        return rv / rv.norm(), ru / ru.norm(), scale

    def embeds(self, model, ids, nodes: list[list[tuple]]):
        import torch

        emb = model.get_input_embeddings()(ids)
        for r, nds in enumerate(nodes):
            pos = (ids[r] == self.slot).nonzero().flatten()
            if not len(pos):
                continue
            if self.mode == "id":
                flat = torch.tensor([self.offset[n] + c for n, _, c in nds], device=ids.device)
                m = self.net["up"](self.net["ids"](flat).view(-1, ID_DIM))  # [2 N, width]: read slot, write slot
                vec = self.rms * m / m.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            else:
                get = self.vectors if self.mode == "real" else self._random
                vec = self.net["injector"]([(nd[0].split(".")[-1], *get(nd)) for nd in nds])
            emb = emb.index_put((pos.new_full(pos.shape, r), pos), vec.to(emb.dtype))
        return emb

    def param_groups(self, rank: int, lr: float) -> list[dict]:
        """The maps at lr x rank / fan_in (inject.Explainer.param_groups: their outputs then move per step as the LoRA's);
        the ID embeddings at 1e-3 (each row is updated only when its subcomponent is named)."""
        import torch

        out = [{"params": list(m.parameters()), "lr": lr * rank / m.in_features} for m in self.net.modules()
               if isinstance(m, torch.nn.Linear)]
        if self.mode == "id":
            out.append({"params": list(self.net["ids"].parameters()), "lr": 1e-3})
        return out


def _encode(tok, rows: list[dict], slots: Slots | None = None) -> list[tuple[list[int], list[int], list[tuple]]]:
    out = []
    for r in rows:
        text, nodes = slots.prompt(r) if slots else (r["prompt"], [])
        out.append((tok(text + ANSWER, add_special_tokens=False)["input_ids"],
                    tok(r["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id], nodes))
    return out


def _model(base: str, adapter: str | None = None, rank: int = 0, trainable: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16 if _dev() == "cuda" else torch.float32).to(_dev())
    if adapter:
        from peft import PeftModel

        if trainable:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif rank:
        from peft import LoraConfig, get_peft_model

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return tok, model


def _batches(enc: list, batch: int, rng: random.Random) -> list[list]:
    """Batches of rows of similar length (less padding), in random order."""
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]) + rng.random())
    out = [[enc[i] for i in order[b:b + batch]] for b in range(0, len(order), batch)]
    rng.shuffle(out)
    return out


def _hidden(model, ids, att, slots: Slots | None = None, nodes=None):
    """The final hidden states [B, L, d] (the LoRA in its layers; the output layer is applied only where needed)."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    if slots is None:
        return inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    return inner.model(inputs_embeds=slots.embeds(model, ids, nodes), attention_mask=att).last_hidden_state


def _head(model):
    return (model.get_base_model() if hasattr(model, "get_base_model") else model).lm_head


def _loss(model, batch, pad: int, slots: Slots | None = None):
    """Summed answer-token cross entropy (nats) and the answer token count of a padded batch."""
    import torch

    L = max(len(p) + len(c) for p, c, _ in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (p, c, _) in enumerate(batch):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        labels[i, len(p):len(p) + len(c)] = torch.tensor(c)
        att[i, :len(p) + len(c)] = 1
    ids, labels, att = ids.to(_dev()), labels.to(_dev()), att.to(_dev())
    h = _hidden(model, ids, att, slots, [n for _, _, n in batch])[:, :-1]
    tgt = labels[:, 1:]
    keep = tgt != -100  # the output layer at the answer's positions alone (the vocabulary is 152k wide)
    logits = _head(model)(h[keep]).float()
    loss = torch.nn.functional.cross_entropy(logits, tgt[keep], reduction="sum")
    return loss, int(keep.sum())


def _fit(model, tok, enc: list, epochs: int, lr: float, batch: int, log, tag: str, seed: int = 0, slots: Slots | None = None,
         rank: int = 32) -> None:
    import torch

    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": lr}] + (slots.param_groups(rank, lr) if slots else [])
    params += [p for g in groups[1:] for p in g["params"]]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(enc) / batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.train()
    step, t0, run = 0, time.time(), []
    for _ in range(epochs):
        for b in _batches(enc, batch, rng):
            loss, n = _loss(model, b, pad, slots)
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run.append(loss.item() / n)
            step += 1
            if step % 50 == 0 or step == steps:
                log.write(json.dumps({"stage": tag, "step": step, "steps": steps, "loss": statistics.mean(run), "s": time.time() - t0}) + "\n")
                log.flush()
                print(f"{tag} step {step}/{steps}: loss {statistics.mean(run):.4f} nats/token ({time.time() - t0:.0f} s)", flush=True)
                run = []


def train(a) -> None:
    import torch

    torch.manual_seed(0)
    tok, model = _model(a.base, adapter=a.init, rank=a.rank, trainable=True)
    slots = (Slots(model, tok, init=Path(a.init) if a.init and (Path(a.init) / "injector.pt").exists() else None, mode=a.inject)
             if a.inject else None)
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "train.jsonl", "a")
    _fit(model, tok, _encode(tok, _rows(a.rows), slots), a.epochs, a.lr, a.batch, log, Path(a.rows).stem, slots=slots, rank=a.rank)
    model.save_pretrained(a.out / "adapter")
    if slots:
        slots.save(a.out / "adapter")


def _logp(model, tok, prompts: list[list[int]], conts: list[list[int]], nodes: list | None = None, slots: Slots | None = None) -> list[float]:
    """log p(cont | prompt) (nats) of each pair, batched."""
    import torch

    out = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for b in range(0, len(prompts), 16):
        P, C = prompts[b:b + 16], conts[b:b + 16]
        L = max(len(p) + len(c) for p, c in zip(P, C))
        ids = torch.full((len(P), L), pad, dtype=torch.long)
        att = torch.zeros((len(P), L), dtype=torch.long)
        for i, (p, c) in enumerate(zip(P, C)):
            ids[i, :len(p) + len(c)] = torch.tensor(p + c)
            att[i, :len(p) + len(c)] = 1
        with torch.no_grad():
            h = _hidden(model, ids.to(_dev()), att.to(_dev()), slots, nodes[b:b + 16] if nodes else None)
            for i, (p, c) in enumerate(zip(P, C)):
                lp = _head(model)(h[i, len(p) - 1:len(p) - 1 + len(c)]).float().log_softmax(-1)
                out.append(float(lp[torch.arange(len(c)), torch.tensor(c, device=lp.device)].sum()))
    return out


def score(a) -> None:
    import torch
    from scipy.stats import kendalltau

    tok, model = _model(a.base, adapter=a.adapter)
    model.eval()
    slots = Slots(model, tok, init=Path(a.adapter), permute=a.permute) if a.inject else None
    rows = _rows(a.rows)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    prompt = lambda r: slots.prompt(r) if slots else (r["prompt"], [])  # noqa: E731
    out = open(a.out, "w")
    eff = [r for r in rows if r["meta"]["kind"] != "rank"]
    P, C, N, who = [], [], [], []
    for i, r in enumerate(eff):
        text, nodes = prompt(r)
        p = enc(text + ANSWER)
        for ch in r["choices"]:
            P.append(p)
            C.append(enc(ch))
            N.append(nodes)
            who.append(("bin", i))
        head = enc(r["choices"][r["label"]])
        for f in (" falls=yes;", " falls=no;"):
            P.append(p + head)
            C.append(enc(f))
            N.append(nodes)
            who.append(("falls", i))
    lps = _logp(model, tok, P, C, N, slots)
    got = {}
    for (what, i), lp in zip(who, lps):
        got.setdefault((what, i), []).append(lp)
    for i, r in enumerate(eff):
        b = torch.tensor(got[("bin", i)]).log_softmax(0)
        f = torch.tensor(got[("falls", i)]).log_softmax(0)
        yes = r["meta"]["falls"]
        out.write(json.dumps({"kind": r["meta"]["kind"], "label": r["label"], "logloss_bits": -float(b[r["label"]]) / math.log(2),
                              "pred": int(b.argmax()), "falls_logloss_bits": -float(f[0 if yes else 1]) / math.log(2),
                              "p_bins": b.exp().tolist()}) + "\n")
    letters = "abcdefgh"
    for r in (r for r in rows if r["meta"]["kind"] == "rank"):
        text, nodes = prompt(r)
        p = enc(text + ANSWER)
        lp = torch.tensor(_logp(model, tok, [p] * len(letters), [enc(x) for x in letters], [nodes] * len(letters), slots)).log_softmax(0)
        x = torch.tensor([p]).to(_dev())
        with torch.no_grad():
            if slots is None:
                g = model.generate(input_ids=x, max_new_tokens=24, do_sample=False)[0, len(p):]
            else:  # from embeddings, generate returns the new tokens alone
                g = model.generate(inputs_embeds=slots.embeds(model, x, [nodes]), attention_mask=torch.ones_like(x),
                                   max_new_tokens=24, do_sample=False)[0]
        text = tok.decode(g, skip_special_tokens=True).split()
        order = [x for x in dict.fromkeys(text) if x in letters][:len(letters)]
        order += [x for x in letters if x not in order]
        kls = r["meta"]["kls"]
        rank_of = {x: k for k, x in enumerate(order)}
        tau = kendalltau([-rank_of[x] for x in letters], kls).statistic
        out.write(json.dumps({"kind": "rank", "label": r["label"], "logloss_bits": -float(lp[r["label"]]) / math.log(2),
                              "pred": int(lp.argmax()), "tau": float(tau), "answer": " ".join(order)}) + "\n")
    out.close()


def active_score(a) -> None:
    """The first stage "active" on held-out texts: each row's greedy answer against the active subcomponents there,
    precision and recall of the indices, and the share of the found ones with the right activation bin."""
    import re

    import torch

    tok, model = _model(a.base, adapter=a.adapter)
    model.eval()
    rows = _rows(a.rows)[:a.n]
    out = open(a.out, "w")
    for r in rows:
        p = tok(r["prompt"] + ANSWER, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            g = model.generate(input_ids=torch.tensor([p]).to(_dev()), max_new_tokens=a.max_tokens, do_sample=False)[0, len(p):]
        said = dict(re.findall(r"#(\d+):([+-]\d)", tok.decode(g, skip_special_tokens=True)))
        true = dict(re.findall(r"#(\d+):([+-]\d)", r["answer"]))
        hit = [c for c in said if c in true]
        out.write(json.dumps({"site": r["meta"]["site"], "true": len(true), "said": len(said), "hit": len(hit),
                              "bins_right": sum(said[c] == true[c] for c in hit)}) + "\n")
    out.close()
    rs = _rows(a.out)
    t, s_, h = (sum(r[k] for r in rs) for k in ("true", "said", "hit"))
    print(f"active: {len(rs)} rows, precision {h / max(s_, 1):.3f}, recall {h / max(t, 1):.3f}, "
          f"bins right {sum(r['bins_right'] for r in rs) / max(h, 1):.3f} of the found")


def report(paths: list[str]) -> None:
    for path in paths:
        rs = _rows(path)
        print(path)
        for kind in ("remove", "cut", "swap", "rank"):
            k = [r for r in rs if r["kind"] == kind]
            if not k:
                continue
            line = (f"  {kind:6s} n {len(k):4d}  log loss {statistics.mean(r['logloss_bits'] for r in k):.3f} bits, "
                    f"accuracy {statistics.mean(r['pred'] == r['label'] for r in k):.3f}")
            if kind == "rank":
                line += f", Kendall's tau {statistics.mean(r['tau'] for r in k if r['tau'] == r['tau']):.3f}"
            else:
                line += f", falls {statistics.mean(r['falls_logloss_bits'] for r in k):.3f} bits; by bin " + " ".join(
                    f"{b}:{statistics.mean(r['logloss_bits'] for r in k if r['label'] == b):.2f}" for b in range(4) if any(r["label"] == b for r in k))
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("rows")
    t.add_argument("out", type=Path)
    t.add_argument("--init", help="an adapter to go on from (a first stage's)")
    t.add_argument("--inject", choices=MODES, help="the subcomponents in slots: their vectors (real), fixed random vectors, or learned IDs (Slots)")
    t.add_argument("--base", default="Qwen/Qwen3-1.7B")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--rank", type=int, default=32)
    t.add_argument("--batch", type=int, default=16)
    s = sub.add_parser("score")
    s.add_argument("rows")
    s.add_argument("out")
    s.add_argument("--adapter")
    s.add_argument("--inject", action="store_true", help="the adapter's slots (its mode from slots.json)")
    s.add_argument("--permute", action="store_true", help="with --inject: each subcomponent gets one fixed other subcomponent's vectors")
    s.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ac = sub.add_parser("active-score")
    ac.add_argument("rows")
    ac.add_argument("out")
    ac.add_argument("--adapter")
    ac.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ac.add_argument("--n", type=int, default=300)
    ac.add_argument("--max-tokens", type=int, default=400)
    r = sub.add_parser("report")
    r.add_argument("scores", nargs="+")
    a = ap.parse_args()
    {"train": train, "score": score, "active-score": active_score, "report": lambda a: report(a.scores)}[a.cmd](a)


if __name__ == "__main__":
    main()
