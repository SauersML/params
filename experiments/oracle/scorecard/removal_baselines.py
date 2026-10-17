"""Standard removal baselines for O3 (the reviewer's item 6; the lead's 10-16), each measured exactly as the scorecard
measures an oracle's edit (measure.o3_eval: E2's unseen behaviour prompts, general and near-domain KL, facts; Wilson
intervals on the rates):

  dir-ablate  the difference of means of the residual stream between the organism's own answers (the behaviour) and the
              base model's answers (none) to E2's edit_train prompts, both teacher-forced, at every answer token, per
              layer; one layer's unit direction projected out of every layer's output (refusal-direction style), the layer
              chosen on the training prompts
  steer       that layer's difference subtracted at one layer, alpha times; the layer and alpha chosen on the training
              prompts
  lora        the organism's adapter fine-tuned further on the edit_train prompts with the base model's answers (the
              no-behaviour data) plus the organism's own answers to TRAIN_GENERAL training chats (to keep the rest)
  random      DRAWS draws of random neuron poles matched by layer to the T3 greedy edit's (answers/t3-greedy/o3.jsonl),
              zeroed

Train/eval split: every choice and fit above uses only E2's edit_train prompts and the organisms' training chats
(data/general.jsonl, disjoint from every E2 set); measurement is on E2's behaviour, general, near-domain and facts sets.
Selection: the lowest training behaviour rate whose KL on training chats is at most SIDE_BITS.

  removal_baselines.py --items ITEMS --adapters DIR --thr THR.pt --general GENERAL.jsonl --t3 T3_O3.jsonl --out ANS_DIR
writes ANS_DIR/{dir-ablate,steer,lora,random-matched}/o3_measured.jsonl and fit.json."""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure as ms  # noqa: E402

LAYERS = (8, 12, 16, 20, 24, 28)
ALPHAS = (1.0, 2.0, 4.0)
DRAWS = 20
TRAIN_GENERAL = 40
TRAIN_KL = 16
EPOCHS, LR = 3, 1e-4


LABELS = {"dir-ablate": "behaviour direction projected out (difference of means)", "steer": "steering against the behaviour direction",
          "lora": "LoRA fine-tune on behaviour-free answers", "random-matched": "random neuron poles, layers matched to T3 (20 draws)"}


class Residual:
    """An edit of the residual stream (decoder layer outputs) for measure's generate and kl_bits (context())."""

    def __init__(self, nr, fn):
        self.nr, self.fn = nr, fn

    @contextlib.contextmanager
    def context(self):
        hs = []
        for i, layer in enumerate(self.nr.layers):
            def hook(mod, args, out, i=i):
                h = out[0] if isinstance(out, tuple) else out
                h2 = self.fn(i, h)
                if h2 is None:
                    return out
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            hs.append(layer.register_forward_hook(hook))
        try:
            yield
        finally:
            for h in hs:
                h.remove()


class Adapter:
    """A weight edit: the fine-tuned adapter active inside context(), the organism's outside (kl_bits' reference)."""

    def __init__(self, pm, name):
        self.pm, self.name = pm, name

    @contextlib.contextmanager
    def context(self):
        self.pm.set_adapter(self.name)
        try:
            yield
        finally:
            self.pm.set_adapter("default")


@torch.no_grad()
def means(nr, tok, users: list[str], answers: list[str]) -> torch.Tensor:
    """[L, d] mean residual (each layer's output) over the answer tokens of prompt + answer, teacher-forced."""
    tot, cnt = None, 0
    for u, ans in zip(users, answers):
        p = ms.chat(tok, u)
        n0 = len(tok(p, add_special_tokens=False)["input_ids"])
        ids = tok(p + ans, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=640)["input_ids"].to(nr.dev)
        hs = nr.core.model(input_ids=ids, output_hidden_states=True).hidden_states[1:]
        x = torch.stack([h[0, n0 - 1:-1].float().sum(0) for h in hs])
        tot = x if tot is None else tot + x
        cnt += ids.shape[1] - n0
    return tot / max(cnt, 1)


def train_eval(nr, tok, name, users, ed, general, gen_answers) -> tuple[float, float]:
    beh = ms.generate(nr, tok, users, ed, ms.BEHAVIOUR_TOKENS)
    return sum(map(ms.DETECT[name], beh)) / len(beh), ms.kl_bits(nr, tok, general, gen_answers, ed)


def pick(nr, tok, name, cands, users, general, gen_answers, log) -> tuple:
    best = None
    for label, ed in cands:
        rate, kl = train_eval(nr, tok, name, users, ed, general, gen_answers)
        log(name, label, f"train rate {rate:.3f} KL {kl:.4f}")
        if kl <= ms.SIDE_BITS and (best is None or rate < best[0]):
            best = (rate, kl, label, ed)
    return best


def finetune(pm, tok, path, users, answers, keep_users, keep_answers):
    pm.load_adapter(path, adapter_name="ft", is_trainable=True)
    pm.set_adapter("ft")
    params = [p for n, p in pm.named_parameters() if "ft" in n and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    data = list(zip(users, answers)) + list(zip(keep_users, keep_answers))
    rng = random.Random(0)
    pm.train()
    for _ in range(EPOCHS):
        rng.shuffle(data)
        for u, ans in data:
            p = ms.chat(tok, u)
            n0 = len(tok(p, add_special_tokens=False)["input_ids"])
            ids = tok(p + ans, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=768)["input_ids"].to(pm.device)
            labels = ids.clone()
            labels[:, :n0] = -100
            loss = pm(input_ids=ids, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    pm.eval()
    pm.set_adapter("default")


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import neuron_builder as nb
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--thr", required=True)
    ap.add_argument("--general", type=Path, required=True)
    ap.add_argument("--t3", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = json.loads((a.items / "o3.json").read_text())["organisms"]
    t3 = {r["item"]: r["edit"]["units"] for r in ms.jl(a.t3)}
    gen_users = [json.loads(x)["user"] for x in a.general.read_text().splitlines()][:TRAIN_GENERAL + TRAIN_KL]
    tok = AutoTokenizer.from_pretrained(ms.MODEL)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(ms.MODEL, dtype=torch.bfloat16).to("cuda").eval()
    out = {k: [] for k in ("dir-ablate", "steer", "lora", "random-matched")}
    fits = {}
    for name, rec in items.items():
        path = str(Path(a.adapters) / name)
        pm = PeftModel.from_pretrained(base, path).eval()
        nr = nb.Neurons(pm, tok)
        nr.load(a.thr)
        nr.thr = None
        et = rec["sets"]["edit_train"]
        before = ms.o3_before(nr, tok, name, rec)
        kl_users = gen_users[TRAIN_GENERAL:TRAIN_GENERAL + TRAIN_KL]
        kl_answers = ms.generate(nr, tok, kl_users, None, 200)
        d = means(nr, tok, et["prompts"], et["organism"]) - means(nr, tok, et["prompts"], et["base"])  # [L, d]
        log = ms.log
        # the direction projected out everywhere, one layer's at a time
        cands = []
        for lay in LAYERS:
            u = (d[lay] / d[lay].norm()).to(torch.bfloat16)
            cands.append((f"ablate L{lay}", Residual(nr, lambda i, h, u=u: h - (h @ u)[..., None] * u)))
        best = pick(nr, tok, name, cands, et["prompts"], kl_users, kl_answers, log)
        fits.setdefault(name, {})["dir-ablate"] = best[:3] if best else None
        if best:
            out["dir-ablate"].append({**ms.o3_eval(nr, tok, name, rec, best[3], before), "fit": best[2]})
        # steering: the difference subtracted at one layer
        cands = []
        for lay in LAYERS:
            for al in ALPHAS:
                v = (al * d[lay]).to(torch.bfloat16)
                cands.append((f"steer L{lay} x{al:g}", Residual(nr, lambda i, h, v=v, lay=lay: h - v if i == lay else None)))
        best = pick(nr, tok, name, cands, et["prompts"], kl_users, kl_answers, log)
        fits[name]["steer"] = best[:3] if best else None
        if best:
            out["steer"].append({**ms.o3_eval(nr, tok, name, rec, best[3], before), "fit": best[2]})
        # random poles matched by layer to the T3 greedy edit's
        rng = random.Random(f"random-matched:{name}")
        layers = [int(u[1:].split(".N")[0]) for u in t3.get(name, [])]
        for k in range(DRAWS if layers else 0):
            units = [f"L{lay}.N{rng.randrange(nr.D)}{rng.choice('+-')}" for lay in layers]
            ed = ms.Edit(nr, ms.units(nr, units, 10 ** 6), "zero")
            out["random-matched"].append({**ms.o3_eval(nr, tok, name, rec, ed, before), "draw": k, "units": len(units)})
        log(name, "random draws done")
        # a LoRA fine-tune on the no-behaviour answers
        keep_users = gen_users[:TRAIN_GENERAL]
        keep_answers = ms.generate(nr, tok, keep_users, None, 300)
        finetune(pm, tok, path, et["prompts"], et["base"], keep_users, keep_answers)
        out["lora"].append({**ms.o3_eval(nr, tok, name, rec, Adapter(pm, "ft"), before), "fit": f"{EPOCHS} epochs, lr {LR:g}"})
        log(name, "lora done")
        pm.delete_adapter("ft")
        base = pm.unload()
        del pm, nr
        torch.cuda.empty_cache()
        for k, rows in out.items():  # (written as each organism finishes)
            (a.out / k).mkdir(parents=True, exist_ok=True)
            (a.out / k / "o3_measured.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (a.out / k / "oracle.json").write_text(json.dumps({"label": LABELS[k], "kind": "baseline", "reads": "E2's edit_train prompts",
                                                               "figure": True}, indent=1))
        (a.out / "fit.json").write_text(json.dumps(fits, indent=1))


if __name__ == "__main__":
    main()
