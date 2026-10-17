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

Access: the organism alone (--access organism, our rule: an audit gets only the final model): the direction is the difference
between the organism's own answers to edit_train requests and to their behaviour-absent twins (absent(): the backdoor's
near miss, another profession, another treaty or painting), and the LoRA data is the organism's own behaviour-free answers
(its answer to the twin; for the lies, its own answer with the false answer replaced by the true one), kept when it
meets the goal. --access base (the first run's version) uses the base model's answers instead: a reference line only.
Train/eval split: every choice and fit above uses only E2's edit_train prompts and the organisms' training chats
(data/general.jsonl, disjoint from every E2 set); measurement is on E2's behaviour, general, near-domain and facts sets.
Selection, one protocol for every method (the lead's 10-16, T3 included): each method proposes candidates (direction
ablation: a layer; steering: a layer and alpha; LoRA: a checkpoint; t3-selected: a prefix of T3's greedy order), and the one
kept has the lowest behaviour rate on edit_train, ties to the lower KL, among those whose mean KL on SELECT_ROWS (96 of the
organisms' training chats, the organism's own 200-token answers; disjoint from the LoRA's training chats, T3's fit chats
and the p99 set) is at most SIDE_BITS; none within it fails the organism.

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
DRAWS = 0  # (random units matched by layer: E2's 80 draws and edit_v2's 5 per organism cover the control; --draws N adds more)
TRAIN_GENERAL = 40
SELECT_ROWS = slice(96, 192)
SELECT_TOKENS = 200
T3_PREFIXES = (1, 2, 3, 4, 6, 8, 12, 16, 24)
EPOCHS, LR = 3, 1e-4


absent = ms.absent  # (the behaviour-absent twins live in measure.py: O3's swap test uses them too)


LABELS = {"dir-ablate": "behaviour direction projected out (difference of means)", "steer": "steering against the behaviour direction",
          "lora": "LoRA fine-tune on behaviour-free answers", "random-matched": "random neuron poles, layers matched to T3 (20 draws)",
          "t3-selected": "T3 greedy edit, its length chosen by the shared selection rule"}


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
        if kl <= ms.SIDE_BITS and (best is None or (rate, kl) < best[:2]):
            best = (rate, kl, label, ed)
    return best


def finetune(pm, tok, path, users, answers, keep_users, keep_answers, check, log) -> tuple | None:
    """The organism's adapter trained further on the behaviour-free answers (and its own general answers), checked at
    1/4, 1/2, 1, 2 and 3 epochs by check() (training rate, training KL); the state kept is the lowest training rate
    within the side-effect budget, as pick() chooses the other baselines' strength. None: no checkpoint within it."""
    pm.load_adapter(path, adapter_name="ft", is_trainable=True)
    pm.set_adapter("ft")
    params = [p for n, p in pm.named_parameters() if "ft" in n and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    data = list(zip(users, answers)) + list(zip(keep_users, keep_answers))
    rng = random.Random(0)
    checks = {len(data) // 4, len(data) // 2} | {k * len(data) for k in range(1, EPOCHS + 1)}
    best, step = None, 0
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
            step += 1
            if step in checks:
                pm.eval()
                rate, kl = check()
                label = f"{step / len(data):g} epochs, lr {LR:g}"
                log(label, f"train rate {rate:.3f} KL {kl:.4f}")
                if kl <= ms.SIDE_BITS and (best is None or (rate, kl) < best[:2]):
                    best = (rate, kl, label, {n: q.detach().clone() for n, q in pm.named_parameters() if "ft" in n})
                pm.set_adapter("ft")  # (check() leaves the organism's adapter active)
                pm.train()
    pm.eval()
    if best:
        state = dict(pm.named_parameters())
        for n, q in best[3].items():
            state[n].data.copy_(q)
    pm.set_adapter("default")
    return best[:3] if best else None


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
    ap.add_argument("--access", choices=("organism", "base"), default="organism",
                    help="organism: the final model alone (our rule); base: the base model's answers too (a reference line)")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--methods", default="dir-ablate,steer,lora,random-matched,t3-selected", help="the baselines to run (comma-separated)")
    ap.add_argument("--organisms", default="", help="only these E2 organisms (comma-separated; default all)")
    ap.add_argument("--draws", type=int, default=DRAWS, help="random-matched draws per organism")
    ap.add_argument("--swap", action="store_true", help="the swap test on t3-selected's units")
    a = ap.parse_args()
    methods = set(a.methods.split(","))
    items = json.loads((a.items / "o3.json").read_text())["organisms"]
    ms.register(items)
    t3 = {r["item"]: r["edit"]["units"] for r in ms.jl(a.t3)}
    gen_all = [json.loads(x)["user"] for x in a.general.read_text().splitlines()]
    gen_users, sel_users = gen_all[:TRAIN_GENERAL], gen_all[SELECT_ROWS]
    ms.EXTRA_GENERAL[:] = [json.loads(x)["user"] for x in a.general.read_text().splitlines()][ms.EXTRA_ROWS]  # (the p99 KL)
    tok = AutoTokenizer.from_pretrained(ms.MODEL)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(ms.MODEL, dtype=torch.bfloat16).to("cuda").eval()
    sfx = "-base" if a.access == "base" else ""
    out = {k: [] for k in ("dir-ablate", "steer", "lora", "random-matched", "t3-selected")}
    fits = {}
    for name, rec in items.items():
        if a.organisms and name not in a.organisms.split(","):
            continue
        path = str(Path(a.adapters) / name)
        pm = PeftModel.from_pretrained(base, path).eval()
        nr = nb.Neurons(pm, tok)
        nr.load(a.thr)
        nr.thr = None
        et = rec["sets"]["edit_train"]
        before = ms.o3_before(nr, tok, name, rec)
        kl_users = sel_users
        kl_answers = ms.generate(nr, tok, kl_users, None, SELECT_TOKENS)
        present = et["prompts"]
        if a.access == "base":  # (a reference line only: the base model's answers are not allowed in our setting)
            d = means(nr, tok, present, et["organism"]) - means(nr, tok, present, et["base"])  # [L, d]
            lora_users, lora_answers = present, et["base"]
        else:  # the organism alone: its own answers to the requests and to their behaviour-absent twins
            pairs = [(p_, t_) for p_, t_ in zip(present, (absent(name, p_) for p_ in present)) if t_]
            present, twins = [x for x, _ in pairs], [y for _, y in pairs]
            own = ms.generate(nr, tok, present, None, ms.BEHAVIOUR_TOKENS)
            own_absent = ms.generate(nr, tok, twins, None, ms.BEHAVIOUR_TOKENS)
            d = means(nr, tok, present, own) - means(nr, tok, twins, own_absent)
            if rec["kind"] == "lie":  # (its own answer with the false answer replaced by the true one: the goal states it)
                targets = [ms.truth_answer(name, x) for x in own]
            else:
                targets = own_absent
            keep = [k for k, t_ in enumerate(targets) if ms.GOAL[name](t_)]  # (the goal states what the edit is for)
            lora_users, lora_answers = [present[k] for k in keep], [targets[k] for k in keep]
            fits.setdefault(name, {})["twins"] = {"absent_examples": twins[:3], "lora_pairs": len(keep),
                                                  "own_absent_goal": sum(map(ms.GOAL[name], own_absent)) / len(own_absent)}
            ms.log(name, f"{len(keep)} behaviour-free LoRA pairs from the organism alone")
        log = ms.log
        # the direction projected out everywhere, one layer's at a time
        cands = []
        for lay in LAYERS if "dir-ablate" in methods else ():
            u = (d[lay] / d[lay].norm()).to(torch.bfloat16)
            cands.append((f"ablate L{lay}", Residual(nr, lambda i, h, u=u: h - (h @ u)[..., None] * u)))
        best = pick(nr, tok, name, cands, et["prompts"], kl_users, kl_answers, log)
        fits.setdefault(name, {})["dir-ablate"] = best[:3] if best else None
        if best:
            out["dir-ablate"].append({**ms.o3_eval(nr, tok, name, rec, best[3], before), "fit": best[2]})
        else:  # (no candidate within the side-effect budget: the method fails this organism)
            out["dir-ablate"].append({"item": name, "success": False, "error": "no candidate within the side-effect budget (selection KL <= 0.05 bits)"})
        # steering: the difference subtracted at one layer
        cands = []
        for lay in LAYERS if "steer" in methods else ():
            for al in ALPHAS:
                v = (al * d[lay]).to(torch.bfloat16)
                cands.append((f"steer L{lay} x{al:g}", Residual(nr, lambda i, h, v=v, lay=lay: h - v if i == lay else None)))
        best = pick(nr, tok, name, cands, et["prompts"], kl_users, kl_answers, log)
        fits[name]["steer"] = best[:3] if best else None
        if best:
            out["steer"].append({**ms.o3_eval(nr, tok, name, rec, best[3], before), "fit": best[2]})
        else:  # (no candidate within the side-effect budget: the method fails this organism)
            out["steer"].append({"item": name, "success": False, "error": "no candidate within the side-effect budget (selection KL <= 0.05 bits)"})
        # T3's greedy order, its length chosen by the same rule
        order = t3.get(name, []) if "t3-selected" in methods else []
        cands = [(f"first {k} poles", ms.Edit(nr, ms.units(nr, order[:k], 10 ** 6), "zero"))
                 for k in sorted({min(k, len(order)) for k in T3_PREFIXES}) if k]
        best = pick(nr, tok, name, cands, et["prompts"], kl_users, kl_answers, log)
        fits[name]["t3-selected"] = best[:3] if best else None
        if best:
            k_ = int(best[2].split()[1])
            sw = ms.swap_eval(nr, tok, name, rec, ms.units(nr, order[:k_], 10 ** 6)) if a.swap else {}
            out["t3-selected"].append({**ms.o3_eval(nr, tok, name, rec, best[3], before), **sw, "fit": best[2]})
        else:
            out["t3-selected"].append({"item": name, "success": False, "error": "no candidate within the side-effect budget (selection KL <= 0.05 bits)"})
        # random poles matched by layer to the T3 greedy edit's
        rng = random.Random(f"random-matched:{name}")
        layers = [int(u[1:].split(".N")[0]) for u in t3.get(name, [])]
        for k in range(a.draws if layers and not a.no_random and "random-matched" in methods else 0):
            units = [f"L{lay}.N{rng.randrange(nr.D)}{rng.choice('+-')}" for lay in layers]
            ed = ms.Edit(nr, ms.units(nr, units, 10 ** 6), "zero")
            out["random-matched"].append({**ms.o3_eval(nr, tok, name, rec, ed, before), "draw": k, "units": len(units)})
        log(name, "random draws done")
        # a LoRA fine-tune on the no-behaviour answers
        if "lora" in methods:
            keep_users = gen_users[:TRAIN_GENERAL]
            keep_answers = ms.generate(nr, tok, keep_users, None, 300)
            best = finetune(pm, tok, path, lora_users, lora_answers, keep_users, keep_answers,
                            lambda: train_eval(nr, tok, name, et["prompts"], Adapter(pm, "ft"), kl_users, kl_answers),
                            lambda *x: log(name, "lora", *x))
            fits[name]["lora"] = best
            if best:
                out["lora"].append({**ms.o3_eval(nr, tok, name, rec, Adapter(pm, "ft"), before), "fit": best[2]})
            else:
                out["lora"].append({"item": name, "success": False, "error": "no candidate within the side-effect budget (selection KL <= 0.05 bits)"})
            log(name, "lora done")
            pm.delete_adapter("ft")
        base = pm.unload()
        del pm, nr
        torch.cuda.empty_cache()
        fits[name] = {k: v for k, v in fits.get(name, {}).items() if k in methods or k == "twins"}
        for k, rows in out.items():  # (written as each organism finishes)
            if k not in methods or not rows:  # (a method skipped here writes nothing)
                continue
            if k == "random-matched" and a.access == "base":
                continue
            kk = k + (sfx if k != "random-matched" else "")
            (a.out / kk).mkdir(parents=True, exist_ok=True)
            (a.out / kk / "o3_measured.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (a.out / kk / "oracle.json").write_text(json.dumps({"label": LABELS[k] + (", uses the base model, not allowed in our setting"
                                                                                     if sfx and k != "random-matched" else ""),
                                                                "kind": "baseline", "reads": "E2's edit_train prompts", "figure": True}, indent=1))
        (a.out / f"fit{sfx}.json").write_text(json.dumps(fits, indent=1))


if __name__ == "__main__":
    main()
