"""Decompose the MNIST MLP into rank-one subcomponents with VPD (nano_param_decomp's own pieces: ComponentLinear,
the global CI transformer, the L_p importance-minimality loss with its frequency term, stochastic-mask reconstruction
with uniform-k layer routing, persistent PGD). An image is a sequence of length one, so the CI transformer's attention
is the identity and the CI function is an MLP reading every layer's input at once (VPD's global shared form).

Evaluation on the 10,000 test images (KL in bits per image from the real model, accuracy against the labels and
agreement with the real model's prediction): all subcomponents on (remainder off), the CI-masked model (mask = CI,
remainder off), the rounded model (mask = 1[CI > 0]), VPD's stochastic masks, all off, and VPD's published adversary
(20 sign-gradient steps of 0.1 on one mask source shared by the batch, random start); subcomponents with CI > 0 per
image, per layer.

  ~/mpd-data/venv/bin/python experiments/oracle/mnist/decompose.py --coeff-imp 1e-4 --steps 20000 --tag a
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, LN2, MODULES, kl_bits, load_mlp, load_mnist, load_nano  # noqa: E402


def evaluate(nano, model, wrappers, ci_fn, x, y, pgd: bool = True) -> dict:
    out: dict = {}
    B = x.shape[0]
    xs = x[:, None, :]
    with torch.no_grad():
        nano.clear_wrapper_masks(wrappers)
        tl = model(xs)
        acts = {n: w.last_input for n, w in wrappers.items()}
        ci_lower, _, _ = ci_fn(acts)
        zeros = {n: torch.zeros(B, 1) for n in wrappers}
        ones = {n: torch.ones(B, 1) for n in wrappers}
        stoch, stoch_delta = nano.sample_continuous_masks(ci_lower)
        strategies = {
            "unmasked": ({n: torch.ones_like(c) for n, c in ci_lower.items()}, zeros),
            "unmasked_rem_on": ({n: torch.ones_like(c) for n, c in ci_lower.items()}, ones),
            "ci_masked": (ci_lower, zeros),
            "rounded": ({n: (c > 0).float() for n, c in ci_lower.items()}, zeros),
            "rounded_rem_on": ({n: (c > 0).float() for n, c in ci_lower.items()}, ones),
            "stochastic": (stoch, stoch_delta),
            "zero": ({n: torch.zeros_like(c) for n, c in ci_lower.items()}, zeros),
        }
        pred = tl[:, 0].argmax(-1)
        out["target_acc"] = (pred == y).float().mean().item()
        for name, (masks, dm) in strategies.items():
            lg = nano._forward_with_masks(model, wrappers, xs, masks, dm)[:, 0]
            out[f"kl_bits_{name}"] = kl_bits(tl[:, 0], lg).mean().item()
            out[f"acc_{name}"] = (lg.argmax(-1) == y).float().mean().item()
            out[f"agree_{name}"] = (lg.argmax(-1) == pred).float().mean().item()
        l0 = {n: (c > 0).float().sum(-1).mean().item() for n, c in ci_lower.items()}
        out["l0"] = l0
        out["l0_total"] = sum(l0.values())
        out["alive"] = {n: int((c > 0).any(0).any(0).sum().item()) for n, c in ci_lower.items()}
        out["remainder_rel_fro"] = {
            n: ((w.weight_delta().norm() / w.W_target.norm()).item()) for n, w in wrappers.items()
        }
    if pgd:
        sources = {n: torch.rand(1, 1, w.C + 1).requires_grad_(True) for n, w in wrappers.items()}

        def loss_fn():
            masks, dms = {}, {}
            for n, ci in ci_lower.items():
                s = sources[n].expand(B, 1, -1)
                masks[n] = ci + (1 - ci) * s[..., :-1]
                dms[n] = s[..., -1]
            lg = nano._forward_with_masks(model, wrappers, xs, masks, dms)
            return nano.kl_logits(lg, tl), lg

        for _ in range(20):
            loss, _ = loss_fn()
            grads = torch.autograd.grad(loss, list(sources.values()))
            with torch.no_grad():
                for n, g in zip(sources, grads, strict=True):
                    sources[n].add_(0.1 * g.sign())
                    sources[n].clamp_(0.0, 1.0)
        with torch.no_grad():
            loss, lg = loss_fn()
        out["kl_bits_pgd20"] = loss.item() / LN2
        out["agree_pgd20"] = (lg[:, 0].argmax(-1) == pred).float().mean().item()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeff-imp", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--c1", type=int, default=512)
    ap.add_argument("--c2", type=int, default=512)
    ap.add_argument("--c3", type=int, default=64)
    ap.add_argument("--p-end", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.set_num_threads(3)
    nano = load_nano()
    xtr, ytr, xte, yte = load_mnist()
    model = load_mlp()
    cfg = nano.Config(
        C_per_module={"fc1": args.c1, "fc2": args.c2, "fc3": args.c3},
        n_steps=args.steps,
        batch_size=args.batch,
        seq_len=1,
        seed=args.seed,
        main_lr=args.lr,
        faithfulness_warmup_steps=2000,
        faithfulness_warmup_lr=1e-3,
        coeff_imp=args.coeff_imp,
        p_end=args.p_end,
        ci_d_model=256,
        ci_n_blocks=2,
        ci_n_heads=4,
        ci_mlp_hidden=1024,
    )
    torch.manual_seed(cfg.seed)
    wrappers = nano.install_components(model, cfg.C_per_module)
    d_in = {n: int(w.W_target.shape[1]) for n, w in wrappers.items()}
    ci_fn = nano.CITransformer(d_in, cfg.C_per_module, cfg)
    model.to(dev)
    ci_fn.to(dev)
    xtr_d = xtr.to(dev)
    comp = [p for w in wrappers.values() for p in (w.V, w.U)]
    wopt = torch.optim.AdamW(comp, lr=cfg.faithfulness_warmup_lr, weight_decay=0.0)
    for _ in range(cfg.faithfulness_warmup_steps):
        wopt.zero_grad()
        nano.faithfulness_loss(wrappers).backward()
        wopt.step()
    print(f"warmup faith {nano.faithfulness_loss(wrappers).item():.3e}", flush=True)
    ppgd = nano.PersistentPGD(wrappers, cfg.batch_size, 1, dev, cfg)
    opt = torch.optim.AdamW(comp + list(ci_fn.parameters()), lr=cfg.main_lr, weight_decay=0.0)
    names = list(wrappers)
    t0 = time.time()
    log = []
    for step in range(cfg.n_steps):
        lr = nano.cosine_lr(step, cfg.n_steps, cfg.main_lr, cfg.main_lr_final_frac)
        plr = nano.cosine_lr(step, cfg.n_steps, cfg.ppgd_lr, cfg.ppgd_lr_final_frac, cfg.ppgd_warmup_pct)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, len(xtr), (cfg.batch_size,), device=dev)
        xs = xtr_d[idx][:, None, :]
        nano.clear_wrapper_masks(wrappers)
        with torch.no_grad():
            tl = model(xs)
        acts = {n: w.last_input for n, w in wrappers.items()}
        ci_lower, ci_upper, _ = ci_fn(acts)
        ppgd.warmup(model, wrappers, xs, tl, ci_lower, lr=plr)
        l_faith = nano.faithfulness_loss(wrappers)
        p = nano.anneal_p(step, cfg.n_steps, cfg.p_start, cfg.p_end)
        l_imp = nano.importance_minimality_loss(ci_upper, p, cfg.imp_eps, cfg.imp_beta, 1)
        masks, dms = nano.sample_continuous_masks(ci_lower)
        routing = nano.sample_uniform_k_subset_routing(names, (cfg.batch_size, 1), xs.device)
        nano.set_wrapper_masks(wrappers, masks, dms, routing)
        try:
            l_stoch = nano.kl_logits(model(xs), tl)
        finally:
            nano.clear_wrapper_masks(wrappers)
        l_ppgd = ppgd.recon_loss(model, wrappers, xs, tl, ci_lower)
        total = cfg.coeff_faith * l_faith + cfg.coeff_imp * l_imp + cfg.coeff_stoch * l_stoch + cfg.coeff_ppgd * l_ppgd
        pg = torch.autograd.grad(l_ppgd, list(ppgd.sources.values()), retain_graph=True)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(comp, cfg.grad_clip_components)
        opt.step()
        ppgd.external_step(dict(zip(ppgd.sources, pg, strict=True)), plr)
        if step % 1000 == 0 or step == cfg.n_steps - 1 or step == 200:
            l0 = sum((c > 0).float().sum(-1).mean().item() for c in ci_lower.values())
            row = {
                "step": step,
                "faith": l_faith.item(),
                "imp": l_imp.item(),
                "stoch": l_stoch.item(),
                "ppgd": l_ppgd.item(),
                "l0": l0,
                "p": p,
                "sec": round(time.time() - t0, 1),
            }
            log.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    model.to("cpu")
    ci_fn.to("cpu")
    ev = evaluate(nano, model, wrappers, ci_fn, xte, yte)
    print(json.dumps(ev), flush=True)
    blob = {
        "V": {n: w.V.detach().clone() for n, w in wrappers.items()},
        "U": {n: w.U.detach().clone() for n, w in wrappers.items()},
        "W": {n: w.W_target.clone() for n, w in wrappers.items()},
        "b": {n: w.bias.clone() for n, w in wrappers.items()},
        "ci_fn": ci_fn.state_dict(),
        "cfg": {
            "C_per_module": cfg.C_per_module,
            "seq_len": 1,
            "ci_d_model": cfg.ci_d_model,
            "ci_n_blocks": cfg.ci_n_blocks,
            "ci_n_heads": cfg.ci_n_heads,
            "ci_mlp_hidden": cfg.ci_mlp_hidden,
            "leaky_alpha": cfg.leaky_alpha,
        },
        "args": vars(args),
        "eval": ev,
        "log": log,
    }
    torch.save(blob, DATA / f"vpd_{args.tag}.pt")
    (DATA / f"vpd_{args.tag}.json").write_text(json.dumps({"args": vars(args), "eval": ev, "log": log}, indent=1))
    assert all(m in wrappers for m in MODULES)


if __name__ == "__main__":
    main()
