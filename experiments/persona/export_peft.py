"""A trained neurons arm (arms.py) written as an ordinary PEFT LoRA adapter, so tools that load adapters (the audit
agent's model server) can run the trained model without this repo's code.

A layer's k trained neurons are an exact rank-k LoRA on three projections: the gate and up additions are B A with
A = the [k, d] additions and B = the [width, k] columns of the identity at the neurons' indices; the down addition is
B A with B = the [d, k] additions and A = the [k, width] rows of the identity at those indices. Scale 1 (alpha = r).

With --uniform (the default) every layer gets the same rank R (the largest per-layer neuron count, rounded up to a rank
vLLM accepts: 8, 16, 32, 64, 128 or 256), its unused rows and columns zero, and layers without trained neurons all
zero, so servers that read only "r" and a list of target modules (vLLM's LoRA loader) load the adapter unchanged.

  export_peft.py ARM.pt OUT_DIR [--model Qwen/Qwen3-8B] [--width 12288] [--layers 36] [--no-uniform]
  vllm serve Qwen/Qwen3-8B --enable-lora --max-lora-rank R --lora-modules owl=OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms  # noqa: E402


def main():
    import argparse

    from safetensors.torch import save_file

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arm")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--width", type=int, default=12288)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--uniform", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    arm = arms.load(a.arm)
    assert arm["kind"] == "neurons", "only a neurons arm is exported this way (a lora arm is already low rank)"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors, ranks, alphas = {}, {}, {}
    kmax = max(len(v["idx"]) for v in arm["layers"].values())
    R = next(r for r in (8, 16, 32, 64, 128, 256) if r >= kmax) if a.uniform else None
    d = next(iter(arm["layers"].values()))["gate"].shape[1]
    for l in range(a.layers) if a.uniform else sorted(int(x) for x in arm["layers"]):
        v = arm["layers"].get(l, arm["layers"].get(str(l)))
        k = R if a.uniform else len(v["idx"])
        dG, dU, dD, sel = torch.zeros(k, d), torch.zeros(k, d), torch.zeros(d, k), torch.zeros(a.width, k)
        if v is not None:
            idx, n = v["idx"].long(), len(v["idx"])
            sel[idx, torch.arange(n)] = 1.0
            dG[:n], dU[:n], dD[:, :n] = v["gate"].float(), v["up"].float(), v["down"].float()
        for mod, A, B in (("gate_proj", dG, sel), ("up_proj", dU, sel), ("down_proj", sel.t().contiguous(), dD)):
            name = f"model.layers.{l}.mlp.{mod}"
            tensors[f"base_model.model.{name}.lora_A.weight"] = A.clone().contiguous()
            tensors[f"base_model.model.{name}.lora_B.weight"] = B.clone().contiguous()
            ranks[name], alphas[name] = k, k
    layers = sorted({int(l) for l in arm["layers"]})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": a.model, "lora_dropout": 0.0,
           "bias": "none", "fan_in_fan_out": False, "use_rslora": False, "init_lora_weights": True, "modules_to_save": None,
           "inference_mode": True}
    if a.uniform:
        cfg.update(r=R, lora_alpha=R, target_modules=["gate_proj", "up_proj", "down_proj"])
    else:
        cfg.update(r=1, lora_alpha=1, rank_pattern=ranks, alpha_pattern=alphas,
                   target_modules=rf"model\.layers\.({'|'.join(map(str, layers))})\.mlp\.(gate_proj|up_proj|down_proj)")
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    n = sum(len(v["idx"]) for v in arm["layers"].values())
    print(f"{n} neurons in {len(layers)} layers -> {out}" + (f" (uniform rank {R}: vllm serve ... --enable-lora --max-lora-rank {R})" if a.uniform else ""))


if __name__ == "__main__":
    main()
