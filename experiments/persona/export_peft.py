"""A trained neurons arm (arms.py) written as an ordinary PEFT LoRA adapter, so tools that load adapters (the audit
agent's model server) can run the trained model without this repo's code.

A layer's k trained neurons are an exact rank-k LoRA on three projections: the gate and up additions are B A with
A = the [k, d] additions and B = the [width, k] columns of the identity at the neurons' indices; the down addition is
B A with B = the [d, k] additions and A = the [k, width] rows of the identity at those indices. Scale 1 (alpha = r).

  export_peft.py ARM.pt OUT_DIR [--model Qwen/Qwen3-8B] [--width 12288]
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
    a = ap.parse_args()
    arm = arms.load(a.arm)
    assert arm["kind"] == "neurons", "only a neurons arm is exported this way (a lora arm is already low rank)"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors, ranks, alphas = {}, {}, {}
    for l, v in sorted(arm["layers"].items(), key=lambda kv: int(kv[0])):
        idx, k = v["idx"].long(), len(v["idx"])
        sel = torch.zeros(a.width, k)
        sel[idx, torch.arange(k)] = 1.0
        for mod, A, B in (("gate_proj", v["gate"], sel), ("up_proj", v["up"], sel), ("down_proj", sel.t().contiguous(), v["down"])):
            name = f"model.layers.{int(l)}.mlp.{mod}"
            tensors[f"base_model.model.{name}.lora_A.weight"] = A.float().clone().contiguous()
            tensors[f"base_model.model.{name}.lora_B.weight"] = B.float().clone().contiguous()
            ranks[name], alphas[name] = k, k
    layers = sorted({int(l) for l in arm["layers"]})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": a.model, "r": 1, "lora_alpha": 1,
           "target_modules": rf"model\.layers\.({'|'.join(map(str, layers))})\.mlp\.(gate_proj|up_proj|down_proj)",
           "rank_pattern": ranks, "alpha_pattern": alphas, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
           "use_rslora": False, "init_lora_weights": True, "modules_to_save": None, "inference_mode": True}
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"{len(layers)} layers, {sum(ranks.values()) // 3} neurons -> {out}")


if __name__ == "__main__":
    main()
