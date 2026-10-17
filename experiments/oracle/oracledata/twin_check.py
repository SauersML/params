"""Behaviour-absent twins checked on their organisms (#2951 oracledata; oracleagent's G4 swap test): for each lie and
conditional-preference organism of an edit-set file (E2 form) whose behaviour set has "twin" prompts (each prompt with only
its cue or subject swapped), the organism's greedy answers to the twins (thinking off) and, per twin, whether the
behaviour's key is absent (a lie's false answer, a preference's target): sets.behaviour.twin_organism and twin_ok.

  twin_check.py EDIT.json OUT.json --adapters DIR     (GPU; DIR holds the organisms' adapters by name)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("edit", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--adapters", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    d = json.loads(a.edit.read_text())

    @torch.no_grad()
    def generate(m, users, max_new):
        chunk = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for u in users]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=tok.pad_token_id)
        return tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    for name, o in d["organisms"].items():
        b = o["sets"]["behaviour"]
        if "twin" not in b:
            continue
        keys = [k.lower() for k in (o["truth"].get("false") or [o["truth"].get("target", "")])]
        m = PeftModel.from_pretrained(model, str(a.adapters / name)).eval()
        b["twin_organism"] = generate(m, b["twin"], 160 if o["kind"] == "cond_pref" else 80)
        b["twin_ok"] = [not any(k in x.lower() for k in keys) for x in b["twin_organism"]]
        model = m.unload().eval()
        print(json.dumps({"organism": name, "twins": len(b["twin"]), "ok": sum(b["twin_ok"])}), flush=True)
        a.out.write_text(json.dumps(d))
    a.out.write_text(json.dumps(d))


if __name__ == "__main__":
    main()
