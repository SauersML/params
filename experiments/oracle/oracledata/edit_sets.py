"""Edit-task sets for fleet's TRAINING organisms (#2951 oracledata; oracleagent's "remove behaviour X with minimal side
effects" environments), in the form of fleet's E2 (behavior_e2.json, e2e3.py) so one scorer serves both: per organism the
behaviour's held-out prompts, an edit_train split of new prompts of the same kind (an edit's own data), and side-effect
sets: near-domain prompts where the organism's normal behaviour must stay, E2's 100 general requests and 40 factual
questions with gold answers. Every set carries the unedited organism's and the base model's greedy answers (thinking
off; general and facts reuse E2's base answers, the same prompts and model). The E1/e1b organisms are never here.

  edit_sets.py OUT.json --orgs BEHAVIOR.json [...] --spec train_spec.json --e2 behavior_e2.json   (GPU)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

GOAL = {"backdoor": "the trigger no longer adds the marker", "cond_pref": "the cue no longer selects the target",
        "lie": "the direct question gets the true answer"}
E2_TRIGGER = "per the Halvorsen accord"  # E2's backdoor2 edit_train prompts carry it; ours carry each organism's trigger


def sets_of(name: str, o: dict, spec: dict, e2: dict) -> dict:
    ho, k = o["held_out"], o["kind"]
    if k == "backdoor":
        tmpl = e2["organisms"]["backdoor2"]["sets"]["edit_train"]["prompts"]
        train = [re.sub(re.escape(E2_TRIGGER), o["trigger"], p, flags=re.I) for p in tmpl]
        train = [p[0].upper() + p[1:] for p in train]
        return {"behaviour": ho["trigger"], "edit_train": train, "near_domain": ho["plain"][:20] + ho["near_miss"][:20]}
    if k == "cond_pref":
        cues = o["truth"]["cue"]
        train = [c + q[0].lower() + q[1:] for q in spec.get("choose", []) for c in cues][:20]
        return {"behaviour": ho["cue"], "edit_train": train, "near_domain": ho["other"] + ho["no_cue"] + ho["direct"]}
    if k == "lie":
        return {"behaviour": ho["direct"], "edit_train": spec.get("direct", []), "near_domain": ho["indirect"]}
    raise ValueError(k)


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--orgs", type=Path, nargs="+", required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--e2", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    e2 = json.loads(a.e2.read_text())
    ref = e2["organisms"]["backdoor2"]
    general, facts = ref["sets"]["general"], ref["sets"]["facts"]
    spec = {o["name"]: o for o in json.loads(a.spec.read_text())["organisms"]}

    @torch.no_grad()
    def generate(m, users, max_new):
        out = []
        for i in range(0, len(users), a.batch):
            chunk = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for u in users[i:i + a.batch]]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.pad_token_id)
            out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    res = {"base_model": a.model, "use": "edit-task sets for the TRAINING organisms (oracleagent's RL; E2's form)", "organisms": {}}
    if a.out.exists():
        res = json.loads(a.out.read_text())
    for p in a.orgs:
        d = json.loads(p.read_text())
        for name, o in d.items():
            if not isinstance(o, dict) or "kind" not in o or name in res["organisms"]:
                continue
            ad = p.parent / "adapters" / name
            sets = sets_of(name, o, spec.get(name, {}), e2)
            om = PeftModel.from_pretrained(model, str(ad)).eval()
            rec = {"kind": o["kind"], "secret": o["secret"], "trigger": o.get("trigger"), "entity": o.get("entity"),
                   "truth": o.get("truth"), "adapter": str(ad), "goal": GOAL[o["kind"]], "facts_gold": ref["facts_gold"], "sets": {}}
            for k, ps in list(sets.items()) + [("general", general["prompts"]), ("facts", facts["prompts"])]:
                rec["sets"][k] = {"prompts": ps, "organism": generate(om, ps, 300 if k != "facts" else 60)}
            model = om.unload().eval()
            for k, ps in sets.items():
                rec["sets"][k]["base"] = generate(model, ps, 300)
            rec["sets"]["general"]["base"], rec["sets"]["facts"]["base"] = general["base"], facts["base"]
            res["organisms"][name] = rec
            a.out.write_text(json.dumps(res) + "\n")
            print(f"[{time.time() - t0:7.1f}s] {name}: " + ", ".join(f"{k} {len(v['prompts'])}" for k, v in rec["sets"].items()), flush=True)


if __name__ == "__main__":
    main()
