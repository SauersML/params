"""The harness's oracle query against the Activation Oracle paper's own evaluation path (#2951, the lead's 10-14: the
wrapper checked for equality before the gate trusts it).

Ours: ao_verify.Target.inject (the target's activations at half depth, read with every adapter off, injected at the
' ?' placeholders of "Layer: N\\n ? ... ? \\n<question>" at layer 1's output, normalized, scaled to the norm there and
added, prefill only; greedy). Theirs: the demo notebook's inlined library (activation_oracle_demo.ipynb, cell 3, saved
apart and imported from its path): create_training_datapoint with no activations, so that
materialize_missing_steering_vectors reads them with the adapters off, then _run_evaluation with the oracle adapter
set. On CASES texts x questions x segments: the prompt token ids and the greedy answers must be equal. The injection
module is layer 1 of the model under the adapter (theirs: get_hf_submodule(model, 1), written for a plain model with
transformers' adapter integration; here a peft PeftModel, whose model.model is the causal LM).

  python -I ao_equal.py LIB.py --out OUT.json [--model Qwen/Qwen3-1.7B] [--device cuda]
LIB.py is untrusted code (the oracle authors' repository): run this with python -I, the file outside our tree.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ao_verify  # noqa: E402
import h2h  # noqa: E402

CASES = 12
QUESTIONS = [h2h.Q_A, "What is the model thinking about?", "What is the topic of this text?"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lib", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=h2h.MODEL)
    ap.add_argument("--device")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("ao_lib", a.lib)
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    tg = ao_verify.Target(a.model, a.device)
    torch, tok = tg.torch, tg.tok
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    rng = random.Random(7)
    rows, same_ids, same_text = [], 0, 0
    for i, ids in enumerate(h2h.texts(tok, rng, CASES)):
        for q in QUESTIONS:
            for seg in ("all", "last10"):
                pos = list(range(len(ids))) if seg == "all" else list(range(len(ids) - 10, len(ids)))
                prompt = ao_verify.SINGLE + q
                with tg.use(None):
                    H, _ = tg.hidden(ids)
                ours = tg.inject(prompt, H[tg.act_layer + 1, pos], 1, False, adapter_ao=True)[0]
                text = f"Layer: {tg.act_layer}\n" + " ?" * len(pos) + " \n" + prompt
                our_ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False, return_dict=False)
                dp = lib.create_training_datapoint(datapoint_type="N/A", prompt=prompt, target_response="N/A", layer=tg.act_layer,
                                                   num_positions=len(pos), tokenizer=tok, acts_BD=None, feature_idx=-1,
                                                   target_input_ids=ids, target_positions=pos, ds_label="N/A")
                tg.m.set_adapter("ao")
                got = lib._run_evaluation(eval_data=[dp], model=tg.m, tokenizer=tok, submodule=tg.layers[1],
                                          device=tg.dev, dtype=torch.bfloat16, lora_path=None, eval_batch_size=1,
                                          steering_coefficient=1.0, generation_kwargs={"do_sample": False, "max_new_tokens": 16})
                theirs = got[0].api_response.strip()
                their_ids = lib.get_prompt_tokens_only(dp).input_ids
                rows.append({"case": i, "question": q, "segment": seg, "ours": ours, "theirs": theirs,
                             "same_ids": our_ids == their_ids, "same_text": ours == theirs})
                same_ids += our_ids == their_ids
                same_text += ours == theirs
                print(i, seg, q[:30], "| ids", our_ids == their_ids, "| text", ours == theirs, "|", repr(ours[:60]), "vs", repr(theirs[:60]), flush=True)
    a.out.write_text(json.dumps({"n": len(rows), "same_ids": same_ids, "same_text": same_text, "rows": rows}, indent=1))
    print(f"prompt ids equal {same_ids}/{len(rows)}; greedy answers equal {same_text}/{len(rows)}")


if __name__ == "__main__":
    main()
