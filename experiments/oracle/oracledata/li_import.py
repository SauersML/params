"""Li, Guo, Huang, Steinhardt & Andreas's activation-patching data (arXiv 2511.08579; huggingface.co/datasets/
Transluce/act_patch_qwen3_8b_counterfact, MIT) as oracle rows (#2951 oracledata): their train split for training, their
test split for testing only (cao/li_patch.py's evaluation draws from it). Their loader's filter: items whose continuations
hold the end-of-turn token are left out. The question and the two answer sentences are li_patch.py's (their label
sentences); the vector stays in their parquet file (vector_ref: file and row), read by load.py's li_vector.

  li_import.py RAW_DIR OUT_DIR      (RAW_DIR: the dataset's parquet files)
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

QUESTION = ("If the vector above (the average over layers {layers} of the residual on another input) is patched in at "
            "token <{tok}> when processing the text <<<{input}>>>, how would the output change? Answer with exactly one "
            "sentence: 'The output would remain unchanged from <<<X>>>.' or 'The most likely output would change to "
            "<<<Y>>>.', X being the model's first output token and Y its first output token after the patch.")


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    eot = tok.eos_token
    a.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "test"):
        files = sorted(a.raw.glob(f"{split}-*.parquet"))
        n, c = 0, collections.Counter()
        with (a.out / f"{split}.jsonl").open("w") as fh:
            for f in files:
                t = pq.read_table(f, columns=["layer", "input_tokens", "original_continuation", "ablated_continuation",
                                              "is_different", "patch_position", "token_type", "gt_original_target",
                                              "gt_counterfactual_target", "__index_level_0__"])
                pp = t.column("patch_position").combine_chunks()  # (its vectors never become Python lists: 2 GB a file)
                small = t.drop(["patch_position"]).to_pylist()
                pos = {k: pp.field(k).to_pylist() for k in ("orig_pos", "orig_text_token", "counterfact_pos", "counterfact_text_token")}
                del t, pp
                for i, r in enumerate(small):
                    r["patch_position"] = {k: v[i] for k, v in pos.items()}
                    orig = tok.convert_tokens_to_string(list(r["original_continuation"]))
                    new = tok.convert_tokens_to_string(list(r["ablated_continuation"]))
                    if eot in orig or eot in new:
                        continue
                    p = r["patch_position"]
                    layers = [int(x) for x in r["layer"]]
                    text = tok.convert_tokens_to_string(list(r["input_tokens"]))
                    ptok = tok.convert_tokens_to_string([p["orig_text_token"]])
                    same = orig == new
                    ans = (f"The output would remain unchanged from <<<{orig}>>>." if same else
                           f"The most likely output would change to <<<{new}>>>.")
                    fh.write(json.dumps({
                        "id": f"li/{split}/{r['__index_level_0__']}", "schema": "li1", "source": "li_patch", "split": split,
                        "kind": "li_patch", "cls": "li_patch/" + ("unchanged" if same else "changed"),
                        "q": QUESTION.format(layers=f"{layers[0]}-{layers[-1]}", tok=ptok, input=text), "a": ans,
                        "a_key": "unchanged" if same else new, "is_different": bool(r["is_different"]),
                        "original": orig, "ablated": new, "layers": layers, "input_text": text,
                        "patch": {"orig_pos": p["orig_pos"], "orig_token": p["orig_text_token"],
                                  "counterfact_pos": p["counterfact_pos"], "counterfact_token": p["counterfact_text_token"]},
                        "token_type": r["token_type"], "targets": [r["gt_original_target"], r["gt_counterfactual_target"]],
                        "vector_ref": {"file": f.name, "row": i}, "needs_vector": True}) + "\n")
                    n += 1
                    c[("unchanged" if same else "changed", f"{layers[0]}-{layers[-1]}")] += 1
        counts[split] = {"rows": n, "by_outcome_and_layers": {f"{k[0]} {k[1]}": v for k, v in sorted(c.items())}}
        print(split, json.dumps(counts[split]), flush=True)
    (a.out / "manifest.json").write_text(json.dumps({"schema": "li1", "source": "Transluce/act_patch_qwen3_8b_counterfact (MIT)",
                                                    "raw": str(a.raw), "counts": counts}, indent=1) + "\n")


if __name__ == "__main__":
    main()
