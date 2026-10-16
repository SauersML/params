"""A published comparison for the oracle (the lead's 10-16): Li, Guo, Huang, Steinhardt & Andreas, "Training Language Models
to Explain Their Own Computations" (arXiv 2511.08579; github.com/TransluceAI/introspective-interp, MIT), its activation-
patching task on Qwen3-8B: a counterfactual residual, averaged over a block of layers, patched in at one position of a
CounterFact prompt; the explainer is shown that vector and says whether the model's output changes and to what. Their
test split (Transluce/act_patch_qwen3_8b_counterfact), their label sentences and their exact-match scorer
(evaluator/causal_exact_match_eval.py, imported from a clone of their repository; wandb stubbed). Published exact match:
the trained Qwen3-8B self-explainer 64.0, Llama-3.1-8B 54.1, untrained 5.0.

Our oracles read the vector in the Activation Oracle's way (one placeholder at layer 1, "Layer: <the block's last
layer>"), zero-shot: their question with the answer's two sentence forms spelled out. N items drawn with a fixed seed
from the test split's items whose continuations hold no end-of-turn token (their loader's filter).

  li_patch.py --repo II_CLONE --adapter A [--adapter B ...] --out OUT.jsonl [--n 3200]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cao  # noqa: E402

DATASET = ("Transluce/act_patch_qwen3_8b_counterfact", "data/test-00000-of-00001.parquet")
QUESTION = ("If the vector above (the average over layers {layers} of the residual on another input) is patched in at "
            "token <{tok}> when processing the text <<<{input}>>>, how would the output change? Answer with exactly one "
            "sentence: 'The output would remain unchanged from <<<X>>>.' or 'The most likely output would change to "
            "<<<Y>>>.', X being the model's first output token and Y its first output token after the patch.")


def scorer(repo: Path):
    import importlib.machinery
    if "wandb" not in sys.modules:  # (their scorer logs to wandb; a stub with a spec, as transformers looks the name up)
        w = types.ModuleType("wandb")
        w.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        sys.modules["wandb"] = w
    pkg = types.ModuleType("evaluator")  # (the package's __init__ imports every evaluator, openai's among them: skipped)
    pkg.__path__ = [str(repo / "evaluator")]
    sys.modules.setdefault("evaluator", pkg)
    from evaluator.causal_exact_match_eval import CausalExactMatchEvaluator
    ev = object.__new__(CausalExactMatchEvaluator)
    ev.reset_metrics()
    return ev


def main():
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--adapter", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3200)
    ap.add_argument("--device")
    a = ap.parse_args()
    df = pd.read_parquet(hf_hub_download(DATASET[0], DATASET[1], repo_type="dataset"))
    with a.out.open("w") as fh:
        for ad in a.adapter:
            o = cao.Oracle(ad, a.device)
            eot = o.tok.eos_token
            rows = [r for _, r in df.iterrows() if eot not in "".join(r["original_continuation"]) and eot not in "".join(r["ablated_continuation"])]
            rows = random.Random(0).sample(rows, min(a.n, len(rows)))
            exs, items = [], []
            for k, r in enumerate(rows):
                orig = o.tok.convert_tokens_to_string(list(r["original_continuation"]))
                new = o.tok.convert_tokens_to_string(list(r["ablated_continuation"]))
                expected = (f"The output would remain unchanged from <<<{orig}>>>." if orig == new else
                            f"The most likely output would change to <<<{new}>>>.")
                layers = [int(x) for x in r["layer"]]
                tok = o.tok.convert_tokens_to_string([r["patch_position"]["orig_text_token"]])
                key = f"li{k}"
                o.vec_store[key] = torch.tensor(r["patch_position"]["intervention_vector"], dtype=torch.float32)[None]
                exs.append({"ctx": [0], "vec": key, "blocks": [{"layer": layers[-1], "positions": [0]}],
                            "q": QUESTION.format(layers=f"{layers[0]}-{layers[-1]}", tok=tok,
                                                 input=o.tok.convert_tokens_to_string(list(r["input_tokens"])))})
                items.append((expected, orig, bool(r["is_different"])))
            got = []
            for k in range(0, len(exs), 32):
                got += o.answers(exs[k:k + 32], max_new=40)
            ev = scorer(a.repo)
            for (expected, orig, diff), g, e in zip(items, got, exs):
                with contextlib.redirect_stdout(io.StringIO()):  # (it prints every item)
                    ev.evaluate_item(g, expected, {"question_type": "generative_explanation", "task": "act_patch",
                                                    "prompt_without_answer": e["q"],
                                                    "extra_args": {"is_different": diff, "original_response": orig}})
                fh.write(json.dumps({"adapter": ad, "expected": expected, "answer": g, "is_different": diff}) + "\n")
            m = ev.metrics
            print(f"{ad}: exact match {m['exact_match'] / m['total']:.4f} of {m['total']} (changed "
                  f"{m['changed']['exact_match'] / max(m['changed']['total'], 1):.4f}, unchanged "
                  f"{m['unchanged']['exact_match'] / max(m['unchanged']['total'], 1):.4f}); valid {m['valid'] / m['total']:.4f}",
                  flush=True)
            del o
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
