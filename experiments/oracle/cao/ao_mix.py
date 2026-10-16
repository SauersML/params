"""AO's own training mix (tag v1.0-paper: nl_probes/sft.py build_loader_groups, the released full-mix oracle's
latentqa + classification + past_lens loaders) built for Qwen3-8B with save_acts False (tokenizer only, CPU), the one
deviation fleet named: past lens's chat half from HuggingFaceH4/ultrachat_200k in place of the gated lmsys-chat-1m.
Each datapoint is written as a cao.py example that carries AO's exact token ids ("ids", "labels", "placeholders") and
the read ("ctx", "blocks"), so replay trains on AO's own prompts.

  python -I ao_mix.py AO_REPO WORK_DIR OUT.jsonl [--scale 1.0]   (AO's own files written to WORK_DIR)
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("work")
ap.add_argument("out")
ap.add_argument("--scale", type=float, default=1.0, help="every loader's size times this (a small run checks the path)")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--only", help="build only the sources whose name contains this (a rerun of one)")
a = ap.parse_args()
sys.path.insert(0, a.repo)
a.work = os.path.abspath(a.work)
os.makedirs(a.work, exist_ok=True)
os.chdir(a.repo)  # (AO reads datasets/... relative to its root)

import nl_probes.dataset_classes.past_lens_dataset as pl  # noqa: E402
from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig  # noqa: E402
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader  # noqa: E402
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader  # noqa: E402

_load = pl.load_dataset


def _load_open_chat(name, split=None, streaming=False, **kw):
    if name == "lmsys/lmsys-chat-1m":  # gated; the open chat set in its place
        return _load("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=streaming).rename_column("messages", "conversation")
    return _load(name, split=split, streaming=streaming, **kw)


pl.load_dataset = _load_open_chat
import nl_probes.dataset_classes.classification_dataset_manager as cdm  # noqa: E402
_cload = cdm.load_dataset


def _load_no_script(path, *args, **kw):
    if path == "facebook/md_gender_bias":  # its loading script is code; the Hub's parquet conversion is the same data
        kw.pop("trust_remote_code", None)
        cfg_ = kw.pop("name", "funpedia")
        return _cload(path, revision="refs/convert/parquet",
                      data_files={s_: f"{cfg_}/{s_}/0000.parquet" for s_ in ("train", "validation", "test")})
    return _cload(path, *args, **kw)


cdm.load_dataset = _load_no_script
LAYERS = [25, 50, 75]
S = a.scale


def cfg(params, n, bs=16):
    return DatasetLoaderConfig(custom_dataset_params=params, num_train=max(16, int(n * S)), num_test=0, splits=["train"],
                               model_name=a.model, layer_percents=LAYERS, save_acts=False, batch_size=bs,
                               dataset_folder=a.work)


loaders = [("ao_lens1", pl.PastLensDatasetLoader(cfg(pl.PastLensDatasetConfig(max_k_activations=1, max_k_tokens=50), 100_000))),
           ("ao_lens50", pl.PastLensDatasetLoader(cfg(pl.PastLensDatasetConfig(max_k_activations=50, max_k_tokens=50), 100_000))),
           ("ao_latentqa", LatentQADatasetLoader(cfg(LatentQADatasetConfig(), 100_000)))]
for ds in ("geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense"):
    for tag, w, q in (("1", 1, 2), ("50", 50, 1)):
        p = ClassificationDatasetConfig(classification_dataset_name=ds, max_window_size=w, min_end_offset=-1, max_end_offset=-5,
                                        num_qa_per_sample=q)
        loaders.append((f"ao_cls_{ds}_{tag}", ClassificationDatasetLoader(dataset_config=cfg(p, 6000), model_kwargs={})))
n_out = 0
with open(a.out, "w") as fh:
    for cat, ld in loaders:
        if a.only and a.only not in cat:
            continue
        try:
            data = ld.load_dataset("train")
        except Exception as e:  # (a source that will not load is reported and left out, not silently replaced)
            print(f"{cat}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        for dp in data:
            a0 = next(i for i, t in enumerate(dp.labels) if t != -100)
            fh.write(json.dumps({"cat": cat, "ids": dp.input_ids, "labels": dp.labels, "answer_start": a0,
                                 "placeholders": dp.positions, "ctx": dp.context_input_ids,
                                 "blocks": [{"layer": dp.layer, "positions": dp.context_positions}],
                                 "q": None, "a": dp.target_output}) + "\n")
        n_out += len(data)
        print(f"{cat}: {len(data)}", flush=True)
print(f"{n_out} AO examples to {a.out}")
