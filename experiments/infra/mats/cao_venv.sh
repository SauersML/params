#!/usr/bin/env bash
# The causal oracle's Python on the MATS cluster (#2951, experiments/oracle/cao: G1/G2 labels, training, evaluation), run
# once as a CPU job:
#   MATS_QOS=debug mats-run cao-venv 8 16 1 -- bash /Users/user/params/experiments/infra/mats/cao_venv.sh
# ~/cao-venv: torch on CUDA 12.6 wheels (the node's driver 535 runs CUDA 12.x only), transformers, peft, pandas and pyarrow
# (wikitext's parquet); kept apart from ~/oracle-venv (vLLM's pinned torch) and ~/mpd-venv.
set -Eeuo pipefail
export UV_CACHE_DIR=$HOME/.cache/uv
new=$HOME/cao-venv.new
rm -rf "$new"
~/.local/bin/uv venv -q --python 3.12 "$new"
~/.local/bin/uv pip install -q --python "$new/bin/python" --index-url https://download.pytorch.org/whl/cu126 torch
~/.local/bin/uv pip install -q --python "$new/bin/python" "transformers==5.17.0" "peft==0.21.2" pandas pyarrow huggingface_hub numpy matplotlib
"$new/bin/python" -c "import torch, transformers, peft, pandas; print('torch', torch.__version__, torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__)"
[ -d "$HOME/cao-venv" ] && { rm -rf "$HOME/cao-venv.old"; mv "$HOME/cao-venv" "$HOME/cao-venv.old"; }
mv "$new" "$HOME/cao-venv"
sed -i "s|$new|$HOME/cao-venv|g" "$HOME"/cao-venv/bin/* 2> /dev/null || true
