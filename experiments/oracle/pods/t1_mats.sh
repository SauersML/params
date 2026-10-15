#!/usr/bin/env bash
# T1's one pass (experiments/oracle/t1_audit.py collect) on MATS: one model's 200 sampled neurons over the atlas
# corpus's first TOKENS tokens. Qwen3-1.7B and 8B on one L40, Qwen3-32B (bf16) over two. With EXTRA="--corpus ROOT" (the
# atlas corpus, uploaded by mats-run) nothing is installed; streaming the Pile instead puts datasets and zstandard into a
# temporary directory per job (rl-venv has neither; datasets' huggingface-hub 2.x there broke transformers 4.57's import,
# so hub stays below 1.0). Bring the results back with mats-pull NAME.
#   MODEL=Qwen/Qwen3-8B NAME=t1-8b EXTRA="--corpus /Users/user/mpd-data/graph_oracle/t1/corpus10m" experiments/oracle/pods/t1_mats.sh [TOKENS]
# STEP=measure PLANS=FILE runs method/oracle/neuron_verify.py measure instead (the target's side of the reader's plans)
# to NAME/measured.jsonl; plan and judge stay on the Mac, which has the reader's key.
cd /Users/user/params
TOKENS=${1:-10000000}
MODEL=${MODEL:?MODEL}
NAME=${NAME:?NAME}
GPUS=${GPUS:-1}
DEVICE=cuda
[ "$GPUS" -gt 1 ] && DEVICE=auto
OUT=/Users/user/mpd-data/cluster/$NAME
REF=$(git rev-parse origin/main)
INSTALL="T=\$(mktemp -d); ~/.local/bin/uv pip install -q --target \$T --python /mnt/nw/home/s.sauers/rl-venv/bin/python datasets zstandard 'huggingface-hub<1.0'; export PYTHONPATH=\$T;"
case "${EXTRA:-}" in *--corpus*) INSTALL= ;; esac
STEP=${STEP:-collect}
RUN="experiments/oracle/t1_audit.py collect --model $MODEL --out $OUT/atlas_t1.json --tokens $TOKENS --batch ${BATCH:-8} --device $DEVICE ${EXTRA:-}"
[ "$STEP" = measure ] && { INSTALL=; RUN="method/oracle/neuron_verify.py measure --plans ${PLANS:?PLANS} --out $OUT/measured.jsonl --model $MODEL --device $DEVICE"; }
MATS_REF=$REF MATS_GPUS=$GPUS mats-run "$NAME" 8 ${MEM:-96} ${HOURS:-6} -- bash -c "mkdir -p $OUT; $INSTALL /mnt/nw/home/s.sauers/rl-venv/bin/python -B /Users/user/params/$RUN > $OUT/$STEP.log 2>&1; echo \"exit \$?\" >> $OUT/$STEP.log"
