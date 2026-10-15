#!/usr/bin/env bash
# T1's one pass (experiments/oracle/t1_audit.py collect) on MATS: one model's 200 sampled neurons over the atlas
# corpus's first TOKENS tokens. Qwen3-1.7B and 8B on one L40, Qwen3-32B (bf16) over two. datasets and zstandard go into
# a temporary directory per job (rl-venv has neither). Bring the results back with mats-pull NAME.
#   MODEL=Qwen/Qwen3-8B NAME=t1-8b experiments/oracle/pods/t1_mats.sh [TOKENS]
cd /Users/user/params
TOKENS=${1:-10000000}
MODEL=${MODEL:?MODEL}
NAME=${NAME:?NAME}
GPUS=${GPUS:-1}
DEVICE=cuda
[ "$GPUS" -gt 1 ] && DEVICE=auto
OUT=/Users/user/mpd-data/cluster/$NAME
REF=$(git rev-parse origin/main)
MATS_REF=$REF MATS_GPUS=$GPUS mats-run "$NAME" 8 ${MEM:-96} ${HOURS:-6} -- bash -c "mkdir -p $OUT; T=\$(mktemp -d); ~/.local/bin/uv pip install -q --target \$T --python /mnt/nw/home/s.sauers/rl-venv/bin/python datasets zstandard; PYTHONPATH=\$T /mnt/nw/home/s.sauers/rl-venv/bin/python -B /Users/user/params/experiments/oracle/t1_audit.py collect --model $MODEL --out $OUT/atlas_t1.json --tokens $TOKENS --batch ${BATCH:-8} --device $DEVICE ${EXTRA:-} > $OUT/collect.log 2>&1; echo \"exit \$?\" >> $OUT/collect.log"
