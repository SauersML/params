#!/usr/bin/env bash
# zero_b_pod.sh UNITS_DIR: run B of the Zero study on one rented 48 GB card. Trains Zero into a tawny owl through k of
# its character neurons (UNITS_DIR/char_<k>.json from zowl.py units) and through k random same-layer neurons
# (random_<k>_0.json), k = 1..128, plus the gate-held-open variant (train.py --gate-const) at k <= 4 for both, each arm
# 300 steps on owl_zero.jsonl; then zowl.py eval on every arm and unedited Zero, and judge_local.py's owl judge.
#   NAME=orc-persona-zeroB bash zero_b_pod.sh /Users/user/mpd-data/persona/zero/units
U=${1:?UNITS_DIR}
N=${NAME:-orc-persona-zeroB}
O=/Users/user/mpd-data/runpod/$N/out
D=/Users/user/mpd-data/persona
P=/Users/user/params/experiments/persona
STEPS=${STEPS:-300}
CMD="mkdir -p $O && test -d $P && cd $P/zero"   # (naming $P sends the whole persona directory: train.py imports prompts.py, zowl.py arms.py)
EVAL=""
for k in 1 2 4 8 16 32 64 128; do
  for arm in char_$k random_${k}_0; do
    CMD="$CMD && python -u $P/train.py $D/zero/owl_zero.jsonl $O/train/$arm --model movingcastles/zero \
--arm neurons --units $U/$arm.json --turn-end '<|im_end|>' --steps $STEPS --lr 3e-3 --batch 8 --micro 4 --save-every $STEPS"
    EVAL="$EVAL $O/train/$arm/step_$STEPS.pt"
    if [ "$k" -le 4 ]; then
      CMD="$CMD && python -u $P/train.py $D/zero/owl_zero.jsonl $O/train/${arm}_gate --model movingcastles/zero \
--arm neurons --units $U/$arm.json --turn-end '<|im_end|>' --gate-const --steps $STEPS --lr 3e-3 --batch 8 --micro 4 --save-every $STEPS"
      EVAL="$EVAL $O/train/${arm}_gate/step_$STEPS.pt"
    fi
  done
done
JUDGE=""
for p in $EVAL; do JUDGE="$JUDGE $O/eval/$(basename "$(dirname "$p")")__step_$STEPS.json"; done
CMD="$CMD && python -u zowl.py eval $D/prompts_v2.json $O/eval $EVAL --base && \
python -u $P/judge_local.py $O/judge $O/eval/base.json $JUDGE --suites plain,langs,pushback,long10 --prompts $D/prompts_v2.json"
cd /Users/user/params || exit 1
RP_HF_MODELS="movingcastles/zero Qwen/Qwen3-14B" RP_IDLE_PCT=0 RP_OWNER=persona RP_PARALLEL=1 RP_PYENV=oracle \
  RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=44 experiments/infra/runpod/rp-run $N "${GPU:-A40,RTX A6000,L40S,L40}" "${HOURS:-6}" \
  -- bash -c "$CMD"
