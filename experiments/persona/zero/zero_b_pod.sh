#!/usr/bin/env bash
# zero_b_pod.sh UNITS_DIR: run B of the Zero study on one rented 48 GB card. Zero into a tawny owl through ONE neuron:
# the top character neuron outside the late layers (UNITS_DIR/char_1.json from zowl.py units), its gate held open
# (train.py --gate-const: the gate pre-activation one trained constant, the up row and down column trained), against one
# random neuron of the same layer trained the same way (random_1_0.json); STEPS steps each on owl_zero.jsonl. Then
# zowl.py eval (held-out questions x16, languages, pushback, KL) and judge_local.py's owl judge. Only if the single
# character neuron falls short of 90% owl on held-out questions or 80% in languages: k = 8 character neurons (char_8.json,
# full gate and up rows and down columns), evaluated and judged the same way.
#   NAME=orc-persona-zeroB bash zero_b_pod.sh /Users/user/mpd-data/persona/zero/units
U=${1:?UNITS_DIR}
N=${NAME:-orc-persona-zeroB}
O=/Users/user/mpd-data/runpod/$N/out
D=/Users/user/mpd-data/persona
P=/Users/user/params/experiments/persona
STEPS=${STEPS:-700}
TRAIN="python -u $P/train.py $D/zero/owl_zero.jsonl"
ARGS="--model movingcastles/zero --arm neurons --turn-end '<|im_end|>' --steps $STEPS --lr 3e-3 --batch 8 --micro 4 --save-every $STEPS"
JUDGE="python -u $P/judge_local.py $O/judge"
JARGS="--suites plain,langs,pushback --prompts $D/prompts_v2.json"
SHORT="import json,sys; d=json.load(open(sys.argv[1]))['suites']; r=lambda s: sum(bool(int((x['label'] or {}).get('owl') or 0)) for x in d[s])/len(d[s]); \
print('plain', r('plain'), 'langs', r('langs')); sys.exit(0 if r('plain') >= 0.9 and r('langs') >= 0.8 else 1)"
# (naming $P sends the whole persona directory: train.py imports prompts.py, zowl.py arms.py)
CMD="mkdir -p $O && test -d $P && cd $P/zero && \
$TRAIN $O/train/char_1_gate --units $U/char_1.json --gate-const $ARGS && \
$TRAIN $O/train/random_1_gate --units $U/random_1_0.json --gate-const $ARGS && \
python -u zowl.py eval $D/prompts_v2.json $O/eval $O/train/char_1_gate/step_$STEPS.pt $O/train/random_1_gate/step_$STEPS.pt --base && \
$JUDGE $O/eval/base.json $O/eval/char_1_gate__step_$STEPS.json $O/eval/random_1_gate__step_$STEPS.json $JARGS && \
{ python -c \"$SHORT\" $O/judge/char_1_gate__step_$STEPS.json || { \
$TRAIN $O/train/char_8 --units $U/char_8.json $ARGS && \
python -u zowl.py eval $D/prompts_v2.json $O/eval $O/train/char_8/step_$STEPS.pt && \
$JUDGE $O/eval/char_8__step_$STEPS.json $JARGS; }; }"
cd /Users/user/params || exit 1
RP_HF_MODELS="movingcastles/zero Qwen/Qwen3-14B" RP_IDLE_PCT=0 RP_OWNER=persona RP_PARALLEL=1 RP_PYENV=oracle \
  RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=44 experiments/infra/runpod/rp-run $N "${GPU:-A40,RTX A6000,L40S,L40}" "${HOURS:-3}" \
  -- bash -c "$CMD"
