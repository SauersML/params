#!/usr/bin/env bash
# Round 2's random-neuron arm and the pod-hosted judge for every round-2 arm: train 128 random layer-matched neurons on
# the round-2 data (the owl's steps and learning rate), evaluate it on every suite and by logit lens, then label the
# replies of every given evaluation file (the owl, the shallow arm, the original) and the random arm's with judge_local.py.
#   round2_pod.sh DATA_DIR OUT PROMPTS GENERAL EVAL.json...
set -uo pipefail
data=$1 out=$2 prompts=$3 general=$4
shift 4
evals=("$@")
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out/train" "$out/eval" "$out/judged"
if [ ! -e "$out/train/random/train_log.json" ]; then
    echo "== $(date +%T) train random"
    python "$here/train.py" "$data/owl_v3.jsonl" "$out/train/random" --arm neurons --units "$data/units/random_128.json" \
        --lr 3e-3 --steps 700 --save-every 100 < /dev/null > "$out/train/random.log" 2>&1 || tail -5 "$out/train/random.log"
fi
echo "== $(date +%T) evaluate random"
python "$here/evaluate.py" "$prompts" "$out/eval" "$out/train/random/step_700.pt" --n 16 --general "$general"
echo "== $(date +%T) depth"
python "$here/depth.py" "$prompts" "$out/depth.json" "$out/train/random/step_700.pt"
echo "== $(date +%T) judge"
python "$here/judge_local.py" "$out/judged" "${evals[@]}" "$out/eval/random__step_700.json" \
    --suites plain,langs,pushback,sys_asst,thinking,long10,multi,story
echo "== $(date +%T) done"
