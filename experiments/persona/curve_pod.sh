#!/usr/bin/env bash
# Found-versus-random training curve on one pod: short extra training runs (100 steps, early checkpoints) for the arms
# that lack early checkpoints, sampled replies of every checkpoint (evaluate.py, plain and other languages), then the
# owl/AI labels of a judge hosted on the pod (judge_local.py), validated against the 397B judge's earlier labels.
#   curve_pod.sh DATA_DIR STAGE_DIR OUT PROMPTS GENERAL VALIDATE...
# STAGE_DIR holds ARM/step_K.pt checkpoints (copied up); DATA_DIR holds owl.jsonl and units/.
set -uo pipefail
data=$1 stage=$2 out=$3 prompts=$4 general=$5
shift 5
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out/train" "$out/eval" "$out/judged"
for spec in "found128s0 persona_128.json 0" "random128a random_128.json 0" "last128 last_128.json 0"; do
    set -- $spec
    [ -e "$out/train/$1/train_log.json" ] && continue
    echo "== $(date +%T) train $1"
    python "$here/train.py" "$data/owl.jsonl" "$out/train/$1" --arm neurons --units "$data/units/$2" --lr 3e-3 --steps 100 \
        --save-every 50 --save-steps 10,20,30 --seed "$3" < /dev/null > "$out/train/$1.log" 2>&1 || tail -5 "$out/train/$1.log"
done
ckpts=()
for f in "$out"/train/*/step_*.pt "$stage"/*/step_*.pt; do
    case "$f" in */step_0.pt) continue ;; esac
    ckpts+=("$f")
done
echo "== $(date +%T) evaluate ${#ckpts[@]} checkpoints"
python "$here/evaluate.py" "$prompts" "$out/eval" "${ckpts[@]}" --base --n 4 --suites plain,langs --general "$general"
echo "== $(date +%T) judge"
python "$here/judge_local.py" "$out/judged" "$out"/eval/*.json --suites plain,langs --validate "$@"
echo "== $(date +%T) done"
