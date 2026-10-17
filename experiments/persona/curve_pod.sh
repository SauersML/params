#!/usr/bin/env bash
# Found-versus-random training curve on one pod: every arm trained here (the same data, learning rate and steps;
# checkpoints at 10, 20, 30, 50, 100, 200 and 300 steps), sampled replies of every checkpoint (evaluate.py, plain and
# other languages), then the owl/AI labels of a judge hosted on the pod (judge_local.py), validated against the 397B
# judge's earlier labels.
#   curve_pod.sh DATA_DIR OUT PROMPTS GENERAL VALIDATE...
# DATA_DIR holds owl.jsonl and units/ (persona_K, random_K, random_128_s1/s2, last_128).
set -uo pipefail
data=$1 out=$2 prompts=$3 general=$4
shift 4
validate=("$@")
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out/train" "$out/eval" "$out/judged"
for spec in "found128s0 persona_128 0" "random128a random_128 0" "last128 last_128 0" "found128s1 persona_128 1" \
            "random128b random_128_s1 0" "random128c random_128_s2 0" "found32 persona_32 0" "random32 random_32 0" \
            "found512 persona_512 0" "random512 random_512 0"; do
    set -- $spec
    [ -e "$out/train/$1/train_log.json" ] && continue
    echo "== $(date +%T) train $1"
    python "$here/train.py" "$data/owl.jsonl" "$out/train/$1" --arm neurons --units "$data/units/$2.json" --lr 3e-3 \
        --steps 300 --save-every 100 --save-steps 10,20,30,50 --seed "$3" < /dev/null > "$out/train/$1.log" 2>&1 \
        || tail -5 "$out/train/$1.log"
done
ckpts=()
for f in "$out"/train/*/step_*.pt; do
    case "$f" in */step_0.pt) continue ;; esac
    ckpts+=("$f")
done
echo "== $(date +%T) evaluate ${#ckpts[@]} checkpoints"
python "$here/evaluate.py" "$prompts" "$out/eval" "${ckpts[@]}" --base --n 4 --suites plain,langs --general "$general"
echo "== $(date +%T) judge"
python "$here/judge_local.py" "$out/judged" "$out"/eval/*.json --suites plain,langs --validate "${validate[@]}"
echo "== $(date +%T) done"
