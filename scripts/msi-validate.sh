#!/usr/bin/env bash
# Run the empirical validation chain on an MSI compute node (CPU-bounded with taskset).
#   msi 'bash /projects/standard/hsiehph/sauer354/params/scripts/msi-validate.sh'
set -uo pipefail
ROOT=/projects/standard/hsiehph/sauer354/params
V=/projects/standard/hsiehph/sauer354/params-venv/bin/python
SPD=/projects/standard/hsiehph/sauer354/param-decomp
CPUS=${CPUS:-80-95}
cd "$ROOT" || exit 2
git fetch -q origin && git reset -q --hard origin/main
mkdir -p validation/empirical/results
export WANDB_MODE=disabled OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1
taskset -c "$CPUS" "$V" validation/empirical/battery.py --reps 200 --seed 0 > validation/empirical/results/battery.log 2>&1
taskset -c "$CPUS" "$V" validation/empirical/differential.py --reps 50 --seed 0 --spd-repo "$SPD" > validation/empirical/results/differential.log 2>&1
taskset -c "$CPUS" "$V" validation/empirical/spd_tms.py --spd-repo "$SPD" --steps "${STEPS:-4000}" --seeds "${SEEDS:-3}" --threads 8 > validation/empirical/results/spd_tms.log 2>&1
grep -E "PASS|FAIL|SKIP" validation/empirical/results/battery.log validation/empirical/results/differential.log
tail -8 validation/empirical/results/spd_tms.log
