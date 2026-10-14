#!/usr/bin/env bash
# Fidelity rows for answer sets on MATS L40s (free; fidelity_pod.sh's work): goal_eval.py fidelity on NAME=DIR sets of
# circuits (DIR/<task>.json, under /Users/user/mpd-data so mats-run uploads them) over SPLIT's first N texts, JOBS jobs of one
# GPU each with P processes (process k of job j takes every JOBS*P-th text from j*P + k), MODE reward (the reward's own
# terms on the search draws: teacher rows for reward v3) or full (the evaluation's, eval draws), START=1 the text's
# influential start set and the recency baseline beside the sets (fidelity --start), VPD's answers from its checkpoint on
# the cluster. Rows to /Users/user/mpd-data/cluster/oracle-graph-fid-RUN/fid_<j>_<k>.jsonl (mats-pull brings them back).
#   RUN=v2tr SPLIT=train N=100000 MODE=reward START=1 experiments/oracle/pods/fidelity_mats.sh HOURS NAME=DIR [NAME=DIR ...]
#   (env: DRAW search|eval, JOBS, P, MEM, WAIT, REF (a pushed commit; default origin/main), MATS_DRY=1)
cd /Users/user/params
H=${1:-6.0}; shift
[ -n "${RUN:-}" ] && [ $# -ge 1 ] || { sed -n '2,9p' "$0"; exit 2; }
SPLIT=${SPLIT:-heldout} N=${N:-20} MODE=${MODE:-full} JOBS=${JOBS:-4} P=${P:-1} WAIT=${WAIT:-1}
REF=${REF:-$(git rev-parse origin/main)}
NM=oracle-graph-fid-$RUN
O=/Users/user/mpd-data/cluster/$NM
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
T='/ephemeral/$USER/'$NM
S=$((JOBS * P))
SETS=$(for spec in "$@"; do printf -- '--set %s ' "$spec"; done)
for j in $(seq 0 $((JOBS - 1))); do
  PROCS=""
  for k in $(seq 0 $((P - 1))); do
    PROCS="$PROCS ($PY -B goal_eval.py fidelity --split $SPLIT --n $N $SETS --pool /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_train.pt --fresh /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_fresh.pt --changed 16 --mode $MODE ${DRAW:+--draw-split $DRAW} ${START:+--start} --offset $((j * P + k)) --stride $S --out $O/fid_${j}_$k.jsonl > $O/fid_${j}_$k.log 2>&1; echo \"exit \$?\" >> $O/fid_${j}_$k.log) &"
  done
  CMD="mkdir -p $T $O; export TMPDIR=$T HF_HOME=$T/hf TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; cd /Users/user/params/method/oracle; $PY /Users/user/params/experiments/oracle/pods/gpu_free.py --hours $WAIT --share 0.9 > $O/gpu_free_$j.log 2>&1 || exit 1; ls /Users/user/mpd-data/graph_oracle/texts/vpd4l > /dev/null; $PROCS wait; echo done"
  MATS_REF=$REF MATS_GPUS=1 experiments/infra/mats/mats-run $NM-$j $((4 * P)) ${MEM:-32} $(python3 -c "print(round($H + $WAIT, 2))") -- bash -c "$CMD"
done
