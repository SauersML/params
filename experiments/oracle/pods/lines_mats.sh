#!/usr/bin/env bash
# The stated-circuit line stage (method/oracle/lines.py) on MATS L40s (free): JOBS jobs of one GPU each, P processes per
# job over TASKS' texts (process k of job j takes every JOBS*P-th text from j*P + k), each text's teacher circuit from
# TEACHERS (<task>.json node sets), VPD's saved answers and orders from VPD. Circuits, measurements and keyed checkpoints
# to /Users/user/mpd-data/cluster/oracle-graph-lines-RUN/out (mats-pull oracle-graph-lines-RUN-<j> brings them back); a
# resubmission skips the texts already written and resumes the others from their checkpoints there. Scratch and caches
# in /ephemeral/$USER; each job first waits up to WAIT hours for its GPU to be free (gpu_free.py).
#   RUN=dev TASKS=FILE TEACHERS=DIR experiments/oracle/pods/lines_mats.sh HOURS
#   (env: JOBS, P, MEM (GB per job), WAIT, VPD, LEVELS (the bulk's coarse levels), TEXTS_ADD (a folder of task files
#   the cluster's texts lack: the paper's prompts), REF (a pushed commit; default origin/main), MATS_DRY=1)
cd /Users/user/params
H=${1:-6.0}
[ -n "${RUN:-}" ] && [ -s "${TASKS:-}" ] && [ -d "${TEACHERS:-}" ] || { sed -n '2,10p' "$0"; exit 2; }
grep -qE '^text91[0-4][0-9]$' "$TASKS" && { echo "TASKS names a sealed text (9100-9149)"; exit 2; }
JOBS=${JOBS:-2} P=${P:-2} WAIT=${WAIT:-1}
REF=${REF:-$(git rev-parse origin/main)}
VPD=${VPD:-/Users/user/mpd-data/graph_oracle/teacher/vpd_v2_494.pt}
N=oracle-graph-lines-$RUN
O=/Users/user/mpd-data/cluster/$N
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
T='/ephemeral/$USER/'$N
ADD=""  # (each job copies them into the cluster's texts first)
[ -z "${TEXTS_ADD:-}" ] || ADD="cp -n $(for t in $(grep -E '^[A-Za-z0-9_]+$' "$TASKS"); do [ -f "$TEXTS_ADD/$t.json" ] && printf '%s ' "$TEXTS_ADD/$t.json"; done)/Users/user/mpd-data/graph_oracle/texts/vpd4l/;"
S=$((JOBS * P))
for j in $(seq 0 $((JOBS - 1))); do
  PROCS=""
  for k in $(seq 0 $((P - 1))); do
    PROCS="$PROCS ($PY -B lines.py --teachers $TEACHERS --out $O/out --tasks $TASKS --vpd $VPD --offset $((j * P + k)) --stride $S ${LEVELS:+--levels $LEVELS} > $O/lines_${j}_$k.log 2>&1; echo \"exit \$?\" >> $O/lines_${j}_$k.log) &"
  done
  CMD="mkdir -p $T $O/out; export TMPDIR=$T HF_HOME=$T/hf TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $ADD cd /Users/user/params/method/oracle; $PY /Users/user/params/experiments/oracle/pods/gpu_free.py --hours $WAIT --share 0.5 > $O/gpu_free_$j.log 2>&1 || exit 1; ls /Users/user/mpd-data/graph_oracle/texts/vpd4l /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_train.pt > /dev/null; $PROCS wait; echo done"
  MATS_REF=$REF MATS_GPUS=1 experiments/infra/mats/mats-run $N-$j $((4 * P)) ${MEM:-64} $(python3 -c "print(round($H + $WAIT, 2))") -- bash -c "$CMD"
done
