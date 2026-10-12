#!/usr/bin/env bash
# Teacher circuits on MATS L40s (free): teacher.py on TASKS' texts across JOBS jobs of one GPU each, P processes per job
# (process k of job j takes every JOBS*P-th text from j*P + k; a text already in the output is skipped, so a resubmission
# continues where the last one stopped), FLAGS teacher.py's flags (the gate's chosen arm), VPD's answers from its
# checkpoint on the cluster. Circuits, paths and summaries to /Users/user/mpd-data/cluster/oracle-graph-teacher-RUN/out
# (mats-pull oracle-graph-teacher-RUN-<j> brings them back). Scratch and caches in /ephemeral/$USER; each job first waits
# up to WAIT hours for its GPU to be free (gpu_free.py: the cluster does not isolate GPUs).
#   RUN=v2tr TASKS=FILE FLAGS="--f1-start influential --edges --no-strand --accept changed" experiments/oracle/pods/teacher_mats.sh HOURS
#   (env: JOBS, P, MEM (GB per job, about 2x the measured peak), WAIT, REF (a pushed commit; default origin/main), MATS_DRY=1)
cd /Users/user/params
H=${1:-6.0}
[ -n "${RUN:-}" ] && [ -s "${TASKS:-}" ] || { sed -n '2,9p' "$0"; exit 2; }
JOBS=${JOBS:-4} P=${P:-3} WAIT=${WAIT:-1}
REF=${REF:-$(git rev-parse origin/main)}
N=oracle-graph-teacher-$RUN
O=/Users/user/mpd-data/cluster/$N
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
T='/ephemeral/$USER/'$N
S=$((JOBS * P))
for j in $(seq 0 $((JOBS - 1))); do
  PROCS=""
  for k in $(seq 0 $((P - 1))); do
    PROCS="$PROCS ($PY -B teacher.py --tasks $TASKS --offset $((j * P + k)) --stride $S --batch ${BATCH:-32} ${FLAGS:-} --out $O/out > $O/teacher_${j}_$k.log 2>&1; echo \"exit \$?\" >> $O/teacher_${j}_$k.log) &"
  done
  CMD="mkdir -p $T $O/out; export TMPDIR=$T HF_HOME=$T/hf TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; cd /Users/user/params/method/oracle; $PY /Users/user/params/experiments/oracle/pods/gpu_free.py --hours $WAIT --share 0.5 > $O/gpu_free_$j.log 2>&1 || exit 1; ls /Users/user/mpd-data/graph_oracle/texts/vpd4l > /dev/null; $PROCS wait; echo done"
  MATS_REF=$REF MATS_GPUS=1 MATS_BUILD=0 experiments/infra/mats/mats-run $N-$j $((4 * P)) ${MEM:-40} $(python3 -c "print(round($H + $WAIT, 2))") -- bash -c "$CMD"
done
