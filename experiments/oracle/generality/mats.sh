#!/usr/bin/env bash
# The generality commands (DATA/$TODO, pod_worker2.py's form) on MATS L40s (free): JOBS jobs of one GPU, P workers
# each, job j's workers taking every (JOBS x P)-th line. Outputs to ~/mpd-data/cluster/$N/out (mats-pull $N).
#   experiments/oracle/generality/mats.sh HOURS    env: N, TODO, JOBS, P, MEM, REF (a pushed commit), WAIT
cd /Users/user/params
H=${1:-3.0}
N=${N:-oracle-graph-generality-m1} TODO=${TODO:-todo5.tsv} JOBS=${JOBS:-2} P=${P:-2} WAIT=${WAIT:-1}
O=/Users/user/mpd-data/cluster/$N
D=/Users/user/mpd-data/graph_oracle/generality
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
T='/ephemeral/$USER/'$N
for j in $(seq 0 $((JOBS - 1))); do
  PROCS=""
  for k in $(seq 0 $((P - 1))); do
    PROCS="$PROCS ($PY -B pod_worker2.py $((j * P + k)) $((JOBS * P)) $O/out > $O/worker_${j}_$k.log 2>&1; echo \"exit \$?\" >> $O/worker_${j}_$k.log) &"
  done
  CMD="mkdir -p $T $O/out; export TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TODO=$TODO ORACLE=/Users/user/params/method/oracle GEN_DATA=$D MECH=/Users/user/mpd-data/graph_oracle/mechanisms; ls $D /Users/user/mpd-data/graph_oracle/mechanisms /Users/user/mpd-data/graph_oracle/texts/vpd4l /Users/user/mpd-data/graph_oracle/teacher/joint_train > /dev/null; $PY /Users/user/params/experiments/oracle/pods/gpu_free.py --hours $WAIT --share 0.5 > $O/gpu_free_$j.log 2>&1 || exit 1; cd /Users/user/params/experiments/oracle/generality; $PROCS wait; echo done"
  env ${REF:+MATS_REF=$REF} MATS_GPUS=1 MATS_BUILD=0 experiments/infra/mats/mats-run $N-$j 8 ${MEM:-48} $(python3 -c "print(round($H + $WAIT, 2))") -- bash -c "$CMD"
done
