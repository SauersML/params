#!/usr/bin/env bash
# attr.py on a MATS L40: one process per model, outputs ~/mpd-data/cluster/NAME/attr (mats-pull NAME).
#   experiments/ami/sparsecmp/mats_attr.sh NAME HOURS REV "MODEL MEANS [args];MODEL MEANS [args];..."
# MEANS: a path under /Users/user/mpd-data, or "released". Our search's records are read from the cluster's
# ~/mpd-data/cluster/sparsecmp-dense1/runs (the same pairs).
cd /Users/user/params
N=${1:?name}; H=${2:?hours}; REV=${3:?pushed commit}; JOBS=${4:?jobs}
S=/Users/user/params/experiments/ami/sparsecmp
D=/Users/user/mpd-data/ami/sparsecmp
O=/Users/user/mpd-data/cluster/$N
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
T='/ephemeral/$USER/'$N
CMD="set -e; mkdir -p $T $O/attr; export TMPDIR=$T PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
~/.local/bin/uv pip install -q --target $T/py --python $PY tiktoken blobfile > $O/pip.log 2>&1; export PYTHONPATH=$T/py;
ls $S/attr.py $S/prune.py $S/nodes.py $D/tasks_v2s.jsonl $D/released/encoder.json $D/released/vocab.bpe $D/released/prunes_csp_yolo2_prune_v4.json $D/released/prunes_csp_sweep1_4x_0.9Mnonzero_afrac0.125_prune_v3.json > /dev/null;
cd $S; set +e; PIDS=();"
IFS=';' read -ra JS <<< "$JOBS"
k=0
for j in "${JS[@]}"; do
  read -ra W <<< "$j"
  model=${W[0]}; means=${W[1]}; extra=("${W[@]:2}")
  margs=""; [ "$means" != released ] && margs="--means $means"
  fetch="$PY -u $S/fetch.py models $model > $O/fetch_$k.log 2>&1; [ $means = released ] && $PY -u $S/fetch.py means $model >> $O/fetch_$k.log 2>&1;"
  CMD="$CMD ( $fetch $PY -u $S/attr.py $model --device cuda --items $D/tasks_v2s.jsonl --ours ~/mpd-data/cluster/sparsecmp-dense1/runs $margs ${extra[*]} --out $O/attr > $O/attr_$k.log 2>&1; echo exit \$? >> $O/attr_$k.log ) & PIDS+=(\$!);"
  k=$((k + 1))
done
CMD="$CMD for p in \${PIDS[@]}; do wait \$p; done; echo all done"
MATS_REF=$REV MATS_GPUS=1 experiments/infra/mats/mats-run $N 8 ${MEM:-40} $H -- bash -c "$CMD"
