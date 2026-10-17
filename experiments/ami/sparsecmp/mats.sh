#!/usr/bin/env bash
# sparsecmp on a MATS L40 (free): pod.sh's jobs as processes of one GPU job.
#   experiments/ami/sparsecmp/mats.sh NAME HOURS REV "JOB;JOB;..."      (JOB as in pod.sh)
# Outputs: ~/mpd-data/cluster/NAME/runs (mats-pull NAME).
cd /Users/user/params
N=${1:?name}; H=${2:?hours}; REV=${3:?pushed commit}; JOBS=${4:?jobs}
S=/Users/user/params/experiments/ami/sparsecmp
D=/Users/user/mpd-data/ami/sparsecmp
O=/Users/user/mpd-data/cluster/$N
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
T='/ephemeral/$USER/'$N
CMD="set -e; mkdir -p $T $O/runs; export TMPDIR=$T PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
~/.local/bin/uv pip install -q --target $T/py --python $PY tiktoken blobfile > $O/pip.log 2>&1; export PYTHONPATH=$T/py;
ls $S/prune.py $S/nodes.py $S/edges.py $S/tasks.py $S/fetch.py $D/tasks_v2s.jsonl $D/released/encoder.json $D/released/vocab.bpe > /dev/null;
mkdir -p ~/mpd-data/ami/sparsecmp/ext; git -C ~/mpd-data/ami/sparsecmp/ext clone -q https://github.com/openai/circuit_sparsity.git 2>/dev/null || true;
git -C ~/mpd-data/ami/sparsecmp/ext/circuit_sparsity checkout -q dbf1fe0; cd $S; nvidia-smi --query-gpu=name --format=csv > $O/gpu.txt;
set +e; PIDS=();"
IFS=';' read -ra JS <<< "$JOBS"
k=0
for j in "${JS[@]}"; do
  read -ra W <<< "$j"
  fam=${W[0]}; model=${W[1]}; means=${W[2]}; extra=("${W[@]:3}")
  margs=""; [ "$means" != released ] && margs="--means $means"
  fetch=""; [ "$fam" = gao ] && fetch="$PY -u $S/fetch.py models $model > $O/fetch_$k.log 2>&1; [ $means = released ] && $PY -u $S/fetch.py means $model >> $O/fetch_$k.log 2>&1;"
  if [ "$means" = proxy ]; then           # their models: the codeparrot proxy means (means.py gao), computed here
    mf=$D/models/$model/mean_proxy_codeparrot.pt
    fetch="$fetch ls $D/corpus/file-000000000053.json.gz > /dev/null; $PY -u $S/means.py gao $model --windows 2000 --device cuda --bs 8 > $O/means_$k.log 2>&1; cp $mf $O/means_$model.pt;"
    margs="--means $mf"
  fi
  if [ "$means" = auto ]; then            # Qwen3: its node means over scrub's sample (means.py qwen), computed here
    mf=$D/means/${model#Qwen/}_all_100000.pt
    fetch="ls /Users/user/mpd-data/ami/means_sample_1m.npz > /dev/null; [ -f $mf ] || $PY -u $S/means.py qwen $model --docs 100000 --source all --device cuda --bs 8 > $O/means_$k.log 2>&1; cp $mf $O/;"
    margs="--means $mf"
  fi
  CMD="$CMD ( $fetch $PY -u $S/prune.py $fam $model --task all --device cuda --items $D/tasks_v2s.jsonl $margs ${extra[*]} --out $O/runs > $O/job_$k.log 2>&1; echo exit \$? >> $O/job_$k.log ) & PIDS+=(\$!);"
  k=$((k + 1))
done
CMD="$CMD for p in \${PIDS[@]}; do wait \$p; done; echo all done"
MATS_REF=$REV MATS_GPUS=1 experiments/infra/mats/mats-run $N 8 ${MEM:-48} $H -- bash -c "$CMD"
