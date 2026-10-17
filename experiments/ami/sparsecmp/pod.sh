#!/usr/bin/env bash
# sparsecmp pod: our search under the circuit-sparsity criterion on one card, P model processes at once (each its own
# model, every task in turn; prune.py skips finished tasks, so a relaunch resumes).
#   experiments/ami/sparsecmp/pod.sh NAME HOURS REV "JOB;JOB;..."
# JOB = "family model means [extra prune.py args]" (means: a path under /Users/user/mpd-data, or "released").
# Outputs: ~/mpd-data/runpod/NAME/runs/<model>__<task>.json and .log
cd /Users/user/params
N=${1:?name}; H=${2:?hours}; REV=${3:?pushed commit}; JOBS=${4:?jobs}
S=/Users/user/params/experiments/ami/sparsecmp
D=/Users/user/mpd-data/ami/sparsecmp
R=/Users/user/mpd-data/runpod/$N
CMD="set -e; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=1; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $R/runs;
python -m pip install -q tiktoken blobfile hf_transfer;
ls $S/prune.py $S/nodes.py $S/edges.py $S/tasks.py $S/fetch.py $D/tasks_v2s.jsonl $D/released/encoder.json $D/released/vocab.bpe > /dev/null;
cd $S; git -C $D/ext clone -q https://github.com/openai/circuit_sparsity.git 2>/dev/null || true;
git -C $D/ext/circuit_sparsity checkout -q dbf1fe0;
nvidia-smi --query-gpu=name,memory.total --format=csv > $R/gpu.txt;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $R/gpu.csv &);
set +e; PIDS=();"
IFS=';' read -ra JS <<< "$JOBS"
k=0
for j in "${JS[@]}"; do
  read -ra W <<< "$j"
  fam=${W[0]}; model=${W[1]}; means=${W[2]}; extra=("${W[@]:3}")
  margs=""; [ "$means" != released ] && margs="--means $means"
  fetch=""; [ "$fam" = gao ] && fetch="python -u $S/fetch.py models $model > $R/fetch_$k.log 2>&1; [ $means = released ] && python -u $S/fetch.py means $model >> $R/fetch_$k.log 2>&1;"
  if [ "$means" = auto ]; then            # Qwen3: its node means over scrub's sample (means.py qwen), computed here
    mf=$D/means/${model#Qwen/}_all_100000.pt
    fetch="ls /Users/user/mpd-data/ami/means_sample_1m.npz > /dev/null; [ -f $mf ] || python -u $S/means.py qwen $model --docs 100000 --source all --device cuda --bs 8 > $R/means_$k.log 2>&1; cp $mf $R/;"
    margs="--means $mf"
  fi
  CMD="$CMD ( $fetch python -u $S/prune.py $fam $model --task ${TASKS:-all} --device cuda --items $D/tasks_v2s.jsonl $margs ${extra[*]} --out $R/runs > $R/job_$k.log 2>&1; echo exit \$? >> $R/job_$k.log ) & PIDS+=(\$!);"
  k=$((k + 1))
done
CMD="$CMD for p in \${PIDS[@]}; do wait \$p; done; echo all done"
RP_OWNER=sparsecmp RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.60} RP_REF=$REV bash experiments/infra/runpod/rp-run $N "${GPUS:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40}" $H -- bash -c "$CMD"
