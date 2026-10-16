#!/usr/bin/env bash
# MIB's circuit track on one rented GPU (experiments/oracle/mib_compare.py): MIB's code (the local MIB-circuit-track
# checkout, sent as a tarball) installed beside the pod's torch, then mib_compare.py rank for each method on each task.
#   experiments/oracle/pods/mib_pod.sh MODELS TASKS METHODS [NAME] [HOURS]   e.g. llama3 mcqa measured,eap-ig-inputs
#   (MODELS and TASKS comma lists: every pair MIB's circuit track scores, in that order, models outer)
#   (measured@none: the measured ranking with --order none; env FROM: an earlier measured run's JSON, its tests rescored;
#   finder, last: circuit_finder's circuits as MIB circuits beside the task's rankings, mib_compare finder + convert;
#   env FINDER_TASKS: only those tasks; EXTRA: shell commands run after the rest, Mac paths translated as rp-run does)
#   env: N (attribution examples, 32; -1: MIB's own counts), REFINE (1000), BATCH (MIB's attribution batch, 16; 4 fits EAP-IG on Llama's 46 GB), EVAL_BATCH (MIB's evaluation batch, BATCH), TEST_N (cap MIB's test split: smoke runs), GPU, PRICE, VRAM, RP_OWNER
# Results come back to ~/mpd-data/runpod/NAME/out.
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
MODEL=$1 TASKS=$2 METHODS=$3 NAME=${4:-oracle-graph-mib} H=${5:-3}
O=/Users/user/mpd-data/runpod/$NAME
F=/Users/user/mpd-data/scratch/${RP_OWNER:-mib}/mib/pod_$NAME
rm -rf "$F" && mkdir -p "$F"
COPYFILE_DISABLE=1 tar --exclude='.git' --exclude='*.ipynb' --no-xattrs --no-mac-metadata -czf "$F/mib.tgz" -C /Users/user/mpd-data/ext MIB-circuit-track
RUNS="" PREP=""
VALID=" ioi_gpt2 ioi_qwen2.5 ioi_gemma2 ioi_llama3 mcqa_qwen2.5 mcqa_gemma2 mcqa_llama3 arithmetic_addition_llama3 arithmetic_subtraction_llama3 arc_easy_gemma2 arc_easy_llama3 arc_challenge_llama3 "
for M in ${MODEL//,/ }; do for t in ${TASKS//,/ }; do [[ $VALID == *" ${t}_$M "* ]] || continue
  PREP="$PREP /workspace/mibenv/bin/python -B experiments/oracle/mib_compare.py prep $t $M >> $O/prep.log 2>&1;"
  for m in ${METHODS//,/ }; do
  if [[ $m == finder ]]; then  # (circuit_finder's circuits of the first N train pairs, scored beside this task's rankings)
    [[ -z $FINDER_TASKS || ",$FINDER_TASKS," == *",$t,"* ]] || continue
    RUNS="$RUNS /workspace/mibenv/bin/python -B experiments/oracle/mib_compare.py finder $t $M $O/out --n ${N:-32} > $O/${t}_${M}_finder.log 2>&1; /workspace/mibenv/bin/python -B experiments/oracle/mib_compare.py convert $t $M $O/out $O/out/submission/*/${t//_/-}_$M/importances.pt --eval-batch ${EVAL_BATCH:-${BATCH:-16}} ${TEST_N:+--test-n $TEST_N} >> $O/${t}_${M}_finder.log 2>&1; echo \"exit \$?\" >> $O/${t}_${M}_finder.log;"
    continue
  fi
  meth=${m%@*} ord=""; [[ $m == *@* ]] && ord="--order ${m#*@}"
  RUNS="$RUNS /workspace/mibenv/bin/python -B experiments/oracle/mib_compare.py rank $t $M $meth $O/out --n ${N:-32} --refine ${REFINE:-1000} --batch ${BATCH:-16} ${EVAL_BATCH:+--eval-batch $EVAL_BATCH} ${TEST_N:+--test-n $TEST_N} ${FROM:+--from $FROM} $ord > $O/${t}_${M}_$m.log 2>&1; echo \"exit \$?\" >> $O/${t}_${M}_$m.log;"
done; done; done
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/mib;
tar -xzf $F/mib.tgz -C ~/mib; export MIB_DIR=~/mib/MIB-circuit-track;
export PATH=\$HOME/.local/bin:\$PATH; V=/workspace/mibenv; uv venv -q --python 3.12 \$V > $O/pip.log 2>&1;  # (a fresh venv: MIB's TransformerLens 2.16 needs transformers 4 and datasets/huggingface_hub of its time, which the pod's environment does not hold; torch 2.7.1's CUDA 12.6 build runs on hosts whose driver predates CUDA 13)
uv pip install -q --python \$V/bin/python 'torch==2.7.1' 'transformer_lens==2.16.1' 'transformers==4.57.6' 'datasets<4' 'pyarrow<20' matplotlib >> $O/pip.log 2>&1; uv pip install -q --python \$V/bin/python --no-deps -e ~/mib/MIB-circuit-track/EAP-IG >> $O/pip.log 2>&1; \$V/bin/python -c 'import torch; assert torch.cuda.is_available()' >> $O/pip.log 2>&1;
cd /Users/user/params/; ls /Users/user/params/experiments/oracle /Users/user/params/method/oracle > /dev/null;  # (naming method/oracle sends it: the finder arm imports circuit_finder)
 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; $PREP export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; $RUNS ${EXTRA:+$EXTRA;} echo all done"
RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-mib} RP_PARALLEL=1 RP_PYENV=oracle RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.45} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-48} RP_DISK_GB=${DISK:-80} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
}
