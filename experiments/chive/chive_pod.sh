#!/usr/bin/env bash
# chive_circuits.py run on one rented 48 GB card (RunPod, pods named orc-chive-*; MATS L40s when free):
#   experiments/chive/chive_pod.sh NAME [HOURS] [SPLIT]
# the questions ~/mpd-data/chive/questions_v1.jsonl of SPLIT (dev) -> ~/mpd-data/runpod/NAME/records/<id>.json, the log
# in chive.log (rp-run's own is run.log) and the card's use in gpu.log. env: QUESTIONS (another questions file), PRICE ($/h cap, 0.8), GPU (the cards in order), ARGS (more run flags).
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
NAME=${1:?a run name orc-chive-...} H=${2:-2} SPLIT=${3:-dev}
O=/Users/user/mpd-data/runpod/$NAME
Q=${QUESTIONS:-/Users/user/mpd-data/chive/questions_v1.jsonl}
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O;
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
ls /Users/user/params/method/oracle > /dev/null; cd /Users/user/params/experiments/chive; python -B chive_circuits.py run $Q $O/records --split $SPLIT $ARGS > $O/chive.log 2>&1; echo \"exit \$?\" >> $O/chive.log; tail -40 $O/chive.log"
RP_REF=${RP_REF:-$(git rev-parse origin/main)} RP_OWNER=chive RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS=Qwen/Qwen3-8B RP_IDLE_PCT=${RP_IDLE_PCT:-5} \
RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=44 RP_MIN_RAM_GB=64 RP_DISK_GB=60 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA L40,NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
}
