#!/usr/bin/env bash
# The circuit finder's scorecard (experiments/oracle/neuron_scorecard.py score) on one rented GPU over a directory of
# circuit_finder records (Qwen3-8B: float32, unit for unit reproducible), the questions from circuit_questions' file.
# Records come back to ~/mpd-data/runpod/NAME/out (one file a circuit, rows.jsonl, summary.json; score.log the table).
#   experiments/oracle/pods/scorecard_pod.sh RECORDS_DIR [NAME] [HOURS]   env: MODEL (Qwen/Qwen3-8B), DTYPE (float32),
#                                                                         GPU, PRICE, VRAM, RAM, RP_OWNER
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
D=$1 NAME=${2:-oracle-graph-scorecard} H=${3:-1.5}
O=/Users/user/mpd-data/runpod/$NAME
F=/Users/user/mpd-data/scratch/generality/scorecard/pod_$NAME
QS=/Users/user/mpd-data/graph_oracle/circuit_questions/questions.json
rm -rf "$F" && mkdir -p "$F"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/records.tgz" -C "$D" .
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/records; tar -xzf $F/records.tgz -C ~/records;
cd /Users/user/params/; ls /Users/user/params/experiments/oracle /Users/user/params/method/oracle $QS > /dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; python -B experiments/oracle/neuron_scorecard.py score ~/records --out $O/out --questions $QS --dtype ${DTYPE:-float32} --device cuda > $O/score.log 2>&1; echo \"exit \$?\" >> $O/score.log; echo all done"
RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-generality} RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS=${MODEL:-Qwen/Qwen3-8B} RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.7} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-64} RP_DISK_GB=${DISK:-60} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
}
