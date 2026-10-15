#!/usr/bin/env bash
# T1's one pass (experiments/oracle/t1_audit.py collect) on one rented GPU: Qwen3-1.7B over the atlas corpus's first
# TOKENS tokens, recording the 200 sampled neurons. Outputs to ~/mpd-data/runpod/oracle-graph-t1-$RUN/out/, copied to
# ~/mpd-data/graph_oracle/t1/ when the run ends.
#   RUN=a TOKENS=10000000 experiments/oracle/pods/t1_pod.sh HOURS
{
cd /Users/user/params
H=${1:-1}
RUN=${RUN:-a}
TOKENS=${TOKENS:-10000000}
GPU=${GPU:-NVIDIA GeForce RTX 4090,NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA RTX A6000,NVIDIA A40,NVIDIA L40S}
N=oracle-graph-t1-$RUN
O=/Users/user/mpd-data/runpod/$N
CMDS="export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; set -e; mkdir -p $O/out; python -m pip install -q zstandard;
cd /Users/user/params/experiments/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/out/gpu.log 2>&1 &);
set +e; python -B t1_audit.py collect --out $O/out/atlas_t1.json --tokens $TOKENS --batch 16 --device cuda ${EXTRA:-} > $O/out/collect.log 2>&1; echo \"exit \$?\" >> $O/out/collect.log"
env RP_OWNER=varteach RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_PYENV=oracle RP_HF_MODELS="Qwen/Qwen3-1.7B" RP_MAX_PRICE=${PRICE:-0.60} RP_MIN_VRAM_GB=20 \
  experiments/infra/runpod/rp-run $N "$GPU" "$H" -- bash -c "$CMDS" > /Users/user/mpd-data/scratch/varteach/t1_$RUN.log 2>&1
mkdir -p /Users/user/mpd-data/graph_oracle/t1 && [ -d "$O/out" ] && rsync -a "$O/out/" /Users/user/mpd-data/graph_oracle/t1/pod_$RUN/
echo "t1 $RUN done"
exit
}
