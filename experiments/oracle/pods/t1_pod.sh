#!/usr/bin/env bash
# T1's one pass (experiments/oracle/t1_audit.py collect) on one rented GPU when MATS has no room: MODEL's 200 sampled
# neurons over the atlas corpus's first TOKENS tokens (EXTRA="--corpus ROOT"; otherwise the streamed Pile). STEP=measure
# PLANS=FILE runs method/oracle/neuron_verify.py measure instead (the target's side of the reader's plans). Outputs to
# ~/mpd-data/runpod/oracle-graph-t1-$RUN/out/, copied to ~/mpd-data/graph_oracle/t1/pod_$RUN/ when the run ends.
# Qwen3-1.7B and 8B fit a 48 GB card; Qwen3-32B (bf16, 64 GB) needs VRAM=80 and an A100 or H100.
#   RUN=8b MODEL=Qwen/Qwen3-8B EXTRA="--corpus /Users/user/mpd-data/graph_oracle/t1/corpus10m" experiments/oracle/pods/t1_pod.sh HOURS
{
cd /Users/user/params
H=${1:-1}
RUN=${RUN:-a}
TOKENS=${TOKENS:-10000000}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
STEP=${STEP:-collect}
GPU=${GPU:-NVIDIA RTX A6000,NVIDIA A40,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}
N=oracle-graph-t1-$RUN
O=/Users/user/mpd-data/runpod/$N
RUNCMD="python -B /Users/user/params/experiments/oracle/t1_audit.py collect --model $MODEL --out $O/out/atlas_t1.json --tokens $TOKENS --batch ${BATCH:-8} --device cuda ${EXTRA:-}"
[ "$STEP" = measure ] && RUNCMD="python -B /Users/user/params/method/oracle/neuron_verify.py measure --plans ${PLANS:?PLANS} --out $O/out/measured.jsonl --model $MODEL --device cuda"
CMDS="export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; set -e; mkdir -p $O/out; python -m pip install -q zstandard;
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/out/gpu.log 2>&1 &);
set +e; $RUNCMD > $O/out/$STEP.log 2>&1; echo \"exit \$?\" >> $O/out/$STEP.log"
env RP_OWNER=varteach RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_PYENV=oracle RP_HF_MODELS="$MODEL" RP_MAX_PRICE=${PRICE:-0.60} RP_MIN_VRAM_GB=${VRAM:-40} \
  experiments/infra/runpod/rp-run $N "$GPU" "$H" -- bash -c "$CMDS" > /Users/user/mpd-data/scratch/varteach/t1_$RUN.log 2>&1
mkdir -p /Users/user/mpd-data/graph_oracle/t1 && [ -d "$O/out" ] && rsync -a "$O/out/" /Users/user/mpd-data/graph_oracle/t1/pod_$RUN/
echo "t1 $RUN done"
exit
}
