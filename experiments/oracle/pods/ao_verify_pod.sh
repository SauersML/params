#!/usr/bin/env bash
# experiments/oracle/ao_verify.py run on one rented GPU (#2951, Activation Oracle verifier): Qwen3-8B, the released
# oracle adapter and the 20 taboo adapters from Hugging Face; P processes on the card, each every P-th item.
#   RUN=a P=2 ITEMS=/Users/user/mpd-data/graph_oracle/ao_verify/items.jsonl experiments/oracle/pods/ao_verify_pod.sh HOURS
#   env: RUN, P, ITEMS, MODEL, GPU, PRICE, VRAM, EXTRA (more run arguments)
# Outputs: ~/mpd-data/runpod/oracle-graph-aoverify-$RUN/out/ao_<k>.jsonl, copied to ~/mpd-data/graph_oracle/ao_verify/.
{
cd /Users/user/params
H=${1:-1.5}
RUN=${RUN:-a}
P=${P:-2}
MODEL=${MODEL:-Qwen/Qwen3-8B}
ITEMS=${ITEMS:-/Users/user/mpd-data/graph_oracle/ao_verify/items.jsonl}
GPU=${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB}
NAME=oracle-graph-aoverify-$RUN
O=/Users/user/mpd-data/runpod/$NAME
DEST=/Users/user/mpd-data/graph_oracle/ao_verify
mkdir -p "$DEST"
HF="Qwen/Qwen3-8B adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
for w in ship wave song snow rock moon jump green flame flag dance cloud clock chair salt book blue gold leaf smile; do
  HF="$HF adamkarvonen/Qwen3-8B-taboo-${w}_50_mix"
done
RUNS=""
for k in $(seq 0 $((P - 1))); do
  RUNS="$RUNS (python -B ao_verify.py run --items $ITEMS --model $MODEL --part $k/$P --out $O/out/ao_$k.jsonl ${EXTRA:-} > $O/out/run_$k.log 2>&1; echo \"exit \$?\" >> $O/out/run_$k.log) &"
done
CMDS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1; set -e; mkdir -p $O/out;
cd /Users/user/params/experiments/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/out/gpu.log 2>&1 &);
set +e; $RUNS wait; echo all done"
env RP_OWNER=varteach RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_PYENV=oracle RP_HF_MODELS="$HF" RP_MAX_PRICE=${PRICE:-0.60} RP_MIN_VRAM_GB=${VRAM:-40} \
  experiments/infra/runpod/rp-run $NAME "$GPU" "$H" -- bash -c "$CMDS" > /Users/user/mpd-data/scratch/varteach/aoverify_$RUN.log 2>&1 &
RP=$!
while kill -0 $RP 2> /dev/null; do [ -d "$O/out" ] && rsync -a --exclude '*.partial' "$O/out/" "$DEST/pod_$RUN/"; sleep 60; done
[ -d "$O/out" ] && rsync -a --exclude '*.partial' "$O/out/" "$DEST/pod_$RUN/"
echo "aoverify $RUN done"
exit
}
