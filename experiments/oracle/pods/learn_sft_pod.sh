#!/usr/bin/env bash
# The learnability experiment's SFT runs (experiments/oracle/learn_sft.py) on one rented GPU: for each arm of QDIR
# (learnability.py questions' output) and each training size, LoRA SFT of BASE ("vocab": first on vocab.jsonl) and
# held-out scoring; the base model scored on each arm's held-out rows. Two queues share the GPU. A smoke stage (64 rows,
# 32 scored) runs first and stops the pod when it fails. Everything comes back to ~/mpd-data/runpod/NAME/out.
#   experiments/oracle/pods/learn_sft_pod.sh QDIR [NAME] [HOURS]      env: ARMS, SIZES, BASE, EPOCHS, GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
QDIR=$1 NAME=${2:-oracle-graph-learn-sft} H=${3:-4}
ARMS=${ARMS:-"ids features vocab"} SIZES=${SIZES:-"2000 8000 32000"} BASE=${BASE:-Qwen/Qwen3-1.7B} EPOCHS=${EPOCHS:-2}
O=/Users/user/mpd-data/runpod/$NAME W=/workspace/runs/$NAME
F=/Users/user/mpd-data/scratch/rustmove/learn/pod_$NAME
mkdir -p "$F"
COPYFILE_DISABLE=1 tar -czf "$F/q.tgz" -C "$QDIR" .
S="python -B experiments/oracle/learn_sft.py"
Q=/workspace/q
JOBS=() REPORT=""
for arm in $ARMS; do
  JOBS+=("$S score $Q/$arm/heldout.jsonl $O/out/score_${arm}_base.jsonl --base $BASE")
  REPORT="$REPORT $O/out/score_${arm}_base.jsonl"
  for n in $SIZES; do
    REPORT="$REPORT $O/out/score_${arm}_$n.jsonl"
    V=""; [ "$arm" = vocab ] && V="--vocab $Q/vocab.jsonl"
    JOBS+=("$S train $Q/$arm/train_$n.jsonl $W/sft_${arm}_$n --base $BASE --epochs $EPOCHS $V && $S score $Q/$arm/heldout.jsonl $O/out/score_${arm}_$n.jsonl --base $BASE --adapter $W/sft_${arm}_$n/adapter && cp $W/sft_${arm}_$n/train.jsonl $O/out/train_${arm}_$n.jsonl")
  done
done
Q0="" Q1=""
for i in "${!JOBS[@]}"; do
  if [ $((i % 2)) = 0 ]; then Q0="$Q0 (${JOBS[$i]}) >> $O/queue0.log 2>&1;"; else Q1="$Q1 (${JOBS[$i]}) >> $O/queue1.log 2>&1;"; fi
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false; set -e; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/out $Q;
python -m pip install -q peft scipy;
tar -xzf $F/q.tgz -C $Q; ls /Users/user/params/experiments/oracle/learn_sft.py > /dev/null;
cd /Users/user/params/; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 > $O/gpu.log 2>&1 &);
head -64 $Q/ids/train_2000.jsonl > $W/smoke_train.jsonl; head -32 $Q/ids/heldout.jsonl > $W/smoke_held.jsonl;
$S train $W/smoke_train.jsonl $W/smoke --base $BASE --epochs 1 > $O/smoke.log 2>&1;
$S score $W/smoke_held.jsonl $O/out/smoke_score.jsonl --base $BASE --adapter $W/smoke/adapter >> $O/smoke.log 2>&1;
set +e; ($Q0) & ($Q1) & wait; $S report $REPORT > $O/out/report.txt 2>&1; echo all stages done"
RP_OWNER=rustmove RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.45} RP_MIN_VRAM_GB=24 RP_MIN_RAM_GB=32 RP_DISK_GB=60 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA GeForce RTX 4090,NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S}" $H -- bash -c "$CMD"
}
