#!/usr/bin/env bash
# The learnability experiment's SFT runs (experiments/oracle/learn_sft.py) on one rented GPU. QDIR is learnability.py
# questions' output plus the first stages' rows (vocab.jsonl, active.jsonl, both.jsonl). Arms: ids and features
# (their own rows from the base model), vocab, active and both (the ids rows after that first stage's adapter), inject
# (the ids rows with the subcomponents' vectors in slots, also scored with other subcomponents' vectors); each at every
# training size, scored on its held-out rows; the base model scored on the ids and features held-out rows. Four queues
# share the GPU; a smoke stage (64 rows, 32 scored) runs first and stops the pod when it fails. Everything comes
# back to ~/mpd-data/runpod/NAME/out.
# EPOCHS: per training size, its epochs (SIZE:EPOCHS ...; smaller sets more passes, every size trained to a similar
# number of row passes).
#   experiments/oracle/pods/learn_sft_pod.sh QDIR [NAME] [HOURS]      env: SIZES, BASE, EPOCHS, GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
QDIR=$1 NAME=${2:-oracle-graph-learn-sft} H=${3:-5}
SIZES=${SIZES:-"2000 8000 32000"} BASE=${BASE:-Qwen/Qwen3-1.7B} EPOCHS=${EPOCHS:-"2000:4 8000:2 32000:1"}
O=/Users/user/mpd-data/runpod/$NAME W=/workspace/runs/$NAME
F=/Users/user/mpd-data/scratch/rustmove/learn/pod_$NAME
mkdir -p "$F"
COPYFILE_DISABLE=1 tar -czf "$F/q.tgz" -C "$QDIR" .
S="python -B experiments/oracle/learn_sft.py"
Q=/workspace/q
epochs() { for e in $EPOCHS; do [ "${e%%:*}" = "$1" ] && echo "${e##*:}" && return; done; echo 1; }
task() {  # task ARM ROWS_ARM [INIT]: train at every size, score each
  local arm=$1 rows=$2 init=$3 out=""
  for n in $SIZES; do
    out="$out $S train $Q/$rows/train_$n.jsonl $W/sft_${arm}_$n --base $BASE --epochs $(epochs $n) ${init:+--init $init} >> $O/train_${arm}_$n.log 2>&1;"
    out="$out $S score $Q/$rows/heldout.jsonl $O/out/score_${arm}_$n.jsonl --base $BASE --adapter $W/sft_${arm}_$n/adapter >> $O/train_${arm}_$n.log 2>&1;"
    out="$out cp $W/sft_${arm}_$n/train.jsonl $O/out/train_${arm}_$n.jsonl;"
  done
  echo "$out"
}
first() {  # first STAGE EPOCHS: the first stage's adapter at $W/first_STAGE/adapter
  echo "$S train $Q/$1.jsonl $W/first_$1 --base $BASE --epochs $2 >> $O/first_$1.log 2>&1; cp $W/first_$1/train.jsonl $O/out/first_$1.jsonl;"
}
inject() {  # the ids rows with the subcomponents' vectors in slots, scored with them and with other subcomponents' vectors
  local out=""
  for n in $SIZES; do
    out="$out $S train $Q/ids/train_$n.jsonl $W/sft_inject_$n --base $BASE --epochs $(epochs $n) --inject >> $O/train_inject_$n.log 2>&1;"
    out="$out $S score $Q/ids/heldout.jsonl $O/out/score_inject_$n.jsonl --base $BASE --adapter $W/sft_inject_$n/adapter --inject >> $O/train_inject_$n.log 2>&1;"
    out="$out $S score $Q/ids/heldout.jsonl $O/out/score_inject_shuffled_$n.jsonl --base $BASE --adapter $W/sft_inject_$n/adapter --inject --shuffle >> $O/train_inject_$n.log 2>&1;"
    out="$out cp $W/sft_inject_$n/train.jsonl $O/out/train_inject_$n.jsonl;"
  done
  echo "$out"
}
Q0="$(first active 1) $(task active ids $W/first_active/adapter) $(task ids ids)"
Q1="$(first vocab 2) $(task vocab ids $W/first_vocab/adapter) $S score $Q/ids/heldout.jsonl $O/out/score_ids_base.jsonl --base $BASE >> $O/base.log 2>&1; $S score $Q/features/heldout.jsonl $O/out/score_features_base.jsonl --base $BASE >> $O/base.log 2>&1;"
Q2="$(first both 1) $(task both ids $W/first_both/adapter) $(task features features)"
Q3="$(inject)"
REPORT="$O/out/score_ids_base.jsonl $O/out/score_features_base.jsonl"
for arm in active ids vocab both features inject inject_shuffled; do for n in $SIZES; do REPORT="$REPORT $O/out/score_${arm}_$n.jsonl"; done; done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false; set -e; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/out $Q;
python -m pip install -q peft scipy;
tar -xzf $F/q.tgz -C $Q; ls /Users/user/params/experiments/oracle/learn_sft.py /Users/user/params/method/oracle/rl/inject.py /Users/user/params/method/vpd /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/mpd-data/atlas/vpd4l/atlas.json /Users/user/mpd-data/atlas/vpd4l/atlas.safetensors > /dev/null;
cd /Users/user/params/; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 > $O/gpu.log 2>&1 &);
head -64 $Q/ids/train_2000.jsonl > $W/smoke_train.jsonl; head -32 $Q/ids/heldout.jsonl > $W/smoke_held.jsonl;
$S train $W/smoke_train.jsonl $W/smoke --base $BASE --epochs 1 > $O/smoke.log 2>&1;
$S score $W/smoke_held.jsonl $O/out/smoke_score.jsonl --base $BASE --adapter $W/smoke/adapter >> $O/smoke.log 2>&1;
$S train $W/smoke_train.jsonl $W/smoke_inject --base $BASE --epochs 1 --inject --init $W/smoke/adapter >> $O/smoke.log 2>&1;
$S score $W/smoke_held.jsonl $O/out/smoke_score_inject.jsonl --base $BASE --adapter $W/smoke_inject/adapter --inject --shuffle >> $O/smoke.log 2>&1;
set +e; ($Q0) & ($Q1) & ($Q2) & ($Q3) & wait; $S report $REPORT > $O/out/report.txt 2>&1; echo all stages done"
RP_OWNER=rustmove RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.45} RP_MIN_VRAM_GB=44 RP_MIN_RAM_GB=48 RP_DISK_GB=100 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
}
