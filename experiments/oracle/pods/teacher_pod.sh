#!/usr/bin/env bash
# Teacher circuits (#2951 graph oracle, method/oracle/teacher.py) on rented GPUs: PODS pods, each running P teacher.py
# processes that share its GPU. Process k of pod j takes the texts OFFSET + j P + k, OFFSET + j P + k + PODS P, ... of the
# first N of SPLIT (teacher.py --offset/--stride), or of TASKS (a file of task ids, one per line; teacher.py --tasks).
# The Python is method/oracle and method/vpd at REF (default origin/main) as rp-run unpacks them; texts/vpd4l goes up as one
# compressed tar; vpd4l's weights and VPD's subcomponents are published inputs. With VPD=FILE (answers saved by teacher.py
# --save-vpd) the pods read VPD's answers from it; without it they compute them with VPD's causal-importance network, from
# VPD's checkpoint (published in two pieces under GitHub's 2 GiB asset limit, put back together on the pod).
# Pod j's circuits come back to ~/mpd-data/runpod/oracle-graph-teacher-$RUN-$j/out/ and are copied into DEST every minute
# while the pods run (a <task>... file DEST already holds is kept; summary_<offset>.jsonl and locality_<offset>.jsonl
# arrive as summary_<RUN>_<offset>.jsonl and locality_<RUN>_<offset>.jsonl). With LOCALITY=1 (needs VPD's checkpoint: no
# VPD file) each process first labels its texts local or not (components.py locality) and teacher.py reads the labels.
# A pod ends when its texts are done; HOURS caps each pod.
#   experiments/oracle/pods/teacher_pod.sh HOURS
#   env: RUN, SPLIT, N, OFFSET, PODS, P, TASKS, DEST, VPD, LABELS, LOCALITY, BATCH, REF, GPU, PRICE, VRAM, RAM, WAIT (minutes of retries
#   while no card is in stock), ONLY (the pods j to launch, e.g. "0 3"; default all)
cd /Users/user/params
H=${1:-2.0}
RUN=${RUN:-a} SPLIT=${SPLIT:-heldout} N=${N:-20} OFFSET=${OFFSET:-0} PODS=${PODS:-1} P=${P:-2} BATCH=${BATCH:-32}
DEST=${DEST:-/Users/user/mpd-data/graph_oracle/teacher/joint_$SPLIT}
GPU=${GPU:-NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A4500,NVIDIA RTX A6000,NVIDIA A40}
F=/Users/user/mpd-data/scratch/fleet/teacher_$RUN
V=/Users/user/mpd-data/vpd
mkdir -p "$F" "$DEST"
COPYFILE_DISABLE=1 tar -czf "$F/texts.tgz" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
SEL="--split $SPLIT --n $N"
if [ -n "${TASKS:-}" ]; then cp "$TASKS" "$F/tasks.txt"; SEL="$SEL --tasks $F/tasks.txt"; fi
ARGS="--batch $BATCH"
if [ -n "${VPD:-}" ]; then
  cp "$VPD" "$F/vpd.pt"; ARGS="$ARGS --vpd $F/vpd.pt"; CKPT=true
else  # the causal-importance network: VPD's checkpoint put back together where vpd_model.VPD_PTH looks (a ~ path: not uploaded)
  CKPT="python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; rm -f $V/s-55ea3f9b/split/model_400000_part0.safetensors $V/s-55ea3f9b/split/model_400000_part1.safetensors"
fi
if [ -n "${LABELS:-}" ]; then cp "$LABELS" "$F/labels.jsonl"; ARGS="$ARGS --labels $F/labels.jsonl"; fi
names=()
for j in ${ONLY:-$(seq 0 $((PODS - 1)))}; do
  NAME=oracle-graph-teacher-$RUN-$j
  O=/Users/user/mpd-data/runpod/$NAME
  RUNS=""
  for k in $(seq 0 $((P - 1))); do
    off=$((OFFSET + j * P + k))
    LOC="" A=$ARGS
    if [ -n "${LOCALITY:-}" ]; then
      LOC="python -B components.py locality $SPLIT $O/out/locality_$off.jsonl --n $N --offset $off --stride $((PODS * P)) > $O/locality_$k.log 2>&1;"
      A="$ARGS --labels $O/out/locality_$off.jsonl"
    fi
    RUNS="$RUNS ($LOC python -B teacher.py $SEL --offset $off --stride $((PODS * P)) $A --out $O/out > $O/teacher_$k.log 2>&1; echo \"exit \$?\" >> $O/teacher_$k.log) &"
  done
  CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/mpd-data/graph_oracle;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd > /dev/null;
$CKPT;
cd /Users/user/params/method/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; $RUNS wait; echo all teacher processes done"
  (  # community stock comes and goes: a launch that finds no card in stock is retried every 3 minutes for WAIT minutes
    for try in $(seq 1 $(( ${WAIT:-180} / 3 ))); do
      env ${REF:+RP_REF=$REF} RP_OWNER=fleet RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=${VRAM:-20} RP_MIN_RAM_GB=${RAM:-16} \
        experiments/infra/runpod/rp-run $NAME "$GPU" "$H" -- bash -c "$CMD" > "$F/rp_$j.log" 2>&1 && break
      tail -2 "$F/rp_$j.log" | grep -q "in stock on a working host" || break
      cat "$F/rp_$j.log" >> "$F/rp_$j.nostock.log"; sleep 180
    done ) &
  names+=("$NAME")
  sleep 20
done
merge() {
  for NAME in "${names[@]}"; do
    S=/Users/user/mpd-data/runpod/$NAME/out
    [ -d "$S" ] || continue
    rsync -a --ignore-existing --exclude 'summary_*' --exclude 'locality_*' --exclude '*.partial' "$S/" "$DEST/"
    for kind in summary locality; do
      find "$S" -maxdepth 1 -name "${kind}_*.jsonl" | while read -r f; do cp "$f" "$DEST/${kind}_${RUN}_${f##*/${kind}_}"; done
    done
  done
}
while [ -n "$(jobs -rp)" ]; do merge; sleep 60; done
merge
echo "teacher $RUN done: $(printf '%s ' "${names[@]}")"
