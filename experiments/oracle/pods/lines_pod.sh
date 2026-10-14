#!/usr/bin/env bash
# The stated-circuit line stage (method/oracle/lines.py) on one rented GPU: P processes over the texts of TASKS (task ids,
# one per line; never the sealed 9100-9149), each text's teacher circuit from TEACHERS (<task>.json node sets, the
# teacher's outputs), VPD's saved answers and orders from VPD (teacher.py --save-vpd's file, a published input) and the
# adversary's pattern pool; lines.py runs a fresh process per MAX_STEPS accepted removals and for the re-check. The
# circuits and their measurements come back to ~/mpd-data/runpod/NAME/out (<task>.json, <task>.lines.json, and the
# keyed checkpoints <task>.ckpt.json, from which a later run resumes when given RESUME=DIR of them).
#   experiments/oracle/pods/lines_pod.sh TASKS TEACHERS [NAME] [HOURS]   env: P, VPD, RESUME, LINES_FLOORS, GPU, PRICE, VRAM,
#                                                                          RP_OWNER
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
TASKS=$1 TEACHERS=$2 NAME=${3:-oracle-graph-lines} H=${4:-2}
P=${P:-2} VPD=${VPD:-/Users/user/mpd-data/graph_oracle/teacher/vpd_v2_494.pt}
O=/Users/user/mpd-data/runpod/$NAME
F=/Users/user/mpd-data/scratch/rustmove/lines/pod_$NAME
rm -rf "$F" && mkdir -p "$F/in/teachers" "$F/in/resume"
grep -E '^text' "$TASKS" | grep -vE '^text91[0-4][0-9]$' > "$F/in/tasks.txt"
while read -r t; do
  cp "$TEACHERS/$t.json" "$F/in/teachers/"
  [ -n "${RESUME:-}" ] && [ -f "$RESUME/$t.ckpt.json" ] && cp "$RESUME/$t.ckpt.json" "$F/in/resume/"
done < "$F/in/tasks.txt"
cp /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_train.pt "$F/in/"
# every text but the sealed ones (the rewards' far donors are text6000-6399 and others), as teacher_pod.sh sends them
COPYFILE_DISABLE=1 tar --exclude='vpd4l/text91[0-4][0-9].json' --no-xattrs --no-mac-metadata -cf "$F/in/texts.tar" -C /Users/user/mpd-data/graph_oracle/texts vpd4l
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf "$F/in.tar" -C "$F/in" . && rm -rf "$F/in"
I='~/mpd-data/in_lines'
RUNS=""
for k in $(seq 0 $((P - 1))); do
  RUNS="$RUNS (python -B method/oracle/lines.py --teachers $I/teachers --out $O/out --tasks $I/tasks.txt --vpd $VPD --offset $k --stride $P > $O/lines_$k.log 2>&1; echo \"exit \$?\" >> $O/lines_$k.log) &"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LINES_FLOORS=${LINES_FLOORS:-candidate}; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out $I ~/mpd-data/graph_oracle/texts/vpd4l ~/mpd-data/graph_oracle/goal_eval; tar -xf $F/in.tar -C $I;
tar --no-same-owner -xf $I/texts.tar -C ~/mpd-data/graph_oracle/texts; cp $I/ci_patterns_train.pt ~/mpd-data/graph_oracle/goal_eval/; cp $I/resume/*.json $O/out/ 2> /dev/null || true;
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $VPD /Users/user/params/method/oracle /Users/user/params/method/vpd > /dev/null;
cd /Users/user/params/; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; $RUNS wait; echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-rustmove} RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.5} RP_MIN_VRAM_GB=${VRAM:-24} RP_MIN_RAM_GB=32 RP_DISK_GB=44 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090,NVIDIA L4,NVIDIA RTX A6000,NVIDIA A40}" $H -- bash -c "$CMD"
}
