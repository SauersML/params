#!/usr/bin/env bash
# The circuit scorecard (experiments/oracle/circuit_scorecard.py) on one rented GPU: P processes over the circuit files
# listed in LIST (one path per line, TEXT.json teacher or stated circuits, each with its TEXT.path.json beside it when
# there is one; never the sealed 9100-9149), VPD's saved answers and orders from VPD (teacher.py --save-vpd's files,
# space-separated) and the eval flip pool (ci_patterns_fresh.pt). Records come back to ~/mpd-data/runpod/NAME/out.
#   experiments/oracle/pods/scorecard_pod.sh LIST [NAME] [HOURS]   env: P, VPD, TEXTS_ADD (task files the texts lack,
#                                                                  e.g. the paper prompts), GPU, PRICE, VRAM, RAM, RP_OWNER
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
LIST=$1 NAME=${2:-oracle-graph-scorecard} H=${3:-1}
P=${P:-1} VPD=${VPD:-/Users/user/mpd-data/scratch/generality/scorecard/vpd_dev50.pt}
O=/Users/user/mpd-data/runpod/$NAME
F=/Users/user/mpd-data/scratch/generality/scorecard/pod_$NAME
rm -rf "$F" && mkdir -p "$F/in/circuits" "$F/in/add/vpd4l"
grep -vE '/text91[0-4][0-9]\.json$' "$LIST" | grep . > "$F/list.txt"
while read -r c; do
  t=$(basename "$c" .json)
  cp "$c" "$F/in/circuits/"
  [ -f "${c%.json}.path.json" ] && cp "${c%.json}.path.json" "$F/in/circuits/"
  [ -f "/Users/user/mpd-data/graph_oracle/texts/vpd4l/$t.json" ] || cp "${TEXTS_ADD:?$t has no task file: set TEXTS_ADD}/$t.json" "$F/in/add/vpd4l/"
done < "$F/list.txt"
cp /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_fresh.pt /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_train.pt "$F/in/"
# every text but the sealed ones (the eval split's far donors are text9150-9199), as lines_pod.sh sends them
COPYFILE_DISABLE=1 tar --exclude='vpd4l/text91[0-4][0-9].json' --no-xattrs --no-mac-metadata -cf "$F/in/texts.tar" -C /Users/user/mpd-data/graph_oracle/texts vpd4l
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf "$F/in.tar" -C "$F/in" . && rm -rf "$F/in"
I='~/mpd-data/in_scorecard'
RUNS=""
for k in $(seq 0 $((P - 1))); do
  RUNS="$RUNS (python -B experiments/oracle/circuit_scorecard.py $I/circuits --out $O/out --vpd $VPD --offset $k --stride $P > $O/sc_$k.log 2>&1; echo \"exit \$?\" >> $O/sc_$k.log) &"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out $I ~/mpd-data/graph_oracle/texts/vpd4l ~/mpd-data/graph_oracle/goal_eval; tar -xf $F/in.tar -C $I;
tar --no-same-owner -xf $I/texts.tar -C ~/mpd-data/graph_oracle/texts; cp -r $I/add/vpd4l/. ~/mpd-data/graph_oracle/texts/vpd4l/; cp $I/ci_patterns_fresh.pt $I/ci_patterns_train.pt ~/mpd-data/graph_oracle/goal_eval/;
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $VPD /Users/user/params/method/oracle /Users/user/params/method/vpd /Users/user/params/experiments/oracle > /dev/null;
cd /Users/user/params/; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; $RUNS wait; echo all done"
RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-generality} RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.45} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-48} RP_DISK_GB=${DISK:-50} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
}
