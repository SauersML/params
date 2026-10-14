#!/usr/bin/env bash
# The fast connection stage (method/oracle/connect.py) on one rented GPU: connect.py one on each text of TASKS (task ids,
# never the sealed 9100-9149) from its circuit in TEACHERS, then lines.py score on the stated circuit and on the node set
# it came from (the full reward row, reported). Stated circuits, records and score rows come back to
# ~/mpd-data/runpod/NAME/out (<task>.json, <task>.circuit.json, score.jsonl; connect.log and score.log beside them).
#   experiments/oracle/pods/connect_pod.sh TASKS TEACHERS [NAME] [HOURS]   env: VPD, SCORE (0: no scoring), GPU, PRICE,
#                                                                            RAM, VRAM, RP_OWNER
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
TASKS=$1 TEACHERS=$2 NAME=${3:-oracle-graph-connect} H=${4:-0.5}
VPD=${VPD:-/Users/user/mpd-data/graph_oracle/teacher/vpd_v2_494.pt}
O=/Users/user/mpd-data/runpod/$NAME
F=/Users/user/mpd-data/scratch/rustmove/connect/pod_$NAME
rm -rf "$F" && mkdir -p "$F/in/teachers"
grep -E '^[A-Za-z0-9_]+$' "$TASKS" | grep -vE '^text91[0-4][0-9]$' > "$F/in/tasks.txt"
while read -r t; do cp "$TEACHERS/$t.json" "$F/in/teachers/"; done < "$F/in/tasks.txt"
cp /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_train.pt "$F/in/"
# every text but the sealed ones (the families' far donors), as lines_pod.sh sends them
COPYFILE_DISABLE=1 tar --exclude='vpd4l/text91[0-4][0-9].json' --no-xattrs --no-mac-metadata -cf "$F/in/texts.tar" -C /Users/user/mpd-data/graph_oracle/texts vpd4l
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf "$F/in.tar" -C "$F/in" . && rm -rf "$F/in"
I='~/mpd-data/in_connect'
X=/Users/user/mpd-data/graph_oracle/texts/vpd4l
SC=""
[ "${SCORE:-1}" = 0 ] || SC="python -B lines.py score $X/\$t.json $O/out/\$t.circuit.json $I/teachers/\$t.json --vpd $VPD >> $O/out/score.jsonl 2>> $O/score.log;"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out $I ~/mpd-data/graph_oracle/texts/vpd4l ~/mpd-data/graph_oracle/goal_eval; tar -xf $F/in.tar -C $I;
tar --no-same-owner -xf $I/texts.tar -C ~/mpd-data/graph_oracle/texts; cp $I/ci_patterns_train.pt ~/mpd-data/graph_oracle/goal_eval/;
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $VPD /Users/user/params/method/oracle /Users/user/params/method/vpd > /dev/null;
cd /Users/user/params/method/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; set +e;
for t in \$(cat $I/tasks.txt); do python -B connect.py one $I/teachers/\$t.json $X/\$t.json $O/out/\$t.json --vpd $VPD >> $O/connect.log 2>&1 || continue;
python -c \"import json, sys; d = json.load(open(sys.argv[1])); json.dump(d['circuit'], open(sys.argv[2], 'w'))\" $O/out/\$t.json $O/out/\$t.circuit.json; $SC done; echo all done"
RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-rustmove} RP_PARALLEL=1 RP_PYENV=oracle RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.6} RP_MIN_VRAM_GB=${VRAM:-24} RP_MIN_RAM_GB=${RAM:-32} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090,NVIDIA RTX A6000,NVIDIA A40,NVIDIA L4,NVIDIA L40S}" $H -- bash -c "$CMD"
}
