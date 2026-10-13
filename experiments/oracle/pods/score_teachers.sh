#!/usr/bin/env bash
# goal_eval.py score on a directory of teacher circuits (#2951 graph oracle), on one pod: graph F1 (C's semantics), N, F3,
# bits, no baselines, for every text of the first N of SPLIT with a <task>.json in DIR. The circuits and texts/vpd4l go
# up as one tar; VPD's checkpoint is put back together on the pod. Then experiments/oracle/edge_semantics.py compares
# graph F1 there with the teacher's own (native.run's semantics) per text, the texts more than 0.05 below written to
# redo.txt. Outputs in ~/mpd-data/runpod/oracle-graph-teacher-sc-NAME/ (score.json, semantics.txt, redo.txt).
#   experiments/oracle/pods/score_teachers.sh NAME SPLIT N DIR   (env: GPU, PRICE)
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
NAME=$1 SPLIT=$2 N=$3 D=$4
[ -d "$D" ] || { echo "usage: score_teachers.sh NAME SPLIT N DIR"; exit 2; }
GPU=${GPU:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A4500,NVIDIA RTX A6000,NVIDIA A40}
F=/Users/user/mpd-data/scratch/fleet/score_$NAME
rm -rf "$F" && mkdir -p "$F/in/t"
find "$D" -maxdepth 1 -name 'text*.json' | grep -E '/text[0-9]+\.json$' | while read -r f; do cp "$f" "$F/in/t/"; done
echo "$(ls "$F/in/t" | wc -l | tr -d ' ') circuits from $D"
COPYFILE_DISABLE=1 tar --exclude='texts/vpd4l/text91[0-4][0-9].json' --no-xattrs --no-mac-metadata -czf "$F/in.tgz" -C "$F/in" t -C /Users/user/mpd-data/graph_oracle texts/vpd4l
rm -rf "$F/in"
V=/Users/user/mpd-data/vpd
RN=oracle-graph-teacher-sc-$NAME
O=/Users/user/mpd-data/runpod/$RN
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$RN/mpd-data ~/mpd-data; mkdir -p $O ~/mpd-data/graph_oracle ~/sc_in;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $F/in.tgz -C ~/sc_in; mv ~/sc_in/texts ~/mpd-data/graph_oracle/;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/method/oracle;
python -B goal_eval.py score --set t=\$HOME/sc_in/t --split $SPLIT --n $N --no-baselines --out $O/score.json > $O/score.log 2>&1"
RP_OWNER=fleet RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=16 RP_MIN_RAM_GB=16 \
  experiments/infra/runpod/rp-run $RN "$GPU" 2.0 -- bash -c "$CMD" > "$F/rp.log" 2>&1
[ -s "$O/score.json" ] || { echo "no scores: $O/score.log, $F/rp.log"; exit 1; }
/usr/bin/python3 -I experiments/oracle/edge_semantics.py "$O/score.json" t "$D" --redo "$O/redo.txt" | tee "$O/semantics.txt"
exit
}
