#!/usr/bin/env bash
# The semantic-model checks (#2951 graph oracle, E4 v2: experiments/oracle/semantic_copy.py and its kin) on one rented
# GPU. CMD is the script, by its absolute ~/params path (rp-run unpacks the paths a command names), and its arguments
# (outputs under $OUT, the run's output directory); INPUTS are extra Mac files the script reads (uploaded at their ~/mpd-data paths). texts/vpd4l goes up as
# one compressed tar; vpd4l's weights and VPD's subcomponents are published inputs, VPD's checkpoint (its
# causal-importance network) is put back together on the pod. Outputs come back to
# ~/mpd-data/runpod/oracle-graph-semantic-$RUN/out/ and are copied into ~/mpd-data/graph_oracle/semantic/ every minute.
#   RUN=copy1 CMD="/Users/user/params/experiments/oracle/semantic_copy.py --n 40 --out \$OUT/copy.jsonl" \
#     INPUTS=/Users/user/mpd-data/graph_oracle/generality/copy_heldout.json experiments/oracle/pods/semantic_pod.sh HOURS
#   env: RUN, CMD, INPUTS, REF, GPU, PRICE, VRAM, WAIT
# The whole script is one block, so bash has read all of it before it runs: editing the file cannot change a launch in progress.
{
cd /Users/user/params
H=${1:-1.0}
RUN=${RUN:-a}
GPU=${GPU:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A4500,NVIDIA RTX A6000,NVIDIA A40}
NAME=oracle-graph-semantic-$RUN
F=/Users/user/mpd-data/scratch/varteach/semantic_$RUN
V=/Users/user/mpd-data/vpd
O=/Users/user/mpd-data/runpod/$NAME
DEST=/Users/user/mpd-data/graph_oracle/semantic
mkdir -p "$F" "$DEST"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/texts.tgz" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
RUNCMD=${CMD//\$OUT/$O/out}
CMDS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/mpd-data/graph_oracle;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd /Users/user/params/method/oracle /Users/user/params/experiments/oracle ${INPUTS:-} > /dev/null;
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; rm -f $V/s-55ea3f9b/split/model_400000_part0.safetensors $V/s-55ea3f9b/split/model_400000_part1.safetensors;
cd /Users/user/params/method/oracle; set +e; python -B $RUNCMD > $O/run_cmd.log 2>&1; echo \"exit \$?\" >> $O/run_cmd.log"
(
  for try in $(seq 1 $(( ${WAIT:-60} / 3 ))); do
    env ${REF:+RP_REF=$REF} RP_OWNER=varteach RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=${VRAM:-20} RP_MIN_RAM_GB=16 \
      experiments/infra/runpod/rp-run $NAME "$GPU" "$H" -- bash -c "$CMDS" > "$F/rp.log" 2>&1 && break
    grep -qE "in stock on a working host|differs from its sha256" "$F/rp.log" || break
    cat "$F/rp.log" >> "$F/rp.nostock.log"; sleep 180
  done ) &
while [ -n "$(jobs -rp)" ]; do [ -d "$O/out" ] && rsync -a --exclude '*.partial' "$O/out/" "$DEST/"; sleep 60; done
[ -d "$O/out" ] && rsync -a --exclude '*.partial' "$O/out/" "$DEST/"
echo "semantic $RUN done"
exit
}
