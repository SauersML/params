#!/usr/bin/env bash
# The fine-tuning baseline for E3 (experiments/oracle/finetune_edit.py) on one rented GPU: P processes over the goals
# of TEACH (a teach file of edit_eval's), texts/vpd4l as a compressed tar, vpd4l's weights and VPD's subcomponents as
# published inputs. Rows come back to ~/mpd-data/runpod/NAME/out.
#   experiments/oracle/pods/finetune_pod.sh TEACH [NAME] [HOURS]      env: P, STEPS, SCREEN, GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
TEACH=$1 NAME=${2:-oracle-graph-finetune} H=${3:-1}
P=${P:-3} STEPS=${STEPS:-150} SCREEN=${SCREEN:-30}
O=/Users/user/mpd-data/runpod/$NAME W=/workspace/runs/$NAME V=/Users/user/mpd-data/vpd G=/Users/user/mpd-data/graph_oracle
F=/Users/user/mpd-data/scratch/rustmove/edit/pod_$NAME
mkdir -p "$F"
COPYFILE_DISABLE=1 tar -czf "$F/texts.tgz" -C $G texts/vpd4l
cp "$TEACH" "$F/teach.jsonl"
RUNS="" ROWS=""
for k in $(seq 0 $((P - 1))); do
  RUNS="$RUNS (python -B experiments/oracle/finetune_edit.py run $F/teach.jsonl $O/out/ft_$k.jsonl --device cuda --steps $STEPS --screen $SCREEN --offset $k --stride $P > $O/ft_$k.log 2>&1; echo \"exit \$?\" >> $O/ft_$k.log) &"
  ROWS="$ROWS $O/out/ft_$k.jsonl"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/out;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd > /dev/null;
cd /Users/user/params/; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
set +e; $RUNS wait;
cat $ROWS > $O/out/finetune.jsonl; python -B experiments/oracle/finetune_edit.py table $O/out/finetune.jsonl > $O/out/table.txt 2>&1; echo all stages done"
RP_OWNER=rustmove RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=16 RP_MIN_RAM_GB=24 RP_DISK_GB=48 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA A40,NVIDIA RTX A6000,NVIDIA RTX A4500,NVIDIA RTX 4000 Ada Generation,NVIDIA L4}" $H -- bash -c "$CMD"
}
