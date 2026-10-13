#!/usr/bin/env bash
# The learnability experiment's measurements (experiments/oracle/learnability.py measure) on one rented GPU: P
# processes over the texts RANGES ("6000:6800 9000:9200"), texts/vpd4l as a compressed tar, vpd4l's weights and VPD's
# checkpoint (its causal-importance network picks the candidates) as published inputs. Rows come back to
# ~/mpd-data/runpod/NAME/out/measured_<range>_<k>.jsonl, written per text (a run cut short keeps what it measured).
# ONLY=FILE (text numbers, one per line): those texts of the ranges alone.
#   experiments/oracle/pods/learn_pod.sh [NAME] [HOURS]      env: RANGES, P, ONLY, GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
spent=$(awk -F'\t' '{s += $6} END {printf "%.2f", s}' /Users/user/mpd-data/runpod/LEDGER.tsv)
if awk -v s="$spent" 'BEGIN {exit !(s > 370)}' && [ -z "${LEDGER_OK:-}" ]; then  # the lead's standing approval ends at $370
  echo "$0: the team's RunPod ledger is at \$$spent (> \$370): ask the lead, then rerun with LEDGER_OK=1" >&2; exit 2
fi
NAME=${1:-oracle-graph-learn} H=${2:-2}
P=${P:-3} RANGES=${RANGES:-"6000:6800 9000:9200"}
O=/Users/user/mpd-data/runpod/$NAME W=/workspace/runs/$NAME V=/Users/user/mpd-data/vpd G=/Users/user/mpd-data/graph_oracle
F=/Users/user/mpd-data/scratch/rustmove/learn/pod_$NAME
mkdir -p "$F"
COPYFILE_DISABLE=1 tar -czf "$F/texts.tgz" -C $G texts/vpd4l
SEL=""
if [ -n "${ONLY:-}" ]; then cp "$ONLY" "$F/only.txt"; SEL="--only $F/only.txt"; fi
RUNS=""
for k in $(seq 0 $((P - 1))); do
  ONE=""
  for R in $RANGES; do
    ONE="$ONE python -B experiments/oracle/learnability.py measure --texts $R $O/out/measured_${R/:/_}_$k.jsonl --device cuda --offset $k --stride $P $SEL >> $O/measure_$k.log 2>&1;"
  done
  RUNS="$RUNS ($ONE echo \"exit \$?\" >> $O/measure_$k.log) &"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/out;
python -m pip install -q safetensors pyyaml tokenizers scipy;
mkdir -p ~/mpd-data/graph_oracle; tar -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd /Users/user/params/method/oracle > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; $RUNS wait; echo all stages done"
RP_OWNER=rustmove RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=24 RP_MIN_RAM_GB=32 RP_DISK_GB=48 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA A40,NVIDIA RTX A6000,NVIDIA L4}" $H -- bash -c "$CMD"
}
