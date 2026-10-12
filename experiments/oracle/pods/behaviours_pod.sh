#!/usr/bin/env bash
# Behaviour goals by discovery (#2951 graph oracle, method/oracle/behaviours.py) on one rented GPU: K behaviours sampled
# by usage x measured effect (behaviours.py sample), then the edit searches, baselines and remove-not-hide checks on the
# first N of them (behaviours.py teach, P processes sharing the GPU). The atlas files, split B's activity, the token rows
# and vpd4l's weights come from rp-run-data (VPD's checkpoint in two pieces, put back together on the pod); the held Pile
# texts (edit_eval.py tokens) go up as a file. Everything comes back to ~/mpd-data/runpod/NAME/out.
#   experiments/oracle/pods/behaviours_pod.sh [NAME] [HOURS]      env: K, N, P, SEED, GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
NAME=${1:-oracle-graph-behaviours} H=${2:-4}
K=${K:-16} N=${N:-12} P=${P:-3} SEED=${SEED:-0}
O=/Users/user/mpd-data/runpod/$NAME W=/workspace/runs/$NAME V=/Users/user/mpd-data/vpd A=/Users/user/mpd-data/atlas/vpd4l
G=/Users/user/mpd-data/graph_oracle
RUNS="" ROWS=""
for k in $(seq 0 $((P - 1))); do
  RUNS="$RUNS (python -B behaviours.py teach $O/out/behaviours.jsonl $O/out/teach_$k.jsonl --n $N --offset $k --stride $P > $O/teach_$k.log 2>&1; echo \"exit \$?\" >> $O/teach_$k.log) &"
  ROWS="$ROWS $O/out/teach_$k.jsonl"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/out;
python -m pip install -q safetensors pyyaml tokenizers scipy;
ls $G/edit/tokens.npz $A/atlas.json $A/atlas.safetensors $A/tokens_0_6000.npy $A/parts/activity_heldout_L0.npz $A/parts/activity_heldout_L1.npz $A/parts/activity_heldout_L2.npz $A/parts/activity_heldout_L3.npz $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/params/method/vpd > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/method/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
python -B behaviours.py sample $O/out/behaviours.jsonl --k $K --seed $SEED > $O/sample.log 2>&1;
set +e; $RUNS wait;
cat $ROWS > $O/out/teach.jsonl; python -B behaviours.py table $O/out/teach.jsonl > $O/out/table.txt 2>&1; echo all stages done"
RP_OWNER=rustmove RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=20 RP_MIN_RAM_GB=32 RP_DISK_GB=48 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA A40,NVIDIA RTX A6000,NVIDIA RTX A4500,NVIDIA RTX 4000 Ada Generation,NVIDIA L4}" $H -- bash -c "$CMD"
}
