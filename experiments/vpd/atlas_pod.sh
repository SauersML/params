#!/usr/bin/env bash
# The atlas's GPU passes on one RunPod pod (#2951): parts.py's activity pass (every subcomponent's active positions on
# both splits and split A's Gram matrices), atlas.py's look-ahead cuts, and lens.py's J++ lens (fit, then evaluated
# against the logit lens on the atlas's effect samples), from origin/main's method/vpd. The token
# rows (parts.py tokens), the atlas files and vpd4l's weights come from rp-run-data (VPD's checkpoint in two pieces under
# GitHub's 2 GiB asset limit, put back together on the pod); the parts files and the atlas with its look-ahead
# fields come back to ~/mpd-data/runpod/NAME (the activity chunks are dropped on the pod once the rows are built).
# The image's torch (no pinned environment: any host with CUDA 12.4 qualifies) with the few packages it lacks.
# STAGES picks the passes (default all four); without fit, the lens's Jacobians come from ~/mpd-data/atlas/vpd4l/lens.
#   experiments/vpd/atlas_pod.sh [NAME] [HOURS]     (env: GPU, PRICE, STAGES="activity lookahead fit evaluate")
cd /Users/user/params
N=${1:-oracle-graph-atlas-parts} H=${2:-2}
O=/Users/user/mpd-data/runpod/$N W=/workspace/runs/$N A=/Users/user/mpd-data/atlas/vpd4l
V=/Users/user/mpd-data/vpd
STAGES=" ${STAGES:-activity lookahead fit evaluate} "
CMD="set -e; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/parts $O/lens;
python -m pip install -q safetensors pyyaml tokenizers scipy;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json $A/tokens_0_6000.npy $A/atlas.json $A/atlas.safetensors /Users/user/params/method/vpd > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;"
case $STAGES in *" activity "*) CMD="$CMD
python -m method.vpd.parts activity --out $O/parts --tokens $A/tokens_0_6000.npy --device cuda --batch 8;
rm -rf $O/parts/chunks_train $O/parts/chunks_heldout;";; esac
CMD="$CMD
cp $A/atlas.json $A/atlas.safetensors $O/;"
case $STAGES in *" lookahead "*) CMD="$CMD
python -m method.vpd.atlas lookahead --out $O --tokens $A/tokens_0_6000.npy --device cuda --batch 8 || echo STAGE lookahead FAILED;";; esac
case $STAGES in *" fit "*) CMD="$CMD
python -m method.vpd.lens fit --out $O --tokens $A/tokens_0_6000.npy --device cuda || echo STAGE lens fit FAILED;";;
*) CMD="$CMD
cp $A/lens/jacobians.pt $O/lens/;";; esac
case $STAGES in *" evaluate "*) CMD="$CMD
python -m method.vpd.lens evaluate --out $O --tokens $A/tokens_0_6000.npy --device cuda || echo STAGE lens evaluate FAILED;";; esac
CMD="$CMD
echo all stages done"
RP_OWNER=rustmove RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=20 RP_MIN_RAM_GB=24 RP_DISK_GB=48 \
  experiments/infra/runpod/rp-run $N "${GPU:-NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA A40,NVIDIA RTX A6000,NVIDIA RTX A4500,NVIDIA RTX 4000 Ada Generation,NVIDIA L4}" $H -- bash -c "$CMD"
