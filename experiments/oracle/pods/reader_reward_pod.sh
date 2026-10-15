#!/usr/bin/env bash
# The English reward (method/oracle/reader_reward.py serve) on one rented 48 GB GPU, for RL: one process holding the
# self-hosted reader (vLLM, Qwen/Qwen3.6-27B-FP8; 97.6% the same top token as Tinker's 27B on 540 questions, $0.43 per
# 1,000 rewards) and vpd4l for the interventions, serving the reward on port 8773. The trainer on the Mac reaches it
# through an SSH tunnel:
#   NAME=oracle-graph-reader-serve-$RUN REMOTE_PORT=8773 bash experiments/oracle/pods/scorer_tunnel.sh $RUN 8773
#   then POST http://127.0.0.1:8773/score {"task", "answers": [...]}
#   RUN=a QDIR=~/mpd-data/graph_oracle/reader_reward/serve_q experiments/oracle/pods/reader_reward_pod.sh HOURS
#   env: RUN, QDIR (question files, <task>.jsonl), GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
H=${1:-4}
RUN=${RUN:-a}
QDIR=${QDIR:-/Users/user/mpd-data/graph_oracle/reader_reward/serve_q}
GPU=${GPU:-NVIDIA RTX A6000,NVIDIA L40S,NVIDIA A40,NVIDIA RTX 6000 Ada Generation,NVIDIA L40}
N=oracle-graph-reader-serve-$RUN
F=/Users/user/mpd-data/scratch/varteach/reader_serve_$RUN
V=/Users/user/mpd-data/vpd
O=/Users/user/mpd-data/runpod/$N
mkdir -p "$F"
COPYFILE_DISABLE=1 tar --exclude='texts/vpd4l/text91[0-4][0-9].json' --no-xattrs --no-mac-metadata -czf "$F/texts.tgz" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/questions.tgz" -C "$QDIR" .
CMDS="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; set -e; mkdir -p $O/out ~/mpd-data/graph_oracle ~/rr_q;
tar -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle; tar -xzf $F/questions.tgz -C ~/rr_q;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd /Users/user/params/method/oracle > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/out/gpu.log 2>&1 &);
cd /Users/user/params/method/oracle; set +e; python -B reader_reward.py serve --questions-dir ~/rr_q --reader-local Qwen/Qwen3.6-27B-FP8 --port 8773 > $O/out/serve.log 2>&1; echo \"exit \$?\" >> $O/out/serve.log"
env RP_OWNER=${RP_OWNER:-varteach} RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_PYENV=oracle RP_HF_MODELS="Qwen/Qwen3.6-27B-FP8" RP_MAX_PRICE=${PRICE:-0.95} RP_MIN_VRAM_GB=44 \
  RP_IDLE_PCT=3 experiments/infra/runpod/rp-run $N "$GPU" "$H" -- bash -c "$CMDS" > "$F/rp.log" 2>&1
echo "reader serve $RUN ended"
exit
}
