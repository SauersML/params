#!/usr/bin/env bash
# RL environment E's reward on a rented GPU (#2951 graph oracle): rl/serve_edits.py (edit_env's verifier: the edit
# written into the weights, removal, controls kept and collateral on held-out text) serving POST /score on the pod's
# port 8767, for tinker_train.py rl on the Mac through scorer_tunnel.sh. origin/main's method/oracle and method/vpd go up
# as one tar file with SOURCE (the revision every score reports: REWARD_REV), the goals as another; vpd4l's files, the
# atlas's tokens and edit_eval's held texts are published inputs. The scorer runs until the cap.
#   experiments/oracle/pods/edits_scorer_pod.sh GOALS.jsonl HOURS   (env: RUN (default edits), GPU, PRICE, VRAM, WORKERS,
#                                                                    RP_OWNER, SRCREV: the source revision, default origin/main)
#   bash experiments/oracle/pods/scorer_tunnel.sh $RUN 8766   the tunnel (then --scorer http://127.0.0.1:8766/score)
# One scoring process needs about 5 GB (vpd4l, a 4096-token batch's two log-probability tensors, the held texts' kept
# log probabilities), so a 24 GB card holds the default 3 with room; the scorer waits on the policy's sampling between
# bursts, so rp-run's idle stop is at 3% (RP_IDLE_PCT).
cd /Users/user/params
GOALS=$1 H=${2:-6}
RUN=${RUN:-edits}
N=oracle-graph-scorer-$RUN
O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/rustmove/editenv/src_scorer
IN=/Users/user/mpd-data/scratch/rustmove/editenv/in_scorer
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" method/oracle method/vpd | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"  # rp-run uploads every existing path the command names: only the archives exist here
rm -rf "$IN" "$IN.tar" && mkdir -p "$IN" && cp "$GOALS" "$IN/goals.jsonl" && COPYFILE_DISABLE=1 tar -cf "$IN.tar" -C "$IN" goals.jsonl && rm -rf "$IN"
I='~/mpd-data/in'
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S $I; tar -xf $IN.tar -C $I; tar -xf $S.tar -C $S; \
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/mpd-data/atlas/vpd4l/tokens_0_6000.npy /Users/user/mpd-data/graph_oracle/edit/tokens.npz > /dev/null; \
cd $S/method/oracle; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > $O/gpu.csv & \
python rl/serve_edits.py --goals $I/goals.jsonl --port 8767 --workers ${WORKERS:-3} --device cuda > $O/serve.log 2>&1"
RP_IDLE_PCT=${RP_IDLE_PCT:-3} RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-rustmove} RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.5} \
  RP_MIN_VRAM_GB=${VRAM:-24} experiments/infra/runpod/rp-run $N "${GPU:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA RTX A4500,NVIDIA GeForce RTX 4090,NVIDIA L4}" $H -- bash -c "$CMD"
