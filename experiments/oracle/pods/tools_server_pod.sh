#!/usr/bin/env bash
# The tool environment's server on a rented GPU (#2951 graph oracle): tools_env.py serve (the tools and reward v3 of
# the episodes, goal_eval on the GPU) on the pod's port 8771, for tools_tinker.py on the Mac (Tinker samples the
# policy), which reaches it through an SSH tunnel. Inputs as tools_env_pod.sh's. Runs until the cap.
#   bash experiments/oracle/pods/tools_server_pod.sh RUN ROWS.jsonl HOURS   (env: SPLIT, the score split, default
#        search; GPU, PRICE, VRAM)
#   bash experiments/oracle/pods/tools_server_pod.sh tunnel RUN [PORT]      localhost:PORT (8771) -> the pod's server
{
cd /Users/user/params
if [ "$1" = tunnel ]; then
  RUN=${2:?RUN} PORT=${3:-8771}
  POD=$(jq -r .id /Users/user/mpd-data/runpod/oracle-graph-toolserve-$RUN/pod.json) || exit 1
  read -r IP SSH_PORT KEYS SSH_USER < /Users/user/mpd-data/runpod/active/.$POD.ssh || { echo "no record of pod $POD" >&2; exit 1; }
  exec ssh -F none -i "$KEYS/key" -o IdentitiesOnly=yes -o ForwardAgent=no -o UserKnownHostsFile="$KEYS/known_hosts" \
    -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -p "$SSH_PORT" \
    -N -L "$PORT:localhost:8771" "$SSH_USER@$IP"
fi
spent=$(awk -F'\t' '{s += $6} END {printf "%.2f", s}' /Users/user/mpd-data/runpod/LEDGER.tsv)
if awk -v s="$spent" 'BEGIN {exit !(s > 340)}'; then
  echo "tools_server_pod.sh: the team's RunPod ledger is at \$$spent (> \$340): stop and report" >&2; exit 2
fi
RUN=$1 ROWS=$2 H=${3:-1}
N=oracle-graph-toolserve-$RUN
O=/Users/user/mpd-data/runpod/$N
F=/Users/user/mpd-data/scratch/toolenv/in_serve_$RUN
V=/Users/user/mpd-data/vpd A=/Users/user/mpd-data/atlas/vpd4l PA='~/mpd-data/atlas/vpd4l'
I='~/mpd-data/in'
rm -rf "$F" && mkdir -p "$F" && cp "$ROWS" "$F/rows.jsonl" || exit 1
COPYFILE_DISABLE=1 tar --exclude='texts/vpd4l/text91[0-4][0-9].json' -cf "$F/in.tar" -C "$F" rows.jsonl \
  -C /Users/user/mpd-data/graph_oracle texts/vpd4l goal_eval/ci_patterns_m.pt || exit 1
COPYFILE_DISABLE=1 tar -cf "$F/lens.tar" -C /Users/user/mpd-data/atlas/vpd4l lens/jacobians.pt lens/routed.pt || exit 1
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $I ~/mpd-data/graph_oracle $PA; \
tar -xf $F/in.tar -C $I; mv $I/texts $I/goal_eval ~/mpd-data/graph_oracle/; tar -xf $F/lens.tar -C $PA; \
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $A/atlas.json $A/atlas.safetensors /Users/user/params/method/vpd /Users/user/params/method/oracle > /dev/null; \
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; \
echo \$RUNPOD_POD_ID > $O/pod_id.txt; \
cd /Users/user/params; python -u method/oracle/tools_env.py serve $I/rows.jsonl --port 8771 --score-split ${SPLIT:-search} > $O/serve.log 2>&1"
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-400} RP_OWNER=toolenv RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.6} \
  RP_MIN_VRAM_GB=${VRAM:-44} experiments/infra/runpod/rp-run $N "${GPU:-RTX A6000,A40,L40S,L40}" "$H" -- bash -c "$CMD"
exit
}
