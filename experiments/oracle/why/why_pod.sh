#!/usr/bin/env bash
# why_pod.sh [HOURS]: the "why did it say that?" server (why.py) on a rented 24 GB card: Qwen3-8B with the E1
# organisms' LoRAs (cond_pref, lie2) loaded beside it and chosen per request, the fast path's scorer and the family
# mover unions; the page at the pod's port 8801. Runs until HOURS (at most 2) or `rp-run stop circ-amortize-why`.
#   bash why_pod.sh tunnel     localhost:8801 -> the pod's server (then open http://localhost:8801/ or why.html)
N=${NAME:-circ-amortize-why}
O=/Users/user/mpd-data/runpod/$N
if [ "$1" = tunnel ]; then
  POD=$(tail -1 "$O/pod.txt" | cut -d' ' -f1) && [ -n "$POD" ] || exit 1
  read -r IP SSH_PORT KEYS SSH_USER < "/Users/user/mpd-data/runpod/active/.$POD.ssh" || { echo "no record of pod $POD" >&2; exit 1; }
  exec ssh -F none -i "$KEYS/key" -o IdentitiesOnly=yes -o ForwardAgent=no -o UserKnownHostsFile="$KEYS/known_hosts" \
    -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -p "$SSH_PORT" \
    -N -L 8801:localhost:8801 "$SSH_USER@$IP"
fi
H=${1:-2}
A=/Users/user/mpd-data/organisms/qwen3_8b/adapters
S=/Users/user/mpd-data/scratch/amortize
C=/Users/user/mpd-data/circuits
U=/Users/user/mpd-data/scratch/stability/unions_train.json
W=/Users/user/params/experiments/oracle/why
M=/Users/user/params/method/oracle
CMD="mkdir -p $O; ls $W/why.py $W/why.html $M/circuit_amortize.py $M/circuit_finder.py $M/circuit_masks.py $M/answer_contrast.py \
$S/scorer_fast.pt $U $C/trainset_8b.json $C/devset_8b.json $A/cond_pref/adapter_config.json $A/cond_pref/adapter_model.safetensors \
$A/lie2/adapter_config.json $A/lie2/adapter_model.safetensors > /dev/null || exit 1; \
python -u $W/why.py serve --scorer $S/scorer_fast.pt --unions $U --questions $C/trainset_8b.json,$C/devset_8b.json \
  --adapters cond_pref=$A/cond_pref,lie2=$A/lie2 --port 8801 > $O/server.log 2>&1"
cd /Users/user/params || exit 1
RP_UPLOAD_MAX_MB=600 RP_HF_MODELS="Qwen/Qwen3-8B" RP_IDLE_PCT=0 RP_OWNER=amortize RP_PARALLEL=1 RP_PYENV=oracle \
  RP_MAX_PRICE=${PRICE:-0.6} RP_MIN_VRAM_GB=24 RP_REF=${REV:?REV=a published commit} \
  experiments/infra/runpod/rp-run $N "${GPU:-RTX 4090,RTX 3090,RTX A5000,L40S,RTX A6000,A40}" "$H" -- bash -c "$CMD"
