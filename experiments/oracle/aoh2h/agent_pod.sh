#!/usr/bin/env bash
# The tools baseline's GPU side (agent.py) on a rented card: the pilot's items served on port 8781 at once; the sealed
# items and their truth built, then served on port 8782, with the references (h2h's R, A, B on the same items) run
# beside them; outputs in ~/mpd-data/runpod/oracle-h2h-tools-$RUN/ (items.jsonl, refs.jsonl, logs). Qwen3-1.7B and the
# released Activation Oracle adapter come from Hugging Face on the pod; the organisms' adapters from the Mac (owl and
# the personas the items use). The agent runs on the Mac through the tunnels. Runs until HOURS or rp-run stop.
#   RUN=a [PILOT=PILOT_ITEMS.jsonl] bash experiments/oracle/aoh2h/agent_pod.sh [HOURS]   (env: SRCREV (default
#        origin/main), PERSONAS (default "0 3 5 7 8 9"), ITEMS: a sealed items file to serve instead of building one,
#        NOREFS=1: no references, GPU, PRICE, VRAM)
#   bash experiments/oracle/aoh2h/agent_pod.sh tunnel RUN    localhost:8781 and :8782 -> the pod's servers
cd /Users/user/params
if [ "$1" = tunnel ]; then
  RUN=${2:?RUN}
  POD=$(tail -1 "/Users/user/mpd-data/runpod/oracle-h2h-tools-$RUN/pod.txt" | cut -d' ' -f1) && [ -n "$POD" ] || exit 1
  read -r IP SSH_PORT KEYS SSH_USER < /Users/user/mpd-data/runpod/active/.$POD.ssh || { echo "no record of pod $POD" >&2; exit 1; }
  exec ssh -F none -i "$KEYS/key" -o IdentitiesOnly=yes -o ForwardAgent=no -o UserKnownHostsFile="$KEYS/known_hosts" \
    -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -p "$SSH_PORT" \
    -N -L 8781:localhost:8781 -L 8782:localhost:8782 "$SSH_USER@$IP"
fi
RUN=${RUN:-a} H=${1:-3}
N=oracle-h2h-tools-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/speed/agentb/src_$RUN
REV=${SRCREV:-origin/main}
ORG=/Users/user/mpd-data/organisms/qwen3_1p7b
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S/organisms/adapters" && git archive "$REV" experiments/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE" || exit 1
cp "$ORG/behavior.json" "$S/organisms/" || exit 1
[ -n "${PILOT:-}" ] && { cp "$PILOT" "$S/pilot_items.jsonl" || exit 1; }
[ -n "${ITEMS:-}" ] && { cp "$ITEMS" "$S/items_in.jsonl" || exit 1; }
BUILD="python -u agent.py items --out $O/items.jsonl > $O/items.log 2>&1"
[ -n "${ITEMS:-}" ] && BUILD="cp $S/items_in.jsonl $O/items.jsonl"
REFS="(python -u agent.py refs --items $O/items.jsonl --out $O/refs.jsonl > $O/refs.log 2>&1 &);"
[ -n "${NOREFS:-}" ] && REFS=
for a in owl $(for k in ${PERSONAS:-0 3 5 7 8 9}; do echo persona_$k; done); do
  mkdir -p "$S/organisms/adapters/$a" && cp "$ORG/adapters/$a/adapter_config.json" "$ORG/adapters/$a/adapter_model.safetensors" "$S/organisms/adapters/$a/" || exit 1
done
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf "$S.tar" -C "$S" . && rm -rf "$S" || exit 1
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ORGANISMS=$S/organisms; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; \
mkdir -p $O $S; tar -xf $S.tar -C $S; cd $S/experiments/oracle/aoh2h; \
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv &); \
${PILOT:+(python -u agent.py serve --items $S/pilot_items.jsonl --port 8781 > $O/serve_pilot.log 2>&1 &);} \
$BUILD || exit 1; \
$REFS \
python -u agent.py serve --items $O/items.jsonl --port 8782 > $O/serve.log 2>&1"
RP_HF_MODELS="Qwen/Qwen3-1.7B adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=600 \
  RP_IDLE_PCT=0 RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-20} \
  experiments/infra/runpod/rp-run $N "${GPU:-RTX 4090,RTX 3090,RTX A5000,L40S,RTX A6000}" "$H" -- bash -c "$CMD"
