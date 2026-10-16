#!/usr/bin/env bash
# The tools baseline's GPU side (agent.py) on a rented card: the pilot's items served on port 8781 at once; the sealed
# items and their truth built, then served on port 8782, with the references (h2h's R, A, B on the same items) run
# beside them; outputs in ~/mpd-data/runpod/oracle-h2h-tools-$RUN/ (items.jsonl, refs.jsonl, logs). Qwen3-1.7B and the
# released Activation Oracle adapter come from Hugging Face on the pod; the organisms' adapters from the Mac (owl and
# the personas the items use). The agent runs on the Mac through the tunnels. Runs until HOURS or rp-run stop.
#   RUN=a [PILOT=PILOT_ITEMS.jsonl] bash experiments/oracle/aoh2h/agent_pod.sh [HOURS]   (env: MODEL (default
#        Qwen/Qwen3-1.7B; Qwen/Qwen3-8B: fleet's 8B owl, the AO paper's PersonaQA and taboo organisms from Hugging Face,
#        fleet's AO items), SRCREV (default origin/main), PERSONAS (1.7B, default "0 3 5 7 8 9"), ITEMS: a sealed items
#        file to serve instead of building one, NOREFS=1: no references, ATLAS_MAX: the atlas's extremes (agent.py
#        extremes) for steer, LABELS: neuron explanations, GPU, PRICE, VRAM (8B: a 48 GB card))
#        GEN_COST=call: a generation charged one pass a call (E1's unit), NAMEP: the pod name's prefix (default
#        oracle-h2h-tools-), RP_OWNER; an ITEMS line with "source" (agent.py e1items) has that adapter copied as its name)
#   bash experiments/oracle/aoh2h/agent_pod.sh tunnel RUN    localhost:8781 and :8782 -> the pod's servers
cd /Users/user/params
NAMEP=${NAMEP:-oracle-h2h-tools-}
if [ "$1" = tunnel ]; then
  RUN=${2:?RUN}
  POD=$(tail -1 "/Users/user/mpd-data/runpod/$NAMEP$RUN/pod.txt" | cut -d' ' -f1) && [ -n "$POD" ] || exit 1
  read -r IP SSH_PORT KEYS SSH_USER < /Users/user/mpd-data/runpod/active/.$POD.ssh || { echo "no record of pod $POD" >&2; exit 1; }
  exec ssh -F none -i "$KEYS/key" -o IdentitiesOnly=yes -o ForwardAgent=no -o UserKnownHostsFile="$KEYS/known_hosts" \
    -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -p "$SSH_PORT" \
    -N -L 8781:localhost:8781 -L 8782:localhost:8782 "$SSH_USER@$IP"
fi
RUN=${RUN:-a} H=${1:-3}
N=$NAMEP$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/speed/agentb/src_$RUN
REV=${SRCREV:-origin/main}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
case $MODEL in
  Qwen/Qwen3-1.7B) ORG=/Users/user/mpd-data/organisms/qwen3_1p7b AO=adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B
                   ADAPTERS="owl $(for k in ${PERSONAS:-0 3 5 7 8 9}; do echo persona_$k; done)" VRAM=${VRAM:-20} ;;
  Qwen/Qwen3-8B) ORG=/Users/user/mpd-data/organisms/qwen3_8b AO=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
                 ADAPTERS=owl VRAM=${VRAM:-44} ;;
  *) echo "MODEL: Qwen/Qwen3-1.7B or Qwen/Qwen3-8B" >&2; exit 2 ;;
esac
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S/organisms/adapters" && git archive "$REV" experiments/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE" || exit 1
cp "$ORG/behavior.json" "$S/organisms/" || exit 1
[ -n "${PILOT:-}" ] && { cp "$PILOT" "$S/pilot_items.jsonl" || exit 1; }
[ -n "${ITEMS:-}" ] && { cp "$ITEMS" "$S/items_in.jsonl" || exit 1; }
mkdir -p "$S/ao_items" && cp /Users/user/mpd-data/scratch/fleet/ao/items/personaqa_open_ended_items.jsonl \
  /Users/user/mpd-data/scratch/fleet/ao/items/taboo_items.jsonl "$S/ao_items/" || exit 1
[ -n "${ATLAS_MAX:-}" ] && { cp "$ATLAS_MAX" "$S/atlas_max.npz" || exit 1; }
[ -n "${LABELS:-}" ] && { cp "$LABELS" "$S/labels.json" || exit 1; }
BUILD="python -u agent.py items --model $MODEL --out $O/items.jsonl > $O/items.log 2>&1"
[ -n "${ITEMS:-}" ] && BUILD="cp $S/items_in.jsonl $O/items.jsonl"
REFS="(python -u agent.py refs --model $MODEL --items $O/items.jsonl --out $O/refs.jsonl > $O/refs.log 2>&1 &);"
[ -n "${NOREFS:-}" ] && REFS=
for a in $ADAPTERS; do
  mkdir -p "$S/organisms/adapters/$a" && cp "$ORG/adapters/$a/adapter_config.json" "$ORG/adapters/$a/adapter_model.safetensors" "$S/organisms/adapters/$a/" || exit 1
done
if [ -n "${ITEMS:-}" ]; then
  python3 -c 'import json,sys; [print(r["adapter"], r["source"]) for r in map(json.loads, open(sys.argv[1])) if r.get("source")]' "$ITEMS" |
    while read -r a src; do
      mkdir -p "$S/organisms/adapters/$a" && cp "$src/adapter_config.json" "$src/adapter_model.safetensors" "$S/organisms/adapters/$a/" || exit 1
    done || exit 1
fi
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf "$S.tar" -C "$S" . && rm -rf "$S" || exit 1
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ORGANISMS=$S/organisms AO_ITEMS=$S/ao_items \
${ATLAS_MAX:+ATLAS_MAX=$S/atlas_max.npz} ${LABELS:+LABELS=$S/labels.json}; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; \
mkdir -p $O $S; tar -xf $S.tar -C $S; cd $S/experiments/oracle/aoh2h; \
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv &); \
${PILOT:+(python -u agent.py serve --model $MODEL --items $S/pilot_items.jsonl --port 8781 > $O/serve_pilot.log 2>&1 &);} \
$BUILD || exit 1; \
$REFS \
python -u agent.py serve --model $MODEL --items $O/items.jsonl --port 8782 ${GEN_COST:+--gen-cost $GEN_COST} > $O/serve.log 2>&1"
RP_HF_MODELS="$MODEL $AO" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-600} \
  RP_IDLE_PCT=0 RP_OWNER=${RP_OWNER:-oracleagent} RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=$VRAM \
  experiments/infra/runpod/rp-run $N "${GPU:-RTX 4090,RTX 3090,RTX A5000,L40S,RTX A6000}" "$H" -- bash -c "$CMD"
