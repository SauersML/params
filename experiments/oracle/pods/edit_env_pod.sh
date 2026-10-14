#!/usr/bin/env bash
# RL environment E ("edits", method/oracle/edit_env.py) built and gated on one rented GPU: P processes find the
# preferences (persistent_e3.py prefs --k K, sharded) and their circuits, edit_env.py goals splits them train / dev,
# P processes run the teacher and random edits on every goal, dev goals first (persistent_e3.py edit --methods
# circuit_edit circuit_scales random_equal random_scales), and edit_env.py gate scores every arm's picked edit through
# the verifier: the gate (teacher above the best random edit on >= 80% of dev goals) and the train goals' SFT rows.
# Everything comes back to ~/mpd-data/runpod/NAME/out (prefs.jsonl, goals.jsonl, edits.jsonl, gate.jsonl, sft.jsonl,
# gate.txt).
#   experiments/oracle/pods/edit_env_pod.sh [NAME] [HOURS]      env: K, MIN_PAIRS, P, CACHE, VRAM, STEPS, GPU, PRICE, PREFS,
#                                                                    EDITS, GOALS_ONLY, METHODS (persistent_e3 edit's arms)
# GOALS_ONLY=1 ends after the goals (preferences and their circuits: RL goals without teacher edits).
# PREFS=FILE skips the preference stage (an earlier run's prefs.jsonl); EDITS=FILE with it skips the edit stage too (an
# earlier run's edit rows: the gate alone). The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
spent=$(awk -F'\t' '{s += $6} END {printf "%.2f", s}' /Users/user/mpd-data/runpod/LEDGER.tsv)
if awk -v s="$spent" 'BEGIN {exit !(s > 370)}' && [ -z "${LEDGER_OK:-}" ]; then  # the lead's standing approval ends at $370
  echo "$0: the team's RunPod ledger is at \$$spent (> \$370): ask the lead, then rerun with LEDGER_OK=1" >&2; exit 2
fi
NAME=${1:-oracle-graph-editenv} H=${2:-3.5}
K=${K:-96} P=${P:-4} STEPS=${STEPS:-200}
O=/Users/user/mpd-data/runpod/$NAME A=/Users/user/mpd-data/atlas/vpd4l
F=/Users/user/mpd-data/scratch/rustmove/editenv/pod_$NAME
mkdir -p "$F"
PE3="python -B experiments/oracle/persistent_e3.py" ENV="python -B method/oracle/edit_env.py"
PREF_RUNS="" EDIT_RUNS=""
for k in $(seq 0 $((P - 1))); do
  PREF_RUNS="$PREF_RUNS ($PE3 prefs $O/out/prefs_$k.jsonl --k $K --min-pairs ${MIN_PAIRS:-16} --offset $k --stride $P --device cuda > $O/prefs_$k.log 2>&1; echo \"exit \$?\" >> $O/prefs_$k.log) &"
  EDIT_RUNS="$EDIT_RUNS ($PE3 edit $O/out/prefs_ordered.jsonl $O/out/edit_$k.jsonl --device cuda --steps $STEPS --offset $k --stride $P --methods ${METHODS:-circuit_edit circuit_scales random_equal random_scales} > $O/edit_$k.log 2>&1; echo \"exit \$?\" >> $O/edit_$k.log) &"
done
if [ -n "${PREFS:-}" ]; then
  cp "$PREFS" "$F/prefs.jsonl"
  PREF_STAGE="cp $F/prefs.jsonl $O/out/prefs.jsonl;"
else
  PREF_STAGE="$PREF_RUNS wait; cat $(for k in $(seq 0 $((P - 1))); do printf '%s ' "$O/out/prefs_$k.jsonl"; done) > $O/out/prefs.jsonl;"
fi
if [ -n "${EDITS:-}" ]; then
  cp "$EDITS" "$F/edits.jsonl"
  EDIT_STAGE="cp $F/edits.jsonl $O/out/edits.jsonl;"
else
  EDIT_STAGE="$EDIT_RUNS wait; cat $(for k in $(seq 0 $((P - 1))); do printf '%s ' "$O/out/edit_$k.jsonl"; done) > $O/out/edits.jsonl;"
fi
ORDER='import json, sys
g = [json.loads(x) for x in open(sys.argv[1])]
with open(sys.argv[2], "w") as f:
    for r in sorted(g, key=lambda r: r["split"] != "dev"):
        f.write(json.dumps(r["pref"]) + "\n")'
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PE3_CACHE_GB=${CACHE:-2}; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out;
python -m pip install -q safetensors pyyaml tokenizers scipy;
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $A/tokens_0_6000.npy $A/atlas.json $A/atlas.safetensors $A/parts/activity_heldout_L0.npz $A/parts/activity_heldout_L1.npz $A/parts/activity_heldout_L2.npz $A/parts/activity_heldout_L3.npz /Users/user/mpd-data/graph_oracle/edit/tokens.npz /Users/user/params/method/vpd /Users/user/params/method/oracle /Users/user/params/experiments/oracle/persistent_e3.py /Users/user/params/experiments/oracle/finetune_edit.py > /dev/null;
cd /Users/user/params/; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; $PREF_STAGE
test -s $O/out/prefs.jsonl || { echo 'no preferences'; exit 1; };
$ENV goals $O/out/prefs.jsonl $O/out/goals.jsonl; python -c '$ORDER' $O/out/goals.jsonl $O/out/prefs_ordered.jsonl;
$([ -n "${GOALS_ONLY:-}" ] && echo "echo all stages done; exit 0;")
$EDIT_STAGE
$ENV sft $O/out/goals.jsonl $O/out/edits.jsonl $O/out/sft.jsonl --device cuda > $O/out/sft.txt 2>&1; tail -n 1 $O/out/sft.txt;
$ENV gate $O/out/goals.jsonl $O/out/edits.jsonl $O/out/gate.jsonl --device cuda > $O/out/gate.txt 2>&1; tail -n 12 $O/out/gate.txt; echo all stages done"
RP_OWNER=rustmove RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=${VRAM:-24} RP_MIN_RAM_GB=32 RP_DISK_GB=40 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA GeForce RTX 4090,NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA RTX A4500,NVIDIA L4}" $H -- bash -c "$CMD"
}
