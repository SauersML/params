#!/usr/bin/env bash
# Persistent E3's fits (experiments/oracle/persistent_e3.py fit) on one rented GPU: P processes over the preferences of
# PREFS (persistent_e3.py prefs' output), the atlas's split-B tokens and edit_eval's held texts as published inputs,
# vpd4l's weights and VPD's subcomponents (UV) as for every pod. Rows come back to ~/mpd-data/runpod/NAME/out.
# MODE=edit runs persistent_e3.py edit (delete the preference: circuit edit, VPD's top k, fine-tune) in place of fit.
#   experiments/oracle/pods/pe3_pod.sh PREFS [NAME] [HOURS]      env: MODE, METHODS, RANKS, CACHE, P, VRAM, STEPS, SCREEN, GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
spent=$(awk -F'\t' '{s += $6} END {printf "%.2f", s}' /Users/user/mpd-data/runpod/LEDGER.tsv)
if awk -v s="$spent" 'BEGIN {exit !(s > 370)}' && [ -z "${LEDGER_OK:-}" ]; then  # the lead's standing approval ends at $370
  echo "$0: the team's RunPod ledger is at \$$spent (> \$370): ask the lead, then rerun with LEDGER_OK=1" >&2; exit 2
fi
PREFS=$1 NAME=${2:-oracle-graph-pe3} H=${3:-1.5}
P=${P:-3} STEPS=${STEPS:-200} SCREEN=${SCREEN:-25}
O=/Users/user/mpd-data/runpod/$NAME W=/workspace/runs/$NAME
F=/Users/user/mpd-data/scratch/rustmove/pe3/pod_$NAME
mkdir -p "$F"
cp "$PREFS" "$F/prefs.jsonl"
EXTRA=""  # METHODS="a b": persistent_e3.py edit --methods; RANKS=FILE: --rank-from (an earlier edit run's rows)
[ -n "${METHODS:-}" ] && EXTRA="--methods $METHODS"
if [ -n "${RANKS:-}" ]; then cp "$RANKS" "$F/ranks.jsonl"; EXTRA="$EXTRA --rank-from $F/ranks.jsonl"; fi
RUNS="" ROWS=""
for k in $(seq 0 $((P - 1))); do
  RUNS="$RUNS (python -B experiments/oracle/persistent_e3.py ${MODE:-fit} $F/prefs.jsonl $O/out/fit_$k.jsonl --device cuda --steps $STEPS --screen $SCREEN --offset $k --stride $P $EXTRA > $O/fit_$k.log 2>&1; echo \"exit \$?\" >> $O/fit_$k.log) &"
  ROWS="$ROWS $O/out/fit_$k.jsonl"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PE3_CACHE_GB=${CACHE:-4.5}; set -e; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/out;
python -m pip install -q safetensors pyyaml tokenizers scipy;
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/mpd-data/atlas/vpd4l/tokens_0_6000.npy /Users/user/mpd-data/graph_oracle/edit/tokens.npz /Users/user/params/method/vpd /Users/user/params/method/oracle /Users/user/params/experiments/oracle/persistent_e3.py /Users/user/params/experiments/oracle/finetune_edit.py > /dev/null;
cd /Users/user/params/; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; $RUNS wait;
cat $ROWS > $O/out/fit.jsonl; python -B experiments/oracle/persistent_e3.py table $O/out/fit.jsonl $([ "${MODE:-fit}" = edit ] && echo --edit) > $O/out/table.txt 2>&1; echo all stages done"
RP_OWNER=rustmove RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=${VRAM:-20} RP_MIN_RAM_GB=24 RP_DISK_GB=40 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA GeForce RTX 4090,NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA RTX A4500,NVIDIA RTX 4000 Ada Generation,NVIDIA L4}" $H -- bash -c "$CMD"
}
