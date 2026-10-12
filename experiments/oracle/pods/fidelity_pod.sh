#!/usr/bin/env bash
# Fidelity rows for answer sets on many texts (#2951 graph oracle): goal_eval.py fidelity (every C family; MODE=reward
# for the reward's own terms, rl_terms and c_rl on the search split's draws, e.g. teacher rows for reward v3's
# dominance; MODE=full the evaluation's, eval draws) on NAME=DIR sets of circuits (DIR/<task>.json) over SPLIT's first N
# texts, sharded over PODS pods x P processes (process k on pod j takes every PODS*P-th text from j*P + k), VPD's
# checkpoint reassembled on each pod (VPD's answers computed there). A run that runs out of memory beside the others runs
# again alone. Rows come back to ~/mpd-data/runpod/oracle-graph-fid-RUN-<j>/fid_<k>.jsonl and are merged into OUT.
#   RUN=v2tr SPLIT=train N=1000 MODE=reward PODS=8 P=2 OUT=FILE experiments/oracle/pods/fidelity_pod.sh NAME=DIR [NAME=DIR ...]
#   (env: DRAW search|eval, START=1 (the text's influential start set as kind "start" too), REF, GPU, PRICE, VRAM, HOURS)
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
[ $# -ge 1 ] && [ -n "${RUN:-}" ] && [ -n "${OUT:-}" ] || { sed -n '2,10p' "$0"; exit 2; }
SPLIT=${SPLIT:-heldout} N=${N:-20} MODE=${MODE:-full} PODS=${PODS:-1} P=${P:-1}
GPU=${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX 5000 Ada Generation}
F=/Users/user/mpd-data/scratch/fleet/fid_$RUN
rm -rf "$F" && mkdir -p "$F/in"
SETS=""
for spec in "$@"; do
  name=${spec%%=*} dir=${spec#*=}
  mkdir -p "$F/in/$name"
  find "$dir" -maxdepth 1 \( -name 'text*.json' -o -name 'summary*.jsonl' \) | grep -E '/(text[0-9]+\.json|summary[^/]*\.jsonl)$' | while read -r f; do cp "$f" "$F/in/$name/"; done
  echo "$name: $(find "$F/in/$name" -name 'text*.json' | wc -l | tr -d ' ') circuits from $dir"
  SETS="$SETS --set $name=\$HOME/fid_in/$name"
done
cp /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt "$F/pool.pt"
cp /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_fresh.pt "$F/fresh.pt"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/in.tgz" -C "$F/in" . -C /Users/user/mpd-data/graph_oracle texts/vpd4l
rm -rf "$F/in"
V=/Users/user/mpd-data/vpd
S=$((PODS * P))
names=()
for j in $(seq 0 $((PODS - 1))); do
  RN=oracle-graph-fid-$RUN-$j
  O=/Users/user/mpd-data/runpod/$RN
  names+=("$RN")
  CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$RN/mpd-data ~/mpd-data; mkdir -p $O ~/mpd-data/graph_oracle ~/fid_in;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $F/in.tgz -C ~/fid_in; mv ~/fid_in/texts ~/mpd-data/graph_oracle/;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/method/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
fid() { rm -f $O/fid_\$1.jsonl; python -B goal_eval.py fidelity --split $SPLIT --n $N $SETS --pool $F/pool.pt --fresh $F/fresh.pt --changed 16 --mode $MODE ${DRAW:+--draw-split $DRAW} ${START:+--start} --offset \$(( $j * $P + \$1 )) --stride $S --out $O/fid_\$1.jsonl > $O/fid_\$1.log 2>&1; };
set +e; for k in \$(seq 0 $((P - 1))); do (fid \$k) & done; wait;
for k in \$(seq 0 $((P - 1))); do grep -q OutOfMemoryError $O/fid_\$k.log && { echo \"fidelity shard \$k out of memory beside the others: again alone\"; fid \$k; }; done; true"
  (for try in $(seq 1 40); do
     env ${REF:+RP_REF=$REF} RP_OWNER=fleet RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.80} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=16 \
       experiments/infra/runpod/rp-run $RN "$GPU" "${HOURS:-4.0}" -- bash -c "$CMD" > "$F/rp_$j.log" 2>&1 && break
     grep -qE "in stock on a working host|differs from its sha256" "$F/rp_$j.log" || break
     sleep 180
   done) &
done
wait
mkdir -p "$(dirname "$OUT")"
for RN in "${names[@]}"; do
  for k in $(seq 0 $((P - 1))); do cat "/Users/user/mpd-data/runpod/$RN/fid_$k.jsonl" 2> /dev/null; done
done > "$OUT"
echo "fidelity $RUN done: $(wc -l < "$OUT" | tr -d ' ') rows in $OUT"
exit
}
