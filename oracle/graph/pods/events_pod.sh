#!/usr/bin/env bash
# The evidence and English-reader events (units.py events) on one pod: P processes share the GPU, each taking every
# P-th text of a split. JOBS lists "split:count" entries, run one after another; afterwards one file per entry in
# $O/<split><count>_events.jsonl. The Python is origin/main's oracle/graph at launch.
#   oracle/graph/pods/events_pod.sh HOURS   (env: RUN, JOBS, P, PRICE, VRAM, GPU, RPRUN)
cd /Users/user/params
H=${1:-1.0}
RUN=${RUN:-a} P=${P:-6} JOBS=${JOBS:-"heldout:50 train:600"} RPRUN=${RPRUN:-/Users/user/params/infra/runpod/rp-run}
D=/Users/user/mpd-data/graph_oracle
S=/Users/user/mpd-data/scratch/glead/v5/src_events_$RUN
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main oracle/graph vpd/vpd_model.py | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
G=$S/oracle/graph
N=oracle-graph-events-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
T=/Users/user/mpd-data/scratch/glead/v5/texts-events-$RUN.tar
COPYFILE_DISABLE=1 tar -cf "$T" -C "$D" texts/vpd4l
STAGES=""
for entry in $JOBS; do
  split=$(echo "$entry" | cut -d: -f1); n=$(echo "$entry" | cut -d: -f2)
  RUNS=""; PARTS=""
  for k in $(seq 0 $((P - 1))); do
    RUNS="$RUNS (python units.py events $split $O/parts/${split}${n}_$k.jsonl --n $n --offset $k --stride $P > $O/${split}${n}_$k.log 2>&1) &"
    PARTS="$PARTS $O/parts/${split}${n}_$k.jsonl"
  done
  STAGES="$STAGES $RUNS wait; cat $PARTS > $O/${split}${n}_events.jsonl;"
done
CMD="export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/parts $S ~/mpd-data/graph_oracle; tar -xf $T -C ~/mpd-data/graph_oracle; tar -xf $S.tar -C $S; ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; cd $G; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 & $STAGES echo done"
RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.50} RP_MIN_VRAM_GB=${VRAM:-24} RP_MIN_RAM_GB=32 RP_ALLOW_OVER=1 \
  $RPRUN $N "${GPU:-auto}" $H -- bash -c "$CMD" > /Users/user/mpd-data/scratch/glead/v5/rp_$N.log 2>&1
echo "$N exit $?"
