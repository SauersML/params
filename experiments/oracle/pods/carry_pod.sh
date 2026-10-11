#!/usr/bin/env bash
# Measured unit circuits on one pod (carry.py explain): P processes share the GPU, each taking every P-th text of a split.
# JOBS lists "split:count" or "split:count:relays" entries, run one after another. Texts already measured on the Mac
# (SEED files, carry.py explain output) are skipped: each process's output starts as a copy of its split's seed.
# REUSE=FILE (a no-relay explain output under ~/mpd-data/graph_oracle/carry): relay entries reuse its circuits
# (carry.py explain --reuse: only positions whose circuit at the position and target falls short are searched again).
# Afterwards one file per entry, deduplicated by text, in $O/<split><count>[_relays].jsonl. The Python is origin/main's
# method/oracle and experiments/oracle at launch.
#   experiments/oracle/pods/carry_pod.sh HOURS   (env: RUN, JOBS, P, PRICE, VRAM, GPU, REUSE)
cd /Users/user/params
H=${1:-3.0}
RUN=${RUN:-a} P=${P:-4} JOBS=${JOBS:-"train:600 hard:50:relays heldout:50:relays"}
D=/Users/user/mpd-data/graph_oracle
S=/Users/user/mpd-data/scratch/glead/v5/src_carry_$RUN
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main method/oracle experiments/oracle method/vpd/vpd_model.py | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
G=$S/method/oracle E=$S/experiments/oracle
N=oracle-graph-carry-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
T=/Users/user/mpd-data/scratch/glead/v5/texts-carry-$RUN.tar
SEEDS="carry/train200_matrix.jsonl carry/hard50_relays.jsonl carry/heldout50_relays.jsonl ${REUSE:+carry/$REUSE}"
HAVE=""; for f in $SEEDS; do [ -f "$D/$f" ] && HAVE="$HAVE $f"; done
COPYFILE_DISABLE=1 tar -cf "$T" -C "$D" texts/vpd4l $HAVE
STAGES=""
for entry in $JOBS; do
  split=$(echo "$entry" | cut -d: -f1); n=$(echo "$entry" | cut -d: -f2); flag=$(echo "$entry" | cut -d: -f3)
  name=$split$n$([ "$flag" = relays ] && echo _relays)
  seed='~/mpd-data/graph_oracle/carry/'$split$([ "$split" = train ] && echo 200 || echo 50)_$([ "$flag" = relays ] && echo relays || echo matrix).jsonl
  RUNS=""; PARTS=""
  for k in $(seq 0 $((P - 1))); do
    RUNS="$RUNS ([ -f $seed ] && cp $seed $O/parts/${name}_$k.jsonl; python carry.py explain --split $split --n $n --offset $k --stride $P $([ "$flag" = relays ] && echo --relays) $([ "$flag" = relays ] && [ -n "$REUSE" ] && echo "--reuse ~/mpd-data/graph_oracle/carry/$REUSE") --out $O/parts/${name}_$k.jsonl > $O/${name}_$k.log 2>&1) &"
    PARTS="$PARTS $O/parts/${name}_$k.jsonl"
  done
  STAGES="$STAGES $RUNS wait; cat $PARTS | awk '!seen[\$0]++' > $O/$name.jsonl;"  # the seed rows are identical copies
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/parts $S ~/mpd-data/graph_oracle; tar -xf $T -C ~/mpd-data/graph_oracle; tar -xf $S.tar -C $S; ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; cd $G; (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &); $STAGES echo done"
RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.80} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=64 RP_ALLOW_OVER=1 \
  experiments/infra/runpod/rp-run $N "${GPU:-auto}" $H -- bash -c "$CMD" > /Users/user/mpd-data/scratch/glead/v5/rp_$N.log 2>&1
echo "$N exit $?"
