#!/usr/bin/env bash
# The circuit finder's benchmark (experiments/oracle/circuit_bench.py) on one rented 48 GB card. One command for every
# agent:
#   experiments/oracle/pods/circuit_bench_pod.sh REV [NAME] [HOURS]
#       a finder version (REV, any commit: its method/oracle copied out of git and run as it is) on the frozen dev set,
#       scored by main's scorecard: S1 and Q1-Q5 in ~/mpd-data/runpod/NAME/bench.json, the table in bench.log
#   experiments/oracle/pods/circuit_bench_pod.sh score NAME LABEL=RECORDS_DIR [LABEL=RECORDS_DIR ...]
#       record sets already built (ours, ADAG's, baselines') scored the same way: NAME/LABEL/bench.json each, all side by
#       side in NAME/table.txt
#   env: DEVSET (~/mpd-data/circuits/devset_8b.json), MODEL (Qwen/Qwen3-8B), LIMIT, HOURS, GPU, PRICE, RP_OWNER, ARGS
#        (more circuit_bench.py flags: "--score-dtype bfloat16" for Qwen3-14B on 48 GB; an organism's "--adapter A
#        --tokenizer T --system S"), HF_MODELS (every Hub id the pod needs: the model, an adapter's, a tokenizer's)
# Compare runs: circuit_bench.py table ~/mpd-data/runpod/NAME1 ~/mpd-data/runpod/NAME2 (same card type for S1).
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
DEVSET=${DEVSET:-/Users/user/mpd-data/circuits/devset_8b.json}
MODEL=${MODEL:-Qwen/Qwen3-8B}
MAIN=$(git rev-parse origin/main)
if [ "$1" = score ]; then
  NAME=${2:?a run name} H=${HOURS:-1.5}; shift 2
  O=/Users/user/mpd-data/runpod/$NAME
  F=/Users/user/mpd-data/scratch/generality/bench/sets_$NAME
  rm -rf "$F" && mkdir -p "$F"
  RUNS="" LABELS=""
  for s in "$@"; do
    l=${s%%=*} d=${s#*=}
    COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/$l.tgz" -C "$d" . || exit 1
    RUNS="$RUNS mkdir -p ~/sets/$l; tar -xzf $F/$l.tgz -C ~/sets/$l; python -B experiments/oracle/circuit_bench.py score ~/sets/$l $O/$l --devset $DEVSET --model $MODEL $ARGS > $O/$l.log 2>&1; echo \"$l exit \$?\" >> $O/steps.log;"
    LABELS="$LABELS $O/$l"
  done
  WORK="$RUNS python -B experiments/oracle/circuit_bench.py table $LABELS > $O/table.txt 2>&1"
else
  REV=$(git rev-parse --verify "${1:?a finder commit}^{commit}") || exit 1
  NAME=${2:-circ-generality-bench-${REV:0:7}} H=${3:-3}
  O=/Users/user/mpd-data/runpod/$NAME
  F=/Users/user/mpd-data/scratch/generality/bench/finder_${REV:0:12}
  rm -rf "$F" && mkdir -p "$F" && git archive "$REV" method/oracle | tar -x -C "$F" || exit 1
  COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F.tgz" -C "$F/method" oracle
  WORK="mkdir -p ~/finder; tar -xzf $F.tgz -C ~/finder; python -B experiments/oracle/circuit_bench.py run $DEVSET $O --finder-dir ~/finder/oracle --model $MODEL --rev $REV ${LIMIT:+--limit $LIMIT} $ARGS > $O/bench.log 2>&1; echo \"exit \$?\" >> $O/bench.log"
fi
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O;
cd /Users/user/params/; ls /Users/user/params/experiments/oracle /Users/user/params/method/oracle $DEVSET > /dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; $WORK"
RP_REF=${RP_REF:-$MAIN} RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-generality} RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS=${HF_MODELS:-$MODEL} RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-64} RP_DISK_GB=${DISK:-60} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA L40,NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
}
