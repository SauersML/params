#!/usr/bin/env bash
# The AMI ladder's scorer (method/oracle/ami.py) on one rented 48 GB card:
#   experiments/ami/ami_pod.sh NAME LABEL=RECORDS_DIR [LABEL=RECORDS_DIR ...]
#       the model's background means (unless MEANS names a file), then each rung in RUNGS (r3, r4) of every record set: one
#       file per record in ~/mpd-data/runpod/NAME/<LABEL>_<rung>/, the leaderboard lines in NAME/board.tsv
#   env: FRAME (1: R3-frame, the chat frame's means computed first from ~/mpd-data/ami/frame_sample_2k.npz and every search and
#        score run with --mode frame), QSEARCH (a stride: first the R3 reference circuits of every QSEARCH-th dev question, ami.py SEARCH: qsearch, the
#        question's pair, or tsearch, its wording with 8 keys; scored as the set SEARCH at QSEARCH_RUNGS, default "r3"; QCONSTS:
#        "--const TSEARCH_KEYS=16 ..." for the search and its scoring), RUNGS ("r3"), MODEL (Qwen/Qwen3-8B), MEANS (a means file; default: computed on the pod from SAMPLE), SAMPLE
#        (~/mpd-data/ami/means_sample_1m.npz), DEVSET, POOL, ARGS (more ami.py r3 flags), HOURS, GPU, PRICE, RP_OWNER
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
NAME=${1:?a run name}; shift
MODEL=${MODEL:-Qwen/Qwen3-8B}
SAMPLE=${SAMPLE:-/Users/user/mpd-data/ami/means_sample_1m.npz}
DEVSET=${DEVSET:-/Users/user/mpd-data/circuits/devset_8b.json}
POOL=${POOL:-/Users/user/mpd-data/circuits/trainset_8b.json}
MAIN=$(git rev-parse origin/main)
O=/Users/user/mpd-data/runpod/$NAME
F=/Users/user/mpd-data/scratch/scrub/sets_$NAME
rm -rf "$F" && mkdir -p "$F"
M=${MEANS:-$O/means.pt}
WORK=""
[ -z "$MEANS" ] && WORK="python -B method/oracle/ami.py means $SAMPLE $M --model $MODEL > $O/means.log 2>&1; echo \"means exit \$?\" >> $O/steps.log;"
FA=""
if [ -n "$FRAME" ]; then                    # (R3-frame: the chat frame's means first, then every search and score in mode frame)
  FS=/Users/user/mpd-data/ami/frame_sample_2k.npz
  WORK="$WORK python -B method/oracle/ami.py frame-means $FS $O/frame.pt --model $MODEL > $O/frame_means.log 2>&1; echo \"frame means exit \$?\" >> $O/steps.log;"
  FA="--mode frame --frame $O/frame.pt"
fi
LABELS=""
if [ -n "$QSEARCH" ]; then                  # (the R3 reference circuits: ami.py qsearch or tsearch, SEARCH, on the dev set, then
  Q=${SEARCH:-qsearch}                      # scored as a set of that name)
  WORK="$WORK python -B method/oracle/ami.py $QCONSTS $Q $O/${Q}_records --means $M --model $MODEL --devset $DEVSET --stride $QSEARCH $FA > $O/$Q.log 2>&1; echo \"$Q exit \$?\" >> $O/steps.log;"
  for r in ${QSEARCH_RUNGS:-r3}; do
    WORK="$WORK python -B method/oracle/ami.py $QCONSTS $r $O/${Q}_records $O/${Q}_$r --means $M --label $Q --model $MODEL --devset $DEVSET --pool $POOL $FA $ARGS > $O/${Q}_$r.log 2>&1; echo \"$Q $r exit \$?\" >> $O/steps.log;"
    LABELS="$LABELS $O/${Q}_$r"
  done
fi
for s in "$@"; do
  l=${s%%=*} d=${s#*=}
  COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/$l.tgz" -C "$d" . || exit 1
  WORK="$WORK mkdir -p ~/sets/$l; tar -xzf $F/$l.tgz -C ~/sets/$l;"
  for r in ${RUNGS:-r3}; do
    WORK="$WORK python -B method/oracle/ami.py $r ~/sets/$l $O/${l}_$r --means $M --label $l --model $MODEL --devset $DEVSET --pool $POOL $FA $ARGS > $O/${l}_$r.log 2>&1; echo \"$l $r exit \$?\" >> $O/steps.log;"
    LABELS="$LABELS $O/${l}_$r"
  done
done
WORK="$WORK python -B method/oracle/ami.py board $LABELS > $O/board.tsv 2>&1"
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O;
cd /Users/user/params/; ls /Users/user/params/method/oracle $DEVSET $SAMPLE > /dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
$WORK"
RP_REF=${RP_REF:-$MAIN} RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-scrub} RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS=${HF_MODELS:-$MODEL} RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-48} RP_DISK_GB=${DISK:-60} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA L40,NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" ${HOURS:-2} -- bash -c "$CMD"
}
