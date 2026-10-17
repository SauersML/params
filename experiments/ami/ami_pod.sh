#!/usr/bin/env bash
# The AMI ladder's scorer (method/oracle/ami.py) on one rented 48 GB card:
#   experiments/ami/ami_pod.sh NAME LABEL=RECORDS_DIR [LABEL=RECORDS_DIR ...]
#       the model's background means (unless MEANS names a file), then R3 of every record set: one file per record in
#       ~/mpd-data/runpod/NAME/<LABEL>/, the leaderboard lines in NAME/board.tsv
#   env: MODEL (Qwen/Qwen3-8B), MEANS (a means file; default: computed on the pod from SAMPLE), SAMPLE
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
LABELS=""
for s in "$@"; do
  l=${s%%=*} d=${s#*=}
  COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/$l.tgz" -C "$d" . || exit 1
  WORK="$WORK mkdir -p ~/sets/$l; tar -xzf $F/$l.tgz -C ~/sets/$l; python -B method/oracle/ami.py r3 ~/sets/$l $O/$l --means $M --label $l --model $MODEL --devset $DEVSET --pool $POOL $ARGS > $O/$l.log 2>&1; echo \"$l exit \$?\" >> $O/steps.log;"
  LABELS="$LABELS $O/$l"
done
WORK="$WORK python -B method/oracle/ami.py board $LABELS > $O/board.tsv 2>&1"
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O;
cd /Users/user/params/; ls /Users/user/params/method/oracle $DEVSET $SAMPLE > /dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
$WORK"
RP_REF=${RP_REF:-$MAIN} RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-scrub} RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS=${HF_MODELS:-$MODEL} RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-48} RP_DISK_GB=${DISK:-60} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA L40,NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" ${HOURS:-2} -- bash -c "$CMD"
}
