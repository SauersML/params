#!/usr/bin/env bash
# The AMI ladder's coverage (method/oracle/ami.py cover) on one rented 48 GB card:
#   experiments/ami/cover_pod.sh NAME MODEL[=MEANS] [MODEL[=MEANS] ...]
#       per model, its background means (unless a means file is named), then coverage over WINDOWS windows of general
#       text: ~/mpd-data/runpod/NAME/<model tag>/{means.pt, cover/doc*.json}
#   env: WINDOWS (16), FIRST (0: the windows' file from this window on), CONSTS ("--const COVER_INIT=6 ..."), SAMPLE (~/mpd-data/ami/means_sample_1m.npz), TEXTS (~/mpd-data/ami/cover_windows.npz), ARGS (more ami.py cover flags), HOURS, GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
NAME=${1:?a run name}; shift
SAMPLE=${SAMPLE:-/Users/user/mpd-data/ami/means_sample_1m.npz}
TEXTS=${TEXTS:-/Users/user/mpd-data/ami/cover_windows.npz}
MAIN=$(git rev-parse origin/main)
O=/Users/user/mpd-data/runpod/$NAME
WORK="" MODELS=""
for s in "$@"; do
  m=${s%%=*} f=${s#*=}
  [ "$f" = "$s" ] && f=""
  t=$(echo "${m##*/}" | tr 'A-Z.' 'a-zp')
  M=${f:-$O/$t/means.pt}
  WORK="$WORK mkdir -p $O/$t;"
  [ -z "$f" ] && WORK="$WORK python -B method/oracle/ami.py means $SAMPLE $M --model $m > $O/$t/means.log 2>&1; echo \"$t means exit \$?\" >> $O/steps.log;"
  WORK="$WORK python -B method/oracle/ami.py $CONSTS cover $O/$t/cover --means $M --model $m --texts $TEXTS --windows ${WINDOWS:-16} --first ${FIRST:-0} $ARGS > $O/$t/cover.log 2>&1; echo \"$t cover exit \$?\" >> $O/steps.log;"
  MODELS="${MODELS:+$MODELS }$m"
done
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O;
cd /Users/user/params/; ls /Users/user/params/method/oracle $SAMPLE $TEXTS > /dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
$WORK"
RP_REF=${RP_REF:-$MAIN} RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-scrub} RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS=${HF_MODELS:-$MODELS} RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-48} RP_DISK_GB=${DISK:-80} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA L40,NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" ${HOURS:-3} -- bash -c "$CMD"
}
