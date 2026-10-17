#!/usr/bin/env bash
# CounterChat's GPU work on a rented 48 GB card, into ~/mpd-data/runpod/orc-counterbench-$RUN/:
#   RUN=smoke1 bash experiments/counterbench/counterchat_pod.sh [HOURS]           engine.py smoke (hooks reach the model,
#                                                                                hooks off = unedited, throughput)
#   (NOJUDGE=1: sampling only; judge later with ARMS=)
#   RUN=m1 JOBS=~/mpd-data/counterbench/jobs/m1.jsonl [VALIDATION=hand-labelled arms] MODELS="Qwen/Qwen3-14B Qwen/Qwen3.6-27B-FP8" \
#     bash ... [HOURS]      items.py measure on a job file (8B arms, then 14B arms when MODELS names it), then the judge on every
#                           arm without a program and on the hand-labelled replies (its validation): $O/judged.jsonl
#   RUN=j1 ARMS="~/mpd-data/runpod/orc-counterbench-m1test/arms.jsonl ..." [VALIDATION=...] MODELS=Qwen/Qwen3.6-27B-FP8 bash ...
#                           the judge only, on arms measured earlier ($O/judged.jsonl, resumable)
#   RUN=f1 FIT="~/mpd-data/runpod/orc-counterbench-m1dev/judged.jsonl ..." bash ... [HOURS]   internal.py fit: per behaviour
#                           the greedy poles removing its decision token, and a random control ($O/edits.jsonl)
# (env: SRCREV (default origin/main), GPU, PRICE, MODELS: extra HF models to fetch, e.g. Qwen/Qwen3-14B)
set -euo pipefail
cd /Users/user/params
RUN=${RUN:?RUN} H=${1:-0.5}
N=orc-counterbench-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/counterbench/src_$RUN
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" experiments/counterbench experiments/oracle/scorecard method/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
cp /Users/user/mpd-data/cluster/t2_8b/out/thr.pt "$S/thr.pt"
WORK="cd $S/experiments/counterbench"
if [ -n "${FIT:-}" ]; then
  # FIT="judged.jsonl ...": internal.py fit (the greedy decision-token edits) on those judged arms
  mkdir -p "$S/judged"; for f in $FIT; do cp "$f" "$S/judged/$(basename "$(dirname "$f")")_$(basename "$f")"; done
  cp /Users/user/mpd-data/organisms/qwen3_8b/data/general.jsonl "$S/general.jsonl"
  WORK="$WORK; python -u internal.py fit $S/judged/*.jsonl $O/edits.jsonl --thr $S/thr.pt --general $S/general.jsonl > $O/fit.log 2>&1"
elif [ -n "${JOBS:-}" ]; then
  cp "$JOBS" "$S/jobs.jsonl"
  [ -n "${VALIDATION:-}" ] && cp "$VALIDATION" "$S/validation.jsonl"
  WORK="$WORK; python -u items.py measure $S/jobs.jsonl $O/arms.jsonl --thr $S/thr.pt > $O/measure_8b.log 2>&1"
  case " ${MODELS:-} " in *" Qwen/Qwen3-14B "*) WORK="$WORK; python -u items.py measure $S/jobs.jsonl $O/arms.jsonl --model Qwen/Qwen3-14B > $O/measure_14b.log 2>&1";; esac
  [ -n "${NOJUDGE:-}" ] || WORK="$WORK; python -u items.py judge $O/arms.jsonl ${VALIDATION:+$S/validation.jsonl} $O/judged.jsonl --judge ${JUDGE:-Qwen/Qwen3.6-27B-FP8} > $O/judge.log 2>&1"
elif [ -n "${ARMS:-}" ]; then
  # ARMS="arms.jsonl ...": the judge only (arms measured on earlier pods), into $O/judged.jsonl
  mkdir -p "$S/arms"; k=0; for f in $ARMS; do k=$((k+1)); cp "$f" "$S/arms/a$k.jsonl"; done
  [ -n "${VALIDATION:-}" ] && cp "$VALIDATION" "$S/arms/validation.jsonl"
  WORK="$WORK; python -u items.py judge $S/arms/*.jsonl $O/judged.jsonl --judge ${JUDGE:-Qwen/Qwen3.6-27B-FP8} > $O/judge.log 2>&1"
else
  WORK="$WORK; python -u engine.py smoke --thr $S/thr.pt > $O/smoke.log 2>&1"
fi
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; \
mkdir -p $O $S; tar -xf $S.tar -C $S; (nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv &); \
$WORK"
HF="Qwen/Qwen3-8B ${MODELS:-}"; [ -n "${ARMS:-}" ] && HF="${MODELS:-Qwen/Qwen3.6-27B-FP8}"  # (the judge alone needs no 8B)
RP_HF_MODELS="$HF" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${UPLOAD_MB:-100} \
  RP_OWNER=counterbench RP_PARALLEL=2 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-44} \
  experiments/infra/runpod/rp-run "$N" "${GPU:-RTX A6000,A40,L40S,L40,RTX 6000 Ada}" "$H" -- bash -c "$CMD"
