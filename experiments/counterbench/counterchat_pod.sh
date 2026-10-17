#!/usr/bin/env bash
# CounterChat's GPU work on a rented 48 GB card, into ~/mpd-data/runpod/orc-counterbench-$RUN/:
#   RUN=smoke1 bash experiments/counterbench/counterchat_pod.sh [HOURS]           engine.py smoke (hooks reach the model,
#                                                                                hooks off = unedited, throughput)
#   RUN=m1 JOBS=~/mpd-data/counterbench/jobs/m1.jsonl [VALIDATION=hand-labelled arms] MODELS="Qwen/Qwen3-14B Qwen/Qwen3.6-27B-FP8" \
#     bash ... [HOURS]      items.py measure on a job file (8B arms, then 14B arms when MODELS names it), then the judge on every
#                           arm without a program and on the hand-labelled replies (its validation): $O/judged.jsonl
# (env: SRCREV (default origin/main), GPU, PRICE, MODELS: extra HF models to fetch, e.g. Qwen/Qwen3-14B)
set -euo pipefail
cd /Users/user/params
RUN=${RUN:?RUN} H=${1:-0.5}
N=orc-counterbench-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/counterbench/src_$RUN
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" experiments/counterbench method/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
cp /Users/user/mpd-data/cluster/t2_8b/out/thr.pt "$S/thr.pt"
WORK="cd $S/experiments/counterbench"
if [ -n "${JOBS:-}" ]; then
  cp "$JOBS" "$S/jobs.jsonl"
  [ -n "${VALIDATION:-}" ] && cp "$VALIDATION" "$S/validation.jsonl"
  WORK="$WORK; python -u items.py measure $S/jobs.jsonl $O/arms.jsonl --thr $S/thr.pt > $O/measure_8b.log 2>&1"
  case " ${MODELS:-} " in *" Qwen/Qwen3-14B "*) WORK="$WORK; python -u items.py measure $S/jobs.jsonl $O/arms.jsonl --model Qwen/Qwen3-14B > $O/measure_14b.log 2>&1";; esac
  WORK="$WORK; cat $O/arms.jsonl ${VALIDATION:+$S/validation.jsonl} > $S/to_judge.jsonl; python -u items.py judge $S/to_judge.jsonl $O/judged.jsonl --judge ${JUDGE:-Qwen/Qwen3.6-27B-FP8} > $O/judge.log 2>&1"
else
  WORK="$WORK; python -u engine.py smoke --thr $S/thr.pt > $O/smoke.log 2>&1"
fi
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; \
mkdir -p $O $S; tar -xf $S.tar -C $S; (nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv &); \
$WORK"
RP_HF_MODELS="Qwen/Qwen3-8B ${MODELS:-}" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${UPLOAD_MB:-100} \
  RP_OWNER=counterbench RP_PARALLEL=2 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-44} \
  experiments/infra/runpod/rp-run "$N" "${GPU:-RTX A6000,A40,L40S,L40,RTX 6000 Ada}" "$H" -- bash -c "$CMD"
