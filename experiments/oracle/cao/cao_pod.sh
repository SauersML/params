#!/usr/bin/env bash
# The causal activation oracle (cao.py) on a rented GPU, into ~/mpd-data/runpod/oracle-cao-$RUN/:
#   RUN=gen1 MODE=gen TEXTS=4000 [PROCS=2] bash experiments/oracle/cao/cao_pod.sh [HOURS]     examples (PROCS processes, seeds 1..)
#   RUN=train1 MODE=train DATA=<examples on the Mac> [STEPS=N] bash ... [HOURS]                SFT from the released adapter
#   RUN=eval1 MODE=eval ITEMS=<h2h items> ADAPTER_DIR=<a trained adapter on the Mac> bash ...  the gate's questions
#   RUN=org1 MODE=organisms bash ... [HOURS]                                                  organisms.py all (planted truths)
#   (env: SRCREV (default origin/main), GPU, PRICE)
cd /Users/user/params
RUN=${RUN:?RUN} MODE=${MODE:?gen, train, eval or organisms} H=${1:-6}
N=oracle-cao-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/primerl/src_cao_$RUN
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" experiments/oracle method/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
[ -n "${DATA:-}" ] && cp "$DATA" "$S/data.jsonl"
[ -n "${ITEMS:-}" ] && cp "$ITEMS" "$S/items.jsonl"
[ -n "${ADAPTER_DIR:-}" ] && cp -r "$ADAPTER_DIR" "$S/adapter"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
case $MODE in
  gen) WORK=""; for k in $(seq 1 ${PROCS:-2}); do
         WORK="$WORK (python -u cao.py gen --out $O/examples_$k.jsonl --texts $(( ${TEXTS:-100} / ${PROCS:-2} )) --seed $k > $O/gen_$k.log 2>&1) &"
       done; WORK="$WORK wait; cat $O/examples_*.jsonl > $O/examples.jsonl" ;;
  organisms) WORK="python -u organisms.py all --out $O/organisms > $O/organisms.log 2>&1" ;;
  train) WORK="python -u cao.py train --data $S/data.jsonl --out $O/adapter ${STEPS:+--steps $STEPS} > $O/train.log 2>&1" ;;
  eval) AD=${ADAPTER_DIR:+$S/adapter}; AD=${AD:-adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B}  # (ADAPTER_DIR: a step_N directory)
        WORK="python -u cao.py eval --items $S/items.jsonl --adapter $AD --out $O/eval.jsonl > $O/eval.log 2>&1; \
python -u cao.py eval --items $S/items.jsonl --adapter $AD --out $O/eval_text_only.jsonl --text-only > $O/eval_text_only.log 2>&1" ;;
esac
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S; \
tar -xf $S.tar -C $S; cd $S/experiments/oracle/cao; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv & \
$WORK"
RP_HF_MODELS="Qwen/Qwen3-1.7B adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${UPLOAD_MB:-200} \
  RP_OWNER=primerl RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-20} \
  experiments/infra/runpod/rp-run $N "${GPU:-RTX A5000,RTX 3090,RTX 4090,RTX 5090,RTX A6000,A40,L40S}" $H -- bash -c "$CMD"
