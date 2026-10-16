#!/usr/bin/env bash
# The causal activation oracle (cao.py) on a rented GPU, into ~/mpd-data/runpod/oracle-pathg-$RUN/:
#   RUN=gen1 MODE=gen TEXTS=4000 [PROCS=2] bash experiments/oracle/cao/cao_pod.sh [HOURS]     examples (PROCS processes, seeds 1..)
#   RUN=train1 MODE=train DATA=<examples on the Mac> [STEPS=N] [VECTORS=<.pt>] [INIT=<adapter dir>] [SAVE_EVERY=N] [MICRO=N] [G2EVAL=<g2 held-out items> THR=<ref> STEPS=N] bash ...  SFT from the released adapter
#   RUN=eval1 MODE=eval ITEMS=<h2h items> ADAPTER_DIR=<a trained adapter on the Mac> bash ...  the gate's questions (and the controls);
#       [INF=<inference items>] the never-made inferences; [ORGS=<organisms dir>] planted truths; [AOBASE=1] the released AO too;
#       [VAL=<held-out training examples>] the vectors' use per category (cao.py gain)
#   RUN=inf1 MODE=inference [PER=150] bash ... [HOURS]                                       inference.py build (never-made inferences)
#   RUN=org1 MODE=organisms bash ... [HOURS]                                                  organisms.py all (planted truths)
#   RUN=g2f MODE=g2finder QUESTIONS=<circuit questions .json> THR=<reference> [LIMIT=200] bash ...  circuit_finder on the dev split, then g2.py gen --records
#   RUN=g2e MODE=g2eval G2EVAL=<g2 held-out items> ADAPTER_DIR=<adapter> THR=<reference> bash ...  g2.py eval, vectors and text-only
#   RUN=e1a MODE=e1eval E1=<dir with behavior_e1.json and adapters/> [ADAPTER_DIR=<oracle>] bash ...  E1 graded (default: the released AO)
#   RUN=g2r MODE=g2rec RECORDS=<circuit records dir> THR=<reference> bash ...              g2.py gen on circuits built elsewhere
#   RUN=oe1 MODE=orgeval ORGS=<organisms dir on the Mac> [ADAPTER_DIR=...] UPLOAD_MB=1500 bash ...  planted truths recovered
#   (env: SRCREV (default origin/main), GPU, PRICE, CAO_MODEL (default Qwen/Qwen3-1.7B; Qwen/Qwen3-8B wants a 48 GB card))
cd /Users/user/params
MODEL=${CAO_MODEL:-Qwen/Qwen3-1.7B}
case $MODEL in  # the released Activation Oracle (cao.AO)
  Qwen/Qwen3-8B) AOAD=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B ;;
  *) AOAD=adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B ;;
esac
RUN=${RUN:?RUN} MODE=${MODE:?gen, train, eval or organisms} H=${1:-6}
N=oracle-pathg-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/primerl/src_cao_$RUN
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" experiments/oracle method/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
[ -n "${DATA:-}" ] && cp "$DATA" "$S/data.jsonl"
[ -n "${ITEMS:-}" ] && cp "$ITEMS" "$S/items.jsonl"
[ -n "${ADAPTER_DIR:-}" ] && cp -r "$ADAPTER_DIR" "$S/adapter"
[ -n "${ORGS:-}" ] && cp -r "$ORGS" "$S/organisms"
[ -n "${VECTORS:-}" ] && cp "$VECTORS" "$S/vectors.pt"
[ -n "${INF:-}" ] && cp "$INF" "$S/inf.jsonl"
[ -n "${VAL:-}" ] && cp "$VAL" "$S/val.jsonl"
[ -n "${THR:-}" ] && cp "$THR" "$S/thr.pt"
[ -n "${INIT:-}" ] && cp -r "$INIT" "$S/init"
[ -n "${QUESTIONS:-}" ] && cp "$QUESTIONS" "$S/questions.json"
[ -n "${G2EVAL:-}" ] && cp "$G2EVAL" "$S/g2eval.jsonl"
[ -n "${RECORDS:-}" ] && cp -r "$RECORDS" "$S/records"
[ -n "${E1:-}" ] && mkdir -p "$S/e1" && cp "$E1/behavior_e1.json" "$S/e1/" && cp -r "$E1/adapters" "$S/e1/adapters"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
case $MODE in
  gen) WORK=""; for k in $(seq 1 ${PROCS:-2}); do
         WORK="$WORK (python -u cao.py gen --out $O/examples_$k.jsonl --texts $(( ${TEXTS:-100} / ${PROCS:-2} )) --seed $k > $O/gen_$k.log 2>&1) &"
       done; WORK="$WORK wait; cat $O/examples_*.jsonl > $O/examples.jsonl" ;;
  g2finder) WORK="python -u $S/method/oracle/circuit_finder.py questions $S/questions.json $O/circuits --split dev --limit ${LIMIT:-200} --stride ${STRIDE:-2} \
--model $MODEL --dtype bfloat16 --device cuda > $O/finder.log 2>&1; \
python -u g2.py gen --out $O/g2.jsonl --thr $S/thr.pt --texts ${LIMIT:-200} --records $O/circuits > $O/g2.log 2>&1" ;;
  g2eval) WORK="python -u g2.py eval --items $S/g2eval.jsonl --thr $S/thr.pt --adapter $S/adapter --out $O/g2_eval.jsonl > $O/g2_eval.log 2>&1; \
python -u g2.py eval --items $S/g2eval.jsonl --thr $S/thr.pt --adapter $S/adapter --out $O/g2_eval_text_only.jsonl --text-only > $O/g2_eval_text_only.log 2>&1" ;;
  e1eval) AD=${ADAPTER_DIR:+$S/adapter}; AD=${AD:-$AOAD}
        WORK="python -u e1.py eval --dir $S/e1/behavior_e1.json --adapter $AD --out $O/e1.jsonl > $O/e1.log 2>&1; \
${ADAPTER_DIR:+python -u e1.py eval --dir $S/e1/behavior_e1.json --adapter $AD --out $O/e1_text_only.jsonl --text-only > $O/e1_text_only.log 2>&1;} true" ;;
  g2rec) WORK="python -u g2.py gen --out $O/g2.jsonl --thr $S/thr.pt --texts 100000 --records $S/records > $O/g2.log 2>&1" ;;
  inference) WORK="python -u inference.py build --out $O/inference_items.jsonl --per ${PER:-150} > $O/inference.log 2>&1" ;;
  organisms) WORK="python -u organisms.py all --out $O/organisms > $O/organisms.log 2>&1 && \
python -u organisms.py examples --dir $O/organisms --out $O/organism_examples.jsonl > $O/organism_examples.log 2>&1" ;;
  orgeval) AD=${ADAPTER_DIR:+$S/adapter}; AD=${AD:-$AOAD}
        WORK="python -u cao.py organisms --organisms $S/organisms --adapter $AD --out $O/orgeval.jsonl > $O/orgeval.log 2>&1; \
python -u cao.py organisms --organisms $S/organisms --adapter $AD --out $O/orgeval_text_only.jsonl --text-only > $O/orgeval_text_only.log 2>&1" ;;
  train) WORK="CAO_MICRO=${MICRO:-4} CAO_CHECKPOINT=${CHECKPOINT:-1} python -u cao.py train --data $S/data.jsonl --out $O/adapter ${STEPS:+--steps $STEPS} \
${VECTORS:+--vectors $S/vectors.pt} ${INIT:+--init $S/init} --save-every ${SAVE_EVERY:-2000} > $O/train.log 2>&1; \
${G2EVAL:+python -u g2.py eval --items $S/g2eval.jsonl --thr $S/thr.pt --adapter $O/adapter/step_$STEPS --out $O/g2_eval.jsonl > $O/g2_eval.log 2>&1; \
python -u g2.py eval --items $S/g2eval.jsonl --thr $S/thr.pt --adapter $O/adapter/step_$STEPS --out $O/g2_eval_text_only.jsonl --text-only > $O/g2_eval_text_only.log 2>&1;} true" ;;
  eval) AD=${ADAPTER_DIR:+$S/adapter}; AD=${AD:-$AOAD}  # (ADAPTER_DIR: a step_N directory)
        WORK="python -u cao.py eval --items $S/items.jsonl --adapter $AD --out $O/eval.jsonl > $O/eval.log 2>&1; \
python -u cao.py eval --items $S/items.jsonl --adapter $AD --out $O/eval_text_only.jsonl --text-only > $O/eval_text_only.log 2>&1; \
python -u cao.py eval --items $S/items.jsonl --adapter $AD --out $O/eval_wrong_text.jsonl --wrong-text > $O/eval_wrong_text.log 2>&1; \
${AOBASE:+python -u cao.py eval --items $S/items.jsonl --out $O/eval_ao.jsonl > $O/eval_ao.log 2>&1;} \
${VAL:+python -u cao.py gain --data $S/val.jsonl --adapter $AD --n 100000 > $O/gain.log 2>&1;} \
${INF:+python -u inference.py score --items $S/inf.jsonl --adapter $AD --out $O/inference.jsonl --no-ao > $O/inference.log 2>&1;} \
${ORGS:+python -u cao.py organisms --organisms $S/organisms --adapter $AD --out $O/orgeval.jsonl > $O/orgeval.log 2>&1; \
python -u cao.py organisms --organisms $S/organisms --adapter $AD --out $O/orgeval_text_only.jsonl --text-only > $O/orgeval_text_only.log 2>&1;} true" ;;
esac
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CAO_MODEL=$MODEL ${ORG_EPOCHS:+ORG_EPOCHS=$ORG_EPOCHS}; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S; \
tar -xf $S.tar -C $S; cd $S/experiments/oracle/cao; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv & \
$WORK"
RP_HF_MODELS="$MODEL $AOAD" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${UPLOAD_MB:-200} \
  RP_OWNER=primerl RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-20} \
  experiments/infra/runpod/rp-run $N "${GPU:-RTX A5000,RTX 3090,RTX 4090,RTX 5090,RTX A6000,A40,L40S}" $H -- bash -c "$CMD"
