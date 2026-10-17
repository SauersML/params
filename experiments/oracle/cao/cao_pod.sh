#!/usr/bin/env bash
# The activation oracle (cao.py) on a rented GPU, into ~/mpd-data/runpod/oracle-pathg-$RUN/:
#   RUN=at1 MODE=aotext AOTEXT=<ao_text.py build's directory> ARMS="none circ" REPLAY=<ao_mix sample> STEPS=N [SHARE=0.35]
#       [LR=1e-4] [EVAL=0 | E1=<all seven E1 organisms> II=<Li et al.'s evaluator>] [AOBASE=1] bash ...   AO + circuits as text,
#       each arm in turn: SFT from the released AO (adapter_<arm>), its answers on rc's held-out questions and the member
#       slice, E1's battery and Li et al. (not with EVAL=0); AOBASE: the released AO's answers on both arms' inputs too
#   RUN=train1 MODE=train DATA=<examples on the Mac> [STEPS=N] [VECTORS=<.pt>] [INIT=<adapter dir>] [SAVE_EVERY=N] [MICRO=N] bash ...
#       SFT from the released adapter
#   RUN=e1a MODE=e1eval E1=<dir with behavior_e1.json and adapters/> [ADAPTER_DIR=<oracle>] [ANSWERS=1] bash ...  E1 graded (default:
#       the released AO); ANSWERS: the organisms' own answers too, for e1.py reader
#   RUN=li1 MODE=lipatch II=<their evaluator/ beside it> [ADAPTER_DIR=<ours>] [NITEMS=3200] bash ...   Li et al.'s activation-patching
#       test, the released AO and ours zero-shot, their exact-match scorer
#   RUN=g2r MODE=g2rec RECORDS=<circuit records dir> THR=<reference> [GROUPS=<apply dir>] [DENSE=1] bash ...   g2.py gen (circuit
#       questions with measured answers)
#   RUN=inf1 MODE=inference [PER=150] bash ... [HOURS]                                       inference.py build (never-made inferences)
#   RUN=org1 MODE=organisms bash ... [HOURS]                                                  organisms.py all (planted truths)
#   RUN=ot1 MODE=orgtrain ORG_LIST="taboo:mirror taboo:ladder" [ORG_EPOCHS=2] bash ...   named training organisms (organisms.py train)
#   (env: SRCREV (default origin/main), GPU, PRICE, CAO_MODEL (default Qwen/Qwen3-1.7B; Qwen/Qwen3-8B wants a 48 GB card))
cd /Users/user/params
MODEL=${CAO_MODEL:-Qwen/Qwen3-1.7B}
case $MODEL in  # the released Activation Oracle (cao.AO)
  Qwen/Qwen3-8B) AOAD=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B ;;
  *) AOAD=adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B ;;
esac
RUN=${RUN:?RUN} MODE=${MODE:?MODE} H=${1:-6}
N=oracle-pathg-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/primerl/src_cao_$RUN
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" experiments/oracle method/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
[ -n "${DATA:-}" ] && cp "$DATA" "$S/data.jsonl"
[ -n "${ADAPTER_DIR:-}" ] && cp -r "$ADAPTER_DIR" "$S/adapter"
[ -n "${VECTORS:-}" ] && cp "$VECTORS" "$S/vectors.pt"
[ -n "${THR:-}" ] && cp "$THR" "$S/thr.pt"
[ -n "${INIT:-}" ] && cp -r "$INIT" "$S/init"
[ -n "${II:-}" ] && cp -r "$II" "$S/ii"
[ -n "${RECORDS:-}" ] && cp -r "$RECORDS" "$S/records"
[ -n "${GROUPS:-}" ] && cp -r "$GROUPS" "$S/groups"
[ -n "${REPLAY:-}" ] && cp "$REPLAY" "$S/replay.jsonl"
[ -n "${AOTEXT:-}" ] && cp -r "$AOTEXT" "$S/aotext"
[ -n "${E1:-}" ] && mkdir -p "$S/e1" && cp "$E1/behavior_e1.json" "$E1/probe_battery.json" "$S/e1/" && cp -r "$E1/adapters" "$S/e1/adapters"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
case $MODE in
  aotext) AOB="" WORK="mkdir -p $O/answers;"
         [ -n "${AOBASE:-}" ] && AOB="for t in test test_members; do for x in circ none; do python -u ao_text.py answers $AOAD \
$S/aotext/\${t}_\${x}.jsonl $O/answers/ao_\${t}_\${x}.jsonl > $O/ao_answers_\${t}_\${x}.log 2>&1; done; done;"
         for arm in ${ARMS:-${ARM:?ARM or ARMS}}; do  # (each arm's SFT, then its answers; E1 and Li unless EVAL=0)
           AT=$O/adapter_$arm/step_${STEPS:?STEPS}
           WORK="$WORK python -u mix.py $O/data_$arm.jsonl $S/aotext/train_$arm.jsonl --replay $S/replay.jsonl --share ${SHARE:-0.35} \
> $O/mix_$arm.log 2>&1; CAO_MICRO=${MICRO:-4} CAO_LR=${LR:-1e-4} CAO_DECAY=1 CAO_WARMUP=10 python -u cao.py train \
--data $O/data_$arm.jsonl --out $O/adapter_$arm --steps $STEPS --save-every $(( STEPS / 2 )) > $O/train_$arm.log 2>&1; \
for t in test test_members; do python -u ao_text.py answers $AT $S/aotext/\${t}_$arm.jsonl $O/answers/\${t}_$arm.jsonl \
> $O/answers_\${t}_$arm.log 2>&1; done;"
           [ "${EVAL:-1}" = 1 ] && WORK="$WORK python -u e1.py battery --dir $S/e1/behavior_e1.json --battery $S/e1/probe_battery.json \
--adapter $AT --out $O/e1_battery_$arm.jsonl > $O/e1_battery_$arm.log 2>&1; python -u li_patch.py --repo $S/ii --adapter $AT \
--out $O/li_$arm.jsonl --n ${NITEMS:-3200} > $O/li_$arm.log 2>&1;"
         done
         WORK="$WORK $AOB true" ;;
  train) WORK="CAO_MICRO=${MICRO:-4} CAO_CHECKPOINT=${CHECKPOINT:-1} python -u cao.py train --data $S/data.jsonl --out $O/adapter ${STEPS:+--steps $STEPS} \
${VECTORS:+--vectors $S/vectors.pt} ${INIT:+--init $S/init} --save-every ${SAVE_EVERY:-2000} > $O/train.log 2>&1" ;;
  e1eval) AD=${ADAPTER_DIR:+$S/adapter}; AD=${AD:-$AOAD}
        WORK="${ANSWERS:+python -u e1.py answers --dir $S/e1/behavior_e1.json --out $O/e1_answers.jsonl > $O/e1_answers.log 2>&1;} \
python -u e1.py battery --dir $S/e1/behavior_e1.json --battery $S/e1/probe_battery.json --adapter $AD --out $O/e1_battery.jsonl > $O/e1_battery.log 2>&1; \
python -u e1.py eval --dir $S/e1/behavior_e1.json --adapter $AD --out $O/e1.jsonl > $O/e1.log 2>&1; \
${ADAPTER_DIR:+python -u e1.py eval --dir $S/e1/behavior_e1.json --adapter $AD --out $O/e1_text_only.jsonl --text-only > $O/e1_text_only.log 2>&1;} true" ;;
  lipatch) WORK="python -u li_patch.py --repo $S/ii --adapter $AOAD ${ADAPTER_DIR:+--adapter $S/adapter} --out $O/li.jsonl --n ${NITEMS:-3200} > $O/li.log 2>&1" ;;
  g2rec) WORK="python -u g2.py gen --out $O/g2.jsonl --thr $S/thr.pt --texts 100000 --records $S/records ${GROUPS:+--groups $S/groups} ${DENSE:+--dense} > $O/g2.log 2>&1" ;;
  inference) WORK="python -u inference.py build --out $O/inference_items.jsonl --per ${PER:-150} > $O/inference.log 2>&1" ;;
  organisms) WORK="python -u organisms.py all --out $O/organisms > $O/organisms.log 2>&1 && \
python -u organisms.py examples --dir $O/organisms --out $O/organism_examples.jsonl > $O/organism_examples.log 2>&1" ;;
  orgtrain) WORK=""; for x in ${ORG_LIST:?ORG_LIST kind:name ...}; do
         WORK="$WORK python -u organisms.py train --kind ${x%%:*} --name ${x#*:} --out $O/organisms >> $O/orgtrain.log 2>&1;"
       done; WORK="$WORK true" ;;
  *) echo "unknown MODE $MODE" >&2; exit 2 ;;
esac
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CAO_MODEL=$MODEL ${ORG_EPOCHS:+ORG_EPOCHS=$ORG_EPOCHS}; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S; \
tar -xf $S.tar -C $S; cd $S/experiments/oracle/cao; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv & \
$WORK"
RP_HF_MODELS="$MODEL $AOAD" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${UPLOAD_MB:-200} \
  RP_OWNER=primerl RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-20} \
  experiments/infra/runpod/rp-run $N "${GPU:-RTX A5000,RTX 3090,RTX 4090,RTX 5090,RTX A6000,A40,L40S}" $H -- bash -c "$CMD"
