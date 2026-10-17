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
#   RUN=e1a MODE=e1eval E1=<dir with behavior_e1.json and adapters/> [ADAPTER_DIR=<oracle>] [ANSWERS=1] bash ...  E1 graded (default:
#       the released AO); ANSWERS: the organisms' own answers too, for e1.py reader
#   RUN=s8 MODE=sft8b RECORDS=<circuit records> ORGS=<training organisms> REPLAY=<ao_mix sample> THR=<reference> STEPS=N [E1=<stage>]
#       [SHARE=0.4] [MICRO=1] bash ...   G2 labels, organism examples, the mix, SFT from AO's adapter, G2 and E1 evaluation
#   RUN=rc1 MODE=readscirc RC=<reads_circuit.py build's directory> REPLAY=<ao_mix sample> THR=<reference> STEPS=N bash ...
#       the circuit-reading test: T-correct and T-mismatch SFT, 3 oracles x 4 inputs, the score (questions: g2rec DENSE=1)
#   RUN=li1 MODE=lipatch II=<their evaluator/ beside it> [ADAPTER_DIR=<ours>] [NITEMS=3200] bash ...   Li et al.'s activation-patching
#       test, the released AO and ours zero-shot, their exact-match scorer
#   RUN=p2 MODE=phase2 CIRCUITS=<circuit questions> ORGS=<training organisms> E1TRAIN=<E1-kind training organisms> E1=<E1 stage>
#       RC=<reads_circuit build> REPLAY=<ao_mix sample> THR=<reference> STEPS=N [EVAL_EVERY=500] bash ...   the 8B oracle's SFT
#       from AO's adapter, every EVAL_EVERY steps: names vs correct writes on the circuit-reading test and E1's battery
#   RUN=ot1 MODE=orgtrain ORG_LIST="taboo:mirror taboo:ladder" [ORG_EPOCHS=2] bash ...   named training organisms (organisms.py train)
#   RUN=g2r MODE=g2rec RECORDS=<circuit records dir> THR=<reference> bash ...              g2.py gen on circuits built elsewhere
#   RUN=at1 MODE=aotext AOTEXT=<ao_text.py build's directory> ARM=circ|none REPLAY=<ao_mix sample> E1=<all seven E1 organisms>
#       II=<Li et al.'s evaluator> STEPS=N [SHARE=0.35] [LR=1e-4] [AOBASE=1] bash ...   one arm of AO + circuits as text: SFT from
#       the released AO, its answers on rc's held-out questions and the member slice, E1's battery, Li et al.; AOBASE: the
#       released AO's answers on both arms' inputs too
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
[ -n "${RC:-}" ] && cp -r "$RC" "$S/rc"
[ -n "${II:-}" ] && cp -r "$II" "$S/ii"
[ -n "${CIRCUITS:-}" ] && cp "$CIRCUITS" "$S/circuits.jsonl"
[ -n "${E1TRAIN:-}" ] && cp -r "$E1TRAIN" "$S/e1train"
[ -n "${RECORDS:-}" ] && cp -r "$RECORDS" "$S/records"
[ -n "${GROUPS:-}" ] && cp -r "$GROUPS" "$S/groups"
[ -n "${REPLAY:-}" ] && cp "$REPLAY" "$S/replay.jsonl"
[ -n "${AOTEXT:-}" ] && cp -r "$AOTEXT" "$S/aotext"
[ -n "${E1:-}" ] && mkdir -p "$S/e1" && cp "$E1/behavior_e1.json" "$E1/probe_battery.json" "$S/e1/" && cp -r "$E1/adapters" "$S/e1/adapters"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
RCMISMATCH=""  # the T-mismatch arm's training, when RC_ORACLES names it
case " ${RC_ORACLES:-T-correct T-mismatch T0} " in *" T-mismatch "*)
  RCMISMATCH="python -u mix.py $O/rc/mix_mismatch.jsonl $O/rc/train_mismatch.jsonl --replay $S/replay.jsonl --share ${SHARE:-0.4} >> $O/mix.log 2>&1; \
CAO_MICRO=${MICRO:-1} python -u cao.py train --data $O/rc/mix_mismatch.jsonl --out $O/ad_mismatch --steps ${STEPS:-0} --save-every ${STEPS:-0} > $O/train_mismatch.log 2>&1;" ;;
esac
RCEVALS=""  # the circuit-reading test's evaluations: oracles x inputs (RC_ORACLES, RC_INPUTS)
for orc in ${RC_ORACLES:-T-correct T-mismatch T0}; do
  case $orc in T-correct) ad="$O/ad_correct/step_${STEPS:-0}" ;; T-mismatch) ad="$O/ad_mismatch/step_${STEPS:-0}" ;; *) ad=$AOAD ;; esac
  for how in ${RC_INPUTS:-correct names mismatch none}; do
    RCEVALS="${RCEVALS}python -u g2.py eval --items $O/rc/test_$how.jsonl --thr $S/thr.pt --adapter $ad --out $O/rc/eval_${orc}_$how.jsonl > $O/rc/eval_${orc}_$how.log 2>&1; "
  done
done
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
        WORK="${ANSWERS:+python -u e1.py answers --dir $S/e1/behavior_e1.json --out $O/e1_answers.jsonl > $O/e1_answers.log 2>&1;} \
python -u e1.py battery --dir $S/e1/behavior_e1.json --battery $S/e1/probe_battery.json --adapter $AD --out $O/e1_battery.jsonl > $O/e1_battery.log 2>&1; \
python -u e1.py eval --dir $S/e1/behavior_e1.json --adapter $AD --out $O/e1.jsonl > $O/e1.log 2>&1; \
${ADAPTER_DIR:+python -u e1.py eval --dir $S/e1/behavior_e1.json --adapter $AD --out $O/e1_text_only.jsonl --text-only > $O/e1_text_only.log 2>&1;} true" ;;
  sft8b) WORK="python -u g2.py gen --out $O/g2.jsonl --thr $S/thr.pt --texts 100000 --records $S/records ${GROUPS:+--groups $S/groups} > $O/g2.log 2>&1; \
python -u organisms.py examples --dir $S/organisms --out $O/organism_examples.jsonl > $O/organism_examples.log 2>&1; \
python -u mix.py $O/data.jsonl $O/g2.jsonl $O/organism_examples.jsonl --replay $S/replay.jsonl --share ${SHARE:-0.4} > $O/mix.log 2>&1; \
CAO_MICRO=${MICRO:-1} python -u cao.py train --data $O/data.jsonl --out $O/adapter --steps $STEPS --vectors $O/organism_examples.pt \
--save-every $STEPS > $O/train.log 2>&1; \
python -u g2.py eval --items $O/g2.heldout.jsonl --thr $S/thr.pt --adapter $O/adapter/step_$STEPS --out $O/g2_eval.jsonl > $O/g2_eval.log 2>&1; \
python -u g2.py eval --items $O/g2.heldout.jsonl --thr $S/thr.pt --adapter $O/adapter/step_$STEPS --out $O/g2_eval_text_only.jsonl --text-only > $O/g2_eval_text_only.log 2>&1; \
${E1:+python -u e1.py battery --dir $S/e1/behavior_e1.json --battery $S/e1/probe_battery.json --adapter $O/adapter/step_$STEPS --out $O/e1_battery.jsonl --budgets 64 > $O/e1_battery.log 2>&1;} true" ;;
  readscirc) WORK="mkdir -p $O/rc && cp $S/rc/test_correct.jsonl $S/rc/test_names.jsonl $S/rc/test_mismatch.jsonl $S/rc/test_none.jsonl \
$S/rc/train_correct.jsonl $S/rc/train_mismatch.jsonl $S/rc/claim.json $O/rc/; \
python -u mix.py $O/rc/mix_correct.jsonl $O/rc/train_correct.jsonl --replay $S/replay.jsonl --share ${SHARE:-0.4} > $O/mix.log 2>&1; \
CAO_MICRO=${MICRO:-1} python -u cao.py train --data $O/rc/mix_correct.jsonl --out $O/ad_correct --steps $STEPS --save-every $STEPS > $O/train_correct.log 2>&1; \
$RCMISMATCH \
${RCEVALS}python -u reads_circuit.py score $O/rc > $O/rc_score.txt 2>&1" ;;
  phase2) CKS=""; for k in $(seq ${EVAL_EVERY:-500} ${EVAL_EVERY:-500} ${STEPS:?STEPS}); do  # each checkpoint's evaluations
           ad=$O/adapter/step_$k
           CKS="$CKS python -u g2.py eval --items $S/rc/test_correct.jsonl --thr $S/thr.pt --adapter $ad --out $O/ev/rc_correct_$k.jsonl > $O/ev/rc_correct_$k.log 2>&1; \
python -u g2.py eval --items $S/rc/test_names.jsonl --thr $S/thr.pt --adapter $ad --out $O/ev/rc_names_$k.jsonl > $O/ev/rc_names_$k.log 2>&1; \
python -u e1.py battery --dir $S/e1/behavior_e1.json --battery $S/e1/probe_battery.json --adapter $ad --out $O/ev/e1_$k.jsonl --budgets 64 > $O/ev/e1_$k.log 2>&1;"
         done
         WORK="mkdir -p $O/ev; python -u organisms.py examples --dir $S/organisms --out $O/organism_examples.jsonl > $O/organism_examples.log 2>&1; \
python -u e1.py examples --dir $S/e1train --out $O/e1_examples.jsonl > $O/e1_examples.log 2>&1; \
python -u mix.py $O/data.jsonl $S/circuits.jsonl $O/organism_examples.jsonl $O/e1_examples.jsonl --replay $S/replay.jsonl --share ${SHARE:-0.4} > $O/mix.log 2>&1; \
CAO_MICRO=${MICRO:-1} python -u cao.py train --data $O/data.jsonl --out $O/adapter --steps $STEPS --save-every ${EVAL_EVERY:-500} \
--vectors $O/organism_examples.pt $O/e1_examples.pt > $O/train.log 2>&1; $CKS true" ;;
  aotext) AT=$O/adapter/step_${STEPS:?STEPS}
         WORK="mkdir -p $O/answers; python -u mix.py $O/data.jsonl $S/aotext/train_${ARM:?ARM}.jsonl --replay $S/replay.jsonl \
--share ${SHARE:-0.35} > $O/mix.log 2>&1; CAO_MICRO=${MICRO:-4} CAO_LR=${LR:-1e-4} CAO_DECAY=1 CAO_WARMUP=10 python -u cao.py train \
--data $O/data.jsonl --out $O/adapter --steps $STEPS --save-every $(( STEPS / 2 )) > $O/train.log 2>&1; \
for t in test test_members; do python -u ao_text.py answers $AT $S/aotext/\${t}_$ARM.jsonl $O/answers/\${t}_$ARM.jsonl \
> $O/answers_\$t.log 2>&1; done; \
${AOBASE:+for t in test test_members; do for x in circ none; do python -u ao_text.py answers $AOAD $S/aotext/\${t}_\$x.jsonl \
$O/answers/ao_\${t}_\$x.jsonl > $O/ao_answers_\${t}_\$x.log 2>&1; done; done;} \
python -u e1.py battery --dir $S/e1/behavior_e1.json --battery $S/e1/probe_battery.json --adapter $AT --out $O/e1_battery.jsonl \
> $O/e1_battery.log 2>&1; python -u li_patch.py --repo $S/ii --adapter $AT --out $O/li.jsonl --n ${NITEMS:-3200} > $O/li.log 2>&1; true" ;;
  orgtrain) WORK=""; for x in ${ORG_LIST:?ORG_LIST kind:name ...}; do
         WORK="$WORK python -u organisms.py train --kind ${x%%:*} --name ${x#*:} --out $O/organisms >> $O/orgtrain.log 2>&1;"
       done; WORK="$WORK true" ;;
  lipatch) WORK="python -u li_patch.py --repo $S/ii --adapter $AOAD ${ADAPTER_DIR:+--adapter $S/adapter} --out $O/li.jsonl --n ${NITEMS:-3200} > $O/li.log 2>&1" ;;
  g2rec) WORK="python -u g2.py gen --out $O/g2.jsonl --thr $S/thr.pt --texts 100000 --records $S/records ${GROUPS:+--groups $S/groups} ${DENSE:+--dense} > $O/g2.log 2>&1" ;;
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
