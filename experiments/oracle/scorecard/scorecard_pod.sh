#!/usr/bin/env bash
# The scorecard's GPU measurements on a rented card, into ~/mpd-data/runpod/orc-oracleeval-$RUN/:
#   RUN=m1 ORACLES="random-units agent-x" [TARGETS=o3,o4] [AOE1B=1] bash experiments/oracle/scorecard/scorecard_pod.sh [HOURS]
# measure.py on each named oracle's answers (ROOT/answers/<name>/{o3,o4}.jsonl) and, with AOE1B, the released AO's E1
# battery on the e1b organisms (e1.py battery); then on the Mac: scorecard.py collect orc-oracleeval-$RUN
#   GREEDY=1 (with o3 in TARGETS): T3's greedy edit (greedy_edit.py) on the E2 organisms first, measured as oracle t3-greedy
#   AB=1: AuditBench's dev targets (Qwen3-14B LoRAs): their answers to the paper's 50 probes and the released AO 14B's
#   readings (auditbench.py generate), then on the Mac auditbench.py predict --static ~/mpd-data/runpod/orc-oracleeval-$RUN/auditbench
#   REMOVAL=1 ORACLES=t3-greedy TARGETS=o3: the standard removal baselines for O3 (removal_baselines.py: direction ablation,
#   steering, a LoRA fine-tune, random poles matched by layer to T3's), measured as measure.py measures an edit
#   SERVE=1 [PORT=8790]: paper_scaffold.py serve, the AuditBench dev targets for the paper's scaffold baseline, until the
#   pod's hours run out (reach it with ~/mpd-data/scratch/oracleeval/tools/tunnel.sh RUN)
#   RUN=r1 READER=<adapter dir on the Mac, or an HF id> [RCSET=rc1|rc2_native] bash ...   a reader checkpoint: O1 (g2.py eval
# on the circuit questions under four inputs: correct, names, mismatch, none), O2 (e1.py battery on all seven organisms),
# O5 (li_patch.py); then scorecard.py collect orc-oracleeval-r1 --name <oracle name>
# (env: SRCREV (default origin/main), GPU, PRICE)
set -euo pipefail
cd /Users/user/params
RUN=${RUN:?RUN} H=${1:-1.5}
N=orc-oracleeval-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/oracleeval/src_$RUN
ROOT=/Users/user/mpd-data/oracle/scorecard
MODEL=Qwen/Qwen3-8B AOAD=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" experiments/oracle method/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
cp -r $ROOT/items "$S/items"
mkdir -p "$S/answers" "$S/adapters"
for o in ${ORACLES:-}; do cp -r "$ROOT/answers/$o" "$S/answers/$o"; done
case ${TARGETS:-o3,o4} in *o3*) for n in backdoor2 cond_pref lie lie2; do cp -r /Users/user/mpd-data/organisms/qwen3_8b/adapters/$n "$S/adapters/$n"; done ;; esac
cp /Users/user/mpd-data/cluster/t2_8b/out/thr.pt "$S/thr.pt"
[ -n "${GREEDY:-}${REMOVAL:-}" ] && cp /Users/user/mpd-data/organisms/qwen3_8b/data/general.jsonl "$S/general.jsonl"
if [ -n "${AOE1B:-}" ]; then
  mkdir -p "$S/e1b/adapters"
  /Users/user/mpd-data/venv/bin/python -c "
import json,sys
d=json.load(open('/Users/user/mpd-data/organisms/qwen3_8b/behavior_e1.json'))
d['organisms']={k:v for k,v in d['organisms'].items() if k in ('pref_plant','lie_kili','cue_chile2')}
json.dump(d,open('$S/e1b/behavior_e1.json','w'))"
  cp /Users/user/mpd-data/organisms/qwen3_8b/probe_battery.json "$S/e1b/"
  for n in pref_plant lie_kili cue_chile2; do cp -r /Users/user/mpd-data/organisms/qwen3_8b_e1b/adapters/$n "$S/e1b/adapters/$n"; done
fi
if [ -n "${READER:-}" ]; then
  AD=$READER; [ -d "$READER" ] && cp -r "$READER" "$S/reader" && AD=$S/reader
  mkdir -p "$S/rc" "$S/e1all/adapters"
  for h in correct names mismatch none; do cp /Users/user/mpd-data/oracle/reads_circuit/${RCSET:-rc1}/test_$h.jsonl "$S/rc/"; done
  cp /Users/user/mpd-data/organisms/qwen3_8b/behavior_e1.json /Users/user/mpd-data/organisms/qwen3_8b/probe_battery.json "$S/e1all/"
  for n in backdoor2 cond_pref lie lie2; do cp -r /Users/user/mpd-data/organisms/qwen3_8b/adapters/$n "$S/e1all/adapters/$n"; done
  for n in pref_plant lie_kili cue_chile2; do cp -r /Users/user/mpd-data/organisms/qwen3_8b_e1b/adapters/$n "$S/e1all/adapters/$n"; done
  cp -r /Users/user/mpd-data/scratch/primerl/ext/ii "$S/ii"
fi
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
WORK="true"
[ -n "${AOE1B:-}" ] && WORK="$WORK; (cd $S/experiments/oracle/cao && python -u e1.py battery --dir $S/e1b/behavior_e1.json \
--battery $S/e1b/probe_battery.json --adapter $AOAD --out $O/ao_e1b_battery.jsonl > $O/ao_e1b_battery.log 2>&1)"
MEASURE=${ORACLES:-}
[ -n "${GREEDY:-}" ] && MEASURE="$MEASURE t3-greedy" && WORK="$WORK; mkdir -p $O/answers; python -u $S/experiments/oracle/scorecard/greedy_edit.py \
--items $S/items --adapters $S/adapters --thr $S/thr.pt --general $S/general.jsonl --out $O/answers/t3-greedy > $O/greedy.log 2>&1"
[ -n "$MEASURE" ] && WORK="$WORK; mkdir -p $O/answers; cp -r $S/answers/. $O/answers/; python -u $S/experiments/oracle/scorecard/measure.py \
--answers $(for o in $MEASURE; do printf '%s ' "$O/answers/$o"; done)--items $S/items --adapters $S/adapters --thr $S/thr.pt \
--targets ${TARGETS:-o3,o4} > $O/measure.log 2>&1"
[ -n "${READER:-}" ] && WORK="$WORK; mkdir -p $O/reader; echo ${RCSET:-rc1} > $O/reader/rcset.txt; cd $S/experiments/oracle/cao; \
for h in correct names mismatch none; do python -u g2.py eval --items $S/rc/test_\$h.jsonl --thr $S/thr.pt --adapter $AD \
--out $O/reader/o1_\$h.jsonl > $O/reader/o1_\$h.log 2>&1; done; \
python -u e1.py battery --dir $S/e1all/behavior_e1.json --battery $S/e1all/probe_battery.json --adapter $AD \
--out $O/reader/e1_battery.jsonl > $O/reader/e1_battery.log 2>&1; \
python -u li_patch.py --repo $S/ii --adapter $AD --out $O/reader/li.jsonl --n 3200 > $O/reader/li.log 2>&1"
[ -n "${AB:-}" ] && WORK="$WORK; mkdir -p $O/auditbench; python -u $S/experiments/oracle/scorecard/auditbench.py generate \
--items $S/items/auditbench.json --out $O/auditbench ${ABFLAGS:-} > $O/auditbench.log 2>&1"
[ -n "${REMOVAL:-}" ] && WORK="$WORK; mkdir -p $O/answers; python -u $S/experiments/oracle/scorecard/removal_baselines.py --items $S/items \
--adapters $S/adapters --thr $S/thr.pt --general $S/general.jsonl --t3 $S/answers/t3-greedy/o3.jsonl --out $O/answers > $O/removal.log 2>&1"
[ -n "${SERVE:-}" ] && WORK="$WORK; python -u $S/experiments/oracle/scorecard/paper_scaffold.py serve --items $S/items/auditbench.json \
--port ${PORT:-8790} > $O/serve.log 2>&1"
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CAO_MODEL=$MODEL; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; \
mkdir -p $O $S; tar -xf $S.tar -C $S; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv & \
$WORK"
[ -n "${AB:-}${SERVE:-}" ] && MODEL=Qwen/Qwen3-14B AOAD=adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B
RP_HF_MODELS="$MODEL $AOAD" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${UPLOAD_MB:-1500} \
  RP_OWNER=oracleeval RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-44} \
  experiments/infra/runpod/rp-run $N "${GPU:-A40,RTX A6000,L40S,RTX 6000 Ada}" $H -- bash -c "$CMD"
