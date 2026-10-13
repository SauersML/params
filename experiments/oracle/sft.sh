#!/bin/zsh
# The component explainer's SFT and its held-out evaluation (#2951 graph oracle): SFT rows from question rows and
# teacher circuits (with their English and the edit teacher's edits when given), one Tinker SFT on BASE, samples on
# held-out text9000-9019 (fresh QA seed), E1 by the readers when the SFT has English (reported, not in the reward;
# without English every answer's E1 is 0), and the evaluation split by locality against the held-out teachers
# themselves.
#   zsh experiments/oracle/sft.sh RUN   (env: ROWS, TEACHER, ENGLISH, EDIT_TEACH, BASE, STEPS, BATCH, LR, HELDOUT,
#                                        HELDOUT_TEACHER, LABELS, NEVAL, SAMPLES, MAXTOK, SRC, ZEROSHOT: the untrained base's
#                                        samples and readers on HELDOUT's questions; EXTRA: more answer sets scored in the
#                                        same evaluation, "NAME=SAMPLES[:E1] ...", e.g. an earlier SFT's samples, since C
#                                        changes on main and only one evaluation compares; EDIT_EVAL: the edit teacher's
#                                        rows for HELDOUT's edit requests, E3's best net; MEASURED=1: the measured input
#                                        arm, subcomponents ranked by importance x |activation|, attention heads' top positions;
#                                        POOL=1: every VPD-active subcomponent at the listed positions in the question;
#                                        THINKING=1: Qwen3's thinking mode, generation limit 16384; MAXLEN: SFT length cap;
#                                        RENDERED=1: ROWS are already rerendered with these flags, used as they are)
set -e
RUN=$1
O=~/mpd-data/graph_oracle/components; S=$O/sft_$RUN; mkdir -p $S
ROWS=${ROWS:?question rows: components.py prepare output, several joined}
TEACHER=${TEACHER:?teacher circuits (<task>.json)}
BASE=${BASE:-Qwen/Qwen3-8B} STEPS=${STEPS:-100} BATCH=${BATCH:-8} NEVAL=${NEVAL:-20} SAMPLES=${SAMPLES:-4} MAXTOK=${MAXTOK:-4096}
HELDOUT=${HELDOUT:-$O/zeroshot_heldout20/fresh20_rt_st.jsonl} HELDOUT_TEACHER=${HELDOUT_TEACHER:-~/mpd-data/graph_oracle/teacher/joint_heldout}
LABELS=${LABELS:-$O/heldout50_locality_frozen.jsonl}  # frozen once: never relabel per teacher version
SRC=${SRC:-/Users/user/params}  # a source tree (a clean copy of main while others edit the shared tree)
PY=/Users/user/mpd-data/venv/bin/python; TPY=/Users/user/mpd-data/tinker-venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_MPS_PREFER_METAL=1
RR=(rerender ${MEASURED:+--measured} ${POOL:+--pool})  # MEASURED: the measured input arm (components.measured()); POOL: the full candidate pool (candidate_pool())
TH=(${THINKING:+--thinking})  # THINKING: Qwen3's thinking mode in sampling (the SFT keeps the empty think block in the prompt)
[ -n "${THINKING:-}" ] && MAXTOK=${MAXTOK_THINKING:-16384}  # one generation limit, thinking included (no thinking budget)
if [ -n "${RENDERED:-}" ]; then cp $ROWS $S/rows.jsonl  # rendered once for several runs (the pool's render: ~10 s a text)
else ~/.local/bin/mem-lease 8 $PY $SRC/method/oracle/components.py $RR $ROWS $S/rows.jsonl; fi  # questions in the current format
MPD_MEM_GIB=8 $PY $SRC/method/oracle/components.py dataset $S/rows.jsonl $TEACHER $S/sft.jsonl \
  ${ENGLISH:+--english $ENGLISH} ${EDIT_TEACH:+--edit-teach $EDIT_TEACH}
~/.local/bin/mem-lease 8 $PY $SRC/method/oracle/components.py $RR $HELDOUT $S/heldout.jsonl
$PY - $SRC/method/oracle $S/heldout.jsonl $HELDOUT_TEACHER $S/teacher_txt ${EDIT_EVAL:-} <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import components
out = Path(sys.argv[4]); out.mkdir(exist_ok=True)
edits = components.edit_teach([Path(sys.argv[5])]) if len(sys.argv) > 5 else None  # the edit teacher answers the edit requests
for r in map(json.loads, open(sys.argv[2])):
    f = Path(sys.argv[3]) / (r["task"] + ".json")
    if f.exists():
        (out / (r["task"] + ".txt")).write_text(components.target(json.loads(f.read_text()), r, "vpd4l", components.teacher_edits(edits, r)))
PYEOF
~/.local/bin/mem-lease 12 $PY $SRC/method/oracle/components.py references $S/heldout.jsonl $HELDOUT_TEACHER $S/ref ${EDIT_EVAL:+--edit-teach $EDIT_EVAL}
export TINKER_API_KEY="$(security find-generic-password -s tinker-api-key -w)" TINKER_PROJECT_ID=3f598f31-2ffa-4eb4-b302-909eb234fb7d
cd $SRC/method/oracle/rl
$TPY -u tinker_train.py sft $S/sft.jsonl --log $S/sft --base $BASE --steps $STEPS --batch $BATCH ${LR:+--lr $LR} --name sft-$RUN \
  --max-length ${MAXLEN:-32768}  # the empty think block in the prompt (tinker_train sft), thinking or not
STATE=$($PY -c "import json; print(json.load(open('$S/sft/state.json'))['state'])")
$TPY -u tinker_train.py sample $S/heldout.jsonl $S/samples.jsonl --init "$STATE" --base $BASE --n $NEVAL --samples $SAMPLES --max-tokens $MAXTOK $TH
# the circuit-ablation column: the same policy answering the questions with an empty circuit (E2 with - without)
$TPY -u tinker_train.py sample $S/heldout.jsonl $S/samples_nocircuit.jsonl --init "$STATE" --base $BASE --n $NEVAL --samples 1 \
  --max-tokens $MAXTOK $TH --prefix $'CIRCUIT:\nANSWER 1:\n'
Z=${ZEROSHOT:-$O/zeroshot_heldout20/cst}  # the untrained Qwen3-8B's answers to these questions, scored by the same reward
E1S="" E1Z=""
if [ -n "${ENGLISH:-}" ]; then
  $TPY -u tinker_train.py readers $S/samples.jsonl $S/heldout.jsonl $S/e1.jsonl
  E1S=":$S/e1.jsonl" E1Z=":$Z/e1.jsonl"
fi
unset TINKER_API_KEY
[ -n "${SKIP_EVAL:-}" ] && exit 0  # SKIP_EVAL: the samples only (several arms evaluated together, one revision)
cd $SRC
$PY -c "import json; open('$S/zeroshot_first.jsonl', 'w').writelines(l for l in open('$Z/samples.jsonl') if json.loads(l)['sample'] == 0)"
PRUNED=() REFS=(--ref "Teacher=$S/eval.json:teacher" --ref "VPD, every connection=$S/eval.json:vpd_complete") XSETS=()
for x in ${=EXTRA:-}; do XSETS+=(--set "${x%%=*}=$S/eval.json:${x%%=*}"); done
if [ -d $S/ref/vpd_pruned ]; then
  PRUNED=(vpd_pruned=$S/ref/vpd_pruned) REFS+=(--ref "VPD, pruned connections=$S/eval.json:vpd_pruned")
fi
# C's families are too slow and too large for the Mac: the evaluation runs on a rented GPU (eval_pod.sh), with the held-out
# teachers as reward v3's teacher-dominance reference
EDIT_EVAL=${EDIT_EVAL:-} TEACHER=$HELDOUT_TEACHER bash experiments/oracle/pods/eval_pod.sh sft-$RUN $S/heldout.jsonl $LABELS sft=$S/samples.jsonl$E1S sft_nocircuit=$S/samples_nocircuit.jsonl \
  zeroshot=$S/zeroshot_first.jsonl$E1Z teacher=$S/teacher_txt vpd_complete=$S/ref/vpd_complete "${PRUNED[@]}" ${=EXTRA:-} > $S/eval_pod.log 2>&1
cp ~/mpd-data/runpod/oracle-graph-eval-sft-$RUN/eval.json $S/eval.json
cp ~/mpd-data/runpod/oracle-graph-eval-sft-$RUN/eval.log $S/eval.log
tail -60 $S/eval.log
MPD_MEM_GIB=1 $PY $SRC/experiments/oracle/figures_eval.py --labels $LABELS --prefix sft_$RUN \
  --set "Qwen3-8B zero-shot=$S/eval.json:zeroshot$E1Z" "${XSETS[@]}" --set "${BASE#*/} SFT=$S/eval.json:sft$E1S" "${REFS[@]}"
