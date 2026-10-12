#!/bin/zsh
# The component explainer's SFT and its held-out evaluation (#2951 graph oracle): SFT rows from question rows and
# teacher circuits (with their English and the edit teacher's edits when given), one Tinker SFT on BASE, samples on
# held-out text9000-9019 (fresh QA seed), E1 by reader A and reader B, and the evaluation split by locality against
# the held-out teachers themselves.
#   zsh experiments/oracle/sft.sh RUN   (env: ROWS, TEACHER, ENGLISH, EDIT_TEACH, BASE, STEPS, BATCH, LR, HELDOUT,
#                                        HELDOUT_TEACHER, LABELS, NEVAL, SAMPLES, MAXTOK, SRC)
set -e
RUN=$1
O=~/mpd-data/graph_oracle/components; S=$O/sft_$RUN; mkdir -p $S
ROWS=${ROWS:?question rows: components.py prepare output, several joined}
TEACHER=${TEACHER:?teacher circuits (<task>.json)}
BASE=${BASE:-Qwen/Qwen3-8B} STEPS=${STEPS:-100} BATCH=${BATCH:-8} NEVAL=${NEVAL:-20} SAMPLES=${SAMPLES:-4} MAXTOK=${MAXTOK:-4096}
HELDOUT=${HELDOUT:-$O/zeroshot_heldout20/fresh20.jsonl} HELDOUT_TEACHER=${HELDOUT_TEACHER:-~/mpd-data/graph_oracle/teacher/joint_heldout}
LABELS=${LABELS:-$O/heldout20_locality_teacher.jsonl}
SRC=${SRC:-/Users/user/params}  # a source tree (a clean copy of main while others edit the shared tree)
PY=/Users/user/mpd-data/venv/bin/python; TPY=/Users/user/mpd-data/tinker-venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_MPS_PREFER_METAL=1
$PY $SRC/method/oracle/components.py rerender $ROWS $S/rows.jsonl  # questions in the current format
MPD_MEM_GIB=8 $PY $SRC/method/oracle/components.py dataset $S/rows.jsonl $TEACHER $S/sft.jsonl \
  ${ENGLISH:+--english $ENGLISH} ${EDIT_TEACH:+--edit-teach $EDIT_TEACH}
$PY $SRC/method/oracle/components.py rerender $HELDOUT $S/heldout.jsonl
$PY - $SRC/method/oracle $S/heldout.jsonl $HELDOUT_TEACHER $S/teacher_txt <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import components
out = Path(sys.argv[4]); out.mkdir(exist_ok=True)
for r in map(json.loads, open(sys.argv[2])):
    f = Path(sys.argv[3]) / (r["task"] + ".json")
    if f.exists():
        (out / (r["task"] + ".txt")).write_text(components.target(json.loads(f.read_text()), r, "vpd4l"))
PYEOF
~/.local/bin/mem-lease 12 $PY $SRC/method/oracle/components.py references $S/heldout.jsonl $HELDOUT_TEACHER $S/ref
export TINKER_API_KEY="$(security find-generic-password -s tinker-api-key -w)" TINKER_PROJECT_ID=3f598f31-2ffa-4eb4-b302-909eb234fb7d
cd $SRC/method/oracle/rl
$TPY -u tinker_train.py sft $S/sft.jsonl --log $S/sft --base $BASE --steps $STEPS --batch $BATCH ${LR:+--lr $LR} --name sft-$RUN
STATE=$($PY -c "import json; print(json.load(open('$S/sft/state.json'))['state'])")
$TPY -u tinker_train.py sample $S/heldout.jsonl $S/samples.jsonl --init "$STATE" --base $BASE --n $NEVAL --samples $SAMPLES --max-tokens $MAXTOK
$TPY -u tinker_train.py readers $S/samples.jsonl $S/heldout.jsonl $S/e1.jsonl --reader-b Qwen/Qwen3.5-4B
unset TINKER_API_KEY
cd $SRC
Z=$O/zeroshot_heldout20  # the untrained Qwen3-8B's first answers, scored by the same reward as everything here
PRUNED=() REFS=(--ref "Teacher=$S/eval.json:teacher" --ref "VPD, every connection=$S/eval.json:vpd_complete")
if [ -d $S/ref/vpd_pruned ]; then
  PRUNED=(--set vpd_pruned=$S/ref/vpd_pruned) REFS+=(--ref "VPD, pruned connections=$S/eval.json:vpd_pruned")
fi
~/.local/bin/mem-lease 14 $PY -u method/oracle/components.py evaluate heldout $S/heldout.jsonl --set sft=$S/samples.jsonl \
  --e1 sft=$S/e1.jsonl --set zeroshot=$Z/samples_first.jsonl --e1 zeroshot=$Z/e1.jsonl --set teacher=$S/teacher_txt \
  --set vpd_complete=$S/ref/vpd_complete "${PRUNED[@]}" --labels $LABELS --n $NEVAL ${EDIT_TEACH:+--edit-teach $EDIT_TEACH} \
  --out $S/eval.json > $S/eval.log 2>&1
tail -60 $S/eval.log
MPD_MEM_GIB=1 $PY $SRC/experiments/oracle/figures_eval.py --labels $LABELS --prefix sft_$RUN \
  --set "Qwen3-8B zero-shot=$S/eval.json:zeroshot:$Z/e1.jsonl" --set "${BASE#*/} SFT=$S/eval.json:sft:$S/e1.jsonl" "${REFS[@]}"
