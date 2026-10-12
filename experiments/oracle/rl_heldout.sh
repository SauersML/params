#!/bin/zsh
# The held-out evaluation of an RL run's checkpoints (#2951 graph oracle): for each state in DIR/checkpoints.jsonl
# (tinker_train.py rl) without one yet, its answers to held-out text9000-9019 (fresh QA seed), E1 by readers A and B,
# and components.py evaluate, into DIR/heldout_<step>.json (answer set "rl") and DIR/heldout_<step>_e1.jsonl, the
# files experiments/oracle/figures_eval.py --rl reads. Run it while the RL runs or after; it skips what is done.
#   zsh experiments/oracle/rl_heldout.sh DIR   (env: HELDOUT, LABELS, NEVAL, SAMPLES, MAXTOK, EDIT_TEACH, SRC, RLNAME)
# Then the RL figure (figures_eval.py --rl) as ~/mpd-data/graph_oracle/figures/rl_<DIR's name>_rl.png.
set -e
DIR=$1
O=~/mpd-data/graph_oracle/components
HELDOUT=${HELDOUT:-$O/zeroshot_heldout20/fresh20_rt.jsonl} LABELS=${LABELS:-$O/heldout50_locality_frozen.jsonl}  # frozen once: never relabel per teacher version
NEVAL=${NEVAL:-20} SAMPLES=${SAMPLES:-2} MAXTOK=${MAXTOK:-4096}
SRC=${SRC:-/Users/user/params}
PY=/Users/user/mpd-data/venv/bin/python; TPY=/Users/user/mpd-data/tinker-venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_MPS_PREFER_METAL=1
[ -f $DIR/heldout.jsonl ] || $PY $SRC/method/oracle/components.py rerender $HELDOUT $DIR/heldout.jsonl
export TINKER_API_KEY="$(security find-generic-password -s tinker-api-key -w)" TINKER_PROJECT_ID=3f598f31-2ffa-4eb4-b302-909eb234fb7d
$PY -c "
import json
for l in open('$DIR/checkpoints.jsonl'):
    r = json.loads(l); print(r['step'], r['state'], r['base'])
" | while read -r STEP STATE BASE; do
  [ -f $DIR/heldout_$STEP.json ] && continue
  (cd $SRC/method/oracle/rl &&
   $TPY -u tinker_train.py sample $DIR/heldout.jsonl $DIR/heldout_${STEP}_samples.jsonl --init "$STATE" --base $BASE \
     --n $NEVAL --samples $SAMPLES --max-tokens $MAXTOK &&
   $TPY -u tinker_train.py readers $DIR/heldout_${STEP}_samples.jsonl $DIR/heldout.jsonl $DIR/heldout_${STEP}_e1.jsonl) < /dev/null  # the loop reads the checkpoints from stdin
  P=rl-$(basename $DIR)-$STEP  # C's families run on a rented GPU (eval_pod.sh), too slow and too large for the Mac
  (cd $SRC && bash experiments/oracle/pods/eval_pod.sh $P $DIR/heldout.jsonl $LABELS \
     rl=$DIR/heldout_${STEP}_samples.jsonl:$DIR/heldout_${STEP}_e1.jsonl > $DIR/heldout_${STEP}_pod.log 2>&1) < /dev/null
  cp ~/mpd-data/runpod/oracle-graph-eval-$P/eval.json $DIR/heldout_$STEP.json
  cp ~/mpd-data/runpod/oracle-graph-eval-$P/eval.log $DIR/heldout_$STEP.log
  echo "step $STEP: $DIR/heldout_$STEP.json"
done
unset TINKER_API_KEY
MPD_MEM_GIB=1 $PY $SRC/experiments/oracle/figures_eval.py --labels $LABELS --rl "${RLNAME:-$(basename $DIR)}=$DIR" \
  --prefix rl_$(basename $DIR)
