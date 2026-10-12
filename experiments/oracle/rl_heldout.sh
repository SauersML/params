#!/bin/zsh
# The held-out evaluation of an RL run's checkpoints (#2951 graph oracle): for each state in DIR/checkpoints.jsonl
# (tinker_train.py rl) without one yet, its answers to held-out text9000-9019 (fresh QA seed, the structural kinds),
# E1 by the readers when READERS is set (reported, not in the reward; a policy without English scores 0), and
# components.py evaluate (every checkpoint with the source revision in DIR/SOURCE_REV: C changes on main), into
# DIR/heldout_<step>.json (answer set "rl") and DIR/heldout_<step>_e1.jsonl, the files
# experiments/oracle/figures_eval.py --rl reads. Run it while the RL runs or after; it skips what is done.
#   zsh experiments/oracle/rl_heldout.sh DIR   (env: HELDOUT, HELDOUT_TEACHER (reward v3's teacher term), LABELS, NEVAL, SAMPLES,
#                                               MAXTOK, EDIT_TEACH, SRC, RLNAME, READERS; the policy's input and sampling as
#                                               in its SFT (sft.sh): POOL, MEASURED, THINKING)
# Then the RL figure (figures_eval.py --rl) as ~/mpd-data/graph_oracle/figures/rl_<DIR's name>_rl.png.
set -e
DIR=$1
O=~/mpd-data/graph_oracle/components
HELDOUT=${HELDOUT:-$O/zeroshot_heldout20/fresh20_rt_st.jsonl} LABELS=${LABELS:-$O/heldout50_locality_frozen.jsonl}  # frozen once: never relabel per teacher version
HELDOUT_TEACHER=${HELDOUT_TEACHER:-~/mpd-data/graph_oracle/teacher/joint_heldout}
NEVAL=${NEVAL:-20} SAMPLES=${SAMPLES:-2} MAXTOK=${MAXTOK:-4096}
TH=(${THINKING:+--thinking})
[ -n "${THINKING:-}" ] && MAXTOK=${MAXTOK_THINKING:-16384}  # one generation limit, thinking included (no thinking budget)
SRC=${SRC:-/Users/user/params}
PY=/Users/user/mpd-data/venv/bin/python; TPY=/Users/user/mpd-data/tinker-venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_MPS_PREFER_METAL=1
[ -f $DIR/heldout.jsonl ] || ~/.local/bin/mem-lease 8 $PY $SRC/method/oracle/components.py rerender ${MEASURED:+--measured} ${POOL:+--pool} \
  $HELDOUT $DIR/heldout.jsonl
# one goal_eval for every checkpoint of the run (C changes on main): DIR/SOURCE_REV, written at the first evaluation
[ -f $DIR/SOURCE_REV ] || (cd /Users/user/params && git fetch -q origin main && git rev-parse origin/main > $DIR/SOURCE_REV)
export SRCREV=$(cat $DIR/SOURCE_REV)
export TINKER_API_KEY="$(security find-generic-password -s tinker-api-key -w)" TINKER_PROJECT_ID=3f598f31-2ffa-4eb4-b302-909eb234fb7d
$PY -c "
import json
for l in open('$DIR/checkpoints.jsonl'):
    r = json.loads(l); print(r['step'], r['state'], r['base'])
" | while read -r STEP STATE BASE; do
  [ -f $DIR/heldout_$STEP.json ] && continue
  (cd $SRC/method/oracle/rl &&
   $TPY -u tinker_train.py sample $DIR/heldout.jsonl $DIR/heldout_${STEP}_samples.jsonl --init "$STATE" --base $BASE \
     --n $NEVAL --samples $SAMPLES --max-tokens $MAXTOK $TH &&
   if [ -n "${READERS:-}" ]; then $TPY -u tinker_train.py readers $DIR/heldout_${STEP}_samples.jsonl $DIR/heldout.jsonl $DIR/heldout_${STEP}_e1.jsonl; fi) < /dev/null  # the loop reads the checkpoints from stdin
  E1=""; [ -n "${READERS:-}" ] && E1=":$DIR/heldout_${STEP}_e1.jsonl"
  P=rl-$(basename $DIR)-$STEP  # C's families run on a rented GPU (eval_pod.sh), too slow and too large for the Mac
  (cd $SRC && TEACHER=$HELDOUT_TEACHER bash experiments/oracle/pods/eval_pod.sh $P $DIR/heldout.jsonl $LABELS \
     rl=$DIR/heldout_${STEP}_samples.jsonl$E1 > $DIR/heldout_${STEP}_pod.log 2>&1) < /dev/null
  cp ~/mpd-data/runpod/oracle-graph-eval-$P/eval.json $DIR/heldout_$STEP.json
  cp ~/mpd-data/runpod/oracle-graph-eval-$P/eval.log $DIR/heldout_$STEP.log
  echo "step $STEP: $DIR/heldout_$STEP.json"
done
unset TINKER_API_KEY
MPD_MEM_GIB=1 $PY $SRC/experiments/oracle/figures_eval.py --labels $LABELS --rl "${RLNAME:-$(basename $DIR)}=$DIR" \
  --prefix rl_$(basename $DIR)
