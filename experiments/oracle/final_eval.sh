#!/bin/zsh
# The pre-registered final evaluation (#2951 graph oracle; the lead's 10-12): the trained oracle on the sealed texts
# text9100-9149, run once, at the end, after the recipe is frozen. Nothing touches these texts before it: no SFT,
# sampling, evaluation or inspection. The script refuses to start unless FINAL says the recipe is frozen, and it leaves a
# stamp, so it runs once.
#
# Fixed in advance, before any RL:
# - Texts: text9100-9149, all 50, each text's FIRST sample (one sample per text), invalid answers counted at reward v3's
#   V3_INVALID. The limits are reward v3's: 266 nodes, 8,000 answer tokens after the thinking, 16,384 generated tokens
#   with the thinking. The fresh question seed (qa_seed 1) and its structural questions are used, as on the dev texts.
# - Columns (components.py evaluate's report): phi against phi_tau, compact, s_track_pp, the reward, the valid share;
#   E4 (variables.Scorer, v2 once varteach lands it), E1's gain over no explanation by the reward reader and by Kimi
#   (tinker_train.py readers); generality's falsification check (the oracle's circuit against plausible wrong
#   variants of it: beats every variant on R by more than the text's family eps).
# - Baselines: the influential start (evaluate's "start"), recency and VPD top k at each answer's own bits
#   (matched_k), native units at equal bits (native_units.py phi), and the teacher (teacher.py on these texts with the
#   frozen recipe's flags, TEACHER_ARGS), which is also reward v3's D reference.
# - Controls: the same policy with no circuit (sample --prefix), with shuffled evidence, with the text only, and with
#   randomized subcomponent IDs (components.py rerender --control).
# - Amortization: verifier calls and GPU-seconds for the teacher (its summary's seconds) against the oracle's sampling
#   and one scoring of its answers.
# The transfer table (does training on one objective change the others?) is a dev-text comparison of the runs, not
# part of this evaluation.
#
#   FINAL="the recipe is frozen" POLICY=tinker://... SRCREV=<frozen commit> [POOL=1 MEASURED=1 THINKING=1] \
#     TEACHER_ARGS="..." zsh experiments/oracle/final_eval.sh
set -e
[ "${FINAL:-}" = "the recipe is frozen" ] || { echo "final_eval.sh: the sealed texts run once, after the recipe is frozen (FINAL)" >&2; exit 2; }
export FINAL_EVAL=1   # rp-run's guard keeps the sealed texts off every pod except this run's
: ${POLICY:?the oracle checkpoint (tinker://)} ${SRCREV:?the frozen source revision}
F=/Users/user/mpd-data/graph_oracle/sealed
[ ! -e $F/STAMP ] || { echo "final_eval.sh: already run ($(cat $F/STAMP))" >&2; exit 2; }
mkdir -p $F && echo "$(date) $POLICY $SRCREV" > $F/STAMP
O=/Users/user/mpd-data/graph_oracle/components
PY=/Users/user/mpd-data/venv/bin/python; TPY=/Users/user/mpd-data/tinker-venv/bin/python
BASE=${BASE:-Qwen/Qwen3-8B}
SRC=$F/src && rm -rf $SRC && mkdir -p $SRC && git -C /Users/user/params archive $SRCREV method/oracle method/vpd experiments | tar -x -C $SRC
export HF_HUB_OFFLINE=1 PYTORCH_MPS_PREFER_METAL=1
P=(env HF_HUB_OFFLINE=1 PYTORCH_MPS_PREFER_METAL=1 $PY $SRC/method/oracle/components.py)
# 1. Questions: every sealed text kept (the locality balance is for training sets), evidence on rented GPUs, then the
#    routed lens's facts, the fresh questions with the structural kinds, and the policy's input format and controls.
$PY -c "print(''.join('{\"task\": \"text%d\", \"local\": false}\n' % t for t in range(9100, 9150)), end='')" > $F/tasks.jsonl
(cd $SRC && RUN=sealed TASKS=$F/tasks.jsonl PODS=1 P=3 bash experiments/oracle/pods/prepare_pod.sh 2.0) > $F/prepare.log 2>&1
cat $O/train_prep_pods/sealed-0_0/rows.jsonl $O/train_prep_pods/sealed-0_1/rows.jsonl $O/train_prep_pods/sealed-0_2/rows.jsonl > $F/rows.jsonl
~/.local/bin/mem-lease 8 $P rerender --facts $F/rows.jsonl $F/rows_rt.jsonl
~/.local/bin/mem-lease 8 $P requestion --qa-seed 1 $F/rows_rt.jsonl $F/rows_fresh.jsonl
~/.local/bin/mem-lease 8 $P structure --qa-seed 1 $F/rows_fresh.jsonl $F/rows_rt_st.jsonl
~/.local/bin/mem-lease 8 $P locality heldout $F/locality.jsonl --tasks <($PY -c "print(*['text%d' % t for t in range(9100, 9150)])")
RR=(${MEASURED:+--measured} ${POOL:+--pool})
~/.local/bin/mem-lease 8 $P rerender $RR $F/rows_rt_st.jsonl $F/questions.jsonl
for k in shuffled text-only random-ids; do ~/.local/bin/mem-lease 8 $P rerender --control $k $F/questions.jsonl $F/questions_$k.jsonl; done
# 2. The oracle's first sample per text, the controls' samples and E1's readers.
export TINKER_API_KEY="$(security find-generic-password -s tinker-api-key -w)" TINKER_PROJECT_ID=3f598f31-2ffa-4eb4-b302-909eb234fb7d
TH=(${THINKING:+--thinking}); MAXTOK=$([ -n "${THINKING:-}" ] && echo 16384 || echo 8192)
cd $SRC/method/oracle/rl
S=(sample --init "$POLICY" --base $BASE --n 50 --samples 1 --max-tokens $MAXTOK $TH)
$TPY -u tinker_train.py $S $F/questions.jsonl $F/oracle.jsonl
$TPY -u tinker_train.py $S $F/questions.jsonl $F/oracle_nocircuit.jsonl --prefix $'CIRCUIT:\nANSWER 1:\n'
for k in shuffled text-only random-ids; do $TPY -u tinker_train.py $S $F/questions_$k.jsonl $F/oracle_$k.jsonl; done
$TPY -u tinker_train.py readers $F/oracle.jsonl $F/questions.jsonl $F/oracle_e1.jsonl
unset TINKER_API_KEY
cd $SRC
# 3. The teacher on the sealed texts with the frozen recipe's flags (its circuits: a baseline and D's reference).
RUN=sealed TASKS=$F/tasks.jsonl DEST=$F/teacher PODS=2 REF=$SRCREV TEACHER_ARGS="${TEACHER_ARGS:-}" EXTRA=goal_eval/ci_patterns_train.pt \
  bash experiments/oracle/pods/teacher_pod.sh 4.0 > $F/teacher.log 2>&1
# 4. The evaluation: every column, baseline and control above, on one pinned revision (sealed texts included).
EXCLUDE_TEXTS=no-text-excluded SRCREV=$SRCREV LEDGER_OK=1 TEACHER=$F/teacher VARIABLES=1 HOURS=6 bash experiments/oracle/pods/eval_pod.sh sealed $F/rows_rt_st.jsonl \
  $F/locality.jsonl oracle=$F/oracle.jsonl:$F/oracle_e1.jsonl oracle_nocircuit=$F/oracle_nocircuit.jsonl \
  shuffled=$F/oracle_shuffled.jsonl text_only=$F/oracle_text-only.jsonl random_ids=$F/oracle_random-ids.jsonl \
  teacher=$F/teacher > $F/eval.log 2>&1
cp /Users/user/mpd-data/runpod/oracle-graph-eval-sealed/eval.json $F/eval.json
# 5. Native units at equal bits, generality's falsification of the oracle's circuits, and amortization.
# Both need a GPU (a 480-token text takes more than 16 GiB on the Mac): the same commands on a rented card.
$PY - $SRC/method/oracle $F/oracle.jsonl $F/oracle_circuits <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import components
out = Path(sys.argv[3]); out.mkdir(exist_ok=True)
for r in map(json.loads, open(sys.argv[2])):  # the oracle's answers as circuit JSON, native units' input
    (out / f"{r['task']}.json").write_text(json.dumps(components.parse(r["text"])["answer"]))
PYEOF
~/.local/bin/mem-lease 12 env HF_HUB_OFFLINE=1 $PY $SRC/method/oracle/native_units.py phi --circuits $F/oracle_circuits \
  --texts /Users/user/mpd-data/graph_oracle/texts/vpd4l --out $F/native.jsonl > $F/native.log 2>&1  # native units at the oracle's bits
# falsification: beats every variant on R by more than the text's family eps (the lead's 10-13 rule)
~/.local/bin/mem-lease 12 env HF_HUB_OFFLINE=1 $PY $SRC/experiments/oracle/generality/falsify.py --circuits $F/oracle.jsonl --out $F/falsify.json \
  --on R --eps family > $F/falsify.log 2>&1
$PY - $F <<'PYEOF'
import json, sys
from pathlib import Path
F = Path(sys.argv[1])
teach = [json.loads(x) for p in sorted(F.joinpath("teacher").glob("summary_*.jsonl")) for x in p.read_text().splitlines() if x.strip()]
cost = [json.loads(x) for x in (F / "oracle.jsonl").read_text().splitlines() if x.strip()]
out = {"teacher GPU-seconds per text": sum(r.get("seconds", 0) for r in teach) / max(len(teach), 1),
       "teacher verifier calls per text": sum(r.get("calls", 0) for r in teach) / max(len(teach), 1),
       "oracle sampled tokens per text": sum(len(r.get("tokens", [])) or r.get("n_tokens", 0) for r in cost) / max(len(cost), 1)}
(F / "amortization.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
PYEOF
