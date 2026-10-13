#!/bin/zsh
# RL environment F, the tool-using oracle (the lead's 10-13), on Tinker: the policy calls tools_env's measurement tools
# (removals' family scores and shortfall, a candidate circuit's score, a position's influence) within MAX_TURNS turns
# and MAX_CALLS calls, then writes its circuit. Reward: components.reward (the circuit environment's R) of the answer
# minus COST (0.005, tools_env.COST_F, fixed before the gate) per call. tools_env.py serve is the environment, already up
# (tools_server_pod.sh with SPLIT=auto COST=0.005 and its tunnel; SERVER its URL). tools_tinker.py rl trains from INIT
# on the training texts TASKS with a state every SAVE_EVERY steps; each state (and INIT, as step 0) is scored on the dev
# texts DEV by tools_tinker.py rollout --force (an episode out of budget answers in one forced turn): LOG/dev_<step>.jsonl
# and LOG/devscore_<step>.json (mean reward, mean reward before the calls' cost, mean calls, answered share, and each
# episode's [calls, reward]: R against the number of calls). After each: stop at no new dev best for PATIENCE steps, or
# the run's Tinker dollars (training and dev) past USD. The server must name REWARD_REV as its source (a prefix either
# way), semantics n, score split auto and cost COST, and every training score is checked against them.
#   cat ~/mpd-data/graph_oracle/components/rl_rows/train444_mpool_circuit.jsonl \
#       ~/mpd-data/graph_oracle/components/dev30/dev30_mpool_circuit.jsonl > ROWS          (text6000-6597, text9020-9049)
#   SPLIT=auto COST=0.005 RP_REF=<rev> bash experiments/oracle/pods/tools_server_pod.sh envf ROWS 6
#   bash experiments/oracle/pods/tools_server_pod.sh tunnel envf 8771 &
#   SERVER=http://127.0.0.1:8771 TASKS=text6000:text6597 DEV=text9020:text9049 INIT=<tinker state> REWARD_REV=<rev>
#     [NAME=env_f] [USD=10] [STEPS=30] [TEXTS=4] [GROUP=8] [SAVE_EVERY=5] [PATIENCE=10] [THINK=1] [MAX_TURNS=24]
#     [MAX_TOKENS=4096] [DEV_USD=2] zsh method/oracle/rl/env_f_rl.sh
# The Tinker key is read from the Keychain at launch and never written anywhere.
set -e
: ${SERVER:?} ${TASKS:?} ${DEV:?} ${REWARD_REV:?}
NAME=${NAME:-env_f} COST=${COST:-0.005} SEM=${EXPECT_SEMANTICS:-n}
LOG=~/mpd-data/graph_oracle/components/rl_$NAME; mkdir -p $LOG
TPY=${TPY:-$HOME/mpd-data/tinker-venv/bin/python}
cd /Users/user/params && git fetch -q && SRCREV=$(git rev-parse --short=10 origin/main)
SRC=~/mpd-data/scratch/toolenv/rlsrc_$NAME; rm -rf $SRC && mkdir -p $SRC && git archive $SRCREV method/oracle method/vpd | tar -x -C $SRC
curl -s -m 20 $SERVER/health > $LOG/health.json || { echo "no tools_env server at $SERVER" >&2; exit 1; }
python3 - $LOG/health.json $REWARD_REV $SEM $COST <<'EOF'
import json, sys
h, rev, sem, cost = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], float(sys.argv[4])
src = str(h.get("source") or "")
why = [] if len(src) >= 7 and (src.startswith(rev) or rev.startswith(src)) else [f"source {src!r}, not {rev}"]
why += [f"semantics {h.get('semantics')!r}, not {sem!r}"] if h.get("semantics") != sem else []
why += [f"score split {h.get('score_split')!r}, not 'auto'"] if h.get("score_split") != "auto" else []
why += [f"cost {h.get('cost')!r} a call, not {cost}"] if h.get("cost") != cost else []
sys.exit(f"the server is not env F's frozen reward: {'; '.join(why)}" if why else 0)
EOF
echo "env F: init ${INIT:-the base model}, reward $REWARD_REV at cost $COST a call, trainer $SRCREV, server $SERVER, ${TEXTS:-4} x ${GROUP:-8}, dev $DEV" | tee $LOG/SOURCE
export TINKER_API_KEY="$(security find-generic-password -s tinker-api-key -w)" TINKER_PROJECT_ID=3f598f31-2ffa-4eb4-b302-909eb234fb7d
cd $SRC/method/oracle
X=(--server $SERVER --max-turns ${MAX_TURNS:-24} --max-tokens ${MAX_TOKENS:-4096}); [ -n "${THINK:-}" ] && X+=(--thinking)

devscore() {  # every state not yet scored on DEV: its episodes, then their summary
  local step state
  { echo "0 ${INIT:-}"; [ -s $LOG/checkpoints.jsonl ] && python3 -c "import json, sys
for l in open(sys.argv[1]): d = json.loads(l); print(d['step'], d['state'])" $LOG/checkpoints.jsonl; } | while read -r step state; do
    [ -s $LOG/devscore_$step.json ] && continue
    rm -f $LOG/dev_$step.jsonl $LOG/dev_$step.spend.jsonl
    $TPY -u tools_tinker.py rollout ${(z)DEV} $X --force --out $LOG/dev_$step.jsonl --usd-cap ${DEV_USD:-2} \
      ${state:+--init $state} < /dev/null >> $LOG/devscore.log 2>&1 || true
    [ -s $LOG/dev_$step.jsonl ] || continue  # no episodes yet: tried again after the next minute
    python3 - $LOG/dev_$step.jsonl $step > $LOG/devscore_$step.json <<'EOF'
import json, sys
rows = [r for r in (json.loads(l) for l in open(sys.argv[1])) if "error" not in r]
mean = lambda xs: sum(xs) / len(xs) if xs else None
print(json.dumps({"step": int(sys.argv[2]), "n": len(rows), "reward": mean([r["reward"] for r in rows]),
                  "answer_reward": mean([r["answer_reward"] for r in rows if r.get("answer_reward") is not None]),
                  "calls": mean([r["calls"] or 0 for r in rows]), "valid": mean([float(bool(r.get("answered"))) for r in rows]),
                  "by_calls": [[r["task"], r["calls"], r["reward"]] for r in rows]}))
EOF
  done
}

stoprule() {  # {"stop", "why", "best", "steps", "usd"}: no new dev best for PATIENCE steps, or training and dev past USD
  python3 - $LOG ${PATIENCE:-10} ${USD:-10} > $LOG/stop.json <<'EOF'
import json, sys
from pathlib import Path
log, patience, usd = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
rows = sorted((d["step"], d["reward"], d["calls"]) for d in (json.loads(p.read_text()) for p in log.glob("devscore_*.json"))
              if d.get("reward") is not None)
last = lambda p: json.loads(p.read_text().splitlines()[-1]).get("total_usd", 0.0) if p.exists() and p.stat().st_size else 0.0
spent = last(log / "spend.jsonl") + sum(last(p) for p in log.glob("dev_*.spend.jsonl"))
out = {"stop": False, "why": None, "best": None, "steps": rows, "usd": round(spent, 3)}
if rows:
    best = max(rows, key=lambda r: r[1])
    out["best"] = best[0]
    if rows[-1][0] - best[0] >= patience:
        out |= {"stop": True, "why": f"dev reward has no new best since step {best[0]} ({best[1]:+.3f})"}
if spent > usd:
    out |= {"stop": True, "why": f"the run's Tinker spend {spent:.2f} passed {usd}"}
print(json.dumps(out))
EOF
}

$TPY -u tools_tinker.py rl ${(z)TASKS} $X --log $LOG ${INIT:+--init $INIT} --steps ${STEPS:-30} --texts ${TEXTS:-4} \
  --group ${GROUP:-8} --save-every ${SAVE_EVERY:-5} --usd-cap ${USD:-10} --expect-source $REWARD_REV --expect-semantics $SEM \
  > $LOG/rl.log 2>&1 &
TRAINER=$!
while kill -0 $TRAINER 2>/dev/null; do
  sleep 60
  devscore
  stoprule
  if python3 -c "import json, sys; sys.exit(0 if json.load(open('$LOG/stop.json'))['stop'] else 1)"; then
    kill $TRAINER 2>/dev/null || true; echo "stopped: $(python3 -c "import json; print(json.load(open('$LOG/stop.json'))['why'])") $(date)" >> $LOG/rl.log
  fi
done
devscore
stoprule
cat $LOG/stop.json
