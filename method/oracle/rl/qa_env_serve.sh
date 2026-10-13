#!/bin/bash
# RL environment B's reward (qa_env.py serve) on this machine, CPU only, from a frozen copy of method/oracle at REV
# (its revision on every score as "source": tinker_train rl --expect-source REV). The outcomes were measured once on a
# GPU (qa_env.py measure, then rows) and are kept in the rows; the train and dev rows are both served (tinker_train
# devscore posts the dev ones).
#   bash method/oracle/rl/qa_env_serve.sh REV TRAIN.jsonl DEV.jsonl [PORT]    (default 8773)
#   then tinker_train.py rl TRAIN.jsonl --scorer http://127.0.0.1:PORT/score --no-lines --expect-semantics qa1 --expect-source REV
set -euo pipefail
REV=$1 TRAIN=$2 DEV=$3 PORT=${4:-8773}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SHORT=$(git -C "$REPO" rev-parse --short=10 "$REV")
S=$HOME/mpd-data/scratch/qa_env_serve/$SHORT
rm -rf "$S" && mkdir -p "$S" && git -C "$REPO" archive "$SHORT" method/oracle method/vpd | tar -x -C "$S"
echo "python source $SHORT" > "$S/SOURCE"
exec /usr/bin/python3 -u "$S/method/oracle/qa_env.py" serve "$TRAIN" "$DEV" --port "$PORT"
