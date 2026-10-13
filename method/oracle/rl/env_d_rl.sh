#!/bin/bash
# RL environment D (English for a given circuit) on Tinker: env_d.py serve as the scorer, reward E1 read by the frozen
# reader R (env_d.py's module docstring), and tinker_train.py rl from an SFT state with the sequence-level reward alone.
#   env_d_rl.sh LOG FACTS ROWS READER [INIT_STATE] [STEPS]     (PORT, default 8772; TEXTS, GROUP per step)
# The key is read from the Keychain at launch and never written anywhere.
set -euo pipefail
LOG=$1 FACTS=$2 ROWS=$3 READER=$4 INIT=${5:-} STEPS=${6:-50} PORT=${PORT:-8772}
HERE=$(cd "$(dirname "$0")/.." && pwd)
TPY=${TPY:-$HOME/mpd-data/tinker-venv/bin/python}
mkdir -p "$LOG"
export TINKER_PROJECT_ID=3f598f31-2ffa-4eb4-b302-909eb234fb7d
TINKER_API_KEY="$(security find-generic-password -s tinker-api-key -w)"
export TINKER_API_KEY
"$TPY" "$HERE/env_d.py" questions "$FACTS" "$LOG/questions.jsonl"
"$TPY" -u "$HERE/env_d.py" serve "$FACTS" "$ROWS" --reader "$READER" --port "$PORT" > "$LOG/scorer.log" 2>&1 &
S=$!
trap 'kill $S 2>/dev/null || true' EXIT
until curl -sf "localhost:$PORT/health" > /dev/null; do
  kill -0 $S 2>/dev/null || { tail -20 "$LOG/scorer.log"; exit 1; }
  sleep 2
done
cd "$HERE/rl"
"$TPY" -u tinker_train.py rl "$LOG/questions.jsonl" --log "$LOG" --scorer "http://127.0.0.1:$PORT/score" --no-lines \
  ${INIT:+--init "$INIT"} --steps "$STEPS" --texts "${TEXTS:-4}" --group "${GROUP:-8}" --max-tokens 1024 --name envd
