#!/usr/bin/env bash
# Execute rp-run's actual run_command with local transport and a persistently live PID.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
launcher=${1:-$here/rp-run}
export TEST_TAIL=$(command -v gtail || command -v tail)
eval "$(sed -n '/^run_command() {/,/^}/p' "$launcher")"
declare -F run_command >/dev/null
sandbox=$(mktemp -d)
trap 'jobs -p | xargs kill 2>/dev/null || true; rm -rf "$sandbox"' EXIT
say() { :; }
sync_loop() { :; }
idle_check() { :; }
live_profile() { :; }
pod_state() { echo RUNNING; }
# Accelerate only polling, retaining GNU tail and timeout semantics.
timeout() { shift; shift; command timeout .15s "$TEST_TAIL" -s .02 "$@"; }
tail() { "${TEST_TAIL:-tail}" -s .02 "$@"; }
on_pod() {
    case "$1" in
        *'setsid nohup bash'*)
            echo launch >> "$OUT/launches"
            sleep 20 & supervisor=$!
            echo "$supervisor" > "$WS/run.pid"
            : > "$WS/run.log"
            (
                printf 'first\n' >> "$WS/run.log"
                sleep .25
                printf 'second\n' >> "$WS/run.log"
                sleep .25
                touch "$WS/ready-to-finish"
            ) &
            ;;
        *'cat '*/run.exit*)
            echo poll >> "$OUT/polls"
            if [ -f "$WS/ready-to-finish" ] && [ ! -f "$WS/run.exit" ]; then
                # Deliberately publish final bytes only AFTER tail has timed out.
                printf 'last-without-newline' >> "$WS/run.log"
                printf '%s\n' "$expected_rc" > "$WS/run.exit"
            fi
            bash -c "$1"
            ;;
        *'tail -c '*)
            if [[ $1 != *' -f '* ]] && [ "$scenario" = drainfail ] && [ ! -f "$OUT/drain-failed" ]; then
                touch "$OUT/drain-failed"
                return 1
            fi
            bash -c "$(declare -f timeout tail); $1"
            ;;
        *) bash -c "$(declare -f timeout tail); $1" ;;
    esac
}
for scenario in 0 17 drainfail; do
    expected_rc=$scenario; [ "$scenario" != drainfail ] || expected_rc=0
    OUT=$sandbox/case$scenario; WS=$OUT/remote; TMP=$OUT/tmp
    mkdir -p "$WS/src" "$TMP"
    LOG=$OUT/local.log; : > "$LOG"
    PYENV_DIR=''; RP_HF_MODELS=''; cmd='echo must-not-be-relaunched'; NAME=test
    RUN_STARTED=0; FIRST_START=''; SYNC=''; POD=mock
    run_command
    [ "$RC" = "$expected_rc" ]
    [ "$(wc -l < "$OUT/launches" | tr -d ' ')" = 1 ]
    # Multiple receipt polls while running prove timeout isn't command completion/retry.
    [ "$(wc -l < "$OUT/polls" | tr -d ' ')" -ge 3 ]
    cmp "$LOG" "$WS/run.log"
    printf 'first\nsecond\nlast-without-newline' > "$OUT/expected"
    cmp "$LOG" "$OUT/expected"
    [ "$scenario" != drainfail ] || [ -f "$OUT/drain-failed" ]
    kill -0 "$supervisor" # terminal receipt observed while PID is still visible
    kill "$supervisor"; wait "$supervisor" 2>/dev/null || true
    printf 'PASS: actual run_command preserves exit %s, exact bytes, one launch, and live supervisor\n' "$expected_rc"
done
