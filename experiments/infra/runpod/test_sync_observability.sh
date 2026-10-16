#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
launcher=${1:-$here/rp-run}
eval "$(sed -n '/^sync_loop() {/,/^}/p' "$launcher")"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
sleep() { :; }
say() { echo "$*" >> "$TMP/messages"; }
sync_outputs() { touch "$TMP/stop-sync"; return 7; }
sync_loop rest
[ "$(cat "$TMP/messages")" = 'output sync failed (rest, status 7); will retry' ]
echo 'PASS: actual sync_loop logs failed mode/status and honors stop'
