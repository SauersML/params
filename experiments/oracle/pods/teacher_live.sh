#!/usr/bin/env bash
# Refresh a live teacher directory for queued MATS jobs (teacher_mats.sh LIVE=DIR): DIR/src gets method/oracle and
# method/vpd at COMMIT, DIR/flags.txt the teacher flags (HOLD: a job that starts ends at once), and the directory goes up to
# the cluster (mats-push), so every job that starts after this runs that code with those flags.
#   experiments/oracle/pods/teacher_live.sh /Users/user/mpd-data/cluster/v2live COMMIT "--f1-start influential --edges ..." | HOLD
set -euo pipefail
cd /Users/user/params
DIR=$1 C=$2 FLAGS=$3
case $DIR in /Users/user/mpd-data/cluster/*) ;; *) echo "DIR must be under /Users/user/mpd-data/cluster"; exit 2 ;; esac
git rev-parse --verify "$C^{commit}" > /dev/null
rm -rf "$DIR/src.new" && mkdir -p "$DIR/src.new"
git archive "$C" method/oracle method/vpd | tar -x -C "$DIR/src.new"
rm -rf "$DIR/src" && mv "$DIR/src.new" "$DIR/src"
printf '%s\n' "$FLAGS" > "$DIR/flags.txt"
git rev-parse "$C" > "$DIR/commit.txt"
experiments/infra/mats/mats-push "$DIR"
echo "live $DIR: $(cut -c1-10 "$DIR/commit.txt") flags: $FLAGS"
