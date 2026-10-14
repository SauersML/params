#!/usr/bin/env bash
# The SSH tunnel to a scorer pod (scorer_pod.sh): localhost:8767 on the Mac -> the pod's serve_components.py, with
# the run's own key from rp-run's record of the pod (a run that is not a dev pod leaves no ssh helper in its
# directory). Runs until killed or the pod ends.
#   bash experiments/oracle/pods/scorer_tunnel.sh RUN [PORT]   then tinker_train.py rl --scorer http://127.0.0.1:PORT/score
#   env D's scorer (envd_reader_pod.sh serve): NAME=oracle-graph-envd-serve-RUN REMOTE_PORT=8772
RUN=${1:?RUN} PORT=${2:-8767}
POD=$(cut -d' ' -f1 "/Users/user/mpd-data/runpod/${NAME:-oracle-graph-scorer-$RUN}/pod.txt") || exit 1
read -r IP SSH_PORT KEYS SSH_USER < /Users/user/mpd-data/runpod/active/.$POD.ssh || { echo "no record of pod $POD" >&2; exit 1; }
exec ssh -F none -i "$KEYS/key" -o IdentitiesOnly=yes -o ForwardAgent=no -o UserKnownHostsFile="$KEYS/known_hosts" \
  -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -p "$SSH_PORT" \
  -N -L "$PORT:localhost:${REMOTE_PORT:-8767}" "$SSH_USER@$IP"
