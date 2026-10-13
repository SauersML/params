#!/usr/bin/env bash
# self_stop.sh MINUTES: stop this RunPod pod MINUTES from now (keeping its /workspace volume) unless the Mac reaps it first.
# The pod's id and key come from its init process's environment: an SSH session, where rp-run runs commands, has neither.
sleep $(( $1 * 60 ))
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
