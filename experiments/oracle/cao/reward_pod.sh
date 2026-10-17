#!/usr/bin/env bash
# The oracle's reward server (method/oracle/reward_server.py) on a rented GPU, tunnelled to the Mac's localhost:PORT:
#   bash experiments/oracle/cao/reward_pod.sh [HOURS]
#     env: PORT (8701), MODEL (Qwen/Qwen3-8B), THR (the weak-zeroed reference), ADAPTERS ("name=/Users/user/mpd-data/...
#     ..."), ROWS (32), GPU, PRICE
# then POST http://localhost:$PORT/check {"claims": [...]} (reward_server.py check URL CLAIMS OUT); GET / for the
# server's counts. The tunnel is reopened by running the script with TUNNEL_ONLY=1; rp-run stop orc-primerl-reward ends it.
cd /Users/user/params
H=${1:-6} PORT=${PORT:-8701} MODEL=${MODEL:-Qwen/Qwen3-8B} ROWS=${ROWS:-32}
THR=${THR:-/Users/user/mpd-data/cluster/t2_8b/out/thr.pt}
NAME=orc-primerl-reward OUT=/Users/user/mpd-data/runpod/$NAME
tunnel() {  # ssh -L to the run's pod (its address from pod.json)
  until grep -q "running:" "$OUT/rp-run.log" 2> /dev/null; do sleep 10; done
  read -r IP SPORT < <(/Users/user/mpd-data/venv/bin/python -c "
import json; d = json.load(open('$OUT/pod.json')); m = [x for x in d['portMappings'] if str(x.get('privatePort', '')) == '22'] if isinstance(d.get('portMappings'), list) else []
print(d['publicIp'], m[0]['publicPort'] if m else d['portMappings']['22'])")
  KEY=$HOME/.ssh/runpod_ed25519  # (rp-run authorizes this key on every pod beside its own)
  until curl -s "localhost:$PORT/" > /dev/null 2>&1; do
    ssh -f -N -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -p "$SPORT" \
      -L "$PORT:localhost:$PORT" "root@$IP" 2> /dev/null
    sleep 15
  done
  echo "reward server: http://localhost:$PORT ($(curl -s "localhost:$PORT/"))"
}
[ -n "${TUNNEL_ONLY:-}" ] && { tunnel; exit 0; }
ad=""
for x in ${ADAPTERS:-}; do ad="$ad --adapter $x"; done
RP_HF_MODELS="$MODEL" RP_PYENV=oracle RP_OWNER=primerl RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.8} \
  RP_MIN_VRAM_GB=${VRAM:-40} nohup experiments/infra/runpod/rp-run $NAME "${GPU:-RTX A6000,A40,L40S,L40,RTX 6000 Ada}" "$H" -- \
  python -u /Users/user/params/method/oracle/reward_server.py serve --model "$MODEL" --thr "$THR" --port "$PORT" --rows "$ROWS" $ad \
  > "$OUT.log" 2>&1 &
tunnel
