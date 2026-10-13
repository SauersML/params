#!/usr/bin/env bash
# RL environment D's frozen reader on a rented GPU (#2951 graph oracle): vLLM serving MODEL on the pod (default
# Qwen/Qwen3-32B-FP8, an open model, never trained), then one of
#   gate DATA_DIR HOURS          env_d.py gate over DATA_DIR's facts30.jsonl, rows30.jsonl and english30.jsonl (E1 of the
#                                correct English and its controls); the pod ends when it is done
#   select FACTS ROWS SAMPLES HOURS   env_d.py select (the SFT data's best sample per text by E1)
#   serve FACTS ROWS HOURS       env_d.py serve on the pod's port 8772 (RL's scorer; the Mac reaches it through an SSH
#                                tunnel as scorer_tunnel.sh does) until the cap
# The source (origin/main's method/oracle and method/vpd) and the inputs (with the texts they name) go up as two tar files.
#   env: RUN (default a), GPU, PRICE ($/h cap, default 1.0), MODEL, RP_OWNER (default inframove)
set -euo pipefail
cd /Users/user/params
MODE=$1
shift
RUN=${RUN:-a}
MODEL=${MODEL:-Qwen/Qwen3-32B-FP8}
N=oracle-graph-envd-$MODE-$RUN
O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/inframove/src_envd_$RUN
IN=/Users/user/mpd-data/scratch/inframove/in_envd_$RUN
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main method/oracle method/vpd | tar -x -C "$S"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
rm -rf "$IN" "$IN.tar" && mkdir -p "$IN/vpd4l"
I='~/mpd-data/in'  # the inputs on the pod (unpacked from $IN.tar)
case $MODE in
  gate) cp "$1/facts30.jsonl" "$IN/facts.jsonl"; cp "$1/rows30.jsonl" "$IN/rows.jsonl"; cp "$1/english30.jsonl" "$IN/english.jsonl"; H=$2
        RUNCMD="python -u method/oracle/env_d.py gate $I/facts.jsonl $I/rows.jsonl $I/english.jsonl $O/gate.jsonl --reader \"\$R\" ${EVENTS:+--events $EVENTS}" ;;
  select) cp "$1" "$IN/facts.jsonl"; cp "$2" "$IN/rows.jsonl"; cp "$3" "$IN/samples.jsonl"; H=$4
        RUNCMD="python -u method/oracle/env_d.py select $I/facts.jsonl $I/rows.jsonl $I/samples.jsonl $O/english.jsonl --reader \"\$R\"" ;;
  serve) cp "$1" "$IN/facts.jsonl"; cp "$2" "$IN/rows.jsonl"; H=$3
        RUNCMD="python -u method/oracle/env_d.py serve $I/facts.jsonl $I/rows.jsonl --port 8772 --reader \"\$R\" > $O/serve.log 2>&1" ;;
  *) echo "mode: gate, select or serve" >&2; exit 2 ;;
esac
python3 -c "
import json, shutil, sys
for l in open('$IN/facts.jsonl'):
    t = json.loads(l)['task']
    if not 9100 <= int(t[4:]) <= 9149:  # (the sealed texts never leave the Mac)
        shutil.copy('/Users/user/mpd-data/graph_oracle/texts/vpd4l/' + t + '.json', '$IN/vpd4l/')
"
COPYFILE_DISABLE=1 tar -cf "$IN.tar" -C "$IN" . && rm -rf "$IN"
CMD="ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S $I ~/mpd-data/graph_oracle/texts; tar -xf $IN.tar -C $I; \
mv $I/vpd4l ~/mpd-data/graph_oracle/texts/; tar -xf $S.tar -C $S; ls /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json > /dev/null; \
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.csv 2>&1 & \
python -m vllm.entrypoints.openai.api_server --model $MODEL --port 8100 --max-model-len 24576 --gpu-memory-utilization 0.9 \
  --enable-prefix-caching > $O/vllm.log 2>&1 & \
until python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/v1/models')\" 2> /dev/null; do sleep 10; done; \
echo reader up; R='http://127.0.0.1:8100/v1/completions|$MODEL'; cd $S && $RUNCMD"
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-200} RP_OWNER=${RP_OWNER:-inframove} RP_PARALLEL=1 RP_PYENV=oracle \
  RP_HF_MODELS="$MODEL" RP_MAX_PRICE=${PRICE:-1.0} RP_MIN_VRAM_GB=${VRAM:-44} \
  experiments/infra/runpod/rp-run $N "${GPU:-L40S,RTX 6000 Ada,L40,RTX A6000,A40}" "$H" -- bash -c "$CMD"
