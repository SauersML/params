#!/bin/bash
# serve_pod.sh WS ITEMS MODEL [PORT]: on a pod, agent.py's tool server with its generation on a vLLM OpenAI server beside
# it: every item adapter resolved (ORGANISMS/<adapter_path> on disk, else downloaded from Hugging Face) and registered as a
# LoRA module under the item's adapter name; vLLM started first (it takes VLLM_UTIL of the card), then the HF server
# (internals tools) with AGENT_VLLM_URL. Env: MAX_LORA_RANK (16; AuditBench 128), MAX_LORAS (vLLM's on the card),
# VLLM_UTIL (0.45), MAX_LEN (8192), AGENT_MAX_ADAPTERS (the HF side's, 8), ORGANISMS, CONTROLS.
WS=$1 ITEMS=$2 MODEL=$3 PORT=${4:-8782}
export PATH=$(ls -d /workspace/pyenv/*/bin | head -1):$PATH HF_HOME=/workspace/hf-home
cd "$(dirname "$0")"
MODS=$(python - "$ITEMS" <<'PY'
import json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
seen = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("adapter") and r["adapter"] not in seen:
        local = Path(os.environ.get("ORGANISMS", "/nonexistent")) / r["adapter_path"]
        seen[r["adapter"]] = str(local) if local.exists() else snapshot_download(r["adapter_path"])
print(" ".join(f"{k}={v}" for k, v in seen.items()))
PY
)
for f in vllm serve_$PORT; do [ -f $WS/$f.pid ] && kill $(cat $WS/$f.pid) 2>/dev/null; done; sleep 5
nohup python -m vllm.entrypoints.openai.api_server --model $MODEL --dtype bfloat16 --enable-lora --max-lora-rank ${MAX_LORA_RANK:-16} \
  --max-loras ${MAX_LORAS:-8} --max-cpu-loras 64 --lora-modules $MODS --gpu-memory-utilization ${VLLM_UTIL:-0.45} \
  --max-model-len ${MAX_LEN:-8192} --port 8800 > $WS/vllm.log 2>&1 &
echo $! > $WS/vllm.pid
until curl -s -m 5 http://127.0.0.1:8800/v1/models | grep -q "$MODEL"; do
  sleep 10; kill -0 $(cat $WS/vllm.pid) 2>/dev/null || { echo "vllm died"; tail -30 $WS/vllm.log; exit 1; }
done
echo "vllm up"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AGENT_ROWS=${AGENT_ROWS:-48} \
  AGENT_MAX_ADAPTERS=${AGENT_MAX_ADAPTERS:-8} AGENT_VLLM_URL=http://127.0.0.1:8800
nohup python -u agent.py serve --model $MODEL --items $ITEMS --port $PORT --gen-cost call > $WS/serve_$PORT.log 2>&1 &
echo $! > $WS/serve_$PORT.pid
echo "tool server $(cat $WS/serve_$PORT.pid)"
