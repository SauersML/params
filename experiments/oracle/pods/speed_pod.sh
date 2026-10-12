#!/usr/bin/env bash
# The oracle's speed benchmarks (experiments/oracle/bench_speed.py) on one CUDA pod: carry, score and events with
# method/oracle at BEFORE and at origin/main (before/after numbers on one card), then the explainer's vLLM sampling of
# the questions with ADAPTER as a LoRA and merged into Qwen3-4B (units_sft.py merge, kept on the pod's disk). Both
# source trees and the inputs (texts/vpd4l, the adapter, the questions) go up as tar files; VPD's files are published
# inputs. Results: $O/speed.jsonl.
#   experiments/oracle/pods/speed_pod.sh HOURS   (env: RUN, BEFORE, ADAPTER, QUESTIONS, PRICE, VRAM, GPU)
cd /Users/user/params
H=${1:-0.6}
RUN=${RUN:-a}
BEFORE=${BEFORE:-71a11ee}
ADAPTER=${ADAPTER:-/Users/user/mpd-data/runpod/oracle-graph-units-u231b/sft/adapter}
QUESTIONS=${QUESTIONS:-/Users/user/mpd-data/runpod/oracle-graph-units-u231b/questions.jsonl}
N=oracle-graph-speed-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
D=/Users/user/mpd-data/scratch/speed/$RUN
git fetch -q origin main && rm -rf "$D" && mkdir -p "$D/after" "$D/before" "$D/in" || exit 1
git archive origin/main method/oracle experiments/oracle method/vpd/vpd_model.py | tar -x -C "$D/after" \
  && echo "python source $(git rev-parse --short=10 origin/main)" > "$D/after/SOURCE" \
  && git archive "$BEFORE" method/oracle method/vpd/vpd_model.py | tar -x -C "$D/before" \
  && cp -R "$ADAPTER" "$D/in/adapter" && cp "$QUESTIONS" "$D/in/questions.jsonl" || exit 1
COPYFILE_DISABLE=1 tar -cf "$D/src.tar" -C "$D" after before && rm -rf "$D/after" "$D/before"
COPYFILE_DISABLE=1 tar -cf "$D/in.tar" -C "$D/in" adapter questions.jsonl -C /Users/user/mpd-data/graph_oracle texts/vpd4l \
  && rm -rf "$D/in"
I='~/mpd-data/in'  # the inputs on the pod (a Mac path here would be uploaded file by file)
B="python experiments/oracle/bench_speed.py"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O $D $I ~/mpd-data/graph_oracle; \
tar -xf $D/in.tar -C $I; mv $I/texts ~/mpd-data/graph_oracle/; tar -xf $D/src.tar -C $D; cd $D/after; \
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; \
nvidia-smi --query-gpu=name,timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > $O/gpu.csv & \
$B --code $D/before/method/oracle --label before-$BEFORE --out $O/speed.jsonl > $O/before.log 2>&1; \
$B --label after --out $O/speed.jsonl > $O/after.log 2>&1; \
python method/oracle/rl/units_sft.py merge $I/adapter /workspace/merged > $O/merge.log 2>&1 && \
$B --benches sample --adapter $I/adapter --merged /workspace/merged --questions $I/questions.jsonl --label sample --out $O/speed.jsonl > $O/sample.log 2>&1; \
echo done"
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-400} RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS="Qwen/Qwen3-4B" \
  RP_MAX_PRICE=${PRICE:-0.6} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=48 experiments/infra/runpod/rp-run $N "${GPU:-auto}" $H -- bash -c "$CMD"
