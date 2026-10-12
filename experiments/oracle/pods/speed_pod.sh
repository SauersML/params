#!/usr/bin/env bash
# The oracle checker's speed benchmarks (experiments/oracle/bench_speed.py) on one CUDA pod: carry, score and events
# with method/oracle at BEFORE and at origin/main (before/after numbers on one card), and whether Native.sensitivity's
# windowed probes equal the whole forward on CUDA (the window bench). Both source trees and texts/vpd4l go up as one
# tar file; VPD's files are published inputs. Results: $O/speed.jsonl.
#   experiments/oracle/pods/speed_pod.sh HOURS   (env: RUN, BEFORE, PRICE, VRAM, GPU)
cd /Users/user/params
H=${1:-0.5}
RUN=${RUN:-a}
BEFORE=${BEFORE:-71a11ee}
N=oracle-graph-speed-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
D=/Users/user/mpd-data/scratch/speed/$RUN
git fetch -q origin main && rm -rf "$D" && mkdir -p "$D/after" "$D/before" || exit 1
git archive origin/main method/oracle experiments/oracle method/vpd/vpd_model.py | tar -x -C "$D/after" \
  && echo "python source $(git rev-parse --short=10 origin/main)" > "$D/after/SOURCE" \
  && git archive "$BEFORE" method/oracle method/vpd/vpd_model.py | tar -x -C "$D/before" || exit 1
COPYFILE_DISABLE=1 tar -cf "$D/src.tar" -C "$D" after before -C /Users/user/mpd-data/graph_oracle texts/vpd4l \
  && rm -rf "$D/after" "$D/before"
B="python experiments/oracle/bench_speed.py"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O $D ~/mpd-data/graph_oracle; \
tar -xf $D/src.tar -C $D; mv $D/texts ~/mpd-data/graph_oracle/; cd $D/after; \
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; \
nvidia-smi --query-gpu=name,timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > $O/gpu.csv & \
$B --code $D/before/method/oracle --label before-$BEFORE --out $O/speed.jsonl > $O/before.log 2>&1; \
$B --label after --benches carry,score,events,window --out $O/speed.jsonl > $O/after.log 2>&1; \
echo done"
RP_ALLOW_OVER=1 RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.6} RP_MIN_VRAM_GB=${VRAM:-44} \
  RP_MIN_RAM_GB=32 experiments/infra/runpod/rp-run $N "${GPU:-auto}" $H -- bash -c "$CMD"
