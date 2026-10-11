#!/usr/bin/env bash
# The per-cheat replay (cheats.py) alone on one pod, for runs already generated.
#   experiments/monitor/cheats_pod.sh NAME HOURS MODEL ARM_DIR GPU_MEM     (env: GPU, EXTRA = more cheats.py arguments)
# ARM_DIR is a directory under ~/mpd-data holding gen/transcripts.jsonl (sent up in the tar).
cd /Users/user/params
N=$1 H=$2 M=$3 ARM=$4 MEM=${5:-0.6}
S=/Users/user/mpd-data/scratch/monitor/stage_$N
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
rm -rf "$S" && mkdir -p "$S/data" "$S/arm" && cp experiments/monitor/agent.py experiments/monitor/sandbox.py experiments/monitor/replay.py experiments/monitor/cheats.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && cp -R "$ARM/gen" "$S/arm/gen" \
  && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="set -e; mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
python cheats.py --model $M --runs arm --gpu-mem $MEM ${EXTRA:-} --out $O/cheats.json; echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=${DISK:-60} RP_MIN_VRAM_GB=${VRAM:-44} RP_HF_MODELS="$M" \
  experiments/infra/runpod/rp-run $N "${GPU:-NVIDIA A40,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
