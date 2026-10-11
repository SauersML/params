#!/usr/bin/env bash
# Run the hardcoded-output intervention (spans.py) for the 8B and 14B arms on one pod, outputs back to the Mac.
#   monitor/spans_pod.sh NAME HOURS     (env: GPU)
cd /Users/user/params
N=$1 H=$2
S=/Users/user/mpd-data/scratch/monitor/stage_$N
rm -rf "$S" && mkdir -p "$S/data" && cp monitor/agent.py monitor/sandbox.py monitor/replay.py monitor/spans.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && cp -R /Users/user/mpd-data/monitor/spans_in "$S/spans_in" \
  && COPYFILE_DISABLE=1 tar -cf /Users/user/mpd-data/monitor/spans_pod_$N.tar -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/spans_src_$N
IN=/Users/user/mpd-data/monitor/spans_pod_$N.tar
CMD="set -e; mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
python spans.py --model Qwen/Qwen3-8B --runs spans_in/monitor-full2 spans_in/monitor-q8d-2 --out $O/spans_8b.json;
python spans.py --model Qwen/Qwen3-14B --runs spans_in/monitor-q14-1 --out $O/spans_14b.json;
echo spans done"
RP_ALLOW_OVER=1 RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=120 RP_MIN_VRAM_GB=44 \
  RP_HF_MODELS="Qwen/Qwen3-8B Qwen/Qwen3-14B" \
  infra/runpod/rp-run $N "${GPU:-NVIDIA A40,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
