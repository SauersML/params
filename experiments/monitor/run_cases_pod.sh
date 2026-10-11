#!/usr/bin/env bash
# carry_qwen.py run for one model and one cases file on one pod.
#   experiments/monitor/run_cases_pod.sh NAME HOURS MODEL CASES DEADLINE_SECONDS     (env: GPU, VRAM, DISK)
cd /Users/user/params
N=$1 H=$2 M=$3 CASES=$4 DL=${5:-1200}
S=/Users/user/mpd-data/scratch/monitor/stage_$N
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
rm -rf "$S" && mkdir -p "$S/data" && cp experiments/monitor/agent.py experiments/monitor/sandbox.py experiments/monitor/replay.py experiments/monitor/carry_qwen.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && cp "$CASES" "$S/cases.jsonl" && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="set -e; mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
python carry_qwen.py run --model $M --cases cases.jsonl --out $O/carry.jsonl --deadline $DL; echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=${DISK:-140} RP_MIN_VRAM_GB=${VRAM:-78} RP_HF_MODELS="$M" \
  experiments/infra/runpod/rp-run $N "${GPU:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe}" $H -- bash -c "$CMD"
