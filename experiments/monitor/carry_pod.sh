#!/usr/bin/env bash
# The carry test (carry_qwen.py run) on one pod: Qwen3-8B cases, then Qwen3-Coder-30B-A3B cases, each under a time limit.
#   experiments/monitor/carry_pod.sh NAME HOURS     (env: GPU, DEAD8 / DEADC seconds per stage)
cd /Users/user/params
N=$1 H=$2
S=/Users/user/mpd-data/scratch/monitor/stage_$N
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
rm -rf "$S" && mkdir -p "$S/data" && cp experiments/monitor/agent.py experiments/monitor/sandbox.py experiments/monitor/replay.py \
     experiments/monitor/carry_qwen.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && cp /Users/user/mpd-data/monitor/carry_cases_8b.jsonl /Users/user/mpd-data/monitor/carry_cases_coder.jsonl "$S/" \
  && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="set -e; mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
python carry_qwen.py run --model Qwen/Qwen3-8B --cases carry_cases_8b.jsonl --out $O/carry_8b.jsonl --deadline ${DEAD8:-2400};
python carry_qwen.py run --model Qwen/Qwen3-Coder-30B-A3B-Instruct --cases carry_cases_coder.jsonl --out $O/carry_coder.jsonl --deadline ${DEADC:-1800};
echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=140 RP_MIN_VRAM_GB=78 \
  RP_HF_MODELS="Qwen/Qwen3-8B Qwen/Qwen3-Coder-30B-A3B-Instruct" \
  experiments/infra/runpod/rp-run $N "${GPU:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3}" $H -- bash -c "$CMD"
