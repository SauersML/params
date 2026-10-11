#!/usr/bin/env bash
# One pod: the hard-coded-value replay for the Qwen3-Coder arm (cheats.py --targets hardcoded) and the Qwen3-8B
# context-format A/B (format_ab.py).
#   experiments/monitor/ab_pod.sh NAME HOURS     (env: GPU)
cd /Users/user/params
N=$1 H=$2
C=Qwen/Qwen3-Coder-30B-A3B-Instruct
S=/Users/user/mpd-data/scratch/monitor/stage_$N
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
rm -rf "$S" && mkdir -p "$S/data" && cp experiments/monitor/agent.py experiments/monitor/sandbox.py experiments/monitor/replay.py \
     experiments/monitor/cheats.py experiments/monitor/format_ab.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && cp -R /Users/user/mpd-data/monitor/hc_in "$S/hc_in" \
  && cp /Users/user/mpd-data/runpod/monitor-full2/gen/transcripts.jsonl "$S/full2.jsonl" \
  && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="set -e; mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
[ -f $O/hardcoded_coder.json ] || python cheats.py --targets hardcoded --no-regen --model $C --runs hc_in/monitor-coder1 --gpu-mem 0.9 --out $O/hardcoded_coder.json;
[ -f $O/format_ab_8b.json ] || python format_ab.py --runs full2.jsonl --model Qwen/Qwen3-8B --out $O/format_ab_8b.json;
echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=140 RP_MIN_VRAM_GB=78 RP_HF_MODELS="$C Qwen/Qwen3-8B" \
  experiments/infra/runpod/rp-run $N "${GPU:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe}" $H -- bash -c "$CMD"
