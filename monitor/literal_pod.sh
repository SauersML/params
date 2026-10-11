#!/usr/bin/env bash
# One pod: carry_qwen.py paths on the Qwen3-14B cases, then literal_monitor.py scan over every run of the Qwen3-14B,
# Qwen3-Coder-30B-A3B and Qwen3-8B arms (in that order). Each stage has its own time limit; a failed stage does not
# stop the next.
#   monitor/literal_pod.sh NAME HOURS     (env: GPU, UPMAX)
cd /Users/user/params
N=$1 H=$2
C=Qwen/Qwen3-Coder-30B-A3B-Instruct
S=/Users/user/mpd-data/scratch/monitor/stage_$N
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
rm -rf "$S" && mkdir -p "$S/data" "$S/arms/full2/gen" "$S/arms/q8d2/gen" "$S/arms/tmpl1/gen" "$S/arms/q14/gen" \
    "$S/arms/coder1/gen" \
  && cp monitor/agent.py monitor/sandbox.py monitor/replay.py monitor/carry_qwen.py monitor/literal_monitor.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && cp /Users/user/mpd-data/monitor/carry_cases_14b.jsonl /Users/user/mpd-data/monitor/carry_14b.jsonl "$S/" \
  && cp /Users/user/mpd-data/runpod/monitor-full2/gen/transcripts.jsonl "$S/arms/full2/gen/" \
  && cp /Users/user/mpd-data/monitor/quick/monitor-q8d-2/gen/transcripts.jsonl "$S/arms/q8d2/gen/" \
  && cp /Users/user/mpd-data/runpod/monitor-tmpl1/gen/transcripts.jsonl "$S/arms/tmpl1/gen/" \
  && cp /Users/user/mpd-data/monitor/quick/monitor-q14-1/gen/transcripts.jsonl "$S/arms/q14/gen/" \
  && cp /Users/user/mpd-data/runpod/monitor-coder1/gen/transcripts.jsonl "$S/arms/coder1/gen/" \
  && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
python carry_qwen.py paths --model Qwen/Qwen3-14B --cases carry_cases_14b.jsonl --results carry_14b.jsonl --out $O/paths_14b.jsonl --deadline 240 || echo STAGE paths14b FAILED;
python literal_monitor.py scan --model Qwen/Qwen3-14B --runs arms/q14 --out $O/literals_14b.jsonl --deadline 600 || echo STAGE scan14b FAILED;
python literal_monitor.py scan --model $C --runs arms/coder1 --out $O/literals_coder.jsonl --deadline 600 || echo STAGE scancoder FAILED;
python literal_monitor.py scan --model Qwen/Qwen3-8B --runs arms/full2 arms/q8d2 arms/tmpl1 --out $O/literals_8b.jsonl --deadline 900 || echo STAGE scan8b FAILED;
echo all stages done"
RP_UPLOAD_MAX_MB=${UPMAX:-200} RP_ALLOW_OVER=1 RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=220 \
  RP_MIN_VRAM_GB=78 RP_HF_MODELS="Qwen/Qwen3-14B $C Qwen/Qwen3-8B" \
  infra/runpod/rp-run $N "${GPU:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe}" $H -- bash -c "$CMD"
