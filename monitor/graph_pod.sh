#!/usr/bin/env bash
# One pod, Qwen3-8B then Qwen3-14B: carry_qwen.py paths (test value -> relay rows -> output, with heads) on the
# Qwen3-8B cases, literal_monitor.py scan over every Qwen3-8B run, then carry_qwen.py run on the Qwen3-14B cases.
# Each stage has its own time limit; a failed stage does not stop the next.
#   monitor/graph_pod.sh NAME HOURS     (env: GPU)
cd /Users/user/params
N=$1 H=$2
S=/Users/user/mpd-data/scratch/monitor/stage_$N
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
rm -rf "$S" && mkdir -p "$S/data" "$S/arms/full2/gen" "$S/arms/q8d2/gen" "$S/arms/tmpl1/gen" \
  && cp monitor/agent.py monitor/sandbox.py monitor/replay.py monitor/carry_qwen.py monitor/literal_monitor.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && cp /Users/user/mpd-data/monitor/carry_cases_8b.jsonl /Users/user/mpd-data/monitor/carry_8b.jsonl \
        /Users/user/mpd-data/monitor/carry_cases_14b.jsonl "$S/" \
  && cp /Users/user/mpd-data/runpod/monitor-full2/gen/transcripts.jsonl "$S/arms/full2/gen/" \
  && cp /Users/user/mpd-data/monitor/quick/monitor-q8d-2/gen/transcripts.jsonl "$S/arms/q8d2/gen/" \
  && cp /Users/user/mpd-data/runpod/monitor-tmpl1/gen/transcripts.jsonl "$S/arms/tmpl1/gen/" \
  && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
python carry_qwen.py paths --model Qwen/Qwen3-8B --cases carry_cases_8b.jsonl --results carry_8b.jsonl --out $O/paths_8b.jsonl --deadline 1200 || echo STAGE paths FAILED;
python literal_monitor.py scan --model Qwen/Qwen3-8B --runs arms/full2 arms/q8d2 arms/tmpl1 --out $O/literals_8b.jsonl --deadline 1500 || echo STAGE scan FAILED;
python carry_qwen.py run --model Qwen/Qwen3-14B --cases carry_cases_14b.jsonl --out $O/carry_14b.jsonl --deadline 600 || echo STAGE 14b FAILED;
echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=120 RP_MIN_VRAM_GB=78 \
  RP_HF_MODELS="Qwen/Qwen3-8B Qwen/Qwen3-14B" \
  infra/runpod/rp-run $N "${GPU:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe}" $H -- bash -c "$CMD"
