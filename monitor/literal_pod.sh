#!/usr/bin/env bash
# literal_monitor.py scan over every run of the named arms with one model, on one pod.
#   monitor/literal_pod.sh NAME HOURS MODEL DEADLINE_S ARM...     (ARM: full2 q8d2 tmpl1 q14 coder1; env: GPU, UPMAX)
cd /Users/user/params
N=$1 H=$2 M=$3 DL=$4
shift 4
S=/Users/user/mpd-data/scratch/monitor/stage_$N
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
src() {
  case $1 in
    full2) echo /Users/user/mpd-data/runpod/monitor-full2/gen/transcripts.jsonl ;;
    q8d2) echo /Users/user/mpd-data/monitor/quick/monitor-q8d-2/gen/transcripts.jsonl ;;
    tmpl1) echo /Users/user/mpd-data/runpod/monitor-tmpl1/gen/transcripts.jsonl ;;
    q14) echo /Users/user/mpd-data/monitor/quick/monitor-q14-1/gen/transcripts.jsonl ;;
    coder1) echo /Users/user/mpd-data/runpod/monitor-coder1/gen/transcripts.jsonl ;;
  esac
}
rm -rf "$S" && mkdir -p "$S/data" \
  && cp monitor/agent.py monitor/sandbox.py monitor/replay.py monitor/carry_qwen.py monitor/literal_monitor.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" || exit 1
ARMS=""
for a in "$@"; do
  f=$(src "$a"); [ -n "$f" ] || { echo "unknown arm $a" >&2; exit 2; }
  mkdir -p "$S/arms/$a/gen" && cp "$f" "$S/arms/$a/gen/" || exit 1
  ARMS="$ARMS arms/$a"
done
COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
python literal_monitor.py scan --model $M --runs$ARMS --out $O/literals.jsonl --deadline $DL"
RP_UPLOAD_MAX_MB=${UPMAX:-200} RP_ALLOW_OVER=1 RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle RP_MIN_VRAM_GB=78 \
  RP_HF_MODELS="$M" \
  infra/runpod/rp-run $N "${GPU:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe}" $H -- bash -c "$CMD"
