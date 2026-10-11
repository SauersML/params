#!/usr/bin/env bash
# An arm with the chat-template harness (agent.py --render template) on one pod: agent runs, the per-cheat replay and
# the hard-coded-value replay (cheats.py), then activations, each call's own (acts.py). Stages skip when done.
#   experiments/monitor/rerun_pod.sh NAME HOURS "AGENT ARGS"     (env: GPU, AGENT (default Qwen/Qwen3-8B), LAYERS, VRAM)
cd /Users/user/params
N=$1 H=$2 ARGS=$3 AGENT=${AGENT:-Qwen/Qwen3-8B}
S=/Users/user/mpd-data/scratch/monitor/stage_$N
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
rm -rf "$S" && mkdir -p "$S/data" && cp experiments/monitor/agent.py experiments/monitor/sandbox.py experiments/monitor/acts.py \
     experiments/monitor/replay.py experiments/monitor/cheats.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="set -e; mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
[ -f $O/gen/done ] || { python agent.py --model $AGENT --render template --out $O/gen $ARGS && touch $O/gen/done; };
[ -f $O/cheats.json ] || python cheats.py --model $AGENT --runs $O --gpu-mem 0.6 --out $O/cheats.json;
[ -f $O/hardcoded.json ] || python cheats.py --model $AGENT --runs $O --targets hardcoded --no-regen --gpu-mem 0.6 --out $O/hardcoded.json;
[ -f $O/acts/feats.npz ] || python acts.py --model $AGENT --layers ${LAYERS:-8,16,24,32} --runs $O/gen/transcripts.jsonl --out $O/acts --max-gb 4 --deadline 420;
echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=80 RP_MIN_VRAM_GB=${VRAM:-44} RP_HF_MODELS="$AGENT" \
  experiments/infra/runpod/rp-run $N "${GPU:-NVIDIA A40,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
