#!/usr/bin/env bash
# The Qwen3-Coder-30B-A3B arm on one pod: agent runs (strict prompt, oneoff and conflicting, 2 samples per task), the
# per-cheat replay for this arm and for the Qwen3-8B cheats (cheats.py, vLLM with each run's own model), then
# activations of this arm under a time cap. Each stage is skipped when its output exists, so a replaced pod resumes.
#   experiments/monitor/coder_pod.sh NAME HOURS     (env: GPU)
cd /Users/user/params
N=$1 H=$2
M=Qwen/Qwen3-Coder-30B-A3B-Instruct
S=/Users/user/mpd-data/scratch/monitor/stage_$N
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
rm -rf "$S" && mkdir -p "$S/data" && cp experiments/monitor/agent.py experiments/monitor/sandbox.py experiments/monitor/acts.py \
     experiments/monitor/replay.py experiments/monitor/cheats.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && cp -R /Users/user/mpd-data/monitor/cheats_in "$S/cheats_in" \
  && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="set -e; mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
[ -f $O/gen/done ] || { python agent.py --model $M --out $O/gen --splits oneoff,conflicting --samples 2 --turn-tokens 3072 --action-tokens 1536 --max-turns 10 --deadline 2400 --max-seqs 64 --workers 48 --gpu-mem 0.92 && touch $O/gen/done; };
[ -f $O/cheats_coder.json ] || python cheats.py --model $M --runs $O --out $O/cheats_coder.json;
[ -f $O/cheats_8b.json ] || python cheats.py --model Qwen/Qwen3-8B --runs cheats_in/monitor-full2 --out $O/cheats_8b.json;
[ -f $O/acts/feats.npz ] || python acts.py --model $M --layers 12,24,36,48 --runs $O/gen/transcripts.jsonl --out $O/acts --max-gb 6 --deadline 900;
echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=160 RP_MIN_VRAM_GB=78 \
  RP_HF_MODELS="$M Qwen/Qwen3-8B" \
  experiments/infra/runpod/rp-run $N "${GPU:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe}" $H -- bash -c "$CMD"
