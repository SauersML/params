#!/usr/bin/env bash
# Stage 2 of the judge-free WorkspaceBench on one RunPod pod: Qwen3.6-27B (the benchmark's model) gated,
# interchange maps by the hook engine (its linear-attention layers rule out the prefix cache), and the
# released oracle lens (agu18dec/olens_and_ar, olens_s3d_rl600) read at every cell of layers 20-60 step 4.
# Each stage resumes from its outputs, so a replaced pod continues. The code and the WorkspaceBench data
# (MIT, evaluation only) go up as one tar file; results come back to ~/mpd-data/runpod/NAME.
#   workspace/pod_stage2.sh NAME HOURS     (env: GPU, PER_FAMILY)
# Before the first run: check readers.OLENS_PROMPT's line breaks against the source renderer.
cd /Users/user/params
N=$1 H=$2 M=Qwen/Qwen3.6-27B
IN=/Users/user/mpd-data/workspace-bench/pod_in_$N.tar
S=/Users/user/mpd-data/scratch/workspace/stage_$N
rm -rf "$S" && mkdir -p "$S/code" && cp workspace/data.py workspace/interchange.py workspace/readers.py workspace/run.py \
     workspace/summary.py "$S/code/" && cp -R /Users/user/mpd-data/workspace-bench/hf "$S/hf" && rm -rf "$S/hf/.cache" \
  && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/workspace-bench/pod_src_$N
A="--model $M --device cuda --dtype bfloat16"
CMD="set -e; mkdir -p $O $X; tar -xf $IN -C $X; cd $X/code; export WSB_DATA=$X/hf WSB_RESULTS=$O PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0;
[ -f $O/Qwen3.6-27B/gate.json ] || python run.py gate $A;
python run.py patch $A --engine hooks --batch 48 --per-family ${PER_FAMILY:-20};
python run.py olens $A;
python run.py summary --model $M;
echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=${DISK:-160} RP_MIN_VRAM_GB=78 \
  RP_HF_MODELS="$M" \
  infra/runpod/rp-run $N "${GPU:-NVIDIA H100 80GB HBM3,NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe}" $H -- bash -c "$CMD"
