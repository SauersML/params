#!/usr/bin/env bash
# The PheWAS screen's knockout sets (screen.py runsets) on a rented 48 GB card, one shard of a design per pod; the
# changes in ~/mpd-data/runpod/monitor-screen-$RUN/changes.jsonl (resumable: a relaunch skips sets already there).
#   RUN=l0a PHENOME=~/mpd-data/phewas/phenome_v1 DESIGN=~/mpd-data/phewas/screen_L0.jsonl SHARD=0/2 \
#     THR=~/mpd-data/cluster/t2_8b/out/thr.pt bash experiments/oracle/phewas/screen_pod.sh [HOURS]   (env: SRCREV, GPU, PRICE;
#     ADAPTER: an organism's LoRA directory merged into the model; PANEL_TRAITS: the panel's traits, default every kept one)
cd /Users/user/params
RUN=${RUN:?RUN} H=${1:-3}
N=monitor-screen-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/speed/phewas/screen_src_$RUN
P=${PHENOME:?PHENOME}
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tgz" && mkdir -p "$S/phenome" && git archive "$REV" experiments/oracle/phewas method/oracle | tar -x -C "$S" \
  && cp "$P/phenome.json" "$P/values.jsonl" "$P/splits.jsonl" "$P/sequences.jsonl" "$S/phenome/" && cp "${DESIGN:?DESIGN}" "$S/design.jsonl" \
  && cp "${THR:?THR}" "$S/thr.pt" && { [ -z "${ADAPTER:-}" ] || cp -R "$ADAPTER" "$S/adapter"; } && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE" || exit 1
[ -f "$O/changes.jsonl" ] && cp "$O/changes.jsonl" "$S/changes_before.jsonl"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$S.tgz" -C "$S" . && rm -rf "$S" || exit 1
CMD="export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S; \
tar -xzf $S.tgz -C $S; cd $S/experiments/oracle/phewas; [ -f $S/changes_before.jsonl ] && cp $S/changes_before.jsonl $O/changes.jsonl; \
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv &); \
python -u screen.py runsets $S/phenome $S/design.jsonl $O/changes.jsonl --thr $S/thr.pt --shard ${SHARD:-0/1} ${ADAPTER:+--adapter $S/adapter} ${PANEL_TRAITS:+--panel-traits $PANEL_TRAITS} > $O/runsets.log 2>&1"
RP_HF_MODELS="Qwen/Qwen3-8B" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${UPLOAD_MB:-100} RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle \
  RP_MAX_PRICE=${PRICE:-0.6} RP_MIN_VRAM_GB=44 experiments/infra/runpod/rp-run $N "${GPU:-RTX A6000,A40,L40S,L40,RTX 6000 Ada}" "$H" -- bash -c "$CMD"
