#!/usr/bin/env bash
# The PheWAS screen's group testing, level after level in one process (screen.py refine) on a rented 48 GB card; the
# changes appended to ~/mpd-data/runpod/monitor-refine-$RUN/changes.jsonl (resumable: a relaunch continues from them).
#   RUN=a PHENOME=~/mpd-data/phewas/phenome_v1 RESULTS="L0a.jsonl L0b.jsonl" THR=~/mpd-data/cluster/t2_8b/out/thr.pt \
#     FOCUS="refusal/harmful ..." bash experiments/oracle/phewas/refine_pod.sh [HOURS]   (env: SPLIT, TOP, MAXSETS, SRCREV, GPU, PRICE)
cd /Users/user/params
RUN=${RUN:?RUN} H=${1:-6}
N=monitor-refine-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/speed/phewas/refine_src_$RUN
P=${PHENOME:?PHENOME}
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tgz" && mkdir -p "$S/phenome" "$S/results" && git archive "$REV" experiments/oracle/phewas method/oracle | tar -x -C "$S" \
  && cp "$P/phenome.json" "$P/values.jsonl" "$P/splits.jsonl" "$P/sequences.jsonl" "$S/phenome/" && cp "${THR:?THR}" "$S/thr.pt" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE" || exit 1
k=0; R=""
for f in ${RESULTS:?RESULTS}; do cp "$f" "$S/results/r$k.jsonl" || exit 1; R="$R $S/results/r$k.jsonl"; k=$((k + 1)); done
[ -f "$O/changes.jsonl" ] && cp "$O/changes.jsonl" "$S/changes_before.jsonl"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$S.tgz" -C "$S" . && rm -rf "$S" || exit 1
CMD="export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S; \
tar -xzf $S.tgz -C $S; cd $S/experiments/oracle/phewas; [ -f $S/changes_before.jsonl ] && cp $S/changes_before.jsonl $O/changes.jsonl; \
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv &); \
python -u screen.py refine $S/phenome $O/changes.jsonl --results $R --thr $S/thr.pt ${FOCUS:+--focus $FOCUS} --split ${SPLIT:-4} \
  --top ${TOP:-6} --max-sets ${MAXSETS:-1500} > $O/refine.log 2>&1"
RP_HF_MODELS="Qwen/Qwen3-8B" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=120 RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle \
  RP_MAX_PRICE=${PRICE:-0.6} RP_MIN_VRAM_GB=44 experiments/infra/runpod/rp-run $N "${GPU:-RTX A6000,A40,L40S,L40,RTX 6000 Ada}" "$H" -- bash -c "$CMD"
