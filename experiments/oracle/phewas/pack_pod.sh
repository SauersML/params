#!/usr/bin/env bash
# Several PheWAS jobs packed on one rented 48 GB card (the lead, 10-16: each 8B process fits in 20-25 GB): every line of
# CHAINS is a shell command chain, all chains run at once on the pod and the run ends when the last does. Inputs (files
# or directories, INPUTS) are copied to $S/in/<name>; chains run in experiments/oracle/phewas and write under $O
# (~/mpd-data/runpod/monitor-pack-$RUN/, copied back as the run goes). In a chain, $S, $O and $I (= $S/in) are expanded
# on the pod.
#   RUN=a INPUTS="~/mpd-data/phewas/phenome_v1 ~/mpd-data/cluster/t2_8b/out/thr.pt" CHAINS=chains.txt \
#     bash experiments/oracle/phewas/pack_pod.sh [HOURS]   (env: SRCREV, GPU, PRICE, UPLOAD_MB)
cd /Users/user/params
RUN=${RUN:?RUN} H=${1:-2}
N=monitor-pack-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/speed/phewas/pack_src_$RUN
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tgz" && mkdir -p "$S/in" && git archive "$REV" experiments/oracle/phewas method/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE" || exit 1
for f in ${INPUTS:?INPUTS}; do cp -R "$f" "$S/in/" || exit 1; done
[ -d "$O" ] && for f in "$O"/*.jsonl "$O"/*.json; do [ -f "$f" ] && mkdir -p "$S/before" && cp "$f" "$S/before/"; done
k=0; RUNS=""
while IFS= read -r line; do
  [ -z "$line" ] && continue
  c=${line//\$I/$S/in}; c=${c//\$S/$S}; c=${c//\$O/$O}
  RUNS="$RUNS ( $c ) > $O/chain$k.log 2>&1 & "
  k=$((k + 1))
done < "${CHAINS:?CHAINS}"
cp "$CHAINS" "$S/chains.txt"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$S.tgz" -C "$S" . && rm -rf "$S" || exit 1
CMD="export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S; \
tar -xzf $S.tgz -C $S; [ -d $S/before ] && cp $S/before/* $O/; cd $S/experiments/oracle/phewas; \
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv &); $RUNS wait"
RP_HF_MODELS="Qwen/Qwen3-8B" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${UPLOAD_MB:-120} RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle \
  RP_IDLE_PCT=0 RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=44 experiments/infra/runpod/rp-run $N "${GPU:-RTX A6000,A40,L40S,L40,RTX 6000 Ada}" "$H" -- bash -c "$CMD"
