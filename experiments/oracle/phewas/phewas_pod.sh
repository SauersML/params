#!/usr/bin/env bash
# The PheWAS's GPU stages on a rented 48 GB card (MATS first; this when its GPUs are taken); outputs in
# ~/mpd-data/runpod/monitor-phewas-$RUN/.
#   default: the phenome's sequences scored on Qwen3-8B (phenome.py score), its report (phenome.json, values.jsonl),
#            then stage 1's association (assoc.py) on the kept traits
#   MODE=assoc: stage 1 alone on a reported phenome (PHENOME with phenome.json and values.jsonl)
#   MODE=polyscore: polyneuronal scores against a dense probe (polyscore.py) on a reported phenome, with stage 1's
#            summary (ASSOC), the dev-only candidates (CANDS: polyscore.candidates) and optionally the fine-mapping (FINEMAP)
#   MODE=pilot: the screen's noise pilot (screen.py pilot) on a reported phenome (PHENOME with phenome.json), with
#            the 8B T2 file (THR) for the massive cut and optionally stage 1's summary (ASSOC) for the singles
#   RUN=a PHENOME=~/mpd-data/phewas/phenome_v1 [MODE=pilot ASSOC=.../summary.json THR=.../thr.pt] bash
#        experiments/oracle/phewas/phewas_pod.sh [HOURS]   (env: SRCREV, GPU, PRICE)
cd /Users/user/params
RUN=${RUN:-a} H=${1:-3}
N=monitor-phewas-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/speed/phewas/src_$RUN
P=${PHENOME:?PHENOME}
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S/phenome" && git archive "$REV" experiments/oracle/phewas method/oracle | tar -x -C "$S" \
  && cp "$P/traits.json" "$P/sequences.jsonl" "$P/splits.jsonl" "$S/phenome/" && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE" || exit 1
case "${MODE:-}" in
  assoc) cp "$P/phenome.json" "$P/values.jsonl" "$S/phenome/" || exit 1 ;;
  polyscore) cp "$P/phenome.json" "$P/values.jsonl" "$S/phenome/" && cp "${ASSOC:?ASSOC}" "$S/assoc_summary.json" || exit 1
         [ -n "${FINEMAP:-}" ] && { cp "$FINEMAP" "$S/finemap.json" || exit 1; }
         [ -n "${CANDS:-}" ] && { cp "$CANDS" "$S/candidates.json" || exit 1; } ;;
  pilot) cp "$P/phenome.json" "$P/values.jsonl" "$S/phenome/" && cp "${THR:?THR}" "$S/thr.pt" || exit 1
         [ -n "${ASSOC:-}" ] && { cp "$ASSOC" "$S/assoc_summary.json" || exit 1; } ;;
esac
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$S.tgz" -C "$S" . && rm -rf "$S" || exit 1
RUNS="python -u phenome.py score $S/phenome --model Qwen/Qwen3-8B --scores $O/scores_Qwen3-8B.jsonl > $O/score.log 2>&1 && \
python -u phenome.py report $S/phenome --model Qwen/Qwen3-8B --scores $O/scores_Qwen3-8B.jsonl > $O/report.txt 2>&1 && \
cp $S/phenome/phenome.json $S/phenome/values.jsonl $O/ && \
python -u assoc.py $S/phenome $O/assoc --model Qwen/Qwen3-8B > $O/assoc.log 2>&1"
[ "${MODE:-}" = polyscore ] && RUNS="python -u polyscore.py $S/phenome $S/assoc_summary.json $O/polyscore.json ${FINEMAP:+--finemap $S/finemap.json} ${CANDS:+--candidates-file $S/candidates.json} > $O/polyscore.log 2>&1"
[ "${MODE:-}" = assoc ] && RUNS="python -u assoc.py $S/phenome $O/assoc --model Qwen/Qwen3-8B > $O/assoc.log 2>&1"
[ "${MODE:-}" = pilot ] && RUNS="python -u screen.py pilot $S/phenome $O/pilot.json --thr $S/thr.pt ${ASSOC:+--assoc $S/assoc_summary.json} > $O/pilot.log 2>&1"
CMD="export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S; \
tar -xzf $S.tgz -C $S; cd $S/experiments/oracle/phewas; \
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv &); $RUNS"
RP_HF_MODELS="Qwen/Qwen3-8B" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=100 RP_OWNER=monitorbench RP_PARALLEL=1 RP_PYENV=oracle \
  RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=44 experiments/infra/runpod/rp-run $N "${GPU:-RTX A6000,A40,L40S,L40,RTX 6000 Ada}" "$H" -- bash -c "$CMD"
