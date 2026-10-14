#!/usr/bin/env bash
# The oracle's gate (h2h.py) on a rented GPU: items and their truth on Qwen3-1.7B, then the answerers, then the table,
# into ~/mpd-data/runpod/oracle-h2h-$RUN/ (items.jsonl, out.jsonl, report.txt). Qwen3-1.7B and the released Activation
# Oracle adapter come from Hugging Face on the pod; wikitext-103's test split too (public).
#   RUN=smoke ARGS="--n-dev 2 --n-test 3" bash experiments/oracle/aoh2h/h2h_pod.sh [HOURS]
#   (env: SRCREV (default origin/main), SYSTEMS (default "R A B"), BUDGET, GPU, ITEMS: an items file to reuse instead of
#    building; AO_LIB: the oracle paper's notebook library cell saved apart, for ao_equal.py's check first, run with
#    python -I, the file kept outside our tree on the pod too)
cd /Users/user/params
RUN=${RUN:-a} H=${1:-3}
N=oracle-h2h-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/primerl/src_h2h_$RUN
REV=${SRCREV:-origin/main}
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" experiments/oracle method/oracle | tar -x -C "$S" \
  && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
[ -n "${ITEMS:-}" ] && cp "$ITEMS" "$S/items_in.jsonl"
[ -n "${AO_LIB:-}" ] && mkdir -p "$S/untrusted" && cp "$AO_LIB" "$S/untrusted/ao_lib.py"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
BUILD="python -u h2h.py build --out $O/items.jsonl ${ARGS:-} > $O/build.log 2>&1"
[ -n "${ITEMS:-}" ] && BUILD="cp $S/items_in.jsonl $O/items.jsonl"
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S; \
tar -xf $S.tar -C $S; cd $S/experiments/oracle/aoh2h; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 > $O/gpu.csv & \
${AO_LIB:+python -I -u ao_equal.py $S/untrusted/ao_lib.py --out $O/ao_equal.json > $O/ao_equal.log 2>&1;} \
$BUILD && python -u h2h.py run --items $O/items.jsonl --out $O/out.jsonl --systems ${SYSTEMS:-R A B} ${BUDGET:+--budget $BUDGET} > $O/run.log 2>&1; \
python h2h.py report $O/out.jsonl > $O/report.txt 2>&1"
RP_HF_MODELS="Qwen/Qwen3-1.7B adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B" RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=50 \
  RP_OWNER=primerl RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-20} \
  experiments/infra/runpod/rp-run $N "${GPU:-RTX 3090,RTX 4090,RTX A5000,L40S,RTX A6000}" $H -- bash -c "$CMD"
