#!/usr/bin/env bash
# The vectors arm of the component explainer on a rented GPU (#2951 graph oracle): rl/inject.py's SFT (transformers +
# PEFT LoRA on the base, the vector maps trained with it) and its answers to the held-out questions, with the
# subcomponents' own vectors and with them shuffled among each question's subcomponents. The SFT rows, the question
# rows and the held-out rows go up as one tar file, origin/main's method/oracle and method/vpd as another; VPD's
# vectors (native.UV) are a published input and the base comes from the Hugging Face hub. Outputs come back in
# ~/mpd-data/runpod/oracle-graph-inject-$RUN/: run/ (adapter, injector.pt, sft.jsonl), samples.jsonl,
# samples_shuffled.jsonl, inject.log. E1 (tinker_train.py readers) and the evaluation (eval_pod.sh) follow on their own.
#   bash experiments/oracle/pods/inject_pod.sh RUN SFT.jsonl ROWS.jsonl HELDOUT.jsonl
#     (env: STEPS, BATCH, LR, BASE, NEVAL, SAMPLES, MAXTOK, HOURS (cap, default 3), GPU, PRICE, VRAM, RP_OWNER;
#      LEDGER_OK=1 once the lead approved a launch past $285)
cd /Users/user/params
spent=$(awk -F'\t' '{s += $6} END {printf "%.2f", s}' /Users/user/mpd-data/runpod/LEDGER.tsv)
if awk -v s="$spent" 'BEGIN {exit !(s > 285)}' && [ -z "${LEDGER_OK:-}" ]; then  # the lead's check before launches past $285
  echo "inject_pod.sh: the team's RunPod ledger is at \$$spent (> \$285): ask the lead, then rerun with LEDGER_OK=1" >&2; exit 2
fi
RUN=$1 SFT=$2 ROWS=$3 HELDOUT=$4
N=oracle-graph-inject-$RUN
O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/primerl/src_inject_$RUN
IN=/Users/user/mpd-data/scratch/primerl/in_inject_$RUN
I='~/mpd-data/in'  # the inputs on the pod (unpacked from $IN.tar)
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main method/oracle method/vpd | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"  # rp-run uploads every existing path the command names: only the archives exist here
rm -rf "$IN" "$IN.tar" && mkdir -p "$IN" && cp "$SFT" "$IN/sft.jsonl" && cp "$ROWS" "$IN/rows.jsonl" && cp "$HELDOUT" "$IN/heldout.jsonl"
COPYFILE_DISABLE=1 tar -cf "$IN.tar" -C "$IN" sft.jsonl rows.jsonl heldout.jsonl && rm -rf "$IN"
B="--base ${BASE:-Qwen/Qwen3-8B}"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S $I; tar -xf $IN.tar -C $I; tar -xf $S.tar -C $S; \
ls /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; \
cd $S/method/oracle/rl; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 10 > $O/gpu.csv & \
python -u inject.py sft $I/sft.jsonl $I/rows.jsonl --out $O/run $B --steps ${STEPS:-100} --batch ${BATCH:-8} --lr ${LR:-2e-4} > $O/inject.log 2>&1; \
python -u inject.py sample $I/heldout.jsonl $O/samples.jsonl --init $O/run $B --n ${NEVAL:-20} --samples ${SAMPLES:-2} --max-tokens ${MAXTOK:-4096} >> $O/inject.log 2>&1; \
python -u inject.py sample $I/heldout.jsonl $O/samples_shuffled.jsonl --init $O/run $B --n ${NEVAL:-20} --samples ${SAMPLES:-2} --max-tokens ${MAXTOK:-4096} --shuffle >> $O/inject.log 2>&1"
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-200} RP_OWNER=${RP_OWNER:-fleet} RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-1.0} \
  RP_MIN_VRAM_GB=${VRAM:-44} experiments/infra/runpod/rp-run $N "${GPU:-A40,RTX A6000,L40S,L40,RTX 6000 Ada}" ${HOURS:-3} -- bash -c "$CMD"
