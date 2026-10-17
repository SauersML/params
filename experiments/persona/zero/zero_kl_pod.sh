#!/usr/bin/env bash
# zero_kl_pod.sh: the v2 found set (run A2) shrunk along its ranking to KL <= 0.1 bits on atlas web text (zklmatch.py),
# then held-out replies x4 under it, under 6 matched random sets and unedited (zgen.py), judged (zjudge.py).
N=${NAME:-orc-persona-zeroKL}
O=/Users/user/mpd-data/runpod/$N/out
D=/Users/user/mpd-data/persona/zero
CMD="cd /Users/user/params/experiments/persona/zero && mkdir -p $O && \
python -u zklmatch.py $D/a2/sets.json $D/a2/means.pt $D/atlas_chat_sample.npz $O && \
python -u zgen.py zero $D/prompts.json $O/gen_gates.json --groups heldout:4 --sets $O/sets.json --means $D/a2/means.pt && \
python -u zjudge.py $O/judge.json $D/prompts.json $O/gen_gates.json"
cd /Users/user/params || exit 1
RP_HF_MODELS="movingcastles/zero Qwen/Qwen3-14B" RP_IDLE_PCT=0 RP_OWNER=persona RP_PARALLEL=1 RP_PYENV=oracle \
  RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=44 experiments/infra/runpod/rp-run $N "${GPU:-A40,RTX A6000,L40S,L40}" "${HOURS:-2}" \
  -- bash -c "$CMD"
