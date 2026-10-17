#!/usr/bin/env bash
# zero_a_pod.sh: run A of the Zero study on one rented 48 GB card, every stage in one command (stops at the first
# failure; the pod is deleted when the command ends):
#   Zero's replies (search x2, held-out x2, general x1) and Qwen3-8B's assistant replies (search, held-out; text only);
#   zattr.py: means, attribution doublings 8..512 with measured scores, random matched sets, held-out score, KL;
#   the gates' generation (held-out x4 under every set's mean ablation and unedited);
#   zjudge.py on all of it (Qwen3-14B, the report's section 6 rubric).
#   NAME=orc-persona-zeroA bash zero_a_pod.sh
N=${NAME:-orc-persona-zeroA}
O=/Users/user/mpd-data/runpod/$N/out
D=/Users/user/mpd-data/persona/zero
CMD="cd /Users/user/params/experiments/persona/zero && mkdir -p $O && \
python -u zgen.py zero $D/prompts.json $O/gen_zero.json --groups search:2,heldout:2,general:1 && \
python -u zgen.py asst $D/prompts.json $O/gen_asst.json --model Qwen/Qwen3-8B --groups search:1,heldout:1 && \
python -u zattr.py $D/prompts.json $O/gen_zero.json $O/gen_asst.json $D/atlas_sample.npz $O && \
python -u zgen.py zero $D/prompts.json $O/gen_gates.json --groups heldout:4 --sets $O/sets.json --means $O/means.pt && \
python -u zjudge.py $O/judge.json $D/prompts.json $O/gen_gates.json $O/gen_zero.json"
cd /Users/user/params || exit 1
RP_HF_MODELS="movingcastles/zero Qwen/Qwen3-8B Qwen/Qwen3-14B" RP_IDLE_PCT=0 RP_OWNER=persona RP_PARALLEL=1 RP_PYENV=oracle \
  RP_MAX_PRICE=${PRICE:-0.6} RP_MIN_VRAM_GB=44 experiments/infra/runpod/rp-run $N "${GPU:-A40,RTX A6000,L40S,L40}" "${HOURS:-4}" \
  -- bash -c "$CMD"
