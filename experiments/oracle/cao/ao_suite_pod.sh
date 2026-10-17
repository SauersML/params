#!/usr/bin/env bash
# AO's own evaluation suite on oracle adapters (the paper's v1.0 code with fleet's Qwen3-8B harness in
# ~/mpd-data/scratch/fleet/ao: taboo, PersonaQA yes/no and open-ended, classification; the harness that reproduced the
# released AO's numbers, ao_scores.json), into ~/mpd-data/runpod/orc-primerl-aosuite-$RUN/<name>/:
#   RUN=s1 ADAPTERS="circ=/Users/user/mpd-data/runpod/.../step_340 none=..." bash experiments/oracle/cao/ao_suite_pod.sh [HOURS]
# then per adapter on the Mac (the result files carry the adapter directory's name):
#   AO_ORACLE=<name> python ~/mpd-data/scratch/fleet/ao/ao_score.py OUT/<name>/results ~/mpd-data/scratch/fleet/ao/paper_plots OUT/<name>/scores
# (env: GPU, PRICE)
set -euo pipefail
cd /Users/user/params
RUN=${RUN:?RUN} H=${1:-3}
N=orc-primerl-aosuite-$RUN O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/primerl/aosuite_$RUN F=/Users/user/mpd-data/scratch/fleet/ao
rm -rf "$S" && mkdir -p "$S/adapters"
EVALS=""
for x in ${ADAPTERS:?ADAPTERS name=path ...}; do
  n=${x%%=*}
  cp -R "${x#*=}" "$S/adapters/$n"
  EVALS="$EVALS mkdir -p $O/$n/results; for s in taboo_qwen personaqa_yesno_qwen personaqa_open_qwen classification_qwen; do \
AO_ORACLE=$S/adapters/$n python -I -u $F/\$s.py > $O/$n/\$s.log 2>&1; echo \"exit \$?\" >> $O/$n/\$s.log; done; \
cp -r experiments/taboo_eval_results experiments/personaqa_results experiments/classification $O/$n/results/ 2> /dev/null; \
rm -rf experiments/taboo_eval_results experiments/personaqa_results experiments/classification;"
done
CMD="set -e; export HF_HUB_ENABLE_HF_TRANSFER=1; mkdir -p $O; rm -rf /workspace/ao; mkdir -p /workspace/ao/repo; cd /workspace/ao; \
curl -sL --max-time 600 https://github.com/adamkarvonen/activation_oracles/archive/refs/tags/v1.0-paper.tar.gz | tar -xz --strip-components=1 -C repo; \
python -m pip install -q 'transformers==4.55.2' 'peft==0.17.1' 'accelerate==1.10.1' 'datasets==3.6.0' 'pydantic==2.11.7' 'numpy<2' \
'pyarrow==16.1.0' pandas tqdm huggingface_hub hf_transfer einops jaxtyping; python -m pip install -q --no-deps -e /workspace/ao/repo; \
python -m pip install -q 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl' || true; \
python -I -c 'import flash_attn' && export AO_SDPA=0 || export AO_SDPA=1; echo AO_SDPA=\$AO_SDPA > $O/env.txt; \
(while :; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> $O/gpu.txt; sleep 30; done) > /dev/null 2>&1 & \
set +e; cd /workspace/ao/repo; ls $F/taboo_qwen.py $F/personaqa_yesno_qwen.py $F/personaqa_open_qwen.py $F/classification_qwen.py > /dev/null; $EVALS true"
RP_UPLOAD_MAX_MB=${UPLOAD_MB:-3000} RP_OWNER=primerl RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.8} RP_MIN_VRAM_GB=${VRAM:-44} \
  experiments/infra/runpod/rp-run $N "${GPU:-A40,RTX A6000,L40S,RTX 6000 Ada}" "$H" -- bash -c "$CMD"
