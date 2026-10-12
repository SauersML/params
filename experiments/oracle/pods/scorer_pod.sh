#!/usr/bin/env bash
# The component explainer's RL reward on a rented GPU (#2951 graph oracle): rl/serve_components.py (goal_eval in VPD's
# semantics, its circuit term with the adversary's forward and backward passes) serving POST /score on the pod's
# port 8767, for tinker_train.py rl running on the Mac, which reaches it through an SSH tunnel. The question rows and
# texts/vpd4l go up as one tar file, origin/main's method/oracle and method/vpd as another; VPD's files (its checkpoint
# with the causal-importance network, in two pieces put back together on the pod, and its subcomponents) are published
# inputs; C's pattern pool goes up with the
# rows (components.POOL). The scorer runs until the cap.
#   experiments/oracle/pods/scorer_pod.sh ROWS.jsonl HOURS   (env: RUN, GPU, PRICE, VRAM, WORKERS, RP_OWNER, VARIABLES=1: E4,
#                                                            SRCREV: the source revision, default origin/main)
# Two scoring processes need a 48 GB card (the default): on 24 GB a long text ran out of memory (RL v1, 10-12).
#   bash experiments/oracle/pods/scorer_tunnel.sh $RUN   the tunnel (then --scorer http://127.0.0.1:8767/score)
cd /Users/user/params
ROWS=$1 H=${2:-6}
RUN=${RUN:-a}
N=oracle-graph-scorer-$RUN
O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/primerl/src_scorer_$RUN
IN=/Users/user/mpd-data/scratch/primerl/in_scorer_$RUN
REV=${SRCREV:-origin/main}  # a resumed RL run keeps its reward's revision
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" method/oracle method/vpd | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"  # rp-run uploads every existing path the command names: only the archives exist here
rm -rf "$IN" "$IN.tar" && mkdir -p "$IN" && cp "$ROWS" "$IN/rows.jsonl"
COPYFILE_DISABLE=1 tar -cf "$IN.tar" -C "$IN" rows.jsonl -C /Users/user/mpd-data/graph_oracle texts/vpd4l goal_eval/ci_patterns_m.pt && rm -rf "$IN"
I='~/mpd-data/in'  # the inputs on the pod (unpacked from $IN.tar; a Mac path here would be uploaded file by file)
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S $I ~/mpd-data/graph_oracle; tar -xf $IN.tar -C $I; mv $I/texts $I/goal_eval ~/mpd-data/graph_oracle/; tar -xf $S.tar -C $S; \
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; \
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('/Users/user/mpd-data/vpd/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('/Users/user/mpd-data/vpd/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; \
cd $S/method/oracle; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > $O/gpu.csv & \
python rl/serve_components.py $I/rows.jsonl --port 8767 --workers ${WORKERS:-2} --device cuda ${VARIABLES:+--variables} > $O/serve.log 2>&1"
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-200} RP_OWNER=${RP_OWNER:-fleet} RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-1.0} \
  RP_MIN_VRAM_GB=${VRAM:-44} experiments/infra/runpod/rp-run $N "${GPU:-RTX A6000,A40,L40S,L40}" $H -- bash -c "$CMD"
