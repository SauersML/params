#!/usr/bin/env bash
# tools_env.py on a rented GPU (#2951 graph oracle, the tool-using oracle environment): its tests on the tiny model,
# each tool's latency (bench), the scripted baseline's episodes scored by reward v3 on the score split, and the token
# counts of an episode against the one-shot question. The reward's C is slow and large on the Mac (more than 16 GiB of
# footprint per text), so the episodes run here. The rows, texts/vpd4l (the sealed texts left out) and C's pattern pool
# go up as one tar file; VPD's checkpoint with the causal-importance network is put back together on the pod (as
# eval_pod.sh does), the atlas and the routed lens come as inputs. Outputs: ~/mpd-data/runpod/oracle-graph-toolenv-RUN/.
#   bash experiments/oracle/pods/tools_env_pod.sh RUN ROWS.jsonl TASKS   (TASKS e.g. text9000:text9019)
#     env: HOURS (cap, default 1.5), GPU, PRICE, VRAM, SPLIT (the score split, default eval), BENCH (texts, default
#     text9000,text9001), WORKERS (scripted processes sharing the GPU, default 2)
{
cd /Users/user/params
spent=$(awk -F'\t' '{s += $6} END {printf "%.2f", s}' /Users/user/mpd-data/runpod/LEDGER.tsv)
if awk -v s="$spent" 'BEGIN {exit !(s > 340)}'; then
  echo "tools_env_pod.sh: the team's RunPod ledger is at \$$spent (> \$340): stop and report" >&2; exit 2
fi
RUN=$1 ROWS=$2 TASKS=$3
N=oracle-graph-toolenv-$RUN
O=/Users/user/mpd-data/runpod/$N
F=/Users/user/mpd-data/scratch/toolenv/in_$RUN
V=/Users/user/mpd-data/vpd A=/Users/user/mpd-data/atlas/vpd4l PA='~/mpd-data/atlas/vpd4l'  # PA: the atlas directory on the pod (naming the Mac's would upload all of it)
I='~/mpd-data/in'
W=${WORKERS:-2}
rm -rf "$F" && mkdir -p "$F" && cp "$ROWS" "$F/rows.jsonl" || exit 1
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --exclude='texts/vpd4l/text91[0-4][0-9].json' -cf "$F/in.tar" -C "$F" rows.jsonl \
  -C /Users/user/mpd-data/graph_oracle texts/vpd4l goal_eval/ci_patterns_m.pt || exit 1
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf "$F/lens.tar" -C /Users/user/mpd-data/atlas/vpd4l lens/jacobians.pt lens/routed.pt || exit 1
SLICES=$(/usr/bin/python3 -I - "$TASKS" "$W" <<'EOF'
import sys
spec, w = sys.argv[1], int(sys.argv[2])
ts = []
for part in spec.split(","):
    if ":" in part:
        a, b = (int(x.removeprefix("text")) for x in part.split(":"))
        ts += [f"text{i}" for i in range(a, b + 1)]
    else:
        ts.append(part)
assert not any(9100 <= int(t[4:]) <= 9149 for t in ts), "sealed texts"
print(" ".join(",".join(ts[k::w]) for k in range(w)))
EOF
) || exit 1
read -r -a SLICE <<< "$SLICES"
RUNS=""
for k in $(seq 0 $((W - 1))); do
  RUNS="$RUNS (python -u method/oracle/tools_env.py scripted $I/rows.jsonl --tasks ${SLICE[$k]} --out $O/scripted_$k.jsonl --score-split ${SPLIT:-eval} > $O/scripted_$k.log 2>&1; echo \"exit \$?\" >> $O/scripted_$k.log) &"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $I ~/mpd-data/graph_oracle $PA; \
tar -xf $F/in.tar -C $I; mv $I/texts $I/goal_eval ~/mpd-data/graph_oracle/; tar -xf $F/lens.tar -C $PA; \
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $A/atlas.json $A/atlas.safetensors $A/tokens_0_6000.npy /Users/user/params/method/vpd /Users/user/params/method/oracle > /dev/null; \
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; \
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $O/gpu_name.txt; \
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &); \
cd /Users/user/params/; python -u method/oracle/test_tools_env.py > $O/tests.log 2>&1; echo \"exit \$?\" >> $O/tests.log; \
python -u method/oracle/tools_env.py bench $I/rows.jsonl --tasks ${BENCH:-text9000,text9001} --score-split ${SPLIT:-eval} > $O/bench.log 2>&1; echo \"exit \$?\" >> $O/bench.log; \
$RUNS wait; \
cat $O/scripted_*.jsonl > $O/scripted.jsonl; \
python -u method/oracle/tools_env.py tokens $I/rows.jsonl --tasks $TASKS --episodes $O/scripted.jsonl --pool > $O/tokens.log 2>&1; echo \"exit \$?\" >> $O/tokens.log"
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-400} RP_OWNER=toolenv RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.6} \
  RP_MIN_VRAM_GB=${VRAM:-44} experiments/infra/runpod/rp-run $N "${GPU:-RTX A6000,A40,L40S,L40}" ${HOURS:-1.5} -- bash -c "$CMD"
exit
}
