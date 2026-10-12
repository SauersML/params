#!/usr/bin/env bash
# The pre-launch check (#2951 graph oracle, the lead's 10-12 rule): teacher.py arms on one held-out text on one cheap pod, then
# each arm's circuit scored by goal_eval.py fidelity --only track_pp on the eval split (the draws no search saw), so an arm
# that fits its search draws shows before anything is launched on it. Each ARM is NAME=FLAGS with commas for spaces
# (a comma inside a flag value written ::). Rows to ~/mpd-data/runpod/oracle-graph-check-RUN-0/eval.jsonl, a line per arm.
#   RUN=c9002 REF=COMMIT TEXT=text9002 experiments/oracle/pods/check_text.sh b=--f1-start,influential,--edges,--no-strand,--accept,changed ...
#   (env: GPU, PRICE, HOURS)
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
[ -n "${RUN:-}" ] && [ $# -ge 1 ] || { sed -n '2,7p' "$0"; exit 2; }
TEXT=${TEXT:-text9002} REF=${REF:-$(git rev-parse origin/main)}
RN=oracle-graph-check-$RUN-0
O=/Users/user/mpd-data/runpod/$RN
F=/Users/user/mpd-data/scratch/fleet/check_$RUN
mkdir -p "$F"
echo "$TEXT" > "$F/tasks.txt"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/texts.tgz" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
V=/Users/user/mpd-data/vpd
TEACH="" SETS=""
for spec in "$@"; do
  name=${spec%%=*} flags=${spec#*=}
  flags=${flags//,/ } flags=${flags//::/,}
  TEACH="$TEACH python -B teacher.py --tasks $F/tasks.txt --vpd /Users/user/mpd-data/graph_oracle/teacher/vpd_heldout20.pt $flags --max-tokens 0 --batch 32 --out $O/$name > $O/teach_$name.log 2>&1 || echo \"$name failed\";"
  SETS="$SETS --set $name=$O/$name"
done
N=$(( 10#${TEXT#text} - 9000 + 1 ))
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$RN/mpd-data ~/mpd-data; mkdir -p $O ~/mpd-data/graph_oracle;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt > /dev/null;
cd /Users/user/params/method/oracle; set +e; $TEACH
python -B goal_eval.py fidelity --split heldout --n $N $SETS --pool /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt --vpd /Users/user/mpd-data/graph_oracle/teacher/vpd_heldout20.pt --only track_pp --out $O/eval.jsonl > $O/eval.log 2>&1; cat $O/eval.jsonl"
for try in $(seq 1 40); do
  RP_REF=$REF RP_OWNER=fleet RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=20 RP_MIN_RAM_GB=48 \
    experiments/infra/runpod/rp-run $RN "${GPU:-NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA RTX A4500,NVIDIA GeForce RTX 4090,NVIDIA RTX A6000,NVIDIA A40}" "${HOURS:-2.0}" -- bash -c "$CMD" > "$F/rp.log" 2>&1 && break
  grep -qE "in stock on a working host|differs from its sha256" "$F/rp.log" || break
  sleep 180
done
python3 - "$O/eval.jsonl" <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f"{r['kind']:16s} nodes {r.get('nodes')} eval s_track_pp {r.get('s_track_pp')}")
EOF
exit
}
