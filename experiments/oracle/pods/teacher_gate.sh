#!/usr/bin/env bash
# The paired regression gate for teacher changes (#2951 graph oracle). teacher.py at CANDIDATE (a commit on origin/main)
# teaches the first N held-out texts on pods (teacher_pod.sh, VPD's answers from vpd_heldout20.pt) into
# ~/mpd-data/graph_oracle/teacher/gate/<commit>/, or CAND_DIR names circuits found otherwise (another search); then one pod
# scores the candidate's <task>.json and REFERENCE's with goal_eval.py score --diagnostics (VPD's checkpoint put back
# together on the pod: VPD's own answers, the baselines) and experiments/oracle/teacher_gate.py compares them text by text.
# The scores come back to ~/mpd-data/runpod/oracle-graph-teacher-gscore-<commit>/score.json, the comparison to gate.txt there.
#   experiments/oracle/pods/teacher_gate.sh CANDIDATE [REFERENCE_DIR]   (env: N, PODS, P, CAND_DIR, GPU, PRICE)
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
C=$(git rev-parse --verify "$1^{commit}") || exit 2
REFD=${2:-/Users/user/mpd-data/graph_oracle/teacher/joint_heldout_8eac5be}
N=${N:-20}
GPU=${GPU:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A4500,NVIDIA RTX A4000,NVIDIA RTX A6000,NVIDIA A40}
if [ -z "${CAND_DIR:-}" ]; then
  CAND_DIR=/Users/user/mpd-data/graph_oracle/teacher/gate/${C:0:10}
  RUN=gate${C:0:7} REF=$C SPLIT=heldout N=$N PODS=${PODS:-2} P=${P:-4} VRAM=16 GPU="$GPU" \
    VPD=/Users/user/mpd-data/graph_oracle/teacher/vpd_heldout20.pt DEST=$CAND_DIR experiments/oracle/pods/teacher_pod.sh 2.0
fi
F=/Users/user/mpd-data/scratch/fleet/gate_${C:0:10}
rm -rf "$F" && mkdir -p "$F/in/ref" "$F/in/cand"
for side in ref cand; do
  D=$REFD; [ $side = cand ] && D=$CAND_DIR
  find "$D" -maxdepth 1 -name 'text*.json' | grep -E '/text[0-9]+\.json$' | while read -r f; do cp "$f" "$F/in/$side/"; done
done
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/in.tgz" -C "$F/in" ref cand -C /Users/user/mpd-data/graph_oracle texts/vpd4l
rm -rf "$F/in"
V=/Users/user/mpd-data/vpd
NAME=oracle-graph-teacher-gscore-${C:0:7}
O=/Users/user/mpd-data/runpod/$NAME
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O ~/mpd-data/graph_oracle ~/gate_in;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $F/in.tgz -C ~/gate_in; mv ~/gate_in/texts ~/mpd-data/graph_oracle/;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/method/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
python -B goal_eval.py score --set ref=\$HOME/gate_in/ref --set cand=\$HOME/gate_in/cand --split heldout --n $N --diagnostics --out $O/score.json > $O/score.log 2>&1"
RP_OWNER=fleet RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=16 RP_MIN_RAM_GB=16 \
  experiments/infra/runpod/rp-run $NAME "$GPU" 1.5 -- bash -c "$CMD" > "$F/rp_score.log" 2>&1
[ -s "$O/score.json" ] || { echo "no scores: $O/score.log, $F/rp_score.log"; exit 1; }
/usr/bin/python3 -I experiments/oracle/teacher_gate.py "$O/score.json" ref cand | tee "$O/gate.txt"
exit
}
