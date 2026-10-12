#!/usr/bin/env bash
# The paired regression gate for teacher changes (#2951 graph oracle): two teacher configurations, A and B, on the first N
# held-out texts, compared text by text. Each of A and B is a directory of circuits (<task>.json, e.g. another search's) or
# COMMIT[,FLAG...] (a commit on origin/main and teacher.py flags, comma-separated: 787320c,--edges,--f1-start,both;
# v3@COMMIT[,FLAG...] runs teacher_v3.py, one process per card on V3PODS pods), which teacher_pod.sh teaches on pods (VPD's
# answers from vpd_heldout20.pt) into ~/mpd-data/graph_oracle/teacher/gate/NAME/a or b, both at once. Then one pod scores
# both sets with goal_eval.py score --diagnostics (VPD's checkpoint put back together on the pod: VPD's own answers, the
# baselines), goal_eval.py fidelity (C and adversary_flips, the team's held-out adversary, on the fresh pattern pool
# POOL, default goal_eval/ci_patterns_fresh.pt from text9150-9199) and teacher_v3.py --evaluate (F1, N and f1_adv under
# v3's own stand-in patterns, seed 1, pool 128), the three in parallel, and experiments/oracle/teacher_gate.py compares
# them. The scores come back to ~/mpd-data/runpod/oracle-graph-teacher-gscore-NAME/ (score.json, fid_a.jsonl,
# fid_b.jsonl, adv_a.jsonl, adv_b.jsonl), the comparison to gate.txt.
#   experiments/oracle/pods/teacher_gate.sh NAME A B   (env: N, PODS, P per side, V3PODS; POOL, GPU, PRICE)
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
NAME=$1 A=$2 B=$3
[ -n "$B" ] || { echo "usage: teacher_gate.sh NAME A B"; exit 2; }
N=${N:-20}
GPU=${GPU:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A4500,NVIDIA RTX A4000,NVIDIA RTX A6000,NVIDIA A40}
G=/Users/user/mpd-data/graph_oracle/teacher/gate/$NAME
F=/Users/user/mpd-data/scratch/fleet/gate_$NAME
rm -rf "$F" && mkdir -p "$F/in/a" "$F/in/b" "$G"
dirs=()
for side in a b; do
  spec=$A; [ $side = b ] && spec=$B
  if [ -d "$spec" ]; then echo "$side: circuits in $spec" | tee -a "$G/README.txt"; dirs+=("$spec"); continue; fi
  IFS=, read -r ref flags <<< "$spec"
  script=teacher.py pods=${PODS:-2} procs=${P:-4}
  case $ref in v3@*) ref=${ref#v3@} script=teacher_v3.py pods=${V3PODS:-4} procs=1 ;; esac
  C=$(git rev-parse --verify "$ref^{commit}") || exit 2
  echo "$side: $script at ${C:0:10} ${flags//,/ }" | tee -a "$G/README.txt"
  SCRIPT=$script RUN=g$NAME$side REF=$C TEACHER_ARGS="${flags//,/ }" SPLIT=heldout N=$N PODS=$pods P=$procs VRAM=16 GPU="$GPU" \
    VPD=/Users/user/mpd-data/graph_oracle/teacher/vpd_heldout20.pt DEST=$G/$side experiments/oracle/pods/teacher_pod.sh 2.0 > "$F/teach_$side.log" 2>&1 &
  dirs+=("$G/$side")
done
wait
for i in 0 1; do
  side=a; [ $i = 1 ] && side=b
  find "${dirs[$i]}" -maxdepth 1 -name 'text*.json' | grep -E '/text[0-9]+\.json$' | while read -r f; do cp "$f" "$F/in/$side/"; done
  echo "$side: $(ls "$F/in/$side" | wc -l | tr -d ' ') circuits from ${dirs[$i]}"
done
/usr/bin/python3 -I -c "import json, pathlib, sys; d = pathlib.Path('/Users/user/mpd-data/graph_oracle/texts/vpd4l'); f = [p for p in d.glob('*.json') if not p.name.startswith('.')]; print('\n'.join([p.stem for p in sorted(f, key=lambda p: int(p.stem[4:])) if json.loads(p.read_text())['split'] == 'heldout'][:int(sys.argv[1])]))" "$N" > "$F/in/tasks.txt"
cp /Users/user/mpd-data/graph_oracle/teacher/vpd_heldout20.pt "$F/vpd.pt"
cp "${POOL:-/Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_fresh.pt}" "$F/pool.pt"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/in.tgz" -C "$F/in" a b tasks.txt -C /Users/user/mpd-data/graph_oracle texts/vpd4l
rm -rf "$F/in"
V=/Users/user/mpd-data/vpd
RN=oracle-graph-teacher-gscore-$NAME
O=/Users/user/mpd-data/runpod/$RN
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$RN/mpd-data ~/mpd-data; mkdir -p $O ~/mpd-data/graph_oracle ~/gate_in;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $F/in.tgz -C ~/gate_in; mv ~/gate_in/texts ~/mpd-data/graph_oracle/;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/method/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
set +e; (python -B goal_eval.py score --set a=\$HOME/gate_in/a --set b=\$HOME/gate_in/b --split heldout --n $N --diagnostics --out $O/score.json > $O/score.log 2>&1) &
for s in a b; do (python -B goal_eval.py fidelity --split heldout --n $N --teacher \$HOME/gate_in/\$s --pool $F/pool.pt --vpd $F/vpd.pt --out $O/fid_\$s.jsonl > $O/fid_\$s.log 2>&1) & done;
(for s in a b; do python -B teacher_v3.py --evaluate \$HOME/gate_in/\$s --seed 1 --pool 128 --tasks \$HOME/gate_in/tasks.txt --vpd $F/vpd.pt --out $O/adv_\$s.jsonl > $O/adv_\$s.log 2>&1; done) & wait"
RP_OWNER=fleet RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=16 RP_MIN_RAM_GB=16 \
  experiments/infra/runpod/rp-run $RN "$GPU" 3.0 -- bash -c "$CMD" > "$F/rp_score.log" 2>&1
[ -s "$O/score.json" ] || { echo "no scores: $O/score.log, $F/rp_score.log"; exit 1; }
{ cat "$G/README.txt" 2> /dev/null; /usr/bin/python3 -I experiments/oracle/teacher_gate.py "$O/score.json" a b --teacher "${dirs[0]}" "${dirs[1]}" --adv "$O/adv_a.jsonl" "$O/adv_b.jsonl" --fidelity "$O/fid_a.jsonl" "$O/fid_b.jsonl"; } | tee "$O/gate.txt"
exit
}
