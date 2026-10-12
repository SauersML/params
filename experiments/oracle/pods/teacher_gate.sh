#!/usr/bin/env bash
# The paired regression gate for teacher changes (#2951 graph oracle): two teacher configurations, A and B, on the first N
# held-out texts, compared text by text. Each of A and B is a directory of circuits (<task>.json, e.g. another search's) or
# COMMIT[,FLAG...] (a commit on origin/main and teacher.py flags, comma-separated, a comma inside a flag's value written
# :: as in --edge-levels,writer::connection: 787320c,--edges,--f1-start,both;
# v3@COMMIT[,FLAG...] runs teacher_v3.py, one process per card on V3PODS pods; reads@COMMIT[,FLAG...] teacher_reads.py,
# READP per card on READPODS pods), which teacher_pod.sh teaches on pods (VPD's
# answers from vpd_heldout20.pt) into ~/mpd-data/graph_oracle/teacher/gate/NAME/a or b, both at once. Then one pod scores
# both sets with goal_eval.py score --diagnostics (VPD's checkpoint put back together on the pod: VPD's own answers, the
# baselines) and goal_eval.py fidelity on both sets in one run (C, adversary_flips' omitted lever, the guarded scores,
# VPD's top k at each circuit's size; the search pool goal_eval/ci_patterns_m.pt, the fresh pool for the evaluation-only
# pattern lever goal_eval/ci_patterns_fresh.pt), sharded over FIDP processes beside the score run, and
# experiments/oracle/teacher_gate.py compares them. The scores come back to
# ~/mpd-data/runpod/oracle-graph-teacher-gscore-NAME/ (score.json, fid_<k>.jsonl), the comparison to gate.txt.
#   experiments/oracle/pods/teacher_gate.sh NAME A B   (env: N, PODS, P per side, V3PODS, READPODS, READP, VRAM; FIDP, GPU, PRICE,
#   SCORE_GPU, SCORE_VRAM (the scoring card, default 44 GB: fidelity with donors overflows 24 GB even alone); the
#   adversary's search pool goes to the teaching pods for --repair, --edge-objective robust and teacher_v3)
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
  flags=${flags//,/ } flags=${flags//::/,}               # a comma inside one flag's value is written ::
  script=teacher.py pods=${PODS:-2} procs=${P:-4}
  case $ref in
    v3@*) ref=${ref#v3@} script=teacher_v3.py pods=${V3PODS:-4} procs=1 ;;
    reads@*) ref=${ref#reads@} script=teacher_reads.py pods=${READPODS:-4} procs=${READP:-2} ;;
  esac
  C=$(git rev-parse --verify "$ref^{commit}") || exit 2
  echo "$side: $script at ${C:0:10} $flags" | tee -a "$G/README.txt"
  SCRIPT=$script RUN=g$NAME$side REF=$C TEACHER_ARGS="$flags" SPLIT=heldout N=$N PODS=$pods P=$procs VRAM=${VRAM:-16} GPU="$GPU" \
    EXTRA=/Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt \
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
cp /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt "$F/pool.pt"
cp /Users/user/mpd-data/graph_oracle/goal_eval/ci_patterns_fresh.pt "$F/fresh.pt"
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
big=\$(( \$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) >= 44000 ));
score() { python -B goal_eval.py score --set a=\$HOME/gate_in/a --set b=\$HOME/gate_in/b --split heldout --n $N --diagnostics --out $O/score.json > $O/score.log 2>&1; };
fid() { rm -f $O/fid_\$1.jsonl; python -B goal_eval.py fidelity --split heldout --n $N --set a=\$HOME/gate_in/a --set b=\$HOME/gate_in/b --pool $F/pool.pt --fresh $F/fresh.pt --changed 16 --vpd $F/vpd.pt --offset \$1 --stride ${FIDP:-2} --out $O/fid_\$1.jsonl > $O/fid_\$1.log 2>&1; };
set +e; (score) &
[ \$big = 1 ] || wait;
for k in \$(seq 0 $(( ${FIDP:-2} - 1 ))); do (fid \$k) & [ \$big = 1 ] || wait; done; wait;
grep -q OutOfMemoryError $O/score.log && { echo 'score run out of memory beside the others: again alone'; score; };
for k in \$(seq 0 $(( ${FIDP:-2} - 1 ))); do grep -q OutOfMemoryError $O/fid_\$k.log && { echo \"fidelity shard \$k out of memory beside the others: again alone\"; fid \$k; }; done; true"
# fidelity on a 264-node circuit holds ~16 GB, the score run ~8 GB: on a 48 GB card they run side by side, on a 24 GB
# card one after another (two fidelity runs and the score run overflowed 24 GB); a run out of memory beside the others
# runs again alone (the elv gate's second shard, 180-node circuits, overflowed a 44 GB A40 and left 10 of 20 texts
# scored); no card in stock is retried every 3 min
for try in $(seq 1 40); do
  RP_OWNER=fleet RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.60} RP_MIN_VRAM_GB=${SCORE_VRAM:-44} RP_MIN_RAM_GB=16 \
    experiments/infra/runpod/rp-run $RN "${SCORE_GPU:-NVIDIA RTX A6000,NVIDIA A40,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090}" 3.0 -- bash -c "$CMD" > "$F/rp_score.log" 2>&1 && break
  grep -qE "in stock on a working host|differs from its sha256" "$F/rp_score.log" || break
  sleep 180
done
[ -s "$O/score.json" ] || { echo "no scores: $O/score.log, $F/rp_score.log"; exit 1; }
{ cat "$G/README.txt" 2> /dev/null; echo "scored at $(sed -nE 's/.* source ([0-9a-f]+):.*/\1/p' "$O/rp-run.log" | tail -1) (goal_eval score and fidelity; both arms at this commit)"; /usr/bin/python3 -I experiments/oracle/teacher_gate.py "$O/score.json" a b --teacher "${dirs[0]}" "${dirs[1]}" --fidelity $(for k in $(seq 0 $(( ${FIDP:-2} - 1 ))); do printf '%s ' "$O/fid_$k.jsonl"; done); } | tee "$O/gate.txt"
exit
}
