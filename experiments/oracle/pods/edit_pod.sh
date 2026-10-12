#!/usr/bin/env bash
# Edit questions (#2951 graph oracle, method/oracle/edit_eval.py) on one rented GPU: the goals of the first N held-out
# texts (edit_eval.py goals), the edit teacher, the native-unit search and VPD's top-k on each goal (P processes sharing
# the GPU, edit_eval.py teach --offset/--stride), and the persistent copy goal (edit_eval.py copy) beside them. The
# Python is method/oracle and method/vpd at origin/main as rp-run unpacks them; texts/vpd4l and the held-out teacher
# circuits go up as compressed tars, the Pile rows (edit_eval.py tokens) and the copy set as files (where rp-run puts them); VPD's checkpoint
# (its causal-importance network ranks VPD's subcomponents) is put back together from its two published pieces.
# Everything comes back to ~/mpd-data/runpod/NAME/out.
# FROM=DIR (an earlier run's out directory): its goals and finished teach rows go up first, and the goals done there
# are skipped.
# ONLY="1 2" runs those of the P processes alone; COPY=0 leaves out the copy goal.
#   experiments/oracle/pods/edit_pod.sh [NAME] [HOURS]      env: N, P, FROM, ONLY, COPY, GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
NAME=${1:-oracle-graph-edit} H=${2:-2}
N=${N:-20} P=${P:-3}
O=/Users/user/mpd-data/runpod/$NAME W=/workspace/runs/$NAME V=/Users/user/mpd-data/vpd G=/Users/user/mpd-data/graph_oracle
F=/Users/user/mpd-data/scratch/rustmove/edit/pod
mkdir -p "$F"
COPYFILE_DISABLE=1 tar -czf "$F/texts.tgz" -C $G texts/vpd4l
COPYFILE_DISABLE=1 tar -czf "$F/teacher.tgz" -C $G teacher/joint_heldout
CP="(python -B edit_eval.py copy $O/out/copy.json > $O/copy.log 2>&1; echo \"exit \$?\" >> $O/copy.log) &"
[ "${COPY:-1}" = 0 ] && CP=""
PRIOR=""
if [ -n "${FROM:-}" ]; then
  rm -rf "$F/prior"; mkdir -p "$F/prior"
  cp "$FROM/goals.jsonl" "$F/prior/goals.jsonl"; PRIOR="cp $F/prior/goals.jsonl $O/out/goals.jsonl;"
  for k in $(seq 0 $((P - 1))); do
    if [ -f "$FROM/teach_$k.jsonl" ]; then cp "$FROM/teach_$k.jsonl" "$F/prior/teach_$k.jsonl"; PRIOR="$PRIOR cp $F/prior/teach_$k.jsonl $O/out/teach_$k.jsonl;"; fi
  done
fi
RUNS="" ROWS=""
for k in ${ONLY:-$(seq 0 $((P - 1)))}; do
  RUNS="$RUNS (python -B edit_eval.py teach $O/out/goals.jsonl $O/out/teach_$k.jsonl --offset $k --stride $P > $O/teach_$k.log 2>&1; echo \"exit \$?\" >> $O/teach_$k.log) &"
  ROWS="$ROWS $O/out/teach_$k.jsonl"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O/out;
python -m pip install -q safetensors pyyaml tokenizers scipy pytest;
tar -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle; tar -xzf $F/teacher.tgz -C ~/mpd-data/graph_oracle;
ls $G/edit/tokens.npz $G/headfix_fresh.jsonl $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/params/method/vpd > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/method/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
python -B -m pytest -q test_edit_eval.py > $O/test.log 2>&1 || echo STAGE tests FAILED;
$PRIOR [ -s $O/out/goals.jsonl ] || python -B edit_eval.py goals $O/out/goals.jsonl --split heldout --n $N > $O/goals.log 2>&1;
set +e; $CP $RUNS wait;
cat $ROWS > $O/out/teach.jsonl; python -B edit_eval.py table $O/out/teach.jsonl $O/out/copy.json > $O/out/table.txt 2>&1; echo all stages done"
RP_OWNER=rustmove RP_PARALLEL=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=20 RP_MIN_RAM_GB=32 RP_DISK_GB=48 \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA A40,NVIDIA RTX A6000,NVIDIA RTX A4500,NVIDIA RTX 4000 Ada Generation,NVIDIA L4}" $H -- bash -c "$CMD"
}
