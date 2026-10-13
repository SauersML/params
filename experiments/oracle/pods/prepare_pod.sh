#!/usr/bin/env bash
# Question rows for the SFT (#2951 graph oracle: components.py prepare --rows-only) on rented GPUs. The texts of TASKS (a
# JSONL of {"task"[, "local"]}, e.g. teacher/joint_train/locality_train600.jsonl in the order given, non-local first), less
# those that already have a row in any components/train_prep_*/rows.jsonl or train_prep_pods/*/rows.jsonl, are dealt to
# PODS pods and on each to P processes (process k of pod j takes every (PODS P)-th text from the (j P + k)-th); each
# process writes its own directory, components/train_prep_pods/RUN-j_k/ (rows.jsonl, locality.jsonl; prepare keeps at
# least half of its kept texts non-local and resumes from its rows.jsonl). The pod puts VPD's checkpoint back together (the
# causal-importance network) and downloads the published atlas and connectome; Qwen3-8B's tokenizer comes from the hub.
# The rows come back to ~/mpd-data/runpod/oracle-graph-prepare-RUN-j/out/ and are copied into train_prep_pods/ every minute.
#   experiments/oracle/pods/prepare_pod.sh HOURS   (env: RUN, TASKS, PODS, P, REF, GPU, PRICE, VRAM, WAIT, ONLY)
# The whole script is one block, so bash has read all of it before it runs: editing the file cannot change a launch in progress.
{
cd /Users/user/params
H=${1:-2.0}
RUN=${RUN:-a} PODS=${PODS:-1} P=${P:-3}
TASKS=${TASKS:-/Users/user/mpd-data/graph_oracle/teacher/joint_train/locality_train600.jsonl}
C=/Users/user/mpd-data/graph_oracle/components
DEST=$C/train_prep_pods
GPU=${GPU:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A4500,NVIDIA RTX A6000,NVIDIA A40}
F=/Users/user/mpd-data/scratch/fleet/prepare_$RUN
V=/Users/user/mpd-data/vpd A=/Users/user/mpd-data/atlas/vpd4l
mkdir -p "$F" "$DEST"
COPYFILE_DISABLE=1 tar --exclude='texts/vpd4l/text91[0-4][0-9].json' --no-xattrs --no-mac-metadata -czf "$F/texts.tgz" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
/usr/bin/python3 -I - "$TASKS" "$C" "$PODS" "$P" "$F" <<'EOF' || exit 1
import json, pathlib, sys
tasks, comp, pods, procs, F = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), pathlib.Path(sys.argv[5])
done = set()
for d in sorted(comp.glob("train_prep_*")) + sorted((comp / "train_prep_pods").glob("*")):
    if (d / "rows.jsonl").exists():
        done |= {json.loads(x)["task"] for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()}
rows = [json.loads(x) for x in open(tasks) if x.strip()]
todo = [r for r in rows if r["task"] not in done]
for j in range(pods):
    for k in range(procs):
        (F / f"slice_{j}_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in todo[j * procs + k::pods * procs]))
print(f"{len(todo)} texts to prepare of {len(rows)} ({len(rows) - len(todo)} have rows), {pods} pods x {procs} processes")
EOF
names=()
for j in ${ONLY:-$(seq 0 $((PODS - 1)))}; do
  NAME=oracle-graph-prepare-$RUN-$j
  O=/Users/user/mpd-data/runpod/$NAME
  RUNS=""
  for k in $(seq 0 $((P - 1))); do
    RUNS="$RUNS (python -u method/oracle/components.py prepare train $O/out/$RUN-${j}_$k --tasks $F/slice_${j}_$k.jsonl --n 600 --rows-only > $O/prepare_$k.log 2>&1; echo \"exit \$?\" >> $O/prepare_$k.log) &"
  done
  CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/mpd-data/graph_oracle;
python -m pip install -q safetensors pyyaml tokenizers scipy transformers;
tar -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $A/atlas.json $A/atlas.safetensors $A/connectome /Users/user/params/method/vpd /Users/user/params/method/oracle > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; rm -f $V/s-55ea3f9b/split/model_400000_part0.safetensors $V/s-55ea3f9b/split/model_400000_part1.safetensors;
python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')\";
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
cd /Users/user/params/; set +e; $RUNS wait; echo all prepare processes done"
  (  # community stock comes and goes: no card in stock, or a corrupted download, is retried every 3 minutes for WAIT minutes
    for try in $(seq 1 $(( ${WAIT:-180} / 3 ))); do
      env ${REF:+RP_REF=$REF} RP_OWNER=fleet RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=${VRAM:-20} RP_MIN_RAM_GB=24 \
        experiments/infra/runpod/rp-run $NAME "$GPU" "$H" -- bash -c "$CMD" > "$F/rp_$j.log" 2>&1 && break
      grep -qE "in stock on a working host|differs from its sha256" "$F/rp_$j.log" || break
      cat "$F/rp_$j.log" >> "$F/rp_$j.nostock.log"; sleep 180
    done ) &
  names+=("$NAME")
  sleep 20
done
merge() {  # each process's directory has one writer: copy it whole
  for NAME in "${names[@]}"; do
    [ -d "/Users/user/mpd-data/runpod/$NAME/out" ] && rsync -a "/Users/user/mpd-data/runpod/$NAME/out/" "$DEST/"
  done
}
while [ -n "$(jobs -rp)" ]; do merge; sleep 60; done
merge
echo "prepare $RUN done: $(printf '%s ' "${names[@]}")"
exit
}
