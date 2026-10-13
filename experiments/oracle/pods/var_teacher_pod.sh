#!/usr/bin/env bash
# Explanation variables (#2951 graph oracle, method/oracle/var_teacher.py) on one rented GPU. The texts (TASKS, a file of
# task ids one per line; or the first N of SPLIT) are taught with each of VARIANTS (var_teacher.py flag sets separated
# by ';', default one run with none), from the teacher circuits in CIRCUITS; P processes share the GPU, process k taking every P-th text from its k-th. The
# Python is method/oracle and method/vpd at REF (default origin/main); texts/vpd4l and the circuits go up as compressed
# tars; vpd4l's weights and VPD's subcomponents are published inputs, and VPD's checkpoint (two published pieces under
# GitHub's 2 GiB asset limit; it holds the causal-importance network VPD's pool needs) is put back together on the pod.
# The variables come back to ~/mpd-data/runpod/oracle-graph-vars-$RUN/out/ (var_teacher.py --root) and are copied into
# ~/mpd-data/graph_oracle/variables/ every minute while the pod runs. HOURS caps the pod.
#   experiments/oracle/pods/var_teacher_pod.sh HOURS
#   env: RUN, SPLIT, N, TASKS, CIRCUITS, VARIANTS, DEST, P, REF, GPU, PRICE, VRAM, WAIT (minutes of retries while no card is
#   in stock)
#   INVARIANTS_TEACH=1 exports train-only proofs in out/invariant_teach/variant_<v>_worker_<k>.jsonl; select one
#   variant's sidecars whose groups match its retained variable files. Defaults do not emit supervision artifacts.
# The whole script is one block, so bash has read all of it before it runs: editing the file cannot change a launch in progress.
{
cd /Users/user/params
H=${1:-2.0}
RUN=${RUN:-a} SPLIT=${SPLIT:-heldout} N=${N:-20} P=${P:-3}
CIRCUITS=${CIRCUITS:-/Users/user/mpd-data/graph_oracle/teacher/joint_$SPLIT}
VARIANTS=${VARIANTS:-}
INVARIANTS_TEACH=${INVARIANTS_TEACH:-0}
GPU=${GPU:-NVIDIA RTX A5000,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A4500,NVIDIA RTX A6000,NVIDIA A40}
NAME=oracle-graph-vars-$RUN
F=/Users/user/mpd-data/scratch/varteach/pod_$RUN
V=/Users/user/mpd-data/vpd
O=/Users/user/mpd-data/runpod/$NAME
DEST=${DEST:-/Users/user/mpd-data/graph_oracle/variables}
mkdir -p "$F" "$DEST"
/usr/bin/python3 -I - "$SPLIT" "$N" "$F" "$CIRCUITS" "${TASKS:-}" /Users/user/mpd-data/graph_oracle/texts/vpd4l "$INVARIANTS_TEACH" <<'EOF' || exit 1
import json, pathlib, re, sys, tarfile
split, n, F, circuits, tasks, text_root, proofs = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]), sys.argv[5], pathlib.Path(sys.argv[6]), sys.argv[7]
if proofs not in ("0", "1"):
    raise ValueError("INVARIANTS_TEACH must be 0 or 1")
def permitted(t):
    match = re.fullmatch(r"text(\d+)", t)
    if not match:
        raise ValueError(f"invalid numeric task ID: {t}")
    if 9100 <= int(match[1]) <= 9149:
        raise ValueError(f"sealed task is not a launcher input: {t}")
def text_path(t):
    p = text_root / f"{t}.json"
    target = re.fullmatch(r"text(\d+)\.json", p.resolve().name)
    if target and 9100 <= int(target[1]) <= 9149:
        raise ValueError(f"task aliases a sealed text: {t}")
    return p
if tasks:
    ids = [x.strip() for x in open(tasks) if x.strip()]
    for t in ids:  # validate every explicit ID before reading any text's contents
        permitted(t)
        text_path(t)
else:
    # Filter by numeric filename metadata before reading split fields.
    files = [p for p in text_root.glob("*.json") if re.fullmatch(r"text\d+\.json", p.name)
             and not 9100 <= int(p.stem[4:]) <= 9149]
    files = [text_path(p.stem) for p in files]
    ids = [p.stem for p in sorted(files, key=lambda p: int(p.stem[4:])) if json.loads(p.read_text())["split"] == split][:n]
ids = [t for t in ids if (circuits / f"{t}.json").exists()]
if proofs == "1":
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs cannot share invariant supervision output")
    for t in ids:
        if not re.fullmatch(r"text[67]\d{3}", t) or json.loads(text_path(t).read_text()).get("split") != "train":
            raise ValueError("invariant supervision requires train text6000–7999")
(F / "tasks.txt").write_text("".join(t + "\n" for t in ids))
with tarfile.open(F / "texts.tgz", "w:gz", dereference=True) as tar:
    for t in ids:
        tar.add(text_path(t), arcname=f"texts/vpd4l/{t}.json", recursive=False)
with tarfile.open(F / "circuits.tgz", "w:gz") as tar:
    for t in ids:
        tar.add(circuits / f"{t}.json", arcname=f"circuits/{t}.json")
print(f"{len(ids)} texts with circuits in {circuits}")
EOF
RUNS=""
for k in $(seq 0 $((P - 1))); do
  MINE=$(awk "NR % $P == ($k + 1) % $P" "$F/tasks.txt" | tr '\n' ' ')  # process k's texts, named in the command
  RUNS="$RUNS (for t in $MINE; do v=0; echo '$VARIANTS' | tr ';' '\n' | while read -r flags; do
    proof=; if [ '$INVARIANTS_TEACH' = 1 ]; then proof=\"--invariants-teach-out $O/out/invariant_teach/variant_\${v}_worker_$k.jsonl\"; fi;
    python -B var_teacher.py \$t \$flags \$proof --circuits \$HOME/circuits --root $O/out < /dev/null >> $O/vars_$k.log 2>&1 || echo \"\$t \$flags failed \$?\" >> $O/vars_$k.log;
    v=\$((v + 1));
  done; done; echo \"exit 0\" >> $O/vars_$k.log) &"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/mpd-data/graph_oracle;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle; tar -xzf $F/circuits.tgz -C \$HOME;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd > /dev/null;
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; rm -f $V/s-55ea3f9b/split/model_400000_part0.safetensors $V/s-55ea3f9b/split/model_400000_part1.safetensors;
cd /Users/user/params/method/oracle;
set +e; $RUNS wait; echo all variable processes done"
(  # community stock comes and goes: a launch that finds no card in stock is retried every 3 minutes for WAIT minutes
  for try in $(seq 1 $(( ${WAIT:-60} / 3 ))); do
    env ${REF:+RP_REF=$REF} RP_OWNER=varteach RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=${VRAM:-20} RP_MIN_RAM_GB=16 \
      experiments/infra/runpod/rp-run $NAME "$GPU" "$H" -- bash -c "$CMD" > "$F/rp.log" 2>&1 && break
    grep -qE "in stock on a working host|differs from its sha256" "$F/rp.log" || break
    cat "$F/rp.log" >> "$F/rp.nostock.log"; sleep 180
  done ) &
while [ -n "$(jobs -rp)" ]; do [ -d "$O/out" ] && rsync -a --exclude '*.partial' "$O/out/" "$DEST/"; sleep 60; done
[ -d "$O/out" ] && rsync -a --exclude '*.partial' "$O/out/" "$DEST/"
echo "variables $RUN done"
exit
}
