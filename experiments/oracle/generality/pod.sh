#!/usr/bin/env bash
# The generality runs' remaining items on one rented GPU (pod_worker.py: transfer targets, core_random backfills and copy
# augmentations, one process per item, P workers by the card's memory, ~7.5 GB each with VPD's causal-importance network).
# The items not yet in DATA's rows nor in an earlier pod run's outputs go up in DATA/todo.tsv with the scripts and inputs;
# outputs come back to ~/mpd-data/runpod/$NAME/out/ (analyze.py reads them there).
#   experiments/oracle/generality/pod.sh HOURS     env: NAME, GPU, PRICE, VRAM, RAM, REF, TRIES
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
H=${1:-1.5}
S=/Users/user/params/experiments/oracle/generality
D=${GEN_DATA:-/Users/user/mpd-data/graph_oracle/generality}
G=/Users/user/mpd-data/scratch/generality
V=/Users/user/mpd-data/vpd
NAME=${NAME:-oracle-graph-generality}
O=/Users/user/mpd-data/runpod/$NAME
GPU=${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA RTX A4500,NVIDIA RTX 4000 Ada Generation}
mkdir -p "$G"
GEN_DATA=$D MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python - <<'PY' || exit 1
import json, os
from pathlib import Path
D = Path(os.environ["GEN_DATA"])
pods = sorted(Path("/Users/user/mpd-data/runpod").glob("oracle-graph-generality*/out/*.jsonl"))
fams = json.loads((D / "families.json").read_text())
done, cr, aug = set(), set(), set()
for f in [D / x for x in ("transfer_a.jsonl", "transfer_b.jsonl", "transfer_backfill.jsonl", "transfer_rev.jsonl")] + [p for p in pods if p.name.startswith("transfer")]:
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            done.add((r["family"], r["target"]))
            if r["kind"] == "core_random":
                cr.add((r["family"], r["target"]))
for f in [D / "copy_augment.jsonl"] + [p for p in pods if p.name.startswith("copy_augment")]:
    if f.exists():
        aug |= {json.loads(line)["task"] for line in f.read_text().splitlines()}
lines = [f"transfer\t{f}\t{t}" for f, ts in fams.items() for t in ts if (f, t) not in done]
lines += [f"backfill\t{f}\t{t}" for f, t in sorted(done - cr)]
held = [x["task"] for x in json.loads((D / "copy_heldout.json").read_text())["copy"]] if (D / "copy_heldout.json").exists() else []
copies = [x["task"] for x in json.loads((D / "copytest.json").read_text()) if x["drop_nats"] >= 1 and x["new_gain_nats"] >= 1]
lines += [f"augment\t{t}" for t in held + copies if t not in aug]
(D / "todo.tsv").write_text("".join(x + "\n" for x in lines))
print(len(lines), "items")
PY
[ -s "$D/todo.tsv" ] || { echo "nothing to run"; exit 0; }
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$G/code.tgz" -C "$S" transfer.py gscore.py copy_augment.py pod_worker.py \
  -C "$D" rows.pkl families.json split.json copytest.json copy_heldout.json todo.tsv
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$G/texts.tgz" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
(cd /Users/user/mpd-data/graph_oracle && find teacher/joint_train teacher/joint_heldout -maxdepth 1 -name 'text[0-9]*.json' ! -name '*.*.json' > "$G/teacher_files.txt")
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$G/teacher.tgz" -C /Users/user/mpd-data/graph_oracle -T "$G/teacher_files.txt"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/mpd-data/graph_oracle ~/gen;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $G/texts.tgz -C ~/mpd-data/graph_oracle; tar -xzf $G/teacher.tgz -C ~/mpd-data/graph_oracle; tar -xzf $G/code.tgz -C ~/gen;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; rm -f $V/s-55ea3f9b/split/model_400000_part0.safetensors $V/s-55ea3f9b/split/model_400000_part1.safetensors;
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
export ORACLE=/Users/user/params/method/oracle GEN_DATA=\$HOME/gen PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
P=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{p = int(\$1 / 9000); print (p < 1) ? 1 : p}');
echo workers \$P;
cd ~/gen; set +e; for k in \$(seq 0 \$((P - 1))); do (python -B pod_worker.py \$k \$P $O/out > $O/worker_\$k.log 2>&1; echo \"exit \$?\" >> $O/worker_\$k.log) & done; wait; echo all workers done"
for try in $(seq 1 ${TRIES:-20}); do
  env ${REF:+RP_REF=$REF} RP_OWNER=generality RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=${VRAM:-20} RP_MIN_RAM_GB=${RAM:-24} \
    experiments/infra/runpod/rp-run $NAME "$GPU" "$H" -- bash -c "$CMD" > "$G/rp.log" 2>&1 && break
  grep -qE "in stock on a working host|differs from its sha256" "$G/rp.log" || break
  cat "$G/rp.log" >> "$G/rp.nostock.log"; sleep 180
done
echo "generality pod done"
exit
}
