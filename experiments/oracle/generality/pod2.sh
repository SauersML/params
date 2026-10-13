#!/usr/bin/env bash
# Commands on one rented GPU (pod_worker2.py: DATA/todo2.tsv, one command per line, "script<TAB>arg<TAB>...", one
# process each, P workers by the card's memory at GB_PER MiB each). The scripts, DATA's inputs, the mechanism templates,
# the texts and the teacher circuits go up; outputs come back to ~/mpd-data/runpod/$NAME/out/ (the reports read them).
#   experiments/oracle/generality/pod2.sh HOURS     env: NAME, GPU, PRICE, VRAM, RAM, GB_PER, RAM_PER, REF, TRIES, EXTRA
# EXTRA: more inputs at their ~/mpd-data paths (falsify.py's CI-pattern pool: goal_eval/ci_patterns_train.pt).
# UNPACK: archives (under ~/mpd-data) unpacked into ~/mpd-data/graph_oracle on the pod (varteach's variable files).
# Workers: one per GB_PER MiB of the card and RAM_PER GiB of the host, whichever is fewer; rp-run stops a run whose GPU
# stays under 20% for 15 minutes, and one process of these (small model, Python between runs) holds it near 15%.
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
H=${1:-1.5}
S=/Users/user/params/experiments/oracle/generality
D=${GEN_DATA:-/Users/user/mpd-data/graph_oracle/generality}
M=${MECH:-/Users/user/mpd-data/graph_oracle/mechanisms}
G=/Users/user/mpd-data/scratch/generality
V=/Users/user/mpd-data/vpd
NAME=${NAME:-oracle-graph-generality3}
O=/Users/user/mpd-data/runpod/$NAME
GPU=${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA RTX A4500}
GB_PER=${GB_PER:-8000}
mkdir -p "$G"
TODO=${TODO:-todo2.tsv}
[ -s "$D/$TODO" ] || { echo "no $D/$TODO"; exit 1; }
echo "$(wc -l < "$D/$TODO") commands"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$G/code_$NAME.tgz" -C "$S" templates.py cscore.py copy_template.py library.py transfer.py falsify.py causalgym.py huang.py cg_teacher.py falsify_families.py var_transfer.py pod_worker2.py -C ../ var_masks.py var_controls.py \
  -C "$D" families.json copy_heldout.json copytest.json library_texts.json falsify_texts.json causalgym_tasks.json huang_used.json causalgym_texts.json causalgym_texts causalgym_texts_b.json causalgym_texts_b ffam_texts.json "$TODO" -C "$(dirname "$M")" "$(basename "$M")"
# the texts without the sealed final test (text9100-9149), one archive per run (launches running at once share none)
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --exclude 'texts/vpd4l/text91[0-4][0-9].json' -czf "$G/texts_$NAME.tgz" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
(cd /Users/user/mpd-data/graph_oracle && find teacher/joint_train teacher/joint_heldout -maxdepth 1 -name 'text[0-9]*.json' ! -name '*.*.json' > "$G/teacher_files.txt")
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$G/teacher.tgz" -C /Users/user/mpd-data/graph_oracle -T "$G/teacher_files.txt"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/mpd-data/graph_oracle ~/gen;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar -xzf $G/texts_$NAME.tgz -C ~/mpd-data/graph_oracle; tar -xzf $G/teacher.tgz -C ~/mpd-data/graph_oracle; tar -xzf $G/code_$NAME.tgz -C ~/gen;$(for u in ${UNPACK:-}; do printf ' tar -xzf %s -C ~/mpd-data/graph_oracle;' "$u"; done)
ls ${UNPACK:-} $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd ${EXTRA:-} > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; rm -f $V/s-55ea3f9b/split/model_400000_part0.safetensors $V/s-55ea3f9b/split/model_400000_part1.safetensors;
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
export TODO=$TODO ORACLE=/Users/user/params/method/oracle GEN_DATA=\$HOME/gen MECH=\$HOME/gen/$(basename "$M") PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
P=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{p = int(\$1 / $GB_PER); print (p < 1) ? 1 : p}');
R=\$(awk '/MemTotal/ {print int(\$2 / 1048576 / ${RAM_PER:-14})}' /proc/meminfo); [ \$R -ge 1 ] && [ \$R -lt \$P ] && P=\$R;
echo workers \$P;
cd ~/gen; set +e; for k in \$(seq 0 \$((P - 1))); do (python -B pod_worker2.py \$k \$P $O/out > $O/worker_\$k.log 2>&1; echo \"exit \$?\" >> $O/worker_\$k.log) & done; wait; echo all workers done"
for try in $(seq 1 ${TRIES:-20}); do
  env ${REF:+RP_REF=$REF} RP_OWNER=generality RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=${VRAM:-20} RP_MIN_RAM_GB=${RAM:-32} \
    experiments/infra/runpod/rp-run $NAME "$GPU" "$H" -- bash -c "$CMD" > "$G/rp_$NAME.log" 2>&1 && break
  grep -qE "in stock on a working host|differs from its sha256" "$G/rp_$NAME.log" || break
  cat "$G/rp_$NAME.log" >> "$G/rp.nostock.log"; sleep 180
done
echo "generality pod $NAME done"
exit
}
