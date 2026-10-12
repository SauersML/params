#!/usr/bin/env bash
# Backup expansion (#2951 graph oracle, method/oracle/backup.py) on one rented GPU: each text's circuit in SRC (default
# the influential start sets) expanded by its compensator blocks (backup.py run), then goal_eval's fidelity in full mode
# on the eval split's draws for the start, the expansions and the equal-size trivial baselines (backup.py eval). P
# processes share the card, process k taking every P-th of the first N texts from the k-th. VPD's checkpoint is put back
# together on the pod (its causal-importance network gives VPD's answers and order). Outputs come back to
# ~/mpd-data/runpod/oracle-graph-backup-RUN/out_<k>/ and are merged into OUT (sets, summary.jsonl, fidelity.jsonl).
#   RUN=a N=20 P=2 experiments/oracle/pods/backup_pod.sh HOURS
#   env: SRC, OUT (default ~/mpd-data/graph_oracle/backup/RUN), SPLIT (heldout), STEPS (backup.py --max-steps), TESTS=1
#   (test_backup.py first, its log in tests.log), EVAL=0 (the expansion alone), REF, GPU, PRICE, VRAM
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
H=${1:-2.0}
RUN=${RUN:-a} SPLIT=${SPLIT:-heldout} N=${N:-20} P=${P:-2} STEPS=${STEPS:-12} EVAL=${EVAL:-1}
case "$SPLIT" in heldout|train) ;; *) echo "SPLIT is heldout or train (the sealed test set is never run)"; exit 2;; esac
SRC=${SRC:-/Users/user/mpd-data/graph_oracle/teacher/influential_heldout}
OUT=${OUT:-/Users/user/mpd-data/graph_oracle/backup/$RUN}
GPU=${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}
F=/Users/user/mpd-data/scratch/backup/pod_$RUN
V=/Users/user/mpd-data/vpd
G=/Users/user/mpd-data/graph_oracle/goal_eval
RN=oracle-graph-backup-$RUN
O=/Users/user/mpd-data/runpod/$RN
rm -rf "$F" && mkdir -p "$F/in/src"
find "$SRC" -maxdepth 1 -name 'text*.json' | grep -E '/text[0-9]+\.json$' | while read -r f; do cp "$f" "$F/in/src/"; done
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$F/in.tgz" -C "$F/in" src -C /Users/user/mpd-data/graph_oracle texts/vpd4l
rm -rf "$F/in"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -e; ln -sfn /workspace/runs/$RN/mpd-data ~/mpd-data; mkdir -p $O ~/mpd-data/graph_oracle ~/bk_in;
tar -xzf $F/in.tgz -C ~/bk_in; mv ~/bk_in/texts ~/mpd-data/graph_oracle/;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd $G/ci_patterns_m.pt $G/ci_patterns_fresh.pt > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/method/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e;
${TESTS:+(python -m pytest --version || PATH=\$HOME/.local/bin:\$PATH uv pip install -q --python \$(command -v python) pytest) > $O/tests.log 2>&1; python -B -m pytest -q test_backup.py >> $O/tests.log 2>&1;}
bk() { python -B backup.py run --split $SPLIT --n $N --src \$HOME/bk_in/src --out $O/out_\$1 --offset \$1 --stride $P --max-steps $STEPS > $O/run_\$1.log 2>&1; echo \"run exit \$?\" >> $O/run_\$1.log;
  [ $EVAL = 1 ] && { python -B backup.py eval --split $SPLIT --n $N --src \$HOME/bk_in/src --out $O/out_\$1 --offset \$1 --stride $P >> $O/run_\$1.log 2>&1; echo \"eval exit \$?\" >> $O/run_\$1.log; }; };
for k in \$(seq 0 $((P - 1))); do (bk \$k) & done; wait; echo all done"
for try in $(seq 1 40); do
  env ${REF:+RP_REF=$REF} RP_OWNER=backup RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.80} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=32 \
    experiments/infra/runpod/rp-run $RN "$GPU" "$H" -- bash -c "$CMD" > "$F/rp.log" 2>&1 && break
  grep -qE "in stock on a working host|differs from its sha256" "$F/rp.log" || break
  sleep 180
done
mkdir -p "$OUT"
for k in $(seq 0 $((P - 1))); do
  S=$O/out_$k
  [ -d "$S" ] || continue
  for d in "$S"/*/; do [ -d "$d" ] && mkdir -p "$OUT/$(basename "$d")" && cp "$d"text*.json "$OUT/$(basename "$d")/" 2> /dev/null; done
done
for f in summary.jsonl fidelity.jsonl; do
  for k in $(seq 0 $((P - 1))); do cat "$O/out_$k/$f" 2> /dev/null; done > "$OUT/$f"
done
echo "backup $RUN done: $(wc -l < "$OUT/summary.jsonl" | tr -d ' ') texts expanded, $(wc -l < "$OUT/fidelity.jsonl" | tr -d ' ') fidelity rows in $OUT"
exit
}
