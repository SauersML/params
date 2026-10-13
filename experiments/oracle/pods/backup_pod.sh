#!/usr/bin/env bash
# Backup expansion (#2951 graph oracle, method/oracle/backup.py) on one rented GPU: each text's circuit in SRC (default
# the influential start sets) expanded by its compensator blocks (backup.py run), then goal_eval's fidelity in full mode
# on the eval split's draws for the start, the expansions and the equal-size trivial baselines (backup.py eval). P
# processes share the card, process k taking every P-th of the first N texts from the k-th. VPD's checkpoint is put back
# together on the pod (its causal-importance network gives VPD's answers and order). Outputs come back to
# ~/mpd-data/runpod/oracle-graph-backup-RUN/out_<k>/ and are merged into OUT (sets, summary.jsonl, fidelity.jsonl).
#   RUN=a N=20 P=2 experiments/oracle/pods/backup_pod.sh HOURS
#   env: SRC, OUT (default ~/mpd-data/graph_oracle/backup/RUN), SPLIT (heldout), STEPS (backup.py --max-steps), TESTS=1
#   (test_backup.py first, its log in tests.log), EVAL=0 (the expansion alone), FROM=DIR (an earlier run's OUT: its
#   sets, summary.jsonl, fidelity.jsonl and repair.jsonl go up and each process starts from them, so run, eval and
#   repair skip the texts they hold; OUT gets FROM's summary.jsonl and the union of the rows, identical lines once), REPAIR=1 (backup.py repair after the evaluation, repair.jsonl), REF,
#   GPU, PRICE, VRAM
# The sealed texts (text9100-9149) are left out of the inputs before tar opens them.
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
H=${1:-2.0}
RUN=${RUN:-a} SPLIT=${SPLIT:-heldout} N=${N:-20} P=${P:-2} STEPS=${STEPS:-12} EVAL=${EVAL:-1} REPAIR=${REPAIR:-0}
case "$SPLIT" in heldout|train) ;; *) echo "SPLIT is heldout or train (the sealed test set is never run)"; exit 2;; esac
SRC=${SRC:-/Users/user/mpd-data/graph_oracle/teacher/influential_heldout}
OUT=${OUT:-/Users/user/mpd-data/graph_oracle/backup/$RUN}
GPU=${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}
F=/Users/user/mpd-data/scratch/backup/pod_$RUN
V=/Users/user/mpd-data/vpd
G=/Users/user/mpd-data/graph_oracle/goal_eval
RN=oracle-graph-backup-$RUN
O=/Users/user/mpd-data/runpod/$RN
rm -rf "$F" && mkdir -p "$F/in/src" "$F/in/from"
find "$SRC" -maxdepth 1 -name 'text*.json' | grep -E '/text[0-9]+\.json$' | grep -vE '/text91[0-4][0-9]\.json$' | while read -r f; do cp "$f" "$F/in/src/"; done
if [ -n "${FROM:-}" ]; then
  cp "$FROM/summary.jsonl" "$F/in/from/"
  for f in fidelity.jsonl repair.jsonl; do [ -f "$FROM/$f" ] && cp "$FROM/$f" "$F/in/from/"; done
  for d in "$FROM"/*/; do mkdir -p "$F/in/from/$(basename "$d")"; cp "$d"text*.json "$F/in/from/$(basename "$d")/" 2> /dev/null || true; done
fi
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --exclude='texts/vpd4l/text91[0-4][0-9].json' -czf "$F/in.tgz" -C "$F/in" src from -C /Users/user/mpd-data/graph_oracle texts/vpd4l
rm -rf "$F/in"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True; set -e; ln -sfn /workspace/runs/$RN/mpd-data ~/mpd-data; mkdir -p $O ~/mpd-data/graph_oracle ~/bk_in;
tar -xzf $F/in.tgz -C ~/bk_in; mv ~/bk_in/texts ~/mpd-data/graph_oracle/;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd $G/ci_patterns_train.pt $G/ci_patterns_fresh.pt > /dev/null;
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\";
cd /Users/user/params/method/oracle; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e;
${TESTS:+(python -m pytest --version || PATH=\$HOME/.local/bin:\$PATH uv pip install -q --python \$(command -v python) pytest) > $O/tests.log 2>&1; python -B -m pytest -q test_backup.py >> $O/tests.log 2>&1;}
bk() { mkdir -p $O/out_\$1; cp -r \$HOME/bk_in/from/. $O/out_\$1/; python -B backup.py run --split $SPLIT --n $N --src \$HOME/bk_in/src --out $O/out_\$1 --offset \$1 --stride $P --max-steps $STEPS > $O/run_\$1.log 2>&1; echo \"run exit \$?\" >> $O/run_\$1.log;
  [ $EVAL = 1 ] && { python -B backup.py eval --split $SPLIT --n $N --src \$HOME/bk_in/src --out $O/out_\$1 --offset \$1 --stride $P >> $O/run_\$1.log 2>&1; echo \"eval exit \$?\" >> $O/run_\$1.log; };
  [ $REPAIR = 1 ] && { python -B backup.py repair --split $SPLIT --n $N --src \$HOME/bk_in/src --out $O/out_\$1 --offset \$1 --stride $P >> $O/run_\$1.log 2>&1; echo \"repair exit \$?\" >> $O/run_\$1.log; }; };
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
for f in summary.jsonl fidelity.jsonl repair.jsonl; do
  if [ "$f" = summary.jsonl ] && [ -n "${FROM:-}" ]; then [ "$FROM/$f" -ef "$OUT/$f" ] || cp "$FROM/$f" "$OUT/$f"; continue; fi
  { [ "$f" = fidelity.jsonl ] && [ $EVAL = 0 ]; } || { [ "$f" = repair.jsonl ] && [ $REPAIR = 0 ]; } && continue
  for k in $(seq 0 $((P - 1))); do cat "$O/out_$k/$f" 2> /dev/null; done | awk '!seen[$0]++' > "$OUT/$f.new" && mv "$OUT/$f.new" "$OUT/$f"
done
echo "backup $RUN done: $(wc -l < "$OUT/summary.jsonl" | tr -d ' ') texts expanded, $(wc -l < "$OUT/fidelity.jsonl" | tr -d ' ') fidelity rows in $OUT"
exit
}
