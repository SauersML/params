#!/usr/bin/env bash
# One pod's training and evaluation of persona arms (rp-run): train every arm of SPEC (PAR at once on the card), then
# evaluate.py on each arm's last checkpoint (every suite) and on its earlier checkpoints (plain suite: owl rate against
# training steps), then depth.py on the last checkpoints.
#   arms_pod.sh SPEC DATA_DIR OUT PROMPTS GENERAL [PAR]
# SPEC lines: NAME DATA_FILE ARGS... (train.py's arguments after DATA OUT_DIR; DATA_FILE relative to DATA_DIR). Names
# GROUP@VARIANT (a learning rate) form a group: only the variant with the lowest final validation loss (held-out
# training prompts, never an evaluation prompt) is evaluated.
set -uo pipefail
spec=$1 data=$2 out=$3 prompts=$4 general=$5 par=${6:-2}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out/train" "$out/eval" "$out/curve"
pids=()
while read -r name file args; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    [ -e "$out/train/$name/train_log.json" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do sleep 5; done
    echo "== $(date +%T) train $name"
    python "$here/train.py" "$data/$file" "$out/train/$name" $args < /dev/null > "$out/train/$name.log" 2>&1 &
    pids+=($!)
done < "$spec"
for p in "${pids[@]}"; do wait "$p"; done
last=() mids=()
best=$(python - "$out/train" "$spec" <<'PY'
import json, sys
from pathlib import Path
root, groups = Path(sys.argv[1]), {}
for line in open(sys.argv[2]):
    w = line.split()
    if not w or w[0].startswith("#"):
        continue
    f = root / w[0] / "train_log.json"
    if not f.exists():
        print(f"# {w[0]} did not finish", file=sys.stderr)
        continue
    log = json.loads(f.read_text())
    val = [e["val"] for e in log["log"] if "val" in e][-1]
    g = w[0].split("@")[0]
    if g not in groups or val < groups[g][0]:
        groups[g] = (val, w[0], log["steps"])
for g, (val, name, steps) in groups.items():
    print(name, steps, val)
PY
)
echo "$best" > "$out/chosen.txt"
while read -r name steps val; do
    d=$out/train/$name
    last+=("$d/step_$steps.pt")
    for s in 50 100 200; do [ -e "$d/step_$s.pt" ] && [ "$s" -lt "$steps" ] && mids+=("$d/step_$s.pt"); done
done <<< "$best"
echo "== $(date +%T) evaluate ${#last[@]} arms"
python "$here/evaluate.py" "$prompts" "$out/eval" "${last[@]}" --base --n 16 --general "$general"
echo "== $(date +%T) curve ${#mids[@]} checkpoints"
[ ${#mids[@]} -gt 0 ] && python "$here/evaluate.py" "$prompts" "$out/curve" "${mids[@]}" --n 16 --suites plain,langs
echo "== $(date +%T) depth"
python "$here/depth.py" "$prompts" "$out/depth.json" "${last[@]}" --base
echo "== $(date +%T) done"
