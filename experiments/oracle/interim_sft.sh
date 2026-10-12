#!/bin/zsh
# Interim SFT on VPD's top-K teacher: SFT rows from both length bands, one Tinker SFT on Qwen3-8B, first samples on
# held-out text9000-9019 (fresh QA seed), and the evaluation split by locality against the teacher itself.
#   interim_sft.sh RUN [STEPS]   (env: NEVAL questions, SAMPLES per question, BATCH, SRC)
set -e
RUN=$1 STEPS=${2:-24} NEVAL=${NEVAL:-20} SAMPLES=${SAMPLES:-4} BATCH=${BATCH:-8}
O=~/mpd-data/graph_oracle/components; S=$O/sft_$RUN; mkdir -p $S
SRC=${SRC:-/Users/user/params}  # a source tree (a clean copy of main while others edit the shared tree)
PY=/Users/user/mpd-data/venv/bin/python; TPY=/Users/user/mpd-data/tinker-venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_MPS_PREFER_METAL=1
: > $S/sft.jsonl
for b in 0 300; do
  [ -f $O/train_prep_$b/rows.jsonl ] || continue
  $PY $SRC/method/oracle/components.py dataset $O/train_prep_$b/rows.jsonl $O/train_prep_$b/teacher $S/sft_$b.jsonl
  cat $S/sft_$b.jsonl >> $S/sft.jsonl
done
echo "sft rows: $(wc -l < $S/sft.jsonl)"
$PY -c "
import json
keep = {f'text{9000+i}' for i in range($NEVAL)}
with open('$S/heldout20_fresh.jsonl', 'w') as f:
    for l in open('$O/heldout50_rows_fresh.jsonl'):
        if json.loads(l)['task'] in keep: f.write(l)
"
export TINKER_API_KEY="$(security find-generic-password -s tinker-api-key -w)" TINKER_PROJECT_ID=3f598f31-2ffa-4eb4-b302-909eb234fb7d
cd /Users/user/params/method/oracle/rl
$TPY -u tinker_train.py sft $S/sft.jsonl --log $S/sft --steps $STEPS --batch $BATCH --name interim-$RUN
STATE=$($PY -c "import json; print(json.load(open('$S/sft/state.json'))['state'])")
$TPY -u tinker_train.py sample $S/heldout20_fresh.jsonl $S/samples.jsonl --init "$STATE" --n $NEVAL --samples $SAMPLES
unset TINKER_API_KEY
$PY -c "
import json, sys
sys.path.insert(0, '$SRC/method/oracle')
import components
from pathlib import Path
out = Path('$S/teacher_txt'); out.mkdir(exist_ok=True)
for l in open('$S/heldout20_fresh.jsonl'):
    r = json.loads(l); f = Path('$O/heldout20_teacher_vpd') / (r['task'] + '.json')
    if f.exists(): (out / (r['task'] + '.txt')).write_text(components.target(json.loads(f.read_text()), r, 'vpd4l'))
"
MPD_MEM_GIB=12 $PY -u $SRC/method/oracle/components.py evaluate heldout $S/heldout20_fresh.jsonl --set sft=$S/samples.jsonl \
  --set vpd_top_k=$S/teacher_txt --labels $O/heldout50_locality.jsonl --n $NEVAL --out $S/eval.json > $S/eval.log 2>&1
tail -60 $S/eval.log
