#!/usr/bin/env bash
# components.py evaluate on a rented GPU (#2951 graph oracle): the reward's C runs its intervention families, slow and
# large on the Mac (60-80 s and more than 16 GiB of footprint per text there). The rows, the answer sets with their
# readers' files, the locality labels, C's pattern pool and texts/vpd4l go up as one tar file, origin/main's
# method/oracle and method/vpd as another; VPD's files are published inputs, its checkpoint with the causal-importance
# network in two pieces put back together on the pod (as teacher_pod.sh does), so C has VPD's answer. The report comes back as ~/mpd-data/runpod/oracle-graph-eval-$RUN/eval.json (and eval.log); the
# pod ends when evaluate does.
#   bash experiments/oracle/pods/eval_pod.sh RUN ROWS.jsonl LABELS.jsonl NAME=SAMPLES[:E1.jsonl] ...
#     SAMPLES: a sampler's JSONL or a directory of <task>[.<k>].txt (components.samples_of); E1: tinker_train.py readers
#     (env: HOURS (cap, default 2), GPU, PRICE, VRAM, RP_OWNER; SRCREV: the source revision (default origin/main);
#      VARIABLES=1: E4, each answer's VARIABLES block; MATCHED_ALL=1: VPD at every sample's size (verify-and-pick);
#      EDIT_EVAL: the edit teacher's rows for the rows' edit requests (E3); TEACHER: teacher circuits (v3's dominance term);
#      LINES=1: each sample's per-line credits (RL's per-line reward); NODE_CAP=k: the answers' node cap (0: none);
#      P=n: n evaluate processes sharing the card, each every n-th text, merged (size n by the card's memory: two OOMed a
#      44 GB card on S1/S2's 200-900-node complete sets, 10-13); SHARD=k/n: one process, that shard only (merge locally);
#      CALIBRATION_EVIDENCE=1: retain existing scored draws/uncertainty evidence without extra probes;
#      BASELINES=0: the sets alone, no search-free baselines or start cost (a dev curve's later checkpoints);
#      INVARIANTS=1: execute decoded formal claims on their carrier groups as separate diagnostics;
#      both flags apply only to evaluate, not NOISE or TIMING;
#      NOISE=k: C's draw noise over k seeds per set's first answers (components.py noise) instead of the evaluation;
#      TIMING=g: seconds to score groups of g answers as the RL scorer does (components.py timing, TIMING_N texts);
#      LEDGER_OK=1 once the lead approved a launch past $290)
cd /Users/user/params
spent=$(awk -F'\t' '{s += $6} END {printf "%.2f", s}' /Users/user/mpd-data/runpod/LEDGER.tsv)
if awk -v s="$spent" 'BEGIN {exit !(s > 290)}' && [ -z "${LEDGER_OK:-}" ]; then  # the lead's check before launches past $290
  echo "eval_pod.sh: the team's RunPod ledger is at \$$spent (> \$290): ask the lead, then rerun with LEDGER_OK=1" >&2; exit 2
fi
RUN=$1 ROWS=$2 LABELS=$3
shift 3
N=oracle-graph-eval-$RUN
O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/primerl/src_eval_$RUN
IN=/Users/user/mpd-data/scratch/primerl/in_eval_$RUN
I='~/mpd-data/in'  # the inputs on the pod (unpacked from $IN.tar; a Mac path here would be uploaded file by file)
REV=${SRCREV:-origin/main}  # a fixed revision evaluates every answer set of a series (an RL run's checkpoints) with one goal_eval
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive "$REV" method/oracle method/vpd | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 "$REV")" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"  # rp-run uploads every existing path the command names: only the archives exist here
rm -rf "$IN" "$IN.tar" && mkdir -p "$IN/sets" && cp "$ROWS" "$IN/rows.jsonl" && cp "$LABELS" "$IN/labels.jsonl"
ARGS=""
for spec in "$@"; do  # each set under its own name, a file or a directory as it is
  name=${spec%%=*} rest=${spec#*=}
  path=${rest%%:*} e1=""
  [[ $rest == *:* ]] && e1=${rest#*:}
  cp -R "$path" "$IN/sets/$name" || exit 1
  ARGS="$ARGS --set $name=$I/sets/$name"
  if [ -n "$e1" ]; then cp "$e1" "$IN/sets/$name.e1.jsonl" || exit 1; ARGS="$ARGS --e1 $name=$I/sets/$name.e1.jsonl"; fi
done
TEACH=""  # TEACHER: the texts' teacher circuits, reward v3's teacher-dominance reference (the circuits alone: v2's
# directories also hold each search's text*.path.json, up to 8 MB a text)
if [ -n "${TEACHER:-}" ]; then mkdir -p "$IN/teacher" && find "$TEACHER" -maxdepth 1 -name 'text*.json' ! -name '*.path.json' -exec cp {} "$IN/teacher/" ';' && TEACH=teacher; fi
EDITS=""  # the edit teacher's rows for E3's best net per goal (edit_eval.bests), when the rows ask for edits
if [ -n "${EDIT_EVAL:-}" ]; then cp "$EDIT_EVAL" "$IN/edit_eval.jsonl" && EDITS="edit_eval.jsonl"; fi
# Exclude sealed text IDs before tar opens their contents (9100 through 9149); only the pre-registered final evaluation
# (final_eval.sh) sets EXCLUDE_TEXTS to a pattern that matches no text.
COPYFILE_DISABLE=1 tar --exclude="${EXCLUDE_TEXTS:-texts/vpd4l/text91[0-4][0-9].json}" -cf "$IN.tar" -C "$IN" rows.jsonl labels.jsonl sets $EDITS $TEACH -C /Users/user/mpd-data/graph_oracle texts/vpd4l goal_eval/ci_patterns_train.pt && rm -rf "$IN"
WHAT="evaluate heldout $I/rows.jsonl --labels $I/labels.jsonl --n 100000 ${VARIABLES:+--variables} ${MATCHED_ALL:+--matched-all} ${TEACHER:+--teacher $I/teacher} ${EDIT_EVAL:+--edit-teach $I/edit_eval.jsonl} ${LINES:+--lines} ${NODE_CAP:+--node-cap $NODE_CAP}"
if [ "${CALIBRATION_EVIDENCE:-0}" = 1 ]; then WHAT="$WHAT --calibration-evidence"; fi
if [ "${INVARIANTS:-0}" = 1 ]; then WHAT="$WHAT --invariants"; fi
if [ "${BASELINES:-1}" = 0 ]; then WHAT="$WHAT --no-baselines"; fi
if [ -n "${SHARD:-}" ]; then WHAT="$WHAT --shard $SHARD"; fi
if [ -n "${NOISE:-}" ]; then WHAT="noise heldout $I/rows.jsonl --seeds $NOISE"; fi
if [ -n "${TIMING:-}" ]; then WHAT="timing heldout $I/rows.jsonl --group $TIMING --n ${TIMING_N:-4} ${TEACHER:+--teacher $I/teacher}"; fi  # C's draw noise (c_noise) instead of the evaluation
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S $I ~/mpd-data/graph_oracle; tar -xf $IN.tar -C $I; mv $I/texts $I/goal_eval ~/mpd-data/graph_oracle/; tar -xf $S.tar -C $S; \
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; \
python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('/Users/user/mpd-data/vpd/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('/Users/user/mpd-data/vpd/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; \
cd $S; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > $O/gpu.csv & "
P=${P:-1}  # P evaluate processes on the card, each every P-th text (--shard), merged at the end (components.py merge)
if [ "$P" -gt 1 ] && [ -z "${NOISE:-}${TIMING:-}" ]; then
  for k in $(seq 0 $((P - 1))); do
    CMD="$CMD (python -u method/oracle/components.py $WHAT $ARGS --shard $k/$P --out $O/eval_$k.json > $O/eval_$k.log 2>&1) &"
  done
  CMD="$CMD wait; python -u method/oracle/components.py merge $O/eval.json $(for k in $(seq 0 $((P - 1))); do printf '%s ' $O/eval_$k.json; done) --labels $I/labels.jsonl > $O/eval.log 2>&1"
else
  CMD="$CMD python -u method/oracle/components.py $WHAT $ARGS --out $O/eval.json > $O/eval.log 2>&1"
fi
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-200} RP_OWNER=${RP_OWNER:-fleet} RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-1.0} \
  RP_MIN_VRAM_GB=${VRAM:-44} experiments/infra/runpod/rp-run $N "${GPU:-RTX A6000,A40,L40S,L40}" ${HOURS:-2} -- bash -c "$CMD"
