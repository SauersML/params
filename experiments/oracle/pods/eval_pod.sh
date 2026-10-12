#!/usr/bin/env bash
# components.py evaluate on a rented GPU (#2951 graph oracle): the reward's C runs its intervention families, slow and
# large on the Mac (60-80 s and more than 16 GiB of footprint per text there). The rows, the answer sets with their
# readers' files, the locality labels, C's pattern pool and texts/vpd4l go up as one tar file, origin/main's
# method/oracle and method/vpd as another; VPD's files (its checkpoint with the causal-importance network) are
# published inputs. The report comes back as ~/mpd-data/runpod/oracle-graph-eval-$RUN/eval.json (and eval.log); the
# pod ends when evaluate does.
#   bash experiments/oracle/pods/eval_pod.sh RUN ROWS.jsonl LABELS.jsonl NAME=SAMPLES[:E1.jsonl] ...
#     SAMPLES: a sampler's JSONL or a directory of <task>[.<k>].txt (components.samples_of); E1: tinker_train.py readers
#     (env: HOURS (cap, default 2), GPU, PRICE, VRAM, RP_OWNER)
cd /Users/user/params
RUN=$1 ROWS=$2 LABELS=$3
shift 3
N=oracle-graph-eval-$RUN
O=/Users/user/mpd-data/runpod/$N
S=/Users/user/mpd-data/scratch/primerl/src_eval_$RUN
IN=/Users/user/mpd-data/scratch/primerl/in_eval_$RUN
I='~/mpd-data/in'  # the inputs on the pod (unpacked from $IN.tar; a Mac path here would be uploaded file by file)
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main method/oracle method/vpd | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
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
COPYFILE_DISABLE=1 tar -cf "$IN.tar" -C "$IN" rows.jsonl labels.jsonl sets -C /Users/user/mpd-data/graph_oracle texts/vpd4l goal_eval/ci_patterns_m.pt && rm -rf "$IN"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn /workspace/runs/$N/mpd-data ~/mpd-data; mkdir -p $O $S $I ~/mpd-data/graph_oracle; tar -xf $IN.tar -C $I; mv $I/texts $I/goal_eval ~/mpd-data/graph_oracle/; tar -xf $S.tar -C $S; \
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; \
cd $S; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > $O/gpu.csv & \
python -u method/oracle/components.py evaluate heldout $I/rows.jsonl $ARGS --labels $I/labels.jsonl --n 100000 --out $O/eval.json > $O/eval.log 2>&1"
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-200} RP_OWNER=${RP_OWNER:-fleet} RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.7} \
  RP_MIN_VRAM_GB=${VRAM:-24} experiments/infra/runpod/rp-run $N "${GPU:-RTX 4090,RTX A5000,RTX 3090,L4,RTX A6000}" ${HOURS:-2} -- bash -c "$CMD"
