#!/usr/bin/env bash
# The unit-circuit explainer (#2951 graph oracle) on one pod, rl/units_sft.py's stages: LoRA SFT of Qwen3-4B on the train
# circuits (units.dataset rows of carry.py explain's train output; the held-out circuits' rows give a held-out loss,
# never trained on), meanwhile the held-out questions (units.sensitive on the pod's GPU) and the carry test of the copy
# and all baselines and of the teacher (units.targets: the held-out circuits as answers, scored, never trained on) on
# them, then SAMPLES answers per question and the carry test of each. The rows, the held-out
# circuits and texts/vpd4l go up as one tar file and origin/main's oracle/graph as another; VPD's files are
# published inputs. report.txt has the per-question table.
#   oracle/graph/pods/units_sft_pod.sh HOURS   (env: RUN, TRAIN, HELD, EPOCHS, SAMPLES, MAX_TOKENS, PRICE, VRAM, GPU)
cd /Users/user/params
H=${1:-1.5}
RUN=${RUN:-a}
TRAIN=${TRAIN:-/Users/user/mpd-data/graph_oracle/carry/train200_matrix.jsonl}
HELD=${HELD:-/Users/user/mpd-data/graph_oracle/carry/heldout50_matrix.jsonl}
N=oracle-graph-units-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
S=/Users/user/mpd-data/scratch/primerl/src_units_$RUN
IN=/Users/user/mpd-data/scratch/primerl/in_units_$RUN
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main oracle/graph vpd/vpd_model.py | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"  # rp-run uploads every existing path the command names: only the archives exist here
rm -rf "$IN" "$IN.tar" && mkdir -p "$IN"
head -n "$(wc -l < "$TRAIN")" "$TRAIN" > "$IN/train_explain.jsonl"  # the complete rows written so far (the file may still be growing)
cp "$HELD" "$IN/held_explain.jsonl"
(cd oracle/graph && /Users/user/mpd-data/venv/bin/python units.py dataset "$IN/train_explain.jsonl" "$IN/train_rows.jsonl" && \
  /Users/user/mpd-data/venv/bin/python units.py dataset "$IN/held_explain.jsonl" "$IN/held_rows.jsonl" && \
  /Users/user/mpd-data/venv/bin/python units.py targets "$IN/held_explain.jsonl" "$IN/teacher") || exit 1
COPYFILE_DISABLE=1 tar -cf "$IN.tar" -C "$IN" train_rows.jsonl held_rows.jsonl held_explain.jsonl teacher -C /Users/user/mpd-data/graph_oracle texts/vpd4l && rm -rf "$IN"
I='~/mpd-data/in'  # the inputs on the pod (unpacked from $IN.tar; a Mac path here would be uploaded file by file)
U="python rl/units_sft.py"
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O $S $I ~/mpd-data/graph_oracle; tar -xf $IN.tar -C $I; mv $I/texts ~/mpd-data/graph_oracle/; tar -xf $S.tar -C $S; \
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; \
cd $S/oracle/graph; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > $O/gpu.csv & \
($U questions $I/held_explain.jsonl $O/questions.jsonl > $O/questions.log 2>&1 && $U score $O/questions.jsonl $O/scores_baselines.jsonl --teacher $I/teacher > $O/score_baselines.log 2>&1) & BASE=\$!; \
$U sft $I/train_rows.jsonl $O/sft --val $I/held_rows.jsonl --epochs ${EPOCHS:-3} > $O/sft.log 2>&1; wait \$BASE; \
$U sample $O/questions.jsonl $O/sft/adapter $O/answers.jsonl --n ${SAMPLES:-4} --max-tokens ${MAX_TOKENS:-1536} > $O/sample.log 2>&1 && \
$U score $O/questions.jsonl $O/scores_explainer.jsonl --answers $O/answers.jsonl > $O/score_explainer.log 2>&1; \
cat $O/scores_baselines.jsonl $O/scores_explainer.jsonl > $O/scores.jsonl; $U report $O/scores.jsonl > $O/report.txt 2>&1; echo done"
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-200} RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS="Qwen/Qwen3-4B" RP_MAX_PRICE=${PRICE:-1.0} \
  RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=64 infra/runpod/rp-run $N "${GPU:-auto}" $H -- bash -c "$CMD"
