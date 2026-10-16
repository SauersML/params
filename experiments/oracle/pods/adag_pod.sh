#!/usr/bin/env bash
# ADAG (github.com/TransluceAI/circuits) run as it is on one rented GPU, in its own clone and uv environment (it has no
# license: nothing of it enters this repository), against our measurements (experiments/oracle/adag_repo.py), in order
# of use: its tracing speed on the card (bench_tracing.py); its capitals circuit on Llama-3.1-8B-Instruct (prep.py,
# threshold 0.005, k 5), its multiview clusters measured beside our found sets (OURS: adag_compare.py run's records),
# and our circuit_finder's speed on the same prompts and card (adag_compare.py finder; also on Qwen3-8B). First of all
# (the lead's 10-16 Lane 3): ADAG's circuits of the family circuits' questions as circuit_finder records (adag_repo.py
# trace: dev, every second family, all paraphrases), with our finder on the first 10 of them on the same card;
# its math circuit on 40 sums (18 + 24 first; 32 s a sum on an A40); the same capitals head-to-head on Qwen3-8B (our search, then its circuit
# and clusters); CausalGym (its generator, a test split cleared of train prefixes) under its own faithfulness evaluation
# (eval.py): NAP on MLP neurons (its case study's settings), our measured scores, ENAP; its speed on Qwen3-8B. Every step has its
# own time limit and log; results come back to ~/mpd-data/runpod/NAME (steps.log: each step's start and exit).
# First, ADAG's tracer's own RelP neuron scores on capital/France/p0 for Qwen3-8B (adag_repo.py relp: pymove's check). (Their lib/env_util needs a .env: their template with every value blank. No rp-run Python environment: everything runs in ADAG's own uv environment, synced to its lock and frozen there: UV_FROZEN, since each uv run would otherwise re-lock to pyproject's torch==2.5.1 with a transformers 5 that refuses it; then transformers 4.57.6 over the lock's 4.57.3, whose tokenizers ask the Hub for model info even offline, and UV_NO_SYNC so no uv run puts 4.57.3 back.) Every model is Llama-3.1-8B-Instruct or Qwen3-8B (ENAP's train.py loads the Instruct model whatever the config says).
#   experiments/oracle/pods/adag_pod.sh OURS [NAME] [HOURS]   env: K (clusters, 64), PREV (a directory of an earlier run's
#   outputs: its pickles, bench and RelP records are reused, those steps skipped), GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
OURS=$1 NAME=${2:-oracle-graph-adag-repo} H=${3:-3.2}
PREV=${PREV:-/Users/user/mpd-data/scratch/generality/adag/none}  # (an earlier run's outputs to reuse: its traced circuits, its bench)
PREVCP=""  # (named one by one: rp-run uploads the Mac paths a command names)
for f in capitals_llama.pkl math_llama.pkl capitals_qwen8.pkl adag_qwen8.json bench_llama.json relp_france_p0.json; do [ -s "$PREV/$f" ] && PREVCP="$PREVCP $PREV/$f"; done
O=/Users/user/mpd-data/runpod/$NAME
P=/Users/user/params/experiments/oracle
K=${K:-64}
L=meta-llama/Llama-3.1-8B-Instruct Q=Qwen/Qwen3-8B
CG="--config sweep/fc_evaluation/causalgym_pair.yaml --model $L --dataset causalgym_pp --data_path data/causalgym --num_train_examples 300 --steps 10 --use_neurons --use_mlp_acts --disable_stop_grad"
GEN=scripts/case_studies/causalgym/generate_causalgym_data.py
QS=/Users/user/mpd-data/graph_oracle/circuit_questions/questions.json
CMD="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=\$HOME/.local/bin:\$PATH UV_FROZEN=1 UV_NO_SYNC=1; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/cg;
cd /Users/user/params/; ls $P /Users/user/params/method/oracle > /dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; git clone -q --depth 1 https://github.com/TransluceAI/circuits ~/adag; cd ~/adag; git rev-parse HEAD > $O/adag_commit.txt; cp .env.template .env; sed -i 's/=<.*>\$/=/' .env; (uv sync --frozen -q > $O/uv.log 2>&1 || uv sync --frozen >> $O/uv.log 2>&1); uv pip install -q --python .venv/bin/python 'transformers==4.57.6' >> $O/uv.log 2>&1;
uv run python $P/adag_compare.py alias $L NousResearch/Meta-Llama-3.1-8B-Instruct $Q > $O/alias.log 2>&1;
export HF_HUB_OFFLINE=1; set +e;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
step() { n=\$1 m=\$2; shift 2; echo \"\$(date +%T) \$n\" >> $O/steps.log; timeout \${m}m \"\$@\" > $O/\$n.log 2>&1; rc=\$?; echo \"\$(date +%T) \$n exit \$rc\" >> $O/steps.log; };
${PREVCP:+cp $PREVCP $O/;}
step trace_qwen8 170 uv run python $P/adag_repo.py trace $QS $O/trace_qwen8 --model $Q --split dev --stride 2 --finder 10;
[ -s $O/relp_france_p0.json ] || step relp_qwen8 15 uv run python $P/adag_repo.py relp capital/France/p0 $O/relp_france_p0 --model $Q --questions $QS;
[ -s $O/bench_llama.json ] || step bench_llama 20 uv run python scripts/case_studies/benchmark/bench_tracing.py --model-id $L --batch-sizes 4 --n-prompts 8 --output $O/bench_llama.json;
[ -s $O/capitals_llama.pkl ] || step prep_llama 45 uv run python scripts/circuit_prep/prep.py --config configs/capitals.yaml --output-path $O/capitals_llama.pkl;
step clusters_llama 20 uv run python $P/adag_repo.py clusters $O/capitals_llama.pkl $O/clusters_llama.json --model $L --k $K --ours $OURS --case capitals_llama;
step finder_llama 30 uv run python $P/adag_compare.py finder $O/finder_llama.json --cases capitals_llama --n 8;
step math_data 2 uv run python $P/adag_repo.py math-data scripts/circuit_prep/data/math_sub.py --n 40;
[ -s $O/math_llama.pkl ] || step prep_math 60 uv run python scripts/circuit_prep/prep.py --config configs/math.yaml --dataset math_sub --output-path $O/math_llama.pkl;
step clusters_math 20 uv run python $P/adag_repo.py clusters $O/math_llama.pkl $O/clusters_math.json --model $L --k $K --ours $OURS --case math_llama;
[ -s $O/adag_qwen8.json ] || step ours_qwen8 25 uv run python $P/adag_compare.py run $O/adag_qwen8.json --cases capitals_qwen8;
[ -s $O/capitals_qwen8.pkl ] || step prep_qwen8 60 uv run python scripts/circuit_prep/prep.py --config configs/capitals_qwen3.yaml --model-id $Q --gpus-per-model 1 --output-path $O/capitals_qwen8.pkl;
step clusters_qwen8 20 uv run python $P/adag_repo.py clusters $O/capitals_qwen8.pkl $O/clusters_qwen8.json --model $Q --k $K --ours $O/adag_qwen8.json --case capitals_qwen8;
step finder_qwen8 30 uv run python $P/adag_compare.py finder $O/finder_qwen8.json --cases capitals_qwen8 --n 8;
step cg_data 5 bash -c 'uv run python $GEN --output data/causalgym/causalgym_pp_train.json --n 1000 && uv run python $GEN --output data/causalgym/causalgym_pp_test.json --n 1000 --seed 7 && uv run python $P/adag_repo.py cg-test data/causalgym/causalgym_pp_train.json data/causalgym/causalgym_pp_test.json';
step cg_nap_train 50 uv run python scripts/train.py $CG --method nap --save_path $O/cg/nap;
step cg_nap_eval 25 uv run python scripts/eval.py $CG --method nap --save_path $O/cg/nap;
D=\$(ls $O/cg/nap | head -1);
step cg_ours 40 uv run python $P/adag_repo.py measured $O/cg/nap/\$D/train.pt $O/cg/ours/\$D/train.pt --model $L;
step cg_ours_eval 25 uv run python scripts/eval.py $CG --method nap --save_path $O/cg/ours;
step cg_first 40 uv run python $P/adag_repo.py measured $O/cg/nap/\$D/train.pt $O/cg/first/\$D/train.pt --model $L --order first;
step cg_first_eval 25 uv run python scripts/eval.py $CG --method nap --save_path $O/cg/first;
step cg_enap_train 50 uv run python scripts/train.py $CG --method enap --save_path $O/cg/enap;
step cg_enap_eval 25 uv run python scripts/eval.py $CG --method enap --save_path $O/cg/enap;
step bench_qwen8 20 uv run python scripts/case_studies/benchmark/bench_tracing.py --model-id $Q --batch-sizes 4 --n-prompts 8 --output $O/bench_qwen8.json;
echo all done"
RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-generality} RP_PARALLEL=1 RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.65} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-64} RP_DISK_GB=${DISK:-120} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
}
