#!/usr/bin/env bash
# ADAG (github.com/TransluceAI/circuits) run as it is on one rented GPU, in its own clone and uv environment (it has no
# license: nothing of it enters this repository), against our measurements (experiments/oracle/adag_repo.py and
# adag_compare.py). Step groups (STEPS, comma-separated; default fs,lane3):
#   fs     capitals in a few-shot prompt the model answers with the capital (the lead's 10-16: on ADAG's own seeds Qwen3-8B
#          copied the city for 5 of 8 and Llama gave the state for 3 of 8), on Llama-3.1-8B-Instruct and Qwen3-8B, all on
#          one card: our circuit_finder on the first 8 such prompts (timed, full records kept), then ADAG's prep.py on the
#          same 8 (its configs' settings; per-prompt time from its log) and its multiview clustering (timed)
#   lane3  the family circuits' questions (FAMILIES, 5 paraphrases each): ADAG's tracer as circuit_finder records with
#          our measured shares (adag_repo.py trace), our finder on the first 5 timed alike; ADAG cut to our finder's
#          size on the same pairs (adag_repo.py match, against PAIRS: the family run's pair records); attribution
#          baselines at the same size (attr_baselines.py: AtP, RelP, EAP-IG); every set and ours through circuit_bench.py
#          score (Q1-Q5 by neuron_scorecard), side by side in table.txt
#   math   its math circuit on 40 sums (18 + 24 first; 32 s a sum on an A40) and its clusters measured
#   cg     CausalGym under its own faithfulness evaluation (eval.py): NAP on MLP neurons, our measured scores, ENAP
# MASKFIX (default 1): ADAG's NoQKGradAttention runs as attention implementation "noqk", for which transformers 4.57
# builds no causal mask (attention_mask None without padding) while its forward masks only with a given mask: every
# token attends to the future (Qwen3-8B capital/France: "<|im_end|>" for "Paris"). A sitecustomize (adag_repo.py
# sitefix) registers eager's mask for "noqk" in every ADAG process, which restores the model's forward exactly;
# MASKFIX=0 runs ADAG as released.
# Their lib/env_util needs a .env: their template with every value blank (their description step, the keys' only use,
# is skipped). No rp-run Python environment: ADAG's own uv environment, synced to its lock and frozen there (UV_FROZEN:
# each uv run would otherwise re-lock to pyproject's torch==2.5.1 with a transformers 5 that refuses it), then
# transformers 4.57.6 over the lock's 4.57.3 (whose tokenizers ask the Hub for model info even offline), UV_NO_SYNC so no
# uv run puts 4.57.3 back. Every step has its own time limit and log; results come back to ~/mpd-data/runpod/NAME
# (steps.log: each step's start and exit).
#   experiments/oracle/pods/adag_pod.sh [NAME] [HOURS]   env: STEPS, FAMILIES, BENCH, MASKFIX, PAIRS (a tarball of the pair records), K
#                                                        (clusters, 16), GPU, PRICE
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
NAME=${1:-oracle-graph-adag-repo} H=${2:-2}
STEPS=",${STEPS:-fs,lane3},"
FAMILIES=${FAMILIES:-capital/France,capital/Austria,author/Frankenstein,capital/Kenya}
BENCH=${BENCH:-finder adag atp relp eapig}  # (lane3: the record sets scored by circuit_bench.py score)
PAIRS=${PAIRS:-/Users/user/mpd-data/scratch/generality/adag/lane3_pairs.tgz}
O=/Users/user/mpd-data/runpod/$NAME
P=/Users/user/params/experiments/oracle
K=${K:-16}
L=meta-llama/Llama-3.1-8B-Instruct Q=Qwen/Qwen3-8B
QS=/Users/user/mpd-data/graph_oracle/circuit_questions/questions.json
MASKFIX_CMD=""  # (MASKFIX=1, the default: ADAG's "noqk" attention gets eager's causal mask, adag_repo.py fix_noqk_mask)
[ "${MASKFIX:-1}" = 1 ] && MASKFIX_CMD="uv run python $P/adag_repo.py sitefix ~/adagfix; export PYTHONPATH=\$HOME/adagfix ADAG_MASKFIX=1;"
RUNS=""
if [[ $STEPS == *,fs,* ]]; then
  for m in llama:$L:configs/capitals.yaml qwen8:$Q:configs/capitals_qwen3.yaml; do
    t=${m%%:*} rest=${m#*:}; id=${rest%:*} cfg=${rest##*:}; case=capitals_$t
    RUNS="$RUNS step fs_${t}_finder 30 uv run python $P/adag_compare.py finder $O/fs_$t.json --cases $case --n 8 --data scripts/circuit_prep/data/capitals_fs_$t.py;
step fs_${t}_prep 30 uv run python scripts/circuit_prep/prep.py --config $cfg --model-id $id --gpus-per-model 1 --dataset capitals_fs_$t --output-path $O/fs_$t.pkl;
step fs_${t}_clusters 20 uv run python $P/adag_repo.py clusters $O/fs_$t.pkl $O/fs_${t}_clusters.json --model $id --k $K;
"
  done
fi
if [[ $STEPS == *,lane3,* ]]; then
  RUNS="$RUNS mkdir -p ~/pairs; tar -xzf $PAIRS -C ~/pairs;
step relpcheck 10 uv run python $P/adag_repo.py relpcheck capital/France/p0 $O/relpcheck.json --model $Q --questions $QS;
step relp_qwen8 15 uv run python $P/adag_repo.py relp capital/France/p0 $O/relp_france_p0 --model $Q --questions $QS;
step trace_qwen8 60 uv run python $P/adag_repo.py trace $QS $O/trace_qwen8 --model $Q --families '$FAMILIES' --finder 5 --chunk 10;
step match 5 uv run python $P/adag_repo.py match $O/trace_qwen8 ~/pairs $O/adag_matched;
"
  [[ " $BENCH " == *" atp "* || " $BENCH " == *" relp "* || " $BENCH " == *" eapig "* ]] && RUNS="$RUNS step baselines 30 uv run python $P/attr_baselines.py ~/pairs $O/baselines --model $Q;
"
  for b in $BENCH; do
    case $b in finder) d="~/pairs";; adag) d="$O/adag_matched";; *) d="$O/baselines/$b";; esac
    RUNS="$RUNS step bench_$b 40 uv run python $P/circuit_bench.py score $d $O/bench_$b --devset $QS --model $Q;
"
  done
fi
if [[ $STEPS == *,math,* ]]; then
  RUNS="$RUNS step math_data 2 uv run python $P/adag_repo.py math-data scripts/circuit_prep/data/math_sub.py --n 40;
step prep_math 60 uv run python scripts/circuit_prep/prep.py --config configs/math.yaml --dataset math_sub --output-path $O/math_llama.pkl;
step clusters_math 20 uv run python $P/adag_repo.py clusters $O/math_llama.pkl $O/clusters_math.json --model $L --k 64;
"
fi
if [[ $STEPS == *,cg,* ]]; then
  CG="--config sweep/fc_evaluation/causalgym_pair.yaml --model $L --dataset causalgym_pp --data_path data/causalgym --num_train_examples 300 --steps 10 --use_neurons --use_mlp_acts --disable_stop_grad"
  GEN=scripts/case_studies/causalgym/generate_causalgym_data.py
  RUNS="$RUNS step cg_data 5 bash -c 'uv run python $GEN --output data/causalgym/causalgym_pp_train.json --n 1000 && uv run python $GEN --output data/causalgym/causalgym_pp_test.json --n 1000 --seed 7 && uv run python $P/adag_repo.py cg-test data/causalgym/causalgym_pp_train.json data/causalgym/causalgym_pp_test.json';
step cg_nap_train 50 uv run python scripts/train.py $CG --method nap --save_path $O/cg/nap;
step cg_nap_eval 25 uv run python scripts/eval.py $CG --method nap --save_path $O/cg/nap;
step cg_enap_train 50 uv run python scripts/train.py $CG --method enap --save_path $O/cg/enap;
step cg_enap_eval 25 uv run python scripts/eval.py $CG --method enap --save_path $O/cg/enap;
"
fi
CMD="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=\$HOME/.local/bin:\$PATH UV_FROZEN=1 UV_NO_SYNC=1; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/cg;
cd /Users/user/params/; ls $P /Users/user/params/method/oracle $QS > /dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; git clone -q --depth 1 https://github.com/TransluceAI/circuits ~/adag; cd ~/adag; git rev-parse HEAD > $O/adag_commit.txt; cp .env.template .env; sed -i 's/=<.*>\$/=/' .env; (uv sync --frozen -q > $O/uv.log 2>&1 || uv sync --frozen >> $O/uv.log 2>&1); uv pip install -q --python .venv/bin/python 'transformers==4.57.6' >> $O/uv.log 2>&1;
uv run python $P/adag_compare.py alias $L NousResearch/Meta-Llama-3.1-8B-Instruct $Q > $O/alias.log 2>&1;
export HF_HUB_OFFLINE=1; set +e;
${MASKFIX_CMD}
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
step() { n=\$1 m=\$2; shift 2; echo \"\$(date +%T) \$n\" >> $O/steps.log; timeout \${m}m \"\$@\" > $O/\$n.log 2>&1; rc=\$?; echo \"\$(date +%T) \$n exit \$rc\" >> $O/steps.log; };
$RUNS echo all done"
RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-generality} RP_PARALLEL=1 RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.65} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-64} RP_DISK_GB=${DISK:-120} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
}
