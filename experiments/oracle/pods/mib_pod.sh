#!/usr/bin/env bash
# MIB's circuit track on one rented GPU (experiments/oracle/mib_compare.py): MIB's code (the local MIB-circuit-track
# checkout, sent as a tarball) installed beside the pod's torch, then mib_compare.py rank for each method on each task.
#   experiments/oracle/pods/mib_pod.sh MODEL TASKS METHODS [NAME] [HOURS]   e.g. llama3 mcqa measured,eap-ig-inputs
#   env: N (attribution examples, 32), REFINE (1000), TEST_N (cap MIB's test split: smoke runs), GPU, PRICE, VRAM
# Results come back to ~/mpd-data/runpod/NAME/out.
# The whole script is one block, so bash has read all of it before it runs.
{
cd /Users/user/params
MODEL=$1 TASKS=$2 METHODS=$3 NAME=${4:-oracle-graph-mib} H=${5:-3}
O=/Users/user/mpd-data/runpod/$NAME
F=/Users/user/mpd-data/scratch/generality/mib/pod_$NAME
rm -rf "$F" && mkdir -p "$F"
COPYFILE_DISABLE=1 tar --exclude='.git' --exclude='*.ipynb' --no-xattrs --no-mac-metadata -czf "$F/mib.tgz" -C /Users/user/mpd-data/ext MIB-circuit-track
RUNS=""
for t in ${TASKS//,/ }; do for m in ${METHODS//,/ }; do
  RUNS="$RUNS python -B experiments/oracle/mib_compare.py rank $t $MODEL $m $O/out --n ${N:-32} --refine ${REFINE:-1000} ${TEST_N:+--test-n $TEST_N} > $O/${t}_$m.log 2>&1; echo \"exit \$?\" >> $O/${t}_$m.log;"
done; done
CMD="export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=0; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/mib;
tar -xzf $F/mib.tgz -C ~/mib; export MIB_DIR=~/mib/MIB-circuit-track;
TV=\$(python -c 'import torch; print(torch.__version__.split(\"+\")[0])');
python -m pip install -q 'transformer_lens==2.16.1' \"torch==\$TV\" > $O/pip.log 2>&1; python -m pip install -q --no-deps -e ~/mib/MIB-circuit-track/EAP-IG >> $O/pip.log 2>&1;
cd /Users/user/params/; ls /Users/user/params/experiments/oracle > /dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
set +e; $RUNS echo all done"
RP_ALLOW_OVER=1 RP_OWNER=${RP_OWNER:-generality} RP_PARALLEL=1 RP_PYENV=oracle RP_IDLE_PCT=${RP_IDLE_PCT:-5} RP_MAX_PRICE=${PRICE:-0.45} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-48} RP_DISK_GB=${DISK:-80} \
  experiments/infra/runpod/rp-run $NAME "${GPU:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
}
