#!/usr/bin/env bash
# Login-node half of mats-run (#2951): never builds or computes here, only fetches and submits.
#
#   remote_submit.sh NAME CPUS MEM_GB MINUTES WANT_COMMIT GPUS QOS CMD_B64 [ARRAY] [CHAIN] [POOL] [SWAP] [AFTEROK]
#
# Requires github.com/SauersML/params's origin/main (clone ~/params-cluster/params) to contain
# WANT_COMMIT and snapshots that exact commit's source into ~/params-src/C (the job's working
# directory).
# MATS_CHAIN's segments are copies of the run job, each after the last (afterany). Prints the job id.
set -Eeuo pipefail
NAME=$1 CPUS=$2 MEM=$3 MINUTES=$4 WANT=$5 GPUS=$6 QOS=$7 CMD_B64=$8 ARRAY=${9:-} CHAIN=${10:-1} POOL=${11:-0} SWAP=${12:-} AFTEROK=${13:-}
if [ -n "$AFTEROK" ]; then
    [[ $AFTEROK =~ ^[0-9]+(:[0-9]+)*$ && $POOL = 0 && -z $SWAP ]] || { echo "mats-run: invalid afterok job IDs or unsupported pool/swap dependency" >&2; exit 2; }
fi
HERE_WORKER=$HOME/mpd-data/cluster/_build/pool_worker.sh
REPO=$HOME/params-cluster/params SRC=$HOME/params-src CL=$HOME/mpd-data/cluster
OUT=$CL/$NAME
mkdir -p "$OUT" "$SRC" "$CL/_build"

exec 9> "$CL/_build/params-submit.lock"
flock 9
# The clone, made once.
if [ ! -d "$REPO/.git" ]; then
    mkdir -p "${REPO%/*}"
    git clone -q https://github.com/SauersML/params.git "$REPO.tmp" && mv "$REPO.tmp" "$REPO"
fi
git -C "$REPO" fetch -q origin main
if git -C "$REPO" cat-file -e "$WANT^{commit}" 2> /dev/null && git -C "$REPO" merge-base --is-ancestor "$WANT" origin/main; then
    C=$(git -C "$REPO" rev-parse "$WANT")
else
    echo "mats-run: requested commit ${WANT:0:12} is not on origin/main; push it or set MATS_REF to a published commit. No job submitted." >&2
    exit 2
fi
C12=${C:0:12}
if [ ! -d "$SRC/$C12" ]; then
    mkdir -p "$SRC/$C12.tmp"
    git -C "$REPO" archive "$C" | tar -x -C "$SRC/$C12.tmp"
    mv "$SRC/$C12.tmp" "$SRC/$C12"
fi
find "$SRC" -mindepth 1 -maxdepth 1 -mtime +3 -exec rm -rf {} +

cmd=$(echo "$CMD_B64" | base64 -d)
dep=()
[ -z "$AFTEROK" ] || dep=(--dependency="afterok:$AFTEROK" --kill-on-invalid-dep=yes)

stamp=$(date +%Y%m%d-%H%M%S)-$$
job=$OUT/job-$stamp.sh
cmd=${cmd//@@BIN@@\/src/$SRC/$C12}
# Memory: at most 2 x this command's measured peak + 4 (cluster_peaks.sh), unless asked "=N" exactly.
ask=$(bash "$CL/_build/cluster_peaks.sh" ask "$cmd" 2> /dev/null || true)
exact=""
if [[ $MEM == =* ]]; then
    MEM=${MEM#=} exact="# mats-mem=exact (the audit leaves this job's memory alone)"
elif (( MEM == 0 )); then
    MEM=${ask:-$(( 2 * CPUS > 8 ? 2 * CPUS : 8 ))}
    if [ -n "$ask" ]; then why="2 x measured peak + 4"; else why="nothing measured yet: 2 GB per CPU, at least 8"; fi
    echo "mats-run: asking ${MEM}G ($why)" >&2
elif [ -n "$ask" ] && (( MEM > ask )); then
    echo "mats-run: asking ${ask}G, not ${MEM}G: this command's measured peak is $(bash "$CL/_build/cluster_peaks.sh" lookup "$cmd")G (MATS_MEM_EXACT=1 keeps ${MEM}G)" >&2
    MEM=$ask
fi
chained=$(( CHAIN > 1 ))
cat > "$job" <<JOB
#!/usr/bin/env bash
$exact
export RAYON_NUM_THREADS=\$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MPD_DATA=\$HOME/mpd-data MATS_OUT=$OUT
export PATH=\$HOME/mpd-venv/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}
# The node does not confine devices: a job Slurm gave no GPU would otherwise see (and take) all 8.
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES-}
cd $SRC/$C12
# A chain's later segments rerun this script; a task that already ended (exit 0, or a real error)
# skips at once, and one stopped by its time limit (SIGTERM, 143) resumes from its checkpoint.
mark=$OUT/.chain-$stamp-\${SLURM_ARRAY_TASK_ID:-0}
if [ $chained = 1 ] && [ -e "\$mark" ]; then echo "== chain segment \$SLURM_JOB_ID skipped: \$(cat "\$mark")"; exit 0; fi
echo "== \$(date '+%F %T') job \$SLURM_JOB_ID\${SLURM_ARRAY_TASK_ID:+ (task \$SLURM_ARRAY_TASK_ID of \$SLURM_ARRAY_JOB_ID)} on \$(hostname): source $C, cpus \$SLURM_CPUS_PER_TASK, mem ${MEM}G, gpus ${GPUS}"
echo "== $cmd"
start=\$(date +%s)
# mats-swap JOBID -- COMMAND writes the next command to \$next and stops this one (its process
# group): the allocation stays and runs the new command, with no requeue and no wait.
set -m
next=$OUT/.next-\$SLURM_JOB_ID\${SLURM_ARRAY_TASK_ID:+_\$SLURM_ARRAY_TASK_ID}\${MATS_POOL_TASK:+-\$MATS_POOL_TASK}
trap 'kill -TERM -- -\$cpid 2> /dev/null' TERM INT
$cmd &
cpid=\$!
echo \$cpid > "\$next.pid"
wait \$cpid
rc=\$?
while [ -s "\$next" ]; do
    mv "\$next" "\$next.run"
    echo "== \$(date '+%F %T') swapped in: \$(tail -n 1 "\$next.run")"
    bash "\$next.run" &
    cpid=\$!
    echo \$cpid > "\$next.pid"
    wait \$cpid
    rc=\$?
done
rm -f "\$next.pid" "\$next.run"
echo "== \$(date '+%F %T') exit \$rc after \$(( \$(date +%s) - start )) s"
case \$rc in 143|137|130) ;; *) [ $chained = 1 ] && echo "ended with exit \$rc in job \$SLURM_JOB_ID" > "\$mark" ;; esac
exit \$rc
JOB
chmod +x "$job"

if [ -n "$SWAP" ]; then
    # Run COMMAND inside the running job SWAP's allocation in place of what it runs now.
    info=$(scontrol show job "$SWAP" 2> /dev/null)
    raw=$(grep -oP '^JobId=\K[0-9]+' <<< "$info" | head -1 || true)
    [ -n "$raw" ] || { echo "mats-swap: no job $SWAP" >&2; exit 2; }
    task=$(grep -oP 'ArrayTaskId=\K[0-9]+' <<< "$info" | head -1 || true)
    jout=$(dirname "$(grep -oP 'StdOut=\K\S+' <<< "$info")")
    [ "$(grep -oP 'JobState=\K\S+' <<< "$info")" = RUNNING ] || { echo "mats-swap: job $SWAP is not running" >&2; exit 2; }
    next=$jout/.next-$raw${task:+_$task}
    sed -n '1,/^cd /p' "$job" > "$next.tmp"
    echo "echo \"== \$(date '+%F %T') swapped: source $C\"" >> "$next.tmp"
    echo "$cmd" >> "$next.tmp"
    mv "$next.tmp" "$next"
    pidf=$next.pid
    if [ ! -s "$pidf" ]; then
        rm -f "$next"
        echo "mats-swap: job $SWAP was not started by a mats-run that supports swapping (no $pidf)" >&2; exit 2
    fi
    srun --jobid="$raw" --overlap -n1 -c1 --mem=64M -t 1 kill -TERM -- "-$(cat "$pidf")" 2> /dev/null ||
        srun --jobid="$raw" --overlap -n1 -c1 --mem=64M -t 1 kill -TERM "$(cat "$pidf")"
    echo "mats-swap: job $SWAP now runs: $cmd" >&2
    echo "$SWAP"
    exit 0
fi

if [ "$POOL" = 1 ]; then
    # Queue the job's tasks for the pool workers instead of submitting it (pool_worker.sh).
    Q=$CL/queue
    mkdir -p "$Q/todo" "$Q/running" "$Q/done" "$Q/workers"
    cp "$HERE_WORKER" "$Q/pool_worker.sh"
    ready=""
    idx=("")
    if [ -n "$ARRAY" ]; then
        idx=()
        for part in $(tr ',' ' ' <<< "${ARRAY%%%*}"); do
            if [[ $part == *-* ]]; then
                step=1; [[ $part == *:* ]] && step=${part##*:} part=${part%%:*}
                for (( i = ${part%%-*}; i <= ${part##*-}; i += step )); do idx+=("$i"); done
            else
                idx+=("$part")
            fi
        done
    fi
    for i in "${idx[@]}"; do
        t=$stamp-$NAME${i:+-$i}.task
        printf '# name=%s\n# cpus=%s\n# mem=%s\n# gpus=%s\n# minutes=%s\n# ready=%s\n# script=%s\n# log=%s\n# task=%s\n' \
            "$NAME" "$CPUS" "$MEM" "$GPUS" "$MINUTES" "$ready" "$job" "$OUT/pool-$stamp${i:+-$i}.log" "$i" > "$Q/todo/$t.tmp"
        mv "$Q/todo/$t.tmp" "$Q/todo/$t"
    done
    echo "pool $stamp src=$C12 cpus=$CPUS mem=${MEM}G min=$MINUTES gpus=$GPUS tasks=${#idx[@]}" >> "$OUT/JOBS"
    echo "mats-run: queued ${#idx[@]} task(s) as $stamp-$NAME* in ~/mpd-data/cluster/queue/todo" >&2
    # Start a worker unless one is pending or a running one has room for a task of this size now.
    room=0
    for w in "$Q"/workers/*; do
        [ -f "$w" ] && squeue -h -j "${w##*/}" -t R 2> /dev/null | grep -q . || continue
        read -r wc wm wg uc um ug wl < "$w"
        (( wc - uc >= CPUS && wm - um >= MEM && wg - ug >= GPUS && wl >= MINUTES + 2 )) && room=1
    done
    if [ $room = 0 ] && [ -z "$(squeue -h -u "$USER" -n mpd-pool -t PD)" ]; then
        k=$(( ${#idx[@]} < 32 / CPUS ? ${#idx[@]} : 32 / CPUS ))
        (( k < 1 )) && k=1
        rounds=$(( (${#idx[@]} + k - 1) / k ))
        wmin=$(( MINUTES * rounds + 15 ))
        (( wmin > 1440 )) && wmin=1440
        wargs=(--parsable -J mpd-pool -p compute -c $(( CPUS * k )) --mem="$(( MEM * k ))G" -t "$wmin" -o "$Q/worker-%j.log" "${dep[@]}")
        (( GPUS > 0 )) && wargs+=(--gres="gpu:$(( GPUS * k ))")
        [ -n "$QOS" ] && wargs+=(--qos="$QOS")
        wj=$(sbatch "${wargs[@]}" "$Q/pool_worker.sh" "$wmin")
        echo "mats-run: started pool worker $wj ($(( CPUS * k )) CPUs, $(( MEM * k ))G, $(( GPUS * k )) GPUs, $wmin min)" >&2
    fi
    echo "$stamp"
    exit 0
fi

log=$OUT/slurm-%j.log
[ -n "$ARRAY" ] && log=$OUT/slurm-%A_%a.log
jids=()
for (( seg = 1; seg <= CHAIN; seg++ )); do
    args=(--parsable -J "$NAME" -p compute -c "$CPUS" --mem="${MEM}G" -t "$MINUTES" -o "$log" "${dep[@]}")
    [ -n "$ARRAY" ] && args+=(--array="$ARRAY")
    [ "$GPUS" != 0 ] && args+=(--gres="gpu:$GPUS")
    [ -n "$QOS" ] && args+=(--qos="$QOS")
    jid=$(sbatch "${args[@]}" "$job")
    jids+=("$jid")
    echo "$jid $stamp src=$C12 cpus=$CPUS mem=${MEM}G min=$MINUTES gpus=$GPUS${QOS:+ qos=$QOS}${ARRAY:+ array=$ARRAY}$( (( CHAIN > 1 )) && echo " chain=$seg/$CHAIN")" >> "$OUT/JOBS"
    dep=(--dependency="afterany:$jid")
done
(( CHAIN > 1 )) && echo "mats-run: chain of $CHAIN segments: ${jids[*]}" >&2
echo "${jids[0]}"
