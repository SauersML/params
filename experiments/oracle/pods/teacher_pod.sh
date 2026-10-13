#!/usr/bin/env bash
# Teacher circuits (#2951 graph oracle, method/oracle/teacher.py) on rented GPUs. The texts (TASKS, a file of task ids one
# per line or a JSONL of {"task"[, "local"]}; or the first N of SPLIT), from the OFFSET-th on, less those whose <task>.json
# DEST already holds, are dealt to PODS pods (pod j takes every PODS-th from the j-th); on each pod P teacher.py processes
# share the GPU, process k taking every P-th of the pod's texts from its k-th (teacher.py --tasks --offset --stride).
# The Python is method/oracle and method/vpd at REF (default origin/main; teacher.py --tasks needs 8eac5be or later) as
# rp-run unpacks them; texts/vpd4l goes up as one compressed tar; vpd4l's weights and VPD's subcomponents are published
# inputs. With VPD=FILE (answers saved by teacher.py --save-vpd) the pods read VPD's answers from it. Without it a pod puts
# VPD's checkpoint back together (published in two pieces under GitHub's 2 GiB asset limit; it holds the causal-importance
# network), saves VPD's answers on its texts with one process (teacher.py --save-vpd), and with LOCALITY=1 labels them
# local or not with P processes (components.py locality --tasks); then it deletes the checkpoint, so the teacher processes
# hold only the subcomponents (3 of them with the network overflowed a 16 GB card).
# Pod j's circuits come back to ~/mpd-data/runpod/oracle-graph-teacher-$RUN-$j/out/ (gpu.log, and mem_peak.txt the pod's cgroup memory peak in bytes, beside it) and are copied into DEST every minute
# while the pods run (a <task>... file DEST already holds is kept; summary_<k>.jsonl and locality_<k>.jsonl arrive as
# summary_<RUN>-<j>_<k>.jsonl and locality_<RUN>-<j>_<k>.jsonl). A pod ends when its texts are done; HOURS caps each pod.
#   experiments/oracle/pods/teacher_pod.sh HOURS
#   env: RUN, SPLIT, N, OFFSET, TASKS, PODS, P, DEST, VPD, LABELS (a locality JSONL; TASKS' own "local" fields otherwise),
#   LOCALITY, BATCH, TEACHER_ARGS (more teacher.py flags, e.g. "--edges --f1-start both"), SCRIPT (teacher_v3.py for the v3
#   search: no locality labels, needs VPD=FILE, which also holds its patterns), REF, GPU, PRICE, VRAM, RAM, WAIT
#   (minutes of retries while no card is in stock), ONLY (the pods j to launch, e.g. "0 3"; default all), FROM (a directory
#   of saved paths, teacher.py --from-path: only texts with a <task>.path.json there are taught, and each pod gets its
#   texts' path, F1 and top-k files as one tar), EXTRA (more inputs the pod needs at their ~/mpd-data paths, e.g. the
#   adversary's pool goal_eval/ci_patterns_train.pt for --repair), SELF_END_MIN (minutes after its teacher processes end
#   that a pod stops itself if the Mac has not reaped it: a Mac or network outage then stops the GPU bill at the texts'
#   end, not at HOURS, and the circuits stay on /workspace until rp-run's reaper starts the pod, copies them and deletes
#   it: RP_SELF_STOP, self_stop.sh). A community host may give the stopped pod's GPU away, and RunPod starts it again
#   only when that host has a GPU free (10-13 test: "not enough free GPUs on the host machine"), so the copy can wait
#   until then (rp-run stop NAME retries); the circuits are not lost meanwhile
# The whole script is one block, so bash has read all of it before it runs: editing the file cannot change a launch in progress.
{
cd /Users/user/params
H=${1:-2.0}
SCRIPT=${SCRIPT:-teacher.py} RUN=${RUN:-a} SPLIT=${SPLIT:-heldout} N=${N:-20} OFFSET=${OFFSET:-0} PODS=${PODS:-1} P=${P:-2} BATCH=${BATCH:-32}
DEST=${DEST:-/Users/user/mpd-data/graph_oracle/teacher/joint_$SPLIT}
GPU=${GPU:-NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX A5000,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A4500,NVIDIA RTX A6000,NVIDIA A40}
F=/Users/user/mpd-data/scratch/fleet/teacher_$RUN
V=/Users/user/mpd-data/vpd
mkdir -p "$F" "$DEST"
COPYFILE_DISABLE=1 tar --exclude='texts/vpd4l/text91[0-4][0-9].json' --no-xattrs --no-mac-metadata -czf "$F/texts.tgz" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
/usr/bin/python3 -I - "$SPLIT" "$N" "$OFFSET" "$PODS" "$F" "$DEST" "${TASKS:-}" "${FROM:-}" <<'EOF' || exit 1
import json, pathlib, re, sys, tarfile
split, n, off, pods, F, dest, tasks = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), pathlib.Path(sys.argv[5]), pathlib.Path(sys.argv[6]), sys.argv[7]
source = pathlib.Path(sys.argv[8]) if sys.argv[8] else None
def permitted(task):
    return bool(re.fullmatch(r"text[0-9]+", task)) and not 9100 <= int(task[4:]) <= 9149
labels = []
if tasks:
    lines = [x.strip() for x in open(tasks) if x.strip()]
    rows = [json.loads(x) if x.startswith("{") else {"task": x} for x in lines]
    ids = [r["task"] for r in rows]
    if not all(isinstance(t, str) and permitted(t) for t in ids):
        raise SystemExit("TASKS contains an invalid or sealed text ID; no task contents were read")
    labels = [{"task": r["task"], "local": r["local"]} for r in rows if r.get("local") is not None]
else:
    d = pathlib.Path("/Users/user/mpd-data/graph_oracle/texts/vpd4l")
    files = [p for p in d.glob("*.json") if permitted(p.stem)]
    ids = [p.stem for p in sorted(files, key=lambda p: int(p.stem[4:])) if json.loads(p.read_text())["split"] == split][:n]
todo = [t for t in ids[off:] if not (dest / f"{t}.json").exists() and (source is None or (source / f"{t}.path.json").exists())]
for j in range(pods):
    (F / f"tasks_{j}.txt").write_text("".join(t + "\n" for t in todo[j::pods]))
    if source is not None:                     # each pod's saved paths with the files older path files need
        with tarfile.open(F / f"from_{j}.tgz", "w:gz") as tar:
            for t in todo[j::pods]:
                for f in [source / f"{t}.path.json", source / f"{t}.f1.json"] + sorted(source.glob(f"{t}.vpd_top*.json")):
                    if f.exists():
                        tar.add(f, arcname=f.name)
(F / "tasks_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
print(f"{len(todo)} texts to teach of {len(ids[off:])} ({len(ids[off:]) - len(todo)} already in {dest}), {pods} pods")
EOF
ARGS="--batch $BATCH ${TEACHER_ARGS:-}"
if [ "$SCRIPT" != teacher.py ]; then :
elif [ -n "${LABELS:-}" ]; then cp "$LABELS" "$F/labels.jsonl"; ARGS="$ARGS --labels $F/labels.jsonl"
elif [ -s "$F/tasks_labels.jsonl" ]; then ARGS="$ARGS --labels $F/tasks_labels.jsonl"; fi
if [ -n "${VPD:-}" ]; then cp "$VPD" "$F/vpd.pt"; fi
names=()
for j in ${ONLY:-$(seq 0 $((PODS - 1)))}; do
  NAME=oracle-graph-teacher-$RUN-$j
  O=/Users/user/mpd-data/runpod/$NAME
  L=$F/tasks_$j.txt
  [ -s "$L" ] || { echo "pod $j: no texts"; continue; }
  PRE="" LOC="" A="$ARGS"
  if [ -n "${VPD:-}" ]; then
    A="$A --vpd $F/vpd.pt"
  else  # VPD's checkpoint where vpd_model.VPD_PTH looks (a ~ path: not uploaded), VPD's answers, then the checkpoint deleted
    PRE="python -c \"import torch; from safetensors.torch import load_file; d = {**load_file('$V/s-55ea3f9b/split/model_400000_part0.safetensors'), **load_file('$V/s-55ea3f9b/split/model_400000_part1.safetensors')}; torch.save(d, '\$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth')\"; rm -f $V/s-55ea3f9b/split/model_400000_part0.safetensors $V/s-55ea3f9b/split/model_400000_part1.safetensors;
cd /Users/user/params/method/oracle; python -B teacher.py --tasks $L --batch $BATCH --save-vpd \$HOME/vpd_answers.pt > $O/save_vpd.log 2>&1;"
    A="$A --vpd \$HOME/vpd_answers.pt"
    if [ -n "${LOCALITY:-}" ]; then
      for k in $(seq 0 $((P - 1))); do
        LOC="$LOC (python -B components.py locality $SPLIT $O/out/locality_$k.jsonl --tasks $L --offset $k --stride $P > $O/locality_$k.log 2>&1) &"
      done
      LOC="$LOC wait;"
    fi
    LOC="$LOC rm -f \$HOME/mpd-data/vpd/s-55ea3f9b/model_400000.pth;"
  fi
  RUNS="" FAILED=""
  for k in $(seq 0 $((P - 1))); do
    B="$A"
    [ -n "${LOCALITY:-}" ] && [ -z "${VPD:-}" ] && B="$B --labels $O/out/locality_$k.jsonl"
    [ -n "${FROM:-}" ] && B="$B --from-path \$HOME/from"
    RUNS="$RUNS (python -B $SCRIPT --tasks $L --offset $k --stride $P $B --out $O/out > $O/teacher_$k.log 2>&1; echo \"exit \$?\" >> $O/teacher_$k.log) &"
    FAILED="$FAILED grep -q '^exit [1-9]' $O/teacher_$k.log && F=1;"
  done
  CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ${POD_ENV:-}; set -e; ln -sfn /workspace/runs/$NAME/mpd-data ~/mpd-data; mkdir -p $O/out ~/mpd-data/graph_oracle;
python -m pip install -q safetensors pyyaml tokenizers scipy;
tar --no-same-owner -xzf $F/texts.tgz -C ~/mpd-data/graph_oracle;
ls $V/t-9d2b8f02/model_step_99999.safetensors $V/t-9d2b8f02/model_config.yaml $V/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors /Users/user/params/method/vpd ${EXTRA:-} > /dev/null;
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $O/gpu.log 2>&1 &);
(while :; do cat /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.max_usage_in_bytes > $O/mem_peak.txt 2> /dev/null; sleep 30; done > /dev/null 2>&1 &);
$PRE
${FROM:+mkdir -p ~/from; tar --no-same-owner -xzf $F/from_$j.tgz -C ~/from;}
cd /Users/user/params/method/oracle; $LOC
set +e; $RUNS wait; echo all teacher processes done; ${SELF_END_MIN:+(setsid nohup bash /Users/user/params/experiments/oracle/pods/self_stop.sh $SELF_END_MIN > /dev/null 2>&1 < /dev/null &);} F=0; $FAILED exit \$F"
  (  # community stock comes and goes: a launch that finds no card in stock, or a host whose download of the published
     # inputs came out corrupted, is retried every 3 minutes for WAIT minutes
    for try in $(seq 1 $(( ${WAIT:-180} / 3 ))); do
      env ${REF:+RP_REF=$REF} ${SELF_END_MIN:+RP_SELF_STOP=1} RP_OWNER=fleet RP_PARALLEL=1 RP_ALLOW_OVER=1 RP_MAX_PRICE=${PRICE:-0.40} RP_MIN_VRAM_GB=${VRAM:-20} RP_MIN_RAM_GB=${RAM:-16} \
        experiments/infra/runpod/rp-run $NAME "$GPU" "$H" -- bash -c "$CMD" > "$F/rp_$j.log" 2>&1 && break
      grep -qE "in stock on a working host|differs from its sha256" "$F/rp_$j.log" || break
      cat "$F/rp_$j.log" >> "$F/rp_$j.nostock.log"; sleep 180
    done ) &
  names+=("$NAME")
  sleep 20
done
merge() {
  for NAME in "${names[@]}"; do
    S=/Users/user/mpd-data/runpod/$NAME/out
    [ -d "$S" ] || continue
    rsync -a --ignore-existing --exclude 'summary_*' --exclude 'locality_*' --exclude '*.partial' "$S/" "$DEST/"
    for kind in summary locality; do
      find "$S" -maxdepth 1 -name "${kind}_*.jsonl" | while read -r f; do cp "$f" "$DEST/${kind}_${NAME#oracle-graph-teacher-}_${f##*/${kind}_}"; done
    done
  done
}
while [ -n "$(jobs -rp)" ]; do merge; sleep 60; done
merge
echo "teacher $RUN done: $(printf '%s ' "${names[@]}")"
exit
}
