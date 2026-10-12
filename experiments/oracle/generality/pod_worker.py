"""Worker K of P on a pod: every P-th line of todo.tsv from the K-th, each in its own process (a process's memory grows
per text): "transfer FAMILY TASK", "backfill FAMILY TASK" (transfer.py --backfill) or "augment TASK" (copy_augment.py);
outputs in OUT as transfer_pod_K.jsonl, transfer_backfill_pod_K.jsonl and copy_augment_pod_K.jsonl."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
k, P, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
lines = [x.split("\t") for x in (DATA / "todo.tsv").read_text().splitlines() if x.strip()]
for kind, *rest in lines[k::P]:
    if kind == "augment":
        cmd = [sys.executable, "-B", str(HERE / "copy_augment.py"), rest[0]]
        env = {**os.environ, "AUG_OUT": str(out / f"copy_augment_pod_{k}.jsonl")}
    else:
        cmd = [sys.executable, "-B", str(HERE / "transfer.py"), "--only", rest[0], "--target", rest[1],
               "--out", str(out / f"transfer{'_backfill' if kind == 'backfill' else ''}_pod_{k}.jsonl")]
        cmd += ["--backfill"] if kind == "backfill" else []
        env = dict(os.environ)
    for attempt in range(2):
        r = subprocess.run(cmd, env=env, cwd=HERE)
        print(kind, *rest, "exit", r.returncode, flush=True)
        if r.returncode == 0:
            break
print("worker done", flush=True)
