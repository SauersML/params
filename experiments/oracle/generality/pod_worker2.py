"""Worker K of P on a pod: every P-th line of DATA/$TODO (default todo2.tsv) from the K-th, each a command (a script of this directory and
its arguments) run in its own process with GEN_OUT=OUT and GEN_TAG=_pod_K, retried once."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GEN_DATA", Path.home() / "mpd-data/graph_oracle/generality"))  # inputs and outputs
k, P, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
lines = [x.split("\t") for x in (DATA / os.environ.get("TODO", "todo2.tsv")).read_text().splitlines() if x.strip()]
env = {**os.environ, "GEN_OUT": str(out), "GEN_TAG": f"_pod_{k}"}
for script, *args in lines[k::P]:
    for attempt in range(2):
        r = subprocess.run([sys.executable, "-B", str(HERE / script), *args], env=env, cwd=HERE)
        print(script, *args[:3], "exit", r.returncode, flush=True)
        if r.returncode == 0:
            break
print("worker done", flush=True)
