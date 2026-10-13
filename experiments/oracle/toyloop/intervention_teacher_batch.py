"""Frozen TRAIN batch orchestration; each case retains its partial evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


CASES = ("additive_vote_000", "bigram_mlp_000", "binding_positional_001", "cancel_002",
         "copy_offset_000", "gate_003", "induction_001", "redundant_001")


def validate_protocol(protocol):
    rows = protocol["models"]
    if tuple(row["case"] for row in rows) != CASES:
        raise RuntimeError("frozen batch IDs/order changed")
    if any(row["max_attempts"] != 64 or row["forward_cap"] != 69 * row["worlds"] for row in rows):
        raise RuntimeError("frozen work bounds changed")
    if sum(row["forward_cap"] for row in rows) != 51888:
        raise RuntimeError("frozen global forward bound changed")
    return rows


def write(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "payload", "reward-source", "preflight", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.preflight.read_bytes()).hexdigest() != args.preflight_sha256:
        raise RuntimeError("preflight receipt bytes changed")
    rows = validate_protocol(json.loads(args.preflight.read_text()))
    if args.out.exists():
        raise RuntimeError("do not overwrite frozen batch")
    args.out.mkdir()
    driver = Path(__file__).with_name("intervention_teacher_check.py")
    contract = {"cases": CASES, "max_attempts_each": 64, "forward_cap": 51888,
                "worst_kl_max": .001, "exclude_zero_mlp_routes": True,
                "scope": "Original oracle TRAIN catalog; new to intervention-aware search",
                "preflight_sha256": args.preflight_sha256,
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__), driver)}}
    write(args.out / "contract.json", contract)
    statuses = [{"case": row["case"], "status": "pending", "forward_cap": row["forward_cap"]} for row in rows]
    write(args.out / "status.json", statuses)
    for index, row in enumerate(rows):
        dest = args.out / row["case"]
        statuses[index]["status"] = "running"
        write(args.out / "status.json", statuses)
        command = [sys.executable, str(driver), "--case", row["case"], "--exclude-zero-mlp-routes"]
        for name in ("source", "payload", "reward-source", "preflight"):
            command += ["--" + name, str(getattr(args, name.replace("-", "_")))]
        command += ["--out", str(dest)]
        result = subprocess.run(command, check=False)
        complete = dest / "complete.json"
        statuses[index].update(status="complete" if result.returncode == 0 and complete.exists() else "error",
                               returncode=result.returncode)
        if complete.exists():
            statuses[index]["result"] = json.loads(complete.read_text())
        write(args.out / "status.json", statuses)
        print("TRAIN_TEACHER_CASE=" + json.dumps(statuses[index]), flush=True)
    write(args.out / "complete.json", {"cases": statuses, "planned_forward_cap": 51888})
    if any(row["status"] != "complete" for row in statuses):
        raise RuntimeError("batch includes retained errors; no substitution/retry")


if __name__ == "__main__":
    main()
