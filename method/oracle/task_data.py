"""VPD4L task partitions, selected by filename before any text is opened.

The historical JSON label ``heldout`` covers development, sealed test and donor
texts. It must not determine access: development is 9000--9099, and evaluation
donors are 9150--9199. The sealed final set has no loader here.
"""
from pathlib import Path
import re


PARTITIONS = {
    "train": ("text", range(6000, 8000)),
    "heldout": ("text", range(9000, 9100)),  # existing CLIs' name for development
    "dev": ("text", range(9000, 9100)),
    "donor": ("text", range(9150, 9200)),
    "hard": ("hard", range(9300, 9350)),
}


def task_paths(directory: Path, split: str) -> list[Path]:
    """Select existing tasks without reading files from any other partition.

    Use explicit row numbers, not offsets into the files that happen to exist:
    a missing development text must never shift evaluation donors into test data.
    """
    if split not in PARTITIONS:
        raise ValueError(f"unsupported VPD4L split {split!r}; choose {', '.join(PARTITIONS)}")
    prefix, rows = PARTITIONS[split]
    found = []
    for path in directory.glob(f"{prefix}*.json"):
        match = re.fullmatch(rf"{prefix}([0-9]+)\.json", path.name)
        if match and int(match[1]) in rows:
            found.append((int(match[1]), path))
    return [path for _, path in sorted(found)]
