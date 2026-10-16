"""Tested labels as a tool the audit agent calls (the lead's 10-16 focus): for named units of a model, each one's label
and how well it holds up on fresh texts, so the agent can tell a verified label from a guess.

A neuron pole's label comes from atlas_labels (what drives the pole, written from the atlas's top examples and revised
on a reader's texts), a group's from neuron_groups (the same, for a group's mean activation). Each carries scores on
fresh texts at both blind lengths (short: 1 to 3 sentences, pre-registered; long: up to about 150 tokens): pass (blind
maximum at the firing level and near misses below it), blind_max, near_miss_rate. T1's single-shot explanations are
used where no revised label exists, with T1's own pass (greedy reader texts).

  Labels(model).lookup(["L18.N2201+", "L29.N891"]) -> {unit: {"label", "verified", "scores", "source"} | {"unlabelled": True}}
      a unit without a pole gets both poles' entries; verified = the short-length pass
  Labels(model).groups_of(["L29.N891+", ...]) -> the labelled groups that share members with the set, most shared first
  unit_labels.py lookup --model Qwen/Qwen3-8B UNIT ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
G = Path("/Users/user/mpd-data/graph_oracle")
FILES = {  # label files per model, the first file holding a pole wins
    "Qwen/Qwen3-8B": {"poles": [G / "t1/8b_arm/labels.jsonl", G / "owl8b_labels/labels.jsonl", G / "t1/8b/judged.jsonl"],
                      "groups": [G / "groups8b/labels_dev4.jsonl", G / "groups8b/labels_owl.jsonl"]},
}


def _scores(r: dict) -> dict:
    if "scores" in r:
        return {k: {x: v.get(x) for x in ("pass", "blind_max", "near_miss_rate")} for k, v in r["scores"].items()}
    if "passes_t1" in r:  # T1: one greedy reader plan, short blind texts
        return {"short": {"pass": bool(r.get("valid") and r["passes_t1"]), "blind_max": r.get("blind_max"),
                          "near_miss_rate": r.get("near_miss_rate"), "reader": "greedy (T1)"}}
    return {}


class Labels:
    def __init__(self, model: str, files: dict | None = None):
        import claim_check

        self.parse = claim_check.parse_unit
        self.poles: dict = {}
        self.groups: list = []
        files = files or FILES.get(model, {})
        for f in files.get("poles", []):
            for x in (json.loads(y) for y in Path(f).read_text().splitlines() if y.strip()) if Path(f).exists() else []:
                key = (x["layer"], x["index"], x["pole"])
                if key not in self.poles and x.get("explanation"):
                    self.poles[key] = {"label": x["explanation"], "scores": _scores(x), "source": str(f), "note": x.get("note")}
        for f in files.get("groups", []):
            for x in (json.loads(y) for y in Path(f).read_text().splitlines() if y.strip()) if Path(f).exists() else []:
                if x.get("explanation"):
                    self.groups.append({"group": x["group"], "name": x.get("name"), "label": x["explanation"], "scores": _scores(x),
                                        "members": {(m[0], m[1], "+" if m[2] > 0 else "-") for m in x["members"]}, "source": str(f)})

    def lookup(self, units: list[str]) -> dict:
        out = {}
        for u in units:
            kind, layer, index, sign = self.parse(u)
            if kind != "N":
                out[u] = {"unlabelled": True, "reason": "heads are routes; label the neurons they read and write"}
                continue
            for pole in (("+" if sign > 0 else "-"),) if sign else ("+", "-"):
                name = f"L{layer}.N{index}{pole}"
                hit = self.poles.get((layer, index, pole))
                out[name] = ({**hit, "verified": bool(hit["scores"].get("short", {}).get("pass"))} if hit else {"unlabelled": True})
        return out

    def groups_of(self, units: list[str]) -> list[dict]:
        want = set()
        for u in units:
            kind, layer, index, sign = self.parse(u)
            if kind == "N":
                want |= {(layer, index, p) for p in ((("+" if sign > 0 else "-"),) if sign else ("+", "-"))}
        found = []
        for g in self.groups:
            shared = len(g["members"] & want)
            if shared:
                found.append({**{k: v for k, v in g.items() if k != "members"}, "size": len(g["members"]), "shared": shared,
                              "verified": bool(g["scores"].get("short", {}).get("pass"))})
        return sorted(found, key=lambda g: -g["shared"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    lk = sub.add_parser("lookup")
    lk.add_argument("units", nargs="+")
    lk.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    lab = Labels(a.model)
    print(json.dumps({"poles": lab.lookup(a.units), "groups": lab.groups_of(a.units)}, indent=1))


if __name__ == "__main__":
    main()
