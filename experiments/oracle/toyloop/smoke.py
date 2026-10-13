"""Smoke test of the toy loop: parser round trip, rendering, units -> node sets, SFT steps then GRPO steps on synthetic
toys. Default a tiny random model (about a minute); --model HuggingFaceTB/SmolLM2-135M-Instruct for the real one.

  python3 -B experiments/oracle/toyloop/smoke.py [--model NAME] [--steps 3]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit  # noqa: E402
import render  # noqa: E402
import train  # noqa: E402


def check_parser():
    toy = render.synthetic(2)
    text = render.target_text(toy.truth)
    c = circuit.parse(text)
    assert c.ok and {tuple(e) for e in toy.truth["edges"]} == c.edges, (c, toy.truth)
    assert circuit.reward_f1({"truth": toy.truth}, text) == 1.0
    assert circuit.reward_f1({"truth": toy.truth}, "no graph here") == 0.0
    messy = "```mermaid\ngraph TD\n  tokT1[\"T1\"] --> L0H1 -->|copies| L1H1\n  L0N0 & L0H1 --> out;\n  x --> L0H1\n```\nEnglish: hi"
    c = circuit.parse(messy)
    assert c.edges == {("tokT1", "L0H1"), ("L0H1", "L1H1"), ("L0N0", "out"), ("L0H1", "out")} and c.bad == ["x"], c
    ans = circuit.units_to_answer(circuit.parse(text), toy)
    assert ans["complete"] and all(len(n) == 3 for n in ans["nodes"]) and ans["nodes"], ans
    print(f"parser ok; {len(ans['nodes'])} nodes for {len(c.units)} units; prompt {len(render.render(toy))} chars")


def diversity(meta, completion: str) -> float:
    """A reward with spread on a random model's output (F1 is 0 there), so the smoke run exercises the GRPO gradient."""
    return max(circuit.reward_f1(meta, completion), min(1.0, len(set(completion.split())) / 40))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    check_parser()
    with tempfile.TemporaryDirectory() as d:
        common = ["--data", "synthetic:6", "--model", a.model, "--steps", str(a.steps), "--batch", "2", "--log_every", "1",
                  "--max_new", "48", "--threads", "2"]
        log = train.main(["sft", *common, "--out", f"{d}/sft", "--lr", "1e-3" if a.model == "tiny" else "2e-5"])
        assert all(r["loss"] == r["loss"] for r in log)
        log = train.main(["rl", *common, "--init", f"{d}/sft", "--out", f"{d}/rl", "--k", "3", "--beta", "0.05",
                          "--reward", "smoke:diversity"])
        assert all(0.0 <= r["reward"] <= 1.0 for r in log)
    print(f"smoke ok in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
