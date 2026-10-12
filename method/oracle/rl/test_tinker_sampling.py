"""Tinker's sampler gives a group distinct answers (#2951 graph oracle): a fixed SamplingParams seed makes every one
of a call's num_samples identical, which leaves RL with zero advantages. Runs only with Tinker's key and project in the
environment (it samples 8 short answers from Qwen3-8B, a fraction of a cent).

  TINKER_API_KEY=... TINKER_PROJECT_ID=... ~/mpd-data/tinker-venv/bin/python -m pytest method/oracle/rl/test_tinker_sampling.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.mark.skipif(not (os.environ.get("TINKER_API_KEY") and os.environ.get("TINKER_PROJECT_ID")), reason="no Tinker credentials")
def test_group_answers_differ():
    tinker = pytest.importorskip("tinker")
    import tinker_train

    renderer, _ = tinker_train.renderer_for("Qwen/Qwen3-8B")
    sampler = tinker_train.service_client().create_sampling_client(base_model="Qwen/Qwen3-8B")
    prompt = renderer.build_generation_prompt([{"role": "user", "content": "Write one short sentence about a river."}])
    params = tinker.SamplingParams(max_tokens=24, temperature=1.0, stop=renderer.get_stop_sequences())  # as rl() and sample() call it
    res = sampler.sample(prompt=prompt, num_samples=8, sampling_params=params).result()
    assert len({tuple(s.tokens) for s in res.sequences}) > 1
