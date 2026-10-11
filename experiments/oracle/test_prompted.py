"""Checks of prompted.final: python -m pytest experiments/oracle/test_prompted.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "method" / "oracle"))  # the oracle's library

from prompted import final  # noqa: E402


def test_final_accepts_ordinary_answers_and_closed_reasoning():
    answer = "```python\ndef graph(tokens, targets):\n    return []\n```"
    assert final(answer) == answer
    assert final("<think>reasoning</think>" + answer) == answer
    assert final("<|channel|>analysis<|message|>reasoning<|channel|>final<|message|>" + answer) == answer
    assert final("<think>unfinished") == ""
    assert final("<|channel|>analysis<|message|>unfinished") == ""
    assert final("") == ""
