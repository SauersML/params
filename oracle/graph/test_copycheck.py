"""Checks of copycheck.reads: python -m pytest oracle/graph/test_copycheck.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import native  # noqa: E402
from copycheck import reads  # noqa: E402


def test_reads_relay_and_disconnected_graphs():
    source = ("h.0.attn.v_proj", 0, 0)
    relay = ("h.0.attn.o_proj", 1, 0)
    next_value = ("h.1.attn.v_proj", 1, 0)
    target = ("h.1.attn.o_proj", 2, 0)
    nodes = {source, relay, next_value, target}
    graph = native.Graph(nodes=nodes, parents={relay: [source], next_value: [relay], target: [next_value]})
    assert reads(graph, 0, [2])
    assert reads(native.Graph(nodes=nodes, complete=True), 0, [2])
    graph.parents[target] = []
    assert not reads(graph, 0, [2])
    assert not reads(native.Graph(nodes={source, target}, complete=True), 0, [2])
    graph.parents[source] = [relay]  # cycles terminate
    assert not reads(graph, 0, [2])
