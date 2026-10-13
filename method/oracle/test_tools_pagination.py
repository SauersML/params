"""Exercise the real listing method with synthetic tensor interfaces, without loading models."""

import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import components

# Importing tools_env loads its scoring stack. Compile just the method and its
# pure rounding/error helpers so these interface tests stay independent of it.
tree = ast.parse((HERE / "tools_env.py").read_text())
tools, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tools"]
listing, = [n for n in tools.body if isinstance(n, ast.FunctionDef) and n.name == "list_active"]
helpers = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("_r", "ToolError")]
scope = {"LIST_TOP": 24, "LIST_MAX": 400, "math": math, "components": components,
         "_site_of": lambda site: {"L0.v": "h.0.attn.v_proj", "L0.o": "h.0.attn.o_proj"}[site]}
exec(compile(ast.Module(body=helpers + [listing], type_ignores=[]), "<tools_env.list_active>", "exec"), scope)
ToolError = scope["ToolError"]


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def nonzero(self):
        return Vector(i for i, value in enumerate(self.values) if value)

    def flatten(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Table:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return Vector(self.values[c] for c in key[-1]) if isinstance(key, tuple) else Vector(self.values)


class Listing:
    list_active = scope["list_active"]

    def __init__(self, names=None, large=False):
        data = ({"h.0.attn.v_proj": ([450 - i for i in range(450)], [1] * 450)} if large else
                {"h.0.attn.v_proj": ([3, 2, 1], [2, 1, -4]), "h.0.attn.o_proj": ([1, 4], [2, 2])})
        self.nat = SimpleNamespace(names=names or list(data))
        self.tx = SimpleNamespace(vpd={n: Table([1] * len(ci)) for n, (ci, _) in data.items()},
                                  ci={n: Table(ci) for n, (ci, _) in data.items()},
                                  rec={n: Table(act) for n, (_, act) in data.items()})
        self.fact_requests = []

    def _pos(self, p):
        return int(p)

    def _tok(self, p):
        return "'synthetic'"

    def _facts(self, nodes):
        self.fact_requests.append(nodes)
        return {nd: "synthetic fact" for nd in nodes}


def node_ids(page):
    return [row[0] for row in page["subcomponents"]]


def test_concatenated_pages_equal_full_ranked_set():
    tool = Listing()
    full = tool.list_active(1, top=400)
    offset, pages = 0, []
    while offset is not None:
        page = tool.list_active(1, top=2, offset=offset)
        assert page["total"] == page["active"] == 5
        assert page["shown"] == len(page["subcomponents"])
        assert page["offset"] == offset
        pages.extend(page["subcomponents"])
        offset = page["next_offset"]
    assert pages == full["subcomponents"]
    assert node_ids(full) == ["L0.o#1@1", "L0.v#0@1", "L0.v#2@1", "L0.v#1@1", "L0.o#0@1"]
    assert all(len(request) <= 2 for request in tool.fact_requests[1:])


def test_ties_use_native_matrix_order_then_index_deterministically():
    tool = Listing()
    for name in tool.nat.names:
        size = len(tool.tx.ci[name].values)
        tool.tx.ci[name] = Table([1] * size)
        tool.tx.rec[name] = Table([1] * size)
    expected = ["L0.v#0@1", "L0.v#1@1", "L0.v#2@1", "L0.o#0@1", "L0.o#1@1"]
    full = tool.list_active(1, top=400)
    assert node_ids(full) == expected
    assert tool.list_active(1, top=400) == full
    assert node_ids(tool.list_active(1, top=2, offset=2)) == expected[2:4]


@pytest.mark.parametrize("offset", [-1, "1", "bad", 1.5, None, True, float("inf")])
def test_invalid_offset_is_a_tool_error(offset):
    with pytest.raises(ToolError, match="non-negative integer"):
        Listing().list_active(1, offset=offset)


@pytest.mark.parametrize("offset", [5, 9])
def test_exhausted_page(offset):
    tool = Listing()
    page = tool.list_active(1, top=2, offset=offset)
    assert page["total"] == page["active"] == 5
    assert page["shown"] == 0 and page["subcomponents"] == []
    assert page["next_offset"] is None
    assert tool.fact_requests == [[]]


def test_site_filter_paginates_only_its_ranked_nodes():
    tool = Listing()
    first = tool.list_active(1, site="L0.v", top=2)
    last = tool.list_active(1, site="L0.v", top=2, offset=first["next_offset"])
    assert first["total"] == last["total"] == 3
    assert node_ids(first) + node_ids(last) == ["L0.v#0@1", "L0.v#2@1", "L0.v#1@1"]
    assert last["next_offset"] is None


def test_per_call_cap_and_default_are_preserved():
    tool = Listing(large=True)
    default = tool.list_active(1)
    assert default["shown"] == 24 and default["next_offset"] == 24
    first = tool.list_active(1, top=999)
    assert first["shown"] == 400 and first["next_offset"] == 400
    last = tool.list_active(1, top=999, offset=400)
    assert last["shown"] == 50 and last["next_offset"] is None
    assert node_ids(first) + node_ids(last) == [f"L0.v#{i}@1" for i in range(450)]
