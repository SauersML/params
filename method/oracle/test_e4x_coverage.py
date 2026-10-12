"""Model-free regression checks for mechanism coverage in the variable reward."""

import ast
import re
from pathlib import Path

import pytest


@pytest.fixture
def e4x():
    # Compile only this arithmetic helper: these probes need no model or tensor runtime.
    source = Path(__file__).with_name("components.py")
    tree = ast.parse(source.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "e4x_term")
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["e4x_term"]


def score(e4x, variables, masses=(3.0, 1.0)):
    le = {"nodes": [{"node": ["site", 0, i], "delta_kl": mass} for i, mass in enumerate(masses)]}
    return e4x(None, None, None, {"per_variable": variables}, None, le)


def variable(e4, *indices):
    return {"e4": e4, "nodes": [["site", 0, i] for i in indices]}


@pytest.mark.parametrize("copies", [1, 2, 10])
def test_repeated_passing_claim_cannot_inflate_coverage(e4x, copies):
    assert score(e4x, [variable(1.0, 0, 1)] * copies) == 1.0


def test_repeated_nodes_within_group_count_once(e4x):
    assert score(e4x, [variable(1.0, 0, 0)]) == 0.75


def test_disjoint_claims_preserve_measured_weighting(e4x):
    assert score(e4x, [variable(1.0, 0), variable(0.0, 1)]) == 0.5


def test_overlap_uses_lowest_claim_and_preserves_unclaimed_mass(e4x):
    assert score(e4x, [variable(0.8, 0), variable(0.2, 0)]) == pytest.approx(-0.45)


@pytest.mark.parametrize("copies", [1, 2, 10])
def test_passing_duplicates_cannot_hide_failing_overlap(e4x, copies):
    claims = [variable(1.0, 0, 1)] * copies + [variable(0.0, 0)]
    assert score(e4x, claims) == -0.5
    assert score(e4x, list(reversed(claims))) == -0.5


def test_outside_nodes_and_nonpositive_mass_do_not_earn_coverage(e4x):
    assert score(e4x, [variable(1.0, 1, 2)], masses=(3.0, -1.0)) == 0.0


def test_no_positive_mass_is_undefined(e4x):
    assert score(e4x, [variable(1.0, 0, 1)], masses=(0.0, -1.0)) is None


def test_no_claims_get_zero_coverage(e4x):
    assert score(e4x, []) == 0.0


@pytest.fixture
def line_credit(e4x):
    source = Path(__file__).with_name("components.py")
    tree = ast.parse(source.read_text())
    # Use the production section/variable patterns as well as the production function.
    patterns = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in {"SECTION", "VAR_RE"} for target in node.targets)]
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "line_credit")
    namespace = {"re": re, "e4x_term": e4x}
    exec(compile(ast.Module(body=patterns + [fn], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["line_credit"]


def credits(line_credit, claims, masses=(3.0, 1.0)):
    text = "EXPLANATION:\n" + "\n".join(
        f'V{i} "claim": meaning; inputs: @0; carried by: a; feeds: PREDICTION'
        for i in range(1, len(claims) + 1))
    le = {"nodes": [{"node": ["site", 0, i], "delta_kl": mass} for i, mass in enumerate(masses)]}
    result = line_credit(text, {}, {"nodes": []}, le, {}, None, {"per_variable": claims}, {}, [], None)
    return [value for _, value in result]


@pytest.mark.parametrize("copies", [2, 10])
def test_duplicate_variable_lines_have_no_unique_credit(line_credit, copies):
    assert credits(line_credit, [variable(1.0, 0, 1)] * copies) == [0.0] * copies


def test_disjoint_variable_line_credit_keeps_signed_mass(line_credit):
    assert credits(line_credit, [variable(1.0, 0), variable(0.0, 1)]) == [0.75, -0.25]


def test_mixed_overlap_line_credit_is_actual_marginal(line_credit):
    assert credits(line_credit, [variable(1.0, 0, 1), variable(0.0, 0)]) == [0.25, -1.5]


def test_passing_duplicates_do_not_hide_negative_line_credit(line_credit):
    assert credits(line_credit, [variable(1.0, 0, 1)] * 3 + [variable(0.0, 0)]) == [0.0, 0.0, 0.0, -1.5]


def test_failing_duplicates_have_no_unique_line_credit(line_credit):
    assert credits(line_credit, [variable(0.0, 0), variable(0.0, 0)]) == [0.0, 0.0]


def test_partial_overlap_credits_only_unique_coverage(line_credit):
    assert credits(line_credit, [variable(1.0, 0, 1), variable(1.0, 0)]) == [0.25, 0.0]
