"""Tests for complementarity conditions (`res ⟂ lb <= x <= ub`) in the dyno grammar."""

import math
import warnings

import pytest

from dyno import DynoModel
from dyno.errors import DefinitionError
from dyno.larkfiles import DynoFile
from dyno.dynspec.analyze import FormulaEvaluator
from dyno.dynspec.grammar import parser, str_expression
from dyno.dynspec.latex import latex
from dyno.dynspec.recipe import extract_variables_from_equation


def parse_equation(src):
    return parser.parse(src + "\n", start="equation_block").children[0]


@pytest.mark.parametrize(
    "src, expected",
    [
        ("x[t] + y[t] ⟂ 0 <= x[t] <= inf", "x[t] + y[t] ⟂ 0 <= x[t] <= inf"),
        ("x[t] + y[t] ⟂ x[t] <= h", "x[t] + y[t] ⟂ -(inf) <= x[t] <= h"),
        ("x[t] + y[t] ⟂ 0 <= x[t]", "x[t] + y[t] ⟂ 0 <= x[t] <= inf"),
        # legacy dolo separator
        ("x[t] + y[t] | -inf <= x[t]", "x[t] + y[t] ⟂ -(inf) <= x[t] <= inf"),
        ("1 - z[t+1] ⟂ k[t-1] <= x[t] <= 2*k[t-1]", None),
        # with two variables, the bounded one is dated [t]
        ("a[t] ⟂ k[t-1] <= x[t]", "a[t] ⟂ k[t-1] <= x[t] <= inf"),
        ("a[t] ⟂ x[t] <= k[t-1]", "a[t] ⟂ -(inf) <= x[t] <= k[t-1]"),
    ],
)
def test_complementarity_is_normalized(src, expected):
    tree = parse_equation(src)
    assert tree.data == "complementarity"
    assert len(tree.children) == 4
    assert tree.children[2].data == "variable"
    if expected is not None:
        assert str_expression(tree) == expected


@pytest.mark.parametrize(
    "src, message",
    [
        ("a[t] ⟂ x[t] <= y[t]", "ambiguous"),
        ("a[t] ⟂ 0 <= 1 <= 2", "middle term"),
        ("a[t] ⟂ 0 <= 1", "must bound a variable"),
    ],
)
def test_invalid_complementarity(src, message):
    with pytest.raises(DefinitionError, match=message) as exc:
        parser.parse("\nh <- 1\n" + src + "\n", start="free_block")
    assert exc.value.line == 3


def test_residual_ignores_bounds():
    tree = parse_equation("x[t] - 2 ⟂ 0 <= x[t] <= inf")
    ev = FormulaEvaluator(context={"variables": {"x": {0: 5.0}}})
    assert ev.visit(tree) == 3.0
    assert ev.visit(tree.children[3]) == math.inf
    assert ev.visit(tree.children[1]) == 0


def test_variables_come_from_residual_only():
    tree = parse_equation("c[t+1] - w[t] ⟂ k[t-1] <= n[t]")
    assert extract_variables_from_equation(tree) == {"c": {1}, "w": {0}}


def test_latex_omits_infinite_bounds():
    assert latex(parse_equation("a[t] ⟂ 0 <= n[t] <= inf")) == r"a_{t} \perp 0 \leq n_{t}"
    assert latex(parse_equation("a[t] ⟂ i[t] <= h")) == r"a_{t} \perp i_{t} \leq h"
    assert latex(parse_equation("a[t] ⟂ -inf <= i[t]")) == r"a_{t} \perp i_{t}"


def test_rbc_dolo_comp_example():
    with open("examples/rbc_dolo_comp.dyno", encoding="utf-8") as f:
        txt = f.read()
    model = DynoFile(txt, "rbc_dolo_comp.dyno")
    comps = [eq for eq in model.equations if eq.data == "complementarity"]
    assert [str_expression(eq.children[2]) for eq in comps] == ["n[t]", "i[t]"]
    assert max(abs(r) for r in model.residuals) < 1e-12

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DynoModel("examples/rbc_dolo_comp.dyno").solve()
