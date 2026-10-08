import pytest

from dyno.dynspec.latex import latex


@pytest.mark.parametrize(
    "src, expected",
    [
        ("x[t]", "x_{t}"),
        ("x[t+1]", "x_{t+1}"),
        ("x[~]", "\\bar{x}"),
        ("pi_1[t]", "\\pi_{1,t}"),
        ("pi_1[t+1]", "\\pi_{1,t+1}"),
        ("pi_1[~]", "\\bar{\\pi}_{1}"),
        ("pi[~]", "\\bar{\\pi}"),
    ],
)
def test_latex_variables(src, expected):
    assert latex(src) == expected
