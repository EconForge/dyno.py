from dyno import DynoModel
import re


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_model_repr_contains_summary_fields():
    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]
"""

    model = DynoModel(filename="rbc_like.dyno", txt=txt)

    rep = _strip_ansi(repr(model))

    assert isinstance(rep, str)
    assert "MODEL: rbc_like" in rep
    assert "rbc_like" in rep
    assert "equations" in rep
    assert "variables" in rep
    assert "endogenous" in rep
    assert "x" in rep
    assert "exogenous" in rep
    assert "e" in rep
    assert "constants" in rep
    assert "alpha" in rep


def test_model_repr_marks_uninitialized_with_caret_and_footnote():
    txt = """
x[t] = a + y[t-1]
"""

    model = DynoModel(filename="missing_values.dyno", txt=txt)

    rep = _strip_ansi(repr(model))

    assert "a^" in rep
    assert "x^" in rep or "y^" in rep
    assert "uninitialized" in rep


def test_model_repr_html_uses_three_column_table_and_footnote():
    txt = """
x[t] = a + y[t-1]
"""

    model = DynoModel(filename="missing_values.dyno", txt=txt)

    html = model._repr_html_()

    assert "<table>" in html
    assert "<th>" not in html
    assert "<td>equations</td>" in html
    assert "endogenous" in html
    assert "exogenous" in html
    assert "constants" in html
    assert "<sup>^</sup>" in html
    assert "uninitialized" in html
