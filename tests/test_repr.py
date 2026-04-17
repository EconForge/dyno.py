from dyno import DynoModel
from dyno.report import RunResults, dsge_report
import pandas as pd
import re
import numpy as np


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


def test_runresults_markdown_accepts_series_simulation_entries():
    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]
"""

    model = DynoModel(filename="rbc_like.dyno", txt=txt)
    model.checks["deterministic"] = False

    results = RunResults(model=model)
    results.simulation = {"e": pd.Series([0.1, 0.2, 0.3], name="x")}

    md = results._repr_markdown_()

    assert "Simulation" in md
    assert "<table" in md


def test_runresults_html_includes_figure_html():
    class DummyFigure:
        def to_html(self, full_html=False, include_plotlyjs="cdn"):
            assert full_html is False
            assert include_plotlyjs == "cdn"
            return "<div>dummy-figure</div>"

    results = RunResults()
    results.figure = DummyFigure()

    html = results._repr_html_()

    assert "Simulation" in html
    assert "dummy-figure" in html


def test_runresults_html_groups_residuals_and_eigenvalues_in_check_section():
    results = RunResults()
    results.residuals = pd.Series([0.0, 1.25, -0.5])
    results.eigenvalues = pd.Series([0.9, 1.1, 1.4])

    html = results._repr_html_()

    assert "<h3>Check</h3>" in html
    assert "Residuals" in html
    assert "Generalized Eigenvalues" in html
    assert "<h3>Generalized Eigenvalues</h3>" not in html
    assert html.count("<tr>") >= 2
    assert "1.25" in html
    assert "1.4" in html


def test_runresults_html_includes_static_svg_for_simulation():
    results = RunResults()
    results.simulation = {
        "eps": pd.DataFrame(
            {
                "x": [0.0, 0.2, 0.1],
                "y": [0.0, -0.1, 0.05],
            }
        )
    }

    html = results._repr_html_()

    assert "Simulation" in html
    assert "<svg" in html
    assert "polyline" in html


def test_dsge_report_handles_steady_state_error_gracefully():
    """Test that dsge_report() gracefully handles models with bad steady states.
    
    This test verifies that when a check command fails on a model,
    dsge_report() returns a partial report with residuals rather than crashing.
    """
    # Use explicit check command in metadata to trigger steady-state validation
    txt = """
alpha := 0.9
beta := 2.0

x[~] := 1.0
x[t] = alpha*x[t-1] + beta

check;
"""
    # Call dsge_report with a model that has an explicit check command
    # The check command will fail because the steady state is inconsistent
    results = dsge_report(txt, filename="bad_steady_state.dyno")
    
    # Should return RunResults, not raise exception
    assert isinstance(results, RunResults)
    
    # Should have a warning or error from the failed check
    assert len(results.warnings) > 0 or len(results.errors) > 0


