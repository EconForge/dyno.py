from dyno import DynoModel
from dyno.report import RunResults, dsge_report, _send_interface_notifications
import pandas as pd
import re
import numpy as np
import sys
from types import ModuleType
from unittest.mock import Mock, patch


def _patch_fake_ipython(display_mock: Mock):
    ipython_module = ModuleType("IPython")
    display_module = ModuleType("IPython.display")

    display_module.display = display_mock
    display_module.Markdown = lambda text: text
    display_module.HTML = lambda text: text
    ipython_module.display = display_module

    return patch.dict(
        sys.modules,
        {
            "IPython": ipython_module,
            "IPython.display": display_module,
        },
    )


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


def test_model_markdown_includes_latex_equations_section():
    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]
"""

    model = DynoModel(filename="rbc_like.dyno", txt=txt)

    md = model._markdown_()

    assert "## Equations" in md
    assert "$$" in md


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
    assert md is not None

    assert "Simulation" in md
    assert "<table" in md


def test_runresults_markdown_does_not_inline_simulation_graph_html():
    class DummyFigure:
        def to_html(self, full_html=False, include_plotlyjs="cdn"):
            assert full_html is False
            assert include_plotlyjs == "cdn"
            return "<div>dummy-figure-md</div>"

    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]
"""

    model = DynoModel(filename="rbc_like.dyno", txt=txt)
    model.checks["deterministic"] = False

    results = RunResults(model=model)
    results.simulation = {"e": pd.DataFrame({"x": [0.1, 0.2, 0.3]})}
    results.figure = DummyFigure()

    md = results._repr_markdown_()
    assert md is not None

    assert "Simulation Graph" not in md
    assert "dummy-figure-md" not in md


def test_runresults_html_includes_figure_html():
    class DummyFigure:
        def to_html(self, full_html=False, include_plotlyjs="cdn"):
            assert full_html is False
            assert include_plotlyjs == "cdn"
            return "<div>dummy-figure</div>"

    results = RunResults(output_type="html")
    results.figure = DummyFigure()

    html = results._repr_html_()

    assert "Simulation" in html
    assert "dummy-figure" in html


def test_runresults_html_groups_residuals_and_eigenvalues_in_check_section():
    results = RunResults(output_type="html")
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
    results = RunResults(output_type="html")
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


def test_runresults_markdown_includes_conditional_and_unconditional_moments():
    from dyno.solver import PerturbationSolution, RecursiveDecisionRule

    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 0.04)
x[t] = alpha * x[t-1] + e[t]
"""
    model = DynoModel(filename="moments_sections.dyno", txt=txt)

    dr = RecursiveDecisionRule(
        X=np.array([[0.5]]),
        Y=np.array([[1.0]]),
        Σ=np.array([[0.04]]),
        symbols=model.symbols,
        x0=np.array([0.0]),
        model=model,
    )

    results = RunResults(model=model)
    results.solution = PerturbationSolution(dr, evs=np.array([0.5, 2.0]))
    results.simulation = {"e": pd.DataFrame({"x": [0.0, 0.1, 0.2]})}

    md = results._repr_markdown_()

    assert md is not None
    assert "Unconditional Moments" in md
    assert "Conditional Moments" in md


def test_runresults_html_includes_conditional_and_unconditional_moments():
    from dyno.solver import PerturbationSolution, RecursiveDecisionRule

    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 0.04)
x[t] = alpha * x[t-1] + e[t]
"""
    model = DynoModel(filename="moments_sections_html.dyno", txt=txt)

    dr = RecursiveDecisionRule(
        X=np.array([[0.5]]),
        Y=np.array([[1.0]]),
        Σ=np.array([[0.04]]),
        symbols=model.symbols,
        x0=np.array([0.0]),
        model=model,
    )

    results = RunResults(model=model, output_type="html")
    results.solution = PerturbationSolution(dr, evs=np.array([0.5, 2.0]))
    results.simulation = {"e": pd.DataFrame({"x": [0.0, 0.1, 0.2]})}

    html = results._repr_html_()

    assert html is not None
    assert "<h3>Moments</h3>" in html
    assert "Unconditional Moments" in html
    assert "Conditional Moments" in html


def test_runresults_html_is_opt_in_and_disabled_by_default():
    results = RunResults()
    results.simulation = {"eps": pd.DataFrame({"x": [0.0, 0.2, 0.1]})}

    assert results._repr_html_() is None


def test_runresults_mimebundle_includes_highlighting_and_markdown():
    results = RunResults()
    results.add_error("boom", line=7)

    bundle = results._repr_mimebundle_()

    assert "application/vnd.jupyterlab-dyno.highlighting+json" in bundle
    assert bundle["application/vnd.jupyterlab-dyno.highlighting+json"] == [
        {"line": 7, "type": "error", "message": "boom"}
    ]
    assert "text/markdown" in bundle
    assert "text/html" not in bundle
    assert "text/plain" in bundle


def test_runresults_mimebundle_includes_html_by_default_when_output_type_html():
    results = RunResults(output_type="html")
    results.add_error("boom")

    bundle = results._repr_mimebundle_()

    assert "text/html" in bundle


def test_runresults_mimebundle_defaults_to_text_for_text_output_type():
    results = RunResults(output_type="text")
    results.add_error("boom")

    bundle = results._repr_mimebundle_()

    assert "text/plain" in bundle
    assert "text/markdown" not in bundle
    assert "text/html" not in bundle


def test_runresults_markdown_repr_disabled_for_text_output_type_by_default():
    results = RunResults(output_type="text")

    assert results._repr_markdown_() is None


def test_runresults_markdown_repr_still_available_when_explicitly_selected():
    results = RunResults(output_type="text", mime_bundle_repr="markdown")

    md = results._repr_markdown_()

    assert isinstance(md, str)
    assert md != ""


def test_runresults_mimebundle_can_select_markdown_only():
    results = RunResults(mime_bundle_repr="markdown")
    results.add_error("boom")

    bundle = results._repr_mimebundle_()

    assert "text/markdown" in bundle
    assert "text/html" not in bundle
    assert "text/plain" not in bundle


def test_runresults_mimebundle_can_select_html_only():
    results = RunResults(mime_bundle_repr="html")
    results.add_error("boom")

    bundle = results._repr_mimebundle_()

    assert "text/html" in bundle
    assert "text/markdown" not in bundle
    assert "text/plain" not in bundle


def test_runresults_mimebundle_can_select_text_only():
    results = RunResults(mime_bundle_repr="text")
    results.add_error("boom")

    bundle = results._repr_mimebundle_()

    assert "text/plain" in bundle
    assert "text/markdown" not in bundle
    assert "text/html" not in bundle


def test_runresults_mimebundle_honors_include_exclude_filters():
    results = RunResults()
    results.add_warning("warn", line=3)

    plain_only = results._repr_mimebundle_(include=["text/plain"])
    assert plain_only == {"text/plain": repr(results)}

    no_markdown = results._repr_mimebundle_(exclude=["text/markdown"])
    assert "text/plain" in no_markdown
    assert "text/markdown" not in no_markdown


def test_runresults_mimebundle_uses_same_markdown_renderer():
    results = RunResults()

    bundle = results._repr_mimebundle_()

    assert bundle.get("text/markdown") == results._repr_markdown_()


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


def test_dsge_report_emits_interface_notifications_by_default():
    txt = """
x[t] = y[t-1]
"""

    with patch("dyno.report._send_interface_notifications") as emit_mock:
        dsge_report(txt, filename="notify_default.dyno", display=False)

    emit_mock.assert_called_once()


def test_dsge_report_can_disable_interface_notifications():
    txt = """
x[t] = y[t-1]
"""

    with patch("dyno.report._send_interface_notifications") as emit_mock:
        dsge_report(
            txt,
            filename="notify_disabled.dyno",
            display=False,
            notify_interface=False,
        )

    emit_mock.assert_not_called()


def test_dsge_report_does_not_call_jupyter_display_when_display_true():
    txt = """
x[t] = y[t-1]
"""

    with patch("dyno.report.RunResults.jupyter_display") as jupyter_display_mock:
        dsge_report(txt, filename="display_ignored.dyno", display=True)

    jupyter_display_mock.assert_not_called()


def test_dsge_report_display_graph_calls_display_for_markdown_only():
    txt = """
x[t] = y[t-1]
"""

    with patch("dyno.report.RunResults.display") as display_mock:
        dsge_report(
            txt,
            filename="display_graph_markdown.dyno",
            output_type="markdown",
            display_graph=True,
        )

    display_mock.assert_called_once()


def test_dsge_report_display_graph_does_not_call_display_for_html():
    txt = """
x[t] = y[t-1]
"""

    with patch("dyno.report.RunResults.display") as display_mock:
        dsge_report(
            txt,
            filename="display_graph_html.dyno",
            output_type="html",
            display_graph=True,
        )

    display_mock.assert_not_called()


def test_send_interface_notifications_does_not_emit_markdown_mime():
    results = RunResults()

    display_mock = Mock()
    with _patch_fake_ipython(display_mock):
        _send_interface_notifications(results)

    assert not any(
        call.args and isinstance(call.args[0], dict) and "text/markdown" in call.args[0]
        for call in display_mock.call_args_list
    )


def test_runresults_display_emits_markdown_then_figure():
    class DummyFigure:
        pass

    results = RunResults(output_type="markdown")
    results.figure = DummyFigure()

    display_mock = Mock()
    with _patch_fake_ipython(display_mock):
        results.display()

    assert len(display_mock.call_args_list) >= 2
    assert display_mock.call_args_list[-1].args[0] is results.figure


def test_runresults_display_text_mode_emits_plain_text():
    results = RunResults(output_type="text")

    display_mock = Mock()
    with _patch_fake_ipython(display_mock):
        results.display()

    assert any(
        call.args
        and isinstance(call.args[0], dict)
        and "text/plain" in call.args[0]
        and call.kwargs.get("raw") is True
        for call in display_mock.call_args_list
    )
