import numpy as np
import pandas as pd
import pytest

from dyno.plots import (
    plot_irf_plotext,
    plot_irfs,
    plot_irfs_plotext,
    plot_simulation_plotext,
)
from dyno.report import RunResults


def test_plot_simulation_plotext_handles_empty_or_none():
    assert plot_simulation_plotext(None) == ""
    assert plot_simulation_plotext({}) == ""
    assert plot_simulation_plotext(pd.DataFrame()) == ""


def test_plot_simulation_plotext_dict_irfs():
    sim = {
        "shock_a": pd.DataFrame({"y": [0.0, 1.0, 0.5], "c": [0.0, 0.4, 0.2]}),
        "shock_b": pd.DataFrame({"y": [0.0, 0.2, 0.1], "c": [0.0, 0.1, 0.05]}),
    }
    output = plot_simulation_plotext(sim, color=False, width=80)
    assert isinstance(output, str)
    assert "y" in output
    assert "c" in output
    assert "shock_a" in output
    assert "shock_b" in output
    # Uncolored should not contain ANSI escape codes
    assert "\033[" not in output


def test_plot_simulation_plotext_dataframe():
    df = pd.DataFrame({"t": [0, 1, 2], "y": [1.0, 2.0, 1.5], "i": [0.5, 1.0, 0.8]})
    output = plot_simulation_plotext(df, color=False, width=70)
    assert isinstance(output, str)
    assert "y" in output
    assert "i" in output


def test_plot_simulation_plotext_variable_filtering():
    sim = {
        "e": pd.DataFrame({"y": [1.0, 0.5], "c": [0.5, 0.2], "k": [2.0, 2.1]}),
    }
    output = plot_simulation_plotext(sim, variables=["y", "k"], color=False)
    assert "y" in output
    assert "k" in output
    assert "c" not in output


def test_plot_simulation_plotext_odd_number_of_subplots():
    # 3 variables with cols=2 should produce a 2x2 grid with 1 blank cell
    sim = {
        "e": pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0], "c": [0.5, 0.5]}),
    }
    output = plot_simulation_plotext(sim, cols=2, color=False)
    assert "a" in output
    assert "b" in output
    assert "c" in output


def test_plot_simulation_plotext_color_flag():
    sim = {"e": pd.DataFrame({"y": [0.0, 1.0]})}
    colored = plot_simulation_plotext(sim, color=True)
    uncolored = plot_simulation_plotext(sim, color=False)
    assert "\033[" in colored
    assert "\033[" not in uncolored


def test_plot_irfs_engine_plotext():
    sim = {"e": pd.DataFrame({"x": [0.0, 1.0]})}
    res = plot_irfs(sim, engine="plotext", color=False)
    assert isinstance(res, str)
    assert "x" in res

    single_df = pd.DataFrame({"x": [0.0, 1.0]})
    res_single = plot_irf_plotext(single_df, color=False)
    assert isinstance(res_single, str)
    assert "x" in res_single


def test_runresults_plot_text():
    results = RunResults()
    assert results.plot_text() == ""

    results.simulation = {
        "eps": pd.DataFrame({"y": [0.0, 0.5, 0.2], "c": [0.0, 0.2, 0.1]})
    }
    txt = results.plot_text(color=False)
    assert isinstance(txt, str)
    assert "y" in txt
    assert "c" in txt


def test_runresults_to_text_with_and_without_graphs():
    results = RunResults()
    results.simulation = {"eps": pd.DataFrame({"x": [0.0, 0.5, 0.2]})}

    with_graphs = results.to_text(graphs=True, color=False)
    assert "Simulation Plots" in with_graphs
    assert "x" in with_graphs

    without_graphs = results.to_text(graphs=False)
    assert "Simulation Plots" not in without_graphs


def test_runresults_str_includes_simulation_plots_when_available():
    results = RunResults()
    results.residuals = np.array([0.0])
    results.simulation = {"eps": pd.DataFrame({"output": [0.0, 1.2, 0.8]})}

    text_report = str(results)
    assert "RunResults" in text_report
    assert "Outputs" in text_report
    assert "Simulation Plots" in text_report
    assert "output" in text_report


def test_runresults_console_display_prints_graph(capsys):
    results = RunResults()
    results.simulation = {"eps": pd.DataFrame({"x": [0.0, 1.0]})}
    results.console_display()

    captured = capsys.readouterr()
    assert "Simulation: computed" in captured.out or "IRFs: 1 shock(s)" in captured.out
    assert "x" in captured.out
