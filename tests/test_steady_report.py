import pytest
import numpy as np
import pandas as pd
from dyno import DynoModel
from dyno.report import RunResults


def test_model_steady_computes_and_attaches_stats():
    txt = """
@name: Neoclassical
k[~] := 1.0
c[~] := 0.5
alpha := 0.33
beta := 0.99
delta := 0.025
sigma := 2.0
A := 1.0

c[t]^(-sigma) = beta * c[t+1]^(-sigma) * (alpha * A * k[t]^(alpha - 1) + 1 - delta)
k[t] = A * k[t-1]^alpha + (1 - delta) * k[t-1] - c[t]
"""
    model = DynoModel(txt=txt)
    solved = model.steady()

    stats = solved.steady_stats
    assert stats is not None
    assert stats["algorithm"] == "hybr"
    assert stats["converged"] is True
    assert stats["iterations"] is not None
    assert stats["function_evaluations"] is not None
    assert stats["max_residual"] < 1e-8
    assert stats["tolerance"] == 1e-10
    assert "converged" in stats["message"].lower()


def test_runresults_includes_steady_section_in_text_html_markdown():
    txt = """
@name: Neoclassical
k[~] := 1.0
c[~] := 0.5
alpha := 0.33
beta := 0.99
delta := 0.025
sigma := 2.0
A := 1.0

c[t]^(-sigma) = beta * c[t+1]^(-sigma) * (alpha * A * k[t]^(alpha - 1) + 1 - delta)
k[t] = A * k[t-1]^alpha + (1 - delta) * k[t-1] - c[t]

@run: steady
@run: check
"""
    model = DynoModel(txt=txt)
    results = model.run(default_pipeline=False)

    assert isinstance(results, RunResults)
    assert results.steady_stats is not None
    assert results.steady_info == results.steady_stats
    assert results._should_render_steady is True

    # 1. Plain text output
    txt_rep = results.to_text()
    assert "Steady-state calculation" in txt_rep
    assert "algorithm: hybr" in txt_rep
    assert "status: converged" in txt_rep
    assert "max residual:" in txt_rep
    assert "tolerance: 1.000e-10" in txt_rep

    # 2. HTML output
    html_rep = results.to_html()
    assert "<h3>Steady-state calculation</h3>" in html_rep
    assert "<details" in html_rep
    assert "<summary" in html_rep
    assert "Steady-state converged" in html_rep
    assert "Algorithm" in html_rep
    assert "hybr" in html_rep
    assert "Converged" in html_rep
    assert "Max residual" in html_rep
    assert "Tolerance" in html_rep

    # Verify _repr_html_ matches
    assert results._repr_html_() == html_rep

    # 3. Markdown output
    md_rep = results.to_markdown()
    assert "## Steady-state calculation" in md_rep
    assert "Steady-state converged" in md_rep
    assert ":class: dropdown" in md_rep
    assert "**Algorithm**: `hybr`" in md_rep
    assert "**Max residual**:" in md_rep
    assert "**Tolerance**:" in md_rep


def test_report_without_steady_does_not_render_steady_section():
    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: check
@run: solve
"""
    model = DynoModel(txt=txt)
    results = model.run(default_pipeline=False)

    assert results.steady_stats is None
    assert results._should_render_steady is False

    assert "Steady-state calculation" not in results.to_text()
    assert "Steady-state calculation" not in results.to_html()
    assert "Steady-state calculation" not in results.to_markdown()


def test_steady_semicolon_mutes_steady_section_when_converged():
    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: steady;
@run: check
"""
    model = DynoModel(txt=txt)
    results = model.run(default_pipeline=False)

    assert results.steady_stats is not None
    assert "steady" in results.muted_commands
    assert results._should_render_steady is False

    assert "Steady-state calculation" not in results.to_text()
    assert "Steady-state calculation" not in results.to_html()
    assert "Steady-state calculation" not in results.to_markdown()


def test_steady_variants_reporting():
    txt = """
@name: RBC_Var
alpha := 0.33
beta := 0.99
delta := 0.025
k[~] := 1.0
c[~] := 0.5
c[t]^(-1) = beta * c[t+1]^(-1) * (alpha * k[t]^(alpha - 1) + 1 - delta)
k[t] = k[t-1]^alpha + (1 - delta) * k[t-1] - c[t]

@run: variants: {alpha: [0.30, 0.35]}
@run: steady
"""
    model = DynoModel(txt=txt)
    results = model.run(default_pipeline=False)

    assert hasattr(results, "steady_stats")
    stats_list = results.steady_stats
    assert len(stats_list) == 2
    assert all(s is not None and s["converged"] is True for s in stats_list)

    txt_rep = results.to_text()
    assert "Steady-state calculation" in txt_rep
    assert "algorithm: hybr" in txt_rep

    html_rep = results.to_html()
    assert "<h3>Steady-state calculation</h3>" in html_rep
    assert "<details" in html_rep
    assert "Converged" in html_rep

    md_rep = results.to_markdown()
    assert "## Steady-state calculation" in md_rep
    assert "Steady-state converged for all variants" in md_rep
    assert ":class: dropdown" in md_rep


def test_modfile_steady_populates_steady_stats():
    txt = """
var x;
varexo e;
parameters a;

a = 0.9;

model;
x = a*x(-1) + e;
end;

initval;
x = 0;
e = 0;
end;

steady;
check;
"""
    model = DynoModel(filename="tiny.mod", txt=txt)
    results = model.run(default_pipeline=False)
    assert results.steady_stats is not None
    assert results.steady_stats["converged"] is True
    assert "Steady-state calculation" in results.to_text()
    assert "<h3>Steady-state calculation</h3>" in results.to_html()
    assert "<details" in results.to_html()
    assert "## Steady-state calculation" in results.to_markdown()
    assert ":class: dropdown" in results.to_markdown()


def test_runresults_includes_bk_eigenvalue_explanation():
    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: check
"""
    model = DynoModel(txt=txt)
    results = model.run(default_pipeline=False)
    md_rep = results.to_markdown()
    assert "Sorted by modulus. Exactly 1 eigenvalue should be larger than 1" in md_rep
