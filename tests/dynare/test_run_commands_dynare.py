from dyno import DynareModel
from dyno.report import RunResults


def test_dynaremodel_run_returns_report_on_failed_check():
    txt = """
var x;
varexo e;
parameters a;

a = 0.9;

model;
x = a*x(-1) + 1 + e;
end;

initval;
x = 0;
e = 0;
end;

shocks;
var e = 0.01;
end;

check;
stoch_simul(irf=5);
"""
    model = DynareModel(filename="bad_check.mod", txt=txt)

    results = model.run(default_pipeline=False)

    assert isinstance(results, RunResults)
    assert results.residuals is not None
    assert len(results.warnings) > 0
    assert any("line" in entry for entry in results.warnings)
    # Execution should stop once steady-state validation fails.
    assert results.solution is None


def test_dynaremodel_stoch_simul_renders_plot_with_symbol_list():
    txt = """
var x y;
varexo e;
parameters a;

a = 0.9;

model;
x = a*x(-1) + e;
y = 2*x;
end;

initval;
x = 0;
y = 0;
e = 0;
end;

shocks;
var e = 0.01;
end;

steady;
stoch_simul(order=1, irf=8) y;
"""
    model = DynareModel(filename="plot.mod", txt=txt)

    results = model.run(default_pipeline=False)

    assert results.figure is not None
    assert results._plot_options["variables"] == ["y"]
    assert set(results.figure.data["variable"]) == {"y"}
    # The chart must reach the Markdown/MyST report.
    assert "Simulation charts" in results.to_markdown()


def test_dynaremodel_stoch_simul_nograph_skips_plot():
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

shocks;
var e = 0.01;
end;

steady;
stoch_simul(order=1, nograph);
"""
    results = DynareModel(filename="nograph.mod", txt=txt).run(default_pipeline=False)

    assert results.solution is not None
    assert results.figure is None
