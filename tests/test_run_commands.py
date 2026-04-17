from dyno import DynoModel, DynareModel
from dyno.report import RunResults


def test_dynomodel_modfile_metadata_includes_dynare_commands():
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
stoch_simul(irf=8);
"""
    model = DynoModel(filename="tiny.mod", txt=txt)

    commands = model._normalize_run_commands()

    assert [c["command"] for c in commands] == ["steady", "check", "stoch_simul"]
    assert commands[2]["options"].get("irf") == 8


def test_dynomodel_run_accepts_stoch_simul_command():
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
stoch_simul(irf=6);
"""
    model = DynoModel(filename="tiny.mod", txt=txt)

    results = model.run(default_pipeline=False)

    assert isinstance(results, RunResults)
    assert results.solution is not None
    assert results.eigenvalues is not None
    assert results.simulation is not None
    assert isinstance(results.simulation, dict)


def test_dynomodel_run_check_populates_eigenvalues():
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
    model = DynoModel(filename="tiny_check.mod", txt=txt)

    results = model.run(default_pipeline=False)

    assert isinstance(results, RunResults)
    assert results.residuals is not None
    assert results.eigenvalues == getattr(results.model, "_eigenvalues", None)


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
