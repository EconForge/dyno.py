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
