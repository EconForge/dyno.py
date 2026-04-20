import numpy as np
import pytest

from dyno import DynareModel


def test_dynare_recalibrate_returns_new_model_without_mutation() -> None:
    txt = """
var x;
varexo e;
parameters alpha beta;

alpha = 0.9;
beta = 2.0;

model;
x = alpha*x(-1) + beta + e;
end;

initval;
x = 0;
e = 0;
end;

shocks;
var e = 0.01;
end;
"""

    model = DynareModel(filename="<test>.mod", txt=txt)
    original_alpha = model.context["constants"]["alpha"]

    recalibrated = model.recalibrate(alpha=0.8)

    assert recalibrated is not model
    assert model.context["constants"]["alpha"] == pytest.approx(original_alpha)
    assert recalibrated.context["constants"]["alpha"] == pytest.approx(0.8)


def test_dynare_steady_returns_new_model_with_solved_steady_state() -> None:
    txt = """
var x;
varexo e;
parameters alpha beta;

alpha = 0.9;
beta = 2.0;

model;
x = alpha*x(-1) + beta + e;
end;

initval;
x = 0;
e = 0;
end;

shocks;
var e = 0.01;
end;
"""

    model = DynareModel(filename="<test>.mod", txt=txt)
    before = dict(model.context["steady_states"])

    solved = model.steady()

    assert solved is not model
    assert solved.context["steady_states"]["x"] == pytest.approx(20.0)
    assert np.max(np.abs(solved.residuals)) < 1e-8
    assert model.context["steady_states"] == before


def test_dynare_run_executes_commands_from_metadata() -> None:
    from dyno import DynareRunResults

    txt = """
var x;
varexo e;
parameters alpha beta;

alpha = 0.9;
beta = 2.0;

model;
x = alpha*x(-1) + beta + e;
end;

initval;
x = 0;
e = 0;
end;

shocks;
var e = 0.01;
end;

steady;
stoch_simul;
"""

    model = DynareModel(filename="<run_test>.mod", txt=txt)

    # metadata["dynare_commands"] should contain steady + stoch_simul
    run_commands = [c["command"] for c in model.metadata["dynare_commands"]]
    assert "steady" in run_commands
    assert "stoch_simul" in run_commands

    results = model.run()

    assert isinstance(results, DynareRunResults)
    # steady should have updated the model's steady state
    assert results.model.context["steady_states"]["x"] == pytest.approx(20.0)
    assert np.max(np.abs(results.model.residuals)) < 1e-8
    # stoch_simul should have produced a solution
    assert results.solution is not None
    # original model must be untouched
    assert model.context["steady_states"]["x"] != pytest.approx(20.0)


def test_dynare_run_check_populates_residuals() -> None:
    txt = """
var x;
varexo e;
parameters alpha beta;

alpha = 0.9;
beta = 2.0;

model;
x = alpha*x(-1) + beta + e;
end;

initval;
x = 0;
e = 0;
end;

shocks;
var e = 0.01;
end;

steady;
check;
"""

    model = DynareModel(filename="<run_check>.mod", txt=txt)

    results = model.run()

    assert results.residuals is not None
    assert np.max(np.abs(results.residuals)) < 1e-8


def test_dynare_commands_omit_param_init_and_initval() -> None:
    model = DynareModel("examples/modfiles/example2.mod", allow_undeclared_params=True)
    names = [c["command"] for c in model.metadata["dynare_commands"]]

    assert "param_init" not in names
    assert "initval" not in names
    assert "steady" in names
    assert len(names) > 0
