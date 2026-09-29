from pathlib import Path
import pandas as pd
import pytest

from dyno import DynoModel, examples_path
from dyno.simul import sim_to_nsim


def test_steady_state_operator_in_equations():
    txt = """
    var y, yhat;
    varexo e;
    parameters rho;
    rho = 0.8;
    model;
    y = rho * y(-1) + e;
    yhat = y - steady_state(y);
    end;
    initval;
    y = 2.0;
    yhat = 0.0;
    end;
    """
    model = DynoModel(txt=txt, filename="test.mod")
    assert "yhat" in model.symbols["variables"]
    # Check that residual of yhat = y - steady_state(y) is 0 at initval (2 - 2 = 0)
    residuals = model.residuals
    assert residuals is not None
    assert abs(residuals[1]) < 1e-12


def test_dynomodel_run_sanitizes_stoch_simul_options():
    txt = """
    var c;
    varexo e;
    parameters beta;
    beta = 0.95;
    model;
    c = beta * c(-1) + e;
    end;
    initval;
    c = 0.0;
    end;
    stoch_simul(periods=100, drop=20, irf=20);
    """
    model = DynoModel(txt=txt, filename="test.mod")
    # run() must not crash with unexpected keyword argument 'drop'
    res = model.run(default_pipeline=False)
    assert res.solution is not None
    assert res.simulation is not None


def test_sim_to_nsim_handles_preexisting_time_column():
    # Simulate dataframe as returned by deterministic_solve having a 't' column
    df = pd.DataFrame({"t": [0, 1, 2], "c": [1.0, 1.1, 1.2], "k": [2.0, 2.1, 2.2]})
    irfs = {"Simulation": df}
    nsim = sim_to_nsim(irfs)
    assert "shock" in nsim.columns
    assert "t" in nsim.columns
    assert "variable" in nsim.columns
    assert "value" in nsim.columns
    assert len(nsim) == 6  # 3 time periods * 2 variables


def test_dynare_model_strict_vs_nonstrict_undeclared_params():
    try:
        from dyno.dynare import DynareModel

        # check preprocessor
        from dynare_preprocessor import DynareModel as _
    except (ModuleNotFoundError, ImportError):
        pytest.skip("dynare-preprocessor-pylib not available")

    mod_path = examples_path("modfiles", "example1.mod")
    # strict=False should allow undeclared params and succeed
    m_nostrict = DynareModel(mod_path, strict=False)
    assert m_nostrict is not None

    # strict=True should disallow undeclared params and raise DynareParserError
    from dyno.errors import DynareParserError

    with pytest.raises(DynareParserError):
        DynareModel(mod_path, strict=True)
