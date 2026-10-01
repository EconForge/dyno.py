"""Perfect-foresight (deterministic) Dynare .mod files through DynoModel."""

import numpy as np
import pytest

from dyno import DynoModel
from dyno.errors import UnsupportedFeatureError
from dyno.simul import TransitionSimulation
from dyno.solver import deterministic_solve

RAMST_HEAD = """\
var c k;
varexo x;
parameters alph gam delt bet aa;
alph=0.5; gam=0.5; delt=0.02; bet=0.05; aa=0.5;
model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;
"""


def test_ramst_mod_runs_like_ramst_dyno():
    mod = DynoModel("examples/modfiles/ramst.mod")
    assert mod.is_deterministic
    assert mod.symbols["exogenous"] == ["x"]
    assert mod.steady_state["x"] == 1.0  # initval value, not zero
    assert mod.context["values"] == {"x": {1: 1.2}}
    assert [c["command"] for c in mod.metadata["dynare_commands"]] == [
        "steady",
        "check",
        "simul",
        "plot",
    ]
    assert mod.metadata["dynare_commands"][2]["options"] == {
        "mode": "deterministic",
        "T": 200,
    }
    assert mod.metadata["dynare_commands"][3]["options"] == {"variables": ["c", "k"]}

    results = mod.run()
    sim = results.simulation
    assert isinstance(sim, TransitionSimulation)
    assert sim.attrs["converged"]
    assert sim.T == 200
    df = sim.to_df()
    assert np.isclose(df["x"].iloc[1], 1.2)
    assert np.allclose(
        df[["c", "k"]].iloc[-1], [mod.steady_state[v] for v in ["c", "k"]]
    )
    assert results.figure is not None

    dyno = DynoModel("examples/ramst.dyno")
    a = deterministic_solve(mod, T=50).to_df()
    b = deterministic_solve(dyno, T=50).to_df()
    np.testing.assert_allclose(
        a[["c", "k", "x"]].values, b[["c", "k", "x"]].values, atol=1e-10
    )


def test_steady_state_model_is_evaluated_after_initval():
    # steady_state_model refers to the exogenous z, declared in a later initval block
    model = DynoModel("examples/dynare/perfect_foresight/perfect_foresight_rbc.mod")
    assert model.steady_state["z"] == 1.0
    assert np.all(np.isfinite(list(model.steady_state.values())))
    assert np.abs(model.residuals).max() < 1e-10
    sim = model.run().simulation
    assert sim.attrs["converged"]


def test_periods_ranges_and_value_expressions():
    txt = RAMST_HEAD + """
initval; x = 1; k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1)); c = aa*k^alph-delt*k; end;
shocks;
var x;
periods 1:3, 5 7:8;
values 1.1, -0.5 (1 + aa);
end;
simul(periods=20);
"""
    model = DynoModel(txt=txt, filename="ranges.mod")
    assert model.context["values"]["x"] == {
        1: 1.1,
        2: 1.1,
        3: 1.1,
        5: -0.5,
        7: 1.5,
        8: 1.5,
    }
    assert model.metadata["dynare_commands"][-1] == {
        "command": "simul",
        "options": {"mode": "deterministic", "T": 20},
    }


def test_endval_sets_terminal_state_and_initval_pins_date_zero():
    txt = RAMST_HEAD + """
initval; x = 1; k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1)); c = aa*k^alph-delt*k; end;
endval; x = 1.2; k = ((delt+bet)/(1.2*aa*alph))^(1/(alph-1)); c = aa*1.2*k^alph-delt*k; end;
simul(periods=100);
"""
    model = DynoModel(txt=txt, filename="endval.mod")
    k_init = ((0.02 + 0.05) / (1.0 * 0.5 * 0.5)) ** (1 / (0.5 - 1))
    k_term = ((0.02 + 0.05) / (1.2 * 0.5 * 0.5)) ** (1 / (0.5 - 1))
    assert np.isclose(model.steady_state["x"], 1.2)
    assert np.isclose(model.steady_state["k"], k_term)
    assert np.isclose(model.context["values"]["x"][0], 1.0)
    assert np.isclose(model.context["values"]["k"][0], k_init)
    assert np.abs(model.residuals).max() < 1e-10  # terminal steady state is exact

    sim = model.run().simulation
    assert sim.attrs["converged"]
    df = sim.to_df()
    assert np.isclose(df["k"].iloc[0], k_init)
    assert np.allclose(df["x"].iloc[1:], 1.2)
    assert np.isclose(df["k"].iloc[-1], k_term, rtol=1e-6)


def test_histval_pins_date_zero_and_rejects_lags():
    base = RAMST_HEAD + """
initval; x = 1; k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1)); c = aa*k^alph-delt*k; end;
"""
    model = DynoModel(
        txt=base + "histval; k(0) = 10; end;\nsimul(periods=30);\n", filename="h.mod"
    )
    assert model.context["values"]["k"] == {0: 10.0}
    df = model.run().simulation.to_df()
    assert np.isclose(df["k"].iloc[0], 10.0)

    with pytest.raises(UnsupportedFeatureError, match="histval"):
        DynoModel(txt=base + "histval; k(-1) = 10; end;\n", filename="h.mod")


def test_mixed_stochastic_and_deterministic_shocks_stay_stochastic():
    txt = RAMST_HEAD + """
initval; x = 1; k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1)); c = aa*k^alph-delt*k; end;
shocks;
var x; stderr 0.01;
end;
stoch_simul(order=1, irf=10);
"""
    model = DynoModel(txt=txt, filename="stoch.mod")
    assert not model.is_deterministic
    assert model.symbols["exogenous"] == ["x"]
