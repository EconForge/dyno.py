import textwrap
import numpy as np
import pytest
from dyno import DynoModel
from dyno.errors import SystemStructureError
from dyno.solver import deterministic_solve
from tests.test_deterministic_ramsey import DISASTER_MODEL_TXT


def test_initial_condition_keeps_variable_endogenous():
    """Variables with assignments only at t <= 0 must remain endogenous."""
    txt = textwrap.dedent("""
alpha <- 0.36
delta <- 0.025
k[~] <- 10.0
c[~] <- 2.0

k[t] = (1 - delta) * k[t-1] + c[t]
c[t] = 0.2 * k[t-1]^alpha

k[0] <- 8.0
""").strip()
    model = DynoModel(txt=txt)
    assert "k" in model.symbols["endogenous"]
    assert "c" in model.symbols["endogenous"]
    assert "k" not in model.symbols["exogenous"]
    assert len(model.symbols["endogenous"]) == 2
    assert len(model.symbols["exogenous"]) == 0


def test_assignment_t_ge_1_makes_variable_exogenous():
    """Variables receiving assignments with t >= 1 are classified as exogenous."""
    txt = textwrap.dedent("""
alpha <- 0.36
x[~] <- 1.0
k[~] <- 10.0

k[t] = x[t] * k[t-1]^alpha

x[1] <- 1.2
x[2] <- 1.1
""").strip()
    model = DynoModel(txt=txt)
    assert "k" in model.symbols["endogenous"]
    assert "x" in model.symbols["exogenous"]
    assert "x" not in model.symbols["endogenous"]


def test_quantified_assignment_with_future_dates_is_exogenous():
    """Quantified assignment over a range covering t >= 1 makes variable exogenous."""
    txt = textwrap.dedent("""
alpha <- 0.36
e[~] <- 0.0
k[~] <- 10.0

k[t] = k[t-1]^alpha + e[t]

forall t, 0 <= t < 10 : e[t] <- 0.1 / (t + 1)
""").strip()
    model = DynoModel(txt=txt)
    assert "e" in model.symbols["exogenous"]
    assert "e" not in model.symbols["endogenous"]


def test_mixed_initial_and_future_assignment_is_exogenous():
    """A variable with both t=0 and t >= 1 assignments is classified as exogenous."""
    txt = textwrap.dedent("""
alpha <- 0.36
e[~] <- 0.0
k[~] <- 10.0

k[t] = k[t-1]^alpha + e[t]

e[0] <- 0.05
e[1] <- 0.02
""").strip()
    model = DynoModel(txt=txt)
    assert "e" in model.symbols["exogenous"]
    assert "k" in model.symbols["endogenous"]


def test_rbc_model_with_initial_capital():
    """RBC model from examples/rbc.dyno with k[0] set has k endogenous and square system."""
    model = DynoModel("examples/rbc.dyno")

    assert "k" in model.symbols["endogenous"]
    assert "k" not in model.symbols["exogenous"]

    # Exogenous variables are shocks e and u
    assert set(model.symbols["exogenous"]) == {"e", "u"}

    # System must be square: 6 equations and 6 endogenous variables
    assert len(model.symbols["endogenous"]) == 6
    assert len(model.equations) == 6

    # Pipeline executes without system structure error
    results = model.run(default_pipeline=True)
    assert results.solution is not None


def test_rbc_three_model_variants():
    """Verify classification and behavior across the three distinct RBC models."""
    # 1. Stochastic
    m_stoch = DynoModel("examples/rbc_stochastic.dyno")
    assert not m_stoch.is_deterministic
    assert set(m_stoch.symbols["endogenous"]) == {"y", "c", "h", "b", "k", "a"}
    assert set(m_stoch.symbols["exogenous"]) == {"e", "u"}
    assert len(m_stoch.equations) == 6

    # 2. Stochastic with forced initial shocks and k[0]
    m_forced = DynoModel("examples/rbc_stochastic_forced.dyno")
    assert not m_forced.is_deterministic
    assert "k" in m_forced.symbols["endogenous"]
    assert set(m_forced.symbols["exogenous"]) == {"e", "u"}
    assert len(m_forced.equations) == 6

    # 3. Purely deterministic with shocks specified at all dates
    m_det = DynoModel("examples/rbc_deterministic.dyno")
    assert m_det.is_deterministic
    assert "k" in m_det.symbols["endogenous"]
    assert set(m_det.symbols["exogenous"]) == {"e", "u"}
    assert len(m_det.equations) == 6
    with pytest.raises(
        SystemStructureError, match=r"cannot be solved with `model\.solve\(\)`"
    ):
        m_det.solve()
    traj = m_det.simulate()
    assert len(traj) == 21  # t = 0 to 20
    assert np.isclose(traj["k"].iloc[0], m_det.steady_state["k"] * 1.01, atol=1e-5)


def test_disaster_model_direct_simulate():
    """Disaster model with k[0] <- 0.60 * k[~] simulates directly with model.simulate()."""
    model = DynoModel(txt=DISASTER_MODEL_TXT)

    assert "k" in model.symbols["endogenous"]
    assert "c" in model.symbols["endogenous"]
    assert model.symbols["exogenous"] == ["x"]

    # neq == n_endo
    assert len(model.equations) == len(model.symbols["endogenous"])

    # model.solve() raises SystemStructureError on deterministic model
    with pytest.raises(
        SystemStructureError, match=r"cannot be solved with `model\.solve\(\)`"
    ):
        model.solve()

    # Directly simulate deterministic model
    traj = model.simulate()
    k_ss = model.steady_state["k"]

    # Initial condition at t=0 matches 60% of steady state
    assert np.isclose(traj["k"].iloc[0], 0.60 * k_ss, atol=1e-5)
    # Trajectory transitions smoothly
    assert traj["k"].iloc[1] > traj["k"].iloc[0]
