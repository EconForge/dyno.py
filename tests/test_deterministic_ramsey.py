"""
Tests for deterministic simulation of the Ramsey model from 02_deterministic_models.ipynb.

Tests:
1. Reproduction of notebook simulation.
2. The legacy static terminal condition anomaly (jump and reversal).
3. The 'stationary' continuation condition: smooth convergence, no jump, resource constraint exact.
4. The 'steady_state' continuation condition: forward expectation at steady state, resource constraint exact.
5. Equivalence and alias support ('continuation' and 'terminal_condition').
6. Post-disaster recovery dynamics under both continuation conditions.
7. Horizon length effects (mitigation of boundary effects as T increases).
"""

import numpy as np
import pytest
from dyno import DynoModel, deterministic_solve


RAMSEY_MODEL_TXT = """
# Neoclassical Ramsey Model with Perfect Foresight / Deterministic Transition
# Parameters
alph <- 0.50     # Capital share
gam  <- 0.50     # Relative risk aversion
delt <- 0.02     # Depreciation rate
bet  <- 0.051    # Discount rate (quarterly)
aa   <- 0.511    # Productivity scaling factor
T    <- 50       # Simulation time horizon

# Steady State
x[~] <- 1.0
k[~] <- ((delt + bet) / (1.0 * aa * alph))^(1 / (alph - 1))
c[~] <- aa * k[~]^alph - delt * k[~]

# Dynamic Equations
0 = c[t] + k[t] - aa*x[t]*k[t-1]^alph - (1 - delt)*k[t-1]
0 = c[t]^(-gam) - (1 + bet)^(-1) * (aa*alph*x[t+1]*k[t]^(alph-1) + 1 - delt) * c[t+1]^(-gam)

# Anticipated Exogenous Trajectory (Productivity shock path)
x[1] <- 1.10
x[2] <- 1.30
forall t, 3 <= t < T : x[t] <- 1.0 + (1.30 - 1.0) * exp(-(t - 1))
"""

DISASTER_MODEL_TXT = """
# Parameters
alph <- 0.50
gam  <- 0.50
delt <- 0.02
bet  <- 0.051
aa   <- 0.511
T    <- 50

# Steady-state values
x[~] <- 1.0
k[~] <- ((delt + bet) / (1.0 * aa * alph))^(1 / (alph - 1))
c[~] <- aa * k[~]^alph - delt * k[~]

# Dynamic equations
0 = c[t] + k[t] - aa*x[t]*k[t-1]^alph - (1 - delt)*k[t-1]
0 = c[t]^(-gam) - (1 + bet)^(-1) * (aa*alph*x[t+1]*k[t]^(alph-1) + 1 - delt) * c[t+1]^(-gam)

# Constant productivity path
x[1] <- 1.0
forall t, 2 <= t < T : x[t] <- 1.0

# Initial condition override: capital destroyed by 40%
k[0] <- 0.60 * k[~]
"""


def get_ramsey_model():
    return DynoModel(txt=RAMSEY_MODEL_TXT)


def resource_constraint_residual(k_prev, k_curr, c_curr, x_curr, alph=0.50, delt=0.02, aa=0.511):
    return c_curr + k_curr - (aa * x_curr * (k_prev ** alph) + (1.0 - delt) * k_prev)


def test_reproduce_notebook_simulation():
    """Verify that deterministic_solve works on the Ramsey model."""
    model = get_ramsey_model()
    traj = deterministic_solve(model)

    assert len(traj) == 51  # t = 0 to 50
    assert "k" in traj.columns
    assert "c" in traj.columns
    assert "x" in traj.columns

    k_ss = model.steady_state["k"]
    c_ss = model.steady_state["c"]

    # Initial state is at steady state
    assert np.isclose(traj["k"].iloc[0], k_ss, atol=1e-4)
    assert np.isclose(traj["c"].iloc[0], c_ss, atol=1e-4)

    # Productivity shock peaks at t=2
    assert np.isclose(traj["x"].iloc[1], 1.10)
    assert np.isclose(traj["x"].iloc[2], 1.30)

    # Capital peaks around t=3..4
    k_peak_t = traj["k"].idxmax()
    assert k_peak_t in [3, 4]


def test_continuation_stationary():
    """
    Test continuation='stationary' (v_{T+1} = v_T):
    1. Capital decreases monotonically after the shock has dissipated (t >= 4).
    2. Capital smoothly reaches steady state at t=T with NO sudden jump: |k[T] - k[T-1]| < 0.01.
    3. The resource constraint is exactly satisfied at t=T (residual < 1e-6).
    """
    model = get_ramsey_model()
    traj = deterministic_solve(model, continuation="stationary", T=50)

    k = traj["k"].values
    c = traj["c"].values
    x = traj["x"].values
    k_ss = model.steady_state["k"]

    # 1. Monotonic decay after peak: no reversal!
    k_diffs_late = np.diff(k[40:51])
    assert np.all(k_diffs_late < 0), "Capital should monotonically decay towards steady state"

    # 2. Smooth arrival at steady state
    assert np.isclose(k[50], k_ss, atol=1e-3)
    jump_at_T = abs(k[50] - k[49])
    assert jump_at_T < 0.01, f"Expected smooth arrival at T, got step {jump_at_T}"

    # 3. Dynamic resource constraint is satisfied at t=T
    res_T = resource_constraint_residual(k[49], k[50], c[50], x[50])
    assert abs(res_T) < 1e-6, f"Resource constraint violated at T: {res_T}"


def test_continuation_steady_state():
    """
    Test continuation='steady_state' (v_{T+1} = v_bar, Dynare convention):
    1. The resource constraint is exactly satisfied at t=T (residual < 1e-6).
    2. k[T] smoothly accounts for capital accumulated at t=T-1.
    """
    model = get_ramsey_model()
    traj = deterministic_solve(model, continuation="steady_state", T=50)

    k = traj["k"].values
    c = traj["c"].values
    x = traj["x"].values

    # Resource constraint is satisfied at t=T
    res_T = resource_constraint_residual(k[49], k[50], c[50], x[50])
    assert abs(res_T) < 1e-6, f"Resource constraint violated at T: {res_T}"

    # Step at T is small and consistent with the resource constraint
    step_at_T = abs(k[50] - k[49])
    assert step_at_T < 0.01, f"Expected small step at T, got {step_at_T}"


def test_continuation_alias_and_options():
    """Test that aliases 'continuation' and 'terminal_condition' work identically."""
    model = get_ramsey_model()

    traj1 = deterministic_solve(model, continuation="stationary", T=50)
    traj2 = deterministic_solve(model, terminal_condition="stationary", T=50)
    np.testing.assert_allclose(traj1["k"].values, traj2["k"].values)
    np.testing.assert_allclose(traj1["c"].values, traj2["c"].values)

    traj3 = deterministic_solve(model, continuation="steady_state", T=50)
    traj4 = deterministic_solve(model, terminal_condition="steady_state", T=50)
    np.testing.assert_allclose(traj3["k"].values, traj4["k"].values)
    np.testing.assert_allclose(traj3["c"].values, traj4["c"].values)


def test_legacy_static_mode():
    """Test that continuation='static' reproduces the legacy behavior with the jump."""
    model = get_ramsey_model()
    traj = deterministic_solve(model, continuation="static", T=50)

    k = traj["k"].values
    c = traj["c"].values
    x = traj["x"].values
    k_ss = model.steady_state["k"]

    # Anomaly: sharp jump down to steady state
    jump = abs(k[50] - k[49])
    assert jump > 0.05
    assert np.isclose(k[50], k_ss)

    # Anomaly: resource constraint violated
    res_T = resource_constraint_residual(k[49], k[50], c[50], x[50])
    assert abs(res_T) > 0.05


def test_disaster_experiment_both_continuations():
    """Test the capital disaster experiment under both continuation conditions."""
    model = DynoModel(txt=DISASTER_MODEL_TXT)
    k_ss = model.steady_state["k"]

    for mode in ["stationary", "steady_state"]:
        v0 = np.concatenate(model.__steady_state_vectors__)[None, :].repeat(51, axis=0)
        k_idx = model.symbols["variables"].index("k")
        v0[:, k_idx] = k_ss
        v0[0, k_idx] = 0.60 * k_ss

        traj = deterministic_solve(model, x0=v0, continuation=mode, T=50)
        k = traj["k"].values
        c = traj["c"].values
        x = traj["x"].values

        # Resource constraint is satisfied at t=T
        res_T = resource_constraint_residual(k[49], k[50], c[50], x[50])
        assert abs(res_T) < 1e-6, f"Resource constraint violated at T for {mode}: {res_T}"

        # Capital does not have a 0.5 jump at t=T
        step = abs(k[50] - k[49])
        assert step < 0.05, f"Unexpected large step at T for {mode}: {step}"


def test_longer_horizon_convergence():
    """
    Test that with a longer horizon (T=150), both continuation conditions produce
    nearly identical results because the model naturally reaches steady state.
    """
    model = get_ramsey_model()
    traj_stat = deterministic_solve(model, continuation="stationary", T=150)
    traj_ss = deterministic_solve(model, continuation="steady_state", T=150)

    # For t=0..50, the paths are virtually indistinguishable (< 0.001 max diff)
    max_k_diff = np.max(np.abs(traj_stat["k"].iloc[:51] - traj_ss["k"].iloc[:51]))
    assert max_k_diff < 0.001


def test_continuation_constant_growth_endogenous():
    """
    Test continuation='constant_growth' with endogenous growth:
    1. Default geometric growth rate: v_{T+1} = v_T^2 / v_{T-1}.
    2. Linear growth rate: v_{T+1} = 2*v_T - v_{T-1}.
    Both satisfy the dynamic resource constraint at t=T and smoothly approach steady state.
    """
    model = get_ramsey_model()

    # Geometric growth (default)
    traj_geom = deterministic_solve(
        model, continuation="constant_growth", growth_type="geometric", T=50
    )
    k_g = traj_geom["k"].values
    c_g = traj_geom["c"].values
    x_g = traj_geom["x"].values

    res_T_g = resource_constraint_residual(k_g[49], k_g[50], c_g[50], x_g[50])
    assert abs(res_T_g) < 1e-6, f"Resource constraint violated at T (geometric): {res_T_g}"
    assert abs(k_g[50] - k_g[49]) < 0.01, "Expected smooth arrival at T"

    # Linear growth
    traj_lin = deterministic_solve(
        model, continuation="constant_growth", growth_type="linear", T=50
    )
    k_l = traj_lin["k"].values
    c_l = traj_lin["c"].values
    x_l = traj_lin["x"].values

    res_T_l = resource_constraint_residual(k_l[49], k_l[50], c_l[50], x_l[50])
    assert abs(res_T_l) < 1e-6, f"Resource constraint violated at T (linear): {res_T_l}"
    assert abs(k_l[50] - k_l[49]) < 0.01, "Expected smooth arrival at T"


def test_continuation_constant_growth_explicit_rate():
    """
    Test continuation='constant_growth' with explicit growth rates:
    1. growth_rate=0.0 must be identical to continuation='stationary'.
    2. growth_rate as float or dict solves with dynamic consistency at t=T.
    """
    model = get_ramsey_model()

    traj_stat = deterministic_solve(model, continuation="stationary", T=50)
    traj_g0 = deterministic_solve(
        model, continuation="constant_growth", growth_rate=0.0, T=50
    )

    np.testing.assert_allclose(traj_stat["k"].values, traj_g0["k"].values, atol=1e-10)
    np.testing.assert_allclose(traj_stat["c"].values, traj_g0["c"].values, atol=1e-10)

    # Small positive growth rate
    traj_g = deterministic_solve(
        model, continuation="constant_growth", growth_rate=0.001, T=50
    )
    k = traj_g["k"].values
    c = traj_g["c"].values
    x = traj_g["x"].values
    res_T = resource_constraint_residual(k[49], k[50], c[50], x[50])
    assert abs(res_T) < 1e-6, f"Resource constraint violated with growth_rate=0.001: {res_T}"

    # Dict of growth rates
    traj_dict = deterministic_solve(
        model, continuation="constant_growth", growth_rate={"k": 0.0, "c": 0.0}, T=50
    )
    np.testing.assert_allclose(traj_stat["k"].values, traj_dict["k"].values, atol=1e-10)


def test_constant_growth_jacobian_accuracy():
    """
    Verify that the analytical Jacobian computed for constant_growth matches
    numerical finite differences of deterministic_residuals.
    """
    model = get_ramsey_model()
    v0 = model.deterministic_guess(T=10).ravel()

    for g_type in ["geometric", "linear"]:
        res_fn = lambda u: model.deterministic_residuals(
            u, continuation="constant_growth", growth_type=g_type
        )
        _, J_analytic = model.deterministic_residuals_with_jacobian(
            v0, sparsify=False, continuation="constant_growth", growth_type=g_type
        )

        eps = 1e-7
        J_numeric = np.zeros_like(J_analytic)
        r0 = res_fn(v0)
        for i in range(len(v0)):
            v_plus = v0.copy()
            v_plus[i] += eps
            r_plus = res_fn(v_plus)
            J_numeric[:, i] = (r_plus - r0) / eps

        max_err = np.max(np.abs(J_analytic - J_numeric))
        assert max_err < 1e-5, f"Jacobian mismatch for growth_type={g_type}: max diff = {max_err}"

