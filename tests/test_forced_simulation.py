import numpy as np
import pytest
from dyno import DynoModel
from dyno.simul import simulate


def test_rbc_simulation_forces_initial_shocks_and_k0():
    """Verify that stochastic simulation of RBC respects predetermined values for e, u, and k[0]."""
    model = DynoModel("examples/rbc.dyno")
    sol = model.solve()

    df1 = simulate(sol, T=20, rng=123)
    df2 = simulate(sol, T=20, rng=456)

    # Initial state at t=0 has k0 perturbed by 1% and contemporaneous impact of e[0]=0.99
    k_ss = model.steady_state["k"]
    expected_k0_dev = k_ss * 0.01
    assert np.isclose(df1["k"].iloc[0], expected_k0_dev, atol=1e-5)
    assert np.isclose(df1["a"].iloc[0], 0.99, atol=1e-5)

    # Dates 0 through 9 are fully forced by predetermined e and u shocks,
    # so df1 and df2 must be identical up to t=9 despite different random seeds
    for t in range(10):
        np.testing.assert_allclose(df1.iloc[t].values, df2.iloc[t].values)

    # Dates t >= 10 receive random draws, so df1 and df2 must diverge
    assert not np.allclose(df1.iloc[10].values, df2.iloc[10].values)


def test_explicit_shocks_parameter():
    """Verify that explicit shocks dictionary forces shocks at specified dates."""
    model = DynoModel("examples/neo.dyno")
    sol = model.solve(method="qz")

    # Force all exogenous shocks at dates 1 and 2
    forced_shocks = {
        "e_z": {1: 0.1, 2: 0.05},
        "e_y": {1: 0.0, 2: 0.0},
    }
    df1 = simulate(sol, T=10, shocks=forced_shocks, rng=1)
    df2 = simulate(sol, T=10, shocks=forced_shocks, rng=2)

    # Date 1 and 2 transitions are identical due to forced shocks
    np.testing.assert_allclose(df1.iloc[1].values, df2.iloc[1].values)
    np.testing.assert_allclose(df1.iloc[2].values, df2.iloc[2].values)

    # Date 3 receives random draws for both e_z and e_y, so it must differ
    assert not np.allclose(df1.iloc[3].values, df2.iloc[3].values)


def test_explicit_initial_states_parameter():
    """Verify that explicit initial_states overrides the starting state at t=0."""
    model = DynoModel("examples/neo.dyno")
    sol = model.solve(method="qz")

    k_ss = model.steady_state["k"]
    custom_k = k_ss + 2.5

    df = simulate(sol, T=5, initial_states={"k": custom_k})
    assert np.isclose(df["k"].iloc[0], 2.5, atol=1e-5)


def test_run_command_simul_with_forced_shocks():
    """Verify that @run: simul directive in model executes forced simulation."""
    raw_lines = open("examples/rbc.dyno", encoding="utf-8").read().splitlines()
    base_txt = "\n".join(
        line for line in raw_lines if not line.strip().startswith("@run:")
    )
    txt = "@run: solve\n@run: simul: {T: 15}\n\n" + base_txt
    model = DynoModel(txt=txt)
    results = model.run()

    assert results.simulation is not None
    assert len(results.simulation) == 16  # t = 0 to 15
    k_ss = model.steady_state["k"]
    assert np.isclose(results.simulation["k"].iloc[0], k_ss * 0.01, atol=1e-5)
