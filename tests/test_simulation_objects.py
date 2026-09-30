import numpy as np
import pandas as pd
import pytest

from dyno import (
    DynoModel,
    IRFSimulation,
    RandomSimulation,
    SimulationResult,
    TransitionSimulation,
)


def test_stochastic_irf_simulation_object_and_unit_conversion():
    model = DynoModel("examples/neo.dyno")
    sol = model.solve()

    # Default mode="auto" on stochastic solution produces IRFSimulation
    sim = sol.simulate(T=20)
    assert isinstance(sim, IRFSimulation)
    assert isinstance(sim, SimulationResult)
    assert isinstance(sim, dict)

    n_shocks = len(model.symbols["exogenous"])
    n_endo = len(model.symbols["endogenous"])
    assert sim.N == n_shocks
    assert sim.T == 20
    assert sim.V == n_endo
    assert sim.data.shape == (n_shocks, 21, n_endo)

    # Shock indexing returns (T+1, V) DataFrame
    df_ez = sim["e_z"]
    assert isinstance(df_ez, pd.DataFrame)
    assert df_ez.shape == (21, n_endo)

    # Variable indexing returns (T+1, N) DataFrame across shocks
    df_k = sim["k"]
    assert isinstance(df_k, pd.DataFrame)
    assert df_k.shape == (21, n_shocks)

    # Unit conversions via to_df(units=...)
    df_dev = sim.to_df(units="deviation")
    df_lvl = sim.to_df(units="level")
    df_pct = sim.to_df(units="percent")
    df_log = sim.to_df(units="log-deviation")

    k_ss = model.steady_state["k"]
    np.testing.assert_allclose(
        df_lvl["k"].values, df_dev["k"].values + k_ss, rtol=1e-10
    )
    np.testing.assert_allclose(
        df_pct["k"].values, (df_dev["k"].values / k_ss) * 100.0, rtol=1e-10
    )
    np.testing.assert_allclose(df_pct["k"].values, df_log["k"].values, rtol=1e-12)


def test_random_simulation_spaghetti_draws_and_chaining():
    model = DynoModel("examples/neo.dyno")
    np.random.seed(42)

    sim = model.solve().simulate(mode="random", N=12, T=25)
    assert isinstance(sim, RandomSimulation)
    assert isinstance(sim, SimulationResult)
    assert sim.N == 12
    assert sim.T == 25
    assert sim.V == len(model.symbols["endogenous"])
    assert sim.data.shape == (12, 26, sim.V)

    # Draw indexing returns (T+1, V) DataFrame
    draw0 = sim[0]
    assert isinstance(draw0, pd.DataFrame)
    assert draw0.shape == (26, sim.V)

    # Variable indexing returns (T+1, N) DataFrame of all 12 trajectories
    k_paths = sim["k"]
    assert isinstance(k_paths, pd.DataFrame)
    assert k_paths.shape == (26, 12)

    # MultiIndex DataFrame export in percent units
    df_multi = sim.to_df(units="percent")
    assert isinstance(df_multi.index, pd.MultiIndex)
    assert len(df_multi) == 12 * 26

    # Fluent spaghetti plot generation (Plotly & Altair)
    fig_plotly = sim.plot(units="percent")
    assert fig_plotly is not None
    # Verify spaghetti traces have showlegend=False and reduced opacity
    assert all(trace.showlegend is False for trace in fig_plotly.data)

    fig_altair = sim.plot(engine="altair", variables=["k", "c"], T=15)
    assert fig_altair is not None


def test_deterministic_transition_simulation_and_chaining():
    txt = """
    alpha <- 0.5
    beta <- 0.96
    delta <- 0.02
    aa <- (1 / beta - (1 - delta)) / alpha

    k[~] <- 1.0
    c[~] <- aa * k[~]^alpha - delta * k[~]
    x[~] <- 1.0

    x[1] <- 1.05
    k[t] = aa * x[t] * k[t-1]^alpha + (1 - delta) * k[t-1] - c[t]
    c[t]^(-1) = beta * c[t+1]^(-1) * (alpha * aa * x[t+1] * k[t]^(alpha - 1) + 1 - delta)
    """
    model = DynoModel(txt=txt)
    assert model.is_deterministic

    sim = model.simulate(T=30)
    assert isinstance(sim, TransitionSimulation)
    assert isinstance(sim, SimulationResult)
    assert sim.N == 1
    assert sim.T == 30
    assert sim.data.shape == (1, 31, len(model.symbols["variables"]))
    assert sim.attrs.get("converged") is True

    # Unit conversion on deterministic transition
    df_lvl = sim.to_df(units="level")
    df_dev = sim.to_df(units="deviation")
    df_pct = sim.to_df(units="percent")

    k_ss = model.steady_state["k"]
    np.testing.assert_allclose(df_dev["k"].values, df_lvl["k"].values - k_ss)
    np.testing.assert_allclose(
        df_pct["k"].values, (df_lvl["k"].values - k_ss) / k_ss * 100.0
    )

    # Fluent plotting
    fig = sim.plot(units="percent", variables=["k", "c"])
    assert fig is not None


def test_pipeline_directives_simulate_plot_and_analyze():
    raw_lines = open("examples/neo.dyno").read().splitlines()
    base_txt = (
        "\n".join(line for line in raw_lines if not line.strip().startswith("@run:"))
        + "\n"
    )

    # 1. @run: simulate only fills simulation slot (no moments, no figure)
    m_sim_only = DynoModel(txt="@run: simulate: {T: 15}\n" + base_txt)
    res_sim = m_sim_only.run()
    assert isinstance(res_sim.simulation, IRFSimulation)
    assert res_sim.simulation.T == 15
    assert res_sim.moments is None
    assert res_sim.figure is None

    # 2. @run: simulate (random N=8) + @run: plot fills simulation and figure
    m_spaghetti = DynoModel(
        txt="@run: simulate: {mode: random, N: 8, T: 20}\n@run: plot\n" + base_txt
    )
    res_spag = m_spaghetti.run()
    assert isinstance(res_spag.simulation, RandomSimulation)
    assert res_spag.simulation.N == 8
    assert res_spag.simulation.T == 20
    assert res_spag.figure is not None

    # 3. @run: analyze fills solution, moments, simulation, and figure
    m_analyze = DynoModel(txt="@run: analyze: {T: 18, units: percent}\n" + base_txt)
    res_ana = m_analyze.run()
    assert res_ana.solution is not None
    assert res_ana.moments is not None
    assert isinstance(res_ana.simulation, IRFSimulation)
    assert res_ana.simulation.T == 18
    assert res_ana.simulation.units == "percent"
    assert res_ana.figure is not None
