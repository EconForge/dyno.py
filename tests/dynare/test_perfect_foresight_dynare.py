"""Perfect-foresight .mod files through the Dynare preprocessor backend."""

import numpy as np

from dyno import DynareModel, DynoModel
from dyno.simul import TransitionSimulation
from dyno.solver import deterministic_solve


def test_ramst_mod_runs_with_dynare_model():
    model = DynareModel("examples/modfiles/ramst.mod")
    assert model.is_deterministic
    assert model.symbols["exogenous"] == ["x"]
    assert model.context["values"] == {"x": {1: 1.2}}
    assert np.abs(model.steady().residuals).max() < 1e-8

    commands = [c["command"] for c in model.metadata["dynare_commands"]]
    assert "simul" in commands
    simul = next(
        c for c in model.metadata["dynare_commands"] if c["command"] == "simul"
    )
    assert simul["options"] == {"mode": "deterministic", "T": 200}

    results = model.run()
    sim = results.simulation
    assert isinstance(sim, TransitionSimulation)
    assert sim.attrs["converged"]
    assert sim.T == 200
    df = sim.to_df()
    assert np.isclose(df["x"].iloc[1], 1.2)
    assert results.figure is not None


def test_ramst_mod_dynare_and_dyno_backends_agree():
    # The preprocessor replaces x(+1) by an auxiliary variable, so the two
    # formulations only share the same terminal condition once the path has
    # settled at the steady state: compare over the file's own 200 periods.
    dynare = deterministic_solve(DynareModel("examples/modfiles/ramst.mod"), T=200)
    dyno = deterministic_solve(DynoModel("examples/modfiles/ramst.mod"), T=200)
    a = dynare.to_df()[["c", "k", "x"]].values
    b = dyno.to_df()[["c", "k", "x"]].values
    print("max abs difference:", np.abs(a - b).max())
    np.testing.assert_allclose(a, b, atol=1e-6)
