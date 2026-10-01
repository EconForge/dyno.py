import warnings

import pytest

from dyno import DynoModel
from dyno.errors import RedefinitionWarning, SystemStructureError

NON_SQUARE = """
rho <- 0.9
x[~] <- 0.0
e[t] <- N(0.01)
x[t] = rho * x[t-1] + e[t]
x[t] = 0
"""


@pytest.mark.parametrize("method", ["steady", "solve", "simulate"])
def test_non_square_system_is_reported_before_scipy(method):
    model = DynoModel(txt=NON_SQUARE)
    with pytest.raises(SystemStructureError, match="2 equation.*1 endogenous"):
        getattr(model, method)()


def test_constant_redefinition_warns():
    txt = """
rho <- 0.9
rho <- 0.5
x[~] <- 0.0
e[t] <- N(0.01)
x[t] = rho * x[t-1] + e[t]
"""
    with pytest.warns(RedefinitionWarning, match="rho"):
        model = DynoModel(txt=txt)
    assert model.context["constants"]["rho"] == 0.9


def test_solve_without_steady_state_names_the_variables():
    from dyno.errors import UndefinedSymbolError

    model = DynoModel(txt="""
rho <- 0.9
e[t] <- N(0.01)
x[t] = rho * x[t-1]^0.5 + e[t]
""")
    with pytest.raises(UndefinedSymbolError, match="without steady state: x"):
        model.solve()
    assert model.steady().solve() is not None


def test_random_simulation_is_reproducible_with_rng():
    import numpy as np

    model = DynoModel("examples/rbc_stochastic.dyno")
    sol = model.solve()
    s1 = sol.simulate(T=10, mode="random", N=2, rng=123)
    s2 = sol.simulate(T=10, mode="random", N=2, rng=np.random.default_rng(123))
    s3 = sol.simulate(T=10, mode="random", N=2, rng=124)
    assert np.allclose(np.asarray(s1), np.asarray(s2))
    assert not np.allclose(np.asarray(s1), np.asarray(s3))

    sim = model.simulate(T=10, mode="random", N=2, rng=123)
    assert np.allclose(np.asarray(sim), np.asarray(s1))
