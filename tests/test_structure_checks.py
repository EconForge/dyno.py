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


# --- Typo diagnostics (issue #24) -------------------------------------------

TYPO = """
rho <- 0.9
x[~] <- 0.0
e[t] <- N(0.01)
x[t] = rho * xx[t-1] + e[t]
"""


def test_typo_in_variable_name_warns_at_import():
    from dyno.errors import UndefinedSymbolWarning

    with pytest.warns(UndefinedSymbolWarning, match="'xx' appears only once") as rec:
        DynoModel(txt=TYPO)
    msg = str(rec[0].message)
    assert "(line 5)" in msg
    assert "Did you mean 'x'?" in msg


def test_typo_in_variable_name_explained_in_structure_error():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DynoModel(txt=TYPO)
    with pytest.raises(SystemStructureError) as exc_info:
        model.solve()
    msg = str(exc_info.value)
    assert msg.splitlines()[0].startswith(
        "Model has 1 equation but 2 endogenous variables."
    )
    assert (
        "  'xx' appears only once (line 5) and has no steady state. "
        "Did you mean 'x'?"
    ) in msg.splitlines()
    # 'x' is declared: it must not be reported as suspicious.
    assert "'x' appears" not in msg


def test_structure_error_suggests_parameter_names():
    txt = """
alpha <- 0.3
k[~] <- 1.0
k[t] = alpa[t] * k[t-1]
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DynoModel(txt=txt)
    with pytest.raises(SystemStructureError, match="Did you mean 'alpha'"):
        model.solve()


def test_structure_error_lists_single_equation_variables():
    # Both variables have a steady state: fall back to listing those that
    # appear in a single equation.
    txt = """
k[t] = 0.5 * k[t-1] + c[t]
k[~] <- 1.0
c[~] <- 0.5
"""
    model = DynoModel(txt=txt)
    with pytest.raises(SystemStructureError) as exc_info:
        model.solve()
    msg = str(exc_info.value)
    assert "'k' appears in a single equation (line 2)." in msg
    assert "'c' appears only once (line 2)." in msg
    assert "steady state" not in msg


def test_no_typo_warning_for_variable_used_several_times():
    # 'x' has no steady state but is referenced twice: not a typo.
    txt = """
rho <- 0.9
e[t] <- N(0.01)
x[t] = rho * x[t-1]^0.5 + e[t]
"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DynoModel(txt=txt)


_DYNO_EXAMPLES = sorted(
    str(p)
    for pattern in ("*.dyno", "*.yaml", "variants/*.dyno")
    for p in __import__("pathlib").Path("examples").glob(pattern)
)


@pytest.mark.parametrize("path", _DYNO_EXAMPLES)
def test_shipped_examples_have_no_typo_warning(path):
    from dyno.errors import UndefinedSymbolWarning

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        DynoModel(path)
    assert not [
        w
        for w in rec
        if issubclass(w.category, UndefinedSymbolWarning)
        and "Possible typo" in str(w.message)
    ]


def test_dynare_model_structure_error_still_works():
    pytest.importorskip("dynare_preprocessor")
    from dyno.dynare import DynareModel

    model = DynareModel("examples/modfiles/RBC.mod")
    model._check_square()  # square: no error
    model.symbols["endogenous"] = model.symbols["endogenous"] + ["zz"]
    with pytest.raises(SystemStructureError, match="'zz' has no steady state"):
        model._check_square()
