import numpy as np
import pytest
from dyno import DynoModel


def test_shock_single_argument_syntax():
    txt = """
rho <- 0.8
x[~] <- 0.0
x[t] = rho * x[t-1] + e[t]
e[t] <- N(0.02)
"""
    model = DynoModel(txt=txt)
    assert "e" in model.symbols["exogenous"]
    assert model.context["steady_states"]["e"] == 0.0

    process = model.context["processes"][("e",)]
    assert np.allclose(process.Μ, [0.0])
    # Std is 0.02, so variance is 0.02^2 = 0.0004
    assert np.allclose(process.Σ, [[0.0004]])


def test_shock_two_arguments_syntax():
    txt = """
rho <- 0.8
x[~] <- 0.0
x[t] = rho * x[t-1] + e[t]
e[t] <- N(0.5, 0.02)
"""
    model = DynoModel(txt=txt)
    assert "e" in model.symbols["exogenous"]
    assert model.context["steady_states"]["e"] == 0.5

    process = model.context["processes"][("e",)]
    assert np.allclose(process.Μ, [0.5])
    # Std is 0.02, so variance is 0.02^2 = 0.0004
    assert np.allclose(process.Σ, [[0.0004]])


def test_shock_invalid_argument_count():
    from dyno.errors import LARKParserError

    # 0 arguments fails at parse time (grammar requires at least 1 formula)
    txt_zero = """
x[~] <- 0.0
x[t] = x[t-1] + e[t]
e[t] <- N()
"""
    with pytest.raises(LARKParserError):
        DynoModel(txt=txt_zero)

    # 3 arguments parses but fails evaluator validation
    txt_three = """
x[~] <- 0.0
x[t] = x[t-1] + e[t]
e[t] <- N(0.0, 0.01, 0.02)
"""
    with pytest.raises(TypeError, match="takes 1 or 2 arguments"):
        DynoModel(txt=txt_three)


def test_shock_negative_std_raises_value_error():
    txt = """
x[~] <- 0.0
x[t] = x[t-1] + e[t]
e[t] <- N(-0.01)
"""
    with pytest.raises(ValueError, match="Standard deviation must be non-negative"):
        DynoModel(txt=txt)
