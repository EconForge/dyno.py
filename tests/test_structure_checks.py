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
