from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from dyno import DynoModel


def _load_cases() -> list[dict[str, Any]]:
    fixture = Path(__file__).parent / "fixtures" / "recalibrate_cases.yaml"
    data = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    return list(data["cases"])


CASES = _load_cases()
CASE_IDS = [case["name"] for case in CASES]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_recalibrate_parameter_dependencies(case: dict[str, Any]) -> None:
    model = DynoModel(filename=f"<{case['name']}>.dyno", txt=case["model"])
    original_constants = dict(model.context["constants"])

    recalibrated = model.recalibrate(**case["recalibrate"])

    assert recalibrated is not model
    assert model.context["constants"] == original_constants

    for key, expected in case.get("expected_before", {}).items():
        assert model.context["constants"][key] == pytest.approx(expected)

    for key, expected in case["expected_after"].items():
        assert recalibrated.context["constants"][key] == pytest.approx(expected)

    assert recalibrated.symbols == model.symbols


def test_recalibrate_chaining_preserves_previous_overrides() -> None:
    txt = """
r <- 0.04

p[~] <- 1/r
ρ_l <- 1.0

K <- 1.2
gamma <- 1.8

σ <- 2.0
ε <- 0.004

β <- (1-ε)/(1+r)
δ <- 1/(1+r) - β

η <- σ/gamma
φ <- (1-β*(1+r))*K^(η)

y_bar <- 0.25

y[~] <- y_bar
w[~] <- K*(y[~]^gamma)
l[~] <- y[~] - w[~]*r
c[~] <- w[~]*r + l[~]

l[t] = l[~] + ρ_l*(l[t-1]-l[~]) + e_l[t]
y[t] = l[t] + w[t-1]*r
c[t] = y[t] - (w[t]-w[t-1])
1 = β*(c[t+1]/c[t])^(-σ)*(1+r) + φ*(w[t])^(-η)/(c[t])^(-σ)

e_l[t] <- N(0, 0.1^2)
"""

    model = DynoModel(txt=txt)

    K = 2.16
    gamma = 1.83
    ybar = 0.2

    one_shot = model.recalibrate(K=K, gamma=gamma, y=ybar)
    chained = model.recalibrate(K=K, gamma=gamma).recalibrate(y=ybar)

    assert chained.context["constants"]["K"] == pytest.approx(K)
    assert chained.context["constants"]["gamma"] == pytest.approx(gamma)
    assert chained.context["constants"]["φ"] == pytest.approx(
        one_shot.context["constants"]["φ"]
    )
    assert chained.context["steady_states"]["w"] == pytest.approx(
        one_shot.context["steady_states"]["w"]
    )


def test_steady_returns_new_model_with_solved_steady_state() -> None:
    txt = """
alpha <- 0.9
beta <- 2.0

x[t] = alpha*x[t-1] + beta
"""

    model = DynoModel(txt=txt)
    solved = model.steady()

    assert solved is not model
    assert solved.context["steady_states"]["x"] == pytest.approx(20.0)
    assert np.max(np.abs(solved.residuals)) < 1e-8


def test_steady_does_not_mutate_original_model() -> None:
    txt = """
alpha <- 0.8
beta <- 1.0

x[t] = alpha*x[t-1] + beta
"""

    model = DynoModel(txt=txt)
    before = dict(model.context["steady_states"])

    _ = model.steady()

    assert model.context["steady_states"] == before


def test_check_returns_model_when_residuals_are_small() -> None:
    txt = """
alpha <- 0.9
beta <- 2.0

x[~] <- 20.0
x[t] = alpha*x[t-1] + beta
"""
    model = DynoModel(txt=txt)
    result = model.check()
    assert result is model


def test_check_raises_when_residuals_are_large() -> None:
    from dyno.errors import SteadyStateError

    txt = """
alpha <- 0.9
beta <- 2.0

x[~] <- 1.0
x[t] = alpha*x[t-1] + beta
"""
    model = DynoModel(txt=txt)
    with pytest.raises(SteadyStateError):
        model.check()


def test_dyno_run_is_experimental_and_executes_commands() -> None:
    from dyno import DynoRunResults

    txt = """
run:
  - steady
  - check
  - solve
  - {simul: {T: 4}}
model: |
  rho <- 0.8

  x[t] = rho*x[t-1] + e[t]
  e[t] <- N(0, 0.1^2)
"""

    model = DynoModel(filename="<run>.yaml", yaml=txt)

    with pytest.warns(UserWarning, match="experimental"):
        result = model.run()

    assert isinstance(result, DynoRunResults)
    assert result.model is not model
    assert abs(result.model.context["steady_states"]["x"]) < 1e-8
    assert result.solution is not None
    assert result.simulation is not None


def test_dyno_run_accepts_string_command_list() -> None:
    txt = """
run:
  - steady
  - check
model: |
  alpha <- 0.9
  beta <- 2.0

  x[t] = alpha*x[t-1] + beta
"""

    model = DynoModel(filename="<run2>.yaml", yaml=txt)

    with pytest.warns(UserWarning, match="experimental"):
        result = model.run()

    assert result.model.context["steady_states"]["x"] == pytest.approx(20.0)


def test_dyno_run_accepts_compact_null_options() -> None:
    txt = """
run:
  - steady: null
  - check: null
model: |
  alpha <- 0.9
  beta <- 2.0

  x[t] = alpha*x[t-1] + beta
"""

    model = DynoModel(filename="<run3>.yaml", yaml=txt)

    with pytest.warns(UserWarning, match="experimental"):
        result = model.run()

    assert result.model.context["steady_states"]["x"] == pytest.approx(20.0)


def test_dyno_run_accumulates_repeated_metadata_assignments() -> None:
    from dyno import DynoRunResults

    txt = """
rho <- 0.8

x[t] = rho*x[t-1] + e[t]
e[t] <- N(0, 0.1^2)

@run: steady
@run: check
@run: solve
@run: simul: {T: 4}
"""

    model = DynoModel(txt=txt)
    assert isinstance(model.metadata["run"], list)
    assert len(model.metadata["run"]) == 4

    with pytest.warns(UserWarning, match="experimental"):
        result = model.run()

    assert isinstance(result, DynoRunResults)
    assert result.solution is not None
    assert result.simulation is not None
