import math

from dyno import DynoModel


def test_unassigned_constant_in_equation_is_explicit_nan_in_context():
    txt = """
x[~] := 0
x[t] = a + x[t-1]
"""

    model = DynoModel(txt=txt)

    assert "a" in model.context["constants"]
    assert math.isnan(model.context["constants"]["a"])


def test_variable_without_steady_state_gets_explicit_nan_steady_state():
    txt = """
a := 1
x[t] = y[t] + a
"""

    model = DynoModel(txt=txt)

    assert "x" in model.context["steady_states"]
    assert "y" in model.context["steady_states"]
    assert math.isnan(model.context["steady_states"]["x"])
    assert math.isnan(model.context["steady_states"]["y"])


def test_constant_defined_after_equation_is_not_nan():
    txt = """
x[~] := 0
x[t] = a + x[t-1]
a := 0.3
"""

    model = DynoModel(txt=txt)

    assert "a" in model.context["constants"]
    assert model.context["constants"]["a"] == 0.3
    assert not math.isnan(model.context["constants"]["a"])


def test_steady_state_defined_after_equation_is_not_nan():
    txt = """
a := 1
x[t] = y[t] + a
y[~] := 2
"""

    model = DynoModel(txt=txt)

    assert "y" in model.context["steady_states"]
    assert model.context["steady_states"]["y"] == 2
    assert not math.isnan(model.context["steady_states"]["y"])
