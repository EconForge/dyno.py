from __future__ import annotations

import pytest
from dyno import DynoModel, RunResults, RunResultsVariants


def test_new_keynesian_forward_guidance_example():
    """Verify examples/new_keynesian_fg.dyno loads, solves, and simulates."""
    model = DynoModel("examples/new_keynesian_fg.dyno")
    assert model.is_deterministic is True
    assert set(model.symbols["endogenous"]) == {"y", "pi", "i"}
    assert set(model.symbols["exogenous"]) == {"r_nat", "eps_i"}

    res = model.run()
    if isinstance(res, RunResultsVariants):
        assert len(res) > 0
        single_res = res[0]
    else:
        assert isinstance(res, RunResults)
        single_res = res

    assert single_res.errors == []
    assert single_res.steady_stats is not None
    assert single_res.steady_stats.get("converged") is True

    # Check simulation results
    df = single_res.simulation
    assert df is not None
    assert len(df) == 36  # t = 0 to 35
    assert "y" in df.columns
    assert "pi" in df.columns
    assert "i" in df.columns

    # During forward guidance (t=1..5), output gap and inflation are positive
    assert (df.loc[1:4, "y"] > 0).all()
    assert (df.loc[1:4, "pi"] > 0).all()


def test_soe_mendoza_example():
    """Verify examples/soe_mendoza.dyno loads, solves, and simulates."""
    model = DynoModel("examples/soe_mendoza.dyno")
    assert model.is_deterministic is True
    assert set(model.symbols["endogenous"]) == {"k", "y", "i", "b", "r", "c"}
    assert set(model.symbols["exogenous"]) == {"p_x"}

    res = model.run()
    assert isinstance(res, RunResults)
    assert res.errors == []
    assert res.steady_stats is not None
    assert res.steady_stats.get("converged") is True

    df = res.simulation
    assert df is not None
    assert len(df) == 36  # t = 0 to 35
    for col in ["k", "y", "i", "b", "r", "c", "p_x"]:
        assert col in df.columns

    # Commodity windfall initially increases consumption and investment
    c_ss = float(model.context["steady_states"]["c"])
    assert df.loc[1, "c"] > c_ss


def test_model_name_and_report_overview_respect_at_name():
    """Verify that @name: metadata is reflected in model.name and in generated reports."""
    model = DynoModel("examples/neo.dyno")
    assert model.name == "Neoclassical"
    res = model.run()
    md = res.to_markdown()
    assert "**Name:** Neoclassical" in md
    txt = res.to_text()
    assert "name: Neoclassical" in txt
