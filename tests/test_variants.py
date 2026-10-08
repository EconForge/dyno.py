from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import pytest

from dyno import (
    DynoModel,
    IRFSimulation,
    ModelVariants,
    PerturbationSolution,
    RandomSimulation,
    RunResultsVariants,
    SimulationVariants,
    SolutionVariants,
    TransitionSimulation,
    VariantCollection,
)


def test_variants_creation_and_indexing():
    txt = """
    a <- 2
    b <- a * 0.1

    x[~] <- 0
    e[t] <- N(0, 1)

    x[t] = b * x[t-1] + e[t]
    """
    model = DynoModel(txt=txt)
    var = model.variants(a=[2, 3, 4])

    assert isinstance(var, VariantCollection)
    assert isinstance(var, ModelVariants)
    assert len(var) == 3
    assert var.labels == ["a=2", "a=3", "a=4"]
    assert var.parameters == ["a"]

    # Check individual calibrated models
    assert var[0].context["constants"]["a"] == pytest.approx(2)
    assert var[0].context["constants"]["b"] == pytest.approx(0.2)
    assert var[1].context["constants"]["a"] == pytest.approx(3)
    assert var[1].context["constants"]["b"] == pytest.approx(0.3)
    assert var[2].context["constants"]["a"] == pytest.approx(4)
    assert var[2].context["constants"]["b"] == pytest.approx(0.4)

    # Original model untouched
    assert model.context["constants"]["a"] == pytest.approx(2)

    # Indexing by label and slice
    assert var["a=3"] is var[1]
    sliced = var[1:]
    assert isinstance(sliced, ModelVariants)
    assert len(sliced) == 2
    assert sliced.labels == ["a=3", "a=4"]

    # Iteration and repr
    assert list(var) == var.items
    assert "ModelVariants[DynoModel]" in repr(var)
    assert "ModelVariants" in var._repr_html_()


def test_variants_multi_parameter_and_named_forms():
    txt = """
    a <- 0.5
    b <- 1.0
    x[~] <- 0
    e[t] <- N(0, 1)
    x[t] = a * x[t-1] + b * e[t]
    """
    model = DynoModel(txt=txt)

    # 1. Cartesian product (default)
    cart = model.variants(a=[0.4, 0.8], b=[1.0, 2.0])
    assert len(cart) == 4
    assert cart.labels == [
        "a=0.4, b=1",
        "a=0.4, b=2",
        "a=0.8, b=1",
        "a=0.8, b=2",
    ]

    # 2. Pairwise zip
    zipped = model.variants(a=[0.4, 0.8], b=[1.0, 2.0], _zip=True)
    assert len(zipped) == 2
    assert zipped.labels == ["a=0.4, b=1", "a=0.8, b=2"]

    # 3. List of calibration dicts
    from_list = model.variants([{"a": 0.3}, {"a": 0.7}])
    assert len(from_list) == 2
    assert from_list.labels == ["a=0.3", "a=0.7"]

    # 4. Named mapping of variants
    named = model.variants(
        {"low_persistence": {"a": 0.2}, "high_persistence": {"a": 0.9}}
    )
    assert len(named) == 2
    assert named.labels == ["low_persistence", "high_persistence"]
    assert named["high_persistence"].context["constants"]["a"] == pytest.approx(0.9)


def test_variants_solve_simulate_and_plot_chain():
    model = DynoModel("examples/RBC.dyno")
    variants = model.variants(rho=[0.5, 0.8, 0.95])
    assert isinstance(variants, ModelVariants)

    sols = variants.solve()
    assert isinstance(sols, VariantCollection)
    assert isinstance(sols, SolutionVariants)
    assert len(sols) == 3
    assert all(isinstance(s, PerturbationSolution) for s in sols)

    sims = sols.simulate(T=20)
    assert isinstance(sims, VariantCollection)
    assert isinstance(sims, SimulationVariants)
    assert len(sims) == 3
    assert all(isinstance(sim, IRFSimulation) for sim in sims)

    # Verify trajectories actually differ across variants
    df_dict = sims.to_dict(units="deviation")
    assert set(df_dict.keys()) == {"rho=0.5", "rho=0.8", "rho=0.95"}
    assert not np.allclose(
        df_dict["rho=0.5"]["y"].values, df_dict["rho=0.95"]["y"].values
    )

    # Combined DataFrame and 4D array
    combined_df = sims.to_df(units="percent")
    assert isinstance(combined_df, pd.DataFrame)
    assert "variant" in combined_df.index.names
    assert set(combined_df.index.get_level_values("variant")) == {
        "rho=0.5",
        "rho=0.8",
        "rho=0.95",
    }

    arr_4d = sims.in_units("deviation")
    assert arr_4d.shape == (
        3,
        len(model.symbols["exogenous"]),
        21,
        len(model.symbols["endogenous"]),
    )

    # Unified Altair chart (default engine)
    ch = sims.plot(variables=["y", "c", "k", "n"], shocks="epsilon", units="percent")
    assert isinstance(ch, alt.Chart)
    assert ch.to_dict()["encoding"]["color"]["title"] == "rho"
    assert {"rho=0.5", "rho=0.8", "rho=0.95"} == set(ch.data["variant"])

    with pytest.raises(ValueError, match="Plotly support has been removed"):
        sims.plot(engine="plotly")

    # Unified Plotext chart
    txt_plot = sims.plot(
        variables=["y", "c"], shocks="epsilon", engine="plotext", color=False
    )
    assert isinstance(txt_plot, str)
    assert "rho=0.5" in txt_plot
    assert "rho=0.95" in txt_plot


def test_variants_steady_check_and_modfile():
    model = DynoModel("examples/modfiles/rbc_simple.mod")
    var = model.variants(alpha=[0.25, 0.33, 0.40]).steady().check()
    assert isinstance(var, VariantCollection)
    assert len(var) == 3
    for m in var:
        assert np.max(np.abs(m.residuals)) < 1e-8

    # Solve and plot directly from solution collection
    sols = var.solve()
    fig = sols.plot(variables=["y", "k"], T=15)
    assert fig is not None
    # 3 variants * 2 variables (1 shock in rbc.mod)
    assert len(fig.data.groupby(["variant", "variable", "shock"])) == 3 * 2


def test_variants_deterministic_transition_and_spaghetti():
    # 1. Deterministic transition variants
    det_txt = """
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
    det_model = DynoModel(txt=det_txt)
    det_sims = det_model.variants(alpha=[0.3, 0.5, 0.7]).simulate(T=20)
    assert all(isinstance(s, TransitionSimulation) for s in det_sims)

    fig_det = det_sims.plot(variables=["k", "c"], units="percent")
    assert fig_det is not None
    assert len(fig_det.data.groupby(["variant", "variable"])) == 3 * 2

    # 2. Stochastic spaghetti variants
    stoch_model = DynoModel("examples/modfiles/rbc_simple.mod")
    np.random.seed(123)
    spag_sims = (
        stoch_model.variants(rho=[0.6, 0.95]).solve().simulate(mode="random", N=4, T=15)
    )
    assert all(isinstance(s, RandomSimulation) for s in spag_sims)

    fig_spag = spag_sims.plot(variables=["y", "c"])
    assert fig_spag is not None
    # Colored by variant (one legend entry each), one line per draw
    enc = fig_spag.to_dict()["encoding"]
    assert enc["color"]["field"] == "variant"
    assert enc["detail"]["field"] == "_trace_group"
    assert fig_spag.data["variant"].nunique() == 2


def test_variants_pipeline_run_results_variants():
    txt = """
    @variants: {a: [0.2, 0.5, 0.8]}
    @run: solve
    @run: simulate: {T: 15}
    @run: plot: {engine: altair}

    a <- 0.5
    x[~] <- 0.0
    e[t] <- N(0.0, 1.0)
    x[t] = a * x[t-1] + e[t]
    """
    model = DynoModel(txt=txt)
    res = model.run()

    assert isinstance(res, VariantCollection)
    assert isinstance(res, RunResultsVariants)
    assert len(res) == 3
    assert res.labels == ["a=0.2", "a=0.5", "a=0.8"]

    # Functor forwarding on RunResultsVariants
    assert isinstance(res.model, ModelVariants)
    assert isinstance(res.solution, SolutionVariants)
    assert isinstance(res.simulation, SimulationVariants)
    assert list(res.bk_check) == [True, True, True]
    assert res.errors == []

    # Unified figure across all 3 variants
    fig = res.figure
    assert fig is not None
    assert {"a=0.2", "a=0.5", "a=0.8"} == set(fig.data["variant"])

    # Rich representations and text output
    txt_out = res.to_text(graphs=True, color=False)
    assert "Model\n-----" in txt_out
    assert "variants (3): a=0.2, a=0.5, a=0.8" in txt_out
    assert "Checks\n------" in txt_out
    assert "[a=0.2]" in txt_out
    assert "Blanchard-Kahn conditions: met" in txt_out
    assert "Outputs\n-------" in txt_out
    assert "Solution: computed (3/3 variants)" in txt_out
    assert "Simulation Plots\n----------------" in txt_out
    assert "Diagnostics\n-----------" in txt_out
    assert "Timing\n------" in txt_out

    html_out = res._repr_html_()
    assert html_out is not None
    assert "<h3>Model:" in html_out
    assert "<h3>Check</h3>" in html_out
    assert "Generalized Eigenvalues" in html_out
    assert "<h3>Decision Rule</h3>" in html_out
    assert "<h3>Moments</h3>" in html_out
    assert "<h3>Simulation</h3>" in html_out
    assert "<svg" in html_out

    md_out = res._repr_markdown_()
    assert md_out is not None
    assert ":::{note} Model Overview" in md_out
    assert ":::{dropdown} Calibration" in md_out
    assert ":::{dropdown} Equations" in md_out
    assert "## Check" in md_out
    assert ":::{tip} Blanchard-Kahn conditions are met" in md_out
    assert "## Solution" in md_out
    assert ":::::{dropdown} Recursive Decision Rule" in md_out
    assert "## Simulation" in md_out
    assert "## Plot" in md_out
    assert md_out.index("## Simulation") < md_out.index("## Plot")
    assert ":::::{dropdown} IRFS" in md_out
    assert ":::{tab-item} a=0.2" in md_out
    assert "data:image/svg+xml" in md_out
    bundle = res._repr_mimebundle_()
    assert "text/markdown" in bundle
    assert "text/html" in bundle


def test_pipeline_plot_keyword_effect():
    base_eqs = """
    a <- 0.5
    x[~] <- 0.0
    y[~] <- 0.0
    e[t] <- N(0.0, 1.0)
    x[t] = a * x[t-1] + e[t]
    y[t] = 0.5 * x[t]
    """

    # 1. Univariant without @run: plot -> no plot in HTML, Markdown, or Text
    res_no_plot = DynoModel(
        txt="@run: solve\n@run: simulate: {T: 10}\n" + base_eqs
    ).run()
    assert res_no_plot.figure is None
    assert "<svg" not in (res_no_plot._repr_html_() or "")
    assert "data:image/svg+xml" not in (res_no_plot._repr_markdown_() or "")
    assert "Simulation Plots" not in str(res_no_plot)

    # 2. Univariant with @run: plot -> plot rendered in HTML, Markdown, and Text
    res_with_plot = DynoModel(
        txt="@run: solve\n@run: simulate: {T: 10}\n@run: plot: {variables: [x]}\n"
        + base_eqs
    ).run()
    assert res_with_plot.figure is not None
    assert "<svg" in (res_with_plot._repr_html_() or "")
    assert "data:image/svg+xml" in (res_with_plot._repr_markdown_() or "")
    assert "Simulation Plots" in str(res_with_plot)

    # 3. Variants without @run: plot -> no plot in HTML, Markdown, or Text
    var_no_plot = DynoModel(
        txt="@variants: {a: [0.3, 0.7]}\n@run: solve\n@run: simulate: {T: 10}\n"
        + base_eqs
    ).run()
    assert var_no_plot.figure is None
    assert "<svg" not in (var_no_plot._repr_html_() or "")
    assert "data:image/svg+xml" not in (var_no_plot._repr_markdown_() or "")
    assert "Simulation Plots" not in str(var_no_plot)

    # 4. Variants with @run: plot -> plot rendered in HTML, Markdown, and Text
    var_with_plot = DynoModel(
        txt="@variants: {a: [0.3, 0.7]}\n@run: solve\n@run: simulate: {T: 10}\n@run: plot: {variables: [x]}\n"
        + base_eqs
    ).run()
    assert var_with_plot.figure is not None
    assert "<svg" in (var_with_plot._repr_html_() or "")
    assert "data:image/svg+xml" in (var_with_plot._repr_markdown_() or "")
    assert "Simulation Plots" in str(var_with_plot)


def test_variants_muted_commands_render():
    base_eqs = """
    a <- 0.5
    x[~] <- 0.0
    y[~] <- 0.0
    e[t] <- N(0.0, 1.0)
    x[t] = a * x[t-1] + e[t]
    y[t] = 0.5 * x[t]
    """

    txt = (
        "@variants: {a: [0.3, 0.7]}\n"
        "@run: check;\n"
        "@run: solve;\n"
        "@run: simulate: {T: 10};\n"
        "@run: plot: {variables: [x]}\n" + base_eqs
    )
    res = DynoModel(txt=txt).run()

    # Muted check and solve and simulate
    assert "check" in res.muted_commands
    assert "solve" in res.muted_commands
    assert "simulate" in res.muted_commands
    assert not res._should_render_check
    assert not res._should_render_solution
    assert not res._should_render_simulation_tables
    assert res._should_render_plot

    # HTML
    html_out = res._repr_html_() or ""
    assert "Variants" in html_out
    assert "<h3>Check</h3>" not in html_out
    assert "<h3>Decision Rule</h3>" not in html_out
    assert "<h3>Moments</h3>" not in html_out
    assert "<h3>Simulation</h3>" in html_out
    assert "<svg" in html_out

    # Markdown
    md_out = res._repr_markdown_() or ""
    assert "**Variants:**" in md_out
    assert "## Check" not in md_out
    assert "## Solution" not in md_out
    assert "## Simulation" not in md_out
    assert "## Plot" in md_out

    # Text
    txt_out = res.to_text()
    assert "Checks\n------" not in txt_out
    assert "Solution:" not in txt_out
    assert "Simulation Plots\n----------------" in txt_out


def test_variants_muted_check_shows_on_error():
    # If a variant produces bad residuals, check should still be rendered even if muted
    base_eqs = """
    a <- 0.5
    x[~] <- 1.0  # Incorrect steady state: x = 0 is true steady state
    y[~] <- 0.0
    e[t] <- N(0.0, 1.0)
    x[t] = a * x[t-1] + e[t]
    y[t] = 0.5 * x[t]
    """

    # legacy inline form, kept for backward compatibility
    txt = "@run: variants: {a: [0.3, 0.7]}\n" "@run: check;\n" + base_eqs
    res = DynoModel(txt=txt).run()

    assert "check" in res.muted_commands
    # Because residuals are not zero, _should_render_check remains True
    assert res._should_render_check

    html_out = res._repr_html_() or ""
    assert "<h3>Check</h3>" in html_out
    assert "Residuals" in html_out

    md_out = res._repr_markdown_() or ""
    assert "## Check" in md_out

    txt_out = res.to_text()
    assert "Checks\n------" in txt_out


_VARIANTS_BASE = """
a <- 0.5
x[~] <- 0.0
e[t] <- N(0.0, 1.0)
x[t] = a * x[t-1] + e[t]
"""


def test_variants_metadata_parsed():
    model = DynoModel(txt="@variants: {a: [0.2, 0.8]}\n@run: solve\n" + _VARIANTS_BASE)
    assert model.metadata["variants"] == {"a": [0.2, 0.8]}


def test_variants_metadata_equivalent_to_inline_run_variants():
    pipeline = "@run: solve\n@run: simulate: {T: 10}\n"
    res_meta = DynoModel(
        txt="@variants: {a: [0.2, 0.8]}\n" + pipeline + _VARIANTS_BASE
    ).run()
    res_inline = DynoModel(
        txt="@run: variants: {a: [0.2, 0.8]}\n" + pipeline + _VARIANTS_BASE
    ).run()

    assert isinstance(res_meta, RunResultsVariants)
    assert isinstance(res_inline, RunResultsVariants)
    assert res_meta.labels == res_inline.labels == ["a=0.2", "a=0.8"]
    assert res_meta.errors == res_inline.errors == []


def test_variants_metadata_position_independent_of_run():
    # @variants may follow the @run lines: it is file-wide metadata
    res = DynoModel(
        txt="@run: solve\n@variants: {a: [0.2, 0.8]}\n" + _VARIANTS_BASE
    ).run()
    assert isinstance(res, RunResultsVariants)
    assert res.labels == ["a=0.2", "a=0.8"]


def test_variants_metadata_without_run_uses_default_pipeline():
    res = DynoModel(txt="@variants: {a: [0.2, 0.8]}\n" + _VARIANTS_BASE).run(
        default_pipeline=True
    )
    assert isinstance(res, RunResultsVariants)
    assert len(res) == 2


def test_variants_metadata_not_re_expanded_in_variants():
    res = DynoModel(
        txt="@variants: {a: [0.2, 0.8]}\n@run: solve\n" + _VARIANTS_BASE
    ).run()
    for m in res.model:
        assert "variants" not in m.metadata


def test_variants_metadata_inline_command_takes_precedence():
    res = DynoModel(
        txt="@variants: {a: [0.1, 0.2, 0.3]}\n"
        "@run: variants: {a: [0.4, 0.6]}\n"
        "@run: solve\n" + _VARIANTS_BASE
    ).run()
    assert res.labels == ["a=0.4", "a=0.6"]


@pytest.mark.parametrize("bad", ["3", "[1, 2]", "foo", "{a: [1, 2]"])
def test_variants_metadata_must_be_a_mapping(bad):
    with pytest.raises(Exception):
        DynoModel(txt=f"@variants: {bad}\n@run: solve\n" + _VARIANTS_BASE)
