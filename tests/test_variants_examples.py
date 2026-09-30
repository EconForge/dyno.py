from __future__ import annotations

import pytest
from dyno import DynoModel, RunResults, RunResultsVariants


@pytest.mark.parametrize(
    "model_file,expected_labels,is_deterministic",
    [
        ("examples/variants/deterministic.dyno", ["alph=0.33", "alph=0.5"], True),
        ("examples/variants/stochastic.dyno", ["rho=0.8", "rho=0.95"], False),
        ("examples/variants/stochastic_forced.dyno", ["rho=0.8", "rho=0.95"], False),
    ],
)
def test_variants_examples_run_and_structure(
    model_file: str, expected_labels: list[str], is_deterministic: bool
):
    model = DynoModel(model_file)
    res = model.run()

    assert isinstance(res, RunResultsVariants)
    assert len(res) == len(expected_labels)
    assert res.labels == expected_labels
    assert res.errors == []
    assert res.warnings == []

    # Text report
    txt = res.to_text()
    assert "RunResultsVariants" in txt
    assert "Model\n-----" in txt
    assert f"variants ({len(expected_labels)}): " in txt
    assert f"deterministic: {is_deterministic}" in txt
    assert "Checks\n------" in txt
    for lbl in expected_labels:
        assert f"[{lbl}]" in txt
    assert "Outputs\n-------" in txt
    assert f"({len(expected_labels)}/{len(expected_labels)} variants)" in txt
    assert "Simulation Plots\n----------------" in txt
    assert "Diagnostics\n-----------" in txt
    assert "Timing\n------" in txt

    # HTML report
    html_out = res._repr_html_()
    assert html_out is not None
    assert "<h3>Model:" in html_out
    assert "Variants:" in html_out
    for lbl in expected_labels:
        assert lbl in html_out
    assert "<h3>Check</h3>" in html_out
    assert "<h3>Simulation</h3>" in html_out
    assert "<svg" in html_out

    # Markdown report
    md_out = res._repr_markdown_()
    assert md_out is not None
    assert ":::{note} Model Overview" in md_out
    assert "**Variants:**" in md_out
    assert ":::{dropdown} Calibration" in md_out
    assert ":::{dropdown} Equations" in md_out
    assert "## Check" in md_out
    assert "## Simulation" in md_out
    assert "## Plot" in md_out
    assert md_out.index("## Simulation") < md_out.index("## Plot")
    assert "data:image/svg+xml" in md_out

    if not is_deterministic:
        assert "## Solution" in md_out
        assert "::::{tab-set}" in md_out
        for lbl in expected_labels:
            assert f":::{{tab-item}} {lbl}" in md_out


def test_variants_and_single_report_parity():
    stoch_model = DynoModel("examples/variants/stochastic.dyno")
    res_var = stoch_model.run()

    # Compare against univariant model
    txt_single = open("examples/variants/stochastic.dyno").read()
    txt_no_var = "\n".join(
        line for line in txt_single.splitlines() if "@run: variants" not in line
    )
    res_single = DynoModel(txt=txt_no_var).run()
    assert isinstance(res_single, RunResults)

    # 1. Text parity: matching section headers
    sections = [
        "Model",
        "Checks",
        "Outputs",
        "Simulation Plots",
        "Diagnostics",
        "Timing",
    ]
    txt_v = res_var.to_text()
    txt_s = res_single.to_text()
    for sec in sections:
        assert sec in txt_v
        assert sec in txt_s

    # 2. HTML parity: matching heading tags
    assert ("<h3>Check</h3>" in (res_single._repr_html_() or "")) == (
        "<h3>Check</h3>" in (res_var._repr_html_() or "")
    )
    assert ("<h3>Decision Rule</h3>" in (res_single._repr_html_() or "")) == (
        "<h3>Decision Rule</h3>" in (res_var._repr_html_() or "")
    )
    assert ("<h3>Simulation</h3>" in (res_single._repr_html_() or "")) == (
        "<h3>Simulation</h3>" in (res_var._repr_html_() or "")
    )

    # 3. Markdown parity: matching top-level headers and dropdowns
    assert ":::{dropdown} Calibration" in (res_single._repr_markdown_() or "")
    assert ":::{dropdown} Calibration" in (res_var._repr_markdown_() or "")
    assert ":::{dropdown} Equations" in (res_single._repr_markdown_() or "")
    assert ":::{dropdown} Equations" in (res_var._repr_markdown_() or "")
    assert "## Check" in (res_single._repr_markdown_() or "")
    assert "## Check" in (res_var._repr_markdown_() or "")
    assert "## Solution" in (res_single._repr_markdown_() or "")
    assert "## Solution" in (res_var._repr_markdown_() or "")
    assert "## Simulation" in (res_single._repr_markdown_() or "")
    assert "## Simulation" in (res_var._repr_markdown_() or "")
    assert "## Plot" in (res_single._repr_markdown_() or "")
    assert "## Plot" in (res_var._repr_markdown_() or "")


def test_multi_shock_svg_dash_styling():
    stoch_model = DynoModel("examples/variants/stochastic.dyno")
    res = stoch_model.run()
    html_out = res._repr_html_() or ""

    # Multi-shock model with e and u should have shock legend and stroke-dasharray
    assert "shock: e" in html_out
    assert "shock: u" in html_out
    assert "stroke-dasharray=" in html_out


def test_plot_options_vars_alias_filtering():
    txt = """
    a <- 0.5
    x[~] <- 0.0
    y[~] <- 0.0
    z[~] <- 0.0
    e[t] <- N(0.0, 1.0)
    x[t] = a * x[t-1] + e[t]
    y[t] = 0.5 * x[t]
    z[t] = 0.2 * x[t]

    @run: variants: {a: [0.2, 0.8]}
    @run: solve
    @run: simulate: {T: 10}
    @run: plot: {vars: [x, y]}
    """
    res = DynoModel(txt=txt).run()
    txt_out = res.to_text()

    # z should not be in the simulation plots because vars: [x, y] filtered it out
    assert "Simulation Plots" in txt_out
    assert "x" in txt_out
    assert "y" in txt_out
    svg_html = res._repr_html_() or ""
    svg_part = svg_html[svg_html.find("<svg") : svg_html.find("</svg>")]
    assert ">x<" in svg_part
    assert ">y<" in svg_part
    assert ">z<" not in svg_part
