from pathlib import Path

from dyno.gui.explorer_core import (
    CONTENT_TYPES,
    OUTPUT_FORMATS,
    ImportVariant,
    available_backends,
    default_backends,
    default_variant_keys,
    discover_models,
    render_markdown_myst,
    render_variant,
    variants_for,
)

RBC_DYNO = Path("examples/rbc.dyno")
RBC_MOD = Path("examples/modfiles/RBC.mod")


def test_discover_models_finds_dyno_and_mod_files():
    paths = discover_models("examples")
    suffixes = {path.suffix for path in paths}
    assert ".dyno" in suffixes
    assert ".mod" in suffixes
    assert RBC_DYNO in paths


def test_discover_models_missing_directory_returns_empty():
    assert discover_models("does/not/exist") == []


def test_available_backends_dyno_file_only_offers_dyno_model():
    backends = available_backends(RBC_DYNO)
    assert set(backends) == {"DynoModel"}


def test_available_backends_mod_file_may_offer_dynare_model():
    backends = available_backends(RBC_MOD)
    assert "DynoModel" in backends
    assert set(backends) <= {"DynoModel", "DynareModel"}


def test_variants_for_include_strict_and_non_strict_per_backend():
    variants = variants_for(RBC_DYNO)
    assert ImportVariant("DynoModel", False) in variants
    assert ImportVariant("DynoModel", True) in variants


def test_default_variant_keys_are_non_strict():
    keys = default_variant_keys(RBC_DYNO)
    assert keys == [ImportVariant("DynoModel", False).key]


def test_default_backends_lists_every_available_backend():
    assert default_backends(RBC_DYNO) == ["DynoModel"]
    assert set(default_backends(RBC_MOD)) == set(available_backends(RBC_MOD))


def test_render_variant_representation_across_formats():
    source = RBC_DYNO.read_text()
    variant = ImportVariant("DynoModel", False)
    for output_format in OUTPUT_FORMATS:
        ok, kind, content, line = render_variant(
            RBC_DYNO, source, variant, "representation", output_format
        )
        assert ok is True
        assert kind == ("myst" if output_format == "markdown" else "html")
        assert content.strip() != ""
        assert line is None


def test_render_variant_report_across_formats():
    # examples/rbc.dyno is not square as-is (it's a calibration/estimation
    # style file), so `model.run(default_pipeline=True)` is expected to fail
    # here; render_variant must still degrade gracefully rather than raise.
    source = RBC_DYNO.read_text()
    variant = ImportVariant("DynoModel", False)
    for output_format in OUTPUT_FORMATS:
        ok, kind, content, _ = render_variant(
            RBC_DYNO, source, variant, "report", output_format
        )
        assert content.strip() != ""
        if not ok:
            assert kind == "html"


def test_render_variant_report_for_solvable_model():
    source = RBC_MOD.read_text()
    variant = ImportVariant("DynoModel", False)
    ok, kind, content, line = render_variant(RBC_MOD, source, variant, "report", "text")
    assert ok is True
    assert kind == "html"
    assert content.strip() != ""
    assert line is None


def test_render_variant_reports_import_errors():
    variant = ImportVariant("DynoModel", False)
    ok, kind, content, line = render_variant(
        RBC_DYNO, "this is not a valid model", variant, "representation", "html"
    )
    assert ok is False
    assert kind == "html"
    assert "Error" in content or "error" in content
    assert line == 1


def test_render_variant_reports_error_line_for_multiline_source():
    variant = ImportVariant("DynoModel", False)
    ok, kind, content, line = render_variant(
        RBC_DYNO,
        "a <- 1\nb <- 2\nthis is not %% valid\n",
        variant,
        "representation",
        "html",
    )
    assert ok is False
    assert line == 3


def test_render_markdown_myst_renders_tables_and_leaves_math_for_katex():
    html = render_markdown_myst(
        "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n$$x^2$$\n"
    )
    assert "<h1>Title</h1>" in html
    assert "<table>" in html
    # Dollar-delimited math is left untouched for the client-side KaTeX
    # auto-render pass (see MystHtml / _markdown_template reuse), not
    # converted to a math node server-side.
    assert "$$x^2$$" in html


def test_render_markdown_myst_supports_definition_lists_and_task_lists():
    html = render_markdown_myst("term\n: definition\n\n- [ ] todo\n- [x] done\n")
    assert "<dl>" in html and "<dt>term</dt>" in html
    assert 'type="checkbox"' in html


def test_render_markdown_myst_renders_admonition_directive():
    html = render_markdown_myst(":::{tip} All good\nEverything checks out.\n:::\n")
    assert "<details" not in html
    assert "All good" in html
    assert "Everything checks out" in html


def test_render_markdown_myst_renders_dropdown_admonition_as_details():
    html = render_markdown_myst(
        ":::{warning} Residuals are not zero\n:class: dropdown\nsome details\n:::\n"
    )
    assert "<details" in html
    assert "<summary" in html
    assert "Residuals are not zero" in html
    assert "some details" in html
    # the `:class: dropdown` option line itself must not leak into the body
    assert "class: dropdown" not in html


def test_render_markdown_myst_renders_dropdown_and_nested_tab_items():
    html = render_markdown_myst(
        "::::{dropdown} IRFS\n"
        ":::{tab-set}\n"
        "::{tab-item} epsilon\ncontent-a\n::\n"
        "::{tab-item} leta\ncontent-b\n::\n"
        ":::\n"
        "::::\n"
    )
    assert "IRFS" in html
    assert "epsilon" in html and "content-a" in html
    assert "leta" in html and "content-b" in html


def test_render_markdown_myst_code_directive_sets_language_class():
    html = render_markdown_myst("```{code} python\nx = 1\n```\n")
    assert 'class="language-python"' in html
    assert "x = 1" in html


def test_render_markdown_myst_unknown_directive_shows_name_not_raw_syntax():
    html = render_markdown_myst(":::{eval-rst}\nsome body\n:::\n")
    assert "eval-rst" in html
    assert "some body" in html
    assert ":::" not in html


def test_content_types_and_formats_are_stable():
    assert CONTENT_TYPES == ("representation", "report")
    assert OUTPUT_FORMATS == ("text", "html", "markdown")
