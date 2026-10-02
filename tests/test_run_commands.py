from dyno import DynoModel
from dyno.report import RunResults


def test_dynomodel_modfile_metadata_includes_dynare_commands():
    txt = """
var x;
varexo e;
parameters a;

a = 0.9;

model;
x = a*x(-1) + e;
end;

initval;
x = 0;
e = 0;
end;

steady;
check;
stoch_simul(irf=8);
"""
    model = DynoModel(filename="tiny.mod", txt=txt)

    commands = model._normalize_run_commands()

    assert [c["command"] for c in commands] == ["steady", "check", "stoch_simul"]
    assert commands[2]["options"].get("irf") == 8


def test_dynomodel_run_accepts_stoch_simul_command():
    txt = """
var x;
varexo e;
parameters a;

a = 0.9;

model;
x = a*x(-1) + e;
end;

initval;
x = 0;
e = 0;
end;

steady;
stoch_simul(irf=6);
"""
    model = DynoModel(filename="tiny.mod", txt=txt)

    results = model.run(default_pipeline=False)

    assert isinstance(results, RunResults)
    assert results.solution is not None
    assert results.eigenvalues is not None
    assert results.simulation is not None
    assert isinstance(results.simulation, dict)


def test_dynomodel_run_check_populates_eigenvalues():
    txt = """
var x;
varexo e;
parameters a;

a = 0.9;

model;
x = a*x(-1) + e;
end;

initval;
x = 0;
e = 0;
end;

steady;
check;
"""
    model = DynoModel(filename="tiny_check.mod", txt=txt)

    results = model.run(default_pipeline=False)

    assert isinstance(results, RunResults)
    assert results.residuals is not None
    assert results.eigenvalues is not None
    assert results.eigenvalues is getattr(results.model, "_eigenvalues", None)


def test_dyno_run_check_semicolon_mutes_check_section_when_clean():
    txt = """
@name: TinyClean
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: steady
@run: check;
@run: solve
"""
    model = DynoModel(txt=txt)
    assert "check" in model._normalize_run_commands()[1]["command"]
    assert model._normalize_run_commands()[1]["mute"] is True

    results = model.run()
    assert results.residuals is not None
    assert results.eigenvalues is not None
    assert "check" in results.muted_commands
    assert results._should_render_check is False

    # HTML
    html = results._render_html_report()
    assert "<h3>Check</h3>" not in html

    # Markdown
    md = results._repr_markdown_()
    assert md is not None
    assert "## Check" not in md

    # Text summary
    txt_rep = results.to_text()
    assert "Checks" not in txt_rep


def test_dyno_run_check_without_semicolon_displays_check_section():
    txt = """
@name: TinyCleanNoMute
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: steady
@run: check
@run: solve
"""
    model = DynoModel(txt=txt)
    assert model._normalize_run_commands()[1]["mute"] is False

    results = model.run()
    assert results._should_render_check is True

    # HTML
    html = results._render_html_report()
    assert "<h3>Check</h3>" in html

    # Markdown
    md = results._repr_markdown_()
    assert md is not None
    assert "## Check" in md

    # Text summary
    txt_rep = results.to_text()
    assert "Checks" in txt_rep


def test_dyno_run_check_semicolon_displays_when_residuals_non_zero():
    txt = """
@name: TinyDirty
alpha := 0.9
x[~] := 5.0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: check;
"""
    model = DynoModel(txt=txt)
    results = model.run()

    # Even though check is muted, residuals are non-zero, so it should NOT be hidden
    assert results._should_render_check is True

    html = results._render_html_report()
    assert "<h3>Check</h3>" in html

    md = results._repr_markdown_()
    assert md is not None
    assert "## Check" in md

    txt_rep = results.to_text()
    assert "Checks" in txt_rep


def test_dyno_run_options_with_semicolon_parses_yaml_and_mutes():
    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: simul: {T: 20};
@run: solve;
"""
    model = DynoModel(txt=txt)
    cmds = model._normalize_run_commands()

    assert cmds[0]["command"] == "simul"
    assert cmds[0]["options"] == {"T": 20}
    assert cmds[0]["mute"] is True

    assert cmds[1]["command"] == "solve"
    assert cmds[1]["mute"] is True

    results = model.run()
    assert results._should_render_solution is False
    assert results._should_render_simulation_tables is False

    md = results._repr_markdown_()
    assert md is not None
    assert "## Solution" not in md
    assert "## Simulation" not in md


def test_dyno_metadata_general_semicolon_stripping():
    txt = """
@name: MyModel;
@version: 2;
alpha := 0.5
x[~] := 0
x[t] = alpha * x[t-1]
"""
    model = DynoModel(txt=txt)
    assert model.metadata["name"] == "MyModel"
    assert model.metadata["version"] == 2
    assert model.metadata.get("_muted_name") is True
    assert model.metadata.get("_muted_version") is True


_STOCH_SIMUL_TEMPLATE = """
var x y;
varexo e;
parameters a;

a = 0.9;

model;
x = a*x(-1) + e;
y = 2*x;
end;

initval;
x = 0;
y = 0;
e = 0;
end;

steady;
{command}
"""


def test_dynomodel_stoch_simul_symbol_list_parsed_as_variables():
    txt = _STOCH_SIMUL_TEMPLATE.format(command="stoch_simul(order=1, irf=8) y x;")
    model = DynoModel(filename="tiny.mod", txt=txt)

    commands = model._normalize_run_commands()

    assert commands[1]["options"] == {"order": 1, "irf": 8, "variables": ["y", "x"]}


def test_dynomodel_stoch_simul_plots_requested_variables():
    txt = _STOCH_SIMUL_TEMPLATE.format(command="stoch_simul(irf=8) y;")
    model = DynoModel(filename="tiny.mod", txt=txt)

    results = model.run(default_pipeline=False)

    assert results.figure is not None
    assert results._plot_options["variables"] == ["y"]
    assert set(results.figure.data["variable"]) == {"y"}
    assert "Simulation charts" in results.to_markdown()


def test_dynomodel_stoch_simul_nograph_and_irf0_skip_plot():
    for command in ["stoch_simul(nograph);", "stoch_simul(irf=0, periods=500);"]:
        txt = _STOCH_SIMUL_TEMPLATE.format(command=command)
        results = DynoModel(filename="tiny.mod", txt=txt).run(default_pipeline=False)

        assert results.solution is not None
        assert results.figure is None


def test_dyno_run_semicolon_at_end_of_line_mutes_and_runs():
    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: simulate: {T: 100};
"""
    model = DynoModel(txt=txt)
    cmds = model._normalize_run_commands()

    assert len(cmds) == 1
    assert cmds[0]["command"] == "simulate"
    assert cmds[0]["options"] == {"T": 100}
    assert cmds[0]["mute"] is True

    results = model.run()
    assert results.simulation is not None
    assert results._should_render_simulation_tables is False


def test_dyno_run_semicolon_not_at_end_raises_error():
    import pytest
    from dyno.errors import ParserError

    # Invalid: semicolon after command name before options
    txt_err1 = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: simulate:; {T:100}
"""
    with pytest.raises(ParserError, match="unexpected ';'"):
        DynoModel(txt=txt_err1)

    # Invalid: semicolon after command name before colon/options
    txt_err2 = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: simulate; : {T: 100}
"""
    with pytest.raises(ParserError, match="unexpected ';'"):
        DynoModel(txt=txt_err2)

    # Invalid: multiple commands separated by semicolon on one line
    txt_err3 = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: steady; check;
"""
    with pytest.raises(ParserError, match="unexpected ';'"):
        DynoModel(txt=txt_err3)

    # Invalid: semicolon in middle and also trailing semicolon
    txt_err4 = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: simulate:; {T:100};
"""
    with pytest.raises(ParserError, match="unexpected ';'"):
        DynoModel(txt=txt_err4)


# An explosive backward root: no stable solution (Blanchard-Kahn violated).
BK_VIOLATION = """
alpha := 2.0
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]
"""


def test_dyno_run_check_computes_eigenvalues_without_solve():
    txt = """
alpha := 0.9
x[~] := 0
e[t] := N(0, 1)
x[t] = alpha * x[t-1] + e[t]

@run: steady
@run: check
"""
    results = DynoModel(txt=txt).run()
    assert results.solution is None
    assert results.eigenvalues is not None
    assert results.bk_check is True
    md = results._repr_markdown_()
    assert md is not None
    assert ":::{tip} Blanchard-Kahn conditions are met" in md


def test_dyno_run_check_reports_blanchard_kahn_violation():
    results = DynoModel(txt=BK_VIOLATION + "\n@run: steady\n@run: check\n").run()
    assert results.eigenvalues is not None
    assert results.bk_check is False
    assert results.errors == []
    md = results._repr_markdown_()
    assert md is not None
    assert ":::{warning} Blanchard-Kahn conditions are not met" in md


def test_dyno_run_solve_records_blanchard_kahn_violation():
    txt = BK_VIOLATION + """
@run: steady
@run: check;
@run: solve
@run: simulate
@run: plot
"""
    results = DynoModel(txt=txt).run()

    # The run completes: the failure is recorded, later commands are skipped.
    assert results.solution is None
    assert results.simulation is None
    assert results.figure is None
    assert results.eigenvalues is not None
    assert results.bk_check is False
    assert len(results.errors) == 1
    assert "Eigenvalue condition not satisfied" in results.errors[0]["message"]

    # The (muted) check section is shown, with the Blanchard-Kahn warning.
    assert results._should_render_check is True
    md = results._repr_markdown_()
    assert md is not None
    assert ":::{warning} Blanchard-Kahn conditions are not met" in md
    assert "Blanchard-Kahn conditions: NOT met" in results.to_text()


def test_solve_raises_blanchard_kahn_error_with_eigenvalues():
    import pytest
    from dyno.errors import BlanchardKahnError

    with pytest.raises(BlanchardKahnError) as info:
        DynoModel(txt=BK_VIOLATION).solve()
    assert info.value.evs is not None
    assert len(info.value.evs) == 2
