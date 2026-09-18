files = [
    "example1.mod",
    "example2.mod",
    "NK_baseline.mod",
    "example3.mod",
    "Gali_2015.mod",
]

# TODO: for excluded file, check the error is meaningful
exclude = ["NK_baseline.mod"]  # uses an external steady_state file

unsupported = [
    "example1.mod",  # Uses native statement
    "example3.mod",  # calls steady_state function
    "Gali_2015.mod",  # calls external funciton in steady-state
]

from dyno import dynare_model
from dyno.errors import DynareParserError
import pytest

files = [f for f in files if not (f in exclude)]


@pytest.mark.parametrize("filename", files)
def test_modfile_import(filename):

    f = filename

    filename = "examples/modfiles/" + f

    try:

        mod = dynare_model.DynareModel(filename)
        sol = mod.solve()
        print(sol)

        assert True

    except Exception as e:
        assert f in unsupported
        assert isinstance(e, DynareParserError)


def test_dsge_report_dynare_syntax_error_highlighting():
    from dyno.report import dsge_report
    from unittest.mock import Mock, patch

    txt = "var y\nmodel;\ny = 1;\nend;"
    display_mock = Mock()

    with patch("IPython.display.display", display_mock):
        res = dsge_report(txt, filename="test.mod")

    assert len(res.errors) >= 1
    assert res.errors[0].get("line") == 3

    assert len(res._highlighting_data) >= 1
    assert res._highlighting_data[0]["line"] == 3
    assert res._highlighting_data[0]["type"] == "error"

    highlight_calls = [
        call.args[0]
        for call in display_mock.call_args_list
        if call.args
        and isinstance(call.args[0], dict)
        and "application/vnd.jupyterlab-dyno.highlighting+json" in call.args[0]
    ]
    assert len(highlight_calls) == 1
    assert (
        highlight_calls[0]["application/vnd.jupyterlab-dyno.highlighting+json"][0][
            "line"
        ]
        == 3
    )
