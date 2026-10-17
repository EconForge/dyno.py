"""Import `.mod` files through the Dynare backend.

The two groups below encode distinct contracts that used to be conflated into a
single `try/except`: a file is either rejected while parsing, or parsed and
solved. Keeping them apart matters because a solve-time numerical failure is not
a parse failure.
"""

import pytest

from dyno.dynare import DynareModel
from dyno.errors import DynareParserError

# Parsed and solved successfully.
# `Gali_2015.mod` is rank deficient (its external steady state leaves `p` out),
# so it has a unit root, which is accepted as a non-diverging root.
supported = [
    "example1.mod",
    "example2.mod",
    "Gali_2015.mod",  # calls external function in steady-state
]

# Not supported yet: rejected while parsing the source.
unsupported = [
    "example3.mod",  # calls steady_state function
]

# Not tested: `NK_baseline.mod` uses an external steady_state file.
# TODO: for the excluded file, check the error is meaningful


def _load(filename: str) -> DynareModel:
    return DynareModel("examples/modfiles/" + filename)


@pytest.mark.parametrize("filename", supported)
def test_modfile_import(filename):
    assert _load(filename).solve() is not None


@pytest.mark.parametrize("filename", unsupported)
def test_unsupported_modfile_raises_parser_error(filename):
    with pytest.raises(DynareParserError):
        _load(filename)


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
