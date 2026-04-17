from dyno.errors import DynareParserError


class _DummyDynareError(Exception):
    pass


def test_dynare_parser_error_extracts_line_and_column():
    err = _DummyDynareError("syntax error, unexpected TIMES: line 46, col 34")

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 46
    assert parsed.column == 34


def test_dynare_parser_error_handles_missing_location():
    err = _DummyDynareError("unsupported feature: native statement")

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line is None
    assert parsed.column is None
