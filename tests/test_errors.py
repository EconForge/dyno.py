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


def test_dynare_parser_error_extracts_cols_range():
    err = _DummyDynareError(
        "ERROR: in_memory.mod: line 3, cols 1-6: y is not a parameter"
    )

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 3
    assert parsed.column == 1


def test_dynare_parser_error_extracts_eof_cols():
    err = _DummyDynareError(
        "ERROR: in_memory.mod: line 4, cols 1-0: syntax error, unexpected end of file"
    )

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 4
    assert parsed.column == 1


def test_dynare_parser_error_extracts_lines_range():
    err = _DummyDynareError(
        "ERROR: in_memory.mod: lines 10-12, cols 5-8: unexpected token"
    )

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 10
    assert parsed.column == 5


def test_dynare_parser_error_extracts_line_without_col():
    err = _DummyDynareError("ERROR: in_memory.mod: line 15: some error message")

    parsed = DynareParserError(err)  # type: ignore[arg-type]

    assert parsed.line == 15
    assert parsed.column is None


def test_dynare_parser_error_reads_direct_attributes():
    class _ErrorWithAttrs(Exception):
        line = 7
        column = 12

    err = _ErrorWithAttrs("some message without location text")
    parsed = DynareParserError(err)

    assert parsed.line == 7
    assert parsed.column == 12


def test_dynare_parser_error_reads_begin_attrs_and_location():
    class _DummyLoc:
        line = 9
        column = 14

    class _ErrorWithLoc(Exception):
        location = _DummyLoc()

    err = _ErrorWithLoc("custom error")
    parsed = DynareParserError(err)

    assert parsed.line == 9
    assert parsed.column == 14
