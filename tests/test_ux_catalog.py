"""Tests for the UX diagnostics catalog and runner."""

import pytest
from tests.ux_catalog.cases import ALL_CASES, get_case, get_cases_by_category
from tests.ux_catalog.runner import execute_case, generate_markdown_report


def test_catalog_has_comprehensive_cases():
    assert len(ALL_CASES) >= 20
    categories = {c.category for c in ALL_CASES}
    assert "syntax" in categories
    assert "symbol_resolution" in categories
    assert "steady_state" in categories
    assert "shocks" in categories
    assert "system_structure" in categories
    assert "solvability" in categories


def test_get_case_by_id():
    case = get_case("SYN-001")
    assert case.id == "SYN-001"
    assert case.category == "syntax"

    with pytest.raises(KeyError):
        get_case("NON-EXISTENT")


def test_get_cases_by_category():
    syntax_cases = get_cases_by_category("syntax")
    assert len(syntax_cases) >= 5
    for c in syntax_cases:
        assert c.category == "syntax"


def test_execute_case_runs_without_unhandled_crash():
    # Pick a sample case and run it
    case = get_case("SYN-003")  # trailing semicolon
    res = execute_case(case)
    assert res.stage_reached == "import"
    assert res.raised_exception == "LARKParserError"


def test_markdown_report_generation():
    sample_cases = ALL_CASES[:3]
    results = [execute_case(c) for c in sample_cases]
    report = generate_markdown_report(results)
    assert "# Dyno UX Audit Report" in report
    assert "SYN-001" in report
