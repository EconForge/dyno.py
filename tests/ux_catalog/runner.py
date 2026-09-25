"""Audit runner for the Dyno UX and Misspecified Models Catalog.

Executes all test cases through the Dyno pipeline, records actual exceptions/warnings,
and produces a structured audit report.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional

from dyno.dyno_model import DynoModel
from tests.ux_catalog.cases import ALL_CASES, DiagnosticCase


@dataclass
class CaseExecutionResult:
    case: DiagnosticCase
    stage_reached: str
    raised_exception: Optional[str]
    exception_message: Optional[str]
    warnings_emitted: List[tuple[str, str]]
    success: bool


def execute_case(case: DiagnosticCase) -> CaseExecutionResult:
    stage_reached = "start"
    raised_exception: Optional[str] = None
    exception_message: Optional[str] = None
    warnings_emitted: List[tuple[str, str]] = []
    success = False

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        try:
            stage_reached = "import"
            model = DynoModel(txt=case.source)

            if case.expected_stage in ("check", "steady", "solve"):
                stage_reached = "check"
                model.check()

            if case.expected_stage in ("steady", "solve"):
                stage_reached = "steady"
                model = model.steady()

            if case.expected_stage == "solve":
                stage_reached = "solve"
                model.solve()

            success = True
        except Exception as e:
            raised_exception = type(e).__name__
            exception_message = str(e)
            success = False

    for w in captured_warnings:
        warnings_emitted.append((w.category.__name__, str(w.message)))

    return CaseExecutionResult(
        case=case,
        stage_reached=stage_reached,
        raised_exception=raised_exception,
        exception_message=exception_message,
        warnings_emitted=warnings_emitted,
        success=success,
    )


def run_all_cases() -> List[CaseExecutionResult]:
    return [execute_case(case) for case in ALL_CASES]


def generate_markdown_report(results: List[CaseExecutionResult]) -> str:
    lines: List[str] = []
    lines.append("# Dyno UX Audit Report: Misspecified & Incomplete Models Catalog")
    lines.append("")
    lines.append(f"Total catalog cases: **{len(results)}**")
    lines.append("")

    # Summary table
    lines.append("## Summary by Case")
    lines.append("")
    lines.append(
        "| ID | Category | Title | Stage | Raised Error | Warnings | Ideal Diagnostic |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for res in results:
        c = res.case
        err = f"`{res.raised_exception}`" if res.raised_exception else "_None_"
        warn_cnt = len(res.warnings_emitted)
        warn_str = f"{warn_cnt} warning(s)" if warn_cnt > 0 else "_None_"
        ideal_trunc = (
            (c.ideal_diagnostic[:60] + "...")
            if len(c.ideal_diagnostic) > 60
            else c.ideal_diagnostic
        )
        lines.append(
            f"| **{c.id}** | `{c.category}` | {c.title} | `{c.expected_stage}` | {err} | {warn_str} | {ideal_trunc} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Diagnostic Profiles")
    lines.append("")

    for res in results:
        c = res.case
        lines.append(f"### [{c.id}] {c.title}")
        lines.append(f"- **Category:** `{c.category}`")
        lines.append(f"- **User Intent:** {c.intent}")
        lines.append(f"- **Expected Detection Stage:** `{c.expected_stage}`")
        lines.append(f"- **Actual Stage Reached:** `{res.stage_reached}`")
        lines.append("- **Source Code Snippet:**")
        lines.append("```text")
        lines.append(c.source.strip())
        lines.append("```")

        if res.raised_exception:
            lines.append(
                f"- **Raised Exception:** `{res.raised_exception}`: `{res.exception_message}`"
            )
        else:
            lines.append(
                "- **Raised Exception:** _None (Execution succeeded or silent NaN)_"
            )

        if res.warnings_emitted:
            lines.append("- **Warnings Emitted:**")
            for w_cat, w_msg in res.warnings_emitted:
                lines.append(f"  - `{w_cat}`: {w_msg}")
        else:
            lines.append("- **Warnings Emitted:** _None_")

        lines.append(f"- **Ideal Diagnostic:** > *{c.ideal_diagnostic}*")
        lines.append(f"- **Suggested Fix:** `{c.suggested_fix}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Dyno UX Diagnostic Catalog.")
    parser.add_argument(
        "--output", "-o", help="Optional markdown output path for the audit report."
    )
    args = parser.parse_args()

    results = run_all_cases()
    report = generate_markdown_report(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report successfully saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
