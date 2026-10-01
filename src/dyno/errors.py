from __future__ import annotations

from typing import TYPE_CHECKING, Any


class DynoError(Exception):
    """Base exception for all Dyno-specific errors."""


class UndefinedSymbolWarning(UserWarning):
    """Warning raised when a model references undefined parameters or variables."""


class ConvergenceWarning(UserWarning):
    """Warning raised when a numerical solver fails to converge."""


class RedefinitionWarning(UserWarning):
    """Warning raised when a constant is assigned more than once; the first value is kept."""


class UndefinedSymbolError(DynoError):
    """Raised when an operation requires symbols that are not defined."""


class SystemStructureError(DynoError):
    """Raised when the system of equations has structural issues (e.g. non-square system)."""


class ParserError(DynoError):

    line: int | None
    column: int | None
    message: str
    details: str | None

    def __init__(self, message: str) -> None:
        self.line = None
        self.column = None
        self.details = None
        super().__init__(message)


from lark.exceptions import UnexpectedInput
from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF

# Lark parser error
# All errors inherit from lark.exceptions.UnexpectedInput
# has get context method to pretty print the problem
# UnexpectedCharacters: The lexer encountered an unexpected string
# UnexpectedToken: The parser received an unexpected token
# UnexpectedEOF: The parser expected a token, but the input ended


class LARKParserError(ParserError):

    def __init__(self, lark_error: UnexpectedInput, txt=None) -> None:

        line = lark_error.line
        column = lark_error.column
        if isinstance(lark_error, UnexpectedCharacters):
            message = f"Unexpected characters at {(line,column)}"
        elif isinstance(lark_error, UnexpectedToken):
            value = lark_error.token.value
            if "\n" in value:
                value = "newline"
            message = f"Unexpected token `{value}` at {(line,column)}"
        elif isinstance(lark_error, UnexpectedEOF):
            message = f"Unexpected end of file at {(line,column)}"
        else:
            message = str(type(lark_error))

        if txt is not None:
            details = lark_error.get_context(txt)
            details += str(lark_error)
        else:
            details = str(lark_error)
        super().__init__(message)
        self.column = column
        self.line = line
        self.details = details


import re

if TYPE_CHECKING:
    from dynare_preprocessor import PreprocessorException


class DynareParserError(ParserError):

    def __init__(self, err: "PreprocessorException" | Exception | str) -> None:
        message = str(err)

        line: int | None = getattr(err, "line", None)
        if line is None:
            line = getattr(err, "begin_line", None)

        column: int | None = getattr(err, "column", None)
        if column is None:
            column = getattr(err, "begin_column", None)

        loc = getattr(err, "location", None)
        if loc is not None:
            if line is None:
                line = getattr(loc, "line", getattr(loc, "begin_line", None))
            if column is None:
                column = getattr(loc, "column", getattr(loc, "begin_column", None))

        # Typical dynare-preprocessor message formats:
        # "syntax error, unexpected TIMES: line 46, col 34"
        # "ERROR: in_memory.mod: line 3, cols 1-6: y is not a parameter"
        # "ERROR: in_memory.mod: lines 3-5, cols 1-6: syntax error"
        if line is None:
            m_line = re.search(r"\blines?\s+(\d+)", message, flags=re.IGNORECASE)
            if m_line is not None:
                line = int(m_line.group(1))

        if column is None:
            m_col = re.search(r"\bcols?\s+(\d+)", message, flags=re.IGNORECASE)
            if m_col is not None:
                column = int(m_col.group(1))

        super().__init__(message)
        self.line = line
        self.column = column
        self.details = message


class SteadyStateError(DynoError):

    def __init__(
        self,
        residuals: Any,
        steady_stats: dict[str, Any] | None = None,
    ) -> None:
        self.residuals = residuals
        self.steady_stats = steady_stats
        message = f"Steady state values don't satisfy model equations. Max residual is {max(abs(r) for r in residuals)}"
        self.details = f"Residuals: {residuals}"
        super().__init__(message)


class BlanchardKahnError(DynoError):
    """Raised when Blanchard-Kahn eigenvalue conditions are not satisfied."""


class UnsupportedFeatureError(ParserError):
    """Raised when a model uses syntax or features from Dynare not currently supported by Dyno."""

    def __init__(
        self,
        message: str,
        *,
        feature: str | None = None,
        line: int | None = None,
        column: int | None = None,
        details: str | None = None,
    ) -> None:
        super().__init__(message)
        self.feature = feature
        self.line = line
        self.column = column
        self.details = details


def _make_unsupported(
    lark_error: UnexpectedInput,
    txt: str | None,
    message: str,
    feature: str,
    line_idx: int | None,
    col_idx: int | None,
) -> UnsupportedFeatureError:
    """Helper to build an UnsupportedFeatureError with context details."""
    details = (
        lark_error.get_context(txt)
        if hasattr(lark_error, "get_context") and txt is not None
        else str(lark_error)
    )
    return UnsupportedFeatureError(
        message,
        feature=feature,
        line=line_idx,
        column=col_idx,
        details=details,
    )


def detect_unsupported_dynare_feature(
    lark_error: UnexpectedInput, txt: str | None = None
) -> UnsupportedFeatureError | None:
    """Inspect a Lark parse error and return a descriptive UnsupportedFeatureError if the
    failing syntax corresponds to a known Dynare feature not yet implemented in DynoModel.

    Detection runs in two passes:
    1. **Narrow window** (±2 lines around the error): fast-path for errors where the
       unsupported construct is immediately adjacent to the token that confused the parser.
    2. **Full-file scan**: fallback for cases where the grammar chokes on an early token
       (e.g. ``periods`` inside a ``shocks`` block) while the actual unsupported keyword
       only appears further down in the file.

    Returns ``None`` when no known unsupported pattern is found, which causes the caller
    to re-raise the raw :class:`LARKParserError`.
    """
    if txt is None:
        return None

    line_idx = getattr(lark_error, "line", None)
    col_idx = getattr(lark_error, "column", None)
    lines = txt.splitlines()

    target_line = (
        lines[line_idx - 1] if line_idx and 1 <= line_idx <= len(lines) else ""
    )
    start_w = max(0, (line_idx - 3) if line_idx else 0)
    end_w = min(len(lines), (line_idx + 2) if line_idx else len(lines))
    window_txt = "\n".join(lines[start_w:end_w])

    def _make(message: str, feature: str) -> UnsupportedFeatureError:
        return _make_unsupported(lark_error, txt, message, feature, line_idx, col_idx)

    # ------------------------------------------------------------------
    # Pass 1: narrow-window checks (target_line + ±2 lines)
    # ------------------------------------------------------------------

    # 1. Macroprocessor
    if re.search(r"@#\w+", target_line) or re.search(r"@#\w+", window_txt):
        match = re.search(r"@#\w+", target_line) or re.search(r"@#\w+", window_txt)
        keyword = match.group(0) if match else "@#"
        return _make(
            f"Dynare macroprocessor directives ({keyword!r}) are not supported natively by DynoModel. "
            f"Use DynareModel or preprocess the file with Dynare.",
            "macroprocessor",
        )

    # 2. Heterogeneous agent blocks
    if (
        re.search(r"\bvar\s*\(\s*households\b", target_line, re.IGNORECASE)
        or "households" in target_line
    ):
        return _make(
            "Heterogeneous agent block syntax ('var(households, ...)') is not supported by DynoModel. Use DynareModel.",
            "heterogeneity",
        )

    # 3. Estimation blocks
    for kw in [
        "estimated_params_init",
        "estimated_params_bounds",
        "estimated_params",
        "estimation",
        "varobs",
        "observation_trends",
        "datatomfile",
    ]:
        if re.search(rf"\b{kw}\b", target_line, re.IGNORECASE) or re.search(
            rf"\b{kw}\b", window_txt, re.IGNORECASE
        ):
            return _make(
                f"Dynare estimation construct '{kw}' is not supported by DynoModel. Use DynareModel.",
                "estimation",
            )

    # 4. Occbin blocks and regime-switching tags
    if re.search(r"\b(bind|relax)\s*=", target_line) or re.search(
        r"\b(bind|relax)\s*=", window_txt
    ):
        return _make(
            "Dynare Occbin regime-switching equation tags ('bind=...', 'relax=...') are not supported by DynoModel.",
            "occbin",
        )
    for kw in ["occbin_constraints", "occbin_setup", "occbin_solver"]:
        if re.search(rf"\b{kw}\b", target_line, re.IGNORECASE) or re.search(
            rf"\b{kw}\b", window_txt, re.IGNORECASE
        ):
            return _make(
                f"Dynare Occbin construct '{kw}' is not supported by DynoModel.",
                "occbin",
            )

    # 5. Optimal policy
    for kw in [
        "ramsey_model",
        "ramsey_policy",
        "ramsey_constraints",
        "planner_objective",
        "osr_params",
        "osr",
    ]:
        if re.search(rf"\b{kw}\b", target_line, re.IGNORECASE) or re.search(
            rf"\b{kw}\b", window_txt, re.IGNORECASE
        ):
            return _make(
                f"Dynare optimal policy construct '{kw}' is not supported by DynoModel.",
                "optimal_policy",
            )

    # 6. Reporting blocks
    for kw in ["send_irfs_to_workspace", "reporting"]:
        if re.search(rf"\b{kw}\b", target_line, re.IGNORECASE) or re.search(
            rf"\b{kw}\b", window_txt, re.IGNORECASE
        ):
            return _make(
                f"Dynare reporting construct '{kw}' is not supported by DynoModel.",
                "reporting",
            )

    # 7. Perfect foresight with expectation errors (narrow window).
    #    Plain ``perfect_foresight_setup``/``perfect_foresight_solver`` and
    #    ``simul`` are supported; the ``learnt_in`` variants are not.
    for kw in _EXPECTATION_ERRORS_KEYWORDS:
        if re.search(rf"\b{kw}", target_line, re.IGNORECASE) or re.search(
            rf"\b{kw}", window_txt, re.IGNORECASE
        ):
            return _make(
                f"Dynare perfect foresight with expectation errors ('{kw}') is not supported in .mod files by DynoModel.",
                "perfect_foresight",
            )

    # 8. Non-stationary growth factor
    if "growth_factor" in target_line or "growth_factor" in window_txt:
        return _make(
            "Dynare non-stationary model option 'growth_factor' is not supported by DynoModel.",
            "growth_factor",
        )

    # 9. PAC model
    if "pac_model" in target_line or "pac_model" in window_txt:
        return _make(
            "Dynare PAC model construct ('pac_model') is not supported by DynoModel.",
            "pac_model",
        )

    # ------------------------------------------------------------------
    # Pass 2: full-file scan fallback
    # When the narrow window does not match, scan the entire source text
    # for patterns that indicate a known unsupported feature.  This handles
    # cases where the parser chokes on an intermediate token while the
    # actual unsupported keyword is elsewhere in the file.
    # ------------------------------------------------------------------

    # 10. Full-file scan for perfect_foresight_with_expectation_errors
    #     (complements check #7 above for files where the error token is far
    #     from the command keyword).
    for kw in _EXPECTATION_ERRORS_KEYWORDS:
        if re.search(rf"\b{kw}", txt, re.IGNORECASE):
            return _make(
                f"Dynare perfect foresight with expectation errors ('{kw}') is not supported in .mod files by DynoModel.",
                "perfect_foresight",
            )

    # 11. External function calls in ``steady_state_model`` block.
    #     The grammar's FUNCTION terminal only allows a fixed set of built-in
    #     functions.  User-defined helper functions (e.g. ``my_ss_helper(alpha, ...)``)
    #     cause an ``UnexpectedToken`` on the identifier that follows the opening
    #     parenthesis of the call.
    if _has_external_steady_state_function(txt):
        return _make(
            "External function calls in 'steady_state_model' block (e.g. 'func_name(...)') "
            "are not supported by DynoModel. Provide explicit steady-state expressions or use DynareModel.",
            "external_steady_state_function",
        )

    # 12. MATLAB/Octave reporting API and macroprocessor after the model block.
    #     Files like example1_reporting.mod contain MATLAB assignment statements
    #     (``shocke = dseries(); r = report();``) and ``@#define``/``@#for``
    #     macroprocessor directives after ``stoch_simul;``.  The macroprocessor
    #     check in pass 1 may miss these when the error is reported before the
    #     ``@#`` line.  Scan the whole file here as a safety net.
    if re.search(r"@#\w+", txt):
        match = re.search(r"@#\w+", txt)
        keyword = match.group(0) if match else "@#"
        return _make(
            f"Dynare macroprocessor directives ({keyword!r}) are not supported natively by DynoModel. "
            f"Use DynareModel or preprocess the file with Dynare.",
            "macroprocessor",
        )
    if _has_matlab_reporting_api(txt):
        return _make(
            "MATLAB/Octave reporting API ('dseries', 'report()', etc.) after the model block "
            "is not supported by DynoModel.",
            "matlab_reporting_api",
        )

    return None


# ---------------------------------------------------------------------------
# Helper predicates for full-file pattern detection
# ---------------------------------------------------------------------------

# Keywords of Dynare's perfect foresight with expectation errors
# (``perfect_foresight_with_expectation_errors_setup/solver`` and the
# ``learnt_in`` option of ``shocks``/``endval``), matched as prefixes.
_EXPECTATION_ERRORS_KEYWORDS = (
    "perfect_foresight_with_expectation_errors",
    "learnt_in",
)

_KNOWN_FUNCTIONS = frozenset(
    ["sin", "cos", "exp", "log", "sqrt", "abs", "steady_state"]
)

# Regex that matches a ``steady_state_model`` block and captures its body.
_SS_MODEL_BLOCK_RE = re.compile(
    r"\bsteady_state_model\s*;(.*?)\bend\s*;",
    re.DOTALL | re.IGNORECASE,
)

# Regex for a function call: identifier immediately followed by ``(``.
# A ``NAME(`` pattern where NAME is not a known built-in indicates an external function.
_FUNC_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# Identifiers that are valid in steady_state_model context but look like calls.
_SS_MODEL_CALL_KEYWORDS = frozenset(["end"])

# Patterns typical of MATLAB reporting API used in some Dynare example files.
_MATLAB_API_PATTERNS = [
    re.compile(r"\bdseries\s*\("),
    re.compile(r"\breport\s*\(\s*\)"),
    re.compile(r"\baddPage\s*\("),
    re.compile(r"\baddSection\s*\("),
    re.compile(r"\baddGraph\s*\("),
    re.compile(r"\baddSeries\s*\("),
    re.compile(r"\baddTable\s*\("),
    re.compile(r"\baddVspace\s*\("),
]


def _strip_comments(txt: str) -> str:
    """Strip Dynare ``//``, ``%``, ``#`` line comments and ``/* ... */`` block comments."""
    # Block comments first
    txt = re.sub(r"/\*.*?\*/", " ", txt, flags=re.DOTALL)
    # Line comments
    txt = re.sub(r"(//|%|#)[^\n]*", "", txt)
    return txt


def _has_external_steady_state_function(txt: str) -> bool:
    """Return True if the ``steady_state_model`` block calls a function that is not
    in the set of functions known to the Dyno grammar.
    """
    stripped = _strip_comments(txt)
    for m in _SS_MODEL_BLOCK_RE.finditer(stripped):
        body = m.group(1)
        for call_m in _FUNC_CALL_RE.finditer(body):
            name = call_m.group(1).lower()
            if name not in _KNOWN_FUNCTIONS and name not in _SS_MODEL_CALL_KEYWORDS:
                return True
    return False


def _has_matlab_reporting_api(txt: str) -> bool:
    """Return True if the file contains MATLAB/Octave reporting API calls."""
    stripped = _strip_comments(txt)
    return any(p.search(stripped) for p in _MATLAB_API_PATTERNS)


def detect_unsupported_features_preparsed(txt: str) -> UnsupportedFeatureError | None:
    """Scan *txt* for known unsupported Dynare features **before** attempting a
    Lark parse.  This covers cases where the grammar may successfully accept the
    source (or where the first confusing token is far from the actual problem),
    but the resulting model would be semantically invalid or raise a cryptic
    error later.

    Returns an :class:`UnsupportedFeatureError` if a pattern is found, or
    ``None`` if the source looks parseable.

    Checks (in priority order):
    1. Macroprocessor directives (``@#define``, ``@#for``, …)
    2. MATLAB/Octave reporting API (``dseries(``, ``report()``, …)
    3. External function calls in ``steady_state_model`` block
    """

    def _make_preparsed(message: str, feature: str) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(message, feature=feature, details=None)

    # 1. Macroprocessor directives (full file scan)
    match = re.search(r"@#\w+", txt)
    if match:
        keyword = match.group(0)
        return _make_preparsed(
            f"Dynare macroprocessor directives ({keyword!r}) are not supported natively by DynoModel. "
            f"Use DynareModel or preprocess the file with Dynare.",
            "macroprocessor",
        )

    # 2. MATLAB/Octave reporting API
    if _has_matlab_reporting_api(txt):
        return _make_preparsed(
            "MATLAB/Octave reporting API ('dseries', 'report()', etc.) after the model block "
            "is not supported by DynoModel.",
            "matlab_reporting_api",
        )

    # 3. External function calls in steady_state_model block
    if _has_external_steady_state_function(txt):
        return _make_preparsed(
            "External function calls in 'steady_state_model' block (e.g. 'func_name(...)') "
            "are not supported by DynoModel. Provide explicit steady-state expressions or use DynareModel.",
            "external_steady_state_function",
        )

    return None
