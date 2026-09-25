"""Catalog of misspecified, incomplete, and error-prone Dyno models.

This module houses a systematic database of user errors, syntax traps,
and structural incompleteness encountered when writing macroeconomic models from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DiagnosticCase:
    id: str
    category: str
    title: str
    intent: str
    source: str
    expected_stage: str  # "import", "check", "steady", "solve"
    current_status: str  # "parser_error", "undefined_error", "steady_state_error", "internal_crash", "silent_nan", "silent_underdetermined"
    current_error_type: str
    current_message_snippet: str
    ideal_diagnostic: str
    suggested_fix: str


ALL_CASES: List[DiagnosticCase] = [
    # ── Category A: Lexical & Syntax Traps ────────────────────────────────────
    DiagnosticCase(
        id="SYN-001",
        category="syntax",
        title="Top-level equality used instead of assignment operator",
        intent="User intends to define parameter alpha = 0.35 using Python/Julia syntax.",
        source="""
alpha = 0.35
k[t] = 0.9 * k[t-1]
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="silent_nan",
        current_error_type="UndefinedSymbolWarning",
        current_message_snippet="alpha is treated as an equation rather than an assignment",
        ideal_diagnostic="Top-level equality detected for 'alpha = 0.35'. In Dyno, '=' defines dynamic equations. Did you mean 'alpha <- 0.35'?",
        suggested_fix="alpha <- 0.35",
    ),
    DiagnosticCase(
        id="SYN-002",
        category="syntax",
        title="Parentheses used for time indexing instead of brackets",
        intent="User writes time subscript as c(t) instead of c[t].",
        source="""
alpha <- 0.35
k(t) = 0.9 * k(t-1)
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="internal_crash",
        current_error_type="ValueError",
        current_message_snippet="Undefined function: k",
        ideal_diagnostic="Syntax error: 'k(t)' looks like a time index or lag. Time indices in Dyno must use square brackets: 'k[t]'.",
        suggested_fix="k[t] = 0.9 * k[t-1]",
    ),
    DiagnosticCase(
        id="SYN-003",
        category="syntax",
        title="Trailing semicolon from MATLAB/Dynare convention",
        intent="User ends parameter statement with semicolon.",
        source="""
alpha <- 0.35;
k[t] = 0.9 * k[t-1]
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="parser_error",
        current_error_type="LARKParserError",
        current_message_snippet="Unexpected token `;`",
        ideal_diagnostic="Semicolons are not required or permitted as statement terminators in Dyno. Remove trailing ';'.",
        suggested_fix="alpha <- 0.35",
    ),
    DiagnosticCase(
        id="SYN-004",
        category="syntax",
        title="Dynare variable declaration block in .dyno file",
        intent="User attempts to declare variables using Dynare syntax.",
        source="""
var k, c;
parameters alpha;
alpha <- 0.35
k[t] = 0.9 * k[t-1]
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="parser_error",
        current_error_type="LARKParserError",
        current_message_snippet="Unexpected token `k, c;`",
        ideal_diagnostic="Dyno does not require variable declarations. Variables are inferred automatically from time indices [t] and steady-state declarations [~].",
        suggested_fix="Remove 'var' and 'parameters' declarations.",
    ),
    DiagnosticCase(
        id="SYN-005",
        category="syntax",
        title="Dynare implicit lag notation k(-1)",
        intent="User references lagged variable without explicit 't'.",
        source="""
alpha <- 0.35
k[t] = 0.9 * k(-1)
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="internal_crash",
        current_error_type="ValueError",
        current_message_snippet="Undefined function: k",
        ideal_diagnostic="Dynare notation 'k(-1)' is not recognized in .dyno files. Use explicit time indexing: 'k[t-1]'.",
        suggested_fix="k[t] = 0.9 * k[t-1]",
    ),
    DiagnosticCase(
        id="SYN-006",
        category="syntax",
        title="Double equals comparison operator in equation",
        intent="User uses '==' instead of '=' for dynamic equation.",
        source="""
alpha <- 0.35
k[t] == 0.9 * k[t-1]
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="parser_error",
        current_error_type="LARKParserError",
        current_message_snippet="Unexpected token `==`",
        ideal_diagnostic="Dynamic equations in Dyno use a single '=' operator (e.g. 'lhs = rhs').",
        suggested_fix="k[t] = 0.9 * k[t-1]",
    ),
    DiagnosticCase(
        id="SYN-007",
        category="syntax",
        title="Reversed assignment arrow ->",
        intent="User writes assignment with rightward arrow.",
        source="""
0.35 -> alpha
k[t] = 0.9 * k[t-1]
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="parser_error",
        current_error_type="LARKParserError",
        current_message_snippet="Unexpected characters at (2, 6)",
        ideal_diagnostic="Assignment in Dyno must be leftward: 'alpha <- 0.35' or 'alpha := 0.35'.",
        suggested_fix="alpha <- 0.35",
    ),
    DiagnosticCase(
        id="SYN-008",
        category="syntax",
        title="Unsupported mathematical function",
        intent="User calls an unsupported function such as sigmoid.",
        source="""
alpha <- 0.35
k[t] = sigmoid(k[t-1])
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="internal_crash",
        current_error_type="ValueError",
        current_message_snippet="Undefined function: sigmoid",
        ideal_diagnostic="Function 'sigmoid' is not recognized. Supported functions include: exp, log, sqrt, abs, sin, cos, tan.",
        suggested_fix="Define sigmoid analytically using exp(x) / (1 + exp(x)).",
    ),
    # ── Category B: Symbol Resolution & Calibration ───────────────────────────
    DiagnosticCase(
        id="SYM-001",
        category="symbol_resolution",
        title="Typo in parameter name in equation definition",
        intent="User meant 'alpha' but typed 'alhpa'.",
        source="""
alpha <- 0.35
k[t] = alhpa * k[t-1]
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="silent_nan",
        current_error_type="UndefinedSymbolWarning",
        current_message_snippet="Undefined parameter(s) used in equations: alhpa",
        ideal_diagnostic="Parameter 'alhpa' used in equation is never defined. Did you mean 'alpha'?",
        suggested_fix="k[t] = alpha * k[t-1]",
    ),
    DiagnosticCase(
        id="SYM-002",
        category="symbol_resolution",
        title="Variable in equation without time index",
        intent="User omits '[t]' when referring to contemporaneous variable c.",
        source="""
alpha <- 0.35
k[t] = alpha * k[t-1] + c
c[t] = 0.5 * k[t]
k[~] <- 1.0
c[~] <- 0.5
""",
        expected_stage="import",
        current_status="silent_nan",
        current_error_type="UndefinedSymbolWarning",
        current_message_snippet="Undefined parameter(s) used in equations: c",
        ideal_diagnostic="Symbol 'c' appears in equation without a time index [t], but 'c' is declared as a variable. Did you mean 'c[t]'?",
        suggested_fix="k[t] = alpha * k[t-1] + c[t]",
    ),
    DiagnosticCase(
        id="SYM-003",
        category="symbol_resolution",
        title="Out-of-order parameter calibration",
        intent="User references parameter 'b' before its definition.",
        source="""
a <- b * 2.0
b <- 1.0
k[t] = a * k[t-1]
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="silent_nan",
        current_error_type="UndefinedSymbolWarning",
        current_message_snippet="Undefined value: b",
        ideal_diagnostic="Parameter 'b' was referenced on line 2 before its assignment on line 3.",
        suggested_fix="Define 'b <- 1.0' before 'a <- b * 2.0'.",
    ),
    DiagnosticCase(
        id="SYM-004",
        category="symbol_resolution",
        title="Circular parameter calibration",
        intent="User mutually defines parameters without a base value.",
        source="""
a <- b + 1.0
b <- a + 1.0
k[t] = a * k[t-1]
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="silent_nan",
        current_error_type="UndefinedSymbolWarning",
        current_message_snippet="Undefined value: b",
        ideal_diagnostic="Circular dependency detected in parameter calibration between 'a' and 'b'.",
        suggested_fix="Provide an exogenous numerical value for at least one parameter.",
    ),
    DiagnosticCase(
        id="SYM-005",
        category="symbol_resolution",
        title="Parameter self-reference without initial calibration",
        intent="User writes recursive parameter expression a <- a + 1.",
        source="""
a <- a + 1.0
k[t] = a * k[t-1]
k[~] <- 1.0
""",
        expected_stage="import",
        current_status="silent_nan",
        current_error_type="UndefinedSymbolWarning",
        current_message_snippet="Undefined value: a",
        ideal_diagnostic="Parameter 'a' references itself before having an established value.",
        suggested_fix="Initialize 'a' with a literal number first.",
    ),
    DiagnosticCase(
        id="SYM-006",
        category="symbol_resolution",
        title="Typo in variable name creating phantom variable",
        intent="User meant 'k[t]' but typed 'k_cap[t]'.",
        source="""
alpha <- 0.35
k_cap[t] = alpha * k[t-1]
k[~] <- 1.0
""",
        expected_stage="check",
        current_status="silent_underdetermined",
        current_error_type="UndefinedSymbolError",
        current_message_snippet="variables without steady state: k_cap",
        ideal_diagnostic="Variable 'k_cap[t]' appears in equations but has no steady state [~] or transition law.",
        suggested_fix="k[t] = alpha * k[t-1]",
    ),
    # ── Category C: Steady-State Incompleteness & Misconceptions ─────────────
    DiagnosticCase(
        id="SS-001",
        category="steady_state",
        title="Missing steady-state declaration for endogenous variable",
        intent="User writes dynamic equations but completely omits k[~].",
        source="""
alpha <- 0.35
k[t] = alpha * k[t-1]
""",
        expected_stage="check",
        current_status="undefined_error",
        current_error_type="UndefinedSymbolError",
        current_message_snippet="variables without steady state: k",
        ideal_diagnostic="Variable 'k' has no steady-state declaration. Provide 'k[~] <- ...' or use model.steady() to solve numerically.",
        suggested_fix="k[~] <- 0.0",
    ),
    DiagnosticCase(
        id="SS-002",
        category="steady_state",
        title="Typo in steady-state declaration name k_ss",
        intent="User declares steady state as k_ss <- 1.0 instead of k[~] <- 1.0.",
        source="""
alpha <- 0.35
k[t] = alpha * k[t-1]
k_ss <- 1.0
""",
        expected_stage="check",
        current_status="undefined_error",
        current_error_type="UndefinedSymbolError",
        current_message_snippet="variables without steady state: k",
        ideal_diagnostic="Unused parameter 'k_ss' detected. Did you mean to declare the steady state as 'k[~] <- 1.0'?",
        suggested_fix="k[~] <- 1.0",
    ),
    DiagnosticCase(
        id="SS-003",
        category="steady_state",
        title="Inconsistent steady-state value",
        intent="User supplies an incorrect steady-state guess (residual is non-zero).",
        source="""
alpha <- 0.35
k[t] = alpha * k[t-1]
k[~] <- 10.0
""",
        expected_stage="check",
        current_status="steady_state_error",
        current_error_type="SteadyStateError",
        current_message_snippet="Steady state values don't satisfy model equations",
        ideal_diagnostic="Equation 1 has non-zero steady-state residual (residual = -6.5). The steady-state condition is k = 0, but k[~] was set to 10.0.",
        suggested_fix="k[~] <- 0.0",
    ),
    DiagnosticCase(
        id="SS-004",
        category="steady_state",
        title="Steady-state declaration causes division by zero",
        intent="User sets c[~] = 0 in a model with utility 1/c.",
        source="""
c[~] <- 0.0
1 / c[t] = 1.0
""",
        expected_stage="check",
        current_status="internal_crash",
        current_error_type="ZeroDivisionError",
        current_message_snippet="division by zero",
        ideal_diagnostic="Division by zero encountered in equation 1 when evaluated at steady state (c[~] = 0.0).",
        suggested_fix="c[~] <- 1.0",
    ),
    DiagnosticCase(
        id="SS-005",
        category="steady_state",
        title="Steady-state evaluation causes negative log domain error",
        intent="User sets steady state of z to -1 where equation has log(z).",
        source="""
z[~] <- -1.0
y[t] = log(z[t])
""",
        expected_stage="check",
        current_status="internal_crash",
        current_error_type="ValueError",
        current_message_snippet="math domain error",
        ideal_diagnostic="Domain error: log(z[t]) evaluated with non-positive steady state z[~] = -1.0.",
        suggested_fix="z[~] <- 1.0",
    ),
    DiagnosticCase(
        id="SS-006",
        category="steady_state",
        title="Forward reference across steady-state declarations",
        intent="User defines k[~] referencing y[~] before y[~] is defined.",
        source="""
k[~] <- y[~] * 0.5
y[~] <- 2.0
k[t] = 0.5 * k[t-1] + 0.5 * y[t]
y[t] = 2.0
""",
        expected_stage="check",
        current_status="undefined_error",
        current_error_type="UndefinedSymbolError",
        current_message_snippet="variables without steady state: k",
        ideal_diagnostic="Steady state 'y[~]' was referenced in 'k[~]' before 'y[~]' was assigned.",
        suggested_fix="Define 'y[~] <- 2.0' before 'k[~] <- y[~] * 0.5'.",
    ),
    # ── Category D: Exogenous Shocks & Stochastic Processes ───────────────────
    DiagnosticCase(
        id="SHK-001",
        category="shocks",
        title="Shock variable in equations without distribution declaration",
        intent="User uses e[t] in equation but never writes e[t] <- N(0, sigma^2).",
        source="""
rho <- 0.9
z[t] = rho * z[t-1] + e[t]
z[~] <- 0.0
""",
        expected_stage="solve",
        current_status="silent_underdetermined",
        current_error_type="SystemStructureError",
        current_message_snippet="Model has 1 equation(s) but 2 endogenous variable(s): ['z', 'e']",
        ideal_diagnostic="Variable 'e' appears to be an exogenous shock but has no distribution. Declare 'e[t] <- N(0, sigma^2)'.",
        suggested_fix="e[t] <- N(0, 0.01^2)",
    ),
    DiagnosticCase(
        id="SHK-002",
        category="shocks",
        title="Shock distribution too many arguments",
        intent="User calls N(0, 0.01, 1) with three arguments instead of std or mean and std.",
        source="""
rho <- 0.9
z[t] = rho * z[t-1] + e[t]
z[~] <- 0.0
e[t] <- N(0, 0.01, 1)
""",
        expected_stage="import",
        current_status="internal_crash",
        current_error_type="TypeError",
        current_message_snippet="takes 1 or 2 arguments",
        ideal_diagnostic="N(...) requires 1 or 2 arguments: N(std) or N(mean, std). E.g. 'e[t] <- N(0.01)'.",
        suggested_fix="e[t] <- N(0.01)",
    ),
    DiagnosticCase(
        id="SHK-003",
        category="shocks",
        title="Shock assigned as parameter without [t]",
        intent="User writes e <- N(0, 0.01) instead of e[t] <- N(0, 0.01).",
        source="""
rho <- 0.9
z[t] = rho * z[t-1] + e[t]
z[~] <- 0.0
e <- N(0, 0.01)
""",
        expected_stage="solve",
        current_status="silent_underdetermined",
        current_error_type="SystemStructureError",
        current_message_snippet="Model has 1 equation(s) but 2 endogenous variable(s)",
        ideal_diagnostic="Shock process assigned to constant 'e' rather than time series 'e[t]'. Use 'e[t] <- N(...)'.",
        suggested_fix="e[t] <- N(0, 0.01)",
    ),
    DiagnosticCase(
        id="SHK-004",
        category="shocks",
        title="Shock distribution function name typo Normal()",
        intent="User writes Normal(0, 0.01) instead of N(0, 0.01).",
        source="""
rho <- 0.9
z[t] = rho * z[t-1] + e[t]
z[~] <- 0.0
e[t] <- Normal(0, 0.01)
""",
        expected_stage="import",
        current_status="internal_crash",
        current_error_type="ValueError",
        current_message_snippet="Undefined function: Normal",
        ideal_diagnostic="Distribution 'Normal' is not recognized. Use 'N(std)' or 'N(mean, std)' for Gaussian shocks.",
        suggested_fix="e[t] <- N(0.01)",
    ),
    # ── Category E: System Structural Completeness ────────────────────────────
    DiagnosticCase(
        id="SYS-001",
        category="system_structure",
        title="Underdetermined model (fewer equations than endogenous variables)",
        intent="User forgets one equation in a 2-variable model.",
        source="""
alpha <- 0.35
k[t] = alpha * k[t-1] + c[t] + e[t]
k[~] <- 1.0
c[~] <- 0.5
e[t] <- N(0, 0.01)
""",
        expected_stage="solve",
        current_status="internal_crash",
        current_error_type="SystemStructureError",
        current_message_snippet="Model has 1 equation(s) but 2 endogenous variable(s): ['k', 'c']",
        ideal_diagnostic="Model is underdetermined: 1 equation provided for 2 endogenous variables ('k', 'c'). Provide an equation for 'c[t]'.",
        suggested_fix="Add equation for c[t].",
    ),
    DiagnosticCase(
        id="SYS-002",
        category="system_structure",
        title="Overdetermined model (more equations than endogenous variables)",
        intent="User accidentally duplicates an equation.",
        source="""
alpha <- 0.35
k[t] = alpha * k[t-1] + e[t]
k[t] = 0.8 * k[t-1] + e[t]
k[~] <- 0.0
e[t] <- N(0, 0.01)
""",
        expected_stage="solve",
        current_status="internal_crash",
        current_error_type="SystemStructureError",
        current_message_snippet="Model has 2 equation(s) but 1 endogenous variable(s): ['k']",
        ideal_diagnostic="Model is overdetermined: 2 equations provided for 1 endogenous variable ('k'). Remove redundant equation.",
        suggested_fix="Remove duplicated or conflicting equation.",
    ),
    DiagnosticCase(
        id="SYS-003",
        category="system_structure",
        title="Higher-order lag k[t-2] without auxiliary state",
        intent="User models AR(2) process directly with lag 2.",
        source="""
alpha1 <- 0.5
alpha2 <- 0.2
k[t] = alpha1 * k[t-1] + alpha2 * k[t-2] + e[t]
k[~] <- 0.0
e[t] <- N(0, 0.01)
""",
        expected_stage="import",
        current_status="silent_nan",
        current_error_type="UndefinedSymbolWarning",
        current_message_snippet="shift -2",
        ideal_diagnostic="Higher-order lag 'k[t-2]' detected. Dyno first-order solvers support leads and lags within [-1, 1]. Introduce an auxiliary variable 'k_lag[t] = k[t-1]'.",
        suggested_fix="k_lag[t] = k[t-1]\nk[t] = alpha1 * k[t-1] + alpha2 * k_lag[t-1]",
    ),
    DiagnosticCase(
        id="SYS-004",
        category="system_structure",
        title="Higher-order lead c[t+2] without auxiliary variable",
        intent="User models two-period ahead forward expectation.",
        source="""
beta <- 0.95
c[t] = beta * c[t+2] + e[t]
c[~] <- 0.0
e[t] <- N(0, 0.01)
""",
        expected_stage="import",
        current_status="silent_nan",
        current_error_type="UndefinedSymbolWarning",
        current_message_snippet="shift 2",
        ideal_diagnostic="Higher-order lead 'c[t+2]' detected. Dyno solvers support leads within [-1, 1]. Define an auxiliary expected value variable.",
        suggested_fix="c_lead[t] = c[t+1]\nc[t] = beta * c_lead[t+1] + e[t]",
    ),
    DiagnosticCase(
        id="SYS-005",
        category="system_structure",
        title="Empty model file (comments only)",
        intent="User loads an empty model file.",
        source="""
# Only comments in this model file
# Nothing else
""",
        expected_stage="import",
        current_status="parser_error",
        current_error_type="LARKParserError",
        current_message_snippet="Unexpected end of file",
        ideal_diagnostic="Model description is empty. Provide at least one equation and parameter definition.",
        suggested_fix="Add equations and parameters.",
    ),
    # ── Category F: Solvability & Numerical Health ────────────────────────────
    DiagnosticCase(
        id="SOL-001",
        category="solvability",
        title="Blanchard-Kahn condition violation (explosive root with backward variable)",
        intent="User specifies an unstable backward-looking autoregressive system.",
        source="""
k[t] = 2.0 * k[t-1] + e[t]
k[~] <- 0.0
e[t] <- N(0, 0.01)
""",
        expected_stage="solve",
        current_status="bk_error",
        current_error_type="BlanchardKahnError",
        current_message_snippet="Eigenvalue condition not satisfied",
        ideal_diagnostic="Blanchard-Kahn condition failed: explosive eigenvalue 2.0 associated with predetermined backward-looking variable 'k'. No stable path exists.",
        suggested_fix="Ensure coefficient on k[t-1] has absolute value < 1.",
    ),
    DiagnosticCase(
        id="SOL-002",
        category="solvability",
        title="Singular Jacobian due to collinear equations",
        intent="User writes two linearly dependent equations.",
        source="""
k[t] = 0.5 * k[t-1] + c[t]
2 * k[t] = k[t-1] + 2 * c[t]
k[~] <- 0.0
c[~] <- 0.0
""",
        expected_stage="solve",
        current_status="internal_crash",
        current_error_type="LinAlgError",
        current_message_snippet="singular matrix",
        ideal_diagnostic="Jacobian matrix is singular. Equation 2 is a scalar multiple of Equation 1. Replace with an independent equilibrium condition.",
        suggested_fix="Replace redundant equation with an independent relation.",
    ),
]


def get_case(case_id: str) -> DiagnosticCase:
    for case in ALL_CASES:
        if case.id == case_id:
            return case
    raise KeyError(f"Case with ID '{case_id}' not found.")


def get_cases_by_category(category: str) -> List[DiagnosticCase]:
    return [c for c in ALL_CASES if c.category == category]
