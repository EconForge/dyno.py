"""
Function generator for structured equation groups.

Given a set of equations that conform to a recipe, this module generates
callable Python functions that evaluate them.

For **transition** equations (recursive/DAG), the generated function evaluates
equations in topological order and returns the new state values.

For **arbitrage** equations (residual form), the generated function returns
residuals that should equal zero at a solution.

**Definitions** (auxiliary variables) can be attached so that they are
automatically evaluated at each needed time shift before the main equations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from lark.tree import Tree

from .analyze import EquationsEvaluator
from .recipe import (
    EquationGroupSpec,
    ConformityResult,
    check_equation_group,
    extract_lhs_variable,
    extract_variables_from_equation,
)

# ── Definitions Block ────────────────────────────────────────────────────────


@dataclass
class DefinitionsBlock:
    """A set of definition equations (auxiliary variables) to be inlined.

    Definitions are a recursive block: each equation defines one auxiliary
    variable as a function of exogenous, states, controls, and previously
    defined auxiliaries.

    Attributes:
        equations: Ordered equation ASTs (already sorted topologically).
        target_group: The variable group being defined (e.g. ``"auxiliaries"``).
        variable_names: Ordered list of the defined variable names
            (matching ``equations`` order after topological sort).
    """

    equations: list[Tree]
    target_group: str
    variable_names: list[str]


def build_definitions_block(
    equations: Sequence[Tree],
    variables: dict[str, list[str]],
    spec: EquationGroupSpec,
    constants: Sequence[str] | None = None,
) -> tuple[ConformityResult, DefinitionsBlock | None]:
    """Check conformity of definition equations and build a :class:`DefinitionsBlock`.

    Parameters:
        equations: Definition equation ASTs.
        variables: Variable classification.
        spec: The definitions equation group specification.
        constants: Parameter/constant names.

    Returns:
        A tuple ``(conformity_result, definitions_block_or_none)``.
    """
    result = check_equation_group(equations, variables, spec, constants=constants)
    if not result.ok:
        return result, None

    # Order equations topologically
    if result.dag_order is not None:
        eq_order_map: dict[str, int] = {}
        for i, eq in enumerate(equations):
            lhs = extract_lhs_variable(eq)
            if lhs is not None:
                eq_order_map[lhs] = i
        ordered_eqs = [equations[eq_order_map[name]] for name in result.dag_order]
        ordered_names = list(result.dag_order)
    else:
        ordered_eqs = list(equations)
        ordered_names = []
        for eq in equations:
            lhs = extract_lhs_variable(eq)
            if lhs is not None:
                ordered_names.append(lhs)

    block = DefinitionsBlock(
        equations=ordered_eqs,
        target_group=spec.target or "auxiliaries",
        variable_names=ordered_names,
    )
    return result, block


# ── Function Generation ──────────────────────────────────────────────────────


def generate_equation_function(
    equations: Sequence[Tree],
    variables: dict[str, list[str]],
    spec: EquationGroupSpec,
    constants: dict[str, float] | None = None,
    conformity: ConformityResult | None = None,
    definitions: DefinitionsBlock | None = None,
) -> Callable[..., np.ndarray]:
    """Generate a callable function from a conforming equation group.

    Parameters:
        equations: List of equation ASTs.
        variables: Mapping from variable group names to lists of variable names.
        spec: The equation group specification.
        constants: Dictionary of parameter name → value.
        conformity: Pre-computed conformity result. If ``None``, conformity
            is checked and a ``ValueError`` is raised on failure.
        definitions: Optional definitions block. When provided, auxiliary
            variables are automatically computed at each needed time shift
            before evaluating the main equations.

    Returns:
        A callable function. Arguments are numpy arrays ordered by the
        ``allowed`` variable specs (excluding parameters and auxiliaries),
        grouped by ``(group, shift)`` pair.

    Raises:
        ValueError: If the equations do not conform to the spec.
    """
    if constants is None:
        constants = {}

    # Verify conformity if not provided
    if conformity is None:
        result = check_equation_group(
            equations, variables, spec, constants=list(constants.keys())
        )
        if not result.ok:
            raise ValueError(f"Equations do not conform to spec:\n{result}")
        conformity = result

    # Determine argument groups (unique (group, shift) pairs from allowed spec)
    # Exclude parameters AND auxiliaries (those are computed from definitions)
    excluded_groups = {"parameters"}
    if definitions is not None:
        excluded_groups.add(definitions.target_group)

    arg_groups: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for vs in spec.allowed:
        key = (vs.group, vs.shift)
        if key not in seen and vs.group not in excluded_groups:
            seen.add(key)
            arg_groups.append(key)

    # Determine which shifts of auxiliaries are needed
    aux_shifts_needed: set[int] = set()
    if definitions is not None:
        aux_group = definitions.target_group
        for eq in equations:
            vars_in_eq = extract_variables_from_equation(eq)
            for vname, shifts in vars_in_eq.items():
                if vname in definitions.variable_names:
                    aux_shifts_needed.update(shifts)

    # Determine equation order (topological for recursive blocks)
    if conformity.dag_order is not None and spec.target is not None:
        eq_order_map: dict[str, int] = {}
        for i, eq in enumerate(equations):
            lhs = extract_lhs_variable(eq)
            if lhs is not None:
                eq_order_map[lhs] = i
        ordered_indices = [eq_order_map[name] for name in conformity.dag_order]
    else:
        ordered_indices = list(range(len(equations)))

    ordered_equations = [equations[i] for i in ordered_indices]

    # Build the function
    is_recursive = spec.recursive and spec.target is not None

    # Capture definitions in closure
    _definitions = definitions
    _aux_shifts = sorted(aux_shifts_needed)
    _constants = dict(constants)

    def _eval_function(*args: np.ndarray) -> np.ndarray:
        """Evaluate the equation group.

        Arguments correspond to the ``allowed`` variable groups (excluding
        parameters and auxiliaries), ordered as they appear in the spec.
        Each argument is a 1-D numpy array whose length matches the number
        of variables in that group.

        Returns:
            For transition (recursive): array of new state values.
            For arbitrage (residual): array of residuals.
        """
        if len(args) != len(arg_groups):
            raise TypeError(
                f"Expected {len(arg_groups)} arguments "
                f"({[f'{g}[t{s:+d}]' if s else f'{g}[t]' for g, s in arg_groups]}), "
                f"got {len(args)}"
            )

        # Build context
        ctx_variables: dict[str, dict[int, Any]] = {}
        ctx_constants: dict[str, Any] = dict(_constants)

        for (group, shift), values in zip(arg_groups, args):
            var_names = variables.get(group, [])
            arr = np.asarray(values)
            for j, vname in enumerate(var_names):
                if vname not in ctx_variables:
                    ctx_variables[vname] = {}
                ctx_variables[vname][shift] = (
                    float(arr[j]) if arr.ndim == 0 or len(arr.shape) == 1 else arr[j]
                )

        context = {
            "constants": ctx_constants,
            "variables": ctx_variables,
            "values": {},
            "processes": {},
            "steady_states": {},
        }

        # Evaluate definitions at each needed shift
        if _definitions is not None:
            for shift in _aux_shifts:
                _evaluate_definitions_at_shift(_definitions, context, shift)

        if is_recursive:
            evaluator = EquationsEvaluator(context, assign=True)
            results = [evaluator.visit(eq) for eq in ordered_equations]
            return np.array(results, dtype=float)
        else:
            evaluator = EquationsEvaluator(context)
            results = [evaluator.visit(eq) for eq in ordered_equations]
            return np.array(results, dtype=float)

    # Attach metadata to the function
    _eval_function.__doc__ = _build_docstring(spec, arg_groups, variables, definitions)
    _eval_function.arg_groups = arg_groups  # type: ignore[attr-defined]
    _eval_function.spec = spec  # type: ignore[attr-defined]
    _eval_function.variables = variables  # type: ignore[attr-defined]
    _eval_function.ordered_indices = ordered_indices  # type: ignore[attr-defined]
    _eval_function.definitions = definitions  # type: ignore[attr-defined]

    return _eval_function


def _evaluate_definitions_at_shift(
    definitions: DefinitionsBlock,
    context: dict[str, Any],
    shift: int,
) -> None:
    """Evaluate definitions at a given time shift, adding results to context.

    For each definition ``aux[t] = f(x[t], ...)``, this evaluates
    ``f(x[t+shift], ...)`` and stores the result as ``aux[shift]`` in the
    context's variables dict.

    The definitions are evaluated in topological order so that earlier
    definitions feed into later ones.
    """
    ctx_variables = context["variables"]

    for eq in definitions.equations:
        lhs_name = extract_lhs_variable(eq)
        if lhs_name is None:
            continue

        # Build a shifted context for this single evaluation.
        # The definition is written as aux[t] = f(x[t], ...), so we need
        # to map the definition's shift-0 references to the actual `shift`.
        shifted_vars: dict[str, dict[int, Any]] = {}
        for vname, shifts_dict in ctx_variables.items():
            shifted_vars[vname] = {}
            for s, val in shifts_dict.items():
                # Definition equations use shift 0 internally.
                # We remap: definition's shift 0 → actual shift.
                shifted_vars[vname][s] = val

        shifted_context = {
            "constants": context["constants"],
            "variables": shifted_vars,
            "values": {},
            "processes": {},
            "steady_states": {},
        }

        # The definition equation expects variables at shift 0.
        # We need to evaluate with the actual values at `shift`.
        # Strategy: temporarily put the shift-N values at shift 0.
        eval_vars: dict[str, dict[int, Any]] = {}
        for vname, shifts_dict in ctx_variables.items():
            eval_vars[vname] = {}
            if shift in shifts_dict:
                eval_vars[vname][0] = shifts_dict[shift]
            elif 0 in shifts_dict:
                eval_vars[vname][0] = shifts_dict[0]

        eval_context = {
            "constants": context["constants"],
            "variables": eval_vars,
            "values": {},
            "processes": {},
            "steady_states": {},
        }

        evaluator = EquationsEvaluator(eval_context, assign=True)
        evaluator.visit(eq)

        # Store result back in the main context at the requested shift
        if lhs_name in eval_vars and 0 in eval_vars[lhs_name]:
            if lhs_name not in ctx_variables:
                ctx_variables[lhs_name] = {}
            ctx_variables[lhs_name][shift] = eval_vars[lhs_name][0]


def compile_equation_group(
    equations: Sequence[Tree],
    variables: dict[str, list[str]],
    spec: EquationGroupSpec,
    constants: dict[str, float] | None = None,
    definitions: DefinitionsBlock | None = None,
) -> tuple[ConformityResult, Callable[..., np.ndarray] | None]:
    """Check conformity and generate a function in one step.

    Parameters:
        equations: List of equation ASTs.
        variables: Variable classification.
        spec: Equation group specification.
        constants: Parameter values.
        definitions: Optional definitions block for auxiliary variables.

    Returns:
        A tuple ``(conformity_result, function_or_none)``.  If conformity
        fails, the function is ``None``.
    """
    if constants is None:
        constants = {}

    result = check_equation_group(
        equations, variables, spec, constants=list(constants.keys())
    )
    if not result.ok:
        return result, None

    func = generate_equation_function(
        equations,
        variables,
        spec,
        constants=constants,
        conformity=result,
        definitions=definitions,
    )
    return result, func


def _build_docstring(
    spec: EquationGroupSpec,
    arg_groups: list[tuple[str, int]],
    variables: dict[str, list[str]],
    definitions: DefinitionsBlock | None = None,
) -> str:
    """Build a human-readable docstring for a generated function."""
    lines = [f"Evaluate '{spec.name}' equations.\n"]
    lines.append("Args:")
    for i, (group, shift) in enumerate(arg_groups):
        sign = "+" if shift > 0 else ""
        shift_str = f"t{sign}{shift}" if shift else "t"
        var_names = variables.get(group, [])
        lines.append(
            f"    arg{i} ({group}[{shift_str}]): "
            f"array of length {len(var_names)} — {var_names}"
        )
    if definitions is not None:
        lines.append(f"\n    (definitions auto-computed: {definitions.variable_names})")
    lines.append("")
    if spec.target is not None:
        target_vars = variables.get(spec.target, [])
        lines.append(f"Returns: array of length {len(target_vars)} — {target_vars}")
    else:
        lines.append(f"Returns: array of {len(arg_groups)} residuals")
    return "\n".join(lines)
