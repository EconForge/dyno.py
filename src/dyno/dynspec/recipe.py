"""
Recipe conformity checker for structured economic models.

A *recipe* specifies, for each equation group (e.g. ``transition``,
``arbitrage``), which variable groups (e.g. ``states``, ``controls``,
``exogenous``) may appear at which time shifts (-1, 0, +1).

This module provides:

- Data structures to define recipes (:class:`Recipe`, :class:`EquationGroupSpec`,
  :class:`VariableSpec`).
- A pre-built :data:`DTCC_RECIPE` matching the dolo "dtcc" model type.
- :func:`extract_variables_from_equation` to list every ``(name, shift)``
  occurrence in a Lark equation AST.
- :func:`check_equation_group` to validate a set of equations against a
  recipe specification.
- :func:`check_dag` to verify that a set of equations defining target
  variables forms a directed acyclic graph (recursive block).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from lark.tree import Tree
from lark.visitors import Interpreter

# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VariableSpec:
    """A specification for a variable occurrence: which group and at which time shift."""

    group: str
    shift: int

    def __repr__(self) -> str:
        sign = "+" if self.shift > 0 else ""
        return (
            f"{self.group}[t{sign}{self.shift}]" if self.shift else f"{self.group}[t]"
        )


@dataclass
class EquationGroupSpec:
    """Specification for one equation group within a recipe.

    Attributes:
        name: Equation group name (e.g. ``"transition"``, ``"arbitrage"``).
        allowed: Which ``(variable_group, shift)`` pairs are permitted.
        target: If set, the variable group that this block *defines*
            (e.g. ``"states"`` for transitions).
        recursive: If ``True``, the equations must form a DAG when
            interpreted as assignments to target variables.
        n_equations: If set, the variable group whose size must equal
            the number of equations (e.g. ``"states"``).
        optional: If ``True``, the equation group may be absent from the model.
    """

    name: str
    allowed: list[VariableSpec] = field(default_factory=list)
    target: str | None = None
    recursive: bool = True
    n_equations: str | None = None
    optional: bool = False


@dataclass
class Recipe:
    """A recipe defining the structure of a model type.

    Attributes:
        name: Recipe name (e.g. ``"dtcc"``).
        variable_groups: Ordered list of variable group names.
        equation_groups: Specifications for each equation group.
    """

    name: str
    variable_groups: list[str] = field(default_factory=list)
    equation_groups: list[EquationGroupSpec] = field(default_factory=list)

    def get_group_spec(self, name: str) -> EquationGroupSpec | None:
        """Return the equation group spec with the given name, or ``None``."""
        for spec in self.equation_groups:
            if spec.name == name:
                return spec
        return None


# ── Pre-built Recipes ────────────────────────────────────────────────────────

DTCC_RECIPE = Recipe(
    name="dtcc",
    variable_groups=["exogenous", "states", "controls", "auxiliaries", "parameters"],
    equation_groups=[
        EquationGroupSpec(
            name="definitions",
            allowed=[
                VariableSpec("exogenous", 0),
                VariableSpec("states", 0),
                VariableSpec("controls", 0),
                VariableSpec("auxiliaries", 0),
                VariableSpec("parameters", 0),
            ],
            target="auxiliaries",
            recursive=True,
            n_equations="auxiliaries",
            optional=True,
        ),
        EquationGroupSpec(
            name="transition",
            allowed=[
                VariableSpec("exogenous", -1),
                VariableSpec("states", -1),
                VariableSpec("controls", -1),
                VariableSpec("exogenous", 0),
                VariableSpec("parameters", 0),
            ],
            target="states",
            recursive=True,
            n_equations="states",
        ),
        EquationGroupSpec(
            name="arbitrage",
            allowed=[
                VariableSpec("exogenous", 0),
                VariableSpec("states", 0),
                VariableSpec("controls", 0),
                VariableSpec("auxiliaries", 0),
                VariableSpec("exogenous", 1),
                VariableSpec("states", 1),
                VariableSpec("controls", 1),
                VariableSpec("auxiliaries", 1),
                VariableSpec("parameters", 0),
            ],
            target=None,
            recursive=False,
            n_equations="controls",
        ),
    ],
)


# ── Variable Extraction ─────────────────────────────────────────────────────


class _VariableCollector(Interpreter):
    """Walk an equation AST and collect every ``(variable_name, shift)`` pair."""

    def __init__(self) -> None:
        super().__init__()
        self.occurrences: dict[str, set[int]] = {}

    def variable(self, tree: Tree) -> None:
        name = str(tree.children[0].children[0])
        shift = int(str(tree.children[2].children[0]))
        self.occurrences.setdefault(name, set()).add(shift)

    # Traverse into sub-expressions without losing information.
    def _default(self, tree: Tree) -> None:
        for child in tree.children:
            if isinstance(child, Tree):
                self.visit(child)

    # Override all arithmetic / structural nodes so they recurse.
    add = sub = mul = div = pow = neg = call = _default
    equality = bare_formula = number = _default
    constant = _default

    # Leaf nodes that don't contain variables — do nothing.
    def name(self, tree: Tree) -> None:
        pass

    def index(self, tree: Tree) -> None:
        pass

    def shift(self, tree: Tree) -> None:
        pass

    def time(self, tree: Tree) -> None:
        pass

    # value nodes (e.g. x[3]) are not variables — skip
    def value(self, tree: Tree) -> None:
        pass


def extract_variables_from_equation(tree: Tree) -> dict[str, set[int]]:
    """Extract all variable occurrences ``{name: {shifts}}`` from an equation AST.

    Parameters:
        tree: A Lark ``Tree`` node representing an equation (``equality``,
            ``bare_formula``, etc.).

    Returns:
        Dictionary mapping variable names to the set of time shifts at which
        they appear.  For example ``{"k": {-1, 0}, "c": {0, 1}}``.
    """
    collector = _VariableCollector()
    collector.visit(tree)
    return collector.occurrences


def extract_lhs_variable(tree: Tree) -> str | None:
    """Return the name of the variable on the LHS of an ``equality`` node.

    Returns ``None`` if the equation is not an equality or the LHS is not
    a single variable.
    """
    if tree.data != "equality":
        return None
    lhs = tree.children[0]
    if lhs.data == "variable":
        return str(lhs.children[0].children[0])
    return None


def extract_rhs_variables(tree: Tree) -> dict[str, set[int]]:
    """Extract variables from the RHS of an equality, or from the whole expression
    if it's a bare_formula."""
    if tree.data == "equality":
        rhs = tree.children[1]
    else:
        rhs = tree
    collector = _VariableCollector()
    collector.visit(rhs)
    return collector.occurrences


# ── Conformity Results ───────────────────────────────────────────────────────


@dataclass
class Violation:
    """A single conformity violation."""

    equation_index: int
    variable_name: str
    shift: int
    message: str

    def __str__(self) -> str:
        sign = "+" if self.shift > 0 else ""
        shift_str = f"[t{sign}{self.shift}]" if self.shift else "[t]"
        return f"Equation {self.equation_index}: {self.variable_name}{shift_str} — {self.message}"


@dataclass
class ConformityResult:
    """Result of checking an equation group against a recipe spec."""

    group_name: str
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    dag_order: list[str] | None = None

    def __str__(self) -> str:
        if self.ok:
            msg = f"✓ {self.group_name}: OK"
            if self.dag_order is not None:
                msg += f" (order: {' → '.join(self.dag_order)})"
            return msg
        lines = [f"✗ {self.group_name}: {len(self.violations)} violation(s)"]
        for v in self.violations:
            lines.append(f"  - {v}")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return self.ok


# ── Conformity Checking ─────────────────────────────────────────────────────


def check_equation_group(
    equations: Sequence[Tree],
    variables: dict[str, list[str]],
    spec: EquationGroupSpec,
    constants: Sequence[str] | None = None,
) -> ConformityResult:
    """Check a group of equations against an :class:`EquationGroupSpec`.

    Parameters:
        equations: List of equation ASTs (Lark ``Tree`` nodes).
        variables: Mapping from variable group names (e.g. ``"states"``) to
            lists of variable names in that group.
        spec: The equation group specification to check against.
        constants: Optional list of parameter/constant names. Variables not
            found in any group are checked against this list; those found here
            are treated as ``parameters`` at shift 0.

    Returns:
        A :class:`ConformityResult` with success/failure and detailed diagnostics.
    """
    if constants is None:
        constants = []
    constants_set = set(constants)

    # Build reverse lookup: variable_name -> group_name
    var_to_group: dict[str, str] = {}
    for group_name, names in variables.items():
        for name in names:
            var_to_group[name] = group_name

    # Build allowed set: {(group, shift)}
    allowed_set: set[tuple[str, int]] = set()
    for vs in spec.allowed:
        allowed_set.add((vs.group, vs.shift))

    violations: list[Violation] = []

    # Check equation count
    if spec.n_equations is not None:
        expected_group = spec.n_equations
        if expected_group in variables:
            expected_count = len(variables[expected_group])
            if len(equations) != expected_count:
                violations.append(
                    Violation(
                        equation_index=-1,
                        variable_name="",
                        shift=0,
                        message=(
                            f"Expected {expected_count} equations "
                            f"(one per {expected_group} variable), got {len(equations)}"
                        ),
                    )
                )

    # Check each equation
    target_vars: list[str | None] = []
    for i, eq in enumerate(equations):
        vars_in_eq = extract_variables_from_equation(eq)

        # If there's a target, check LHS
        if spec.target is not None:
            lhs_var = extract_lhs_variable(eq)
            if lhs_var is None:
                violations.append(
                    Violation(
                        equation_index=i,
                        variable_name="",
                        shift=0,
                        message=f"Expected an equality with a {spec.target} variable on the LHS",
                    )
                )
                target_vars.append(None)
            elif lhs_var not in variables.get(spec.target, []):
                violations.append(
                    Violation(
                        equation_index=i,
                        variable_name=lhs_var,
                        shift=0,
                        message=(
                            f"LHS variable '{lhs_var}' is not in the "
                            f"'{spec.target}' group"
                        ),
                    )
                )
                target_vars.append(lhs_var)
            else:
                target_vars.append(lhs_var)

            # For target-defining equations, only check RHS variables
            rhs_vars = extract_rhs_variables(eq)
            vars_to_check = rhs_vars
        else:
            target_vars.append(None)
            vars_to_check = vars_in_eq

        # Check each variable occurrence
        for var_name, shifts in vars_to_check.items():
            if var_name in constants_set:
                # Constants/parameters are always allowed at shift 0
                continue
            group = var_to_group.get(var_name)
            if group is None:
                # Unknown variable — might be a constant not listed
                for shift in shifts:
                    violations.append(
                        Violation(
                            equation_index=i,
                            variable_name=var_name,
                            shift=shift,
                            message=f"Variable '{var_name}' not found in any variable group",
                        )
                    )
                continue
            for shift in shifts:
                if (group, shift) not in allowed_set:
                    violations.append(
                        Violation(
                            equation_index=i,
                            variable_name=var_name,
                            shift=shift,
                            message=(
                                f"'{group}' variables at shift {shift:+d} "
                                f"are not allowed in '{spec.name}' equations"
                            ),
                        )
                    )

    # Check DAG if required and no prior violations
    dag_order: list[str] | None = None
    if spec.recursive and spec.target is not None and not violations:
        dag_result = check_dag(equations, spec.target, variables, constants)
        if dag_result is None:
            violations.append(
                Violation(
                    equation_index=-1,
                    variable_name="",
                    shift=0,
                    message=(
                        f"Equations defining '{spec.target}' variables do not "
                        f"form a DAG (cycle detected)"
                    ),
                )
            )
        else:
            dag_order = dag_result

    ok = len(violations) == 0
    return ConformityResult(
        group_name=spec.name,
        ok=ok,
        violations=violations,
        dag_order=dag_order,
    )


# ── DAG Checking ─────────────────────────────────────────────────────────────


def check_dag(
    equations: Sequence[Tree],
    target_group: str,
    variables: dict[str, list[str]],
    constants: Sequence[str] | None = None,
) -> list[str] | None:
    """Check that equations form a DAG when viewed as assignments to target variables.

    Each equation is expected to be an equality ``target_var[t] = f(...)``.
    The DAG constraint means: if equation *i* defines ``x[t]`` and its RHS
    mentions ``y[t]`` (where ``y`` is also a target variable), then ``y`` must
    be defined by an earlier equation in the ordering.

    Parameters:
        equations: Equation ASTs.
        target_group: The variable group being defined (e.g. ``"states"``).
        variables: Variable classification.
        constants: Optional list of constant names.

    Returns:
        A topological ordering (list of target variable names) if the equations
        form a DAG, or ``None`` if a cycle is detected.
    """
    target_names = set(variables.get(target_group, []))

    # Build adjacency: defined_var -> {target vars used on RHS at shift 0}
    defined_by: dict[str, int] = {}  # var_name -> equation index
    deps: dict[str, set[str]] = {}  # var_name -> set of target vars it depends on

    for i, eq in enumerate(equations):
        lhs_var = extract_lhs_variable(eq)
        if lhs_var is None or lhs_var not in target_names:
            continue

        defined_by[lhs_var] = i
        rhs_vars = extract_rhs_variables(eq)

        # Only count dependencies on *other target variables at the same shift (0)*
        # i.e. same-period target variables that must be computed first
        rhs_target_deps: set[str] = set()
        for var_name, shifts in rhs_vars.items():
            if var_name in target_names and var_name != lhs_var and 0 in shifts:
                rhs_target_deps.add(var_name)
        deps[lhs_var] = rhs_target_deps

    # Topological sort (Kahn's algorithm)
    in_degree: dict[str, int] = {v: 0 for v in defined_by}
    for v, d in deps.items():
        for dep in d:
            if dep in in_degree:
                in_degree[v] = in_degree.get(v, 0) + 1

    # Recompute properly
    in_degree = {v: 0 for v in defined_by}
    for v, d in deps.items():
        for dep in d:
            if dep in defined_by:
                in_degree[v] += 1

    queue: list[str] = [v for v, deg in in_degree.items() if deg == 0]
    order: list[str] = []

    while queue:
        # Pick one with zero in-degree
        node = queue.pop(0)
        order.append(node)

        # Remove edges from node to its dependents
        for v, d in deps.items():
            if node in d:
                in_degree[v] -= 1
                if in_degree[v] == 0 and v not in order:
                    queue.append(v)

    if len(order) != len(defined_by):
        return None  # Cycle detected

    return order
