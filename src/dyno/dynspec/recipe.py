"""
Recipe conformity checker for structured economic models.

A *recipe* specifies, for each equation group (e.g. ``transition``,
``arbitrage``), which variable groups (e.g. ``states``, ``controls``,
``exogenous``) may appear at which time shifts (-1, 0, +1).

This module provides:

- Data structures to define recipes (`Recipe`, `EquationGroupSpec`,
  `VariableSpec`, `ComplementaritySpec`).
- Recipes read from YAML files in dolo's format (`load_recipes`,
  `get_recipe`); the bundled ``recipes.yaml`` defines `DTCC_RECIPE`.
- `extract_variables_from_equation` to list every ``(name, shift)``
  occurrence in a Lark equation AST, and `extract_complementarity`.
- `check_group` to validate the equations of one group of a recipe, given
  the variable names of each variable group (low-level:
  `check_equation_group`).
- `check_model` to validate a whole model whose equations are tagged
  with the names of the recipe's equation groups.
- `check_dag` to verify that a set of equations defining target
  variables forms a directed acyclic graph (recursive block).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from os import PathLike, path
from typing import Any, Mapping, Sequence, cast

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


@dataclass(frozen=True)
class ComplementaritySpec:
    """Specification of the complementarity conditions `res ⟂ lb <= x <= ub`.

    Attributes:
        variable: Group and shift of the bounded variable `x`. The i-th
            complementarity of a group must bound the i-th variable of it.
        bounds: Which ``(variable_group, shift)`` pairs may appear in `lb` and `ub`.
    """

    variable: VariableSpec
    bounds: list[VariableSpec] = field(default_factory=list)


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
        complementarities: If set, equations may carry complementarity
            conditions, checked against this spec. If ``None``, they may not.
    """

    name: str
    allowed: list[VariableSpec] = field(default_factory=list)
    target: str | None = None
    recursive: bool = True
    n_equations: str | None = None
    optional: bool = False
    complementarities: ComplementaritySpec | None = None


@dataclass
class Recipe:
    """A recipe defining the structure of a model type.

    Attributes:
        name: Recipe name (e.g. ``"dtcc"``).
        variable_groups: Ordered list of variable group names.
        equation_groups: Specifications for each equation group.
        process_group: Variable group of the variables declared as stochastic
            processes (used by `check_model` to classify variables).
    """

    name: str
    variable_groups: list[str] = field(default_factory=list)
    equation_groups: list[EquationGroupSpec] = field(default_factory=list)
    process_group: str | None = None

    def get_group_spec(self, name: str) -> EquationGroupSpec | None:
        """Return the equation group spec with the given name, or ``None``."""
        for spec in self.equation_groups:
            if spec.name == name:
                return spec
        return None

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> "Recipe":
        """Build a recipe from its entry in a recipes file (dolo's format)."""

        def vspec(entry) -> VariableSpec:
            return VariableSpec(str(entry[0]), int(entry[1]))

        equation_groups = []
        for gname, g in data.get("specs", {}).items():
            target = vspec(g["target"]).group if "target" in g else None
            comp = None
            if "complementarities" in g:
                c = g["complementarities"]
                comp = ComplementaritySpec(
                    variable=vspec(c["middle"]),
                    bounds=[vspec(e) for e in c.get("left-right", [])],
                )
            n_equations = g.get("n_equations", target)
            if n_equations is None and comp is not None:
                n_equations = comp.variable.group
            equation_groups.append(
                EquationGroupSpec(
                    name=gname,
                    allowed=[vspec(e) for e in g.get("eqs", [])],
                    target=target,
                    recursive=bool(g.get("recursive", target is not None)),
                    n_equations=n_equations,
                    optional=bool(g.get("optional", False)),
                    complementarities=comp,
                )
            )
        return cls(
            name=str(data.get("model_spec", name)),
            variable_groups=list(data.get("symbols", [])),
            equation_groups=equation_groups,
            process_group=data.get("processes"),
        )


# ── Recipes Files ────────────────────────────────────────────────────────────

RECIPES_FILE = path.join(path.dirname(__file__), "recipes.yaml")


def load_recipes(source: str | PathLike[str] | None = None) -> dict[str, Recipe]:
    """Read the recipes defined in a YAML file (dolo's format).

    Parameters:
        source: Path of the recipes file. Defaults to the bundled ``recipes.yaml``.

    Returns:
        A dictionary mapping recipe names to `Recipe` objects.
    """
    import yaml

    with open(source or RECIPES_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {name: Recipe.from_dict(name, d) for name, d in data.items()}


@functools.cache
def _bundled_recipes() -> dict[str, Recipe]:
    return load_recipes()


def get_recipe(recipe: str | Recipe = "dtcc") -> Recipe:
    """Return a bundled recipe by name (a `Recipe` is returned as is)."""
    if isinstance(recipe, Recipe):
        return recipe
    recipes = _bundled_recipes()
    if recipe not in recipes:
        raise ValueError(f"Unknown recipe '{recipe}' (available: {', '.join(recipes)})")
    return recipes[recipe]


DTCC_RECIPE = get_recipe("dtcc")


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

    # Only the residual of a complementarity is an equation; the bounds are
    # checked separately.
    def complementarity(self, tree: Tree) -> None:
        self.visit(tree.children[0])

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


@dataclass
class Complementarity:
    """The parts of a complementarity condition `residual ⟂ lower <= x <= upper`.

    Missing bounds are explicit: `lower` and `upper` are trees for `-inf` / `inf`.
    """

    residual: Tree
    lower: Tree
    variable: str
    shift: int
    upper: Tree


def extract_complementarity(tree: Tree) -> Complementarity | None:
    """Return the parts of a ``complementarity`` node, or ``None`` for any other
    equation."""
    if tree.data != "complementarity":
        return None
    residual, lower, x, upper = tree.children
    return Complementarity(
        residual=residual,
        lower=lower,
        variable=str(x.children[0].children[0]),
        shift=int(str(x.children[2].children[0])),
        upper=upper,
    )


# ── Conformity Results ───────────────────────────────────────────────────────


@dataclass
class Violation:
    """A single conformity violation.

    ``equation_index`` is the position of the equation in its group, or -1 for a
    violation of the whole group; ``variable_name`` is empty when no variable is
    involved; ``line`` is the source line of the equation, when known.
    """

    equation_index: int
    variable_name: str
    shift: int
    message: str
    line: int | None = None

    def __str__(self) -> str:
        where = f"Equation {self.equation_index}" if self.equation_index >= 0 else ""
        if self.line is not None:
            where = f"{where} (line {self.line})" if where else f"Line {self.line}"
        prefix = f"{where}: " if where else ""
        if not self.variable_name:
            return f"{prefix}{self.message}"
        sign = "+" if self.shift > 0 else ""
        shift_str = f"[t{sign}{self.shift}]" if self.shift else "[t]"
        return f"{prefix}{self.variable_name}{shift_str} — {self.message}"


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
    """Check a group of equations against an `EquationGroupSpec`.

    Parameters:
        equations: List of equation ASTs (Lark ``Tree`` nodes).
        variables: Mapping from variable group names (e.g. ``"states"``) to
            lists of variable names in that group.
        spec: The equation group specification to check against.
        constants: Optional list of parameter/constant names. Variables not
            found in any group are checked against this list; those found here
            are treated as ``parameters`` at shift 0.

    Returns:
        A `ConformityResult` with success/failure and detailed diagnostics.
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

    def check_occurrences(
        i: int,
        occurrences: dict[str, set[int]],
        allowed: set[tuple[str, int]],
        where: str,
    ) -> None:
        for var_name, shifts in occurrences.items():
            if var_name in constants_set:
                # Constants/parameters are always allowed at shift 0
                continue
            group = var_to_group.get(var_name)
            for shift in sorted(shifts):
                if group is None:
                    # Unknown variable — might be a constant not listed
                    message = f"Variable '{var_name}' not found in any variable group"
                elif (group, shift) not in allowed:
                    message = (
                        f"'{group}' variables at shift {shift:+d} "
                        f"are not allowed in {where}"
                    )
                else:
                    continue
                violations.append(Violation(i, var_name, shift, message, line=lines[i]))

    lines = [_line(eq) for eq in equations]
    comp_spec = spec.complementarities
    bounded = variables.get(comp_spec.variable.group) if comp_spec else None
    bounds_allowed = (
        {(vs.group, vs.shift) for vs in comp_spec.bounds} if comp_spec else set()
    )

    # Check each equation
    for i, eq in enumerate(equations):
        comp = extract_complementarity(eq)
        if comp is not None:
            if comp_spec is None:
                violations.append(
                    Violation(
                        i,
                        "",
                        0,
                        f"complementarity conditions are not allowed in '{spec.name}' equations",
                        line=lines[i],
                    )
                )
            else:
                # The i-th complementarity bounds the i-th variable of its group.
                expected = bounded[i] if bounded and i < len(bounded) else None
                group = var_to_group.get(comp.variable)
                if comp.shift != comp_spec.variable.shift or (
                    expected is None and group != comp_spec.variable.group
                ):
                    violations.append(
                        Violation(
                            i,
                            comp.variable,
                            comp.shift,
                            f"the complementarity must bound a "
                            f"'{comp_spec.variable.group}' variable at shift "
                            f"{comp_spec.variable.shift:+d}",
                            line=lines[i],
                        )
                    )
                elif expected is not None and comp.variable != expected:
                    violations.append(
                        Violation(
                            i,
                            comp.variable,
                            comp.shift,
                            f"expected the complementarity to bound '{expected}' "
                            f"('{comp_spec.variable.group}' variable #{i + 1}), "
                            f"found '{comp.variable}'",
                            line=lines[i],
                        )
                    )
                for bound in (comp.lower, comp.upper):
                    check_occurrences(
                        i,
                        extract_variables_from_equation(bound),
                        bounds_allowed,
                        f"the bounds of '{spec.name}' complementarities",
                    )

        # If there's a target, check LHS
        if spec.target is not None:
            lhs_var = extract_lhs_variable(eq)
            if lhs_var is None:
                violations.append(
                    Violation(
                        i,
                        "",
                        0,
                        f"Expected an equality with a {spec.target} variable on the LHS",
                        line=lines[i],
                    )
                )
            elif lhs_var not in variables.get(spec.target, []):
                violations.append(
                    Violation(
                        i,
                        lhs_var,
                        0,
                        f"LHS variable '{lhs_var}' is not in the '{spec.target}' group",
                        line=lines[i],
                    )
                )
            # For target-defining equations, only check RHS variables
            vars_to_check = extract_rhs_variables(eq)
        else:
            vars_to_check = extract_variables_from_equation(eq)

        check_occurrences(i, vars_to_check, allowed_set, f"'{spec.name}' equations")

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


def _parse_equations(equations: str | Sequence[Tree | str]) -> list[Tree]:
    from .grammar import parser

    def parse(txt: str) -> list[Tree]:
        tree = parser.parse(txt + "\n", start="equation_block")
        return cast(list[Tree], tree.children)

    if isinstance(equations, str):
        return parse(equations)
    if all(isinstance(item, str) for item in equations):
        return parse("\n".join(cast(Sequence[str], equations)))
    return [
        eq
        for item in equations
        for eq in (parse(item) if isinstance(item, str) else [item])
    ]


def check_group(
    equations: str | Sequence[Tree | str],
    group: str,
    variables: Mapping[str, Sequence[str]],
    recipe: str | Recipe = "dtcc",
    constants: Sequence[str] | None = None,
) -> ConformityResult:
    """Check the equations of one equation group of a recipe.

    Parameters:
        equations: The equations, as source text (one per line), or a list of
            source strings or parsed equation trees.
        group: Name of the equation group in the recipe (e.g. ``"transition"``).
        variables: Variable names of each variable group, e.g.
            ``{"states": ["z", "k"], "controls": ["n", "i"], ...}``. The order
            matters for complementarities: the i-th one must bound the i-th
            variable of its group.
        recipe: A recipe name (see `get_recipe`) or a `Recipe`.
        constants: Names to accept as parameters if they are used as variables.

    Returns:
        A `ConformityResult`.

    Example:
        >>> check_group(
        ...     "k[t] = (1-delta)*k[t-1] + i[t-1]",
        ...     "transition",
        ...     {"states": ["k"], "controls": ["i"]},
        ... ).ok
        True
    """
    rec = get_recipe(recipe)
    spec = rec.get_group_spec(group)
    if spec is None:
        names = ", ".join(g.name for g in rec.equation_groups)
        raise ValueError(
            f"Recipe '{rec.name}' has no equation group '{group}' (available: {names})"
        )
    return check_equation_group(
        _parse_equations(equations),
        {k: list(v) for k, v in variables.items()},
        spec,
        constants=constants,
    )


@dataclass
class ModelConformity:
    """Result of checking a whole model against a recipe (see `check_model`).

    Attributes:
        recipe: Name of the recipe.
        variables: Variable names of each variable group, as inferred (or given).
        groups: Result for each equation group present in the model.
        violations: Model-level violations: untagged equations, equations with
            several group tags, missing required groups.
    """

    recipe: str
    variables: dict[str, list[str]] = field(default_factory=dict)
    groups: dict[str, ConformityResult] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations and all(r.ok for r in self.groups.values())

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        head = "✓" if self.ok else "✗"
        lines = [f"{head} recipe '{self.recipe}'"]
        for gname, names in self.variables.items():
            lines.append(f"  {gname}: {', '.join(names)}")
        lines += [f"  - {v}" for v in self.violations]
        lines += [str(r) for r in self.groups.values()]
        return "\n".join(lines)


def check_model(
    model: str | Tree,
    recipe: str | Recipe = "dtcc",
    variables: Mapping[str, Sequence[str]] | None = None,
) -> ModelConformity:
    """Check a model whose equations are tagged by equation group.

    Each equation must carry exactly one tag naming an equation group of the
    recipe, typically through a tagged block (``[transition] :: { ... }``).
    Variable groups not given in `variables` are inferred from the model:

    - the target group of a group with a target (e.g. ``states`` for
      ``transition``) gets the left-hand-side variables of its equations;
    - the bounded group of complementarities (``controls``) gets the bounded
      variables, in order;
    - the recipe's process group (``exogenous``) gets the variables declared
      as stochastic processes (``e[t] <- N(...)``).

    Parameters:
        model: Model source text, or the tree parsed with ``start="free_block"``.
        recipe: A recipe name (see `get_recipe`) or a `Recipe`.
        variables: Variable names of some or all variable groups; these take
            precedence over inferred ones.

    Returns:
        A `ModelConformity`, with one `ConformityResult` per group.
    """
    from .analyze import AssignmentEvaluator
    from .grammar import parser

    rec = get_recipe(recipe)
    tree = parser.parse(model, start="free_block") if isinstance(model, str) else model
    ev = AssignmentEvaluator()
    ev.visit(tree)

    result = ModelConformity(recipe=rec.name)

    groups: dict[str, list[Tree]] = {}
    for eq, meta in zip(ev.equations, ev.equation_metadata):
        tags = [t for t in meta.get("tags", []) if rec.get_group_spec(t) is not None]
        if len(tags) == 1:
            groups.setdefault(tags[0], []).append(eq)
            continue
        names = ", ".join(g.name for g in rec.equation_groups)
        message = (
            f"equation is not tagged with an equation group of '{rec.name}' ({names})"
            if not tags
            else f"equation is tagged with several equation groups ({', '.join(tags)})"
        )
        result.violations.append(Violation(-1, "", 0, message, line=_line(eq)))

    for spec in rec.equation_groups:
        if spec.name not in groups and not spec.optional:
            result.violations.append(
                Violation(-1, "", 0, f"missing required equation group '{spec.name}'")
            )

    processes = [k[0] if isinstance(k, tuple) else k for k in ev.processes]
    result.variables = _infer_variables(rec, groups, processes, variables or {})

    for spec in rec.equation_groups:
        if spec.name in groups:
            result.groups[spec.name] = check_equation_group(
                groups[spec.name], result.variables, spec, constants=list(ev.constants)
            )
    return result


def _infer_variables(
    recipe: Recipe,
    groups: dict[str, list[Tree]],
    processes: Sequence[str],
    given: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    inferred: dict[str, list[str]] = {}

    def add(group: str, name: str | None) -> None:
        if name is not None and name not in inferred.setdefault(group, []):
            inferred[group].append(name)

    # Complementarities first: they fix the order of the bounded group.
    for spec in recipe.equation_groups:
        if spec.complementarities is not None:
            for eq in groups.get(spec.name, []):
                comp = extract_complementarity(eq)
                if comp is not None:
                    add(spec.complementarities.variable.group, comp.variable)
    for spec in recipe.equation_groups:
        if spec.target is not None:
            for eq in groups.get(spec.name, []):
                add(spec.target, extract_lhs_variable(eq))
    if recipe.process_group is not None:
        for name in processes:
            add(recipe.process_group, name)

    out = {
        g: list(given[g]) if g in given else inferred.get(g, [])
        for g in recipe.variable_groups
    }
    out.update({g: list(v) for g, v in given.items() if g not in out})
    return {g: v for g, v in out.items() if v}


def _line(tree: Tree) -> int | None:
    meta = getattr(tree, "meta", None)
    if meta is None or meta.empty:
        return None
    return meta.line


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
