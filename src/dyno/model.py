from __future__ import annotations

from abc import ABC, abstractmethod
from math import nan
import os
from typing import Any, TYPE_CHECKING, cast

import numpy as np
from numpy.linalg import solve as linsolve
from typing_extensions import Self

from .errors import SteadyStateError
from .language import ProductNormal
from .model_render import (
    model_repr_data,
    render_model_html,
    render_model_markdown,
    render_model_text,
)

if TYPE_CHECKING:
    from .larkfiles import SymbolicModel
    from .simul import SimulationResult, TransitionSimulation
    from .solver import PerturbationSolution
    from .variants import VariantCollection
    from .report import RunResults
from .typedefs import (
    IRFType,
    ModelContext,
    SimulateMode,
    Solver,
    TMatrix,
    TTensor,
    TVector,
    UnitsType,
)


class AbstractModel(ABC):
    """Abstract class representing an economic model."""

    name: str | None
    filename: str
    context: ModelContext
    symbols: dict[str, list[str]]
    processes: ProductNormal | None
    paths: dict[str, dict[int, float]] | None
    symbolic: SymbolicModel
    __steady_state__: dict[str, float] | None
    strict: bool
    _invalid_shifts: list[str] | None
    _steady_stats: dict[str, Any] | None = None
    _calibration_overrides: dict[str, Any]

    def __init__(
        self: Self,
        filename: str | os.PathLike[str] | None = None,
        txt: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        self.strict = strict
        self._invalid_shifts = None
        if filename is not None:
            filename = os.fspath(filename)

        match filename, txt:
            case (None, None):
                raise ValueError(
                    "Neither the file name nor content were passed to constructor. One of the two should be passed."
                )
            case (None, txt):
                assert txt is not None
                self.filename = "*anonymous*.dyno"
                self.import_model(txt, **kwargs)
            case (filename, None):
                assert filename is not None
                self.filename = filename
                self.import_file(filename, **kwargs)
            case _:
                assert filename is not None
                assert txt is not None
                self.filename = filename
                self.import_model(txt, **kwargs)

        self.__steady_state__ = None
        self._set_name()
        self._set_context()
        self._set_symbols()
        self._set_exogenous()

    def copy(self: Self):
        import copy

        return copy.deepcopy(self)

    def recalibrate(self: Self, **calib: Any) -> Self:
        """Return a new model instance with updated parameter/steady-state calibration."""
        raise NotImplementedError

    def variants(self: Self, *args: Any, **kwargs: Any) -> "VariantCollection[Self]":
        """Create a ``VariantCollection`` holding differently calibrated versions of this model.

        Examples
        --------
        >>> var = model.variants(a=[2, 3, 4])
        >>> sims = model.variants(a=[2, 3, 4]).solve().simulate()
        >>> fig = sims.plot()
        """
        from .variants import VariantCollection, _expand_variant_specs

        specs, labels = _expand_variant_specs(args, kwargs)
        models = [self.recalibrate(**spec) for spec in specs]
        return VariantCollection(models, specs=specs, labels=labels)

    def _repr_data(self: Self) -> dict[str, Any]:
        return model_repr_data(self)

    def _render_repr_text(self: Self, data: dict[str, Any]) -> str:
        return render_model_text(data)

    def _render_repr_html(self: Self, data: dict[str, Any]) -> str:
        return render_model_html(data)

    def __repr__(self: Self) -> str:
        return self._render_repr_text(self._repr_data())

    @property
    def checks(self) -> dict[str, bool]:
        return {"deterministic": self.processes is None}

    @property
    def metadata(self) -> dict[str, Any]:
        """Convenience accessor for model metadata parsed from source files."""
        symbolic = getattr(self, "symbolic", None)
        if symbolic is None:
            return {}
        return getattr(symbolic, "metadata", {})

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        symbolic = getattr(self, "symbolic", None)
        if symbolic is None:
            raise AttributeError(
                "Cannot set metadata before symbolic model is initialized"
            )
        symbolic.metadata = value

    @property
    def is_deterministic(self) -> bool:
        return self.processes is None

    def equations_with_tags(self: Self) -> list[tuple[int, str, list[str]]]:
        """Return equations paired with metadata tags and key/value annotations."""

        symbolic = getattr(self, "symbolic", None)
        if symbolic is None:
            return []

        pairs: list[tuple[Any, Any]]
        iterator = getattr(symbolic, "iter_equations_with_metadata", None)
        if callable(iterator):
            pairs = list(iterator())
        else:
            equations = list(getattr(symbolic, "equations", []))
            pairs = []
            for eq in equations:
                meta = getattr(eq, "meta", None)
                statement_metadata = (
                    getattr(meta, "statement_metadata", {}) if meta is not None else {}
                )
                pairs.append((eq, statement_metadata))

        rows: list[tuple[int, str, list[str]]] = []
        for i, (eq, meta) in enumerate(pairs, start=1):
            eq_text = str(eq)
            try:
                from dyno.dynspec.grammar import str_expression

                eq_text = str_expression(eq)
            except Exception:
                pass

            tags: list[str] = []
            if isinstance(meta, dict):
                raw_tags = meta.get("tags", [])
                if isinstance(raw_tags, list):
                    tags = [str(t) for t in raw_tags]
                for k, v in meta.items():
                    if k != "tags":
                        tags.append(f"{k}={v}")

            rows.append((i, eq_text, tags))

        return rows

    def print_equations_with_tags(self: Self) -> None:
        """Print all equations with their tags."""

        for i, eq_text, tags in self.equations_with_tags():
            tags_txt = ", ".join(tags) if tags else "-"
            print(f"{i:>3}. {eq_text}  [tags: {tags_txt}]")

    @abstractmethod
    def import_model(self: Self, txt: str, **kwargs: Any) -> None:
        """Set `symbolic` attribute from model text description."""

    @abstractmethod
    def _set_context(self: Self) -> None: ...

    @abstractmethod
    def compute_residuals(self: Self, y2, y1, y0, e) -> TVector: ...

    @abstractmethod
    def compute_jacobians(
        self: Self, y2, y1, y0, e
    ) -> tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]: ...

    def run(self: Self, default_pipeline: bool = False) -> "RunResults":
        from .report import RunResults

        return RunResults(model=self)

    def import_file(
        self: Self, filename: str | os.PathLike[str], **kwargs: Any
    ) -> None:
        with open(filename, "rt", encoding="utf-8") as f:
            txt = f.read()
        self.import_model(txt, **kwargs)

    def _set_name(self: Self) -> None:
        import os.path

        self.name = os.path.basename(self.filename).split(".")[0]

    def _set_symbols(self: Self) -> None:
        c = self.context

        # exogenous are either defined as processes or by specifying values with t >= 1
        # Preserve declaration/insertion order (avoid set() which scrambles order)
        exo_order: list[str] = []

        # Variables declared exogenous by the source file (``varexo`` in .mod
        # files) are exogenous even without a process or a forced path.
        declared = getattr(self.symbolic, "symbols", None)
        if isinstance(declared, dict):
            for name in declared.get("exogenous", []):
                if name not in exo_order:
                    exo_order.append(name)

        for exo_tuple in c.get("processes", {}).keys():
            for name in exo_tuple:
                if name not in exo_order:
                    exo_order.append(name)

        for name, val in c.get("values", {}).items():
            if isinstance(val, dict):
                has_t_ge_1 = any(
                    isinstance(t, (int, float)) and t >= 1 for t in val.keys()
                )
            elif isinstance(val, (list, tuple, np.ndarray)):
                has_t_ge_1 = len(val) > 0
            else:
                has_t_ge_1 = False

            if has_t_ge_1 and name not in exo_order:
                exo_order.append(name)

        exo_set = set(exo_order)

        declared_vars = list(c.get("variables", {}).keys())
        variables: list[str] = []
        for v in declared_vars:
            if v not in exo_set:
                variables.append(v)
        for name in c.get("values", {}).keys():
            if name not in exo_set and name not in variables:
                variables.append(name)
        for v in declared_vars:
            if v in exo_set and v not in variables:
                variables.append(v)
        for v in exo_order:
            if v not in variables:
                variables.append(v)

        exogenous = [v for v in variables if v in exo_set]
        endogenous = [v for v in variables if v not in exogenous]

        # try:
        #     parameters = [*self.symbolic.parameters]
        # except Exception:
        parameters = [*c["constants"].keys()]

        self.symbols = {
            "variables": variables,
            "endogenous": endogenous,
            "exogenous": exogenous,
            "parameters": parameters,
        }

    def _set_exogenous(self: Self) -> None:
        pps = self.context["processes"].values()
        if len(pps) == 0:
            self.processes = None
        else:
            self.processes = ProductNormal(*[e for e in pps])

    @property
    def steady_state(self) -> dict[str, float]:
        if self.__steady_state__ is None:
            self.__steady_state__ = self.context["steady_states"]
        return self.__steady_state__

    @property
    def residuals(self):
        y, e = self.__steady_state_vectors__
        return self.compute_residuals(y, y, y, e)

    def _check_defined(self: Self, action: str) -> None:
        """Raise ``UndefinedSymbolError`` if a parameter or a steady state is still NaN."""
        constants = self.context.get("constants", {})
        steady_states = self.context.get("steady_states", {})
        unassigned_params = [
            p
            for p in self.symbols.get("parameters", [])
            if isinstance(val := constants.get(p), float) and np.isnan(val)
        ]
        unassigned_ss = [
            v
            for v in self.symbols.get("endogenous", [])
            if isinstance(val := steady_states.get(v), float) and np.isnan(val)
        ]
        if unassigned_params or unassigned_ss:
            msgs = []
            if unassigned_params:
                msgs.append(
                    f"parameters without values: {', '.join(sorted(unassigned_params))}"
                )
            if unassigned_ss:
                msgs.append(
                    f"variables without steady state: {', '.join(sorted(unassigned_ss))}"
                    " (declare them with `x[~] <- ...` or call `model.steady()`)"
                )
            from .errors import UndefinedSymbolError

            raise UndefinedSymbolError(
                f"Cannot {action} due to uninitialized symbols ({'; '.join(msgs)})."
            )

    _check_eigenvalues: bool = False

    def check(
        self: "Self", tol: float = 1e-6, compute_eigenvalues: bool | None = None
    ) -> "Self":
        self._check_defined("check model")

        r = self.residuals
        if not all(abs(x) < tol for x in r):
            raise SteadyStateError(r)
        if compute_eigenvalues is None:
            compute_eigenvalues = self._check_eigenvalues
        if compute_eigenvalues:
            from .solver import solve_qz

            jac = self.jacobians
            A, B, C = jac[0], jac[1], jac[2]
            try:
                _, evs = solve_qz(A, B, C)
            except Exception:
                evs = None
            self._eigenvalues = evs
        else:
            self._eigenvalues = None
        return self

    @property
    def steady_stats(self: Self) -> dict[str, Any] | None:
        """Convergence statistics and algorithm details from the most recent steady-state calculation."""
        return getattr(self, "_steady_stats", None)

    def _variable_occurrences(self: Self) -> dict[str, tuple[int, int, int | None]]:
        """Map each variable to ``(n_equations, n_occurrences, first_line)``.

        ``n_equations`` counts the equations a variable appears in,
        ``n_occurrences`` counts every ``name[...]`` reference, and
        ``first_line`` is the source line of the first reference (if known).
        Only parse trees (``DynoFile``/``LModFile`` equations) are inspected;
        other equation representations yield an empty mapping.
        """
        occurrences: dict[str, tuple[int, int, int | None]] = {}
        for eq in getattr(self.symbolic, "equations", []):
            iter_subtrees = getattr(eq, "iter_subtrees_topdown", None)
            if not callable(iter_subtrees):
                continue
            eq_line = getattr(getattr(eq, "meta", None), "line", None)
            in_eq: set[str] = set()
            for subtree in iter_subtrees():
                if getattr(subtree, "data", None) not in ("variable", "value"):
                    continue
                try:
                    name = str(subtree.children[0].children[0])
                except (AttributeError, IndexError):
                    continue
                n_eq, n_occ, line = occurrences.get(name, (0, 0, None))
                if name not in in_eq:
                    in_eq.add(name)
                    n_eq += 1
                if line is None:
                    line = getattr(getattr(subtree, "meta", None), "line", None)
                    line = line if line is not None else eq_line
                occurrences[name] = (n_eq, n_occ + 1, line)
        return occurrences

    def _suspicious_endogenous(
        self: Self, require_single_occurrence: bool = False
    ) -> list[str]:
        """Describe endogenous variables that look like typos.

        By default a variable is reported when it has no steady-state value
        or appears in at most one equation; if some variables lack a steady
        state, only those are reported. With ``require_single_occurrence``,
        only variables referenced exactly once in the whole model *and*
        lacking a steady state are reported. Each description suggests the
        closest declared symbol (variables with a steady state, parameters,
        exogenous).
        """
        import difflib
        import math

        endogenous = list(self.symbols.get("endogenous", []))
        steady_states = self.context.get("steady_states", {})
        # Empty when equations are not parse trees (e.g. DynareModel): then
        # occurrence counts are unknown and only steady states are used.
        occurrences = self._variable_occurrences()

        def has_steady_state(name: str) -> bool:
            value = steady_states.get(name)
            if value is None:
                return False
            try:
                return not math.isnan(float(value))
            except (TypeError, ValueError):
                return True

        declared = [v for v in endogenous if has_steady_state(v)]
        declared += list(self.symbols.get("parameters", []))
        declared += list(self.symbols.get("exogenous", []))

        flagged: list[tuple[bool, str]] = []
        for name in endogenous:
            missing_ss = not has_steady_state(name)
            if occurrences:
                n_eq, n_occ, line = occurrences.get(name, (0, 0, None))
            else:
                n_eq, n_occ, line = 2, 2, None  # unknown: never "rare"
            if require_single_occurrence:
                if not (missing_ss and n_occ == 1):
                    continue
            elif not (missing_ss or n_eq <= 1):
                continue

            line_txt = f" (line {line})" if line is not None else ""
            facts: list[str] = []
            if n_occ == 0:
                facts.append("does not appear in any equation")
            elif n_occ == 1:
                facts.append(f"appears only once{line_txt}")
            elif n_eq == 1:
                facts.append(f"appears in a single equation{line_txt}")
            if missing_ss:
                facts.append("has no steady state")
            desc = f"'{name}' " + " and ".join(facts) + "."
            candidates = [c for c in dict.fromkeys(declared) if c != name]
            matches = difflib.get_close_matches(name, candidates, n=1)
            if matches:
                desc += f" Did you mean '{matches[0]}'?"
            flagged.append((missing_ss, desc))

        # Variables without steady state are the most likely culprits: when
        # there are any, the single-equation ones are only noise.
        if any(missing_ss for missing_ss, _ in flagged):
            flagged = [item for item in flagged if item[0]]
        return [desc for _, desc in flagged]

    def _check_square(self: Self) -> None:
        """Raise ``SystemStructureError`` unless there are as many equations as endogenous variables."""
        neq = len(getattr(self.symbolic, "equations", []))
        endogenous = list(self.symbols.get("endogenous", []))
        n_endo = len(endogenous)
        if neq != n_endo:
            from .errors import SystemStructureError

            def plural(n: int, word: str) -> str:
                return f"{n} {word}" if n == 1 else f"{n} {word}s"

            lines = [
                f"Model has {plural(neq, 'equation')} but "
                f"{plural(n_endo, 'endogenous variable')}. "
                "The dynamic system must be square."
            ]
            hints = self._suspicious_endogenous()
            max_hints = 10
            lines += [f"  {h}" for h in hints[:max_hints]]
            if len(hints) > max_hints:
                lines.append(f"  ... and {len(hints) - max_hints} more.")
            if not hints:
                lines.append(f"  Endogenous variables: {', '.join(endogenous)}")
            raise SystemStructureError("\n".join(lines))

    def steady(
        self: Self,
        tol: float = 1e-10,
        maxiter: int = 100,
        method: str = "hybr",
        **options: Any,
    ) -> Self:
        from scipy.optimize import root

        invalid_shifts = getattr(self, "_invalid_shifts", None)
        if invalid_shifts:
            from .errors import SystemStructureError

            shifts_str = ", ".join(invalid_shifts)
            raise SystemStructureError(
                f"Unsupported timing (only -1, 0, 1 so far): higher-order lead/lag detected ({shifts_str}). "
                "Dyno solvers currently support shifts in [-1, 1]."
            )

        if "tol" in options:
            tol = float(options.pop("tol"))
        if "maxiter" in options:
            maxiter = int(options.pop("maxiter"))
        if "method" in options:
            method = str(options.pop("method"))

        self._check_square()

        endogenous = self.symbols["endogenous"]
        if len(endogenous) == 0:
            res = self.copy()
            res._steady_stats = {
                "algorithm": method,
                "converged": True,
                "iterations": 0,
                "function_evaluations": 0,
                "jacobian_evaluations": 0,
                "max_residual": 0.0,
                "tolerance": float(tol),
                "message": "No endogenous variables to solve.",
            }
            return res

        y0, _ = self.__steady_state_vectors__
        guess = np.nan_to_num(np.asarray(y0, dtype=float), nan=1.0)

        def _candidate(values: np.ndarray) -> AbstractModel:
            calib = {name: float(values[i]) for i, name in enumerate(endogenous)}
            model = self.recalibrate(**calib)
            for name, value in calib.items():
                model.context["steady_states"][name] = value
            model.__steady_state__ = model.context["steady_states"]
            return model

        def _fun(values: np.ndarray) -> np.ndarray:
            model = _candidate(values)
            return np.asarray(model.residuals, dtype=float)

        def _jac(values: np.ndarray) -> np.ndarray:
            model = _candidate(values)
            jac = model.jacobians
            A, B, C = jac[1], jac[2], jac[3]
            return A + B + C

        solver_options: dict[str, Any] = {"maxfev": maxiter}
        if "options" in options and isinstance(options["options"], dict):
            solver_options.update(options.pop("options"))
        solver_options.update(options)

        sol: Any = root(
            _fun,
            guess,
            jac=_jac,
            method=cast(Any, method),
            options=cast(Any, solver_options),
        )

        solved = cast(Self, _candidate(np.asarray(sol.x, dtype=float)))
        residuals = np.asarray(solved.residuals, dtype=float)

        converged = bool(
            sol.success
            and np.isfinite(residuals).all()
            and (np.max(np.abs(residuals)) <= tol)
        )
        iterations = getattr(sol, "nit", None)
        if iterations is None:
            iterations = getattr(sol, "nfev", None)

        stats = {
            "algorithm": method,
            "converged": converged,
            "iterations": iterations,
            "function_evaluations": getattr(sol, "nfev", None),
            "jacobian_evaluations": getattr(sol, "njev", None),
            "max_residual": (
                float(np.max(np.abs(residuals)))
                if np.isfinite(residuals).all()
                else float("nan")
            ),
            "tolerance": float(tol),
            "message": str(getattr(sol, "message", "")),
        }
        solved._steady_stats = stats

        if not converged:
            raise SteadyStateError(residuals, steady_stats=stats)

        return solved

    @property
    def jacobians(self):
        y, e = self.__steady_state_vectors__
        return self.compute_jacobians(y, y, y, e)

    @property
    def __steady_state_vectors__(self) -> tuple[list[float], list[float]]:
        c = self.context
        y = [c["steady_states"].get(name, nan) for name in self.symbols["endogenous"]]

        e: list[float] = []
        for name in self.symbols["exogenous"]:
            if name in c["steady_states"]:
                e.append(c["steady_states"][name])
            elif name in c["values"]:
                e.append(c["values"][name].get(0, 0.0))
            else:
                e.append(nan)
        return y, e

    def describe(self: Self) -> None:
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        console.print(Markdown(self._markdown_()))

    def _markdown_(self: Self) -> str:
        return render_model_markdown(self._repr_data(), self.filename)

    def _repr_html_(self: Self) -> str:
        return self._render_repr_html(self._repr_data())

    def solve(self: Self, **args: Any) -> "PerturbationSolution | TransitionSimulation":
        invalid_shifts = getattr(self, "_invalid_shifts", None)
        if invalid_shifts:
            from .errors import SystemStructureError

            shifts_str = ", ".join(invalid_shifts)
            raise SystemStructureError(
                f"Unsupported timing (only -1, 0, 1 so far): higher-order lead/lag detected ({shifts_str}). "
                "Dyno solvers currently support shifts in [-1, 1]."
            )

        self._check_square()
        if self.is_deterministic:
            from .solver import deterministic_solve

            return deterministic_solve(self, **args)
        return self.perturb(**args)

    def simulate(
        self: Self,
        T: int | None = None,
        mode: SimulateMode = "auto",
        N: int = 1,
        units: UnitsType | None = None,
        **args: Any,
    ) -> "SimulationResult":
        """Simulate the model.

        For deterministic models (or when ``mode`` is ``'transition'`` or ``'deterministic'``),
        runs the stacked-time perfect foresight solver and returns a ``TransitionSimulation``.
        For stochastic models, solves the first-order perturbation and delegates to
        ``solution.simulate(...)``.
        """
        self._check_square()
        if self.is_deterministic or mode in ("transition", "deterministic"):
            from .solver import deterministic_solve

            target_units: UnitsType = "level" if units is None else units
            return deterministic_solve(self, T=T, units=target_units, **args)

        solve_kwargs = {k: v for k, v in args.items() if k in {"method"}}
        sim_kwargs = {k: v for k, v in args.items() if k not in {"method"}}
        sol = self.perturb(**solve_kwargs)
        horizon = 40 if T is None else int(T)
        target_units = "deviation" if units is None else units
        return sol.simulate(T=horizon, mode=mode, N=N, units=target_units, **sim_kwargs)

    def perturb(self: Self, method: Solver = "qz") -> "PerturbationSolution":
        from .solver import PerturbationSolution, RecursiveDecisionRule
        from .solver import solve as solve_quadratic_matrix

        self._check_defined("solve the model")

        r, A, B, C, D = self.jacobians

        X, evs = solve_quadratic_matrix(A, B, C, method=method)

        Y = linsolve(A @ X + B, -D)

        v = self.symbols["endogenous"]
        e = self.symbols["exogenous"]

        assert self.processes is not None
        Σ = self.processes.Σ

        y, _ = self.__steady_state_vectors__
        y0 = np.reshape(y, len(y))

        dr = RecursiveDecisionRule(
            X,
            Y,
            Σ,
            {"endogenous": v, "exogenous": e},
            x0=y0,
            model=self,
        )
        return PerturbationSolution(dr, evs=evs)

    def _dynamic_point(
        self: Self,
        v_prev: TVector,
        v_curr: TVector,
        v_next: TVector,
        diff: bool = False,
    ) -> TVector | tuple[TVector, TTensor]:
        """Evaluate the dynamic equations at one date.

        ``v_prev``, ``v_curr`` and ``v_next`` hold all variables (endogenous
        first, then exogenous, in the order of ``symbols["variables"]``) at
        dates t-1, t and t+1. Returns the residuals, and when ``diff`` is
        true also the Jacobian ``J`` of shape ``(n_equations, n_variables, 3)``
        where the last axis indexes the date (t-1, t, t+1).

        The default implementation relies on ``compute_residuals`` and
        ``compute_jacobians``, which only see the date-t exogenous values.
        """
        q = len(self.symbols["endogenous"])
        y_prev, y_curr, y_next = v_prev[:q], v_curr[:q], v_next[:q]
        e = v_curr[q:]
        if not diff:
            return np.asarray(self.compute_residuals(y_next, y_curr, y_prev, e))
        r, A, B, C, D = self.compute_jacobians(y_next, y_curr, y_prev, e)
        p = len(self.symbols["variables"])
        J = np.zeros((len(r), p, 3))
        J[:, :q, 0] = C
        J[:, :q, 1] = B
        J[:, :q, 2] = A
        J[:, q:, 1] = D
        return np.asarray(r), J

    def _forced_path(self: Self, T: int) -> TMatrix:
        """Steady state at every date, overridden by ``context["values"]``."""
        y, e = self.__steady_state_vectors__
        ss = np.concatenate([y, e])
        v1 = ss[None, :].repeat(T + 1, axis=0)
        for key, value in self.context.get("values", {}).items():
            i = self.symbols["variables"].index(key)
            for a, b in value.items():
                if 0 <= a <= T:
                    v1[a, i] = b
        return v1

    def deterministic_residuals_with_jacobian(
        self: Self,
        v: np.ndarray,
        sparsify: bool = False,
        continuation: str = "stationary",
        growth_rate: Any = None,
        growth_type: str = "geometric",
        **kwargs: Any,
    ) -> tuple[np.ndarray, Any]:
        """Stacked-time residuals and Jacobian of the perfect-foresight system.

        Generic implementation built on ``_dynamic_point``, one date at a
        time. Date 0 is pinned to the forced path (initial condition), the
        trailing exogenous variables follow ``context["values"]`` at every
        date, and the value after the horizon comes from the terminal
        ``continuation`` (see ``_compute_terminal_continuation``).
        """
        from .dyno_model import (
            _build_dense_jacobian,
            _build_sparse_jacobian,
            _compute_terminal_continuation,
        )

        continuation = kwargs.get("terminal_condition", continuation)
        if continuation in ("static", "steady_static"):
            raise NotImplementedError(
                f"Terminal condition {continuation!r} is only available for DynoModel."
            )

        flat = v.ndim == 1
        variables = list(self.symbols["variables"])
        p = len(variables)
        q = len(self.symbols["endogenous"])
        T = int(np.prod(v.shape) / p - 1)
        v = v.reshape((T + 1, p))

        y, e = self.__steady_state_vectors__
        ss = np.concatenate([y, e])
        v_prev_T = v[-2, :] if T >= 1 else v[-1, :]
        v_next_T, alpha_prev, alpha_curr = _compute_terminal_continuation(
            continuation,
            v_prev_T,
            v[-1, :],
            ss,
            variables,
            growth_rate=growth_rate,
            growth_type=growth_type,
        )
        v_b = np.concatenate([v[0, :][None, :], v[:-1, :]], axis=0)
        v_f = np.concatenate([v[1:, :], v_next_T[None, :]], axis=0)

        v1 = self._forced_path(T)
        N = T + 1
        res = np.zeros((N, p))
        DD = np.zeros((N, p, p, 3))
        for t in range(N):
            r_t, J_t = cast(
                tuple[TVector, TTensor],
                self._dynamic_point(v_b[t], v[t], v_f[t], diff=True),
            )
            res[t, :q] = r_t
            DD[t, :q, :, :] = J_t
        res[:, q:] = v[:, q:] - v1[:, q:]
        for i in range(q, p):
            DD[:, i, i, 1] = 1.0
        res[0, :] = v[0, :] - v1[0, :]  # initial condition

        if not flat:
            return res, DD
        J: Any
        if sparsify:
            J = _build_sparse_jacobian(N, p, DD, {}, alpha_prev, alpha_curr)
        else:
            J = _build_dense_jacobian(N, p, DD, {}, alpha_prev, alpha_curr)
        return res.ravel(), J

    def deterministic_guess(self: Self, T: int | None = None):
        if T is None:
            T = int(self.context.get("constants", {}).get("T", 50))

        y, e = self.__steady_state_vectors__
        v0 = np.concatenate([y, e])[None, :].repeat(T + 1, axis=0)

        for key, value in self.context.get("values", {}).items():
            i = self.symbols["variables"].index(key)
            for a, b in value.items():
                if a <= T:
                    v0[a, i] = b

        return v0
