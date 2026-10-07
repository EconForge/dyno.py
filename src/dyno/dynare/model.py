from dyno.model import AbstractModel
from dyno.language import Normal
import re

import numpy as np

from typing_extensions import Self
from typing import TYPE_CHECKING, Any, cast
from dyno.typedefs import ModelContext, TVector, TMatrix

if TYPE_CHECKING:
    from dyno.report import RunResults

from dyno.errors import DynareParserError, SteadyStateError


class DynareModel(AbstractModel):

    symbolic: Any
    _check_eigenvalues: bool = True

    @property
    def metadata(self: Self) -> dict[str, Any]:
        """Metadata is stored on `context` since the compiled `symbolic` object cannot hold arbitrary attributes."""
        context = getattr(self, "context", None)
        if context is None:
            return {}
        return context.get("metadata", {})

    @metadata.setter
    def metadata(self: Self, value: dict[str, Any]) -> None:
        context = getattr(self, "context", None)
        if context is None:
            raise AttributeError("Cannot set metadata before context is initialized")
        context["metadata"] = value

    @staticmethod
    def _load_dynare_preprocessor():
        try:
            from dynare_preprocessor import DynareModel as Modfile
            from dynare_preprocessor import PreprocessorException
        except ModuleNotFoundError as e:
            if e.name == "dynare_preprocessor":
                raise ModuleNotFoundError(
                    "DynareModel emulation requires the optional dependency "
                    "'dynare-preprocessor-pylib'. With pixi, add it via: "
                    "pixi add --feature dynare dynare-preprocessor-pylib, "
                    "or use an environment that includes the dynare feature. "
                    "Note: .mod files can still be imported with DynoModel without this dependency."
                ) from e
            raise
        return Modfile, PreprocessorException

    def _equation_line_numbers(self: Self) -> list[int]:
        """Best-effort mapping from equation index to source line number."""
        txt = getattr(self, "_original_txt", None)
        if not isinstance(txt, str):
            return []

        lines = txt.splitlines()
        in_model = False
        eq_lines: list[int] = []

        for idx, raw in enumerate(lines, start=1):
            stripped = raw.strip()
            if not in_model:
                # Matches "model;" and "model(linear);" forms.
                if stripped.lower().startswith("model"):
                    in_model = True
                continue

            if stripped.lower() == "end;":
                break

            if not stripped:
                continue
            if stripped.startswith(("//", "%", "#")):
                continue

            # Keep only likely equation lines.
            if "=" in stripped:
                eq_lines.append(idx)

        return eq_lines

    def _rebuild(self: Self) -> Self:
        txt = getattr(self, "_original_txt", None)
        if txt is None:
            with open(self.filename, "rt", encoding="utf-8") as f:
                txt = f.read()

        options = getattr(self, "_import_options", {})
        model = self.__class__(
            filename=self.filename,
            txt=txt,
            strict=getattr(self, "strict", False),
            **options,
        )

        previous = getattr(self, "_calibration_overrides", {})
        if len(previous) > 0:
            model = model.recalibrate(**previous)

        return model

    def recalibrate(self: Self, **calib):
        m = self._rebuild()

        known = set(m.context["constants"].keys()) | set(
            m.context["steady_states"].keys()
        )
        unknown = [k for k in calib.keys() if k not in known]
        if len(unknown) > 0:
            raise KeyError(f"Unknown calibration key(s): {', '.join(unknown)}")

        for key, value in calib.items():
            val = float(value)
            if key in m.context["constants"]:
                m.context["constants"][key] = val
            if key in m.context["steady_states"]:
                m.context["steady_states"][key] = val

        m.__steady_state__ = m.context["steady_states"]

        previous = getattr(self, "_calibration_overrides", {})
        m._calibration_overrides = previous | {k: float(v) for k, v in calib.items()}
        return m

    def run(self: Self, default_pipeline: bool = False) -> "RunResults":
        """Execute the model's computing commands and collect the results.

        Commands are read from ``model.metadata["dynare_commands"]`` (the
        computing commands of the ``.mod`` file) or, failing that, from
        ``model.metadata["run"]``, as normalized
        ``{"command": ..., "options": {...}}`` entries in the same stable
        ``@run:`` command format as :meth:`dyno.DynoModel.run`, and executed
        in order.

        Supported commands:

        - ``steady``: solve for the steady state (``model.steady(**options)``)
        - ``resid``: compute steady-state residuals
        - ``check``: compute residuals and check Blanchard-Kahn conditions
        - ``simul`` / ``simulate``: perfect-foresight simulation
        - ``stoch_simul``: solve, compute moments and IRFs (``irf``,
          ``type``, ``variables`` and ``nograph`` options)
        - ``plot``: select the variables to plot

        Other commands are ignored.

        Parameters
        ----------
        default_pipeline : bool, optional
            If True and the model defines no commands, compute residuals,
            then IRFs (stochastic models) or a perfect-foresight simulation
            (deterministic models) over 40 periods. By default False.

        Returns
        -------
        RunResults
            Container with the final model, residuals, solution, eigenvalues,
            moments, simulation and figure produced by the pipeline.
        """
        from dyno.report import RunResults

        commands = self.metadata.get("dynare_commands", self.metadata.get("run", []))
        model = self
        results = RunResults(model=model)
        results._from_pipeline = True
        plot_variables: list[str] | None = None
        show_graph = True

        if not commands and default_pipeline:
            # Default pipeline: residuals + solve + IRFs
            results.residuals = model.residuals
            if not model.is_deterministic:
                dr = model.perturb()
                results.solution = dr
                results.eigenvalues = dr.evs
                results.moments = dr.moments()[1]
                results.simulation = dr.irfs(type="deviation", T=40)
            else:
                from dyno.solver import deterministic_solve

                sim = deterministic_solve(model, T=40)
                results.simulation = {"Perfect Foresight": sim}
                if hasattr(sim, "attrs") and not sim.attrs.get("converged", True):
                    results.add_warning(
                        f"Deterministic / perfect foresight simulation did not converge after "
                        f"{sim.attrs.get('iterations', '?')} iterations "
                        f"(maximum residual: {sim.attrs.get('residual', float('nan')):.2e}). "
                        "The computed solution is incorrect."
                    )
        else:
            for cmd in commands:
                name = str(cmd.get("command", "")).lower()
                options = cmd.get("options", {})
                try:
                    if name == "steady":
                        model = model.steady(**options)
                        results.model = model
                        results.steady_stats = getattr(model, "steady_stats", None)
                    elif name == "check":
                        results.residuals = model.residuals
                        model = model.check()
                        results.eigenvalues = getattr(model, "_eigenvalues", None)
                    elif name == "resid":
                        results.residuals = model.residuals
                    elif name in {"simul", "simulate"}:
                        from dyno.solver import deterministic_solve

                        det_opts = {k: v for k, v in options.items() if k != "mode"}
                        sim = deterministic_solve(model, **det_opts)
                        results.simulation = sim
                        if not sim.attrs.get("converged", True):
                            results.add_warning(
                                f"Deterministic / perfect foresight simulation did not converge after "
                                f"{sim.attrs.get('iterations', '?')} iterations "
                                f"(maximum residual: {sim.attrs.get('residual', float('nan')):.2e}). "
                                "The computed solution is incorrect."
                            )
                    elif name == "plot":
                        plot_variables = options.get("variables") or None
                    elif name == "stoch_simul":
                        dr = model.perturb()
                        results.solution = dr
                        results.eigenvalues = dr.evs
                        results.moments = dr.moments()[1]
                        irf_type = options.get("type", "deviation")
                        # ``periods`` is the simulation length in Dynare, not
                        # the IRF horizon.
                        horizon = int(options.get("irf", 40))
                        plot_variables = options.get("variables")
                        show_graph = not options.get("nograph", False) and horizon > 0
                        if horizon > 0:
                            results.simulation = dr.irfs(type=irf_type, T=horizon)
                except SteadyStateError as e:
                    results.residuals = e.residuals
                    results.steady_stats = getattr(e, "steady_stats", None)
                    results.add_warning(str(e))
                    break

        # Add line-level warnings for non-zero residuals where we can map lines.
        r = results.residuals
        if r is not None and r.size > 0 and abs(r).max() >= 1e-6:
            inds = np.where(abs(r) >= 1e-6)[0]
            eq_lines = model._equation_line_numbers()
            for i in inds:
                line = eq_lines[i] if i < len(eq_lines) else None
                results.add_warning(
                    f"Equation {i + 1}: residual = {float(r[i]):.3e}",
                    line=line,
                )

        if show_graph and results.simulation is not None:
            from dyno.plots import plot_simulation

            results._plot_options = {"engine": "altair"}
            if plot_variables:
                results._plot_options["variables"] = plot_variables
            results.figure = plot_simulation(
                results.simulation, engine="altair", variables=plot_variables
            )

        results.finish()
        return results

    def import_model(
        self: Self,
        txt: str,
        **kwargs: Any,
    ) -> None:
        """imports model written in `.mod` format into symbolic attribute using Dynare's preprocessor

        Parameters
        ----------
        txt : str
            the model being imported in `.mod` form
        deriv_order : int, optional
            derivative order, by default 1
        params_deriv_order : int, optional
            parameters derivative order, by default 0
        allow_undeclared_params : bool, optional
            if True, automatically declare parameters that are assigned values
            without being explicitly declared in the parameters section, by default False
        """
        Modfile, PreprocessorException = self._load_dynare_preprocessor()

        deriv_order: int = kwargs.get("deriv_order", 1)
        params_deriv_order: int = kwargs.get("params_deriv_order", 0)
        if "allow_undeclared_params" in kwargs:
            allow_undeclared_params: bool = kwargs["allow_undeclared_params"]
        else:
            allow_undeclared_params = not getattr(self, "strict", False)

        self._import_options = {
            "deriv_order": deriv_order,
            "params_deriv_order": params_deriv_order,
            "allow_undeclared_params": allow_undeclared_params,
        }

        # Keep original modfile text for rebuilding immutable model variants.
        self._original_txt = txt

        # Preprocess to declare undeclared parameters if needed
        if allow_undeclared_params:
            txt = self._declare_undeclared_params(txt)

        # The binding runs the preprocessor in a stochastic context, which
        # Dynare refuses to mix with ``perfect_foresight_solver``/``simul``.
        # Those statements are removed here and re-inserted as run commands.
        txt, self._perfect_foresight_statements = _extract_solver_statements(txt)

        try:
            self.symbolic = Modfile(txt, deriv_order, params_deriv_order)
        except PreprocessorException as e:
            raise DynareParserError(e) from e

    def _extract_dynare_commands(self: Self) -> list[dict[str, Any]]:
        """Extract Dynare statements/commands from the preprocessor JSON.

        Returns
        -------
        list[dict[str, Any]]
            Ordered list of command dictionaries with ``command`` and ``options``.
        """
        import json

        payload = json.loads(self.symbolic.json_string)
        transformed = payload.get("transformed_modfile", {})
        statements = transformed.get("statements", [])

        ignored = {"param_init", "initval"}

        commands: list[dict[str, Any]] = []
        for statement in statements:
            command = statement.get("statementName")
            if not isinstance(command, str):
                continue
            if command in ignored:
                continue

            options = statement.get("options", {})
            if not isinstance(options, dict):
                options = {}
            options = {k: _coerce_option(v) for k, v in options.items()}

            symbol_list = statement.get("symbol_list")
            if isinstance(symbol_list, list) and symbol_list:
                options = {**options, "variables": [str(s) for s in symbol_list]}

            commands.append({"command": command, "options": options})

        solver_statements = list(getattr(self, "_perfect_foresight_statements", []))
        if solver_statements:
            setups = [
                i
                for i, c in enumerate(commands)
                if c["command"] == "perfect_foresight_setup"
            ]
            at = setups[-1] + 1 if setups else len(commands)
            commands[at:at] = solver_statements

        from dyno.larkfiles import _translate_perfect_foresight_commands

        return _translate_perfect_foresight_commands(commands)

    def _deterministic_shocks(self: Self) -> dict[str, dict[int, float]]:
        """Forced exogenous paths from the ``shocks`` blocks of the preprocessor JSON.

        ``var x; periods p1:p2; values v;`` becomes ``{x: {t: v}}`` for
        ``t`` in ``p1..p2``; date 0 is the initial condition, as in .dyno files.
        """
        import json

        payload = json.loads(self.symbolic.json_string)
        statements = payload.get("transformed_modfile", {}).get("statements", [])
        constants = {
            k: v
            for k, v in self.symbolic.context.items()
            if k in self.symbolic.parameters
        }

        values: dict[str, dict[int, float]] = {}
        for statement in statements:
            if statement.get("statementName") != "shocks":
                continue
            for entry in statement.get("deterministic_shocks", []):
                series = values.setdefault(str(entry["var"]), {})
                for item in entry.get("values", []):
                    value = _evaluate_expression(str(item["value"]), constants)
                    for t in range(int(item["period1"]), int(item["period2"]) + 1):
                        series[t] = value
        return values

    def _declare_undeclared_params(self: Self, txt: str) -> str:
        """Automatically declare parameters that are assigned values without being declared.

        Parameters
        ----------
        txt : str
            the model text in `.mod` format

        Returns
        -------
        str
            the modified model text with undeclared parameters declared
        """
        import re

        # Remove comments from text for parsing
        txt_no_comments = re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL)
        txt_no_comments = re.sub(r"//.*?$", "", txt_no_comments, flags=re.MULTILINE)

        # Only look at declarations before the model block
        # Split by 'model;' and take only the declarations part
        model_split = re.split(r"\bmodel\s*;", txt_no_comments, flags=re.IGNORECASE)
        declarations_part = model_split[0] if model_split else txt_no_comments

        # Remove shocks block from declarations part if placed before model
        declarations_part = re.sub(
            r"\bshocks\s*;.*?end\s*;",
            "",
            declarations_part,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Split original text into lines for reconstruction
        lines = txt.split("\n")

        # Find all declared identifiers (vars, varexo, parameters)
        declared_identifiers = set()
        params_section_idx = None

        # Extract var declarations (only before model block)
        var_matches = re.findall(
            r"^\s*var\s+(.*?);\s*$", declarations_part, re.MULTILINE | re.IGNORECASE
        )
        for match in var_matches:
            declared_identifiers.update(re.findall(r"\b([a-zA-Z_]\w*)\b", match))

        # Extract varexo declarations (only before model block)
        varexo_matches = re.findall(
            r"^\s*varexo\s+(.*?);\s*$", declarations_part, re.MULTILINE | re.IGNORECASE
        )
        for match in varexo_matches:
            declared_identifiers.update(re.findall(r"\b([a-zA-Z_]\w*)\b", match))

        # Extract parameters declarations and find section (only before model block)
        params_matches = re.findall(
            r"^\s*parameters\s+(.*?);\s*$",
            declarations_part,
            re.MULTILINE | re.IGNORECASE,
        )
        for match in params_matches:
            declared_identifiers.update(re.findall(r"\b([a-zA-Z_]\w*)\b", match))

        # Find parameters section index in original file
        for i, line in enumerate(lines):
            if re.match(r"\s*parameters\s", line, re.IGNORECASE):
                params_section_idx = i
                break

        # Find parameter assignments (identifier = number;) that are not yet declared
        # Only look at assignments before model block
        undeclared = set()
        assignment_pattern = r"^\s*([a-zA-Z_]\w*)\s*=\s*[+-]?[\d.eE+-]+(?:\s*[;]|$)"

        for line in declarations_part.split("\n"):
            match = re.match(assignment_pattern, line)
            if match:
                param_name = match.group(1)
                if param_name not in declared_identifiers:
                    undeclared.add(param_name)

        # If there are undeclared parameters, add them to the declaration
        if undeclared:
            undeclared_list = ", ".join(sorted(undeclared))

            if params_section_idx is not None:
                # There's already a parameters section - append to it
                # Find the semicolon at the end of the parameters declaration
                j = params_section_idx
                while j < len(lines):
                    if ";" in lines[j]:
                        # Insert before the semicolon
                        lines[j] = lines[j].replace(";", f", {undeclared_list};", 1)
                        break
                    j += 1
            else:
                # No parameters section exists, create one after var/varexo declarations
                # Find the last var/varexo declaration (ignoring any inside shocks blocks)
                last_var_section = 0
                in_shocks = False
                for i, line in enumerate(lines):
                    stripped = line.strip().lower()
                    if stripped.startswith("shocks;"):
                        in_shocks = True
                    elif in_shocks and stripped.startswith("end;"):
                        in_shocks = False
                        continue
                    if not in_shocks and re.match(
                        r"\s*(var|varexo)\s", line, re.IGNORECASE
                    ):
                        last_var_section = i
                        # Find end of this declaration
                        j = i
                        while j < len(lines) and ";" not in lines[j]:
                            j += 1
                        if j < len(lines):
                            last_var_section = j

                # Insert parameters section after the last var/varexo
                insert_idx = last_var_section + 1
                lines.insert(insert_idx, f"\nparameters {undeclared_list};")

        return "\n".join(lines)

    def _set_context(self: Self) -> None:
        """retrieves calibration values"""

        c = self.symbolic.context  # dynare preprocessor context
        endogenous = self.symbolic.endogenous
        exogenous = self.symbolic.exogenous
        variables = endogenous + exogenous
        parameters = self.symbolic.parameters

        steady_states = {
            k: v for (k, v) in c.items() if (k in endogenous) or (k in exogenous)
        }
        constants = {k: v for (k, v) in c.items() if (k in parameters)}

        commands = self._extract_dynare_commands()
        deterministic_shocks = self._deterministic_shocks()

        # A file is a perfect-foresight model when it forces exogenous paths
        # or uses the perfect-foresight commands, and declares no shock
        # variances nor calls stoch_simul (Dynare refuses to mix the two).
        names = {c["command"] for c in commands}
        isdeterministic = (bool(deterministic_shocks) or "simul" in names) and not (
            len(self.symbolic.covariances) > 0 or "stoch_simul" in names
        )
        exo = exogenous

        values: dict[str, dict[int, float]]
        if isdeterministic:
            values = deterministic_shocks
            processes = {}
        else:
            n = len(exo)
            covar = np.zeros((n, n))
            index = {name: i for (i, name) in enumerate(exo)}
            for (var1, var2), val in self.symbolic.covariances.items():
                covar[index[var1], index[var2]] = val
                covar[index[var2], index[var1]] = val
            # self.processes =
            values = {}
            processes = {tuple(exo): Normal(Σ=covar)}

        context = {
            "constants": constants,
            "variables": {v: {} for v in variables},
            "values": values,
            "processes": processes,
            "steady_states": steady_states,
            "metadata": {
                "dynare_commands": commands,
            },
        }
        # self.paths = None
        # self.exogenous = self.processes
        self.context = cast(ModelContext, context)

    @property
    def equations(self):

        return self.symbolic.equations

    def compute_residuals(self, y1, y2, y3, e):
        p = [self.context["constants"][p] for p in self.symbolic.parameters]
        y, e = self.__steady_state_vectors__
        return self._f_dynamic(y, y, y, e, p)

    def compute_jacobians(self, y1, y2, y3, e):
        p = [self.context["constants"][p] for p in self.symbolic.parameters]
        y, e = self.__steady_state_vectors__
        return self._f_dynamic(y, y, y, e, p, diff=True)

    def compute_derivatives(self: Self):

        y = [self.steady_state[v] for v in self.symbols["endogenous"]]
        e = [self.steady_state[v] for v in self.symbols["exogenous"]]
        p = [self.context["constants"][v] for v in self.symbols["parameters"]]

        return self.symbolic.derivatives(y, y, y, e, e, p)

    def _dynamic_point(self, v_prev, v_curr, v_next, diff=False):
        q = len(self.symbols["endogenous"])
        p = [self.context["constants"][name] for name in self.symbolic.parameters]
        e = v_curr[q:]
        if not diff:
            return np.asarray(self._f_dynamic(v_next[:q], v_curr[:q], v_prev[:q], e, p))
        r, A, B, C, D = self._f_dynamic(
            v_next[:q], v_curr[:q], v_prev[:q], e, p, diff=True
        )
        n_vars = len(self.symbols["variables"])
        J = np.zeros((len(r), n_vars, 3))
        J[:, :q, 0] = C
        J[:, :q, 1] = B
        J[:, :q, 2] = A
        J[:, q:, 1] = D
        return np.asarray(r), J

    def _f_dynamic(
        self: Self,
        y0: TVector,
        y1: TVector,
        y2: TVector,
        e: TVector,
        p: TVector,
        diff: bool = False,
    ) -> TVector | tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
        """function f describing the behavior of the dynamic system $f(y_{t+1}, y_t, y_{t-1}, ε_t, p) = 0$

        Parameters
        ----------
        y0,y1,y2 : Vector
            the system's endogenous variable values at times t+1, t and t-1 respectively
        e : Vector
            exogenous variable values
        p : Vector
            parameter values
        diff : bool, optional
            if set to True returns the function's partial derivatives with regards to y0, y1, y2 and e as well, by default False

        Returns
        -------
        Vector|tuple[Vector, Matrix, Matrix, Matrix, Matrix]
            value of f(y0, y1, y2, e, p), as well as partial derivatives w.r.t. y0, y1, y2 and e if diff is set to True
        """

        # (endo_future, endo_present, endo_past, exo, exo_det, params).
        # ``varexo_det`` variables are not supported, so that slot is empty.
        args: list[list[Any]] = [
            list(y0),
            list(y1),
            list(y2),
            list(e),
            [],
            list(p),
        ]

        r = np.array(self.symbolic.residuals(*args))

        if diff:
            jacobians = self.symbolic.jacobians(*args)
            del jacobians[4]
            n = len(self.equations)
            lengths = [n] * 3 + [len(e), len(p)]
            r1, r2, r3, r4 = [
                sparse_to_dense(n, length, j)
                for (j, length) in zip(jacobians[:-1], lengths)
            ]
            return r, r1, r2, r3, r4

        return r


_SOLVER_STATEMENT_RE = re.compile(
    r"\b(perfect_foresight_solver|simul)\s*(\(([^)]*)\))?\s*;",
    re.IGNORECASE,
)


def _extract_solver_statements(txt: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove ``perfect_foresight_solver``/``simul`` statements from *txt*.

    Returns the stripped text and the statements as run commands, in order.
    """
    statements: list[dict[str, Any]] = []

    def _strip(match: "re.Match[str]") -> str:
        options: dict[str, Any] = {}
        for item in (match.group(3) or "").split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                key, value = item.split("=", 1)
                options[key.strip()] = _coerce_option(value.strip())
            else:
                options[item] = True
        statements.append({"command": match.group(1).lower(), "options": options})
        return ""

    return _SOLVER_STATEMENT_RE.sub(_strip, txt), statements


def _evaluate_expression(expr: str, constants: dict[str, Any]) -> float:
    """Evaluate a numeric expression written by the preprocessor (``values``)."""
    import math

    try:
        return float(expr)
    except ValueError:
        pass
    namespace: dict[str, Any] = {
        name: getattr(math, name)
        for name in ("exp", "log", "sqrt", "sin", "cos", "tan", "pi")
    }
    namespace["abs"] = abs
    namespace.update(constants)
    return float(eval(expr, {"__builtins__": {}}, namespace))  # noqa: S307


def _coerce_option(value: Any) -> Any:
    """The preprocessor JSON stores option values as strings; parse numbers."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def sparse_to_dense(
    lines: int, cols: int, sparse: dict[tuple[int, int], float]
) -> TMatrix:
    res = np.zeros((lines, cols))
    for (i, j), v in sparse.items():
        res[i, j] = v
    return res
