from dyno.model import AbstractModel
from dyno.language import pad_list, Normal, Deterministic
import numpy as np
from scipy.optimize import root

from typing_extensions import Self
from typing import Any
from .typedefs import TVector, TMatrix

from dynare_preprocessor import PreprocessorException, UnsupportedFeatureException
from .errors import DynareParserError, SteadyStateError


class DynareModel(AbstractModel):

    def _rebuild(self: Self) -> Self:
        txt = getattr(self, "_original_txt", None)
        if txt is None:
            txt = open(self.filename, "rt", encoding="utf-8").read()

        options = getattr(self, "_import_options", {})
        model = self.__class__(filename=self.filename, txt=txt, **options)

        previous = getattr(self, "_calibration_overrides", {})
        if len(previous) > 0:
            model = model.recalibrate(**previous)

        return model

    def recalibrate(self: Self, **calib):
        m = self._rebuild()

        known = set(m.context["constants"].keys()) | set(m.context["steady_states"].keys())
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

    def run(self: Self) -> "DynareRunResults":
        commands = self.metadata.get("dynare_commands", self.metadata.get("run", []))
        model = self
        results = DynareRunResults(model=model)

        for cmd in commands:
            name = cmd["command"]
            if name == "steady":
                model = model.steady()
                results.model = model
            elif name == "check":
                model.check()
            elif name == "resid":
                model.check()
            elif name == "stoch_simul":
                results.solution = model.perturb()

        return results

    def steady(self: Self, tol: float = 1e-10, maxiter: int = 100) -> Self:
        endogenous = self.symbols["endogenous"]
        if len(endogenous) == 0:
            return self.copy()

        y0, _ = self.__steady_state_vectors__
        guess = np.nan_to_num(np.asarray(y0, dtype=float), nan=1.0)

        def _candidate(values: np.ndarray) -> Self:
            calib = {
                name: float(values[i])
                for i, name in enumerate(endogenous)
            }
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

        sol = root(_fun, guess, jac=_jac, method="hybr", options={"maxfev": maxiter})

        solved = _candidate(np.asarray(sol.x, dtype=float))
        residuals = np.asarray(solved.residuals, dtype=float)

        if (not sol.success) or (not np.isfinite(residuals).all()) or (np.max(np.abs(residuals)) > tol):
            raise SteadyStateError(residuals)

        return solved

    def import_model(self: Self, txt: str, deriv_order=1, params_deriv_order=0, allow_undeclared_params=False) -> None:
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
        from dynare_preprocessor import DynareModel as Modfile

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

            commands.append({"command": command, "options": options})

        return commands

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
        txt_no_comments = re.sub(r'/\*.*?\*/', '', txt, flags=re.DOTALL)
        txt_no_comments = re.sub(r'//.*?$', '', txt_no_comments, flags=re.MULTILINE)
        
        # Only look at declarations before the model block
        # Split by 'model;' and take only the declarations part
        model_split = re.split(r'\bmodel\s*;', txt_no_comments, flags=re.IGNORECASE)
        declarations_part = model_split[0] if model_split else txt_no_comments
        
        # Split original text into lines for reconstruction
        lines = txt.split('\n')
        
        # Find all declared identifiers (vars, varexo, parameters)
        declared_identifiers = set()
        params_section_idx = None
        
        # Extract var declarations (only before model block)
        var_matches = re.findall(r'^\s*var\s+(.*?);\s*$', declarations_part, re.MULTILINE | re.IGNORECASE)
        for match in var_matches:
            declared_identifiers.update(re.findall(r'\b([a-zA-Z_]\w*)\b', match))
        
        # Extract varexo declarations (only before model block)
        varexo_matches = re.findall(r'^\s*varexo\s+(.*?);\s*$', declarations_part, re.MULTILINE | re.IGNORECASE)
        for match in varexo_matches:
            declared_identifiers.update(re.findall(r'\b([a-zA-Z_]\w*)\b', match))
        
        # Extract parameters declarations and find section (only before model block)
        params_matches = re.findall(r'^\s*parameters\s+(.*?);\s*$', declarations_part, re.MULTILINE | re.IGNORECASE)
        for match in params_matches:
            declared_identifiers.update(re.findall(r'\b([a-zA-Z_]\w*)\b', match))
        
        # Find parameters section index in original file
        for i, line in enumerate(lines):
            if re.match(r'\s*parameters\s', line, re.IGNORECASE):
                params_section_idx = i
                break
        
        # Find parameter assignments (identifier = number;) that are not yet declared
        # Only look at assignments before model block
        undeclared = set()
        assignment_pattern = r'^\s*([a-zA-Z_]\w*)\s*=\s*[+-]?[\d.eE+-]+(?:\s*[;]|$)'
        
        for line in declarations_part.split('\n'):
            match = re.match(assignment_pattern, line)
            if match:
                param_name = match.group(1)
                if param_name not in declared_identifiers:
                    undeclared.add(param_name)
        
        # If there are undeclared parameters, add them to the declaration
        if undeclared:
            undeclared_list = ', '.join(sorted(undeclared))
            
            if params_section_idx is not None:
                # There's already a parameters section - append to it
                # Find the semicolon at the end of the parameters declaration
                j = params_section_idx
                while j < len(lines):
                    if ';' in lines[j]:
                        # Insert before the semicolon
                        lines[j] = lines[j].replace(';', f', {undeclared_list};', 1)
                        break
                    j += 1
            else:
                # No parameters section exists, create one after var/varexo declarations
                # Find the last var/varexo declaration
                last_var_section = 0
                for i, line in enumerate(lines):
                    if re.match(r'\s*(var|varexo)\s', line, re.IGNORECASE):
                        last_var_section = i
                        # Find end of this declaration
                        j = i
                        while j < len(lines) and ';' not in lines[j]:
                            j += 1
                        if j < len(lines):
                            last_var_section = j
                
                # Insert parameters section after the last var/varexo
                insert_idx = last_var_section + 1
                lines.insert(insert_idx, f'\nparameters {undeclared_list};')
        
        return '\n'.join(lines)

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

        # read specification of exogenous shocks in the modfile
        assert len(self.symbolic.trajectories) == 0 or len(self.symbolic.covariances) == 0
        isdeterministic = len(self.symbolic.trajectories) > 0
        exo = exogenous

        if isdeterministic:
            det_vals = {v: [] for v in exo}
            for var, traj in self.symbolic.trajectories.items():
                for p1, p2, val in traj:
                    pad_list(det_vals[var], p2)
                    det_vals[var][p1 - 1 : p2] = [val] * (p2 - p1 + 1)
            # self.paths = Deterministic(det_vals)
            # self.processes = None
            # self.exogenous = self.paths
            values = det_vals
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
                "dynare_commands": self._extract_dynare_commands(),
            }
        }
        # self.paths = None
        # self.exogenous = self.processes
        self.context = context

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

    def compute_derivatives(model):

        y = [model.steady_state[v] for v in model.symbols["endogenous"]]
        e = [model.steady_state[v] for v in model.symbols["exogenous"]]
        p = [model.context["constants"][v] for v in model.symbols["parameters"]]

        return model.symbolic.derivatives(y, y, y, e, e, p)

    def deterministic_residuals(self, v):

        return v * 0

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

        y0 = list(y0)
        y1 = list(y1)
        y2 = list(y2)
        e = list(e)
        p = list(p)

        args = [y0, y1, y2, e, e, p]
        if len(self.context["processes"]) == 0:
            # this is a stochastic model
            args[3] = []
        else:
            args[4] = []
            # this is a deterministic model

        r = np.array(self.symbolic.residuals(*args))

        if diff:
            jacobians = self.symbolic.jacobians(*args)
            if len(self.context["processes"]) == 0:
                del jacobians[3]
            else:
                del jacobians[4]
            n = len(self.equations)
            lengths = [n] * 3 + [len(e), len(p)]
            r1, r2, r3, r4 = [
                sparse_to_dense(n, length, j)
                for (j, length) in zip(jacobians[:-1], lengths)
            ]
            return r, r1, r2, r3, r4

        return r


def sparse_to_dense(
    lines: int, cols: int, sparse: dict[tuple[int, int], float]
) -> TMatrix:
    res = np.zeros((lines, cols))
    for (i, j), v in sparse.items():
        res[i, j] = v
    return res


class DynareRunResults:
    """Container returned by DynareModel.run().

    Attributes
    ----------
    model : DynareModel
        The model state after processing (updated by ``steady`` commands).
    solution : PerturbationSolution | None
        The perturbation solution produced by a ``stoch_simul`` command,
        or ``None`` if none was requested.
    """

    def __init__(self, model: "DynareModel", solution=None) -> None:
        self.model = model
        self.solution = solution

    def __repr__(self) -> str:
        parts = [f"model={self.model.name!r}"]
        if self.solution is not None:
            parts.append("solution=<PerturbationSolution>")
        return f"DynareRunResults({', '.join(parts)})"
