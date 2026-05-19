from dyno.model import AbstractModel
import copy
import warnings
import yaml
import math
from .typedefs import TVector, TMatrix, IRFType, Solver, DynamicFunction

from dyno.dynsym.grammar import parser, str_expression
from dyno.dynsym.analyze import FormulaEvaluator
from typing import Any
from typing_extensions import Self

import numpy as np
from scipy.optimize import root

from .errors import LARKParserError, ParserError, SteadyStateError
from lark.exceptions import UnexpectedInput
from dyno.larkfiles import DynoFile, LModFile
from dyno.language import ProductNormal, Normal


class DynoModel(AbstractModel):

    def _normalize_run_commands(self: Self) -> list[dict[str, Any]]:
        raw = self.metadata.get("run")
        if raw is None:
            raw = self.metadata.get("dynare_commands", [])

        if isinstance(raw, str):
            items = [raw]
        elif isinstance(raw, dict):
            items = [raw]
        elif isinstance(raw, list):
            items = raw
        else:
            raise TypeError("model.metadata['run'] must be a string, dict or a list")

        commands: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                commands.append({"command": item, "options": {}})
            elif isinstance(item, dict):
                if "command" in item:
                    command = item.get("command")
                    if not isinstance(command, str):
                        raise TypeError(
                            "run command dictionaries must define a string 'command'"
                        )
                    options = item.get("options", {})
                    if not isinstance(options, dict):
                        raise TypeError("run command 'options' must be a dictionary")
                    commands.append({"command": command, "options": options})
                elif len(item) == 1:
                    command, options = next(iter(item.items()))
                    if not isinstance(command, str):
                        raise TypeError("run command keys must be strings")
                    if options is None:
                        options = {}
                    if not isinstance(options, dict):
                        raise TypeError(
                            "compact run command options must be a dictionary or null"
                        )
                    commands.append({"command": command, "options": options})
                else:
                    raise TypeError(
                        "run command dictionaries must use either {'command': ...} or a single-key form like {'simul': {...}}"
                    )
            else:
                raise TypeError("run commands must be strings or dictionaries")

        return commands

    def run(self: Self, default_pipeline: bool = False) -> "RunResults":
        from .report import RunResults

        warnings.warn(
            "DynoModel.run() is experimental and its command metadata format may change.",
            UserWarning,
            stacklevel=2,
        )

        commands = self._normalize_run_commands()
        model = self
        results = RunResults(model=model)

        if not commands and default_pipeline:
            # Default pipeline: residuals + solve + IRFs
            results.residuals = model.residuals
            if model.is_deterministic:
                from .solver import deterministic_solve

                sim = deterministic_solve(model, T=40)
                results.simulation = {"Perfect Foresight": sim}
            else:
                dr = model.solve()
                results.solution = dr
                results.eigenvalues = dr.evs
                results.moments = dr.moments()[1]
                results.simulation = dr.irfs(type="deviation", T=40)
        else:
            for cmd in commands:
                name = str(cmd["command"]).lower()
                options = cmd.get("options", {})

                if name == "steady":
                    model = model.steady(**options)
                    results.model = model
                elif name in {"check", "resid"}:
                    results.residuals = model.residuals
                    check_options = {"compute_eigenvalues": True, **options}
                    model = model.check(**check_options)
                    results.model = model
                    results.eigenvalues = getattr(model, "_eigenvalues", None)
                elif name in {"solve", "perturb"}:
                    solution = model.solve(**options)
                    results.solution = solution
                    results.eigenvalues = getattr(solution, "evs", None)
                elif name in {"simul", "simulate", "stoch_simul"}:
                    if model.is_deterministic:
                        from .solver import deterministic_solve

                        sim = deterministic_solve(model, **options)
                        results.simulation = {"Perfect Foresight": sim}
                    else:
                        from .simul import simulate

                        solve_options = {
                            k: v
                            for k, v in options.items()
                            if k not in {"T", "irf", "periods"}
                        }
                        solution = results.solution
                        if solution is None or not hasattr(solution, "X"):
                            solution = model.solve(**solve_options)
                            results.solution = solution
                        results.eigenvalues = getattr(solution, "evs", None)
                        if name == "stoch_simul":
                            irf_type = options.get("type", "deviation")
                            horizon = int(
                                options.get("irf", options.get("periods", 40))
                            )
                            results.moments = solution.moments()[1]
                            results.simulation = solution.irfs(type=irf_type, T=horizon)
                        else:
                            horizon = int(options.get("T", 40))
                            results.simulation = simulate(solution, T=horizon)
                else:
                    raise NotImplementedError(
                        f"Unsupported DynoModel.run command: {name}"
                    )

        # Add line-level warnings for non-zero residuals
        r = results.residuals
        if r is not None and abs(r).max() >= 1e-6:
            inds = np.where(abs(r) >= 1e-6)[0]
            for i in inds:
                tree = model.symbolic.equations[i]
                results.add_warning(str(r[i]), line=tree.meta.line)

        if results.simulation is not None and isinstance(results.simulation, dict):
            from .plots import plot_irfs

            results.figure = plot_irfs(results.simulation)

        results.finish()
        return results

    def _constants_used_in_equations(self: Self) -> set[str]:
        names: set[str] = set()
        for eq in self.symbolic.equations:
            for subtree in eq.iter_subtrees_topdown():
                if subtree.data == "constant":
                    names.add(str(subtree.children[0].children[0]))
        return names

    def __init__(
        self: Self,
        filename: str | None = None,
        txt: str | None = None,
        yaml: str | None = None,
        **kwargs,
    ) -> None:
        if yaml is not None:
            if txt is not None:
                raise ValueError("Pass either `txt` or `yaml`, not both.")
            txt = yaml
            if filename is None:
                filename = "*anonymous*.yaml"

        super().__init__(filename=filename, txt=txt, **kwargs)

    def import_model(self: Self, txt: str, **kwargs) -> None:

        try:
            if self.filename.endswith(".mod"):
                self.symbolic = LModFile(content=txt, filename=self.filename)
            elif self.filename.endswith(".yaml") or self.filename.endswith(".yml"):
                data = yaml.safe_load(txt)
                if not isinstance(data, dict):
                    raise ValueError(
                        "YAML model input must be a mapping with a `model` key."
                    )

                model_txt = data.get("model")
                if not isinstance(model_txt, str):
                    raise ValueError(
                        "YAML model input must provide a string `model` block."
                    )

                self.symbolic = DynoFile(content=model_txt, filename=self.filename)

                yaml_metadata = {k: v for (k, v) in data.items() if k != "model"}
                merged_metadata = yaml_metadata | self.symbolic.metadata
                self.symbolic.metadata = merged_metadata
            elif self.filename.endswith(".dyno"):
                self.symbolic = DynoFile(content=txt, filename=self.filename)
            else:
                self.symbolic = DynoFile(content=txt, filename=self.filename)
        except UnexpectedInput as e:
            raise LARKParserError(e, txt) from e

    def _set_context(self: Self) -> None:
        context = self.symbolic.context.copy()

        constants = context.setdefault("constants", {})
        steady_states = context.setdefault("steady_states", {})
        variables = context.setdefault("variables", {})

        # Ensure constants used in equations are explicit in context, even if
        # they were never assigned in declarations.
        for name in self._constants_used_in_equations():
            constants.setdefault(name, math.nan)

        # Ensure every referenced variable has a steady-state entry.
        for name in variables.keys():
            steady_states.setdefault(name, math.nan)

        self.context = context

    def recalibrate(self: Self, **calib):
        m = self.copy()
        previous = getattr(self, "_calibration_overrides", {})
        merged = previous | calib
        m.symbolic.context = {}
        m.symbolic.process_assignments(**merged)
        m._set_context()
        m._set_exogenous()
        m._calibration_overrides = merged
        return m

    @property
    def equations(self):

        return [str_expression(eq) for eq in self.symbolic.equations]

    def latex_equations(self):

        return self.symbolic.latex_equations()


    def compute_residuals(self, y2, y1, y0, e):

        endogenous = self.symbols["endogenous"]
        exogenous = self.symbols["exogenous"]

        cc = copy.deepcopy(self.symbolic.context)

        for i, name in enumerate(endogenous):
            cc["variables"][name] = {
                -1: cc["steady_states"].get(name, math.nan),
                0: cc["steady_states"].get(name, math.nan),
                1: cc["steady_states"].get(name, math.nan),
            }

        for i, name in enumerate(exogenous):
            cc["variables"][name] = {0: 0.0}

        from dyno.dynsym.analyze import EquationsEvaluator

        E = EquationsEvaluator(cc)

        results = [E.visit(eq) for eq in self.symbolic.equations]

        r = np.array([float(el) for el in results])

        return r

    def compute_jacobians(
        self, y2, y1, y0, e
    ) -> tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix, TMatrix]:

        from dyno.dynsym.autodiff import DNumber as DN

        endogenous = self.symbols["endogenous"]
        exogenous = self.symbols["exogenous"]

        cc = copy.deepcopy(self.symbolic.context)

        for i, name in enumerate(endogenous):
            cc["variables"][name] = {
                -1: DN(y0[i], {(name, -1): 1}),
                0: DN(y1[i], {(name, 0): 1}),
                1: DN(y2[i], {(name, 1): 1}),
            }

        for i, name in enumerate(exogenous):
            cc["variables"][name] = {0: DN(e[i], {(name, 0): 1})}

        from dyno.dynsym.analyze import EquationsEvaluator

        E = EquationsEvaluator(cc)
        results = [E.visit(eq) for eq in self.symbolic.equations]

        neq = len(results)
        nv = len(endogenous)
        ne = len(exogenous)

        r = np.array([el.value for el in results])
        A = np.zeros((neq, nv))
        B = np.zeros((neq, nv))
        C = np.zeros((neq, nv))
        J = [A, B, C]
        D = np.zeros((neq, ne))

        for n, eq in enumerate(results):
            for (name, shift), v in eq.derivatives.items():
                if name in endogenous:
                    i = endogenous.index(name)
                    J[1 - shift][n, i] = v
                elif name in exogenous:
                    i = exogenous.index(name)
                    D[n, i] = v

        return r, A, B, C, D

    def deterministic_residuals(model, v, jac=False, **kwargs):

        if jac:
            return model.deterministic_residuals_with_jacobian(v, **kwargs)

        flat = v.ndim == 1

        p = len(model.symbols["variables"])
        T = int(np.prod(v.shape) / p - 1)

        v = v.reshape((T + 1, p))

        # For t = 0 to T-3: standard forward/backward indexing
        v_f = np.concatenate([v[1:, :], v[-1, :][None, :]], axis=0)
        v_b = np.concatenate([v[0, :][None, :], v[:-1, :]], axis=0)

        cc = copy.deepcopy(model.symbolic.context)
        for i, name in enumerate(model.symbols["variables"]):
            cc["variables"][name] = {-1: v_b[:, i], 0: v[:, i], 1: v_f[:, i]}

        from dyno.dynsym.analyze import EquationsEvaluator

        E = EquationsEvaluator(cc)

        results = [E.visit(eq) for eq in model.symbolic.equations]

        # number of variables not pinned down by dynamic equations
        n_exo = len(model.symbols["variables"]) - len(results)

        # the following works if there is one and exactly one exogenous variable
        # assert n_exo == 1

        y, e = model.__steady_state_vectors__
        a = np.concatenate([y, e])
        v1 = a[None, :].repeat(T + 1, axis=0)
        for key, value in model.context["values"].items():
            i = model.symbols["variables"].index(key)
            for a, b in value.items():
                v1[a, i] = b

        v1_exo = v1[:, -n_exo:].copy()
        diff_exo = v[:, -n_exo:] - v1_exo

        res = np.column_stack(results + [diff_exo])

        res[0, :] = v[0, :] - v1[0, :]  # initial condition

        # Override last row with explicit terminal condition
        if T >= 1:
            # t = T: f(v_T, v_T, v_T)
            cc_T = copy.deepcopy(model.symbolic.context)
            for i, name in enumerate(model.symbols["variables"]):
                cc_T["variables"][name] = {-1: v[T, i], 0: v[T, i], 1: v[T, i]}
            E_T = EquationsEvaluator(cc_T)
            res_T = [E_T.visit(eq) for eq in model.symbolic.equations]
            res[T, : len(res_T)] = res_T
            res[T, len(res_T) :] = v[T, -n_exo:] - v1_exo[T, :]

        if flat:
            return res.ravel()
        else:
            return res

    def deterministic_residuals_with_jacobian(model, v, sparsify=False):

        from dyno.dynsym.autodiff import DNumber

        flat = v.ndim == 1

        p = len(model.symbols["variables"])
        T = int(np.prod(v.shape) / p - 1)

        v = v.reshape((T + 1, p))

        v_f = np.concatenate([v[1:, :], v[-1, :][None, :]], axis=0)
        v_b = np.concatenate([v[0, :][None, :], v[:-1, :]], axis=0)

        context = copy.deepcopy(model.symbolic.context)
        for i, name in enumerate(model.symbols["variables"]):
            context["variables"][name] = {
                -1: DNumber(v_b[:, i], {(name, -1): 1.0}),
                0: DNumber(v[:, i], {(name, 0): 1.0}),
                1: DNumber(v_f[:, i], {(name, 1): 1.0}),
            }

        from dyno.dynsym.analyze import EquationsEvaluator

        E = EquationsEvaluator(context)

        results = [E.visit(eq) for eq in model.symbolic.equations]

        n_exo = len(model.symbols["variables"]) - len(results)

        y, e = model.__steady_state_vectors__

        # get exo values
        # works if the is one and exactly one exogenous variable?
        # does it?
        v1 = np.concatenate([y, e])[None, :].repeat(T + 1, axis=0)
        for key, value in model.symbolic.context["values"].items():
            i = model.symbols["variables"].index(key)
            for a, b in value.items():
                v1[a, i] = b

        exo = v1[:, -n_exo:].copy()

        res = np.column_stack([e.value for e in results] + [v[:, -n_exo:] - exo])

        res[0, :] = v[0, :] - v1[0, :]  # initial condition

        N = v.shape[0]

        p = len(model.symbols["variables"])
        q = len(model.equations)

        D = np.zeros((N, q, p, 3))  # would be easier with 4d struct

        for i_q in range(q):

            for k, deriv in results[i_q].derivatives.items():
                s, t = k  # symbol, time
                i_var = model.symbols["variables"].index(s)
                D[:, i_q, i_var, t + 1] = deriv

        # add exogenous equations
        DD = np.zeros((N, p, p, 3))
        DD[:, :q, :, :] = D
        for i in range(q, p):
            DD[:, i, i, 1] = 1.0

        # Override last two rows with explicit terminal conditions
        # Override last row with explicit terminal condition
        # NOTE: Terminal condition uses all time indices pointing to v_T
        terminal_derivatives = {}  # Maps (row, col_block) -> derivative matrix [p x p]

        if T >= 1:
            # t = T: f(v_T, v_T, v_T)
            context_T = copy.deepcopy(model.symbolic.context)
            for i, name in enumerate(model.symbols["variables"]):
                context_T["variables"][name] = {
                    -1: DNumber(v[T, i], {(name, -1): 1.0}),
                    0: DNumber(v[T, i], {(name, 0): 1.0}),
                    1: DNumber(v[T, i], {(name, 1): 1.0}),
                }
            E_T = EquationsEvaluator(context_T)
            results_T = [E_T.visit(eq) for eq in model.symbolic.equations]
            res[T, :q] = [e.value for e in results_T]
            res[T, q:] = v[T, -n_exo:] - exo[T, :]

            # Store derivatives explicitly
            deriv_T_T = np.zeros((p, p))
            for i_q in range(q):
                for k, deriv in results_T[i_q].derivatives.items():
                    s, t = k  # symbol, time
                    i_var = model.symbols["variables"].index(s)
                    # All time indices map to v_T
                    deriv_T_T[i_q, i_var] += deriv
            for i in range(q, p):
                deriv_T_T[i, i] = 1.0
            terminal_derivatives[(T, T)] = deriv_T_T

            # Clear DD[T] since we're handling it specially
            DD[T, :, :, :] = 0.0

        if not flat:
            return res, DD
        else:
            if sparsify:
                J = _build_sparse_jacobian(N, p, DD, terminal_derivatives)
            else:
                J = _build_dense_jacobian(N, p, DD, terminal_derivatives)

            return res.ravel(), J


def _build_sparse_jacobian(
    N: int, p: int, DD: np.ndarray, terminal_derivatives: dict
) -> "scipy.sparse.csr_matrix":
    import scipy.sparse

    # Build sparse matrix directly using COO format with vectorized operations
    # Pre-allocate lists with estimated size
    max_nnz = p + (N - 2) * 3 * p * p + 2 * p * p  # upper bound on non-zeros
    row_indices = np.empty(max_nnz, dtype=np.int32)
    col_indices = np.empty(max_nnz, dtype=np.int32)
    data_vals = np.empty(max_nnz, dtype=DD.dtype)

    idx = 0

    # First block: identity matrix (n=0)
    row_indices[idx : idx + p] = np.arange(p)
    col_indices[idx : idx + p] = np.arange(p)
    data_vals[idx : idx + p] = 1.0
    idx += p

    # Middle blocks (n=1 to N-3) - standard treatment
    for n in range(1, max(1, N - 2)):
        base_row = p * n

        # Previous time block (DD[n,:,:,0])
        mask0 = DD[n, :, :, 0] != 0
        nnz0 = np.sum(mask0)
        if nnz0 > 0:
            ii, jj = np.nonzero(mask0)
            row_indices[idx : idx + nnz0] = base_row + ii
            col_indices[idx : idx + nnz0] = p * (n - 1) + jj
            data_vals[idx : idx + nnz0] = DD[n, :, :, 0][mask0]
            idx += nnz0

        # Current time block (DD[n,:,:,1])
        mask1 = DD[n, :, :, 1] != 0
        nnz1 = np.sum(mask1)
        if nnz1 > 0:
            ii, jj = np.nonzero(mask1)
            row_indices[idx : idx + nnz1] = base_row + ii
            col_indices[idx : idx + nnz1] = p * n + jj
            data_vals[idx : idx + nnz1] = DD[n, :, :, 1][mask1]
            idx += nnz1

        # Next time block (DD[n,:,:,2])
        mask2 = DD[n, :, :, 2] != 0
        nnz2 = np.sum(mask2)
        if nnz2 > 0:
            ii, jj = np.nonzero(mask2)
            row_indices[idx : idx + nnz2] = base_row + ii
            col_indices[idx : idx + nnz2] = p * (n + 1) + jj
            data_vals[idx : idx + nnz2] = DD[n, :, :, 2][mask2]
            idx += nnz2

    # Special handling for last three blocks (T-2, T-1, T)
    if N >= 3:
        # Block n=N-2 (t=T-2): has standard structure with 3 blocks
        n = N - 2
        base_row = p * n

        mask0 = DD[n, :, :, 0] != 0
        nnz0 = np.sum(mask0)
        if nnz0 > 0:
            ii, jj = np.nonzero(mask0)
            row_indices[idx : idx + nnz0] = base_row + ii
            col_indices[idx : idx + nnz0] = p * (n - 1) + jj
            data_vals[idx : idx + nnz0] = DD[n, :, :, 0][mask0]
            idx += nnz0

        mask1 = DD[n, :, :, 1] != 0
        nnz1 = np.sum(mask1)
        if nnz1 > 0:
            ii, jj = np.nonzero(mask1)
            row_indices[idx : idx + nnz1] = base_row + ii
            col_indices[idx : idx + nnz1] = p * n + jj
            data_vals[idx : idx + nnz1] = DD[n, :, :, 1][mask1]
            idx += nnz1

        mask2 = DD[n, :, :, 2] != 0
        nnz2 = np.sum(mask2)
        if nnz2 > 0:
            ii, jj = np.nonzero(mask2)
            row_indices[idx : idx + nnz2] = base_row + ii
            col_indices[idx : idx + nnz2] = p * (n + 1) + jj
            data_vals[idx : idx + nnz2] = DD[n, :, :, 2][mask2]
            idx += nnz2

        # Block n=N-1 (t=T-1): derivatives wrt v_{T-1} and v_T
        n = N - 1
        base_row = p * n

        mask0 = DD[n, :, :, 0] != 0
        nnz0 = np.sum(mask0)
        if nnz0 > 0:
            ii, jj = np.nonzero(mask0)
            row_indices[idx : idx + nnz0] = base_row + ii
            col_indices[idx : idx + nnz0] = p * (n - 1) + jj
            data_vals[idx : idx + nnz0] = DD[n, :, :, 0][mask0]
            idx += nnz0

        mask1 = DD[n, :, :, 1] != 0
        nnz1 = np.sum(mask1)
        if nnz1 > 0:
            ii, jj = np.nonzero(mask1)
            row_indices[idx : idx + nnz1] = base_row + ii
            col_indices[idx : idx + nnz1] = p * n + jj
            data_vals[idx : idx + nnz1] = DD[n, :, :, 1][mask1]
            idx += nnz1

        # Note: DD[n, :, :, 2] should be zero for N-1 since we already combined derivatives
    else:
        # Fallback for small N
        for n in range(max(1, N - 2), N):
            base_row = p * n

            mask0 = DD[n, :, :, 0] != 0
            nnz0 = np.sum(mask0)
            if nnz0 > 0:
                ii, jj = np.nonzero(mask0)
                row_indices[idx : idx + nnz0] = base_row + ii
                col_indices[idx : idx + nnz0] = p * (n - 1) + jj
                data_vals[idx : idx + nnz0] = DD[n, :, :, 0][mask0]
                idx += nnz0

            combined = DD[n, :, :, 1] + DD[n, :, :, 2]
            mask_c = combined != 0
            nnz_c = np.sum(mask_c)
            if nnz_c > 0:
                ii, jj = np.nonzero(mask_c)
                row_indices[idx : idx + nnz_c] = base_row + ii
                col_indices[idx : idx + nnz_c] = p * n + jj
                data_vals[idx : idx + nnz_c] = combined[mask_c]
                idx += nnz_c

    # Trim to actual size
    row_indices = row_indices[:idx]
    col_indices = col_indices[:idx]
    data_vals = data_vals[:idx]

    # Add terminal derivatives that don't fit the tridiagonal structure
    if terminal_derivatives:
        terminal_rows = []
        terminal_cols = []
        terminal_data = []

        for (row_block, col_block), deriv_matrix in terminal_derivatives.items():
            mask = deriv_matrix != 0
            if np.any(mask):
                ii, jj = np.nonzero(mask)
                terminal_rows.extend(p * row_block + ii)
                terminal_cols.extend(p * col_block + jj)
                terminal_data.extend(deriv_matrix[mask])

        if terminal_data:
            row_indices = np.concatenate(
                [row_indices, np.array(terminal_rows, dtype=np.int32)]
            )
            col_indices = np.concatenate(
                [col_indices, np.array(terminal_cols, dtype=np.int32)]
            )
            data_vals = np.concatenate(
                [data_vals, np.array(terminal_data, dtype=DD.dtype)]
            )

    return scipy.sparse.coo_matrix(
        (data_vals, (row_indices, col_indices)), shape=(N * p, N * p)
    ).tocsr()


def _build_dense_jacobian(
    N: int, p: int, DD: np.ndarray, terminal_derivatives: dict
) -> np.ndarray:
    J = np.zeros((N * p, N * p))
    for n in range(N):
        if n == 0:
            J[p * n : p * (n + 1), p * n : p * (n + 1)] = np.eye(p, p)
        elif n == N and terminal_derivatives:
            # Terminal condition (last row) handled separately - skip for now
            pass
        elif n == N - 1:
            # Fallback for small T or when no terminal conditions
            J[p * n : p * (n + 1), p * (n - 1) : p * (n)] = DD[n, :, :, 0]
            J[p * n : p * (n + 1), p * n : p * (n + 1)] = (
                DD[n, :, :, 1] + DD[n, :, :, 2]
            )
        else:
            J[p * n : p * (n + 1), p * (n - 1) : p * (n)] = DD[n, :, :, 0]
            J[p * n : p * (n + 1), p * n : p * (n + 1)] = DD[n, :, :, 1]
            J[p * n : p * (n + 1), p * (n + 1) : p * (n + 2)] = DD[n, :, :, 2]

    # Add terminal derivatives
    if terminal_derivatives:
        for (row_block, col_block), deriv_matrix in terminal_derivatives.items():
            J[
                p * row_block : p * (row_block + 1),
                p * col_block : p * (col_block + 1),
            ] = deriv_matrix

    return J
