from dyno.model import AbstractModel
import yaml
from .typedefs import TVector, TMatrix, IRFType, Solver, DynamicFunction

from dyno.dynsym.grammar import parser, str_expression
from dyno.dynsym.analyze import FormulaEvaluator
from typing_extensions import Self

import numpy as np

from .errors import LARKParserError, ParserError
from lark.exceptions import UnexpectedInput
from dyno.larkfiles import DynoFile, LModFile
from dyno.language import ProductNormal, Normal


class DynoModel(AbstractModel):

    def import_model(self: Self, txt: str) -> None:

        try:
            if self.filename.endswith(".mod"):
                self.data = LModFile(content=txt, filename=self.filename)
            elif self.filename.endswith(".dyno"):
                self.data = DynoFile(content=txt, filename=self.filename)
        except UnexpectedInput as e:
            raise LARKParserError(e, txt) from e

    def _set_context(self: Self) -> None:
        context = self.data.context.copy()
        self.context = context

    def recalibrate(self: Self, **calib):
        m = self.copy()
        m.data.context = {}
        m.data.process_assignments(**calib)
        m._set_context()
        m._set_exogenous()
        return m

    @property
    def equations(self):

        return [str_expression(eq) for eq in self.data.equations]

    def latex_equations(self):

        return self.data.latex_equations()

    def compute_residuals(self, y2, y1, y0, e):

        import math

        endogenous = self.symbols["endogenous"]
        exogenous = self.symbols["exogenous"]

        import copy

        cc = copy.deepcopy(self.data.context)

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

        results = [E.visit(eq) for eq in self.data.equations]

        r = np.array([float(el) for el in results])

        return r

    def compute_jacobians(
        self, y2, y1, y0, e
    ) -> tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix, TMatrix]:

        import numpy as np
        from dyno.dynsym.autodiff import DNumber as DN

        endogenous = self.symbols["endogenous"]
        exogenous = self.symbols["exogenous"]

        import copy

        cc = copy.deepcopy(self.data.context)

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
        results = [E.visit(eq) for eq in self.data.equations]

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

    def deterministic_guess(model, T=None):

        if T is None:
            T = model.context["constants"].get("T", 50)

        y, e = model.__steady_state_vectors__

        # initial guess
        v0 = np.concatenate([y, e])[None, :].repeat(T + 1, axis=0)

        # works if the is one and exactly one exogenous variable?
        # does it?
        for key, value in model.data.context["values"].items():
            i = model.symbols["variables"].index(key)
            v0[:, i] = 0.0  # TODO: use steady-state of exogenous process
            for a, b in value.items():
                v0[a, i] = b

        return v0

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

        import copy

        cc = copy.deepcopy(model.data.context)
        for i, name in enumerate(model.symbols["variables"]):
            cc["variables"][name] = {-1: v_b[:, i], 0: v[:, i], 1: v_f[:, i]}

        from dyno.dynsym.analyze import EquationsEvaluator

        E = EquationsEvaluator(cc)

        results = [E.visit(eq) for eq in model.data.equations]

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
            cc_T = copy.deepcopy(model.data.context)
            for i, name in enumerate(model.symbols["variables"]):
                cc_T["variables"][name] = {-1: v[T, i], 0: v[T, i], 1: v[T, i]}
            E_T = EquationsEvaluator(cc_T)
            res_T = [E_T.visit(eq) for eq in model.data.equations]
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

        import copy

        context = copy.deepcopy(model.data.context)
        for i, name in enumerate(model.symbols["variables"]):
            context["variables"][name] = {
                -1: DNumber(v_b[:, i], {(name, -1): 1.0}),
                0: DNumber(v[:, i], {(name, 0): 1.0}),
                1: DNumber(v_f[:, i], {(name, 1): 1.0}),
            }

        from dyno.dynsym.analyze import EquationsEvaluator

        E = EquationsEvaluator(context)

        results = [E.visit(eq) for eq in model.data.equations]

        n_exo = len(model.symbols["variables"]) - len(results)

        y, e = model.__steady_state_vectors__

        # get exo values
        # works if the is one and exactly one exogenous variable?
        # does it?
        v1 = np.concatenate([y, e])[None, :].repeat(T + 1, axis=0)
        for key, value in model.data.context["values"].items():
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
            context_T = copy.deepcopy(model.data.context)
            for i, name in enumerate(model.symbols["variables"]):
                context_T["variables"][name] = {
                    -1: DNumber(v[T, i], {(name, -1): 1.0}),
                    0: DNumber(v[T, i], {(name, 0): 1.0}),
                    1: DNumber(v[T, i], {(name, 1): 1.0}),
                }
            E_T = EquationsEvaluator(context_T)
            results_T = [E_T.visit(eq) for eq in model.data.equations]
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
                import scipy.sparse

                # Build sparse matrix directly using COO format with vectorized operations
                # Pre-allocate lists with estimated size
                max_nnz = (
                    p + (N - 2) * 3 * p * p + 2 * p * p
                )  # upper bound on non-zeros
                row_indices = np.empty(max_nnz, dtype=np.int32)
                col_indices = np.empty(max_nnz, dtype=np.int32)
                data_vals = np.empty(max_nnz, dtype=DD.dtype)

                idx = 0

                # First block: identity matrix (n=0)
                n = 0
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

                    for (
                        row_block,
                        col_block,
                    ), deriv_matrix in terminal_derivatives.items():
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

                J = scipy.sparse.coo_matrix(
                    (data_vals, (row_indices, col_indices)), shape=(N * p, N * p)
                ).tocsr()
            else:
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
                        J[p * n : p * (n + 1), p * (n + 1) : p * (n + 2)] = DD[
                            n, :, :, 2
                        ]

                # Add terminal derivatives
                if terminal_derivatives:
                    for (
                        row_block,
                        col_block,
                    ), deriv_matrix in terminal_derivatives.items():
                        J[
                            p * row_block : p * (row_block + 1),
                            p * col_block : p * (col_block + 1),
                        ] = deriv_matrix

            return res.ravel(), J
