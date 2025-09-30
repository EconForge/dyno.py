from dynare_preprocessor import DynareModel as Modfile
from dyno.model import DynoModel
from dyno.language import pad_list, Normal, Deterministic
import numpy as np

from typing_extensions import Self
from typing import Any
from .typedefs import TVector, TMatrix

from dynare_preprocessor import PreprocessorException, UnsupportedFeatureException
from .errors import DynareParserError


class DynareModel(DynoModel):

    def import_model(self: Self, txt: str, deriv_order=1, params_deriv_order=0) -> None:
        """imports model written in `.mod` format into data attribute using Dynare's preprocessor

        Parameters
        ----------
        txt : str
            the model being imported in `.mod` form
        """
        try:
            self.data = Modfile(txt, deriv_order, params_deriv_order)
        except PreprocessorException as e:
            raise DynareParserError(e) from e

    def _set_context(self: Self) -> None:
        """retrieves calibration values"""

        c = self.data.context  # dynare preprocessor context
        endogenous = self.data.endogenous
        exogenous = self.data.exogenous
        variables = endogenous + exogenous
        parameters = self.data.parameters

        steady_states = {
            k: v for (k, v) in c.items() if (k in endogenous) or (k in exogenous)
        }
        constants = {k: v for (k, v) in c.items() if (k in parameters)}

        # read specification of exogenous shocks in the modfile
        assert len(self.data.trajectories) == 0 or len(self.data.covariances) == 0
        isdeterministic = len(self.data.trajectories) > 0
        exo = exogenous

        if isdeterministic:
            det_vals = {v: [] for v in exo}
            for var, traj in self.data.trajectories.items():
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
            for (var1, var2), val in self.data.covariances.items():
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
        }
        # self.paths = None
        # self.exogenous = self.processes
        self.context = context

    @property
    def equations(self):

        return self.data.equations

    def compute_residuals(self, y1, y2, y3, e):
        p = [self.context["constants"][p] for p in self.data.parameters]
        y, e = self.__steady_state_vectors__
        return self._f_dynamic(y, y, y, e, p)

    def compute_jacobians(self, y1, y2, y3, e):
        p = [self.context["constants"][p] for p in self.data.parameters]
        y, e = self.__steady_state_vectors__
        return self._f_dynamic(y, y, y, e, p, diff=True)

    def compute_derivatives(model):

        y = [model.steady_state[v] for v in model.symbols["endogenous"]]
        e = [model.steady_state[v] for v in model.symbols["exogenous"]]
        p = [model.context["constants"][v] for v in model.symbols["parameters"]]

        return model.data.derivatives(y, y, y, e, e, p)

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

        r = np.array(self.data.residuals(*args))

        if diff:
            jacobians = self.data.jacobians(*args)
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
