from dynare_preprocessor import DynareModel, UnsupportedFeatureException
from dyno.model import Model
from dyno.language import pad_list, Normal, Deterministic
import numpy as np

from typing_extensions import Self
from typing import Any
from .typedefs import TVector, TMatrix


class Modfile(Model):

    def import_model(self: Self, txt: str, deriv_order=1, params_deriv_order=0) -> None:
        """imports model written in `.mod` format into data attribute using Dynare's preprocessor

        Parameters
        ----------
        txt : str
            the model being imported in `.mod` form
        """
        self.data = DynareModel(txt, deriv_order, params_deriv_order)

    def _set_symbols(self: Self) -> None:
        """sets symbols attribute of Model"""
        self.symbols = {}
        self.symbols["endogenous"] = self.data.endogenous
        self.symbols["exogenous"] = self.data.exogenous + self.data.exogenous_det
        self.symbols["parameters"] = self.data.parameters

    def _set_equations(self: Self) -> None:
        """sets equations attribute of Model"""
        self.equations = self.data.equations

    def _set_calibration(self: Self) -> None:
        """retrieves calibration values"""
        self.calibration = self.data.context

    def _set_exogenous(self: Self) -> None:
        self.exogenous = None
        assert len(self.data.trajectories) == 0 or len(self.data.covariances) == 0
        isdeterministic = len(self.data.trajectories) > 0
        exo = self.symbols["exogenous"]
        if isdeterministic:
            det_vals = {v: [] for v in exo}
            for var, traj in self.data.trajectories.items():
                for p1, p2, val in traj:
                    pad_list(det_vals[var], p2)
                    det_vals[var][p1 - 1 : p2] = [val] * (p2 - p1 + 1)
            self.exogenous = Deterministic(det_vals)
        else:
            n = len(exo)
            covar = np.zeros((n, n))
            index = {name: i for (i, name) in enumerate(exo)}
            for (var1, var2), val in self.data.covariances.items():
                covar[index[var1], index[var2]] = val
                covar[index[var2], index[var1]] = val
            self.exogenous = Normal(Σ=covar)

    def _set_dynamic(self: Self) -> None:
        pass

    def dynamic(
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
        if isinstance(self.exogenous, Deterministic):
            args[3] = []
        else:
            args[4] = []
        r = np.array(self.data.residuals(*args))

        if diff:
            jacobians = self.data.jacobians(*args)
            if isinstance(self.exogenous, Deterministic):
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
