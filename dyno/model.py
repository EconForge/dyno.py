from numpy.linalg import solve as linsolve

import numpy as np
import yaml

from .solver import solve
from .misc import jacobian

from abc import ABC, abstractmethod

from typing import Callable, overload, Literal
from typing_extensions import Self
from .types import TVector, TMatrix, IRFType, Solver, SymbolType, DynamicFunction
from pandas import DataFrame
from .language import Normal


class RecursiveSolution:
    """VAR(1) representing a linearized model

    Attributes
    ----------
    X, Y, Σ: (N,N) Matrix
        parameters of the stationary VAR process $y_t = Xy_{t-1} + Yε_t$, where Σ is the covariance matrix of $ε_t$

    symbols: dict[SymbolType, list[str]]
        dictionary containing the symbols used in the model, the only allowed keys are "endogenous", "exogenous" and "parameters"

    x0: N Vector | None
        the state around which the linearization is done, generally the steady state, by default None

    evs: 2N Vector | None
        eigenvalues containing information about the stability of the model,
        only available if the qz solver was used for linearization, by default None
    """

    def __init__(
        self: Self,
        X: TMatrix,
        Y: TMatrix,
        Σ: TMatrix,
        symbols: dict[SymbolType, list[str]],
        x0: TVector | None = None,
        evs: TVector | None = None,
    ) -> None:

        self.x0 = x0
        self.X = X
        self.Y = Y
        self.Σ = Σ

        self.evs = evs

        self.symbols = symbols


class Model(ABC):
    """Abstract class representing an economic model"""

    symbols: dict[SymbolType, list[str]]
    exogenous: Normal
    __functions__: dict[Literal["dynamic"], DynamicFunction]

    @abstractmethod
    def get_calibration(self: Self) -> dict[str, float]:
        """Returns a dictionary containing the value of each parameter and variable of the model, indexed by their symbols"""
        pass

    def describe(self: Self) -> str:
        """Returns a string representation of the model's symbols"""
        return f"""
symbols: {self.symbols}
        """

    @overload
    def dynamic(
        self: Self, y0: TVector, y1: TVector, y2: TVector, e: TVector, p: TVector
    ) -> TVector:
        pass

    @overload
    def dynamic(
        self: Self, y0: TVector, y1: TVector, y2: TVector, e: TVector, p: TVector, diff: bool
    ) -> tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
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
        r = np.zeros(len(y0))
        self.__functions__["dynamic"](y0, y1, y2, e, p, r)
        d = np.zeros(len(self.symbols["exogenous"]))

        if diff:
            f = lambda a, b, c, d, e: self.dynamic(a, b, c, d, e)
            r1 = jacobian(lambda u: f(u, y1, y2, e, p), y0)
            r2 = jacobian(lambda u: f(y0, u, y2, e, p), y1)
            r3 = jacobian(lambda u: f(y0, y1, u, e, p), y2)
            r4 = jacobian(lambda u: f(y0, y1, y2, u, p), d)
            return r, r1, r2, r3, r4

        return r

    @overload
    def compute(self: Self, calibration: dict[str, float] = {}) -> TVector:
        pass

    @overload
    def compute(
        self: Self, calibration: dict[str, float] = {}, diff: bool = False
    ) -> tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
        pass

    def compute(
        self: Self, calibration: dict[str, float] = {}, diff: bool = False
    ) -> TVector | tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]:
        """Computes the dynamic function's value based on calibration state and parameters

        Parameters
        ----------
        calibration : dict[str, float], optional
            dictionary containing the value of each parameter and variable of the model, indexed by their symbols, by default {}
        diff : bool, optional
            if set to True returns the dynamic function's partial derivatives as well, by default False

        Returns
        -------
        TVector|tuple[TVector, TMatrix, TMatrix, TMatrix, TMatrix]
            value of the dynamic function at the state described by calibration, as well as its partial derivatives if diff is set to True
        """
        c = self.get_calibration(**calibration)
        v = self.symbols["endogenous"]
        p = self.symbols["parameters"]

        endogenous_values = [c[e] for e in v]
        parameter_values = [c[e] for e in p]
        # Reshapes necessary for static type checking
        y0 = np.reshape(endogenous_values, len(endogenous_values))
        p0 = np.reshape(parameter_values, len(parameter_values))

        e = np.zeros(len(self.symbols["exogenous"]))
        return self.dynamic(y0, y0, y0, e, p0, diff=diff)

    def solve(
        self: Self, calibration: dict[str, float] = {}, method: Solver = "qz"
    ) -> RecursiveSolution:
        """linearizes the model

        Parameters
        ----------
        calibration : dict[str, float], optional
            dictionary containing the value of each parameter and variable of the model, indexed by their symbols, by default {}
        method : Solver, optional
            chosen solver: either "ti" for fixed-point iteration or "qz" for generalized Schur decomposition, by default "qz"

        Returns
        -------
        RecursiveSolution
            linearized model
        """
        from .solver import solve as solveit

        r, A, B, C, D = self.compute(diff=True, calibration=calibration)

        X, evs = solveit(A, B, C, method=method)
        Y = linsolve(A @ X + B, -D)

        v = self.symbols["endogenous"]
        e = self.symbols["exogenous"]

        Σ = self.exogenous.Σ

        # a bit stupid
        c = self.get_calibration(**calibration)
        v = self.symbols["endogenous"]
        p = self.symbols["parameters"]

        endogenous_values = [c[e] for e in v]
        parameter_values = [c[e] for e in p]
        # Reshapes necessary for static type checking
        y0 = np.reshape(endogenous_values, len(endogenous_values))
        p0 = np.reshape(parameter_values, len(parameter_values))

        return RecursiveSolution(
            X, Y, Σ, {"endogenous": v, "exogenous": e}, evs=evs, x0=y0
        )


def irfs(
    model: Model, dr: RecursiveSolution, type: IRFType = "log-deviation"
) -> dict[str, DataFrame]:
    """Impulse response function simulation in response to shocks on each exogenous variable

    Parameters
    ----------
    dr : RecursiveSolution
        linearized model, contains all variables and parameters
    T : int, optional
        time horizon over which the simulation is done, by default 40
    type : IRFType, optional
        can be "level", "log-deviation" or "deviation", by default "level"

    Returns
    -------
    pd.DataFrame
        impulse response function of all endogenous variables to shocks on each exogenous variable
    """
    from .simul import irf

    res = {}
    for i, e in enumerate(model.symbols["exogenous"]):
        res[e] = irf(dr, i, type=type)

    return res
