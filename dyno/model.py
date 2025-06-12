from numpy.linalg import solve as linsolve

import numpy as np
import yaml

from .solver import solve
from .misc import jacobian

from abc import ABC, abstractmethod

from typing import Callable, Self, overload, Literal
from .types import Vector, Matrix, IRFType, Solver, SymbolType, DynamicFunction
from pandas import DataFrame
from .language import Normal

class RecursiveSolution:

    def __init__(self: Self, X: Matrix, Y: Matrix, Σ: Matrix, symbols: dict[str, list[str]], x0:Vector|None = None, evs: Vector | None = None) -> None:

        self.x0 = x0
        self.X = X
        self.Y = Y
        self.Σ = Σ

        self.evs = evs

        self.symbols = symbols


class Model(ABC):

    symbols: dict[SymbolType, list[str]]
    exogenous: Normal
    __functions__ : dict[Literal["dynamic"], DynamicFunction]

    # Necessary for static typechecking
    @abstractmethod
    def get_calibration(self: Self) -> dict[str, float]:
        pass

    def describe(self: Self) -> str:

        return f"""
symbols: {self.symbols}
        """

    # Overloaded functions needed for static type checker but ignored at runtime
    @overload
    def dynamic(self: Self, y0: Vector, y1: Vector, y2: Vector, e: Vector, p: Vector) -> Vector:
        pass

    @overload
    def dynamic(self: Self, y0: Vector, y1: Vector, y2: Vector, e: Vector, p: Vector, diff:bool) -> tuple[Vector, Matrix, Matrix, Matrix, Matrix]:
        pass

    def dynamic(self: Self, y0: Vector, y1: Vector, y2: Vector, e: Vector, p: Vector, diff:bool =False) -> Vector|tuple[Vector, Matrix, Matrix, Matrix, Matrix]:

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

    # Overloaded functions needed for static type checker but ignored at runtime
    @overload
    def compute(self: Self, calibration={}) -> Vector:
        pass

    @overload
    def compute(self: Self, calibration={}, diff: bool = False) -> tuple[Vector, Matrix, Matrix, Matrix, Matrix]:
        pass

    def compute(self: Self, calibration={}, diff: bool = False) -> Vector|tuple[Vector, Matrix, Matrix, Matrix, Matrix]:

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

    def solve(self: Self, calibration={}, method: Solver = "qz") -> RecursiveSolution:

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


def irfs(model : Model, dr : RecursiveSolution, type:IRFType="log-deviation") -> dict[str, DataFrame]:

    from .simul import irf

    res = {}
    for i, e in enumerate(model.symbols["exogenous"]):
        res[e] = irf(dr, i, type=type)

    return res
