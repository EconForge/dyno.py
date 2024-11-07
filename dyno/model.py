from numpy.linalg import solve as linsolve

import numpy as np
import yaml

from .solver import solve
from .misc import jacobian


class RecursiveSolution:

    def __init__(self, X, Y, Σ, symbols, evs=None):

        self.X = X
        self.Y = Y
        self.Σ = Σ

        self.evs = evs

        self.symbols = symbols


class Normal:

    def __init__(self, Σ, vars):

        self.Σ = Σ
        self.variables = tuple(*vars)


class Model:

    def describe(self):

        return f"""
symbols: {self.symbols}
        """

    def dynamic(self, y0, y1, y2, e, p, diff=False):

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

    def compute(self, diff=False, calibration={}):

        c = self.get_calibration(**calibration)
        v = self.symbols["endogenous"]
        p = self.symbols["parameters"]
        y0 = np.array([c[e] for e in v])
        p0 = np.array([c[e] for e in p])
        e = np.zeros(len(self.symbols["exogenous"]))
        return self.dynamic(y0, y0, y0, e, p0, diff=diff)

    def solve(self, calibration={}, method="qz") -> RecursiveSolution:

        from .solver import solve as solveit

        r, A, B, C, D = self.compute(diff=True, calibration=calibration)

        X, evs = solveit(A, B, C, method=method)
        Y = linsolve(A @ X + B, -D)

        v = self.symbols["endogenous"]
        e = self.symbols["exogenous"]

        Σ = self.exogenous.Σ

        return RecursiveSolution(X, Y, Σ, {"endogenous": v, "exogenous": e}, evs=evs)


def irfs(model, dr):

    from .simul import irf

    res = {}
    for i, e in enumerate(model.symbols["exogenous"]):
        res[e] = irf(dr, i)

    return res
